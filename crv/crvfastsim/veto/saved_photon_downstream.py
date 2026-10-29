"""Bounded four-arm downstream timing control on saved photons, never an F estimator."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from ..response import sipm as SI
from ..reco import waveform as WF, pulsefit
from . import optical_timing_probe as OP, optical_timing_report as OR
from . import replay_sipm_control as SC, replay_waveform_control as WC
from .replay import IDS, sha256

MAP_SHA = "b8cd7588eac90f72df56cfb95fbaa102671647a46e7df403af267ad76200f236"
TEMPLATE_SHA = "0642a5a492d5afcc6174927a5fa769a90e0b3850e7998d603138444332475190"
STAGES = {
    "charges": (["event", "chan", "root", "dark", "generation", "px", "py"], ["time", "charge", "pe"]),
    "digis": (["event", "chan", "starttdc", "sample"], ["adc"]),
    "pulses": (["event", "chan", "starttdc", "seq"], ["pe", "time_relative", "time_reco", "failed", "zerondf", "channel_ignored"]),
}
SCOPE = ("Bounded census-selected extreme-time frames, one saved optical draw and one downstream seed per input. "
         "Fixed-integer pixels, grouped fast crosstalk, zero traps; no stock queue/RNG equivalence. "
         "Thermal OFF/ON only on the saved optical channel union, not all detector channels. "
         "No new optics, stock response, coincidence finding, F estimate, finite-library error or C4 promotion.")


def compare_rows(left, right, keys, fields):
    """Exact identity joins; waveform start/sequence matches are structural, not pulse truth matches."""
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("duplicate stage comparison key")
    for frame in (left, right):
        if any(not np.isfinite(frame[k].to_numpy()).all() for k in keys + fields):
            raise ValueError("nonfinite stage comparison value")
    merged = left[keys + fields].merge(right[keys + fields], on=keys, how="outer", suffixes=("_a", "_b"),
                                       indicator=True, validate="one_to_one", sort=True)
    both = merged._merge.eq("both")
    changed = ~both
    maxima = {}
    for field in fields:
        a, b = merged.loc[both, f"{field}_a"], merged.loc[both, f"{field}_b"]
        changed.loc[both] |= a.to_numpy() != b.to_numpy()
        delta = b.to_numpy(dtype=float) - a.to_numpy(dtype=float)
        maxima[field] = float(np.max(np.abs(delta))) if len(delta) else None
    diff = merged.loc[changed].reset_index(drop=True)
    return {"rows_a": len(left), "rows_b": len(right), "common": int(both.sum()),
            "only_a": int(merged._merge.eq("left_only").sum()), "only_b": int(merged._merge.eq("right_only").sum()),
            "changed_common": int((changed & both).sum()), "exact": not bool(changed.any()),
            "max_abs_delta_common": maxima,
            "changed_channels": diff[["event", "chan"]].drop_duplicates().to_dict(orient="records")}, diff


def conditions(stock, events, q, nbars=1280):
    """Restrict the legacy control's event-only key to a unique fixed run/subrun domain."""
    if (events.run.nunique() != 1 or events.subrun.nunique() != 1 or events.event.duplicated().any()
            or events.event.min() < 0 or events.event.max() >= 2**20 or len(events) > 10):
        raise ValueError("requires bounded unique single-run/subrun events")
    wanted = pd.MultiIndex.from_frame(events[list(IDS)])
    if not pd.MultiIndex.from_frame(q[list(IDS)]).isin(wanted).all():
        raise ValueError("photon outside full event census")
    with uproot.open(stock) as root:
        c = root["crvdump/calib"].arrays(library="pd")
    cols = ["channel", "pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb"]
    c = c[cols].sort_values("channel").rename(columns={"channel": "chan"}).reset_index(drop=True)
    if (not np.array_equal(c.chan, np.arange(nbars * 4)) or not np.isfinite(c.to_numpy()).all()
            or (c[["pulsearea", "pulseheight"]] <= 0).any().any()):
        raise ValueError("invalid complete calibration")
    for col in ("chan", "status", "roc", "feb"):
        if c[col].dtype.kind not in "iu" or (c[col] < 0).any():
            raise ValueError("invalid integer calibration")
    if not np.array_equal(q.live_channel, (c.status.to_numpy()[q.chan.to_numpy()] & 5) == 0):
        raise ValueError("saved live-channel mask changed")
    union = q[["event", "chan"]].drop_duplicates()
    cond = union.merge(c, on="chan", validate="many_to_one")
    cond = cond.merge(events[["event", "pbtime"]], on="event", validate="many_to_one")
    cond["w"] = -cond.pbtime.to_numpy().astype(np.float32).astype(float)
    return cond[(cond.status & 5) == 0].sort_values(["event", "chan"]).reset_index(drop=True)


def encode_digis(d):
    n = d["adc"].shape[1]
    return pd.DataFrame({"event": np.repeat(d["chan"] // WC.MULT, n), "chan": np.repeat(d["chan"] % WC.MULT, n),
                         "starttdc": np.repeat(d["starttdc"], n), "sample": np.tile(np.arange(n), len(d["chan"])),
                         "adc": d["adc"].ravel()})


def run_arm(q, cond, pmap, cfg, wcfg, seed, arm, dark):
    if arm not in OP.ARMS:
        raise ValueError("unknown timing arm")
    time_key = "time_historical_relative" if arm in ("historical", "rejection_only") else "time_raw_wrapped_relative"
    selected = q.loc[q[f"{arm}_kept"], ["event", "chan", "photon_id", time_key]].rename(columns={time_key: "time"})
    optical = SC.optical_candidates(selected, pmap, cfg, seed, "fixed_integer")
    candidates = pd.concat([optical, dark], ignore_index=True)
    charges, attempts = SC.simulate_candidates(candidates, cfg, seed, end=1500.)
    d, samples, clusters, phases = WC.synthesize(charges, cond, seed, cfg=wcfg)
    pulses, waveforms, fit_samples = WC.reconstruct(d, cond)
    pulses = pulses.merge(waveforms[["wf", "starttdc"]], on="wf", how="left", validate="many_to_one")
    if pulses.starttdc.isna().any():
        raise ValueError("pulse missing waveform identity")
    products = dict(candidates=candidates, attempts=attempts, charges=charges, digis=encode_digis(d),
                    pulses=pulses, samples=samples, clusters=clusters, phases=phases,
                    waveforms=waveforms, fit_samples=fit_samples)
    return products, {"photons": len(selected), "dark_candidates": len(dark), "charges": len(charges),
                      "digis": len(d["chan"]), "pulses": len(pulses)}


def persist(frame, path, outputs):
    frame.to_parquet(path, index=False)
    outputs[path.name] = sha256(path)


def comparisons(products, out, prefix, outputs):
    result = {}
    for arm in ("arrival_only", "rejection_only", "stock_timing"):
        result[f"historical_vs_{arm}"] = {}
        for stage, (keys, fields) in STAGES.items():
            stats, diff = compare_rows(products["historical"][stage], products[arm][stage], keys, fields)
            result[f"historical_vs_{arm}"][stage] = stats
            persist(diff, out / f"{prefix}_historical_vs_{arm}_{stage}_diff.parquet", outputs)
    result["rejection_only_vs_stock_timing"] = {}
    for stage, (keys, fields) in STAGES.items():
        stats, diff = compare_rows(products["rejection_only"][stage], products["stock_timing"][stage], keys, fields)
        result["rejection_only_vs_stock_timing"][stage] = stats
        persist(diff, out / f"{prefix}_rejection_only_vs_stock_timing_{stage}_diff.parquet", outputs)
    return result


def run(probe_dir, map_path, template, out, seed=2027000):
    probe_dir, out = Path(probe_dir), Path(out)
    if out.exists():
        raise ValueError("choose a new output directory")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    OP.bound(map_path, MAP_SHA)
    OP.bound(template, TEMPLATE_SHA)
    with uproot.open(map_path) as root:
        _, x, y = root["photonMap"].to_numpy()
    if not np.array_equal(x, np.arange(41)) or not np.array_equal(y, np.arange(41)):
        raise ValueError("requires unit-width 0..40 photon-map axes")
    summary_path = probe_dir / "summary.json"
    input_hash = sha256(summary_path)
    prior = json.loads(summary_path.read_text())
    if not 0 < sum(len(b["selected_events"]) for b in prior["blocks"]) <= 10:
        raise ValueError("requires one to ten saved frames")
    out.mkdir(parents=True)
    OR.run(probe_dir, out / "optical_verified.json")
    pmap = SI.photon_map(str(map_path))
    wcfg = dataclasses.replace(WF.WaveformConfig(), single_pe_file=str(template))
    outputs, blocks = {"optical_verified.json": sha256(out / "optical_verified.json")}, []
    for i, b in enumerate(prior["blocks"]):
        q = pd.read_parquet(probe_dir / b["artifacts"]["photons"]["path"])
        events = pd.read_parquet(probe_dir / b["artifacts"]["events"]["path"])
        stock = [p for p in b["input_bindings"] if p.endswith("/stock.root")]
        if len(stock) != 1:
            raise ValueError("missing/ambiguous stock conditions")
        cond = conditions(stock[0], events, q)
        persist(cond, out / f"input{i}_conditions.parquet", outputs)
        models = {}
        for thermal in (0., 1e-4):
            prefix = f"input{i}_thermal{int(thermal > 0)}"
            cfg = dataclasses.replace(SI.SiPMConfig(), thermal_rate=thermal)
            dark = SC.thermal_candidates(cond, cfg, seed + i, start=100., end=1500.)
            stages, records = {}, {}
            for arm in OP.ARMS:
                start = time.perf_counter()
                stages[arm], counts = run_arm(q, cond, pmap, cfg, wcfg, seed + i, arm, dark)
                for name, frame in stages[arm].items():
                    persist(frame, out / f"{prefix}_{arm}_{name}.parquet", outputs)
                records[arm] = counts | {"elapsed_seconds": time.perf_counter() - start}
            models[prefix] = {"arms": records, "comparisons": comparisons(stages, out, prefix, outputs)}
        blocks.append({"input_sha256": b["input_sha256"], "seed": seed + i, "events": events[list(IDS)].to_dict(orient="records"),
                       "channels": len(cond), "models": models})
    OP.bound(summary_path, input_hash)
    OP.bound(map_path, MAP_SHA)
    OP.bound(template, TEMPLATE_SHA)
    code = [Path(__file__), Path(SC.__file__), Path(WC.__file__), Path(OR.__file__),
            Path(SI.__file__), Path(WF.__file__), Path(pulsefit.__file__)]
    result = {"schema": "crv-saved-photon-downstream-v1", "scope": SCOPE, "blocks": blocks,
              "input_summary": str(summary_path), "input_summary_sha256": input_hash,
              "assets_sha256": {str(map_path): MAP_SHA, str(template): TEMPLATE_SHA},
              "code_sha256": {str(p): sha256(p) for p in code}, "output_sha256": outputs,
              "sipm_configs": [dataclasses.asdict(dataclasses.replace(SI.SiPMConfig(), thermal_rate=t)) for t in (0., 1e-4)],
              "waveform_config": dataclasses.asdict(wcfg), "pulse_config": dataclasses.asdict(pulsefit.RecoConfig()),
              "hash_version": SC.HASH_VERSION, "random_contract": SC.RANDOM_CONTRACT,
              "rng_scope": "Input-specific seed; optical root/channel/event avalanche; event/ROC/FEB phase; event/channel/TDC noise. Fixed optical candidate IDs.",
              "pulse_comparison": "Waveform start TDC and sequence structural identity, not pulse truth matching.",
              "F": None, "finite_library_uncertainty": None, "timing_closure": False}
    with (out / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def verify(directory, out):
    """Recompute saved stage contrasts exactly; no detector or optical regeneration."""
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise ValueError("choose a new verification directory")
    path = directory / "summary.json"
    digest = sha256(path)
    saved = json.loads(path.read_text())
    if saved["schema"] != "crv-saved-photon-downstream-v1" or saved["F"] is not None or saved["timing_closure"] is not False:
        raise ValueError("wrong diagnostic contract")
    OP.bound(saved["input_summary"], saved["input_summary_sha256"])
    for file, value in (saved["assets_sha256"] | saved["code_sha256"]).items():
        OP.bound(file, value)
    if not saved["blocks"] or any(not b["models"] for b in saved["blocks"]):
        raise ValueError("empty saved comparison domain")
    required = {"optical_verified.json"}
    for i, block in enumerate(saved["blocks"]):
        if set(block["models"]) != {f"input{i}_thermal{n}" for n in (0, 1)}:
            raise ValueError("incomplete thermal comparison domain")
        required.update(f"{prefix}_{arm}_{stage}.parquet" for prefix in block["models"] for arm in OP.ARMS for stage in STAGES)
    if not required <= saved["output_sha256"].keys():
        raise ValueError("missing required artifact hashes")
    for name, value in saved["output_sha256"].items():
        if Path(name).name != name:
            raise ValueError("invalid artifact path")
        OP.bound(directory / name, value)
    out.mkdir(parents=True)
    outputs, result = {}, []
    for block in saved["blocks"]:
        models = {}
        for prefix, model in block["models"].items():
            stages = {arm: {stage: pd.read_parquet(directory / f"{prefix}_{arm}_{stage}.parquet") for stage in STAGES}
                      for arm in OP.ARMS}
            comp = comparisons(stages, out, prefix, outputs)
            if comp != model["comparisons"]:
                raise ValueError("saved comparisons disagree")
            models[prefix] = comp
        result.append({"input_sha256": block["input_sha256"], "models": models})
    OP.bound(path, digest)
    report = {"summary_sha256": digest, "comparisons_reproduced_exactly": True, "blocks": result,
              "scope": SCOPE, "regenerated_response": False, "verifier_sha256": sha256(__file__)}
    with (out / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe")
    ap.add_argument("--photon-map")
    ap.add_argument("--template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2027000)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        result = verify(a.probe, a.out)
    else:
        if not a.photon_map or not a.template:
            ap.error("--photon-map and --template are required for downstream generation")
        result = run(a.probe, a.photon_map, a.template, a.out, a.seed)
    print(json.dumps(result["blocks"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
