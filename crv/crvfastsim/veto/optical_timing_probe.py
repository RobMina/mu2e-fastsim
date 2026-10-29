"""Common-photon diagnostic of census-selected extreme-time frames, never production response."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .. import geometry
from ..response import photons
from . import fastframes
from .replay import EVENT_TYPES, IDS, STEP_TYPES, sha256
from .replay_check import check_events
from .replay_metrics import verify_metadata
from .replay_policy import INTENDED, apply_policy
from .replay_timing import DIGI_END, PERIOD, stock_step_mask, wrap_arrivals
from .step_timing_audit import COUNTERS_SHA, STOCK_SHA, read_input

ASSETS = {
    "CRVResponse/src/CrvPhotonGenerator_module.cc": "d448ab0d278681b0f8095ec7eba4ce0e6adf54aad87dee5ab4740358d41c227b",
    "CRVResponse/src/MakeCrvPhotons.cc": "f3c50b2ef4e38e973d8043450adb7f20a484acd6177467d05a2ed355c4e15201",
    "Mu2eG4/geom/crv_counters_run1a_v01.txt": COUNTERS_SHA,
}
LUT_SHA = "1c39bfdafc617d6edc6b703bc7f320729a1c30c5d4a990765a924ab5f50b6077"
ARMS = ("historical", "arrival_only", "rejection_only", "stock_timing")
RNG_CONTRACT = ("PCG64 fixed ordered subset, identical initial states for raw/folded production optics; "
                "assert exact step/channel output identities and identical final RNG state. "
                "Photon keys are input hash/seed/source row/channel/ordinal. Post-draw masks preserve keys. "
                "Not a subset-, chunk-, or permutation-invariant keyed optical generator; not stock RNG recovery.")
SCOPE = ("Frames selected by prior step-fold roundoff, not representative exposure. Full optical channel union; "
         "no SiPM, thermal, waveform, pulse, coincidence or F calculation. Timing operations only; "
         "not stock photons, stock queue/RNG equivalence, finite-library uncertainty or C4 promotion.")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bound(path, digest):
    path = Path(path)
    if sha256(path) != digest:
        raise ValueError(f"changed bound input: {path}")
    return path


def timing_source_bridge(census_source, fixed_source):
    """Prove the event-processing timing body is byte-identical across the two pinned releases."""
    old = bound(census_source, STOCK_SHA).read_text()
    new = bound(fixed_source, ASSETS["CRVResponse/src/CrvPhotonGenerator_module.cc"]).read_text()
    marker = "  void CrvPhotonGenerator::produce(art::Event& event)"
    if old.count(marker) != 1 or new.count(marker) != 1 or old[old.index(marker):] != new[new.index(marker):]:
        raise ValueError("census/fixed-stock event processing differs")
    return {"census_source_sha256": STOCK_SHA, "fixed_source_sha256": ASSETS["CRVResponse/src/CrvPhotonGenerator_module.cc"],
            "produce_body_sha256": hashlib.sha256(old[old.index(marker):].encode()).hexdigest(),
            "produce_body_byte_identical": True,
            "scope": "Event-processing body including timing only; fixed stock has a separate per-sector yield-sharing fix."}


def validate_stock_events(expected, actual):
    """TTree numpy arrays are a dict; use a row-oriented record for the completed-event count."""
    return check_events(expected, ak.Array(actual), len(expected))


def select_table(tree, events):
    """Basket-bounded full-tuple selection preserves original ROOT row, including repeated steps."""
    wanted = pd.MultiIndex.from_frame(events[list(IDS)])
    if wanted.has_duplicates or not len(wanted):
        raise ValueError("requires unique nonempty selected events")
    chunks, offset = [], 0
    for columns in tree.iterate(step_size="32 MB", library="np"):
        frame = pd.DataFrame(columns)
        mask = pd.MultiIndex.from_frame(frame[list(IDS)]).isin(wanted)
        chunks.append(frame.loc[mask].assign(source_row=np.flatnonzero(mask) + offset))
        offset += len(frame)
    return pd.concat(chunks, ignore_index=True)


def join_particles(steps, particles):
    """Exact full-event SimParticle identity only; no time/distance association."""
    keys = [*IDS, "sp"]
    parts = particles[[*IDS, "key", "pdg"]].rename(columns={"key": "sp"})
    if parts.duplicated(keys).any():
        raise ValueError("duplicate particle identity")
    joined = steps.merge(parts, on=keys, how="left", sort=False, validate="many_to_one")
    if joined.pdg.isna().any() or not np.array_equal(joined.source_row, steps.source_row):
        raise ValueError("missing particle identity or changed step order")
    joined["pdg"] = joined.pdg.astype(np.int64)
    return joined


def common_photons(steps, events, geom, cfg, tables, deviation, status, seed):
    """Two exact same-state calls to unchanged production optics; masks never consume RNG."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ids = pd.MultiIndex.from_frame(events[list(IDS)])
    if ids.has_duplicates or not len(ids):
        raise ValueError("duplicate or empty event census")
    event_row = ids.get_indexer(pd.MultiIndex.from_frame(steps[list(IDS)]))
    if np.any(event_row < 0) or steps.source_row.duplicated().any():
        raise ValueError("step outside event census or duplicate source row")
    for key in (*IDS, "source_row", "bar", "sp", "pdg"):
        a = steps[key].to_numpy()
        if a.dtype.kind not in "iu" or (key != "pdg" and np.any(a < 0)):
            raise ValueError(f"invalid integer step identity: {key}")
    if steps.bar.max() >= geom.nbars or len(status) != geom.nbars * 4 or len(deviation) != len(status):
        raise ValueError("geometry/conditions domain mismatch")
    if not np.isfinite(deviation).all() or not np.isfinite(events.pbtime).all():
        raise ValueError("nonfinite conditions")
    w = -events.pbtime.to_numpy().astype(np.float32).astype(float)[event_row]
    t0, t1 = steps.t0.to_numpy(), steps.t1.to_numpy()
    accepted = stock_step_mask(t0, t1, w)
    shift = np.floor((t0 - w) / PERIOD) * PERIOD
    folded = steps.assign(t0=t0 - w - shift, t1=t1 - w - shift)
    raw_rng, folded_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    raw_s, raw_c, raw_t = photons.make_photons(steps, steps.pdg.to_numpy(), geom, cfg, tables, deviation, raw_rng)
    si, ch, folded_t = photons.make_photons(folded, steps.pdg.to_numpy(), geom, cfg, tables, deviation, folded_rng)
    if not np.array_equal(raw_s, si) or not np.array_equal(raw_c, ch):
        raise ValueError("time-only coordinate change altered photon identities")
    if digest_json(raw_rng.bit_generator.state) != digest_json(folded_rng.bit_generator.state):
        raise ValueError("time-only coordinate change altered RNG consumption")
    if (np.asarray(si).dtype.kind not in "iu" or np.asarray(ch).dtype.kind not in "iu"
            or np.any(si < 0) or np.any(si >= len(steps)) or np.any(ch < 0) or np.any(ch >= len(status))):
        raise ValueError("invalid emitted photon index")
    if np.any(ch // 4 != steps.bar.to_numpy()[si]) or not np.isfinite(raw_t).all() or not np.isfinite(folded_t).all():
        raise ValueError("photon bar identity or finite-time violation")
    wrapped_abs = wrap_arrivals(raw_t, w[si])
    wrapped_rel = wrapped_abs - w[si]
    folded_wrapped = wrap_arrivals(folded_t)
    live = (np.asarray(status)[ch] & 0b101) == 0
    q = pd.DataFrame({k: steps[k].to_numpy()[si] for k in IDS})
    q["source_step_row"], q["step_index"], q["chan"], q["draw_seed"] = steps.source_row.to_numpy()[si], si, ch, seed
    q["ordinal"] = q.groupby(["source_step_row", "chan"], sort=False).cumcount()
    q["photon_id"] = np.arange(len(q), dtype=np.int64)
    q["event_row"], q["origin"] = event_row[si], w[si]
    q["time_raw_absolute"], q["time_historical_relative"] = raw_t, folded_t
    q["time_raw_wrapped_absolute"], q["time_raw_wrapped_relative"] = wrapped_abs, wrapped_rel
    q["time_folded_wrapped_relative"] = folded_wrapped
    residual = wrapped_rel - folded_wrapped
    q["raw_folded_residual_ns"] = residual
    q["raw_folded_circular_residual_ns"] = np.minimum(abs(residual), abs(abs(residual) - PERIOD))
    q["live_channel"], q["stock_step_kept"] = live, accepted[si]
    q["historical_kept"] = live & (folded_t <= DIGI_END)
    q["arrival_only_kept"] = live & (wrapped_abs <= w[si] + DIGI_END)
    q["rejection_only_kept"] = q.historical_kept & q.stock_step_kept
    q["stock_timing_kept"] = q.arrival_only_kept & q.stock_step_kept
    q["folded_arrival_kept"] = live & (folded_wrapped <= DIGI_END)
    q["raw_folded_upper_flip"] = q.arrival_only_kept != q.folded_arrival_kept
    q["raw_folded_late_flip"] = live & ((wrapped_abs >= w[si] + 1400.) != (folded_wrapped >= 1400.))
    q["rescued_by_arrival"] = q.arrival_only_kept & ~q.historical_kept
    q["removed_by_arrival"] = q.historical_kept & ~q.arrival_only_kept
    q["rescued_survives_step"] = q.rescued_by_arrival & q.stock_step_kept
    q["accepted_time_changed"] = q.historical_kept & q.stock_timing_kept & (folded_t != wrapped_rel)
    return q, {"final_rng_sha256": digest_json(raw_rng.bit_generator.state), "same_photon_identities": True,
               "same_rng_consumption": True, "contract": RNG_CONTRACT}


FLAGS = tuple(f"{arm}_kept" for arm in ARMS) + ("folded_arrival_kept", "raw_folded_upper_flip", "raw_folded_late_flip",
    "rescued_by_arrival", "removed_by_arrival", "rescued_survives_step", "accepted_time_changed")


def summaries(q, events, geom):
    """Every selected frame plus the union of all emitted channels, not retained member channels."""
    group = q.groupby("event_row", sort=True)
    result = events.reset_index(drop=True).copy()
    result["photons"] = group.size().reindex(range(len(events)), fill_value=0).to_numpy()
    for name in FLAGS:
        result[name] = group[name].sum().reindex(range(len(events)), fill_value=0).to_numpy(dtype=np.int64)
    channels = q.groupby([*IDS, "chan"], sort=True)[list(FLAGS)].sum().reset_index()
    channels["sector"] = np.asarray(geom.sector_names)[geom.sector[channels.chan.to_numpy(dtype=int) // 4]]
    return result, channels


def load_selection(audit_dir, verified):
    """Select every frame with measured endpoint-fold residual from an exactly verified census."""
    audit_dir = Path(audit_dir)
    verification = json.loads(Path(verified).read_text())
    if verification.get("all_ledgers_reclassified_exactly") is not True or verification.get("pinned_geometry_verified") is not True:
        raise ValueError("requires exact saved-census verification")
    path = bound(audit_dir / "summary.json", verification["summary_sha256"])
    report = json.loads(path.read_text())
    if report["schema"] != "crv-step-timing-audit-v1" or report["counters"]["sha256"] != COUNTERS_SHA:
        raise ValueError("wrong census contract")
    output = []
    for i, row in enumerate(report["inputs"]):
        artifact = row["artifacts"]["steps"]
        if artifact["path"] != f"input{i}_steps.parquet":
            raise ValueError("unexpected census artifact path")
        ledger = pd.read_parquet(bound(audit_dir / artifact["path"], artifact["sha256"]))
        if len(ledger) != artifact["rows"] or int(ledger.fold_start_roundoff.sum()) != row["counts"]["fold_start_roundoff"]:
            raise ValueError("census rows or exception count disagreement")
        targets = ledger.loc[ledger.fold_start_roundoff, list(IDS)].drop_duplicates()
        if not len(targets):
            continue
        if len(targets) > 10 or sum(len(item["targets"]) for item in output) + len(targets) > 10:
            raise ValueError("bounded probe supports at most ten selected frames")
        selected = ledger[pd.MultiIndex.from_frame(ledger[list(IDS)]).isin(pd.MultiIndex.from_frame(targets))].copy()
        output.append({"record": row, "targets": targets, "ledger": selected})
    if not output:
        raise ValueError("no roundoff-selected frames")
    return output


def block_probe(item, paired_row, photon_dev, geom, cfg, contract, tabs, seed, out, number):
    """Use bound replay and fixed-stock conditions only; stock observables are not regenerated."""
    started = time.perf_counter()
    record = item["record"]
    input_path = Path(record["path"])
    check_path = bound(paired_row["check"], paired_row["check_sha256"])
    check = json.loads(check_path.read_text())
    meta_path = bound(input_path.with_name("fast.json"), check["fast_metadata_sha256"])
    meta = json.loads(meta_path.read_text())
    identity, _, _ = verify_metadata(check, meta, sha256(meta_path))
    if identity["response_contract"] != contract or check.get("conditions_equal") is not True:
        raise ValueError("requires fixed-stock intended response identity")
    bound(photon_dev, identity["photon_dev_sha256"])
    stock_path = bound(check["stock"], check["stock_sha256"])
    if check["input_sha256"] != record["sha256"] or check["input_manifest_sha256"] != record["manifest_sha256"]:
        raise ValueError("census and fixed-stock check input disagree")
    input_bindings = {str(p): sha256(p) for p in (input_path, input_path.with_suffix(".json"),
                      check_path, meta_path, stock_path, Path(photon_dev))}
    _, all_events, binding = read_input(input_path)
    if binding["sha256"] != record["sha256"] or binding["manifest_sha256"] != record["manifest_sha256"]:
        raise ValueError("census payload binding changed")
    with uproot.open(stock_path) as root:
        actual = root["crvdump/events"].arrays(library="np")
        validate_stock_events(all_events, actual)
        calib = root["crvdump/calib"].arrays(library="pd")
    if (len(calib) != geom.nbars * 4 or not np.array_equal(np.sort(calib.channel), np.arange(geom.nbars * 4))
            or calib.status.dtype.kind not in "iu" or np.any(calib.status < 0)):
        raise ValueError("incomplete or invalid channel calibration")
    selected_index = pd.MultiIndex.from_frame(item["targets"][list(IDS)])
    events = all_events[pd.MultiIndex.from_frame(all_events[list(IDS)]).isin(selected_index)].reset_index(drop=True)
    with uproot.open(input_path) as root:
        steps, parts = (select_table(root[f"dump/{name}"], events) for name in ("steps", "particles"))
    expected = item["ledger"]
    if not np.array_equal(steps.source_row, expected.step_row):
        raise ValueError("changed selected source row identity")
    pd.testing.assert_frame_equal(steps[[*IDS, "bar", "t0", "t1", "edep"]].reset_index(drop=True),
                                  expected[[*IDS, "bar", "t0", "t1", "edep"]].reset_index(drop=True))
    steps = join_particles(steps, parts)
    steps["endpoint_roundoff"] = expected.fold_start_roundoff.to_numpy()
    dev = np.loadtxt(photon_dev, delimiter=",", comments=("#", "TABLE"))
    if (dev.ndim != 2 or dev.shape != (geom.nbars * 4, 2) or not np.isfinite(dev).all()
            or not np.array_equal(np.sort(dev[:, 0]), np.arange(geom.nbars * 4))):
        raise ValueError("requires complete unique photon deviations")
    cond = fastframes.conditions_from_dump(str(stock_path), str(photon_dev))
    q, rng = common_photons(steps, events, geom, cfg, tabs, cond.deviation, cond.status, seed)
    q["endpoint_roundoff_step"] = steps.endpoint_roundoff.to_numpy()[q.step_index]
    frames, channels = summaries(q, events, geom)
    artifacts = {}
    for name, frame in (("steps", steps), ("photons", q), ("events", frames), ("channels", channels)):
        target = out / f"input{number}_{name}.parquet"
        frame.to_parquet(target, index=False)
        artifacts[name] = {"path": target.name, "sha256": sha256(target), "rows": len(frame)}
    for path, digest in input_bindings.items():
        bound(path, digest)
    extreme = q[q.endpoint_roundoff_step]
    return {"input_sha256": record["sha256"], "input": str(input_path), "seed": seed, "rng": rng,
            "selected_events": frames.to_dict(orient="records"), "artifacts": artifacts,
            "counts": {k: int(q[k].sum()) for k in FLAGS}, "photons": len(q),
            "extreme_step_photons": len(extreme), "extreme_step_counts": {k: int(extreme[k].sum()) for k in FLAGS},
            "max_raw_folded_circular_residual_ns": float(q.raw_folded_circular_residual_ns.max()) if len(q) else None,
            "photon_channels": len(channels), "elapsed_seconds": time.perf_counter() - started,
            "input_bindings": input_bindings}


def run(audit_dir, verified, paired_path, photon_dev, offline, lut_dir, out, seed):
    out, offline, lut_dir = Path(out), Path(offline), Path(lut_dir)
    if out.exists():
        raise ValueError("choose a new output directory")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    assets = {str(bound(offline / rel, sha)): sha for rel, sha in ASSETS.items()}
    assets[str(bound(lut_dir / "LookupTable_6000_0", LUT_SHA))] = LUT_SHA
    evidence = {str(p): sha256(p) for p in (Path(audit_dir) / "summary.json", Path(verified), Path(paired_path))}
    census_report = json.loads((Path(audit_dir) / "summary.json").read_text())
    bridge_source = bound(census_report["stock_source"]["path"], census_report["stock_source"]["sha256"])
    bridge = timing_source_bridge(bridge_source, offline / "CRVResponse/src/CrvPhotonGenerator_module.cc")
    assets[str(bridge_source)] = STOCK_SHA
    selections = load_selection(audit_dir, verified)
    paired = json.loads(Path(paired_path).read_text())
    checks = {row["input_sha256"]: row for row in paired["blocks"]}
    if len(checks) != len(paired["blocks"]) or any(s["record"]["sha256"] not in checks for s in selections):
        raise ValueError("missing or duplicate paired input")
    cfg, _, _, _ = fastframes.run_config("run1a_v01")
    cfg, contract = apply_policy(cfg, INTENDED)
    geom = geometry.from_offline_config(str(offline / "Mu2eG4/geom/crv_counters_run1a_v01.txt"))
    tabs = photons.load_tables(cfg, str(lut_dir))
    out.mkdir(parents=True)
    records = [block_probe(item, checks[item["record"]["sha256"]], photon_dev, geom, cfg, contract, tabs,
                           seed + i, out, i) for i, item in enumerate(selections)]
    for path, digest in (assets | evidence).items():
        bound(path, digest)
    code_paths = [Path(__file__), Path(photons.__file__), Path(geometry.__file__),
                  *[Path(__file__).with_name(n) for n in ("replay_timing.py", "step_timing_audit.py", "fastframes.py", "replay_policy.py")]]
    report = {"schema": "crv-optical-timing-probe-v1", "scope": SCOPE, "rng_contract": RNG_CONTRACT,
              "selection": "all frames with endpoint floor-fold/corrected-fmod residual in verified census",
              "arms": {"historical": "folded optics, no step rejection, no final wrap",
                       "arrival_only": "raw optics then corrected-fmod arrival wrap, no step rejection",
                       "rejection_only": "folded optics with stock step eligibility, no final wrap",
                       "stock_timing": "raw optics, corrected-fmod arrival wrap, stock step eligibility"},
              "response_contract": contract, "assets": assets, "timing_source_bridge": bridge, "blocks": records,
              "inputs": evidence,
              "code_sha256": {str(p): sha256(p) for p in code_paths},
              "F": None, "finite_library_uncertainty": None, "timing_closure": False}
    with (out / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "verified", "paired", "photon-dev", "offline", "lut-dir", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--seed", type=int, default=1027000)
    a = parser.parse_args(argv)
    result = run(a.audit, a.verified, a.paired, a.photon_dev, a.offline, a.lut_dir, a.out, a.seed)
    print(json.dumps([{k: b[k] for k in ("seed", "selected_events", "photons", "extreme_step_photons", "counts",
                         "extreme_step_counts", "max_raw_folded_circular_residual_ns", "elapsed_seconds")}
                      for b in result["blocks"]], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
