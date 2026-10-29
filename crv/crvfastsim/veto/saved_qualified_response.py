"""Bounded all-live downstream validation using saved keyed photons, never optical regeneration."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import pandas as pd
import scipy
import uproot

from ..response import sipm as SI
from ..reco import waveform as WF
from . import bounded_response as R, batch_artifacts as A
from . import keyed_optical_smoke as M, retained_plan_report as PR, frame_registry as F
from .replay import sha256

SCHEMA = "saved-qualified-response-smoke-v1"
SEEDS = (3027000, 3027001)
THERMAL_RATE = 1e-4
ROOT_CAP = 2 * 1024**2
BATCH_CAP = 32 * 1024**2
MAX_PHOTONS_UNION = 600000
FROZEN_WAVEFORM_SHA = "09db394906de171e92ec3ad3bd4018990d1590066e5ef3b53cfabfbc646a61ba"
VARIANTS = ("full", "part0", "part1")
SCOPE = ("Four saved smoke frames and two optical/downstream seed pairs; diagnostic all-live thermal response. "
         "No optical redraw, stock response, coincidence, F, independent exposure or finite-library uncertainty.")


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def code_paths():
    return [Path(m.__file__) for m in (R, R.S, R.S.S, R.W, R.W.W, R.P, R.P.LM, A, F, R.Q, R.D, SI, WF, R.P.P, PR)] + [Path(__file__)]


def _table(directory, record):
    name = record["path"]
    if Path(name).name != name or not name.endswith(".parquet"):
        raise ValueError("unsafe saved table path")
    table = pd.read_parquet(M.P.bound(Path(directory) / name, record["sha256"]))
    if len(table) != record["rows"]:
        raise ValueError("saved table row count changed")
    return table


def admission(plan, smoke, expected_plan_sha256, expected_smoke_sha256):
    plan, smoke = Path(plan), Path(smoke)
    ppath, spath = M.P.bound(plan / "summary.json", expected_plan_sha256), M.P.bound(smoke / "summary.json", expected_smoke_sha256)
    p, s = json.loads(ppath.read_text()), json.loads(spath.read_text())
    if Path(p["smoke"]).resolve() != smoke.resolve() or p["inputs"].get(str(spath)) != expected_smoke_sha256:
        raise ValueError("plan and saved photons differ")
    if p["registry"]["frames"] != 9000 or s["seeds"] != list(SEEDS):
        raise ValueError("requires complete retained registry and frozen two-seed smoke")
    with tempfile.TemporaryDirectory(prefix="crv-response-admission-") as tmp:
        plan_verified = PR.verify(plan, Path(tmp) / "plan.json")
        smoke_verified = M.verify(smoke, Path(tmp) / "smoke.json")
    registry = _table(plan, p["artifacts"]["registry"])
    F.validate(registry)
    if F.digest(registry) != p["registry"]["registry_sha256"]:
        raise ValueError("wrong frozen registry")
    frames, calibration, photons, bindings = [], [], {seed: [] for seed in SEEDS}, {}
    for i, block in enumerate(s["blocks"]):
        digest = block["input_sha256"]
        events = _table(smoke, block["artifacts"]["events"]).assign(input_sha256=digest)
        encoded = F.encode(events, registry)
        selected = registry.iloc[encoded.event.to_numpy()].copy()
        frames.append(selected)
        stock = [path for path in block["input_bindings"] if path.endswith("/stock.root")]
        if len(stock) != 1:
            raise ValueError("missing unique fixed-stock calibration")
        M.P.bound(stock[0], block["input_bindings"][stock[0]])
        with uproot.open(stock[0]) as root:
            c = root["crvdump/calib"].arrays(library="pd").rename(columns={"channel": "chan"})
        columns = ["chan", "pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb"]
        calibration.append(c[columns].assign(input_sha256=digest))
        for record in block["artifacts"].values():
            bindings[str(smoke / record["path"])] = record["sha256"]
        for replica in block["replicas"]:
            seed = replica["seed"]
            record = replica["artifacts"]["photons"]
            if record["rows"] > R.Limits().photons:
                raise ValueError("saved photon cap exceeded before read")
            photons[seed].append(_table(smoke, record))
            for a in replica["artifacts"].values():
                bindings[str(smoke / a["path"])] = a["sha256"]
    selected = pd.concat(frames, ignore_index=True).sort_values("frame_id").reset_index(drop=True)
    R.selected_frames(registry, selected, F.digest(registry))
    if len(selected) != 4:
        raise ValueError("requires exactly the four saved smoke frames")
    c = pd.concat(calibration, ignore_index=True)
    q = {seed: pd.concat(parts, ignore_index=True) for seed, parts in photons.items()}
    if sum(len(table) for table in q.values()) > MAX_PHOTONS_UNION:
        raise ValueError("saved photon union cap exceeded")
    union = pd.concat([table[list(M.K.KEYS) + ["photon_id"]] for table in q.values()], ignore_index=True)
    np.testing.assert_array_equal(union.photon_id, M.roots(union))
    inputs = {str(ppath): expected_plan_sha256, str(spath): expected_smoke_sha256,
              **{str(plan / a["path"]): a["sha256"] for a in p["artifacts"].values()}, **bindings}
    for mapping in (p["inputs"], p["code_sha256"], s["inputs"], s["assets"], s["code_sha256"], *(b["input_bindings"] for b in s["blocks"])):
        M.check_bindings(inputs, mapping)
        inputs.update(mapping)
    return registry, selected, c, q, inputs, {"plan": plan_verified, "smoke": smoke_verified}


def _photon_subset(q, selected):
    return q[pd.MultiIndex.from_frame(q[list(F.FRAME_KEYS)]).isin(pd.MultiIndex.from_frame(selected[list(F.FRAME_KEYS)]))]


def structural_comparisons(left, right, registry):
    result = {}
    for name, (keys, fields) in R.D.STAGES.items():
        a, b = (F.encode(p[name].drop(columns="frame_id"), registry) for p in (left, right))
        result[name] = R.D.compare_rows(a, b, keys, fields)[0]
    return result


def run(plan, smoke, out, *, expected_plan_sha256, expected_smoke_sha256, photon_map, template):
    out = Path(out)
    if out.exists():
        raise ValueError("choose a new saved-response directory")
    registry, selected, calibration, photons, inputs, verified = admission(plan, smoke, expected_plan_sha256, expected_smoke_sha256)
    pmap_path = M.P.bound(photon_map, R.D.MAP_SHA)
    template = M.P.bound(template, R.D.TEMPLATE_SHA)
    with uproot.open(pmap_path) as root:
        pmap, x, y = root["photonMap"].to_numpy()
    if not np.array_equal(x, np.arange(41)) or not np.array_equal(y, np.arange(41)):
        raise ValueError("requires unit-width photon map")
    inputs.update({str(pmap_path): R.D.MAP_SHA, str(template): R.D.TEMPLATE_SHA})
    M.P.bound(R.W.W.__file__, FROZEN_WAVEFORM_SHA)
    code = {str(path): sha256(path) for path in code_paths()}
    M.check_bindings(inputs, code)
    digest = F.digest(registry)
    cfg, wcfg = replace(SI.SiPMConfig(), thermal_rate=THERMAL_RATE), replace(WF.WaveformConfig(), single_pe_file=str(template))
    context = {"registry_sha256": digest, "batch_identity": {"plan_sha256": expected_plan_sha256,
               "smoke_sha256": expected_smoke_sha256, "contract": R.CONTRACT}, "arms": list(M.P.ARMS), "seeds": list(SEEDS)}
    tags = [f"seed{seed}_{arm}" for seed in SEEDS for arm in M.P.ARMS]
    names = ["summary.json", "registry.parquet", "selected.parquet", "calibration.parquet", "admission.json"]
    names += [f"{tag}.json" for tag in tags] + [f"code{i}.txt" for i in range(len(code))]
    out.mkdir(parents=True)
    records, comparisons = [], {}
    with A.BatchArtifacts(out / "run", max_bytes=ROOT_CAP, context=context, expected_artifacts=names) as top:
        for name, table in (("registry", registry), ("selected", selected), ("calibration", calibration)):
            top.write_parquet(name + ".parquet", table)
        top.write_json("admission.json", verified)
        for i, path in enumerate(code):
            with top.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
        for seed in SEEDS:
            reference = None
            for arm in M.P.ARMS:
                tag = f"seed{seed}_{arm}"
                expected = [f"{v}_{s}.parquet" for v in VARIANTS for s in R.STAGES] + ["validation.json"]
                child_context = {**context, "batch_identity": {**context["batch_identity"], "frame_ids": selected.frame_id.tolist(),
                                 "variants": list(VARIANTS)}, "arms": [arm], "seeds": [seed]}
                start = time.perf_counter()
                with A.BatchArtifacts(out / tag, max_bytes=BATCH_CAP, context=child_context, expected_artifacts=expected) as tx:
                    products, receipts = {}, {}
                    domains = {"full": selected, "part0": selected.iloc[::2].iloc[::-1], "part1": selected.iloc[1::2].iloc[::-1]}
                    for variant, sub in domains.items():
                        q = _photon_subset(photons[seed], sub)
                        if variant != "full":
                            q = q.iloc[::-1]
                        products[variant], receipts[variant] = R.run_arm(q, registry, sub, calibration,
                            registry_digest=digest, optical_seed=seed, response_seed=seed, arm=arm,
                            pmap=pmap, sipm_cfg=cfg, waveform_cfg=wcfg)
                        for stage, table in products[variant].items():
                            tx.write_parquet(f"{variant}_{stage}.parquet", table)
                    partition = R.compare_partition(products["full"], [products["part0"], products["part1"]])
                    if partition["frame_ids"] != sorted(int(x) for x in selected.frame_id):
                        raise ValueError("partition does not cover every declared frame")
                    subset = R.compare_partition(products["full"], [products["part0"]])
                    tx.write_json("validation.json", {"partition": partition, "subset": subset, "receipts": receipts})
                    current = products["full"]
                    if arm == "historical":
                        reference = {s: current[s].copy() for s in R.D.STAGES}
                    else:
                        comparisons[f"seed{seed}_historical_vs_{arm}"] = structural_comparisons(reference, current, registry)
                top.write_json(tag + ".json", json.loads((out / tag / A.COMPLETE).read_text()))
                records.append({"path": tag, "seed": seed, "arm": arm, "receipt_sha256": sha256(out / tag / A.COMPLETE),
                                "elapsed_seconds": time.perf_counter() - start, "counts": receipts["full"],
                                "partition": partition, "subset": subset})
                del products
        M.check_bindings(inputs, code)
        summary = {"schema": SCHEMA, "scope": SCOPE, "registry_sha256": digest, "frames": len(selected),
                   "registry_frames": len(registry), "seeds": list(SEEDS), "arms": list(M.P.ARMS), "thermal_rate": THERMAL_RATE,
                   "sipm_config": asdict(cfg), "waveform_config": asdict(wcfg), "inputs": inputs, "code_sha256": code,
                   "records": records, "structural_comparisons": comparisons,
                   "byte_caps": {"run": ROOT_CAP, "each_arm_seed": BATCH_CAP, "maximum_total": ROOT_CAP + len(tags)*BATCH_CAP},
                   "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "uproot": uproot.__version__},
                   "all_tested_physical_partition_fields_exact": True, "execution_admitted": False,
                   "F": None, "finite_library_uncertainty": None, "timing_closure": False}
        top.write_json("summary.json", summary)
    return summary


def preflight(plan, smoke, out, *, expected_plan_sha256, expected_smoke_sha256, photon_map, template):
    """Repeat real saved-input admission and all-live census without downstream allocation."""
    out = Path(out)
    if out.exists():
        raise ValueError("choose a new preflight result")
    r, selected, c, q, inputs, verified = admission(plan, smoke, expected_plan_sha256, expected_smoke_sha256)
    for path, digest in ((photon_map, R.D.MAP_SHA), (template, R.D.TEMPLATE_SHA), (R.W.W.__file__, FROZEN_WAVEFORM_SHA)):
        M.P.bound(path, digest)
        inputs[str(path)] = digest
    cond = R.Q.conditions(r, selected.drop(columns="frame_id"), c, F.digest(r))
    for seed in SEEDS:
        R.Q.admit(q[seed], cond, r, F.digest(r), seed)
    result = {"schema": SCHEMA + "-admission", "registry_sha256": F.digest(r), "registry_frames": len(r),
              "frames": selected.to_dict(orient="records"), "calibration_rows": len(c), "live_channels": len(cond),
              "candidate_photons": {str(seed): len(q[seed]) for seed in SEEDS}, "inputs": inputs,
              "saved_verified": verified, "code_sha256": {str(p): sha256(p) for p in code_paths()},
              "execution_admitted": False, "response_executed": False}
    M.check_bindings(inputs, result["code_sha256"])
    M._json(out, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "smoke", "out", "expected-plan-sha256", "expected-smoke-sha256", "photon-map", "template"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--preflight", action="store_true")
    args = vars(p.parse_args(argv))
    action = preflight if args.pop("preflight") else run
    print(json.dumps(action(args.pop("plan"), args.pop("smoke"), args.pop("out"), **args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
