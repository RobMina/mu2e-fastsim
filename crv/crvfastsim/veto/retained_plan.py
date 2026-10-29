"""Read-only whole-frame resource census and batch plan for a fixed retained replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from .. import geometry
from ..response import photons
from . import frame_registry as F, keyed_optical_smoke as M
from .replay import EVENT_TYPES, IDS, sha256

SCHEMA = "crv-retained-keyed-plan-v1"
DEFAULT_LIMITS = {"frames": 4, "steps": 20000, "simulated_substeps": 2000000, "live_pairs": 20480}
SEEDS = (3027000, 3027001)
MAX_TOTAL_SUBSTEPS = 10**12
MAX_LIMIT = 10**8


def workload(tree, events, ledger, geom, cfg, status):
    """Scan ROOT rows in bounded baskets; accumulate per-frame work without optical draws."""
    index = pd.MultiIndex.from_frame(events[list(IDS)])
    if index.has_duplicates or not len(index) or np.any(ledger.step_row.to_numpy() != np.arange(len(ledger))):
        raise ValueError("invalid full event/source-row census")
    n = len(events)
    steps = np.zeros(n, np.int64)
    finite_count, work = steps.copy(), steps.copy()
    first, last = np.full(n, len(ledger), np.int64), np.full(n, -1, np.int64)
    pairs, offset = [], 0
    ref = np.array([cfg.sectors[name].reflector for name in geom.sector_names])[geom.sector]
    names = [*IDS, "bar", "t0", "t1", "edep", "x0", "y0", "z0", "x1", "y1", "z1"]
    for arrays in tree.iterate(names, step_size="16 MB", library="np"):
        st = pd.DataFrame(arrays)
        size = len(st)
        expected = ledger.iloc[offset:offset + size]
        cols = [*IDS, "bar", "t0", "t1", "edep"]
        pd.testing.assert_frame_equal(st[cols].reset_index(drop=True), expected[cols].reset_index(drop=True), check_exact=True)
        er = index.get_indexer(pd.MultiIndex.from_frame(st[list(IDS)]))
        bar = st.bar.to_numpy()
        if np.any(er < 0) or bar.dtype.kind not in "iu" or np.any(bar < 0) or np.any(bar >= geom.nbars):
            raise ValueError("step outside event/geometry domain")
        t0, t1 = st.t0.to_numpy(), st.t1.to_numpy()
        if np.isinf(t0).any() or np.isinf(t1).any():
            raise ValueError("infinite raw endpoint")
        finite = np.isfinite(t0) & np.isfinite(t1)
        s, e = st[["x0", "y0", "z0"]].to_numpy(), st[["x1", "y1", "z1"]].to_numpy()
        if not np.isfinite(s[finite]).all() or not np.isfinite(e[finite]).all():
            raise ValueError("nonfinite active geometry")
        rows = np.arange(offset, offset + size)
        np.add.at(steps, er, 1)
        np.minimum.at(first, er, rows)
        np.maximum.at(last, er, rows)
        np.add.at(finite_count, er[finite], 1)
        b = bar[finite]
        l2w = geom.local_to_world[b]
        ri = np.arange(len(b))[:, None]
        start = (s[finite] - geom.center[b])[ri, l2w]
        end = (e[finite] - geom.center[b])[ri, l2w]
        sub = np.maximum(np.floor(np.linalg.norm(end - start, axis=1) / photons.PRECISION_MM), 1)
        if not np.isfinite(sub).all() or np.any(sub > 1e12) or sub.sum() > 2**52:
            raise ValueError("substep resource overflow")
        weighted = sub.astype(np.int64) * np.where(ref[b] == 0, 4, 2)
        if work.sum(dtype=np.float64) + weighted.sum(dtype=np.float64) > MAX_TOTAL_SUBSTEPS:
            raise ValueError("total substep census limit exceeded")
        np.add.at(work, er[finite], weighted)
        for sipm in range(4):
            allowed = (ref[b] == 0) | ((ref[b] < 0) if sipm % 2 else (ref[b] > 0))
            ch = b * 4 + sipm
            allowed &= (status[ch] & 5) == 0
            pairs.append(pd.DataFrame({"event_row": er[finite][allowed], "chan": ch[allowed]}).drop_duplicates())
        offset += size
    if offset != len(ledger) or offset != tree.num_entries:
        raise ValueError("raw/census row coverage disagreement")
    if not np.array_equal(steps, np.bincount(ledger.event_row, minlength=n)):
        raise ValueError("frame/source-row counts disagree")
    unique = pd.concat(pairs, ignore_index=True).drop_duplicates() if pairs else pd.DataFrame({"event_row": [], "chan": []})
    support = np.bincount(unique.event_row.to_numpy(dtype=int), minlength=n)
    out = events.copy()
    out["steps"], out["finite_steps"], out["simulated_substeps"] = steps, finite_count, work
    out["source_first"], out["source_stop"] = np.where(steps > 0, first, -1), np.where(steps > 0, last + 1, -1)
    out["contiguous"] = (steps == 0) | (last - first + 1 == steps)
    out["optical_support_pairs"] = support
    out["live_pairs"] = np.count_nonzero((status & 5) == 0)
    out["sipm_state_bytes"] = out.live_pairs * 41 * 41 * 8
    return out


def batch_plan(frames, limits=None):
    """Vectorized bisection of adjacent frame groups; never split a frame's physical history."""
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    if set(limits) != set(DEFAULT_LIMITS) or any(type(v) is not int or not 0 < v <= MAX_LIMIT for v in limits.values()):
        raise ValueError("invalid batch limits")
    if not 0 < len(frames) < 2**20 or frames.frame_id.duplicated().any():
        raise ValueError("empty or duplicate frame domain")
    ordered = frames.sort_values("frame_id").reset_index(drop=True)
    values = np.column_stack([np.ones(len(ordered), np.int64), ordered.steps, ordered.simulated_substeps, ordered.live_pairs])
    caps = np.array([limits[name] for name in DEFAULT_LIMITS], np.int64)
    if np.any(values < 0) or np.any(values > caps):
        bad = ordered.loc[np.any(values > caps, axis=1), [*F.FRAME_KEYS, "frame_id"]]
        raise ValueError(f"single frame exceeds proposed cap: {bad.head(10).to_dict(orient='records')}")
    cumulative = np.vstack([np.zeros(4, np.int64), np.cumsum(values, axis=0)])
    first = np.r_[0, np.flatnonzero(ordered.input_sha256.to_numpy()[1:] != ordered.input_sha256.to_numpy()[:-1]) + 1]
    last = np.r_[first[1:], len(ordered)]
    while True:
        total = cumulative[last] - cumulative[first]
        split = np.any(total > caps, axis=1)
        if not split.any():
            break
        middle = (first[split] + last[split]) // 2
        first = np.r_[first[~split], first[split], middle]
        last = np.r_[last[~split], middle, last[split]]
        order = np.argsort(first)
        first, last = first[order], last[order]
    total = cumulative[last] - cumulative[first]
    batches = pd.DataFrame(total, columns=list(DEFAULT_LIMITS))
    batches["batch"] = np.arange(len(first), dtype=np.int64)
    batches["first_frame_id"] = ordered.frame_id.to_numpy()[first]
    batches["last_frame_id"] = ordered.frame_id.to_numpy()[last - 1]
    batches["input_sha256"] = ordered.input_sha256.to_numpy()[first]
    assignment = ordered.copy()
    assignment["batch"] = np.repeat(np.arange(len(first)), last - first)
    starts = first
    batches["source_first"] = np.minimum.reduceat(np.where(assignment.source_first >= 0, assignment.source_first, np.iinfo(np.int64).max), starts)
    batches["source_stop"] = np.maximum.reduceat(assignment.source_stop.to_numpy(), starts)
    empty = batches.steps == 0
    batches.loc[empty, ["source_first", "source_stop"]] = -1
    batches["ranged_rows"] = np.where(empty, 0, batches.source_stop - batches.source_first)
    batches["range_has_no_extra_rows"] = batches.ranged_rows == batches.steps
    batches["sipm_state_bytes"] = batches.live_pairs * 41 * 41 * 8
    if assignment.frame_id.duplicated().any() or len(assignment) != len(frames):
        raise ValueError("batch coverage disagreement")
    return assignment, batches


def describe(frame, columns):
    return {name: {"sum": int(frame[name].sum()), "max": int(frame[name].max()),
                   "p50": float(frame[name].quantile(.5)), "p95": float(frame[name].quantile(.95)),
                   "p99": float(frame[name].quantile(.99))} for name in columns}


def run(smoke, smoke_verified, out, limits=None):
    smoke, smoke_verified, out = Path(smoke), Path(smoke_verified), Path(out)
    if out.exists():
        raise ValueError("choose a new plan directory")
    verification = json.loads(smoke_verified.read_text())
    if verification.get("saved_ledgers_exact") is not True or verification.get("root_union_collision_free") is not True:
        raise ValueError("requires successful keyed-smoke verification")
    smoke_summary = M.P.bound(smoke / "summary.json", verification["summary_sha256"])
    prior = json.loads(smoke_summary.read_text())
    if prior["schema"] != M.SCHEMA:
        raise ValueError("wrong smoke schema")
    out.mkdir(parents=True)
    M.verify(smoke, out / "smoke_reverified.json")
    inputs = {str(smoke_summary): sha256(smoke_summary), str(smoke_verified): sha256(smoke_verified)}
    census_path = Path(prior["paths"]["audit"]) / "summary.json"
    M.P.bound(census_path, prior["inputs"][str(census_path)])
    census = json.loads(census_path.read_text())
    counters = [path for path, h in prior["assets"].items() if h == M.P.COUNTERS_SHA]
    geom = geometry.from_offline_config(counters[0])
    cfg, _ = M.P.apply_policy(M.P.fastframes.run_config("run1a_v01")[0], M.P.INTENDED)
    collected, records = [], []
    for i, record in enumerate(census["inputs"]):
        started = time.perf_counter()
        path = M.P.bound(record["path"], record["sha256"])
        _, events, binding = M.P.read_input(path)
        if binding["sha256"] != record["sha256"] or binding["manifest_sha256"] != record["manifest_sha256"]:
            raise ValueError("raw input differs from census")
        artifact = record["artifacts"]["steps"]
        ledger_path = M.P.bound(census_path.parent / artifact["path"], artifact["sha256"])
        ledger = pd.read_parquet(ledger_path)
        block = [b for b in prior["blocks"] if b["input_sha256"] == record["sha256"]]
        if len(block) != 1:
            raise ValueError("missing admitted input")
        stock = [p for p in block[0]["input_bindings"] if p.endswith("/stock.root")]
        if len(stock) != 1:
            raise ValueError("ambiguous stock conditions")
        with uproot.open(stock[0]) as root:
            calib = root["crvdump/calib"].arrays(library="pd").sort_values("channel")
            M.P.validate_stock_events(events, root["crvdump/events"].arrays(library="np"))
        if not np.array_equal(calib.channel, np.arange(geom.nbars * 4)) or calib.status.dtype.kind not in "iu":
            raise ValueError("invalid full channel calibration")
        with uproot.open(path) as root:
            work = workload(root["dump/steps"], events, ledger, geom, cfg, calib.status.to_numpy())
        work.insert(0, "input_sha256", record["sha256"])
        collected.append(work)
        records.append({"input_sha256": record["sha256"], "path": str(path), "raw_bytes": path.stat().st_size,
                        "frames": len(work), "steps": len(ledger), "elapsed_seconds": time.perf_counter() - started})
        inputs.update({str(path): record["sha256"], str(path.with_suffix(".json")): record["manifest_sha256"],
                       str(ledger_path): artifact["sha256"]})
    frames = pd.concat(collected, ignore_index=True)
    registry = F.freeze(frames)
    frames = frames.merge(registry[[*F.FRAME_KEYS, "frame_id"]], on=list(F.FRAME_KEYS), validate="one_to_one")
    assignments, batches = batch_plan(frames, limits)
    artifacts = {name: M.save_table(out, name, table) for name, table in (("registry", registry), ("frames", assignments), ("batches", batches))}
    code = {str(p): sha256(p) for p in (Path(__file__), Path(F.__file__), Path(M.__file__), Path(geometry.__file__), Path(photons.__file__))}
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    report = {"schema": SCHEMA, "scope": "Read-only actual full-frame workload and proposed batches; no response or runtime extrapolation.",
              "registry": F.manifest(registry, SEEDS), "limits": limits, "inputs": inputs, "code_sha256": code,
              "smoke": str(smoke), "smoke_verified": str(smoke_verified), "input_records": records, "artifacts": artifacts,
              "frame_stats": describe(assignments, ["steps", "finite_steps", "simulated_substeps", "optical_support_pairs", "live_pairs"]),
              "batch_stats": describe(batches, list(DEFAULT_LIMITS) + ["sipm_state_bytes", "ranged_rows"]),
              "noncontiguous_frames": int((~assignments.contiguous).sum()),
              "batches_with_extra_range_rows": int((~batches.range_has_no_extra_rows).sum()),
              "thermal_domain": "all status-live calibration channels per frame; not optical union",
              "execution_admitted": False, "open_resource_gates": ["photon/cascade tail bounds at runtime", "waveform preallocation guard",
              "output-byte cap and atomic batch completion", "real-asset qualified downstream validation", "pulse-fit batch-composition numerical gate"],
              "runtime_projection": None, "F": None, "finite_library_uncertainty": None}
    report["admission_bindings"] = {"assets": prior["assets"], "inputs": prior["inputs"], "code": prior["code_sha256"],
                                    "blocks": [b["input_bindings"] for b in prior["blocks"]]}
    M.check_bindings(inputs, code, prior["assets"], prior["inputs"], prior["code_sha256"],
                     *(b["input_bindings"] for b in prior["blocks"]))
    M._json(out / "summary.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", required=True)
    p.add_argument("--smoke-verified", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    print(json.dumps(run(a.smoke, a.smoke_verified, a.out), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
