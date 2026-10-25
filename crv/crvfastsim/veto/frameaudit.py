"""Audit saved mixed-frame campaigns without rerunning response or changing the BO objective.

Run with --selection SEL.parquet --reference deadtime_stats.json --out report.json
followed by frames_mix_*.parquet. The reference supplies the frozen card and momentum
selection. Each input file is one job/subset block; repeated event IDs across jobs are
allowed, but duplicate IDs within a job and inconsistent sidecars are errors.

The block error measures observed job/subset variation conditional on the available
Cat library and selected-track sample. It is not a finite-library uncertainty estimate:
subsets can overlap, and historical sidecars do not identify their source tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np

from .apply import gate
from .card import VetoCard
from .deadtime import frame_coverage, read_coincs


def track_histogram(times, step):
    """Use the existing method-F grid, rejecting an empty or nonfinite selection."""
    times = np.asarray(times, dtype=float)
    if not len(times) or not np.isfinite(times).all():
        raise ValueError("selected track times must be nonempty and finite")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("grid spacing must be positive and finite")
    lo, hi = np.floor(times.min()) - step, np.ceil(times.max()) + step
    edges = np.arange(lo, hi + step, step) - step / 2
    weights = np.histogram(times, edges)[0].astype(float)
    return weights / weights.sum(), edges


def audit_frame(frames, metadata, card, weights, edges):
    """Validate a job's denominator and return coverage moments including empty frames."""
    n = len(frames)
    if not n:
        raise ValueError("empty frame file")
    ids = np.column_stack([ak.to_numpy(frames[k]) for k in ("run", "subrun", "event")])
    if np.unique(ids, axis=0).shape[0] != n:
        raise ValueError("duplicate (run, subrun, event) within a job")
    for field in ("events", "frames"):
        if metadata.get(field) != n:
            raise ValueError(f"sidecar {field}={metadata.get(field)} differs from {n} saved frames")
    co = frames.coincs
    counts = ak.to_numpy(ak.num(co.time))
    for field in ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType"):
        if not np.array_equal(ak.to_numpy(ak.num(co[field])), counts):
            raise ValueError(f"inconsistent coincidence lengths: {field}")
        if not np.isfinite(ak.to_numpy(ak.flatten(co[field]))).all():
            raise ValueError(f"nonfinite coincidence field: {field}")
    if metadata.get("coincs") != int(counts.sum()):
        raise ValueError("sidecar coincidence count differs from saved coincidences")
    if not np.isclose(metadata.get("frac_frames_with_coinc", np.nan), (counts > 0).mean(), rtol=0, atol=1e-12):
        raise ValueError("sidecar occupancy differs from saved frames")
    for name, pool in metadata.get("pools", {}).items():
        if pool["frames"] <= 0 or not 0 <= pool["stepped"] <= pool["frames"]:
            raise ValueError(f"invalid pool frame denominator: {name}")
    coverage = frame_coverage(co, card, weights, edges)
    fail = ~gate(co, card)
    empty = counts == 0
    if np.any(coverage[empty] != 0) or np.any(fail[empty]):
        raise ValueError("empty frames must have zero coverage and pass the gate")
    return {"frames": n, "empty_frames": int(empty.sum()), "coincs": int(counts.sum()),
            "frames_with_coinc": int((~empty).sum()), "gate_fail": int(fail.sum()),
            "coverage_sum": float(coverage.sum()), "coverage_sq_sum": float(coverage @ coverage),
            "frac": float(coverage.mean()), "gate_fail_frac": float(fail.mean()),
            "passing_gate_coverage_frac": float(coverage[~fail].sum() / n),
            "seed": metadata.get("seed"), "pools": metadata.get("pools"),
            "mean_intensity": metadata.get("mean_intensity"),
            "mean_frames_per_event": metadata.get("mean_frames_per_event")}


def aggregate(jobs):
    """Pooled IID error and independent-job sandwich error for a frame-weighted mean."""
    if not jobs:
        raise ValueError("no jobs")
    n = np.asarray([r["frames"] for r in jobs], dtype=float)
    sums = np.asarray([r["coverage_sum"] for r in jobs])
    total = float(n.sum())
    mean = float(sums.sum() / total)
    ss = sum(r["coverage_sq_sum"] for r in jobs)
    iid = float(np.sqrt(max(0.0, ss - total * mean**2) / (total * (total - 1)))) if total > 1 else None
    g = len(jobs)
    block = float(np.sqrt(g / (g - 1) * np.sum((sums - n * mean)**2)) / total) if g > 1 else None
    return {"jobs": g, "frames": int(total), "frac": mean, "err_iid_frames": iid,
            "err_job_blocks": block,
            "empty_frames": sum(r["empty_frames"] for r in jobs),
            "coincs": sum(r["coincs"] for r in jobs),
            "frames_with_coinc": sum(r["frames_with_coinc"] for r in jobs),
            "gate_fail": sum(r["gate_fail"] for r in jobs),
            "gate_fail_frac": sum(r["gate_fail"] for r in jobs) / total,
            "passing_gate_coverage_frac": mean - sum(r["gate_fail"] for r in jobs) / total}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--selection", nargs="+", required=True)
    ap.add_argument("--reference", required=True, help="saved method-F JSON supplying card, pmin/pmax and grid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = [Path(p).resolve() for p in a.frames]
    if len(set(paths)) != len(paths):
        ap.error("repeated frame file")
    ref = json.loads(Path(a.reference).read_text())
    card_dict = dict(ref["card"])
    recorded_sha = card_dict.pop("sha", None)
    card = VetoCard.from_dict(card_dict)
    if recorded_sha is not None and card.sha() != recorded_sha:
        raise ValueError("reference card hash does not match its contents")
    ev, _ = read_coincs(a.selection)
    times = ak.to_numpy(ak.flatten(ev.t_mid))
    momentum = ak.to_numpy(ak.flatten(ev.p_mid))
    sel = np.ones(len(times), dtype=bool)
    if ref.get("pmin") is not None:
        sel &= momentum > ref["pmin"]
    if ref.get("pmax") is not None:
        sel &= momentum < ref["pmax"]
    if int(sel.sum()) != ref["selected"]:
        raise ValueError("selected track count differs from reference")
    step = ref["F"]["grid_ns"]
    weights, edges = track_histogram(times[sel], step)
    rows = []
    for path in paths:
        start = time.perf_counter()
        meta_path = path.with_suffix(".json")
        row = audit_frame(ak.from_parquet(path), json.loads(meta_path.read_text()), card, weights, edges)
        row.update(path=str(path), sidecar_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                   wall_s=time.perf_counter() - start)
        rows.append(row)
        print(f"{path.name}: {row['frames']} frames, {row['empty_frames']} empty, F={row['frac']:.8f}, {row['wall_s']:.2f} s", flush=True)
    seeds = [r["seed"] for r in rows]
    if None in seeds or len(set(seeds)) != len(seeds):
        raise ValueError("missing or repeated job seed")
    pooled = aggregate(rows)
    report = {"card": card.__dict__ | {"sha": card.sha()}, "selected": int(sel.sum()), "grid_ns": step,
              "selection": a.selection, "reference": a.reference,
              "reference_sha256": hashlib.sha256(Path(a.reference).read_bytes()).hexdigest(),
              "pooled": pooled, "reference_F": ref["F"],
              "delta_from_reference_F": pooled["frac"] - ref["F"]["frac"], "jobs": rows,
              "uncertainty_scope": "Conditional on selected tracks and available Cat library. Job-block error assumes independent jobs; overlapping subsets and finite-library uncertainty are not resolved.",
              "source_table_identity": "Unavailable in historical sidecars; pool counts do not identify source tables."}
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(pooled, indent=2))


if __name__ == "__main__":
    main()
