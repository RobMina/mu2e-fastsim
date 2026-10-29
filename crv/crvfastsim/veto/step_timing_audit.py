"""Read-only all-frame step timing census; no photons, response, F estimate or promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import uproot

from .. import geometry
from .exposure import require_legacy_response
from .mixframes import PBTIMES
from .replay import EVENT_TYPES, IDS, STEP_TYPES, sha256
from .replay_timing import DIGI_END, DIGI_START, PERIOD, STEP_MARGIN, stock_step_mask, wrap_arrivals

STOCK_SHA = "1b5aef320f6f344d607fbf3c18e92b075b9d4daed84532d61ef78938dccd826e"
COUNTERS_SHA = "e70a819bb0fe1a3a4c7253b9b32e87dace8f68de27aa82f054d9c3cdfc11ec57"
FLAGS = ("finite", "nan_endpoint", "stock_keep", "reject_low", "reject_high", "reject_wrap",
         "historical_only_eligible", "start_outside_period", "end_outside_historical_period",
         "crosses_period", "negative_duration", "multiperiod_duration", "float32_origin_gate_flip",
         "fold_start_roundoff")
SCOPE = ("Step eligibility and numerical timing census only. Steps/energy are not photons, pulses or veto loss. "
         "No response simulation, optical RNG, selected-track weighting, finite-library uncertainty or timing closure. "
         "Source-qualified row identities are replay identities, not independent upstream showers.")


def _integer(frame, columns, maximum):
    for name in columns:
        value = frame[name].to_numpy()
        if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0) or np.any(value > maximum):
            raise ValueError(f"invalid integer identity: {name}")


def classify(steps, events, geom):
    """Vectorized full-tuple joins and mutually exclusive stock rejection reasons."""
    _integer(events, IDS, np.iinfo(np.uint32).max)
    _integer(steps, IDS, np.iinfo(np.uint32).max)
    _integer(steps, ("bar",), geom.nbars - 1)
    if not len(events) or events.duplicated(list(IDS)).any():
        raise ValueError("empty or duplicate event census")
    pb = events.pbtime.to_numpy(dtype=float)
    if not np.isfinite(pb).all() or not np.isin(pb, PBTIMES).all():
        raise ValueError("invalid bunch phase")
    index = pd.MultiIndex.from_frame(events[list(IDS)])
    event_row = index.get_indexer(pd.MultiIndex.from_frame(steps[list(IDS)]))
    if np.any(event_row < 0):
        raise ValueError("step absent from full-tuple event census")
    t0, t1 = steps.t0.to_numpy(dtype=float), steps.t1.to_numpy(dtype=float)
    edep = steps.edep.to_numpy(dtype=float)
    if np.isinf(t0).any() or np.isinf(t1).any() or not np.isfinite(edep).all() or np.any(edep < 0):
        raise ValueError("infinite endpoint or invalid deposited energy")
    origin = -pb.astype(np.float32).astype(float)
    w = origin[event_row]
    finite = np.isfinite(t0) & np.isfinite(t1)
    a = wrap_arrivals(np.where(finite, t0, w), w)
    b = wrap_arrivals(np.where(finite, t1, w), w)
    lo, hi = w + DIGI_START - STEP_MARGIN, w + DIGI_END
    low = finite & (a < lo) & (b < lo)
    high = finite & (a > hi) & (b > hi)
    wrap = finite & (a > hi) & (b < lo)
    keep = stock_step_mask(t0, t1, w)
    if not np.array_equal(keep, finite & ~(low | high | wrap)):
        raise ValueError("stock predicate/reason disagreement")
    with np.errstate(invalid="ignore", over="raise"):
        relative = t0 - w
        shift = np.floor(relative / PERIOD) * PERIOD
        folded0, folded1 = relative - shift, t1 - w - shift
        duration = t1 - t0
        crosses = np.floor(relative / PERIOD) != np.floor((t1 - w) / PERIOD)
    roundoff = folded0 + w - a
    flags = dict(finite=finite, nan_endpoint=~finite, stock_keep=keep,
                 reject_low=low, reject_high=high, reject_wrap=wrap,
                 historical_only_eligible=finite & ~keep,
                 start_outside_period=finite & ((relative < 0) | (relative >= PERIOD)),
                 end_outside_historical_period=finite & ((folded1 < 0) | (folded1 >= PERIOD)),
                 crosses_period=finite & crosses, negative_duration=finite & (duration < 0),
                 multiperiod_duration=finite & (np.abs(duration) >= PERIOD),
                 float32_origin_gate_flip=keep != stock_step_mask(t0, t1, -pb[event_row]),
                 fold_start_roundoff=finite & (roundoff != 0))
    ledger = steps[[*IDS, "bar", "t0", "t1", "edep"]].reset_index(drop=True).copy()
    ledger["step_row"] = np.arange(len(steps), dtype=np.int64)
    ledger["event_row"] = event_row
    ledger["sector"] = np.asarray(geom.sector_names)[geom.sector[steps.bar.to_numpy(dtype=np.int64)]]
    ledger["pbtime"] = pb[event_row]
    ledger["origin"] = w
    ledger["historical_t0_relative"], ledger["historical_t1_relative"] = folded0, folded1
    ledger["stock_t0_absolute"], ledger["stock_t1_absolute"] = np.where(finite, a, np.nan), np.where(finite, b, np.nan)
    ledger["fold_start_residual_ns"] = np.where(finite, roundoff, np.nan)
    for name, values in flags.items():
        ledger[name] = values
    frames = events.reset_index(drop=True).copy()
    frames["event_row"] = np.arange(len(events), dtype=np.int64)
    frames["steps"] = np.bincount(event_row, minlength=len(events))
    for name in FLAGS:
        frames[name] = np.bincount(event_row[flags[name]], minlength=len(events))
    return ledger, frames


def summarize(ledger, frames):
    """Counts and deposited energy only; neither is a response-weighted statistic."""
    return {"frames": len(frames), "steps": len(ledger), "empty_frames": int((frames.steps == 0).sum()),
            "counts": {k: int(ledger[k].sum()) for k in FLAGS},
            "frames_with": {k: int((frames[k] > 0).sum()) for k in FLAGS},
            "edep_MeV": {k: float(ledger.loc[ledger[k], "edep"].sum()) for k in FLAGS},
            "max_abs_fold_start_residual_ns": float(ledger.fold_start_residual_ns.abs().max())
            if ledger.finite.any() else None}


def sector_phase_summary(ledger, sector_names):
    """Include zero-count sector/phase combinations rather than selecting observed channels."""
    groups = []
    for sector in sector_names:
        for phase in PBTIMES:
            sub = ledger[(ledger.sector == sector) & (ledger.pbtime == phase)]
            groups.append({"sector": sector, "pbtime": float(phase), "steps": len(sub),
                           "counts": {k: int(sub[k].sum()) for k in FLAGS}})
    return groups


def aggregate(records):
    totals = {k: sum(r[k] for r in records) for k in ("frames", "steps", "empty_frames")}
    for key in ("counts", "frames_with", "edep_MeV"):
        totals[key] = {k: sum(r[key][k] for r in records) for k in FLAGS}
    return totals


def read_input(path):
    path = Path(path)
    sidecar = path.with_suffix(".json")
    sidecar_hash = sha256(sidecar)
    manifest = json.loads(sidecar.read_text())
    require_legacy_response(manifest)
    digest = sha256(path)
    if manifest.get("output_sha256") != digest or manifest.get("raw_times") is not True:
        raise ValueError("changed input or non-raw replay")
    with uproot.open(path) as root:
        counts = {n: root[f"dump/{n}"].num_entries for n in ("steps", "particles", "events")}
        steps = root["dump/steps"].arrays([*IDS, "bar", "t0", "t1", "edep"], library="pd")
        events = root["dump/events"].arrays(list(EVENT_TYPES), library="pd")
    if counts != manifest["entries"] or len(events) != manifest["events"]:
        raise ValueError("manifest/tree census disagreement")
    for table, types in ((steps, STEP_TYPES), (events, EVENT_TYPES)):
        if any(table[k].dtype != np.dtype(types[k]) for k in table):
            raise ValueError("replay branch dtype disagreement")
    if not np.isfinite(events.intensity).all() or np.any(events.intensity < 0):
        raise ValueError("invalid saved intensity")
    if (not np.all(events.run == manifest["run"]) or not np.all(events.subrun == manifest["subrun"])
            or not np.array_equal(events.event, np.arange(manifest["first_event"], manifest["first_event"] + len(events)))):
        raise ValueError("manifest event identity disagreement")
    return steps, events, {"path": str(path.resolve()), "sha256": digest, "manifest_sha256": sidecar_hash,
                           "entries": counts, "schema": manifest["schema"]}


def run(inputs, counters, stock_source, out):
    """One file at a time, exclusive outputs, persisted ledgers including every empty frame."""
    out = Path(out)
    if out.exists():
        raise ValueError("choose a new output directory")
    if not inputs or sha256(stock_source) != STOCK_SHA:
        raise ValueError("inputs and pinned stock photon-generator source required")
    counters_hash = sha256(counters)
    if counters_hash != COUNTERS_SHA:
        raise ValueError("requires pinned Run1A counters geometry")
    geom = geometry.from_offline_config(str(counters))
    if geom.sector_names != ["T1", "T2"]:
        raise ValueError("requires the Run1A T1/T2 geometry, without sector filtering")
    out.mkdir(parents=True)
    records, seen = [], set()
    for number, path in enumerate(inputs):
        started = time.perf_counter()
        steps, events, binding = read_input(path)
        if binding["sha256"] in seen:
            raise ValueError("repeated replay payload is not another frame sample")
        seen.add(binding["sha256"])
        ledger, frames = classify(steps, events, geom)
        groups = sector_phase_summary(ledger, geom.sector_names)
        if sha256(path) != binding["sha256"] or sha256(Path(path).with_suffix(".json")) != binding["manifest_sha256"]:
            raise ValueError("input changed during audit")
        artifacts = {}
        for name, frame in (("steps", ledger), ("frames", frames)):
            target = out / f"input{number}_{name}.parquet"
            frame.to_parquet(target, index=False)
            artifacts[name] = {"path": target.name, "sha256": sha256(target), "rows": len(frame)}
        records.append(binding | summarize(ledger, frames) | {"sector_phase": groups, "artifacts": artifacts,
                                                             "elapsed_seconds": time.perf_counter() - started})
        del steps, events, ledger, frames
    if sha256(counters) != counters_hash or sha256(stock_source) != STOCK_SHA:
        raise ValueError("geometry or stock source changed during audit")
    totals = aggregate(records)
    result = {"schema": "crv-step-timing-audit-v1", "inputs": records, "totals": totals,
              "counters": {"path": str(counters), "sha256": counters_hash, "sectors": geom.sector_names},
              "stock_source": {"path": str(stock_source), "sha256": STOCK_SHA},
              "timing": {"period_ns": PERIOD, "digi_start_ns": DIGI_START, "digi_end_ns": DIGI_END,
                         "step_margin_ns": STEP_MARGIN, "origin": "-float32(pbtime)",
                         "historical": "subtract origin; subtract floor((t0-origin)/period)*period from BOTH endpoints",
                         "stock": "corrected fmod separately per endpoint, then three rejection clauses; NaNs skip",
                         "identity": "input SHA256 + full event tuple + step_row (original zero-based ROOT row)"},
              "finite_library_uncertainty": None, "F": None, "timing_closure": False, "scope": SCOPE,
              "code_sha256": {n: sha256(Path(__file__).with_name(n)) for n in
                              ("step_timing_audit.py", "replay_timing.py", "fastframes.py", "replay.py", "exposure.py")}}
    with (out / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def verify_saved(directory, counters, stock_source, out):
    """Revalidate persisted ledgers and expose exceptional rows without rereading physics trees."""
    directory, out = Path(directory), Path(out)
    if out.exists():
        raise ValueError("choose a new verification output")
    if sha256(counters) != COUNTERS_SHA or sha256(stock_source) != STOCK_SHA:
        raise ValueError("requires pinned geometry and stock source")
    summary_path = directory / "summary.json"
    summary_hash = sha256(summary_path)
    report = json.loads(summary_path.read_text())
    if (report["schema"] != "crv-step-timing-audit-v1" or report["counters"]["sha256"] != COUNTERS_SHA
            or report["stock_source"]["sha256"] != STOCK_SHA or report["timing_closure"] is not False):
        raise ValueError("saved audit contract disagreement")
    geom = geometry.from_offline_config(str(counters))
    checks, seen = [], set()
    for number, item in enumerate(report["inputs"]):
        if item["sha256"] in seen:
            raise ValueError("repeated replay payload")
        seen.add(item["sha256"])
        for path, digest in ((Path(item["path"]), item["sha256"]),
                             (Path(item["path"]).with_suffix(".json"), item["manifest_sha256"])):
            if sha256(path) != digest:
                raise ValueError("saved input hash disagreement")
        tables = {}
        for name in ("steps", "frames"):
            artifact = item["artifacts"][name]
            if artifact["path"] != f"input{number}_{name}.parquet":
                raise ValueError("unexpected artifact path")
            path = directory / artifact["path"]
            if sha256(path) != artifact["sha256"]:
                raise ValueError("saved artifact hash disagreement")
            tables[name] = pd.read_parquet(path)
            if len(tables[name]) != artifact["rows"]:
                raise ValueError("saved artifact count disagreement")
        steps, frames = tables["steps"], tables["frames"]
        rebuilt, reframe = classify(steps, frames[list(EVENT_TYPES)], geom)
        pd.testing.assert_frame_equal(steps, rebuilt)
        pd.testing.assert_frame_equal(frames, reframe)
        stats = summarize(steps, frames)
        if (any(stats[k] != item[k] for k in stats)
                or sector_phase_summary(steps, geom.sector_names) != item["sector_phase"]):
            raise ValueError("saved summary disagreement")
        exceptions = steps[steps.fold_start_roundoff]
        columns = [*IDS, "step_row", "bar", "sector", "t0", "t1", "origin", "edep",
                   "historical_t0_relative", "historical_t1_relative", "stock_t0_absolute",
                   "stock_t1_absolute", "fold_start_residual_ns", "stock_keep"]
        checks.append({"input_sha256": item["sha256"], "frames": len(frames), "steps": len(steps),
                       "max_abs_fold_start_residual_ns": stats["max_abs_fold_start_residual_ns"],
                       "roundoff_rows_total": len(exceptions), "roundoff_rows_limit": 100,
                       "roundoff_rows": exceptions[columns].head(100).to_dict(orient="records"),
                       "no_stock_eligible_step_frames": frames.loc[frames.stock_keep == 0, list(IDS)].head(100).to_dict(orient="records")})
        del tables, steps, frames, rebuilt, reframe, exceptions
    if aggregate(report["inputs"]) != report["totals"]:
        raise ValueError("saved totals disagreement")
    if sha256(summary_path) != summary_hash:
        raise ValueError("saved summary changed during verification")
    result = {"summary_sha256": summary_hash, "checks": checks, "all_ledgers_reclassified_exactly": True,
              "pinned_geometry_verified": True, "verifier_sha256": sha256(__file__), "scope": SCOPE}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--verify", help="verify saved census ledgers instead of making a new census")
    parser.add_argument("--counters", required=True)
    parser.add_argument("--stock-source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.verify:
        if args.inputs:
            parser.error("--verify does not accept new inputs")
        result = verify_saved(args.verify, args.counters, args.stock_source, args.out)
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        result = run(args.inputs, args.counters, args.stock_source, args.out)
        print(json.dumps(result["totals"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
