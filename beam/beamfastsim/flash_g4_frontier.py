"""Conservative causal frontier of first birth-foil photon exits, not S2.

All foil births and the full input-event ledger remain in the output. An exiting
photon is covered by an ancestor only if that ancestor exited BEFORE the birth of
its immediate child on this ancestry branch. Using the final descendant's birth
would incorrectly merge branches emitted before an ancestor crossed. Time bounds,
not chord/c times, determine strict order. Overlap, missing ancestry, non-exiting
foil ancestors, and unresolved spatial evidence remain unknown. This deliberately
conservative diagnostic never establishes zero detector response or full exposure
completeness. No tracker deposits are assigned here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .flash_inclusive_check import EVENT, PRODUCT, keys, join
from .flash_g4_photons import FIELDS as PHOTON_FIELDS, extract
from .flash_g4_timing import crossing_intervals, float32_interval

NO_EXIT, FRONTIER, COVERED, UNRESOLVED = range(4)
FIELDS = {**PHOTON_FIELDS, "particles": (*PHOTON_FIELDS["particles"], "time", "end_time", "end_defined")}


def classify(tables, photons, intervals, *, max_depth=128):
    """Classify all birth rows with vectorized, bounded ancestry traversal.

    Status 0=no recorded exit, 1=resolved frontier, 2=covered by a resolved
    ancestor, 3=unresolved. ``covering_photon_row`` indexes this same birth table
    and is -1 unless status=2. Any unknown competitor conservatively quarantines
    the row, including when another ancestor is eligible; it is not silently
    excluded from an exposure denominator. Depth exhaustion includes cycles.
    """
    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    a = tables["particles"]
    ak = keys(a, (*PRODUCT, "key"))
    pk = keys(photons, (*PRODUCT, "key"))
    join(ak, ak)
    join(pk, pk)
    particle = join(pk, ak)
    if np.any(particle < 0):
        raise ValueError("foil photon missing from particles")
    if np.any(a["pdg"][particle] != 22):
        raise ValueError("foil photon identity is not a photon")
    parent = join(keys(a, (*EVENT, "parent_product", "parent_key")), ak)
    candidate = join(ak, pk)
    birth_lo, birth_hi = float32_interval(a["time"])
    n = len(pk)
    ex = np.asarray(photons["exit"], bool)
    lo, hi, valid = (np.asarray(intervals[k]) for k in ("lower_ns", "upper_ns", "valid"))
    if any(x.shape != (n,) for x in (lo, hi, valid)):
        raise ValueError("crossing interval shapes must match photon rows")
    valid = valid.astype(bool) & np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    spatial = (np.asarray(photons["birth_inside_geometry"], bool)
               & ~np.asarray(photons["ambiguous"], bool)
               & (np.asarray(photons["recorded_invalid_segments"]) == 0))
    own_invalid = ex & (~valid | ~spatial)
    unknown = own_invalid.copy()
    missing_parent = np.zeros(n, bool)
    uncertain_ancestor = np.zeros(n, bool)
    time_overlap = np.zeros(n, bool)
    depth_exhausted = np.zeros(n, bool)
    covering = np.full(n, -1, np.int64)
    child = particle.copy()
    active = ex.copy()
    depth = np.zeros(n, np.int64)
    for _ in range(max_depth):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        c = child[rows]
        state = a["parent_state"][c]
        par = parent[c]
        root = state == 0
        missing = ~root & ((state != 1) | (par < 0))
        missing_parent[rows[missing]] = True
        unknown[rows[missing]] = True
        active[rows[root | missing]] = False
        rows, c, par = rows[~(root | missing)], c[~(root | missing)], par[~(root | missing)]
        depth[rows] += 1
        k = candidate[par]
        found = k >= 0
        r, c_for, k = rows[found], c[found], k[found]
        usable = ex[k] & valid[k] & spatial[k]
        bvalid = np.isfinite(birth_lo[c_for]) & np.isfinite(birth_hi[c_for])
        eligible = usable & bvalid & (hi[k] < birth_lo[c_for])
        excluded = usable & bvalid & (lo[k] > birth_hi[c_for])
        ambiguous = ~(eligible | excluded)
        unknown[r[ambiguous]] = True
        uncertain_ancestor[r[~usable]] = True
        time_overlap[r[usable & ambiguous]] = True
        covering[r[eligible]] = k[eligible]
        child[rows] = par
    depth_exhausted[active] = True
    unknown |= depth_exhausted
    status = np.full(n, NO_EXIT, np.int8)
    status[ex & ~unknown & (covering < 0)] = FRONTIER
    status[ex & ~unknown & (covering >= 0)] = COVERED
    status[ex & unknown] = UNRESOLVED
    covering[status != COVERED] = -1
    return dict(frontier_status=status, covering_photon_row=covering, ancestry_depth=depth,
                frontier_own_invalid=own_invalid, frontier_missing_parent=missing_parent,
                frontier_uncertain_ancestor=uncertain_ancestor, frontier_time_overlap=time_overlap,
                frontier_depth_exhausted=depth_exhausted)


def read_tables(path):
    import awkward as ak
    import uproot

    with uproot.open(path) as root:
        return {name: {field: ak.to_numpy(arrays[field]) for field in fields}
                for name, fields in FIELDS.items()
                for arrays in [root["export/" + name].arrays(list(fields), library="ak")]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--geom", required=True)
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--events", required=True, type=int)
    parser.add_argument("--boundary-tolerance-mm", type=float, default=0.)
    parser.add_argument("--max-depth", type=int, default=128)
    args = parser.parse_args(argv)
    if args.events < 0:
        parser.error("--events must be nonnegative")
    from .flash_geometry import from_offline

    start = perf_counter()
    tables = read_tables(args.input)
    photons = extract(tables, from_offline(args.geom, args.offline_root),
                      boundary_tolerance_mm=args.boundary_tolerance_mm)
    intervals = crossing_intervals(tables, photons)
    result = classify(tables, photons, intervals, max_depth=args.max_depth)
    status = result["frontier_status"]
    ek = keys(tables["events"], EVENT)
    event = join(keys(photons, EVENT), ek)
    if np.any(event < 0):
        raise ValueError("photon missing from input event ledger")
    summary = json.loads(str(photons["report_json"]))
    summary.update(scope="Conservative first-foil causal frontier only; NOT S2 or full exposure certification",
                   detector_response_valid_S2=False, input_events=len(ek), expected_events=args.events,
                   foil_births=len(status), spatial_exits=int(photons["exit"].sum()),
                   time_brackets_valid=int(intervals["valid"].sum()),
                   status_codes={"no_exit": NO_EXIT, "frontier": FRONTIER, "covered": COVERED, "unresolved": UNRESOLVED},
                   status_counts=np.bincount(status, minlength=4).tolist(), max_depth=args.max_depth)
    for name in ("own_invalid", "missing_parent", "uncertain_ancestor", "time_overlap", "depth_exhausted"):
        summary[name] = int(result["frontier_" + name].sum())
    for name, code in (("frontier", FRONTIER), ("covered", COVERED), ("unresolved", UNRESOLVED)):
        result["event_" + name] = np.bincount(event[status == code], minlength=len(ek))
    summary["elapsed_seconds"] = perf_counter() - start
    summary["input"] = str(args.input)
    result.update(photons)
    result.update({"crossing_" + k: v for k, v in intervals.items()})
    result["frontier_report_json"] = np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        np.savez_compressed(stream, **result)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{len(ek)}/{args.events} events; frontier/covered/unresolved="
          f"{summary['status_counts'][1:]}; {summary['elapsed_seconds']:.3f} s/file. NOT S2. {args.out}")
    return int(len(ek) != args.events)


if __name__ == "__main__":
    raise SystemExit(main())
