"""Inclusive first-foil raw tracker response accounting, NOT board/S2 closure.

Left-join exclusive, causally attributed tracker ionization to the full photon
frontier, including zero deposits. Unknown ancestry/timing is separate from the
definite non-interface residual. Any unresolved deposit marks every frontier
exposure in that event as potentially incomplete, not a known zero. This is a
geometry-dependent reference diagnostic, not a transferable fitted model. Raw
tracker steps have neither board aggregation nor EarlyFlash event selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .flash_g4_frontier import FIELDS as FRONTIER_FIELDS, FRONTIER, classify
from .flash_g4_photons import extract
from .flash_g4_timing import crossing_intervals, float32_interval
from .flash_inclusive_check import EVENT, PRODUCT, keys, join, matched

FIELDS = {**FRONTIER_FIELDS, "steps": (*FRONTIER_FIELDS["steps"], "total_edep", "nonionizing_edep"),
          "statuses": (*PRODUCT, "status", "overflow", "killed_step", "killed_field"),
          "completion": ("complete", "events", "subrun_products")}
RESIDUAL, OWNED, UNKNOWN = range(3)


def recording_checks(tables):
    """Per-event recorded-integrity checks, not a transport-completeness proof."""
    e, p, st, c = (tables[k] for k in ("events", "products", "statuses", "completion"))
    ek = keys(e, EVENT)
    pe = join(keys(p, EVENT), ek)
    ep = np.flatnonzero(p["scope"] == 0)
    ok = np.asarray(e["ptr_failures"]) == 0
    if np.any((p["scope"] == 0) & (pe < 0)):
        ok[:] = False
    for kind, instance, name in ((1, "", "particles"), (3, "photonSteps", "steps"),
                                  (3, "tracker", "steps"), (4, "", "statuses")):
        role = ((p["scope"] == 0) & (p["kind"] == kind) & (p["label"] == "g4run")
                & (p["instance"] == instance) & (pe >= 0))
        join(keys(p, EVENT)[role], keys(p, EVENT)[role])
        index = join(keys(tables[name], PRODUCT), keys(p, PRODUCT)[ep])
        index[index >= 0] = ep[index[index >= 0]]
        if np.any(index < 0):
            ok[:] = False
        rows = matched(index, role)
        count = np.bincount(index[rows], minlength=len(p["product"]))
        good = count[role] == p["size"][role]
        if kind == 1 or instance == "photonSteps":
            good &= p["size"][role] < 1_000_000
        present = np.zeros(len(ek), bool)
        present[pe[role]] = good
        ok &= present
        if kind == 4:
            se = join(keys(st, EVENT)[rows], ek)
            bad = ((st["status"][rows] != 0) | (st["overflow"][rows] != 0)
                   | (st["killed_step"][rows] != 0) | (st["killed_field"][rows] != 0))
            ok[se[bad]] = False
            status_count = np.bincount(se, minlength=len(ek))
            ok &= status_count == 1
    complete = (len(c["complete"]) == 1 and c["complete"][0] == 1
                and c["events"][0] == len(ek) and c["subrun_products"][0] > 0)
    if not complete:
        ok[:] = False
    return ok


def attribute(tables, photons, intervals, frontier, *, max_depth=128):
    """Return one owner or explicit residual per raw tracker step, with zeros.

    Proper descendants compare crossing bounds with the immediate child's birth
    interval; same-track deposits require their raw pre-time strictly after the
    crossing upper bound. No attempt is made to split a straddling deposit. The
    rootmost eligible crossing wins, while any uncertain competitor quarantines
    the step. Bounded ancestry traversal uses arrays, never per-event loops.
    """
    if isinstance(max_depth, bool) or not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")
    e, a, s, products = (tables[name] for name in ("events", "particles", "steps", "products"))
    ek, ak = keys(e, EVENT), keys(a, (*PRODUCT, "key"))
    pk = keys(photons, (*PRODUCT, "key"))
    join(ek, ek)
    join(ak, ak)
    join(pk, pk)
    ep = np.flatnonzero(products["scope"] == 0)
    si = join(keys(s, PRODUCT), keys(products, PRODUCT)[ep])
    si[si >= 0] = ep[si[si >= 0]]
    role = ((products["scope"] == 0) & (products["kind"] == 3)
            & (products["label"] == "g4run") & (products["instance"] == "tracker"))
    product_event = join(keys(products, EVENT), ek)
    if np.any(product_event[role] < 0):
        raise ValueError("tracker product missing event")
    join(keys(products, EVENT)[role], keys(products, EVENT)[role])
    sr = np.flatnonzero(matched(si, role))
    row_count = np.bincount(si[sr], minlength=len(products["product"]))
    observed = np.zeros(len(ek), bool)
    observed[product_event[role]] = row_count[role] == products["size"][role]
    particle = join(keys(s, (*EVENT, "particle_product", "particle_key"))[sr], ak)
    se = join(keys(s, EVENT)[sr], ek)
    pe = join(keys(photons, EVENT), ek)
    if np.any(se < 0) or np.any(pe < 0):
        raise ValueError("step or photon missing event")
    parent = join(keys(a, (*EVENT, "parent_product", "parent_key")), ak)
    candidate = join(ak, pk)
    lo, hi, valid = (np.asarray(intervals[k]) for k in ("lower_ns", "upper_ns", "valid"))
    good = (np.asarray(photons["exit"], bool) & valid & np.isfinite(lo) & np.isfinite(hi)
            & (hi >= lo) & photons["birth_inside_geometry"] & ~photons["ambiguous"]
            & (photons["recorded_invalid_segments"] == 0))
    birth_lo, birth_hi = float32_interval(a["time"])
    m = len(sr)
    owner = np.full(m, -1, np.int64)
    unknown = (particle < 0) | (s["particle_state"][sr] != 1) | ~np.isfinite(s["time"][sr])
    active = ~unknown
    rows = np.flatnonzero(active)
    k = candidate[particle[rows]]
    r, k = rows[k >= 0], k[k >= 0]
    after = good[k] & (s["time"][sr[r]] > hi[k])
    owner[r[after]] = k[after]
    unknown[r[~after]] = True
    child = particle.copy()
    for _ in range(max_depth):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        c = child[rows]
        ps, par = a["parent_state"][c], parent[c]
        root = ps == 0
        missing = ~root & ((ps != 1) | (par < 0))
        unknown[rows[missing]] = True
        active[rows[root | missing]] = False
        rows, c, par = rows[~(root | missing)], c[~(root | missing)], par[~(root | missing)]
        k = candidate[par]
        found = k >= 0
        r, c_for, k = rows[found], c[found], k[found]
        bvalid = np.isfinite(birth_lo[c_for]) & np.isfinite(birth_hi[c_for])
        eligible = good[k] & bvalid & (hi[k] < birth_lo[c_for])
        excluded = good[k] & bvalid & (lo[k] > birth_hi[c_for])
        unknown[r[~(eligible | excluded)]] = True
        owner[r[eligible]] = k[eligible]
        child[rows] = par
    unknown |= active
    report = json.loads(str(photons["report_json"]))
    exposure_keys = ("new_photons_without_volume", "photons_without_stage", "photons_after_destination_stage",
                     "particles_without_event", "steps_without_event", "particles_without_product",
                     "steps_without_product", "unresolved_photon_step_pointers")
    extraction_ok = all(name in report and report[name] == 0 for name in exposure_keys)
    if not extraction_ok:
        unknown[:] = True
    owned = owner >= 0
    unknown[owned] |= frontier["frontier_status"][owner[owned]] != FRONTIER
    status = np.where(unknown, UNKNOWN, np.where(owned, OWNED, RESIDUAL)).astype(np.int8)
    owner[status != OWNED] = -1
    total = s["total_edep"][sr].astype(np.float64)
    nonion = s["nonionizing_edep"][sr].astype(np.float64)
    edep = total - nonion
    if not np.all(np.isfinite(edep)) or np.any(edep < 0):
        raise ValueError("invalid raw tracker ionization; retain input and diagnose, do not drop steps")
    use = status == OWNED
    response = np.bincount(owner[use], weights=edep[use], minlength=len(pk))
    counts = np.bincount(owner[use], minlength=len(pk))
    event_unknown = np.bincount(se[status == UNKNOWN], minlength=len(ek)) > 0
    checked = recording_checks(tables)
    exposure_ok = np.full(len(ek), extraction_ok, bool)
    suspect = (photons["ambiguous"] | ~photons["birth_inside_geometry"]
               | (photons["recorded_invalid_segments"] > 0)
               | (photons["recorded_steps"] == 0))
    exposure_ok[pe[suspect]] = False
    known = ((frontier["frontier_status"] == FRONTIER) & observed[pe] & ~event_unknown[pe]
             & checked[pe] & exposure_ok[pe])
    event_energy = {name: np.bincount(se[status == code], weights=edep[status == code], minlength=len(ek))
                    for name, code in (("owned", OWNED), ("residual", RESIDUAL), ("unknown", UNKNOWN))}
    return dict(tracker_step_row=sr, tracker_owner_photon_row=owner, tracker_ownership_status=status,
                tracker_ionizing_mev=edep, photon_assigned_ionizing_mev=response,
                photon_assigned_tracker_steps=counts, photon_response_known=known,
                event_tracker_product_complete=observed, event_ownership_unknown=event_unknown,
                event_recording_checks_passed=checked, event_exposure_checks_passed=exposure_ok,
                event_tracker_ionizing_mev=np.bincount(se, weights=edep, minlength=len(ek)),
                **{"event_" + k + "_ionizing_mev": v for k, v in event_energy.items()})


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
    frontier = classify(tables, photons, intervals, max_depth=args.max_depth)
    result = attribute(tables, photons, intervals, frontier, max_depth=args.max_depth)
    summary = json.loads(str(photons["report_json"]))
    summary.update(scope="Inclusive first-foil raw response accounting; NOT board closure or transferable S2",
                   detector_response_valid_S2=False, input_events=int(photons["input_event_count"]),
                   expected_events=args.events, foil_births=len(photons["key"]),
                   frontier_status_counts=np.bincount(frontier["frontier_status"], minlength=4).tolist(),
                   tracker_status_counts=np.bincount(result["tracker_ownership_status"], minlength=3).tolist(),
                   tracker_status_codes={"residual": RESIDUAL, "owned": OWNED, "unknown": UNKNOWN},
                   frontier_known_responses=int(result["photon_response_known"].sum()),
                   known_zero_responses=int(np.count_nonzero(result["photon_response_known"] &
                                                            (result["photon_assigned_ionizing_mev"] == 0))),
                   incomplete_tracker_events=int(np.count_nonzero(~result["event_tracker_product_complete"])),
                   failed_recording_events=int(np.count_nonzero(~result["event_recording_checks_passed"])),
                   uncertain_exposure_events=int(np.count_nonzero(~result["event_exposure_checks_passed"])),
                   elapsed_seconds=perf_counter() - start)
    for key in ("tracker", "owned", "residual", "unknown"):
        summary[key + "_ionizing_mev"] = float(result["event_" + key + "_ionizing_mev"].sum())
    summary["max_event_accounting_error_mev"] = float(np.max(np.abs(result["event_tracker_ionizing_mev"] -
        result["event_owned_ionizing_mev"] - result["event_residual_ionizing_mev"] -
        result["event_unknown_ionizing_mev"]), initial=0.))
    result.update(photons)
    result.update(frontier)
    result.update({"crossing_" + k: v for k, v in intervals.items()})
    result["response_report_json"] = np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        np.savez_compressed(stream, **result)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{summary['input_events']}/{args.events} events; raw/owned/residual/unknown MeV="
          f"{[summary[k + '_ionizing_mev'] for k in ('tracker', 'owned', 'residual', 'unknown')]}; "
          f"{summary['elapsed_seconds']:.3f} s/file. NOT validated S2. {args.out}")
    return int(summary["input_events"] != args.events)


if __name__ == "__main__":
    raise SystemExit(main())
