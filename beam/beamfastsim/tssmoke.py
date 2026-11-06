"""Inspect unfiltered single-stage POT VD/Beam records; not a physics validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from .tsreference import _tree, art_inventory, ids, lookup, read_table, subset, summary

COMPACT_PDGS = (-211, -13, 13, 211, 2212)


def census(particles, vd, beam, *, compact=False):
    """Trace recorded states, marking genealogy gaps in species-filtered diagnostics."""
    pk = ids(particles)
    lookup(pk, pk)
    parent_keys = pk.copy()
    parent_keys[:, 3] = ids(particles, ("parent_id",))[:, 0]
    parent = lookup(pk, parent_keys)
    missing = (parent_keys[:, 3] > 0) & (parent < 0)
    if compact:
        if np.any(~np.isin(particles["pdg"], COMPACT_PDGS)):
            raise ValueError("compact ntpart contains an unselected species")
        parent_pdg = ids(particles, ("parent_pdg",))[:, 0]
        if np.any(missing & np.isin(parent_pdg, COMPACT_PDGS)):
            raise ValueError("ntpart lacks a selected positive parent key")
        retained = parent >= 0
        if np.any(parent_pdg[retained] != particles["pdg"][parent[retained]]):
            raise ValueError("parent/ntpart PDG disagreement")
    elif np.any(missing):
        raise ValueError("ntpart lacks a positive parent key")
    omitted_hits = []
    for hits in (vd, beam):
        hit_rows = lookup(pk, ids(hits))
        required = np.isin(hits["pdg"], COMPACT_PDGS) if compact else np.ones(len(hit_rows), bool)
        if np.any(required & (hit_rows < 0)):
            raise ValueError("recorded hit lacks ntpart identity")
        selected = np.isin(hits["pdg"], COMPACT_PDGS)
        if np.any(hits["pdg"][selected] != particles["pdg"][hit_rows[selected]]):
            raise ValueError("hit/ntpart PDG disagreement")
        omitted_hits.append(int(np.sum(hit_rows < 0)))
    mu_pi = np.isin(np.abs(vd["pdg"]), [13, 211])
    coll1 = vd["sid"] == 1
    forward = coll1 & mu_pi & (vd["pz"] > 0)
    source = subset(vd, forward)
    sk = ids(source)
    unique_keys, inverse, counts = np.unique(sk, axis=0, return_inverse=True, return_counts=True)
    order = np.lexsort((source["time"], inverse))
    first = order[np.r_[True, inverse[order][1:] != inverse[order][:-1]]] if len(order) else order
    source = subset(source, first)
    source_rows = lookup(pk, ids(source))
    source_at = np.full(len(pk), -1, dtype=np.int64)
    source_at[source_rows] = np.arange(len(source_rows))
    ends = subset(beam, np.isin(np.abs(beam["pdg"]), [13, 211]))
    end_rows = lookup(pk, ids(ends))
    current = end_rows.copy()
    root = np.full(len(current), -1, dtype=np.int64)
    incomplete = np.zeros(len(current), dtype=bool)
    depth = 0
    while np.any(current >= 0):
        if depth > len(pk):
            raise ValueError("cyclic particle ancestry")
        active = np.flatnonzero(current >= 0)
        found = source_at[current[active]]
        matched = active[found >= 0]
        root[matched] = found[found >= 0]
        current[matched] = -1
        unresolved = active[found < 0]
        incomplete[unresolved] |= missing[current[unresolved]]
        current[unresolved] = parent[current[unresolved]]
        depth += 1
    reached = np.zeros(len(source_rows), dtype=bool)
    reached[root[root >= 0]] = True
    nends = np.bincount(root[root >= 0], minlength=len(source_rows))
    primary = (particles["parent_id"] <= 0) & (particles["pdg"] == 2212)
    event_ids = np.unique(pk[:, :3], axis=0)
    source_events = np.unique(ids(source)[:, :3], axis=0)
    vd_ids, vd_counts = np.unique(vd["sid"].astype(np.int64), return_counts=True)
    late = np.zeros(len(root), dtype=bool)
    matched = root >= 0
    late[matched] = ends["time"][matched] < source["time"][root[matched]]
    return {
        "diagnostic_mode": "compact" if compact else "full",
        "omitted_parent_links": int(np.sum(missing)),
        "vd_rows_without_particle_identity": omitted_hits[0],
        "beam_rows_without_particle_identity": omitted_hits[1],
        "ds_mu_pi_ancestry_incomplete": int(np.sum(incomplete)),
        "particle_rows": len(pk), "events_in_ntpart": len(event_ids),
        "primary_proton_rows": int(np.sum(primary)),
        "primary_proton_events": len(np.unique(pk[primary, :3], axis=0)),
        "vd_rows": len(vd["trk"]), "vd_counts": dict(zip(map(str, vd_ids), map(int, vd_counts))),
        "coll1_mu_pi_all_direction_rows": int(np.sum(coll1 & mu_pi)),
        "coll1_mu_pi_forward_rows": int(np.sum(forward)),
        "coll1_mu_pi_forward_tracks": len(unique_keys),
        "tracks_with_repeated_forward_records": int(np.sum(counts > 1)),
        "events_with_forward_mu_pi": len(source_events),
        "events_without_forward_mu_pi_in_ntpart": len(event_ids) - len(source_events),
        "ds_beam_all_species_rows": len(beam["trk"]), "ds_mu_pi_rows": len(root),
        "ds_mu_pi_linked_to_forward_source": int(np.sum(matched)),
        "ds_mu_pi_without_forward_source": int(np.sum(~matched & ~incomplete)),
        "source_tracks_with_ds_mu_pi_descendant": int(np.sum(reached)),
        "source_tracks_without_linked_ds_mu_pi_descendant": int(np.sum(~reached)),
        "source_tracks_without_ds_mu_pi_descendant": None if np.any(incomplete) else int(np.sum(~reached)),
        "source_tracks_with_multiple_ds_descendants": int(np.sum(nends > 1)),
        "linked_ds_rows_preceding_source": int(np.sum(late)),
        "source_species": {str(pdg): {"inputs": int(np.sum(source["pdg"] == pdg)),
                           "with_ds_mu_pi": int(np.sum((source["pdg"] == pdg) & reached))}
                           for pdg in (-211, -13, 13, 211)},
        "source_states": summary(source), "ds_states": summary(ends),
        "ds_all_species_states": summary(beam),
        "limits": ["smoke census, not a transmission measurement or physics validation",
                   "VD1 pz>0; first time per track; repeated records counted, not silently discarded",
                   "DS genealogy uses nearest source ancestor; additional production/return branches need interpretation",
                   "unmatched DS particles may originate downstream or outside the charged source cohort",
                   "compact diagnostics omit nonselected parents; full ancestry remains in raw art",
                   "an unresolved DS ancestry makes the definitive source-loss count null"]}


def inspect(art, ntuple, expected_events=None, *, compact=False):
    _, totals, files = art_inventory([art])
    particles, pm = read_table(ntuple, "ntpart")
    vd, vm = read_table(ntuple, "ntvd")
    beam, bm = read_table(ntuple, "nttvd")
    report = census(particles, vd, beam, compact=compact)
    with uproot.open(art) as f:
        events, _ = _tree(f, "Events")
        branches = list(events.keys(recursive=False))
        report["status_branches"] = [k for k in events.keys() if "StatusG4" in k]
        subruns, _ = _tree(f, "SubRuns") if "SubRuns" in f else (None, None)
        report["subrun_branches"] = list(subruns.keys()) if subruns is not None else []
    report.update(art_files=files, finished_art_events=int(totals.sum()),
                  ntuple=str(ntuple), trees={"particles": pm, "vd": vm, "beam": bm},
                  art_event_branches=branches)
    checks = {"art_equals_ntuple_events": report["finished_art_events"] == report["events_in_ntpart"],
              "one_primary_proton_per_event": report["primary_proton_rows"] == report["finished_art_events"]
                  == report["primary_proton_events"],
              "no_linked_endpoint_before_source": report["linked_ds_rows_preceding_source"] == 0}
    if expected_events is not None:
        checks["requested_event_count"] = report["finished_art_events"] == expected_events
    report["checks"] = checks
    report["checks_passed"] = all(checks.values())
    return report


def compare_diagnostics(baseline, compact):
    """Require identical retained scalar records and all Beam rows for a same-seed rerun."""
    result = {}
    for name in ("ntpart", "ntvd", "nttvd"):
        before, _ = read_table(baseline, name)
        after, _ = read_table(compact, name)
        original_rows = len(before["trk"])
        if name != "nttvd":
            before = subset(before, np.isin(before["pdg"], COMPACT_PDGS))
        changed = [k for k in before.keys() & after.keys()
                   if not np.array_equal(before[k], after[k], equal_nan=True)]
        result[name] = {"baseline_rows": original_rows, "expected_retained_rows": len(before["trk"]),
                        "compact_rows": len(after["trk"]), "changed_scalar_columns": sorted(changed),
                        "scalar_columns_equal": before.keys() == after.keys(),
                        "exact_selected_scalar_match": before.keys() == after.keys() and not changed}
    return {"baseline": str(baseline), "compact": str(compact), "trees": result,
            "checks_passed": all(t["exact_selected_scalar_match"] for t in result.values()),
            "limits": "compares diagnostic scalar records, not raw art serialization or jagged ntpart arrays"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--art", required=True)
    parser.add_argument("--ntuple", required=True)
    parser.add_argument("--expected-events", type=int)
    parser.add_argument("--compact", action="store_true", help="mu/pi/proton diagnostic with incomplete ancestry")
    parser.add_argument("--baseline-ntuple", help="same-seed full diagnostic to compare with compact output")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.baseline_ntuple and not args.compact:
        parser.error("--baseline-ntuple requires --compact")
    report = inspect(args.art, args.ntuple, args.expected_events, compact=args.compact)
    if args.baseline_ntuple:
        report["baseline_comparison"] = compare_diagnostics(args.baseline_ntuple, args.ntuple)
        report["checks"]["same_seed_selected_records"] = report["baseline_comparison"]["checks_passed"]
        report["checks_passed"] = all(report["checks"].values())
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("trees", "art_event_branches")},
                     indent=2, allow_nan=False))
    raise SystemExit(0 if report["checks_passed"] else 1)


if __name__ == "__main__":
    main()
