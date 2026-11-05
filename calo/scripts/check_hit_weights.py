"""Check stock CaloHit weights on linked stock reco digis, not full hit finding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from calofastsim.response import weighted_reco_digis


def leaf(record, name):
    for candidate in (name, name + "_"):
        if candidate in ak.fields(record):
            return record[candidate]
    matches = [field for field in ak.fields(record)
               if field.rsplit(".", 1)[-1] in (name, name + "_")]
    if len(matches) == 1:
        return record[matches[0]]
    raise KeyError(f"missing or ambiguous {name}; fields={ak.fields(record)}")


def flat(array):
    return ak.to_numpy(ak.flatten(array, axis=None))


def records(tree, name):
    for candidate in (name + ".", name):
        try:
            branch = tree[candidate]
        except KeyError:
            continue
        return branch.array()
    raise KeyError(f"missing record branch {name}")


def check(path, tree_name, expected_events):
    with uproot.open(path) as source:
        tree = source[tree_name]
        if tree.num_entries != expected_events:
            raise ValueError(f"expected {expected_events} entries, got {tree.num_entries}")
        hits = records(tree, "calohits")
        reco = records(tree, "calorecodigis")
    n_hits = ak.to_numpy(ak.num(leaf(hits, "nSiPMs")))
    hit_offsets = np.r_[0, np.cumsum(n_hits)[:-1]]
    local_group = leaf(reco, "caloHitIdx")
    if bool(ak.any((local_group < -1) | (local_group >= ak.Array(n_hits)))):
        raise ValueError("reco-digi hit link outside its event")
    global_group = local_group + ak.Array(hit_offsets)
    keep = flat(local_group) >= 0
    group = flat(global_group)[keep].astype(np.int64)
    time = flat(leaf(reco, "time"))[keep]
    order = np.lexsort((time, group))
    same_group = group[order][1:] == group[order][:-1]
    if np.any(same_group & (time[order][1:] == time[order][:-1])):
        raise ValueError("equal-time members have no specified stock sort order")
    group = group[order]
    count = int(n_hits.sum())
    members = np.bincount(group, minlength=count)
    expected_members = flat(leaf(hits, "nSiPMs"))
    if not np.array_equal(members, expected_members):
        bad = np.flatnonzero(members != expected_members)
        raise ValueError("linked reco-digi multiplicities do not match CaloHit nSiPMs: "
                         f"bad_hits={len(bad)}/{count}, linked={int(keep.sum())}, "
                         f"expected={int(expected_members.sum())}, "
                         f"first_indices={bad[:8]}, observed={members[bad[:8]]}, "
                         f"expected_first={expected_members[bad[:8]]}")
    if count == 0:
        raise ValueError("no CaloHits: empty sample cannot certify weighting")
    report = {"events": expected_events, "hits": count,
              "reco_digis": len(keep), "linked_reco_digis": int(keep.sum()),
              "scope": "inverse-variance reduction with stock membership only",
              "quantities": {}}
    for quantity in ("time", "eDep"):
        values = flat(leaf(reco, quantity))[keep][order]
        errors = flat(leaf(reco, quantity + "Err"))[keep][order]
        pred, err = weighted_reco_digis(values, errors, group, count)
        ref = flat(leaf(hits, quantity))
        referr = flat(leaf(hits, quantity + "Err"))
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(referr)):
            raise ValueError(f"nonfinite stock {quantity}")
        report["quantities"][quantity] = {
            "max_abs_value_difference": float(np.max(np.abs(pred - ref))),
            "max_abs_error_difference": float(np.max(np.abs(err - referr))),
            "values_float32_equal": bool(np.array_equal(pred, ref)),
            "errors_float32_equal": bool(np.array_equal(err, referr)),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--tree", default="EventNtuple/ntuple")
    parser.add_argument("--expected-events", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.input, args.tree, args.expected_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(q["values_float32_equal"] and q["errors_float32_equal"]
               for q in result["quantities"].values()):
        raise SystemExit("stored float32 values differ from stock")


if __name__ == "__main__":
    main()
