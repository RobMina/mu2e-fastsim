"""Compare source-algorithm hit grouping on stock reco digis, not saved link repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from check_hit_weights import flat, leaf, records
from calofastsim.hits import make_hits


def check(path, tree_name, expected_events, *, n_crystals, time4merge):
    with uproot.open(path) as source:
        tree = source[tree_name]
        if tree.num_entries != expected_events:
            raise ValueError(f"expected {expected_events} entries, got {tree.num_entries}")
        hits = records(tree, "calohits")
        reco = records(tree, "calorecodigis")
    count = ak.to_numpy(ak.num(leaf(reco, "time")))
    event = np.repeat(np.arange(expected_events), count)
    sipm = flat(leaf(reco, "SiPMID"))
    predicted = make_hits(event, sipm, flat(leaf(reco, "time")), flat(leaf(reco, "timeErr")),
                          flat(leaf(reco, "eDep")), flat(leaf(reco, "eDepErr")),
                          n_events=expected_events, n_crystals=n_crystals, time4merge=time4merge)
    nh = ak.to_numpy(ak.num(leaf(hits, "nSiPMs")))
    report = {"events": expected_events, "stock_hits": int(nh.sum()),
              "predicted_hits": len(predicted["nSiPMs"]), "reco_digis": len(event),
              "scope": "same stock reco digis to hit groups/weights; no digi or TemplateFit closure",
              "time4merge": time4merge, "n_crystals": n_crystals, "quantities": {}}
    layout = {"event_index": np.repeat(np.arange(expected_events), nh),
              "crystal": flat(leaf(hits, "crystalId")), "nSiPMs": flat(leaf(hits, "nSiPMs"))}
    report["layout_equal"] = all(np.array_equal(predicted[name], values) for name, values in layout.items())
    if not report["layout_equal"]:
        report["certified_scope"] = False
        return report
    for name, field in (("time", "time"), ("time_error", "timeErr"),
                        ("energy", "eDep"), ("energy_error", "eDepErr")):
        reference = flat(leaf(hits, field))
        report["quantities"][name] = {
            "float32_equal": bool(np.array_equal(predicted[name], reference)),
            "mismatches": int(np.count_nonzero(predicted[name] != reference)),
            "max_abs_difference": float(np.max(np.abs(predicted[name] - reference), initial=0)),
        }
    member = predicted["member_index"]
    group = np.repeat(np.arange(len(predicted["nSiPMs"])), predicted["nSiPMs"])
    times = flat(leaf(reco, "time"))[member]
    tied = (group[1:] == group[:-1]) & (times[1:] == times[:-1])
    nonidentical_fit = np.zeros(len(tied), dtype=bool)
    for field in ("timeErr", "eDep", "eDepErr"):
        values = flat(leaf(reco, field))[member]
        nonidentical_fit |= values[1:] != values[:-1]
    report["equal_time_hits"] = int(np.count_nonzero(predicted["equal_time_order_unspecified"]))
    report["nonidentical_equal_time_pairs"] = int(np.count_nonzero(tied & nonidentical_fit))
    report["certified_scope"] = (not report["nonidentical_equal_time_pairs"]
                                 and all(item["float32_equal"] for item in report["quantities"].values()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--tree", default="EventNtuple/ntuple")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--n-crystals", type=int, required=True)
    parser.add_argument("--time4merge", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.input, args.tree, args.expected_events,
                   n_crystals=args.n_crystals, time4merge=args.time4merge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["certified_scope"]:
        raise SystemExit("hit layout/values/order domain did not pass")


if __name__ == "__main__":
    main()
