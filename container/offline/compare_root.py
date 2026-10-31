"""Compare highest-cycle histograms and columnar trees, ignoring ROOT serialization metadata."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def compare(reference, candidate):
    """Return an exact content comparison and explicit unsupported-object failures."""
    report = {"reference": str(reference), "candidate": str(candidate), "objects": [], "equal": True}
    with uproot.open(reference) as left, uproot.open(candidate) as right:
        lc, rc = left.classnames(cycle=False), right.classnames(cycle=False)
        for key in sorted(set(lc) | set(rc)):
            item = {"key": key, "class": lc.get(key), "equal": True}
            if lc.get(key) != rc.get(key):
                item.update(equal=False, reason="missing object or different class", candidate_class=rc.get(key))
            elif lc[key].startswith("TDirectory"):
                continue
            elif lc[key] == "TNamed":
                a, b = left[key], right[key]
                item["equal"] = all(a.member(m) == b.member(m) for m in ("fName", "fTitle"))
                item["title"] = a.member("fTitle")
            elif lc[key].startswith(("TH1", "TH2", "TH3", "TProfile")):
                a, b = left[key], right[key]
                item["equal"] = bool(np.array_equal(a.values(flow=True), b.values(flow=True), equal_nan=True))
                item["equal"] &= bool(np.array_equal(a.variances(flow=True), b.variances(flow=True), equal_nan=True))
                item["equal"] &= all(np.array_equal(x.edges(), y.edges()) for x, y in zip(a.axes, b.axes))
                item["equal"] &= len(a.axes) == len(b.axes)
                members = ("fEntries", "fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2",
                           "fTsumwy", "fTsumwy2", "fTsumwxy", "fTsumwz", "fTsumwz2", "fTsumwxz", "fTsumwyz")
                for member in members:
                    if member in a.all_members or member in b.all_members:
                        item["equal"] &= a.member(member) == b.member(member)
                item["entries"] = float(a.member("fEntries"))
            elif lc[key] == "TTree" or "RNTuple" in lc[key]:
                a, b = left[key], right[key]
                item.update(reference_entries=a.num_entries, candidate_entries=b.num_entries)
                item["equal"] = a.num_entries == b.num_entries and set(a.keys()) == set(b.keys())
                item["different_branches"] = []
                for branch in sorted(set(a.keys()) & set(b.keys())):
                    if not ak.almost_equal(a[branch].array(), b[branch].array(), rtol=0, atol=0):
                        item["different_branches"].append(branch)
                        item["equal"] = False
            else:
                item.update(equal=False, reason="unsupported object: requires explicit review")
            item["equal"] = bool(item["equal"])
            report["equal"] &= item["equal"]
            report["objects"].append(item)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"equal": result["equal"], "objects": len(result["objects"]),
                      "different": [x["key"] for x in result["objects"] if not x["equal"]]}))
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
