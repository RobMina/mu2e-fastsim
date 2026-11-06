"""Inspect TS transport profiles and compare saved numerical-rewrite outputs."""
import argparse
import json
from pathlib import Path
import pstats

import numpy as np


def compare(reference, candidate):
    """Require the same rows and compare every saved quantity except elapsed cost."""
    with np.load(reference) as data:
        before = {key: data[key] for key in data.files}
    with np.load(candidate) as data:
        after = {key: data[key] for key in data.files}
    if before.keys() != after.keys():
        raise ValueError("output fields differ")
    for key in ("id", "shard"):
        if key in before and not np.array_equal(before[key], after[key]):
            raise ValueError(f"input alignment differs: {key}")
    fields = {}
    for key, a in before.items():
        if key == "seconds":
            continue
        b = after[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"output shape or dtype differs: {key}")
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        entry = {"exact": bool(equal.all()), "different_elements": int((~equal).sum())}
        if a.dtype.kind in "fi":
            finite = np.isfinite(a) & np.isfinite(b)
            entry["max_abs_difference"] = float(np.max(np.abs(a[finite] - b[finite]), initial=0))
        fields[key] = entry
    changed = (np.any(before['x'] != after['x'], axis=1) | (before['fate'] != after['fate']))
    rows = {key: before[key][changed].tolist() for key in ('id', 'pdg') if key in before}
    for label, arrays in (('before', before), ('after', after)):
        for key in ('x', 'fate'):
            rows[label + '_' + key] = arrays[key][changed].tolist()
    baseline_seconds, candidate_seconds = float(before["seconds"]), float(after["seconds"])
    return {"reference": str(reference), "candidate": str(candidate), "n": len(before["pdg"]),
            "all_exact": all(entry["exact"] for entry in fields.values()), "fields": fields,
            "position_or_fate_changed_rows": rows,
            "baseline_seconds": baseline_seconds, "candidate_seconds": candidate_seconds,
            "measured_ratio": baseline_seconds / candidate_seconds if candidate_seconds else None,
            "candidate_inputs_per_second": len(after["pdg"]) / candidate_seconds if candidate_seconds else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.profile:
        pstats.Stats(args.profile).strip_dirs().sort_stats("cumulative").print_stats(args.top)
    if args.reference or args.candidate:
        if not (args.reference and args.candidate):
            parser.error("--reference and --candidate are required together")
        result = compare(args.reference, args.candidate)
        text = json.dumps(result, indent=2, allow_nan=False)
        print(text)
        if args.out:
            Path(args.out).write_text(text + "\n")
    elif not args.profile:
        parser.error("supply --profile or a reference/candidate pair")


if __name__ == "__main__":
    main()
