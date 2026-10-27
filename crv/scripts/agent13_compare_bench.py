"""Compare completed same-backend benchmark arms without treating throughput as a physics certificate."""
import argparse
import json
from pathlib import Path

import numpy as np


def array_differences(left, right):
    result = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for name in sorted(set(a.files) | set(b.files)):
            if name not in a or name not in b:
                result[name] = {"missing_column": True}
                continue
            x, y = a[name], b[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                result[name] = {"shape_dtype_mismatch": [str(x.shape), str(y.shape), str(x.dtype), str(y.dtype)]}
            elif x.tobytes() != y.tobytes():
                mismatch = x != y
                result[name] = {"different_values": int(np.count_nonzero(mismatch)), "size": x.size}
                if x.dtype.kind in "fiu":
                    delta = np.abs(x.astype(float) - y.astype(float))
                    result[name]["max_abs_diff"] = float(np.max(delta))
    return result


DERIVED_ATOL = {"cl_pe": 1e-8, "cl_avgtime": 1e-8, "cl_pos": 1e-8, "cl_slope": 1e-8}


def rounding_only(differences):
    return all(name.split("__")[-1] in DERIVED_ATOL and "max_abs_diff" in detail
               and np.isfinite(detail["max_abs_diff"])
               and detail["max_abs_diff"] <= DERIVED_ATOL[name.split("__")[-1]]
               for diff in differences for name, detail in diff.items())


def compare(left, right):
    a, b = (json.loads((Path(p) / "benchmark.json").read_text()) for p in (left, right))
    if any(x.get("status") != "complete" or not x.get("repeat_fingerprints_identical") for x in (a, b)):
        raise ValueError("both arms need completed, repeat-identical benchmarks")
    for field in ("boundary", "arguments", "device"):
        if a[field] != b[field]:
            raise ValueError(f"incompatible {field}")
    for field in ("input_sha256", "assets_sha256"):
        if a["provenance"][field] != b["provenance"][field]:
            raise ValueError(f"incompatible {field}")
    result = {"same_backend_regression": True, "exact_membership_bounded_properties": True,
              "derived_property_atol": DERIVED_ATOL, "stock_physics_validation": "not evaluated", "passes": []}
    if len(a["passes"]) != len(b["passes"]):
        raise ValueError("pass count differs")
    for before, after in zip(a["passes"], b["passes"]):
        if len(before["files"]) != len(after["files"]):
            raise ValueError("input output count differs")
        identical = [f["fingerprint"] for f in before["files"]] == [f["fingerprint"] for f in after["files"]]
        result["same_backend_regression"] &= identical
        differences = []
        if not identical:
            for fa, fb in zip(before["files"], after["files"]):
                pa = Path(left) / f"pass_{before['index']:03d}" / fa["npz"]
                pb = Path(right) / f"pass_{after['index']:03d}" / fb["npz"]
                differences.append(array_differences(pa, pb))
        bounded = identical or rounding_only(differences)
        result["exact_membership_bounded_properties"] &= bounded
        result["passes"].append({"stage": before["stage"], "identical": identical,
                                 "exact_membership_bounded_properties": bounded, "array_differences": differences,
                                 "baseline_elapsed_s": before["elapsed_s"], "candidate_elapsed_s": after["elapsed_s"],
                                 "speedup": before["elapsed_s"] / after["elapsed_s"]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-derived-rounding", action="store_true")
    args = parser.parse_args(argv)
    result = compare(args.baseline, args.candidate)
    with open(args.out, "x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    gate = "exact_membership_bounded_properties" if args.allow_derived_rounding else "same_backend_regression"
    if not result[gate]:
        parser.exit(1, f"comparison gate failed: {gate}\n")


if __name__ == "__main__":
    main()
