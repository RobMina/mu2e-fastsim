"""Compare two replay partitions by raw identities and algorithmic bucket/generation, not fit matching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calofastsim.ancestry import SIM_FIELDS
from calofastsim.identities import EVENT_FIELDS, keys


GROUP_FIELDS = SIM_FIELDS + ("crystal", "bucket", "generation")
RAW_FIELDS = EVENT_FIELDS + ("product_id", "index")


def partition(arrays):
    group_columns = {name: arrays[f"predicted_{name}"] if f"predicted_{name}" in arrays else arrays[name] for name in GROUP_FIELDS}
    group_ids = keys(group_columns, GROUP_FIELDS)
    if len(np.unique(group_ids)) != len(group_ids):
        raise ValueError("duplicate algorithmic compression group")
    raw_ids = keys({name: arrays[f"raw_{name}"] for name in RAW_FIELDS}, RAW_FIELDS)
    offsets = np.asarray(arrays["member_offsets"])
    if (offsets.dtype.kind not in "iu" or offsets.ndim != 1 or offsets.size != len(group_ids) + 1
            or offsets[0] != 0 or offsets[-1] != len(raw_ids) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("invalid membership offsets")
    if len(np.unique(raw_ids)) != len(raw_ids):
        raise ValueError("raw identity occurs in more than one constituent slot")
    counts = np.diff(offsets).astype(np.int64)
    if not np.array_equal(counts, arrays["predicted_n_compress"]):
        raise ValueError("constituent count mismatch")
    member_group = np.repeat(np.arange(len(group_ids)), counts)
    order = np.argsort(raw_ids)
    return raw_ids[order], group_ids[member_group[order]], group_ids


def compare(left, right):
    pleft, pright = (json.loads(str(arrays["provenance_json"])) for arrays in (left, right))
    for name in ("input_sha256", "expanded_fcl_sha256", "configuration", "crystal_length_mm"):
        if pleft.get(name) != pright.get(name) or name not in pleft:
            raise ValueError("cannot compare different replay source/configuration")
    raw_left, label_left, group_left = partition(left)
    raw_right, label_right, group_right = partition(right)
    same_raw = np.array_equal(raw_left, raw_right)
    same_group = np.array_equal(np.sort(group_left), np.sort(group_right))
    different = int(np.count_nonzero(label_left != label_right)) if same_raw else None
    return {"raw_identity_population_equal": same_raw, "algorithmic_group_population_equal": same_group,
            "raw_constituent_group_assignment_equal": bool(same_raw and different == 0),
            "different_raw_assignments": different, "raw_rows": len(raw_left), "groups": len(group_left),
            "group_output_order_equal": bool(np.array_equal(group_left, group_right)),
            "left_equal_time_order": pleft.get("equal_time_order", "stable"),
            "right_equal_time_order": pright.get("equal_time_order", "stable"),
            "stock_constituent_identity_certified": False,
            "scope": "two explicit replay alternatives only; equality is not independent observation of C++ constituent links"}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("refuse to overwrite membership comparison")
    with np.load(args.left, allow_pickle=False) as left, np.load(args.right, allow_pickle=False) as right:
        result = compare(left, right)
    result["left_sha256"], result["right_sha256"] = digest(args.left), digest(args.right)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["raw_constituent_group_assignment_equal"]:
        raise SystemExit("replay partitions differ")


if __name__ == "__main__":
    main()
