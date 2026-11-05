"""Census existing truth tables for a pre-calorimeter tracker filter witness.

This diagnoses a candidate restricted library population, NOT a certification of
VD coverage, original uncompressed filter equivalence or unbiased sampling.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot


EVENT = ["run", "subrun", "event"]
BOUNDARY_VDS = [73, 74, 75, 76, 77, 78, 79, 80, 102, 103, 104, 105, 106, 107, 108, 109]


def read_tree(source, name, columns):
    choices = [key for key, kind in source.classnames(cycle=False).items()
               if key.split("/")[-1] == name and kind == "TTree"]
    if len(choices) != 1:
        raise ValueError(f"expected one {name} tree; found {choices}")
    return source[choices[0]].arrays(columns, library="np")


def keys(table, fields):
    arrays = [np.asarray(table[field], dtype=np.int64) for field in fields]
    result = np.empty(len(arrays[0]), dtype=[(field, "i8") for field in fields])
    for field, values in zip(fields, arrays):
        result[field] = values
    return result


def audit(path):
    with uproot.open(path) as source:
        vd = read_tree(source, "vdsteps", EVENT + ["vdid", "sp", "t", "x", "y", "z", "px", "py", "pz"])
        sg = read_tree(source, "sgsteps", EVENT + ["sp", "t", "px", "py", "pz", "eion"])
        part = read_tree(source, "particles", EVENT + ["key", "pdg", "ismcprim", "et"])
    boundary = np.isin(vd["vdid"], BOUNDARY_VDS)
    events, inverse = np.unique(keys(vd, EVENT), return_inverse=True)
    first = np.full(len(events), np.inf)
    np.minimum.at(first, inverse[boundary], vd["t"][boundary])
    sg_event = keys(sg, EVENT)
    slot = np.searchsorted(events, sg_event)
    valid = slot < len(events)
    matched = np.zeros(len(slot), dtype=bool)
    matched[valid] = events[slot[valid]] == sg_event[valid]
    first_sg = np.full(len(slot), -np.inf)
    first_sg[matched] = first[slot[matched]]
    momentum = np.sqrt(sg["px"]**2 + sg["py"]**2 + sg["pz"]**2)
    good = (np.isfinite(first_sg) & (sg["t"] < first_sg) &
            (sg["eion"] > 1e-6) & (momentum > 50) & (momentum < 1e6))
    particle_keys = keys(sg, EVENT + ["sp"])
    unique, counts = np.unique(particle_keys[good], return_counts=True)
    witness = unique[counts >= 12]
    vd_keys = keys(vd, EVENT + ["sp"])
    vd_witness = np.isin(vd_keys, witness)
    part["sp"] = part["key"]
    prim_keys = keys(part, EVENT + ["sp"])[(part["pdg"] == 11) & (part["ismcprim"] != 0)]
    primary_entry = np.isin(vd_keys, prim_keys) & np.isin(vd["vdid"], [73, 75])
    ids, count = np.unique(vd["vdid"], return_counts=True)
    return {
        "source": str(path), "vd_rows": len(inverse), "tracker_rows": len(slot),
        "boundary_events": int(np.isfinite(first).sum()),
        "primary_electron_front_rows": int(primary_entry.sum()),
        "primary_front_rows_with_preboundary_witness": int((primary_entry & vd_witness).sum()),
        "events_with_witness": len(np.unique(witness[EVENT])),
        "vd_counts": {str(int(i)): int(n) for i, n in zip(ids, count)},
        "certified_unbiased": False,
        "warning": "StrawGasStep earliest time can hide later returns; witness is diagnostic only",
        "remaining_gates": ["full VD boundary coverage before any calo material",
                            "original filter thresholds/config and compression equivalence",
                            "first crossing per particle; re-entries and material histories",
                            "persistent product identity; key-only table is single-collection only",
                            "lossless shower steps including zero-deposit entries"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = [audit(path) for path in args.inputs]
    result = {"schema_version": 1, "files": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
