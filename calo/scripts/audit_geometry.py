"""Audit exported geometry frames and dimensions, not material-boundary completeness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from calofastsim.identities import keys


AXES = ("x", "y", "z")


def vector(table, prefix):
    return np.column_stack([table[f"{prefix}_{axis}"] for axis in AXES])


def rotations(table):
    return np.stack([vector(table, f"axis_{axis}") for axis in AXES], axis=2)


def audit(crystals, disks, virtualdetectors, *, expected_crystals, expected_disks, tolerance=1e-9):
    if len(crystals["crystal"]) != expected_crystals or len(disks["disk"]) != expected_disks:
        raise ValueError("geometry counts do not match explicit configuration")
    disk_keys = keys(disks, ("run", "disk"))
    crystal_keys = keys(crystals, ("run", "crystal"))
    vd_keys = keys(virtualdetectors, ("run", "id"))
    if (len(np.unique(disk_keys)) != len(disk_keys) or len(np.unique(crystal_keys)) != len(crystal_keys)
            or len(np.unique(vd_keys)) != len(vd_keys)):
        raise ValueError("duplicate geometry identity")
    r_disk, r_vd = rotations(disks), rotations(virtualdetectors)
    for name, r in (("disk", r_disk), ("virtualdetector", r_vd)):
        if (not np.all(np.isfinite(r)) or not np.allclose(np.swapaxes(r, 1, 2) @ r, np.eye(3), atol=tolerance, rtol=0)
                or not np.allclose(np.linalg.det(r), 1, atol=tolerance, rtol=0)):
            raise ValueError(f"invalid {name} frame")
    row = keys(crystals, ("run", "disk"))
    order = np.argsort(disk_keys)
    slot = np.searchsorted(disk_keys[order], row)
    if np.any(slot >= len(disks["disk"])) or not np.array_equal(disk_keys[order][slot], row):
        raise ValueError("crystal refers to absent disk")
    slot = order[slot]
    position = vector(crystals, "position")
    local = vector(crystals, "local_position")
    predicted = vector(disks, "origin")[slot] + np.einsum("nij,nj->ni", r_disk[slot], local)
    discrepancy = np.abs(position - predicted)
    if not np.all(np.isfinite(discrepancy)) or np.any(discrepancy > tolerance):
        raise ValueError("crystal disk/global transform mismatch")
    for table in (crystals, disks):
        size = vector(table, "size")
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise ValueError("invalid geometry dimensions")
    if (not np.all(np.isfinite(vector(disks, "front_face_origin")))
            or not np.all(np.isfinite(vector(virtualdetectors, "position")))
            or not np.all(np.isfinite(virtualdetectors["half_length"]))
            or np.any(virtualdetectors["half_length"] <= 0)):
        raise ValueError("invalid geometry coordinates")
    return {"crystals": expected_crystals, "disks": expected_disks, "virtualdetectors": len(vd_keys),
            "crystals_per_disk": {str(int(disk)): int(count) for disk, count in
                                  zip(*np.unique(crystals["disk"], return_counts=True))},
            "max_crystal_transform_difference_mm": float(np.max(discrepancy, initial=0)),
            "frame_dimension_consistency": True, "simulation_geometry_equality_certified": False,
            "material_boundary_coverage_certified": False,
            "scope": "numeric frame/identity/dimension consistency only, not historical geometry equality"}


def check(path, *, expected_crystals, expected_disks):
    with uproot.open(path) as source:
        fields = {
            "crystals": ["run", "crystal", "disk", "local_id"],
            "disks": ["run", "disk"], "virtualdetectors": ["run", "id", "half_length"],
        }
        prefixes = {"crystals": ["position", "local_position", "size"],
                    "disks": ["origin", "front_face_origin", "size", "axis_x", "axis_y", "axis_z"],
                    "virtualdetectors": ["position", "axis_x", "axis_y", "axis_z"]}
        tables = {name: source[f"dump/{name}"].arrays(base + [f"{prefix}_{axis}" for prefix in prefixes[name] for axis in AXES],
                                                     library="np") for name, base in fields.items()}
    return audit(**tables, expected_crystals=expected_crystals, expected_disks=expected_disks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-crystals", type=int, required=True)
    parser.add_argument("--expected-disks", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.input, expected_crystals=args.expected_crystals, expected_disks=args.expected_disks)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
