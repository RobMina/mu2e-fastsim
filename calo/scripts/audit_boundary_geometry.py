"""Source-derived pilot envelope checks and an uncovered cable-entry witness, not full G4 navigation."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import uproot


REQUIRED = ("calorimeter.caloDiskRadiusIn", "calorimeter.caloDiskRadiusOut", "calorimeter.vdThickness",
            "calorimeter.crateZLength", "calorimeter.crateFShieldDeltaZ", "calorimeter.crateFShieldThickness",
            "calorimeter.FEBToDiskZOffset", "calorimeter.caloMotherZ1", "calorimeter.hasCrates",
            "ds.hasCableRunCal", "ds.CableRunCal.UpRin", "ds.CableRunCal.UpRout", "ds.CableRunCal.dPhi")


def read_config(path):
    text = path.read_text()
    result = {}
    for name in REQUIRED:
        matches = re.findall(r"^Geom: (?:double|bool) " + re.escape(name) + r" = ([^;]+);$", text, re.M)
        if len(matches) != 1:
            raise ValueError(f"expected one resolved value for {name}")
        result[name] = {"true": True, "false": False}.get(matches[0], matches[0])
        if not isinstance(result[name], bool):
            result[name] = float(result[name])
    return result


def describe(config, disks, virtualdetectors):
    if len(disks["disk"]) != 2 or not np.array_equal(np.sort(disks["disk"]), [0, 1]):
        raise ValueError("two disks required")
    if not (config["calorimeter.hasCrates"] and config["ds.hasCableRunCal"]):
        raise ValueError("cable-entry witness requires enabled crates and cables")
    for axis in "xyz":
        for component in "xyz":
            if not np.array_equal(disks[f"axis_{axis}_{component}"], np.full(2, float(axis == component))):
                raise ValueError("source-specific witness requires unrotated disks")
    if not all(np.all(np.isfinite(value)) for value in disks.values()):
        raise ValueError("nonfinite disk geometry")
    if not all(np.isfinite(value) for value in config.values()):
        raise ValueError("nonfinite resolved configuration")
    if not (0 < config["ds.CableRunCal.UpRin"] < config["ds.CableRunCal.UpRout"]
            and config["calorimeter.caloDiskRadiusOut"] < config["ds.CableRunCal.UpRin"]):
        raise ValueError("unsupported cable/disk radial topology")
    order = np.argsort(disks["disk"])
    z = disks["origin_z"][order]
    half = disks["size_z"][order] / 2
    thickness = config["calorimeter.vdThickness"]
    feb_half = (config["calorimeter.crateZLength"] + config["calorimeter.crateFShieldDeltaZ"]
                + config["calorimeter.crateFShieldThickness"]) / 2 + thickness
    feb_center = z + feb_half - half - config["calorimeter.FEBToDiskZOffset"]
    front, back = z - half, z + half
    feb_front, feb_back = feb_center - feb_half, feb_center + feb_half
    cable_half = (z[1] - z[0]) / 2 - feb_half
    cable_front, cable_back = feb_back[0], feb_back[0] + 2 * cable_half
    cable_radius = (config["ds.CableRunCal.UpRin"] + config["ds.CableRunCal.UpRout"]) / 2
    gap_start = max(back[0], feb_back[0], cable_front)
    gap_end = min(front[1], feb_front[1], cable_back)
    if not gap_end > gap_start or not cable_half > 0:
        raise ValueError("no uncovered inter-disk cable interval in this geometry")
    if not config["ds.CableRunCal.dPhi"] > 0:
        raise ValueError("empty cable phi interval")
    expected_ids = np.array([73, 74, 75, 76, 77, 78, 79, 80, 102, 103, 104, 105, 106, 107, 108, 109])
    expected_z = np.r_[front[0] + thickness / 2, back[0] - thickness / 2,
                       front[1] + thickness / 2, back[1] - thickness / 2,
                       z[0], z[0], z[1], z[1], feb_front[0] + thickness / 2, feb_back[0] - thickness / 2,
                       feb_front[1] + thickness / 2, feb_back[1] - thickness / 2,
                       feb_center[0], feb_center[0], feb_center[1], feb_center[1]]
    vd_order = np.argsort(virtualdetectors["id"])
    if not np.array_equal(virtualdetectors["id"][vd_order], expected_ids):
        raise ValueError("unexpected calorimeter VD registry")
    differences = virtualdetectors["position_z"][vd_order] - expected_z
    return {
        "disk_z_extents_mm": np.column_stack([front, back]).tolist(),
        "feb_z_extents_mm": np.column_stack([feb_front, feb_back]).tolist(),
        "inter_disk_cable_z_extent_mm": [float(cable_front), float(cable_back)],
        "exported_vd_minus_constructed_shell_z_mm": {str(int(vd)): float(delta) for vd, delta in zip(expected_ids, differences)},
        "uncovered_cable_entry_witness": {
            "local_z_mm": float((gap_start + gap_end) / 2), "local_radius_inside_cable_mm": float(cable_radius),
            "local_phi_degrees": 90., "direction": "radially inward from larger radius at fixed z and phi",
            "cable_outer_radius_mm": config["ds.CableRunCal.UpRout"],
            "gap_interval_outside_all_disk_and_feb_shells_mm": [float(gap_start), float(gap_end)],
            "claim": "this ray enters enabled CaloCableRunGap_0 without crossing any disk/FEB VD shell first",
            "limit": "analytic source-derived topology witness, not an observed pilot trajectory or full G4 navigation test",
        },
        "existing_disk_feb_vds_cover_all_calo_passive_material": False,
        "single_entrance_whole_event_library_certified": False,
        "scope": "released constructDiskCalorimeter/constructVirtualDetectors formulas with resolved pilot inputs; no transport or complete material navigation",
    }


def check(path, geometry_log):
    with uproot.open(path) as source:
        fields = ("disk", "origin_z", "size_z") + tuple(f"axis_{axis}_{component}" for axis in "xyz" for component in "xyz")
        disks = source["dump/disks"].arrays(fields, library="np")
        vd = source["dump/virtualdetectors"].arrays(["id", "position_z"], library="np")
    return describe(read_config(geometry_log), disks, vd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--geometry-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.input, args.geometry_log)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
