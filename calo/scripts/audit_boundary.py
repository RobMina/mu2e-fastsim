"""Diagnose recorded calo shell direction and raw-step ancestry/time coverage, not certify entrances."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from calofastsim.ancestry import SIM_FIELDS, forest
from calofastsim.identities import EVENT_FIELDS, contains, keys


SURFACES = np.array([73, 74, 75, 76, 77, 78, 79, 80, 102, 103, 104, 105, 106, 107, 108, 109])
DISKS = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
NORMALS = np.array([-1, 1, -1, 1, -2, 2, -2, 2, -1, 1, -1, 1, -2, 2, -2, 2])


def lookup(available, requested):
    order = np.argsort(available)
    if not np.all(contains(available[order], requested)):
        raise ValueError("unresolved exact identity")
    return order[np.searchsorted(available[order], requested)]


def vector(table, prefix):
    return np.column_stack([table[f"{prefix}_{axis}"] for axis in "xyz"])


def directions(steps, disks):
    """Classify momentum relative to outward normals, not Geant4 boundary status."""
    volume = steps["volume"]
    if not np.all(np.isin(volume, SURFACES)):
        raise ValueError("unknown calorimeter shell")
    slot = np.searchsorted(SURFACES, volume)
    disk = DISKS[slot]
    disk_slot = lookup(keys(disks, ("run", "disk")), keys({**steps, "disk": disk}, ("run", "disk")))
    rotation = np.stack([vector(disks, f"axis_{axis}") for axis in "xyz"], axis=2)[disk_slot]
    local = np.einsum("nji,nj->ni", rotation, vector(steps, "position") - vector(disks, "origin")[disk_slot])
    momentum = np.einsum("nji,nj->ni", rotation, vector(steps, "momentum"))
    normal = NORMALS[slot]
    radius = np.hypot(local[:, 0], local[:, 1])
    radial = np.abs(normal) == 2
    if np.any(radial & (radius == 0)):
        raise ValueError("radial shell at zero radius")
    dot = np.sign(normal) * np.where(radial, np.sum(local[:, :2] * momentum[:, :2], axis=1) / np.where(radius > 0, radius, 1), momentum[:, 2])
    if not np.all(np.isfinite(dot)):
        raise ValueError("nonfinite shell state")
    return dot, disk


def grouped(table, fields, values):
    unique, inverse = np.unique(keys(table, fields), return_inverse=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return unique, sums


def causal_times(parent, earliest, birth):
    """Propagate only crossings no later than every intervening child's birth bound."""
    best, jump, cutoff = earliest.copy(), parent.copy(), birth.astype(np.float64).copy()
    for _ in range(len(parent).bit_length() + 1):
        inherited = best[jump]
        best = np.minimum(best, np.where(inherited <= cutoff, inherited, np.inf))
        cutoff = np.minimum(cutoff, cutoff[jump])
        jump = jump[jump]
    return best


def describe(events, particles, steps, showers, disks):
    if np.any(events["g4_status"] != 0):
        raise ValueError("status-zero completed export required")
    event_ids = keys(events, EVENT_FIELDS)
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("duplicate event")
    parent = forest(particles)
    particle_ids = keys(particles, SIM_FIELDS)
    boundary = (steps["kind"] == 0) & np.isin(steps["volume"], SURFACES)
    shell = {name: value[boundary] for name, value in steps.items()}
    dot, disk = directions(shell, disks)
    incoming = dot < 0
    shell_particle = lookup(particle_ids, keys(shell, SIM_FIELDS))
    earliest = np.full(len(particle_ids), np.inf)
    np.minimum.at(earliest, shell_particle[incoming], shell["time"][incoming])
    own_earliest = earliest.copy()
    jump = parent.copy()
    for _ in range(len(particle_ids).bit_length() + 1):
        earliest = np.minimum(earliest, earliest[jump])
        jump = jump[jump]
    birth = particles["start_time"].astype(np.float32)
    birth_upper = np.nextafter(birth, np.float32(np.inf)).astype(np.float64)
    causal = causal_times(parent, own_earliest, birth_upper)
    birth_lower = np.nextafter(birth, np.float32(-np.inf)).astype(np.float64)
    causal_lower = causal_times(parent, own_earliest, birth_lower)
    inherited = causal[parent]
    first_lineage = np.isfinite(own_earliest) & ((parent == np.arange(len(parent))) | (inherited > birth_upper))
    lineage_counts = np.bincount(lookup(event_ids, keys(particles, EVENT_FIELDS)[first_lineage]), minlength=len(event_ids))
    raw = {name: value[steps["kind"] == 1] for name, value in steps.items()}
    raw_slot = lookup(particle_ids, keys(raw, SIM_FIELDS))
    shower_slot = lookup(particle_ids, keys(showers, SIM_FIELDS))
    raw_covered = earliest[raw_slot] <= raw["time"]
    shower_covered = earliest[shower_slot] <= showers["time"]
    raw_causal = causal[raw_slot] <= raw["time"]
    shower_causal = causal[shower_slot] <= showers["time"]
    raw_lower = causal_lower[raw_slot] <= raw["time"]
    shower_lower = causal_lower[shower_slot] <= showers["time"]
    raw_events = lookup(event_ids, keys(raw, EVENT_FIELDS))
    raw_counts = np.bincount(raw_events, minlength=len(event_ids))
    counts = np.bincount(lookup(event_ids, keys(shell, EVENT_FIELDS)[incoming]), minlength=len(event_ids))
    bad_raw_events = np.unique(keys(raw, EVENT_FIELDS)[~raw_covered])
    bad_shower_events = np.unique(keys(showers, EVENT_FIELDS)[~shower_covered])
    raw_groups, raw_energy = grouped({**raw, "crystal": raw["volume"]}, EVENT_FIELDS + ("crystal",), raw["energy_g4"])
    shower_groups, shower_energy = grouped(showers, EVENT_FIELDS + ("crystal",), showers["energy_g4"])
    all_groups = np.union1d(raw_groups, shower_groups)
    totals = np.zeros((len(all_groups), 2))
    totals[np.searchsorted(all_groups, raw_groups), 0] = raw_energy
    totals[np.searchsorted(all_groups, shower_groups), 1] = shower_energy
    raw_only = contains(raw_groups, all_groups) & ~contains(shower_groups, all_groups)
    deficit = totals[:, 0] - totals[:, 1]
    out = {
        "events": len(event_ids), "raw_crystal_rows": len(raw_slot), "shower_rows": len(shower_slot),
        "shell_rows": len(dot), "incoming_momentum_rows": int(np.count_nonzero(incoming)),
        "outgoing_momentum_rows": int(np.count_nonzero(dot > 0)), "tangent_momentum_rows": int(np.count_nonzero(dot == 0)),
        "events_with_incoming_momentum_rows": int(np.count_nonzero(counts)),
        "empty_shower_events_with_incoming_momentum_rows": int(np.count_nonzero((events["n_showers"] == 0) & (counts > 0))),
        "empty_shower_events_with_raw_crystal_rows": int(np.count_nonzero((events["n_showers"] == 0) & (raw_counts > 0))),
        "events_with_multiple_birth_order_distinct_entrance_lineages": int(np.count_nonzero(lineage_counts > 1)),
        "entrance_lineage_count_histogram": {str(int(n)): int(count) for n, count in zip(*np.unique(lineage_counts, return_counts=True))},
        "raw_steps_without_birth_order_compatible_incoming_ancestor": int(np.count_nonzero(~raw_causal)),
        "shower_rows_without_birth_order_compatible_incoming_ancestor": int(np.count_nonzero(~shower_causal)),
        "birth_order_gap_event_columns": {name: np.unique(keys(raw, EVENT_FIELDS)[~raw_causal])[name].tolist() for name in EVENT_FIELDS},
        "raw_steps_depending_on_float_birth_ambiguity": int(np.count_nonzero(raw_causal & ~raw_lower)),
        "shower_rows_depending_on_float_birth_ambiguity": int(np.count_nonzero(shower_causal & ~shower_lower)),
        "birth_time_policy": "one float32 ULP lower/upper birth bounds; crossing time is double; necessary diagnostic only",
        "raw_steps_without_earlier_incoming_ancestor_row": int(np.count_nonzero(~raw_covered)),
        "raw_step_gap_event_columns": {name: bad_raw_events[name].tolist() for name in EVENT_FIELDS},
        "shower_rows_without_earlier_incoming_ancestor_row": int(np.count_nonzero(~shower_covered)),
        "shower_gap_event_columns": {name: bad_shower_events[name].tolist() for name in EVENT_FIELDS},
        "raw_minus_shower_crystal_energy_range_mev": [float(np.min(deficit, initial=0)), float(np.max(deficit, initial=0))],
        "raw_only_event_crystal_groups": int(np.count_nonzero(raw_only)),
        "raw_only_total_energy_mev": float(np.sum(totals[raw_only, 0])),
        "raw_constituent_count": len(raw_slot), "stored_compressed_constituent_count": int(np.sum(showers["n_compress"], dtype=np.int64)),
        "shell_direction_counts": {str(int(vd)): {"incoming": int(np.count_nonzero((shell["volume"] == vd) & incoming)),
                                                   "outgoing": int(np.count_nonzero((shell["volume"] == vd) & (dot > 0)))} for vd in SURFACES},
        "scope": "momentum direction, ancestry/time and float birth-order bounds only; no boundary-status, crossing-visit or constituent assignment certificate",
        "independent_incident_manifest_complete": False, "material_exterior_coverage_certified": False,
        "compressed_constituent_causality_certified": False, "zero_response_certified": False,
    }
    return out


def check(path):
    with uproot.open(path) as source:
        fields = {
            "events": EVENT_FIELDS + ("g4_status", "n_showers"),
            "particles": SIM_FIELDS + ("parent_valid", "parent_product_id", "parent_key", "start_time"),
            "steps": SIM_FIELDS + ("kind", "volume", "time", "energy_g4") + tuple(f"{p}_{a}" for p in ("position", "momentum") for a in "xyz"),
            "showers": SIM_FIELDS + ("crystal", "time", "energy_g4", "n_compress"),
            "disks": ("run", "disk") + tuple(f"{p}_{a}" for p in ("origin", "axis_x", "axis_y", "axis_z") for a in "xyz"),
        }
        tables = {name: source[f"dump/{name}"].arrays(names, library="np") for name, names in fields.items()}
    return describe(**tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.input)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
