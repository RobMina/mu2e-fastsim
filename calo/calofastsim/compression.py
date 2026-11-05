"""Source-algorithm raw crystal step compression with explicit constituent indices."""
from __future__ import annotations

import operator

import numpy as np

from .ancestry import SIM_FIELDS, forest
from .identities import EVENT_FIELDS, contains, keys


def lookup(available, requested):
    order = np.argsort(available)
    if len(np.unique(available)) != len(available) or not np.all(contains(available[order], requested)):
        raise ValueError("duplicate or unresolved exact identity")
    return order[np.searchsorted(available[order], requested)]


def vector(table, prefix):
    return np.column_stack([table[f"{prefix}_{axis}"] for axis in "xyz"]).astype(np.float64)


def frame(disks):
    rotation = np.stack([vector(disks, f"axis_{axis}") for axis in "xyz"], axis=2)
    if (not np.all(np.isfinite(rotation))
            or not np.allclose(np.swapaxes(rotation, 1, 2) @ rotation, np.eye(3), atol=1e-12, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-12, rtol=0)):
        raise ValueError("invalid disk frame")
    return rotation


def containment(particles, crystals, disks):
    """Return start-in-any-cell and start/end-in-same-disk flags from numeric geometry.

    Crystal-cell union is queried in x-sorted batches, not a nearest-crystal
    approximation. Disk radius is narrowed to float32 like Disk::isInsideDisk.
    The tested numeric geometry is external; this is not native G4 navigation.
    """
    start, end = vector(particles, "start"), vector(particles, "end")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        raise ValueError("nonfinite particle state")
    r = frame(disks)
    if len(np.unique(keys(disks, ("run", "disk")))) != len(disks["disk"]):
        raise ValueError("duplicate disk identity")
    if not np.all(np.isin(particles["run"], disks["run"])):
        raise ValueError("particle run has no geometry")
    for table in (crystals, disks):
        size = vector(table, "size")
        if not np.all(np.isfinite(size)) or np.any(size[:, 1:] <= 0) or np.any(size[:, 0] < 0):
            raise ValueError("invalid geometry dimensions")
    if not np.all(np.isfinite(vector(crystals, "local_position"))):
        raise ValueError("nonfinite crystal positions")
    if not np.all(np.isfinite(vector(disks, "origin"))) or not np.all(np.isfinite(vector(disks, "front_face_origin"))):
        raise ValueError("nonfinite disk origin")
    inside_cell = np.zeros(len(start), dtype=bool)
    same_disk = np.zeros(len(start), dtype=bool)
    particle_run = np.asarray(particles["run"])
    for disk in range(len(disks["disk"])):
        rows = np.flatnonzero(particle_run == disks["run"][disk])
        local_start = (start[rows] - vector(disks, "origin")[disk]) @ r[disk]
        local_end = (end[rows] - vector(disks, "origin")[disk]) @ r[disk]
        size = vector(disks, "size")[disk]
        start_rad = np.sqrt(local_start[:, 0] ** 2 + local_start[:, 1] ** 2).astype(np.float32)
        end_rad = np.sqrt(local_end[:, 0] ** 2 + local_end[:, 1] ** 2).astype(np.float32)
        start_disk = (np.abs(local_start[:, 2]) <= size[2] / 2) & (start_rad >= size[0]) & (start_rad <= size[1])
        end_disk = (np.abs(local_end[:, 2]) <= size[2] / 2) & (end_rad >= size[0]) & (end_rad <= size[1])
        same_disk[rows] |= start_disk & end_disk
        ff = (vector(disks, "front_face_origin")[disk] - vector(disks, "origin")[disk]) @ r[disk]
        local_ff = local_start - ff
        order = np.argsort(local_ff[:, 0])
        x = local_ff[order, 0]
        crystal_rows = np.flatnonzero((crystals["run"] == disks["run"][disk]) & (crystals["disk"] == disks["disk"][disk]))
        positions, sizes = vector(crystals, "local_position"), vector(crystals, "size")
        for crystal in crystal_rows:
            center, cell_size = positions[crystal], sizes[crystal]
            low = np.searchsorted(x, center[0] - cell_size[0] / 2, side="left")
            high = np.searchsorted(x, center[0] + cell_size[0] / 2, side="right")
            candidate = order[low:high]
            inside = ((np.abs(local_ff[candidate, 1] - center[1]) <= cell_size[1] / 2)
                      & (local_ff[candidate, 2] >= -1e-6) & (local_ff[candidate, 2] <= cell_size[2])
                      & (start_rad[candidate] >= size[0]) & (start_rad[candidate] <= size[1]))
            inside_cell[rows[candidate[inside]]] = True
    return inside_cell, same_disk


def ancestor_rows(particles, steps, crystals, disks):
    """Resolve only the ancestor closure needed by supplied raw steps."""
    parent = forest(particles)
    ids = keys(particles, SIM_FIELDS)
    step_slot = lookup(ids, keys(steps, SIM_FIELDS))
    needed = np.zeros(len(ids), dtype=bool)
    needed[step_slot] = True
    jump = parent.copy()
    for _ in range(len(ids).bit_length() + 1):
        needed[jump[needed]] = True
        jump = jump[jump]
    rows = np.flatnonzero(needed)
    subset = {name: value[rows] for name, value in particles.items()}
    inside, same_disk = containment(subset, crystals, disks)
    jump = np.arange(len(ids), dtype=np.int64)
    jump[rows] = np.where(inside & same_disk & subset["parent_valid"], parent[rows], rows)
    for _ in range(len(ids).bit_length() + 1):
        jump = jump[jump]
    return jump[step_slot]


def compress(particles, steps, crystals, disks, *, num_z_slices, delta_time, energy_threshold, crystal_length,
             equal_time_order="stable"):
    """Replay compressData=true and retain each accepted raw row's group membership.

    Equal-time sorting is stable by default; reverse is a diagnostic alternative,
    not C++ std::sort emulation. That domain stays flagged. Aggregate sums are
    double sequential scatters. CSS time narrows through its float constructor.
    """
    if equal_time_order not in ("stable", "reverse"):
        raise ValueError("unsupported equal-time ordering diagnostic")
    num_z_slices = operator.index(num_z_slices)
    if num_z_slices <= 0 or num_z_slices > np.iinfo(np.uint32).max:
        raise ValueError("num_z_slices outside unsigned domain")
    delta = float(np.float32(delta_time))
    if not np.isfinite(delta) or delta < 0 or not np.isfinite(energy_threshold) or energy_threshold < 0:
        raise ValueError("invalid compression thresholds")
    if not np.isfinite(crystal_length) or crystal_length <= 0:
        raise ValueError("invalid crystal length")
    for name in ("time", "energy_g4", "energy_visible"):
        if np.asarray(steps[name]).ndim != 1 or not np.all(np.isfinite(steps[name])):
            raise ValueError("invalid step values")
    if np.any(steps["energy_g4"] <= 0) or np.any(steps["energy_visible"] < 0):
        raise ValueError("positive raw crystal energy required")
    if len(np.unique(keys(crystals, ("run", "crystal")))) != len(crystals["crystal"]):
        raise ValueError("duplicate crystal identity")
    ancestor = ancestor_rows(particles, steps, crystals, disks)
    columns = {field: particles[field][ancestor] for field in SIM_FIELDS}
    columns["crystal"] = steps["volume"]
    base_fields = SIM_FIELDS + ("crystal",)
    base, cell = np.unique(keys(columns, base_fields), return_inverse=True)
    energy = np.zeros(len(base), dtype=np.float64)
    np.add.at(energy, cell, steps["energy_g4"])
    raw_rows = np.flatnonzero(energy[cell] >= energy_threshold)
    kept = {name: value[raw_rows] for name, value in steps.items()}
    cell = cell[raw_rows]
    crystal_slot = lookup(keys(crystals, ("run", "crystal")), keys({**kept, "crystal": kept["volume"]}, ("run", "crystal")))
    disk_slot = lookup(keys(disks, ("run", "disk")), keys({**kept, "disk": crystals["disk"][crystal_slot]}, ("run", "disk")))
    rotations = frame(disks)[disk_slot]
    local = np.einsum("nji,nj->ni", rotations, vector(kept, "position") - vector(disks, "origin")[disk_slot]) - vector(crystals, "local_position")[crystal_slot]
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(vector(kept, "momentum"))):
        raise ValueError("nonfinite raw state")
    slice_size = float(crystal_length) / float(np.float32(num_z_slices)) + 1e-5
    depth_ratio = np.maximum(1e-6, local[:, 2]) / slice_size
    if np.any(depth_ratio > np.iinfo(np.uint32).max):
        raise ValueError("raw depth exceeds source unsigned conversion domain")
    bucket = np.minimum(depth_ratio.astype(np.uint64), num_z_slices - 1).astype(np.int64)
    tie_order = np.arange(len(cell))
    if equal_time_order == "reverse":
        tie_order = -tie_order
    order = np.lexsort((tie_order, kept["time"], cell))
    sorted_cell = cell[order]
    _, bucket_cell = np.unique(np.rec.fromarrays([cell, bucket], names="cell,bucket"), return_inverse=True)
    bc = bucket_cell[order]
    n_cells = int(np.max(bc, initial=-1)) + 1
    n = len(order)
    anchor = np.full(n_cells, n, dtype=np.int64)
    np.minimum.at(anchor, bc, np.arange(n))
    pending = np.ones(n, dtype=bool)
    generation = np.full(n, -1, dtype=np.int64)
    round_index = 0
    while np.any(pending):
        current = np.flatnonzero(pending)
        take = current[kept["time"][order[current]] - kept["time"][order[anchor[bc[current]]]] <= delta]
        generation[take] = round_index
        pending[take] = False
        anchor.fill(n)
        remaining = np.flatnonzero(pending)
        np.minimum.at(anchor, bc[remaining], remaining)
        round_index += 1
    group_keys = np.rec.fromarrays([bc, generation], names="bucket_cell,generation")
    unique, group, counts = np.unique(group_keys, return_inverse=True, return_counts=True)
    ng = len(unique)
    first = np.full(ng, n, dtype=np.int64)
    np.minimum.at(first, group, np.arange(n))
    last_in_bucket = np.r_[unique.bucket_cell[1:] != unique.bucket_cell[:-1], True] if ng else np.zeros(0, dtype=bool)
    trigger = np.full(ng, n, dtype=np.int64)
    next_group = np.flatnonzero(~last_in_bucket) + 1
    trigger[~last_in_bucket] = first[next_group]
    output_order = np.lexsort((bucket[order[first]], trigger, sorted_cell[first]))
    inverse_output = np.empty(ng, dtype=np.int64)
    inverse_output[output_order] = np.arange(ng)
    group = inverse_output[group]
    counts = counts[output_order]
    first = first[output_order]
    output_base = base[sorted_cell[first]]
    result = {name: output_base[name].copy() for name in base_fields}
    result["n_compress"] = counts
    result["bucket"] = bucket[order[first]]
    result["generation"] = unique.generation[output_order]
    result["flush_is_final"] = last_in_bucket[output_order]
    trigger_in_output = trigger[output_order]
    result["flush_trigger_time"] = np.full(ng, np.inf)
    nonfinal = trigger_in_output < n
    result["flush_trigger_time"][nonfinal] = kept["time"][order[trigger_in_output[nonfinal]]]
    weight = kept["energy_g4"][order].astype(np.float64)
    total = np.zeros(ng, dtype=np.float64)
    np.add.at(total, group, weight)
    visible = np.zeros(ng, dtype=np.float64)
    np.add.at(visible, group, kept["energy_visible"][order].astype(np.float64))
    result["energy_g4"], result["energy_birks"] = total.astype(np.float32), visible.astype(np.float32)
    for name, values in (("time", kept["time"][order]), ("position_x", local[order, 0]),
                         ("position_y", local[order, 1]), ("position_z", local[order, 2])):
        sums = np.zeros(ng, dtype=np.float64)
        np.add.at(sums, group, values * weight)
        result[name] = sums / total
    result["time"] = result["time"].astype(np.float32).astype(np.float64)
    momentum = vector(kept, "momentum")[order]
    pin = np.zeros(ng, dtype=np.float64)
    np.maximum.at(pin, group, np.sqrt((momentum[:, 0] ** 2 + momentum[:, 1] ** 2) + momentum[:, 2] ** 2))
    result["momentum_in"] = pin.astype(np.float32)
    tied = np.r_[False, (sorted_cell[1:] == sorted_cell[:-1]) & (kept["time"][order][1:] == kept["time"][order][:-1])] if n else np.zeros(0, dtype=bool)
    tie_cell = np.unique(sorted_cell[tied])
    result["equal_time_order_unspecified"] = np.isin(sorted_cell[first], tie_cell)
    member_order = np.argsort(group, kind="stable")
    result["member_offsets"] = np.r_[0, np.cumsum(counts)].astype(np.int64)
    result["member_index"] = raw_rows[order[member_order]]
    result["excluded_raw_index"] = np.setdiff1d(np.arange(len(steps["time"])), raw_rows)
    return result
