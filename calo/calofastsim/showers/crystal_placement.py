"""Opt-in discrete crystal-pattern remapping, retaining sub-crystal donor coordinates."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .response import place


def place_crystals(library, entries, draw):
    """Move donor crystal centers as a whole, rather than rotating each deposit into gaps.

    Each transformed center chooses a nearby real crystal in the same disk,
    bounded by half the target cell diagonal. Donor local offsets and depths
    are retained without rotation or scaling inside that target crystal.
    Raw deposits outside the disk radial/axial envelope remain losses; missing
    cells, excessive center moves and failed containment also remain losses.
    This is a discrete shower model, not a geometric round-trip correction.
    """
    ids = library["geometry_crystal_crystal"]
    sort = np.argsort(ids)
    slot = np.searchsorted(ids[sort], draw["crystal"])
    if np.any(slot >= len(ids)) or not np.array_equal(ids[sort[slot]], draw["crystal"]):
        raise ValueError("donor crystal is missing from saved geometry")
    source = sort[slot]
    centers = np.column_stack([library["geometry_crystal_position_" + a] for a in "xyz"])
    sizes = np.column_stack([library["geometry_crystal_size_" + a] for a in "xyz"])
    disk_ids = library["geometry_disk_disk"]
    origins = np.column_stack([library["geometry_disk_origin_" + a] for a in "xyz"])
    axes = np.stack([np.column_stack([library["geometry_disk_axis_" + a + "_" + b] for b in "xyz"])
                     for a in "xyz"], axis=2)
    if np.any(np.diff(disk_ids) <= 0) or not np.allclose(axes[:, :, 2], (0, 0, 1)):
        raise ValueError("crystal remapping requires sorted z-aligned disk geometry")
    row = draw["entry"]
    donor = draw["donor"][row]
    dslot = np.searchsorted(disk_ids, entries["disk"][row])
    source_xy = library["entry_position"][donor, :2] - origins[dslot, :2]
    target_xy = entries["position"][row, :2] - origins[dslot, :2]
    angle = np.arctan2(target_xy[:, 1], target_xy[:, 0]) - np.arctan2(source_xy[:, 1], source_xy[:, 0])
    c, s = np.cos(angle), np.sin(angle)
    offset = centers[source] - library["entry_position"][donor]
    rotated = offset.copy()
    rotated[:, 0] = c * offset[:, 0] - s * offset[:, 1]
    rotated[:, 1] = s * offset[:, 0] + c * offset[:, 1]
    ideal_center = entries["position"][row] + rotated
    delta = draw["position"] - ideal_center
    unrotated = delta.copy()
    unrotated[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    unrotated[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    position = draw["position"].copy()
    accepted = np.zeros(len(row), bool)
    displacement = np.full(len(row), np.nan)
    target_id = np.full(len(row), -1, dtype=np.int64)
    loss_reason = np.zeros(len(row), dtype=np.int8)
    for disk in np.unique(draw["disk"]):
        take = np.flatnonzero(draw["disk"] == disk)
        pool = np.flatnonzero(library["geometry_crystal_disk"] == disk)
        ds = np.flatnonzero(disk_ids == disk)
        if not len(pool) or len(ds) != 1:
            raise ValueError("geometry must contain each donor disk exactly once")
        ds = ds[0]
        r = axes[ds]
        dist, near = cKDTree(centers[pool, :2]).query(ideal_center[take, :2])
        target = pool[near]
        local = unrotated[take] @ r
        raw = (draw["position"][take] - origins[ds]) @ r
        center_local = (ideal_center[take] - origins[ds]) @ r
        crystal_z = ((centers[pool] - origins[ds]) @ r)[:, 2]
        radial = np.linalg.norm(raw[:, :2], axis=1)
        center_radial = np.linalg.norm(center_local[:, :2], axis=1)
        lo, hi = library["geometry_disk_size_x"][ds], library["geometry_disk_size_y"][ds]
        in_envelope = ((radial >= lo) & (radial <= hi) & (center_radial >= lo) & (center_radial <= hi) &
                       (raw[:, 2] >= crystal_z.min() - 1e-6) &
                       (raw[:, 2] <= np.max(crystal_z + sizes[pool, 2]) + 1e-6))
        close = dist <= np.linalg.norm(sizes[target, :2], axis=1) / 2 + 1e-6
        contains = ((np.abs(local[:, :2]) <= sizes[target, :2] / 2 + 1e-6).all(axis=1) &
                    (local[:, 2] >= -1e-6) & (local[:, 2] <= sizes[target, 2] + 1e-6))
        proposed = centers[target] + local @ r.T
        proposed_radius = np.linalg.norm(((proposed - origins[ds]) @ r)[:, :2], axis=1)
        in_envelope &= (proposed_radius >= lo) & (proposed_radius <= hi)
        valid = in_envelope & close & contains
        position[take[valid]] = proposed[valid]
        accepted[take] = valid
        loss_reason[take] = np.where(~in_envelope, 1, np.where(~close, 2, np.where(~contains, 3, 0)))
        target_id[take[valid]] = ids[target[valid]]
        displacement[take] = dist
    result = place(library, dict(draw, position=position))
    result["mapped"] &= accepted
    if np.any(result["crystal"][result["mapped"]] != target_id[result["mapped"]]):
        raise ValueError("remapped deposit did not land in its selected crystal")
    result["crystal"] = np.where(result["mapped"], result["crystal"], -1)
    result["raw_position"] = draw["position"]
    result["center_shift_mm"] = displacement
    result["crystal_loss_reason"] = loss_reason
    return result
