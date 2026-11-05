"""Shower placement and a simple cluster-sum interface to calorimeter response."""
from __future__ import annotations

import numpy as np


def place(library, draw):
    """Assign translated CSS to crystal boxes; retain explicit unmapped energy.

    Crystal-local z starts at the front face; x/y are centered. Nearest-center
    search is followed by containment, never a snap of an outside deposit into
    the detector. The donor's disk is retained for each CSS, including deposits
    in the other disk. Returned positions are global and crystal-local.
    """
    from scipy.spatial import cKDTree

    center = np.column_stack([library["geometry_crystal_position_" + a] for a in "xyz"])
    size = np.column_stack([library["geometry_crystal_size_" + a] for a in "xyz"])
    rotation = np.stack([np.column_stack([library["geometry_disk_axis_" + a + "_" + b] for b in "xyz"])
                         for a in "xyz"], axis=2)
    n = len(draw["entry"])
    crystal = np.full(n, -1, dtype=np.int64)
    local = np.full((n, 3), np.nan)
    for disk in np.unique(draw["disk"]):
        rows = np.flatnonzero(draw["disk"] == disk)
        candidates = np.flatnonzero(library["geometry_crystal_disk"] == disk)
        disk_row = np.flatnonzero(library["geometry_disk_disk"] == disk)
        if not len(candidates) or len(disk_row) != 1:
            raise ValueError("geometry must contain one run and each shower disk")
        r = rotation[disk_row[0]]
        xyz = draw["position"][rows] @ r
        origins = center[candidates] @ r
        _, near = cKDTree(origins[:, :2]).query(xyz[:, :2], k=[1, 2, 3, 4])
        valid = near < len(candidates)
        near = np.minimum(near, len(candidates) - 1)
        chosen = candidates[near]
        delta = xyz[:, None, :] - origins[near]
        inside = (valid & (np.abs(delta[:, :, :2]) <= size[chosen, :2] / 2 + 1e-6).all(axis=2) &
                  (delta[:, :, 2] >= -1e-6) & (delta[:, :, 2] <= size[chosen, 2] + 1e-6))
        match = inside.any(axis=1)
        slot = inside.argmax(axis=1)
        crystal[rows[match]] = library["geometry_crystal_crystal"][chosen[np.arange(len(rows)), slot][match]]
        local[rows] = delta[np.arange(len(rows)), slot]
    result = dict(draw)
    result.update(crystal=crystal, local_position=local, mapped=crystal >= 0)
    return result


def cluster_sum(steps, n_entries, *, energies=None, accepted=None):
    """One energy-weighted cluster per entry, explicitly not stock clustering.

    With no response supplied this is the visible-CSS-energy diagnostic. A
    caller may supply calorimeter corrected energies and its timing acceptance mask.
    Missing/empty responses have energy zero and time/position NaN.
    """
    row = np.asarray(steps["entry"])
    energy = np.asarray(steps["energy_birks"] if energies is None else energies, dtype=float)
    keep = np.ones(len(row), bool) if accepted is None else np.asarray(accepted, dtype=bool)
    if energy.shape != row.shape or keep.shape != row.shape or not np.all(np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("invalid response energy or mask")
    if row.dtype.kind not in "iu" or np.any((row < 0) | (row >= n_entries)):
        raise ValueError("invalid cluster entry index")
    weights = np.where(keep, energy, 0)
    total = np.bincount(row, weights=weights, minlength=n_entries)
    time = np.bincount(row, weights=weights * steps["time"], minlength=n_entries)
    position = np.column_stack([np.bincount(row, weights=weights * steps["position"][:, a], minlength=n_entries)
                                for a in range(3)])
    time = np.divide(time, total, out=np.full(n_entries, np.nan), where=total > 0)
    position = np.divide(position, total[:, None], out=np.full((n_entries, 3), np.nan), where=total[:, None] > 0)
    return dict(energy=total, time=time, position=position)
