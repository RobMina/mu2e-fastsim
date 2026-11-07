"""Opt-in cechain-to-disk adapter; never replaces the cesob proxy by default."""
from __future__ import annotations

import numpy as np

from mu2efastsim import ehelix

from .library import sample
from .response import place

ME = .51099895
C_MM_NS = 299.792458


def disk_front(field, library, x, u, kinetic_energy, time, *, step_mm=10., path_limit_mm=30000., radius_limit_mm=1000.):
    """Helix-to-front states, using cesob's midpoint-field vacuum stepping.

    This first-entry adapter does not transport tracker/passive material or
    side-entry interactions. Those limitations are returned in the hand-off
    documentation, not hidden as a full calorimeter-acceptance replacement.
    """
    x, u = np.array(x, float), np.array(u, float)
    kinetic_energy, time = np.asarray(kinetic_energy, float), np.array(time, float)
    n = len(x)
    if x.shape != (n, 3) or u.shape != (n, 3) or kinetic_energy.shape != (n,) or time.shape != (n,):
        raise ValueError("invalid incident-state shapes")
    if any(not np.all(np.isfinite(a)) for a in (x, u, kinetic_energy, time)) or np.any(kinetic_energy <= 0):
        raise ValueError("incident states must be finite, with positive kinetic energy")
    if step_mm <= 0 or path_limit_mm <= 0 or radius_limit_mm <= 0:
        raise ValueError("path and step limits must be positive")
    norm = np.linalg.norm(u, axis=1)
    if np.any(norm == 0):
        raise ValueError("zero direction")
    u /= norm[:, None]
    disk_ids = library["geometry_disk_disk"]
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xyz"])
    axis = np.column_stack([library["geometry_disk_axis_z_" + a] for a in "xyz"])
    if not np.allclose(axis, (0, 0, 1)):
        raise ValueError("adapter requires z-aligned disks")
    fronts = []
    for disk in disk_ids:
        vd = np.flatnonzero(library["geometry_vd_id"] == 73 + 2 * disk)
        if len(vd) != 1:
            raise ValueError("one front VD per disk is required")
        fronts.append(library["geometry_vd_position_z"][vd[0]])
    fronts = np.asarray(fronts)
    inner = library["geometry_disk_size_x"]
    outer = library["geometry_disk_size_y"]
    momentum = np.sqrt(kinetic_energy * (kinetic_energy + 2 * ME))
    speed = C_MM_NS * momentum / (kinetic_energy + ME)
    distance = np.zeros(n)
    found = np.full(n, -1, dtype=np.int64)
    for d, z in enumerate(fronts):
        radius = np.linalg.norm(x[:, :2] - origin[d, :2], axis=1)
        at_front = ((found < 0) & (np.abs(x[:, 2] - z) <= 1e-9) & (u[:, 2] > 0) &
                    (radius >= inner[d]) & (radius <= outer[d]))
        found[at_front] = d
    live = found < 0
    while live.any():
        i = np.flatnonzero(live)
        h = np.minimum(step_mm, path_limit_mm - distance[i])
        b0 = ehelix.field_at(field, x[i])
        midpoint = ehelix.helix(x[i], u[i], momentum[i], -1., b0, h / 2)[0]
        bm = ehelix.field_at(field, midpoint)
        xn, un = ehelix.helix(x[i], u[i], momentum[i], -1., bm, h)
        chosen = np.full(len(i), -1, dtype=np.int64)
        used = h.copy()
        hit_x, hit_u = xn.copy(), un.copy()
        for d, z in enumerate(fronts):
            crossing = (x[i, 2] <= z) & (xn[:, 2] >= z) & (un[:, 2] > 0)
            j = np.flatnonzero(crossing)
            if not len(j):
                continue
            length = ehelix.to_plane(x[i[j]], u[i[j]], momentum[i[j]], -1., bm[j], z, h[j])
            valid = np.isfinite(length)
            j, length = j[valid], length[valid]
            at, direction = ehelix.helix(x[i[j]], u[i[j]], momentum[i[j]], -1., bm[j], length)
            radius = np.linalg.norm(at[:, :2] - origin[d, :2], axis=1)
            inside = (radius >= inner[d]) & (radius <= outer[d]) & (length <= used[j])
            j, length, at, direction = j[inside], length[inside], at[inside], direction[inside]
            chosen[j], used[j], hit_x[j], hit_u[j] = d, length, at, direction
        x[i], u[i] = hit_x, hit_u
        time[i] += used / speed[i]
        distance[i] += used
        hit = chosen >= 0
        found[i[hit]] = chosen[hit]
        radius = np.linalg.norm(x[i, :2] - origin[0, :2], axis=1)
        live[i] = ~hit & (distance[i] < path_limit_mm) & (x[i, 2] <= fronts.max()) & (radius <= radius_limit_mm)
    keep = np.flatnonzero(found >= 0)
    d = found[keep]
    return dict(event_row=keep, disk=disk_ids[d], position=x[keep], direction=u[keep],
                energy=kinetic_energy[keep] + ME, cosine=u[keep, 2],
                radius=np.linalg.norm(x[keep, :2] - origin[d, :2], axis=1), time=time[keep])


def from_cechain(result, field, library, rng, *, plane_fate=1, placement="geometric", **options):
    """Return query states, sampled CSS, and counts; no cesob default changes.

    ``result`` is cechain's Result, including absolute ``t``; old cesob NPZs
    omit time and cannot be used for a timing-aware response without that input.
    """
    if placement not in ("geometric", "crystal"):
        raise ValueError("placement must be geometric or crystal")
    selected = np.flatnonzero(np.asarray(result.fate) == plane_fate)
    entries = disk_front(field, library, result.x[selected], result.u[selected], result.T[selected], result.t[selected], **options)
    entries["event_row"] = selected[entries["event_row"]]
    draw = sample(library, entries, rng, orientation="disk")
    if placement == "crystal":
        from .crystal_placement import place_crystals
        draw = place_crystals(library, entries, draw)
    else:
        draw = place(library, draw)
    return dict(entries=entries, steps=draw, generated_events=len(result.fate),
                tracker_entry_events=len(selected), disk_front_events=len(entries["energy"]))
