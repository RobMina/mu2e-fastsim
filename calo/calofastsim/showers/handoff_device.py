"""Opt-in device twin of :func:`calofastsim.showers.handoff.disk_front`: helix steps to the calorimeter disk front.

From device electron states (e.g. the tracker transport's exit states) to the first forward crossing of a disk front
plane inside that disk's annulus, with the same midpoint-field helix steps as the host adapter (vacuum, no
material, first passage only). All lanes step together in one ``lax.while_loop``; lanes that found a disk, left the
radial limit, passed the last front or used the path limit stop. Output feeds :func:`calofastsim.device_chain`.
"""
from __future__ import annotations

import numpy as np

ME = .51099895
C_MM_NS = 299.792458


def disk_geometry(library, dtype=np.float64):
    """Front-plane z, origins, inner and outer radii per disk, from a shower library's saved geometry."""
    disk_ids = np.asarray(library["geometry_disk_disk"])
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
    return dict(disk_ids=disk_ids, front=np.asarray(fronts, dtype), origin=origin.astype(dtype),
                inner=np.asarray(library["geometry_disk_size_x"], dtype),
                outer=np.asarray(library["geometry_disk_size_y"], dtype))


def disk_front(maps, geo, x, u, kinetic_energy, time, active, *, step_mm=10.0, path_limit_mm=30000.0,
               radius_limit_mm=1000.0, max_steps=None):
    """Device disk-front states. ``maps``: device field maps (``beamfastsim.target_geometry_jax.device_maps``) or a
    constant field 3-vector [T];
    ``geo``: :func:`disk_geometry`; x, u (unit), kinetic energy [MeV], time [ns]: device arrays; ``active``: lanes to
    step. ``max_steps`` caps the helix steps (default: the path limit over the step). Returns a dict of per-lane
    arrays: ``found``, ``disk``, ``position``, ``direction``, ``energy`` (total), ``cosine``, ``radius``, ``time``,
    ``path`` (meaningful where ``found``), and the loop's ``iterations``."""
    import jax.numpy as jnp
    from jax import lax
    from mu2efastsim import eslab_jax as SJ
    dt = x.dtype
    front = jnp.asarray(geo["front"], dt)
    origin = jnp.asarray(geo["origin"], dt)
    inner, outer = jnp.asarray(geo["inner"], dt), jnp.asarray(geo["outer"], dt)
    disk_ids = jnp.asarray(geo["disk_ids"])
    T = kinetic_energy.astype(dt)
    p = jnp.sqrt(T * (T + 2 * ME))
    speed = C_MM_NS * p / (T + ME)
    n = x.shape[0]
    steps = int(np.ceil(path_limit_mm / step_mm)) + 1 if max_steps is None else int(max_steps)

    if isinstance(maps, (tuple, list)):
        from beamfastsim import target_geometry_jax as TG

        def field(pos):
            return TG.field(maps, None, pos)
    else:
        const = jnp.asarray(maps, dt)

        def field(pos):
            return jnp.broadcast_to(const, pos.shape)

    def cond(s):
        return jnp.any(s["live"]) & (s["it"] < steps)

    def body(s):
        x, u, live = s["x"], s["u"], s["live"]
        h = jnp.minimum(step_mm, path_limit_mm - s["dist"]).astype(dt)
        b0 = field(x)
        mid = SJ.helix(x, u, p, -1.0, b0, h / 2)[0]
        bm = field(mid)
        xn, un = SJ.helix(x, u, p, -1.0, bm, h)
        chosen = jnp.full(n, -1, jnp.int32)
        used, hx, hu = h, xn, un
        for d in range(front.shape[0]):
            z = front[d]
            crossing = (x[:, 2] <= z) & (xn[:, 2] >= z) & (un[:, 2] > 0)
            length = SJ.to_plane(x, u, p, -1.0, bm, jnp.full(n, z, dt), h)
            ok = crossing & jnp.isfinite(length)
            length = jnp.where(ok, length, h)
            at, di = SJ.helix(x, u, p, -1.0, bm, length)
            r = jnp.linalg.norm(at[:, :2] - origin[d, :2], axis=1)
            take = ok & (r >= inner[d]) & (r <= outer[d]) & (length <= used)
            chosen = jnp.where(take, d, chosen)
            used = jnp.where(take, length, used)
            hx = jnp.where(take[:, None], at, hx)
            hu = jnp.where(take[:, None], di, hu)
        x = jnp.where(live[:, None], hx, x)
        u = jnp.where(live[:, None], hu, u)
        t = jnp.where(live, s["t"] + used / speed, s["t"])
        dist = jnp.where(live, s["dist"] + used, s["dist"])
        hit = live & (chosen >= 0)
        disk = jnp.where(hit, chosen, s["disk"])
        r0 = jnp.linalg.norm(x[:, :2] - origin[0, :2], axis=1)
        live = live & ~hit & (dist < path_limit_mm) & (x[:, 2] <= jnp.max(front)) & (r0 <= radius_limit_mm)
        return dict(x=x, u=u, t=t, dist=dist, disk=disk, live=live, it=s["it"] + 1)

    s = dict(x=x, u=u / jnp.linalg.norm(u, axis=1, keepdims=True), t=time.astype(dt), dist=jnp.zeros(n, dt),
             disk=jnp.full(n, -1, jnp.int32), live=active, it=jnp.int32(0))
    s = lax.while_loop(cond, body, s)
    found = s["disk"] >= 0
    d = jnp.maximum(s["disk"], 0)
    return dict(found=found, disk=disk_ids[d], position=s["x"], direction=s["u"], energy=T + ME, path=s["dist"],
                iterations=s["it"],
                cosine=jnp.clip(s["u"][:, 2], 0, 1), radius=jnp.linalg.norm(s["x"][:, :2] - origin[d, :2], axis=1),
                time=s["t"])
