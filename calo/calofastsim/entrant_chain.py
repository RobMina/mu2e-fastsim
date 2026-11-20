"""W7's default device chain: the per-entrant shower library (:mod:`calofastsim.showers.entrants`, g4fastsim P2).

The same library file (LIBSPEC npz, :func:`calofastsim.showers.entrants.load`) that the Mu2eG4 host loads drives the
W7 device chain: per batch, calorimeter entrants (front or inner-edge entry, several per event) -> donor draw ->
placed CaloShowerSteps -> crystal placement -> the device response (:mod:`calofastsim.device_chain`: parametric
default or waveform) -> hits -> clusters, one jitted call.

Entry state: what a Geant4 fast-simulation model sees at the disk envelope, in the disk frame: ``pdg``, ``disk``,
``face`` (:data:`entrants.FRONT` or :data:`entrants.INNER`), ``local_position`` and ``local_momentum`` (mm, MeV/c,
disk frame) and the global ``time``. The device computes the entry features of :func:`entrants.entry_features`
(total energy, face cosine, radius or depth key, azimuth) in double precision, looks the cell up in a nearest
occupied-cell table built on the host with the rule of :func:`entrants.sample`, and draws ``order[start + floor(u
count)]``: with the same uniforms the donors equal the host sampler's. Placement turns the donor pattern about the
disk axis to the query azimuth and translates it to the query entry (the host rule), in the library ``dtype``;
the step positions then go through :func:`device_chain.place_steps` (``placement="crystal"``, the W7 setting, or
``"geometric"``).

Edge rule (opt-in, decision D4): front entrants whose entry radius is within ``d_edge`` of the disk's inner or outer
radius are not drawn (``edge``); a host gives them to full simulation. In this chain the caller may pass their
Geant4 CaloShowerSteps as ``g4`` (:func:`device_chain.steps_batch` layout), which join the drawn steps before the
response, the stand-in for Geant4 used when validating against stock. With an ``entrant`` column (batch row of the
step's entrant, -1 padding) only the steps of entrants the chain leaves to Geant4 join, so the steps of every entrant
can be passed. ``d_edge=0`` (default) draws every entrant.

W7 default (:data:`W7_DEFAULT`, :data:`W7_LIBRARY_HASH`): library ``406bd7b4...`` (the Mu2eG4 P2 library, front and
inner-edge faces), crystal placement, draws among the 8 entries nearest in the position key, no edge rule, every
front and inner-edge entrant of an event drawn. The front-only library path (:func:`device_chain.make_chain`, one
disk-front entry per event) stays available unchanged; :func:`pack_entrants`' own keyword defaults are the host
sampler's (:func:`entrants.sample`), so the W7 setting is passed explicitly.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import device_chain as dc
from .showers import entrants as E

__all__ = ["pack_entrants", "entrants_batch", "sample_entrants", "near_edge", "make_entrant_chain", "STAT_NAMES",
           "W7_DEFAULT", "W7_LIBRARY_HASH"]

# W7's default shower source (decision 2026-11-20, after the held-out comparison and the A100 rate check): the
# pack_entrants settings and the content hash of the library they were validated with
W7_DEFAULT = dict(placement="crystal", key_neighbors=8, d_edge=0.0, psi_bins=None)
W7_LIBRARY_HASH = "406bd7b463e3e0a6fb74764da73ee6bb2d4464992ec8d953fdf68a730c09e898"

SPECIES_GROUPS = len(E.SPECIES_NAMES)        # e-, e+, gamma, other
FACE_GROUPS = len(E.SUPPORTED_FACES)         # front, inner
STAT_NAMES = ("draws", "fallback", "edge", "far", "unsupported", "steps", "mapped", "g4_steps")


def _edges(library):
    return tuple(np.asarray(library["bins_" + k], np.float64) for k in ("energy", "cosine", "front_radius", "inner_depth"))


def _cell_tables(library, disk_ids, psi_bins=None):
    """Nearest occupied cell per (species, face, disk slot) group and flat query cell, as :func:`entrants.sample`.

    Returns table (G, ne*nc*npos) of occupied-cell indices (-1: the group has no entrants), fallback (G, ...) and the
    occupied cells' order/starts/counts (the host's ``np.unique`` order, so draws agree)."""
    e, c, r, z = _edges(library)
    ne, nc, nr, nz = len(e) - 1, len(c) - 1, len(r) - 1, len(z) - 1
    npos = max(nr, nz)
    npsi = 1 if psi_bins is None else len(psi_bins) - 1
    group = np.column_stack((library["entry_species"], library["entry_face"], library["entry_disk"]))
    labels = np.column_stack((group, library["entry_cell"]))
    if psi_bins is not None:
        labels = np.column_stack((labels, E.psi_cells(library["entry_face"], library["entry_psi"], psi_bins)))
    occupied, inverse = np.unique(labels, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(occupied))
    starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int64)
    G = SPECIES_GROUPS * FACE_GROUPS * len(disk_ids)
    table = np.full((G, ne * nc * npos * npsi), -1, np.int64)
    fallback = np.ones((G, ne * nc * npos * npsi), bool)
    for s in range(SPECIES_GROUPS):
        for f, face in enumerate(E.SUPPORTED_FACES):
            npf = nz if face == E.INNER else nr
            nps = npsi if face == E.FRONT else 1
            axes_ = [np.arange(ne), np.arange(nc), np.arange(npf)] + ([np.arange(nps)] if psi_bins is not None else [])
            grid = np.stack(np.meshgrid(*axes_, indexing="ij"), -1).reshape(-1, len(axes_))
            flat = ((grid[:, 0] * nc + grid[:, 1]) * npos + grid[:, 2]) * npsi + (grid[:, 3] if psi_bins is not None else 0)
            for slot, disk in enumerate(disk_ids):
                g = (s * FACE_GROUPS + f) * len(disk_ids) + slot
                pool = np.flatnonzero(np.all(occupied[:, :3] == (s, face, disk), axis=1))
                if not len(pool):
                    continue
                distance = np.sum((grid[:, None, :] - occupied[pool, 3:][None, :, :]) ** 2, axis=2)
                near = np.argmin(distance, axis=1)
                table[g, flat] = pool[near]
                fallback[g, flat] = distance[np.arange(len(grid)), near] != 0
    return table, fallback, order, starts, counts, (ne, nc, npos, npsi)


def pack_entrants(library, *, placement="crystal", key_neighbors=0, psi_bins=None, pixel=8.0, candidate_width=12,
                  d_edge=0.0, selection="cell", nearest_k=1, max_entry_distance=20.0, far="g4", dtype=np.float64):
    """Pack an entrant library (:func:`entrants.build` / :func:`entrants.load`) into a
    :class:`device_chain.DeviceLibrary` for :func:`make_entrant_chain`.

    ``placement`` as :func:`device_chain.pack_library`; ``key_neighbors`` as :func:`entrants.sample` (opt-in: in a
    cell with more than K entrants, draw among the K nearest in the position key); ``psi_bins`` as
    :func:`entrants.sample` (opt-in psi cell axis of front entries); ``selection``, ``nearest_k``,
    ``max_entry_distance`` and ``far`` as :func:`entrants.sample` (``selection="nearest"``: the Mu2eG4 host's nearest
    donor in entry point; far entrants are edge entrants with ``far="g4"``); ``d_edge`` [mm] the edge rule (front entrants within this
    distance of the disk's inner or outer radius are not drawn); ``dtype`` the precision of positions, frames and
    geometry (cell lookup, draws and times stay double). The disks must share the global axes (the Mu2e layout):
    the host rotation about the disk axis then equals the device's rotation in global x-y."""
    import jax.numpy as jnp
    if placement not in ("crystal", "geometric"):
        raise ValueError("placement must be crystal or geometric")
    if (isinstance(key_neighbors, (bool, np.bool_)) or not isinstance(key_neighbors, (int, np.integer))
            or key_neighbors < 0):
        raise ValueError("key_neighbors must be a nonnegative integer")
    if not np.isfinite(d_edge) or d_edge < 0:
        raise ValueError("d_edge must be a nonnegative distance")
    if selection not in E.SELECTIONS or far not in E.FAR_RULES:
        raise ValueError(f"selection must be one of {E.SELECTIONS} and far one of {E.FAR_RULES}")
    if selection == "nearest":
        if (isinstance(nearest_k, (bool, np.bool_)) or not isinstance(nearest_k, (int, np.integer))
                or nearest_k < 1):
            raise ValueError("nearest_k must be a positive integer")
        if not (np.isfinite(max_entry_distance) and max_entry_distance >= 0):
            raise ValueError("the device nearest rule needs a finite max_entry_distance")
        if psi_bins is not None:
            raise ValueError("psi_bins applies to the cell rule only")
    disk_ids = np.asarray(library["geometry_disk_disk"])
    if np.any(np.diff(disk_ids) <= 0):
        raise ValueError("disk geometry must be sorted")
    axes = np.stack([np.column_stack([library["geometry_disk_axis_" + a + "_" + b] for b in "xyz"])
                     for a in "xyz"], axis=2)
    if not np.allclose(axes, np.eye(3)[None]):
        raise ValueError("the entrant device chain requires disks aligned with the global axes")
    if psi_bins is not None:
        if "entry_psi" not in library:
            raise ValueError("psi_bins needs a library built with entry_psi")
        psi_bins = tuple(float(v) for v in psi_bins)
    table, fallback, order, starts, counts, shape = _cell_tables(library, disk_ids, psi_bins)
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xyz"])
    entry_slot = np.searchsorted(disk_ids, library["entry_disk"])
    local = np.asarray(library["entry_local_position"], np.float64)
    phi = np.asarray(library["entry_azimuth"], np.float64)
    c, s = np.cos(phi), np.sin(phi)
    frame = np.zeros((len(phi), 3, 3))
    frame[:, 0, 0], frame[:, 0, 1], frame[:, 1, 0], frame[:, 1, 1], frame[:, 2, 2] = c, -s, s, c, 1.0
    centers = np.column_stack([library["geometry_crystal_position_" + a] for a in "xyz"])
    sizes = np.column_stack([library["geometry_crystal_size_" + a] for a in "xyz"])
    ids = np.asarray(library["geometry_crystal_crystal"])
    cdisk = np.asarray(library["geometry_crystal_disk"])
    sort = np.argsort(ids)
    step_crystal = np.asarray(library["step_crystal"])
    slot = np.searchsorted(ids[sort], step_crystal)
    if np.any(slot >= len(ids)) or not np.array_equal(ids[sort[np.minimum(slot, len(ids) - 1)]], step_crystal):
        raise ValueError("donor crystal is missing from saved geometry")
    lo, hi = np.asarray(library["geometry_disk_size_x"], np.float64), np.asarray(library["geometry_disk_size_y"], np.float64)
    extent = float(np.max(hi)) + 4 * pixel
    n_pix = int(np.ceil(2 * extent / pixel))
    pix, zmin, zmax = [], [], []
    for d, disk in enumerate(disk_ids):
        pool = np.flatnonzero(cdisk == disk)
        if not len(pool):
            raise ValueError("geometry must contain each disk's crystals")
        loc = centers[pool] - origin[d]
        cand = dc._pixel_candidates(loc[:, :2], np.array([-extent, -extent]), pixel, n_pix, candidate_width,
                                    (lo[d], hi[d]))
        pix.append(np.where(cand >= 0, pool[np.maximum(cand, 0)], -1))
        zmin.append(loc[:, 2].min())
        zmax.append(np.max(loc[:, 2] + sizes[pool, 2]))
    e, cs, r, z = _edges(library)
    key = np.asarray(library["entry_key"], np.float64)
    key_order = np.concatenate([order[s0:s0 + n0][np.argsort(key[order[s0:s0 + n0]], kind="stable")]
                                for s0, n0 in zip(starts, counts)]) if len(counts) else np.zeros(0, np.int64)
    a = dict(
        edges_energy=e, edges_cosine=cs, edges_front_radius=r, edges_inner_depth=z,
        edges_psi=np.asarray(psi_bins if psi_bins is not None else (-np.pi, np.pi), np.float64),
        disk_ids=disk_ids, cell_table=table, cell_fallback=fallback, order=order, starts=starts, counts=counts,
        key_order=key_order, entry_key=key, key_sorted=key[key_order], offsets=np.asarray(library["offsets"], np.int64),
        donor_frame=frame, donor_angle=phi, donor_position=origin[entry_slot] + local,
        step_energy_g4=np.asarray(library["step_energy_g4"], np.float64),
        step_energy_birks=np.asarray(library["step_energy_birks"], np.float64),
        step_relative_position=np.asarray(library["step_relative_position"], np.float64),
        step_relative_time=np.asarray(library["step_relative_time"], np.float64),
        step_disk=np.asarray(library["step_disk"], np.int64), step_source=sort[slot],
        origin=origin, axes=axes, inner=lo, outer=hi, zmin=np.asarray(zmin), zmax=np.asarray(zmax),
        centers=centers, sizes=sizes, crystal_ids=ids, crystal_disk=cdisk, pixel_candidates=np.stack(pix),
    )
    for k in ("donor_frame", "donor_angle", "donor_position", "step_relative_position", "origin", "axes", "inner",
              "outer", "zmin", "zmax", "centers", "sizes"):
        a[k] = np.asarray(a[k], dtype)
    nearest_meta = {}
    if selection == "nearest":
        a.update(_nearest_tables(library, disk_ids, shape, float(max_entry_distance)))
        nearest_meta = dict(max_group=int(a["near_count"].max(initial=0)), window=int(a.pop("near_window")))
    meta = dict(entrants=int(len(phi)), steps=int(len(step_crystal)), occupied_cells=int(len(counts)),
                max_cell=int(counts.max(initial=0)), max_steps=int(np.diff(a["offsets"]).max(initial=0)),
                d_edge=float(d_edge), key_neighbors=int(key_neighbors),
                psi_bins=psi_bins, dtype=np.dtype(dtype).name, selection=selection, nearest_k=int(nearest_k),
                max_entry_distance=float(max_entry_distance), far=far, library="entrants", **nearest_meta)
    return dc.DeviceLibrary({k: jnp.asarray(v) for k, v in a.items()}, shape, int(key_neighbors), placement, n_pix,
                            float(pixel), meta)


def _nearest_tables(library, disk_ids, shape, max_distance):
    """Device tables of the nearest rule: library rows sorted by (species, face, disk slot, energy bin, cosine bin)
    group, then position key, then row; per group start and count; and the window width, the most entrants of one
    group within 2 ``max_distance`` in key. The entry distance is at least the key difference (front: the radius
    difference; inner edge: the depth difference), so every donor within ``max_distance`` of a query lies in the
    query's key window, which the device searches exactly."""
    ne, nc = shape[0], shape[1]
    face_index = np.searchsorted(np.asarray(E.SUPPORTED_FACES), library["entry_face"])
    slot = np.searchsorted(disk_ids, library["entry_disk"])
    cell = np.asarray(library["entry_cell"])
    g = (np.asarray(library["entry_species"]) * FACE_GROUPS + face_index) * len(disk_ids) + slot
    group = (g * ne + cell[:, 0]) * nc + cell[:, 1]
    n_groups = SPECIES_GROUPS * FACE_GROUPS * len(disk_ids) * ne * nc
    key = np.asarray(library["entry_key"], np.float64)
    rows = np.arange(len(key))
    order = np.lexsort((rows, key, group))
    count = np.bincount(group, minlength=n_groups).astype(np.int64)
    start = np.r_[0, np.cumsum(count[:-1])].astype(np.int64)
    ks = key[order]
    window = 1
    for s0, c0 in zip(start[count > 0], count[count > 0]):
        kg = ks[s0:s0 + c0]
        window = max(window, int(np.max(np.searchsorted(kg, kg + 2 * max_distance, side="right") - np.arange(c0))))
    return dict(near_order=order, near_start=start, near_count=count, near_key=ks,
                near_phi=np.asarray(library["entry_azimuth"], np.float64)[order], near_window=window)


def near_edge(face, local_position, inner, outer, d_edge):
    """Host edge rule: front entries with radius within ``d_edge`` of ``inner``/``outer`` (per entry, mm)."""
    rho = np.hypot(np.asarray(local_position)[:, 0], np.asarray(local_position)[:, 1])
    return (np.asarray(face) == E.FRONT) & (np.minimum(rho - inner, outer - rho) < d_edge) if d_edge > 0 else \
        np.zeros(len(rho), bool)


def entrants_batch(entrants, timing, n_entrants, n_events):
    """Pad entrants (several per event allowed) and per-event timing for :func:`make_entrant_chain`.

    ``entrants``: ``event`` (batch-local), ``pdg``, ``face``, ``disk``, ``time``, ``local_position`` (N,3),
    ``local_momentum`` (N,3). ``timing`` as :func:`device_chain.entries_batch`. Returns (q, t) of NumPy arrays."""
    m = len(entrants["event"])
    if m > n_entrants:
        raise ValueError("more entrants than the batch capacity")
    if m and (np.min(entrants["event"]) < 0 or np.max(entrants["event"]) >= n_events):
        raise ValueError("entrant event outside the batch")
    q = {"valid": np.zeros(n_entrants, bool)}
    q["valid"][:m] = True
    for k, shape, fill, dtype in (("event", (), 0, np.int64), ("pdg", (), 0, np.int64), ("face", (), E.FRONT, np.int64),
                                  ("disk", (), 0, np.int64), ("time", (), 0.0, np.float64),
                                  ("local_position", (3,), 0.0, np.float64), ("local_momentum", (3,), 0.0, np.float64)):
        a = np.full((n_entrants,) + shape, fill, dtype)
        a[:m] = np.asarray(entrants[k], dtype)
        q[k] = a
    empty = {k: np.zeros((0, 3) if k in ("position", "direction") else 0)
             for k in ("energy", "cosine", "radius", "disk", "time", "position", "direction")}
    _, t = dc.entries_batch(empty, timing, np.zeros(0, np.int64), n_events)
    return q, t


def _features(q):
    """Device twin of :func:`entrants.entry_features` (float64): energy, cosine, key, azimuth, radius, species."""
    import jax.numpy as jnp
    lp = q["local_position"].astype(jnp.float64)
    lm = q["local_momentum"].astype(jnp.float64)
    pdg = q["pdg"]
    species = jnp.where(pdg == 11, 0, jnp.where(pdg == -11, 1, jnp.where(pdg == 22, 2, 3)))
    p = jnp.sqrt(jnp.sum(lm * lm, axis=1))
    mass = jnp.where(species < 2, E.MASS[11], 0.0)
    energy = jnp.sqrt(p ** 2 + mass ** 2)
    phi = jnp.arctan2(lp[:, 1], lp[:, 0])
    rho = jnp.hypot(lp[:, 0], lp[:, 1])
    safe = jnp.where(p > 0, p, 1.0)
    along = jnp.where(p > 0, lm[:, 2] / safe, 0.0)
    outward = jnp.where(p > 0, (lm[:, 0] * jnp.cos(phi) + lm[:, 1] * jnp.sin(phi)) / safe, 0.0)
    inner = q["face"] == E.INNER
    cosine = jnp.clip(jnp.where(inner, outward, along), 0.0, 1.0)
    key = jnp.where(inner, lp[:, 2], rho)
    psi = jnp.angle(jnp.exp(1j * (jnp.arctan2(lm[:, 1], lm[:, 0]) - phi)))
    return energy, cosine, key, phi, rho, species, psi


def _bsearch(values, lo, hi, x, side, iters):
    """:func:`device_chain._bsearch` (segmented searchsorted, absolute index) as a device loop: the same steps and
    result. The unrolled version fused with the entry features makes the XLA CPU compile take hours."""
    import jax
    import jax.numpy as jnp

    def step(_, carry):
        lo, hi = carry
        mid = (lo + hi) // 2
        v = values[jnp.clip(mid, 0, values.shape[0] - 1)]
        go = (v < x) if side == "left" else (v <= x)
        active = lo < hi
        return jnp.where(active & go, mid + 1, lo), jnp.where(active & ~go, mid, hi)
    lo, hi = jax.lax.fori_loop(0, iters, step, (lo, jnp.broadcast_to(hi, lo.shape)))
    return lo


def sample_entrants(lib: dc.DeviceLibrary, q, uniform, return_far=False):
    """Donor row per entrant (-1: no donor), fallback (another cell or outside the bins), edge (left to full
    simulation by the edge rule, or far from any donor under the nearest rule with ``far="g4"``), entry-disk slot and
    the global entry position (library dtype); with ``return_far`` also the nearest rule's far flag."""
    import jax.numpy as jnp
    a = lib.arrays
    pdt = a["centers"].dtype
    energy, cosine, key, phi, rho, species, psi = _features(q)

    def locate(edges, v):
        idx = jnp.clip(jnp.searchsorted(edges, v, side="right") - 1, 0, edges.shape[0] - 2)
        return idx, (v < edges[0]) | (v > edges[-1])

    inner = q["face"] == E.INNER
    ie, oe = locate(a["edges_energy"], energy)
    ic, oc = locate(a["edges_cosine"], cosine)
    ir, orr = locate(a["edges_front_radius"], key)
    iz, oz = locate(a["edges_inner_depth"], key)
    ip, op = jnp.where(inner, iz, ir), jnp.where(inner, oz, orr)
    ne, nc, npos, npsi = lib.shape
    flat = (ie * nc + ic) * npos + ip
    if lib.meta.get("psi_bins") is not None:
        ips, _ = locate(a["edges_psi"], psi)
        flat = flat * npsi + jnp.where(q["face"] == E.FRONT, ips, 0)
    nd = a["disk_ids"].shape[0]
    slot = jnp.clip(jnp.searchsorted(a["disk_ids"], q["disk"]), 0, nd - 1)
    known = (a["disk_ids"][slot] == q["disk"]) & ((q["face"] == E.FRONT) | inner)
    g = (species * FACE_GROUPS + jnp.where(inner, 1, 0)) * nd + slot
    occ = jnp.where(known, a["cell_table"][g, flat], -1)
    fallback = oe | oc | op | ~known | a["cell_fallback"][g, flat]
    d_edge = lib.meta["d_edge"]
    if d_edge > 0:
        lo, hi = a["inner"][slot].astype(jnp.float64), a["outer"][slot].astype(jnp.float64)
        edge = (q["face"] == E.FRONT) & (jnp.minimum(rho - lo, hi - rho) < d_edge) & q["valid"]
    else:
        edge = jnp.zeros(q["valid"].shape, bool)
    o = jnp.maximum(occ, 0)
    start, count = a["starts"][o], a["counts"][o]
    donor = a["order"][start + (uniform * count).astype(jnp.int64)]
    K = lib.radius_neighbors          # key_neighbors (DeviceLibrary's neighbour field)
    if K:
        # the K nearest in the position key within the cell, as device_chain.sample_donors' radius neighbours
        draw = (uniform * K).astype(jnp.int64)
        kk = a["key_sorted"]          # packed on the host: a device gather here makes the CPU compile take hours
        iters = int(max(lib.meta["max_cell"], 1)).bit_length() + 1
        split = _bsearch(kk, start, start + count, key, "left", iters) - start
        ar = jnp.arange(K)
        left = split[:, None] - K + ar
        right = split[:, None] + ar
        valid_left = left >= 0
        valid_right = right < count[:, None]
        left = jnp.maximum(left, 0)
        lv = kk[jnp.clip(start[:, None] + left, 0, kk.shape[0] - 1)]
        first = _bsearch(kk, jnp.broadcast_to(start[:, None], left.shape),
                            jnp.broadcast_to((start + count)[:, None], left.shape), lv, "left", iters) - start[:, None]
        left = first + left - jnp.maximum(first, jnp.maximum(split - K, 0)[:, None])
        pos = jnp.concatenate([left, jnp.minimum(right, count[:, None] - 1)], axis=1)
        cand = a["key_order"][jnp.clip(start[:, None] + pos, 0, a["key_order"].shape[0] - 1)]
        dist = jnp.abs(a["entry_key"][cand] - key[:, None])
        dist = jnp.where(jnp.concatenate([valid_left, valid_right], axis=1), dist, jnp.inf)
        rank = jnp.lexsort((cand, dist), axis=1)[:, :K]
        eligible = jnp.sort(jnp.take_along_axis(cand, rank, axis=1), axis=1)
        pooled = jnp.take_along_axis(eligible, jnp.minimum(draw, K - 1)[:, None], axis=1)[:, 0]
        donor = jnp.where(count <= K, donor, pooled)
    far = jnp.zeros(q["valid"].shape, bool)
    if lib.meta.get("selection") == "nearest":
        W, kq = lib.meta["window"], lib.meta["nearest_k"]
        dmax = lib.meta["max_entry_distance"]
        g5 = (g * ne + ie) * nc + ic
        g5 = jnp.clip(g5, 0, a["near_start"].shape[0] - 1)
        st5 = a["near_start"][g5]
        ct5 = jnp.where(known, a["near_count"][g5], 0)
        nk = a["near_key"]
        iters = int(max(lib.meta["max_group"], 1)).bit_length() + 1
        lo = _bsearch(nk, st5, st5 + ct5, key - dmax, "left", iters)
        hi = _bsearch(nk, st5, st5 + ct5, key + dmax, "right", iters)
        idx = lo[:, None] + jnp.arange(W)
        inwin = idx < hi[:, None]
        ci = jnp.clip(idx, 0, nk.shape[0] - 1)
        dphi = phi[:, None] - a["near_phi"][ci]
        dphi = dphi - jnp.round(dphi / jnp.pi) * jnp.pi
        dk = nk[ci]
        kk_, rr = key[:, None], rho[:, None]
        d2 = jnp.where(inner[:, None], (kk_ - dk) ** 2 + (rr * dphi) ** 2,
                       kk_ * kk_ + dk * dk - 2.0 * kk_ * dk * jnp.cos(dphi))
        d2 = jnp.where(inwin, jnp.maximum(d2, 0.0), jnp.inf)
        rows = a["near_order"][ci]
        rank = jnp.lexsort((rows, d2), axis=1)
        m = jnp.minimum(kq, ct5)
        pick = jnp.minimum((uniform * m).astype(jnp.int64), jnp.maximum(m - 1, 0))
        chosen = jnp.take_along_axis(rank, jnp.minimum(pick, W - 1)[:, None], axis=1)[:, 0]
        d2c = jnp.take_along_axis(d2, chosen[:, None], axis=1)[:, 0]
        nearest = jnp.take_along_axis(rows, chosen[:, None], axis=1)[:, 0]
        far = (~(d2c <= dmax * dmax) | (pick >= W) | (ct5 == 0)) & q["valid"]
        if lib.meta["far"] == "cell":
            donor = jnp.where(far, donor, nearest)
            fallback = jnp.where(far, fallback, oe | oc | op | ~known)
        else:
            donor = nearest
            fallback = oe | oc | op | ~known
            edge = edge | far
    donor = jnp.where((occ >= 0) & q["valid"] & ~edge, donor, -1)
    position = a["origin"][slot] + q["local_position"].astype(pdt)
    if return_far:
        return donor, fallback & q["valid"], edge, slot, position, far
    return donor, fallback & q["valid"], edge, slot, position


def make_entrant_chain(lib: dc.DeviceLibrary, resp: dc.DeviceResponse, caps: dc.Caps = dc.Caps(),
                       digi_response="parametric", truth=False):
    """Return ``chain(q, t, key, uniform=None, g4=None)``: entrants -> clusters, one jitted call per batch.

    ``q``/``t`` from :func:`entrants_batch` (``t`` has ``caps.events`` rows; ``q`` any number of entrant rows);
    ``uniform`` (per entrant, in [0,1)) replaces the draw's random numbers, for tests; ``g4`` (optional) Geant4
    CaloShowerSteps in the :func:`device_chain.steps_batch` layout that join the drawn steps (the edge entrants'
    full simulation). ``caps.steps`` bounds the drawn steps. The result has the layout of
    :func:`device_chain.make_chain` plus ``edge`` (per entrant) and the stats :data:`STAT_NAMES`.
    ``digi_response`` and ``truth`` as :func:`device_chain.make_chain`. The library arrays are an argument of the
    jitted call, not constants, so a full-size library (about a million steps) compiles quickly."""
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise ValueError("the device chain requires caller-enabled JAX x64")
    if lib.meta.get("library") != "entrants":
        raise ValueError("make_entrant_chain needs a library packed by pack_entrants")
    table = dc._digi_table(digi_response)
    B, nc = caps.events, resp.n_crystals

    def chain(arrays, q, t, key, uniform, g4):
        lb = dataclasses.replace(lib, arrays=arrays)
        k_draw, k_pe, k_digi, k_pbt = jax.random.split(key, 4)
        if uniform is None:
            uniform = jax.random.uniform(k_draw, q["valid"].shape, dtype=jnp.float64)
        donor, fallback, edge, slot, position, far = sample_entrants(lb, q, uniform, return_far=True)
        qq = dict(q, position=position, time=q["time"].astype(jnp.float64))
        st, over_steps = dc.gather_steps(lb, qq, donor, slot, caps.steps)
        crystal, depth, mapped = dc.place_steps(lb, qq, st)
        ev = q["event"][st["row"]]
        time, eg4, ebirks, valid = st["time"], st["energy_g4"], st["energy_birks"], st["valid"]
        n_g4 = jnp.int64(0)
        if g4 is not None:
            g_ev, g_cr = g4["event"].astype(jnp.int64), g4["crystal"].astype(jnp.int64)
            g_ok = g_cr >= 0
            if "entrant" in g4:          # steps of every entrant: keep those of the entrants left to Geant4
                g_ok = g_ok & edge[jnp.clip(g4["entrant"].astype(jnp.int64), 0, edge.shape[0] - 1)] & (g4["entrant"] >= 0)
            n_g4 = jnp.sum(g_ok)
            ev = jnp.concatenate([ev, g_ev])
            time = jnp.concatenate([time, g4["time"].astype(time.dtype)])
            eg4 = jnp.concatenate([eg4, g4["energy_g4"].astype(eg4.dtype)])
            ebirks = jnp.concatenate([ebirks, g4["energy_birks"].astype(ebirks.dtype)])
            valid = jnp.concatenate([valid, g_ok])
            crystal = jnp.concatenate([crystal, jnp.where(g_ok, g_cr, -1)])
            depth = jnp.concatenate([depth, g4["z_local"].astype(depth.dtype)])
            mapped = jnp.concatenate([mapped, g_ok])
        n = ev.shape[0]
        sc = dict(row=jnp.arange(n), time=time, energy_g4=eg4, energy_birks=ebirks, valid=valid)

        def with_truth(hits):
            if not truth:
                return hits
            use = valid & mapped
            cell = jnp.where(use, ev * nc + jnp.maximum(crystal, 0), B * nc)
            e = jnp.zeros(B * nc + 1, jnp.float64).at[cell].add(jnp.where(use, ebirks, 0.0).astype(jnp.float64))
            tm = jnp.full(B * nc + 1, jnp.inf, jnp.float64).at[cell].min(jnp.where(use, time, jnp.inf).astype(jnp.float64))
            hc = jnp.where(hits["ok"], hits["event"] * nc + hits["crystal"], B * nc)
            return dict(hits, etrue=e[hc], ttrue=tm[hc])

        common = dict(draws=jnp.sum(donor >= 0), fallback=jnp.sum(fallback & (donor >= 0)), edge=jnp.sum(edge),
                      far=jnp.sum(far),
                      unsupported=jnp.sum(q["valid"] & ~edge & (donor < 0)), steps=jnp.sum(st["valid"]),
                      mapped=jnp.sum(mapped), g4_steps=n_g4)
        if table is not None:
            hits, over_resp, stats = dc.param_hits(resp, table, ev, t, sc, crystal, depth, mapped, k_pe,
                                                   dc._smeared_pbt(resp, t, k_pbt), caps)
            cl, over_cl = dc.clusters(resp, with_truth(hits), B, caps)
            over = dict(steps=over_steps, **over_resp, **over_cl)
            stats.update(common, photoelectrons=jnp.float64(0), digis=jnp.int64(0), hits=jnp.sum(hits["ok"]))
        else:
            pe_ev, sipm, pe_t, weight, pe_ok, over_pe = dc.photoelectrons(resp, {"event": ev}, t, sc, crystal, depth,
                                                                          mapped, k_pe, caps.photoelectrons)
            dg, over_dg = dc.digitise(resp, t, pe_ev, sipm, pe_t, weight, pe_ok, k_digi, caps)
            hits, over_hits, stats = dc.fit_and_hits(resp, t, dg, dc._smeared_pbt(resp, t, k_pbt), caps)
            cl, over_cl = dc.clusters(resp, with_truth(hits), B, caps)
            over = dict(steps=over_steps, photoelectrons=over_pe, hits=over_hits, refits=stats.pop("refits"),
                        fit_buckets=stats.pop("fit_buckets"), **over_dg, **over_cl)
            stats.update(common, photoelectrons=jnp.sum(jnp.where(pe_ok, weight, 0.0)), digis=jnp.sum(dg["ok"]),
                         hits=jnp.sum(hits["ok"]))
        return dict(clusters=cl, overflow=jnp.stack([jnp.asarray(over[k], jnp.int64) for k in dc.OVERFLOW_NAMES]),
                    stats=dc._with_cluster_stats(stats, over_cl), donor=donor, edge=edge)

    jitted = jax.jit(chain, compiler_options=dc.COMPILER_OPTIONS)
    arrays = lib.arrays

    def call(q, t, key, uniform=None, g4=None):
        return jitted(arrays, q, t, key, uniform, g4)

    return call


def run(chain, q, t, key, *, g4=None, tolerate_overflow=False):
    """:func:`device_chain.run` for :func:`make_entrant_chain` (optional ``g4`` steps, host arrays)."""
    import jax
    import jax.numpy as jnp
    dev = lambda d: None if d is None else {k: jnp.asarray(v) for k, v in d.items()}  # noqa: E731
    out = chain(dev(q), dev(t), key, None, dev(g4))
    out = jax.tree_util.tree_map(np.asarray, out)
    over = dict(zip(dc.OVERFLOW_NAMES, out["overflow"].tolist()))
    if not tolerate_overflow and any(over.values()):
        raise RuntimeError(f"device chain capacity exceeded: { {k: v for k, v in over.items() if v} }")
    out["overflow"] = over
    return out
