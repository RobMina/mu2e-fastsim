"""Opt-in on-device calorimeter chain: calorimeter-entrance states -> clusters in one jitted call per batch.

Stages, all on the device with the per-event data resident between them:

1. whole-shower library draw (:func:`calofastsim.showers.library.sample`, ``orientation="disk"``): cell lookup,
   one uniform draw per query, and a gather from the packed library (offsets plus flat step arrays);
2. crystal placement (:func:`calofastsim.showers.crystal_placement.place_crystals`) or the geometric placement
   (:func:`calofastsim.showers.response.place`); nearest crystals come from a per-disk pixel table of candidates
   followed by the exact distance and containment tests, instead of a k-d tree;
3. crystal response and photoelectrons (:mod:`calofastsim.readout`): time folding, Birks/LRU, Poisson statistics
   and per-photoelectron propagation time;
4. digitisation (:mod:`calofastsim.digi` rules): the waveform is built by histogramming each row's photoelectrons
   in (sample, fine phase) over a short window and multiplying by the pulse table, late photoelectrons are added
   one by one; then noise, ADC, peak search (as :mod:`calofastsim.digi_device`) and digi extraction;
5. template fit (:mod:`calofastsim.templatefit`), reco digis, hits (:mod:`calofastsim.hits`) and clusters
   (:mod:`calofastsim.cluster`).

Nothing here changes a default; the host modules stay the reference and are not modified. Every stage follows the
host rules; differences, by design:

- random numbers come from ``jax.random``, so with photon statistics, travel time, noise or the reco bunch-time
  smear on, only distributions agree with the host path (the library draw takes explicit uniforms for tests);
- sums run in another order (waveforms, fit reductions, weighted means), so values agree to roundoff;
- every collection has a static capacity (:class:`Caps`). Entries beyond a capacity are dropped and counted in
  ``overflow``; :func:`run` raises unless the caller asks to tolerate it. Pick capacities from a first batch.

The jitted chains compile without XLA command buffers (:data:`COMPILER_OPTIONS`), whatever the process's
device-loop setting (``mu2efastsim.devloops``). JAX x64 must be enabled by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cluster import ClusterConfig, Geometry
from .digi import DigiConfig, noise_pedestal
from .pulse import N_STEPS, PulseCache
from .readout import PropagationTable, ReadoutConfig
from .showers.library import Binning, cells, frame
from .templatefit import FitConfig

__all__ = ["Caps", "COMPILER_OPTIONS", "pack_library", "pack_response", "make_chain", "make_steps_chain", "run",
           "entries_batch", "steps_batch", "flat_clusters", "OVERFLOW_NAMES"]

OVERFLOW_NAMES = ("steps", "photoelectrons", "rows", "outliers", "digis", "digi_length", "hits", "event_hits",
                  "main_rounds", "split_rounds", "refits", "fit_buckets")
TF32_EXACT = "default"   # matmul precision for products TF32 computes exactly (0/1 operands, integer sums)
# XLA options of the chain's jit: no command buffers. With XLA's loop and conditional command buffers on (the
# mu2efastsim.devloops setting where on, --xla_gpu_enable_command_buffer=+CONDITIONAL,+WHILE) the clustering's nested device
# loops gave wrong clusters on A100 GPUs, in about a quarter of the events of a batch and differently from run to
# run; compiled without command buffers they match the CPU result (float64 identical). Set per jit, so other programs
# keep the process setting.
COMPILER_OPTIONS = {"xla_gpu_enable_command_buffer": ""}


def _physics(*xs):
    """Matmul precision of a physics product: "highest" when an operand is float32 (XLA's default runs float32 on TF32
    tensor cores on Ampere and later, a 10-bit mantissa), XLA's default otherwise (float64 is unaffected)."""
    return "highest" if any(x.dtype == np.float32 for x in xs) else None


@dataclass(frozen=True)
class Caps:
    """Static capacities of one batch (each one is a compiled shape)."""

    events: int = 1024
    steps: int = 65536
    photoelectrons: int = 1 << 23
    rows: int = 65536
    digis: int = 65536
    digi_length: int = 96
    hits: int = 32768
    event_hits: int = 64
    rounds: int = 8
    window: int = 8              # samples per row histogrammed for the pulse matmul
    outliers: int = 1 << 18      # photoelectrons outside a row's window (scattered one by one)
    recluster: bool = True       # events with more main or split clusters than ``rounds``: cluster again (clusters)
    refits: int = 0              # > 0: fit every digi without the multi-peak refit, then fit again only the digis
    #                              that need it (at most this many per batch); 0: one pass with the refit for all
    fit_short_length: int = 0    # > 0: fit digis up to this length on that many columns (bucket), the rest on all
    fit_short: int = 0           # capacity of the short bucket (digis per batch)
    fit_long: int = 0            # capacity of the long bucket
    wave_window: int = 0         # > 0: waveform samples per row (window from the row's first photoelectron), the rest
    #                              of the waveform is zero; 0: the full digitisation window for every row
    wide_rows: int = 0           # rows whose signal does not fit the window: digitised on the full waveform
    wide_digis: int = 0          # digis of those rows (capacity, added to ``digis``)


# ----------------------------------------------------------------------------------------------- host packing

def _nearest_cell_table(library, disk_ids, binning):
    """(disk slot, flat query cell) -> occupied cell index (or -1) and a fallback flag, as library.sample."""
    edges = binning.arrays()
    shape = tuple(len(e) - 1 for e in edges)
    labels = np.column_stack((library["entry_disk"], library["entry_cell"]))
    occupied, inverse = np.unique(labels, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    table = np.full((len(disk_ids), len(grid)), -1, np.int64)
    fallback = np.ones((len(disk_ids), len(grid)), bool)
    for slot, disk in enumerate(disk_ids):
        pool = np.flatnonzero(occupied[:, 0] == disk)
        if not len(pool):
            continue
        distance = np.sum((grid[:, None, :] - occupied[pool, 1:][None, :, :]) ** 2, axis=2)
        near = np.argmin(distance, axis=1)
        table[slot] = pool[near]
        fallback[slot] = distance[np.arange(len(grid)), near] != 0
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(occupied))
    starts = np.r_[0, np.cumsum(counts[:-1])]
    return table, fallback, order, starts, counts, shape


def _pixel_candidates(centers_xy, lo_xy, pixel, n_pix, width, annulus):
    """Per pixel: every crystal that can be nearest to some point of the pixel (index into ``centers_xy``).

    Far from the crystals (the central hole, outside the disk) only the ``width`` nearest are kept; such points
    are outside the radial envelope and never mapped. A truncated pixel inside ``annulus`` (inner, outer radius,
    widened by two pixels) is an error.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(centers_xy)
    ij = np.stack(np.meshgrid(np.arange(n_pix), np.arange(n_pix), indexing="ij"), -1).reshape(-1, 2)
    mid = lo_xy + (ij + 0.5) * pixel
    dmin, _ = tree.query(mid)
    reach = tree.query_ball_point(mid, dmin + pixel * np.sqrt(2.0) + 1e-6)
    r = np.hypot(mid[:, 0], mid[:, 1])
    inside = (r >= annulus[0] - 2 * pixel) & (r <= annulus[1] + 2 * pixel)
    out = np.full((len(mid), width), -1, np.int64)
    for p, cand in enumerate(reach):
        cand = np.asarray(cand, np.int64)
        if len(cand) > width:
            if inside[p]:
                raise ValueError(f"pixel candidate width {width} too small ({len(cand)}) inside the disk")
            cand = cand[np.argsort(np.linalg.norm(centers_xy[cand] - mid[p], axis=1), kind="stable")[:width]]
        out[p, :len(cand)] = np.sort(cand)
    return out


@dataclass
class DeviceLibrary:
    arrays: dict
    shape: tuple
    radius_neighbors: int
    placement: str
    n_pix: int
    pixel: float
    meta: dict = field(default_factory=dict)


def pack_library(library, *, radius_neighbors=0, placement="crystal", pixel=8.0, candidate_width=12,
                 dtype=np.float64):
    """Pack a :mod:`calofastsim.showers.library` dict (with saved geometry) into device arrays.

    ``placement`` is ``"crystal"`` (:func:`place_crystals`) or ``"geometric"`` (:func:`place`). ``dtype`` is the
    precision of positions, frames and geometry (cell lookup, draws and times stay double).
    """
    import jax.numpy as jnp
    if placement not in ("crystal", "geometric"):
        raise ValueError("placement must be crystal or geometric")
    if (isinstance(radius_neighbors, (bool, np.bool_)) or not isinstance(radius_neighbors, (int, np.integer))
            or radius_neighbors < 0):
        raise ValueError("radius_neighbors must be a nonnegative integer")
    binning = Binning(*(library["bins_" + k] for k in ("energy", "cosine", "radius")))
    disk_ids = np.asarray(library["geometry_disk_disk"])
    if np.any(np.diff(disk_ids) <= 0):
        raise ValueError("disk geometry must be sorted")
    axes = np.stack([np.column_stack([library["geometry_disk_axis_" + a + "_" + b] for b in "xyz"])
                     for a in "xyz"], axis=2)
    if not np.allclose(axes[:, :, 2], (0, 0, 1)):
        raise ValueError("device placement requires z-aligned disks")
    table, fallback, order, starts, counts, shape = _nearest_cell_table(library, disk_ids, binning)
    radius = np.asarray(library["entry_radius"], np.float64)
    radius_order = np.concatenate([order[s:s + c][np.argsort(radius[order[s:s + c]], kind="stable")]
                                   for s, c in zip(starts, counts)]) if len(counts) else np.zeros(0, np.int64)
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xyz"])
    entry_disk_slot = np.searchsorted(disk_ids, library["entry_disk"])
    source_xy = np.asarray(library["entry_position"])[:, :2] - origin[np.minimum(entry_disk_slot, len(disk_ids) - 1), :2]
    centers = np.column_stack([library["geometry_crystal_position_" + a] for a in "xyz"])
    sizes = np.column_stack([library["geometry_crystal_size_" + a] for a in "xyz"])
    ids = np.asarray(library["geometry_crystal_crystal"])
    cdisk = np.asarray(library["geometry_crystal_disk"])
    sort = np.argsort(ids)
    step_crystal = np.asarray(library["step_crystal"])
    slot = np.searchsorted(ids[sort], step_crystal)
    if np.any(slot >= len(ids)) or not np.array_equal(ids[sort[np.minimum(slot, len(ids) - 1)]], step_crystal):
        raise ValueError("donor crystal is missing from saved geometry")
    lo, hi = np.asarray(library["geometry_disk_size_x"]), np.asarray(library["geometry_disk_size_y"])
    extent = float(np.max(hi)) + 4 * pixel
    n_pix = int(np.ceil(2 * extent / pixel))
    pix, zmin, zmax = [], [], []
    for d, disk in enumerate(disk_ids):
        pool = np.flatnonzero(cdisk == disk)
        if not len(pool):
            raise ValueError("geometry must contain each disk's crystals")
        r = axes[d]
        local = (centers[pool] - origin[d]) @ r
        cand = _pixel_candidates(local[:, :2], np.array([-extent, -extent]), pixel, n_pix, candidate_width,
                                (lo[d], hi[d]))
        pix.append(np.where(cand >= 0, pool[np.maximum(cand, 0)], -1))
        zmin.append(local[:, 2].min())
        zmax.append(np.max(local[:, 2] + sizes[pool, 2]))
    a = dict(
        edges_energy=binning.arrays()[0], edges_cosine=binning.arrays()[1], edges_radius=binning.arrays()[2],
        disk_ids=disk_ids, cell_table=table, cell_fallback=fallback, order=order, starts=starts, counts=counts,
        radius_order=radius_order, entry_radius=radius,
        offsets=np.asarray(library["offsets"], np.int64),
        donor_frame=frame(library["entry_direction"]), donor_angle=np.arctan2(source_xy[:, 1], source_xy[:, 0]),
        donor_position=np.asarray(library["entry_position"], np.float64),
        step_energy_g4=np.asarray(library["step_energy_g4"]), step_energy_birks=np.asarray(library["step_energy_birks"]),
        step_relative_position=np.asarray(library["step_relative_position"], np.float64),
        step_relative_time=np.asarray(library["step_relative_time"], np.float64),
        step_disk=np.asarray(library["step_disk"]), step_source=sort[slot],
        origin=origin, axes=axes, inner=lo, outer=hi, zmin=np.asarray(zmin), zmax=np.asarray(zmax),
        centers=centers, sizes=sizes, crystal_ids=ids, crystal_disk=cdisk, pixel_candidates=np.stack(pix),
    )
    meta = dict(entries=int(len(radius)), steps=int(len(step_crystal)), occupied_cells=int(len(counts)),
                max_cell=int(counts.max(initial=0)))
    for k in ("donor_frame", "donor_angle", "donor_position", "step_relative_position", "origin", "axes", "inner",
              "outer", "zmin", "zmax", "centers", "sizes"):
        a[k] = np.asarray(a[k], dtype)
    meta["dtype"] = np.dtype(dtype).name
    return DeviceLibrary({k: jnp.asarray(v) for k, v in a.items()}, shape, int(radius_neighbors), placement,
                         n_pix, float(pixel), meta)


@dataclass
class DeviceResponse:
    arrays: dict
    pulse: PulseCache
    ro_cfg: ReadoutConfig
    digi_cfg: DigiConfig
    fit_cfg: FitConfig
    cluster_cfg: ClusterConfig
    max_chi2: float
    time4merge: float
    pbt_reco_sigma: float
    add_noise: bool
    noise_size: int
    n_crystals: int
    dtype: str = "float64"
    fit_hessian: str | None = "analytic"   # template fit Newton step: "analytic" = closed form, None = autodiff


def pack_response(sim, pulse: PulseCache, noise_wf, table: PropagationTable, geom: Geometry, *, adc2mev=None,
                  ro_cfg=ReadoutConfig(), digi_cfg=DigiConfig(), fit_cfg=FitConfig(), cluster_cfg=ClusterConfig(),
                  add_noise=True, max_chi2=2.0, time4merge=6.0, pbt_reco_sigma=0.5, dtype=np.float64,
                  fit_hessian="analytic"):
    """Calorimeter conditions as device arrays (``sim`` from :func:`calofastsim.stockio.load_sim_crystal`).

    ``dtype`` is the precision of the waveform, noise, template fit and cluster sums (the source uses double).
    Time folding, photoelectron times, keys and reco digi weighting keep the source types. ``fit_hessian`` picks the
    template fit's Newton gradient and Hessian (:func:`calofastsim.templatefit._compiled`): "analytic" (default: the
    closed form, the same quantities as autodiff up to roundoff and faster on GPUs) or None (autodiff)."""
    import jax.numpy as jnp
    if fit_hessian not in (None, "analytic"):
        raise ValueError("fit_hessian must be None or 'analytic'")
    n = geom.disk.size
    sipm = np.arange(2 * n)
    c, i = sipm // 2, sipm % 2
    scale = (sim["adc_per_mev"][c, i] / sim["pe_per_mev"][c, i]).astype(np.float64)
    reco = np.full(2 * n, 0.0625) if adc2mev is None else np.asarray(adc2mev(sipm))
    noise = np.zeros(1) if noise_wf is None else np.asarray(noise_wf, np.float64)
    ped = np.float64(noise_pedestal(noise)) if add_noise else np.float64(0.0)
    a = dict(lru=np.asarray(sim["lru"], np.float32)[:n], yields=np.asarray(sim["pe_per_mev"], np.float32)[:n],
             adc_per_pe=scale, adc2mev=reco, noise=noise.astype(dtype), pulse_vec=np.asarray(pulse.pulse_vec, dtype),
             pedestal=np.asarray(ped, dtype),
             cdf=np.asarray(table.cdf, np.float32), times=np.asarray(table.times, np.float32),
             position=np.asarray(geom.position, dtype), disk=np.asarray(geom.disk),
             neighbors=np.asarray(geom.neighbors, bool), front_face=np.asarray(geom.front_face, dtype))
    out = {k: jnp.asarray(v) for k, v in a.items()}
    out["dz"] = float(table.dz)
    return DeviceResponse(out, pulse, ro_cfg, digi_cfg, fit_cfg, cluster_cfg, float(max_chi2), float(time4merge),
                          float(pbt_reco_sigma), bool(add_noise), int(noise.size), int(n), np.dtype(dtype).name,
                          fit_hessian)


def entries_batch(entries, timing, event_row, n_events):
    """Pad calorimeter-entrance states (one per event at most) and per-event timing to ``n_events`` rows.

    ``entries``: dict of energy, cosine, radius, disk, position (N,3), direction (N,3), time; ``event_row`` the
    batch-local event of each entry. ``timing``: per-event pbtime, mbtime, event_length, onspill, marker,
    crystal_length (length <= n_events). Returns the two dicts of NumPy arrays :func:`run` takes.
    """
    m = len(entries["energy"])
    nt = len(timing["pbtime"])
    if m > n_events or nt > n_events:
        raise ValueError("batch larger than its event capacity")
    if len(np.unique(event_row)) != m:
        raise ValueError("one entry per event is required")
    q = {"valid": np.zeros(n_events, bool), "event": np.zeros(n_events, np.int64)}
    q["valid"][:m] = True
    q["event"][:m] = event_row
    for k, shape, fill in (("energy", (), 1.0), ("cosine", (), 1.0), ("radius", (), 0.0), ("disk", (), 0),
                           ("time", (), 0.0), ("position", (3,), 0.0), ("direction", (3,), 0.0)):
        v = np.asarray(entries[k])
        a = np.full((n_events,) + shape, fill, np.int64 if k == "disk" else np.float64)
        if k == "direction":
            a[:, 2] = 1.0
        a[:m] = v
        q[k] = a
    t = {}
    for k, fill, dtype in (("pbtime", 0.0, np.float32), ("mbtime", 1695.0, np.float32),
                           ("event_length", 1695.0, np.float64), ("onspill", True, bool),
                           ("marker", 200.0, np.float32), ("crystal_length", 200.0, np.float32)):
        a = np.full(n_events, fill, dtype)
        a[:nt] = np.asarray(timing[k], dtype)
        t[k] = a
    return q, t


# ----------------------------------------------------------------------------------------------- device stages

def _bsearch(values, lo, hi, x, side, iters):
    """Vectorised searchsorted of ``x`` in ``values[lo:hi]`` (per-element segments); returns an absolute index."""
    import jax.numpy as jnp
    for _ in range(iters):
        mid = (lo + hi) // 2
        v = values[jnp.clip(mid, 0, values.shape[0] - 1)]
        go = (v < x) if side == "left" else (v <= x)
        active = lo < hi
        lo = jnp.where(active & go, mid + 1, lo)
        hi = jnp.where(active & ~go, mid, hi)
    return lo


def sample_donors(lib: DeviceLibrary, q, uniform):
    """Donor library row per query (-1 without disk support), fallback flag; ``uniform`` in [0,1) per query."""
    import jax.numpy as jnp
    a = lib.arrays
    values = jnp.stack([q["energy"], q["cosine"], q["radius"]], axis=1)
    edges = (a["edges_energy"], a["edges_cosine"], a["edges_radius"])
    idx, outside = [], jnp.zeros(values.shape[0], bool)
    for k, e in enumerate(edges):
        idx.append(jnp.clip(jnp.searchsorted(e, values[:, k], side="right") - 1, 0, e.shape[0] - 2))
        outside = outside | (values[:, k] < e[0]) | (values[:, k] > e[-1])
    ne, nc, nr = lib.shape
    flat = (idx[0] * nc + idx[1]) * nr + idx[2]
    slot = jnp.clip(jnp.searchsorted(a["disk_ids"], q["disk"]), 0, a["disk_ids"].shape[0] - 1)
    known = a["disk_ids"][slot] == q["disk"]
    occ = jnp.where(known, a["cell_table"][slot, flat], -1)
    fallback = outside | ~known | a["cell_fallback"][slot, flat]
    supported = occ >= 0
    o = jnp.maximum(occ, 0)
    start, count = a["starts"][o], a["counts"][o]
    k_neighbors = lib.radius_neighbors
    if not k_neighbors:
        draw = (uniform * count).astype(jnp.int64)
        donor = a["order"][start + draw]
    else:
        size = jnp.minimum(k_neighbors, count)
        draw = (uniform * size).astype(jnp.int64)
        plain = a["order"][start + draw]
        rr = a["entry_radius"][a["radius_order"]]
        iters = int(max(lib.meta["max_cell"], 1)).bit_length() + 1
        qr = q["radius"]
        split = _bsearch(rr, start, start + count, qr, "left", iters) - start
        K = k_neighbors
        ar = jnp.arange(K)
        left = split[:, None] - K + ar
        right = split[:, None] + ar
        valid_left = left >= 0
        valid_right = right < count[:, None]
        left = jnp.maximum(left, 0)
        lv = rr[jnp.clip(start[:, None] + left, 0, rr.shape[0] - 1)]
        first = _bsearch(rr, jnp.broadcast_to(start[:, None], left.shape), jnp.broadcast_to((start + count)[:, None],
                         left.shape), lv, "left", iters) - start[:, None]
        left = first + left - jnp.maximum(first, jnp.maximum(split - K, 0)[:, None])
        pos = jnp.concatenate([left, jnp.minimum(right, count[:, None] - 1)], axis=1)
        cand = a["radius_order"][jnp.clip(start[:, None] + pos, 0, a["radius_order"].shape[0] - 1)]
        dist = jnp.abs(a["entry_radius"][cand] - qr[:, None])
        dist = jnp.where(jnp.concatenate([valid_left, valid_right], axis=1), dist, jnp.inf)
        rank = jnp.lexsort((cand, dist), axis=1)[:, :K]
        eligible = jnp.sort(jnp.take_along_axis(cand, rank, axis=1), axis=1)
        pooled = jnp.take_along_axis(eligible, jnp.minimum(draw, K - 1)[:, None], axis=1)[:, 0]
        donor = jnp.where(count <= K, plain, pooled)
    donor = jnp.where(supported & q["valid"], donor, -1)
    return donor, fallback, slot


def gather_steps(lib: DeviceLibrary, q, donor, slot, n_steps):
    """Sampled CSS (disk orientation) in flat capacity ``n_steps``; returns the step dict and the overflow count."""
    import jax.numpy as jnp
    a = lib.arrays
    ok = donor >= 0
    d = jnp.maximum(donor, 0)
    length = jnp.where(ok, a["offsets"][d + 1] - a["offsets"][d], 0)
    end = jnp.cumsum(length)
    total = end[-1]
    i = jnp.arange(n_steps)
    row = jnp.clip(jnp.searchsorted(end, i, side="right"), 0, length.shape[0] - 1)
    valid = i < total
    src = a["offsets"][d[row]] + i - (end[row] - length[row])
    src = jnp.where(valid, src, 0)
    rel = a["step_relative_position"][src]
    dframe = a["donor_frame"][d[row]]
    delta = jnp.einsum("nij,nj->ni", dframe, rel, precision=_physics(dframe, rel))
    s = slot[row]
    target_xy = q["position"][row, :2] - a["origin"][s, :2]
    angle = jnp.arctan2(target_xy[:, 1], target_xy[:, 0]) - a["donor_angle"][d[row]]
    c, sn = jnp.cos(angle), jnp.sin(angle)
    x, y = delta[:, 0], delta[:, 1]
    delta = jnp.stack([c * x - sn * y, sn * x + c * y, delta[:, 2]], axis=1)
    steps = dict(valid=valid, row=row, donor=d[row], angle_c=c, angle_s=sn, source=a["step_source"][src],
                 position=q["position"][row] + delta, time=q["time"][row] + a["step_relative_time"][src],
                 energy_g4=a["step_energy_g4"][src], energy_birks=a["step_energy_birks"][src],
                 disk=a["step_disk"][src])
    return steps, jnp.maximum(total - n_steps, 0)


def _nearest_crystal(lib: DeviceLibrary, dslot, xy_local):
    """Nearest same-disk crystal centre (geometry index) and its distance, from the pixel candidate table."""
    import jax.numpy as jnp
    a = lib.arrays
    extent = lib.n_pix * lib.pixel / 2
    ij = jnp.clip(jnp.floor((xy_local + extent) / lib.pixel).astype(jnp.int64), 0, lib.n_pix - 1)
    cand = a["pixel_candidates"][dslot, ij[:, 0] * lib.n_pix + ij[:, 1]]
    rel, axes = a["centers"][jnp.maximum(cand, 0)] - a["origin"][dslot][:, None, :], a["axes"][dslot]
    cl = jnp.einsum("nkj,nji->nki", rel, axes, precision=_physics(rel, axes))
    dist = jnp.linalg.norm(cl[:, :, :2] - xy_local[:, None, :], axis=2)
    dist = jnp.where(cand >= 0, dist, jnp.inf)
    k = jnp.argmin(dist, axis=1)
    return jnp.take_along_axis(cand, k[:, None], axis=1)[:, 0], jnp.take_along_axis(dist, k[:, None], axis=1)[:, 0]


def place_steps(lib: DeviceLibrary, q, st):
    """Crystal index (geometry row, -1 unmapped) and crystal-local depth of each sampled step."""
    import jax.numpy as jnp
    a = lib.arrays
    eps = 1e-6
    ds = jnp.clip(jnp.searchsorted(a["disk_ids"], st["disk"]), 0, a["disk_ids"].shape[0] - 1)
    r = a["axes"][ds]
    org = a["origin"][ds]

    def to_local(v):
        return jnp.einsum("nj,nji->ni", v, r, precision=_physics(v, r))

    if lib.placement == "geometric":
        # every pixel candidate is tested for containment; the closest containing centre wins, as place()
        xyz = to_local(st["position"] - org)
        extent = lib.n_pix * lib.pixel / 2
        ij = jnp.clip(jnp.floor((xyz[:, :2] + extent) / lib.pixel).astype(jnp.int64), 0, lib.n_pix - 1)
        cand = a["pixel_candidates"][ds, ij[:, 0] * lib.n_pix + ij[:, 1]]
        cc = jnp.maximum(cand, 0)
        rel = st["position"][:, None, :] - a["centers"][cc]
        local = jnp.einsum("nkj,nji->nki", rel, r, precision=_physics(rel, r))
        size = a["sizes"][cc]
        inside = ((cand >= 0) & (jnp.abs(local[:, :, :2]) <= size[:, :, :2] / 2 + eps).all(axis=2) &
                  (local[:, :, 2] >= -eps) & (local[:, :, 2] <= size[:, :, 2] + eps))
        dist = jnp.where(inside, jnp.linalg.norm(local[:, :, :2], axis=2), jnp.inf)
        k = jnp.argmin(dist, axis=1)
        mapped = inside.any(axis=1) & st["valid"]
        target = jnp.take_along_axis(cc, k[:, None], axis=1)[:, 0]
        depth = jnp.take_along_axis(local[:, :, 2], k[:, None], axis=1)[:, 0]
        return jnp.where(mapped, target, -1), depth, mapped
    c, s = st["angle_c"], st["angle_s"]
    offset = a["centers"][st["source"]] - a["donor_position"][st["donor"]]
    rotated = jnp.stack([c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1], offset[:, 2]], 1)
    ideal = q["position"][st["row"]] + rotated
    delta = st["position"] - ideal
    unrot = jnp.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1], delta[:, 2]], 1)
    center_local = to_local(ideal - org)
    target, dist = _nearest_crystal(lib, ds, center_local[:, :2])
    local = to_local(unrot)
    raw = to_local(st["position"] - org)
    lo, hi = a["inner"][ds], a["outer"][ds]
    radial = jnp.linalg.norm(raw[:, :2], axis=1)
    center_radial = jnp.linalg.norm(center_local[:, :2], axis=1)
    envelope = ((radial >= lo) & (radial <= hi) & (center_radial >= lo) & (center_radial <= hi) &
                (raw[:, 2] >= a["zmin"][ds] - eps) & (raw[:, 2] <= a["zmax"][ds] + eps))
    size = a["sizes"][target]
    close = dist <= jnp.linalg.norm(size[:, :2], axis=1) / 2 + eps
    contains = ((jnp.abs(local[:, :2]) <= size[:, :2] / 2 + eps).all(axis=1) &
                (local[:, 2] >= -eps) & (local[:, 2] <= size[:, 2] + eps))
    proposed = a["centers"][target] + jnp.einsum("ni,nji->nj", local, r, precision=_physics(local, r))
    pr = jnp.linalg.norm(to_local(proposed - org)[:, :2], axis=1)
    envelope = envelope & (pr >= lo) & (pr <= hi)
    mapped = envelope & close & contains & st["valid"]
    depth = to_local(proposed - a["centers"][target])[:, 2]
    return jnp.where(mapped, target, -1), depth, mapped


def photoelectrons(resp: DeviceResponse, q, t, st, crystal, depth, mapped, key, n_pe):
    """Per-SiPM photoelectrons (readout.make_readouts): returns event, sipm, float32 time, weight, valid, overflow."""
    import jax
    import jax.numpy as jnp
    from . import response_jax as rj
    from .propagation_jax import propagation_time
    a = resp.arrays
    cfg = resp.ro_cfg
    ev = q["event"][st["row"]]
    cr = jnp.maximum(crystal, 0)
    hit, accepted = rj.fold_step_time(
        st["time"], pbtime=t["pbtime"][ev], mbtime=t["mbtime"][ev], event_length=t["event_length"][ev],
        onspill=t["onspill"][ev], digitization_start=cfg.digitization_start, digitization_end=cfg.digitization_end,
        digitization_buffer=cfg.digitization_buffer, time_from_protons_to_dr_marker=t["marker"][ev])
    keep = mapped & accepted
    energy = rj.corrected_energy(st["energy_g4"], st["energy_birks"], depth, t["crystal_length"][ev], a["lru"][cr],
                                 birks=cfg.birks, longitudinal=cfg.lru)
    mean = rj.photoelectron_mean(energy, a["yields"][cr])
    k_pois, k_trav = jax.random.split(key)
    if cfg.pe_stat:
        count = jax.random.poisson(k_pois, mean.astype(jnp.float64), mean.shape).astype(jnp.int64)
    else:
        count = rj.nonnegative_lround(mean).astype(jnp.int64)
    count = jnp.where(keep[:, None], count, 0).reshape(-1)
    group_step = jnp.arange(count.shape[0]) // 2
    group_sipm = 2 * cr[group_step] + jnp.arange(count.shape[0]) % 2
    t32 = hit.astype(jnp.float32)
    if not cfg.travel:
        valid = count > 0
        return ev[group_step], group_sipm, t32[group_step], count.astype(jnp.float64), valid, jnp.int64(0)
    end = jnp.cumsum(count)
    total = end[-1]
    j = jnp.arange(n_pe)
    g = jnp.clip(jnp.searchsorted(end, j, side="right"), 0, count.shape[0] - 1)
    valid = j < total
    s = group_step[g]
    dpt = t["crystal_length"][ev[s]].astype(jnp.float32) - depth.astype(jnp.float32)[s]
    u = jax.random.uniform(k_trav, (n_pe,), dtype=jnp.float32)
    dt = propagation_time(dpt, u, a["cdf"], a["times"], a["dz"])
    return ev[s], group_sipm[g], (t32[s] + dt).astype(jnp.float32), jnp.ones(n_pe), valid, jnp.maximum(total - n_pe, 0)


def _waveforms(resp: DeviceResponse, row, time32, weight, n_rows, window, outlier_cap, size=None, shift=None):
    """Waveforms (n_rows, size), float64, as digi_device: each PE adds ``pulse_vec[first + 100 k]`` at sample
    ``start + k``. PEs within ``window`` samples of their row's earliest start are histogrammed by (sample offset,
    fine phase) and convolved with the pulse table in one matmul; the rest (late or pre-window PEs) are compacted
    (at most ``outlier_cap``) and scattered per PE. Only the summation order differs from the per-PE scatter.

    ``size``/``shift`` (opt-in): each row holds only ``size`` samples starting at its global sample ``shift[row]``
    (a window that must contain the row's signal; see :func:`digitise`)."""
    import jax.numpy as jnp
    from jax import lax
    cfg = resp.digi_cfg
    pulse = resp.pulse
    nshape = pulse.n_bin_shape
    size = cfg.waveform_size if size is None else size
    vec = resp.arrays["pulse_vec"]
    fdt = jnp.dtype(resp.dtype)
    weight = weight.astype(fdt)
    start = jnp.trunc(time32 / jnp.float32(cfg.digi_sampling)).astype(jnp.int64)
    keep = start >= 0
    q = jnp.trunc(time32.astype(jnp.float64) / pulse.digi_step).astype(jnp.int64)
    first = N_STEPS - jnp.fmod(q, N_STEPS)
    w = jnp.where(keep, weight, 0.0)
    big = jnp.iinfo(jnp.int64).max
    s0 = jnp.full(n_rows, big, jnp.int64).at[row].min(jnp.where(keep, start, big))
    ds = start - s0[row]
    inwin = keep & (q >= 0) & (ds < window)
    # windowed part: H[row, ds, phase] -> A[row, ds, k] = sum_phase H * vec[phase + 100 k]
    flat = (row * window + jnp.where(inwin, ds, 0)) * N_STEPS + jnp.where(inwin, first - 1, 0)
    H = jnp.zeros(n_rows * window * N_STEPS, fdt).at[flat].add(jnp.where(inwin, w, 0.0))
    V = vec[jnp.clip(jnp.arange(1, N_STEPS + 1)[:, None] + N_STEPS * jnp.arange(nshape)[None, :], 0, vec.size - 1)]
    A = jnp.matmul(H.reshape(n_rows * window, N_STEPS), V, precision=_physics(H, V)).reshape(n_rows, window, nshape)
    width = size + window + nshape
    wave = jnp.zeros((n_rows, width), fdt)
    if shift is not None:                    # window coordinates: samples counted from the row's window origin
        start = start - shift[row]
        s0 = jnp.where(s0 < big, s0 - shift, big)
    base = jnp.where(s0 < big, s0, size)[:, None]
    for d in range(window):
        cols = jnp.minimum(base + d + jnp.arange(nshape)[None, :], width - 1)
        wave = wave.at[jnp.arange(n_rows)[:, None], cols].add(A[:, d, :])
    # outliers: compacted per-PE scatter
    out = keep & ~inwin
    pos = jnp.cumsum(out) - 1
    n_out = jnp.sum(out)
    slot = jnp.where(out & (pos < outlier_cap), pos, outlier_cap)
    o_row = jnp.zeros(outlier_cap + 1, jnp.int64).at[slot].set(row)[:outlier_cap]
    o_start = jnp.zeros(outlier_cap + 1, jnp.int64).at[slot].set(start)[:outlier_cap]
    o_first = jnp.zeros(outlier_cap + 1, jnp.int64).at[slot].set(first)[:outlier_cap]
    o_w = jnp.zeros(outlier_cap + 1, fdt).at[slot].set(w)[:outlier_cap]
    flat_wave = wave.reshape(-1)

    def add_sample(k, acc):
        sample = o_start + k
        ok = sample < size
        idx = jnp.where(ok, o_row * width + sample, n_rows * width - 1)
        return acc.at[idx].add(jnp.where(ok, vec[jnp.clip(o_first + N_STEPS * k, 0, vec.size - 1)] * o_w,
                                         jnp.zeros((), fdt)))

    flat_wave = lax.fori_loop(0, nshape, add_sample, flat_wave)
    return flat_wave.reshape(n_rows, width)[:, :size], jnp.maximum(n_out - outlier_cap, 0)


def _first_peaks_window(wf, cfg, offset, size_g):
    """digi.first_peaks on window rows whose sample 0 is global sample ``offset[row]``: the peak range
    [nBinsPeak + bufferDigi, size - nBinsPeak) is checked in global samples (the window holds the whole signal, so the
    samples outside it are zero and below every threshold)."""
    import jax.numpy as jnp
    from jax import lax
    n, size = wf.shape
    nb = cfg.n_bins_peak
    thr = wf >= cfg.min_peak_adc
    pad = jnp.concatenate([jnp.full((n, nb), -1, dtype=wf.dtype), wf, jnp.full((n, nb), -1, dtype=wf.dtype)], axis=1)
    cand = thr
    for d in range(1, nb + 1):
        cand = cand & (wf > pad[:, nb - d:nb - d + size]) & (wf >= pad[:, nb + d:nb + d + size])
    pos = jnp.arange(size)
    pos_g = offset[:, None] + pos[None, :]
    cand = cand & (pos_g >= nb + cfg.buffer_digi) & (pos_g + nb < size_g)
    prev = jnp.concatenate([jnp.zeros((n, 1), dtype=bool), thr[:, :-1]], axis=1)
    run_start = thr & ~prev
    csum = jnp.cumsum(cand.astype(jnp.int64), axis=1)
    base = lax.cummax(jnp.where(run_start, csum - cand.astype(jnp.int64), 0), axis=1)
    first = cand & ((csum - base) == 1)
    run_end = lax.cummin(jnp.where(~thr, pos, size), axis=1, reverse=True)
    return first, run_end


def _digi_post(resp: DeviceResponse, wave, key, window_origin=None):
    """digi_device's noise, ADC and peak search on given waveforms (same code path, factored). ``window_origin``
    (opt-in): window rows (:func:`_waveforms` ``shift``), peaks bounded in global samples."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from .digi import adc_waveform, first_peaks, noise_coverage
    cfg = resp.digi_cfg
    a = resp.arrays
    n_rows, size = wave.shape
    noise_size = resp.noise_size
    cover = noise_coverage(wave, cfg, backend="jax")
    pos = jnp.arange(size)
    prev = jnp.concatenate([jnp.zeros((n_rows, 1), bool), cover[:, :-1]], axis=1)
    nxt = jnp.concatenate([cover[:, 1:], jnp.zeros((n_rows, 1), bool)], axis=1)
    starts = cover & ~prev
    last = lax.cummin(jnp.where(cover & ~nxt, pos, size), axis=1, reverse=True)
    length = last - pos + 1
    u = jax.random.uniform(key, (n_rows, size), dtype=jnp.float64)
    offset = jnp.trunc(u * (noise_size - length)).astype(jnp.int64)
    anchor = lax.cummax(jnp.where(starts, pos, -1), axis=1)
    an = jnp.maximum(anchor, 0)
    idx = jnp.take_along_axis(offset, an, axis=1) + pos - an
    if resp.add_noise:
        wave = wave + jnp.where(cover, a["noise"][jnp.clip(idx, 0, noise_size - 1)], 0.0)
    wf = adc_waveform(wave, a["pedestal"], cfg, backend="jax")
    if window_origin is None:
        peaks, run_end = first_peaks(wf, cfg, backend="jax")
    else:
        peaks, run_end = _first_peaks_window(wf, cfg, window_origin, cfg.waveform_size)
    stored = jnp.minimum(jnp.trunc(wave), cfg.max_adc).astype(jnp.int32)
    return stored, peaks, run_end.astype(jnp.int32)


def digitise(resp: DeviceResponse, t, ev, sipm, pe_t, weight, valid, key, caps: Caps):
    """Waveforms, noise, ADC and peaks (digi_device kernel) on (event, SiPM) rows; then digis in flat capacity."""
    import jax.numpy as jnp
    a = resp.arrays
    cfg = resp.digi_cfg
    f32 = jnp.float32
    nsipm = 2 * resp.n_crystals
    big = jnp.iinfo(jnp.int64).max
    k = jnp.where(valid, ev * nsipm + sipm, big)
    sk = jnp.sort(k)
    distinct = jnp.sum((sk != big) & jnp.concatenate([jnp.ones(1, bool), sk[1:] != sk[:-1]]))
    ukey = jnp.unique(k, size=caps.rows, fill_value=big)
    row = jnp.clip(jnp.searchsorted(ukey, k), 0, caps.rows - 1)
    row_ok = (ukey[row] == k) & valid
    tt = pe_t + t["pbtime"][ev].astype(f32)
    tt = tt - f32(cfg.digitization_start)
    tt = tt + t["marker"][ev].astype(f32)
    tt = (tt + cfg.start_time_buffer).astype(f32)
    tt = jnp.where(row_ok, tt, f32(-1e6))
    w = jnp.where(row_ok, weight * a["adc_per_pe"][sipm], 0.0).astype(jnp.dtype(resp.dtype))
    if caps.wave_window:
        return _digitise_window(resp, t, ukey, row, tt, w, key, caps, distinct)
    wave, over_out = _waveforms(resp, row, tt, w, caps.rows, caps.window, caps.outliers)
    stored, peaks, run_end = _digi_post(resp, wave, key)
    dg, n_digi, too_long = _extract_digis(resp, t, stored, peaks, run_end, None, ukey, caps.digis, caps.digi_length)
    overflow = dict(rows=jnp.maximum(distinct - caps.rows, 0), outliers=over_out,
                    digis=jnp.maximum(n_digi - caps.digis, 0), digi_length=too_long)
    return dg, overflow


def _extract_digis(resp: DeviceResponse, t, stored, peaks, run_end, offset, row_key, D, L):
    """digi.extract_digis on device rows: merge peaks whose buffered start overlaps the previous stop, then cut the
    samples (at most ``D`` digis, ``L`` samples each). ``offset``: per-row global sample of column 0 (None: 0).
    ``row_key``: event * nSiPM + SiPM of each row. Returns the digi dict, the digi count and the over-long count."""
    import jax.numpy as jnp
    from jax import lax
    cfg = resp.digi_cfg
    f32 = jnp.float32
    nsipm = 2 * resp.n_crystals
    n_rows, size = stored.shape
    pos = jnp.arange(size)
    m = jnp.where(peaks, run_end, -1)
    excl = jnp.concatenate([jnp.full((n_rows, 1), -1, m.dtype), lax.cummax(m, axis=1)[:, :-1]], axis=1)
    new = peaks & ((pos - cfg.buffer_digi)[None, :] > excl)
    flat_new = new.reshape(-1)
    d_of = jnp.cumsum(flat_new) - 1
    n_digi = d_of[-1] + 1
    sink = jnp.where(peaks.reshape(-1) & (d_of < D), d_of, D)
    first = jnp.where(flat_new & (d_of < D), d_of, D)
    rr = jnp.broadcast_to(jnp.arange(n_rows)[:, None], peaks.shape).reshape(-1)
    pp = jnp.broadcast_to(pos[None, :], peaks.shape).reshape(-1)
    d_row = jnp.zeros(D + 1, jnp.int64).at[first].set(rr)[:D]
    d_start = jnp.zeros(D + 1, jnp.int64).at[first].set(pp - cfg.buffer_digi)[:D]          # column of the row
    d_stop = jnp.zeros(D + 1, jnp.int64).at[sink].max(run_end.reshape(-1).astype(jnp.int64))[:D]
    d_ok = jnp.arange(D) < n_digi
    length = jnp.where(d_ok, d_stop - d_start, 0)
    too_long = jnp.sum(length > L)
    length = jnp.minimum(length, L)
    u_row = row_key[d_row]
    d_event = jnp.where(d_ok, u_row // nsipm, 0)
    d_sipm = jnp.where(d_ok, u_row % nsipm, 0)
    marker = t["marker"][d_event].astype(f32)
    g_start = d_start if offset is None else d_start + offset[d_row]                        # global sample
    t0 = jnp.trunc(g_start.astype(f32) * f32(cfg.digi_sampling) + f32(cfg.digitization_start) - marker
                   - cfg.start_time_buffer).astype(jnp.int64)
    kk = jnp.arange(L)
    col = jnp.clip(d_start[:, None] + kk[None, :], 0, size - 1)
    samples = jnp.where(kk[None, :] < length[:, None], stored[d_row[:, None], col], 0)
    return dict(ok=d_ok, event=d_event, sipm=d_sipm, t0=t0, samples=samples, length=length), n_digi, too_long


def _digitise_window(resp: DeviceResponse, t, ukey, row, tt, w, key, caps: Caps, distinct):
    """digitise with ``caps.wave_window`` samples per row (opt-in): each row's window starts bufferDigi + nBinsPeak
    samples before its first photoelectron and holds its signal; rows whose photoelectrons spread past the window
    (late deposits) are digitised on the full waveform, compacted (at most ``caps.wide_rows``). The digis are those
    of the full-waveform path; only the noise offsets come from other random numbers (same distribution)."""
    import jax
    import jax.numpy as jnp
    cfg = resp.digi_cfg
    R, Ww, size = caps.rows, caps.wave_window, cfg.waveform_size
    nshape = resp.pulse.n_bin_shape
    big = jnp.iinfo(jnp.int64).max
    start = jnp.trunc(tt / jnp.float32(cfg.digi_sampling)).astype(jnp.int64)
    keep = start >= 0
    s0 = jnp.full(R, big, jnp.int64).at[row].min(jnp.where(keep, start, big))
    s1 = jnp.full(R, -1, jnp.int64).at[row].max(jnp.where(keep, start, -1))
    used = s1 >= 0
    ws = jnp.where(used, jnp.clip(s0 - (cfg.buffer_digi + cfg.n_bins_peak), 0, max(size - Ww, 0)), 0)
    wide = used & (jnp.minimum(s1 + nshape, size) > ws + Ww)
    k_n, k_w = jax.random.split(key)
    narrow_pe = keep & ~wide[row]
    wave_n, over_n = _waveforms(resp, row, jnp.where(narrow_pe, tt, jnp.float32(-1e6)), w, R, caps.window,
                                caps.outliers, size=Ww, shift=ws)
    stored_n, peaks_n, run_end_n = _digi_post(resp, wave_n, k_n, window_origin=ws)
    dn, nd_n, long_n = _extract_digis(resp, t, stored_n, peaks_n, run_end_n, ws, ukey, caps.digis, caps.digi_length)
    Rw = caps.wide_rows
    wpos = jnp.cumsum(wide) - 1
    n_wide = jnp.sum(wide)
    wslot = jnp.where(wide & (wpos < Rw), wpos, Rw)
    wide_pe = keep & (wslot[row] < Rw)
    wave_w, over_w = _waveforms(resp, jnp.where(wide_pe, wslot[row], 0), jnp.where(wide_pe, tt, jnp.float32(-1e6)),
                                w, Rw, caps.window, caps.outliers)
    stored_w, peaks_w, run_end_w = _digi_post(resp, wave_w, k_w)
    wrow = jnp.zeros(Rw + 1, jnp.int64).at[wslot].set(jnp.arange(R))[:Rw]
    peaks_w = peaks_w & (jnp.arange(Rw) < n_wide)[:, None]
    dw, nd_w, long_w = _extract_digis(resp, t, stored_w, peaks_w, run_end_w, None, ukey[wrow], caps.wide_digis,
                                      caps.digi_length)
    dg = {k: jnp.concatenate([dn[k], dw[k]]) for k in dn}
    overflow = dict(rows=jnp.maximum(distinct - R, 0) + jnp.maximum(n_wide - Rw, 0), outliers=over_n + over_w,
                    digis=jnp.maximum(nd_n - caps.digis, 0) + jnp.maximum(nd_w - caps.wide_digis, 0),
                    digi_length=long_n + long_w)
    return dg, overflow


def _fit_digis(resp: DeviceResponse, cfg, caps: Caps, fit_dt, x, y, mask, length, ok, L):
    """Template fit of padded digis (columns ``L``): one pass with the refit, or (``caps.refits``) every digi without
    the refit and again, with it, only the digis that need it. Returns the fit dict and the refit overflow."""
    import jax
    import jax.numpy as jnp
    from .templatefit import _compiled
    if not caps.refits:
        with jax.ensure_compile_time_eval():   # the cached fit must not close over this trace
            fitter = _compiled(resp.pulse, cfg, L, fit_dt, resp.fit_hessian)
        return fitter(x, y, mask, length), jnp.int64(0)
    # the refit minimisation only where it is used: the same arithmetic per digi as the single pass
    with jax.ensure_compile_time_eval():
        first = _compiled(resp.pulse, cfg, L, fit_dt, resp.fit_hessian, refit=False)
        full = _compiled(resp.pulse, cfg, L, fit_dt, resp.fit_hessian)
    fit = first(x, y, mask, length)
    need = fit.pop("needs_refit") & ok
    R = caps.refits
    pos = jnp.cumsum(need) - 1
    n_need = jnp.sum(need)
    D0 = need.shape[0]
    sel = jnp.zeros(R + 1, jnp.int64).at[jnp.where(need & (pos < R), pos, R)].set(jnp.arange(D0))[:R]
    live = jnp.arange(R) < n_need
    again = full(x[sel], y[sel], mask[sel], length[sel])
    dest = jnp.where(live, sel, D0)
    return {k: v.at[dest].set(again[k], mode="drop") for k, v in fit.items()}, jnp.maximum(n_need - R, 0)


def fit_and_hits(resp: DeviceResponse, t, dg, pbt_reco, caps: Caps):
    """Template fit of every digi, reco digis (chain.run_chain rules), then hits (hits.make_hits rules)."""
    import jax
    import jax.numpy as jnp
    from .response_jax import weighted_reco_digis
    cfg = resp.fit_cfg
    L = caps.digi_length
    idx = jnp.arange(L)
    fdt = jnp.dtype(resp.dtype)
    x = dg["t0"].astype(fdt)[:, None] + (idx + 0.5) * cfg.digi_sampling
    mask = idx[None, :] < dg["length"][:, None]
    fit_dt = None if fdt == jnp.float64 else fdt
    y = dg["samples"].astype(fdt)
    over_bucket = jnp.int64(0)
    if not caps.fit_short_length:
        fit, over_refit = _fit_digis(resp, cfg, caps, fit_dt, x, y, mask, dg["length"], dg["ok"], L)
    else:
        # two length buckets: digis of at most fit_short_length samples are fitted on that many columns (the fit only
        # reads samples below a digi's length), the rest on all digi_length; results written back in digi order
        Ls = caps.fit_short_length
        short = dg["ok"] & (dg["length"] <= Ls)
        long_ = dg["ok"] & ~short
        D0 = short.shape[0]
        fit, over_refit = None, jnp.int64(0)
        for sel_b, cap_b, Lb in ((short, caps.fit_short, Ls), (long_, caps.fit_long, L)):
            pos = jnp.cumsum(sel_b) - 1
            idx = jnp.zeros(cap_b + 1, jnp.int64).at[jnp.where(sel_b & (pos < cap_b), pos, cap_b)].set(
                jnp.arange(D0))[:cap_b]
            live = jnp.arange(cap_b) < jnp.sum(sel_b)
            fb, ob = _fit_digis(resp, cfg, caps, fit_dt, x[idx, :Lb], y[idx, :Lb], mask[idx, :Lb],
                                jnp.where(live, dg["length"][idx], 0), live, Lb)
            dest = jnp.where(live, idx, D0)
            if fit is None:
                fit = {k: jnp.zeros((D0,) + v.shape[1:], v.dtype) for k, v in fb.items()}
            fit = {k: v.at[dest].set(fb[k], mode="drop") for k, v in fit.items()}
            over_refit = over_refit + ob
            over_bucket = over_bucket + jnp.maximum(jnp.sum(sel_b) - cap_b, 0)
    ftime = fit["peak_time"] - resp.pulse.delta_t - 0.5 * resp.pulse.digi_step * N_STEPS
    chi_ok = (fit["chi2"] / jnp.maximum(fit["ndf"], 1).astype(jnp.float32)) <= resp.max_chi2
    sel = fit["valid"] & chi_ok[:, None] & dg["ok"][:, None]
    D, K = sel.shape
    f32 = jnp.float32
    scale = resp.arrays["adc2mev"][dg["sipm"]][:, None]
    ev = jnp.broadcast_to(dg["event"][:, None], (D, K))
    e = (fit["amp"] * scale).astype(f32)
    e_err = (fit["amp_err"] * scale).astype(f32)
    tr = (ftime - pbt_reco[ev].astype(jnp.float64)).astype(f32)
    t_err = fit["time_err"].astype(f32)
    finite = jnp.isfinite(e) & jnp.isfinite(e_err) & jnp.isfinite(tr) & jnp.isfinite(t_err) & (e_err > 0) & (t_err > 0)
    n_nonfinite = jnp.sum(sel & ~finite)
    sel = (sel & finite).reshape(-1)
    ev, e, e_err, tr, t_err = (v.reshape(-1) for v in (ev, e, e_err, tr, t_err))
    crystal = jnp.broadcast_to((dg["sipm"] // 2)[:, None], (D, K)).reshape(-1)
    # hits: sort by (event, crystal, float32 time); drop cells with < 2 reco digis; group rounds per cell
    big = jnp.iinfo(jnp.int64).max
    cellkey = jnp.where(sel, ev * resp.n_crystals + crystal, big)
    order = jnp.lexsort((jnp.where(sel, tr, jnp.inf), cellkey))
    ck, ts = cellkey[order], tr[order]
    M = ck.shape[0]
    newcell = jnp.concatenate([jnp.ones(1, bool), ck[1:] != ck[:-1]])
    cell = jnp.cumsum(newcell) - 1
    ccount = jax.ops.segment_sum(jnp.ones(M, jnp.int64), cell, num_segments=M)
    keep = (ck != big) & (ccount[cell] >= 2)
    thr = resp.time4merge
    i = jnp.arange(M)

    def body(state):
        pending, rnd, r = state
        anchor = jax.ops.segment_min(jnp.where(pending, i, M), cell, num_segments=M)
        a = jnp.minimum(anchor[cell], M - 1)
        take = pending & ((ts - ts[a]).astype(jnp.float64) <= thr)
        return pending & ~take, jnp.where(take, r, rnd), r + 1

    _, rnd, _ = jax.lax.while_loop(lambda s: jnp.any(s[0]), body, (keep, jnp.full(M, -1, jnp.int64), jnp.int64(0)))
    prev_ok = jnp.concatenate([jnp.zeros(1, bool), keep[:-1]])
    newgroup = keep & (newcell | ~prev_ok | (rnd != jnp.concatenate([jnp.full(1, -2, jnp.int64), rnd[:-1]])))
    gid = jnp.cumsum(newgroup) - 1
    n_hits = jnp.where(M > 0, gid[-1] + 1, 0)
    H = caps.hits
    g = jnp.where(keep & (gid < H), gid, H)
    terr, ee, eerr = t_err[order], e[order], e_err[order]
    safe = lambda v: jnp.where(keep, v, 1.0)   # noqa: E731 - padded members go to the sink group
    h_t, h_terr = weighted_reco_digis(jnp.where(keep, ts, 0.0), safe(terr), g, H + 1)
    h_e, h_eerr = weighted_reco_digis(jnp.where(keep, ee, 0.0), safe(eerr), g, H + 1)
    nsipm = jax.ops.segment_sum(jnp.ones(M, jnp.int64), g, num_segments=H + 1)[:H]
    first = jax.ops.segment_min(jnp.where(keep, i, M), g, num_segments=H + 1)[:H]
    fi = jnp.minimum(first, M - 1)
    h_ok = jnp.arange(H) < n_hits
    hits = dict(ok=h_ok, event=jnp.where(h_ok, ck[fi] // resp.n_crystals, 0),
                crystal=jnp.where(h_ok, ck[fi] % resp.n_crystals, 0), nsipm=nsipm,
                t=h_t[:H], t_err=h_terr[:H], e=h_e[:H], e_err=h_eerr[:H])
    stats = dict(n_reco=jnp.sum(sel), n_nonfinite=n_nonfinite, n_refit_skipped=jnp.sum(fit["refit_skipped"] & dg["ok"]),
                 refits=over_refit, fit_buckets=over_bucket)
    return hits, jnp.maximum(n_hits - H, 0), stats


def param_hits(resp: DeviceResponse, table, ev, t, st, crystal, depth, mapped, key, pbt_reco, caps: Caps):
    """Hits from the parametric digi response (:mod:`calofastsim.param_response`; approximation
    ``digi_response=parametric``) instead of photoelectrons, digitisation and the template fit: per (event, crystal)
    cell the visible energy and energy-weighted time of the accepted steps (the rules of :func:`photoelectrons`),
    then one table draw. Returns (hits as :func:`fit_and_hits`, overflow dict, stats)."""
    import jax.numpy as jnp
    from . import param_response as PR
    from . import response_jax as rj
    a = resp.arrays
    cfg = resp.ro_cfg
    cr = jnp.maximum(crystal, 0)
    hit_t, accepted = rj.fold_step_time(
        st["time"], pbtime=t["pbtime"][ev], mbtime=t["mbtime"][ev], event_length=t["event_length"][ev],
        onspill=t["onspill"][ev], digitization_start=cfg.digitization_start, digitization_end=cfg.digitization_end,
        digitization_buffer=cfg.digitization_buffer, time_from_protons_to_dr_marker=t["marker"][ev])
    keep = mapped & accepted
    energy = rj.corrected_energy(st["energy_g4"], st["energy_birks"], depth, t["crystal_length"][ev], a["lru"][cr],
                                 birks=cfg.birks, longitudinal=cfg.lru)
    ok, cev, ccr, E, T, over_cells = PR.cells(ev, cr, energy, hit_t, keep, resp.n_crystals, caps.rows)
    hit, e, th = PR.draw(table, key, ok, E, T, T - window_end(cfg, t)[cev])
    # the table's times carry no reco bunch-time smear: subtract this event's (pbt_reco - pbtime)
    shift = pbt_reco.astype(jnp.float64) - t["pbtime"].astype(jnp.float32).astype(jnp.float64)
    th = (th.astype(jnp.float64) - shift[cev]).astype(jnp.float32)
    # compact the hits (cell order: event, crystal) into the hit capacity, as fit_and_hits leaves them
    n = hit.shape[0]
    order = jnp.argsort(jnp.where(hit, jnp.arange(n), n))
    H = caps.hits
    sel = order[:H] if n >= H else jnp.concatenate([order, jnp.full(H - n, n - 1, order.dtype)])
    n_hits = jnp.sum(hit)
    h_ok = jnp.arange(H) < jnp.minimum(n_hits, H)
    f32 = jnp.float32
    hits = dict(ok=h_ok, event=jnp.where(h_ok, cev[sel], 0), crystal=jnp.where(h_ok, ccr[sel], 0),
                nsipm=jnp.where(h_ok, 2, 0).astype(jnp.int64), t=jnp.where(h_ok, th[sel], f32(0)),
                t_err=jnp.where(h_ok, f32(1), f32(0)), e=jnp.where(h_ok, e[sel], f32(0)),
                e_err=jnp.where(h_ok, f32(1), f32(0)))
    over = dict(photoelectrons=jnp.int64(0), rows=over_cells, outliers=jnp.int64(0), digis=jnp.int64(0),
                digi_length=jnp.int64(0), hits=jnp.maximum(n_hits - H, 0), refits=jnp.int64(0),
                fit_buckets=jnp.int64(0))
    stats = dict(n_reco=jnp.int64(0), n_nonfinite=jnp.int64(0), n_refit_skipped=jnp.int64(0), cells=jnp.sum(ok))
    return hits, over, stats


def window_end(cfg, t):
    """Per event, the end of the digitisation window in folded step time (fold_step_time's corrected end)."""
    import jax.numpy as jnp
    f32 = jnp.float32
    return ((f32(cfg.digitization_end) - t["pbtime"].astype(f32)) - t["marker"].astype(f32)).astype(jnp.float64)


def _digi_table(digi_response):
    """None for the waveform response; the device table for "parametric" (the shipped table) or a table path."""
    if digi_response in (None, "waveform"):
        return None
    from . import param_response as PR
    return PR.to_device(PR.load("default" if digi_response == "parametric" else digi_response))


def _smeared_pbt(resp, t, k_pbt):
    import jax
    import jax.numpy as jnp
    pbt = t["pbtime"].astype(jnp.float32)
    if resp.pbt_reco_sigma > 0:
        pbt = (pbt.astype(jnp.float64) + resp.pbt_reco_sigma *
               jax.random.normal(k_pbt, pbt.shape, dtype=jnp.float64)).astype(jnp.float32)
    return pbt


def _cluster_props(label, n_cl, t, e):
    import jax.numpy as jnp
    member = label[:, :, None] == jnp.arange(n_cl)[None, None, :]            # (B, H, C)
    score = jnp.where(member, e[:, :, None].astype(jnp.float64), -jnp.inf)
    lead = jnp.argmax(score, axis=1)                                           # first max = lowest slot
    exists = member.any(axis=1)
    lead = jnp.where(exists, lead, -1)
    esum = jnp.sum(jnp.where(member, e[:, :, None].astype(jnp.float64), 0.0), axis=1)
    time = jnp.where(exists, jnp.take_along_axis(t.astype(jnp.float64), jnp.maximum(lead, 0), axis=1), jnp.inf)
    return lead, time, esum, member


def _closest_pairs(member_a, member_b, dist):
    """(B, Ca, Cb) minimum crystal distance between hits of clusters a and b."""
    import jax.numpy as jnp
    rows = []
    for i in range(member_a.shape[2]):
        da = jnp.where(member_a[:, :, i, None], dist, jnp.inf).min(axis=1)        # (B, H): min over a-hits
        rows.append(jnp.where(member_b, da[:, :, None], jnp.inf).min(axis=1))      # (B, Cb)
    return jnp.stack(rows, axis=1)


def clusters(resp: DeviceResponse, hits, n_events, caps: Caps):
    """cluster.make_clusters on padded per-event hits; returns per-event clusters sorted by energy, and overflows.

    Main and split clusters grow in ``caps.rounds`` rounds, and the result has ``caps.rounds`` slots. With
    ``caps.recluster`` (default) an event that still has seeds after those rounds is clustered again with
    ``caps.event_hits`` rounds, which always suffices (each round takes its seed hit); the result keeps that event's
    ``caps.rounds`` most energetic clusters. This branch runs only in batches that need it and replaces only those
    events, so the other events are unchanged. ``over`` also holds ``recluster`` (events clustered again) and
    ``cluster_slots`` (of those, events with more clusters than slots); ``main_rounds`` / ``split_rounds`` count the
    events left with seeds (0 with ``recluster``).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    cfg = resp.cluster_cfg
    a = resp.arrays
    B, Hm, R0 = n_events, caps.event_hits, caps.rounds
    ok = hits["ok"]
    ev = jnp.where(ok, hits["event"], B)
    hidx = jnp.arange(ev.shape[0])
    estart = jax.ops.segment_min(jnp.where(ok, hidx, ev.shape[0]), ev, num_segments=B + 1)
    slot = hidx - estart[ev]
    put = ok & (slot < Hm)
    over_hits = jnp.sum(ok & (slot >= Hm))
    ei, si = jnp.where(put, ev, B), jnp.where(put, slot, 0)

    def pad(v, fill, dtype):
        return jnp.full((B + 1, Hm), fill, dtype).at[ei, si].set(v.astype(dtype))[:B]

    valid = pad(jnp.ones_like(ok), False, bool)
    crystal = pad(hits["crystal"], 0, jnp.int64)
    t = pad(hits["t"], 0.0, jnp.float32)
    e = pad(hits["e"], 0.0, jnp.float32)
    vv = valid[:, :, None] & valid[:, None, :]
    same = (crystal[:, :, None] == crystal[:, None, :]) & vv
    adj = a["neighbors"][crystal[:, :, None], crystal[:, None, :]] & vv
    pos = a["position"][crystal]
    dist = jnp.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
    avail = valid & (e >= cfg.enoise_cut)
    t64 = t.astype(jnp.float64)
    adjf, samef = adj.astype(jnp.float32), same.astype(jnp.float32)
    rows = jnp.arange(B)

    def grow(seed, avail):
        has = seed >= 0
        s = jnp.maximum(seed, 0)
        ts = t64[rows, s]
        intime = avail & (jnp.abs(t64 - ts[:, None]) < cfg.delta_time)
        visited = same[rows, s] & has[:, None]
        mem = (jnp.arange(Hm)[None, :] == s[:, None]) & has[:, None]

        def cond(st):
            return st[3]

        def body(st):
            visited, expanding, mem, _ = st
            reach = jnp.einsum("ei,eij->ej", expanding.astype(jnp.float32), adjf, precision=TF32_EXACT) > 0
            newc = reach & ~visited
            visited = visited | newc
            new = newc & intime
            mem = mem | new
            grow_ = new & (e > cfg.expand_cut)
            expanding = jnp.einsum("ei,eij->ej", grow_.astype(jnp.float32), samef, precision=TF32_EXACT) > 0
            return visited, expanding, mem, newc.any()

        _, _, mem, _ = lax.while_loop(cond, body, (visited, visited, mem, jnp.bool_(True)))
        return mem

    def cluster_events(R):
        """Per-event clusters with R rounds and R slots; per event, seeds left after the main / split rounds."""
        def rounds(seedable, avail):
            seedable = seedable & avail

            def body(r, st):
                label, seedable, avail, seed_t = st
                has = seedable.any(axis=1)
                seed = jnp.where(has, jnp.argmax(seedable, axis=1), -1)
                mem = grow(seed, avail)
                label = jnp.where(mem, r, label)
                avail = avail & ~mem
                seedable = seedable & ~mem
                seed_t = seed_t.at[:, r].set(jnp.where(has, t64[rows, jnp.maximum(seed, 0)], jnp.nan))
                return label, seedable, avail, seed_t

            label, seedable, avail, seed_t = lax.fori_loop(
                0, R, body, (jnp.full((B, Hm), -1, jnp.int64), seedable, avail, jnp.full((B, R), jnp.nan)))
            return label, seed_t, avail, seedable.any(axis=1)

        main, seed_t, left, over_main = rounds(avail & (e > cfg.emin_seed), avail)
        keep = jnp.any(seed_t[:, None, :] - t64[:, :, None] < cfg.delta_time, axis=2)
        left = left & keep
        split, _, _, over_split = rounds(left, left)
        m_lead, m_time, _, _ = _cluster_props(main, R, t, e)
        s_lead, s_time, _, s_mem = _cluster_props(split, R, t, e)
        m_order = jnp.argsort(m_time, axis=1, stable=True)
        rank = jnp.zeros_like(m_order).at[rows[:, None], m_order].set(jnp.broadcast_to(jnp.arange(R), (B, R)))
        main_sorted = jnp.where(main >= 0, jnp.take_along_axis(rank, jnp.maximum(main, 0), axis=1), -1)
        ms_time = jnp.take_along_axis(m_time, m_order, axis=1)
        ms_lead = jnp.take_along_axis(m_lead, m_order, axis=1)
        ms_mem = main_sorted[:, :, None] == jnp.arange(R)[None, None, :]
        # split -> closest main within time and distance (first minimum in time order)
        d_ms = _closest_pairs(ms_mem, s_mem, dist)                                  # (B, main j, split k)
        pl = lambda lead: a["position"][jnp.take_along_axis(crystal, jnp.maximum(lead, 0), axis=1)]  # noqa: E731
        d0 = jnp.linalg.norm(pl(ms_lead)[:, :, None, :] - pl(s_lead)[:, None, :, :], axis=-1)
        okp = ((s_lead >= 0)[:, None, :] & (ms_lead >= 0)[:, :, None] &
               (s_time[:, None, :] - ms_time[:, :, None] <= cfg.delta_time) &
               (ms_time[:, :, None] - s_time[:, None, :] <= cfg.delta_time) & (d0 <= 2.5 * cfg.max_dist_split))
        cand = jnp.where(okp & (d_ms < cfg.max_dist_split) & (d_ms < 1e6), d_ms, jnp.inf)
        best = jnp.argmin(cand, axis=1)
        split_to = jnp.where(jnp.isfinite(jnp.min(cand, axis=1)), best, -1)        # (B, split k)
        temp = main_sorted
        sj = jnp.where(split >= 0, jnp.take_along_axis(split_to, jnp.maximum(split, 0), axis=1), -1)
        temp = jnp.where((split >= 0) & (sj >= 0), sj, temp)
        t_mem = temp[:, :, None] == jnp.arange(R)[None, None, :]
        d_mm = _closest_pairs(t_mem, t_mem, dist)                                    # (B, i, j)
        ii, jj = jnp.arange(R)[:, None], jnp.arange(R)[None, :]
        okm = ((ii < jj)[None] & (ms_lead >= 0)[:, :, None] & (ms_lead >= 0)[:, None, :] &
               ((ms_time[:, None, :] - ms_time[:, :, None]) <= cfg.delta_time) & (d_mm <= cfg.max_dist_main))
        parent = jnp.where(okm.any(axis=1), jnp.argmax(okm, axis=1), -1)            # first i for each j
        root = jnp.broadcast_to(jnp.arange(R), (B, R))
        for _ in range(R):
            p = jnp.take_along_axis(parent, root, axis=1)
            root = jnp.where(p >= 0, p, root)
        final = jnp.where(temp >= 0, jnp.take_along_axis(root, jnp.maximum(temp, 0), axis=1), -1)
        merged = jnp.zeros((B, R + 1), bool).at[rows[:, None], jnp.where(parent >= 0, root, R)].set(True)[:, :R]
        has_split = jnp.zeros((B, R + 1), bool).at[rows[:, None], jnp.where((split >= 0) & (sj >= 0), final, R)].set(
            True)[:, :R]
        lead, ctime, esum, f_mem = _cluster_props(final, R, t, e)
        exists = lead >= 0
        lead_disk = a["disk"][jnp.take_along_axis(crystal, jnp.maximum(lead, 0), axis=1)]  # (B, C)
        hit_disk = a["disk"][crystal]
        on_disk = f_mem & (hit_disk[:, :, None] == lead_disk[:, None, :])
        w = jnp.where(on_disk, e[:, :, None].astype(a["position"].dtype), 0.0)                     # (B, H, C)
        sw = w.sum(axis=1)
        sxyz = jnp.einsum("bhc,bhk->bck", w, pos, precision=_physics(w, pos))
        cog = sxyz / jnp.where(sw > 0, sw, 1.0)[..., None] - a["front_face"][lead_disk]
        size = f_mem.sum(axis=1)
        order = jnp.argsort(jnp.where(exists, -esum, jnp.inf), axis=1, stable=True)
        take = lambda v: jnp.take_along_axis(v, order, axis=1)  # noqa: E731
        out = dict(valid=take(exists), energy=take(esum), time=take(ctime), x=take(cog[..., 0]), y=take(cog[..., 1]),
                   z=take(cog[..., 2]), size=take(size), disk=take(lead_disk), split=take(has_split | merged))
        if "etrue" in hits:   # opt-in truth: the member hits' true deposits (sum) and earliest true time (min)
            etr = pad(hits["etrue"], 0.0, jnp.float64)
            ttr = pad(hits["ttrue"], jnp.inf, jnp.float64)
            out["etrue"] = take(jnp.einsum("bhc,bh->bc", f_mem.astype(jnp.float64), etr))
            out["ttrue"] = take(jnp.min(jnp.where(f_mem, ttr[:, :, None], jnp.inf), axis=1))
        return out, over_main, over_split

    out, left_main, left_split = cluster_events(R0)
    redo = left_main | left_split
    n_redo = jnp.sum(redo)
    n_slots = jnp.int64(0)
    if caps.recluster and Hm > R0:
        def again():
            big, bm, bs = cluster_events(Hm)
            pick = lambda x, y: jnp.where(redo.reshape((B,) + (1,) * (x.ndim - 1)), y[:, :R0], x)  # noqa: E731
            n_over = jnp.sum(redo & (jnp.sum(big["valid"], axis=1) > R0))
            return {k: pick(v, big[k]) for k, v in out.items()}, left_main & bm, left_split & bs, n_over

        out, left_main, left_split, n_slots = lax.cond(
            n_redo > 0, again, lambda: (out, left_main, left_split, n_slots))
    else:
        n_redo = jnp.int64(0)
    over = dict(event_hits=over_hits, main_rounds=jnp.sum(left_main), split_rounds=jnp.sum(left_split),
                recluster=n_redo, cluster_slots=n_slots)
    return out, over


# ----------------------------------------------------------------------------------------------- driver

def _with_cluster_stats(stats, over_cl):
    """stats plus the clustering's ``recluster`` / ``cluster_slots`` counts (:func:`clusters`)."""
    return dict(stats, recluster=over_cl["recluster"], cluster_slots=over_cl["cluster_slots"])


def make_chain(lib: DeviceLibrary, resp: DeviceResponse, caps: Caps = Caps(), digi_response="waveform", truth=False):
    """Return ``chain(q, timing, key, uniform=None)``: one jitted call from entrance states to clusters.

    ``q``/``timing`` come from :func:`entries_batch` (``caps.events`` rows). The result stays on the device:
    ``clusters`` (per event, sorted by energy, ``caps.rounds`` slots with ``valid``), ``overflow`` (int64 per
    :data:`OVERFLOW_NAMES`) and ``stats`` (counts of draws, mapped steps, photoelectrons, digis, hits, and the
    events clustered again, ``recluster`` / ``cluster_slots``: :func:`clusters`).
    ``uniform`` (per query, in [0,1)) replaces the library draw's random numbers, for tests.
    ``digi_response``: "waveform" (default: photoelectrons, digis, template fit) or the opt-in approximation
    "parametric" (or a table path): hits drawn per crystal from :mod:`calofastsim.param_response` (:func:`param_hits`);
    no digis.
    ``truth`` (opt-in) adds per cluster ``etrue``, the sampled steps' visible (Birks) energy in the cluster's hit
    crystals, and ``ttrue``, the earliest of those steps' times (the stock CaloClusterMC etot / earliest deposit);
    the other outputs are unchanged.
    """
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise ValueError("the device chain requires caller-enabled JAX x64")

    pdt = lib.arrays["centers"].dtype

    def with_truth(hits, q, st, crystal, mapped):
        if not truth:
            return hits
        nc, B = resp.n_crystals, caps.events
        use = st["valid"] & mapped
        cell = jnp.where(use, q["event"][st["row"]] * nc + jnp.maximum(crystal, 0), B * nc)
        e = jnp.zeros(B * nc + 1, jnp.float64).at[cell].add(jnp.where(use, st["energy_birks"], 0.0).astype(jnp.float64))
        tm = jnp.full(B * nc + 1, jnp.inf, jnp.float64).at[cell].min(jnp.where(use, st["time"], jnp.inf).astype(jnp.float64))
        hc = jnp.where(hits["ok"], hits["event"] * nc + hits["crystal"], B * nc)
        return dict(hits, etrue=e[hc], ttrue=tm[hc])
    table = _digi_table(digi_response)

    def chain(q, t, key, uniform=None):
        q = dict(q, position=q["position"].astype(pdt), direction=q["direction"].astype(pdt))
        k_draw, k_pe, k_digi, k_pbt = jax.random.split(key, 4)
        if uniform is None:
            uniform = jax.random.uniform(k_draw, q["energy"].shape, dtype=jnp.float64)
        donor, fallback, slot = sample_donors(lib, q, uniform)
        st, over_steps = gather_steps(lib, q, donor, slot, caps.steps)
        crystal, depth, mapped = place_steps(lib, q, st)
        if table is not None:
            hits, over_resp, stats = param_hits(resp, table, q["event"][st["row"]], t, st, crystal, depth, mapped,
                                                k_pe, _smeared_pbt(resp, t, k_pbt), caps)
            cl, over_cl = clusters(resp, with_truth(hits, q, st, crystal, mapped), caps.events, caps)
            over = dict(steps=over_steps, **over_resp, **over_cl)
            stats.update(draws=jnp.sum(donor >= 0), fallback=jnp.sum(fallback & q["valid"]),
                         steps=jnp.sum(st["valid"]), mapped=jnp.sum(mapped), photoelectrons=jnp.float64(0),
                         digis=jnp.int64(0), hits=jnp.sum(hits["ok"]))
            return dict(clusters=cl, overflow=jnp.stack([jnp.asarray(over[k], jnp.int64) for k in OVERFLOW_NAMES]),
                        stats=_with_cluster_stats(stats, over_cl), donor=donor)
        ev, sipm, pe_t, weight, pe_ok, over_pe = photoelectrons(resp, q, t, st, crystal, depth, mapped, k_pe,
                                                                caps.photoelectrons)
        dg, over_dg = digitise(resp, t, ev, sipm, pe_t, weight, pe_ok, k_digi, caps)
        pbt = t["pbtime"].astype(jnp.float32)
        if resp.pbt_reco_sigma > 0:
            pbt = (pbt.astype(jnp.float64) + resp.pbt_reco_sigma *
                   jax.random.normal(k_pbt, pbt.shape, dtype=jnp.float64)).astype(jnp.float32)
        hits, over_hits, stats = fit_and_hits(resp, t, dg, pbt, caps)
        cl, over_cl = clusters(resp, with_truth(hits, q, st, crystal, mapped), caps.events, caps)
        over = dict(steps=over_steps, photoelectrons=over_pe, hits=over_hits, refits=stats.pop("refits"),
                    fit_buckets=stats.pop("fit_buckets"), **over_dg, **over_cl)
        stats.update(draws=jnp.sum(donor >= 0), fallback=jnp.sum(fallback & q["valid"]), steps=jnp.sum(st["valid"]),
                     mapped=jnp.sum(mapped), photoelectrons=jnp.sum(jnp.where(pe_ok, weight, 0.0)),
                     digis=jnp.sum(dg["ok"]), hits=jnp.sum(hits["ok"]))
        return dict(clusters=cl, overflow=jnp.stack([jnp.asarray(over[k], jnp.int64) for k in OVERFLOW_NAMES]),
                    stats=_with_cluster_stats(stats, over_cl), donor=donor)

    return jax.jit(chain, compiler_options=COMPILER_OPTIONS)


def make_steps_chain(resp: DeviceResponse, caps: Caps = Caps(), digi_response="waveform"):
    """Return ``chain(steps, timing, key)``: saved CaloShowerSteps -> clusters (the stock calo digi + reco stage).

    ``steps`` from :func:`steps_batch`: batch-local ``event``, ``crystal`` (-1 = padding), ``energy_g4``,
    ``energy_birks``, ``time``, ``z_local``; same result layout as :func:`make_chain`. ``digi_response`` as
    :func:`make_chain` (the opt-in "parametric" draws hits per crystal: steps -> hits + clusters, no digis).
    """
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise ValueError("the device chain requires caller-enabled JAX x64")
    table = _digi_table(digi_response)

    def chain(steps, t, key, uniform=None):
        k_pe, k_digi, k_pbt = jax.random.split(key, 3)
        n = steps["event"].shape[0]
        ok = steps["crystal"] >= 0
        st = dict(row=jnp.arange(n), time=steps["time"], energy_g4=steps["energy_g4"],
                  energy_birks=steps["energy_birks"], valid=ok)
        if table is not None:
            hits, over_resp, stats = param_hits(resp, table, steps["event"], t, st, steps["crystal"],
                                                steps["z_local"], ok, k_pe, _smeared_pbt(resp, t, k_pbt), caps)
            cl, over_cl = clusters(resp, hits, caps.events, caps)
            over = dict(steps=jnp.int64(0), **over_resp, **over_cl)
            stats.update(steps=jnp.sum(ok), photoelectrons=jnp.float64(0), digis=jnp.int64(0),
                         hits=jnp.sum(hits["ok"]))
            return dict(clusters=cl, overflow=jnp.stack([jnp.asarray(over[k], jnp.int64) for k in OVERFLOW_NAMES]),
                        stats=_with_cluster_stats(stats, over_cl))
        ev, sipm, pe_t, weight, pe_ok, over_pe = photoelectrons(resp, {"event": steps["event"]}, t, st,
                                                                steps["crystal"], steps["z_local"], ok, k_pe,
                                                                caps.photoelectrons)
        dg, over_dg = digitise(resp, t, ev, sipm, pe_t, weight, pe_ok, k_digi, caps)
        pbt = t["pbtime"].astype(jnp.float32)
        if resp.pbt_reco_sigma > 0:
            pbt = (pbt.astype(jnp.float64) + resp.pbt_reco_sigma *
                   jax.random.normal(k_pbt, pbt.shape, dtype=jnp.float64)).astype(jnp.float32)
        hits, over_hits, stats = fit_and_hits(resp, t, dg, pbt, caps)
        cl, over_cl = clusters(resp, hits, caps.events, caps)
        over = dict(steps=jnp.int64(0), photoelectrons=over_pe, hits=over_hits, refits=stats.pop("refits"),
                    fit_buckets=stats.pop("fit_buckets"), **over_dg, **over_cl)
        stats.update(steps=jnp.sum(ok), photoelectrons=jnp.sum(jnp.where(pe_ok, weight, 0.0)),
                     digis=jnp.sum(dg["ok"]), hits=jnp.sum(hits["ok"]))
        return dict(clusters=cl, overflow=jnp.stack([jnp.asarray(over[k], jnp.int64) for k in OVERFLOW_NAMES]),
                    stats=_with_cluster_stats(stats, over_cl))

    return jax.jit(chain, compiler_options=COMPILER_OPTIONS)


def steps_batch(steps, n_steps):
    """Pad flat CaloShowerSteps (batch-local ``event``) to ``n_steps`` rows for :func:`make_steps_chain`."""
    m = len(steps["event"])
    if m > n_steps:
        raise ValueError("more steps than the batch capacity")
    out = {}
    for k, fill, dtype in (("event", 0, np.int64), ("crystal", -1, np.int64), ("energy_g4", 0.0, np.float32),
                           ("energy_birks", 0.0, np.float32), ("time", 0.0, np.float64), ("z_local", 0.0, np.float32)):
        a = np.full(n_steps, fill, dtype)
        a[:m] = np.asarray(steps[k], dtype)
        out[k] = a
    return out


def run(chain, q, t, key, *, tolerate_overflow=False):
    """Call ``chain`` on host arrays; download the clusters and check the capacities (one sync per batch)."""
    import jax
    import jax.numpy as jnp
    out = chain({k: jnp.asarray(v) for k, v in q.items()}, {k: jnp.asarray(v) for k, v in t.items()}, key)
    out = jax.tree_util.tree_map(np.asarray, out)
    over = dict(zip(OVERFLOW_NAMES, out["overflow"].tolist()))
    if not tolerate_overflow and any(over.values()):
        raise RuntimeError(f"device chain capacity exceeded: { {k: v for k, v in over.items() if v} }")
    out["overflow"] = over
    return out


def flat_clusters(cl):
    """Per-event cluster slots -> flat arrays (event, energy, time, x, y, z, size, disk, split), event-major."""
    ev, slot = np.nonzero(np.asarray(cl["valid"]))
    out = {k: np.asarray(v)[ev, slot] for k, v in cl.items() if k != "valid"}
    out["event"] = ev
    return out
