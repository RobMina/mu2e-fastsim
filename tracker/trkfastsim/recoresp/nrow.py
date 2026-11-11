"""Sampled tracker rows (opt-in approximation W2-W4 ``tracker_rows=sampled``): the response layer's ``nrow`` drawn
from a library instead of transported.

The reco response (:mod:`trkfastsim.recoresp.device`) reads a track's StrawGasStep rows only through the bin of
``nrow`` it falls in. When the tracker steps themselves are not needed (track observables without digis), the
tracker and albedo transport can be replaced by a draw of ``nrow`` given the entry helix: the library holds, per
cell of (|p|, tan dip, r_max) at the tracker entry, the rows the default device chain produced (tracker transport
plus the calorimeter albedo) on training electrons. A cell without training tracks (fewer than ``min_count``, default
1) falls back to its (|p|, tan dip) cell, then to every track. A fallback that drops r_max biases the sparse
low-momentum cells: with 20 instead of 1, the RPC fit efficiency came out 3 % low.

    tab = build(has_entry, nrow, x, p, axis_xy, bz)        # host, from a training run of the default chain
    save(path, tab); dn = to_device(load(path))
    nrow = sample(dn, key, has_entry, x, p)                 # device, per event (0 without a tracker entry)

The shipped library (``DEFAULT``) was built from the default chain on Run 1A geometry and field (conversion, DIO and
RPC electrons, training draws disjoint from the benchmark inputs); its README records the inputs.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np

from trkfastsim.recoresp import table as RT

SCHEMA = 1
DEFAULT = Path(__file__).with_name("data") / "nrow_run1a.npz"
NAMED = {"default": DEFAULT}
# cell edges (values outside go to the first or last cell)
P_EDGES = np.concatenate([[0.0, 30.0], np.arange(40.0, 96.0, 2.5), [97.0, 99.0, 100.5, 101.5, 102.5, 103.25, 104.0,
                                                                   104.75, 106.0, 108.0, 110.0, 115.0, 120.0, 200.0]])
TANDIP_EDGES = np.round(np.arange(0.30, 1.501, 0.05), 6)
RMAX_EDGES = np.round(np.arange(400.0, 760.1, 10.0), 6)


def resolve(library):
    """Path of a library: a shipped name (NAMED) or a file."""
    if isinstance(library, str) and library in NAMED:
        return NAMED[library]
    return Path(library)


def _bins(edges, v):
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)


def build(has_entry, nrow, x, p, axis_xy, bz, *, p_edges=P_EDGES, tandip_edges=TANDIP_EDGES, rmax_edges=RMAX_EDGES,
          min_count=1, reference=""):
    """Library from a training run (host arrays): has_entry [n] bool, nrow [n] rows of the default chain, x, p [n, 3]
    entry state (Mu2e frame), axis_xy and bz as the response table's."""
    has = np.asarray(has_entry, bool)
    nrow = np.asarray(nrow)[has].astype(np.int64)
    if nrow.size == 0:
        raise ValueError("no training track with a tracker entry")
    if nrow.max() > np.iinfo(np.int16).max:
        raise ValueError("nrow beyond int16")
    pm, td, _, rmax = RT.entry_truth(np.asarray(x, float)[has], np.asarray(p, float)[has], axis_xy, bz)
    ip, it, ir = _bins(p_edges, pm), _bins(tandip_edges, td), _bins(rmax_edges, rmax)
    npb, ntb, nrb = len(p_edges) - 1, len(tandip_edges) - 1, len(rmax_edges) - 1
    fine = (ip * ntb + it) * nrb + ir
    coarse = ip * ntb + it
    pools, start = [], 0

    def add(values):
        nonlocal start
        pools.append(np.sort(values))
        s = start
        start += len(values)
        return s, len(values)

    fine_order = np.argsort(fine, kind="stable")
    fine_cells, fine_first, fine_n = np.unique(fine[fine_order], return_index=True, return_counts=True)
    fine_at = {}
    for c, f, k in zip(fine_cells, fine_first, fine_n):
        if k >= min_count:
            fine_at[int(c)] = add(nrow[fine_order[f:f + k]])
    coarse_at = {}
    for c in np.unique(coarse):
        v = nrow[coarse == c]
        if len(v) >= min_count:
            coarse_at[int(c)] = add(v)
    every = add(nrow)
    n_fine = npb * ntb * nrb
    cell_start = np.empty(n_fine, np.int64)
    cell_count = np.empty(n_fine, np.int64)
    level = np.empty(n_fine, np.int8)          # 0 fine, 1 coarse, 2 every track
    for c in range(n_fine):
        if c in fine_at:
            (cell_start[c], cell_count[c]), level[c] = fine_at[c], 0
        elif c // nrb in coarse_at:
            (cell_start[c], cell_count[c]), level[c] = coarse_at[c // nrb], 1
        else:
            (cell_start[c], cell_count[c]), level[c] = every, 2
    used = level[fine]
    return dict(schema=np.int64(SCHEMA), p_edges=np.asarray(p_edges, float), tandip_edges=np.asarray(tandip_edges, float),
                rmax_edges=np.asarray(rmax_edges, float), cell_start=cell_start, cell_count=cell_count,
                cell_level=level, pool=np.concatenate(pools).astype(np.int16), axis_xy=np.asarray(axis_xy, float),
                bz=np.float64(bz), min_count=np.int64(min_count), n_training=np.int64(len(nrow)),
                training_level_fraction=np.array([np.mean(used == k) for k in range(3)]), reference=np.str_(reference))


def save(path, tab):
    np.savez_compressed(Path(path), **tab)


def load(library):
    with np.load(resolve(library), allow_pickle=False) as f:
        tab = {k: f[k] for k in f.files}
    if int(tab["schema"]) != SCHEMA:
        raise ValueError(f"nrow library schema {int(tab['schema'])}, expected {SCHEMA}")
    return tab


def sample_numpy(tab, rng, has_entry, x, p):
    """Host twin of :func:`sample` (tests)."""
    has = np.asarray(has_entry, bool)
    pm, td, _, rmax = RT.entry_truth(np.asarray(x, float), np.asarray(p, float), tab["axis_xy"], float(tab["bz"]))
    ntb, nrb = len(tab["tandip_edges"]) - 1, len(tab["rmax_edges"]) - 1
    c = (_bins(tab["p_edges"], pm) * ntb + _bins(tab["tandip_edges"], td)) * nrb + _bins(tab["rmax_edges"], rmax)
    cnt = tab["cell_count"][c]
    j = tab["cell_start"][c] + np.minimum((rng.uniform(size=len(c)) * cnt).astype(np.int64), cnt - 1)
    return np.where(has, tab["pool"][j], 0).astype(np.int64)


class DeviceNrow(NamedTuple):
    p_edges: object
    tandip_edges: object
    rmax_edges: object
    cell_start: object
    cell_count: object
    pool: object
    axis_xy: object
    bz: float


def to_device(tab, dtype=np.float64):
    import jax.numpy as jnp
    dt = np.dtype(dtype)
    return DeviceNrow(jnp.asarray(tab["p_edges"], dt), jnp.asarray(tab["tandip_edges"], dt),
                      jnp.asarray(tab["rmax_edges"], dt), jnp.asarray(tab["cell_start"], jnp.int32),
                      jnp.asarray(tab["cell_count"], jnp.int32), jnp.asarray(tab["pool"], jnp.int32),
                      jnp.asarray(tab["axis_xy"], dt), float(tab["bz"]))


def sample(dn, key, has_entry, x, p):
    """nrow per event (int32 device array; 0 without a tracker entry): one uniform draw from the event's cell."""
    import jax
    import jax.numpy as jnp
    from trkfastsim.recoresp import device as RD
    dt = x.dtype
    pm, td, _, rmax = RD.entry_truth(x, p, dn.axis_xy.astype(dt), dn.bz)
    ntb, nrb = dn.tandip_edges.shape[0] - 1, dn.rmax_edges.shape[0] - 1
    c = (RD._bin(dn.p_edges.astype(dt), pm) * ntb + RD._bin(dn.tandip_edges.astype(dt), td)) * nrb \
        + RD._bin(dn.rmax_edges.astype(dt), rmax)
    cnt = dn.cell_count[c]
    u = jax.random.uniform(key, (x.shape[0],), dt)
    j = dn.cell_start[c] + jnp.minimum((u * cnt).astype(jnp.int32), cnt - 1)
    return jnp.where(has_entry, dn.pool[j], 0)
