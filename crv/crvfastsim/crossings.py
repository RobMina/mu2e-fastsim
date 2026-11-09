"""CRV bar crossings: the generic slab method of mu2efastsim.crossings applied to the CRV bar boxes.

V0 found every bar axis-aligned and none rotated, so a bar is the box center +- halflen (geometry.py).
"""
from mu2efastsim import crossings as _core
from mu2efastsim.crossings import HAVE_JAX, slab_crossings  # noqa: F401  (re-exported for the CRV code and tests)


def bar_bounds(geom):
    return geom.center - geom.halflen, geom.center + geom.halflen


def crossings(origin, direction, geom, chunk: int = 32768, backend=None, per_module_above=1000):
    """Sparse crossings with the CRV bars: rows (track index, bar index, t_in, t_out, path length).

    Geometries with more than per_module_above bars (Run 1A 1,280; Run 2 5,344) are cut in two levels: tracks against
    the (sector, module) boxes, then each track against the bars of the modules it enters only (same rows, same order;
    the [tracks, bars] slab array of the full CRV would be ~10 GB per chunk)."""
    lo, hi = bar_bounds(geom)
    if geom.nbars <= per_module_above:
        return _core.crossings(origin, direction, lo, hi, chunk=chunk, backend=backend)
    import numpy as np
    mod = geom.sector.astype(np.int64) * 1000 + geom.module
    um, inv = np.unique(mod, return_inverse=True)
    mlo = np.full((len(um), 3), np.inf)
    mhi = np.full((len(um), 3), -np.inf)
    np.minimum.at(mlo, inv, lo)
    np.maximum.at(mhi, inv, hi)
    mx = _core.crossings(origin, direction, mlo, mhi, chunk=chunk, backend=backend)      # (track, module) pairs
    if not len(mx):
        return np.zeros((0, 5))
    bars_of = np.argsort(inv, kind="stable")
    start = np.r_[0, np.cumsum(np.bincount(inv, minlength=len(um)))]
    it_m, im = mx[:, 0].astype(np.int64), mx[:, 1].astype(np.int64)
    cnt = start[im + 1] - start[im]
    it = np.repeat(it_m, cnt)
    ib = bars_of[np.repeat(start[im], cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
    out = []
    for s in range(0, len(it), 1 << 22):
        a, b = it[s:s + (1 << 22)], ib[s:s + (1 << 22)]
        o, d = origin[a], direction[a]
        inv_d = 1.0 / np.where(d == 0.0, 1e-300, d)
        t1, t2 = (lo[b] - o) * inv_d, (hi[b] - o) * inv_d
        t_in = np.maximum(np.minimum(t1, t2).max(1), 0.0)
        t_out = np.maximum(t1, t2).min(1)
        k = t_out > t_in
        out.append(np.stack([a[k], b[k], t_in[k], t_out[k], t_out[k] - t_in[k]], 1))
    x = np.concatenate(out)
    return x[np.lexsort((x[:, 1], x[:, 0]))]
