"""Fixed-shape strict-v3 crossing metadata without host event grouping."""
from __future__ import annotations

import numpy as np

from . import tier1b as T


def prepare(geom, libbins):
    """Upload immutable geometry and bin edges; no event-dependent preparation."""
    from .tier1b_jax import _jax
    _, j = _jax()
    lid, width = T.layer_slots(geom)
    out = dict(lid=j.asarray(lid), width=j.asarray(width), rank=j.asarray(T.layer_rank(geom)),
               group=j.asarray(T._group_index(geom, np.arange(geom.nbars))),
               module=j.asarray(geom.sector.astype(np.int64) * 1000 + geom.module),
               axis=j.asarray(geom.local_to_world, dtype=j.int64))
    for g in range(len(T.GROUPS)):
        for kind, name in enumerate(("p", "l")):
            out[f"{name}{g}"] = j.asarray(libbins.edge[(libbins.kind == kind) & (libbins.group == g)].to_numpy())
    return out


def metadata(x, g):
    """Return padded visits, bin/context and UV cells; invalid rows never participate."""
    from .tier1b_jax import _jax
    jax, j = _jax()
    valid, bar = x["valid"], x["bar"]
    n = len(bar)
    ii, maximum = j.arange(n, dtype=j.int64), j.iinfo(j.int64).max
    identity = j.stack([x[c] for c in T.KEY + ["sp"]], axis=1)
    order = j.lexsort((ii, *[identity[:, k] for k in reversed(range(identity.shape[1]))], ~valid))
    ids, ok = identity[order], valid[order]
    new = ok & j.concatenate((j.ones(1, bool), j.any(ids[1:] != ids[:-1], axis=1)))
    group = j.maximum(j.cumsum(new) - 1, 0)
    first = j.full(n, maximum, j.int64).at[group].min(j.where(ok, order, maximum))
    rank = j.empty(n, j.int64).at[j.argsort(first, stable=True)].set(ii)
    mu = j.empty(n, j.int64).at[order].set(j.where(ok, rank[group], -1))
    safe_mu = j.maximum(mu, 0)
    order = j.lexsort((ii, x["t0"], mu, ~valid))
    ms, bs, ts, ok = mu[order], bar[order], x["t0"][order], valid[order]
    lids = g["lid"][bs]
    first_mu = j.concatenate((j.ones(1, bool), ms[1:] != ms[:-1]))
    new = ok & (first_mu | j.concatenate((j.ones(1, bool), lids[1:] != lids[:-1])))
    vg = j.maximum(j.cumsum(new) - 1, 0)
    base = j.full(n, maximum, j.int64).at[j.maximum(ms, 0)].min(j.where(ok, vg, maximum))
    vs = j.where(ok, vg - base[j.maximum(ms, 0)], 0)
    vis = j.empty(n, j.int64).at[order].set(vs)
    first = j.full(n, maximum, j.int64).at[vg].min(j.where(ok, ii, maximum))
    present = first < n
    oi = order[j.minimum(first, n - 1)]
    width = g["width"][bs]
    wmin = j.full(n, maximum, j.int64).at[vg].min(j.where(ok, width, maximum))
    wmax = j.full(n, -1, j.int64).at[vg].max(j.where(ok, width, -1))
    visits = dict(key=j.where(present, mu[oi] * 100000 + vis[oi], maximum),
                  lid=j.where(present, g["lid"][bar[oi]], 0),
                  wmin=j.where(present, wmin, 0), wmax=j.where(present, wmax, 0),
                  entry=j.where(present[:, None], x["entry"][oi], 0.),
                  vkey=j.sort(j.where(valid, mu * 1000000 + g["lid"][bar], maximum)))
    counts = j.zeros(n, j.int64).at[safe_mu].add(valid.astype(j.int64))
    pos, mod = x["entry"][order], g["module"][bs]
    def cond(state):
        return state[0] < j.max(counts)
    def body(state):
        k, upstream = state
        prev = j.maximum(ii - k, 0)
        dist = j.sqrt(j.sum((pos - pos[prev]) ** 2, axis=1))
        hit = ok & (ii >= k) & (ms == ms[prev]) & (ts[prev] < ts) & (mod[prev] != mod) & (dist < T.UPSTREAM_MM)
        return k + 1, upstream | hit
    _, upstream = jax.lax.while_loop(cond, body, (j.asarray(1), j.zeros(n, bool)))
    up = j.empty(n, bool).at[order].set(upstream)
    ctx = j.minimum(g["rank"][bar], 3) * 2 + up.astype(j.int64)
    p = j.linalg.norm(x["mom"], axis=1)
    bins = j.zeros(n, j.int64)
    for gi in range(len(T.GROUPS)):
        b = (gi * T.N_PBINS + j.searchsorted(g[f"p{gi}"], p, side="left")) * T.N_LBINS
        b += j.searchsorted(g[f"l{gi}"], x["length"], side="left")
        bins = j.where(g["group"][bar] == gi, b, bins)
    last_t = j.full(n, -j.inf).at[safe_mu].max(j.where(valid, x["t0"], -j.inf))
    last = valid & (x["t0"] == last_t[safe_mu])
    bits = j.left_shift(j.ones(n, j.int64), j.maximum(g["group"][bar], 0))
    mask = j.zeros(n, j.int64)
    for gi in range(len(T.GROUPS)):
        seen = j.zeros(n, j.int64).at[safe_mu].max((valid & (bits == (1 << gi))).astype(j.int64))
        mask += seen[safe_mu] * (1 << gi)
    along = j.take_along_axis(x["entry"], g["axis"][bar, 2:3], axis=1)[:, 0]
    pc = j.searchsorted(j.asarray(T.UV_P_EDGES), p, side="right")
    tail = last.astype(j.int64) * (len(T.UV_P_EDGES) + 1) + pc
    nt = 2 * (len(T.UV_P_EDGES) + 1)
    cells = []
    for wb, am in zip(T.UV_W_BARS, T.UV_A_MM):
        cell = (g["lid"][bar] * 1000 + g["width"][bar] // wb) * 100000 + j.floor(along / am).astype(j.int64) + 50000
        cells.append((cell * 16 + mask) * nt + tail)
    cells.extend(((g["lid"][bar] * 16 + mask) * nt + tail, g["lid"][bar] * nt + tail,
                  (g["group"][bar] + 1) * nt + tail))
    fx = dict(x, mu=mu, vis=vis)
    return fx, visits, dict(bins=bins, ctx=ctx, cells=j.stack(cells, axis=1), last=last, mask=mask, valid=valid)


def crossing_arrays(frame):
    """Compatibility ingress for host tables; device callers can supply this mapping directly."""
    from .tier1b_jax import _bucket, _jax, _pad
    _, j = _jax()
    n, size = len(frame), _bucket(len(frame))
    out = {c: j.asarray(_pad(frame[c].to_numpy().astype(np.int64), size)) for c in T.KEY + ["sp", "bar"]}
    out.update({c: j.asarray(_pad(frame[v].to_numpy(), size)) for c, v in
                (("entry", ["x0", "y0", "z0"]), ("exit", ["x1", "y1", "z1"]),
                 ("mom", ["px0", "py0", "pz0"]), ("t0", "t0"), ("length", "len"))})
    out["valid"] = j.arange(size) < n
    return out
