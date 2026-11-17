"""Muon crossings of a designed CRV and the per-muon probability that no sector makes a coincidence.

Crossings: per sector, the muon lines are rotated into the sector-local frame, where every bar is an axis-aligned box,
and crossed with the core slab method (tracks first culled by the sector's bounding box).

Veto probability, response fluctuations averaged analytically: per (muon, sector, layer, read-out end) the expected
PE of the crossed counters are summed (the finder's same-plus-adjacent-counter PE sum); the layer is hit if either
end has at least the sector's PE threshold (Poisson); the sector has a coincidence if at least k of its N layers are
hit (Poisson-binomial), k = Design.coincidence_layers, or by default N - (4 - stock required layers), at least 1.
P_miss = prod over sectors (1 - P_coinc); a muon that crosses no counter has P_miss = 1. The finder's time, slope
and distance cuts are not applied.
"""
from __future__ import annotations

import numpy as np
from scipy.special import pdtr

from mu2efastsim import crossings as XC


def sector_crossings(dg, origin, direction, dtype=np.float64, chunk=4096):
    """Rows of crossings: track, bar, local entry [3], local exit [3] (counter-local, mm): (int [m, 2], float [m, 6])."""
    origin = np.asarray(origin, dtype)
    direction = np.asarray(direction, dtype)
    idx, pos = [], []
    for s in range(len(dg.sector_names)):
        bars = np.flatnonzero(dg.sector == s)
        if not len(bars):
            continue
        A = dg.axes[s].astype(dtype)
        o, d = origin @ A.T, direction @ A.T
        c = (dg.center[bars] @ dg.axes[s].T).astype(dtype)
        h = dg.halflen[bars].astype(dtype)
        lo, hi = c - h, c + h
        _, keep = _box_hits(o, d, lo.min(0), hi.max(0))
        tk = np.flatnonzero(keep)
        if not len(tk):
            continue
        x = XC.crossings(o[tk], d[tk], lo, hi, chunk=chunk)
        if not len(x):
            continue
        it = tk[x[:, 0].astype(np.int64)]
        ib = x[:, 1].astype(np.int64)
        p_in = o[it] + d[it] * x[:, 2:3].astype(dtype) - c[ib]
        p_out = o[it] + d[it] * x[:, 3:4].astype(dtype) - c[ib]
        idx.append(np.stack([it, bars[ib]], 1))
        pos.append(np.concatenate([p_in, p_out], 1))
    if not idx:
        return np.zeros((0, 2), np.int64), np.zeros((0, 6), dtype)
    return np.concatenate(idx), np.concatenate(pos).astype(dtype)


def _box_hits(o, d, lo, hi):
    inv = 1.0 / np.where(d == 0, 1e-30, d)
    t1, t2 = (lo - o) * inv, (hi - o) * inv
    t_in = np.maximum(np.minimum(t1, t2).max(1), 0)
    t_out = np.maximum(t1, t2).min(1)
    return t_in, t_out > t_in


def required_layers(n_layers, stock_required, k=None):
    """k of the k-of-N rule: ``k`` if given (Design.coincidence_layers), else N minus the stock spare layers
    (stock: 3 of 4 or 4 of 4), at least 1. Raises if k exceeds N."""
    n = np.asarray(n_layers)
    if k is None:
        return np.maximum(1, n - (4 - np.asarray(stock_required)))
    if np.any(k > n):
        raise ValueError(f"coincidence_layers {k} exceeds the layer count {n_layers}")
    return np.full(n.shape, int(k), np.int64)


def p_coincidence(mu_layer_side, thr, k):
    """mu_layer_side [g, N, 2] expected PE per group (muon, sector), layer and end; thr [g] PE; k [g] required layers.
    Returns P(at least k layers with an end >= thr) [g] (float64)."""
    mu = np.asarray(mu_layer_side, np.float64)
    kk = (np.ceil(np.asarray(thr, np.float64)) - 1.0)[:, None, None]
    below = np.where(mu > 0, pdtr(kk, mu), 1.0)            # P(N <= thr - 1) per end
    p = 1.0 - below.prod(2)                                 # [g, N]
    g, n = p.shape
    dist = np.zeros((g, n + 1))
    dist[:, 0] = 1.0
    for j in range(n):
        q = p[:, j:j + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - q) + dist[:, :-1] * q
        dist[:, 0] *= (1 - q[:, 0])
    cum = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]         # cum[:, m] = P(>= m)
    return cum[np.arange(g), np.asarray(k, np.int64)]


def p_miss(n_tracks, track, sector, layer, pe_sides, n_layers_sec, thr_sec, k_sec):
    """Per track P(no sector coincidence) [n_tracks] and per (track, sector) P_coinc rows (track, sector, p).
    track, sector, layer [m] per crossing; pe_sides [m, 2]."""
    nsec = len(n_layers_sec)
    nmax = int(np.max(n_layers_sec)) if nsec else 1
    out = np.ones(n_tracks)
    if not len(track):
        return out, (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    grp = np.asarray(track, np.int64) * nsec + np.asarray(sector, np.int64)
    ug, gi = np.unique(grp, return_inverse=True)
    mu = np.zeros((len(ug), nmax, 2))
    np.add.at(mu, (gi, np.asarray(layer, np.int64)), np.asarray(pe_sides, np.float64))
    gs = ug % nsec
    pc = p_coincidence(mu, np.asarray(thr_sec)[gs], np.asarray(k_sec)[gs])
    gt = ug // nsec
    np.multiply.at(out, gt, 1.0 - pc)
    return out, (gt, gs, pc)
