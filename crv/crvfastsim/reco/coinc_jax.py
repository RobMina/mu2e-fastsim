"""JAX twin of reco/coinc.py (the port of Offline CrvCoincidenceFinder), jitted with fixed shapes.

Layout: one row per (event, sector type) group, hits padded to H per group in collection order; one jitted call per
sector type (cuts are static per type). Equivalences used to get fixed shapes:
  - stock findClusters joins until no hit is added and its link test is symmetric, so a cluster is the connected
    component of its seed; components come from the boolean transitive closure (log2 H squarings)
  - checkCoincidence combinations are a dense tensor over K filtered hits per (side, layer) slot; the maxDistance
    widening is a max over passing combinations (order-independent); the stable x-sort is reproduced with ranks
  - only n == coincidenceLayers combinations are checked (cuts are uniform within a sector type: asserted)
Groups with more than H hits, or a (side, layer) with more than K filtered hits, are flagged and redone by coinc.py.
"""
import functools
import os
from functools import partial
from itertools import combinations

import numpy as np

import jax
import jax.numpy as jnp

from . import coinc as C

jax.config.update("jax_enable_x64", True)

H_DEFAULT = 32
K_DEFAULT = 8
TIERS = ((16, 6, 16384), (32, 8, 8192), (64, 12, 2048), (128, 16, 256), (256, 24, 32), (512, 48, 4), (1024, 96, 1))
# oneAPI (Intel PVC): the H=32 kernel aborts the SPIR-V compiler (32-component vector needs SPV_EXT_long_vector)
ONEAPI_SKIP_H = (32,)


def backend_tiers(tiers):
    """Tiers usable on the current backend: on oneAPI those with H in ONEAPI_SKIP_H are dropped (rows go one tier up)."""
    return tuple(t for t in tiers if t[0] not in ONEAPI_SKIP_H) if is_oneapi() else tuple(tiers)


def is_oneapi():
    """True on the JAX oneAPI plugin (Intel GPUs), which reports its platform as 'gpu'."""
    try:
        d = jax.devices()[0]
    except Exception:
        return False
    tags = (getattr(d, "platform", ""), type(d).__name__, getattr(d, "device_kind", ""), getattr(d.client, "platform_version", ""))
    return any(k in str(t).lower() for t in tags for k in ("oneapi", "sycl", "level_zero", "intel"))


K_MAX_4LAYER = 32   # the 4-layer combination tensor is [2, K, K, K, K]: cap K there (overflow -> next tier/numpy)   # (H, K, groups per call)


CUT_COLS = ("pe_threshold", "max_dt_adjacent", "max_dt", "min_slope", "max_slope", "max_slope_diff",
            "coincidence_layers", "min_cluster_pes", "initial_max_distance", "final_max_distance")
TP_COLS = ("max_dt", "p0", "p1", "both", "fiber_signal_speed", "max_time_interval", "time_offset",
           "add_opposite_side_pulses")


class TypeParams:
    """One sector type: compile-time structure (length axis `ld`, combination sizes `ns`) and the type-level settings
    as data (`tp`, TP_COLS). Per-sector cuts travel with each hit (hit_tables()['cuts'], CUT_COLS)."""
    def __init__(self, ld, ns, tp, cuts):
        self.ld, self.ns, self.tp, self.cuts = ld, ns, tp, cuts
        self.uniform = len(set(cuts)) == 1      # all sectors of the type share their cuts: scalar fast path
        self.cu = np.array([float(getattr(cuts[0], c)) for c in CUT_COLS], np.float64)

    @property
    def max_layers(self):
        return max(self.ns)


def type_params(G, cfg=C.FinderConfig()):
    """TypeParams per sector type. The cuts may differ between the sectors of a type (as Run 2 per-sector thresholds):
    the kernel applies them per hit as the stock finder does."""
    lim = C._type_limits(G, cfg)
    out = {}
    for t, L in lim.items():
        secs = [s for s in np.flatnonzero(G.sector_type == t) if G.sector_names[s] in cfg.cuts]
        cuts = [cfg.cuts[G.sector_names[s]] for s in secs]
        lds = {int(G.length_dir[s]) for s in secs}
        assert len(lds) == 1
        nl = [c.coincidence_layers for c in cuts]
        assert set(nl) <= {2, 3, 4}
        ns = tuple(n for n, need in ((2, min(nl) == 2), (3, min(nl) <= 3 <= max(nl)), (4, max(nl) == 4)) if need)
        tp = np.array([L["max_dt"], L["p0"], L["p1"], float(L["side0"] and L["side1"]), cfg.fiber_signal_speed,
                       cfg.max_time_interval, cfg.time_offset, float(cfg.add_opposite_side_pulses)], np.float64)
        out[t] = TypeParams(lds.pop(), ns, tp, cuts)
    return out


def hit_tables(G, status, cfg=C.FinderConfig()):
    """Per-channel (bar*4+sipm) lookups: sector type (-1 = no cuts), time correction, PE factor."""
    lim = C._type_limits(G, cfg)
    nch = 4 * len(G.sector)
    ch = np.arange(nch)
    bar, sipm = ch // 4, ch % 4
    s = G.sector[bar]
    st = G.sector_type[s]
    ok = np.array([G.sector_names[k] in cfg.cuts for k in s])
    p0 = np.array([lim[int(t)]["p0"] if o else 0.0 for t, o in zip(st, ok)])
    p1 = np.array([lim[int(t)]["p1"] if o else 0.0 for t, o in zip(st, ok)])
    tc = np.where(sipm % 2 == 0, (G.pos_side0[s] - p0), (p1 - G.pos_side1[s])) / cfg.fiber_signal_speed
    other = bar * 4 + np.where(sipm < 2, sipm + 2, sipm - 2)
    st_o = np.asarray(status)[other].astype(np.int64)
    comp = np.zeros(nch, bool)
    for b in cfg.compensate_status_bits:
        comp |= ((st_o >> b) & 1).astype(bool)
    cut_tab = np.zeros((len(G.sector_names), len(CUT_COLS)))
    for k, name in enumerate(G.sector_names):
        if name in cfg.cuts:
            cut_tab[k] = [float(getattr(cfg.cuts[name], c)) for c in CUT_COLS]
    return dict(stype=np.where(ok, st, -1), tcorr=tc, pefac=np.where(comp, 2.0, 1.0), cuts=cut_tab[s])


def pack(ev, idx, bar, sipm, time, pe, seq, G, tables, t_type, H=H_DEFAULT, cfg=C.FinderConfig(), events=None):
    """Flat pulses (sorted by event, collection order within) -> padded [Ng, H] hit arrays of one sector type
    (optionally only the given sorted events). Returns arrays, the event of each group row, and the overflow events."""
    ch = bar * 4 + sipm
    m = tables["stype"][ch] == t_type
    if cfg.ignore_secondary_pulses:
        m &= seq == 0
    if events is not None:
        m &= np.isin(ev, events)
    e = ev[m]
    gev, first, cnt = np.unique(e, return_index=True, return_counts=True)
    g = np.repeat(np.arange(len(gev)), cnt)
    r = np.arange(len(e)) - first[g]
    keep = r < H
    g, r = g[keep], r[keep]
    sel = np.flatnonzero(m)[keep]
    b = bar[sel]
    Ng = len(gev)

    def dense(v, fill=0.0, dt=np.float64):
        a = np.full((Ng, H) + v.shape[1:], fill, dt)
        a[g, r] = v
        return a
    out = dict(valid=dense(np.ones(len(sel), bool), False, bool),
               idx=dense(idx[sel], -1, np.int64),
               side=dense(sipm[sel] % 2, 0, np.int64),
               bar=dense(b, -1, np.int64), sipm=dense(sipm[sel], -1, np.int64),
               first=dense(seq[sel] == 0, False, bool),
               layer=dense(G.layer[b], 0, np.int64),
               counter=dense(G.counter_in_type[b], 0, np.int64),
               x=dense(G.x[b]), y=dense(G.y[b]),
               t=dense(time[sel] + tables["tcorr"][ch[sel]]),
               pe=dense(pe[sel] * tables["pefac"][ch[sel]]),
               pos=dense(G.pos[b]),
               cut=dense(tables["cuts"][ch[sel]]))
    return out, gev, gev[cnt > H]


def link_scales(G, P):
    """Largest time and x separation over which any finder step can link two hits of one sector type: filterHits
    (maxTimeDifferenceAdjacentPulses), clustering (maxTimeDifference + opposite-side fibre delay; initialMaxDistance,
    finalMaxDistance, and maxDistance widened at most to |slope| x layer-stack thickness)."""
    max_dt, p0, p1, both, v = P.tp[:5]
    dt_adj = max(c.max_dt_adjacent for c in P.cuts)
    maxd0 = max(c.initial_max_distance for c in P.cuts)
    maxd_f = max(c.final_max_distance for c in P.cuts)
    smax = max(max(abs(c.min_slope), abs(c.max_slope)) for c in P.cuts)
    t_link = max(dt_adj, max_dt + (2.0 * (p1 - p0) / 2.0 / v if both else 0.0))
    return t_link, maxd0, maxd_f, smax


def split_blocks(key, t, x, t_link, x_link, max_iter=16):
    """Refine group ids so that hits are only in one block if they are chained by gaps <= the link scales, alternately
    in time and in x, until stable. Hits in different blocks can never be linked by the finder (exact split)."""
    key = np.unique(key, return_inverse=True)[1].ravel()
    nb = key.max() + 1 if len(key) else 0
    for _ in range(max_iter):
        before = nb
        for v, lim in ((t, t_link), (x, x_link)):
            o = np.lexsort((v, key))
            ko, vo = key[o], v[o]
            new = np.r_[True, (ko[1:] != ko[:-1]) | (np.diff(vo) > lim)]
            key = np.empty_like(key)
            key[o] = np.cumsum(new) - 1
            nb = int(key.max()) + 1 if len(key) else 0
        if nb == before:
            break
    return key


def _closure(A):
    """Reachability of a symmetric boolean adjacency [..., H, H] (diagonal set for valid hits)."""
    R = A.astype(jnp.float32)
    n = int(np.ceil(np.log2(max(A.shape[-1], 2))))
    for _ in range(n):
        R = (jnp.matmul(R, R, precision=jax.lax.Precision.HIGHEST) > 0).astype(jnp.float32)
    return R > 0


def _labels(A, valid):
    R = _closure(A)
    return jnp.where(valid, jnp.argmax(R, axis=-1), -1)


def _attach_opposite(lab, kept, filt, comp, idx, bar, sipm, first, t, enabled, window):
    """Snapshot attachment; O(G H²) storage, with stock seed-order and candidate-order ties."""
    H = lab.shape[1]
    ordinal = jnp.arange(H)
    side = sipm % 2
    members = kept[:, None, :] & (lab[:, None, :] == ordinal[None, :, None])
    same_bar = bar[:, :, None] == bar[:, None, :]
    same_side = side[:, :, None] == side[:, None, :]
    occupied = jnp.matmul((same_bar & same_side).astype(jnp.float32),
                          jnp.swapaxes(members.astype(jnp.float32), 1, 2)) > 0
    anchor_occupied = jnp.take_along_axis(occupied, jnp.maximum(lab, 0)[:, None, :], axis=2)
    eligible = (enabled & filt & ~kept & first)[:, :, None] & (kept & first)[:, None, :]
    eligible &= same_bar & ~same_side & ~anchor_occupied
    dt = jnp.abs(t[:, :, None] - t[:, None, :])
    eligible &= dt < window
    distance = jnp.where(eligible, dt, jnp.inf)
    best_dt = jnp.min(distance, -1)
    # Stock seeds follow side, initial component seed, then pulse Ptr order.
    order = jnp.lexsort((idx, comp, side), axis=-1)
    rank = jnp.argsort(order, axis=-1)
    root_rank = jnp.min(jnp.where(members, rank[:, None, :], H), -1)
    anchor_rank = jnp.take_along_axis(root_rank, jnp.maximum(lab, 0), axis=1)
    tied = eligible & (distance == best_dt[:, :, None])
    choice = jnp.argmin(jnp.where(tied, anchor_rank[:, None, :], H), -1)
    chosen_lab = jnp.take_along_axis(lab, choice, axis=1)
    candidate = jnp.isfinite(best_dt)
    same_slot = (same_bar & (sipm[:, :, None] == sipm[:, None, :]) &
                 (chosen_lab[:, :, None] == chosen_lab[:, None, :]))
    better = (best_dt[:, None, :] < best_dt[:, :, None]) | (
        (best_dt[:, None, :] == best_dt[:, :, None]) & (ordinal[None, None, :] < ordinal[None, :, None]))
    selected = candidate & ~jnp.any(same_slot & candidate[:, None, :] & better, -1)
    return jnp.where(selected, chosen_lab, lab), kept | selected


@partial(jax.jit, static_argnames=("ld", "ns", "uniform", "K"))
def find_dense(valid, idx, side, bar, sipm, first, layer, counter, x, y, t, pe, pos, cut, tp, cu, ld, ns, uniform, K=K_DEFAULT):
    """One sector type, [G, H] hits -> per-hit cluster label (root hit, -1 = no cluster) and per-root properties.
    cut [G, H, len(CUT_COLS)]: each hit's sector cuts; cu [len(CUT_COLS)]: the shared cuts when `uniform`;
    tp [len(TP_COLS)]: type-level settings. Static: ld (length axis), ns (combination sizes that can be needed),
    uniform (one set of cuts for the whole type: scalar path, same result), K (slots per side and layer)."""
    max_dt, p0, p1, both, v, mti, toff, attach = (tp[i] for i in range(len(TP_COLS)))
    both = both > 0.5
    src = cu if uniform else cut
    (thr, dt_adj, cmax_dt, lo_h, hi_h, msd_h, ncl_h, minpe_h, maxd0_h, maxdf_h) = (src[..., i] for i in range(len(CUT_COLS)))
    pair = (lambda a: a) if uniform else (lambda a: jnp.maximum(a[:, :, None], a[:, None, :]))
    row = (lambda a: a) if uniform else (lambda a: a[:, :, None])
    Gn, H = valid.shape
    I = jnp.eye(H, dtype=bool)
    vv = valid[:, :, None] & valid[:, None, :]
    dx = jnp.abs(x[:, :, None] - x[:, None, :])
    dtt = jnp.abs(t[:, :, None] - t[:, None, :])
    same_side = side[:, :, None] == side[:, None, :]
    # filterHits (per side, same layer, |dt| <= maxTimeDifferenceAdjacentPulses)
    near = vv & same_side & (layer[:, :, None] == layer[:, None, :]) & (dtt <= row(dt_adj))
    dc = counter[:, None, :] - counter[:, :, None]
    pej = jnp.broadcast_to(pe[:, None, :], near.shape)
    s0 = jnp.sum(jnp.where(near & (dc == 0), pej, 0.0), -1)
    s1 = jnp.sum(jnp.where(near & (dc == -1), pej, 0.0), -1)
    s2 = jnp.sum(jnp.where(near & (dc == 1), pej, 0.0), -1)
    filt = valid & ((s0 + s1 >= thr) | (s0 + s2 >= thr))
    # first clustering per side: |dx| <= initialMaxDistance, |dt| < maxTimeDifference
    ff = filt[:, :, None] & filt[:, None, :]
    A1 = ff & same_side & (((dx <= pair(maxd0_h)) & (dtt < max_dt)) | I)
    comp = _labels(A1, filt)
    if not uniform:   # coincidence layers required in each first-stage cluster: min and max over its hits
        same_c = filt[:, :, None] & filt[:, None, :] & (comp[:, :, None] == comp[:, None, :])
        nmin = jnp.min(jnp.where(same_c, ncl_h[:, None, :], 99.0), -1)
        nmax = jnp.max(jnp.where(same_c, ncl_h[:, None, :], -1.0), -1)
    # slot of each filtered hit within its (side, layer)
    sl = filt[:, None, :] & same_side & (layer[:, :, None] == layer[:, None, :]) & \
        (jnp.arange(H)[None, None, :] < jnp.arange(H)[None, :, None])
    slot = jnp.sum(sl, -1)
    overflow = jnp.any(filt & (slot >= K), -1)
    # slot table [G, 2, 4, K] -> hit index (H = empty)
    put = filt & (slot < K)
    flat = jnp.where(put, (side * C.N_LAYERS + layer) * K + slot, 2 * C.N_LAYERS * K)
    tab = jnp.full((Gn, 2 * C.N_LAYERS * K + 1), H, jnp.int64)
    tab = tab.at[jnp.arange(Gn)[:, None], flat].set(jnp.arange(H)[None, :])[:, :-1].reshape(Gn, 2, C.N_LAYERS, K)
    pad = lambda a, f: jnp.concatenate([a, jnp.full((Gn, 1), f, a.dtype)], 1)
    gi = jnp.arange(Gn)[:, None, None, None]
    sx, sy, st_ = pad(x, 0.0)[gi, tab], pad(y, 0.0)[gi, tab], pad(t, 0.0)[gi, tab]
    sc, sv = pad(comp, -2)[gi, tab], tab < H
    if not uniform:
        sdt, slo, shi, smsd, sncl = (pad(a, 0.0)[gi, tab] for a in (cmax_dt, lo_h, hi_h, msd_h, ncl_h))
        snmin, snmax = pad(nmin, 99.0)[gi, tab], pad(nmax, -1.0)[gi, tab]
    wid = jnp.full((Gn, 2, C.N_LAYERS, K), -jnp.inf)
    for layers in (c for n_ in ns for c in combinations(range(C.N_LAYERS), n_)):
        n = len(layers)

        def at(a, p, l):   # slot attribute of layer l on combo axis p: [G, 2, 1.., K, ..1]
            shp = [Gn, 2] + [1] * n
            shp[2 + p] = K
            return a[:, :, l, :].reshape(shp)
        X = [at(sx, p, l) for p, l in enumerate(layers)]
        Y = [at(sy, p, l) for p, l in enumerate(layers)]
        T = [at(st_, p, l) for p, l in enumerate(layers)]
        ok = at(sv, 0, layers[0])
        for p in range(1, n):
            ok = ok & at(sv, p, layers[p]) & (at(sc, p, layers[p]) == at(sc, 0, layers[0]))
        tmax, tmin = T[0], T[0]
        for p in range(1, n):
            tmax, tmin = jnp.maximum(tmax, T[p]), jnp.minimum(tmin, T[p])
        if uniform:
            c_dt, c_lo, c_hi, msd = cmax_dt, lo_h, hi_h, msd_h
        else:
            A_ = lambda a: [at(a, p, l) for p, l in enumerate(layers)]
            red = lambda f, xs: functools.reduce(f, xs)
            c_dt, c_lo, c_hi, msd = (red(f, A_(a)) for f, a in ((jnp.maximum, sdt), (jnp.minimum, slo),
                                                                (jnp.maximum, shi), (jnp.maximum, smsd)))
            # this combination size is checked for the cluster (nmin/nmax) and not every hit asks for more layers
            nmn, nmx = at(snmin, 0, layers[0]), at(snmax, 0, layers[0])
            need = (nmn == 2) if n == 2 else ((nmn <= 3) & (nmx >= 3)) if n == 3 else (nmx == 4)
            ok = ok & need & ~red(jnp.logical_and, [c > n for c in A_(sncl)])
        ok = ok & ~(tmax - tmin > c_dt)
        S = []
        for p in range(n - 1):
            s = (X[p + 1] - X[p]) / jnp.where(Y[p + 1] == Y[p], 1.0, Y[p + 1] - Y[p])
            ok = ok & ~((s < c_lo) | (s > c_hi))
            S.append(s)
        if n > 2:
            ok = ok & ~(jnp.abs(S[0] - S[1]) > msd)
        if n > 3:
            ok = ok & ~((jnp.abs(S[0] - S[2]) > msd) | (jnp.abs(S[1] - S[2]) > msd))
        rank = [sum(((X[q] < X[p]) | ((X[q] == X[p]) & (q < p))).astype(jnp.int64) for q in range(n) if q != p)
                for p in range(n)]
        for p, l in enumerate(layers):
            w = jnp.zeros_like(ok, dtype=jnp.float64)
            for q in range(n):
                if q == p:
                    continue
                w = jnp.maximum(w, jnp.where(rank[q] == rank[p] + 1, X[q] - X[p], 0.0))
                w = jnp.maximum(w, jnp.where(rank[q] == rank[p] - 1, X[p] - X[q], 0.0))
            w = jnp.where(ok, jnp.broadcast_to(w, ok.shape), -jnp.inf)
            axes = tuple(2 + q for q in range(n) if q != p)
            red = jnp.max(w, axis=axes).reshape(Gn, 2, K)
            wid = wid.at[:, :, l, :].max(red)
    hw = jnp.where(put, wid.reshape(Gn, -1)[jnp.arange(Gn)[:, None], jnp.minimum(flat, 2 * C.N_LAYERS * K - 1)],
                   -jnp.inf)
    kept = put & (hw > -jnp.inf)
    maxd = jnp.maximum(maxdf_h, hw)
    # final clustering across both readout sides
    kk = kept[:, :, None] & kept[:, None, :]
    mt = max_dt + jnp.where(both & ~same_side, 2.0 * (p1 - p0) / 2.0 / v, 0.0)
    md = jnp.maximum(maxd[:, :, None], maxd[:, None, :])
    A2 = kk & (((dx <= md) & (dtt < mt)) | I)
    lab = _labels(A2, kept)
    lab, kept = _attach_opposite(lab, kept, filt, comp, idx, bar, sipm, first, t,
                                 both & (attach > 0.5), max_dt + 2.0 * (p1 - p0) / 2.0 / v)
    # clusterProperties per root
    M = (lab[:, None, :] == jnp.arange(H)[None, :, None]) & kept[:, None, :]
    Mf = M.astype(jnp.float64)
    sm = lambda a: jnp.einsum("grh,gh->gr", Mf, a)
    tot = sm(pe)
    is_root = jnp.any(M, -1)
    min_pe = minpe_h if uniform else jnp.min(jnp.where(M, minpe_h[:, None, :], jnp.inf), -1)
    ok_r = is_root & ~(tot < min_pe)
    safe = jnp.where(tot != 0, tot, 1.0)
    posw = jnp.einsum("grh,ghk->grk", Mf, pos * pe[..., None]) / safe[..., None]
    spx, spy = sm(pe * x), sm(pe * y)
    slope = (tot * sm(pe * x * y) - spx * spy) / (tot * sm(pe * y * y) - spy ** 2)
    big = jnp.inf
    start = jnp.min(jnp.where(M, t[:, None, :], big), -1)
    end = jnp.max(jnp.where(M, t[:, None, :], -big), -1)
    nh, spe, stt, tmn, tmx = [], [], [], [], []
    for s in (0, 1):
        Ms = M & ((side == s) & (pe > 0))[:, None, :]
        nh.append(jnp.sum(Ms, -1))
        e_ = jnp.einsum("grh,gh->gr", Ms.astype(jnp.float64), pe)
        spe.append(e_)
        stt.append(jnp.where(e_ > 0, jnp.einsum("grh,gh->gr", Ms.astype(jnp.float64), t * pe) /
                             jnp.where(e_ > 0, e_, 1.0), 0.0))
        tmn.append(jnp.min(jnp.where(Ms, t[:, None, :], big), -1))
        tmx.append(jnp.max(jnp.where(Ms, t[:, None, :], -big), -1))
    has0, has1 = nh[0] > 0, nh[1] > 0
    org_c = 0.5 * (v * (stt[0] - stt[1]) + (p0 + p1))
    calc = has0 & has1 & (p0 < org_c) & (org_c < p1) & (tmx[0] - tmn[0] < mti) & (tmx[1] - tmn[1] < mti)
    org_p = posw[..., ld]
    org = jnp.where(calc, org_c, org_p)
    t0 = stt[0] - (org - p0) / v
    t1 = stt[1] - (p1 - org) / v
    avg = jnp.where(calc, t0, jnp.where(has0 & has1, 0.5 * (t0 + t1), jnp.where(has0, t0, t1)))
    posw = posw.at[..., ld].set(org)
    lay = jnp.stack([jnp.any(M & (layer == l)[:, None, :], -1) for l in range(C.N_LAYERS)], -1)
    return dict(label=jnp.where(kept, lab, -1), root=ok_r, pe=tot, start=start, end=end, slope=slope,
                avgtime=avg - toff, poscalc=calc, pos=posw, sidepe=jnp.stack(spe, -1), sidetime=jnp.stack(stt, -1),
                sidehits=jnp.stack(nh, -1), layers=lay, overflow=overflow)


SPLIT_MIN = 64   # only (event, sector type) groups larger than this are split into blocks
OUT_KEYS = ("label", "root", "pe", "start", "end", "slope", "avgtime", "poscalc", "overflow")
POS_KEYS = OUT_KEYS + ("pos",)   # with_pos: cluster positions too (transferred only when asked for)


def _compact_dispatch():
    """Experimental host dispatch only; unset/any value other than '1' keeps the baseline."""
    return os.environ.get("CRVFASTSIM_COMPACT_DISPATCH") == "1"


class _DispatchOutputs(list):
    """Keep the dispatch mode with pending outputs, even if the environment changes before collection."""
    def __init__(self, compact=None):
        super().__init__()
        self.compact = _compact_dispatch() if compact is None else compact


def _row_capacity(n, batch, compact):
    """Bound partial row padding by the caller/backend cap; never change hit/slot capacities."""
    return min(batch, 1 << (max(1, int(n)) - 1).bit_length()) if compact else batch


def _real_rows_to_host(outs, keys):
    """Slice the leading *row* axis on device, then transfer all requested output leaves together.

    Labels are hit-column roots, not flattened padded-row indices; do not compact the hit axis.
    Pulse outputs likewise retain every pulse slot and every per-row recovery flag.
    """
    return jax.device_get([{k: d[k][:n] for k in keys} for d, n in outs])


def _dispatch_tier(arr, P, K, batch, compact=None):
    """Launch the jitted kernel on fixed-shape batches without waiting (JAX async dispatch)."""
    Ng = len(arr["valid"])
    outs = _DispatchOutputs(compact)
    for a in range(0, Ng, batch):
        sl = {k: v[a:a + batch] for k, v in arr.items()}
        n = len(sl["valid"])
        capacity = _row_capacity(n, batch, outs.compact)
        if n < capacity:   # baseline: fixed tier batch; opt-in: bounded power-of-two tail
            sl = {k: np.concatenate([v, np.zeros((capacity - n,) + v.shape[1:], v.dtype)]) for k, v in sl.items()}
        d = find_dense(*[jnp.asarray(sl[k]) for k in ("valid", "idx", "side", "bar", "sipm", "first", "layer", "counter", "x", "y", "t",
                                                      "pe", "pos", "cut")], jnp.asarray(P.tp), jnp.asarray(P.cu), ld=P.ld, ns=P.ns,
                       uniform=P.uniform, K=K)
        outs.append(({k: d[k] for k in POS_KEYS}, n))
    return outs


def _collect_tier(outs, keys=OUT_KEYS):
    if getattr(outs, "compact", False):
        host = _real_rows_to_host(outs, keys)
        return {k: np.concatenate([d[k] for d in host]) for k in keys}
    host = [({k: np.asarray(d[k]) for k in keys}, n) for d, n in outs]
    return {k: np.concatenate([d[k][:n] for d, n in host]) for k in keys}


def _run_tier(arr, P, K, batch, keys=OUT_KEYS, compact=None):
    return _collect_tier(_dispatch_tier(arr, P, K, batch, compact), keys)


def _kfor(P, K):
    return min(K, K_MAX_4LAYER) if P.max_layers == 4 else K


def _groups(tt, P, ev, idx, bar, sipm, time, pe, seq, G, tables, cfg, split, split_min, timers):
    """Hits of one sector type as kernel groups: (event) or, for groups above split_min hits, exact blocks."""
    import time as _time
    ch = bar * 4 + sipm
    m = tables["stype"][ch] == tt
    if cfg.ignore_secondary_pulses:
        m &= seq == 0
    sel = np.flatnonzero(m)
    e = ev[sel]
    if not split or not len(sel):
        return dict(g=e, sel=sel, gev_of=None)
    t0 = _time.perf_counter()
    ue, inv, cnt = np.unique(e, return_inverse=True, return_counts=True)
    big = cnt[inv] > split_min
    gid = inv.astype(np.int64)
    if big.any():
        t_link, maxd0, maxd_f, smax = link_scales(G, P)
        bars = np.flatnonzero(tables["stype"][np.arange(0, 4 * len(G.sector), 4)] == tt)
        ys = G.y[bars]
        x_link = max(maxd0, maxd_f, smax * (ys.max() - ys.min()))
        bs = sel[big]
        blk = split_blocks(e[big], time[bs] + tables["tcorr"][ch[bs]], G.x[bar[bs]], t_link, x_link)
        gid[big] = len(ue) + blk
    o = np.lexsort((sel, gid))
    sel, gid = sel[o], gid[o]
    g, gid = np.unique(gid, return_inverse=True)
    gev_of = np.zeros(len(g), ev.dtype)
    gev_of[gid] = ev[sel]
    timers["split_s"] = timers.get("split_s", 0.0) + _time.perf_counter() - t0
    timers[f"groups_t{tt}"] = int(len(g))
    return dict(g=gid, sel=sel, gev_of=gev_of)


def find_async(ev, idx, bar, sipm, time, pe, seq, G, status, cfg=C.FinderConfig(), tiers=TIERS, split=True,
               split_min=SPLIT_MIN, timers=None, params=None, tables=None, with_pos=False):
    """Host preparation and kernel launch for all sector types and first-choice tiers; returns a handle for finish().
    The GPU starts on the first type while the host prepares the next. with_pos: finish() also returns cl_pos [n, 3]
    (Mu2e frame, length axis from the timing when cl_poscalc). CRVFASTSIM_COMPACT_DISPATCH=1 opts into
    bounded row padding and real-row-only grouped host transfer; H/K and recovery are unchanged."""
    import time as _time
    tiers = backend_tiers(tiers)
    compact = _compact_dispatch()
    timers = timers if timers is not None else {}
    params = params or type_params(G, cfg)
    tables = tables or hit_tables(G, status, cfg)
    jobs = []
    for tt, P in sorted(params.items()):
        gr = _groups(tt, P, ev, idx, bar, sipm, time, pe, seq, G, tables, cfg, split, split_min, timers)
        sel = gr["sel"]
        gcols = (gr["g"], idx[sel], bar[sel], sipm[sel], time[sel], pe[sel], seq[sel])
        gev, cnt = np.unique(gr["g"], return_counts=True)
        tier = np.searchsorted([h for h, _, _ in tiers], cnt)       # smallest tier with H >= count
        for ti, (H, K, batch) in enumerate(tiers):
            todo = tier == ti
            if not todo.any():
                continue
            t0 = _time.perf_counter()
            arr, tg, _ = pack(*gcols, G, tables, tt, H, cfg, events=gev[todo])
            timers["pack_s"] = timers.get("pack_s", 0.0) + _time.perf_counter() - t0
            t0 = _time.perf_counter()
            jobs.append(dict(tt=tt, P=P, ti=ti, arr=arr, tg=tg, outs=_dispatch_tier(arr, P, _kfor(P, K), batch, compact)))
            timers["dispatch_s"] = timers.get("dispatch_s", 0.0) + _time.perf_counter() - t0
            timers[f"groups_H{H}"] = timers.get(f"groups_H{H}", 0) + len(tg)
        jobs.append(dict(tt=tt, meta=True, gcols=gcols, gev_of=gr["gev_of"], gev=gev, tier=tier))
    return dict(jobs=jobs, cols=(ev, idx, bar, sipm, time, pe, seq), G=G, status=status, cfg=cfg, tiers=tiers,
                tables=tables, timers=timers, with_pos=with_pos, compact=compact)


def finish(h, fallback=True):
    """Collect the kernels of a find_async handle; overflowing groups are promoted to the next tier (synchronously),
    groups beyond the last tier send their event to coinc.py. Returns cl_* per cluster and the member table mem_*."""
    import time as _time
    G, cfg, tiers, tables, timers = h["G"], h["cfg"], h["tiers"], h["tables"], h["timers"]
    keys = POS_KEYS if h.get("with_pos") else OUT_KEYS
    rows, mems, redo = [], [], set()
    ncl = [0]

    def take(d, arr, tg, tt, to_ev):
        good = ~d["overflow"]
        g, r = np.nonzero(d["root"] & good[:, None])
        rows.append(dict(cl_event=to_ev(tg[g]), cl_type=np.full(len(g), tt), **{f"cl_{k}": d[k][g, r] for k in
                         ("pe", "start", "end", "slope", "avgtime", "poscalc") + (("pos",) if "pos" in d else ())}))
        cmap = np.full(d["root"].shape, -1, np.int64)
        cmap[g, r] = ncl[0] + np.arange(len(g))
        ncl[0] += len(g)
        hg, hh = np.nonzero((d["label"] >= 0) & good[:, None])
        cl = cmap[hg, d["label"][hg, hh]]
        mems.append(dict(mem_cl=cl[cl >= 0], mem_idx=arr["idx"][hg, hh][cl >= 0], mem_event=to_ev(tg[hg][cl >= 0])))
        return tg[d["overflow"]]

    t0 = _time.perf_counter()
    pend = {}
    for j in h["jobs"]:
        if j.get("meta"):
            continue
        meta = next(m for m in h["jobs"] if m.get("meta") and m["tt"] == j["tt"])
        to_ev = (lambda b, m=meta: m["gev_of"][b]) if meta["gev_of"] is not None else (lambda b: b)
        ov = take(_collect_tier(j["outs"], keys), j["arr"], j["tg"], j["tt"], to_ev)
        if len(ov):
            pend.setdefault(j["tt"], []).append((j["ti"], ov))
    timers["kernel_s"] = timers.get("kernel_s", 0.0) + _time.perf_counter() - t0
    for tt, lst in pend.items():
        meta = next(m for m in h["jobs"] if m.get("meta") and m["tt"] == tt)
        to_ev = (lambda b, m=meta: m["gev_of"][b]) if meta["gev_of"] is not None else (lambda b: b)
        P = type_params(G, cfg)[tt]
        for ti0, ov in lst:
            ti = ti0 + 1
            while len(ov) and ti < len(tiers):
                H, K, batch = tiers[ti]
                arr, tg, _ = pack(*meta["gcols"], G, tables, tt, H, cfg, events=ov)
                d = _run_tier(arr, P, _kfor(P, K), batch, keys, h.get("compact", False))
                timers[f"promoted_H{H}"] = timers.get(f"promoted_H{H}", 0) + len(tg)
                ov = take(d, arr, tg, tt, to_ev)
                ti += 1
            redo.update(to_ev(ov).tolist())
        # groups larger than the last tier never ran
    for m in h["jobs"]:
        if m.get("meta"):
            to_ev = (lambda b, mm=m: mm["gev_of"][b]) if m["gev_of"] is not None else (lambda b: b)
            redo.update(to_ev(m["gev"][m["tier"] >= len(tiers)]).tolist())
    cat = lambda L, k: np.concatenate([x[k] for x in L]) if L else np.zeros(0)
    out = {k: cat(rows, k) for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_slope", "cl_avgtime",
                                      "cl_poscalc")}
    out.update({k: cat(mems, k) for k in ("mem_cl", "mem_idx", "mem_event")})
    if h.get("with_pos"):
        out["cl_pos"] = np.concatenate([x["cl_pos"] for x in rows]) if rows else np.zeros((0, 3))
    redo = np.array(sorted(redo), np.int64)
    out["fallback_events"] = redo
    if fallback and len(redo):
        t0 = _time.perf_counter()
        _replace_with_numpy(out, redo, *h["cols"], G, h["status"], cfg)
        timers["fallback_s"] = timers.get("fallback_s", 0.0) + _time.perf_counter() - t0
    return out


def find(ev, idx, bar, sipm, time, pe, seq, G, status, cfg=C.FinderConfig(), tiers=TIERS, fallback=True,
         timers=None, split=True, split_min=SPLIT_MIN):
    """Flat pulses of many events (sorted by event, collection order within) -> clusters (find_async + finish).
    split: (event, sector type) groups above split_min hits are cut into independent blocks at time and x gaps larger
    than any link (split_blocks, exact); each kernel row runs in the smallest (H, K) tier it fits, promoted on
    overflow; rows beyond the last tier send their event to coinc.py ('fallback_events')."""
    return finish(find_async(ev, idx, bar, sipm, time, pe, seq, G, status, cfg, tiers, split, split_min, timers),
                  fallback)


def _replace_with_numpy(out, redo, ev, idx, bar, sipm, time, pe, seq, G, status, cfg):
    """Drop the JAX clusters of the redo events and append coinc.find_event results for them."""
    keep_cl = ~np.isin(out["cl_event"], redo)
    newid = np.cumsum(keep_cl) - 1
    km = keep_cl[out["mem_cl"].astype(np.int64)] if len(out["mem_cl"]) else np.zeros(0, bool)
    pos = "cl_pos" in out
    for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_slope", "cl_avgtime", "cl_poscalc") + \
            (("cl_pos",) if pos else ()):
        out[k] = out[k][keep_cl]
    out["mem_cl"] = newid[out["mem_cl"].astype(np.int64)[km]]
    out["mem_idx"], out["mem_event"] = out["mem_idx"][km], out["mem_event"][km]
    lo, hi = np.searchsorted(ev, redo), np.searchsorted(ev, redo, side="right")
    add = {k: [] for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_slope", "cl_avgtime",
                           "cl_poscalc", "mem_cl", "mem_idx", "mem_event")}
    nxt = len(out["cl_event"])
    for e, a, b in zip(redo, lo, hi):
        p = dict(idx=idx[a:b], bar=bar[a:b], sipm=sipm[a:b], time=time[a:b], pe=pe[a:b], seq=seq[a:b])
        for r in C.find_event(p, G, status, cfg, vectorized=True):
            for k, src in (("cl_type", "sectortype"), ("cl_pe", "pe"), ("cl_start", "start"), ("cl_end", "end"),
                           ("cl_slope", "slope"), ("cl_avgtime", "avgtime"), ("cl_poscalc", "poscalc")):
                add[k].append(r[src])
            add["cl_event"].append(e)
            if pos:
                add.setdefault("cl_pos", []).append(np.asarray(r["pos"], float))
            add["mem_cl"] += [nxt] * len(r["pulses"])
            add["mem_idx"] += list(r["pulses"])
            add["mem_event"] += [e] * len(r["pulses"])
            nxt += 1
    for k, v in add.items():
        out[k] = np.concatenate([out[k], np.asarray(v, dtype=out[k].dtype if len(out[k]) else None)]) if v else out[k]


def cluster_keys(out):
    """Set of (event, sectortype, sorted pulse idx tuple) for comparisons."""
    o = np.lexsort((out["mem_idx"], out["mem_cl"]))
    cl, ix = out["mem_cl"][o].astype(np.int64), out["mem_idx"][o].astype(np.int64)
    b = np.flatnonzero(np.diff(cl)) + 1
    groups = np.split(ix, b)
    ids = cl[np.r_[0, b]] if len(cl) else []
    return {(int(out["cl_event"][i]), int(out["cl_type"][i]), tuple(g.tolist())) for i, g in zip(ids, groups)}
