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
K_MAX_4LAYER = 32   # the 4-layer combination tensor is [2, K, K, K, K]: cap K there (overflow -> next tier/numpy)   # (H, K, groups per call)


def type_params(G, cfg=C.FinderConfig()):
    """Static per-sector-type parameters (hashable tuples) and per-bar/channel tables for hit building."""
    lim = C._type_limits(G, cfg)
    out = {}
    for t, L in lim.items():
        secs = [s for s in np.flatnonzero(G.sector_type == t) if G.sector_names[s] in cfg.cuts]
        cuts = {cfg.cuts[G.sector_names[s]] for s in secs}
        assert len(cuts) == 1, f"sector type {t}: non-uniform cuts {cuts}"
        lds = {int(G.length_dir[s]) for s in secs}
        assert len(lds) == 1
        c = cuts.pop()
        assert c.coincidence_layers in (2, 3, 4)
        out[t] = (float(c.pe_threshold), float(c.max_dt_adjacent), float(L["max_dt"]), float(c.min_slope),
                  float(c.max_slope), float(c.max_slope_diff), int(c.coincidence_layers), float(c.min_cluster_pes),
                  float(c.initial_max_distance), float(c.final_max_distance), float(L["p0"]), float(L["p1"]),
                  bool(L["side0"] and L["side1"]), lds.pop(), float(cfg.fiber_signal_speed),
                  float(cfg.max_time_interval), float(cfg.time_offset))
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
    return dict(stype=np.where(ok, st, -1), tcorr=tc, pefac=np.where(comp, 2.0, 1.0))


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
               layer=dense(G.layer[b], 0, np.int64),
               counter=dense(G.counter_in_type[b], 0, np.int64),
               x=dense(G.x[b]), y=dense(G.y[b]),
               t=dense(time[sel] + tables["tcorr"][ch[sel]]),
               pe=dense(pe[sel] * tables["pefac"][ch[sel]]),
               pos=dense(G.pos[b]))
    return out, gev, gev[cnt > H]


def link_scales(G, P):
    """Largest time and x separation over which any finder step can link two hits of one sector type: filterHits
    (maxTimeDifferenceAdjacentPulses), clustering (maxTimeDifference + opposite-side fibre delay; initialMaxDistance,
    finalMaxDistance, and maxDistance widened at most to |slope| x layer-stack thickness)."""
    (thr, dt_adj, max_dt, lo, hi, msd, ncl, min_pe, maxd0, maxd_f, p0, p1, both, ld, v, mti, toff) = P
    t_link = max(dt_adj, max_dt + (2.0 * (p1 - p0) / 2.0 / v if both else 0.0))
    return t_link, maxd0, maxd_f, max(abs(lo), abs(hi))


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


@partial(jax.jit, static_argnames=("P", "K"))
def find_dense(valid, idx, side, layer, counter, x, y, t, pe, pos, P, K=K_DEFAULT):
    """One sector type, [G, H] hits -> per-hit cluster label (root hit, -1 = no cluster) and per-root properties."""
    (thr, dt_adj, max_dt, lo, hi, msd, ncl, min_pe, maxd0, maxd_f, p0, p1, both, ld, v, mti, toff) = P
    Gn, H = valid.shape
    I = jnp.eye(H, dtype=bool)
    vv = valid[:, :, None] & valid[:, None, :]
    dx = jnp.abs(x[:, :, None] - x[:, None, :])
    dtt = jnp.abs(t[:, :, None] - t[:, None, :])
    same_side = side[:, :, None] == side[:, None, :]
    # filterHits (per side, same layer, |dt| <= maxTimeDifferenceAdjacentPulses)
    near = vv & same_side & (layer[:, :, None] == layer[:, None, :]) & (dtt <= dt_adj)
    dc = counter[:, None, :] - counter[:, :, None]
    pej = jnp.broadcast_to(pe[:, None, :], near.shape)
    s0 = jnp.sum(jnp.where(near & (dc == 0), pej, 0.0), -1)
    s1 = jnp.sum(jnp.where(near & (dc == -1), pej, 0.0), -1)
    s2 = jnp.sum(jnp.where(near & (dc == 1), pej, 0.0), -1)
    filt = valid & ((s0 + s1 >= thr) | (s0 + s2 >= thr))
    # first clustering per side: |dx| <= initialMaxDistance, |dt| < maxTimeDifference
    ff = filt[:, :, None] & filt[:, None, :]
    A1 = ff & same_side & (((dx <= maxd0) & (dtt < max_dt)) | I)
    comp = _labels(A1, filt)
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
    wid = jnp.full((Gn, 2, C.N_LAYERS, K), -jnp.inf)
    for layers in combinations(range(C.N_LAYERS), ncl):
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
        ok = ok & ~(tmax - tmin > max_dt)
        S = []
        for p in range(n - 1):
            s = (X[p + 1] - X[p]) / jnp.where(Y[p + 1] == Y[p], 1.0, Y[p + 1] - Y[p])
            ok = ok & ~((s < lo) | (s > hi))
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
    maxd = jnp.maximum(maxd_f, hw)
    # final clustering across both readout sides
    kk = kept[:, :, None] & kept[:, None, :]
    mt = max_dt + jnp.where(both & ~same_side, 2.0 * (p1 - p0) / 2.0 / v, 0.0)
    md = jnp.maximum(maxd[:, :, None], maxd[:, None, :])
    A2 = kk & (((dx <= md) & (dtt < mt)) | I)
    lab = _labels(A2, kept)
    # clusterProperties per root
    M = (lab[:, None, :] == jnp.arange(H)[None, :, None]) & kept[:, None, :]
    Mf = M.astype(jnp.float64)
    sm = lambda a: jnp.einsum("grh,gh->gr", Mf, a)
    tot = sm(pe)
    is_root = jnp.any(M, -1)
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


def _run_tier(arr, P, K, batch):
    Ng = len(arr["valid"])
    outs = []
    for a in range(0, Ng, batch):
        sl = {k: v[a:a + batch] for k, v in arr.items()}
        n = len(sl["valid"])
        if n < batch:   # one static shape per tier
            sl = {k: np.concatenate([v, np.zeros((batch - n,) + v.shape[1:], v.dtype)]) for k, v in sl.items()}
        d = find_dense(*[jnp.asarray(sl[k]) for k in ("valid", "idx", "side", "layer", "counter", "x", "y", "t",
                                                      "pe", "pos")], P=P, K=K)
        outs.append({k: np.asarray(v)[:n] for k, v in d.items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def find(ev, idx, bar, sipm, time, pe, seq, G, status, cfg=C.FinderConfig(), tiers=TIERS, fallback=True,
         timers=None, split=True):
    """Flat pulses of many events (sorted by event, collection order within) -> clusters.
    split: each (event, sector type) group is first cut into independent blocks at time and x gaps larger than any
    link (split_blocks); blocks are the kernel rows and their clusters are mapped back to the event.
    Each row runs in the smallest (H, K) tier it fits, promoted on overflow; rows beyond the last tier send their
    event to coinc.py ('fallback_events'). Returns cl_* per cluster and the member table mem_*."""
    import time as _time
    timers = timers if timers is not None else {}
    params = type_params(G, cfg)
    tables = hit_tables(G, status, cfg)
    rows, mems, redo = [], [], set()
    ch = bar * 4 + sipm
    for tt, P in sorted(params.items()):
        m = tables["stype"][ch] == tt
        if cfg.ignore_secondary_pulses:
            m &= seq == 0
        sel = np.flatnonzero(m)
        if split:
            t0 = _time.perf_counter()
            t_link, maxd0, maxd_f, smax = link_scales(G, P)
            bars = np.flatnonzero(tables["stype"][np.arange(0, 4 * len(G.sector), 4)] == tt)
            ys = G.y[bars]
            x_link = max(maxd0, maxd_f, smax * (ys.max() - ys.min()))
            tt_hit = time[sel] + tables["tcorr"][ch[sel]]
            blk = split_blocks(ev[sel], tt_hit, G.x[bar[sel]], t_link, x_link)
            o = np.lexsort((sel, blk))
            sel, blk = sel[o], blk[o]
            blk_ev = np.zeros(blk.max() + 1 if len(blk) else 0, ev.dtype)
            blk_ev[blk] = ev[sel]
            timers["split_s"] = timers.get("split_s", 0.0) + _time.perf_counter() - t0
            timers[f"blocks_t{tt}"] = int(len(blk_ev))
            g_ev, g_idx, g_bar, g_sipm, g_time, g_pe, g_seq = blk, idx[sel], bar[sel], sipm[sel], time[sel], pe[sel], \
                seq[sel]
        else:
            g_ev, g_idx, g_bar, g_sipm, g_time, g_pe, g_seq = ev[sel], idx[sel], bar[sel], sipm[sel], time[sel], \
                pe[sel], seq[sel]
            blk_ev = None
        to_ev = (lambda b: blk_ev[b]) if split else (lambda b: b)
        gev, cnt = np.unique(g_ev, return_counts=True)
        pending = np.zeros(len(gev), bool)
        hprev = 0
        for H, K, batch in tiers:
            todo = ((cnt > hprev) & (cnt <= H)) | (pending & (cnt <= H))
            pending &= ~todo
            hprev = H
            if not todo.any():
                continue
            t0 = _time.perf_counter()
            arr, tev, _ = pack(g_ev, g_idx, g_bar, g_sipm, g_time, g_pe, g_seq, G, tables, tt, H, cfg,
                               events=gev[todo])
            timers["pack_s"] = timers.get("pack_s", 0.0) + _time.perf_counter() - t0
            t0 = _time.perf_counter()
            d = _run_tier(arr, P, min(K, K_MAX_4LAYER) if P[6] == 4 else K, batch)
            timers["kernel_s"] = timers.get("kernel_s", 0.0) + _time.perf_counter() - t0
            timers[f"groups_H{H}"] = timers.get(f"groups_H{H}", 0) + len(tev)
            ov = d["overflow"]
            pending |= np.isin(gev, tev[ov])
            good = ~ov
            g, r = np.nonzero(d["root"] & good[:, None])
            cid0 = sum(len(x["cl_event"]) for x in rows)
            rows.append(dict(cl_event=to_ev(tev[g]), cl_type=np.full(len(g), tt), **{f"cl_{k}": d[k][g, r] for k in
                             ("pe", "start", "end", "slope", "avgtime", "poscalc")}))
            cmap = np.full(d["root"].shape, -1, np.int64)
            cmap[g, r] = cid0 + np.arange(len(g))
            hg, hh = np.nonzero((d["label"] >= 0) & good[:, None])
            cl = cmap[hg, d["label"][hg, hh]]
            mems.append(dict(mem_cl=cl[cl >= 0], mem_idx=arr["idx"][hg, hh][cl >= 0], mem_event=to_ev(tev[hg][cl >= 0])))
        redo.update(to_ev(gev[pending | (cnt > hprev)]).tolist())
    cat = lambda L, k: np.concatenate([x[k] for x in L]) if L else np.zeros(0)
    out = {k: cat(rows, k) for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_slope", "cl_avgtime",
                                      "cl_poscalc")}
    out.update({k: cat(mems, k) for k in ("mem_cl", "mem_idx", "mem_event")})
    redo = np.array(sorted(redo), np.int64)
    out["fallback_events"] = redo
    if fallback and len(redo):
        t0 = _time.perf_counter()
        _replace_with_numpy(out, redo, ev, idx, bar, sipm, time, pe, seq, G, status, cfg)
        timers["fallback_s"] = timers.get("fallback_s", 0.0) + _time.perf_counter() - t0
    return out


def _replace_with_numpy(out, redo, ev, idx, bar, sipm, time, pe, seq, G, status, cfg):
    """Drop the JAX clusters of the redo events and append coinc.find_event results for them."""
    keep_cl = ~np.isin(out["cl_event"], redo)
    newid = np.cumsum(keep_cl) - 1
    km = keep_cl[out["mem_cl"].astype(np.int64)] if len(out["mem_cl"]) else np.zeros(0, bool)
    for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_slope", "cl_avgtime", "cl_poscalc"):
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
