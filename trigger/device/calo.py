"""Calorimeter trigger stage on the device: ports of the stock CaloHitMakerFast and CaloClusterFast (Offline v13_41_01)
and of the calo trigger filters: CaloClusterCounterFilter, CaloLikelihood, CaloNNEval + FilterEcalNNTrigger,
CaloCosmicCalib, and DigiFilter on the calo digi count.

The port follows the stock arithmetic, as the mpr port does: float32 values with the C++ double steps (traced under
`exact.strict`, so float64 in both precisions of the chain), products guarded against fused multiply-add, and the
stock orders where they decide a result:

- digis in (SiPM id, t0) order; hits in crystal order and, within a crystal, in pulse creation order;
- the cluster seeds in (time, crystal) order (stock's comparator with its 1 ps tie tolerance), each window found with
  stock's iterator scan (including its quirks: the window never reaches back to the first hit, and a hit whose index
  equals the number of selected hits counts as used, stock's sentinel value), the cluster grown breadth first in stock's neighbour order (first ring, then second), so
  the members come in stock's discovery order, then libstdc++'s std::sort by energy (`mpr._stock_sort`);
- every sum in the stock order.

Where an input would need an order this port does not reproduce, the event is counted: `tie` (two digis with the same
(SiPM, t0), or two seed-candidate hits within 1 ps but not equal in time), `fallback` (an energy sort that reaches
introsort's heap fallback), `overflow` (more digis, hits or cluster members than the slots).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .exact import m as _m, strict
from .mpr import _argmax, _d, _f, _stock_sort

F32, F64 = jnp.float32, jnp.float64
N_CRYSTALS = 1348
PERIOD = 5.0                          # CaloConst::_digitizationPeriod (ns)


# --- inputs ---------------------------------------------------------------------------------------------------------

def pack_digis(digis, n_events, slots=None):
    """Flat stock CaloDigis (`caloart.decode_digis`) -> padded per-event arrays [E, D] with the samples the hit
    maker reads (the first four, the peak sample, the length). `overflow` [E] counts events with more than `slots`."""
    n = np.asarray(digis["n"], np.int64)
    if n.size != n_events:
        raise ValueError("digi counts do not match the event count")
    D = int(max(n.max(initial=0), 1)) if slots is None else int(slots)
    ev = np.asarray(digis["entry"], np.int64)
    slot = np.arange(ev.size) - np.repeat(np.cumsum(n) - n, n)
    keep = slot < D
    out = {"valid": np.zeros((n_events, D), bool)}
    out["valid"][ev[keep], slot[keep]] = True
    s = np.asarray(digis["samples"], np.int64)
    start, ln, pk = (np.asarray(digis[k], np.int64) for k in ("wf_start", "wf_len", "peakpos"))
    cols = {"sipm": digis["sipm"], "t0": digis["t0"], "peakpos": pk, "wf_len": ln}
    for i in range(4):
        cols[f"s{i}"] = np.where(i < ln, s[np.minimum(start + i, max(s.size - 1, 0))], 0)
    cols["s_peak"] = np.where((pk >= 0) & (pk < ln), s[np.clip(start + pk, 0, max(s.size - 1, 0))], 0)
    for k, v in cols.items():
        a = np.zeros((n_events, D), np.int32)
        a[ev[keep], slot[keep]] = np.asarray(v, np.int64)[keep]
        out[k] = a
    out["n"] = n.astype(np.int32)
    out["overflow"] = n > D
    return out


def settings(cfg_hits, cfg_clusters, calib, geo):
    """Static settings from the menu (CaloHitMakerFast, CaloClusterFast), CalCalib and the geometry."""
    adc = np.where(geo["caphri"], np.float32(calib["ADC2MeVlyso"]), np.float32(calib["ADC2MeVCsI"])).astype(np.float32)
    nbr = np.concatenate([geo["neighbors"], geo["next_neighbors"]], axis=1) if cfg_clusters["extendSearch"] \
        else np.asarray(geo["neighbors"])
    return {"deltaT": np.float32(cfg_hits["deltaTPulses"]), "npe": np.float32(cfg_hits["nPEperMeV"]),
            "noise2": np.float32(np.float32(cfg_hits["noiseLevelMeV"]) * np.float32(cfg_hits["noiseLevelMeV"])),
            "nsigma": np.float32(cfg_hits["nSigmaNoise"]), "minE": np.float32(cfg_hits.get("minEDep", 0.0)),
            "minWfm": int(cfg_hits.get("minWfmLength", 5)), "timeoffset": np.float32(calib.get("timeoffset", 0.0)),
            "adc2mev": adc, "caphri": np.asarray(geo["caphri"]),
            "eminSeed": float(cfg_clusters["EminSeed"]), "enoise": float(cfg_clusters["EnoiseCut"]),
            "expand": float(cfg_clusters["ExpandCut"]), "dt": float(cfg_clusters["deltaTime"]),
            "toff": float(cfg_clusters["timeOffset"]), "minSiPM": int(cfg_clusters["minSiPMPerHit"]),
            "nbr": nbr.astype(np.int32), "disk": np.asarray(geo["disk"], np.int32),
            "lx": np.asarray(geo["local_x"], np.float64), "ly": np.asarray(geo["local_y"], np.float64),
            "pos": np.asarray(geo["position"], np.float64), "ff": np.asarray(geo["ff"], np.float64),
            "crystal_size": np.float32(geo["crystal_size"]), "origin": np.asarray(geo["origin"], np.float64),
            "next": np.asarray(geo["next_neighbors"], np.int32), "first": np.asarray(geo["neighbors"], np.int32)}


# --- CaloHitMakerFast -----------------------------------------------------------------------------------------------

def _hits_event(dg, pbt, s, H):
    D = dg["sipm"].shape[0]
    crystal = dg["sipm"] // 2
    ok = dg["valid"] & (crystal < N_CRYSTALS) & (dg["wf_len"] >= s["minWfm"])
    key = jnp.where(dg["valid"], dg["sipm"].astype(jnp.int64) * (1 << 32) + (dg["t0"].astype(jnp.int64) + (1 << 31)),
                    jnp.iinfo(jnp.int64).max)
    order = jnp.argsort(key, stable=True)
    ks = key[order]
    tie = jnp.any((ks[1:] == ks[:-1]) & dg["valid"][order][1:])
    g = {k: v[order] for k, v in dg.items()}
    ok = ok[order]
    crystal = crystal[order]
    pk = g["peakpos"]
    nped = jnp.where(pk > 3, 4, jnp.maximum(pk - 1, 1))
    base = _d(0.0)
    for i in range(4):
        base = base + jnp.where(i < nped, _d(g[f"s{i}"]), 0.0)
    base = base / _d(nped)
    a2m = _d(jnp.asarray(s["adc2mev"])[jnp.clip(crystal, 0, N_CRYSTALS - 1)])
    edep = _m(_d(g["s_peak"]) - base, a2m)
    time = _m(_d(g["t0"] + pk), PERIOD) - _d(pbt) - _d(s["timeoffset"])
    ok = ok & ~(edep < _d(s["minE"]))
    t_in, e_in = _f(time), _f(edep)
    P = D
    st = (jnp.full(P, -1, jnp.int32), jnp.zeros(P, jnp.int32), jnp.zeros(P, F32), jnp.zeros(P, F32), jnp.int32(0))
    npe, noise2, nsig, dtp = (jnp.asarray(s[k]) for k in ("npe", "noise2", "nsigma", "deltaT"))

    def body(k, st):
        pc, pn, pt, pe, cnt = st
        c, t, e = crystal[k], t_in[k], e_in[k]
        live = jnp.arange(P) < cnt
        intime = live & (pc == c) & ~(jnp.abs(pt - t) > dtp)
        ratio = (e - pe) / (e + pe)
        emean = _f(_d(e + pe) / 2.0)
        sig = _f(_m(0.707, jnp.sqrt(1.0 / _d(emean) / _d(npe) + _d(noise2 / emean / emean))))
        compat = intime & (jnp.abs(ratio) < _m(nsig, sig))
        j = _argmax(compat)
        hit = ok[k] & jnp.any(compat)
        nj = _f(pn[j])
        tj = (_m(nj, pt[j]) + t) / _f(pn[j] + 1)
        ej = (_m(nj, pe[j]) + e) / _f(pn[j] + 1)
        new = ok[k] & ~jnp.any(compat)
        at = jnp.where(hit, j, jnp.where(new, cnt, P))
        pc = pc.at[at].set(c, mode="drop")
        pn = pn.at[at].set(jnp.where(hit, pn[j] + 1, 1), mode="drop")
        pt = pt.at[at].set(jnp.where(hit, tj, t), mode="drop")
        pe = pe.at[at].set(jnp.where(hit, ej, e), mode="drop")
        return pc, pn, pt, pe, cnt + new.astype(jnp.int32)

    pc, pn, pt, pe, cnt = jax.lax.fori_loop(0, D, body, st)
    live = jnp.arange(P) < cnt
    calo = live & ~jnp.asarray(s["caphri"])[jnp.clip(pc, 0, N_CRYSTALS - 1)]
    # CaloHitCollection "calo": the non-CAPHRI pulses in order, gathered into H slots
    take = jnp.argsort(~calo, stable=True)[:H] if P >= H else jnp.pad(jnp.argsort(~calo, stable=True), (0, H - P))
    valid = calo[take] & (jnp.arange(H) < jnp.sum(calo))
    return {"crystal": jnp.where(valid, pc[take], -1), "nsipm": jnp.where(valid, pn[take], 0),
            "time": jnp.where(valid, pt[take], 0), "edep": jnp.where(valid, pe[take], 0), "valid": valid,
            "n": jnp.sum(calo), "overflow": jnp.sum(calo) > H, "tie": tie}


# --- CaloClusterFast ------------------------------------------------------------------------------------------------

def _sort_lt(lt, sel):
    """libstdc++ std::sort (GCC 13 / 14) of the selected slots, taken in slot order, with the comparator given as a
    matrix lt[i, j] = comp(slot i, slot j): introsort (median of three to the first element, unguarded Hoare
    partition, recursion above 16 elements) and the final insertion sort, done literally (the comparator need not be a
    strict weak order). Returns (perm, n, fallback): perm [L] lists the selected slots in that order, then the others;
    fallback where introsort would switch to heap sort (not reproduced)."""
    L = sel.shape[0]
    n = jnp.sum(sel).astype(jnp.int32)
    a = jnp.argsort(~sel, stable=True).astype(jnp.int32)
    S = 2 * max(L, 2).bit_length() + 4
    depth0 = jnp.where(n > 0, 2 * (31 - jax.lax.clz(jnp.maximum(n, 1))), 0)
    pos = jnp.arange(L, dtype=jnp.int32)
    T = 16                                                   # libstdc++ _S_threshold

    def c(a, i, j):
        return lt[a[i], a[j]]

    def swap(a, i, j):
        ai, aj = a[i], a[j]
        return a.at[i].set(aj).at[j].set(ai)

    def partition(a, f, l, go):
        mid = f + (l - f) // 2
        x, y, z = f + 1, mid, l - 1
        sw = jnp.where(c(a, x, y), jnp.where(c(a, y, z), y, jnp.where(c(a, x, z), z, x)),
                       jnp.where(c(a, x, z), x, jnp.where(c(a, y, z), z, y)))
        a = swap(a, f, sw)
        pv = a[f]

        def cond(st):
            return go & ~st[3]

        def body(st):
            a, first, last, _ = st
            cand = (pos >= first) & ~lt[a, pv]
            first = jnp.where(jnp.any(cand), _argmax(cand), L).astype(jnp.int32)
            last = (L - 1 - _argmax(((pos <= last - 1) & ~lt[pv, a])[::-1])).astype(jnp.int32)
            done = ~(first < last)
            a = jnp.where(done, a, swap(a, first, last))
            return a, jnp.where(done, first, first + 1), last, done

        a, cut, _, _ = jax.lax.while_loop(cond, body, (a, f + 1, l, jnp.bool_(False)))
        return a, cut

    def icond(st):
        return st[4] > 0

    def ibody(st):
        a, sf, sl, sd, top, fb = st
        top = top - 1
        f, l, d = sf[top], sl[top], sd[top]
        big = (l - f) > T
        fb = fb | (big & (d == 0))
        go = big & (d > 0)
        a2, cut = partition(a, f, l, go)
        a = jnp.where(go, a2, a)
        sf = jnp.where(go, sf.at[top].set(cut).at[top + 1].set(f, mode="drop"), sf)
        sl = jnp.where(go, sl.at[top].set(l).at[top + 1].set(cut, mode="drop"), sl)
        sd = jnp.where(go, sd.at[top].set(d - 1).at[top + 1].set(d - 1, mode="drop"), sd)
        return a, sf, sl, sd, jnp.where(go, top + 2, top), fb

    z = jnp.zeros(S, jnp.int32)
    a, _, _, _, _, fb = jax.lax.while_loop(icond, ibody, (a, z, z.at[0].set(n), z.at[0].set(depth0), jnp.int32(1),
                                                         jnp.bool_(False)))

    def insert(i, a):
        """__insertion_sort on [0, min(n, 16)), then __unguarded_linear_insert for the rest."""
        live = i < n
        v = a[i]
        guarded = (i < T) | (n <= T)
        to_front = guarded & lt[v, a[0]]
        b = ((pos < i) & lt[v, a]) | (pos >= i)
        k = jnp.sum(jnp.cumprod(b[::-1].astype(jnp.int32))) - (L - i)   # run of comp(v, a[j]) for j = i-1, i-2, ...
        at = jnp.where(to_front, 0, i - k)
        shifted = jnp.where(pos < at, a, jnp.where(pos == at, v, jnp.where(pos <= i, a[jnp.maximum(pos - 1, 0)], a)))
        return jnp.where(live, shifted, a)

    a = jax.lax.fori_loop(1, L, insert, a)
    return a, n, fb


def _clusters_event(h, s, C):
    H = h["crystal"].shape[0]
    nbr = jnp.asarray(s["nbr"])
    sel = h["valid"] & (h["crystal"] < N_CRYSTALS) & (_d(h["edep"]) > s["enoise"]) & (h["nsipm"] >= s["minSiPM"])
    # stock's comparator: (dt = t_a - t_b in float) |dt| < 1e-3 -> crystal id order, else time order. It is not a
    # strict weak order (times a few ulps apart tie), so libstdc++'s std::sort is emulated on it, not on a key.
    dtm = h["time"][:, None] - h["time"][None, :]
    lt = jnp.where(_d(jnp.abs(dtm)) < 1e-3, h["crystal"][:, None] < h["crystal"][None, :], dtm < 0)
    o, _, fb_t = _sort_lt(lt, sel)
    ns = jnp.sum(sel).astype(jnp.int32)
    cry, t, e = h["crystal"][o], h["time"][o], h["edep"][o]
    pos = jnp.arange(H)
    inn = pos < ns
    tie = fb_t

    def seed_body(sp, st):
        used, out, nc, fb, ovf = st
        go = (sp < ns) & ~used[sp] & ~(_d(e[sp]) < s["eminSeed"])
        t0 = _d(t[sp])
        stop_cond = used | (_d(t) - t0 < s["dt"])
        after = (pos >= sp) & inn & ~stop_cond
        stop = jnp.where(jnp.any(after), _argmax(after), ns)
        before = (pos >= 1) & (pos <= sp) & ~(used | (t0 - _d(t) < s["dt"]))
        start = jnp.where(jnp.any(before), H - 1 - _argmax(before[::-1]), 0) + 1
        start = jnp.minimum(start, sp + 1)
        win = (pos >= start) & (pos < stop) & inn
        seedc = cry[sp]
        used2 = used.at[sp].set(used[sp] | go)
        mem = jnp.full(H, -1, jnp.int32).at[0].set(sp)
        queue = jnp.full(H + 1, -1, jnp.int32).at[0].set(seedc)
        visited = jnp.zeros(N_CRYSTALS, bool)

        def bfs_cond(b):
            return b[5] & (b[3] < b[4])

        def bfs_body(b):
            used, mem, nm, q, qlen, live, queue, visited = b
            v = queue[q]
            visited = visited.at[v].set(True)
            def nbr_step(k, c):
                used, mem, nm, qlen, queue, visited = c
                nid = nbr[v, k]
                fresh = (nid >= 0) & ~visited[jnp.maximum(nid, 0)]
                visited = visited.at[jnp.maximum(nid, 0)].set(visited[jnp.maximum(nid, 0)] | fresh)
                join = fresh & win & ~used & (cry == nid)
                slot = nm + jnp.cumsum(join) - 1
                mem = mem.at[jnp.where(join, slot, H)].set(pos.astype(jnp.int32), mode="drop")
                nm = nm + jnp.sum(join).astype(jnp.int32)
                used = used | join
                push = jnp.any(join & (_d(e) > s["expand"]))
                queue = queue.at[jnp.where(push, qlen, H + 1)].set(nid, mode="drop")
                return used, mem, nm, qlen + push.astype(jnp.int32), queue, visited

            used, mem, nm, qlen, queue, visited = jax.lax.fori_loop(
                0, nbr.shape[1], nbr_step, (used, mem, nm, qlen, queue, visited))
            return used, mem, nm, q + 1, qlen, live, queue, visited

        b = (used2, mem, jnp.int32(1), jnp.int32(0), jnp.int32(1), go, queue, visited)
        used3, mem, nm, _, _, _, _, _ = jax.lax.while_loop(bfs_cond, bfs_body, b)
        used = jnp.where(go, used3, used)
        # std::sort of the members by energy (descending), from the discovery order
        msel = jnp.arange(H) < nm
        ekey = jnp.where(msel, -e[jnp.maximum(mem, 0)], 0)
        perm, _, fb1 = _stock_sort(ekey, msel)
        mem = jnp.where(msel, mem[perm], -1)

        def acc(i, a):
            tot, x, y = a
            m = mem[i]
            ok = i < nm
            em = _d(e[jnp.maximum(m, 0)])
            c = cry[jnp.maximum(m, 0)]
            tot = jnp.where(ok, tot + em, tot)
            x = jnp.where(ok, x + _m(jnp.asarray(s["lx"])[c], em), x)
            y = jnp.where(ok, y + _m(jnp.asarray(s["ly"])[c], em), y)
            return tot, x, y

        tot, x, y = jax.lax.fori_loop(0, H, acc, (_d(0.0), _d(0.0), _d(0.0)))
        seed_hit = mem[0]
        row = {"disk": jnp.asarray(s["disk"])[cry[seed_hit]], "time": _f(_d(t[seed_hit]) + s["toff"]),
               "energy": _f(tot), "cog_x": x / tot, "cog_y": y / tot, "size": nm,
               "members": o[jnp.maximum(mem[:H], 0)].astype(jnp.int32) * (mem[:H] >= 0) - (mem[:H] < 0)}
        at = jnp.where(go, nc, C)
        out = jax.tree.map(lambda a, v: a.at[at].set(v, mode="drop"), out, row)
        return used, out, nc + go.astype(jnp.int32), fb | (go & fb1), ovf | (go & (nc >= C))

    out0 = {"disk": jnp.zeros(C, jnp.int32), "time": jnp.zeros(C, F32), "energy": jnp.zeros(C, F32),
            "cog_x": jnp.zeros(C, F64), "cog_y": jnp.zeros(C, F64), "size": jnp.zeros(C, jnp.int32),
            "members": jnp.full((C, H), -1, jnp.int32)}
    # stock marks a used hit by overwriting its index with the number of selected hits: a selected hit whose hit index
    # equals that number counts as used from the start
    st = (inn & (o == ns), out0, jnp.int32(0), jnp.bool_(False), jnp.bool_(False))
    used, out, nc, fb, ovf = jax.lax.fori_loop(0, H, seed_body, st)
    out["valid"] = jnp.arange(C) < nc
    out.update(n=nc, fallback=fb, overflow=ovf, tie=tie)
    return out


def reconstruct(dg, pbt, s, H=None, C=None):
    """Packed digis [E, D] + ProtonBunchTime [E] -> calo hits [E, H] and clusters [E, C] (stock order)."""
    D = dg["sipm"].shape[1]
    H = D if H is None else H
    C = H if C is None else C
    keys = ("valid", "sipm", "t0", "peakpos", "wf_len", "s0", "s1", "s2", "s3", "s_peak")
    with strict(dg["sipm"]):
        d = {k: jnp.asarray(dg[k]) for k in keys}
        hits = jax.vmap(lambda g, p: _hits_event(g, p, s, H))(d, jnp.asarray(pbt, F32))
        cl = jax.vmap(lambda h: _clusters_event(h, s, C))(hits)
    return hits, cl


# --- filters --------------------------------------------------------------------------------------------------------

def counter_pass(cl, cfg):
    """CaloClusterCounterFilter."""
    r = _f(jnp.sqrt(_m(cl["cog_x"], cl["cog_x"]) + _m(cl["cog_y"], cl["cog_y"])))
    disk_ok = jnp.zeros(cl["disk"].shape, bool)
    for dsk in cfg["DiskID"]:
        disk_ok = disk_ok | (cl["disk"] == int(dsk))
    e = _d(cl["energy"])
    ok = (cl["valid"] & disk_ok & (e >= float(cfg["MinClusterEnergy"])) & (e <= float(cfg["MaxClusterEnergy"]))
          & (cl["size"] >= int(cfg["MinNCel"])) & (cl["size"] <= int(cfg["MaxNCel"]))
          & (_d(r) >= float(cfg["MinClusterRadius"])))
    return jnp.sum(ok, axis=-1) >= int(cfg["MinNCl"])


def digi_count_pass(n_digis, cfg):
    """DigiFilter on the calo digi count (useCaloDigi only, no energy cut)."""
    if cfg.get("useStrawDigi", False) or not cfg.get("useCaloDigi", False) or float(cfg.get("maxCaloEnergy", -1)) >= 0:
        raise ValueError("device calo DigiFilter supports useCaloDigi true, useStrawDigi false, no energy cut")
    return (n_digis >= int(cfg["minNCaloDigi"])) & (n_digis <= int(cfg["maxNCaloDigi"]))


def _member_view(cl, hits, s):
    """Per cluster member [E, C, H]: crystal, energy (float), valid, in the stored (energy-sorted) order."""
    m = cl["members"]
    ok = m >= 0
    idx = jnp.maximum(m, 0)
    take = lambda a: jnp.take_along_axis(a[:, None, :], idx, axis=2)
    return take(hits["crystal"]), take(hits["edep"]), ok & cl["valid"][..., None]


def _seq(vals, use, init, op=lambda a, b: a + b):
    """Sequential masked reduction over the last axis, in index order (the stock loop order)."""
    init = jnp.broadcast_to(jnp.asarray(init, vals.dtype), vals.shape[:-1])
    return jax.lax.fori_loop(0, vals.shape[-1],
                             lambda i, acc: jnp.where(use[..., i], op(acc, vals[..., i]), acc), init)


def _ff_sums(cl, hits, s):
    """ClusterUtils::fill (Linear): the energy-weighted Mu2e-frame sums over the members on the first hit's disk."""
    cr, en, ok = _member_view(cl, hits, s)
    pos, disk = jnp.asarray(s["pos"]), jnp.asarray(s["disk"])
    d0 = disk[jnp.maximum(cr[..., 0], 0)]
    use = ok & (disk[jnp.maximum(cr, 0)] == d0[..., None])
    w = _d(en)
    p = pos[jnp.maximum(cr, 0)]
    x, y = p[..., 0], p[..., 1]
    z = _d(0.0)
    sums = {"sw": _seq(w, use, z), "sx": _seq(_m(x, w), use, z), "sy": _seq(_m(y, w), use, z),
            "sx2": _seq(_m(x, x, w), use, z), "sy2": _seq(_m(y, y, w), use, z)}
    return sums, d0, (cr, en, ok)


def load_templates(signal_path, bkg_path, min_rdist, rdist_step, n_cor=10):
    """CaloLikelihood's templates as arrays: the rDist TH1F (with flow bins) and, for the seven 2D variables, the
    TH2F's y projections over each radial slice (ProjectionY over the stock's 0-based x-bin range, area over bins
    1..N, scaled by 1 / area) -> dict of numpy arrays indexed [kind (0 sig, 1 bkg), disk, ...]."""
    import uproot
    names2 = ("eRDist", "e1RDist", "e2RDist", "e2e1RDist", "e3RDist", "e4RDist", "nCrRDist")
    r1, corr, c_low, c_w, c_nb = [], [], [], [], []
    for path, sub, suffix in ((signal_path, "cluster_electron", "2"), (bkg_path, "cluster_all", "0")):
        with uproot.open(path) as fh:
            rk, ck = [], []
            for disk in (0, 1):
                h = fh[f"ceAna/{sub}/clDisk{disk}_rDist{suffix}"]
                e = h.axis().edges()
                rk.append((h.values(flow=True).astype(np.float64), e[0], e[1] - e[0], len(e) - 1))
                per = []
                for nm in names2:
                    h2 = fh[f"ceAna/{sub}/clDisk{disk}_{nm}{suffix}"]
                    v = h2.values(flow=True).astype(np.float64)
                    ex, ey = h2.axis(0).edges(), h2.axis(1).edges()
                    binsize, binlow, nx = ex[1] - ex[0], ex[0], len(ex) - 1
                    sl = []
                    for i in range(n_cor):
                        start = min_rdist + i * rdist_step
                        b0 = int((start - binlow) / binsize)
                        b1 = int(b0 + rdist_step / binsize)
                        b0 = max(b0, 0)
                        b1 = nx + 1 if (b1 < 0 or b1 > nx + 1) else b1
                        proj = np.zeros(v.shape[1])
                        for bx in range(b0, b1 + 1):
                            proj = proj + v[bx]
                        area = 0.0
                        for by in range(1, len(ey)):
                            area += proj[by]
                        sl.append(proj * (1 / area))
                    per.append((np.stack(sl), ey[0], ey[1] - ey[0], len(ey) - 1))
                ck.append(per)
            r1.append(rk)
            corr.append(ck)
    nb1 = max(x[3] for k in r1 for x in k) + 2
    nb2 = max(x[3] for k in corr for d in k for x in d) + 2
    out = {"r_vals": np.zeros((2, 2, nb1)), "r_low": np.zeros((2, 2)), "r_w": np.ones((2, 2)),
           "r_nb": np.zeros((2, 2), np.int32), "c_vals": np.zeros((2, 2, 7, n_cor, nb2)),
           "c_low": np.zeros((2, 2, 7)), "c_w": np.ones((2, 2, 7)), "c_nb": np.zeros((2, 2, 7), np.int32)}
    for kd in range(2):
        for dk in range(2):
            v, lo, w, nb = r1[kd][dk]
            out["r_vals"][kd, dk, :v.size] = v
            out["r_low"][kd, dk], out["r_w"][kd, dk], out["r_nb"][kd, dk] = lo, w, nb
            for k in range(7):
                v, lo, w, nb = corr[kd][dk][k]
                out["c_vals"][kd, dk, k, :, :v.shape[1]] = v
                out["c_low"][kd, dk, k], out["c_w"][kd, dk, k], out["c_nb"][kd, dk, k] = lo, w, nb
    return out


def _prob(var, vals, low, width, nb):
    """CaloLikelihood::calculateProb: bin int((var - low edge) / width) (0-based, used as ROOT's bin number)."""
    q = (var - low) / width
    idx = jnp.clip(q, -2.0e9, 2.0e9).astype(jnp.int32)
    p = jnp.take_along_axis(vals, jnp.clip(idx, 0, nb + 1)[..., None].astype(jnp.int32), axis=-1)[..., 0]
    p = jnp.where(idx > nb, 1e-17, p)
    return jnp.where(p < 1e-10, 1e-17, p)


def likelihood_pass(cl, hits, s, tpl, cfg):
    """CaloLikelihood (TriggerInfo not kept): any cluster with lhValue above the disk's cut."""
    min_e, e_step = float(cfg["MinClusterEnergy"]), float(cfg["ClusterEnergyStep"])
    min_r, r_step = float(cfg["MinClusterRadialDist"]), float(cfg["ClusterRadialDistStep"])
    cuts = jnp.asarray([float(c) for c in cfg["MinLikelihoodCut"]], F64)
    sums, _, (cr, en, ok) = _ff_sums(cl, hits, s)
    disk = cl["disk"]
    orig = jnp.asarray(s["origin"])[jnp.maximum(disk, 0)]
    x = sums["sx"] / sums["sw"] - orig[..., 0]
    y = sums["sy"] / sums["sw"] - orig[..., 1]
    r = jnp.sqrt(_m(x, x) + _m(y, y))
    E = _d(cl["energy"])
    ev = jnp.where(ok, _d(en), -jnp.inf)
    H = ev.shape[-1]
    j = jnp.arange(H)
    i1 = _argmax(jnp.where(ok, ev, -jnp.inf) == jnp.max(jnp.where(ok, ev, 0.0), axis=-1, keepdims=True), axis=-1)
    m1 = jnp.maximum(jnp.max(jnp.where(ok, ev, 0.0), axis=-1), 0.0)

    def nth(excl):
        cand = ok & ~excl
        top = jnp.max(jnp.where(cand, ev, -jnp.inf), axis=-1)
        has = jnp.any(cand & (ev > -1.0), axis=-1)
        idx = _argmax(cand & (ev == top[..., None]), axis=-1)
        return jnp.where(has, top, -1.0), jnp.where(has, idx, -1)

    m2, i2 = nth(j == i1[..., None])
    m3, i3 = nth((j == i1[..., None]) | (j == i2[..., None]))
    m4, _ = nth((j == i1[..., None]) | (j == i2[..., None]) | (j == i3[..., None]))
    n = cl["size"]
    t = {k: jnp.asarray(v) for k, v in tpl.items()}
    dk = jnp.maximum(disk, 0)

    def p1(kind):
        return _prob(r, t["r_vals"][kind][dk], t["r_low"][kind][dk], t["r_w"][kind][dk], t["r_nb"][kind][dk])

    def p2(kind, k, var, mn, st):
        out = jnp.full(r.shape, 1e-17)
        found = jnp.zeros(r.shape, bool)
        for jj in range(t["c_vals"].shape[3]):
            inside = ~found & (r >= mn + float(jj) * st) & (r < mn + float(jj + 1) * st)
            p = _prob(var, t["c_vals"][kind, :, k, jj][dk], t["c_low"][kind, :, k][dk], t["c_w"][kind, :, k][dk],
                      t["c_nb"][kind, :, k][dk])
            out = jnp.where(inside, p, out)
            found = found | inside
        return out

    lr = lambda a, b: jnp.log(a / b)
    l_r = lr(p1(0), p1(1))
    two = lambda k, var, mn, st: lr(p2(0, k, var, mn, st), p2(1, k, var, mn, st))
    l_e = two(0, r, min_e, e_step)
    l_e1, l_e2, l_e21 = two(1, m1 / E, min_r, r_step), two(2, m2 / E, min_r, r_step), two(3, m2 / m1, min_r, r_step)
    l_e3, l_e4, l_n = two(4, m3 / E, min_r, r_step), two(5, m4 / E, min_r, r_step), two(6, _d(n), min_r, r_step)
    lh = l_r + l_e + l_e1 + l_r + l_n
    lh = jnp.where(n >= 2, lh + (l_e2 + l_e21), lh)
    lh = jnp.where(n >= 3, lh + l_e3, lh)
    lh = jnp.where(n >= 4, lh + l_e4, lh)
    sel = cl["valid"] & ~(E < min_e)
    if cfg.get("DropSecondDisk", False):
        sel = sel & (disk != 1)
    return jnp.any(sel & (lh > cuts[dk]), axis=-1)


def _mlp(net, v):
    """MVATools::evalMVA (float, sequential sums in the stock order) on v [..., nvar] -> score (float)."""
    x = [_m(_f(v[..., i]) - net["offset"][i], net["scale"][i]) - F32(1.0) if net["norm"] else _f(v[..., i])
         for i in range(v.shape[-1])] + [jnp.ones(v.shape[:-1], F32)]
    for k, w in enumerate(net["layers"]):
        y = []
        for jj in range(w.shape[1]):
            acc = jnp.zeros(v.shape[:-1], F32)
            for i in range(w.shape[0]):
                acc = acc + _m(F32(w[i, jj]), x[i])
            y.append(acc)
        if k < len(net["layers"]) - 1:
            if net["act"] == "relu":
                y = [jnp.maximum(F32(0.0), a) for a in y]
            elif net["act"] == "sigmoid":
                y = [_f(1.0 / (1.0 + _d(_f(jnp.exp(-_d(a)))))) for a in y]
            else:
                raise ValueError("device calo MLP supports relu and sigmoid layers")
            x = y + [jnp.ones(v.shape[:-1], F32)]
        else:
            return _f(1.0 / (1.0 + _d(_f(jnp.exp(-_d(y[0]))))))


def nn_pass(cl, hits, s, net, cfg_eval, cfg_filter):
    """CaloNNEval + FilterEcalNNTrigger."""
    def window(c):
        e, t = cl["energy"], cl["time"]
        r = _f(jnp.sqrt(_m(cl["cog_x"], cl["cog_x"]) + _m(cl["cog_y"], cl["cog_y"])))
        return (~(e < F32(c["minEtoTest"])) & ~(e > F32(c["maxEtoTest"])) & ~(r < F32(c["minRtoTest"]))
                & ~(r > F32(c["maxRtoTest"])) & ~(t < F32(c["minTtoTest"])) & ~(t > F32(c["maxTtoTest"]))), r
    test, r = window(cfg_eval)
    sums, _, (cr, en, ok) = _ff_sums(cl, hits, s)
    E = _d(cl["energy"])
    ed = _d(en)
    e1 = ed[..., 0]
    e2 = jnp.where(cl["size"] > 1, ed[..., 0] + ed[..., 1], ed[..., 0])
    c0 = jnp.maximum(cr[..., 0], 0)
    first, nxt = jnp.asarray(s["first"])[c0], jnp.asarray(s["next"])[c0]
    in1 = jnp.any((cr[..., :, None] == first[..., None, :]) & (first[..., None, :] >= 0), axis=-1) & ok
    in2 = jnp.any((cr[..., :, None] == nxt[..., None, :]) & (nxt[..., None, :] >= 0), axis=-1) & ok
    e9 = _seq(ed, in1, e1)
    e25 = _seq(ed, in2, e9)
    sw, sx, sy, sx2, sy2 = (sums[k] for k in ("sw", "sx", "sy", "sx2", "sy2"))
    m2 = (sx2 - _m(sx, sx) / sw + sy2 - _m(sy, sy) / sw) / sw
    v = jnp.stack([r, _f(cl["size"]), _f(e1 / E), _f(e2 / E), _f(e9 / E), _f(e25 / E), _f(m2)], axis=-1)
    score = jnp.where(test, _mlp(net, v), F32(-1.0))
    sel, _ = window(cfg_filter)
    return jnp.any(cl["valid"] & sel & ~(score < F32(cfg_filter["minMVAScore"])), axis=-1)


def cosmic_pass(cl, hits, s, cfg):
    """CaloCosmicCalib (its `abs` calls resolve to the float overload, as the stock replays on cosmic digis show)."""
    cr, en, ok = _member_view(cl, hits, s)
    mincel = F32(cfg["MinCelEneCut"])
    outr, innr = F32(cfg["OutRadius"]), F32(cfg["InnerRadius"])
    ff = jnp.asarray(s["ff"])[jnp.maximum(cr, 0)]
    x, y = _f(ff[..., 0]), _f(ff[..., 1])
    rr = _f(jnp.sqrt(_m(_d(x), _d(x)) + _m(_d(y), _d(y))))
    mip = ok & (en > mincel)
    nmip = jnp.sum(mip, axis=-1)
    eo = _seq(en, mip & (rr > outr), jnp.zeros(cl["energy"].shape, F32))
    ei = _seq(en, mip & (rr < innr), jnp.zeros(cl["energy"].shape, F32))
    ymax_v = jnp.maximum(jnp.max(jnp.where(mip, y, -jnp.inf), axis=-1), F32(-999))
    ymin_v = jnp.minimum(jnp.min(jnp.where(mip, y, jnp.inf), axis=-1), F32(999))
    imax = _argmax(mip & (y == ymax_v[..., None]) & (y > F32(-999)), axis=-1)
    imin = _argmax(mip & (y == ymin_v[..., None]) & (y < F32(999)), axis=-1)
    hmax = jnp.any(mip & (y > F32(-999)), axis=-1)
    hmin = jnp.any(mip & (y < F32(999)), axis=-1)
    pick = lambda a, i: jnp.take_along_axis(a, i[..., None], axis=-1)[..., 0]
    xmax = jnp.where(hmax, pick(x, imax), F32(-999)); rmax = jnp.where(hmax, pick(rr, imax), F32(-999))
    xmin = jnp.where(hmin, pick(x, imin), F32(999)); rmin = jnp.where(hmin, pick(rr, imin), F32(-999))
    flag = jnp.where((eo > mincel) & (ei > mincel) & (nmip > int(cfg["MinNumCelinout"])), 1, 0)
    dx, dy = xmax - xmin, ymax_v - ymin_v
    dx, dy = jnp.abs(dx), jnp.abs(dy)
    diag = jnp.where(dx > 0, dy / jnp.where(dx > 0, dx, F32(1)), F32(0))
    dg = nmip > int(cfg["MinNumCeldiagver"])
    flag = flag + jnp.where(dg & (dx < s["crystal_size"]), 100, 0)
    flag = flag + jnp.where(dg & (diag > F32(cfg["diagminval"])) & (diag < F32(cfg["diagmaxval"])), 200, 0)
    flag = flag + jnp.where(dg & (dy > F32(cfg["dyminval"])) & (rmax > outr) & (rmin > outr), 10, 0)
    E = _d(cl["energy"])
    sel = (cl["valid"] & (E >= float(cfg["MinEnergy"])) & (E <= float(cfg["MaxEnergy"]))
           & (cl["size"] >= int(cfg["MinNCrystalHits"])))
    return jnp.any(sel & (flag > 0), axis=-1)
