"""TimeClusterFinder (TrkPatRec) on the device, without calorimeter seeds: per event, vectorised over events.

Steps as the stock module: time spectrum of TOF-corrected hit times weighted by straw count, greedy peaks above
Ymin (bins of width (Tmax-Tmin)/rint((Tmax-Tmin)/Tbin), +-PeakWidth bins), hit assignment to the nearest peak within
DtMax + t0err, then per cluster: initial weighted median, azimuth prefilter, weighted mean, MVA refinement (remove
the worst hit below MinKeepHitMVA), MVA recovery (add hits above MinAddHitMVA), and the MinNHits cut.

Differences from stock, all small by construction and measured by the per-path comparison:
- the initial time/position is the exact weighted median (stock: boost's P^2 weighted-median estimate);
- azimuths use atan2 (stock: polyAtan2) and the MVA sums run in a different float order;
- calorimeter-seeded clusters are not made (the evaluated samples have no calorimeter digis).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import mva

F32, F64 = jnp.float32, jnp.float64
C_LIGHT = 299.792458          # mm/ns (CLHEP)
NCAND = 32                    # candidate peak bins examined per event
NTC = 4                       # clusters kept per event


def settings(m, pitch=None):
    if m.get("TestFlag", False):
        raise ValueError("device time clustering implements TestFlag=false (all hits)")
    return {"tmin": float(m["Tmin"]), "tmax": float(m["Tmax"]), "tbin": float(m["Tbin"]), "ymin": float(m["Ymin"]),
            "npeak": int(m["PeakWidth"]), "maxdt": float(m["DtMax"]), "minnhits": int(m["MinNHits"]),
            "maxdphi": float(m["MaxdPhi"]), "minkeep": float(m["MinKeepHitMVA"]), "minadd": float(m["MinAddHitMVA"]),
            "pitch": float(m["AveragePitch"] if pitch is None else pitch), "prefilter": bool(m["PrefilterCluster"]),
            "refine": bool(m["RefineClusters"]), "recover": bool(m["RecoverHits"])}


def _dphi(a, b):
    d = a - b
    return jnp.where(d > jnp.pi, d - 2 * jnp.pi, jnp.where(d < -jnp.pi, d + 2 * jnp.pi, d))


def _wmedian(v, w, m):
    """Weighted median of v (weights w, mask m): smallest value whose cumulative weight reaches half the total."""
    key = jnp.where(m, v, jnp.inf)
    o = jnp.argsort(key)
    vs, ws = key[o], jnp.where(m, w, 0.0)[o]
    cw = jnp.cumsum(ws)
    i = jnp.argmax(cw >= 0.5 * cw[-1])
    return vs[i]


def _peaks(hit, s):
    """hit: dict of [H] arrays for one event -> the per-event arrays the clusters use (`ev`) and the peak-assigned
    candidate clusters: members [NTC, H], oks [NTC] (a peak was made for the slot)."""
    valid = hit["valid"]
    nsh_i = jnp.where(valid, hit["nsh"], 0).astype(F64)
    # the time-of-flight denominator is pitch x c in double precision (a Python float, or the same value as an array
    # when producers that differ only in pitch share one batched call: time_clusters_multi)
    den = s["tof_den"] if "tof_den" in s else s["pitch"] * C_LIGHT
    cht = (hit["time"].astype(F64) - hit["z"].astype(F64) / den).astype(F32)
    nbins = int(round((s["tmax"] - s["tmin"]) / s["tbin"]))
    width = (s["tmax"] - s["tmin"]) / nbins
    b = jnp.floor((cht.astype(F64) - s["tmin"]) / width).astype(jnp.int64) + 1          # TH1 bin, 1..nbins
    inr = valid & (cht >= s["tmin"]) & (cht < s["tmax"])
    hist = jnp.zeros(nbins + 2, F32).at[jnp.where(inr, b, 0)].add(jnp.where(inr, nsh_i, 0.0).astype(F32))
    centre = s["tmin"] + (jnp.arange(nbins + 2) - 0.5) * width
    # candidates: bins >= Ymin by content (descending), ties by bin number; exact in float32 (integer contents)
    key = jnp.where((jnp.arange(nbins + 2) >= 1) & (jnp.arange(nbins + 2) <= nbins) & (hist >= s["ymin"]),
                    hist * 16384.0 - jnp.arange(nbins + 2), -jnp.inf)
    _, cand = jax.lax.top_k(key, NCAND)
    cand_ok = jnp.isfinite(key[cand])
    npk = s["npeak"]

    def peak(state, k):
        used, ntc, t0s, oks = state
        bc = cand[k]
        lo, hi = jnp.maximum(1, bc - npk), jnp.minimum(nbins + 1, bc + npk + 1)
        win = (jnp.arange(nbins + 2) >= lo) & (jnp.arange(nbins + 2) < hi)
        go = cand_ok[k] & ~used[bc]
        nsh = jnp.sum(jnp.where(win, hist, 0.0).astype(F32))
        t0 = (jnp.sum(jnp.where(win, centre * hist, 0.0)) / nsh).astype(F32)
        used = jnp.where(go, used | win, used)
        make = go & (nsh > s["minnhits"]) & (ntc < NTC)
        t0s = jnp.where(make, t0s.at[ntc].set(t0), t0s)
        oks = jnp.where(make, oks.at[ntc].set(True), oks)
        return (used, ntc + make, t0s, oks), None

    # candidates with content >= Ymin come first (top_k order) and the rest never make a peak: stop after them
    ncand = jnp.sum(cand_ok)

    def peak_while(state):
        k, st = state
        return k + 1, peak(st, k)[0]

    _, (_, _, t0s, oks) = jax.lax.while_loop(
        lambda state: state[0] < ncand, peak_while,
        (jnp.zeros((), ncand.dtype), (jnp.zeros(nbins + 2, bool), 0, jnp.zeros(NTC, F32), jnp.zeros(NTC, bool))))
    t0err0 = F32(s["tbin"] * 0.5)
    dt = jnp.abs(cht[:, None] - t0s[None, :])                                           # [H, NTC]
    elig = oks[None, :] & (dt < s["maxdt"] + t0err0)
    best = jnp.argmin(jnp.where(elig, dt, jnp.inf), axis=1)
    assigned = valid & jnp.any(elig, axis=1)
    members = assigned[None, :] & (best[None, :] == jnp.arange(NTC)[:, None])           # [NTC, H]
    phi_h = jnp.arctan2(hit["y"], hit["x"]).astype(F32)
    rho2 = (hit["x"].astype(F32) ** 2 + hit["y"].astype(F32) ** 2)
    feats_static = jnp.stack([rho2, hit["nsh"].astype(F32), hit["plane"].astype(F32),
                              jnp.sqrt(hit["uvar"]).astype(F32), jnp.abs(hit["wdist"]).astype(F32)], axis=-1)
    pos = jnp.stack([hit["x"], hit["y"], hit["z"]], -1).astype(F32)
    ev = {"valid": valid, "nsh_i": nsh_i, "cht": cht, "phi_h": phi_h, "feats": feats_static, "pos": pos}
    return ev, members, oks


def _cluster(ev, m0, ok0, s, net):
    """One candidate cluster (members m0 [H], ok0) of an event (`ev` from _peaks) -> keep, nsh, t0, t0err, members."""
    H = ev["valid"].shape[0]
    valid, nsh_i, cht, phi_h, feats_static, pos = (ev[k] for k in ("valid", "nsh_i", "cht", "phi_h", "feats", "pos"))

    def mva_of(t0, pphi):
        d = jnp.abs(cht - t0)
        dphi = jnp.abs(_dphi(phi_h, pphi))
        return mva.evaluate(net, jnp.concatenate([d[:, None], dphi[:, None], feats_static], -1)), d, dphi

    def remove(c, i):
        m, nsh, t0, t0err, p = c
        ni = nsh_i[i].astype(F32)
        denom = nsh - ni
        ok = denom > 0
        newt0 = (t0 * nsh - cht[i] * ni) / jnp.where(ok, denom, 1.0)
        var = t0err * t0err * nsh - (cht[i] - newt0) * (cht[i] - t0) * ni
        t0err = jnp.where(ok & (var > 0), jnp.sqrt(jnp.maximum(var, 0) / jnp.where(ok, denom, 1.0)), t0err)
        t0 = jnp.where(ok, newt0, t0)
        p = jnp.where(ok, (p * nsh - pos[i] * ni) / jnp.where(ok, denom, 1.0), p)
        nsh = jnp.where(ok, denom, nsh)
        return (m.at[i].set(False), nsh, t0, t0err, p)

    def add(c, i):
        m, nsh, t0, t0err, p = c
        ni = nsh_i[i].astype(F32)
        denom = nsh + ni
        newt0 = (t0 * nsh + cht[i] * ni) / denom
        t0err = jnp.sqrt((t0err * t0err * nsh + (cht[i] - newt0) * (cht[i] - t0) * ni) / denom)
        p = (p * nsh + pos[i] * ni) / denom
        return (m.at[i].set(True), denom, newt0, t0err, p)

    def cluster(m0, ok0):
        w = nsh_i.astype(F32)
        nsh = jnp.sum(jnp.where(m0, w, 0.0))
        t0 = _wmedian(cht, w, m0)
        tmn, tmx = jnp.min(jnp.where(m0, cht, jnp.inf)), jnp.max(jnp.where(m0, cht, -jnp.inf))
        nstrs = jnp.sum(m0)
        t0err = ((tmx - tmn) * F32(1.0 / np.sqrt(12.0)) / jnp.sqrt(jnp.maximum(nstrs, 1).astype(F32))).astype(F32)
        p = jnp.stack([_wmedian(pos[:, k], w, m0) for k in range(3)])
        c = (m0, nsh, t0, t0err, p)

        if s["prefilter"]:
            def pf_cond(st):
                return st[1] & (st[0][1] > 0)

            def pf_body(st):
                c, _ = st
                pphi = jnp.arctan2(c[4][1], c[4][0])
                ad = jnp.where(c[0], jnp.abs(_dphi(phi_h, pphi)), -1.0)
                i = jnp.argmax(ad)
                hit_ = ad[i] > s["maxdphi"]
                return jax.lax.cond(hit_, lambda: remove(c, i), lambda: c), hit_
            c, _ = jax.lax.while_loop(pf_cond, pf_body, (c, True))

        def finish(c):
            m = c[0]
            wsum = jnp.sum(jnp.where(m, w, 0.0))
            mean = jnp.sum(jnp.where(m, w * cht, 0.0)) / wsum
            var = jnp.sum(jnp.where(m, w * cht * cht, 0.0)) / wsum - mean * mean
            cnt = jnp.sum(m).astype(F32)
            t0err = jnp.sqrt(jnp.maximum(F32(1.0), 2.0 * var) / cnt).astype(F32)
            p = jnp.stack([jnp.sum(jnp.where(m, w * pos[:, k], 0.0)) / wsum for k in range(3)])
            c = (m, c[1], mean.astype(F32), t0err, p.astype(F32))
            if s["refine"]:
                def rf_cond(st):
                    return st[1] & (st[0][1] > 0)

                def rf_body(st):
                    c, _ = st
                    pphi = jnp.arctan2(c[4][1], c[4][0])
                    v, _, _ = mva_of(c[2], pphi)
                    v = jnp.where(c[0], v, F32(100.0))
                    i = jnp.argmin(v)
                    bad = v[i] < s["minkeep"]
                    return jax.lax.cond(bad, lambda: remove(c, i), lambda: c), bad
                c, _ = jax.lax.while_loop(rf_cond, rf_body, (c, True))
            if s["recover"]:
                def rc_cond(st):
                    return st[1] & (st[0][1] > 0)

                def rc_body(st):
                    c, _ = st
                    pphi = jnp.arctan2(c[4][1], c[4][0])
                    dphi_all = jnp.abs(_dphi(phi_h, pphi))
                    idx = jnp.arange(H)

                    # The stock pass visits hits 0..H-1 in order and adds a hit when it passes the cuts with the
                    # cluster as it is at that moment. A hit that fails the time or azimuth window (or is invalid or
                    # already a member) is not added whatever its MVA, so each iteration jumps to the next hit at or
                    # after `start` that passes those cuts with the current cluster and evaluates the MVA there, with
                    # the same single-row call as the per-hit loop: the same adds, in the same order and state.
                    def step(st2):
                        c, start, changed = st2
                        dti_all = jnp.abs(cht - c[2])
                        cheap = valid & ~c[0] & (dti_all < s["maxdt"] + c[3]) & (dphi_all < s["maxdphi"]) & (idx >= start)
                        any_ = jnp.any(cheap)
                        i = jnp.argmax(cheap)
                        dti, dphi = jnp.abs(cht[i] - c[2]), jnp.abs(_dphi(phi_h[i], pphi))
                        x = jnp.concatenate([jnp.stack([dti, dphi]), feats_static[i]])
                        v = mva.evaluate(net, x[None, :])[0]
                        go = any_ & (v > s["minadd"])
                        c = jax.lax.cond(go, lambda: add(c, i), lambda: c)
                        return c, jnp.where(any_, i + 1, H), changed | go
                    c, _, changed = jax.lax.while_loop(lambda st2: st2[1] < H, step,
                                                       (c, jnp.zeros((), idx.dtype), False))
                    return c, changed
                c, _ = jax.lax.while_loop(rc_cond, rc_body, (c, True))
            return c

        c = jax.lax.cond(c[1] >= s["minnhits"], finish, lambda c: c, c)
        keep = ok0 & (c[1] >= s["minnhits"])
        return keep, c[1], c[2], c[3], c[0] & keep

    return cluster(m0, ok0)


def _one_event(hit, s, net):
    """hit: dict of [H] arrays for one event -> clusters [NTC] (valid, nsh, t0, t0err, members [NTC, H])."""
    ev, members, oks = _peaks(hit, s)
    keep, nsh, t0, t0err, mem = jax.vmap(lambda m, o: _cluster(ev, m, o, s, net))(members, oks)
    return {"valid": keep, "nsh": nsh, "t0": t0, "t0err": t0err, "members": mem}


def _hits(ph):
    hit = {k: ph[k] for k in ("valid", "time", "z", "x", "y", "uvar", "wdist", "plane")}
    hit["nsh"] = ph["ncombo"]
    return hit


def time_clusters(ph, s, net):
    """Panel hits [E, H] (hits.panel_hits) -> clusters per event [E, NTC]."""
    return jax.vmap(lambda h: _one_event(h, s, net))(_hits(ph))


def time_clusters_multi(ph, ss, net):
    """time_clusters for several producers whose settings differ only in AveragePitch (stock TrigTimeClusterFinder and
    TrigTimeClusterFinderUe), in one batched call: each producer's clusters are the same computation as its own
    time_clusters call, and the producers share the loops' iterations. Returns one cluster dict per settings dict."""
    base = {k: v for k, v in ss[0].items() if k != "pitch"}
    if any({k: v for k, v in s_.items() if k != "pitch"} != base for s_ in ss):
        raise ValueError("time_clusters_multi: settings differ in more than the pitch")
    den = jnp.asarray([s_["pitch"] * C_LIGHT for s_ in ss])          # double-precision products, rounded once
    out = jax.vmap(lambda d: time_clusters(ph, {**ss[0], "tof_den": d}, net))(den)
    return [{k: v[i] for k, v in out.items()} for i in range(len(ss))]


def time_clusters_multi_pairs(ph, ss, net, K):
    """time_clusters_multi with the cluster refinement run only on the (producer, event, slot) pairs that have a peak
    candidate, gathered up to K pairs; when a batch has more than K candidates, the batched full path runs instead.
    Each candidate's cluster is the same computation as in time_clusters; slots without a candidate get valid False,
    nsh 0, t0 0, t0err 0 and no members (time_clusters gives them valid False too, and every consumer reads clusters
    through `valid`)."""
    base = {k: v for k, v in ss[0].items() if k != "pitch"}
    if any({k: v for k, v in s_.items() if k != "pitch"} != base for s_ in ss):
        raise ValueError("time_clusters_multi_pairs: settings differ in more than the pitch")
    den = jnp.asarray([s_["pitch"] * C_LIGHT for s_ in ss])
    s0 = ss[0]
    ev, members, oks = jax.vmap(lambda d: candidates(ph, {**s0, "tof_den": d}))(den)   # [P, E, ...]
    P, E, C = oks.shape

    def full():
        def one(ev_p, mem_p, ok_p):
            def per_event(ev_e, mem_e, ok_e):
                keep, nsh, t0, t0err, mem = jax.vmap(lambda m, o: _cluster(ev_e, m, o, s0, net))(mem_e, ok_e)
                return {"valid": keep, "nsh": nsh, "t0": t0, "t0err": t0err, "members": mem}
            return jax.vmap(per_event)(ev_p, mem_p, ok_p)
        return jax.vmap(one)(ev, members, oks)

    def pairs():
        flat = jnp.nonzero(oks.reshape(-1), size=K, fill_value=P * E * C)[0]
        use = flat < P * E * C
        f = jnp.minimum(flat, P * E * C - 1)
        p, e, c = f // (E * C), (f // C) % E, f % C
        evp = {k: v[p, e] for k, v in ev.items()}
        keep, nsh, t0, t0err, mem = jax.vmap(lambda ev_, m, o: _cluster(ev_, m, o, s0, net))(
            evp, members[p, e, c], oks[p, e, c] & use)
        dst = jnp.where(use, flat, P * E * C)

        def put(v, fill):
            out = jnp.full((P * E * C + 1,) + v.shape[1:], fill, v.dtype).at[dst].set(v)
            return out[:P * E * C].reshape((P, E, C) + v.shape[1:])
        return {"valid": put(keep & use, False), "nsh": put(nsh, 0), "t0": put(t0, 0), "t0err": put(t0err, 0),
                "members": put(mem & use[:, None], False)}

    out = jax.lax.cond(jnp.sum(oks) <= K, pairs, full)
    return [{k: v[i] for k, v in out.items()} for i in range(P)]


def candidates(ph, s):
    """Stage 1 of the compacted time clusters: per-event arrays and the candidate clusters [E, NTC] (members, oks)."""
    return jax.vmap(lambda h: _peaks(h, s))(_hits(ph))


def refine_pairs(ev, members, oks, s, net, K):
    """Stage 2 of the compacted time clusters: _cluster on the first K candidate (event, slot) pairs of `oks` [E, NTC]
    only (K >= their count: the caller reads it), scattered back to [E, NTC]. Each cluster is the same computation as
    in time_clusters; slots without a candidate get valid False, nsh 0, t0 0, t0err 0 and no members (time_clusters
    gives them valid False too, and every consumer reads clusters through `valid`)."""
    E, C = oks.shape
    flat = jnp.nonzero(oks.reshape(-1), size=K, fill_value=E * C)[0]
    use = flat < E * C
    e, c = jnp.minimum(flat // C, E - 1), flat % C
    evp = {k: v[e] for k, v in ev.items()}
    keep, nsh, t0, t0err, mem = jax.vmap(lambda ev_, m, o: _cluster(ev_, m, o, s, net))(evp, members[e, c], oks[e, c] & use)
    dst = jnp.where(use, flat, E * C)

    def put(v, fill):
        out = jnp.full((E * C + 1,) + v.shape[1:], fill, v.dtype).at[dst].set(v)
        return out[:E * C].reshape((E, C) + v.shape[1:])
    return {"valid": put(keep & use, False), "nsh": put(nsh, 0), "t0": put(t0, 0), "t0err": put(t0err, 0),
            "members": put(mem & use[:, None], False)}
