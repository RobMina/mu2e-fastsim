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


def _one_event(hit, s, net):
    """hit: dict of [H] arrays for one event -> clusters [NTC] (valid, nsh, t0, t0err, members [NTC, H])."""
    H = hit["valid"].shape[0]
    valid = hit["valid"]
    nsh_i = jnp.where(valid, hit["nsh"], 0).astype(F64)
    cht = (hit["time"].astype(F64) - hit["z"].astype(F64) / (s["pitch"] * C_LIGHT)).astype(F32)
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

    (_, _, t0s, oks), _ = jax.lax.scan(peak, (jnp.zeros(nbins + 2, bool), 0, jnp.zeros(NTC, F32), jnp.zeros(NTC, bool)),
                                       jnp.arange(NCAND))
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

                    def step(i, st2):
                        c, changed = st2
                        dti = jnp.abs(cht[i] - c[2])
                        dphi = jnp.abs(_dphi(phi_h[i], pphi))
                        x = jnp.concatenate([jnp.stack([dti, dphi]), feats_static[i]])
                        v = mva.evaluate(net, x[None, :])[0]
                        go = valid[i] & ~c[0][i] & (dti < s["maxdt"] + c[3]) & (dphi < s["maxdphi"]) & (v > s["minadd"])
                        return jax.lax.cond(go, lambda: add(c, i), lambda: c), changed | go
                    return jax.lax.fori_loop(0, H, step, (c, False))
                c, _ = jax.lax.while_loop(rc_cond, rc_body, (c, True))
            return c

        c = jax.lax.cond(c[1] >= s["minnhits"], finish, lambda c: c, c)
        keep = ok0 & (c[1] >= s["minnhits"])
        return keep, c[1], c[2], c[3], c[0] & keep

    keep, nsh, t0, t0err, mem = jax.vmap(cluster)(members, oks)
    return {"valid": keep, "nsh": nsh, "t0": t0, "t0err": t0err, "members": mem}


def time_clusters(ph, s, net):
    """Panel hits [E, H] (hits.panel_hits) -> clusters per event [E, NTC]."""
    hit = {k: ph[k] for k in ("valid", "time", "z", "x", "y", "uvar", "wdist", "plane")}
    hit["nsh"] = ph["ncombo"]
    return jax.vmap(lambda h: _one_event(h, s, net))(hit)
