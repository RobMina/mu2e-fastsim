"""TimeClusterFinder (TrkPatRec, Offline v13_41_01) on the device, with the stock arithmetic: per event, vectorised
over events, including the calorimeter-seeded clusters (UseCaloCluster, UseCaloClusterPosition).

A port of the stock module as the literal transcription tests/time_cluster_reference.py writes it, traced under
`exact.strict` in both precisions of the chain: float32 where stock computes in float, float64 where a C++ double
promotes (TrkT0's t0 and t0err, the time-of-flight correction, the TH1F axis, double literals, the calo cog); the
products an add follows are guarded (`exact.m`, `exact.g`) and float64 divisions by a scalar are elementwise
(rhelix._div). Steps, in the stock order:

- calo seeds (with calo clusters given and UseCaloCluster): the clusters with energy > CaloClusterMinE, in cluster
  order, t0 = caloClusterTime (the cog moved to Mu2e and the tracker frame), t0err = CaloTimeErr. They come first
  in the collection and blank PeakWidth bins around FindBin(t0) (underflow and overflow included) before the peaks;
- the time spectrum: a TH1F of the float TOF-corrected hit times weighted by the straw count (ROOT 6.38's
  TAxis::FindBin and GetBinCenter), the bins >= Ymin in the order libstdc++'s std::sort leaves them (content
  descending, ties as introsort leaves them: mpr._stock_sort), greedy peaks of +-PeakWidth bins with float sums
  of the double contents and centres, kept when nsh > MinNHits;
- assignHits: each hit to the closest cluster (collection order, strict <) with fabs(float - double) stored in a
  float below DtMax + t0err (double);
- per cluster: boost's weighted P^2 median (float samples and weights, the calo cog x and y after the hits) for t0
  (not a calo cluster) and the position, t0err from the time range; the polyAtan2 / Angles::deltaPhi prefilter;
  clusterMean (float sums in hit order, the calo cog after the hits); the MVA refinement (ClusterMVA, or
  ClusterCaloMVA for a calo cluster: mva.evaluate_stock, glibc's expf) removing the first worst hit below
  MinKeepHitMVA; the recovery adding hits above MinAddHitMVA in index order with the cluster as it is at that
  moment; the MinNHits cut on the uint16 straw count. removeHit / addHit use TrkT0's double t0 and t0err with the
  float / unsigned operands and XYZVectorF arithmetic (v / a as v * (1 / a)); a calo cluster's t0 never changes.

Slots per event: the calo clusters' slots first (slot c is calo cluster c, valid when it seeds a kept cluster), then
NPEAK peak slots in peak order, so the valid slots in slot order are the stock collection. Fixed capacities, flagged
per event in `overflow` (never dropped silently): 1 more bins >= Ymin than NCAND, 2 more peak clusters than NPEAK,
4 std::sort's heap-sort fallback (not reproduced). The calo cog must reach the port in double (calo.reconstruct makes
it so on the device; a float64 jit argument is rounded to float32 when x64 is off).

The pair paths (time_clusters_multi_pairs, refine_pairs) run the gathered clusters one after the other (`LANES`), so
each cluster's loops run its own iteration counts; refine evaluates the MVA on the members gathered to REFINE_CAP rows
(more members: on all hits, the same values).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import mva
from .exact import g as _g, m as _m, strict
from .mpr import _argmax, _d, _f, _stock_sort, poly_atan2
from .rhelix import _angles_dphi, _div

F32, F64, I32, I64 = jnp.float32, jnp.float64, jnp.int32, jnp.int64
C_LIGHT = (2.99792458e+8 * 1000.0) / 1.e+9   # CLHEP::c_light, mm/ns
DETECTOR_SYSTEM_Z0 = 10171.0                 # mu2e.detectorSystemZ0 (geom_run1_a.txt), mm
NCAND = 64                    # bins >= Ymin per event (dumps: at most 25, cosmics)
NPEAK = 16                    # peak cluster slots per event (dumps: at most 10, cosmics)
NTC = NPEAK                   # cluster slots per event without calo clusters
OVF_CAND, OVF_PEAKS, OVF_SORT = 1, 2, 4
LANES = 1                     # gathered clusters per mapped step (1: each cluster runs its own loop counts; CPU)
REFINE_CAP = 128              # refine's MVA on the members gathered to this many rows (more members: on all hits)
ORDER_NONE = np.int32(2 ** 30)       # TimeCluster hit-order key of a non-member
INVSQRT12 = np.float32(1.0 / math.sqrt(12.0))
FLT_MAX = np.float32(np.finfo(np.float32).max)
_P = np.float32(0.5)                 # the median's quantile probability (float)


def geometry():
    """Calo disk origin z (Mu2e) per disk and the tracker centre z (the disks are not rotated)."""
    from . import calogeom
    return {"disk_z": tuple(float(v) for v in np.asarray(calogeom.build()["origin"], np.float64)[:, 2]),
            "tracker_z": DETECTOR_SYSTEM_Z0}


def settings(m, pitch=None, geom=None):
    """TimeClusterFinder module table -> static settings. The calo seeds run when the call is given calo clusters and
    UseCaloCluster is true."""
    if m.get("TestFlag", False):
        raise ValueError("device time clustering implements TestFlag=false (all hits)")
    t0c = m.get("T0Calculator", {})
    if not t0c.get("UseTOTDrift", True):
        raise ValueError("device time clustering implements UseTOTDrift true")
    g = geom or geometry()
    return {"tmin": float(m["Tmin"]), "tmax": float(m["Tmax"]), "tbin": float(m["Tbin"]), "ymin": float(m["Ymin"]),
            "npeak": int(m["PeakWidth"]), "maxdt": float(m["DtMax"]), "minnhits": int(m["MinNHits"]),
            "maxdphi": float(m["MaxdPhi"]), "minkeep": float(m["MinKeepHitMVA"]), "minadd": float(m["MinAddHitMVA"]),
            "pitch": float(m["AveragePitch"] if pitch is None else pitch), "prefilter": bool(m["PrefilterCluster"]),
            "refine": bool(m["RefineClusters"]), "recover": bool(m["RecoverHits"]),
            "usecc": bool(m.get("UseCaloCluster", False)), "useccpos": bool(m.get("UseCaloClusterPosition", False)),
            "ccmine": float(m.get("CaloClusterMinE", 0.0)), "ccwt": float(m.get("CaloClusterWeight", 0.0)),
            "beta": float(t0c.get("StrawHitBeta", 1.0)), "calo_zoff": float(t0c.get("CaloShowermaxZ", -120.0)),
            "calo_toff": float(t0c.get("TrkToCaloTimeOffset", -0.4)),
            "calo_terr": float(t0c.get("CaloTimeErr", 0.5)), "disk_z": g["disk_z"], "tracker_z": g["tracker_z"]}


def _c(s):
    """Derived constants (float settings rounded to float as the module's float members)."""
    f = np.float32
    tmin, tmax, tbin = f(s["tmin"]), f(s["tmax"]), f(s["tbin"])
    n = int(np.rint(f(f(tmax - tmin) / tbin)))
    return {"n": n, "xmin": float(tmin), "xmax": float(tmax), "width": (float(tmax) - float(tmin)) / n,
            "tbin": float(tbin), "ymin": float(f(s["ymin"])), "maxdt": float(f(s["maxdt"])),
            "maxdphi": f(s["maxdphi"]), "minkeep": f(s["minkeep"]), "minadd": f(s["minadd"]),
            "minnhits": int(s["minnhits"]), "npeak": int(s["npeak"]), "ccmine": f(s.get("ccmine", 0.0)),
            "ccwt": f(s.get("ccwt", 0.0)), "beta": float(s.get("beta", 1.0))}


def tof_den(s):
    """pitch * beta * c_light (double) with the float AveragePitch."""
    return float(np.float32(s["pitch"])) * float(s.get("beta", 1.0)) * C_LIGHT


# --- TH1F axis --------------------------------------------------------------------------------------------------------

def _find_bin(x, k):
    """TAxis::FindBin (ROOT 6.38, fixed bins) of double x."""
    xmin, xmax, w, n = k["xmin"], k["xmax"], k["width"], k["n"]
    q = _div(x - xmin, w)
    a = jnp.trunc(jnp.where((x >= xmin) & (x < xmax), q, 0.0))
    ai = a.astype(I32)
    b = 1 + ai - (x < xmin + _g(w * a)).astype(I32) + (xmin + _g(w * (a + 1.0)) <= x).astype(I32)
    return jnp.where(x < xmin, 0, jnp.where(~(x < xmax), n + 1, b)).astype(I32)


def _center(b, k):
    """TAxis::GetBinCenter (ROOT 6.38) of int bins b."""
    return k["xmin"] + _g((b.astype(F64) - 0.5) * ((k["xmax"] - k["xmin"]) / float(k["n"])))


# --- stage 1: seeds, peaks, hit assignment ------------------------------------------------------------------------------

def _hits(ph):
    hit = {k: ph[k] for k in ("valid", "time", "z", "x", "y", "uvar", "wdist", "plane")}
    hit["nsh"] = ph["ncombo"]
    return hit


def _calo_in(cl, E):
    keys = ("valid", "disk", "time", "energy", "cog_x", "cog_y")
    out = {k: cl[k] for k in keys}
    out["cog_z"] = cl["cog_z"] if "cog_z" in cl else jnp.zeros(cl["time"].shape, F64)
    return out


def _peaks(hit, cc, s):
    """One event: hit dict of [H] arrays (+ calo clusters cc [C] or None) -> per-event arrays (`ev`), the per-slot
    seeds (members [K, H], t0, t0err, calo, cog [K, 2]) and oks [K]."""
    k = _c(s)
    valid = hit["valid"]
    H = valid.shape[0]
    nsh = jnp.where(valid, jnp.asarray(hit["nsh"]).astype(I32), 0)
    den = s["tof_den"] if "tof_den" in s else tof_den(s)
    t, x, y, z = (_f(hit[q]) for q in ("time", "x", "y", "z"))
    cht = _f(_d(t) - _div(_d(z), den))
    nb = k["n"] + 1
    # time spectrum
    hb = _find_bin(_d(cht), k)
    hist = jnp.zeros(nb + 1, F32).at[jnp.where(valid, hb, nb + 1)].add(nsh.astype(F32), mode="drop")
    hist_d = _d(hist)
    # calo seeds
    if cc is not None and s.get("usecc", False):
        C = cc["time"].shape[0]
        dz = jnp.asarray(s["disk_z"], F64)[jnp.clip(jnp.asarray(cc["disk"]).astype(I32), 0, len(s["disk_z"]) - 1)]
        zc = (dz + _d(cc["cog_z"])) - s["tracker_z"]
        t0c = (_d(_f(cc["time"])) - _div(zc + s["calo_zoff"], den)) + s["calo_toff"]
        seed = jnp.asarray(cc["valid"]) & (_f(cc["energy"]) > k["ccmine"])
        cog = jnp.stack([_d(cc["cog_x"]), _d(cc["cog_y"])], -1)
    else:
        C = 0
        t0c, seed, cog = jnp.zeros(0, F64), jnp.zeros(0, bool), jnp.zeros((0, 2), F64)
    npk = k["npeak"]
    used = jnp.zeros(nb + 1, bool)                    # alreadyUsed[0 .. nbins]; index nb is a dump slot
    if C:
        ib = _find_bin(t0c, k)
        for o in range(-npk, npk + 1):
            j = ib + o
            ok = seed & (j >= 1) & (j < nb)
            used = used.at[jnp.where(ok, j, nb)].set(True)
    # candidate bins >= Ymin in bin order, sorted as std::sort (content descending)
    bins_all = jnp.arange(nb + 1, dtype=I32)
    sel = (bins_all >= 1) & (bins_all < nb) & (hist_d >= k["ymin"])
    ncand = jnp.sum(sel).astype(I32)
    cb = jnp.nonzero(sel, size=NCAND, fill_value=0)[0].astype(I32)
    slot = jnp.arange(NCAND, dtype=I32)
    csel = slot < ncand
    key = jnp.where(csel, -hist[cb].astype(I32), jnp.iinfo(jnp.int32).max)
    perm, _, fallback = _stock_sort(key, csel)
    order_bins = cb[perm]

    def body(i, st):
        used, npeaks, pt0 = st
        b = order_bins[i]
        go = (i < jnp.minimum(ncand, NCAND)) & ~used[b]
        nshf, t0 = np.float32(0.0), np.float32(0.0)
        nshf, t0 = jnp.asarray(nshf, F32), jnp.asarray(t0, F32)
        for o in range(-npk, npk + 1):
            j = b + o
            inw = (j >= 1) & (j < nb)
            jj = jnp.clip(j, 0, nb)
            c = hist_d[jj]
            nshf = jnp.where(inw, _f(_d(nshf) + c), nshf)
            t0 = jnp.where(inw, _f(_d(t0) + _g(_center(jj, k) * c)), t0)
            used = used.at[jnp.where(go & inw, jj, nb)].set(True)
        t0 = t0 / jnp.where(go, nshf, np.float32(1.0))
        make = go & (nshf > np.float32(k["minnhits"]))
        pt0 = jnp.where(make, pt0.at[jnp.minimum(npeaks, NPEAK)].set(t0, mode="drop"), pt0)
        return used, npeaks + make.astype(I32), pt0

    _, npeaks, pt0 = jax.lax.fori_loop(0, NCAND, body, (used, jnp.int32(0), jnp.zeros(NPEAK, F32)))
    overflow = (jnp.where(ncand > NCAND, OVF_CAND, 0) | jnp.where(npeaks > NPEAK, OVF_PEAKS, 0)
                | jnp.where(fallback, OVF_SORT, 0)).astype(I32)
    # the collection: calo seeds, then peaks
    pk_ok = jnp.arange(NPEAK) < npeaks
    t0s = jnp.concatenate([t0c, _d(pt0)])
    t0e = jnp.concatenate([jnp.full(C, s.get("calo_terr", 0.5), F64), jnp.full(NPEAK, k["tbin"] * 0.5, F64)])
    oks = jnp.concatenate([seed, pk_ok])
    calo = jnp.concatenate([jnp.arange(C, dtype=I32), jnp.full(NPEAK, -1, I32)])
    cogs = jnp.concatenate([cog, jnp.zeros((NPEAK, 2), F64)])
    # assignHits
    dt = _f(jnp.abs(_d(cht)[:, None] - t0s[None, :]))                                  # [H, K]
    elig = oks[None, :] & (_d(dt) < k["maxdt"] + t0e[None, :]) & (dt < np.float32(1e5))
    best = jax.lax.argmin(jnp.where(elig, dt, np.float32(np.inf)), 1, I32)
    assigned = valid & jnp.any(elig, axis=1)
    Kc = C + NPEAK
    members = assigned[None, :] & (best[None, :] == jnp.arange(Kc)[:, None])              # [K, H]
    # per-hit inputs of the clusters
    phi = poly_atan2(y, x)
    uvar, wdist = _f(hit["uvar"]), _f(hit["wdist"])
    feats = jnp.stack([_m(x, x) + _m(y, y), nsh.astype(F32), jnp.asarray(hit["plane"]).astype(F32),
                       jnp.sqrt(uvar), jnp.abs(wdist)], -1)
    pos = jnp.stack([x, y, z], -1)
    ev = {"valid": valid, "nsh": nsh, "cht": cht, "phi": phi, "feats": feats, "pos": pos, "overflow": overflow}
    sd = {"members": members, "t0": t0s, "t0err": t0e, "calo": calo, "cog": cogs}
    return ev, sd, oks


# --- stage 2: one cluster -------------------------------------------------------------------------------------------

def _ub(h, x):
    """std::upper_bound over the 5 heights h [L, 5] (float) of x [L] (double): libstdc++'s binary search."""
    L = h.shape[0]
    first, length = jnp.zeros(L, I32), jnp.full(L, 5, I32)
    for _ in range(3):
        half = length >> 1
        mid = first + half
        less = x < _d(jnp.take_along_axis(h, jnp.clip(mid, 0, 4)[:, None], 1)[:, 0])
        go = length > 0
        first = jnp.where(go & ~less, mid + 1, first)
        length = jnp.where(go, jnp.where(less, half, length - half - 1), length)
    return first


def _wp2_sort5(h, ap):
    """The fifth sample's initialisation: heights sorted with min_element swaps (the weights follow), then the
    weights summed into the actual positions. h, ap [L, 5]."""
    L = h.shape[0]
    lanes = jnp.arange(L)
    for b in range(5):
        m = b + jax.lax.argmin(h[:, b:], 1, I32)
        hb, hm = h[:, b], h[lanes, m]
        ab, am = ap[:, b], ap[lanes, m]
        h = h.at[lanes, m].set(hb).at[:, b].set(hm)
        ap = ap.at[lanes, m].set(ab).at[:, b].set(am)
    for i in range(1, 5):
        ap = ap.at[:, i].set(ap[:, i] + ap[:, i - 1])
    return h, ap


def _wp2_late(st, x, w):
    """A sample after the fifth: x [L] (double: the C++ argument type; stored as float), w [L] (float)."""
    cnt, sw, h, ap, dp = st
    sw1 = sw + w
    xf = _f(x)
    lo = x < _d(h[:, 0])
    top = ~lo & (_d(h[:, 4]) <= x)
    kc = jnp.where(lo, 1, jnp.where(top, 4, _ub(h, x)))
    hl = jnp.where(lo[:, None] & (jnp.arange(5) == 0), xf[:, None], h)
    hl = jnp.where(top[:, None] & (jnp.arange(5) == 4), xf[:, None], hl)
    al = jnp.where(lo[:, None] & (jnp.arange(5) == 0), w[:, None], ap)
    al = jnp.where(jnp.arange(5)[None, :] >= kc[:, None], al + w[:, None], al)
    S, a0 = sw1, al[:, 0]
    sa = S - a0
    d1 = _f(_d(_m(sa, _P)) / 2. + _d(a0))
    d2 = _m(sa, _P) + a0
    d3 = _f(_g(_d(sa) * (1. + float(_P))) / 2. + _d(a0))
    dl = jnp.stack([a0, d1, d2, d3, S], -1)
    one = np.float32(1.0)
    for i in (1, 2, 3):
        d = dl[:, i] - al[:, i]
        dpp = al[:, i + 1] - al[:, i]
        dm = al[:, i - 1] - al[:, i]
        hp = _g((hl[:, i + 1] - hl[:, i]) / dpp)
        hm_ = _g((hl[:, i - 1] - hl[:, i]) / dm)
        cnd = ((d >= one) & (dpp > one)) | ((d <= -one) & (dm < -one))
        sd = jnp.where(d > 0, one, -one)
        hh = hl[:, i] + _m(sd / (dpp - dm), _m(sd - dm, hp) + _m(dpp - sd, hm_))
        nh = jnp.where((hl[:, i - 1] < hh) & (hh < hl[:, i + 1]), hh,
                       jnp.where(d > 0, hl[:, i] + hp, jnp.where(d < 0, hl[:, i] - hm_, hl[:, i])))
        hl = hl.at[:, i].set(jnp.where(cnd, nh, hl[:, i]))
        al = al.at[:, i].set(jnp.where(cnd, al[:, i] + sd, al[:, i]))
    return cnt + 1, sw1, hl, al, dl


def _wp2_step(st, x, w, on):
    """boost weighted_p_square_quantile_impl (median) for L accumulators at once, any sample: x [L] (double),
    w [L] (float), on [L]: the first five samples are stored (sorted at the fifth), the later ones update."""
    cnt, sw, h, ap, dp = st
    L = h.shape[0]
    lanes = jnp.arange(L)
    cnt1 = cnt + 1
    idx = jnp.clip(cnt1 - 1, 0, 4)
    hi = h.at[lanes, idx].set(_f(x))
    ai = ap.at[lanes, idx].set(w)
    hs, as_ = _wp2_sort5(hi, ai)
    five = (cnt1 == 5)[:, None]
    hi, ai = jnp.where(five, hs, hi), jnp.where(five, as_, ai)
    _, _, hl, al, dl = _wp2_late(st, x, w)
    early = (cnt1 <= 5)[:, None]
    h2, a2, d2_ = jnp.where(early, hi, hl), jnp.where(early, ai, al), jnp.where(early, dp, dl)
    o = on[:, None]
    return (jnp.where(on, cnt1, cnt), jnp.where(on, sw + w, sw), jnp.where(o, h2, h), jnp.where(o, a2, ap),
            jnp.where(o, d2_, dp))


def _wp2_run(samples, w, mi, n):
    """The P^2 accumulators [L] after the samples samples[mi[0 .. n-1]] ([H, L] double, weights w [H] float): the
    first five stored and sorted once, then a loop over the later ones only."""
    L = samples.shape[1]
    on = jnp.arange(5) < jnp.minimum(n, 5)
    j5 = mi[:5] if mi.shape[0] >= 5 else jnp.pad(mi, (0, 5 - mi.shape[0]))
    x5 = jnp.where(on[:, None], _f(samples[j5]), np.float32(0.0)).T                   # [L, 5]
    w5 = jnp.where(on, w[j5], np.float32(0.0))
    sw = jnp.zeros((), F32)
    for i in range(5):
        sw = sw + w5[i]
    a5 = jnp.broadcast_to(w5, (L, 5))
    hs, as_ = _wp2_sort5(x5, a5)
    full = n >= 5
    h, ap = jnp.where(full, hs, x5), jnp.where(full, as_, a5)
    st = (jnp.full(L, jnp.minimum(n, 5), I32), jnp.full(L, sw, F32), h, ap, jnp.zeros((L, 5), F32))

    def body(c):
        j, st = c
        i = mi[j]
        return j + 1, _wp2_late(st, samples[i], jnp.full(L, w[i], F32))
    _, st = jax.lax.while_loop(lambda c: c[0] < n, body, (jnp.int32(5), st))
    return st


def _wp2_init(L):
    z = jnp.zeros((L, 5), F32)
    return jnp.zeros(L, I32), jnp.zeros(L, F32), z, z, z


def _members(m):
    """The member indices in index order (padded) and their count."""
    return jnp.nonzero(m, size=m.shape[0], fill_value=0)[0].astype(I32), jnp.sum(m).astype(I32)


def _seq_scan(fn, c0, xs):
    c, _ = jax.lax.scan(lambda c, a: (fn(c, *a), None), c0, xs)
    return c


def _nets(net, cnet):
    """The two nets' weight matrices stacked [2, ...] (ClusterMVA, ClusterCaloMVA) when both are given."""
    wn = mva.stock_layers(net)
    if cnet is None:
        return wn, None
    wc = mva.stock_layers(cnet)
    if [w.shape for w in wn] != [w.shape for w in wc] or bool(net["norm"]) != bool(cnet["norm"]):
        raise ValueError("ClusterMVA and ClusterCaloMVA differ in layout")
    return wn, [np.stack([a, b]) for a, b in zip(wn, wc)]


def _cluster(ev, sd, ok0, s, net, cnet=None):
    """One candidate cluster slot (sd: members [H], t0, t0err, calo, cog [2]) of an event -> keep, nsh, t0, t0err,
    pos, members, calo, and with s["order"] the stock TimeCluster hit order as a key per hit (_strawHitIdxs: the
    assigned hits in index order, then the recovered hits in the order recoverHits adds them; ORDER_NONE else)."""
    k = _c(s)
    H = ev["valid"].shape[0]
    order = bool(s.get("order", False))
    valid, nsh_i, cht, phi_h, feats, pos = (ev[q] for q in ("valid", "nsh", "cht", "phi", "feats", "pos"))
    is_calo = sd["calo"] >= 0
    cog = sd["cog"]
    useccpos = bool(s.get("useccpos", False))
    wn, wboth = _nets(net, cnet)
    if wboth is None:
        net_c, layers = net, wn
    else:                                         # ClusterCaloMVA for a calo cluster: the same layout, other numbers
        pick = lambda a, b: jnp.where(is_calo, jnp.asarray(b, F32), jnp.asarray(a, F32))
        layers = [jnp.where(is_calo, w[1], w[0]) for w in wboth]
        net_c = {"act": "sigmoid", "norm": net["norm"], "offset": pick(net["offset"], cnet["offset"]),
                 "scale": pick(net["scale"], cnet["scale"])}
    wf = nsh_i.astype(F32)
    idx = jnp.arange(H, dtype=I32)

    def mva_of(x):
        return mva.evaluate_stock(net_c, x, layers)

    def inputs(t0, pphi, rows=None):
        c_, p_, f_ = (cht, phi_h, feats) if rows is None else (cht[rows], phi_h[rows], feats[rows])
        dt = _f(jnp.abs(_d(c_) - t0))
        dphi = jnp.abs(_angles_dphi(p_, pphi))
        return jnp.concatenate([dt[..., None], dphi[..., None], f_], -1)

    # state: members, nsh (uint16 as int32), t0, t0err (double), pos (float [3]), order keys, next order key
    def remove(c, i):
        m, nsh, t0, t0err, p = c[:5]
        n = nsh_i[i]
        nf = n.astype(F32)
        denom = ((nsh.astype(I64) - n.astype(I64)) & 0xFFFFFFFF).astype(F32)      # float(unsigned)
        ok = denom > 0
        dd = _d(jnp.where(ok, denom, np.float32(1.0)))
        newt0 = _f((_m(t0, _d(nsh)) - _d(_m(cht[i], nf))) / dd)
        var = _m(t0err, t0err, _d(nsh)) - _m(_d(_f(cht[i] - newt0)), _d(cht[i]) - t0, _d(n))
        t0err2 = jnp.where(var > 0.0, jnp.sqrt(jnp.where(var > 0.0, var, 1.0) / dd), t0err)
        upd = ok & ~is_calo
        t0 = jnp.where(upd, _d(newt0), t0)
        t0err = jnp.where(upd, t0err2, t0err)
        rec = _g(np.float32(1.0) / jnp.where(ok, denom, np.float32(1.0)))
        p2 = _m(_m(p, nsh.astype(F32)) - _m(pos[i], nf), rec)
        p = jnp.where(ok, p2, p)
        nsh = jnp.where(ok, (nsh - n) & 0xFFFF, nsh)
        return (m.at[i].set(False), nsh, t0, t0err, p) + tuple(c[5:])

    def add(c, i):
        m, nsh, t0, t0err, p = c[:5]
        n = nsh_i[i]
        nf = n.astype(F32)
        denom = (nsh + n).astype(F32)
        dd = _d(denom)
        newt0 = _f((_m(t0, _d(nsh)) + _d(_m(cht[i], nf))) / dd)
        t0err2 = jnp.sqrt((_m(t0err, t0err, _d(nsh)) + _m(_d(_f(cht[i] - newt0)), _d(cht[i]) - t0, _d(n))) / dd)
        t0 = jnp.where(is_calo, t0, _d(newt0))
        t0err = jnp.where(is_calo, t0err, t0err2)
        p = _m(_m(p, nsh.astype(F32)) + _m(pos[i], nf), _g(np.float32(1.0) / denom))
        nsh = (nsh + n) & 0xFFFF
        o = (c[5].at[i].set(c[6]), c[6] + 1) if order else ()
        return (m.at[i].set(True), nsh, t0, t0err, p) + o

    m0 = sd["members"] & ok0
    # initCluster
    nsh0 = jnp.sum(jnp.where(m0, nsh_i, 0)) & 0xFFFF
    samples = jnp.stack([_d(cht), _d(pos[:, 0]), _d(pos[:, 1]), _d(pos[:, 2])], -1)          # [H, 4]
    mi0, n0 = _members(m0)
    st = _wp2_run(samples, wf, mi0, n0)
    if useccpos:
        z0 = jnp.zeros((), F64)
        cx = jnp.stack([z0, _d(cog[0]), _d(cog[1]), z0])
        con = is_calo & (jnp.arange(4) >= 1) & (jnp.arange(4) <= 2)
        st = _wp2_step(st, cx, jnp.full(4, np.float32(k["ccwt"]), F32), con)
    med = st[2][:, 2]
    tmn = jnp.min(jnp.where(m0, cht, FLT_MAX))
    tmx = jnp.max(jnp.where(m0, cht, -FLT_MAX))
    nstrs = jnp.sum(m0).astype(F64)
    t0e_init = _d(_m(tmx - tmn, INVSQRT12)) / _g(jnp.sqrt(nstrs))
    t0 = jnp.where(is_calo, sd["t0"], _d(med[0]))
    t0err = jnp.where(is_calo, sd["t0err"], t0e_init)
    c = (m0, nsh0.astype(I32), t0, t0err, med[1:4].astype(F32))
    if order:
        c = c + (jnp.where(m0, idx, ORDER_NONE), jnp.int32(H))

    if s["prefilter"]:
        def pf_body(st):
            c, _ = st
            pphi = poly_atan2(c[4][1], c[4][0])
            ad = jnp.where(c[0], jnp.abs(_angles_dphi(phi_h, pphi)), np.float32(-1.0))
            mx = jnp.max(ad)
            i = _argmax(c[0] & (ad == mx))
            hit_ = mx > k["maxdphi"]
            return jax.lax.cond(hit_, lambda: remove(c, i), lambda: c), hit_
        c, _ = jax.lax.while_loop(lambda st: st[1] & (st[0][1] > 0), pf_body, (c, jnp.bool_(True)))

    def finish(c):
        m = c[0]
        # clusterMean: float sums in hit order
        mi, cnt = _members(m)

        def acc(a):
            j, sw, ws, wm2, sx = a
            i = mi[j]
            w, x, p = wf[i], cht[i], pos[i]
            return j + 1, sw + w, ws + _m(x, w), wm2 + _m(w, _m(x, x)), sx + _m(p, w)
        z1 = jnp.zeros((), F32)
        _, sw, ws, wm2, sx = jax.lax.while_loop(lambda a: a[0] < cnt, acc,
                                                (jnp.int32(0), z1, z1, z1, jnp.zeros(3, F32)))
        swx = jnp.stack([sw, sw, sw])
        if useccpos:
            cw = np.float32(k["ccwt"])
            on2 = is_calo & (jnp.arange(3) < 2)
            sx = jnp.where(on2, _f(_d(sx) + _g(jnp.concatenate([cog, jnp.zeros(1, F64)]) * float(cw))), sx)
            swx = jnp.where(on2, swx + cw, swx)
        mean = ws / sw
        var = _g(wm2 / sw) - _m(mean, mean)
        t0e_m = _d(jnp.sqrt(_f(jnp.maximum(1.0, 2.0 * _d(var)) / _d(cnt))))
        t0 = jnp.where(is_calo, c[2], _d(mean))
        t0err = jnp.where(is_calo, c[3], t0e_m)
        c = (m, c[1], t0, t0err, _div(sx, swx)) + tuple(c[5:])      # elementwise: no reciprocal product
        if s["refine"]:
            cap = min(H, REFINE_CAP)

            def worst_all(c, pphi):
                v = jnp.where(c[0], mva_of(inputs(c[2], pphi)), np.float32(np.inf))
                mn = jnp.min(v)
                return mn, _argmax(c[0] & (v == mn))

            def worst_members(c, pphi):              # the members in index order: the first lowest is the same hit
                mi = jnp.nonzero(c[0], size=cap, fill_value=0)[0].astype(I32)
                v = jnp.where(jnp.arange(cap) < jnp.sum(c[0]), mva_of(inputs(c[2], pphi, mi)), np.float32(np.inf))
                mn = jnp.min(v)
                return mn, mi[_argmax(v == mn)]

            def rf_body(st):
                c, _ = st
                pphi = poly_atan2(c[4][1], c[4][0])
                if cap < H:
                    mn, i = jax.lax.cond(jnp.sum(c[0]) <= cap, worst_members, worst_all, c, pphi)
                else:
                    mn, i = worst_all(c, pphi)
                bad = (mn < np.float32(100.0)) & (mn < k["minkeep"])
                return jax.lax.cond(bad, lambda: remove(c, i), lambda: c), bad
            c, _ = jax.lax.while_loop(lambda st: st[1] & (st[0][1] > 0), rf_body, (c, jnp.bool_(True)))
        if s["recover"]:
            def rc_pass(st):
                c, _ = st
                pphi = poly_atan2(c[4][1], c[4][0])
                dphi_all = jnp.abs(_angles_dphi(phi_h, pphi))

                # recoverHits visits the hits 0..H-1 with the cluster as it is at that moment; a hit outside the
                # time or azimuth window (or invalid, or a member) is skipped whatever its MVA, so each step jumps to
                # the next hit at or after `start` inside the windows and evaluates the MVA there
                def step(st2):
                    c, start, changed = st2
                    dti = _f(jnp.abs(_d(cht) - c[2]))
                    cheap = (valid & ~c[0] & (_d(dti) < k["maxdt"] + c[3]) & (dphi_all < k["maxdphi"])
                             & (idx >= start))
                    any_ = jnp.any(cheap)
                    i = _argmax(cheap)
                    v = mva_of(inputs(c[2], pphi, i))
                    go = any_ & (v > k["minadd"])
                    c = jax.lax.cond(go, lambda: add(c, i), lambda: c)
                    return c, jnp.where(any_, i + 1, H).astype(I32), changed | go
                c, _, changed = jax.lax.while_loop(lambda st2: st2[1] < H, step, (c, jnp.int32(0), jnp.bool_(False)))
                return c, changed
            c = jax.lax.cond(c[1] > 0, lambda c: jax.lax.while_loop(lambda st: st[1], rc_pass, (c, jnp.bool_(True)))[0],
                             lambda c: c, c)
        return c

    c = jax.lax.cond(c[1] >= k["minnhits"], finish, lambda c: c, c)
    keep = ok0 & (c[1] >= k["minnhits"])
    out = {"valid": keep, "nsh": c[1], "t0": c[2], "t0err": c[3], "pos": c[4], "members": c[0] & keep,
           "calo": jnp.where(keep, sd["calo"], -1).astype(I32)}
    if order:
        out["order"] = jnp.where(c[0] & keep, c[5], ORDER_NONE)
    return out


# --- public entry points --------------------------------------------------------------------------------------------

def _zsrc(ph):
    return jnp.asarray(ph["ncombo"])


def candidates(ph, s, cl=None):
    """Stage 1: per-event arrays `ev` (with `overflow` [E]), the per-slot seeds [E, K] (members, t0, t0err, calo, cog)
    and oks [E, K]. Calo clusters cl [E, C] (valid, disk, time, energy, cog_x, cog_y[, cog_z]) give the calo seeds
    (with UseCaloCluster); without them the slots are the NPEAK peak slots only."""
    with strict(_zsrc(ph)):
        hit = _hits(ph)
        if cl is not None and s.get("usecc", False):
            cc = _calo_in(cl, ph["x"].shape[0])
            return jax.vmap(lambda h, c: _peaks(h, c, s))(hit, cc)
        return jax.vmap(lambda h: _peaks(h, None, s))(hit)


def _all_slots(ev, sd, oks, s, net, cnet):
    def per_event(ev_e, sd_e, ok_e):
        return jax.vmap(lambda sd_, o: _cluster(ev_e, sd_, o, s, net, cnet))(sd_e, ok_e)
    out = jax.vmap(per_event)(ev, sd, oks)
    return {**out, "overflow": ev["overflow"]}


def time_clusters(ph, s, net, cl=None, cnet=None):
    """Panel hits [E, H] (hits.panel_hits; float columns hold float values) -> clusters per event [E, K]: valid, nsh,
    t0 and t0err (float64, TrkT0's doubles), pos [E, K, 3] (float32), members [E, K, H], calo (cluster index or -1),
    order (with s["order"]); overflow [E]. cnet: the ClusterCaloMVA net (needed with calo clusters)."""
    if cl is not None and s.get("usecc", False) and cnet is None:
        raise ValueError("calo-seeded time clusters need the ClusterCaloMVA net (cnet)")
    ev, sd, oks = candidates(ph, s, cl)
    with strict(_zsrc(ph)):
        return _all_slots(ev, sd, oks, s, net, cnet)


def _check_multi(ss):
    base = {k: v for k, v in ss[0].items() if k != "pitch"}
    if any({k: v for k, v in s_.items() if k != "pitch"} != base for s_ in ss):
        raise ValueError("settings differ in more than the pitch")


def time_clusters_multi(ph, ss, net, cl=None, cnet=None):
    """time_clusters for several producers whose settings differ only in AveragePitch (stock TrigTimeClusterFinder and
    TrigTimeClusterFinderUe), in one batched call. Returns one cluster dict per settings dict."""
    _check_multi(ss)
    with strict(_zsrc(ph)):
        den = jnp.asarray([tof_den(s_) for s_ in ss], F64)
    out = jax.vmap(lambda d: time_clusters(ph, {**ss[0], "tof_den": d}, net, cl, cnet))(den)
    return [{k: v[i] for k, v in out.items()} for i in range(len(ss))]


def time_clusters_multi_pairs(ph, ss, net, K, cl=None, cnet=None):
    """time_clusters_multi with the cluster refinement run only on the (producer, event, slot) pairs that have a
    candidate, gathered up to K pairs; when a batch has more than K candidates, the batched full path runs instead.
    Each candidate's cluster is the same computation as in time_clusters; slots without a candidate get valid False,
    nsh 0, t0 0, t0err 0, calo -1 and no members (every consumer reads clusters through `valid`)."""
    _check_multi(ss)
    if cl is not None and ss[0].get("usecc", False) and cnet is None:
        raise ValueError("calo-seeded time clusters need the ClusterCaloMVA net (cnet)")
    s0 = ss[0]
    with strict(_zsrc(ph)):
        den = jnp.asarray([tof_den(s_) for s_ in ss], F64)
    ev, sd, oks = jax.vmap(lambda d: candidates(ph, {**s0, "tof_den": d}, cl))(den)   # [P, E, ...]
    P, E, C = oks.shape
    with strict(_zsrc(ph)):
        def full():
            out = jax.vmap(lambda e_, s_, o_: _all_slots(e_, s_, o_, s0, net, cnet))(ev, sd, oks)
            return out

        def pairs():
            flat = jnp.nonzero(oks.reshape(-1), size=K, fill_value=P * E * C)[0]
            use = flat < P * E * C
            f = jnp.minimum(flat, P * E * C - 1)
            p, e, c = f // (E * C), (f // C) % E, f % C
            evp = {k: v[p, e] for k, v in ev.items()}
            sdp = {k: v[p, e, c] for k, v in sd.items()}
            r = _map_clusters(lambda ev_, sd_, o: _cluster(ev_, sd_, o, s0, net, cnet), evp, sdp, oks[p, e, c] & use)
            dst = jnp.where(use, flat, P * E * C)

            def put(v, fill):
                out = jnp.full((P * E * C + 1,) + v.shape[1:], fill, v.dtype).at[dst].set(v)
                return out[:P * E * C].reshape((P, E, C) + v.shape[1:])
            return {**_put_all(r, use, put), "overflow": ev["overflow"]}

        out = jax.lax.cond(jnp.sum(oks) <= K, pairs, full)
    return [{k: v[i] for k, v in out.items()} for i in range(P)]


def refine_pairs(ev, sd, oks, s, net, K, cnet=None):
    """Stage 2 of the compacted time clusters: _cluster on the first K candidate (event, slot) pairs of `oks` [E, K]
    only (K >= their count: the caller reads it), scattered back to [E, K]. Each cluster is the same computation as
    in time_clusters; slots without a candidate get valid False, nsh 0, t0 0, t0err 0, calo -1 and no members."""
    E, C = oks.shape
    if C > NPEAK and s.get("usecc", False) and cnet is None:
        raise ValueError("calo-seeded time clusters need the ClusterCaloMVA net (cnet)")
    with strict(ev["nsh"]):
        flat = jnp.nonzero(oks.reshape(-1), size=K, fill_value=E * C)[0]
        use = flat < E * C
        e, c = jnp.minimum(flat // C, E - 1), flat % C
        evp = {k: v[e] for k, v in ev.items()}
        sdp = {k: v[e, c] for k, v in sd.items()}
        r = _map_clusters(lambda ev_, sd_, o: _cluster(ev_, sd_, o, s, net, cnet), evp, sdp, oks[e, c] & use)
        dst = jnp.where(use, flat, E * C)

        def put(v, fill):
            out = jnp.full((E * C + 1,) + v.shape[1:], fill, v.dtype).at[dst].set(v)
            return out[:E * C].reshape((E, C) + v.shape[1:])
        return {**_put_all(r, use, put), "overflow": ev["overflow"]}


def _map_clusters(fn, evp, sdp, ok):
    """fn over the gathered clusters, LANES at a time (a vmapped while loop runs as long as its slowest lane)."""
    return jax.lax.map(lambda a: fn(*a), (evp, sdp, ok), batch_size=LANES if LANES > 1 else None)


def _put_all(r, use, put):
    """Scatter the gathered clusters' results back to the full slots (no candidate: valid False, nsh 0, t0 0,
    t0err 0, pos 0, calo -1, no members, ORDER_NONE)."""
    out = {"valid": put(r["valid"] & use, False), "nsh": put(r["nsh"], 0), "t0": put(r["t0"], 0),
           "t0err": put(r["t0err"], 0), "pos": put(r["pos"], 0), "calo": put(jnp.where(use, r["calo"], -1), -1),
           "members": put(r["members"] & use[:, None], False)}
    if "order" in r:
        out["order"] = put(jnp.where(use[:, None], r["order"], ORDER_NONE), ORDER_NONE)
    return out
