"""RobustHelixFinder (TrkPatRec, with TrkReco RobustHelixFit), Offline v13_41_01, on the device: the helix seeds of
the tpr paths (TrigHelixFinder, TrigHelixFinderUe), and MergeHelices (TrigHelixMergerDe / Ue).

A port of the stock module per time cluster, with the stock arithmetic, as the literal transcription
tests/robust_helix_reference.py writes it (median circle init and fit, no triple radius, no hit-error weights, the
circle fit's calo cluster off, initFZ, no hit MVA, no stereo update, one output collection):

- float32 where stock computes in float, float64 where a C++ double promotes (double literals, M_PI and
  CLHEP::twopi, std::pow(float, int), circleAzimuth's double z, the TH1F axis of extractFZ0, the time-of-flight
  correction), traced under `exact.strict` in both precisions of the chain; products that an add follows are guarded
  (`exact.m`), and XLA's division rewrites are blocked where they would change a rounding: x / sqrt(y), 1 / sqrt(y)
  and the quotients an add follows keep their rounding (`exact.g`), and float64 divisions by a scalar or a constant
  are made elementwise (`_div`: XLA compiles them as reciprocal products, off in about a third of the cases).
  RobustHelix's cos / sin of a float are float32 jnp.cos / jnp.sin (XLA's CPU float trig equals glibc's); the
  triplet weight cbrtf of a product of straw-hit counts is a table of the host libm's cbrtf;
- the hits of a time cluster are laid out as fillFaceOrderedHits leaves _chHitsToProcess: the cluster's hits with a
  HitSelectionBits bit and no HitBackgroundBits bit, in stored order, sorted by unique panel with libstdc++'s
  std::sort (`_psort`: mpr._stock_sort's introsort with the heap-sort fallback, then the final insertion sort; ties
  included). A face's range runs from its first to its last hit, so it also covers the other face's hits of the same
  plane: filterCircleHits and filterHits visit the two faces of a plane one after the other (the even face index
  first), which the port evaluates in two stages, all planes at once; the per-face chi2 sums run in face order;
- fitCircleMedian walks the triplets (i, j, k) in stock order, one i per loop step with all (j, k) at once, until
  ntripleMax + 1 are accepted; its weighted medians sort by value (stable for ties; the straw-hit-count weights of the
  radius and the z-phi medians are exact integer sums) and accumulate the float weights in that order;
- prefilterHits' boost P^2 medians are a scan over the hits; initFZ, extractFZ0 (the TH1F with ROOT 6.38's bin
  arithmetic) and fitFZ (its break leaves the inner pair loop) are pair matrices; updateT0 is boost's weighted mean
  and lazy weighted variance as float sums of double terms (the samples stay double): the calo cluster first
  (HelixSeed::caloCluster() is the time cluster's), then the hits in order;
- each time cluster fits both helicities at once (a vmap), then pickBestHelix keeps one: a time cluster gives at
  most one seed.

Time clusters are independent, so the finder runs on the valid clusters of all events, B lanes at a time. Fixed
capacities (bits of `overflow`: 1 a time cluster with more selected hits than NH (truncated), 2 more seeds in an
event than K).

`merge` is MergeHelices on these seeds: calhelix._merge_event's findchisq and HelixTool quantities, with the
survivors of compareHelices including UseCalo (a seed whose time cluster has a calo cluster beats an overlapping one
without: the tpr time clusters carry calo clusters).
"""
from __future__ import annotations

import ctypes
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .calhelix import _cint, _mask, _merge_event
from .exact import g as _g, m as _m, strict
from .mpr import SORT_THRESHOLD, _argmax, _d, _f, poly_atan2

F32, F64, I32, I64 = jnp.float32, jnp.float64, jnp.int32, jnp.int64
PI = math.pi
TWO_PI = 2. * math.pi                              # CLHEP::twopi
C_LIGHT = (2.99792458e+8 * 1000.0) / 1.e+9         # CLHEP::c_light, mm/ns
OUTLIER = 1 << 8                                   # StrawHitFlag::outlier
RESPHI = 1 << 18                                   # StrawHitFlag::resolvedphi
NFACES = 72                                        # StrawId::_ntotalfaces
TC_HITS = 96                                       # selected hits per time cluster (dumps: at most 80)
K_SEEDS = 16                                       # seed slots per event (at most one per time cluster)
LANES = 1                                          # time clusters per vmapped call (1 is fastest on CPU)
ORDER_NONE = np.int32(2 ** 30)                     # timecluster's order key of a non-member
DETECTOR_SYSTEM_Z0 = 10171.0                       # mu2e.detectorSystemZ0 (geom_run1_a.txt), mm
OVF_HITS, OVF_SLOTS = 1, 2
MAX_NSH = 8                                        # ComboHit::MaxNCombo: straw hits per panel hit
_INF = np.float32(np.inf)


def _host_cbrtf():
    """cbrtf(n) for the products n of three straw-hit counts (the platform libm's float cbrt)."""
    n = np.arange(MAX_NSH ** 3 + 1, dtype=np.float32)
    fn = None
    if sys.platform.startswith("linux"):
        try:
            fn = ctypes.CDLL("libm.so.6").cbrtf
            fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]
        except (OSError, AttributeError):
            fn = None
    if fn is None:
        return np.cbrt(n).astype(np.float32)
    return np.array([fn(float(v)) for v in n], np.float32)


# --- settings (host) --------------------------------------------------------------------------------------------------

def geometry():
    """The calorimeter disk origins in Mu2e (the disks are not rotated) and the tracker centre (-solenoidOffset, 0,
    detectorSystemZ0)."""
    from . import calogeom
    cg = calogeom.CALO_GEOM
    return {"disk_origin": np.asarray(calogeom.build()["origin"], np.float64),
            "tracker_center": np.array([-cg["solenoidOffset"], 0.0, DETECTOR_SYSTEM_Z0])}


def settings(m, geom=None):
    """RobustHelixFinder module table (and the geometry, `geometry()` by default) -> kernel constants (numpy
    scalars, static); raises for settings the port does not cover."""
    hf = m["HelixFitter"]
    bad = []
    for k, v in (("CircleInitType", 0), ("CircleFitType", 0), ("TripleRadius", False), ("HitErrorWeight", False),
                 ("UseCaloCluster", False), ("use_initFZ_from_dzFrequency", False)):
        if hf.get(k, v) != v:
            bad.append(k)
    for k, v in (("UseHitMVA", False), ("UpdateStereo", False), ("doSingleOutput", True)):
        if m.get(k, v) != v:
            bad.append(k)
    if list(m["SaveHelixFlag"]) != ["HelixOK"]:
        bad.append("SaveHelixFlag")
    if [int(h) for h in m["Helicities"]] != [-1, 1]:
        bad.append("Helicities " + str(m["Helicities"]))
    if bad:
        raise ValueError("device RobustHelixFinder does not support " + ", ".join(bad))
    t0c = m.get("T0Calculator", {})
    f = lambda d, k: np.float32(d[k])
    s = {"minnsh": int(m["MinNStrawHits"]), "maxphisep": f(m, "MaxPhiHitSeparation"), "maxdr": f(m, "MaxRadiusDiff"),
         "maxrpull": f(m, "MaxRPull"), "rpullscale": f(m, "RPullScaleF"), "maxniter": int(m["MaxIterations"]),
         "cradres": f(m, "CentralRadialResolution"), "cperpres": f(m, "CentralPerpResolution"),
         "maxdwire": f(m, "MaxWireDistance"), "maxdtrans": f(m, "MaxTransDistance"),
         "maxchisq": f(m, "MaxChisquared"), "maxrwdot": f(m, "MaxRWDot"), "prefilter": bool(m["PrefilterHits"]),
         "tcon_init": bool(m["targetconsistent_init"]), "tcon": bool(m["targetconsistent"]),
         "area_wt": bool(m["UseTripletArea"]), "hsel": _mask(m["HitSelectionBits"]),
         "hbkg": _mask(m["HitBackgroundBits"]), "maxedep": f(m, "maxEDepAvg"),
         "hwt": np.float32((1.0 / float(t0c.get("StrawHitTimeErr", 9.7))) ** 2),
         "beta": float(t0c.get("StrawHitBeta", 1.0)), "tot": bool(t0c.get("UseTOTDrift", True)),
         "avgdrift": float(t0c.get("AverageDriftTime", 22.5)),
         "cwt": np.float32((1.0 / float(t0c.get("CaloTimeErr", 0.5))) ** 2),
         "calo_toff": float(t0c.get("TrkToCaloTimeOffset", -0.4)),
         "calo_zoff": float(t0c.get("CaloShowermaxZ", -120.0)),
         "dontuse": _mask(hf["DontUseFlag"]), "minnhit": int(hf["minNHit"]), "nloops": int(hf["nLoopsdfdz"]),
         "nphibins": int(hf["NPhiHistBins"]), "phifactor": f(hf, "PhiHistRangeFactor"),
         "minnphi": int(hf["MinNPhi"]), "fmaxniter": int(hf["maxniter"]), "minzsep": f(hf, "minzsep"),
         "maxzsep": f(hf, "maxzsep"), "mindphi": f(hf, "mindphi"), "rmin": f(hf, "minR"), "rmax": f(hf, "maxR"),
         "rcmin": f(hf, "minCenterR"), "rcmax": f(hf, "maxCenterR"), "lmin": f(hf, "minAbsLambda"),
         "lmax": f(hf, "maxAbsLambda"), "targetradius": f(hf, "targetradius"),
         "trackerradius": f(hf, "trackerradius"), "ntripmin": int(hf["ntripleMin"]),
         "ntripmax": int(hf["ntripleMax"]), "iminl": f(hf, "initFZMinLambda"), "imaxl": f(hf, "initFZMaxLambda"),
         "istepl": f(hf, "initFZStepLambda"), "fminl": f(hf, "fitFZMinLambda"), "fmaxl": f(hf, "fitFZMaxLambda"),
         "fstepl": f(hf, "fitFZStepLambda")}
    geom = geometry() if geom is None else geom
    s["disk_oz"] = np.asarray(geom["disk_origin"], np.float64)[:, 2]
    s["tracker_z"] = float(geom["tracker_center"][2])
    md, xd, ma = f(hf, "mindist"), f(hf, "maxdist"), f(hf, "minArea")
    s.update(mind2=np.float32(md * md), maxd2=np.float32(xd * xd), minarea2=np.float32(ma * ma),
             maxdphi=np.float32(s["minzsep"] / s["iminl"]),
             inbins=int((s["imaxl"] - s["iminl"]) / s["istepl"] + np.float32(1)),
             fnbins=int((s["fmaxl"] - s["fminl"]) / s["fstepl"] + np.float32(1)),
             hxmin=-float(s["phifactor"]) * math.pi, hxmax=float(s["phifactor"]) * math.pi,
             cbrt=_host_cbrtf())
    return s


def merge_settings(merger):
    """MergeHelices module table -> the constants calhelix._merge_event reads."""
    if not merger.get("SelectBest", True):
        raise ValueError("device MergeHelices supports SelectBest true")
    return {"minnover": int(merger["MinNHitOverlap"]), "minfrac": np.float32(merger["MinHitOverlapFraction"]),
            "deltanh": int(merger["deltanh"]), "scalexy": np.float32(merger["scaleXY"]),
            "scalezphi": np.float32(merger["scaleZPhi"]), "bad": _mask(merger.get("BadHitFlags", ["Outlier"])),
            "usecalo": bool(merger.get("UseCalo", True))}


# --- small stock functions --------------------------------------------------------------------------------------------

def _w(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


def _div(a, b):
    """a / b, IEEE: XLA compiles a float64 division by a broadcast scalar (or a constant) as a product with the
    reciprocal, which differs from the quotient in about a third of the cases. The divisor is made per element by
    ORing its bits with a zero the compiler cannot see through (min(|bits(a) >> 1|, 0), elementwise)."""
    a = jnp.asarray(a)
    b = jnp.broadcast_to(jnp.asarray(b, a.dtype), a.shape)
    it = I64 if a.dtype == F64 else I32
    z = jnp.minimum(jnp.abs(jax.lax.bitcast_convert_type(a, it) >> 1), 0)
    return a / jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(b, it) | z, a.dtype)


def _angles_dphi(phi, ref):
    """Angles::deltaPhi(float&, float), in double."""
    d = _d(phi) - _d(ref)
    for _ in range(2):
        d = jnp.where(d > PI, d - TWO_PI, d)
    for _ in range(2):
        d = jnp.where(d <= -PI, d + TWO_PI, d)
    return _f(d)


def _fit_dphi(phi1, phi2):
    """RobustHelixFit::deltaPhi(float, float)."""
    dp = _f(jnp.fmod(_d(phi2 - phi1), TWO_PI))
    dp = jnp.where(_d(dp) > PI, _f(_d(dp) - TWO_PI), dp)
    return jnp.where(_d(dp) < -PI, _f(_d(dp) + TWO_PI), dp)


def _azimuth(st, z):
    """RobustHelix::circleAzimuth(double z) -> float."""
    lam = st["lam"]
    return jnp.where(lam != 0, _f(_d(st["fz0"]) + _g(_div(_d(z), _d(jnp.where(lam != 0, lam, np.float32(1)))))),
                     np.float32(0))


def _center(st):
    return _m(st["rcent"], jnp.cos(st["fcent"])), _m(st["rcent"], jnp.sin(st["fcent"]))


def _fsum(v, mask, init=None):
    """init + v[i] over the masked slots, one at a time in slot order (float32)."""
    L = mask.shape[0]
    idx = jnp.nonzero(mask, size=L, fill_value=0)[0]
    cnt = jnp.sum(mask).astype(I32)
    vv = v[idx]
    z = jnp.zeros(v.shape[1:], v.dtype) if init is None else init
    return jax.lax.fori_loop(0, cnt, lambda i, s: s + vv[i], z)


def _first_min(v, mask, start, last=False):
    """The index kept by `if (v < best)` (`<=` with last=True) from best = start in index order, along the last axis:
    (index or -1, its value)."""
    mk = mask & (v == v)
    vv = jnp.where(mk, v, jnp.asarray(np.inf, v.dtype))
    mn = jnp.min(vv, axis=-1, keepdims=True)
    hit = mk & (vv == mn) & ((mn <= start) if last else (mn < start))
    L = v.shape[-1]
    i = (L - 1 - _argmax(hit[..., ::-1], axis=-1)) if last else _argmax(hit, axis=-1)
    i = jnp.where(jnp.any(hit, axis=-1), i, -1)
    return i, mn[..., 0]


def _wmedian_f(val, w, n):
    """MedianCalculator::weightedMedian of the first n (val, w) in push order, float weights."""
    T = val.shape[0]
    ok = jnp.arange(T) < n
    tot = _fsum(w, ok)
    o = jnp.argsort(jnp.where(ok, val, _INF), stable=True)
    vs, ws = val[o], w[o]

    def cond(c):
        i, s = c
        return (_d(s) < _m(0.5, _d(tot))) & (i < n - 1)

    i, _ = jax.lax.while_loop(cond, lambda c: (c[0] + 1, c[1] + ws[c[0] + 1]), (I32(0), ws[0]))
    return jnp.where(n == 1, val[0], vs[i])


def _wmedian_i(val, w, mask):
    """MedianCalculator::weightedMedian with integer weights (straw-hit counts): exact sums."""
    o = jnp.argsort(jnp.where(mask, val, _INF), stable=True)
    ws = jnp.where(mask, w, 0)[o]
    tot = jnp.sum(ws)
    cum = jnp.cumsum(ws)
    n = jnp.sum(mask)
    i = _argmax(~(2 * cum < tot) & (jnp.arange(val.shape[0]) < n))
    return val[o][i]


# --- the hit sort: std::sort with the heap-sort fallback --------------------------------------------------------------

def _heap_sort(a, f, l, key, on):
    """std::__partial_sort(first, last, last) on the slots [f, l) of the arrangement a (operator< on key): __make_heap,
    then __sort_heap, through __adjust_heap and __push_heap; a no-op where `on` is false."""
    ln = jnp.where(on, l - f, 0).astype(I32)

    def adjust(a, hole, length, value):
        top = hole

        def c1(c):
            return c[2] < (length - 1) // 2

        def b1(c):
            a, hole, second = c
            second = 2 * (second + 1)
            second = jnp.where(key[a[f + second]] < key[a[f + second - 1]], second - 1, second)
            return a.at[f + hole].set(a[f + second]), second, second

        a, hole, second = jax.lax.while_loop(c1, b1, (a, hole, hole))
        ev = ((length & 1) == 0) & (second == (length - 2) // 2)
        s2 = 2 * (second + 1)
        a = jnp.where(ev, a.at[f + hole].set(a[jnp.maximum(f + s2 - 1, 0)]), a)
        hole = jnp.where(ev, s2 - 1, hole)
        par = lambda h: jnp.maximum(h - 1, 0) // 2

        def c2(c):
            a, hole = c
            return (hole > top) & (key[a[f + par(hole)]] < key[value])

        def b2(c):
            a, hole = c
            return a.at[f + hole].set(a[f + par(hole)]), par(hole)

        a, hole = jax.lax.while_loop(c2, b2, (a, hole))
        return a.at[f + hole].set(value)

    p0 = (ln - 2) // 2
    a = jax.lax.fori_loop(0, jnp.where(ln >= 2, p0 + 1, 0),
                          lambda t, a: adjust(a, p0 - t, ln, a[f + p0 - t]), a)

    def pop(t, a):
        last = l - 1 - t
        value = a[last]
        return adjust(a.at[last].set(a[f]), I32(0), last - f, value)

    return jax.lax.fori_loop(0, jnp.maximum(ln - 1, 0), pop, a)


def _psort(key, sel):
    """libstdc++ std::sort by `key` (operator<) of the selected slots, taken in slot order: introsort, heap sort past
    the depth limit, final insertion sort. Returns (perm, n, heap): heap is True when a range was heap sorted."""
    L = sel.shape[0]
    n = jnp.sum(sel).astype(I32)
    a = jnp.argsort(~sel, stable=True).astype(I32)
    S = 2 * max(L, 2).bit_length() + 4
    depth0 = jnp.where(n > 0, 2 * (31 - jax.lax.clz(jnp.maximum(n, 1))), 0)
    pos = jnp.arange(L, dtype=I32)

    def lt(i, j, a):
        return key[a[i]] < key[a[j]]

    def swap(a, i, j):
        ai, aj = a[i], a[j]
        return a.at[i].set(aj).at[j].set(ai)

    def partition(a, f, l, go):
        mid = f + (l - f) // 2
        x, y, z = f + 1, mid, l - 1
        sw = jnp.where(lt(x, y, a), jnp.where(lt(y, z, a), y, jnp.where(lt(x, z, a), z, x)),
                       jnp.where(lt(x, z, a), x, jnp.where(lt(y, z, a), z, y)))
        a = swap(a, f, sw)
        pk = key[a[f]]

        def body(st):
            a, first, last, _ = st
            ka = key[a]
            cand = (pos >= first) & ~(ka < pk)
            first = jnp.where(jnp.any(cand), _argmax(cand), L).astype(I32)
            last = (L - 1 - _argmax(((pos <= last - 1) & ~(pk < ka))[::-1])).astype(I32)
            done = ~(first < last)
            a = jnp.where(done, a, swap(a, first, last))
            return a, jnp.where(done, first, first + 1), last, done

        a, cut, _, _ = jax.lax.while_loop(lambda st: go & ~st[3], body, (a, f + 1, l, jnp.bool_(False)))
        return a, cut

    def body(st):
        a, sf, sl, sd, top, hs = st
        top = top - 1
        f, l, d = sf[top], sl[top], sd[top]
        big = (l - f) > SORT_THRESHOLD
        heap = big & (d == 0)
        a = _heap_sort(a, f, l, key, heap)
        go = big & (d > 0)
        a2, cut = partition(a, f, l, go)
        a = jnp.where(go, a2, a)
        sf = jnp.where(go, sf.at[top].set(cut).at[top + 1].set(f, mode="drop"), sf)
        sl = jnp.where(go, sl.at[top].set(l).at[top + 1].set(cut, mode="drop"), sl)
        sd = jnp.where(go, sd.at[top].set(d - 1).at[top + 1].set(d - 1, mode="drop"), sd)
        return a, sf, sl, sd, jnp.where(go, top + 2, top), hs | heap

    z = jnp.zeros(S, I32)
    st = (a, z, z.at[0].set(n), z.at[0].set(depth0), I32(1), jnp.bool_(False))
    a, _, _, _, _, hs = jax.lax.while_loop(lambda st: st[4] > 0, body, st)
    fin = jnp.argsort(jnp.where(pos < n, key[a], jnp.inf), stable=True)          # the final insertion sort
    return a[fin].astype(I32), n, hs


# --- prefilterHits: boost P^2 median ----------------------------------------------------------------------------------

def _psq_step(c, x, on):
    n, h, pos, des = c
    one = np.float32(1)
    inc = jnp.array([0., 0.25, 0.5, 0.75, 1.], F32)
    n1 = n + 1
    # the first five samples, sorted at the fifth
    hi = h.at[jnp.clip(n1 - 1, 0, 4)].set(x)
    hi = jnp.where(n1 == 5, jnp.sort(hi), hi)
    # later samples
    lo, hi4 = x < h[0], h[4] <= x
    k = jnp.where(lo, 1, jnp.where(hi4, 4, jnp.sum(~(x < h)).astype(I32)))
    hl = jnp.where(lo, h.at[0].set(x), jnp.where(hi4, h.at[4].set(x), h))
    pl = pos + jnp.where(jnp.arange(5) >= k, one, np.float32(0))
    dl = des + inc
    for i in (1, 2, 3):
        d = dl[i] - pl[i]
        dp = pl[i + 1] - pl[i]
        dm = pl[i - 1] - pl[i]
        hp = _g((hl[i + 1] - hl[i]) / dp)
        hm = _g((hl[i - 1] - hl[i]) / dm)
        cnd = ((d >= one) & (dp > one)) | ((d <= -one) & (dm < -one))
        sd = jnp.where(d > 0, one, -one)
        hh = hl[i] + _m(sd / (dp - dm), _m(sd - dm, hp) + _m(dp - sd, hm))
        nh = jnp.where((hl[i - 1] < hh) & (hh < hl[i + 1]), hh,
                       jnp.where(d > 0, hl[i] + hp, jnp.where(d < 0, hl[i] - hm, hl[i])))
        hl = jnp.where(cnd, hl.at[i].set(nh), hl)
        pl = jnp.where(cnd, pl.at[i].add(sd), pl)
    early = n1 <= 5
    h2 = jnp.where(early, hi, hl)
    p2 = jnp.where(early, pos, pl)
    d2 = jnp.where(early, des, dl)
    return (jnp.where(on, n1, n), jnp.where(on, h2, h), jnp.where(on, p2, pos), jnp.where(on, d2, des)), None


def _psq_median(v, on):
    c0 = (I32(0), jnp.zeros(5, F32), jnp.arange(1, 6, dtype=F32), jnp.array([1., 2., 3., 4., 5.], F32))
    (_, h, _, _), _ = jax.lax.scan(lambda c, a: _psq_step(c, a[0], a[1]), c0, (v, on))
    return h[2]


def _prefilter(h, flag, go, S):
    phi = poly_atan2(h["y"], h["x"])

    def body(c):
        flag, _, _, rem = c
        on = h["valid"] & ((flag & OUTLIER) == 0)
        nhit = jnp.sum(on).astype(I32)
        mphi = poly_atan2(_psq_median(h["y"], on), _psq_median(h["x"], on))
        dphi = jnp.abs(_angles_dphi(phi, mphi))
        mx = jnp.max(jnp.where(on, dphi, np.float32(0)))
        hit = on & (dphi == mx) & (mx > 0)
        worst = _argmax(hit)
        ch = mx > S["maxphisep"]
        flag = jnp.where(ch, flag.at[worst].set(flag[worst] | OUTLIER), flag)
        return flag, ch, nhit, rem + jnp.where(ch, h["nsh"][worst], 0)

    flag, _, _, rem = jax.lax.while_loop(lambda c: c[1] & (c[2] > 0), body, (flag, go, h["n"], I32(0)))
    return flag, rem


# --- fitCircleMedian --------------------------------------------------------------------------------------------------

def _circle_median(h, st, force, go, S):
    """RobustHelixFit::fitCircleMedian on the hits not flagged DontUse: (state, number of triplets)."""
    N = h["x"].shape[0]
    T = S["ntripmax"] + 1
    use = h["valid"] & ((st["flag"] & S["dontuse"]) == 0)
    X, Y, F = h["x"], h["y"], h["wface"]
    n = h["n"]
    p = jnp.arange(N, dtype=I32)
    D2, R2 = h["d2"], h["r2"]
    okd = ~(D2 < S["mind2"]) & ~(D2 > S["maxd2"])
    pj = use[:, None] & (p[:, None] < n - 1)
    jk = (p[None, :] > p[:, None]) & use[None, :] & (F[:, None] != F[None, :]) & (p[None, :] < n) & okd
    W = h["nsh"]
    cb = jnp.asarray(S["cbrt"])

    def body(c):
        i, cnt, bx, by, bw = c
        ri2 = R2[i]
        mi = use[i] & (i < n - 2)
        m = mi & (p[:, None] > i) & pj & (F[i] != F)[:, None] & okd[i][:, None] & jk & okd[i][None, :]
        dij, dik, djk = D2[i][:, None], D2[i][None, :], D2
        area2 = _f(_d(_m(dij, djk) + _m(dik, djk) + _m(dij, dik)) -
                   _m(0.5, _d(_m(dij, dij) + _m(djk, djk) + _m(dik, dik))))
        xi, yi = X[i], Y[i]
        xj, yj, xk, yk = X[:, None], Y[:, None], X[None, :], Y[None, :]
        rj2, rk2 = R2[:, None], R2[None, :]
        delta = _m(xk - xj, yj - yi) - _m(xj - xi, yk - yj)
        nx = _m(yk - yj, ri2) + _m(yi - yk, rj2) + _m(yj - yi, rk2)
        ny = _m(xk - xj, ri2) + _m(xi - xk, rj2) + _m(xj - xi, rk2)
        cx = _f(_g(_m(0.5, _d(nx))) / _d(delta))
        cy = _f(_g(_m(-0.5, _d(ny))) / _d(delta))
        ex, ey = xi - cx, yi - cy
        rho = jnp.sqrt(_m(ex, ex) + _m(ey, ey))
        rc = jnp.sqrt(_m(cx, cx) + _m(cy, cy))
        acc = m & ~(area2 < S["minarea2"]) & (rc > S["rcmin"]) & (rc < S["rcmax"]) & (rho > S["rmin"]) & \
            (rho < S["rmax"]) & (rc + rho < S["trackerradius"])
        if force:
            acc &= jnp.abs(rc - rho) < S["targetradius"]
        wt = area2 if S["area_wt"] else cb[jnp.clip(W[i] * W[:, None] * W[None, :], 0, MAX_NSH ** 3)]
        a = acc.reshape(-1)
        rank = cnt + jnp.cumsum(a) - 1
        dst = jnp.where(a & (rank < T), rank, T)
        bx = bx.at[dst].set(cx.reshape(-1), mode="drop")
        by = by.at[dst].set(cy.reshape(-1), mode="drop")
        bw = bw.at[dst].set(jnp.broadcast_to(wt, acc.shape).reshape(-1), mode="drop")
        return i + 1, cnt + jnp.sum(a).astype(I32), bx, by, bw

    z = jnp.zeros(T, F32)
    _, cnt, bx, by, bw = jax.lax.while_loop(lambda c: go & (c[0] < n - 2) & (c[1] < T), body, (I32(0), I32(0), z, z,
                                                                                                z))
    nt = jnp.minimum(cnt, T)
    run = go & (nt > S["ntripmin"])
    ntm = jnp.maximum(nt, 1)
    centx = _wmedian_f(bx, bw, ntm)
    centy = _wmedian_f(by, bw, ntm)
    dx, dy = _d(X - centx), _d(Y - centy)
    rho = jnp.sqrt(_f(_m(dx, dx) + _m(dy, dy)))
    run &= jnp.any(use)
    rad = _wmedian_i(rho, W, use)
    ok = run & (rad > S["rmin"]) & (rad < S["rmax"])
    new = {**st, "rcent": jnp.sqrt(_m(centx, centx) + _m(centy, centy)), "fcent": poly_atan2(centy, centx),
           "radius": rad, "cok": ok}
    return _w(run, new, {**st, "cok": jnp.bool_(False)}), nt


def _fit_circle(h, st, force, go, S):
    """RobustHelixFit::fitCircle (median init and fit: an initialisation is the fit)."""
    st, nt = _circle_median(h, st, force, go, S)
    return {**st, "cinit": st["cinit"] | st["cok"]}, nt


# --- initFZ / fitFZ ---------------------------------------------------------------------------------------------------

def _good_lambda(lam, hel, S):
    return jnp.where(hel == -1, (lam > -S["lmax"]) & (lam < -S["lmin"]),
                     jnp.where(hel == 1, (lam > S["lmin"]) & (lam < S["lmax"]), False))


def _good_fz(st, S):
    lam = st["lam"]
    hl = jnp.where(lam != 0, jnp.where(jnp.signbit(lam), -1, 1), 0)
    return (hl == st["hel"]) & (jnp.abs(lam) > S["lmin"]) & (jnp.abs(lam) < S["lmax"])


def _good_helix(st, S):
    return (st["radius"] > S["rmin"]) & (st["radius"] < S["rmax"]) & _good_fz(st, S)


def _resolve(h, st):
    """resolvePhi on every hit."""
    phiex = _azimuth(st, h["z"])
    nl = _cint(jnp.rint(_div(_d(phiex - st["hphi"]), TWO_PI)))
    hp = _f(_d(st["hphi"]) + _m(_d(nl), TWO_PI))
    v = h["valid"]
    return {**st, "hphi": jnp.where(v, hp, st["hphi"]), "flag": jnp.where(v, st["flag"] | RESPHI, st["flag"])}, \
        jnp.any(v & (nl != 0))


def _find_bin(x, S):
    n = S["nphibins"]
    xmin, xmax = S["hxmin"], S["hxmax"]
    width = (xmax - xmin) / n
    a = _cint(_div(x - xmin, width))
    ad = _d(a)
    b = 1 + a - (x < xmin + _m(width, ad)).astype(I64) + (xmin + _m(width, ad + 1.0) <= x).astype(I64)
    return jnp.where(x < xmin, 0, jnp.where(~(x < xmax), n + 1, b))


def _extract_fz0(h, st, use, S):
    n = S["nphibins"]
    dphi = _d(_fit_dphi(_azimuth(st, h["z"]), st["hphi"]))
    bins = jnp.stack([_find_bin(dphi, S), _find_bin(dphi - TWO_PI, S), _find_bin(dphi + TWO_PI, S)])
    c = jnp.zeros(n + 2, I32).at[jnp.where(use[None, :], bins, n + 2)].add(1, mode="drop")
    imax = 1 + _argmax(c[1:n + 1] == jnp.max(c[1:n + 1]))
    width = (S["hxmax"] - S["hxmin"]) / float(n)
    cnt, fz0 = I32(0), np.float32(0)
    for t in range(3):
        b = jnp.maximum(0, imax - 1) + t
        on = b <= jnp.minimum(imax + 1, n)
        cb = c[jnp.minimum(b, n + 1)]
        cen = S["hxmin"] + _m(_d(b) - 0.5, width)
        cnt = jnp.where(on, cnt + cb, cnt)
        fz0 = jnp.where(on, _f(_d(fz0) + _m(_d(cb), cen)), fz0)
    return cnt >= S["minnphi"], fz0 / cnt.astype(F32)


def _init_fz(h, st, S):
    """initFZ: (state, ok)."""
    N = h["x"].shape[0]
    p = jnp.arange(N)
    st = {**st, "lam": np.float32(1.0e12), "fz0": np.float32(0)}
    use = h["valid"] & ((st["flag"] & S["dontuse"]) == 0)
    cx, cy = _center(st)
    hp = jnp.where(use, poly_atan2(h["y"] - cy, h["x"] - cx), st["hphi"])
    st = {**st, "hphi": hp}
    hel = st["hel"]
    sign = jnp.where(hel == -1, np.float32(-1), jnp.where(hel == 1, np.float32(1), np.float32(0)))
    pr = use[:, None] & use[None, :] & (p[None, :] > p[:, None]) & (h["uface"][:, None] != h["uface"][None, :])
    dz = h["z"][None, :] - h["z"][:, None]
    pr &= ~(jnp.abs(dz) < S["minzsep"]) & ~(jnp.abs(dz) > S["maxzsep"])
    dphi = _d(_fit_dphi(hp[:, None], hp[None, :]))
    hist = jnp.zeros(S["inbins"], I32)
    blast = jnp.full(dphi.shape, -1, I64)
    alive = pr
    counter = I32(0)
    for lp in range(S["nloops"]):
        dn = dphi + _m(float(lp), 2.0, PI, _d(sign))
        cont = (dn * _d(hel) < 0) | (jnp.abs(dn) < float(S["mindphi"])) | (jnp.abs(dn) > float(S["maxdphi"]))
        lam = _f(_d(dz) / jnp.where(dn != 0, dn, 1.0))
        ls = _m(lam, sign)
        c2 = ls >= S["imaxl"]
        brk = ~c2 & (ls <= S["iminl"])
        bn = _cint((ls - S["iminl"]) / S["istepl"])
        skip = (blast > 0) & (blast - bn <= 2)
        fill = alive & ~cont & ~c2 & ~brk & ~skip
        hist = hist.at[jnp.where(fill, bn, S["inbins"])].add(1, mode="drop")
        counter = counter + jnp.sum(fill).astype(I32)
        blast = jnp.where(fill, bn, blast)
        alive = alive & ~(~cont & brk)
    ok = ~(counter < S["minnhit"])
    sw = (hist[:-2] + hist[1:-1] + hist[2:]).astype(F32)
    mx = jnp.max(sw)
    ix = _argmax(sw == mx)
    st_d = float(S["istepl"])
    ixd = _d(ix)
    num = _m(st_d, ixd + 0.5, _d(hist[ix])) + _m(st_d, ixd + 1.0 + 0.5, _d(hist[ix + 1])) + \
        _m(st_d, ixd + 2.0 + 0.5, _d(hist[ix + 2]))
    xmp = jnp.where(mx > 0, _f(_d(S["iminl"]) + _g(num / _d(sw[ix]))), np.float32(0))
    lam = _m(xmp, sign)
    ok &= _good_lambda(lam, hel, S)
    st2 = {**st, "lam": lam}
    eok, fz0 = _extract_fz0(h, st2, use, S)
    st3, _ = _resolve(h, {**st2, "fz0": _fit_dphi(np.float32(0), fz0)})
    res = _w(eok, st3, st2)
    return _w(ok, res, st), ok & eok


def _fit_fz(h, st, go, S):
    """RobustHelixFit::fitFZ."""
    N = h["x"].shape[0]
    p = jnp.arange(N)
    st = {**st, "pok": jnp.bool_(False)}
    need = go & ~st["pinit"]
    sti, iok = _init_fz(h, st, S)
    st = _w(need, {**sti, "pinit": iok}, st)
    run = go & st["pinit"]
    hel = st["hel"]
    sign = jnp.where(hel == -1, np.float32(-1), jnp.where(hel == 1, np.float32(1), np.float32(0)))
    upper = p[None, :] > p[:, None]
    face = h["uface"][:, None] != h["uface"][None, :]
    dz = h["z"][None, :] - h["z"][:, None]
    st_d = float(S["fstepl"])
    nb = S["fnbins"]

    def body(c):
        st, _, it, _ = c
        use = h["valid"] & ((st["flag"] & S["dontuse"]) == 0)
        hp = st["hphi"]
        ok = use[:, None] & use[None, :] & upper & face
        dphi = hp[None, :] - hp[:, None]
        ok &= ~(jnp.abs(dphi) < S["mindphi"])
        lam = dz / jnp.where(dphi != 0, dphi, np.float32(1))
        good = ok & _good_lambda(lam, hel, S)
        ls = _m(lam, sign)
        c2 = ls >= S["fmaxl"]
        brk = good & ~c2 & (ls <= S["fminl"])
        fb = jnp.where(jnp.any(brk, axis=1), _argmax(brk, axis=1), N)
        fill = good & ~c2 & ~brk & (p[None, :] < fb[:, None])
        bn = _cint((ls - S["fminl"]) / S["fstepl"])
        hist = jnp.zeros(nb, I32).at[jnp.where(fill, bn, nb)].add(1, mode="drop")
        cs = jnp.concatenate([jnp.zeros(1, I32), jnp.cumsum(hist)])
        sw = (cs[10:nb] - cs[:nb - 10]).astype(F32)
        mx = jnp.max(sw)
        ix = _argmax(sw == mx)
        xmp = np.float32(0)
        for l in range(10):
            xmp = _f(_d(xmp) + _m(st_d, _d(ix) + float(l) + 0.5, _d(hist[ix + l])))
        xmp = _g(xmp / sw[ix]) + S["fminl"]
        lam_new = jnp.where(mx > 0, _m(xmp, sign), _m(np.float32(0), sign))
        s1 = {**st, "lam": lam_new}
        any_use = jnp.any(use)
        dph = _fit_dphi(_azimuth(s1, h["z"]), hp)
        med = _wmedian_i(dph, h["nsh"], use)
        s2, ch = _resolve(h, {**s1, "fz0": _fit_dphi(np.float32(0), s1["fz0"] + med)})
        return _w(any_use, s2, s1), ch & any_use, it + 1, ~any_use

    st, _, it, ret = jax.lax.while_loop(lambda c: c[1] & (c[2] < S["fmaxniter"]), body,
                                        (st, run, I32(0), jnp.bool_(False)))
    return {**st, "pok": run & ~ret & _good_fz(st, S)}


# --- the module's hit filters -----------------------------------------------------------------------------------------

def _geom(h, st):
    cx, cy = _center(st)
    dx, dy = h["x"] - cx, h["y"] - cy
    m2 = _m(dx, dx) + _m(dy, dy)
    r = _g(jnp.sqrt(m2))
    inv = np.float32(1) / jnp.where(r == 0, np.float32(1), r)
    cdx = jnp.where(r == 0, dx, _m(dx, inv))
    cdy = jnp.where(r == 0, dy, _m(dy, inv))
    return cx, cy, dx, dy, r, cdx, cdy


def _span(h):
    """[72, N] membership of the face ranges [idChBegin, idChEnd)."""
    p = jnp.arange(h["x"].shape[0])
    return (h["fb"][:, None] >= 0) & (p[None, :] >= h["fb"][:, None]) & (p[None, :] < h["fe"][:, None])


def _filter_circle(h, st, S):
    """filterCircleHits: (state, changed count)."""
    N = h["x"].shape[0]
    p = jnp.arange(N)
    _, _, dx, dy, r, cdx, cdy = _geom(h, st)
    ux, uy = h["udx"], h["udy"]
    rwdot = _m(ux, cdx) + _m(uy, cdy)
    dr = r - st["radius"]
    rw2 = _m(rwdot, rwdot)
    werr, terr = jnp.sqrt(h["uvar"]), jnp.sqrt(h["vvar"])
    rres = _g(jnp.sqrt(_f(_d(_m(werr, werr, rw2)) + _m(_d(_m(terr, terr)), 1.0 - _d(rw2)))))
    rpull = _m(jnp.abs(dr / rres), S["rpullscale"])
    cand = ~(rwdot > S["maxrwdot"]) & ~(jnp.abs(dr) > S["maxdr"]) & ~(rpull > S["maxrpull"])
    sp = _span(h)
    best, mn = _first_min(rpull[None, :], sp & cand[None, :], S["maxrpull"])            # [72]
    hasb = best >= 0
    bs = jnp.maximum(best, 0)
    out0 = (st["flag"] & OUTLIER) != 0
    # the last face (in index order) whose range covers a hit decides its flag
    fidx = jnp.arange(NFACES)
    lastf = jnp.max(jnp.where(sp, fidx[:, None], -1), axis=0)
    keep = (lastf >= 0) & (best[jnp.maximum(lastf, 0)] == p)
    covered = lastf >= 0
    out1 = jnp.where(covered, ~keep, out0)
    # oldout of a face's best: after the plane's even face when that covers it, else the incoming flag
    ev = fidx % 2 == 1
    part = fidx - 1
    in_prev = ev & sp[jnp.maximum(part, 0), bs]
    old = jnp.where(in_prev, best[jnp.maximum(part, 0)] != bs, out0[bs])
    changed = jnp.sum(hasb & old).astype(I32)
    ngood = jnp.sum(jnp.where(hasb, h["nsh"][bs], 0)).astype(I32)
    chi2 = _fsum(_m(mn, mn)[:, None], hasb)[0]
    cnt = jnp.sum(hasb).astype(I32)
    cntf = jnp.where(cnt > 0, cnt.astype(F32), np.float32(1e-10))
    flag = jnp.where(out1, st["flag"] | OUTLIER, st["flag"] & ~OUTLIER)
    return {**st, "flag": flag, "nxysh": ngood, "chi2xy": chi2 / cntf}, changed


def _filter_hits(h, st, S):
    """filterHits: (state, changed)."""
    N = h["x"].shape[0]
    p = jnp.arange(N)
    cx, cy, dx, dy, r, cdx, cdy = _geom(h, st)
    ux, uy = h["udx"], h["udy"]
    dphi = jnp.abs(_angles_dphi(h["phi"], st["fcent"]))
    az = _azimuth(st, h["z"])
    hx = cx + _m(st["radius"], jnp.cos(az))
    hy = cy + _m(st["radius"], jnp.sin(az))
    dhx, dhy = h["x"] - hx, h["y"] - hy
    dtrans = jnp.abs(_m(dhx, -uy) + _m(dhy, ux))
    dwire = jnp.abs(_m(dhx, ux) + _m(dhy, uy))
    sq = lambda v: _m(_d(v), _d(v))
    cw = _m(cdx, ux) + _m(cdy, uy)
    pw = _m(-cdy, ux) + _m(cdx, uy)
    ct = _m(cdx, -uy) + _m(cdy, ux)
    pt = _m(-cdy, -uy) + _m(cdx, ux)
    wres2 = _f(sq(jnp.sqrt(h["uvar"])) + sq(_m(S["cradres"], cw)) + sq(_m(S["cperpres"], pw)))
    wtres2 = _f(sq(jnp.sqrt(h["vvar"])) + sq(_m(S["cradres"], ct)) + sq(_m(S["cperpres"], pt)))
    chisq = _g(_m(dwire, dwire) / wres2) + _g(_m(dtrans, dtrans) / wtres2)
    fail = (dphi > S["maxphisep"]) | (dwire > S["maxdwire"]) | (dtrans > S["maxdtrans"]) | (chisq > S["maxchisq"])
    sp = _span(h)
    out = (st["flag"] & OUTLIER) != 0
    changed = jnp.bool_(False)
    bests = jnp.full(NFACES, -1, I32)
    mins = jnp.zeros(NFACES, F32)
    for par in (0, 1):                                  # the even faces of all planes, then the odd ones
        fsel = (jnp.arange(NFACES) % 2) == par
        vis = sp & fsel[:, None] & ~out[None, :]
        b, mn = _first_min(chisq[None, :], vis, S["maxchisq"], last=True)
        changed = changed | jnp.any(vis & fail[None, :])
        bests = jnp.where(fsel, b, bests)
        mins = jnp.where(fsel, mn, mins)
        visited = jnp.any(vis, axis=0)
        isbest = jnp.any(vis & (b[:, None] == p[None, :]), axis=0)
        out = jnp.where(visited, ~isbest, out)
    hasb = bests >= 0
    bs = jnp.maximum(bests, 0)
    ngood = jnp.sum(jnp.where(hasb, h["nsh"][bs], 0)).astype(I32)
    chi2 = _fsum(mins[:, None], hasb)[0]
    cnt = jnp.sum(hasb).astype(I32)
    cntf = jnp.where(cnt > 0, cnt.astype(F32), np.float32(1e-10))
    flag = jnp.where(out, st["flag"] | OUTLIER, st["flag"] & ~OUTLIER)
    return {**st, "flag": flag, "nzphish": ngood, "chi2zphi": chi2 / cntf}, changed


def _update_t0(h, st, S):
    r, lam = st["radius"], st["lam"]
    pitch = _f(_g(_m(jnp.where(jnp.signbit(r), -1.0, 1.0), _d(lam))) / _d(_g(jnp.sqrt(_m(r, r) + _m(lam, lam)))))
    tflt = _g(_div(_d(h["z"]), _g(_m(_d(pitch), S["beta"], C_LIGHT))))
    t = _d(h["time"]) - tflt
    if not S["tot"]:
        t = t - S["avgdrift"]
    on = h["valid"] & ((st["flag"] & OUTLIER) == 0)
    hw, cw = S["hwt"], S["cwt"]
    # the calo cluster's sample first; boost keeps the samples double: sample * weight is a double term
    xc = (h["ct"] - _g((h["cz"] + S["calo_zoff"]) / _g(_m(_d(pitch), S["beta"], C_LIGHT)))) + S["calo_toff"]
    z3 = jnp.zeros(3, F32)

    def add(c, x, w):
        sw, ws, ws2 = c[0], c[1], c[2]
        return jnp.stack([sw + w, _f(_d(ws) + _m(x, _d(w))), _f(_d(ws2) + _m(_d(w), _m(x, x)))])

    c0 = jnp.where(h["hascalo"], add(z3, xc, cw), z3)
    N = t.shape[0]
    idx = jnp.nonzero(on, size=N, fill_value=0)[0]
    tt = t[idx]
    sw, ws, ws2 = jax.lax.fori_loop(0, jnp.sum(on).astype(I32), lambda i, c: add(c, tt[i], hw), c0)
    cnt = (jnp.sum(on) + h["hascalo"].astype(I32)).astype(F32)
    mean = ws / sw
    var = _g(ws2 / sw) - _m(mean, mean)
    err = jnp.sqrt(jnp.maximum(np.float32(0), var) / cnt)
    upd = sw > 0
    return {**st, "t0": jnp.where(upd, _d(mean), st["t0"]), "t0err": jnp.where(upd, _d(err), st["t0err"])}


def _fit_helix(h, st, go, S):
    """RobustHelixFinder::fitHelix for one helicity."""
    mx = S["maxniter"]

    def xy_body(c):
        st, nxy, _, first = c
        st, _ = _fit_circle(h, st, S["tcon"], jnp.bool_(True), S)
        st, ch = _filter_circle(h, st, S)
        return st, nxy + 1, ch > 0, jnp.bool_(False)

    def fz_body(c):
        st, nfz, _, first = c
        st = _fit_fz(h, st, jnp.bool_(True), S)
        st, ch = _filter_hits(h, st, S)
        return st, nfz + 1, ch, jnp.bool_(False)

    def outer(c):
        st, niter, _, fzch, first = c
        st, _, xych, _ = jax.lax.while_loop(lambda c: c[3] | (c[0]["cok"] & (c[1] < mx) & c[2]), xy_body,
                                            (st, I32(0), jnp.bool_(True), jnp.bool_(True)))
        low = st["nxysh"] < S["minnsh"]
        st = {**st, "cok": st["cok"] & ~low}
        niter = jnp.where(low, mx, niter)
        doz = st["cok"]
        stz, _, fzc, _ = jax.lax.while_loop(lambda c: c[3] | (c[0]["pok"] & (c[1] < mx) & c[2]), fz_body,
                                            (st, I32(0), jnp.bool_(False), doz))
        lowz = doz & (stz["nzphish"] < S["minnsh"])
        stz = {**stz, "pok": stz["pok"] & ~lowz}
        st = _w(doz, stz, st)
        fzch = jnp.where(doz, fzc, fzch)
        niter = jnp.where(lowz, mx, niter) + 1
        return st, niter, fzch | xych, fzch, jnp.bool_(False)

    st, _, _, _, _ = jax.lax.while_loop(
        lambda c: c[4] | (_good_helix(c[0], S) & (c[1] < mx) & c[2]), outer,
        (st, I32(0), jnp.bool_(True), jnp.bool_(True), go))
    hok = go & _good_helix(st, S) & st["cok"] & st["pok"]
    st = _w(hok, _update_t0(h, st, S), st)
    return {**st, "hok": hok}


# --- one time cluster -------------------------------------------------------------------------------------------------

def _one_tc(h, S):
    N = h["x"].shape[0]
    flag = h["flag"]
    go = h["go"] & ~(h["nsh_tot"] < S["minnsh"])
    rem = I32(0)
    if S["prefilter"]:
        flag, rem = _prefilter(h, flag, go, S)
    go &= ~(h["nsh_tot"] - rem < S["minnsh"])
    st = {"flag": flag, "hphi": jnp.zeros(N, F32), "rcent": np.float32(-1), "fcent": np.float32(0),
          "radius": np.float32(-1), "lam": np.float32(0), "fz0": np.float32(0), "chi2xy": np.float32(0),
          "chi2zphi": np.float32(0), "cinit": jnp.bool_(False), "cok": jnp.bool_(False),
          "pinit": jnp.bool_(False), "pok": jnp.bool_(False), "hok": jnp.bool_(False), "nxysh": I32(0),
          "nzphish": I32(0), "t0": _d(h["t0"]), "t0err": _d(h["t0err"]), "hel": I32(0)}
    st, nt0 = _fit_circle(h, st, S["tcon_init"], go, S)
    go2 = go & st["cok"]
    sts = jax.tree.map(lambda a: jnp.stack([a, a]), st)
    sts = {**sts, "hel": jnp.array([-1, 1], I32)}
    r = jax.vmap(lambda s_: _fit_helix(h, s_, go2, S))(sts)
    mask = h["valid"][None, :] & ((r["flag"] & OUTLIER) == 0)
    ed = jax.vmap(lambda mm: _fsum(_m(h["edep"], h["nsh"].astype(F32))[:, None], mm)[0])(mask)
    nshk = jnp.sum(jnp.where(mask, h["nsh"][None, :], 0), axis=1)
    edep = _f(_d(ed) / (_d(nshk) + 1e-10))
    pas = r["hok"] & ~(edep > S["maxedep"])
    nh = jnp.sum(mask, axis=1)
    both = pas[0] & pas[1]
    pick = jnp.where(both, jnp.where(nh[1] > nh[0], 1, jnp.where(nh[0] > nh[1], 0,
                                                                 jnp.where(r["chi2zphi"][0] < r["chi2zphi"][1], 0, 1))),
                     jnp.where(pas[0], 0, 1))
    sel = jax.tree.map(lambda a: a[pick], {**r, "mask": mask, "edep_avg": edep})
    sel["ok"] = go2 & jnp.any(pas)
    sel["dropped"] = jnp.sum(r["hok"] & ~pas).astype(I32)
    sel["ntrip0"] = nt0
    return sel


def _fill(hev, key, t0, t0err, calo, cev, S, NH):
    """fillFaceOrderedHits of one time cluster: the slot arrays [NH]."""
    H = key.shape[0]
    o0 = jnp.argsort(key, stable=True).astype(I32)                       # the cluster's hits in stored order
    mem = key[o0] < ORDER_NONE
    fl = jnp.asarray(hev["flag"]).astype(I64)[o0]
    sel = mem & ((fl & S["hsel"]) != 0) & ((fl & S["hbkg"]) == 0)
    sid = jnp.asarray(hev["sid"]).astype(I32)[o0]
    up = (((sid & 0xFC00) >> 10) * 6 + ((sid & 0x380) >> 7)).astype(F32)
    perm, nm, heap = _psort(up, sel)
    hi = o0[perm]
    if H < NH:                                                           # fewer hit slots than NH: pad (masked below)
        hi = jnp.concatenate([hi, jnp.zeros(NH - H, I32)])
    hi = hi[:NH]
    n = jnp.minimum(nm, NH)
    valid = jnp.arange(NH) < n
    g = lambda k: jnp.where(valid, _f(hev[k][hi]), np.float32(0))
    h = {k: g(k) for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar", "edep", "time")}
    h["nsh"] = jnp.where(valid, jnp.asarray(hev["ncombo"])[hi].astype(I32), 0)
    sd = jnp.where(valid, jnp.asarray(hev["sid"])[hi].astype(I32), 0)
    h["flag"] = jnp.where(valid, jnp.asarray(hev["flag"])[hi].astype(I64) & ~RESPHI, 0)
    h["idx"] = jnp.where(valid, hi, -1).astype(I32)
    h["sid"] = sd
    station, plane, panel = (sd & 0xF800) >> 11, ((sd & 0xFC00) >> 10) % 2, (sd & 0x380) >> 7
    xf = panel % 2
    of = jnp.where(station % 2 == 0, jnp.where(plane == 0, 1 - xf, xf + 2), jnp.where(plane == 0, xf, 3 - xf))
    h["wface"] = jnp.where(valid, of, -1 - jnp.arange(NH)).astype(I32)
    h["uface"] = jnp.where(valid, ((sd & 0xFC00) >> 10) * 2 + ((sd & 0x80) >> 7), -1 - jnp.arange(NH)).astype(I32)
    face = jnp.where(valid, of + station * 4, NFACES)
    p = jnp.arange(NH, dtype=I32)
    h["fb"] = jnp.full(NFACES, NH, I32).at[face].min(p, mode="drop")
    h["fe"] = jnp.zeros(NFACES, I32).at[face].max(p + 1, mode="drop")
    h["fb"] = jnp.where(h["fb"] < NH, h["fb"], -1)
    h["valid"] = valid
    h["n"] = n.astype(I32)
    h["nsh_tot"] = jnp.sum(jnp.where(mem & sel, jnp.asarray(hev["ncombo"])[o0], 0)).astype(I32)
    h["t0"], h["t0err"] = t0, t0err
    c = jnp.maximum(calo, 0)
    h["hascalo"] = calo >= 0
    h["ct"] = _d(_f(cev["time"][c]))
    h["cz"] = (jnp.asarray(S["disk_oz"])[cev["disk"][c]] + 0.0) - S["tracker_z"]      # diskToMu2e, mu2eToTracker
    # per-TC constants
    X, Y = h["x"], h["y"]
    dX, dY = X[:, None] - X[None, :], Y[:, None] - Y[None, :]
    h["d2"] = _m(dX, dX) + _m(dY, dY)
    h["r2"] = _m(X, X) + _m(Y, Y)
    h["phi"] = poly_atan2(Y, X)
    return h, nm > NH, heap


def _helices(ph, tc, cl, S, K, NH, B):
    E, H = ph["x"].shape
    KT = tc["valid"].shape[1]
    hev = {k: ph[k] for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar", "edep", "time", "ncombo", "flag", "sid")}
    cev = {"disk": jnp.asarray(cl["disk"]).astype(I32), "time": _f(cl["time"])}
    calo = jnp.where(jnp.asarray(tc["valid"]), jnp.asarray(tc["calo"]).astype(I32), -1)
    fill = jax.vmap(lambda he, kk, a, b, cc, ce: jax.vmap(lambda k1, a1, b1, c1: _fill(he, k1, a1, b1, c1, ce, S, NH))(
        kk, a, b, cc))
    hs, ovh, heap = fill(hev, jnp.asarray(tc["order"]).astype(I32), _d(tc["t0"]), _d(tc["t0err"]), calo, cev)
    hs = {**hs, "go": jnp.asarray(tc["valid"])}
    NI = E * KT
    flat = {k: v.reshape((NI,) + v.shape[2:]) for k, v in hs.items()}
    gof = flat["go"]
    order = jnp.concatenate([jnp.nonzero(gof, size=NI, fill_value=NI)[0].astype(I32), jnp.full(B, NI, I32)])
    nchunk = ((jnp.sum(gof) + B - 1) // B).astype(I32)
    run = jax.vmap(lambda h: _one_tc(h, S))
    spec = jax.eval_shape(run, jax.tree.map(lambda a: a[:B], flat))
    buf0 = jax.tree.map(lambda a: jnp.zeros((NI + 1,) + a.shape[1:], a.dtype), spec)

    def chunk(i, buf):
        sel = jax.lax.dynamic_slice(order, (i * B,), (B,))
        ok = sel < NI
        si = jnp.minimum(sel, NI - 1)
        r = run({**jax.tree.map(lambda a: a[si], flat), "go": flat["go"][si] & ok})
        dst = jnp.where(ok, sel, NI)
        return jax.tree.map(lambda b, a: b.at[dst].set(a), buf, r)

    buf = jax.lax.fori_loop(0, nchunk, chunk, buf0)
    r = jax.tree.map(lambda a: a[:NI].reshape((E, KT) + a.shape[1:]), buf)
    cand = r["ok"] & jnp.asarray(tc["valid"])                                     # [E, KT]
    pos = jnp.cumsum(cand, axis=1) - 1
    nseed = jnp.sum(cand, axis=1).astype(I32)
    rows = jnp.arange(E)[:, None]
    dst = jnp.where(cand & (pos < K), pos, K)
    src = jnp.zeros((E, K + 1), I32).at[rows, dst].set(jnp.arange(KT, dtype=I32)[None, :], mode="drop")[:, :K]
    ok = jnp.arange(K)[None, :] < jnp.minimum(nseed, K)[:, None]
    tk = lambda a: a[rows, src]
    out = {k: jnp.where(ok, tk(r[k]), jnp.zeros((), r[k].dtype)) for k in (
        "t0", "t0err", "rcent", "fcent", "radius", "lam", "fz0", "chi2xy", "chi2zphi", "edep_avg", "hel")}
    out["lambda"] = out.pop("lam")
    out["helicity"] = out.pop("hel")
    out["edep"] = out["edep_avg"]
    mask = tk(r["mask"]) & ok[..., None]
    out.update(ok=ok, mask=mask, hphi=jnp.where(mask, tk(r["hphi"]), np.float32(0)), tc=jnp.where(ok, src, -1),
               calo=jnp.where(ok, calo[rows, src], -1),
               flag=jnp.where(mask, tk(r["flag"]), 0))
    for k in ("x", "y", "z", "uvar", "vvar", "udx", "udy"):
        out[k] = tk(hs[k])
    out["ci"] = jnp.maximum(tk(hs["idx"]), 0)
    out["nsh"] = tk(hs["nsh"])
    out["xyw"] = jnp.zeros((E, K, NH), F32)
    out["zw"] = jnp.zeros((E, K, NH), F32)
    ovf = jnp.where(jnp.any(ovh & tc["valid"], axis=1), OVF_HITS, 0) | jnp.where(nseed > K, OVF_SLOTS, 0)
    out["overflow"] = ovf.astype(I32)
    out["nseeds"] = nseed
    v = jnp.asarray(tc["valid"])
    diag = {"nhits": jnp.where(v, hs["n"], 0), "heapsort": (heap & v).astype(I32),
            "ntriplets": jnp.where(v, r["ntrip0"], 0), "dropped_edep": jnp.where(v, r["dropped"], 0)}
    return out, diag


def helices(ph, tc, cl, S, K=K_SEEDS, NH=TC_HITS, B=LANES):
    """Panel hits ph [E, H] (x, y, z, udx, udy, uvar, vvar, edep, time, ncombo, sid, flag; stock collection order),
    the time clusters tc (valid [E, KT], order [E, KT, H]: timecluster's per-hit order key, ORDER_NONE for
    non-members; t0, t0err, calo [E, KT]: the cluster's calo cluster index or -1) and the calo clusters cl [E, C]
    (disk, time) -> (seeds, diag). seeds: the helix seeds of each event in collection order in K
    slots: ok [E, K], t0, t0err, helicity, rcent, fcent, radius, lambda, fz0, chi2xy, chi2zphi, edep_avg and edep
    (HelixSeed::_eDepAvg), tc and calo (the time cluster and its calo cluster), and the time cluster's face-ordered
    hit slots [E, K, NH] in the
    layout calhelix._merge_event reads (mask: the seed's hits; ci, nsh, x, y, z, uvar, vvar, udx, udy, flag, hphi,
    xyw, zw), nseeds [E], overflow [E] (bits in the module docstring). diag: per (event, time cluster) counters (the
    selected hits, a heap-sorted hit sort, the triplets of the first circle fit, helicities dropped by maxEDepAvg)."""
    ft = F64 if jax.config.jax_enable_x64 else F32
    with strict(ph["ncombo"]):
        out, diag = _helices(ph, tc, cl, S, K, NH, B)
    conv = lambda v: v.astype(ft) if jnp.issubdtype(v.dtype, jnp.floating) else \
        (v.astype(I32) if v.dtype == I64 else v)
    return {k: conv(v) for k, v in out.items()}, diag


def _alive(sd, H, S, cxy, czp):
    """The MergeHelices list-erase loop with compareHelices (UseCalo: a seed whose time cluster has a calo cluster
    beats one without); the survivors."""
    n = sd["mask"].shape[0]
    use = sd["mask"] & ((sd["flag"] & S["bad"]) == 0)
    gm = jnp.zeros((n, H + 1), I32).at[jnp.arange(n)[:, None], jnp.where(use, sd["ci"], H)].set(
        jnp.where(use, sd["nsh"], 0))[:, :H]
    nh = jnp.sum(gm, axis=1)
    nover = jnp.einsum("ah,bh->ab", gm, (gm > 0).astype(I32))
    chi = cxy + czp
    hc = sd["calo"] >= 0

    def comp(i, j):
        minh = jnp.minimum(nh[i], nh[j])
        ov = nover[i, j]
        same = (ov >= S["minnover"]) & (ov.astype(F32) / minh.astype(F32) > S["minfrac"])
        c1 = hc[i] & ~hc[j]
        c2 = hc[j] & ~hc[i]
        if not S["usecalo"]:
            c1 = c2 = jnp.bool_(False)
        first = (nh[i] > nh[j]) & (nh[i] - nh[j] > S["deltanh"])
        second = (nh[j] > nh[i]) & (nh[j] - nh[i] > S["deltanh"])
        f1 = c1 | (~c2 & (first | (~second & (chi[i] < chi[j]))))
        return jnp.where(same, jnp.where(f1, 1, 2), 0)

    def nxt(alive, k):
        cand = alive & (jnp.arange(n) > k)
        return jnp.where(jnp.any(cand), _argmax(cand), n)

    def body(st):
        i, j, alive = st
        c = jnp.where(j < n, comp(i, jnp.minimum(j, n - 1)), -1)
        a2 = jnp.where(c == 1, alive.at[jnp.minimum(j, n - 1)].set(False), alive)
        a2 = jnp.where(c == 2, a2.at[i].set(False), a2)
        i2 = jnp.where((c == -1) | (c == 2), nxt(a2, i), i)
        j2 = jnp.where((c == -1) | (c == 2), nxt(a2, i2), nxt(a2, j))
        return i2, j2, a2

    i0 = nxt(sd["ok"], -1)
    _, _, alive = jax.lax.while_loop(lambda st: st[0] < n, body, (i0, nxt(sd["ok"], i0), sd["ok"]))
    return alive


def merge(seeds, H, S, fz):
    """MergeHelices on the seeds of `helices` [E, K] (the finder's collection order) with merge_settings S: the
    calhelix._merge_event quantities (findchisq's chi2xy / chi2zphi, the HelixTool values) with `ok` the survivors
    of the UseCalo comparison. H is the panel-hit axis length, fz the face z of the hit ratio."""
    ft = F64 if jax.config.jax_enable_x64 else F32
    with strict(seeds["nsh"]):
        sd = {k: v for k, v in seeds.items() if k not in ("overflow", "nseeds")}
        sd = {k: (_f(v) if jnp.issubdtype(v.dtype, jnp.floating) and k not in ("t0", "t0err") else v)
              for k, v in sd.items()}
        sd["t0"] = _d(sd["t0"])
        sd["flag"] = jnp.asarray(sd["flag"]).astype(I64)

        def one(s_):
            o = _merge_event(s_, H, S, fz)
            o["ok"] = _alive(s_, H, S, o["chi2xy"], o["chi2zphi"])
            return o

        out = jax.vmap(one)(sd)
    conv = lambda v: v.astype(ft) if jnp.issubdtype(v.dtype, jnp.floating) else \
        (v.astype(I32) if v.dtype == I64 else v)
    return {k: conv(v) for k, v in out.items()}


def order_keys(hit_lists, E, KT, H):
    """Host helper: time clusters as hit-index lists (stock order) -> timecluster's order key [E, KT, H]."""
    key = np.full((E, KT, H), ORDER_NONE, np.int32)
    for e, tcs in enumerate(hit_lists):
        for k, hl in enumerate(tcs):
            key[e, k, np.asarray(hl, np.int64)] = np.arange(len(hl), dtype=np.int32)
    return key
