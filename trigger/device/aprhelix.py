"""AgnosticHelixFinder (CalPatRec), Offline v13_41_01, on the device: the helix seeds of the apr paths
(TrigAprHelixFinder), and MergeHelices with UseCalo for TrigAprHelixMerger.

A port of the stock module per time cluster, with the stock arithmetic, as the literal transcription
tests/agnostic_helix_reference.py writes it (no timeout, all three helix directions valid, maxSeedCircleHits < 0):

- float32 where stock computes in float, float64 where a C++ double promotes (`_twopi`, double literals,
  std::pow(float, int), LsqSums2 / LsqSums4 sums and results, CLHEP perp() / phi()), traced under `exact.strict` in
  both precisions of the chain; products that an add follows are guarded (`exact.m`), and XLA's division rewrites
  are blocked where they would change a rounding (x / sqrt(y), chained divisions: `exact.g`);
- the hits of a time cluster are laid out as tcHitsFill leaves _tcHits: the stopping target (0, 0, FLT_MAX), the
  calorimeter cluster (cog to Mu2e to tracker, z - caloZOffset), then the cluster's hits sorted by z descending with
  libstdc++'s std::sort (`_zsort`: mpr._stock_sort's introsort, with the heap sort libstdc++ falls back to at the
  depth limit, then the final insertion sort; ties included). Every walk and every double or float sum runs in that
  slot order;
- findHelix is a state machine per time cluster (jax.lax while loops): a search advances the triplet cursor
  (i, j, k) with the stock BREAK / CONTINUE rules, vectorised over the candidates of one loop level, to the next
  triplet whose circle passes the pt window; the pipeline then runs initSeedCircle ... saveHelix on it.
  findSeedPhiLines builds the positive and negative lines of every seed at once (one scan over the test hits: the
  lines' last-added z stays the seed's, as stock, so the z-window and gap breaks are masks), then gates the seeds
  in order on notOnSegment; resolve2PiAmbiguities is vectorised over (line, hit) (stock fixes each line's slope and
  intercept before its loop); refinePhiLine runs on all lines at once until none removes a hit; recoverPoints is a
  while loop, one recovered hit per iteration;
- stock state that crosses triplets and helices is kept: notOnSegment, uselessTripletSeed, used, circleError2,
  averagedOut (setFlags vectorised: the isolation counts and the averaging walk, whose j range skips the hits
  between the isolation decision and i), the helixPhi / helixPhiCorrection writes of recoverPoints, inHelix (also
  for seeds then dropped by maxEDepAvg).

Time clusters are independent in stock (tcHitsFill rebuilds _tcHits), so the finder runs on the valid clusters of
all events, B lanes at a time (the while loops of a vmapped call run as long as its slowest lane: on CPU one or two
lanes are fastest). Fixed capacities (counted in `overflow`, bits: 1 a time cluster with more than NH hits
(truncated), 2 more than LC accepted seed phi lines in one triplet, 4 more than M saved seeds in one time cluster, 8
more than K seeds in one event).

`merge` is MergeHelices on these seeds: calhelix._merge_event's findchisq and HelixTool quantities, with the
survivors of compareHelices including UseCalo (a seed whose time cluster has a calo cluster beats an overlapping one
without), which the apr seeds need and the calo-seeded cpr seeds never reach.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .calhelix import _cint, _lsq, _merge_event, _or, _seqsum, _terms as _t4, _mask
from .exact import g as _g, m as _m, strict
from .mpr import SORT_THRESHOLD, _argmax, _d, _f, _round, poly_atan2
from .tzcluster import _chi2dof, _dydx, _terms as _t2raw, _y0

F32, F64, I32, I64 = jnp.float32, jnp.float64, jnp.int32, jnp.int64
PI = math.pi
TWO_PI = 2. * math.pi                              # static constexpr double _twopi = 2.*M_PI
FLT_MAX = np.float32(np.finfo(np.float32).max)
C_LIGHT = (2.99792458e+8 * 1000.0) / 1.e+9         # CLHEP::c_light, mm/ns
MMT = np.float32(C_LIGHT / 1000.0)                 # _mmTconversion (float)
BZ0 = 1.0117056584                                 # BFieldManager z at the detector origin (stock BField job), T
ST, CALO, PAD = -2, -1, -3                         # HitType::STOPPINGTARGET, HitType::CALOCLUSTER, empty slot
TC_HITS = 192                                      # hits per time cluster (dumps: at most 162)
LINES = 16                                         # accepted seed phi lines per triplet (dumps: 9)
SEEDS_TC = 4                                       # saved seeds per time cluster (dumps: 3)
K_SEEDS = 8                                        # seed slots per event (dumps: 6)
LANES = 2                                          # time clusters per vmapped call
OVF_HITS, OVF_LINES, OVF_SEEDS_TC, OVF_SLOTS = 1, 2, 4, 8
M_I, M_J, M_K, M_RUN, M_DONE = 0, 1, 2, 3, 4
DIRECTIONS = {"downstream", "upstream", "unknown"}
DETECTOR_SYSTEM_Z0 = 10171.0                       # mu2e.detectorSystemZ0 (geom_run1_a.txt), mm
_INF = np.float32(np.inf)


# --- settings and geometry (host) -------------------------------------------------------------------------------------

def geometry(bz0=BZ0):
    """The geometry inputs: the field z at the detector origin, caloZOffset = (diskCaseZLength + BPPipeZOffset +
    BPHoleZLength + FEEZLength) / 2 from the calorimeter G4 parameters (diskCaseZLength = crystalZLength +
    crystalCapZLength), the disk origins in Mu2e (the disks are not rotated) and the tracker centre
    (-solenoidOffset, 0, detectorSystemZ0)."""
    from . import calogeom
    cg = calogeom.CALO_GEOM
    off = cg["crystalZLength"] + cg["crystalCapZLength"]
    off += cg["BPPipeZOffset"]
    off += cg["BPHoleZLength"]
    off += cg["FEEZLength"]
    off /= 2.0
    return {"bz0": bz0, "caloZOffset": off, "disk_origin": np.asarray(calogeom.build()["origin"], np.float64),
            "tracker_center": np.array([-cg["solenoidOffset"], 0.0, DETECTOR_SYSTEM_Z0])}


def settings(m, geom=None):
    """AgnosticHelixFinder module table (and the geometry, `geometry()` by default) -> kernel constants (numpy
    scalars, static); raises for settings the port does not cover."""
    bad = []
    if float(m.get("timeoutMs", 0) or 0) > 0:
        bad.append("timeoutMs")
    if set(m["validHelixDirections"]) != DIRECTIONS:
        bad.append("validHelixDirections " + str(m["validHelixDirections"]))
    if int(m.get("maxSeedCircleHits", -1)) >= 0:
        bad.append("maxSeedCircleHits " + str(m["maxSeedCircleHits"]))
    if bad:
        raise ValueError("device AgnosticHelixFinder does not support " + ", ".join(bad))
    geom = geometry() if geom is None else geom
    f = lambda k: np.float32(m[k])
    sq = lambda k: np.float32(f(k) * f(k))
    bz0 = np.float32(geom["bz0"])
    conv = np.float32(bz0 * MMT)
    return {"iso": bool(m["doIsolationFlag"]), "avg": bool(m["doAverageFlag"]), "edf": bool(m["doEDepFlag"]),
            "st": bool(m["useStoppingTarget"]), "multi": bool(m["findMultipleHelices"]),
            "ievt": int(m["intenseEventThresh"]), "iclu": int(m["intenseClusterThresh"]),
            "isomin": int(m["isoMinHitsNear"]), "isorad2": sq("isoRad"), "mindist2": sq("minDistCut"),
            "edthr": f("eDepFlagThresh"), "minseedz": f("minTripletSeedZ"), "mindz": f("minTripletDz"),
            "maxdz": f("maxTripletDz"), "mintd2": sq("minTripletDist"), "maxres2": sq("maxSeedCircleResidual"),
            "minseed": int(m["minSeedCircleHits"]), "mdz": f("maxDphiDz"), "gap": f("maxSeedLineGap"),
            "win": f("maxZWindow"), "minseg": int(m["minLineSegmentHits"]), "segm2": sq("segMultiplier"),
            "segchi2": f("maxSegmentChi2"), "amb": f("max2PiAmbigResidual"), "phizres": f("maxPhiZResidual"),
            "minfinal": int(m["minFinalSeedHits"]), "ratio": f("maxNHitsRatio"),
            "csig": f("maxCircleRecoverSigma"), "lsig": f("maxLineRecoverSigma"), "calo2": sq("caloClusterSigma"),
            "minsh": int(m["minNHelixStrawHits"]), "minch": int(m["minNHelixComboHits"]),
            "minpt": f("minHelixPerpMomentum"), "maxpt": f("maxHelixPerpMomentum"), "minp2": sq("minHelixMomentum"),
            "maxp2": sq("maxHelixMomentum"), "chi2line": f("chi2LineSaveThresh"), "maxedep": f("maxEDepAvg"),
            "conv": conv, "conv2": np.float32(conv * conv), "zoff": np.float32(geom["caloZOffset"]),
            "disk_origin": np.asarray(geom["disk_origin"], np.float64),
            "tracker_center": np.asarray(geom["tracker_center"], np.float64)}


def merge_settings(merger):
    """MergeHelices module table -> the constants calhelix._merge_event and the UseCalo comparison read."""
    if not merger.get("SelectBest", True):
        raise ValueError("device MergeHelices supports SelectBest true")
    return {"minnover": int(merger["MinNHitOverlap"]), "minfrac": np.float32(merger["MinHitOverlapFraction"]),
            "deltanh": int(merger["deltanh"]), "scalexy": np.float32(merger["scaleXY"]),
            "scalezphi": np.float32(merger["scaleZPhi"]), "bad": _mask(merger.get("BadHitFlags", ["Outlier"])),
            "usecalo": bool(merger.get("UseCalo", True))}


# --- LsqSums2 extras and small helpers --------------------------------------------------------------------------------

def _t2(X, Y, W):
    X, Y, W = jnp.broadcast_arrays(_d(X), _d(Y), _d(W))
    return _t2raw(X, Y, W)


def _dydx_err(F):
    qn, sw, sx, sx2 = F[..., 0], F[..., 1], F[..., 2], F[..., 4]
    D = _m(sw, sx2) - _m(sx, sx)
    return jnp.where(qn > 2, jnp.sqrt(_g(sw / D)), 0.0)


def _y0_err(F):
    qn, sw, sx, sx2 = F[..., 0], F[..., 1], F[..., 2], F[..., 4]
    D = _m(sw, sx2) - _m(sx, sx)
    t = _m(0.0, sw) - sx
    return jnp.where(qn > 2, jnp.sqrt(_m(_g(1.0 / sw), 1.0 + _g(_m(t, t) / D))), 0.0)


def _first(mask):
    """Index of the first True (len if none)."""
    return jnp.where(jnp.any(mask), _argmax(mask), mask.shape[-1]).astype(I32)


def _nint(x):
    """int n = std::round(double)."""
    return _cint(_round(x)).astype(I32)


def _circle_err2(h, xC, yC, S):
    """computeCircleError2 of every slot (the calorimeter slot: caloClusterSigma^2)."""
    dx, dy = h["x"] - xC, h["y"] - yC
    dxn = _m(dx, -h["udy"]) + _m(dy, h["udx"])
    c2 = _m(dxn, dxn) / (_m(dx, dx) + _m(dy, dy))
    s2 = np.float32(1) - c2
    return jnp.where(h["calo"], S["calo2"], _m(h["uvar"], s2) + _m(h["vvar"], c2))


def _residual2(h, xC, yC, rC, ce2):
    x, y = h["x"] - xC, h["y"] - yC
    dd = rC - jnp.sqrt(_m(x, x) + _m(y, y))
    return _m(dd, dd) / ce2


def _helix_phi(h, xC, yC, S):
    """computeHelixPhi of every slot: (helixPhi, helixPhiError2)."""
    X, Y = h["x"] - xC, h["y"] - yC
    Rsq = _m(X, X) + _m(Y, Y)
    hp = poly_atan2(Y, X)
    hp = jnp.where(hp < 0, _f(_d(hp) + TWO_PI), hp)
    R = _g(jnp.sqrt(Rsq))
    tx, ty = Y / R, -X / R
    pw = _m(jnp.sqrt(h["uvar"]), _m(h["udx"], tx) + _m(h["udy"], ty))
    pt = _m(jnp.sqrt(h["vvar"]), _m(h["udy"], tx) + _m(-h["udx"], ty))
    ds2 = jnp.where(h["calo"], S["calo2"], _m(pw, pw) + _m(pt, pt))
    return hp, ds2 / Rsq


def _mom2(r, dphidz, S):
    lam = np.float32(1) / dphidz
    return _m(_m(r, r) + _m(lam, lam), S["conv2"])


def _circ(F):
    q = _lsq(F)
    return _f(q["x0"]), _f(q["y0"]), _f(q["r"])


def _under(p1, e1, p2, e2, dz):
    """underEstimateSlope"""
    a, b = jnp.sqrt(e1), jnp.sqrt(e2)
    return jnp.where(p2 > p1, jnp.abs(((p2 - b) - (p1 + a)) / dz), jnp.abs(((p2 + b) - (p1 - a)) / dz))


def _w(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


# --- the z sort: std::sort with the heap-sort fallback ------------------------------------------------------------

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


def _zsort(key, sel):
    """mpr._stock_sort (libstdc++ std::sort by `key` with operator<, the selected slots taken in slot order) with the
    introsort ranges past the depth limit heap sorted, as libstdc++ does. Returns (perm, n, heap): heap is True when
    a range was heap sorted."""
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


# --- setFlags / resetFlags --------------------------------------------------------------------------------------------

def _set_flags(h, st, S):
    if not (S["iso"] or S["avg"]):              # findHelix calls setFlags only with isolation or averaging on
        return st
    N = h["x"].shape[0]
    p = jnp.arange(N, dtype=I32)
    el = h["valid"] & ~h["special"] & ~st["inH"]
    hie = jnp.where(el, jnp.bool_(S["edf"]) & (h["edep"] > S["edthr"]), st["hie"])
    dx = _d(h["x"][:, None] - h["x"][None, :])                  # std::pow(float, 2) in double
    dy = _d(h["y"][:, None] - h["y"][None, :])
    perp2 = _f(_m(dx, dx) + _m(dy, dy))
    elj = el[None, :] & (p[:, None] != p[None, :])
    if S["iso"]:
        near = elj & (perp2 < S["isorad2"])
        cnt = jnp.cumsum(near, axis=1)
        reach = cnt >= S["isomin"]
        j0 = jnp.where(jnp.any(reach, axis=1), _argmax(reach, axis=1), N).astype(I32)
        iso_i = ~jnp.any(reach, axis=1)
    else:
        j0 = jnp.full(N, -1, I32)
        iso_i = jnp.zeros(N, bool)
    iso = jnp.where(el, iso_i, st["iso"])
    ao = st["ao"]
    if S["avg"]:
        # j visited by i's loop: up to the isolation decision j0, then above max(j0, i)
        vis = elj & ((p[None, :] <= j0[:, None]) | (p[None, :] > jnp.maximum(j0, p)[:, None]))
        close = vis & (perp2 <= S["mindist2"])

        def step(i, ao):
            a = el[i] & ~ao[i]
            return ao | (a & close[i])

        ao = jax.lax.fori_loop(0, h["n"], step, ao)
    return {**st, "hie": hie, "iso": iso, "ao": ao}


def _reset_flags(h, st):
    m = h["valid"] & ~st["inH"] & ~h["st"]
    return {**st, "ao": st["ao"] & ~m, "nos": st["nos"] | m}


def _passes(st, S):
    p = ~st["inH"] & ~st["useless"]
    if S["iso"]:
        p &= ~st["iso"]
    if S["avg"]:
        p &= ~st["ao"]
    if S["edf"]:
        p &= ~st["hie"]
    return p


# --- triplet search ---------------------------------------------------------------------------------------------------

def _search(h, st, S):
    """Advance the triplet cursor to the next triplet whose circle passes the pt window (mode M_RUN), or end the
    findHelix call (M_DONE); one loop level per iteration."""
    N = h["x"].shape[0]
    p = jnp.arange(N, dtype=I32)
    n = h["n"]

    def body(st):
        mode, ci, cj, ck = st["mode"], st["ci"], st["cj"], st["ck"]
        pf = _passes(st, S)
        # setTripletI
        rng = (p >= ci) & (p < n - 2)
        brk = h["valid"] & ((h["z"] < S["minseedz"]) | (h["intense"] & ~h["special"]))
        b = _first(rng & brk)
        ni = _first(rng & (p < b) & pf)
        fi = ni < N
        # setTripletJ
        xi, yi, zi, si, sti = h["x"][ci], h["y"][ci], h["z"][ci], h["special"][ci], h["st"][ci]
        rng = (p > cj) & (p < n - 1)
        dz12 = zi - h["z"]
        brk = (~si & (dz12 > S["maxdz"])) | (h["intense"] & sti & ~h["calo"])
        ddx, ddy = _d(xi - h["x"]), _d(yi - h["y"])
        cont = ~pf | (dz12 < S["mindz"]) | (_m(ddx, ddx) + _m(ddy, ddy) < float(S["mintd2"]))
        b = _first(rng & brk)
        nj = _first(rng & (p < b) & ~cont)
        fj = nj < N
        # setTripletK and initTriplet's pt window
        xj, yj, zj, sj = h["x"][cj], h["y"][cj], h["z"][cj], h["special"][cj]
        rng = (p > ck) & (p < n)
        dz23 = zj - h["z"]
        brk = ~sj & (dz23 > S["maxdz"])
        ax, ay, bx, by = xi - h["x"], yi - h["y"], xj - h["x"], yj - h["y"]
        cont = ~pf | (dz23 < S["mindz"]) | (_m(ax, ax) + _m(ay, ay) < S["mintd2"]) | \
            (_m(bx, bx) + _m(by, by) < S["mintd2"])
        one = np.float32(1)
        F3 = (_t4(xi, yi, one) + _t4(xj, yj, one))[None, :] + _t4(h["x"], h["y"], one)
        pt = _m(_f(_lsq(F3)["r"]), S["conv"])
        ptok = ~(pt < S["minpt"]) & ~(pt > S["maxpt"])
        b = _first(rng & brk)
        nk = _first(rng & (p < b) & ~cont & ptok)
        fk = nk < N
        st = dict(st)
        st["nsearch"] = st["nsearch"] + 1
        # M_I -> M_J (a seed i) or M_DONE; M_J -> M_K (a j) or M_I (the j loop ended: uselessTripletSeed);
        # M_K -> M_RUN (a triplet) or M_J
        new_mode = jnp.where(mode == M_I, jnp.where(fi, M_J, M_DONE),
                             jnp.where(mode == M_J, jnp.where(fj, M_K, M_I), jnp.where(fk, M_RUN, M_J)))
        st["useless"] = jnp.where((mode == M_J) & ~fj, st["useless"].at[ci].set(st["uls"]), st["useless"])
        st["ci"] = jnp.where(mode == M_I, jnp.where(fi, ni, ci), jnp.where((mode == M_J) & ~fj, ci + 1, ci))
        st["uls"] = jnp.where((mode == M_I) & fi, True, st["uls"])
        st["cj"] = jnp.where((mode == M_I) & fi, ni, jnp.where((mode == M_J) & fj, nj, cj))
        st["ck"] = jnp.where((mode == M_J) & fj, nj, jnp.where((mode == M_K) & fk, nk, ck))
        st["mode"] = new_mode.astype(I32)
        return st

    return jax.lax.while_loop(lambda st: st["mode"] < M_RUN, body, st)


# --- the pipeline on one triplet --------------------------------------------------------------------------------------

def _seed_lines(h, st, rC, S, LC):
    """findSeedPhiLines: the accepted lines (membership, sums, corrections 0) in stock order, and notOnSegment."""
    N = h["x"].shape[0]
    p = jnp.arange(N, dtype=I32)
    z, hp, he = h["z"], st["hphi"], st["hphie2"]
    ej = h["valid"] & ~st["inH"] & st["used"] & ~h["special"]
    cs = ej & (p < h["n"] - S["minseg"])
    w = np.float32(1) / he
    T = _t2(z, hp, w)
    mdz, gap, win = S["mdz"], S["gap"], S["win"]

    def step(j, c):
        lpP, lpE, lnP, lnE, Fp, Fn, AP, AN = c
        tz, tP, tE = z[j], hp[j], he[j]
        dd = z - tz                                  # seedZ - testZ = positive and negative line dz (z never moves)
        on = cs & (p < j) & ej[j] & ~(dd > win) & ~(dd > gap)
        pd = lpP - tP
        pdSq, pdE = _m(pd, pd), lpE + tE
        nd = lnP - tP
        ndSq, ndE = _m(nd, nd), lnE + tE
        dz_ok = ~(dd > gap) & (_d(dd) > 1.e-10)
        ap = on & ((lpP > tP) | (pdSq < pdE)) & dz_ok
        ap &= (pdSq < pdE) | (jnp.abs(pd / dd) <= mdz) | (_under(lpP, lpE, tP, tE, dd) <= mdz)
        an = on & ((tP > lnP) | (ndSq < ndE)) & dz_ok
        an &= (ndSq < ndE) | (jnp.abs(nd / dd) <= mdz) | (_under(lnP, lnE, tP, tE, dd) <= mdz)
        Fp = jnp.where(ap[:, None], Fp + T[j], Fp)
        Fn = jnp.where(an[:, None], Fn + T[j], Fn)
        return (jnp.where(ap, tP, lpP), jnp.where(ap, tE, lpE), jnp.where(an, tP, lnP), jnp.where(an, tE, lnE),
                Fp, Fn, AP.at[:, j].set(AP[:, j] | ap), AN.at[:, j].set(AN[:, j] | an))

    eye = p[:, None] == p[None, :]
    lpP, _, lnP, _, Fp, Fn, memP, memN = jax.lax.fori_loop(0, h["n"], step, (hp, he, hp, he, T, T, eye, eye))

    def accept(F, last, slope_ok):
        ok = cs & (F[:, 0] >= S["minseg"]) & slope_ok
        p2 = _mom2(rC, _f(_dydx(F)), S)
        dphi = hp - last
        return ok & (_m(dphi, dphi) > _m(S["segm2"], he)) & (p2 > S["minp2"]) & (p2 < S["maxp2"]) & \
            (_chi2dof(F) <= float(S["segchi2"]))

    okP = accept(Fp, lpP, _dydx(Fp) <= float(mdz))
    okN = accept(Fn, lnP, jnp.abs(_dydx(Fn)) <= float(mdz))

    def gate(q, c):
        nos, act = c
        a = cs[q] & nos[q]
        nos = nos & ~(a & okP[q] & memP[q]) & ~(a & okN[q] & memN[q])
        return nos, act.at[q].set(a)

    nos, act = jax.lax.fori_loop(0, h["n"], gate, (st["nos"], jnp.zeros(N, bool)))
    sel = jnp.stack([act & okP, act & okN], 1).reshape(-1)
    nl = jnp.sum(sel).astype(I32)
    li = jnp.nonzero(sel, size=LC, fill_value=0)[0]
    lv = jnp.arange(LC) < nl
    mem = jnp.stack([memP, memN], 1).reshape(2 * N, N)[li] & lv[:, None]
    F = jnp.stack([Fp, Fn], 1).reshape(2 * N, 7)[li]
    return nos, nl, lv, mem, F


def _resolve(h, st, lv, mem, F, S):
    """resolve2PiAmbiguities: (sums, membership, corrections, list ranks)."""
    N = h["x"].shape[0]
    z, hp, he = h["z"], st["hphi"], st["hphie2"]
    slope, icpt = _f(_dydx(F))[:, None], _f(_y0(F))[:, None]
    serr, ierr = _f(_dydx_err(F))[:, None], _f(_y0_err(F))[:, None]
    mj = (h["valid"] & ~st["inH"] & ~h["st"] & st["used"])[None, :] & ~mem & lv[:, None]
    dp = _m(slope, z) + icpt - hp
    nn = _nint(_d(dp) / TWO_PI)
    tp = _f(_d(hp) + _m(TWO_PI, _d(nn)))
    dp2 = jnp.abs(_m(slope, z) + icpt - tp)
    e2 = _m(z, z, serr, serr) + _m(ierr, ierr) + he
    add = mj & (_m(dp2, dp2) < _m(S["amb"], S["amb"], e2))
    T = _t2(z[None, :], tp, _f(1.0 / _d(he))[None, :])

    F = jax.lax.fori_loop(0, h["n"], lambda j, F: jnp.where(add[:, j, None], F + T[:, j], F), F)
    norig = jnp.sum(mem, axis=1)
    rank = jnp.where(mem, jnp.cumsum(mem, axis=1) - 1, jnp.where(add, norig[:, None] + jnp.cumsum(add, axis=1) - 1,
                                                                   N * 4))
    return F, mem | add, jnp.where(add, nn, 0), rank


def _refine(h, st, lv, mem, F, corr, rank, S):
    """refinePhiLine, all lines with at least minFinalSeedHits hits at once, until none removes a hit."""
    N = h["x"].shape[0]
    z, hp, he = h["z"], st["hphi"], st["hphie2"]
    thr = np.float32(S["phizres"] * S["phizres"])
    pP = _f(_d(hp)[None, :] + _m(TWO_PI, _d(corr)))
    act0 = lv & (jnp.sum(mem, axis=1) >= S["minfinal"])

    def body(c):
        F, mem, act, nrm = c
        lp = _f(_m(_dydx(F)[:, None], _d(z)[None, :]) + _y0(F)[:, None])
        dph = jnp.abs(lp - pP)
        r2 = _m(dph, dph) / he[None, :]
        vm = mem & (r2 == r2)
        mx = jnp.max(jnp.where(vm, r2, -_INF), axis=1)
        pick = vm & (r2 == mx[:, None]) & (mx[:, None] > 0)
        rm = _argmax(-jnp.where(pick, rank, N * 8).astype(I32), axis=1)        # the first in list order
        largest = jnp.where(jnp.any(pick, axis=1), mx, np.float32(0))
        rem = act & (largest > thr)
        Tr = _t2(z[rm], jnp.take_along_axis(pP, rm[:, None], 1)[:, 0], _f(1.0 / _d(he[rm])))
        F = jnp.where(rem[:, None], F - Tr, F)
        mem = mem & ~(rem[:, None] & (jnp.arange(N)[None, :] == rm[:, None]))
        return F, mem, rem, nrm + jnp.sum(rem).astype(I32)

    F, mem, _, nrm = jax.lax.while_loop(lambda c: jnp.any(c[2]), body, (F, mem, act0, I32(0)))
    return F, mem, nrm


def _pipeline(h, st, S, LC, M):
    """initTriplet ... saveHelix on the triplet (ci, cj, ck) of a lane in mode M_RUN."""
    N = h["x"].shape[0]
    act = st["mode"] == M_RUN
    ci, cj, ck = st["ci"], st["cj"], st["ck"]
    one = np.float32(1)
    F3 = _t4(h["x"][ci], h["y"][ci], one) + _t4(h["x"][cj], h["y"][cj], one) + _t4(h["x"][ck], h["y"][ck], one)
    xC, yC, rC = _circ(F3)
    # initSeedCircle
    m1 = h["valid"] & ~st["inH"] & ~h["st"]
    e2 = _circle_err2(h, xC, yC, S)
    take = m1 & (_residual2(h, xC, yC, rC, e2) < S["maxres2"])
    F1 = _seqsum(_t4(h["x"], h["y"], one / e2), take, jnp.zeros(16, F64))
    ok1 = ~(F1[0] < S["minseed"])
    s1 = {**st, "ce2": jnp.where(m1, e2, st["ce2"]),
          "used": jnp.where(m1, take, jnp.where(h["valid"] & h["st"] & ~st["inH"], False, st["used"])),
          "uls": st["uls"] & ~ok1, "npipe": st["npipe"] + 1}
    # initHelixPhi
    xC1, yC1, rC1 = _circ(F1)
    m2 = h["valid"] & ~s1["inH"] & ~h["st"] & s1["used"]
    hp, he = _helix_phi(h, xC1, yC1, S)
    s2 = {**s1, "hphi": jnp.where(m2, hp, s1["hphi"]), "hphie2": jnp.where(m2, he, s1["hphie2"]),
          "hcorr": jnp.where(m2, 0, s1["hcorr"])}
    # findSeedPhiLines
    nos, nl, lv, mem, F = _seed_lines(h, s2, rC1, S, LC)
    ok3 = nl > 0
    s3 = {**s2, "nos": nos, "ovf": s2["ovf"] | jnp.where(nl > LC, OVF_LINES, 0),
          "maxlines": jnp.maximum(s2["maxlines"], nl)}
    # resolve2PiAmbiguities, refinePhiLine
    F, mem, corr, rank = _resolve(h, s3, lv, mem, F, S)
    F, mem, nrm = _refine(h, s3, lv, mem, F, corr, rank, S)
    # initFinalSeed
    cnt = jnp.sum(mem, axis=1).astype(I32)
    c2 = _chi2dof(F)

    def bstep(c, l):
        best, most, cf = c
        gt = lv[l] & (cnt[l] > most)
        eq = lv[l] & (cnt[l] == most) & (c2[l] < _d(cf))
        tk = gt | eq
        return (jnp.where(tk, l, best), jnp.where(gt, cnt[l], most), jnp.where(tk, _f(c2[l]), cf)), None

    (best, _, _), _ = jax.lax.scan(bstep, (I32(0), I32(0), np.float32(0)), jnp.arange(LC, dtype=I32))
    m6 = h["valid"] & ~s3["inH"] & ~h["st"]
    bm = mem[best]
    Fc = _seqsum(_t4(h["x"], h["y"], _f(1.0 / _d(s3["ce2"]))), m6 & bm, jnp.zeros(16, F64))
    Fl = F[best]
    ratio = _f(F1[0]) / _f(Fl[0])
    ok6 = ~(Fc[0] < S["minfinal"]) & ~(ratio > S["ratio"])
    s4 = {**s3, "used": jnp.where(m6, bm, s3["used"]), "hcorr": jnp.where(m6 & bm, corr[best], s3["hcorr"]),
          "nrefine": s3["nrefine"] + nrm}
    s4 = _w(ok3, s4, s3)
    # recoverPoints (only on the lanes that reach it)
    r7 = act & ok1 & ok3 & ok6
    s5, Fc, Fl, nrec = _recover(h, s4, Fc, Fl, S, r7)
    s5 = {**s5, "nrec": s5["nrec"] + nrec}
    # checkHelixViability
    radius = _f(_lsq(Fc)["r"])
    pt = _m(radius, S["conv"])
    mom2 = _mom2(radius, _f(_dydx(Fl)), S)
    ok8 = ~(_chi2dof(Fl) > float(S["chi2line"])) & ~(pt < S["minpt"]) & ~(pt > S["maxpt"]) & \
        ~(mom2 < S["minp2"]) & ~(mom2 > S["maxp2"])
    cm = h["valid"] & ~s5["inH"] & ~h["special"]
    nsh_t = jnp.sum(jnp.where(cm, h["nsh"], 0)).astype(I32)
    nch_t = jnp.sum(cm).astype(I32)
    sm = cm & s5["used"]
    nsh_h = jnp.sum(jnp.where(sm, h["nsh"], 0)).astype(I32)
    nch_h = jnp.sum(sm).astype(I32)
    save = r7 & ok8 & (nsh_h >= S["minsh"]) & (nch_h >= S["minch"])
    another = ((nsh_t - nsh_h) >= S["minsh"]) & ((nch_t - nch_h) >= S["minch"])
    # saveHelix
    tres = jnp.sqrt(h["tvar"])
    Ft = _seqsum(_t2(h["z"], h["time"], one / _m(tres, tres)), sm, jnp.zeros(7, F64))
    esum = _seqsum(_m(h["edep"], h["nsh"].astype(F32))[:, None], sm, jnp.zeros(1, F32))[0]
    edep = _f(_d(esum) / (_d(nsh_h) + 1e-10))
    q = _lsq(Fc)
    cx, cy = _d(_f(q["x0"])), _d(_f(q["y0"]))
    dfdz = _f(_dydx(Fl))
    fz0 = _f(jnp.fmod(_y0(Fl), TWO_PI))
    fz0 = jnp.where(_d(fz0) > PI, _f(_d(fz0) - TWO_PI), jnp.where(_d(fz0) < -PI, _f(_d(fz0) + TWO_PI), fz0))
    lam = _f(1. / _d(dfdz))
    seed = {"t0": _y0(Ft), "t0err": _y0_err(Ft), "rcent": _f(jnp.sqrt(_m(cx, cx) + _m(cy, cy))),
            "fcent": _f(jnp.where((cx == 0.0) & (cy == 0.0), 0.0, jnp.arctan2(cy, cx))), "radius": radius,
            "lambda": lam, "fz0": fz0, "helicity": jnp.where(lam > 0, 1, -1).astype(I32),
            "chi2xy": _f(q["chi2c"]), "chi2zphi": _f(_chi2dof(Fl)), "edep_avg": edep}
    hphi_s = _f(_d(s5["hphi"]) + _m(_d(s5["hcorr"]), TWO_PI))
    keep = save & ~(edep > S["maxedep"])
    k = jnp.where(keep & (s5["nout"] < M), s5["nout"], M)
    out = {kk: s5["o_" + kk].at[k].set(v, mode="drop") for kk, v in seed.items()}
    s6 = {**s5, **{"o_" + kk: v for kk, v in out.items()},
          "o_mask": s5["o_mask"].at[k].set(sm, mode="drop"), "o_hphi": s5["o_hphi"].at[k].set(hphi_s, mode="drop"),
          "nout": s5["nout"] + keep.astype(I32), "nsave": s5["nsave"] + save.astype(I32),
          "ovf": s5["ovf"] | jnp.where(keep & (s5["nout"] >= M), OVF_SEEDS_TC, 0),
          "inH": s5["inH"] | (save & sm)}
    if S["avg"]:
        s6 = _w(save & another, _reset_flags(h, s6), s6)
    nxt = save & another & jnp.bool_(S["multi"])
    s7 = _w(nxt, _set_flags(h, s6, S), s6)
    mode = jnp.where(save, jnp.where(nxt, M_I, M_DONE), M_K).astype(I32)
    s7 = {**s7, "mode": mode, "ci": jnp.where(nxt, 0, s7["ci"]).astype(I32)}
    # lanes not in M_RUN keep their state; the state writes of a stage need the stage reached
    r_cont1 = act & ~ok1                        # initSeedCircle failed: its writes stay
    s1m = {**s1, "mode": jnp.int32(M_K)}
    r_cont3 = act & ok1 & ~ok3                  # no line: initHelixPhi and the notOnSegment writes stay
    s3m = {**s3, "mode": jnp.int32(M_K)}
    r_cont6 = act & ok1 & ok3 & ~ok6            # initFinalSeed failed
    s4m = {**s4, "mode": jnp.int32(M_K)}
    res = _w(r7, s7, st)
    res = _w(r_cont6, s4m, res)
    res = _w(r_cont3, s3m, res)
    res = _w(r_cont1, s1m, res)
    return res


def _recover(h, st, Fc, Fl, S, on):
    """while (recoverPoints()) on the lanes `on` (the others leave the loop at once)."""
    csig2 = np.float32(S["csig"] * S["csig"])
    lsig2 = np.float32(S["lsig"] * S["lsig"])
    base = h["valid"] & ~st["inH"] & ~h["st"]

    def body(c):
        st, Fc, Fl, _, nrec = c
        xC, yC, rC = _circ(Fc)
        slope, icpt = _f(_dydx(Fl)), _f(_y0(Fl))
        c0 = base & ~st["used"]
        cres2 = _residual2(h, xC, yC, rC, st["ce2"])
        c1 = c0 & ~(cres2 > csig2)
        hp, he = _helix_phi(h, xC, yC, S)
        phd = _m(slope, h["z"]) + icpt - hp
        nn = _nint(_d(phd) / TWO_PI)
        php = _f(_d(hp) + _m(TWO_PI, _d(nn)))
        phd2 = jnp.abs(_m(slope, h["z"]) + icpt - php)
        pres2 = _m(phd2, phd2) / he
        c2 = c1 & ~(pres2 > lsig2)
        res2 = pres2 + cres2
        rec = jnp.any(c2)
        L = res2.shape[0]
        first = jnp.minimum(_first(c2), L - 1)
        mn = jnp.min(jnp.where(c2 & (res2 == res2), res2, _INF))
        fnan = res2[first] != res2[first]
        add = jnp.where(fnan, first, jnp.minimum(_first(c2 & (res2 == mn)), L - 1))
        s1 = {**st, "hphi": jnp.where(c1, hp, st["hphi"]), "hphie2": jnp.where(c1, he, st["hphie2"]),
              "hcorr": jnp.where(c1, nn, st["hcorr"])}
        used = s1["used"].at[add].set(True)
        Fa = Fc + _t4(h["x"][add], h["y"][add], _f(1.0 / _d(s1["ce2"][add])))
        xa, ya, _ = _circ(Fa)
        m7 = base & used
        ce2 = jnp.where(m7, _circle_err2(h, xa, ya, S), s1["ce2"])
        Fn = _seqsum(_t4(h["x"], h["y"], _f(1.0 / _d(ce2))), m7, jnp.zeros(16, F64), rec)
        xb, yb, _ = _circ(Fn)
        hp3, he3 = _helix_phi(h, xb, yb, S)
        dp = _m(slope, h["z"]) + icpt - hp3
        corr = _nint(_d(dp) / TWO_PI)
        php3 = _f(_d(hp3) + _m(_d(corr), TWO_PI))
        Fln = _seqsum(_t2(h["z"], php3, _f(1.0 / _d(he3))), m7, jnp.zeros(7, F64), rec)
        new = {**s1, "used": used, "ce2": ce2, "hphi": jnp.where(m7, hp3, s1["hphi"]),
               "hphie2": jnp.where(m7, he3, s1["hphie2"]), "hcorr": jnp.where(m7, corr, s1["hcorr"])}
        # no recovery: only the candidate writes of the scan stay
        st = _w(rec, new, s1)
        return st, jnp.where(rec, Fn, Fc), jnp.where(rec, Fln, Fl), rec, nrec + rec.astype(I32)

    st, Fc, Fl, _, nrec = jax.lax.while_loop(lambda c: c[3], body, (st, Fc, Fl, on, I32(0)))
    return st, Fc, Fl, nrec


# --- one time cluster -------------------------------------------------------------------------------------------------

def _state(h, M):
    N = h["x"].shape[0]
    zb = jnp.zeros(N, bool)
    zf = jnp.zeros(N, F32)
    st = {"inH": zb, "used": zb, "iso": zb, "ao": zb, "hie": zb, "useless": zb, "nos": jnp.ones(N, bool),
          "ce2": jnp.ones(N, F32), "hphi": zf, "hphie2": zf, "hcorr": jnp.zeros(N, I32),
          "ci": I32(0), "cj": I32(0), "ck": I32(0), "uls": jnp.bool_(True),
          "mode": jnp.where(h["go"], M_I, M_DONE).astype(I32),
          "nout": I32(0), "nsave": I32(0), "ovf": I32(0), "npipe": I32(0), "nsearch": I32(0), "nrec": I32(0),
          "nrefine": I32(0), "maxlines": I32(0)}
    for k in ("t0", "t0err"):
        st["o_" + k] = jnp.zeros(M, F64)
    for k in ("rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi", "edep_avg"):
        st["o_" + k] = jnp.zeros(M, F32)
    st["o_helicity"] = jnp.zeros(M, I32)
    st["o_mask"] = jnp.zeros((M, N), bool)
    st["o_hphi"] = jnp.zeros((M, N), F32)
    return st


def _one_tc(h, S, LC, M):
    st = _state(h, M)
    st = _w(h["go"], _set_flags(h, st, S), st)

    def body(st):
        st = _search(h, st, S)
        return _pipeline(h, st, S, LC, M)

    return jax.lax.while_loop(lambda st: st["mode"] != M_DONE, body, st)


def _fill(hev, order, calo, clev, S, NH):
    """tcHitsFill of one time cluster: the slot arrays [NH + 2]."""
    H = order.shape[0]
    N = NH + 2
    mem = order >= 0
    nm = jnp.sum(mem).astype(I32)
    oc = jnp.clip(order, 0, H - 1)
    perm, _, fb = _zsort(-_f(hev["z"])[oc], mem)
    hi = oc[perm][:NH]
    hv = jnp.arange(NH) < nm
    off = I32(1 if S["st"] else 0) + (calo >= 0).astype(I32)
    p = jnp.arange(N, dtype=I32)
    src = jnp.clip(p - off, 0, NH - 1)
    real = (p >= off) & (p - off < NH) & hv[src]
    hidx = hi[src]
    c = jnp.maximum(calo, 0)
    o = jnp.asarray(S["disk_origin"])[clev["disk"][c]]
    tcen = S["tracker_center"]
    cx = _f((o[0] + _d(clev["cog_x"][c])) - tcen[0])
    cy = _f((o[1] + _d(clev["cog_y"][c])) - tcen[1])
    cz = _f(((o[2] + 0.0) - tcen[2]) - _d(S["zoff"]))
    is_st = (p == 0) & jnp.bool_(S["st"])
    is_calo = (calo >= 0) & (p == (1 if S["st"] else 0))
    gf = lambda k: jnp.where(real, _f(hev[k][hidx]), np.float32(0))
    h = {k: gf(k) for k in ("udx", "udy", "uvar", "vvar", "edep", "time", "timevar")}
    h["tvar"] = h.pop("timevar")
    h["x"] = jnp.where(real, _f(hev["x"][hidx]), jnp.where(is_calo, cx, np.float32(0)))
    h["y"] = jnp.where(real, _f(hev["y"][hidx]), jnp.where(is_calo, cy, np.float32(0)))
    h["z"] = jnp.where(real, _f(hev["z"][hidx]), jnp.where(is_calo, cz, jnp.where(is_st, FLT_MAX, np.float32(0))))
    h["nsh"] = jnp.where(real, jnp.asarray(hev["ncombo"])[hidx].astype(I32), 0)
    h["flag"] = jnp.where(real, jnp.asarray(hev["flag"])[hidx].astype(I64), 0)
    h["idx"] = jnp.where(real, hidx, jnp.where(is_calo, CALO, jnp.where(is_st, ST, PAD))).astype(I32)
    h["valid"] = real | is_st | is_calo
    h["st"], h["calo"] = is_st, is_calo
    h["special"] = is_st | is_calo
    h["n"] = off + jnp.minimum(nm, NH)
    return h, nm, (nm > NH), fb


def _helices(ph, tc, cl, S, K, NH, M, LC, B):
    E, H = ph["x"].shape
    KT = tc["valid"].shape[1]
    hev = {k: ph[k] for k in ("x", "y", "z", "udx", "udy", "uvar", "vvar", "edep", "time", "timevar", "ncombo",
                              "flag")}
    clev = {"disk": jnp.asarray(cl["disk"]).astype(I32), "cog_x": cl["cog_x"], "cog_y": cl["cog_y"]}
    calo = jnp.where(tc["valid"], jnp.asarray(tc["calo"]).astype(I32), -1)
    fill = jax.vmap(lambda he, o, c, ce: jax.vmap(lambda oo, cc: _fill(he, oo, cc, ce, S, NH))(o, c))
    hs, nm, ovh, fb = fill(hev, jnp.asarray(tc["order"]).astype(I32), calo, clev)
    ntc = jnp.sum(tc["valid"], axis=1)
    intense = (ntc > S["ievt"])[:, None] | (nm > S["iclu"])
    go = tc["valid"] & ~(intense & (calo < 0)) & ~(nm < 3) & ~(nm < S["minseed"]) & ~(nm < S["minseg"]) & \
        ~(nm < S["minch"])
    hs = {**hs, "intense": intense, "go": go}
    NI = E * KT
    flat = {k: v.reshape((NI,) + v.shape[2:]) for k, v in hs.items()}
    gof = go.reshape(-1)
    order = jnp.concatenate([jnp.nonzero(gof, size=NI, fill_value=NI)[0].astype(I32), jnp.full(B, NI, I32)])
    nchunk = ((jnp.sum(gof) + B - 1) // B).astype(I32)
    run = jax.vmap(lambda h: _one_tc(h, S, LC, M))
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
    # the seeds of an event in collection order: time clusters in order, seeds in save order
    N = NH + 2
    cand = jnp.arange(M)[None, None, :] < jnp.minimum(r["nout"], M)[..., None]               # [E, KT, M]
    cf = cand.reshape(E, KT * M)
    pos = jnp.cumsum(cf, axis=1) - 1
    nseed = jnp.sum(cf, axis=1).astype(I32)
    rows = jnp.arange(E)[:, None]
    dst = jnp.where(cf & (pos < K), pos, K)
    src = jnp.zeros((E, K + 1), I32).at[rows, dst].set(jnp.arange(KT * M, dtype=I32)[None, :], mode="drop")[:, :K]
    ok = jnp.arange(K)[None, :] < jnp.minimum(nseed, K)[:, None]
    kt, mm = src // M, src % M
    tk = lambda a: a[rows, kt]                                                             # item arrays [E, K, ...]
    sk = lambda a: a[rows, kt, mm]                                                         # seed arrays
    out = {k: jnp.where(ok, sk(r["o_" + k]), jnp.zeros((), r["o_" + k].dtype)) for k in (
        "t0", "t0err", "rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi", "edep_avg", "helicity")}
    mask = sk(r["o_mask"]) & ok[..., None]
    out.update(ok=ok, mask=mask, hphi=jnp.where(mask, sk(r["o_hphi"]), np.float32(0)), tc=jnp.where(ok, kt, -1),
               calo=jnp.where(ok, calo[rows, kt], -1), edep=jnp.zeros((E, K), F32))
    for k in ("x", "y", "z", "uvar", "vvar", "udx", "udy"):
        out[k] = tk(hs[k])
    out["ci"] = jnp.maximum(tk(hs["idx"]), 0)
    out["nsh"] = tk(hs["nsh"])
    out["flag"] = tk(hs["flag"])
    out["xyw"] = jnp.zeros((E, K, N), F32)
    out["zw"] = jnp.zeros((E, K, N), F32)
    ovf = jnp.zeros(E, I32)
    ovf = ovf | jnp.where(jnp.any(ovh & tc["valid"], axis=1), OVF_HITS, 0)
    ovf = ovf | _or(jnp.where(go, r["ovf"], 0), 1) | jnp.where(nseed > K, OVF_SLOTS, 0)
    out["overflow"] = ovf.astype(I32)
    out["nseeds"] = nseed
    diag = {k: jnp.where(go, r[k], 0) for k in ("npipe", "nsearch", "nrec", "nrefine", "maxlines", "nsave")}
    diag["nhits"] = jnp.where(tc["valid"], nm, 0)
    diag["heapsort"] = (fb & tc["valid"]).astype(I32)
    return out, diag


def helices(ph, tc, cl, S, K=K_SEEDS, NH=TC_HITS, M=SEEDS_TC, LC=LINES, B=LANES):
    """Panel hits ph [E, H] (x, y, z, udx, udy, uvar, vvar, edep, time, timevar, ncombo, flag; stock collection
    order), the electron time clusters tc (valid [E, KT], order [E, KT, H]: the member hit indices in stock order,
    then -1, as tzcluster.clusters returns them; calo [E, KT]: the cluster's calo cluster index or -1) and the calo
    clusters cl [E, C] (disk, cog_x, cog_y) -> (seeds, diag). seeds: the saved helix seeds of each event in
    collection order in K slots: ok [E, K], t0, t0err, helicity, rcent, fcent, radius, lambda, fz0, chi2xy,
    chi2zphi, edep_avg (the eDepAvg cut value), edep (the stored HelixSeed::_eDepAvg: 0, stock does not set it), tc
    and calo (the time cluster and its calo cluster), and the time cluster's slots [E, K, NH + 2] in tcHits order in
    the layout calhelix._merge_event reads (mask: the seed's hits; ci, nsh, x, y, z, uvar, vvar, udx, udy, flag,
    hphi, xyw, zw), nseeds [E], overflow [E] (bits in the module docstring). diag: per (event, time cluster) work
    counters (pipelines run, search steps, recoveries, refine removals, the most lines of a triplet, saves, hits,
    z sorts that took the heap-sort fallback)."""
    ft = F64 if jax.config.jax_enable_x64 else F32
    with strict(ph["ncombo"]):
        out, diag = _helices(ph, tc, cl, S, K, NH, M, LC, B)
    conv = lambda v: v.astype(ft) if jnp.issubdtype(v.dtype, jnp.floating) else \
        (v.astype(I32) if v.dtype == I64 else v)
    return {k: conv(v) for k, v in out.items()}, diag


# --- MergeHelices (UseCalo) -------------------------------------------------------------------------------------------

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
