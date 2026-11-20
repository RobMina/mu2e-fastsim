"""CalHelixFinder (CalPatRec, with CalHelixFinderAlg) and MergeHelices (TrkReco), Offline v13_41_01, on the device,
for the cpr paths.

A port of the stock finder per (event, time peak) and helicity, with the stock arithmetic, as the literal
transcriptions tests/cal_helix_reference.py and tests/merge_helices_reference.py write it:

- float32 where stock computes in float, float64 where a C++ double promotes (double literals, M_PI, LsqSums4 /
  LsqSums2, HelixParams), traced under `exact.strict` in both precisions of the chain; every product that an add
  follows is guarded (`exact.m`) against fused multiply-add;
- the float libm calls (cosf, sinf, atan2f) are float32 jnp.cos / jnp.sin / jnp.arctan2 (XLA's CPU float trig
  equals glibc's), the double ones float64 jnp functions;
- the hits of a peak are its selected ComboHits ordered as fillFaceOrderedHits does (libstdc++ std::sort by unique
  panel, `mpr._stock_sort`), then laid out in the order every stock face/panel/hit walk visits them (faces
  descending, panels ascending, sorted order within a panel): a walk "from a seed hit" is then the suffix that starts
  at the seed, and every float or double sum runs in that order;
- XLA's division rewrites are blocked where they would change a rounding: (a / b) / c and x / sqrt(y) keep their
  intermediate (`exact.g`);
- stock state that crosses calls is kept: the triplet search's shared `_hdfdz` (the use-mpv search starts each track
  from the slope the last findDfDz left), the trial add/remove rounding of rescueHits, the counters rescueHits does
  not take back, and the hit buffer of the Helix copy (see the reference's docstring).

Per-hit loops are vectorised over the peak's hit slots; the sequential walks (faces, the triplet seeds, refits, the
cleanup loops, double sums in hit order) are jax.lax while loops, so each piece is traced once.

Stock behaviour this port flags instead of reproducing (`err` bits per peak, summed per event): findDfDz's throw on
a negative station dz with a calo cluster (1), a doLinearFitPhiZ cleanup that does not terminate within LOOP_CAP
iterations (2), more than NCHOICES phi-z slope choices for one station pair (4), the calculateTrackParameters 2 pi
loops past LOOP_CAP (8), and a hit sort that would fall back to heap sort (16).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .exact import g as _g, m as _m, strict
from .mpr import _argmax, _d, _f, _stock_sort, poly_atan2

F32, F64, I32, I64 = jnp.float32, jnp.float64, jnp.int32, jnp.int64
PI = math.pi                                 # M_PI
TWO_PI = 2.0 * math.pi
FPI = np.float32(math.pi)
FTWOPI = np.float32(np.float32(2) * FPI)
FHALFPI = np.float32(np.float64(FPI) / 2.0)
NTOTALFACES, KNPANELS, NSTATIONS = 72, 3, 18
CHF_HITS = 128                               # selected hit slots per time peak (more: counted in `overflow`)
NCHOICES = 32                                # phi-z slope choices per station pair in findDfDz
LOOP_CAP = 1000                              # iteration cap of the loops that stock does not bound
E_MASS = 0.510998910                         # e- mass (ParticleList), MeV
C_LIGHT = (2.99792458e+8 * 1000.0) / 1.e+9   # CLHEP c_light, mm/ns
# StrawHitFlag bit positions (RecoDataProducts/inc/StrawHitFlag.hh, v13_41_01)
SHF_BITS = {
    "Stereo": 0, "EnergySelection": 1, "RadiusSelection": 2, "TimeSelection": 3, "BackgroundCluster": 5,
    "Background": 6, "Isolated": 7, "Outlier": 8, "OtherBackground": 9, "TimeDivision": 10, "TimeCluster": 11,
    "CalorimeterSelection": 12, "StrawXTalk": 13, "ElectronicsXTalk": 14, "TrackerSelection": 15, "Active": 16,
    "GoodDOCA": 17, "ResolvedPhi": 18, "CalorimeterPreSelection": 19, "InTime": 20, "PanelCombo": 21, "Track": 22,
    "Dead": 23, "Noisy": 24, "NHitSelection": 25, "StereoLineFit": 26,
}
BKG_MASK = 1 << SHF_BITS["Background"]
OUTLIER = 1 << SHF_BITS["Outlier"]
ERR_THROW, ERR_CLEANUP, ERR_CHOICES, ERR_DPHI, ERR_SORT = 1, 2, 4, 8, 16
MM2MEV = np.float32(3. / 10.)                # HelixFilter's `float mm2MeV = 3./10.`


def _mask(names):
    m = 0
    for n in names:
        m |= 1 << SHF_BITS[n]
    return m


# --- settings and geometry (host) ------------------------------------------------------------------------------------

def settings(finder, merger):
    """CalHelixFinder and MergeHelices module tables -> kernel constants (numpy scalars, static)."""
    a = finder["HelixFinderAlg"]
    if int(a["initDfDz"]) != 0:
        raise ValueError("device CalHelixFinder supports initDfDz 0")
    if not merger.get("UseCalo", True) or not merger.get("SelectBest", True):
        raise ValueError("device MergeHelices supports UseCalo true, SelectBest true")
    f = lambda k: np.float32(a[k])
    minarea = f("minArea")
    sig = f("sigmaPhi")
    te = np.float32(5. / math.sqrt(12.))
    te2 = np.float32(np.float64(te) * 1.5)
    nb = 50
    minx, maxx = np.float32(0.0), np.float32(0.025)
    stepx = np.float32((maxx - minx) / np.float32(nb))
    hel = tuple(int((h > 0) - (h < 0)) for h in finder["Helicities"])
    if not 1 <= len(hel) <= 2:
        raise ValueError("device CalHelixFinder supports one or two helicities")
    s = {"hsel": _mask(a["HelixFitSelectionBits"]), "bkgsel": _mask(a["BackgroundSelectionBits"]),
         "maxe": f("maxElectronHitEnergy"), "minnh": int(a["minNHit"]), "absmp": f("mostProbableDfDz"),
         "maxdfdz": f("maxDfDz"), "mindfdz": f("minDfDz"), "sigphi": sig, "wxy": f("weightXY"),
         "targetcon": int(a["targetconsistent"]), "wzphi": f("weightZPhi"), "w3d": f("weight3D"),
         "maxxdphi": f("maxXDPhi"), "maxpdphi": f("maxPanelToHelixDPhi"), "distpr": f("distPatRec"),
         "mindist": f("mindist"), "pmin": f("minP"), "pmax": f("maxP"), "tdmin": f("minAbsTanDip"),
         "tdmax": f("maxAbsTanDip"), "filter": bool(a["filter"]), "hitchi2": f("hitChi2Max"),
         "chi2xy": f("chi2xyMax"), "chi2zphi": f("chi2zphiMax"), "chi2h3d": f("chi2hel3DMax"),
         "dfdzerr": f("dfdzErr"), "maxratio": f("maxNHitsRatio"), "procall": bool(a["procAllTCs"]),
         "slopelim": f("slopeRatioLimit"), "minarea2": np.float32(minarea * minarea),
         "wphi0": np.float32(1. / np.float64(np.float32(sig * sig))), "tv1": np.float32(te * te),
         "tv2": np.float32(te2 * te2), "minx": minx, "maxx": maxx, "stepx": stepx, "nb": nb,
         "ixmax": int(np.float32(maxx / stepx)), "maxedep": np.float32(finder["maxEDepAvg"]),
         "minnhtc": int(finder["minNHitsTimeCluster"]), "helicities": hel,
         # MergeHelices
         "minnover": int(merger["MinNHitOverlap"]), "minfrac": np.float32(merger["MinHitOverlapFraction"]),
         "deltanh": int(merger["deltanh"]), "scalexy": np.float32(merger["scaleXY"]),
         "scalezphi": np.float32(merger["scaleZPhi"]), "bad": _mask(merger.get("BadHitFlags", ["Outlier"]))}
    return s


def _poly_atan2_np(y, x):
    f = np.float32
    y, x = f(y), f(x)
    at = lambda z: (f(0.97239411) + f(-0.19194795) * z * z) * z
    if x != 0:
        if abs(x) > abs(y):
            z = f(y / x)
            if x > 0:
                return at(z)
            return f(float(at(z)) + math.pi) if y >= 0 else f(float(at(z)) - math.pi)
        z = f(x / y)
        return -at(z) + f(math.pi / 2.) if y > 0 else -at(z) - f(math.pi / 2.)
    return f(math.pi / 2.) if y > 0 else (-f(math.pi / 2.) if y < 0 else f(0.))


def _order_id(station, plane, panel):
    face = 0 if panel % 2 == 0 else 1
    if station % 2 == 0:
        oface = 1 - face if plane == 0 else face + 2
    else:
        oface = face if plane == 0 else 3 - face
    return station, oface, int(panel / 2)


def module_geometry(straw_mid_z, panel_origin):
    """CalHelixFinder::beginRun: zFace [72] and phiPanel [216] (float32) from the straw-0 / straw-1 midpoint z of each
    (plane, panel) and the panel origin (x, y); each face index keeps its last panel's write, as stock."""
    zface = np.zeros(NTOTALFACES, np.float32)
    phipanel = np.zeros(NTOTALFACES * KNPANELS, np.float32)
    for ipl in range(36):
        for ipn in range(6):
            os_, of, op = _order_id(ipl // 2, ipl % 2, ipn)
            fid = of + os_ * 4
            zface[fid] = np.float32((float(straw_mid_z[ipl][ipn][0]) + float(straw_mid_z[ipl][ipn][1])) / 2.0)
            ox, oy = panel_origin[ipl][ipn]
            p = float(_poly_atan2_np(np.float32(oy), np.float32(ox)))
            while p >= TWO_PI:
                p -= TWO_PI
            while p < 0.0:
                p += TWO_PI
            phipanel[fid * KNPANELS + op] = np.float32(p)
    return zface, phipanel


def geometry(cond, bz=1.0):
    """The geometry inputs of the finder, in one place:
    - zFace from the nominal straw midpoint z (helix.nominal_straw_z, the geometry HelixTool reads);
    - phiPanel from the aligned straw-0 midpoints of the conditions (PROVISIONAL: stock reads Panel::origin());
    - bz: the field at the tracker origin (PROVISIONAL 1.0 T; stock reads the BFieldManager);
    - fCaloOffset from the calorimeter G4 parameters (calogeom.CALO_GEOM) as CalHelixFinderAlg::setCalorimeter;
    - the disk origins (calogeom.build) and the tracker centre (-solenoidOffset, 0, detectorSystemZ0)."""
    from . import calogeom, cpr, helix
    g = helix.TRACKER_GEOM
    mz = [[[helix.nominal_straw_z(g, p, q, s) for s in (0, 1)] for q in range(6)] for p in range(36)]
    sx, sy = np.asarray(cond["straw"]["midx"]), np.asarray(cond["straw"]["midy"])
    po = [[(float(sx[(p * 6 + q) * 96]), float(sy[(p * 6 + q) * 96])) for q in range(6)] for p in range(36)]
    zf, pp = module_geometry(mz, po)
    cg = calogeom.CALO_GEOM
    case = cg["crystalZLength"] + cg["crystalCapZLength"]
    off = np.float32(case / 2.0 + (cg["BPPipeZOffset"] + cg["BPHoleZLength"] + cg["FEEZLength"]) / 2.0
                     - cg["FPCarbonZLength"] - cg["FPFoamZLength"])
    return {"zFace": zf, "phiPanel": pp, "bz": float(bz), "fCaloOffset": off,
            "disk_origin": np.asarray(calogeom.build()["origin"], np.float64),
            "tracker_center": np.array([-cg["solenoidOffset"], 0.0, cpr.DETECTOR_SYSTEM_Z0])}


def _radius_limits(s, bz):
    pb = np.float32(abs((C_LIGHT * 1e-3) / (float(np.float32(bz)) * -1.0)))
    rmin = np.float32(float(s["pmin"]) / (float(pb) * math.sqrt(1.0 + float(np.float32(s["tdmax"] * s["tdmax"])))))
    rmax = np.float32(float(s["pmax"]) / (float(pb) * math.sqrt(1.0 + float(np.float32(s["tdmin"] * s["tdmin"])))))
    return rmin, rmax


# --- small helpers ---------------------------------------------------------------------------------------------------

def _w(c, a, b):
    return jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)


def _cint(x):
    """C++ float/double -> int (truncation; x86 gives INT_MIN for NaN and out-of-range values), as int64."""
    xd = _d(x)
    bad = (xd != xd) | (xd >= 2147483648.0) | (xd <= -2147483649.0)
    return jnp.where(bad, jnp.int64(-2147483648), jnp.trunc(jnp.where(bad, 0.0, xd)).astype(I64))


def _nrot(dphi):
    return _cint(jnp.abs((_d(dphi) + PI) / TWO_PI))


def _twopi(n):
    return _m(_d(n), 2.0, PI)                # D(n) * D(2.) * M_PI


def _pos2pi(phi):
    return jnp.where(phi < 0, _f(_d(phi) + TWO_PI), phi)


def _wrap_d(dp):
    dp = jnp.where(_d(dp) > PI, _f(_d(dp) - TWO_PI), dp)
    return jnp.where(_d(dp) < -PI, _f(_d(dp) + TWO_PI), dp)


def _fmin(v, mask, start):
    """`if (v < best) {best = v; i = k;}` from best = start in index order: (index or -1, best)."""
    mk = mask & (v == v)
    vv = jnp.where(mk, v, jnp.asarray(np.inf, v.dtype))
    mn = jnp.min(vv)
    hit = mk & (vv == mn) & (mn < start)
    i = jnp.where(jnp.any(hit), _argmax(hit), -1)
    return i, jnp.where(i >= 0, mn, jnp.asarray(start, v.dtype))


def _fmax(v, mask, start):
    mk = mask & (v == v)
    vv = jnp.where(mk, v, jnp.asarray(-np.inf, v.dtype))
    mx = jnp.max(vv)
    hit = mk & (vv == mx) & (mx > start)
    i = jnp.where(jnp.any(hit), _argmax(hit), -1)
    return i, jnp.where(i >= 0, mx, jnp.asarray(start, v.dtype))


def _seqsum(t, mask, init, on=True):
    """init + t[i] over the masked rows, one at a time in row order (the stock accumulation order)."""
    L = mask.shape[0]
    idx = jnp.nonzero(mask, size=L, fill_value=0)[0]
    cnt = jnp.where(on, jnp.sum(mask), 0).astype(I32)
    tt = t[idx]
    return jax.lax.fori_loop(0, cnt, lambda i, s: s + tt[i], init)


def _flag(cond, bit):
    return jnp.where(cond, I32(bit), I32(0))


# --- LsqSums4 (double, offsets 0): [qn, sw, sx, sy, sx2, sxy, sy2, sx3, sx2y, sxy2, sy3, sx4, sx3y, sx2y2, sxy3, sy4]

def _terms(X, Y, W):
    X, Y, W = _d(X), _d(Y), _d(W)
    X2, Y2, XY = _m(X, X), _m(Y, Y), _m(X, Y)
    WX, WY, WX2, WY2 = _m(W, X), _m(W, Y), _m(W, X2), _m(W, Y2)
    t = [jnp.ones_like(X), W, WX, WY, WX2, _m(W, XY), WY2, _m(WX2, X), _m(WX2, Y), _m(WY2, X), _m(WY2, Y),
         _m(WX2, X2), _m(WX2, XY), _m(WX2, Y2), _m(WY2, XY), _m(WY2, Y2)]
    return jnp.stack(jnp.broadcast_arrays(*t), -1)


def _ls_add(s, X, Y, W, on=True):
    return jnp.where(on, s + _terms(X, Y, W), s)


def _ls_rem(s, X, Y, W, on=True):
    return jnp.where(on, s - _terms(X, Y, W), s)


def _lsq(s):
    """The LsqSums4 results: x0, y0, radius, chi2DofCircle, chi2DofLine, phi0, dfdz (double)."""
    qn, sw, sx, sy, sx2, sxy, sy2, sx3, sx2y, sxy2, sy3, sx4, sx3y, sx2y2, sxy3, sy4 = jnp.moveaxis(s, -1, 0)
    xm, ym, x2m, xym, y2m = sx / sw, sy / sw, sx2 / sw, sxy / sw, sy2 / sw
    x3m, x2ym, xy2m, y3m = sx3 / sw, sx2y / sw, sxy2 / sw, sy3 / sw
    x4m, x2y2m, y4m = sx4 / sw, sx2y2 / sw, sy4 / sw
    sXX, sXY, sYY = x2m - _m(xm, xm), xym - _m(xm, ym), y2m - _m(ym, ym)
    sX2X, sX2Y, sXY2, sYY2 = x3m - _m(xm, x2m), x2ym - _m(ym, x2m), xy2m - _m(xm, y2m), y3m - _m(ym, y2m)
    sX2X2, sX2Y2, sY2Y2 = x4m - _m(x2m, x2m), x2y2m - _m(x2m, y2m), y4m - _m(y2m, y2m)
    det = _m(sXX, sYY) - _m(sXY, sXY)
    a, b = sX2X + sXY2, sX2Y + sYY2
    # chained divisions are guarded: XLA rewrites (a / b) / c as a / (b * c) (and x / sqrt(y) as x * rsqrt(y))
    x0 = _g((_m(sYY, a) - _m(sXY, b)) / 2.0) / det
    y0 = _g((_m(sXX, b) - _m(sXY, a)) / 2.0) / det
    dx, dy = xm - x0, ym - y0
    r = jnp.sqrt(sXX + sYY + _m(dx, dx) + _m(dy, dy))
    c2 = sX2X2 + _m(2.0, sX2Y2) + sY2Y2 - (_m(sYY, a, a) + _m(sXX, b, b) - _m(2.0, sXY, a, b)) / det
    c2 = _m(c2 / _m(4.0, r, r), sw / qn)
    l2 = _m((_m(sYY, sXX) - _m(sXY, sXY)) / sXX, sw / qn)
    dn = _m(sw, sx2) - _m(sx, sx)
    return {"x0": x0, "y0": y0, "r": r, "chi2c": c2, "chi2l": l2,
            "phi0": (_m(sx2, sy) - _m(sx, sxy)) / dn, "dfdz": (_m(sw, sxy) - _m(sy, sx)) / dn}


def _circ(s):
    """(f(x0), f(y0), f(radius)) of the sums."""
    q = _lsq(s)
    return _f(q["x0"]), _f(q["y0"]), _f(q["r"])


# --- hit weights -----------------------------------------------------------------------------------------------------

def _dxn_c2(P, cx, cy):
    dx, dy = P["x"] - cx, P["y"] - cy
    dxn = _m(dx, -P["udy"]) + _m(dy, P["udx"])
    return _m(dxn, dxn) / (_m(dx, dx) + _m(dy, dy))


def _calc_weight(P, cx, cy, S):
    tv = jnp.where(P["nsh"] > 1, S["tv2"], S["tv1"])
    c2 = _dxn_c2(P, cx, cy)
    e2 = _m(P["uvar"], np.float32(1) - c2) + _m(tv, c2)
    return _m(_f(1.0 / _d(e2)), S["wxy"])


def _calc_phi_weight(P, cx, cy, R, S):
    c2 = _dxn_c2(P, cx, cy)
    e2 = _m(P["uvar"], c2) + _m(P["vvar"], np.float32(1) - c2)
    return _m(_m(R, R) / e2, S["wzphi"])


def _radial(P, cx, cy, R):
    dx, dy = P["x"] - cx, P["y"] - cy
    return jnp.sqrt(_m(dx, dx) + _m(dy, dy)) - R


def _resolve(phih, phi_ref):
    """resolve2PiAmbiguity on the hit azimuths phih: (dphi_min, the rotated azimuth written to _hphi)."""
    dphi = phi_ref - phih
    n = _nrot(dphi)
    neg = _d(dphi) < -PI
    dmin = jnp.where(neg, _f(_d(dphi) + _twopi(n + 1)), _f(_d(dphi) - _twopi(n)))
    prot = jnp.where(neg, _f(_d(phih) - _twopi(n + 1)), _f(_d(phih) + _twopi(n)))
    return dmin, prot


# --- the finder state ------------------------------------------------------------------------------------------------

def _cleared(L, cx, cy):
    """CalHelixFinderData after clearResults (the centre is not cleared)."""
    z16 = jnp.zeros(16, F64)
    return {"used": jnp.zeros(L, I32), "sxy": z16, "szphi": z16, "nst": I32(0), "ncomb": I32(0), "nxy": I32(0),
            "nzphi": I32(0), "hchi2": F32(1e10), "cx": cx, "cy": cy, "r": F32(-1.), "cchi2": F32(1e10),
            "dfdz": F32(-1.e6), "fz0": F32(-1.e6), "dchi2": F32(1e10), "seed": I32(-1)}


FIT_KEYS = ("used", "sxy", "szphi", "nst", "ncomb", "nxy", "nzphi", "hchi2", "cx", "cy", "r", "cchi2", "dfdz", "fz0",
            "dchi2", "seed")


def _dwcf(P, S, T, s0, hcx, hcy, R, on):
    """doWeightedCircleFit from the walk start s0: (T, hcx, hcy, R) as stock writes them through its references."""
    pos = jnp.arange(P["x"].shape[0])
    init = jnp.zeros(16, F64)
    n0 = I32(0)
    cal = P["ct"] > 0
    init = _ls_add(init, P["cX"], P["cY"], 1.0 / 100.0, cal)
    n0 = n0 + cal.astype(I32)
    if S["targetcon"] == 1:
        init = _ls_add(init, 0.0, 0.0, 1.0 / 900.0)
        n0 = n0 + 1
    mk = (T["used"] == 1) & (pos >= s0) & P["valid"]
    wt = _calc_weight(P, hcx, hcy, S)
    sxy = _seqsum(_terms(P["x"], P["y"], wt), mk, init, on)
    q = _lsq(sxy)
    cx, cy, r, c2 = _f(q["x0"]), _f(q["y0"]), _f(q["r"]), _f(q["chi2c"])
    T2 = {**T, "sxy": sxy, "nxy": n0 + jnp.sum(jnp.where(mk, P["nsh"], 0)).astype(I32),
          "ncomb": n0 + jnp.sum(mk).astype(I32), "cx": cx, "cy": cy, "r": r, "cchi2": c2,
          "xyw": jnp.where(mk, wt, T["xyw"])}
    return _w(on, T2, T), jnp.where(on, cx, hcx), jnp.where(on, cy, hcy), jnp.where(on, r, R)


def _remove_xy(P, T, iw, on):
    i = jnp.maximum(iw, 0)
    return _w(on, {**T, "sxy": _ls_rem(T["sxy"], P["x"][i], P["y"][i], T["xyw"][i]),
                   "nxy": T["nxy"] - P["nsh"][i], "used": T["used"].at[i].set(0), "ncomb": T["ncomb"] - 1,
                   "nst": T["nst"] - P["nsh"][i]}, T)


def _refine(P, S, T, s0, on):
    """refineHelixParameters (its failure branch is unreachable: the residual loop exits with no worst hit), as one
    loop with one refit: step 0 the weighted fit from the sums' circle; step 1 the entry test and cleanUpCircle
    (whose minimum starts at -1); then the residual loop, a refit after each removal."""
    pos = jnp.arange(P["x"].shape[0])

    def cond(c):
        return c[5]

    def body(c):
        T, hcx, hcy, r, ph, live = c
        mk = (T["used"] == 1) & (pos >= s0) & P["valid"]
        dr = _radial(P, hcx, hcy, r)
        iw, wv = _fmax(_m(dr, dr, T["xyw"]), mk, S["hitchi2"])
        rows = T["sxy"][None, :] - _terms(P["x"], P["y"], T["xyw"])
        ic, _ = _fmin(_f(_lsq(rows)["chi2c"]), mk, np.float32(-1.))
        chi2 = _f(_lsq(T["sxy"])["chi2c"])
        enter = ~((chi2 <= S["chi2xy"]) & (wv <= S["hitchi2"]))
        rmi = jnp.where(ph == 1, ic, jnp.where(ph == 2, iw, -1))
        rm = live & (ph > 0) & ~((ph == 1) & ~enter) & (rmi >= 0)
        T = _remove_xy(P, T, rmi, rm)
        hx, hy, hr = _circ(T["sxy"])
        T, hcx, hcy, r = _dwcf(P, S, T, s0, hx, hy, hr, live & ((ph == 0) | rm))
        live2 = jnp.where(ph == 0, True, jnp.where(ph == 1, enter, rm))
        return T, hcx, hcy, r, jnp.where(ph == 0, I32(1), I32(2)), live & live2

    hcx, hcy, r = _circ(T["sxy"])
    T, _, _, _, _, _ = jax.lax.while_loop(cond, body, (T, hcx, hcy, r, I32(0), on))
    q = _lsq(T["sxy"])
    return _w(on, {**T, "cx": _f(q["x0"]), "cy": _f(q["y0"]), "r": _f(q["r"]), "cchi2": _f(q["chi2c"])}, T)


def _find_dfdz(P, A, S, T, s0, on):
    """findDfDz: (hdfdz, hphi0, err)."""
    L = P["x"].shape[0]
    pos = jnp.arange(L)
    cx, cy = T["cx"], T["cy"]
    ns = NSTATIONS
    phicl = _pos2pi(poly_atan2(P["cY"] - cy, P["cX"] - cx))
    pv = jnp.zeros(ns + 1, F32).at[ns].set(phicl)
    zv = jnp.zeros(ns + 1, F32).at[ns].set(P["cZ"])
    nh = jnp.zeros(ns + 1, I32).at[ns].set(1)
    mk = (T["used"] == 1) & (pos >= s0) & P["valid"]
    phis = _pos2pi(poly_atan2(P["y"] - cy, P["x"] - cx))
    idx = jnp.nonzero(mk, size=L, fill_value=0)[0]
    cnt = jnp.where(on, jnp.sum(mk), 0).astype(I32)

    def hit(i, c):
        pv, zv, nh, nsw = c
        j = idx[i]
        st = P["station"][j]
        phi = phis[j]
        zv = zv.at[st].set(zv[st] + P["z"][j])
        first = nh[st] == 0
        dphi = phi - pv[st]
        n = _nrot(dphi)
        phc = jnp.where(_d(dphi) > PI, _f(_d(phi) - _twopi(n)),
                        jnp.where(_d(dphi) < -PI, _f(_d(phi) + _twopi(n + 1)), phi))
        pvn = (_m(pv[st], _f(nh[st])) + phc) / _f(nh[st] + 1)
        pv = pv.at[st].set(jnp.where(first, phi, pvn))
        return pv, zv, nh.at[st].add(1), nsw + first.astype(I32)

    pv, zv, nh, nsw = jax.lax.fori_loop(0, cnt, hit, (pv, zv, nh, I32(0)))
    zv = jnp.where((jnp.arange(ns + 1) < ns) & (nh > 1), zv / _f(nh), zv)
    ii, jj = np.triu_indices(ns + 1, 1)
    ii, jj = ii[ii < ns], jj[ii < ns]
    pok = (nh[ii] != 0) & (nh[jj] != 0)
    dphi = pv[jj] - pv[ii]
    dz = zv[jj] - zv[ii]
    skip = _d(jnp.abs(dz)) < 10e-10
    throw = jnp.any(pok & ~skip & (_d(dz) < 0.0))
    dphidz = _m(dphi / dz, A["sign"])
    dx = _f(TWO_PI / _d(dz))
    nminf = (S["mindfdz"] - dphidz) / dx
    nmaxf = (S["maxdfdz"] - dphidz) / dx
    nmin = _cint(jnp.where(nminf < 0, nminf, nminf + np.float32(1)))
    nmax = _cint(jnp.where(nmaxf < 0, nmaxf - np.float32(1), nmaxf))
    nch = jnp.where((nmaxf < nminf) | (nmax < nmin), 0, nmax - nmin + 1)
    use = pok & ~skip & (nch > 0)
    ovf = jnp.any(use & (nch > NCHOICES))
    c = jnp.arange(NCHOICES)
    cuse = use[:, None] & (c[None, :] < nch[:, None])
    xx = dphidz[:, None] + _m(_f(nmin[:, None] + c[None, :]), dx[:, None])
    b = jnp.clip(_cint((xx - S["minx"]) / S["stepx"]), 0, S["nb"] - 1)
    hist = jnp.zeros(S["nb"] + 1, I32).at[jnp.where(cuse, b, S["nb"])].add(1)[:S["nb"]].astype(F32)
    nix = max(S["ixmax"] - 1, 0)
    xmp = np.float32(0)
    if nix > 0:
        ix = jnp.arange(nix)
        sw = hist[ix] + hist[ix + 1]
        k, swk = _fmax(sw, jnp.ones(nix, bool), np.float32(0))
        kk = jnp.maximum(k, 0)
        xm_ = _f((_m(_d(S["stepx"]), _d(kk) + 0.5, _d(hist[kk])) + _m(_d(S["stepx"]), _d(kk + 1) + 0.5,
                                                                       _d(hist[kk + 1]))) / _d(swk))
        xmp = jnp.where(k >= 0, xm_, np.float32(0))
    hdfdz = jnp.where(nsw < 2, A["mp"], _m(xmp, A["sign"]))
    nz = nh != 0
    i0 = _argmax(nz)
    phi0 = pv[i0] - _m(zv[i0], hdfdz)
    sh = phi0 + _m(zv, hdfdz)
    dp = pv - sh
    n = _nrot(dp)
    dmin = jnp.where(_d(dp) < -PI, _f(_d(dp) + _twopi(n + 1)), _f(_d(dp) - _twopi(n)))
    rest = nz & (jnp.arange(ns + 1) != i0)
    sdphi = jax.lax.fori_loop(0, ns + 1, lambda i, s: jnp.where(rest[i], s + dmin[i], s), np.float32(0))
    hphi0 = phi0 + sdphi / _f(jnp.sum(nz))
    err = _flag(on & throw, ERR_THROW) | _flag(on & ovf, ERR_CHOICES)
    return hdfdz, hphi0, err


def _dlfp(P, A, S, T, s0, uiw, on):
    """doLinearFitPhiZ (cleanup 1) from the walk start s0: (T, success, err)."""
    L = P["x"].shape[0]
    pos = jnp.arange(L)
    saved = T["used"]
    hcx, hcy, rad = _circ(T["sxy"])
    has = T["sxy"][0] > 0
    hcx, hcy, rad = jnp.where(has, hcx, T["cx"]), jnp.where(has, hcy, T["cy"]), jnp.where(has, rad, T["r"])
    # addCaloClusterToFitPhiZ
    phicl0 = poly_atan2(P["cY"] - hcy, P["cX"] - hcx)
    dphi = (T["fz0"] + _m(P["cZ"], T["dfdz"])) - phicl0
    n = _nrot(dphi)
    neg = _d(dphi) < -PI
    dmin = jnp.where(neg, _f(_d(dphi) + _twopi(n + 1)), _f(_d(dphi) - _twopi(n)))
    phicl = jnp.where(neg, _f(_d(phicl0) - _twopi(n + 1)), _f(_d(phicl0) + _twopi(n)))
    addc = _d(jnp.abs(dmin) / S["sigphi"]) < 2.0 * float(S["maxxdphi"])
    szphi = _ls_add(jnp.zeros(16, F64), P["cZ"], phicl, float(np.float32(784.)), addc)
    nzphi = addc.astype(I32)
    phih = poly_atan2(P["y"] - hcy, P["x"] - hcx)
    if uiw:
        wv = _calc_phi_weight(P, hcx, hcy, rad, S)
        err = jnp.where(_d(wv) > 0, _f(1.0 / _d(_g(jnp.sqrt(wv)))), S["sigphi"])
    else:
        wv = jnp.full(L, S["wphi0"], F32)
        err = jnp.full(L, S["sigphi"], F32)
    k0 = P["fk"][s0]

    def fcond(c):
        return on & (c[0] < P["nf"])

    def fbody(c):
        k, used, szphi, nzphi, cnt, idf, ip0, zlast, hphi, zw = c
        zfc = P["zfk"][k]
        inf_ = (pos >= P["fstart"][k]) & (pos < P["fend"][k]) & (pos >= s0)
        dz = zfc - zlast
        dmin, prot = _resolve(phih, _m(zfc, idf) + ip0)
        hphi = jnp.where(inf_, prot, hphi)
        zw = jnp.where(inf_, wv, zw)
        xd = jnp.abs(dmin) / err
        u1 = inf_ & (used == 1)
        rem = u1 & (xd > S["maxxdphi"])
        used = jnp.where(rem, 0, used)
        g, fchi2 = _fmin(xd, u1 & ~rem, np.float32(1e10))
        fd = g >= 0
        gi = jnp.maximum(g, 0)
        szphi = _ls_add(szphi, zfc, hphi[gi], zw[gi], fd)
        nzphi = nzphi + jnp.where(fd, P["nsh"][gi], 0)
        cnt = cnt + fd.astype(I32)
        one = fd & (cnt == 1)
        zlast = jnp.where(one, zfc, zlast)
        dz = jnp.where(one, np.float32(0), dz)
        q = _lsq(szphi)
        upd = fd & (cnt >= 5) & (_d(fchi2) < 2.0) & (jnp.abs(_d(idf) - q["dfdz"]) < 8.e-4) & \
            (_m(q["dfdz"], _d(A["sign"])) > 0.0) & (-_d(dz) >= _d(S["mindist"]))
        idf = jnp.where(upd, _f(q["dfdz"]), idf)
        ip0 = jnp.where(upd, _f(q["phi0"]), ip0)
        zlast = jnp.where(upd, zfc, zlast)
        return k + 1, used, szphi, nzphi, cnt, idf, ip0, zlast, hphi, zw

    c0 = (k0, T["used"], szphi, nzphi, I32(0), T["dfdz"], T["fz0"], np.float32(0), T["hphi"], T["zw"])
    _, used, szphi, nzphi, _, _, _, _, hphi, zw = jax.lax.while_loop(fcond, fbody, c0)
    # cleanup: worst chi2 hit, then the residual loop, until the chi2 is good or too few hits remain
    base = (pos >= s0) & P["valid"]
    q = _lsq(szphi)
    enter = on & (szphi[0] >= 5.0) & (q["chi2l"] > _d(S["chi2zphi"]))

    def ccond(c):
        return c[4] & (c[5] < LOOP_CAP)

    def cbody(c):
        used, szphi, nzphi, chi2min, live, it, phase = c
        mk = (used == 1) & base
        rows = szphi[None, :] - _terms(P["zf"], hphi, zw)
        i0_, c0_ = _fmin(_f(_lsq(rows)["chi2l"]), mk, np.float32(1e10))
        q = _lsq(szphi)
        ph = _f(_m(_d(P["zf"]), q["dfdz"]) + q["phi0"])
        i1, _ = _fmax(jnp.abs(hphi - ph) / err, mk, S["maxxdphi"])
        p0 = phase == 0
        iw = jnp.where(p0, i0_, i1)
        rm = (iw >= 0) & (nzphi > S["minnh"])
        wi = jnp.maximum(iw, 0)
        used2 = jnp.where(rm, used.at[wi].set(0), used)
        sz2 = _ls_rem(szphi, P["zf"][wi], hphi[wi], zw[wi], rm)
        nz2 = nzphi - jnp.where(rm, P["nsh"][wi], 0)
        c_rm = _f(_lsq(sz2)["chi2l"])
        # phase 0: chi2min from the removal or the search; then the residual phase
        chi_p0 = jnp.where(rm, c_rm, c0_)
        # phase 1 without a removal: the end of the residual loop
        end1 = ~p0 & ~rm
        chi_e = jnp.where(nz2 <= S["minnh"], c_rm, chi2min)
        again = ((chi_e >= S["chi2zphi"]) | (iw >= 0)) & (nz2 > S["minnh"])
        chi2min = jnp.where(p0, chi_p0, jnp.where(rm, c_rm, chi_e))
        live2 = ~end1 | again
        phase2 = jnp.where(end1, I32(0), I32(1))
        return used2, sz2, nz2, chi2min, live & live2, it + 1, phase2

    cc = (used, szphi, nzphi, np.float32(0), enter, I32(0), I32(0))
    used, szphi, nzphi, _, live, _, _ = jax.lax.while_loop(ccond, cbody, cc)
    q = _lsq(szphi)
    ok = (nzphi >= S["minnh"]) & (q["chi2l"] < _d(S["chi2zphi"]))
    bad = (szphi[0] < 5.0) | (_m(q["dfdz"], _d(A["sign"])) < 0.0)
    ok = ok & ~bad
    T2 = {**T, "used": jnp.where(ok, used, saved), "szphi": szphi, "nzphi": nzphi,
          "fz0": jnp.where(ok, _f(q["phi0"]), T["fz0"]), "dfdz": jnp.where(ok, _f(q["dfdz"]), T["dfdz"]),
          "dchi2": jnp.where(ok, _f(q["chi2l"]), np.float32(-1.)), "hphi": hphi, "zw": zw}
    return _w(on, T2, T), on & ok, _flag(on & live, ERR_CLEANUP)


# --- calculateTrackParameters and findTrack --------------------------------------------------------------------------

def _track_params(P, s, A, S):
    """calculateTrackParameters(p1 = target (0, 0, 0), p2 = seed hit, p3 = calo cluster): (ok, cx, cy, r, phi0,
    dfdz, err)."""
    z0 = np.float32(0)
    p1 = (z0, z0, z0)
    p2 = (P["x"][s], P["y"][s], P["z"][s])
    p3 = (P["cX"], P["cY"], P["cZ"])

    def mag2(a, b):
        dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
        return _m(dx, dx) + _m(dy, dy) + _m(dz, dz)

    dij, dik, djk = mag2(p1, p2), mag2(p1, p3), mag2(p2, p3)
    area2 = _f(_d(_m(dij, djk) + _m(dik, djk) + _m(dij, dik))
               - _m(0.5, _d(_m(dij, dij) + _m(djk, djk) + _m(dik, dik))))
    ok = ~(area2 < S["minarea2"])
    half = lambda a, b: _f(_d(a + b) / 2.0)
    slope = lambda a, b: _f(-_d(a[0] - b[0]) / _d(a[1] - b[1]))
    x_m, y_m = half(p3[0], p1[0]), half(p3[1], p1[1])
    x_n, y_n = half(p3[0], p2[0]), half(p3[1], p2[1])
    m = slope(p3, p1)
    c = y_m - _m(x_m, m)
    k = slope(p3, p2)
    t = y_n - _m(x_n, k)
    fin = jnp.isfinite(m) & jnp.isfinite(k)
    ksafe = jnp.where(k != 0, k, np.float32(1))
    q = m / ksafe
    alt = fin & (_d(jnp.abs(k)) > 1e-9) & (q > 0) & ((q - _cint(q).astype(F32)) > S["slopelim"])
    x_n = jnp.where(alt, half(p1[0], p2[0]), x_n)
    y_n = jnp.where(alt, half(p1[1], p2[1]), y_n)
    k = jnp.where(alt, slope(p1, p2), k)
    t = jnp.where(alt, y_n - _m(x_n, k), t)
    bad = ~jnp.isfinite(m) | ~jnp.isfinite(k) | (_d(jnp.abs(m - k)) < 1e-6)
    ok = ok & ~bad
    x0 = (t - c) / (m - k)
    y0 = _m(m, x0) + c
    dx3, dy3, dz32 = p3[0] - x0, p3[1] - y0, p3[2] - p2[2]
    radius = jnp.sqrt(_m(dx3, dx3) + _m(dy3, dy3))
    phi0 = poly_atan2(p2[1] - y0, p2[0] - x0)
    dphi32 = poly_atan2(dy3, dx3) - phi0
    dphi32 = jnp.where(_d(_m(dphi32, A["sign"])) < 0.0, _f(_d(dphi32) + TWO_PI), dphi32)
    mp = _d(A["mp"])
    dfdz = dphi32 / dz32
    diff = jnp.abs(dfdz - A["mp"])

    def loop(d, dfdz, diff, sg):
        dt = lambda d: _f(jnp.abs((_d(d) + sg * TWO_PI) / _d(dz32) - mp))

        def cond(c):
            return ok & (c[3] < c[2]) & (c[4] < LOOP_CAP)

        def body(c):
            d, _, _, _, it = c
            d = _f(_d(d) + sg * TWO_PI)
            df = d / dz32
            return d, df, jnp.abs(df - A["mp"]), dt(d), it + 1

        return jax.lax.while_loop(cond, body, (d, dfdz, diff, dt(d), I32(0)))

    dphi32, dfdz, diff, _, it1 = loop(dphi32, dfdz, diff, 1.0)
    dphi32, dfdz, diff, _, it2 = loop(dphi32, dfdz, diff, -1.0)
    dfdz = jnp.where((jnp.abs(dfdz) < S["mindfdz"]) | (jnp.abs(dfdz) > S["maxdfdz"]), A["mp"], dfdz)
    err = _flag(ok & ((it1 >= LOOP_CAP) | (it2 >= LOOP_CAP)), ERR_DPHI)
    return ok, x0, y0, radius, phi0, dfdz, err


def _dphidz2(P, ulist, cnt, X0, Y0, on):
    """calculateDphiDz_2 over the used hits (seed first, then one per face, in walk order)."""
    phi = poly_atan2(P["y"] - Y0, P["x"] - X0)
    phi0 = phi[ulist[0]]
    phi = jnp.where(_d(phi - phi0) > PI, _f(_d(phi) - TWO_PI), phi)
    phi = jnp.where(_d(phi - phi0) < -PI, _f(_d(phi) + TWO_PI), phi)
    X, Y = _d(P["z"]), _d(phi)
    t = jnp.stack([X, Y, _m(X, X), _m(X, Y)], -1)[ulist]
    n = jnp.where(on, cnt, 0)
    sx, sy, sx2, sxy = jax.lax.fori_loop(0, n, lambda i, s: s + t[i], jnp.zeros(4, F64))
    sw = _d(cnt)
    dydx = (_m(sw, sxy) - _m(sy, sx)) / (_m(sw, sx2) - _m(sx, sx))
    return _f(jnp.where(sw > 1.0, dydx, 0.0))


def _find_track(P, A, S, s, use_mpv, hd_in, cxp, cyp, act):
    """findTrack from seed hit s on a cleared TmpHelix: (T, reached findDfDz, hdfdz, hphi0, err)."""
    L = P["x"].shape[0]
    pos = jnp.arange(L)
    ok, cx, cy, rad, phi0, dfdz, err = _track_params(P, s, A, S)
    ok = ok & act
    toll = lambda d: jnp.where(d != 0, _f(jnp.abs(TWO_PI / _d(d))), np.float32(1.e6))
    tmax = toll(dfdz)
    dfdz = jnp.where(use_mpv, hd_in, dfdz)
    tmax = jnp.where(use_mpv, toll(dfdz), tmax)
    sxy = _ls_add(jnp.zeros(16, F64), P["x"][s], P["y"][s], 1.0)
    sxy = _ls_add(sxy, P["cX"], P["cY"], 1.0)
    sxy = _ls_add(sxy, 0.0, 0.0, float(np.float32(0.1)))
    used = (pos == s).astype(I32)
    sfz = P["zf"][s]
    ulist = jnp.zeros(L, I32).at[0].set(s)

    def cond(c):
        return ok & (c[0] < P["nf"])

    def body(c):
        k, cx, cy, rad, phi0, dfdz, sxy, used, npt, nco, ulist, zphi0, lastz, mode, rt, trip = c
        zfc = P["zfk"][k]
        dz_ = zfc - lastz
        phi = phi0 + _m(dz_, dfdz)
        hx = cx + _m(rad, jnp.cos(phi))
        hy = cy + _m(rad, jnp.sin(phi))
        hphi = _pos2pi(poly_atan2(hy, hx))
        inf_ = (pos >= P["fstart"][k]) & (pos < P["fend"][k]) & (pos > s)
        pok = ~(jnp.abs(_wrap_d(hphi - P["pp"])) > S["maxpdphi"])
        ddx, ddy = hx - P["x"], hy - P["y"]
        dist = jnp.sqrt(_m(ddx, ddx) + _m(ddy, ddy))
        dmax = S["distpr"] + _m(S["dfdzerr"], jnp.abs(dz_))
        g, _ = _fmin(dist, inf_ & pok & (dist <= dmax), np.float32(1e10))
        fd = g >= 0
        gi = jnp.maximum(g, 0)
        used = jnp.where(fd, used.at[gi].set(1), used)
        ulist = jnp.where(fd, ulist.at[nco].set(gi), ulist)
        nco = nco + fd.astype(I32)
        npt = npt + jnp.where(fd, P["nsh"][gi], 0)
        sxy = _ls_add(sxy, P["x"][gi], P["y"][gi], 1.0, fd)
        ccx, ccy, ccr = _circ(sxy)
        m1 = fd & (mode == 1)
        cx, cy, rad = jnp.where(m1, ccx, cx), jnp.where(m1, ccy, cy), jnp.where(m1, ccr, rad)
        phi0 = jnp.where(fd, poly_atan2(P["y"][gi] - cy, P["x"][gi] - cx), phi0)
        zphi0 = jnp.where(fd, zfc, zphi0)
        lastz = jnp.where(fd, zfc, lastz)
        dzs = _f(-_d(zfc - sfz))
        trip = fd & rt & (dzs > np.float32(100.)) & (dzs < tmax)       # `trip` is reset at every face
        blk = fd & trip & (nco >= 2) & ~use_mpv
        cx, cy, rad = jnp.where(blk, ccx, cx), jnp.where(blk, ccy, cy), jnp.where(blk, ccr, rad)
        phi0 = jnp.where(blk, poly_atan2(P["y"][gi] - cy, P["x"][gi] - cx), phi0)
        db = _dphidz2(P, ulist, nco, cx, cy, blk)
        rng = (jnp.abs(db) > S["maxdfdz"]) | (jnp.abs(db) < S["mindfdz"])
        dfdz = jnp.where(blk, jnp.where(rng, A["mp"], db), dfdz)
        good = blk & ~rng
        rt = rt & ~good
        mode = jnp.where(good, 1, mode)
        return k + 1, cx, cy, rad, phi0, dfdz, sxy, used, npt, nco, ulist, zphi0, lastz, mode, rt, trip

    c0 = (P["fk"][s], cx, cy, rad, phi0, dfdz, sxy, used, P["nsh"][s], I32(1), ulist, P["z"][s], sfz, I32(0),
          jnp.bool_(True), jnp.bool_(False))
    _, cx, cy, rad, phi0, dfdz, sxy, used, npt, nco, _, zphi0, _, _, _, _ = jax.lax.while_loop(cond, body, c0)
    go = ok & ~(npt < S["minnh"])
    q = _lsq(sxy)
    T = {**_cleared(L, cxp, cyp), "used": used, "sxy": sxy, "nxy": npt, "r": _f(q["r"]), "cx": _f(q["x0"]),
         "cy": _f(q["y0"]), "cchi2": _f(q["chi2c"]), "nst": npt, "ncomb": nco, "dfdz": dfdz,
         "fz0": phi0 - _m(dfdz, zphi0), "dchi2": np.float32(-1.),
         "hphi": jnp.zeros(L, F32), "xyw": jnp.zeros(L, F32), "zw": jnp.zeros(L, F32)}
    T = _refine(P, S, T, s, go)
    r_end, npt2 = T["r"], T["nst"]
    hd, hp, e1 = _find_dfdz(P, A, S, T, s, go)
    T = {**T, "dfdz": hd, "fz0": hp, "dchi2": np.float32(-1.)}
    T, rc, e2 = _dlfp(P, A, S, T, s, False, go)
    mk = (T["used"] > 0) & (pos >= s) & P["valid"]
    nco = jnp.where(rc, jnp.sum(mk).astype(I32), nco)
    npt = jnp.where(rc, jnp.sum(jnp.where(mk, P["nsh"], 0)).astype(I32), npt2)
    q = _lsq(T["sxy"])
    Tf = {**T, "nst": npt, "ncomb": nco, "hchi2": jnp.where(nco > 2, _f(q["chi2c"]), np.float32(-1)),
          "seed": I32(s), "cx": _f(q["x0"]), "cy": _f(q["y0"]), "r": r_end, "cchi2": _f(q["chi2c"]),
          "fz0": jnp.where(rc, T["fz0"], hp), "dfdz": jnp.where(rc, T["dfdz"], hd),
          "dchi2": jnp.where(rc, T["dchi2"], np.float32(-1.))}
    early = {**_cleared(L, cxp, cyp), "used": used}
    out = _w(go, {k: Tf[k] for k in FIT_KEYS}, early)
    e = jnp.where(act, err | jnp.where(go, e1 | e2, I32(0)), I32(0))
    return out, go, hd, hp, e


def _search(P, A, S):
    """The two searchBestTriplet passes (most-probable slope off, then on): the best TmpHelix, hdfdz/hphi0 as the
    last findDfDz left them, err."""
    L = P["x"].shape[0]
    n, nsh_tot = P["n"], P["nfs"]
    H0 = _cleared(L, np.float32(0), np.float32(0))

    def runnable(k, tested, H):
        return (k < n) & ~(H["nst"] > nsh_tot - tested) & ~(nsh_tot - tested < S["minnh"])

    def norm(c):
        """Skip to the next search when this one stops (its stop condition only grows), then to the end."""
        ph, k, tested, H = c[:4]
        adv = (ph == 0) & ~runnable(k, tested, H)
        ph, k, tested = jnp.where(adv, 1, ph), jnp.where(adv, 0, k), jnp.where(adv, 0, tested)
        ph = jnp.where((ph == 1) & ~runnable(k, tested, H), 2, ph)
        return (ph, k, tested) + tuple(c[3:])

    def cond(c):
        return c[0] < 2

    def body(c):
        ph, k, tested, H, tcx, tcy, hd, hp, err = c
        T, reached, hd2, hp2, e = _find_track(P, A, S, k, ph == 1, hd, tcx, tcy, ph < 2)
        better = (T["nst"] > H["nst"]) | ((T["nst"] == H["nst"]) & (T["hchi2"] < H["hchi2"]))
        H = _w(better, T, H)
        c = (ph, k + 1, tested + P["nsh"][k], H, T["cx"], T["cy"], jnp.where(reached, hd2, hd),
             jnp.where(reached, hp2, hp), err | e)
        return norm(c)

    c0 = norm((I32(0), I32(0), I32(0), H0, np.float32(0), np.float32(0), A["mp"], np.float32(-9999.), I32(0)))
    _, _, _, H, _, _, hd, hp, err = jax.lax.while_loop(cond, body, c0)
    return H, hd, hp, err


# --- doPatternRecognition after the triplet search -------------------------------------------------------------------

def _rescue_before_seed(P, S, H, on):
    L = P["x"].shape[0]
    pos = jnp.arange(L)
    sd = jnp.maximum(H["seed"], 0)
    kf = P["fk"][sd]
    dfdz = H["dfdz"]
    phi0 = H["fz0"] + _m(dfdz, P["zf"][sd])

    def cond(c):
        return on & (c[0] >= 0)

    def body(c):
        k, H, x0, y0, rad, phi0, lastz, rsh, rpt = c
        zfc = P["zfk"][k]
        dz_ = zfc - lastz
        phi = phi0 + _m(dz_, dfdz)
        hx = x0 + _m(rad, jnp.cos(phi))
        hy = y0 + _m(rad, jnp.sin(phi))
        hph = _pos2pi(poly_atan2(hy, hx))
        inf_ = (pos >= P["fstart"][k]) & (pos < P["fend"][k]) & \
            ~((k == kf) & (P["op"] == P["op"][sd]) & (pos >= sd))
        pok = ~(jnp.abs(_wrap_d(hph - P["pp"])) > S["maxpdphi"])
        ddx, ddy = hx - P["x"], hy - P["y"]
        dist = jnp.sqrt(_m(ddx, ddx) + _m(ddy, ddy))
        dmax = S["distpr"] + _m(S["dfdzerr"], jnp.abs(dz_))
        cand = inf_ & pok & ~(H["used"] >= 1) & (dist <= dmax) & (dist == dist)
        vv = jnp.where(cand, dist, np.inf)
        mn = jnp.min(vv)
        hit = cand & (vv == mn) & (mn < np.float32(1e10))
        fd = jnp.any(hit)
        gi = L - 1 - _argmax(hit[::-1])
        gi = jnp.where(fd, gi, 0)
        wt = _calc_weight({kk: v[gi] for kk, v in P.items() if kk in ("x", "y", "udx", "udy", "uvar", "nsh")},
                          x0, y0, S)
        sxy = _ls_add(H["sxy"], P["x"][gi], P["y"][gi], wt, fd)
        nx, ny, nr = _circ(sxy)
        H2 = {**H, "sxy": sxy, "nxy": H["nxy"] + P["nsh"][gi], "xyw": H["xyw"].at[gi].set(wt),
              "used": H["used"].at[gi].set(1)}
        H = _w(fd, H2, H)
        return (k - 1, H, jnp.where(fd, nx, x0), jnp.where(fd, ny, y0), jnp.where(fd, nr, rad),
                jnp.where(fd, H["hphi"][gi], phi0), jnp.where(fd, zfc, lastz),
                rsh + jnp.where(fd, P["nsh"][gi], 0), rpt + fd.astype(I32))

    c0 = (kf, H, H["cx"], H["cy"], H["r"], phi0, P["zf"][sd], I32(0), I32(0))
    _, H2, x0, y0, rad, _, _, rsh, rpt = jax.lax.while_loop(cond, body, c0)
    H2 = {**H2, "cx": x0, "cy": y0, "r": rad, "cchi2": _f(_lsq(H2["sxy"])["chi2c"]), "ncomb": H2["ncomb"] + rpt,
          "nst": H2["nst"] + rsh}
    return _w(on, H2, H)


def _rescue(P, S, H, use_phi, on):
    """rescueHits from the first hit (seed = all faces)."""
    L = P["x"].shape[0]
    pos = jnp.arange(L)
    # step 0: the weighted fit and the chi2 test; then one hit search per step, a refit after each added hit

    def cond(c):
        return c[5] & (c[6] < LOOP_CAP)

    def body(c):
        H, hcx, hcy, r, df_ph0, live, it, passed = c
        dfdz, phi0 = df_ph0
        p1 = it > 0
        u1 = (H["used"] == 1) & P["valid"]
        fu = jnp.zeros(L + 1, I32).at[jnp.where(P["valid"], P["fk"], L)].max(u1.astype(I32))[P["fk"]] > 0
        cand0 = P["valid"] & ~fu & ~(H["used"] >= 1)
        dr = _radial(P, hcx, hcy, r)
        wt = _calc_weight(P, hcx, hcy, S)
        drc = _m(dr, dr, wt)
        phi_pred = _m(P["zf"], dfdz) + phi0
        dphi = phi_pred - H["hphi"]
        pw1 = _calc_phi_weight(P, hcx, hcy, r, S)
        dpc1 = _m(dphi, dphi, pw1)
        xp = hcx + _m(r, jnp.cos(phi_pred))
        yp = hcy + _m(r, jnp.sin(phi_pred))
        whel = _m(_calc_weight(P, xp, yp, S), S["w3d"])
        xr2 = _m(P["x"] - xp, P["x"] - xp)
        yr2 = _m(P["y"] - yp, P["y"] - yp)
        hc1 = _m(xr2 + yr2, whel)
        hc = jnp.where(use_phi, hc1, drc)
        dpc = jnp.where(use_phi, dpc1, np.float32(0))
        pw = jnp.where(use_phi, pw1, np.float32(-9999.))
        stat = cand0 & (hc < S["hitchi2"]) & (drc < S["hitchi2"]) & (dpc < S["hitchi2"])
        idx = jnp.nonzero(stat, size=L, fill_value=0)[0].astype(I32)
        cnt = jnp.where(live & p1, jnp.sum(stat), 0).astype(I32)

        def trial(i, t):
            sxy, szphi, nxy, nzphi, xyw, zw, cmin, ib, wb, pb = t
            j = idx[i]
            go = hc[j] < cmin
            x, y, zf_, hp = P["x"][j], P["y"][j], P["zf"][j], H["hphi"][j]
            sxy = _ls_add(sxy, x, y, wt[j], go)
            szphi = _ls_add(szphi, zf_, hp, pw[j], go)
            nxy = nxy + jnp.where(go, P["nsh"][j], 0)
            nzphi = nzphi + jnp.where(go, P["nsh"][j], 0)
            xyw = jnp.where(go, xyw.at[j].set(wt[j]), xyw)
            zw = jnp.where(go, zw.at[j].set(pw[j]), zw)
            acc = go & (_lsq(sxy)["chi2c"] < _d(S["chi2xy"])) & \
                (~use_phi | (_lsq(szphi)["chi2l"] < _d(S["chi2zphi"])))
            cmin = jnp.where(acc, hc[j], cmin)
            ib = jnp.where(acc, j, ib)
            wb = jnp.where(acc, wt[j], wb)
            pb = jnp.where(acc & use_phi, pw[j], pb)
            sxy = _ls_rem(sxy, x, y, wt[j], go)
            szphi = _ls_rem(szphi, zf_, hp, pw[j], go)
            return sxy, szphi, nxy, nzphi, xyw, zw, cmin, ib, wb, pb

        t0 = (H["sxy"], H["szphi"], H["nxy"], H["nzphi"], H["xyw"], H["zw"], S["hitchi2"], I32(-1),
              np.float32(-1), np.float32(-1))
        sxy, szphi, nxy, nzphi, xyw, zw, _, ib, wb, pb = jax.lax.fori_loop(0, cnt, trial, t0)
        H = _w(live & p1, {**H, "sxy": sxy, "szphi": szphi, "nxy": nxy, "nzphi": nzphi, "xyw": xyw, "zw": zw}, H)
        fd = live & p1 & (ib >= 0)
        bi = jnp.maximum(ib, 0)
        samep = P["valid"] & (P["fk"] == P["fk"][bi]) & (P["op"] == P["op"][bi]) & (H["used"] == 1)
        same = fd & jnp.any(samep)
        H = _w(same, {**H, "used": H["used"].at[bi].set(10)}, H)
        add = fd & ~same
        sxy = _ls_add(H["sxy"], P["x"][bi], P["y"][bi], wb, add)
        szphi = _ls_add(H["szphi"], P["zf"][bi], H["hphi"][bi], pb, add & use_phi)
        q = _lsq(szphi)
        dfdz = jnp.where(add & use_phi, _f(q["dfdz"]), dfdz)
        phi0 = jnp.where(add & use_phi, _f(q["phi0"]), phi0)
        H = _w(add, {**H, "sxy": sxy, "szphi": szphi, "nxy": H["nxy"] + P["nsh"][bi],
                     "nzphi": H["nzphi"] + jnp.where(use_phi, P["nsh"][bi], 0), "used": H["used"].at[bi].set(1)}, H)
        fit = live & (~p1 | add)
        nx, ny, nr = _circ(H["sxy"])
        H, hcx, hcy, r = _dwcf(P, S, H, I32(0), jnp.where(fit, nx, hcx), jnp.where(fit, ny, hcy),
                               jnp.where(fit, nr, r), fit)
        ok0 = ~(_f(_lsq(H["sxy"])["chi2c"]) >= S["chi2xy"])
        passed = jnp.where(p1, passed, live & ok0)
        return H, hcx, hcy, r, (dfdz, phi0), jnp.where(p1, fd, live & ok0), it + 1, passed

    hcx, hcy, r = _circ(H["sxy"])
    H, _, _, _, _, _, _, on2 = jax.lax.while_loop(cond, body, (H, hcx, hcy, r, (H["dfdz"], H["fz0"]), on, I32(0),
                                                               jnp.bool_(False)))
    q = _lsq(H["sxy"])
    return _w(on2, {**H, "cx": _f(q["x0"]), "cy": _f(q["y0"]), "r": _f(q["r"]), "cchi2": _f(q["chi2c"])}, H)


def _pattern(P, A, S, H):
    """doPatternRecognition after the triplet search, as a loop over its ten steps (each piece traced once)."""
    alive0 = (H["seed"] >= 0) & ~(H["nxy"] < S["minnh"])

    def body(t, c):
        H, alive, rc, rc6, uphi, err = c
        c4 = (H["nxy"] - 1) != H["nzphi"]
        run_dl = alive & ((t == 0) | ((t == 4) & c4) | (t == 6) | ((t == 8) & rc6 & c4))
        H2, ok, e1 = _dlfp(P, A, S, H, I32(0), True, run_dl)
        # the step-0 checks: hit ratio, empty fit, failed fit -> return
        ratio = _f(_d(_f(H["sxy"][0])) / (_d(_f(H2["szphi"][0])) + 1e-6))
        alive = jnp.where(alive & (t == 0), ~(ratio > S["maxratio"]) & ~(H2["szphi"][0] == 0.0) & ok, alive)
        rc = jnp.where(run_dl & (t != 8), ok, rc)
        rc6 = jnp.where(alive & (t == 6), ok, rc6)
        chi_up = alive & (((t == 4) & ~c4 & (uphi == 1)) | ((t == 8) & rc6 & ~c4))
        H = {**H2, "dchi2": jnp.where(chi_up, _f(_lsq(H2["szphi"])["chi2l"]), H2["dchi2"])}
        H = _rescue_before_seed(P, S, H, alive & (t == 1))
        H = _refine(P, S, H, I32(0), alive & ((t == 2) | ((t == 9) & rc6)))
        uphi = jnp.where(t == 3, jnp.where((H["nzphi"] < S["minnh"]) | ~rc, I32(0), I32(1)),
                         jnp.where(t == 7, I32(1), uphi))
        H = _rescue(P, S, H, uphi == 1, alive & ((t == 3) | ((t == 7) & rc6)))
        run_df = alive & (t == 5) & ((H["nzphi"] < S["minnh"]) | ~rc)
        hd, hp, e2 = _find_dfdz(P, A, S, H, I32(0), run_df)
        H = _w(run_df, {**H, "dfdz": hd, "fz0": hp, "dchi2": np.float32(-1.)}, H)
        return H, alive, rc, rc6, uphi, err | e1 | e2

    H, alive, _, _, _, err = jax.lax.fori_loop(0, 10, body, (H, alive0, jnp.bool_(False), jnp.bool_(False),
                                                             I32(0), I32(0)))
    # filterUsingPatternRecognition
    on = alive & (H["seed"] >= 0) & (H["dfdz"] != 0)
    u = (H["used"] == 1) & P["valid"]
    pp = H["fz0"] + _m(P["zf"], H["dfdz"])
    xp = H["cx"] + _m(H["r"], jnp.cos(pp))
    yp = H["cy"] + _m(H["r"], jnp.sin(pp))
    wgt = _m(_calc_weight(P, xp, yp, S), S["w3d"])
    xr2 = _m(P["x"] - xp, P["x"] - xp)
    yr2 = _m(P["y"] - yp, P["y"] - yp)
    res = _m(xr2 + yr2, wgt)
    out = P["valid"] & (~u | (res > S["chi2h3d"]))
    H = _w(on, {**H, "flag": jnp.where(out, H["flag"] | OUTLIER, H["flag"]),
                "nst": jnp.sum(jnp.where(u & ~(res > S["chi2h3d"]), P["nsh"], 0)).astype(I32)}, H)
    return H, err


def _helicity(P, A, S, G, R):
    """findHelix for one helicity, defineHelixParams, initHelixSeed and the eDepAvg cut: the seed of this lane."""
    L = P["x"].shape[0]
    H, _, _, err = _search(P, A, S)
    H = {**H, "hphi": jnp.zeros(L, F32), "xyw": jnp.zeros(L, F32), "zw": jnp.zeros(L, F32), "flag": P["flag"]}
    H, e2 = _pattern(P, A, S, H)
    rmin, rmax = R
    ad = jnp.abs(H["dfdz"])
    fit = ~(H["nst"] < S["minnh"]) & ~((H["r"] < rmin) | (H["r"] > rmax)) & \
        ~((H["nxy"] < S["minnh"]) | (H["cchi2"] > S["chi2xy"])) & \
        ~((H["nzphi"] < S["minnh"]) | (H["dchi2"] < 0) | (H["dchi2"] > S["chi2zphi"]) | (ad < S["mindfdz"]) |
          (ad > S["maxdfdz"]))
    radius = H["r"]
    hok = fit & ~((radius == 0) | (H["dfdz"] == 0))
    am = jnp.where(A["hel"] == 1, np.float32(1), np.float32(-1))
    sr = jnp.where(radius != 0, radius, np.float32(1))
    omega = _d(am / sr)
    ph0 = _d(jnp.arctan2(-_m(am, H["cx"]), _m(am, H["cy"])))          # atan2f(dy, dx)
    d0 = _d(_m(am, jnp.sqrt(_m(H["cx"], H["cx"]) + _m(H["cy"], H["cy"])) - radius))
    td = _d(am / _m(sr, jnp.where(H["dfdz"] != 0, H["dfdz"], np.float32(1))))
    p1 = H["fz0"] + _m(am, FHALFPI)
    dph = _f(jnp.fmod(_d(_f(ph0) - p1), _d(FTWOPI)))
    dph = jnp.where(dph > FPI, dph - FTWOPI, dph)
    dph = jnp.where(dph < -FPI, dph + FTWOPI, dph)
    z0 = _m(_d(dph), td) / omega
    # initHelixSeed
    Rd = 1.0 / jnp.abs(omega)
    a = 1.0 / omega + d0
    x0 = _m(-a, jnp.sin(ph0))
    y0 = _m(a, jnp.cos(ph0))
    dfd = _g(1.0 / td) / Rd
    sign = _d(A["sign"])
    rcent = jnp.sqrt(_m(x0, x0) + _m(y0, y0))
    fcent = jnp.where((x0 == 0) & (y0 == 0), 0.0, jnp.arctan2(y0, x0))
    mm2mev = (3.0 / 10.0) * float(G["bz"])
    mom = _m(Rd, mm2mev) / jnp.cos(jnp.arctan(td))
    beta = jnp.abs(mom) / _g(jnp.sqrt(_m(mom, mom) + E_MASS * E_MASS))     # opaque: XLA makes x / sqrt(y) x * rsqrt(y)
    pitch = PI / 2.0 - jnp.arctan(td)
    t0 = _d(P["ct"]) - _g((P["tz"] - z0) / jnp.sin(pitch)) / _m(beta, C_LIGHT)
    good = (H["used"] == 1) & P["valid"]
    esum = _seqsum(_m(P["edep"], P["nsh"].astype(F32)), good, np.float32(0))
    nsh = jnp.sum(jnp.where(good, P["nsh"], 0))
    edep = _f(_d(esum) / (_d(nsh) + 1e-10))
    ok = hok & ~(edep > S["maxedep"])
    return {"ok": ok, "helicity": A["hel"], "rcent": _f(rcent), "fcent": _f(fcent), "radius": _f(Rd),
            "lambda": _f(_m(1.0 / dfd, sign)), "fz0": _f(ph0 - _m(PI / 2.0, sign) - _m(z0, omega) / td),
            "chi2xy": _f(_lsq(H["sxy"])["chi2c"]), "chi2zphi": _f(_lsq(H["szphi"])["chi2l"]), "t0": t0,
            "edep": edep, "mask": good, "hphi": H["hphi"], "xyw": H["xyw"], "zw": H["zw"], "flag": H["flag"],
            "err": err | e2}


# --- one time peak ---------------------------------------------------------------------------------------------------

def _peak(h, mem, clu, S, G, L):
    """fillFaceOrderedHits on one time peak: the selected hits in walk order and the face structure."""
    H = mem.shape[0]
    o = jnp.asarray(G["disk_origin"])[clu["disk"]]
    tcen = np.asarray(G["tracker_center"], np.float64)
    tx = (o[0] + _d(clu["cog_x"])) - tcen[0]
    ty = (o[1] + _d(clu["cog_y"])) - tcen[1]
    tz = (o[2] + 0.0) - tcen[2]
    ct = _f(clu["time"])
    cX, cY, cZ = _f(tx), _f(ty), _f(tz - float(np.float32(G["fCaloOffset"])))
    clphi = jnp.where(ct > 0, poly_atan2(cY, cX), np.float32(-9999.))
    flag = h["flag"]
    ngood = jnp.sum(jnp.where(mem & ((flag & BKG_MASK) == 0), h["nsh"], 0))
    sel = mem & ((flag & S["hsel"]) == S["hsel"]) & ((flag & S["bkgsel"]) == 0) & ~(h["edep"] > S["maxe"])
    if S["filter"]:
        dphi = poly_atan2(h["y"], h["x"]) - clphi
        dphi = jnp.where(dphi > FPI, dphi - FTWOPI, dphi)
        dphi = jnp.where(dphi < -FPI, dphi + FTWOPI, dphi)
        sel = sel & ~(jnp.abs(dphi) > FPI / np.float32(2))
    nsel = jnp.sum(sel)
    slot = jnp.nonzero(sel, size=L, fill_value=0)[0]
    valid0 = jnp.arange(L) < nsel
    hs = {k: v[slot] for k, v in h.items()}
    hs["ci"] = slot.astype(I32)
    perm, _, fb = _stock_sort(hs["upanel"], valid0)
    hs = {k: v[perm] for k, v in hs.items()}
    sid = hs["sid"]
    plane, panel = (sid & 0xFC00) >> 10, (sid & 0x380) >> 7
    station = (sid & 0xF800) >> 11
    face = panel % 2
    pl = plane % 2
    even = station % 2 == 0
    oface = jnp.where(even, jnp.where(pl == 0, 1 - face, face + 2), jnp.where(pl == 0, face, 3 - face))
    fid = (oface + station * 4).astype(I32)
    op = (panel // 2).astype(I32)
    key = jnp.where(valid0, (NTOTALFACES - 1 - fid) * 3 + op, 10 ** 6)
    it = jnp.argsort(key, stable=True)
    P = {k: v[it] for k, v in hs.items()}
    fid, op, station = fid[it], op[it], station[it].astype(I32)
    valid = valid0
    pos = jnp.arange(L)
    new = valid & ((pos == 0) | (fid != jnp.roll(fid, 1)))
    fk = jnp.where(valid, jnp.cumsum(new) - 1, 0).astype(I32)
    nf = jnp.sum(new).astype(I32)
    fstart = jnp.full(L + 1, L, I32).at[jnp.where(valid, fk, L)].min(pos.astype(I32))[:L]
    fend = jnp.zeros(L + 1, I32).at[jnp.where(valid, fk, L)].max(pos.astype(I32) + 1)[:L]
    zface = jnp.asarray(G["zFace"], F32)
    zfk = jnp.zeros(L + 1, F32).at[jnp.where(new, fk, L)].set(zface[fid])[:L]
    P.update(valid=valid, n=jnp.minimum(nsel, L).astype(I32), nfs=jnp.sum(jnp.where(valid, P["nsh"], 0)).astype(I32),
             station=station, op=op, fid=fid, fk=fk, nf=nf, fstart=fstart, fend=fend, zfk=zfk,
             zf=zface[fid], pp=jnp.asarray(G["phiPanel"], F32)[fid * 3 + op], ct=ct, cX=cX, cY=cY, cZ=cZ, tz=tz)
    go = ~(ngood < S["minnhtc"])
    return P, go, nsel > L, fb


def _one_peak(h, mem, clu, act, S, G, L):
    """CalHelixFinder's per-peak body for both helicities and pickBestHelix: the seeds this peak pushes [2]."""
    P, go, ovf, fb = _peak(h, mem, clu, S, G, L)
    go = go & act
    P = {**P, "n": jnp.where(go, P["n"], 0), "nf": jnp.where(go, P["nf"], 0)}
    hel = S["helicities"]
    A = {"hel": jnp.asarray(hel, I32), "sign": jnp.asarray([np.float32(1) if x == 1 else np.float32(-1) for x in hel]),
         }
    A["mp"] = A["sign"] * S["absmp"]
    R = _radius_limits(S, G["bz"])
    out = jax.vmap(lambda a: _helicity(P, a, S, G, R))(A)
    ok = out["ok"] & go
    if len(hel) == 2:
        both = ok[0] & ok[1]
        nh = jnp.sum(out["mask"], axis=1).astype(F32)
        natc = jnp.sum(out["mask"][0] & out["mask"][1]).astype(F32)
        over = (natc > _m(nh[0], np.float32(0.5))) | (natc > _m(nh[1], np.float32(0.5)))
        pick = jnp.where(nh[1] > nh[0], 1, jnp.where(nh[0] > nh[1], 0,
                                                     jnp.where(out["chi2zphi"][0] < out["chi2zphi"][1], 0, 1)))
        keep = ok & (~both | ~over | (jnp.arange(2) == pick))
    else:
        keep = ok
    err = jnp.where(go, _or(out.pop("err"), 0) | _flag(fb, ERR_SORT), I32(0))
    peak = {k: P[k] for k in ("ci", "x", "y", "z", "uvar", "vvar", "udx", "udy", "nsh")}
    return {**out, "ok": keep}, peak, ovf & act, err


def _or(x, axis):
    return jax.lax.reduce(x, jnp.zeros((), x.dtype), jax.lax.bitwise_or, (axis % x.ndim,))


# --- MergeHelices and the HelixFilter quantities ---------------------------------------------------------------------

def _findchisq(sd, S):
    """findchisq of the seeds [n] over their hits in order: (chi2dXY, chi2dZPhi)."""
    rc, fc = sd["rcent"], sd["fcent"]
    cx = _m(rc, jnp.cos(fc))[:, None]
    cy = _m(rc, jnp.sin(fc))[:, None]
    r = sd["radius"][:, None]
    use = sd["mask"] & ((sd["flag"] & S["bad"]) == 0)
    dx, dy = sd["x"] - cx, sd["y"] - cy
    dxn = _m(dx, -sd["udy"]) + _m(dy, sd["udx"])
    c2 = _m(dxn, dxn) / (_m(dx, dx) + _m(dy, dy))
    s2 = np.float32(1) - c2
    wxy = _m(_f(1. / _d(_m(sd["uvar"], s2) + _m(sd["vvar"], c2))), S["scalexy"])
    wzp = _m(_m(r, r) / (_m(sd["uvar"], c2) + _m(sd["vvar"], s2)), S["scalezphi"])
    txy = _terms(sd["x"], sd["y"], wxy)
    tzp = _terms(sd["z"], sd["hphi"], wzp)
    z = jnp.zeros(sd["x"].shape[:1] + (16,), F64)

    def step(c, a):
        sxy, szp = c
        u, t1, t2 = a
        return (jnp.where(u[:, None], sxy + t1, sxy), jnp.where(u[:, None], szp + t2, szp)), None

    (sxy, szp), _ = jax.lax.scan(step, (z, z), (use.T, jnp.moveaxis(txy, 1, 0), jnp.moveaxis(tzp, 1, 0)))
    return _f(_lsq(sxy)["chi2c"]), _f(_lsq(szp)["chi2l"])


def _merge_event(sd, H, S, fz):
    """MergeHelices on one event's seeds [n] (peak order, then helicity order) -> survivors and helix quantities."""
    n, L = sd["mask"].shape
    use = sd["mask"] & ((sd["flag"] & S["bad"]) == 0)
    gm = jnp.zeros((n, H + 1), I32).at[jnp.arange(n)[:, None], jnp.where(use, sd["ci"], H)].set(
        jnp.where(use, sd["nsh"], 0))[:, :H]
    nh = jnp.sum(gm, axis=1)
    nover = jnp.einsum("ah,bh->ab", gm, (gm > 0).astype(I32))
    cxy, czp = _findchisq(sd, S)
    chi = cxy + czp

    def comp(i, j):
        minh = jnp.minimum(nh[i], nh[j])
        ov = nover[i, j]
        same = (ov >= S["minnover"]) & (ov.astype(F32) / minh.astype(F32) > S["minfrac"])
        first = (nh[i] > nh[j]) & (nh[i] - nh[j] > S["deltanh"])
        second = (nh[j] > nh[i]) & (nh[j] - nh[i] > S["deltanh"])
        f1 = first | (~second & (chi[i] < chi[j]))
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
    # HelixTool over the non-outlier hits
    good = sd["mask"] & ((sd["flag"] & OUTLIER) == 0)
    nsh = jnp.sum(jnp.where(good, sd["nsh"], 0), axis=1)
    anyg = jnp.any(good, axis=1)
    zf = jnp.where(anyg, jnp.take_along_axis(sd["z"], _argmax(good, axis=1)[:, None], 1)[:, 0], np.float32(0))
    il = L - 1 - _argmax(good[:, ::-1], axis=1)
    zl = jnp.where(anyg, jnp.take_along_axis(sd["z"], il[:, None], 1)[:, 0], np.float32(0))
    lam, R = sd["lambda"], sd["radius"]
    nloops = _f(jnp.abs(_d(zl - zf) / _m(_d(lam), 2., PI)))
    lam_d = jnp.where(lam != 0, _d(lam), 1.0)
    az = jnp.where(lam[:, None] != 0, _f(_d(sd["fz0"])[:, None] + _d(jnp.asarray(fz, F32))[None, :] / lam_d[:, None]),
                   np.float32(0))
    ccx, ccy = _m(sd["rcent"], jnp.cos(sd["fcent"])), _m(sd["rcent"], jnp.sin(sd["fcent"]))
    px = ccx[:, None] + _m(R[:, None], jnp.cos(az))
    py = ccy[:, None] + _m(R[:, None], jnp.sin(az))
    rf = jnp.sqrt(_m(px, px) + _m(py, py))
    from .helix import TRACKER_RIN, TRACKER_ROUT, expected_faces
    nface = jnp.sum(~(rf < TRACKER_RIN) & ~(rf > TRACKER_ROUT), axis=1)
    expected = jnp.asarray(expected_faces(len(fz)))[nface]
    return {"ok": alive, "helicity": sd["helicity"].astype(F32), "R": R, "rcent": sd["rcent"], "fcent": sd["fcent"],
            "lambda": lam, "fz0": sd["fz0"], "t0": sd["t0"], "edep": sd["edep"],
            "p": _m(jnp.sqrt(_m(R, R) + _m(lam, lam)), MM2MEV), "pt": _m(R, MM2MEV), "d0": sd["rcent"] - R,
            "chi2xy": cxy, "chi2zphi": czp, "chi2xy_finder": sd["chi2xy"], "chi2zphi_finder": sd["chi2zphi"],
            "nsh": nsh.astype(F32), "nloops": nloops, "hitratio": nsh.astype(F32) / expected,
            "tandip": jnp.abs(lam) / R,
            "hits": jnp.where(sd["mask"], sd["ci"], -1), "hphi": sd["hphi"], "flag": sd["flag"],
            "xyw": sd["xyw"], "zw": sd["zw"]}


def helices(ph, tcs, cl, S, G, fz, K=None, L=CHF_HITS):
    """Panel hits [E, H], time-peak sets [{valid [E, C], members [E, C, H]}] of finders with the settings S, the
    calo clusters cl [E, C] (disk, time, cog_x, cog_y; the peaks are in cluster order) and the geometry G -> one
    merged helix-seed dict per set, each [E, nhel C] (seed slots: peak, then helicity order; `ok` marks the
    MergeHelices survivors), with `overflow` [E] (peaks with more selected hits than L: truncated) and `err` [E]
    (OR of the flags in the module docstring). The finder runs on the valid peaks only, K at a time."""
    ft = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    with strict(ph["sid"]):
        outs = _helices(ph, tcs, cl, S, G, fz, K, L)
    conv = lambda v: v.astype(ft) if jnp.issubdtype(v.dtype, jnp.floating) else \
        (v.astype(I32) if v.dtype == I64 else v)
    return [{k: conv(v) for k, v in o.items()} for o in outs]


def _out_spec(nh, L):
    """(shape, dtype) of _one_peak's results for one peak (written out: an eval_shape would trace the finder twice)."""
    f, fl = ((nh,), F32), ((nh, L), F32)
    seeds = {"ok": ((nh,), jnp.bool_), "helicity": ((nh,), I32), **{k: f for k in (
        "rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi", "edep")}, "t0": ((nh,), F64),
        "mask": ((nh, L), jnp.bool_), "hphi": fl, "xyw": fl, "zw": fl, "flag": ((nh, L), I64)}
    peak = {"ci": ((L,), I32), "nsh": ((L,), I32), **{k: ((L,), F32) for k in ("x", "y", "z", "uvar", "vvar", "udx",
                                                                               "udy")}}
    return seeds, peak, ((), jnp.bool_), ((), I32)


def _helices(ph, tcs, cl, S, G, fz, K, L):
    E, H = ph["x"].shape
    upanel = ((ph["sid"] >> 10) * 6 + ((ph["sid"] & 0x380) >> 7)).astype(I32)
    h = {k: _f(ph[k]) for k in ("x", "y", "z", "time", "edep", "uvar", "vvar", "udx", "udy")}
    h.update(nsh=jnp.asarray(ph["ncombo"]).astype(I32), sid=jnp.asarray(ph["sid"]).astype(I64),
             flag=jnp.asarray(ph["flag"]).astype(I64), upanel=upanel)
    nP = len(tcs)
    C = tcs[0]["valid"].shape[1]
    nh = len(S["helicities"])
    val = jnp.stack([t["valid"] for t in tcs])                               # [P, E, C]
    mem = jnp.stack([t["members"] for t in tcs]) & ph["valid"][None, :, None, :]
    N = nP * E * C
    K = E if K is None else K
    flat = val.reshape(-1)
    order = jnp.concatenate([jnp.nonzero(flat, size=N, fill_value=N)[0].astype(I32), jnp.full(K, N, I32)])
    nchunk = ((jnp.sum(flat) + K - 1) // K).astype(I32)
    clu = {"disk": jnp.asarray(cl["disk"]).astype(I32), "time": _f(cl["time"]), "cog_x": _d(cl["cog_x"]),
           "cog_y": _d(cl["cog_y"])}
    run = jax.vmap(lambda hh, mm, cc, aa: _one_peak(hh, mm, cc, aa, S, G, L))
    buf0 = jax.tree.map(lambda a: jnp.zeros((N + 1,) + a[0], a[1]), _out_spec(nh, L),
                        is_leaf=lambda a: isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], tuple))

    def chunk(i, buf):
        sel = jax.lax.dynamic_slice(order, (i * K,), (K,))
        ok = sel < N
        si = jnp.minimum(sel, N - 1)
        p, e, c = si // (E * C), (si // C) % E, si % C
        r = run({k: v[e] for k, v in h.items()}, mem[p, e, c], {k: v[e, c] for k, v in clu.items()}, ok)
        dst = jnp.where(ok, sel, N)
        return jax.tree.map(lambda b, a: b.at[dst].set(a), buf, r)

    buf = jax.lax.fori_loop(0, nchunk, chunk, buf0)
    seeds, peak, ovf, err = jax.tree.map(lambda a: a[:N].reshape((nP, E, C) + a.shape[1:]), buf)
    seeds = {**seeds, "ok": seeds["ok"] & val[..., None]}
    outs = []
    for p in range(nP):
        sd = {k: v[p].reshape((E, C * nh) + v.shape[4:]) for k, v in seeds.items()}
        sd.update({k: jnp.repeat(v[p], nh, axis=1) for k, v in peak.items()})
        o = jax.vmap(lambda s_: _merge_event(s_, H, S, fz))(sd)
        o["overflow"] = jnp.sum(ovf[p], axis=1)
        o["err"] = _or(err[p], 1)
        outs.append(o)
    return outs
