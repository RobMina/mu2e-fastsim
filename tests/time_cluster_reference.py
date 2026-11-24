"""A literal transcription of the stock TimeClusterFinder module (TrkPatRec, Mu2e Offline v13_41_01, Apache License
2.0) with TrkTimeCalculator (TrkReco), MVATools::evalMVA (Mu2eUtilities), the boost accumulators it uses (min, max,
weighted_median(with_p_square_quantile), weighted_mean, weighted_variance(lazy)), ROOT's TH1F time spectrum and
glibc's expf, event by event, as the reference for the device port (trigger/device/timecluster.py) and the stock
TimeCluster comparison. Covered configuration: TestFlag false (every hit is used), UseTOTDrift true, sigmoid MLPs
with the N input normalisation (TMVA >= 262657); the calo seeds (UseCaloCluster, UseCaloClusterPosition) on or off.

Typing follows the C++: numpy float32 scalars (`f`) for every C++ float, so each operation rounds as float does;
Python floats (IEEE double, no fused multiply-add) where C++ uses double or promotes (TrkT0's t0 and t0err, the
time-of-flight correction, double literals, M_PI, the TH1F axis and bin contents read back as double, sqrt of an
unsigned); a value is rounded to float exactly where stock stores it in a float. Float32 scalars are never mixed
with Python floats in arithmetic (NumPy 2 would round the Python float to float32).

Stock details kept:
- comboHitTime and caloClusterTime: double, (pitch * beta) * c_light with the float pitch; the calo cog is moved to
  Mu2e (disk origin + cog, the disks are not rotated) and to the tracker frame (minus the tracker centre); the hit
  time is stored in a float by every caller.
- the collection: the calo seeds first (clusters with energyDep > CaloClusterMinE, in cluster order, t0 the double
  caloClusterTime, t0err CaloTimeErr), then the histogram peaks. findPeaks blanks the bins max(1, b - PeakWidth) ..
  min(nbins + 1, b + PeakWidth + 1) - 1 around each seed's FindBin(t0) (b can be the underflow 0 or the overflow
  nbins + 1), then visits the bins >= Ymin in the order libstdc++'s std::sort leaves them (content descending, not
  stable: `std_sort`), accumulating nsh and t0 in float from the double GetBinContent / GetBinCenter.
- the TH1F: rint((Tmax - Tmin) / Tbin) bins of the float settings, ROOT 6.38's TAxis::FindBin (width-based with the
  +-1 correction) and GetBinCenter (`ROOT_AXIS` = "6.36" selects the older bin arithmetic); float contents.
- assignHits: dt = fabs(float - double) stored in a float, the cut dt < DtMax + t0err in double, the closest cluster
  in collection order with a strict <, over calo and peak clusters alike.
- initCluster: boost's weighted P^2 median (`WeightedP2`, float samples and float weights: the first five samples
  sorted with min_element swaps and their weights summed, the minimum replacing actual_positions[0] by the new
  weight, the desired positions with the double literals, std::upper_bound), of the hit times (not for a calo
  cluster) and the positions, the calo cog x and y (as float, weight CaloClusterWeight) after the hits; with fewer
  than five samples the result is the third sample in input order (0 below three). t0err = float((max - min) *
  invsqrt12) / sqrt(unsigned) in double. _nsh is a uint16_t.
- removeHit / addHit: float newt0 = (double * uint16 - float * unsigned) / float, t0err in double (removeHit only
  when the variance is positive; addHit takes the sqrt as it is), the position as XYZVectorF arithmetic: v * float,
  and v / a as v * (1 / a) (ROOT's operator/= scales by the float reciprocal). A calo cluster's t0 never changes.
- clusterMean: float sums in _strawHitIdxs order (weighted_sum x * w, weighted_moment<2> w * (x * x), the sum of
  weights), the calo cog after the hits for x and y; var = m2 - mean * mean in float, t0err = sqrtf(max(1.0,
  2.0 * var) / count).
- prefilter, refine and recover: polyAtan2 azimuths (float) and Angles::deltaPhi's float overload (double
  arithmetic); the cluster azimuth is taken once per pass; refine removes the first lowest MVA (strict <) below
  MinKeepHitMVA; recover visits the hits in index order with the cluster as it is at that moment and appends the
  added hits.
- MVATools::evalMVA: inputs (v - offset) * scale - 1.0 (float, double literal), each neuron 0.0f plus the products
  w * x in input order (bias last), sigmoid 1.0 / (1.0 + expf(-x)) in double from glibc's expf, stored as float;
  offsets, maxima and weights read with strtof, scale = 2.0 / (max - min) in double, stored as float.
- glibc 2.28+ expf (`expf`): the generic C algorithm (32-entry 2^(i/32) table, degree-3 polynomial in double), as
  the build without fused multiply-add computes it; `expf_fma` is the -mfma variant x86-64 selects at run time on
  CPUs with FMA (the two differ in a few results per 10^7, measured by the stock comparison).
"""
import math
import xml.etree.ElementTree as ET
from fractions import Fraction

import numpy as np

from mpr_reference import poly_atan2
from robust_helix_reference import std_sort

f = np.float32
F64 = np.float64
PI = math.pi
TWOPI = 2 * math.pi                          # Angles: static const double twopi = 2*M_PI
C_LIGHT = (2.99792458e+8 * 1000.0) / 1.e+9   # CLHEP::c_light, mm/ns
INVSQRT12 = f(1.0 / math.sqrt(12.0))         # static float invsqrt12(1.0/sqrt(12.0))
FLT_MAX = f(np.finfo(np.float32).max)
DETECTOR_SYSTEM_Z0 = 10171.0                 # mu2e.detectorSystemZ0 (geom_run1_a.txt), mm
ROOT_AXIS = "6.38"


# --- settings and geometry --------------------------------------------------------------------------------------------

def settings(m, pitch=None):
    """The TimeClusterFinder module table -> the module's members (float where the Config atom is float)."""
    if m.get("TestFlag", False):
        raise ValueError("transcription covers TestFlag false")
    t0c = m.get("T0Calculator", {})
    if not t0c.get("UseTOTDrift", True):
        raise ValueError("transcription covers UseTOTDrift true")
    return {"tmin": f(m["Tmin"]), "tmax": f(m["Tmax"]), "tbin": f(m["Tbin"]), "ymin": f(m["Ymin"]),
            "npeak": int(m["PeakWidth"]), "maxdt": f(m["DtMax"]), "minnhits": int(m["MinNHits"]),
            "maxdphi": f(m["MaxdPhi"]), "minkeep": f(m["MinKeepHitMVA"]), "minadd": f(m["MinAddHitMVA"]),
            "pitch": f(m["AveragePitch"] if pitch is None else pitch), "prefilter": bool(m["PrefilterCluster"]),
            "refine": bool(m["RefineClusters"]), "recover": bool(m["RecoverHits"]),
            "usecc": bool(m.get("UseCaloCluster", False)), "useccpos": bool(m.get("UseCaloClusterPosition", False)),
            "ccmine": f(m.get("CaloClusterMinE", 0.0)), "ccwt": f(m.get("CaloClusterWeight", 0.0)),
            "beta": float(t0c.get("StrawHitBeta", 1.0)), "calo_zoff": float(t0c.get("CaloShowermaxZ", -120.0)),
            "calo_toff": float(t0c.get("TrkToCaloTimeOffset", -0.4)),
            "calo_terr": float(t0c.get("CaloTimeErr", 0.5))}


def geometry():
    """The calorimeter disk origins in Mu2e (the disks are not rotated) and the tracker centre (-solenoidOffset, 0,
    detectorSystemZ0)."""
    from trigger.device import calogeom
    cg = calogeom.CALO_GEOM
    return {"disk_origin": np.asarray(calogeom.build()["origin"], np.float64),
            "tracker_center": np.array([-cg["solenoidOffset"], 0.0, DETECTOR_SYSTEM_Z0])}


# --- MVATools ---------------------------------------------------------------------------------------------------------

def strtof(s):
    """strtof: the decimal string correctly rounded to float (nearest, ties to even)."""
    q = Fraction(s.strip())
    c = f(float(q))
    best = None
    for v in (np.nextafter(c, f(-np.inf)), c, np.nextafter(c, f(np.inf))):
        key = (abs(Fraction(float(v)) - q), int(v.view(np.uint32)) & 1)
        if best is None or key < best[0]:
            best = (key, v)
    return f(best[1])


def load_mva(path):
    """MVATools::initMVA on a TMVA MLP XML: offsets, scales and the layer weights [n_source incl. bias, n_target]."""
    root = ET.parse(path).getroot()
    rel = next(i.get("value") for i in root.iter("Info") if i.get("name") == "TMVA Release")
    if int(rel[rel.find("[") + 1:rel.find("]")]) < 262657:
        raise ValueError("old TMVA networks are not covered")
    opts = {o.get("name"): (o.text or "", o.get("modified")) for o in root.iter("Option")}
    if "sigmoid" not in opts["NeuronType"][0]:
        raise ValueError("transcription covers sigmoid networks")
    norm = "Yes" in (opts.get("VarTransform", ("", ""))[1] or "")
    off, scl = [], []
    for v in root.find("Variables"):
        vmin, vmax = strtof(v.get("Min")), strtof(v.get("Max"))
        off.append(vmin)
        scl.append(f(2.0 / float(f(vmax - vmin))))
    layers = []
    for layer in root.find("Weights").find("Layout"):
        rows = [[strtof(t) for t in (n.text or "").split()] for n in layer.findall("Neuron")]
        if rows and rows[0]:
            layers.append(np.array(rows, np.float32))
    return {"norm": norm, "offset": np.array(off, np.float32), "scale": np.array(scl, np.float32),
            "layers": layers}


_EXP2F_T = np.array([
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540], np.uint64)
EXP2F_N = 32
EXP2F_INVLN2N = float.fromhex("0x1.71547652b82fep+0") * EXP2F_N
EXP2F_SHIFT = float.fromhex("0x1.8p+52")
EXP2F_C = (float.fromhex("0x1.c6af84b912394p-5") / EXP2F_N / EXP2F_N / EXP2F_N,
           float.fromhex("0x1.ebfce50fac4f3p-3") / EXP2F_N / EXP2F_N,
           float.fromhex("0x1.62e42ff0c52d6p-1") / EXP2F_N)
EXPF_OFLOW = f(float.fromhex("0x1.62e42ep6"))
EXPF_UFLOW = f(-float.fromhex("0x1.9fe368p6"))


def expf(x):
    """glibc's __expf (sysdeps/ieee754/flt-32/e_expf.c, generic C, no fused multiply-add), elementwise."""
    x = np.asarray(x, np.float32)
    xd = x.astype(F64)
    z = EXP2F_INVLN2N * xd
    kd = z + EXP2F_SHIFT
    ki = kd.view(np.uint64)
    kd = kd - EXP2F_SHIFT
    with np.errstate(invalid="ignore"):              # x = -inf: replaced below, as stock's early return
        r = z - kd
    t = _EXP2F_T[(ki % np.uint64(EXP2F_N)).astype(np.int64)] + (ki << np.uint64(52 - 5))
    s = t.view(F64)
    z = EXP2F_C[0] * r + EXP2F_C[1]
    r2 = r * r
    y = EXP2F_C[2] * r + 1
    y = z * r2 + y
    y = y * s
    with np.errstate(over="ignore"):
        out = y.astype(np.float32)
    out = np.where(x > EXPF_OFLOW, np.float32(np.inf), np.where(x < EXPF_UFLOW, np.float32(0), out))
    return out.astype(np.float32)


def expf_fma(x):
    """The -mfma build of the same function: GCC contracts z = InvLn2N * xd into both of its uses (z + SHIFT and
    z - kd) and the three polynomial multiply-adds into fused multiply-adds."""
    fma = math.fma
    x = np.asarray(x, np.float32)
    out = np.empty(x.shape, np.float32)
    for idx, xv in np.ndenumerate(x):
        xd = float(xv)
        kd = fma(EXP2F_INVLN2N, xd, EXP2F_SHIFT)
        ki = int(np.float64(kd).view(np.uint64))
        kd = kd - EXP2F_SHIFT
        r = fma(EXP2F_INVLN2N, xd, -kd)
        t = (int(_EXP2F_T[ki % EXP2F_N]) + (ki << 47)) & 0xFFFFFFFFFFFFFFFF
        s = float(np.uint64(t).view(F64))
        z = fma(EXP2F_C[0], r, EXP2F_C[1])
        y = fma(EXP2F_C[2], r, 1.0)
        y = fma(z, r * r, y)
        with np.errstate(over="ignore"):
            out[idx] = f(y * s)
        if xv > EXPF_OFLOW:
            out[idx] = f(np.inf)
        elif xv < EXPF_UFLOW:
            out[idx] = f(0)
    return out


EXPF = {"fn": expf}                          # the expf the MVA uses (expf or expf_fma)


def _sigmoid(a):
    e = EXPF["fn"](-a).astype(F64)
    return (1.0 / (1.0 + e)).astype(np.float32)


def eval_mva(net, X):
    """MVATools::evalMVA for the rows of X [n, nvar] (float32) -> [n] float32."""
    X = np.asarray(X, np.float32)
    n = X.shape[0]
    if net["norm"]:
        x = ((X - net["offset"]) * net["scale"]).astype(F64) - 1.0
        x = x.astype(np.float32)
    else:
        x = X
    x = np.concatenate([x, np.ones((n, 1), np.float32)], 1)
    L = len(net["layers"])
    for k, W in enumerate(net["layers"]):
        y = np.zeros((n, W.shape[1]), np.float32)
        for i in range(W.shape[0]):
            y = y + W[i][None, :] * x[:, i:i + 1]
        y = _sigmoid(y)
        if k < L - 1:
            x = np.concatenate([y, np.ones((n, 1), np.float32)], 1)
        else:
            x = y
    return x[:, 0]


# --- boost accumulators -----------------------------------------------------------------------------------------------

def _upper_bound(h, x):
    first, length = 0, len(h)
    while length > 0:
        half = length >> 1
        mid = first + half
        if x < float(h[mid]):
            length = half
        else:
            first, length = mid + 1, length - half - 1
    return first


class WeightedP2:
    """boost weighted_p_square_quantile_impl<float, float, for_median> (with count and sum_of_weights)."""

    def __init__(self):
        self.cnt = 0
        self.sw = f(0)
        self.h = [f(0)] * 5
        self.ap = [f(0)] * 5
        self.dp = [f(0)] * 5
        self.p = f(0.5)

    def add(self, x, w, diag=None):
        # the sample keeps the type the caller passes: a double cog is compared in double, stored as float
        x = float(x)
        w = f(w)
        self.cnt += 1
        self.sw = f(self.sw + w)
        h, ap, dp, cnt = self.h, self.ap, self.dp, self.cnt
        if cnt <= 5:
            h[cnt - 1] = f(x)
            ap[cnt - 1] = w
            if cnt == 5:
                for b in range(5):
                    m = b
                    for j in range(b + 1, 5):                   # std::min_element: the first smallest
                        if h[j] < h[m]:
                            m = j
                    h[b], h[m] = h[m], h[b]
                    ap[b], ap[m] = ap[m], ap[b]
                for i in range(1, 5):
                    ap[i] = f(ap[i] + ap[i - 1])
            return
        if x < float(h[0]):
            h[0] = f(x)
            ap[0] = w
            k = 1
        elif float(h[4]) <= x:
            h[4] = f(x)
            k = 4
        else:
            k = _upper_bound(h, x)
        for i in range(k, 5):
            ap[i] = f(ap[i] + w)
        S, a0, p = self.sw, ap[0], self.p
        dp[0] = a0
        dp[1] = f(float(f((S - a0) * p)) / 2. + float(a0))
        dp[2] = f(f((S - a0) * p) + a0)
        dp[3] = f(float(f(S - a0)) * (1. + float(p)) / 2. + float(a0))
        dp[4] = S
        for i in (1, 2, 3):
            d = f(dp[i] - ap[i])
            dpp = f(ap[i + 1] - ap[i])
            dm = f(ap[i - 1] - ap[i])
            with np.errstate(all="ignore"):
                hp = f(f(h[i + 1] - h[i]) / dpp)
                hm = f(f(h[i - 1] - h[i]) / dm)
            if (d >= 1. and dpp > 1.) or (d <= -1. and dm < -1.):
                sd = f(int(d / abs(d)))                          # short sign_d
                hh = f(h[i] + f(sd / f(dpp - dm)) * f(f(f(sd - dm) * hp) + f(f(dpp - sd) * hm)))
                if h[i - 1] < hh and hh < h[i + 1]:
                    h[i] = hh
                else:
                    if diag is not None:
                        diag["p2_linear"] = diag.get("p2_linear", 0) + 1
                    if d > 0:
                        h[i] = f(h[i] + hp)
                    if d < 0:
                        h[i] = f(h[i] - hm)
                ap[i] = f(ap[i] + sd)

    def result(self):
        return self.h[2]


# --- ROOT TH1F ----------------------------------------------------------------------------------------------------------

class TimeSpectrum:
    """TH1F("timespec", ..., nbins, Tmin, Tmax) with float contents."""

    def __init__(self, s):
        self.n = int(np.rint(f(f(s["tmax"] - s["tmin"]) / s["tbin"])))
        self.xmin, self.xmax = float(s["tmin"]), float(s["tmax"])
        self.c = [f(0)] * (self.n + 2)

    def find_bin(self, x):
        if x < self.xmin:
            return 0
        if not (x < self.xmax):
            return self.n + 1
        if ROOT_AXIS == "6.36":
            return 1 + int(self.n * (x - self.xmin) / (self.xmax - self.xmin))
        width = (self.xmax - self.xmin) / self.n
        a = int((x - self.xmin) / width)
        return 1 + a - int(x < self.xmin + width * a) + int(self.xmin + width * (a + 1) <= x)

    def fill(self, x, w):
        b = self.find_bin(x)
        self.c[b] = f(self.c[b] + f(w))

    def content(self, b):
        return float(self.c[b])

    def center(self, b):
        width = (self.xmax - self.xmin) / float(self.n)
        if ROOT_AXIS == "6.36":
            return self.xmin + (b - 1) * width + 0.5 * width
        return self.xmin + (b - 0.5) * width


# --- the module -------------------------------------------------------------------------------------------------------

def delta_phi(phi, ref):
    """Angles::deltaPhi(float&, float): double arithmetic, returned as float."""
    d = float(phi) - float(ref)
    while d > PI:
        d -= TWOPI
    while d <= -PI:
        d += TWOPI
    return f(d)


class _TC:
    def __init__(self, t0, t0err, calo=-1):
        self.hits, self.t0, self.t0err, self.pos, self.nsh, self.calo = [], float(t0), float(t0err), None, 0, calo
        self.cog = None


class Finder:
    """One event of TimeClusterFinder::produce."""

    def __init__(self, hits, cc, s, nets, geom=None, diag=None):
        self.s, self.nets, self.diag = s, nets, ({} if diag is None else diag)
        self.T = np.asarray(hits["time"], np.float32)
        self.X, self.Y, self.Z = (np.asarray(hits[k], np.float32) for k in ("x", "y", "z"))
        self.NSH = [int(v) for v in hits["ncombo"]]
        self.PLANE = [int(v) for v in hits["plane"]]
        self.UVAR = np.asarray(hits["uvar"], np.float32)
        self.WDIST = np.asarray(hits["wdist"], np.float32)
        self.N = len(self.T)
        den = float(s["pitch"]) * s["beta"] * C_LIGHT
        self.den = den
        self.CHT = [f(float(self.T[i]) - float(self.Z[i]) / den) for i in range(self.N)]
        self.PHI = [poly_atan2(self.Y[i], self.X[i]) for i in range(self.N)]
        # the static MVA inputs: rho = Perp2(), nsh, plane, werr = wireRes(), |wireDist()|
        self.STATIC = np.stack([self.X * self.X + self.Y * self.Y, np.asarray(self.NSH, np.float32),
                                np.asarray(self.PLANE, np.float32), np.sqrt(self.UVAR), np.abs(self.WDIST)], 1)
        self.cc = cc
        self.geom = geom

    # -- collection
    def calo_seeds(self, tccol):
        s, cc = self.s, self.cc
        g = self.geom if self.geom is not None else geometry()
        for k in range(len(cc["time"])):
            if f(cc["energy"][k]) > s["ccmine"]:
                o, c = g["disk_origin"][int(cc["disk"][k])], g["tracker_center"]
                cz = float(cc["cog_z"][k]) if "cog_z" in cc else 0.0
                zc = (o[2] + cz) - c[2]
                t0 = (float(f(cc["time"][k])) - (zc + s["calo_zoff"]) / self.den) + s["calo_toff"]
                tc = _TC(t0, s["calo_terr"], k)
                tc.cog = (float(cc["cog_x"][k]), float(cc["cog_y"][k]))     # cog3Vector: double
                tccol.append(tc)

    def find_peaks(self, tccol):
        s, hs = self.s, self.hist
        nbins = hs.n + 1
        npk = s["npeak"]
        used = [False] * nbins
        for tc in tccol:
            ib = hs.find_bin(tc.t0)
            for j in range(max(1, ib - npk), min(nbins, ib + npk + 1)):
                used[j] = True
        bcv = [(hs.c[b], b) for b in range(1, nbins) if hs.content(b) >= float(s["ymin"])]
        self.diag["max_cand"] = max(self.diag.get("max_cand", 0), len(bcv))
        if len(set(c for c, _ in bcv)) < len(bcv):
            self.diag["sort_ties"] = self.diag.get("sort_ties", 0) + 1
        std_sort(bcv, lambda x, y: x[0] > y[0], self.diag)
        npeaks = 0
        for c, b in bcv:
            if used[b]:
                continue
            nsh, t0 = f(0.0), f(0.0)
            for ib in range(max(1, b - npk), min(nbins, b + npk + 1)):
                nsh = f(float(nsh) + hs.content(ib))
                t0 = f(float(t0) + hs.center(ib) * hs.content(ib))
                used[ib] = True
            t0 = f(t0 / nsh)
            if nsh > f(s["minnhits"]):
                tc = _TC(float(t0), float(s["tbin"]) * 0.5)
                tccol.append(tc)
                npeaks += 1
        self.diag["max_peaks"] = max(self.diag.get("max_peaks", 0), npeaks)

    def assign_hits(self, tccol):
        s = self.s
        for i in range(self.N):
            time = self.CHT[i]
            mindt = f(1e5)
            best = None
            for tc in tccol:
                dt = f(abs(float(time) - tc.t0))
                if float(dt) < float(s["maxdt"]) + tc.t0err and dt < mindt:
                    mindt, best = dt, tc
            if best is not None:
                best.hits.append(i)

    # -- per cluster
    def init_cluster(self, tc):
        s = self.s
        tmin, tmax = FLT_MAX, -FLT_MAX
        acc = [WeightedP2() for _ in range(4)]
        nstrs = len(tc.hits)
        tc.nsh = 0
        for i in tc.hits:
            n = self.NSH[i]
            tc.nsh = (tc.nsh + n) & 0xFFFF
            ht, hw = self.CHT[i], f(n)
            if ht < tmin:
                tmin = ht
            if tmax < ht:
                tmax = ht
            for a, v in zip(acc, (ht, self.X[i], self.Y[i], self.Z[i])):
                a.add(v, hw, self.diag)
        if tc.calo >= 0:
            if s["useccpos"]:
                acc[1].add(tc.cog[0], s["ccwt"], self.diag)
                acc[2].add(tc.cog[1], s["ccwt"], self.diag)
        else:
            tc.t0 = float(acc[0].result())
            with np.errstate(all="ignore"):
                tc.t0err = float(F64(f(f(tmax - tmin) * INVSQRT12)) / np.sqrt(F64(nstrs)))
        tc.pos = np.array([acc[1].result(), acc[2].result(), acc[3].result()], np.float32)

    def _pos_update(self, tc, i, sign, denom):
        p = np.array([self.X[i], self.Y[i], self.Z[i]], np.float32)
        a = tc.pos * f(tc.nsh)
        b = p * f(self.NSH[i])
        v = a - b if sign < 0 else a + b
        return v * f(f(1) / denom)                    # XYZVectorF / a == * (1 / a)

    def remove_hit(self, tc, k):
        i = tc.hits[k]
        n = self.NSH[i]
        denom = f((tc.nsh - n) & 0xFFFFFFFF)          # float(uint16 - unsigned): unsigned arithmetic
        if denom > 0:
            if tc.calo < 0:
                cht = self.CHT[i]
                newt0 = f((tc.t0 * tc.nsh - float(f(cht * f(n)))) / float(denom))
                var = tc.t0err * tc.t0err * tc.nsh - float(f(cht - newt0)) * (float(cht) - tc.t0) * n
                if var > 0.0:
                    tc.t0err = math.sqrt(var / float(denom))
                tc.t0 = float(newt0)
            tc.pos = self._pos_update(tc, i, -1, denom)
            tc.nsh = (tc.nsh - n) & 0xFFFF
        del tc.hits[k]

    def add_hit(self, tc, i):
        n = self.NSH[i]
        denom = f(tc.nsh + n)
        if tc.calo < 0:
            cht = self.CHT[i]
            newt0 = f((tc.t0 * tc.nsh + float(f(cht * f(n)))) / float(denom))
            with np.errstate(invalid="ignore"):
                tc.t0err = float(np.sqrt(F64((tc.t0err * tc.t0err * tc.nsh
                                              + float(f(cht - newt0)) * (float(cht) - tc.t0) * n) / float(denom))))
            tc.t0 = float(newt0)
        tc.pos = self._pos_update(tc, i, 1, denom)
        tc.nsh = (tc.nsh + n) & 0xFFFF
        tc.hits.append(i)

    def prefilter(self, tc):
        s = self.s
        changed = True
        while changed and tc.nsh > 0:
            changed = False
            pphi = poly_atan2(tc.pos[1], tc.pos[0])
            worst, maxad = None, s["maxdphi"]
            for k, i in enumerate(tc.hits):
                ad = abs(delta_phi(self.PHI[i], pphi))
                if ad > maxad:
                    worst, maxad = k, ad
            if worst is not None:
                changed = True
                self.diag["prefiltered"] = self.diag.get("prefiltered", 0) + 1
                self.remove_hit(tc, worst)

    def cluster_mean(self, tc):
        s = self.s
        sw = ws = wm2 = f(0)
        sx = [f(0)] * 3
        swx = [f(0)] * 3
        cnt = 0
        for i in tc.hits:
            w = f(self.NSH[i])
            x = self.CHT[i]
            cnt += 1
            sw = f(sw + w)
            ws = f(ws + f(x * w))
            wm2 = f(wm2 + f(w * f(x * x)))
            for k, v in enumerate((self.X[i], self.Y[i], self.Z[i])):
                swx[k] = f(swx[k] + w)
                sx[k] = f(sx[k] + f(v * w))
        if tc.calo >= 0:
            if s["useccpos"]:
                for k in (0, 1):
                    swx[k] = f(swx[k] + s["ccwt"])
                    sx[k] = f(float(sx[k]) + tc.cog[k] * float(s["ccwt"]))   # double sample * float weight
        else:
            mean = f(ws / sw)
            var = f(f(wm2 / sw) - f(mean * mean))
            tc.t0 = float(mean)
            tc.t0err = float(np.sqrt(f(max(1.0, 2.0 * float(var)) / cnt)))
        tc.pos = np.array([f(sx[k] / swx[k]) for k in range(3)], np.float32)

    def _mva_rows(self, tc, idx, pphi):
        rows = []
        for i in idx:
            dt = f(abs(float(self.CHT[i]) - tc.t0))
            dphi = abs(delta_phi(self.PHI[i], pphi))
            rows.append(np.concatenate([[dt, dphi], self.STATIC[i]]).astype(np.float32))
        return np.array(rows, np.float32).reshape(len(idx), 7)

    def net(self, tc):
        return self.nets[1] if tc.calo >= 0 else self.nets[0]

    def refine(self, tc):
        s = self.s
        changed = True
        while changed and tc.nsh > 0:
            changed = False
            worst, worstmva = None, f(100.0)
            pphi = poly_atan2(tc.pos[1], tc.pos[0])
            if tc.hits:
                v = eval_mva(self.net(tc), self._mva_rows(tc, tc.hits, pphi))
                for k in range(len(tc.hits)):
                    if v[k] < worstmva:
                        worstmva, worst = v[k], k
            if worstmva < s["minkeep"]:
                changed = True
                self.diag["refined"] = self.diag.get("refined", 0) + 1
                self.remove_hit(tc, worst)

    def recover(self, tc):
        s = self.s
        changed = True
        while changed:
            changed = False
            pphi = poly_atan2(tc.pos[1], tc.pos[0])
            for i in range(self.N):
                if i in tc.hits:
                    continue
                dt = f(abs(float(self.CHT[i]) - tc.t0))
                if float(dt) < float(s["maxdt"]) + tc.t0err:
                    dphi = abs(delta_phi(self.PHI[i], pphi))
                    if dphi < s["maxdphi"]:
                        x = np.concatenate([[dt, dphi], self.STATIC[i]]).astype(np.float32)[None, :]
                        mva = eval_mva(self.net(tc), x)[0]
                        self.diag["recover_evals"] = self.diag.get("recover_evals", 0) + 1
                        if mva > s["minadd"]:
                            self.add_hit(tc, i)
                            self.diag["recovered"] = self.diag.get("recovered", 0) + 1
                            changed = True

    def run(self):
        s = self.s
        tccol = []
        if s["usecc"]:
            self.calo_seeds(tccol)
        self.hist = TimeSpectrum(s)
        for i in range(self.N):
            self.hist.fill(float(self.CHT[i]), self.NSH[i])
        self.find_peaks(tccol)
        self.assign_hits(tccol)
        out = []
        for tc in tccol:
            self.init_cluster(tc)
            if s["prefilter"]:
                self.prefilter(tc)
            if tc.nsh >= s["minnhits"]:
                self.cluster_mean(tc)
                if s["refine"]:
                    self.refine(tc)
                if s["recover"] and tc.nsh > 0:
                    self.recover(tc)
            if tc.nsh >= s["minnhits"]:
                out.append(tc)
        self.diag["max_tc"] = max(self.diag.get("max_tc", 0), len(tccol))
        self.diag["max_hits"] = max(self.diag.get("max_hits", 0), self.N)
        return [{"hits": list(tc.hits), "t0": tc.t0, "t0err": tc.t0err, "pos": np.asarray(tc.pos, np.float32),
                 "nsh": int(tc.nsh), "calo": int(tc.calo)} for tc in out]


def time_clusters(hits, cc, s, nets, geom=None, diag=None):
    """One event: hits (the ComboHits in collection order: time = correctedTime, x, y, z, ncombo = nStrawHits,
    plane, uvar, wdist), its calo clusters cc (disk, time, energy, cog_x, cog_y[, cog_z]: cog3Vector in the disk
    frame), the settings and the nets (ClusterMVA, ClusterCaloMVA) -> the TimeClusterCollection in stock order:
    [{hits (_strawHitIdxs order), t0, t0err (double), pos (float32 [3]), nsh, calo (index or -1)}]."""
    return Finder(hits, cc, s, nets, geom, diag).run()
