"""A literal transcription of the stock RobustHelixFinder module (TrkPatRec, Mu2e Offline v13_41_01, Apache License
2.0) with RobustHelixFit, RobustHelixFinderData / TrkFaceData and MedianCalculator (TrkReco, Mu2eUtilities), time
cluster by time cluster, as the reference for the device port (trigger/device/rhelix.py) and the stock helix-seed
comparison. Covered configuration: median circle init and fit, no triple radius, no hit-error weights, the circle
fit's calo cluster off (UseCaloCluster false), initFZ (not the dz-frequency variant), no hit MVA, no stereo update,
a single output collection; the module's diagnostics are off and the HelixTool direction of propagation (written to
the seed, read by nothing downstream of the comparison) is not transcribed.

Typing follows the C++: numpy float32 scalars (`f`) for every C++ float, so each operation rounds as float does;
Python floats (IEEE double, no fused multiply-add) where C++ uses double or promotes (double literals, M_PI,
CLHEP::twopi, std::pow(float, int), RobustHelix::circleAzimuth's double z, the TH1F axis); a value is rounded to
float exactly where stock stores it in a float. Float32 scalars are never mixed with Python floats in arithmetic
(NumPy 2 would round the Python float to float32). The float libm calls (cosf, sinf of RobustHelix's centre and
position, cbrtf of the triplet weight) are the platform libm's (cal_helix_reference); sqrt of a float is the
correctly rounded float sqrt.

Stock details kept:
- fillFaceOrderedHits: the time cluster's hits with a HitSelectionBits bit and no HitBackgroundBits bit, in stored
  order, sorted by unique panel with libstdc++'s std::sort (ties: introsort, heap-sort fallback, final insertion
  sort; `std_sort`, as in the AgnosticHelixFinder transcription). A face's range [idChBegin, idChEnd) runs from
  its first to its last hit in that order, so it also covers the other face's hits of the same plane (the panels of
  the two faces interleave): every face walk (filterCircleHits, filterHits) visits them in both faces, in face order.
- XYWVec's face is the ordered face within the station (0-3), so the triplet loop of fitCircleMedian skips pairs on
  the same ordered face of different stations; the third hit is only tested against the second.
- prefilterHits: boost's P^2 median (tag::median(with_p_square_quantile), float) of the x and y of the non-outliers.
- MedianCalculator: std::sort by value, the float running weight against 0.5 * the float total (in double).
- initFZ: the TH1F phi histogram with ROOT 6.38's TAxis::FindBin and GetBinCenter; fitFZ's break leaves the inner
  pair loop; resolvePhi runs on every hit (outliers included); a hit's helixPhi is set by initPhi only for the hits
  used at initFZ time (the others keep the input value 0) and is only 2 pi shifted afterwards.
- updateT0: boost's weighted mean and lazy weighted variance (float sums of double terms, the samples staying the
  double times the caller passes): the time cluster's calo cluster first (HelixSeed::caloCluster() is the time
  cluster's), at weight 1/CaloTimeErr^2 and time cc.time() - (z + CaloShowermaxZ) / (pitch beta c) +
  TrkToCaloTimeOffset, the cog moved to Mu2e and the tracker frame as diskToMu2e and mu2eToTracker do; then the hits
  in order, at weight 1/StrawHitTimeErr^2.
- pickBestHelix: of the two helicities' seeds, more hits wins, then the smaller chi2dZPhi (ties: the second).
"""
import ctypes
import math
import sys

import numpy as np

from cal_helix_reference import SHF_BITS, cosf, sinf
from mpr_reference import poly_atan2

f = np.float32
PI = math.pi
TWOPI = 2. * math.pi                         # CLHEP::twopi, static const double twopi = 2*M_PI
OUTLIER = 1 << SHF_BITS["Outlier"]
RESOLVEDPHI = 1 << SHF_BITS["ResolvedPhi"]
NFACES_TOTAL = 72                            # StrawId::_ntotalfaces
C_LIGHT = (2.99792458e+8 * 1000.0) / 1.e+9   # CLHEP::c_light, mm/ns


def _load_cbrtf():
    if not sys.platform.startswith("linux"):
        return None
    try:
        fn = ctypes.CDLL("libm.so.6").cbrtf
        fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]
        return fn
    except (OSError, AttributeError):
        return None


_CBRTF = _load_cbrtf()


def cbrtf(x):
    return f(_CBRTF(float(x))) if _CBRTF is not None else np.cbrt(f(x))


def _mask(names):
    m = 0
    for n in names:
        m |= 1 << SHF_BITS[n]
    return m


def std_sort(a, less, diag=None):
    """libstdc++ (GCC 13) std::sort on the list a, in place: introsort (median of three to the first element,
    unguarded Hoare partition, threshold 16) whose ranges past the depth limit 2 lg(n) are heap sorted
    (std::__partial_sort(first, last, last): __make_heap, then __sort_heap, with __adjust_heap / __push_heap), then
    the final insertion sort. `diag["heap_sort"]` counts the heap-sorted ranges."""
    def lg(n):
        return n.bit_length() - 1

    def move_median_to_first(r, x, y, z):
        if less(a[x], a[y]):
            if less(a[y], a[z]): a[r], a[y] = a[y], a[r]
            elif less(a[x], a[z]): a[r], a[z] = a[z], a[r]
            else: a[r], a[x] = a[x], a[r]
        elif less(a[x], a[z]): a[r], a[x] = a[x], a[r]
        elif less(a[y], a[z]): a[r], a[z] = a[z], a[r]
        else: a[r], a[y] = a[y], a[r]

    def partition(first, last, pivot):
        while True:
            while less(a[first], a[pivot]): first += 1
            last -= 1
            while less(a[pivot], a[last]): last -= 1
            if not first < last:
                return first
            a[first], a[last] = a[last], a[first]
            first += 1

    def adjust_heap(first, hole, length, value):
        top = second = hole
        while second < (length - 1) // 2:
            second = 2 * (second + 1)
            if less(a[first + second], a[first + second - 1]):
                second -= 1
            a[first + hole] = a[first + second]
            hole = second
        if (length & 1) == 0 and second == (length - 2) // 2:
            second = 2 * (second + 1)
            a[first + hole] = a[first + second - 1]
            hole = second - 1
        parent = (hole - 1) // 2                                     # __push_heap
        while hole > top and less(a[first + parent], value):
            a[first + hole] = a[first + parent]
            hole = parent
            parent = (hole - 1) // 2
        a[first + hole] = value

    def heap_sort(first, last):
        length = last - first
        if length >= 2:                                              # __make_heap
            parent = (length - 2) // 2
            while True:
                adjust_heap(first, parent, length, a[first + parent])
                if parent == 0:
                    break
                parent -= 1
        while last - first > 1:                                      # __sort_heap: __pop_heap(first, last, last)
            last -= 1
            value = a[last]
            a[last] = a[first]
            adjust_heap(first, 0, last - first, value)
        if diag is not None:
            diag["heap_sort"] = diag.get("heap_sort", 0) + 1

    def introsort(first, last, depth):
        while last - first > 16:
            if depth == 0:
                heap_sort(first, last)
                return
            depth -= 1
            mid = first + (last - first) // 2
            move_median_to_first(first, first + 1, mid, last - 1)
            cut = partition(first + 1, last, first)
            introsort(cut, last, depth)
            last = cut

    def linear_insert(i):
        v = a[i]; j = i - 1
        while less(v, a[j]):
            a[j + 1] = a[j]; j -= 1
        a[j + 1] = v

    def insertion(first, last):
        for i in range(first + 1, last):
            if less(a[i], a[first]):
                v = a[i]; a[first + 1:i + 1] = a[first:i]; a[first] = v
            else:
                linear_insert(i)

    n = len(a)
    if n:
        introsort(0, n, lg(n) * 2)
        if n > 16:
            insertion(0, 16)
            for i in range(16, n):
                linear_insert(i)
        else:
            insertion(0, n)
    return a


DETECTOR_SYSTEM_Z0 = 10171.0                       # mu2e.detectorSystemZ0 (geom_run1_a.txt), mm


def geometry():
    """The calorimeter disk origins in Mu2e (the disks are not rotated) and the tracker centre (-solenoidOffset, 0,
    detectorSystemZ0)."""
    from trigger.device import calogeom
    cg = calogeom.CALO_GEOM
    return {"disk_origin": np.asarray(calogeom.build()["origin"], np.float64),
            "tracker_center": np.array([-cg["solenoidOffset"], 0.0, DETECTOR_SYSTEM_Z0])}


def settings(m, geom=None):
    """RobustHelixFinder module table (and the geometry, `geometry()` by default) -> scalars (floats as the stock
    float members); raises for the
    configurations the transcription does not cover."""
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
    if bad:
        raise ValueError("RobustHelixFinder reference does not cover " + ", ".join(bad))
    t0c = m.get("T0Calculator", {})
    F = lambda d, k: f(d[k])
    s = {"minnsh": int(m["MinNStrawHits"]), "maxphisep": F(m, "MaxPhiHitSeparation"),
         "maxdr": F(m, "MaxRadiusDiff"), "maxrpull": F(m, "MaxRPull"), "rpullscale": F(m, "RPullScaleF"),
         "maxniter": int(m["MaxIterations"]), "cradres": F(m, "CentralRadialResolution"),
         "cperpres": F(m, "CentralPerpResolution"), "maxdwire": F(m, "MaxWireDistance"),
         "maxdtrans": F(m, "MaxTransDistance"), "maxchisq": F(m, "MaxChisquared"), "maxrwdot": F(m, "MaxRWDot"),
         "prefilter": bool(m["PrefilterHits"]), "tcon_init": bool(m["targetconsistent_init"]),
         "tcon": bool(m["targetconsistent"]), "area_wt": bool(m["UseTripletArea"]),
         "hsel": _mask(m["HitSelectionBits"]), "hbkg": _mask(m["HitBackgroundBits"]),
         "helicities": [int(h) for h in m["Helicities"]], "maxedep": F(m, "maxEDepAvg"),
         "sherr": float(t0c.get("StrawHitTimeErr", 9.7)), "beta": float(t0c.get("StrawHitBeta", 1.0)),
         "tot": bool(t0c.get("UseTOTDrift", True)), "avgdrift": float(t0c.get("AverageDriftTime", 22.5)),
         "calo_terr": float(t0c.get("CaloTimeErr", 0.5)), "calo_toff": float(t0c.get("TrkToCaloTimeOffset", -0.4)),
         "calo_zoff": float(t0c.get("CaloShowermaxZ", -120.0)),
         # RobustHelixFit
         "dontuse": _mask(hf["DontUseFlag"]), "minnhit": int(hf["minNHit"]), "nloops": int(hf["nLoopsdfdz"]),
         "nphibins": int(hf["NPhiHistBins"]), "phifactor": F(hf, "PhiHistRangeFactor"), "minnphi": int(hf["MinNPhi"]),
         "fmaxniter": int(hf["maxniter"]), "minzsep": F(hf, "minzsep"), "maxzsep": F(hf, "maxzsep"),
         "mindphi": F(hf, "mindphi"), "mindist": F(hf, "mindist"), "maxdist": F(hf, "maxdist"),
         "rmin": F(hf, "minR"), "rmax": F(hf, "maxR"), "rcmin": F(hf, "minCenterR"), "rcmax": F(hf, "maxCenterR"),
         "lmin": F(hf, "minAbsLambda"), "lmax": F(hf, "maxAbsLambda"), "targetradius": F(hf, "targetradius"),
         "trackerradius": F(hf, "trackerradius"), "ntripmin": int(hf["ntripleMin"]),
         "ntripmax": int(hf["ntripleMax"]), "iminl": F(hf, "initFZMinLambda"), "imaxl": F(hf, "initFZMaxLambda"),
         "istepl": F(hf, "initFZStepLambda"), "fminl": F(hf, "fitFZMinLambda"), "fmaxl": F(hf, "fitFZMaxLambda"),
         "fstepl": F(hf, "fitFZStepLambda")}
    geom = geometry() if geom is None else geom
    s["disk_origin"] = np.asarray(geom["disk_origin"], np.float64)
    s["tracker_center"] = np.asarray(geom["tracker_center"], np.float64)
    s["maxdphi"] = s["minzsep"] / s["iminl"]
    ma = F(hf, "minArea")
    s["minarea2"] = ma * ma
    s["inbins"] = int((s["imaxl"] - s["iminl"]) / s["istepl"] + f(1))
    s["fnbins"] = int((s["fmaxl"] - s["fminl"]) / s["fstepl"] + f(1))
    return s


# --- small stock functions --------------------------------------------------------------------------------------------

def angles_dphi(phi, ref):
    """Angles::deltaPhi(float&, float): in double, the float argument not updated."""
    d = float(phi) - float(ref)
    while d > PI:
        d -= TWOPI
    while d <= -PI:
        d += TWOPI
    return f(d)


def fit_dphi(phi1, phi2):
    """RobustHelixFit::deltaPhi(float, float)."""
    dphi = f(math.fmod(float(phi2 - phi1), TWOPI))
    if float(dphi) > PI:
        dphi = f(float(dphi) - TWOPI)
    if float(dphi) < -PI:
        dphi = f(float(dphi) + TWOPI)
    return dphi


def c_rint(x):
    """rint: round half to even."""
    return float(np.rint(x))


class Median:
    """MedianCalculator (weighted)."""

    def __init__(self):
        self.v = []
        self.tot = f(0)

    def push(self, val, w):
        self.v.append((f(val), f(w)))
        self.tot = f(self.tot + f(w))

    def wmedian(self):
        v = self.v
        if len(v) == 1:
            return v[0][0]
        std_sort(v, lambda a, b: a[0] < b[0])
        i, s = 0, v[0][1]
        while float(s) < 0.5 * float(self.tot):
            i += 1
            s = f(s + v[i][1])
        return v[i][0]


class PSquare:
    """boost accumulators p_square_quantile_impl, probability 0.5, float."""

    def __init__(self):
        self.n = 0
        self.h = [f(0)] * 5
        self.pos = [f(i) + f(1.) for i in range(5)]
        p = f(0.5)
        self.des = [f(1.), f(1.) + f(2.) * p, f(1.) + f(4.) * p, f(3.) + f(2.) * p, f(5.)]
        self.inc = [f(0.), p / f(2.), p, (f(1.) + p) / f(2.), f(1.)]

    def add(self, x):
        x = f(x)
        self.n += 1
        h, pos, des, inc = self.h, self.pos, self.des, self.inc
        if self.n <= 5:
            h[self.n - 1] = x
            if self.n == 5:
                std_sort(h, lambda a, b: a < b)
            return
        if x < h[0]:
            h[0] = x
            k = 1
        elif h[4] <= x:
            h[4] = x
            k = 4
        else:
            k = next(i for i in range(5) if x < h[i])          # std::upper_bound
        for i in range(k, 5):
            pos[i] = pos[i] + f(1)
        for i in range(5):
            des[i] = des[i] + inc[i]
        for i in (1, 2, 3):
            d = des[i] - pos[i]
            dp = pos[i + 1] - pos[i]
            dm = pos[i - 1] - pos[i]
            with np.errstate(all="ignore"):
                hp = (h[i + 1] - h[i]) / dp
                hm = (h[i - 1] - h[i]) / dm
            if (d >= f(1.) and dp > f(1.)) or (d <= f(-1.) and dm < f(-1.)):
                sd = f(int(d / abs(d)))                          # short sign_d
                hh = h[i] + sd / (dp - dm) * ((sd - dm) * hp + (dp - sd) * hm)
                if h[i - 1] < hh and hh < h[i + 1]:
                    h[i] = hh
                else:
                    if d > f(0):
                        h[i] = h[i] + hp
                    if d < f(0):
                        h[i] = h[i] - hm
                pos[i] = pos[i] + sd

    def result(self):
        return self.h[2]


class PhiHist:
    """The TH1F _hphi(nbins, -phifactor pi, phifactor pi) of extractFZ0 (ROOT 6.38 TAxis, float contents)."""

    def __init__(self, nbins, factor):
        self.n = nbins
        self.xmin = -float(factor) * math.pi
        self.xmax = float(factor) * math.pi
        self.c = [f(0)] * (nbins + 2)

    def find_bin(self, x):
        if x < self.xmin:
            return 0
        if not (x < self.xmax):
            return self.n + 1
        width = (self.xmax - self.xmin) / self.n
        a = int((x - self.xmin) / width)
        return 1 + a - int(x < self.xmin + width * a) + int(self.xmin + width * (a + 1) <= x)

    def fill(self, x):
        b = self.find_bin(x)
        self.c[b] = self.c[b] + f(1)

    def center(self, b):
        width = (self.xmax - self.xmin) / float(self.n)
        return self.xmin + (b - 0.5) * width

    def max_bin(self):
        best, mx = 0, -float(np.finfo(np.float32).max)
        for b in range(1, self.n + 1):
            if float(self.c[b]) > mx:
                mx, best = float(self.c[b]), b
        return best


# --- the helix (RobustHelix) ------------------------------------------------------------------------------------------

class Helix:
    __slots__ = ("rcent", "fcent", "radius", "lam", "fz0", "chi2xy", "chi2zphi", "hel")

    def __init__(self):
        self.rcent, self.fcent, self.radius, self.lam, self.fz0 = f(-1.0), f(0.0), f(-1.0), f(0.0), f(0.0)
        self.chi2xy = self.chi2zphi = f(0.0)
        self.hel = 0

    def copy(self):
        c = Helix()
        for k in self.__slots__:
            setattr(c, k, getattr(self, k))
        return c

    def cx(self):
        return self.rcent * cosf(self.fcent)

    def cy(self):
        return self.rcent * sinf(self.fcent)

    def azimuth(self, z):
        """circleAzimuth(double zpos) -> Float_t"""
        if float(self.lam) != 0.0:
            return f(float(self.fz0) + float(z) / float(self.lam))
        return f(0.0)

    def pitch(self):
        return f(math.copysign(1.0, float(self.radius)) * float(self.lam) /
                 float(np.sqrt(self.radius * self.radius + self.lam * self.lam)))

    def valid_helicity(self):
        h = 0 if float(self.lam) == 0.0 else int(math.copysign(1.0, float(self.lam)))
        return h == self.hel


class _TC:
    """The finder data of one time cluster (_hfResult / tmpResult): the ordered hits and the fit state."""

    def __init__(self, hits, tc, cc, s, diag):
        self.s, self.diag = s, diag
        sel = [int(i) for i in tc["hits"] if (int(hits["flag"][i]) & s["hsel"]) and not (int(hits["flag"][i]) &
                                                                                      s["hbkg"])]
        sid = hits["sid"]
        up = lambda i: ((int(sid[i]) & 0xFC00) >> 10) * 6 + ((int(sid[i]) & 0x380) >> 7)
        if len(sel) > 16 and len(set(up(i) for i in sel)) < len(sel):
            diag["sort_ties"] = diag.get("sort_ties", 0) + 1
        std_sort(sel, lambda a, b: up(a) < up(b), diag)
        n = self.n = len(sel)
        self.idx = sel
        g = lambda k: [f(hits[k][i]) for i in sel]
        self.X, self.Y, self.Z = g("x"), g("y"), g("z")
        self.UDX, self.UDY, self.UVAR, self.VVAR = g("udx"), g("udy"), g("uvar"), g("vvar")
        self.EDEP, self.TIME = g("edep"), g("time")
        self.NSH = [int(hits["ncombo"][i]) for i in sel]
        self.flag = [int(hits["flag"][i]) & ~RESOLVEDPHI for i in sel]
        self.hphi = [f(0.0)] * n
        self.wface, self.uface, self.face = [], [], []
        self.fb, self.fe = [-1] * NFACES_TOTAL, [-1] * NFACES_TOTAL
        for k, i in enumerate(sel):
            v = int(sid[i])
            station, plane, panel = (v & 0xF800) >> 11, ((v & 0xFC00) >> 10) % 2, (v & 0x380) >> 7
            xface = 0 if panel % 2 == 0 else 1                      # orderID
            if station % 2 == 0:
                of = 1 - xface if plane == 0 else xface + 2
            else:
                of = xface if plane == 0 else 3 - xface
            fid = of + station * 4
            self.wface.append(of)
            self.uface.append(((v & 0xFC00) >> 10) * 2 + ((v & 0x80) >> 7))
            self.face.append(fid)
            if self.fb[fid] < 0:
                self.fb[fid] = k
            self.fe[fid] = k + 1
        self.nsh_tot = sum(self.NSH)
        diag["max_hits"] = max(diag.get("max_hits", 0), n)
        self.hx = Helix()
        self.st = set()
        self.t0, self.t0err = float(tc["t0"]), float(tc["t0err"])
        self.calo = None
        k = self.calo_idx = int(tc.get("calo", -1))
        if k >= 0:                                     # diskToMu2e (cog z 0), then mu2eToTracker
            o, c = s["disk_origin"][int(cc["disk"][k])], s["tracker_center"]
            self.calo = (float(np.float32(cc["time"][k])), (o[2] + 0.0) - c[2])
        self.nxysh = self.nxych = self.nzphish = 0

    def copy(self):
        c = object.__new__(_TC)
        c.__dict__.update(self.__dict__)
        c.flag, c.hphi, c.hx, c.st = list(self.flag), list(self.hphi), self.hx.copy(), set(self.st)
        return c

    def use(self, i):
        return not (self.flag[i] & self.s["dontuse"])

    def out(self, i):
        return bool(self.flag[i] & OUTLIER)

    # --- prefilterHits ------------------------------------------------------------------------------------------------
    def prefilter(self):
        s = self.s
        changed, nhit, removed = True, self.n, 0
        while changed and nhit > 0:
            nhit, changed = 0, False
            ax, ay = PSquare(), PSquare()
            for i in range(self.n):
                if self.out(i):
                    continue
                ax.add(self.X[i])
                ay.add(self.Y[i])
                nhit += 1
            mphi = poly_atan2(ay.result(), ax.result())
            maxdphi, worst = f(0.0), -1
            for i in range(self.n):
                if self.out(i):
                    continue
                phi = poly_atan2(self.Y[i], self.X[i])
                dphi = abs(angles_dphi(phi, mphi))
                if dphi > maxdphi:
                    maxdphi, worst = dphi, i
            if maxdphi > s["maxphisep"]:
                self.flag[worst] |= OUTLIER
                removed += self.NSH[worst]
                changed = True
                self.diag["prefiltered"] = self.diag.get("prefiltered", 0) + 1
        return removed

    # --- RobustHelixFit::fitCircleMedian ------------------------------------------------------------------------------
    def circle_median(self, force):
        s, H = self.s, self.hx
        mind2, maxd2 = s["mindist"] * s["mindist"], s["maxdist"] * s["maxdist"]
        accx, accy, accr = Median(), Median(), Median()
        X, Y, W, F = self.X, self.Y, self.NSH, self.wface
        n, nt = self.n, 0
        d2 = lambda a, b: (X[a] - X[b]) * (X[a] - X[b]) + (Y[a] - Y[b]) * (Y[a] - Y[b])
        done = False
        for i in range(n - 2):
            if done:
                break
            if not self.use(i):
                continue
            ri2 = X[i] * X[i] + Y[i] * Y[i]
            for j in range(i + 1, n - 1):
                if done:
                    break
                if not self.use(j) or F[i] == F[j]:
                    continue
                dij = d2(i, j)
                if dij < mind2 or dij > maxd2:
                    continue
                rj2 = X[j] * X[j] + Y[j] * Y[j]
                for k in range(j + 1, n):
                    if not self.use(k) or F[j] == F[k]:
                        continue
                    dik, djk = d2(i, k), d2(j, k)
                    if dik < mind2 or djk < mind2 or dik > maxd2 or djk > maxd2:
                        continue
                    area2 = f(float(dij * djk + dik * djk + dij * dik) -
                              0.5 * float(dij * dij + djk * djk + dik * dik))
                    if area2 < s["minarea2"]:
                        continue
                    delta = (X[k] - X[j]) * (Y[j] - Y[i]) - (X[j] - X[i]) * (Y[k] - Y[j])
                    rk2 = X[k] * X[k] + Y[k] * Y[k]
                    with np.errstate(all="ignore"):
                        num = (Y[k] - Y[j]) * ri2 + (Y[i] - Y[k]) * rj2 + (Y[j] - Y[i]) * rk2
                        cx = f(_ddiv(0.5 * float(num), float(delta)))
                        num = (X[k] - X[j]) * ri2 + (X[i] - X[k]) * rj2 + (X[j] - X[i]) * rk2
                        cy = f(_ddiv(-0.5 * float(num), float(delta)))
                        ex, ey = X[i] - cx, Y[i] - cy
                        rho = np.sqrt(ex * ex + ey * ey)
                        rc = np.sqrt(cx * cx + cy * cy)
                    rmin = abs(rc - rho)
                    rmax = rc + rho
                    if rc > s["rcmin"] and rc < s["rcmax"] and rho > s["rmin"] and rho < s["rmax"] and \
                            rmax < s["trackerradius"] and (not force or rmin < s["targetradius"]):
                        nt += 1
                        wt = cbrtf(f(W[i]) * f(W[j]) * f(W[k])) if not s["area_wt"] else area2
                        accx.push(cx, wt)
                        accy.push(cy, wt)
                        if nt > s["ntripmax"]:
                            done = True
                            break
        self.diag["max_triplets"] = max(self.diag.get("max_triplets", 0), nt)
        if nt > s["ntripmin"]:
            centx, centy = accx.wmedian(), accy.wmedian()
            for i in range(n):
                if not self.use(i):
                    continue
                dx, dy = float(X[i] - centx), float(Y[i] - centy)
                accr.push(np.sqrt(f(dx * dx + dy * dy)), f(W[i]))
            if len(accr.v) == 0:
                return
            rho = accr.wmedian()
            H.rcent = np.sqrt(centx * centx + centy * centy)
            H.fcent = poly_atan2(centy, centx)
            H.radius = rho
            if self.good_circle():
                self.st.add("circleOK")

    def good_circle(self):
        r = self.hx.radius
        return r > self.s["rmin"] and r < self.s["rmax"]

    def good_fz(self):
        H, s = self.hx, self.s
        return H.valid_helicity() and abs(H.lam) > s["lmin"] and abs(H.lam) < s["lmax"]

    def good_helix(self):
        return self.good_circle() and self.good_fz()

    def good_lambda(self, lam):
        s = self.s
        if self.hx.hel == -1:
            return lam > -s["lmax"] and lam < -s["lmin"]
        if self.hx.hel == 1:
            return lam > s["lmin"] and lam < s["lmax"]
        return False

    def fit_circle(self, force):
        self.st.discard("circleOK")
        if "circleInit" not in self.st:
            self.circle_median(force)
            if "circleOK" in self.st:
                self.st.add("circleInit")
            return                                  # init == fit type (median): done
        self.circle_median(force)

    # --- initFZ / fitFZ -------------------------------------------------------------------------------------------
    def resolve(self, i):
        H = self.hx
        phiex = H.azimuth(self.Z[i])
        nloop = int(c_rint(float(phiex - self.hphi[i]) / TWOPI))
        self.hphi[i] = f(float(self.hphi[i]) + nloop * TWOPI)
        self.flag[i] |= RESOLVEDPHI
        return nloop != 0

    def init_fz(self):
        s, H = self.s, self.hx
        H.lam = f(1.0e12)
        H.fz0 = f(0.0)
        cx, cy = H.cx(), H.cy()
        for i in range(self.n):
            if not self.use(i):
                continue
            self.hphi[i] = poly_atan2(self.Y[i] - cy, self.X[i] - cx)
        hist = [0] * s["inbins"]
        counter = 0
        sign = f(-1.) if H.hel == -1 else (f(1.) if H.hel == 1 else f(0))
        for a in range(self.n - 1):
            if not self.use(a):
                continue
            for b in range(a + 1, self.n):
                if not self.use(b) or self.uface[a] == self.uface[b]:
                    continue
                dz = self.Z[b] - self.Z[a]
                if abs(dz) < s["minzsep"] or abs(dz) > s["maxzsep"]:
                    continue
                dphi = fit_dphi(self.hphi[a], self.hphi[b])
                bin_last = -1
                for lp in range(s["nloops"]):
                    dphi_n = float(dphi) + float(lp) * 2 * PI * float(sign)
                    if dphi_n * H.hel < 0 or abs(dphi_n) < float(s["mindphi"]) or abs(dphi_n) > float(s["maxdphi"]):
                        continue
                    lam = f(float(dz) / dphi_n)
                    if lam * sign >= s["imaxl"]:
                        continue
                    elif lam * sign <= s["iminl"]:
                        break
                    bn = int((lam * sign - s["iminl"]) / s["istepl"])
                    if bin_last > 0 and bin_last - bn <= 2:
                        continue
                    hist[bn] += 1
                    counter += 1
                    bin_last = bn
        if counter < s["minnhit"]:
            return False
        swmax, xmp = f(0), f(0)
        st = float(s["istepl"])
        for ix in range(s["inbins"] - 2):
            sw = f(hist[ix] + hist[ix + 1] + hist[ix + 2])
            if sw > swmax:
                xmp = f(float(s["iminl"]) + (st * (ix + 0.5) * hist[ix] + st * (ix + 1 + 0.5) * hist[ix + 1] +
                                             st * (ix + 2 + 0.5) * hist[ix + 2]) / float(sw))
                swmax = sw
        lam = xmp * sign
        if not self.good_lambda(lam):
            return False
        H.lam = lam
        ok, fz0 = self.extract_fz0()
        if ok:
            H.fz0 = fit_dphi(f(0.0), fz0)
            for i in range(self.n):
                self.resolve(i)
            return True
        return False

    def extract_fz0(self):
        s, H = self.s, self.hx
        h = PhiHist(s["nphibins"], s["phifactor"])
        for i in range(self.n):
            if not self.use(i):
                continue
            dphi = fit_dphi(H.azimuth(self.Z[i]), self.hphi[i])
            h.fill(float(dphi))
            h.fill(float(dphi) - TWOPI)
            h.fill(float(dphi) + TWOPI)
        imax = h.max_bin()
        count, fz0 = 0, f(0)
        for b in range(max(0, imax - 1), min(imax + 1, s["nphibins"]) + 1):
            count = int(count + float(h.c[b]))
            fz0 = f(float(fz0) + float(h.c[b]) * h.center(b))
        with np.errstate(all="ignore"):
            fz0 = fz0 / f(count)
        return count >= s["minnphi"], fz0

    def fit_fz(self):
        s, H = self.s, self.hx
        self.st.discard("phizOK")
        if "phizInit" not in self.st:
            if self.init_fz():
                self.st.add("phizInit")
            else:
                return
        sign = f(-1.) if H.hel == -1 else (f(1.) if H.hel == 1 else f(0))
        niter, changed = 0, True
        while changed and niter < s["fmaxniter"]:
            changed = False
            hist = [0] * s["fnbins"]
            for a in range(self.n - 1):
                if not self.use(a):
                    continue
                for b in range(a + 1, self.n):
                    if not self.use(b) or self.uface[a] == self.uface[b]:
                        continue
                    dz = self.Z[b] - self.Z[a]
                    dphi = self.hphi[b] - self.hphi[a]
                    if abs(dphi) < s["mindphi"]:
                        continue
                    with np.errstate(all="ignore"):
                        lam = dz / dphi
                    if self.good_lambda(lam):
                        if lam * sign >= s["fmaxl"]:
                            continue
                        elif lam * sign <= s["fminl"]:
                            break
                        bn = int((lam * sign - s["fminl"]) / s["fstepl"])
                        hist[bn] += 1
            swmax, xmp = f(0), f(0)
            st = float(s["fstepl"])
            for ix in range(s["fnbins"] - 10):
                sw = f(0)
                for l in range(10):
                    sw = sw + f(hist[ix + l])
                if sw > swmax:
                    xmp = f(0)
                    for l in range(10):
                        xmp = f(float(xmp) + st * (ix + l + 0.5) * hist[ix + l])
                    xmp = xmp / sw
                    xmp = xmp + s["fminl"]
                    swmax = sw
            H.lam = xmp * sign
            acc = Median()
            for i in range(self.n):
                if not self.use(i):
                    continue
                acc.push(fit_dphi(H.azimuth(self.Z[i]), self.hphi[i]), f(self.NSH[i]))
            if len(acc.v) == 0:
                return
            dphi = acc.wmedian()
            H.fz0 = fit_dphi(f(0.0), H.fz0 + dphi)
            for i in range(self.n):
                changed |= self.resolve(i)
            niter += 1
        self.diag["max_fz_iter"] = max(self.diag.get("max_fz_iter", 0), niter)
        if self.good_fz():
            self.st.add("phizOK")

    # --- the module's hit filters ---------------------------------------------------------------------------------
    def _geom(self, i, cx, cy):
        """wdir, cvec, cdir for hit i (ROOT GenVector float arithmetic; Unit() scales by 1/R)."""
        ux, uy = self.UDX[i], self.UDY[i]
        dx, dy = self.X[i] - cx, self.Y[i] - cy
        with np.errstate(all="ignore"):
            r = np.sqrt(dx * dx + dy * dy + f(0) * f(0))
            if r == 0:
                cdx, cdy = dx, dy
            else:
                inv = f(1) / r
                cdx, cdy = dx * inv, dy * inv
        return ux, uy, dx, dy, cdx, cdy

    def filter_circle(self):
        s, H = self.s, self.hx
        changed, ngood = 0, 0
        chi2, cnt = f(0), f(1e-10)
        cx, cy = H.cx(), H.cy()
        for fc in range(NFACES_TOTAL):
            if self.fb[fc] < 0:
                continue
            minchi2 = s["maxrpull"]
            best, oldbest = -1, False
            for i in range(self.fb[fc], self.fe[fc]):
                oldout = self.out(i)
                self.flag[i] &= ~OUTLIER
                ux, uy, dx, dy, cdx, cdy = self._geom(i, cx, cy)
                rwdot = ux * cdx + uy * cdy + f(0) * f(0)
                if rwdot > s["maxrwdot"]:
                    self.flag[i] |= OUTLIER
                    continue
                with np.errstate(all="ignore"):
                    dr = np.sqrt(dx * dx + dy * dy + f(0) * f(0)) - H.radius
                if abs(dr) > s["maxdr"]:
                    self.flag[i] |= OUTLIER
                    continue
                rw2 = rwdot * rwdot
                werr, terr = np.sqrt(self.UVAR[i]), np.sqrt(self.VVAR[i])
                with np.errstate(all="ignore"):
                    rres = np.sqrt(f(float(werr * werr * rw2) + float(terr * terr) * (1.0 - float(rw2))))
                    rpull = abs(dr / rres) * s["rpullscale"]
                if rpull > s["maxrpull"]:
                    self.flag[i] |= OUTLIER
                    continue
                if rpull < minchi2:
                    minchi2, best, oldbest = rpull, i, oldout
                self.flag[i] |= OUTLIER
            if best >= 0:
                self.flag[best] &= ~OUTLIER
                ngood += self.NSH[best]
                chi2 = chi2 + minchi2 * minchi2
                if oldbest:
                    changed += 1
                cnt = cnt + f(1.)
        self.nxysh = ngood
        self.nxych = int(cnt)
        H.chi2xy = chi2 / cnt
        return changed

    def filter_hits(self):
        s, H = self.s, self.hx
        changed, ngood = False, 0
        chi2, cnt = f(0), f(1e-10)
        cx, cy = H.cx(), H.cy()
        for fc in range(NFACES_TOTAL):
            if self.fb[fc] < 0:
                continue
            minchi2, best = s["maxchisq"], -1
            for i in range(self.fb[fc], self.fe[fc]):
                if self.out(i):
                    continue
                hphi = poly_atan2(self.Y[i], self.X[i])
                dphi = abs(angles_dphi(hphi, H.fcent))
                ux, uy, dx, dy, cdx, cdy = self._geom(i, cx, cy)
                wtx, wty = -uy, ux                                   # zaxis x wdir
                cpx, cpy = -cdy, cdx                                 # zaxis x cdir
                az = H.azimuth(self.Z[i])
                hx = cx + H.radius * cosf(az)
                hy = cy + H.radius * sinf(az)
                dhx, dhy = self.X[i] - hx, self.Y[i] - hy
                z0 = f(0) * f(0)
                dtrans = abs(dhx * wtx + dhy * wty + z0)
                dwire = abs(dhx * ux + dhy * uy + z0)
                sq = lambda v: float(v) * float(v)                   # std::pow(float, int), double
                wres2 = f(sq(np.sqrt(self.UVAR[i])) + sq(s["cradres"] * (cdx * ux + cdy * uy + z0)) +
                          sq(s["cperpres"] * (cpx * ux + cpy * uy + z0)))
                wtres2 = f(sq(np.sqrt(self.VVAR[i])) + sq(s["cradres"] * (cdx * wtx + cdy * wty + z0)) +
                           sq(s["cperpres"] * (cpx * wtx + cpy * wty + z0)))
                with np.errstate(all="ignore"):
                    chisq = dwire * dwire / wres2 + dtrans * dtrans / wtres2
                if dphi > s["maxphisep"] or abs(dwire) > s["maxdwire"] or abs(dtrans) > s["maxdtrans"] or \
                        chisq > s["maxchisq"]:
                    changed = True
                if chisq <= minchi2:
                    minchi2, best = chisq, i
                self.flag[i] |= OUTLIER
            if best >= 0:
                self.flag[best] &= ~OUTLIER
                ngood += self.NSH[best]
                chi2 = chi2 + minchi2
                cnt = cnt + f(1.)
        self.nzphish = ngood
        H.chi2zphi = chi2 / cnt
        return changed

    def update_t0(self):
        s = self.s
        pitch = self.hx.pitch()
        hwt = f((1.0 / s["sherr"]) ** 2)
        sw, wsum, wsum2, cnt = f(0), f(0), f(0), 0
        samples = []                                    # (double sample, float weight): boost keeps the double
        if self.calo is not None:
            ct, cz = self.calo
            cwt = f((1.0 / s["calo_terr"]) ** 2)
            samples.append((ct - (cz + s["calo_zoff"]) / (float(pitch) * s["beta"] * C_LIGHT) + s["calo_toff"], cwt))
        for i in range(self.n):
            if self.out(i):
                continue
            tflt = float(self.Z[i]) / (float(pitch) * s["beta"] * C_LIGHT)
            t = float(self.TIME[i]) - tflt if s["tot"] else float(self.TIME[i]) - tflt - s["avgdrift"]
            samples.append((t, hwt))
        for x, w in samples:                            # sum_of_weights, weighted_sum, weighted_moment<2>
            cnt += 1
            sw = sw + w
            wsum = f(float(wsum) + x * float(w))
            wsum2 = f(float(wsum2) + float(w) * (x * x))
        if float(sw) > 0.0:
            with np.errstate(all="ignore"):
                mean = wsum / sw
                var = wsum2 / sw - mean * mean
                self.t0 = float(mean)
                self.t0err = float(np.sqrt(max(f(0.0), var) / f(cnt)))

    # --- RobustHelixFinder::fitHelix -----------------------------------------------------------------------------
    def fit_helix(self):
        s, st = self.s, self.st
        niter, changed, xych, fzch = 0, True, True, True
        while True:
            nxy = 0
            while True:
                self.fit_circle(s["tcon"])
                xych = self.filter_circle() > 0
                nxy += 1
                if not ("circleOK" in st and nxy < s["maxniter"] and xych):
                    break
            if self.nxysh < s["minnsh"]:
                st.discard("circleOK")
                niter = s["maxniter"]
            if "circleOK" in st:
                nfz, fzch = 0, False
                while True:
                    self.fit_fz()
                    fzch = self.filter_hits()
                    nfz += 1
                    if not ("phizOK" in st and nfz < s["maxniter"] and fzch):
                        break
                if self.nzphish < s["minnsh"]:
                    st.discard("phizOK")
                    niter = s["maxniter"]
            niter += 1
            changed = fzch or xych
            if not (self.good_helix() and niter < s["maxniter"] and changed):
                break
        if self.good_helix() and "circleOK" in st and "phizOK" in st:
            st.add("helixOK")
            self.update_t0()

    def seed(self, itc):
        H = self.hx
        keep = [i for i in range(self.n) if not self.out(i)]
        esum, nsh = f(0.0), 0
        for i in keep:
            esum = esum + self.EDEP[i] * f(self.NSH[i])
            nsh += self.NSH[i]
        edep = f(float(esum) / (nsh + 1e-10))
        return {"tc": itc, "t0": self.t0, "t0err": self.t0err, "helicity": H.hel, "rcent": H.rcent,
                "fcent": H.fcent, "radius": H.radius, "lambda": H.lam, "fz0": H.fz0, "chi2xy": H.chi2xy,
                "chi2zphi": H.chi2zphi, "hits": [self.idx[i] for i in keep], "hphi": [self.hphi[i] for i in keep],
                "flags": [self.flag[i] for i in keep], "slots": keep, "edep_avg": edep,
                "calo": self.calo_idx}


def _ddiv(a, b):
    if b != 0:
        return a / b
    if a != a or a == 0:
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def robust_helices(hits, tcs, cc, s, diag=None):
    """One event: panel hits (x y z udx udy uvar vvar edep time float32; ncombo, sid, flag; collection order), the
    input time clusters (stock order: {hits: ComboHit indices in stored order, t0, t0err, calo: the calo cluster
    index or -1}) and the calo clusters cc (disk, time) -> the helix seeds in
    collection order: {tc, t0, t0err (double), helicity, rcent, fcent, radius, lambda, fz0, chi2xy, chi2zphi,
    edep_avg (float32), hits (ComboHit indices in hhits order), hphi (float32), flags, slots (positions in the
    face-ordered hits)}. `diag` counts the work (time clusters fitted, prefilter removals, the largest hit and
    triplet counts, z-phi iterations, seeds dropped by maxEDepAvg, sorts with ties above the insertion threshold,
    heap-sorted ranges)."""
    diag = {} if diag is None else diag
    out = []
    for itc, tc in enumerate(tcs):
        R = _TC(hits, tc, cc, s, diag)
        if R.nsh_tot < s["minnsh"]:
            continue
        nrm = R.prefilter() if s["prefilter"] else 0
        if R.nsh_tot - nrm < s["minnsh"]:
            continue
        R.st.add("hitsOK")
        R.fit_circle(s["tcon_init"])
        diag["circle_fits"] = diag.get("circle_fits", 0) + 1
        if "circleOK" not in R.st:
            continue
        vec = []
        for hel in s["helicities"]:
            T = R.copy()
            T.hx.hel = hel
            T.fit_helix()
            if "helixOK" in T.st:
                sd = T.seed(itc)
                if sd["edep_avg"] > s["maxedep"]:
                    diag["dropped_edep"] = diag.get("dropped_edep", 0) + 1
                    continue
                vec.append(sd)
        if not vec:
            continue
        if len(vec) == 1:
            best = 0
        else:
            n1, n2 = len(vec[0]["hits"]), len(vec[1]["hits"])
            if n2 > n1:
                best = 1
            elif n1 > n2:
                best = 0
            else:
                best = 0 if vec[0]["chi2zphi"] < vec[1]["chi2zphi"] else 1
            diag["both_helicities"] = diag.get("both_helicities", 0) + 1
        out.append(vec[best])
    return out
