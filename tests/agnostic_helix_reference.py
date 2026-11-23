"""A literal transcription of the stock AgnosticHelixFinder module (CalPatRec, Mu2e Offline v13_41_01, Apache License
2.0), time cluster by time cluster, as the reference for the device port (trigger/device/aprhelix.py) and the stock
helix-seed comparison. No timeout (timeoutMs unset), all three helix directions valid (HelixTool's direction of
propagation then decides nothing and is not transcribed), diagnostics off.

Typing follows the C++: numpy float32 scalars (`f`) for every C++ float, so each operation rounds as float does;
Python floats (IEEE double, no fused multiply-add) where C++ uses double or promotes (double literals, M_PI, `_twopi`,
std::pow(float, int), the double members and results of LsqSums2 / LsqSums4, CLHEP Hep3Vector perp() / phi()); a
value is rounded to float exactly where stock stores it in a float. Float32 scalars are never mixed with Python floats
in arithmetic or comparisons (NumPy 2 would round the Python float to float32): mixed comparisons are made in double.
std::round is halves away from zero; double -> int conversions truncate (x86: INT_MIN when out of range). Division by
zero gives IEEE inf / nan as in C++ (`_div`), never an exception. The per-hit arithmetic of initSeedCircle (the
circle errors and residuals of all hits against the triplet circle) is evaluated with numpy float32 array operations,
element by element the same IEEE operations as the scalar code; every accumulation runs hit by hit in stock order.

The hits of a time cluster (tcHitsFill): the stopping target (0, 0, FLT_MAX) first, then the calorimeter cluster of
the time cluster (its cog moved to Mu2e, then to the tracker frame, z minus caloZOffset), then the cluster's
ComboHits in their stored order, sorted by z descending with libstdc++'s std::sort (`std_sort`: introsort with its
heap-sort fallback at the depth limit, then the final insertion sort; ties included) from the first ComboHit on.

Stock state that crosses triplets and helices is kept: notOnSegment (set by every accepted seed phi line, reset only
by resetFlags), uselessTripletSeed (set after a triplet seed's loops, never reset within a time cluster), the hit
flags `used` and `circleError2` (rewritten by initSeedCircle; recoverPoints reads the triplet-circle errors of the
unused hits), averagedOut (setFlags only sets it; resetFlags clears it when another helix is searched), the
helixPhi / helixPhiCorrection writes of recoverPoints on every candidate, and inHelix (saveHelix marks the hits even
when the seed is then dropped by maxEDepAvg). Quirks kept: the negative-line slope test divides by the positive
line's dz, the last-added z of the seed phi lines is never updated, the calorimeter hit takes part in the 2 pi
resolution, the final circle and the recovery, and maxSeedCircleHits -1 disables its cap.
"""
import math

import numpy as np

from mpr_reference import poly_atan2

f = np.float32
TWOPI = 2. * math.pi                           # static constexpr double _twopi = 2.*M_PI
FLT_MAX = f(np.finfo(np.float32).max)
C_LIGHT = (2.99792458e+8 * 1000.0) / 1.e+9     # CLHEP::c_light, mm/ns
MMT = f(C_LIGHT / 1000.0)                      # static constexpr float _mmTconversion = CLHEP::c_light/1000.0
BZ0 = 1.0117056584                             # BFieldManager z at the detector origin (stock BField job), T
ST, CALO = -2, -1                              # HitType::STOPPINGTARGET, HitType::CALOCLUSTER
CONTINUE, BREAK, GOOD = 0, 1, 2
INT_MIN = -2147483648
DIRECTIONS = {"downstream", "upstream", "unknown"}
DETECTOR_SYSTEM_Z0 = 10171.0                       # mu2e.detectorSystemZ0 (geom_run1_a.txt), mm


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


def _div(a, b):
    """IEEE double division (Python raises on a zero divisor)."""
    a, b = float(a), float(b)
    if b != 0:
        return a / b
    if a != a or a == 0:
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _sqrt(x):
    return math.sqrt(x) if x >= 0 else (x if x != x else math.nan)


def c_round(x):
    """std::round (double): halves away from zero."""
    x = float(x)
    if x != x or math.isinf(x):
        return x
    t = float(math.trunc(x))
    return t + math.copysign(1.0, x) if abs(x - t) >= 0.5 else t


def c_int(x):
    """double -> int conversion (truncation; x86 gives INT_MIN for NaN and out-of-range values)."""
    x = float(x)
    if x != x or x >= 2147483648.0 or x <= -2147483649.0:
        return INT_MIN
    return int(math.trunc(x))


def settings(m, geom=None):
    """AgnosticHelixFinder module table (and the geometry, `geometry()` by default) -> scalars, floats as the stock
    float members."""
    if float(m.get("timeoutMs", 0) or 0) > 0:
        raise ValueError("AgnosticHelixFinder reference does not model timeouts")
    if set(m["validHelixDirections"]) != DIRECTIONS:
        raise ValueError("AgnosticHelixFinder reference needs all three validHelixDirections (HelixTool not "
                         "transcribed)")
    geom = geometry() if geom is None else geom
    fl = lambda k: f(m[k])
    s = {k: fl(k) for k in (
        "isoRad", "eDepFlagThresh", "minDistCut", "minTripletSeedZ", "minTripletDz", "maxTripletDz",
        "minTripletDist", "maxSeedCircleResidual", "maxDphiDz", "maxSeedLineGap", "maxZWindow", "segMultiplier",
        "maxSegmentChi2", "max2PiAmbigResidual", "maxPhiZResidual", "maxNHitsRatio", "maxCircleRecoverSigma",
        "maxLineRecoverSigma", "caloClusterSigma", "minHelixPerpMomentum", "maxHelixPerpMomentum",
        "minHelixMomentum", "maxHelixMomentum", "chi2LineSaveThresh", "maxEDepAvg")}
    s.update({k: int(m[k]) for k in (
        "intenseEventThresh", "intenseClusterThresh", "isoMinHitsNear", "minSeedCircleHits", "minLineSegmentHits",
        "minFinalSeedHits", "minNHelixStrawHits", "minNHelixComboHits")})
    s["maxSeedCircleHits"] = int(m.get("maxSeedCircleHits", -1))
    s.update({k: bool(m[k]) for k in ("findMultipleHelices", "useStoppingTarget", "doIsolationFlag", "doAverageFlag",
                                      "doEDepFlag")})
    # the cached squares of the constructor (float)
    s["minTripletDistSq"] = f(s["minTripletDist"] * s["minTripletDist"])
    s["maxSeedCircleResidualSq"] = f(s["maxSeedCircleResidual"] * s["maxSeedCircleResidual"])
    s["caloClusterSigmaSq"] = f(s["caloClusterSigma"] * s["caloClusterSigma"])
    s["isoRadSq"] = f(s["isoRad"] * s["isoRad"])
    s["minDistCutSq"] = f(s["minDistCut"] * s["minDistCut"])
    s["minHelixMomentumSq"] = f(s["minHelixMomentum"] * s["minHelixMomentum"])
    s["maxHelixMomentumSq"] = f(s["maxHelixMomentum"] * s["maxHelixMomentum"])
    s["segMultiplierSq"] = f(s["segMultiplier"] * s["segMultiplier"])
    # beginRun
    s["bz0"] = f(geom["bz0"])
    s["bz0Conv"] = f(s["bz0"] * MMT)
    s["bz0ConvSq"] = f(s["bz0Conv"] * s["bz0Conv"])
    s["caloZOffset"] = f(geom["caloZOffset"])
    s["disk_origin"] = np.asarray(geom["disk_origin"], np.float64)
    s["tracker_center"] = np.asarray(geom["tracker_center"], np.float64)
    return s


def geometry(bz0=BZ0):
    """The geometry inputs: the field z at the detector origin, caloZOffset = (diskCaseZLength + BPPipeZOffset +
    BPHoleZLength + FEEZLength) / 2 from the calorimeter G4 parameters (diskCaseZLength = crystalZLength +
    crystalCapZLength), the disk origins in Mu2e (the disks are not rotated) and the tracker centre
    (-solenoidOffset, 0, detectorSystemZ0)."""
    from trigger.device import calogeom
    cg = calogeom.CALO_GEOM
    off = cg["crystalZLength"] + cg["crystalCapZLength"]
    off += cg["BPPipeZOffset"]
    off += cg["BPHoleZLength"]
    off += cg["FEEZLength"]
    off /= 2.0
    return {"bz0": bz0, "caloZOffset": off, "disk_origin": np.asarray(calogeom.build()["origin"], np.float64),
            "tracker_center": np.array([-cg["solenoidOffset"], 0.0, DETECTOR_SYSTEM_Z0])}


class LsqSums2:
    """Mu2eUtilities LsqSums2 with the default zero offsets, in double."""
    __slots__ = ("qn", "sw", "sx", "sy", "sx2", "sxy", "sy2")

    def __init__(self):
        self.clear()

    def clear(self):
        self.qn = self.sw = self.sx = self.sy = self.sx2 = self.sxy = self.sy2 = 0.0

    def copy(self):
        c = LsqSums2()
        for k in self.__slots__:
            setattr(c, k, getattr(self, k))
        return c

    def addPoint(self, XX, YY, W=1.0):
        X = float(XX) - 0.0
        Y = float(YY) - 0.0
        W = float(W)
        self.qn += 1
        self.sw += W
        self.sx += X * W
        self.sy += Y * W
        self.sx2 += X * X * W
        self.sxy += X * Y * W
        self.sy2 += Y * Y * W

    def removePoint(self, XX, YY, W=1.0):
        X = float(XX) - 0.0
        Y = float(YY) - 0.0
        W = float(W)
        self.qn -= 1
        self.sw -= 1 * W
        self.sx -= X * W
        self.sy -= Y * W
        self.sx2 -= X * X * W
        self.sxy -= X * Y * W
        self.sy2 -= Y * Y * W

    def dydx(self):
        if self.qn > 1:
            return _div(self.sw * self.sxy - self.sy * self.sx, self.sw * self.sx2 - self.sx * self.sx)
        return 0.0

    def dydxErr(self):
        if self.qn > 2:
            N = self.sw
            D = self.sw * self.sx2 - self.sx * self.sx
            return _sqrt(_div(N, D))
        return 0.0

    def y0(self):
        dx = 0.0 - 0.0
        return (0.0 + _div(self.sy, self.sw)) + self.dydx() * (dx - _div(self.sx, self.sw))

    def y0Err(self):
        if self.qn > 2:
            D = self.sw * self.sx2 - self.sx * self.sx
            dx = 0.0 - 0.0
            return _sqrt(_div(1, self.sw) * (1 + _div((dx * self.sw - self.sx) * (dx * self.sw - self.sx), D)))
        return 0.0

    def chi2Dof(self):
        if self.qn > 2:
            xm = _div(self.sx, self.sw)
            ym = _div(self.sy, self.sw)
            sxx = _div(self.sx2, self.sw) - xm * xm
            sxy = _div(self.sxy, self.sw) - xm * ym
            syy = _div(self.sy2, self.sw) - ym * ym
            chi2 = syy * sxx - sxy * sxy
            chi2 = _div(chi2, sxx)
            chi2 *= _div(self.sw, self.qn - 2)
            return chi2
        return 0.0


class LsqSums4:
    """Mu2eUtilities LsqSums4 (Kasa circle fit) with the default zero offsets, every sum and result in double."""
    __slots__ = ("qn", "sw", "sx", "sy", "sx2", "sxy", "sy2", "sx3", "sx2y", "sxy2", "sy3", "sx4", "sx3y", "sx2y2",
                 "sxy3", "sy4")

    def __init__(self):
        self.clear()

    def clear(self):
        for k in self.__slots__:
            setattr(self, k, 0.0)

    def addPoint(self, XX, YY, W=1.0):
        X = float(XX) - 0.0
        Y = float(YY) - 0.0
        W = float(W)
        X2, Y2, XY = X * X, Y * Y, X * Y
        W_X, W_Y, W_X2, W_Y2 = W * X, W * Y, W * X2, W * Y2
        self.qn += 1
        self.sw += W
        self.sx += W_X
        self.sy += W_Y
        self.sx2 += W_X2
        self.sxy += W * XY
        self.sy2 += W_Y2
        self.sx3 += W_X2 * X
        self.sx2y += W_X2 * Y
        self.sxy2 += W_Y2 * X
        self.sy3 += W_Y2 * Y
        self.sx4 += W_X2 * X2
        self.sx3y += W_X2 * XY
        self.sx2y2 += W_X2 * Y2
        self.sxy3 += W_Y2 * XY
        self.sy4 += W_Y2 * Y2

    def _sig(self):
        m = lambda s: _div(s, self.sw)
        xm, ym, x2m, y2m = m(self.sx), m(self.sy), m(self.sx2), m(self.sy2)
        return {"XX": x2m - xm * xm, "XY": m(self.sxy) - xm * ym, "YY": y2m - ym * ym,
                "X2X": m(self.sx3) - xm * x2m, "X2Y": m(self.sx2y) - ym * x2m, "XY2": m(self.sxy2) - xm * y2m,
                "YY2": m(self.sy3) - ym * y2m, "X2X2": m(self.sx4) - x2m * x2m, "X2Y2": m(self.sx2y2) - x2m * y2m,
                "Y2Y2": m(self.sy4) - y2m * y2m, "xm": xm, "ym": ym}

    @staticmethod
    def _det(s):
        return s["XX"] * s["YY"] - s["XY"] * s["XY"]

    def x0(self):
        s = self._sig()
        return _div((s["YY"] * (s["X2X"] + s["XY2"]) - s["XY"] * (s["X2Y"] + s["YY2"])) / 2, self._det(s)) + 0.0

    def y0(self):
        s = self._sig()
        return _div((s["XX"] * (s["X2Y"] + s["YY2"]) - s["XY"] * (s["X2X"] + s["XY2"])) / 2, self._det(s)) + 0.0

    def radius(self):
        s = self._sig()
        x_0 = self.x0() - 0.0
        y_0 = self.y0() - 0.0
        dx = s["xm"] - x_0
        dy = s["ym"] - y_0
        return _sqrt(s["XX"] + s["YY"] + dx * dx + dy * dy)

    def chi2DofCircle(self):
        s = self._sig()
        r = self.radius()
        sx2 = s["X2X"] + s["XY2"]
        sy2 = s["X2Y"] + s["YY2"]
        chi2 = s["X2X2"] + 2. * s["X2Y2"] + s["Y2Y2"] - _div(
            s["YY"] * sx2 * sx2 + s["XX"] * sy2 * sy2 - 2 * s["XY"] * sx2 * sy2, self._det(s))
        chi2 = _div(chi2, 4 * r * r)
        chi2 *= _div(self.sw, self.qn)
        return chi2


class Line:
    """lineInfo."""
    __slots__ = ("zMin", "zMax", "fitter", "idx", "corr")

    def __init__(self):
        self.fitter = LsqSums2()
        self.clear()

    def clear(self):
        self.zMin = f(0)
        self.zMax = f(0)
        self.fitter.clear()
        self.idx = []
        self.corr = []

    def copy(self):
        c = Line()
        c.zMin, c.zMax, c.fitter, c.idx, c.corr = self.zMin, self.zMax, self.fitter.copy(), list(self.idx), list(
            self.corr)
        return c


class _Finder:
    """The module state for one time cluster: _tcHits (struct of arrays, tcHits order) and the fitters."""

    def __init__(self, hits, tc, cc, s, intense, diag):
        self.s, self.diag, self.intense = s, diag, intense
        self.H = hits
        self.circle, self.line = LsqSums4(), LsqSums2()
        self.lines = []
        self.tfill(tc, cc)

    # --- tcHitsFill ---------------------------------------------------------------------------------------------
    def tfill(self, tc, cc):
        s, H = self.s, self.H
        head = []
        if s["useStoppingTarget"]:
            head.append((ST, f(0), f(0), FLT_MAX))
        if int(tc["calo"]) >= 0:
            k = int(tc["calo"])
            o = s["disk_origin"][int(cc["disk"][k])]
            c = s["tracker_center"]
            gx, gy, gz = o[0] + float(cc["cog_x"][k]), o[1] + float(cc["cog_y"][k]), o[2] + 0.0   # diskToMu2e
            tx, ty, tz = gx - c[0], gy - c[1], gz - c[2]                                            # mu2eToTracker
            head.append((CALO, f(tx), f(ty), f(tz - float(s["caloZOffset"]))))
        body = [(int(i), H["x"][i], H["y"][i], H["z"][i]) for i in tc["hits"]]
        if len(body) > 16 and len(set(float(b[3]) for b in body)) < len(body):
            self.diag["sort_ties"] = self.diag.get("sort_ties", 0) + 1
        std_sort(body, lambda a, b: a[3] > b[3], self.diag)
        rows = head + body
        n = self.n = len(rows)
        self.idx = [r[0] for r in rows]
        self.X = np.array([r[1] for r in rows], np.float32)
        self.Y = np.array([r[2] for r in rows], np.float32)
        self.Z = np.array([r[3] for r in rows], np.float32)
        hi = np.array([max(i, 0) for i in self.idx], np.int64)
        isreal = np.array([i >= 0 for i in self.idx])
        g = lambda k, dt=np.float32: np.where(isreal, H[k][hi], 0).astype(dt)
        self.UVAR, self.VVAR, self.UDX, self.UDY = g("uvar"), g("vvar"), g("udx"), g("udy")
        self.EDEP, self.TIME, self.TVAR = g("edep"), g("time"), g("timevar")
        self.NSH = g("ncombo", np.int64)
        self.ce2 = np.full(n, f(1.0), np.float32)
        self.hphi = np.zeros(n, np.float32)
        self.hphie2 = np.zeros(n, np.float32)
        self.hcorr = [0] * n
        self.inHelix = [False] * n
        self.used = [False] * n
        self.isolated = [False] * n
        self.averagedOut = [False] * n
        self.highEDep = [False] * n
        self.notOnLine = [True] * n
        self.useless = [False] * n
        self.notOnSegment = [True] * n
        self.special = [i < 0 for i in self.idx]
        self.diag["max_tchits"] = max(self.diag.get("max_tchits", 0), n)

    # --- hit helpers ----------------------------------------------------------------------------------------------
    def circle_error2(self, n, xC, yC):
        """computeCircleError2"""
        if self.idx[n] == CALO:
            self.ce2[n] = self.s["caloClusterSigmaSq"]
        elif self.idx[n] >= 0:
            transVar = self.VVAR[n]
            x, y = self.X[n], self.Y[n]
            dx = x - xC
            dy = y - yC
            vx, vy = -self.UDY[n], self.UDX[n]                   # vDir() = (-udir.y, udir.x)
            dxn = dx * vx + dy * vy
            costh2 = dxn * dxn / (dx * dx + dy * dy)
            sinth2 = f(1) - costh2
            self.ce2[n] = self.UVAR[n] * sinth2 + transVar * costh2

    def circle_residual2(self, n, xC, yC, rC):
        """computeCircleResidual2"""
        x = self.X[n] - xC
        y = self.Y[n] - yC
        dd = rC - np.sqrt(x * x + y * y)
        return dd * dd / self.ce2[n]

    def helix_phi(self, n, xC, yC):
        """computeHelixPhi"""
        X = self.X[n] - xC
        Y = self.Y[n] - yC
        Rsq = X * X + Y * Y
        hp = poly_atan2(Y, X)
        if float(hp) < 0.:
            hp = f(float(hp) + TWOPI)
        self.hphi[n] = hp
        if self.idx[n] == CALO:
            dS2 = self.s["caloClusterSigmaSq"]
        else:
            R = np.sqrt(Rsq)
            ux, uy = self.UDX[n], self.UDY[n]
            tanX = Y / R
            tanY = -X / R
            wireErr = np.sqrt(self.UVAR[n])
            pW = wireErr * (ux * tanX + uy * tanY)
            transErr = np.sqrt(self.VVAR[n])
            trX, trY = uy, -ux
            pT = transErr * (trX * tanX + trY * tanY)
            dS2 = pW * pW + pT * pT
        self.hphie2[n] = dS2 / Rsq

    def momentum2(self, radius, dphidz):
        """computeHelixMomentum2"""
        lam = f(1) / dphidz
        return (radius * radius + lam * lam) * self.s["bz0ConvSq"]

    def perp_momentum(self, radius):
        return radius * self.s["bz0Conv"]

    def passes(self, n):
        """passesFlags"""
        s = self.s
        if self.inHelix[n] or self.useless[n]:
            return False
        if s["doIsolationFlag"] and self.isolated[n]:
            return False
        if s["doAverageFlag"] and self.averagedOut[n]:
            return False
        if s["doEDepFlag"] and self.highEDep[n]:
            return False
        return True

    # --- setFlags / resetFlags ----------------------------------------------------------------------------------
    def set_flags(self):
        s = self.s
        iso, avg, edf = s["doIsolationFlag"], s["doAverageFlag"], s["doEDepFlag"]
        if not iso and not avg and not edf:
            return
        n = self.n
        for i in range(n):
            if self.inHelix[i] or self.special[i]:
                continue
            self.highEDep[i] = edf and self.EDEP[i] > s["eDepFlagThresh"]
            if iso or avg:
                near = 0
                sx, sy = self.X[i], self.Y[i]
                self.isolated[i] = iso
                j = 0
                while j < n:
                    if j == i:
                        j += 1
                        continue
                    if not self.isolated[i]:
                        if not avg or self.averagedOut[i]:
                            break
                        if j < i:                                # j = i; continue (then j++)
                            j = i + 1
                            continue
                    if not self.isolated[i] and self.averagedOut[j]:
                        j += 1
                        continue
                    if self.inHelix[j] or self.special[j]:
                        j += 1
                        continue
                    ddx, ddy = float(sx - self.X[j]), float(sy - self.Y[j])   # std::pow(float, 2) in double
                    perp2 = f(ddx * ddx + ddy * ddy)
                    if self.isolated[i]:
                        if perp2 < s["isoRadSq"]:
                            near += 1
                        self.isolated[i] = near < s["isoMinHitsNear"]
                    if avg:
                        if self.averagedOut[i] or self.averagedOut[j]:
                            j += 1
                            continue
                        if perp2 <= s["minDistCutSq"]:
                            self.averagedOut[j] = True
                    j += 1

    def reset_flags(self):
        for i in range(self.n):
            if self.inHelix[i] or self.idx[i] == ST:
                continue
            self.averagedOut[i] = False
            self.notOnLine[i] = True
            self.notOnSegment[i] = True

    # --- triplets -------------------------------------------------------------------------------------------------
    def trip_i(self, n):
        z, index = self.Z[n], self.idx[n]
        if z < self.s["minTripletSeedZ"]:
            return BREAK
        if self.intense and index != CALO and index != ST:
            return BREAK
        if not self.passes(n):
            return CONTINUE
        self.ti = (self.X[n], self.Y[n], z, index)
        return GOOD

    def trip_j(self, n):
        s = self.s
        x, y, z, index = self.X[n], self.Y[n], self.Z[n], self.idx[n]
        ix, iy, iz, ii = self.ti
        dz12 = iz - z
        if ii != CALO and ii != ST and dz12 > s["maxTripletDz"]:
            return BREAK
        if self.intense and ii == ST and index != CALO:
            return BREAK
        ddx, ddy = float(ix - x), float(iy - y)
        if not self.passes(n) or dz12 < s["minTripletDz"] or ddx * ddx + ddy * ddy < float(s["minTripletDistSq"]):
            return CONTINUE
        self.tj = (x, y, z, index)
        return GOOD

    def trip_k(self, n):
        s = self.s
        x, y, z, index = self.X[n], self.Y[n], self.Z[n], self.idx[n]
        ix, iy, _, _ = self.ti
        jx, jy, jz, jj = self.tj
        dz23 = jz - z
        if jj != CALO and jj != ST and dz23 > s["maxTripletDz"]:
            return BREAK
        p2 = lambda ax, ay: (ax - x) * (ax - x) + (ay - y) * (ay - y)        # XYZVectorF Perp2, float
        if not self.passes(n) or dz23 < s["minTripletDz"] or p2(ix, iy) < s["minTripletDistSq"] or \
                p2(jx, jy) < s["minTripletDistSq"]:
            return CONTINUE
        self.tk = (x, y, z, index)
        return GOOD

    def init_triplet(self):
        s = self.s
        for p in (self.ti, self.tj, self.tk):
            self.circle.addPoint(p[0], p[1])
        radius = f(self.circle.radius())
        pt = self.perp_momentum(radius)
        if pt < s["minHelixPerpMomentum"] or pt > s["maxHelixPerpMomentum"]:
            return CONTINUE
        return GOOD

    def init_seed_circle(self):
        s = self.s
        xC, yC, rC = f(self.circle.x0()), f(self.circle.y0()), f(self.circle.radius())
        self.circle.clear()
        n = self.n
        # computeCircleError2 and computeCircleResidual2 of every hit against the triplet circle (float32 arrays)
        isc = np.array([i == CALO for i in self.idx])
        with np.errstate(all="ignore"):
            dx, dy = self.X - xC, self.Y - yC
            dxn = dx * -self.UDY + dy * self.UDX
            costh2 = dxn * dxn / (dx * dx + dy * dy)
            sinth2 = f(1) - costh2
            e2 = np.where(isc, s["caloClusterSigmaSq"], self.UVAR * sinth2 + self.VVAR * costh2).astype(np.float32)
            dd = rC - np.sqrt(dx * dx + dy * dy)
            res2 = dd * dd / e2
        for i in range(n):
            if self.inHelix[i]:
                continue
            if self.idx[i] == ST:
                self.used[i] = False
                continue
            self.ce2[i] = e2[i]
            if (s["maxSeedCircleHits"] < 0 or self.circle.qn < s["maxSeedCircleHits"]) and \
                    res2[i] < s["maxSeedCircleResidualSq"]:
                wP = f(1) / self.ce2[i]
                self.circle.addPoint(self.X[i], self.Y[i], wP)
                self.used[i] = True
            else:
                self.used[i] = False
        if self.circle.qn < s["minSeedCircleHits"]:
            return CONTINUE
        return GOOD

    def init_helix_phi(self):
        xC, yC = f(self.circle.x0()), f(self.circle.y0())
        for i in range(self.n):
            if self.inHelix[i] or self.idx[i] == ST or not self.used[i]:
                continue
            self.helix_phi(i, xC, yC)
            self.hcorr[i] = 0

    # --- phi-z lines ------------------------------------------------------------------------------------------------
    @staticmethod
    def under_slope(phi1, e1, phi2, e2, dz):
        """underEstimateSlope"""
        p1e, p2e = np.sqrt(e1), np.sqrt(e2)
        if phi2 > phi1:
            return abs(((phi2 - p2e) - (phi1 + p1e)) / dz)
        return abs(((phi2 + p2e) - (phi1 - p1e)) / dz)

    def find_seed_phi_lines(self):
        s = self.s
        self.lines = []
        rC = f(self.circle.radius())
        pos, neg = Line(), Line()
        n = self.n
        max_index = n - s["minLineSegmentHits"]
        gap, win, mdz = s["maxSeedLineGap"], s["maxZWindow"], s["maxDphiDz"]
        for i in range(max_index):
            if self.inHelix[i] or not self.used[i] or not self.notOnSegment[i] or self.special[i]:
                continue
            seedZ, seedPhi, seedE2 = self.Z[i], self.hphi[i], self.hphie2[i]
            seedW = f(1) / seedE2
            for L in (pos, neg):
                L.clear()
                L.idx.append(i)
                L.corr.append(0)
                L.zMax = seedZ
                L.fitter.addPoint(seedZ, seedPhi, seedW)
            lnPhi, lnE2, lnZ = seedPhi, seedE2, seedZ
            lpPhi, lpE2, lpZ = seedPhi, seedE2, seedZ
            for j in range(i + 1, n):
                if self.inHelix[j] or not self.used[j] or self.special[j]:
                    continue
                testZ = self.Z[j]
                if seedZ - testZ > win:
                    break
                pdz = lpZ - testZ
                ndz = lnZ - testZ
                if pdz > gap and ndz > gap:
                    break
                tPhi, tE2 = self.hphi[j], self.hphie2[j]
                tW = f(1) / tE2
                pd = lpPhi - tPhi
                pdSq = pd * pd
                pdE = lpE2 + tE2
                nd = lnPhi - tPhi
                ndSq = nd * nd
                ndE = lnE2 + tE2
                with np.errstate(all="ignore"):
                    add_p = (lpPhi > tPhi or pdSq < pdE) and pdz <= gap and float(pdz) > 1.e-10
                    add_p &= (pdSq < pdE or abs((lpPhi - tPhi) / pdz) <= mdz or
                              self.under_slope(lpPhi, lpE2, tPhi, tE2, pdz) <= mdz)
                    if add_p:
                        pos.idx.append(j)
                        pos.corr.append(0)
                        pos.zMin = testZ
                        pos.fitter.addPoint(testZ, tPhi, tW)
                        lpPhi, lpE2 = tPhi, tE2
                    add_n = (tPhi > lnPhi or ndSq < ndE) and ndz <= gap and float(ndz) > 1.e-10
                    add_n &= (ndSq < ndE or abs((lnPhi - tPhi) / pdz) <= mdz or         # stock: / positiveDeltaZ
                              self.under_slope(lnPhi, lnE2, tPhi, tE2, ndz) <= mdz)
                    if add_n:
                        neg.idx.append(j)
                        neg.corr.append(0)
                        neg.zMin = testZ
                        neg.fitter.addPoint(testZ, tPhi, tW)
                        lnPhi, lnE2 = tPhi, tE2
            for L, last, slope_ok in ((pos, lpPhi, lambda v: v <= float(mdz)),
                                      (neg, lnPhi, lambda v: abs(v) <= float(mdz))):
                if L.fitter.qn >= s["minLineSegmentHits"] and slope_ok(L.fitter.dydx()):
                    dphidz = f(L.fitter.dydx())
                    with np.errstate(all="ignore"):
                        p2 = self.momentum2(rC, dphidz)
                    dphi = seedPhi - last
                    if dphi * dphi > s["segMultiplierSq"] * seedE2 and p2 > s["minHelixMomentumSq"] and \
                            p2 < s["maxHelixMomentumSq"] and L.fitter.chi2Dof() <= float(s["maxSegmentChi2"]):
                        self.lines.append(L.copy())
                        for q in L.idx:
                            self.notOnSegment[q] = False
        self.diag["max_lines"] = max(self.diag.get("max_lines", 0), len(self.lines))
        return CONTINUE if len(self.lines) == 0 else GOOD

    def resolve_2pi(self):
        s = self.s
        for L in self.lines:
            slope, icpt = f(L.fitter.dydx()), f(L.fitter.y0())
            serr, ierr = f(L.fitter.dydxErr()), f(L.fitter.y0Err())
            for j in range(self.n):
                if self.inHelix[j] or self.idx[j] == ST or not self.used[j]:
                    continue
                testZ, testPhi, testE2 = self.Z[j], self.hphi[j], self.hphie2[j]
                testW = f(_div(1.0, testE2))
                if j in L.idx:
                    continue
                deltaPhi = slope * testZ + icpt - testPhi
                nn = c_int(c_round(_div(deltaPhi, TWOPI)))
                testPhi = f(float(testPhi) + TWOPI * nn)
                deltaPhi = abs(slope * testZ + icpt - testPhi)
                error2 = testZ * testZ * serr * serr + ierr * ierr + testE2
                if deltaPhi * deltaPhi < s["max2PiAmbigResidual"] * s["max2PiAmbigResidual"] * error2:
                    L.fitter.addPoint(testZ, testPhi, testW)
                    L.idx.append(j)
                    L.corr.append(nn)
                    if testZ < L.zMin:
                        L.zMin = testZ
                    if testZ > L.zMax:
                        L.zMax = testZ

    def refine(self, L):
        """refinePhiLine"""
        s = self.s
        largest, rm = f(0.0), 0
        for i, q in enumerate(L.idx):
            pointPhi = f(float(self.hphi[q]) + TWOPI * L.corr[i])
            pointZ = self.Z[q]
            linePhi = f(L.fitter.dydx() * float(pointZ) + L.fitter.y0())
            deltaPhi = abs(linePhi - pointPhi)
            with np.errstate(all="ignore"):
                res2 = deltaPhi * deltaPhi / self.hphie2[q]
            if res2 > largest:
                largest, rm = res2, i
        if largest > s["maxPhiZResidual"] * s["maxPhiZResidual"]:
            q = L.idx[rm]
            z = self.Z[q]
            phi = f(float(self.hphi[q]) + TWOPI * L.corr[rm])
            w = f(_div(1.0, self.hphie2[q]))
            L.fitter.removePoint(z, phi, w)
            del L.idx[rm]
            del L.corr[rm]
            self.diag["refine_removed"] = self.diag.get("refine_removed", 0) + 1
            return True
        return False

    def init_final_seed(self):
        s = self.s
        seedCircleHits = f(self.circle.qn)
        best, most, chi2dof = 0, 0, f(0.0)
        for i, L in enumerate(self.lines):
            if len(L.idx) < most:
                continue
            if len(L.idx) > most:
                most, best, chi2dof = len(L.idx), i, f(L.fitter.chi2Dof())
                continue
            if len(L.idx) == most and L.fitter.chi2Dof() < float(chi2dof):
                best, chi2dof = i, f(L.fitter.chi2Dof())
        self.circle.clear()
        self.line.clear()
        B = self.lines[best]
        self.line = B.fitter.copy()
        for i in range(self.n):
            if self.inHelix[i] or self.idx[i] == ST:
                continue
            self.used[i] = False
            for j, q in enumerate(B.idx):
                if q == i:
                    self.used[i] = True
                    self.hcorr[i] = B.corr[j]
                    wP = f(_div(1.0, self.ce2[i]))
                    self.circle.addPoint(self.X[i], self.Y[i], wP)
                    break
        nLineHits = f(self.line.qn)
        ratio = seedCircleHits / nLineHits
        if self.circle.qn < s["minFinalSeedHits"] or ratio > s["maxNHitsRatio"]:
            return CONTINUE
        return GOOD

    def recover(self):
        """recoverPoints"""
        s = self.s
        xC, yC, rC = f(self.circle.x0()), f(self.circle.y0()), f(self.circle.radius())
        slope, icpt = f(self.line.dydx()), f(self.line.y0())
        smallest, add, rec = f(0.0), 0, False
        csig2 = s["maxCircleRecoverSigma"] * s["maxCircleRecoverSigma"]
        lsig2 = s["maxLineRecoverSigma"] * s["maxLineRecoverSigma"]
        with np.errstate(all="ignore"):
            for i in range(self.n):
                if self.used[i] or self.inHelix[i] or self.idx[i] == ST:
                    continue
                cres2 = self.circle_residual2(i, xC, yC, rC)
                if cres2 > csig2:
                    continue
                self.helix_phi(i, xC, yC)
                phiP, zP, phiS2 = self.hphi[i], self.Z[i], self.hphie2[i]
                phiD = slope * zP + icpt - phiP
                nn = c_int(c_round(_div(phiD, TWOPI)))
                self.hcorr[i] = nn
                phiP = f(float(phiP) + TWOPI * nn)
                phiD = abs(slope * zP + icpt - phiP)
                pres2 = phiD * phiD / phiS2
                if pres2 > lsig2:
                    continue
                res2 = pres2 + cres2
                if res2 < smallest or not rec:
                    smallest, add = res2, i
                rec = True
            if rec:
                self.used[add] = True
                w = f(_div(1.0, self.ce2[add]))
                self.circle.addPoint(self.X[add], self.Y[add], w)
                xC, yC, rC = f(self.circle.x0()), f(self.circle.y0()), f(self.circle.radius())
                self.circle.clear()
                for i in range(self.n):
                    if not self.used[i] or self.inHelix[i] or self.idx[i] == ST:
                        continue
                    self.circle_error2(i, xC, yC)
                    wP = f(_div(1.0, self.ce2[i]))
                    self.circle.addPoint(self.X[i], self.Y[i], wP)
                xC, yC, rC = f(self.circle.x0()), f(self.circle.y0()), f(self.circle.radius())
                self.line.clear()
                for i in range(self.n):
                    if not self.used[i] or self.inHelix[i] or self.idx[i] == ST:
                        continue
                    zP = self.Z[i]
                    self.helix_phi(i, xC, yC)
                    phiW = f(_div(1.0, self.hphie2[i]))
                    phiP = self.hphi[i]
                    dP = slope * zP + icpt - phiP
                    self.hcorr[i] = c_int(c_round(_div(dP, TWOPI)))
                    phiP = f(float(phiP) + self.hcorr[i] * TWOPI)
                    self.line.addPoint(zP, phiP, phiW)
                self.diag["recovered"] = self.diag.get("recovered", 0) + 1
        return rec

    def viability(self):
        """checkHelixViability"""
        s = self.s
        if self.line.chi2Dof() > float(s["chi2LineSaveThresh"]):
            return CONTINUE
        radius = f(self.circle.radius())
        pt = self.perp_momentum(radius)
        if pt < s["minHelixPerpMomentum"] or pt > s["maxHelixPerpMomentum"]:
            return CONTINUE
        slope = f(self.line.dydx())
        with np.errstate(all="ignore"):
            mom2 = self.momentum2(radius, slope)
        if mom2 < s["minHelixMomentumSq"] or mom2 > s["maxHelixMomentumSq"]:
            return CONTINUE
        return GOOD

    # --- saveHelix ----------------------------------------------------------------------------------------------
    def save(self, itc, tc, out):
        s = self.s
        fit = LsqSums2()
        hits, hphi, pos = [], [], []
        esum, nsum = f(0.0), 0
        for i in range(self.n):
            if self.inHelix[i] or not self.used[i] or self.special[i]:
                continue
            self.inHelix[i] = True
            tres = np.sqrt(self.TVAR[i])
            with np.errstate(all="ignore"):
                fit.addPoint(self.Z[i], self.TIME[i], f(1) / (tres * tres))
            hits.append(self.idx[i])
            hphi.append(f(float(self.hphi[i]) + self.hcorr[i] * TWOPI))
            pos.append(i)
            esum = esum + self.EDEP[i] * f(self.NSH[i])            # ComboHitCollection::eDepAvg
            nsum += int(self.NSH[i])
        edep = f(_div(esum, nsum + 1e-10))
        x0, y0 = f(self.circle.x0()), f(self.circle.y0())
        cx, cy = float(x0), float(y0)
        dfdz = f(self.line.dydx())
        fz0 = f(math.fmod(self.line.y0(), TWOPI))
        if float(fz0) > math.pi:
            fz0 = f(float(fz0) - TWOPI)
        elif float(fz0) < -math.pi:
            fz0 = f(float(fz0) + TWOPI)
        lam = f(_div(1., dfdz))
        seed = {"tc": itc, "t0": fit.y0(), "t0err": fit.y0Err(), "rcent": f(math.sqrt(cx * cx + cy * cy)),
                "fcent": f(0.0 if (cx == 0.0 and cy == 0.0) else math.atan2(cy, cx)),
                "radius": f(self.circle.radius()), "lambda": lam, "fz0": fz0,
                "helicity": 1 if float(lam) > 0 else -1,
                "chi2xy": f(self.circle.chi2DofCircle()), "chi2zphi": f(self.line.chi2Dof()),
                "hits": hits, "hphi": hphi, "slots": pos, "edep_avg": edep, "calo": int(tc["calo"])}
        if edep > s["maxEDepAvg"]:
            self.diag["dropped_edep"] = self.diag.get("dropped_edep", 0) + 1
            return
        out.append(seed)

    # --- findHelix ------------------------------------------------------------------------------------------------
    def find_helix(self, itc, tc, out):
        s, dg = self.s, self.diag
        self.circle.clear()
        self.line.clear()
        if s["doIsolationFlag"] or s["doAverageFlag"]:
            self.set_flags()
        n = self.n
        for i in range(n - 2):
            useless = True
            c = self.trip_i(i)
            if c == CONTINUE:
                continue
            if c == BREAK:
                break
            for j in range(i + 1, n - 1):
                c = self.trip_j(j)
                if c == CONTINUE:
                    continue
                if c == BREAK:
                    break
                for k in range(j + 1, n):
                    c = self.trip_k(k)
                    if c == CONTINUE:
                        continue
                    if c == BREAK:
                        break
                    self.circle.clear()
                    self.line.clear()
                    dg["triplets"] = dg.get("triplets", 0) + 1
                    with np.errstate(all="ignore"):
                        if self.init_triplet() == CONTINUE:
                            continue
                        if self.init_seed_circle() == CONTINUE:
                            continue
                        useless = False
                        dg["seed_circles"] = dg.get("seed_circles", 0) + 1
                        self.init_helix_phi()
                        if self.find_seed_phi_lines() == CONTINUE:
                            continue
                        self.resolve_2pi()
                        for L in self.lines:
                            if len(L.idx) < s["minFinalSeedHits"]:
                                continue
                            while self.refine(L):
                                pass
                        if self.init_final_seed() == CONTINUE:
                            continue
                        dg["final_seeds"] = dg.get("final_seeds", 0) + 1
                        while self.recover():
                            pass
                        if self.viability() == CONTINUE:
                            continue
                    nsh_h = nch_h = nsh_t = nch_t = 0
                    for q in range(n):
                        if self.inHelix[q] or self.special[q]:
                            continue
                        nsh_t += int(self.NSH[q])
                        nch_t += 1
                        if not self.used[q]:
                            continue
                        nsh_h += int(self.NSH[q])
                        nch_h += 1
                    if nsh_h >= s["minNHelixStrawHits"] and nch_h >= s["minNHelixComboHits"]:
                        dg["saved"] = dg.get("saved", 0) + 1
                        with np.errstate(all="ignore"):
                            self.save(itc, tc, out)
                        another = (nsh_t - nsh_h) >= s["minNHelixStrawHits"] and \
                            (nch_t - nch_h) >= s["minNHelixComboHits"]
                        if another and s["doAverageFlag"]:
                            self.reset_flags()
                        return another
            self.useless[i] = useless
        return False


def good_time_cluster(tc, intense, s):
    if intense and int(tc["calo"]) < 0:
        return False
    nh = len(tc["hits"])
    if nh < 3:
        return False
    return not (nh < s["minSeedCircleHits"] or nh < s["minLineSegmentHits"] or nh < s["minNHelixComboHits"])


def agnostic_helices(hits, tcs, cc, s, diag=None):
    """One event: panel hits (x y z udx udy uvar vvar edep time timevar float32, ncombo, in collection order), the
    input time clusters tcs (stock order: {hits: ComboHit indices in stored order, calo: cluster index or -1}) and
    the calo clusters cc (disk, cog_x, cog_y) -> the saved helix seeds in collection order: {tc, t0, t0err (double),
    rcent, fcent, radius, lambda, fz0, chi2xy, chi2zphi, edep_avg (float32; the eDepAvg of the maxEDepAvg cut),
    helicity, hits (ComboHit indices in hhits order), hphi (float32, per hit), slots (their tcHits positions),
    calo}. `diag` (a dict) counts the work: triplets,
    seed circles, final seeds, recoveries, refine removals, saves, seeds dropped by maxEDepAvg, the largest tcHits
    and line counts, helices per time cluster, sorts with ties above the insertion threshold, heap-sorted
    ranges of the z sort."""
    diag = {} if diag is None else diag
    out = []
    intense_event = len(tcs) > s["intenseEventThresh"]
    for itc, tc in enumerate(tcs):
        intense = intense_event or len(tc["hits"]) > s["intenseClusterThresh"]
        if not good_time_cluster(tc, intense, s):
            continue
        F = _Finder(hits, tc, cc, s, intense, diag)
        nh = 0
        while True:
            before = len(out)
            more = F.find_helix(itc, tc, out)
            nh += len(out) - before
            if not (more and s["findMultipleHelices"]):
                break
        diag["max_helices_tc"] = max(diag.get("max_helices_tc", 0), nh)
        diag["findhelix_calls"] = diag.get("findhelix_calls", 0) + 1
    diag["max_seeds_event"] = max(diag.get("max_seeds_event", 0), len(out))
    return out
