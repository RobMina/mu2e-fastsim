"""A literal transcription of the stock CalHelixFinder module and CalHelixFinderAlg (Mu2e Offline v13_41_01, Apache
License 2.0), time cluster by time cluster, as the reference for the device port and the stock helix-seed comparison.

Typing follows the C++: numpy float32 scalars (`f`) for every C++ float, so each operation rounds as float does;
numpy float64 (`D`) where C++ uses double or promotes (double literals, M_PI, functions returning double, the double
members of LsqSums4 / LsqSums2, HelixParams); a value is rounded to float exactly where stock stores it in a float.
Float32 scalars are never mixed with Python floats (NumPy 2 would round the Python float to float32): every double
operand is wrapped in `D`, every float literal in `f`. C++ int conversions truncate toward zero (`c_int`, x86
semantics for out-of-range values). Loops run in the stock order: faces descending, panels ascending, hits in
_chHitsToProcess order.

The ComboHit buffer of each CalHelixFinderData is modelled as stock leaves it in memory. CalHelixFinderData::
clearResults() calls _chHitsToProcess.clear(), so after `Helix = TmpHelix` in searchBestTriplet the vector of Helix is
empty but its storage keeps the hits (ComboHit is trivially destructible, nothing reallocates), and all later
`_chHitsToProcess[index]` accesses read and write that storage. Consequences reproduced here: (1) the hit fields that
TmpHelix's findTrack writes (_hphi, _xyWeight, _zphiWeight) never reach Helix; (2) the copies in HelSeed._hhits carry
what doPatternRecognition wrote on the Helix storage after the triplet search (and any input value never
overwritten). A copy construction copies the buffer; an assignment copies every member except the buffer contents.

float32 libm calls (cosf, sinf, atan2f, reached through libstdc++'s <math.h> `using std::cos` etc. on float
arguments) go to the platform libm through ctypes on Linux (`LIBM_FLOAT`), otherwise they are evaluated in double and
rounded. Double functions (sin, cos, atan, atan2, sqrt) use Python's math module, i.e. the platform libm.
"""
import ctypes
import math
import sys

import numpy as np

from mpr_reference import poly_atan2, std_sort

f = np.float32
D = np.float64

F0 = f(0.0)
F1 = f(1.0)
PI = D(math.pi)                                     # M_PI
TWO_PI = D(2.0) * PI                                # 2.*M_PI and 2*M_PI
HALF_PI = PI / D(2.0)                               # M_PI/2.
FPI = f(PI)                                         # static const float pi(M_PI)
FTWOPI = f(2) * FPI                                 # static const float twopi(2*pi)
FHALFPI = f(D(FPI) / D(2.0))                        # static const float halfpi(pi/2.0)

# ------------------------------------------------------------------------------------------------- constants
# RecoDataProducts/inc/StrawHitFlag.hh (StrawHitFlagDetail::bit_type) and src/StrawHitFlag.cc (bitNames), v13_41_01
SHF_BITS = {
    "Stereo": 0, "EnergySelection": 1, "RadiusSelection": 2, "TimeSelection": 3, "BackgroundCluster": 5,
    "Background": 6, "Isolated": 7, "Outlier": 8, "OtherBackground": 9, "TimeDivision": 10, "TimeCluster": 11,
    "CalorimeterSelection": 12, "StrawXTalk": 13, "ElectronicsXTalk": 14, "TrackerSelection": 15, "Active": 16,
    "GoodDOCA": 17, "ResolvedPhi": 18, "CalorimeterPreSelection": 19, "InTime": 20, "PanelCombo": 21, "Track": 22,
    "Dead": 23, "Noisy": 24, "NHitSelection": 25, "StereoLineFit": 26,
}
BKG_MASK = 1 << SHF_BITS["Background"]              # StrawHitFlag::bkg (module goodHitsTimeCluster)
OUTLIER_MASK = 1 << SHF_BITS["Outlier"]             # StrawHitFlag::outlier (filterUsingPatternRecognition)

# DataProducts/inc/StrawId.hh
NPLANES, NPANELS, NFACES, NSTATIONS = 36, 6, 2, 18
NTOTALFACES = NFACES * NPLANES                      # 72
NUPANELS = NPANELS * NPLANES                        # 216
# TrkReco/inc/TrkFaceData.hh (FaceZ_t)
KNPANELS, KNPLANESPERSTATION = 3, 2
# CalPatRec/inc/CalHelixFinderData.hh
KNMAXCHHITS = 5000
# GlobalConstantsService/data/ParticleList.txt line 19 (e-, mass MeV, charge), read by TrkParticle::mass/charge
E_MASS = D(0.510998910)
E_CHARGE = D(-1.0)
# CLHEP/Units/PhysicalConstants.h: c_light = 2.99792458e+8 * m/s with m = 1000 mm, s = 1e9 ns
C_LIGHT = (D(2.99792458e+8) * D(1000.0)) / D(1.e+9)


# ------------------------------------------------------------------------------------------------- float libm
def _load_libm():
    if not sys.platform.startswith("linux"):
        return None
    try:
        lib = ctypes.CDLL("libm.so.6")
        for name, nargs in (("cosf", 1), ("sinf", 1), ("atan2f", 2)):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_float
            fn.argtypes = [ctypes.c_float] * nargs
        return lib
    except (OSError, AttributeError):
        return None


_LIBM = _load_libm()
LIBM_FLOAT = _LIBM is not None


def cosf(x):
    return f(_LIBM.cosf(float(x))) if LIBM_FLOAT else f(math.cos(float(x)))


def sinf(x):
    return f(_LIBM.sinf(float(x))) if LIBM_FLOAT else f(math.sin(float(x)))


def atan2f(y, x):
    return f(_LIBM.atan2f(float(y), float(x))) if LIBM_FLOAT else f(math.atan2(float(y), float(x)))


def sqrtf(x):
    return np.sqrt(f(x))


def c_int(x):
    """C++ float/double -> int: truncation toward zero; x86 cvtt* gives INT_MIN for NaN and out-of-range values."""
    x = float(x)
    if x != x or x >= 2147483648.0 or x <= -2147483649.0:
        return -2147483648
    return int(x)


def nrot_of(dphi):
    """const int nrot = std::abs((dphi + M_PI) / (2.*M_PI)); (std::abs of a double, then truncation)"""
    return c_int(abs((D(dphi) + PI) / TWO_PI))


def phi_0_2pi(x):
    """ROOT TVector2::Phi_0_2pi (double)."""
    x = D(x)
    if x != x:
        return x
    while x >= TWO_PI:
        x -= TWO_PI
    while x < D(0.0):
        x += TWO_PI
    return x


def flag_mask(names):
    """StrawHitFlag(std::vector<std::string>): OR of the named bits; an unknown name throws in stock."""
    m = 0
    for n in names:
        m |= 1 << SHF_BITS[n]
    return m


# StrawId accessors (sid is the uint16 StrawId)
def sid_plane(sid):
    return (sid & 0xFC00) >> 10


def sid_panel(sid):
    return (sid & 0x380) >> 7


def sid_face(sid):
    return (sid & 0x80) >> 7


def sid_station(sid):
    return (sid & 0xF800) >> 11


def sid_upanel(sid):
    return sid_plane(sid) * NPANELS + sid_panel(sid)


def order_id(station, plane, panel):
    """CalHelixFinderData::orderID -> (station, ordered face, ordered panel)."""
    face = 0 if panel % 2 == 0 else 1
    if station % 2 == 0:
        oface = 1 - face if plane == 0 else face + 2
    else:
        oface = face if plane == 0 else 3 - face
    return station, oface, int(panel / 2)


# ------------------------------------------------------------------------------------------------- geometry inputs
def module_geometry(straw_mid_z, panel_origin):
    """CalHelixFinder::beginRun: (zFace[72], phiPanel[216]) as float32.

    straw_mid_z[plane][panel][straw]: z of Straw::getMidPoint() (double) for straws 0 and 1 of each panel;
    panel_origin[plane][panel] = (x, y): Panel::straw0MidPoint(), which at v13_41_01 is Panel::origin() (the panel
    UVW origin, not straw 0's midpoint). Panels loop as stock: each face index is written by its three panels and
    keeps the last write."""
    zface = [F0] * NTOTALFACES
    phipanel = [F0] * NUPANELS
    for ipl in range(NPLANES):
        for ipn in range(NPANELS):
            os_, of, op = order_id(ipl // KNPLANESPERSTATION, ipl % KNPLANESPERSTATION, ipn)
            face_id = of + os_ * NFACES * KNPLANESPERSTATION
            zface[face_id] = f((D(straw_mid_z[ipl][ipn][0]) + D(straw_mid_z[ipl][ipn][1])) / D(2.0))
            ox, oy = panel_origin[ipl][ipn]
            phipanel[face_id * KNPANELS + op] = f(phi_0_2pi(poly_atan2(f(D(oy)), f(D(ox)))))
    return zface, phipanel


def calo_offset(diskCaseZLength, BPPipeZOffset, BPHoleZLength, FEEZLength, FPCarbonZLength, FPFoamZLength):
    """CalHelixFinderAlg::setCalorimeter: fCaloOffset from the calorimeter G4Info doubles (stored as float).
    diskCaseZLength = calorimeter.crystalZLength + calorimeter.crystalCapZLength (DiskCalorimeterMaker)."""
    return f(D(diskCaseZLength) / D(2.0) + (D(BPPipeZOffset) + D(BPHoleZLength) + D(FEEZLength)) / D(2.0)
             - D(FPCarbonZLength) - D(FPFoamZLength))


# ------------------------------------------------------------------------------------------------- fit sums
_LS4 = ("qn", "sw", "sx", "sy", "sx2", "sxy", "sy2", "sx3", "sx2y", "sxy2", "sy3", "sx4", "sx3y", "sx2y2", "sxy3",
        "sy4", "X0", "Y0")


class LsqSums4:
    """Mu2eUtilities LsqSums4, all double; clear() also zeroes the offsets fX0, fY0."""
    __slots__ = _LS4

    def __init__(self, X0=0.0, Y0=0.0):
        self.clear()
        self.X0, self.Y0 = D(X0), D(Y0)

    def clear(self):
        for k in _LS4:
            setattr(self, k, D(0.0))

    def copy(self):
        o = LsqSums4.__new__(LsqSums4)
        for k in _LS4:
            setattr(o, k, getattr(self, k))
        return o

    def _terms(self, XX, YY, W):
        X, Y, W = D(XX) - self.X0, D(YY) - self.Y0, D(W)
        X2, Y2, XY = X * X, Y * Y, X * Y
        WX, WY, WX2, WY2 = W * X, W * Y, W * X2, W * Y2
        return (W, WX, WY, WX2, W * XY, WY2, WX2 * X, WX2 * Y, WY2 * X, WY2 * Y, WX2 * X2, WX2 * XY, WX2 * Y2,
                WY2 * XY, WY2 * Y2)

    def add(self, XX, YY, W=1.0):
        t = self._terms(XX, YY, W)
        self.qn += D(1.0)
        for k, v in zip(_LS4[1:16], t):
            setattr(self, k, getattr(self, k) + v)

    def remove(self, XX, YY, W=1.0):
        t = self._terms(XX, YY, W)
        self.qn -= D(1.0)
        for k, v in zip(_LS4[1:16], t):
            setattr(self, k, getattr(self, k) - v)

    # means and moments
    def xm(self): return self.sx / self.sw
    def ym(self): return self.sy / self.sw
    def x2m(self): return self.sx2 / self.sw
    def xym(self): return self.sxy / self.sw
    def y2m(self): return self.sy2 / self.sw
    def x3m(self): return self.sx3 / self.sw
    def x2ym(self): return self.sx2y / self.sw
    def xy2m(self): return self.sxy2 / self.sw
    def y3m(self): return self.sy3 / self.sw
    def x4m(self): return self.sx4 / self.sw
    def x2y2m(self): return self.sx2y2 / self.sw
    def y4m(self): return self.sy4 / self.sw
    def sigXX(self): return self.x2m() - self.xm() * self.xm()
    def sigXY(self): return self.xym() - self.xm() * self.ym()
    def sigYY(self): return self.y2m() - self.ym() * self.ym()
    def sigX2X(self): return self.x3m() - self.xm() * self.x2m()
    def sigX2Y(self): return self.x2ym() - self.ym() * self.x2m()
    def sigXY2(self): return self.xy2m() - self.xm() * self.y2m()
    def sigYY2(self): return self.y3m() - self.ym() * self.y2m()
    def sigX2X2(self): return self.x4m() - self.x2m() * self.x2m()
    def sigX2Y2(self): return self.x2y2m() - self.x2m() * self.y2m()
    def sigY2Y2(self): return self.y4m() - self.y2m() * self.y2m()
    def det(self): return self.sigXX() * self.sigYY() - self.sigXY() * self.sigXY()

    def x0(self):
        x = (self.sigYY() * (self.sigX2X() + self.sigXY2()) - self.sigXY() * (self.sigX2Y() + self.sigYY2())) \
            / D(2) / self.det()
        return x + self.X0

    def y0(self):
        y = (self.sigXX() * (self.sigX2Y() + self.sigYY2()) - self.sigXY() * (self.sigX2X() + self.sigXY2())) \
            / D(2) / self.det()
        return y + self.Y0

    def radius(self):
        x_0, y_0 = self.x0() - self.X0, self.y0() - self.Y0
        dx, dy = self.xm() - x_0, self.ym() - y_0
        return np.sqrt(self.sigXX() + self.sigYY() + dx * dx + dy * dy)

    def phi0(self):
        Dn = self.sw * self.sx2 - self.sx * self.sx
        p = self.sx2 * self.sy - self.sx * self.sxy
        return p / Dn

    def dfdz(self):
        Dn = self.sw * self.sx2 - self.sx * self.sx
        s = self.sw * self.sxy - self.sy * self.sx
        return s / Dn

    def chi2_circle(self):
        r = self.radius()
        sx2 = self.sigX2X() + self.sigXY2()          # locals shadowing the members, as in stock
        sy2 = self.sigX2Y() + self.sigYY2()
        chi2 = self.sigX2X2() + D(2.) * self.sigX2Y2() + self.sigY2Y2() \
            - (self.sigYY() * sx2 * sx2 + self.sigXX() * sy2 * sy2 - D(2) * self.sigXY() * sx2 * sy2) / self.det()
        chi2 = chi2 / (D(4) * r * r)
        chi2 *= self.sw / self.qn
        return chi2

    def chi2_line(self):
        chi2 = self.sigYY() * self.sigXX() - self.sigXY() * self.sigXY()
        chi2 /= self.sigXX()
        chi2 *= self.sw / self.qn
        return chi2


class LsqSums2:
    """Mu2eUtilities LsqSums2 (double), only what calculateDphiDz_2 uses (offsets 0, unit weights)."""

    def __init__(self):
        self.qn = self.sw = self.sx = self.sy = self.sx2 = self.sxy = self.sy2 = D(0.0)

    def add(self, XX, YY, W=1.0):
        X, Y, W = D(XX) - D(0.0), D(YY) - D(0.0), D(W)
        self.qn += D(1.0)
        self.sw += W
        self.sx += X * W
        self.sy += Y * W
        self.sx2 += X * X * W
        self.sxy += X * Y * W
        self.sy2 += Y * Y * W

    def dydx(self):
        if self.qn > D(1):
            return (self.sw * self.sxy - self.sy * self.sx) / (self.sw * self.sx2 - self.sx * self.sx)
        return D(0.0)


# ------------------------------------------------------------------------------------------------- data structures
class HI:
    """HitInfo_t: face, panel, panelHitIndex (the index into _chHitsToProcess)."""
    __slots__ = ("face", "panel", "phi")

    def __init__(self, face=-1, panel=-1, phi=-1):
        self.face, self.panel, self.phi = face, panel, phi

    def set(self, face, panel, phi):
        self.face, self.panel, self.phi = face, panel, phi

    def copy(self):
        return HI(self.face, self.panel, self.phi)


class HFData:
    """CalHelixFinderData. `buf` is the storage of _chHitsToProcess (see the module docstring)."""

    def __init__(self):
        self.buf = []
        self.oT = None                               # 72 faces: {"b", "e", "pz": [[b, e]] * 3}, shared once filled
        self.zFace = self.phiPanel = None
        self.has_calo = False
        self.helix = None
        self.helicity = 0
        self.goodhits = []
        self.seed, self.cand = HI(), HI()
        self.nStrawHits = self.nComboHits = self.nXYSh = self.nZPhiSh = 0
        self.nFiltPoints = self.nFiltStrawHits = 0
        self.helixChi2 = f(1e10)
        self.sxy, self.szphi = LsqSums4(), LsqSums4()
        self.cx = self.cy = F0                       # _center (XYZVectorF, default 0)
        self.radius = f(-1.)
        self.circle_chisq = f(1e10)
        self.dfdz = f(-1.e6)
        self.fz0 = f(-1.e6)
        self.dfdz_chisq = f(1e10)
        self.used = [0] * KNMAXCHHITS                # std::array<int, kNMaxChHits> _hitsUsed

    def _copy_members(self, o):
        self.oT, self.zFace, self.phiPanel, self.has_calo = o.oT, o.zFace, o.phiPanel, o.has_calo
        self.helix = None if o.helix is None else dict(o.helix)
        self.helicity = o.helicity
        self.goodhits = list(o.goodhits)
        self.seed, self.cand = o.seed.copy(), o.cand.copy()
        self.nStrawHits, self.nComboHits, self.nXYSh, self.nZPhiSh = o.nStrawHits, o.nComboHits, o.nXYSh, o.nZPhiSh
        self.nFiltPoints, self.nFiltStrawHits = o.nFiltPoints, o.nFiltStrawHits
        self.helixChi2 = o.helixChi2
        self.sxy, self.szphi = o.sxy.copy(), o.szphi.copy()
        self.cx, self.cy, self.radius, self.circle_chisq = o.cx, o.cy, o.radius, o.circle_chisq
        self.dfdz, self.fz0, self.dfdz_chisq = o.dfdz, o.fz0, o.dfdz_chisq
        self.used = list(o.used)

    def copy(self):
        """Copy construction: the vector copy carries the hits."""
        n = HFData()
        n._copy_members(self)
        n.buf = [dict(h) for h in self.buf]
        return n

    def assign(self, o):
        """Copy assignment from an object whose _chHitsToProcess is empty: own storage untouched."""
        self._copy_members(o)

    def _clear_fit(self):
        self.goodhits = []
        self.sxy.clear()
        self.szphi.clear()
        self.radius = f(-1.)
        self.circle_chisq = f(1e10)
        self.dfdz = f(-1.e6)
        self.fz0 = f(-1.e6)
        self.dfdz_chisq = f(1e10)
        self.nXYSh = self.nZPhiSh = self.nStrawHits = self.nComboHits = 0
        self.helixChi2 = f(1e10)
        self.seed, self.cand = HI(), HI()

    def clear_temp_variables(self):
        self.buf = []
        self._clear_fit()
        self.nFiltPoints = self.nFiltStrawHits = 0
        self.oT = [{"best": -1, "b": -1, "e": -1, "pz": [[-1, -1] for _ in range(KNPANELS)]}
                   for _ in range(NTOTALFACES)]
        self.used = [0] * KNMAXCHHITS

    def clear_helix_info(self):
        self._clear_fit()

    def clear_results(self):
        # _chHitsToProcess.clear(): storage kept (see the module docstring)
        self._clear_fit()
        self.used = [0] * KNMAXCHHITS


def _iter_from_seed(H, seed):
    """The common face/panel/hit walk: faces seed.face..0, panels (seed panel on the seed face)..2, hits from the
    seed hit on the seed panel."""
    for fc in range(seed.face, -1, -1):
        face = H.oT[fc]
        first = seed.panel if fc == seed.face else 0
        for p in range(first, KNPANELS):
            b, e = face["pz"][p]
            n = e - b
            if n == 0:
                continue
            spi = seed.phi - b if (fc == seed.face and p == seed.panel and seed.phi >= 0) else 0
            for i in range(spi, n):
                yield fc, p, b + i


# ------------------------------------------------------------------------------------------------- algorithm
class CalHelixFinderAlg:
    def __init__(self, a, geom):
        """a: the HelixFinderAlg fhicl table. Settings without a reader in the algorithm (usetarget, xyWeights,
        zWeights, nHitsMaxPerPanel, maxZTripletSearch, dzOverHelPitchCut, minDeltaNShPatRec, plotall) are not
        stored."""
        self.hsel = flag_mask(a["HelixFitSelectionBits"])
        self.bkgsel = flag_mask(a["BackgroundSelectionBits"])
        self.maxHitEnergy = f(a["maxElectronHitEnergy"])
        self.minNHits = int(a["minNHit"])
        self.absMpDfDz = f(a["mostProbableDfDz"])
        self.initDfDz = int(a["initDfDz"])
        self.maxDfDz = f(a["maxDfDz"])
        self.minDfDz = f(a["minDfDz"])
        self.sigmaPhi = f(a["sigmaPhi"])
        self.weightXY = f(a["weightXY"])
        self.targetcon = int(a["targetconsistent"])
        self.weightZPhi = f(a["weightZPhi"])
        self.weight3D = f(a["weight3D"])
        self.maxXDPhi = f(a["maxXDPhi"])
        self.maxPanelToHelixDPhi = f(a["maxPanelToHelixDPhi"])
        self.distPatRec = f(a["distPatRec"])
        self.mindist = f(a["mindist"])
        self.pmin, self.pmax = f(a["minP"]), f(a["maxP"])
        self.tdmin, self.tdmax = f(a["minAbsTanDip"]), f(a["maxAbsTanDip"])
        self.filter = bool(a["filter"])
        self.hitChi2Max = f(a["hitChi2Max"])
        self.chi2xyMax = f(a["chi2xyMax"])
        self.chi2zphiMax = f(a["chi2zphiMax"])
        self.chi2hel3DMax = f(a["chi2hel3DMax"])
        self.dfdzErr = f(a["dfdzErr"])
        self.maxNHitsRatio = f(a["maxNHitsRatio"])
        self.procAllTCs = bool(a["procAllTCs"])
        self.slopeRatioLimit = f(a["slopeRatioLimit"])
        minarea = f(a["minArea"])
        self.minarea2 = minarea * minarea
        # geometry and services
        self.bz = f(D(geom["bz"]))                   # mutable float _bz (cached field z at the tracker origin)
        if "fCaloOffset" in geom:
            self.caloOffset = f(geom["fCaloOffset"])
        else:
            self.caloOffset = calo_offset(**geom["calo_g4info"])
        self.disk_origin = geom["disk_origin"]
        self.tracker_center = geom["tracker_center"]
        # mutable algorithm state
        self.caloTime = self.caloX = self.caloY = self.caloZ = F0
        self.rmin = self.rmax = self.dfdzsign = self.mpDfDz = F0
        self.hdfdz = self.hphi0 = F0
        self.phiCorrectedDefined = 0                 # uninitialised in stock; set before its only reader
        self.fUseDefaultDfDz = 0

    # ---------------------------------------------------------------- small helpers
    def calc_weight(self, h, cx, cy):
        te = f(D(5.) / D(math.sqrt(12.)))
        if h["nsh"] > 1:
            te = f(D(te) * D(1.5))
        tv = te * te
        dx, dy = h["x"] - cx, h["y"] - cy
        dxn = dx * (-h["udy"]) + dy * h["udx"]       # vDir() = (-udir.y, udir.x)
        c2 = dxn * dxn / (dx * dx + dy * dy)
        s2 = F1 - c2
        e2 = h["uvar"] * s2 + tv * c2
        wt = f(D(1.) / D(e2))
        wt = wt * self.weightXY
        return wt

    def calc_phi_weight(self, h, cx, cy, R):
        dx, dy = h["x"] - cx, h["y"] - cy
        dxn = dx * (-h["udy"]) + dy * h["udx"]
        c2 = dxn * dxn / (dx * dx + dy * dy)
        s2 = F1 - c2
        e2 = h["uvar"] * c2 + h["vvar"] * s2
        wt = R * R / e2
        wt = wt * self.weightZPhi
        return wt

    @staticmethod
    def radial_dist(h, cx, cy, R):
        dx, dy = h["x"] - cx, h["y"] - cy
        return sqrtf(dx * dx + dy * dy) - R

    @staticmethod
    def resolve_2pi(h, cx, cy, phi_ref):
        dx, dy = h["x"] - cx, h["y"] - cy
        phi = poly_atan2(dy, dx)
        dphi = phi_ref - phi
        nrot = nrot_of(dphi)
        if D(dphi) < -PI:
            dphi_min = f(D(dphi) + D(nrot + 1) * D(2.) * PI)
            phi_rot = f(D(phi) - D(nrot + 1) * D(2.) * PI)
        else:
            dphi_min = f(D(dphi) - D(nrot) * D(2.) * PI)
            phi_rot = f(D(phi) + D(nrot) * D(2.) * PI)
        h["hphi"] = phi_rot
        return dphi_min

    @staticmethod
    def delta_phi(phi1, phi2):
        dphi = f(math.fmod(float(phi2 - phi1), float(FTWOPI)))
        if dphi > FPI:
            dphi = dphi - FTWOPI
        if dphi < -FPI:
            dphi = dphi + FTWOPI
        return dphi

    # ---------------------------------------------------------------- calorimeter / hit ordering
    def set_calo_cluster(self, H, hits, tc):
        if not tc["has_calo"]:
            idx = tc["hits"]
            if len(idx) != 0 and self.procAllTCs:
                zmax, imax = f(-1e10), 0
                for i, ll in enumerate(idx):
                    hz = hits[ll]["z"]
                    if hz > zmax:
                        zmax, imax = hz, i
                ch = hits[idx[imax]]
                self.caloTime, self.caloX, self.caloY, self.caloZ = ch["time"], ch["x"], ch["y"], ch["z"]
            else:
                self.caloTime = self.caloX = self.caloY = self.caloZ = f(-9999.)
            return
        gpos, tpos = self._cluster_tracker_pos(tc)
        self.caloTime = f(tc["time"])
        self.caloX = f(tpos[0])
        self.caloY = f(tpos[1])
        self.caloZ = f(tpos[2] - D(self.caloOffset))

    def _cluster_tracker_pos(self, tc):
        """diskToMu2e (origin + inverseRotation*cog, rotation = identity) then mu2eToTracker (- trackerCenter)."""
        o = self.disk_origin[tc["disk"]]
        cog = (D(tc["cog_x"]), D(tc["cog_y"]), D(0.0))
        gpos = tuple(D(o[k]) + cog[k] for k in range(3))
        tpos = tuple(gpos[k] - D(self.tracker_center[k]) for k in range(3))
        return gpos, tpos

    def fill_face_ordered_hits(self, H, hits, tc):
        self.set_calo_cluster(H, hits, tc)
        clPhi = f(-9999.)
        if self.caloTime > 0:
            clPhi = poly_atan2(self.caloY, self.caloX)
        ordered = []
        for loc in tc["hits"]:
            ch = hits[loc]
            flag = ch["flag"]
            good = (flag & self.hsel) == self.hsel
            bkg = (flag & self.bkgsel) != 0
            if good and not bkg:
                if ch["edep"] > self.maxHitEnergy:
                    continue
                if self.filter:
                    chPhi = poly_atan2(ch["y"], ch["x"])
                    dphi = chPhi - clPhi
                    if dphi > FPI:
                        dphi = dphi - FTWOPI
                    if dphi < -FPI:
                        dphi = dphi + FTWOPI
                    if abs(dphi) > FPI / f(2):
                        continue
                ordered.append(dict(ch))
        std_sort(ordered, lambda a, b: sid_upanel(a["sid"]) < sid_upanel(b["sid"]))
        nfp, nfs = 0, 0
        for ch in ordered:
            sid = ch["sid"]
            os_, of, op = order_id(sid_station(sid), sid_plane(sid) % 2, sid_panel(sid))
            face_id = of + os_ * NFACES * KNPLANESPERSTATION
            fz = H.oT[face_id]
            pz = fz["pz"][op]
            H.buf.append(dict(ch))
            n = len(H.buf)
            if pz[0] < 0:
                pz[0], pz[1] = n - 1, n
            else:
                pz[1] = n
            if fz["b"] < 0:
                fz["b"], fz["e"] = n - 1, n
            else:
                fz["e"] = n
            nfp += 1
            nfs += ch["nsh"]
        if len(H.buf) > KNMAXCHHITS:
            raise RuntimeError("more ComboHits than _hitsUsed holds (stock writes past the array)")
        H.nFiltPoints, H.nFiltStrawHits = nfp, nfs

    # ---------------------------------------------------------------- top level
    def find_helix(self, H):
        pb = f(abs((C_LIGHT * D(1e-3)) / (D(self.bz) * E_CHARGE)))
        self.rmin = f(D(self.pmin) / (D(pb) * D(math.sqrt(D(1.0) + D(self.tdmax * self.tdmax)))))
        self.rmax = f(D(self.pmax) / (D(pb) * D(math.sqrt(D(1.0) + D(self.tdmin * self.tdmin)))))
        self.dfdzsign = f(1) if H.helicity == 1 else f(-1)
        self.mpDfDz = self.dfdzsign * self.absMpDfDz
        return self.fit_helix(H)

    def fit_helix(self, H):
        retval = False
        self.fUseDefaultDfDz = 0                     # resetTrackParamters
        self.hphi0 = f(-9999.)
        self.hdfdz = self.mpDfDz
        self.do_pattern_recognition(H)
        if H.nStrawHits < self.minNHits:
            pass
        elif H.radius < self.rmin or H.radius > self.rmax:
            pass
        elif H.nXYSh < self.minNHits or H.circle_chisq > self.chi2xyMax:
            pass
        elif (H.nZPhiSh < self.minNHits or H.dfdz_chisq < F0 or H.dfdz_chisq > self.chi2zphiMax
              or abs(H.dfdz) < self.minDfDz or abs(H.dfdz) > self.maxDfDz):
            pass
        else:
            H.goodhits = []
            for fc in range(NTOTALFACES - 1, -1, -1):
                for p in range(KNPANELS):
                    b, e = H.oT[fc]["pz"][p]
                    for i in range(e - b):
                        if H.used[b + i] != 1:
                            continue
                        H.goodhits.append(b + i)
            self.define_helix_params(H)
            retval = H.helix is not None
        return retval

    def define_helix_params(self, H):
        radius = H.radius
        if radius == F0 or H.dfdz == F0:
            H.helix = None
            return
        amsign = f(1) if H.helicity == 1 else f(-1)
        omega = D(amsign / radius)
        dx = amsign * H.cy
        dy = -amsign * H.cx
        phi0 = D(atan2f(dy, dx))
        d0 = D(amsign * (sqrtf(H.cx * H.cx + H.cy * H.cy) - H.radius))
        tandip = D(amsign / (radius * H.dfdz))
        dphi = self.delta_phi(H.fz0 + amsign * FHALFPI, f(phi0))
        z0 = D(dphi) * tandip / omega
        H.helix = {"d0": d0, "phi0": phi0, "omega": omega, "z0": z0, "tanDip": tandip}

    def do_pattern_recognition(self, H):
        use_mpv, use_iw = 1, 1
        trip = H.copy()
        trip.helix = None
        self.search_best_triplet(H, trip)
        if self.fUseDefaultDfDz == 0:
            self.search_best_triplet(H, trip, use_mpv)
        ALL = lambda: HI(NTOTALFACES - 1, 0, -1)
        if H.seed.panel < 0 or H.nXYSh < self.minNHits:
            return
        circle_hits = f(H.sxy.qn)
        rc = self.do_linear_fit_phiz(H, ALL(), use_iw)
        phi_hits = f(H.szphi.qn)
        ratio = f(D(circle_hits) / (D(phi_hits) + D(1e-6)))
        if ratio > self.maxNHitsRatio:
            return
        if H.szphi.qn == D(0.):
            return
        if not rc:
            return
        self.rescue_hits_before_seed(H)
        self.refine_helix_parameters(H, ALL())
        use_phi = 0 if (H.nZPhiSh < self.minNHits or not rc) else 1
        self.rescue_hits(H, ALL(), use_phi)
        if (H.nXYSh - 1) != H.nZPhiSh:
            rc = self.do_linear_fit_phiz(H, ALL(), use_iw)
        elif use_phi:
            H.dfdz_chisq = f(H.szphi.chi2_line())
        if H.nZPhiSh < self.minNHits or not rc:
            rs = self.find_dfdz(H, ALL())
            if rs == 1:
                H.dfdz, H.fz0, H.dfdz_chisq = self.hdfdz, self.hphi0, f(-1.)
        rc = self.do_linear_fit_phiz(H, ALL(), use_iw)
        if rc:
            use_phi = 1
            self.rescue_hits(H, ALL(), use_phi)
            if (H.nXYSh - 1) != H.nZPhiSh:
                rc = self.do_linear_fit_phiz(H, ALL(), use_iw)
            elif use_phi:
                H.dfdz_chisq = f(H.szphi.chi2_line())
            self.refine_helix_parameters(H, ALL())
        # (the _diag > 0 block removes/re-adds the target point in _sxy; diagLevel 0 skips it)
        self.filter_using_pattern_recognition(H)

    def search_best_triplet(self, H, T, use_mpv=0):
        nSh = H.nFiltStrawHits
        tested = 0
        for fc in range(NTOTALFACES - 1, -1, -1):
            for p in range(KNPANELS):
                b, e = H.oT[fc]["pz"][p]
                for i in range(e - b):
                    if H.nStrawHits > (nSh - tested):
                        continue
                    if (nSh - tested) < self.minNHits:
                        continue
                    T.clear_results()
                    self.find_track(HI(fc, p, b + i), T, use_mpv)
                    tested += H.buf[b + i]["nsh"]
                    if (T.nStrawHits > H.nStrawHits) or \
                            (T.nStrawHits == H.nStrawHits and T.helixChi2 < H.helixChi2):
                        H.assign(T)

    # ---------------------------------------------------------------- findTrack and its pieces
    def calculate_track_parameters(self, p1, p2, p3):
        """-> None, or (cx, cy, radius, phi0, dfdz32)."""
        def mag2(a, b):
            dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
            return dx * dx + dy * dy + dz * dz
        d_ij, d_ik, d_jk = mag2(p1, p2), mag2(p1, p3), mag2(p2, p3)
        area2 = f(D(d_ij * d_jk + d_ik * d_jk + d_ij * d_ik)
                  - D(0.5) * D(d_ij * d_ij + d_jk * d_jk + d_ik * d_ik))
        if area2 < self.minarea2:
            return None
        x_m = f(D(p3[0] + p1[0]) / D(2.))
        y_m = f(D(p3[1] + p1[1]) / D(2.))
        x_n = f(D(p3[0] + p2[0]) / D(2.))
        y_n = f(D(p3[1] + p2[1]) / D(2.))
        m = f(D(-1.) * D(p3[0] - p1[0]) / D(p3[1] - p1[1]))
        c = y_m - x_m * m
        k = f(D(-1.) * D(p3[0] - p2[0]) / D(p3[1] - p2[1]))
        t = y_n - x_n * k
        if np.isfinite(m) and np.isfinite(k) and D(abs(k)) > D(1e-9):
            q = m / k
            if q > F0 and (q - f(c_int(q))) > self.slopeRatioLimit:
                x_n = f(D(p1[0] + p2[0]) / D(2.))
                y_n = f(D(p1[1] + p2[1]) / D(2.))
                k = f(D(-1.) * D(p1[0] - p2[0]) / D(p1[1] - p2[1]))
                t = y_n - x_n * k
        if not np.isfinite(m) or not np.isfinite(k) or D(abs(m - k)) < D(1e-6):
            return None
        x0 = (t - c) / (m - k)
        y0 = m * x0 + c
        dx3, dy3, dz32 = p3[0] - x0, p3[1] - y0, p3[2] - p2[2]
        radius = sqrtf(dx3 * dx3 + dy3 * dy3)
        dx2, dy2 = p2[0] - x0, p2[1] - y0
        phi0 = poly_atan2(dy2, dx2)
        dphi32 = poly_atan2(dy3, dx3) - phi0
        if D(dphi32 * self.dfdzsign) < D(0.):
            dphi32 = f(D(dphi32) + TWO_PI)
        mp = D(self.mpDfDz)
        dfdz = dphi32 / dz32
        diff = abs(dfdz - self.mpDfDz)
        diff_plus = f(abs((D(dphi32) + TWO_PI) / D(dz32) - mp))
        while diff_plus < diff:
            dphi32 = f(D(dphi32) + TWO_PI)
            dfdz = dphi32 / dz32
            diff = abs(dfdz - self.mpDfDz)
            diff_plus = f(abs((D(dphi32) + TWO_PI) / D(dz32) - mp))
        diff_minus = f(abs((D(dphi32) - TWO_PI) / D(dz32) - mp))
        while diff_minus < diff:
            dphi32 = f(D(dphi32) - TWO_PI)
            dfdz = dphi32 / dz32
            diff = abs(dfdz - self.mpDfDz)
            diff_minus = f(abs((D(dphi32) - TWO_PI) / D(dz32) - mp))
        if abs(dfdz) < self.minDfDz or abs(dfdz) > self.maxDfDz:
            dfdz = self.mpDfDz
        return x0, y0, radius, phi0, dfdz

    def calculate_dphidz_2(self, H, seed, nhits, X0, Y0):
        s = LsqSums2()
        phi0 = F0
        counter = 0
        first = True
        for fc in range(seed.face, -1, -1):
            face = H.oT[fc]
            fp = seed.panel if fc == seed.face else 0
            for p in range(fp, KNPANELS):
                if counter > nhits:
                    break
                b, e = face["pz"][p]
                n = e - b
                if n == 0:
                    continue
                spi = seed.phi - b if (fc == seed.face and p == seed.panel and seed.phi >= 0) else 0
                for i in range(spi, n):
                    idx = b + i
                    if H.used[idx] != 1:
                        continue
                    h = H.buf[idx]
                    phi = poly_atan2(h["y"] - Y0, h["x"] - X0)
                    if first:
                        phi0 = phi
                        first = False
                    if D(phi - phi0) > PI:
                        phi = f(D(phi) - TWO_PI)
                    if D(phi - phi0) < -PI:
                        phi = f(D(phi) + TWO_PI)
                    s.add(h["z"], phi)
                    counter += 1
        return f(s.dydx())

    def find_dfdz(self, H, seed):
        nb = 50
        hist = [F0] * nb
        minX, maxX = F0, f(0.025)
        stepX = (maxX - minX) / f(nb)
        cx, cy = H.cx, H.cy
        ns = NSTATIONS
        nh = [0] * (ns + 1)
        pv = [F0] * (ns + 1)
        zv = [F0] * (ns + 1)
        nsw = 0
        zCl = self.caloZ
        phiCl = poly_atan2(self.caloY - cy, self.caloX - cx)
        if phiCl < F0:
            phiCl = f(D(phiCl) + TWO_PI)
        pv[ns], zv[ns], nh[ns] = phiCl, zCl, 1
        for fc, p, idx in _iter_from_seed(H, seed):
            if H.used[idx] != 1:
                continue
            h = H.buf[idx]
            ist = sid_station(h["sid"])
            phi = poly_atan2(h["y"] - cy, h["x"] - cx)
            if phi < F0:
                phi = f(D(phi) + TWO_PI)
            zv[ist] = zv[ist] + h["z"]
            if nh[ist] == 0:
                pv[ist] = phi
                nsw += 1
            else:
                dphi = phi - pv[ist]
                nrot = nrot_of(dphi)
                if D(dphi) > PI:
                    phi = f(D(phi) - D(nrot) * D(2.) * PI)
                elif D(dphi) < -PI:
                    phi = f(D(phi) + D(nrot + 1) * D(2.) * PI)
                pv[ist] = (pv[ist] * f(nh[ist]) + phi) / f(nh[ist] + 1)
            nh[ist] += 1
        for i in range(ns):
            if nh[i] > 1:
                zv[i] = zv[i] / f(nh[i])
        for i in range(ns):
            if nh[i] == 0:
                continue
            phi_ref, z_ref = pv[i], zv[i]
            for j in range(i + 1, ns + 1):
                if nh[j] == 0:
                    continue
                dphi = pv[j] - phi_ref
                dz = zv[j] - z_ref
                if D(abs(dz)) < D(10e-10):
                    continue
                if D(dz) < D(0.):
                    if not H.has_calo:
                        dz = -dz
                    else:
                        raise RuntimeError("CalHelixFinderAlg::findDfDz: Delta z < 0 (stock throws)")
                dphidz = dphi / dz * self.dfdzsign
                x = dphidz
                dx = f(TWO_PI / D(dz))
                nmin_f = (self.minDfDz - x) / dx
                nmax_f = (self.maxDfDz - x) / dx
                nmin = c_int(nmin_f if nmin_f < F0 else nmin_f + F1)
                nmax = c_int(nmax_f - F1 if nmax_f < F0 else nmax_f)
                nchoices = 0 if (nmax_f < nmin_f or nmax < nmin) else nmax - nmin + 1
                if nchoices <= 0:
                    continue
                weight = F1
                for n in range(nmin, nmax + 1):
                    xx = dphidz + f(n) * dx
                    b = max(0, min(nb - 1, c_int((xx - minX) / stepX)))
                    hist[b] = hist[b] + weight
        ixmax = c_int(maxX / stepX)
        swmax, xmp = F0, F0
        for ix in range(ixmax - 1):
            sw = hist[ix] + hist[ix + 1]
            if sw > swmax:
                xmp = f((D(stepX) * (D(ix) + D(0.5)) * D(hist[ix])
                         + D(stepX) * (D(ix + 1) + D(0.5)) * D(hist[ix + 1])) / D(sw))
                swmax = sw
        if nsw < 2:
            self.hdfdz = self.mpDfDz
        else:
            self.hdfdz = xmp * self.dfdzsign
        phi0, sdphi, sn = F0, F0, 0
        for i in range(ns + 1):
            if nh[i] == 0:
                continue
            if sn == 0:
                phi0 = pv[i] - zv[i] * self.hdfdz
                sdphi = F0
                sn = 1
            else:
                shifted = phi0 + zv[i] * self.hdfdz
                dphi = pv[i] - shifted
                nrot = nrot_of(dphi)
                if D(dphi) < -PI:
                    dmin = f(D(dphi) + D(nrot + 1) * D(2.) * PI)
                else:
                    dmin = f(D(dphi) - D(nrot) * D(2.) * PI)
                sdphi = sdphi + dmin
                sn += 1
        self.hphi0 = phi0 + sdphi / f(sn)
        return 1

    def count_used_hits(self, H, seed):
        nch, npt = 0, 0
        for fc, p, idx in _iter_from_seed(H, seed):
            if H.used[idx] > 0:
                nch += 1
                npt += H.buf[idx]["nsh"]
        return nch, npt

    def find_track(self, seed, H, use_mpv=0):
        mode, mode0 = 0, 0
        seed_hit = H.buf[seed.phi]
        seed_facez = H.zFace[seed.face]
        H.used[seed.phi] = 1
        toll_min = f(100.)
        weight, wtarget = f(1.), f(0.1)
        remove_target = True
        dfdz = self.mpDfDz
        last_facez = seed_facez
        p1 = (F0, F0, F0)
        p2 = (seed_hit["x"], seed_hit["y"], seed_hit["z"])
        p3 = (self.caloX, self.caloY, self.caloZ)
        pd = (p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2])
        pd_r = sqrtf(pd[0] * pd[0] + pd[1] * pd[1] + pd[2] * pd[2])
        if not H.has_calo and D(pd_r) < D(1e-10):
            return
        tp = self.calculate_track_parameters(p1, p2, p3)
        if tp is None:
            return
        cx, cy, radius, phi0, dfdz = tp
        if self.initDfDz == 1:
            if self.find_dfdz(H, seed) == 1:
                dfdz = self.hdfdz
        toll_max = f(abs(TWO_PI / D(dfdz))) if dfdz != F0 else f(1.e6)
        if use_mpv == 1:
            dfdz = self.hdfdz
            toll_max = f(abs(TWO_PI / D(dfdz))) if dfdz != F0 else f(1.e6)
        last_index = HI()
        sxy = LsqSums4()
        sxy.add(p2[0], p2[1], D(1.))
        sxy.add(p3[0], p3[1], D(1.))
        sxy.add(D(0.), D(0.), wtarget)
        npoints = seed_hit["nsh"]
        ncombo = 1
        z_phi0 = p2[2]
        for fc in range(seed.face, -1, -1):
            face = H.oT[fc]
            fp = seed.panel if fc == seed.face else 0
            dz_ = H.zFace[fc] - last_facez
            phi = phi0 + dz_ * dfdz
            he_x = cx + radius * cosf(phi)
            he_y = cy + radius * sinf(phi)
            he_phi = poly_atan2(he_y, he_x)
            if he_phi < F0:
                he_phi = f(D(he_phi) + TWO_PI)
            good, trip = HI(), HI()
            face_chi2 = f(1e10)
            for p in range(fp, KNPANELS):
                b, e = face["pz"][p]
                n = e - b
                if n == 0:
                    continue
                spi = seed.phi + 1 - b if (fc == seed.face and p == seed.panel and seed.phi >= 0) else 0
                dp = he_phi - H.phiPanel[fc * KNPANELS + p]
                if D(dp) > PI:
                    dp = f(D(dp) - TWO_PI)
                if D(dp) < -PI:
                    dp = f(D(dp) + TWO_PI)
                if abs(dp) > self.maxPanelToHelixDPhi:
                    continue
                for i in range(spi, n):
                    h = H.buf[b + i]
                    dx, dy = he_x - h["x"], he_y - h["y"]
                    dist = sqrtf(dx * dx + dy * dy)
                    dxy_max = self.distPatRec + self.dfdzErr * abs(dz_)
                    if dist <= dxy_max:
                        if dist < face_chi2:
                            face_chi2 = dist
                            good.set(fc, p, b + i)
            if good.face < 0:
                continue
            h = H.buf[good.phi]
            H.used[good.phi] = 1
            ncombo += 1
            npoints += h["nsh"]
            sxy.add(h["x"], h["y"], weight)
            if mode == 1:
                cx, cy, radius = f(sxy.x0()), f(sxy.y0()), f(sxy.radius())
            phi0 = poly_atan2(h["y"] - cy, h["x"] - cx)
            z_phi0 = H.zFace[good.face]
            last_facez = z_phi0
            if mode == 0:
                mode0 += 1
            dz_seed = f(D(-1.) * D(z_phi0 - seed_facez))
            if dz_seed > toll_min and dz_seed < toll_max:
                if remove_target:
                    trip = good.copy()
            if trip.face >= 0 and ncombo >= 2 and use_mpv == 0:
                p1 = (h["x"], h["y"], h["z"])
                cx, cy, radius = f(sxy.x0()), f(sxy.y0()), f(sxy.radius())
                phi0 = poly_atan2(h["y"] - cy, h["x"] - cx)
                if use_mpv == 0:
                    dfdz = self.calculate_dphidz_2(H, seed, ncombo, cx, cy)
                elif use_mpv == 1:
                    dfdz = self.hdfdz
                if abs(dfdz) > self.maxDfDz or abs(dfdz) < self.minDfDz:
                    p1 = (F0, F0, F0)
                    dfdz = self.mpDfDz
                else:
                    remove_target = False
                    mode = 1
                    last_index = trip.copy()
        if npoints < self.minNHits:
            return
        H.sxy = sxy.copy()
        H.nXYSh = npoints
        H.radius = f(sxy.radius())
        H.cx, H.cy = f(sxy.x0()), f(sxy.y0())
        H.circle_chisq = f(sxy.chi2_circle())
        H.nStrawHits = npoints
        H.nComboHits = ncombo
        H.dfdz = dfdz
        H.fz0 = phi0 - dfdz * z_phi0
        H.dfdz_chisq = f(-1.)
        rc = self.refine_helix_parameters(H, seed)
        if rc < 0:
            return
        radius_end = H.radius
        npoints = H.nStrawHits
        if self.find_dfdz(H, seed) == 1:
            H.dfdz, H.fz0, H.dfdz_chisq = self.hdfdz, self.hphi0, f(-1.)
        rc_phiz = self.do_linear_fit_phiz(H, seed)
        if rc_phiz:
            dfdz_end, phi0_end = H.dfdz, H.fz0
            ncombo, npoints = self.count_used_hits(H, seed)
        else:
            dfdz_end, phi0_end = self.hdfdz, self.hphi0
        chi2 = f(H.sxy.chi2_circle()) if ncombo > 2 else f(-1)
        H.nStrawHits = npoints
        H.nComboHits = ncombo
        H.helixChi2 = chi2
        H.seed = seed.copy()
        H.cand = last_index
        H.cx, H.cy = f(H.sxy.x0()), f(H.sxy.y0())
        H.radius = radius_end
        H.circle_chisq = f(H.sxy.chi2_circle())
        H.fz0 = phi0_end
        H.dfdz = dfdz_end
        if not rc_phiz:
            H.dfdz_chisq = f(-1.)

    # ---------------------------------------------------------------- phi-z line fit
    def add_calo_to_fit_phiz(self, H):
        dfdz, phi0 = H.dfdz, H.fz0
        hcx, hcy = H.cx, H.cy
        if H.sxy.qn > D(0):
            hcx, hcy = f(H.sxy.x0()), f(H.sxy.y0())
        zCl = self.caloZ
        phiCl0 = poly_atan2(self.caloY - hcy, self.caloX - hcx)
        shifted = phi0 + zCl * dfdz
        dphi = shifted - phiCl0
        nrot = nrot_of(dphi)
        if D(dphi) < -PI:
            dmin = f(D(dphi) + D(nrot + 1) * D(2.) * PI)
            phiCl = f(D(phiCl0) - D(nrot + 1) * D(2.) * PI)
        else:
            dmin = f(D(dphi) - D(nrot) * D(2.) * PI)
            phiCl = f(D(phiCl0) + D(nrot) * D(2.) * PI)
        xdphi = abs(dmin) / self.sigmaPhi
        if D(xdphi) < D(2.) * D(self.maxXDPhi):
            H.szphi.add(zCl, phiCl, f(784.))
            H.nZPhiSh += 1

    def find_good_face_hit(self, H, info, good, chi2):
        fid = info["faceId"]
        face = H.oT[fid]
        z = H.zFace[fid]
        hcx, hcy, radius = H.cx, H.cy, H.radius
        weight = info["weight"]
        sd = info["seed"]
        if H.sxy.qn > D(0):
            hcx, hcy, radius = f(H.sxy.x0()), f(H.sxy.y0()), f(H.sxy.radius())
        fp = sd.panel if fid == sd.face else 0
        for p in range(fp, KNPANELS):
            b, e = face["pz"][p]
            n = e - b
            if n == 0:
                continue
            spi = sd.phi - b if (fid == sd.face and p == sd.panel and sd.phi >= 0) else 0
            for i in range(spi, n):
                idx = b + i
                h = H.buf[idx]
                info["dz"] = z - info["zlast"]
                phi_ref = z * info["dfdz"] + info["phi0"]
                dphi = abs(self.resolve_2pi(h, hcx, hcy, phi_ref))
                err = self.sigmaPhi
                if info["uiw"] == 1:
                    weight = self.calc_phi_weight(h, hcx, hcy, radius)
                    if D(weight) > D(0.):
                        err = f(D(1.) / D(sqrtf(weight)))
                h["zw"] = weight
                xdphi = dphi / err
                if H.used[idx] != 1:
                    continue
                if info["cleanup"] == 1 and xdphi > self.maxXDPhi:
                    H.used[idx] = 0
                    info["nrem"] += 1
                    continue
                if xdphi < chi2:
                    chi2 = xdphi
                    good.set(fid, p, idx)
        return chi2

    def find_worst_chi2(self, H, info, worst, hchi2):
        weight = info["weight"]
        for fc, p, idx in _iter_from_seed(H, info["seed"]):
            if H.used[idx] != 1:
                continue
            h = H.buf[idx]
            z = H.zFace[fc]
            s = H.szphi.copy()
            phi = h["hphi"]
            if info["uiw"] == 1:
                weight = h["zw"]
            s.remove(z, phi, weight)
            c2 = f(s.chi2_line())
            if c2 < hchi2:
                worst.set(fc, p, idx)
                hchi2 = c2
        return hchi2

    def find_worst_resid(self, H, info, worst, resid):
        weight = info["weight"]
        for fc, p, idx in _iter_from_seed(H, info["seed"]):
            if H.used[idx] != 1:
                continue
            h = H.buf[idx]
            z = H.zFace[fc]
            phi = f(D(z) * H.szphi.dfdz() + H.szphi.phi0())
            dphi = abs(h["hphi"] - phi)
            err = self.sigmaPhi
            if info["uiw"] == 1:
                weight = h["zw"]
                if D(weight) > D(0.):
                    err = f(D(1.) / D(sqrtf(weight)))
            xdphi = dphi / err
            if xdphi > resid:
                worst.set(fc, p, idx)
                resid = xdphi
        return resid

    def _remove_phiz(self, H, w):
        h = H.buf[w.phi]
        H.used[w.phi] = 0
        H.szphi.remove(H.zFace[w.face], h["hphi"], h["zw"])
        H.nZPhiSh -= h["nsh"]
        return f(H.szphi.chi2_line())

    def do_linear_fit_phiz(self, H, seed, uiw=0, cleanup=1):
        saved = list(H.used)
        success = False
        min_fit = 5
        weight = F0
        if uiw == 0:
            weight = f(D(1.) / D(self.sigmaPhi * self.sigmaPhi))
        info = {"dfdz": H.dfdz, "phi0": H.fz0, "seed": seed, "weight": weight, "uiw": uiw, "dz": F0, "zlast": F0,
                "nrem": 0, "cleanup": cleanup, "faceId": -1}
        H.szphi.clear()
        H.nZPhiSh = 0
        if H.has_calo:
            self.add_calo_to_fit_phiz(H)
        count = 0
        for fc in range(seed.face, -1, -1):
            good = HI()
            info["faceId"] = fc
            fchi2 = self.find_good_face_hit(H, info, good, f(1e10))
            if good.face < 0:
                continue
            h = H.buf[good.phi]
            z = H.zFace[good.face]
            H.szphi.add(z, h["hphi"], h["zw"])
            H.nZPhiSh += h["nsh"]
            count += 1
            if count == 1:
                info["zlast"], info["dz"] = z, F0
            if (count >= min_fit and D(fchi2) < D(2.)
                    and abs(D(info["dfdz"]) - H.szphi.dfdz()) < D(8.e-4)
                    and H.szphi.dfdz() * D(self.dfdzsign) > D(0.)
                    and D(-1.) * D(info["dz"]) >= D(self.mindist)):
                info["dfdz"] = f(H.szphi.dfdz())
                info["phi0"] = f(H.szphi.phi0())
                info["zlast"] = z
        self.phiCorrectedDefined = 1
        if cleanup == 1 and H.szphi.qn >= D(min_fit):
            if H.szphi.chi2_line() > D(self.chi2zphiMax):
                guard = 0
                while True:                                           # NEXT_ITERATION
                    guard += 1
                    if guard > 100000:
                        raise RuntimeError("doLinearFitPhiZ cleanup does not terminate (stock would loop)")
                    iw = HI()
                    chi2min = self.find_worst_chi2(H, info, iw, f(1e10))
                    if iw.panel >= 0 and H.nZPhiSh > self.minNHits:
                        chi2min = self._remove_phiz(H, iw)
                    while True:                                       # CHECK_RESIDUALS
                        iw = HI()
                        self.find_worst_resid(H, info, iw, self.maxXDPhi)
                        if iw.panel >= 0 and H.nZPhiSh > self.minNHits:
                            chi2min = self._remove_phiz(H, iw)
                            continue
                        break
                    if H.nZPhiSh <= self.minNHits:
                        chi2min = f(H.szphi.chi2_line())
                    if (chi2min >= self.chi2zphiMax or iw.panel >= 0) and H.nZPhiSh > self.minNHits:
                        continue
                    break
        if H.nZPhiSh >= self.minNHits and H.szphi.chi2_line() < D(self.chi2zphiMax):
            success = True
        if H.szphi.qn < D(min_fit) or H.szphi.dfdz() * D(self.dfdzsign) < D(0.):
            success = False
        elif success:
            H.fz0 = f(H.szphi.phi0())
            H.dfdz = f(H.szphi.dfdz())
            H.dfdz_chisq = f(H.szphi.chi2_line())
        if not success:
            H.used = saved
            H.dfdz_chisq = f(-1.)
        return success

    # ---------------------------------------------------------------- circle fits
    def do_weighted_circle_fit(self, H, seed, hcx, hcy, R):
        """Returns the updated (HelCenter.x, HelCenter.y, Radius) that stock writes through its references."""
        H.sxy.clear()
        H.nXYSh = 0
        H.nComboHits = 0
        if self.caloTime > F0:
            H.sxy.add(self.caloX, self.caloY, D(1.) / D(100.))
            H.nXYSh += 1
            H.nComboHits += 1
        if self.targetcon == 1:
            H.sxy.add(D(0.), D(0.), D(1.) / D(900.))
            H.nXYSh += 1
            H.nComboHits += 1
        for fc, p, idx in _iter_from_seed(H, seed):
            if H.used[idx] != 1:
                continue
            h = H.buf[idx]
            wt = self.calc_weight(h, hcx, hcy)
            h["xyw"] = wt
            H.sxy.add(h["x"], h["y"], wt)
            H.nXYSh += h["nsh"]
            H.nComboHits += 1
        R = f(H.sxy.radius())
        hcx, hcy = f(H.sxy.x0()), f(H.sxy.y0())
        H.cx, H.cy = f(H.sxy.x0()), f(H.sxy.y0())
        H.radius = f(H.sxy.radius())
        H.circle_chisq = f(H.sxy.chi2_circle())
        return hcx, hcy, R

    def search_worst_hit_circle(self, H, seed, hcx, hcy, R):
        worst_chi2 = self.hitChi2Max
        iw = HI()
        for fc, p, idx in _iter_from_seed(H, seed):
            if H.used[idx] != 1:
                continue
            h = H.buf[idx]
            dr = self.radial_dist(h, hcx, hcy, R)
            c2 = dr * dr * h["xyw"]
            if c2 > worst_chi2:
                worst_chi2 = c2
                iw.set(fc, p, idx)
        return iw, worst_chi2

    def clean_up_circle(self, H, seed):
        chi2_min = f(-1.)
        iw = HI()
        for fc, p, idx in _iter_from_seed(H, seed):
            if H.used[idx] != 1:
                continue
            h = H.buf[idx]
            s = H.sxy.copy()
            s.remove(h["x"], h["y"], h["xyw"])
            c2 = f(s.chi2_circle())
            if c2 < chi2_min:                         # never true for chi2 >= 0 (stock starts at -1)
                chi2_min = c2
                iw.set(fc, p, idx)
        return iw

    def _remove_xy(self, T, iw):
        h = T.buf[iw.phi]
        T.sxy.remove(h["x"], h["y"], h["xyw"])
        T.nXYSh -= h["nsh"]
        T.used[iw.phi] = 0
        T.nComboHits -= 1
        T.nStrawHits -= h["nsh"]

    def refine_helix_parameters(self, T, seed):
        saved = list(T.used)
        r = f(T.sxy.radius())
        hcx, hcy = f(T.sxy.x0()), f(T.sxy.y0())
        hcx0, hcy0, r0 = hcx, hcy, r
        rc = 0
        hcx, hcy, r = self.do_weighted_circle_fit(T, seed, hcx, hcy, r)
        iw, worst = self.search_worst_hit_circle(T, seed, hcx, hcy, r)
        chi2 = f(T.sxy.chi2_circle())
        chi2_min = chi2
        if not (chi2 <= self.chi2xyMax and worst <= self.hitChi2Max):
            while True:                                               # NEXT_ITERATION
                iw = self.clean_up_circle(T, seed)
                if iw.panel >= 0:
                    self._remove_xy(T, iw)
                    r = f(T.sxy.radius())
                    hcx, hcy = f(T.sxy.x0()), f(T.sxy.y0())
                    hcx, hcy, r = self.do_weighted_circle_fit(T, seed, hcx, hcy, r)
                    chi2_min = f(T.sxy.chi2_circle())
                while True:                                           # CHECK_RESIDUALS
                    iw, worst = self.search_worst_hit_circle(T, seed, hcx, hcy, r)
                    if iw.panel >= 0:
                        self._remove_xy(T, iw)
                        r = f(T.sxy.radius())
                        hcx, hcy = f(T.sxy.x0()), f(T.sxy.y0())
                        hcx, hcy, r = self.do_weighted_circle_fit(T, seed, hcx, hcy, r)
                        chi2_min = f(T.sxy.chi2_circle())
                        continue
                    break
                if chi2_min >= self.chi2xyMax and iw.panel >= 0:
                    if T.nStrawHits > self.minNHits:
                        continue
                    rc = -1
                elif chi2_min < self.chi2xyMax:
                    rc = 1
                break
        if rc >= 0:
            T.cx, T.cy = f(T.sxy.x0()), f(T.sxy.y0())
            T.radius = f(T.sxy.radius())
            T.circle_chisq = f(T.sxy.chi2_circle())
        else:
            T.used = saved
            self.do_weighted_circle_fit(T, seed, hcx0, hcy0, r0)
        return rc

    # ---------------------------------------------------------------- hit rescue
    def rescue_hits_before_seed(self, H):
        sd = H.seed
        dfdz = H.dfdz
        phi0 = H.fz0 + dfdz * H.zFace[sd.face]
        x0, y0, radius = H.cx, H.cy, H.radius
        hcx, hcy = x0, y0
        last_facez = H.zFace[sd.face]
        first = sd.face if sd.face >= 0 else 0
        resc_sh, resc_pt = 0, 0
        for fc in range(first, NTOTALFACES):
            face = H.oT[fc]
            face_chi2 = f(1e10)
            good = HI()
            dz_ = H.zFace[fc] - last_facez
            phi = phi0 + dz_ * dfdz
            he_x = x0 + radius * cosf(phi)
            he_y = y0 + radius * sinf(phi)
            he_phi = poly_atan2(he_y, he_x)
            if he_phi < F0:
                he_phi = f(D(he_phi) + TWO_PI)
            for p in range(KNPANELS - 1, -1, -1):
                b, e = face["pz"][p]
                n = e - b
                if n == 0:
                    continue
                if fc == sd.face and p == sd.panel and sd.phi >= 0:
                    n = sd.phi - b
                dp = he_phi - H.phiPanel[fc * KNPANELS + p]
                if D(dp) > PI:
                    dp = f(D(dp) - TWO_PI)
                if D(dp) < -PI:
                    dp = f(D(dp) + TWO_PI)
                if abs(dp) > self.maxPanelToHelixDPhi:
                    continue
                for i in range(n - 1, -1, -1):
                    idx = b + i
                    if H.used[idx] >= 1:
                        continue
                    h = H.buf[idx]
                    dx, dy = he_x - h["x"], he_y - h["y"]
                    dist = sqrtf(dx * dx + dy * dy)
                    max_dist = self.distPatRec + self.dfdzErr * abs(dz_)
                    if dist <= max_dist:
                        if dist < face_chi2:
                            good.set(fc, p, idx)
                            face_chi2 = dist
            if good.panel < 0:
                continue
            h = H.buf[good.phi]
            last_facez = H.zFace[fc]
            weight = self.calc_weight(h, hcx, hcy)
            H.sxy.add(h["x"], h["y"], weight)
            H.nXYSh += h["nsh"]
            x0, y0, radius = f(H.sxy.x0()), f(H.sxy.y0()), f(H.sxy.radius())
            hcx, hcy = x0, y0
            phi0 = h["hphi"]
            h["xyw"] = weight
            H.used[good.phi] = 1
            resc_sh += h["nsh"]
            resc_pt += 1
        H.cx, H.cy = x0, y0
        H.radius = radius
        H.circle_chisq = f(H.sxy.chi2_circle())
        H.nComboHits += resc_pt
        H.nStrawHits += resc_sh

    @staticmethod
    def is_face_used(H, face):
        for p in range(KNPANELS):
            b, e = face["pz"][p]
            for i in range(e - b):
                if H.used[b + i] == 1:
                    return True
        return False

    def rescue_hits(self, H, seed, use_phi):
        phiwt = f(-9999.)
        dfdz, phi0 = H.dfdz, H.fz0
        r = f(H.sxy.radius())
        hcx, hcy = f(H.sxy.x0()), f(H.sxy.y0())
        hcx, hcy, r = self.do_weighted_circle_fit(H, seed, hcx, hcy, r)
        chi2 = f(H.sxy.chi2_circle())
        if chi2 >= self.chi2xyMax:
            return
        while True:                                                   # NEXT_ITERATION
            chi2_min = self.hitChi2Max
            dphi_chi2 = F0
            ib = HI()
            wt_best, phiwt_best = f(-1), f(-1)
            for fc in range(seed.face, -1, -1):
                face = H.oT[fc]
                fp = seed.panel if fc == seed.face else 0
                if self.is_face_used(H, face):
                    continue
                for p in range(fp, KNPANELS):
                    b, e = face["pz"][p]
                    n = e - b
                    if n == 0:
                        continue
                    spi = seed.phi - b if (fc == seed.face and p == seed.panel and seed.phi >= 0) else 0
                    for i in range(spi, n):
                        idx = b + i
                        h = H.buf[idx]
                        if H.used[idx] >= 1:
                            continue
                        dr = self.radial_dist(h, hcx, hcy, r)
                        wt = self.calc_weight(h, hcx, hcy)
                        dr_chi2 = dr * dr * wt
                        if use_phi == 1 and self.phiCorrectedDefined:
                            phi_pred = H.zFace[fc] * dfdz + phi0
                            dphi = phi_pred - h["hphi"]
                            phiwt = self.calc_phi_weight(h, hcx, hcy, r)
                            dphi_chi2 = dphi * dphi * phiwt
                            x_pred = hcx + r * cosf(phi_pred)
                            y_pred = hcy + r * sinf(phi_pred)
                            w_hel = self.calc_weight(h, x_pred, y_pred) * self.weight3D
                            xr2 = (h["x"] - x_pred) * (h["x"] - x_pred)
                            yr2 = (h["y"] - y_pred) * (h["y"] - y_pred)
                            hit_chi2 = (xr2 + yr2) * w_hel
                        else:
                            hit_chi2 = dr_chi2
                        if hit_chi2 < chi2_min and dr_chi2 < self.hitChi2Max and dphi_chi2 < self.hitChi2Max:
                            x, y = h["x"], h["y"]
                            H.sxy.add(x, y, wt)
                            H.szphi.add(H.zFace[fc], h["hphi"], phiwt)
                            H.nXYSh += h["nsh"]           # stock does not take these back after the trial
                            H.nZPhiSh += h["nsh"]
                            h["xyw"] = wt
                            h["zw"] = phiwt
                            if H.sxy.chi2_circle() < D(self.chi2xyMax):
                                if use_phi == 1:
                                    if H.szphi.chi2_line() < D(self.chi2zphiMax):
                                        chi2_min = hit_chi2
                                        ib.set(fc, p, idx)
                                        wt_best, phiwt_best = wt, phiwt
                                else:
                                    chi2_min = hit_chi2
                                    ib.set(fc, p, idx)
                                    wt_best = wt
                            H.sxy.remove(x, y, wt)
                            H.szphi.remove(H.zFace[fc], h["hphi"], phiwt)
            if ib.panel < 0:
                break
            n_used, id_same = 0, -1
            b, e = H.oT[ib.face]["pz"][ib.panel]
            for l in range(e - b):
                if H.used[b + l] == 1:
                    n_used += 1
                    id_same = l
            h = H.buf[ib.phi]
            if n_used >= 1 or id_same >= 0:
                H.used[ib.phi] = 10
                continue
            H.sxy.add(h["x"], h["y"], wt_best)
            H.nXYSh += h["nsh"]
            if use_phi == 1:
                H.szphi.add(H.zFace[ib.face], h["hphi"], phiwt_best)
                H.nZPhiSh += h["nsh"]
                dfdz = f(H.szphi.dfdz())
                phi0 = f(H.szphi.phi0())
            H.used[ib.phi] = 1
            r = f(H.sxy.radius())
            hcx, hcy = f(H.sxy.x0()), f(H.sxy.y0())
            hcx, hcy, r = self.do_weighted_circle_fit(H, seed, hcx, hcy, r)
        H.cx, H.cy = f(H.sxy.x0()), f(H.sxy.y0())
        H.radius = f(H.sxy.radius())
        H.circle_chisq = f(H.sxy.chi2_circle())

    def filter_using_pattern_recognition(self, H):
        if H.seed.panel < 0:
            return
        if H.dfdz == F0:
            return
        n_hel = 0
        xc, yc, radius, fz0, dfdz = H.cx, H.cy, H.radius, H.fz0, H.dfdz
        for fc in range(NTOTALFACES - 1, -1, -1):
            for p in range(KNPANELS):
                b, e = H.oT[fc]["pz"][p]
                for i in range(e - b):
                    idx = b + i
                    h = H.buf[idx]
                    if H.used[idx] != 1:
                        h["flag"] = h["flag"] | OUTLIER_MASK
                    else:
                        x, y = h["x"], h["y"]
                        z = H.zFace[fc]
                        phi_pred = fz0 + z * dfdz
                        x_pred = xc + radius * cosf(phi_pred)
                        y_pred = yc + radius * sinf(phi_pred)
                        weight = self.calc_weight(h, x_pred, y_pred) * self.weight3D
                        xr2 = (x - x_pred) * (x - x_pred)
                        yr2 = (y - y_pred) * (y - y_pred)
                        resid2 = (xr2 + yr2) * weight
                        if resid2 > self.chi2hel3DMax:
                            h["flag"] = h["flag"] | OUTLIER_MASK
                        else:
                            n_hel += h["nsh"]
        H.nStrawHits = n_hel


# ------------------------------------------------------------------------------------------------- module
def _input_hit(h, ci):
    return {
        "x": f(h["x"]), "y": f(h["y"]), "z": f(h["z"]), "time": f(h["time"]), "edep": f(h["edep"]),
        "uvar": f(h["uvar"]), "vvar": f(h["vvar"]), "udx": f(h["udx"]), "udy": f(h["udy"]),
        "sid": int(h["sid"]), "flag": int(h["flag"]) & 0xFFFFFFFF, "nsh": int(h["nsh"]),
        # _hphi / _xyWeight / _zphiWeight are written by the algorithm; the input value survives only on a hit
        # never written (ComboHit default 0)
        "hphi": f(h.get("hphi", 0.0)), "xyw": f(h.get("xyWeight", 0.0)), "zw": f(h.get("zphiWeight", 0.0)),
        "ci": ci, "pind0": int(h.get("pind0", ci)),
    }


def _init_helix_seed(alg, H, tc, bz0):
    pv = H.helix
    om, d0, ph0, td, z0 = pv["omega"], pv["d0"], pv["phi0"], pv["tanDip"], pv["z0"]
    R = D(1.) / abs(om)
    x0 = -(D(1) / om + d0) * D(math.sin(ph0))
    y0 = (D(1) / om + d0) * D(math.cos(ph0))
    dfdz = D(1.) / td / R
    sign = D(alg.dfdzsign)
    rcent = D(math.sqrt(x0 * x0 + y0 * y0))
    fcent = D(0.0) if (x0 == 0.0 and y0 == 0.0) else D(math.atan2(y0, x0))
    seed = {
        "rcent": f(rcent), "fcent": f(fcent), "radius": f(R), "lambda": f(D(1.) / dfdz * sign),
        "fz0": f(ph0 - PI / D(2.) * sign - z0 * om / td), "helicity": int(H.helicity),
        "chi2xy": f(H.sxy.chi2_circle()), "chi2zphi": f(H.szphi.chi2_line()),
    }
    mm2mev = (D(3) / D(10.)) * D(bz0)
    mom = R * mm2mev / D(math.cos(math.atan(td)))
    beta = abs(mom) / D(math.sqrt(mom * mom + E_MASS * E_MASS))
    pitch = PI / D(2.) - D(math.atan(td))
    hh = [H.buf[i] for i in H.goodhits]
    if tc["has_calo"]:
        _, tpos = alg._cluster_tracker_pos(tc)
        t0 = D(f(tc["time"])) - (tpos[2] - z0) / D(math.sin(pitch)) / (beta * C_LIGHT)
    else:
        if len(hh) == 0:
            t0 = D(0.)
        else:
            s = D(0.)
            for h in hh:
                s += D(h["time"]) - (D(h["z"]) - z0) / D(math.sin(pitch)) / (beta * C_LIGHT)
            t0 = s / D(len(hh))
    esum, nsh = F0, 0
    for h in hh:
        esum = esum + h["edep"] * f(h["nsh"])
        nsh += h["nsh"]
    seed["t0"] = t0
    seed["eDepAvg"] = f(D(esum) / (D(nsh) + D(1e-10)))
    seed["hits"] = [h["ci"] for h in hh]
    seed["hhits"] = [{"ci": h["ci"], "pind0": h["pind0"], "x": h["x"], "y": h["y"], "z": h["z"], "time": h["time"],
                      "edep": h["edep"], "nsh": h["nsh"], "sid": h["sid"], "flag": np.uint32(h["flag"]),
                      "hphi": h["hphi"], "xyWeight": h["xyw"], "zphiWeight": h["zw"]} for h in hh]
    return seed


def _pick_best_helix(vec):
    if len(vec) == 1:
        return 0
    h1, h2 = vec[0]["hhits"], vec[1]["hhits"]
    nh1, nh2 = len(h1), len(h2)
    s1 = {h["pind0"] for h in h1}                    # ComboHit::index() = _pind[0]
    natc = sum(1 for h in h2 if h["pind0"] in s1)
    thr = f(0.5)
    if f(natc) > f(nh1) * thr or f(natc) > f(nh2) * thr:
        if nh2 > nh1:
            return 1
        if nh1 > nh2:
            return 0
        return 0 if vec[0]["chi2zphi"] < vec[1]["chi2zphi"] else 1
    return 2


def cal_helix_finder(hits, tc, cfg, geom):
    """One time cluster through CalHelixFinder::produce's per-peak body: goodHitsTimeCluster, fillFaceOrderedHits,
    findHelix for each helicity in cfg['Helicities'] order, initHelixSeed, the eDepAvg cut, pickBestHelix. Returns
    the seeds the module pushes for this time cluster (0, 1 or 2), each a dict with the RobustHelix fields (rcent,
    fcent, radius, lambda, fz0, chi2xy, chi2zphi, helicity) as float32 / int, t0 (double), eDepAvg, `hits` (the
    collection indices of HelSeed._hhits in their order) and `hhits` (the ComboHit copies in HelSeed._hhits: hphi,
    xyWeight, zphiWeight float32, flag uint32, nsh int, plus position, time, edep, sid, pind0).

    hits: the ComboHit collection (list of dicts, collection order); tc: dict(hits, has_calo, disk, time, cog_x,
    cog_y); cfg: the module fhicl table (HelixFinderAlg sub-table for the algorithm); geom: dict(zFace, phiPanel, bz,
    disk_origin, tracker_center, and fCaloOffset or calo_g4info) -- see module_geometry / calo_offset."""
    with np.errstate(all="ignore"):
        alg = CalHelixFinderAlg(cfg["HelixFinderAlg"], geom)
        bz0 = D(geom["bz"])
        if not tc["has_calo"] and not alg.procAllTCs:
            return []
        ngood = 0                                    # goodHitsTimeCluster
        for i in tc["hits"]:
            if int(hits[i]["flag"]) & BKG_MASK:
                continue
            ngood += int(hits[i]["nsh"])
        if ngood < int(cfg["minNHitsTimeCluster"]):
            return []
        chits = {i: _input_hit(hits[i], i) for i in tc["hits"]}
        hlist = _HitView(chits)
        base = HFData()
        base.clear_temp_variables()
        base.zFace = [f(z) for z in geom["zFace"]]
        base.phiPanel = [f(p) for p in geom["phiPanel"]]
        base.has_calo = bool(tc["has_calo"])
        alg.fill_face_ordered_hits(base, hlist, tc)
        max_edep = f(cfg["maxEDepAvg"])
        seeds = []
        for hv in cfg["Helicities"]:
            H = base.copy()
            H.clear_helix_info()
            H.helicity = (hv > 0) - (hv < 0)
            if not alg.find_helix(H):
                continue
            if H.helix is None:
                continue
            s = _init_helix_seed(alg, H, tc, bz0)
            if s["eDepAvg"] > max_edep:
                continue
            seeds.append(s)
        if not seeds:
            return []
        best = _pick_best_helix(seeds)
        if 0 <= best < 2:
            return [seeds[best]]
        return [seeds[k] for k in range(len(cfg["Helicities"]))]


class _HitView:
    """Collection access by index for the hits a time cluster references."""

    def __init__(self, d):
        self._d = d

    def __getitem__(self, i):
        return self._d[i]
