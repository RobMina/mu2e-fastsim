"""Counter geometry and lookup-table bin definitions of the CRV photon lookup-table generator.

Stock values follow Mu2e/CRVStandalone wls/src/WLSDetectorConstruction.cc (the default, non-TESTBEAM, non-LDMX build):
a 19.78 x 51.34 mm extrusion with a 0.25 mm TiO2 coating and 2.0 mm outer corner radius, two elliptical fiber holes
(semi-axes 2.0 mm along the thickness, 1.0 mm along the width) 26 mm apart, 1.4 mm fibers with two cladding layers of
3 % of the diameter each, 10 mm fiber guide bars at both ends, and 2 x 2 mm SiPMs (or aluminium reflectors) 1 um
from the fiber ends. Coordinates: x thickness, y width, z length, origin at the counter centre; SiPM 0 reads the
fiber at y = -separation/2 at the -z end.

``Counter`` makes the compile-time dimensions of the stock program runtime parameters. The stock cross-section keeps
the stock bin edges exactly; any other cross-section gets edges from ``general_xy_edges`` (the stock pattern moved to
the new dimensions).
"""
import math
from dataclasses import dataclass, replace

import numpy as np

STOCK_THICKNESS = 19.78
STOCK_WIDTH = 51.34
STOCK_SEPARATION = 26.0
STOCK_FIBER_DIAMETER = 1.4

# stock edges (mm): x uses only x >= 0 (symmetry), y covers the full width
_STOCK_X = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 9.0, 9.5, 9.89)
_STOCK_Y = (-25.67, -25.2, -24.7, -23.5, -20.5, -17.5, -15.5, -15.0, -14.5, -14.0, -13.5, -13.0, -12.5, -12.0, -11.5,
            -11.0, -10.5, -8.5, -6.5, -4.5, -1.5, 1.5, 4.5, 6.5, 8.5, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
            14.5, 15.0, 15.5, 17.5, 20.5, 23.5, 24.7, 25.2, 25.67)
_INV_BETA = (2.20, 1.80, 1.64, 1.58, 1.24, 1.0)

SIM_TYPES = {0: "scintillation in scintillator", 1: "Cerenkov in scintillator", 2: "Cerenkov in fiber"}


@dataclass(frozen=True)
class Counter:
    """One counter: dimensions in mm. ``reflector``: 0 SiPMs at both ends, 1 reflector at +z, 2 black tape at +z."""
    length: float
    reflector: int = 0
    thickness: float = STOCK_THICKNESS
    width: float = STOCK_WIDTH
    fiber_separation: float = STOCK_SEPARATION
    fiber_diameter: float = STOCK_FIBER_DIAMETER
    clad_fraction: float = 0.03          # each cladding layer, as a fraction of the fiber diameter
    hole_rx: float = 2.0                 # fiber hole semi-axis along x (thickness)
    hole_ry: float = 1.0                 # fiber hole semi-axis along y (width)
    coating: float = 0.25
    corner_radius: float = 2.0           # outer (extrusion) corner radius
    guide_bar: float = 10.0
    # end readout of the "gap" optical model (current CRVStandalone); the "v6" model puts a SiPM / reflector disc of
    # the fiber radius directly on the fiber end
    sipm_width: float = 2.0
    sipm_window: float = 0.1             # G4Box half-length of the epoxy window (as in the stock code)
    sipm_length: float = 0.2
    air_gap: float = 0.001

    def __post_init__(self):
        if self.reflector not in (0, 1, 2):
            raise ValueError("reflector must be 0 (none), 1 (reflector at +z) or 2 (black tape at +z)")
        if self.r_clad2 >= self.hole_ry or self.r_clad2 >= self.hole_rx:
            raise ValueError("the fiber must fit inside its hole")
        if self.fiber_separation / 2 + self.hole_ry >= self.hy or self.hole_rx >= self.hx:
            raise ValueError("the fiber holes must lie inside the scintillator")
        if self.length <= 600.0:
            raise ValueError("the stock z binning needs a counter longer than 600 mm")

    # derived dimensions (stock expressions, same floating-point order)
    @property
    def hx(self):
        return self.thickness / 2.0 - self.coating

    @property
    def hy(self):
        return self.width / 2.0 - self.coating

    @property
    def half_length(self):
        return self.length / 2.0

    @property
    def rc(self):
        return self.corner_radius - self.coating

    @property
    def r_clad2(self):
        return self.fiber_diameter / 2.0

    @property
    def r_clad1(self):
        return self.fiber_diameter / 2.0 - self.clad_fraction * self.fiber_diameter

    @property
    def r_core(self):
        c = self.clad_fraction * self.fiber_diameter
        return self.fiber_diameter / 2.0 - c - c

    @property
    def is_stock_section(self):
        return (self.thickness, self.width, self.fiber_separation, self.fiber_diameter, self.hole_rx, self.hole_ry) == \
            (STOCK_THICKNESS, STOCK_WIDTH, STOCK_SEPARATION, STOCK_FIBER_DIAMETER, 2.0, 1.0)

    @property
    def name(self):
        return f"{int(round(self.length))}_{self.reflector}"


def stock(length, reflector=0):
    return Counter(length=float(length), reflector=int(reflector))


def design(thickness, width, length, reflector=0, fiber_diameter=2.0, hole_clearance=(1.3, 0.3)):
    """A design counter: two fibers at the stock separation ratio (26 / 51.34 of the width), holes keeping the stock
    clearance around the fiber (semi-axes r + 1.3 along x, r + 0.3 along y)."""
    r = fiber_diameter / 2.0
    return Counter(length=float(length), reflector=int(reflector), thickness=thickness, width=width,
                   fiber_separation=width * STOCK_SEPARATION / STOCK_WIDTH, fiber_diameter=fiber_diameter,
                   hole_rx=r + hole_clearance[0], hole_ry=r + hole_clearance[1])


def z_edges(c):
    """WLSDetectorConstruction z bins: 10 mm and 25 mm bins over 300 mm at each end, ~100 mm bins in between."""
    hl = c.length / 2.0
    n = int(np.rint((c.length - 600.0) / 100.0))     # lrint: round half to even
    bw = (c.length - 600.0) / n if n > 0 else 0.0
    z = [-hl + 10.0 * i for i in range(6)]
    z += [-hl + 50.0 + 25.0 * i for i in range(1, 11)]
    z += [-hl + 300.0 + bw * i for i in range(1, n + 1)]
    z += [hl - 300.0 + 25.0 * i for i in range(1, 11)]
    z += [hl - 50.0 + 10.0 * i for i in range(1, 6)]
    return np.array(z, np.float64)


def general_xy_edges(c):
    """x and y edges for a non-stock cross-section: the stock pattern moved to the new dimensions.

    x (>= 0): 0 to 3 mm in 0.5 mm steps (the hole region), equal steps of at most 3 mm up to T/2 - 0.89, then
    T/2 - 0.39 and T/2. y (both halves, mirrored): 0.5 mm steps within 2.5 mm of each fiber centre; a central bin
    [-1.5, 1.5]; 2 mm steps from the fiber region towards the centre while they stay 2.5 mm above 1.5; equal steps of
    at most 3 mm between the fiber region and W/2 - 2.17; then W/2 - {0.97, 0.47, 0}.
    """
    h, w2, yc = c.thickness / 2.0, c.width / 2.0, c.fiber_separation / 2.0
    x = [0.5 * i for i in range(7)]
    lo, hi = 3.0, h - 0.89
    if hi <= lo:
        raise ValueError("counter too thin for the x binning rule")
    n = max(1, math.ceil((hi - lo) / 3.0 - 1e-9))
    x += [lo + (hi - lo) * i / n for i in range(1, n + 1)] + [h - 0.39, h]
    f_lo, f_hi = yc - 2.5, yc + 2.5
    e_lo = w2 - 2.17
    if f_lo <= 1.5 + 0.5 or f_hi >= e_lo:
        raise ValueError("fiber region overlaps the centre or edge bins")
    pos = [1.5]
    v = f_lo - 2.0
    inner = []
    while v - 1.5 >= 2.5:
        inner.append(v)
        v -= 2.0
    pos += sorted(inner)
    pos += [f_lo + 0.5 * i for i in range(11)]
    n = max(1, math.ceil((e_lo - f_hi) / 3.0 - 1e-9))
    pos += [f_hi + (e_lo - f_hi) * i / n for i in range(1, n + 1)]
    pos += [w2 - 0.97, w2 - 0.47, w2]
    y = [-p for p in reversed(pos)] + pos
    return np.array(x, np.float64), np.array(y, np.float64)


def edges(c):
    """All bin edges of the table: dict x, y, z, beta, theta, phi, r (r: core radius / 5 steps, as stock)."""
    if c.is_stock_section:
        x, y = np.array(_STOCK_X), np.array(_STOCK_Y)
    else:
        x, y = general_xy_edges(c)
    beta = np.array([1.0 / b for b in _INV_BETA])
    theta = np.array([0.0] + [math.pi / 20.0 + math.pi * i / 10.0 for i in range(10)] + [math.pi])
    phi = np.array([0.0] + [math.pi / 8.0 + math.pi * i / 4.0 for i in range(4)] + [math.pi])
    r = np.array([c.r_core * i / 5.0 for i in range(6)])
    return {"x": x, "y": y, "z": z_edges(c), "beta": beta, "theta": theta, "phi": phi, "r": r}


def n_bins(c, simtype, e=None):
    e = e or edges(c)
    n = {k: len(v) - 1 for k, v in e.items()}
    if simtype == 0:
        return n["x"] * n["y"] * n["z"]
    if simtype == 1:
        return n["x"] * n["y"] * n["z"] * n["beta"]
    if simtype == 2:
        return n["z"] * n["r"] * n["phi"] * n["theta"] * n["beta"]
    raise ValueError(simtype)


def bin_boxes(c, simtype, bins, e=None):
    """Per bin: the sampling ranges WLSPrimaryGeneratorAction::SetNextBins uses (lo, hi arrays per variable)."""
    e = e or edges(c)
    b = np.asarray(bins, np.int64)
    n = {k: len(v) - 1 for k, v in e.items()}
    out = {}
    if simtype in (0, 1):
        if simtype == 0:
            xb, yb, zb = (b // (n["z"] * n["y"])) % n["x"], (b // n["z"]) % n["y"], b % n["z"]
        else:
            nb = n["beta"]
            xb = (b // (nb * n["z"] * n["y"])) % n["x"]
            yb = (b // (nb * n["z"])) % n["y"]
            zb = (b // nb) % n["z"]
            bb = b % nb
            out["beta"] = (e["beta"][bb], e["beta"][bb + 1])
        for k, i in (("x", xb), ("y", yb), ("z", zb)):
            out[k] = (e[k][i], e[k][i + 1])
    else:
        nz, nr, nph, nth = n["z"], n["r"], n["phi"], n["theta"]
        out["beta"] = tuple(e["beta"][(b // (nz * nr * nph * nth)) % n["beta"] + d] for d in (0, 1))
        out["theta"] = tuple(e["theta"][(b // (nz * nr * nph)) % nth + d] for d in (0, 1))
        out["phi"] = tuple(e["phi"][(b // (nz * nr)) % nph + d] for d in (0, 1))
        out["r"] = tuple(e["r"][(b // nz) % nr + d] for d in (0, 1))
        out["z"] = tuple(e["z"][b % nz + d] for d in (0, 1))
    return out


def with_length(c, length, reflector):
    return replace(c, length=float(length), reflector=int(reflector))
