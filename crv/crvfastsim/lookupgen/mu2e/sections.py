"""Mu2e adapter: the CRV counters and the bin definitions of the Mu2e CRV lookup tables.

Stock values follow CRVStandalone wls/src/WLSDetectorConstruction.cc (the default, non-TESTBEAM, non-LDMX build): a
19.78 x 51.34 mm extrusion with a 0.25 mm TiO2 coating and 2.0 mm outer corner radius, two elliptical fiber holes
(semi-axes 2.0 mm along the thickness, 1.0 mm along the width) 26 mm apart, 1.4 mm fibers with two cladding layers of
3 % of the diameter each, 10 mm fiber guide bars at both ends, and 2 x 2 mm SiPMs (or aluminium reflectors) 1 um
from the fiber ends. SiPM 0 reads the fiber at y = -separation/2 at the -z end.

The bins follow the stock table: x (x >= 0), y, z edges per cross-section (the stock cross-section keeps the stock
edges exactly; any other gets ``general_xy_edges``, the stock pattern moved to the new dimensions), z edges from the
counter length (``z_edges``), and fixed beta, theta, phi and r edges; bin numbers are row-major in ORDER.
"""
import math

import numpy as np

from ..core.binning import SIM_TYPES, Binning  # noqa: F401  (SIM_TYPES re-exported)
from ..core.counter import Counter, with_length  # noqa: F401

STOCK_THICKNESS = 19.78
STOCK_WIDTH = 51.34
STOCK_SEPARATION = 26.0
STOCK_FIBER_DIAMETER = 1.4
# the rest of the stock counter (CRVStandalone)
STOCK = dict(clad_fraction=0.03, hole_rx=2.0, hole_ry=1.0, coating=0.25, corner_radius=2.0, guide_bar=10.0,
             sipm_width=2.0, sipm_window=0.1, sipm_length=0.2, air_gap=0.001)

# stock edges (mm): x uses only x >= 0 (symmetry), y covers the full width
_STOCK_X = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 9.0, 9.5, 9.89)
_STOCK_Y = (-25.67, -25.2, -24.7, -23.5, -20.5, -17.5, -15.5, -15.0, -14.5, -14.0, -13.5, -13.0, -12.5, -12.0, -11.5,
            -11.0, -10.5, -8.5, -6.5, -4.5, -1.5, 1.5, 4.5, 6.5, 8.5, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
            14.5, 15.0, 15.5, 17.5, 20.5, 23.5, 24.7, 25.2, 25.67)
_INV_BETA = (2.20, 1.80, 1.64, 1.58, 1.24, 1.0)

# bin numbering of the stock table (slowest variable first)
ORDER = {0: ("x", "y", "z"), 1: ("x", "y", "z", "beta"), 2: ("beta", "theta", "phi", "r", "z")}


def stock(length, reflector=0):
    return Counter(length=float(length), reflector=int(reflector), thickness=STOCK_THICKNESS, width=STOCK_WIDTH,
                   fiber_separation=STOCK_SEPARATION, fiber_diameter=STOCK_FIBER_DIAMETER, **STOCK)


def design(thickness, width, length, reflector=0, fiber_diameter=2.0, hole_clearance=(1.3, 0.3)):
    """A design counter: two fibers at the stock separation ratio (26 / 51.34 of the width), holes keeping the stock
    clearance around the fiber (semi-axes r + 1.3 along x, r + 0.3 along y)."""
    r = fiber_diameter / 2.0
    kw = dict(STOCK, hole_rx=r + hole_clearance[0], hole_ry=r + hole_clearance[1])
    return Counter(length=float(length), reflector=int(reflector), thickness=thickness, width=width,
                   fiber_separation=width * STOCK_SEPARATION / STOCK_WIDTH, fiber_diameter=fiber_diameter, **kw)


def is_stock(c):
    """True for the stock cross-section (any length and end)."""
    return (c.thickness, c.width, c.fiber_separation, c.fiber_diameter, c.hole_rx, c.hole_ry) == \
        (STOCK_THICKNESS, STOCK_WIDTH, STOCK_SEPARATION, STOCK_FIBER_DIAMETER, 2.0, 1.0)


def name(c):
    """The table name: <length>_<reflector> (LookupTable_<name>)."""
    return f"{int(round(c.length))}_{c.reflector}"


def section_tag(c):
    """Cross-section tag, e.g. t24p71_w25p00_f2p0_s12p66 (thickness, width, fiber diameter, fiber separation)."""
    def f(v, d):
        return f"{v:.{d}f}".replace(".", "p")
    return f"t{f(c.thickness, 2)}_w{f(c.width, 2)}_f{f(c.fiber_diameter, 1)}_s{f(c.fiber_separation, 2)}"


def z_edges(c):
    """WLSDetectorConstruction z bins: 10 mm and 25 mm bins over 300 mm at each end, ~100 mm bins in between."""
    if c.length <= 600.0:
        raise ValueError("the Mu2e z binning needs a counter longer than 600 mm")
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
    if is_stock(c):
        x, y = np.array(_STOCK_X), np.array(_STOCK_Y)
    else:
        x, y = general_xy_edges(c)
    beta = np.array([1.0 / b for b in _INV_BETA])
    theta = np.array([0.0] + [math.pi / 20.0 + math.pi * i / 10.0 for i in range(10)] + [math.pi])
    phi = np.array([0.0] + [math.pi / 8.0 + math.pi * i / 4.0 for i in range(4)] + [math.pi])
    r = np.array([c.r_core * i / 5.0 for i in range(6)])
    return {"x": x, "y": y, "z": z_edges(c), "beta": beta, "theta": theta, "phi": phi, "r": r}


def binning(c):
    """The Binning of counter c's table."""
    return Binning(edges(c), ORDER)


def n_bins(c, simtype, e=None):
    return Binning(e or edges(c), ORDER).n_bins(simtype)


def bin_boxes(c, simtype, bins, e=None):
    """Per bin: the sampling ranges WLSPrimaryGeneratorAction::SetNextBins uses (lo, hi arrays per variable)."""
    return Binning(e or edges(c), ORDER).boxes(simtype, bins)
