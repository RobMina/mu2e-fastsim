"""Parametric CRV module designs on the sector layout of a stock counters file.

A Design sets the module stack: number of layers, counter thickness, width and maximum length, the passive absorber
between layers, the module tilt and the WLS fiber diameter. It is applied to every sector of a base counters file
(Run 2: crv_counters_v10) through that sector's frame: layer direction n, gap direction g, length axis l, the
first-counter position and the stock extent along g and l.

- Counters follow the stock layout formula (CosmicRayShieldMaker): within a module they alternate the small and large
  gaps, modules are separated by the module gap, layer j sits at j (thickness + absorber) along n plus j times the
  layer offset along the offset direction. The default Design reproduces the stock bars exactly.
- Modules are re-tiled to cover the stock sector extent along g; a sector longer than ``max_length`` is split along l
  into equal segments, each a counter with its own SiPMs.
- A tilted module is rotated by ``tilt_deg`` about l through its centre (n' = cos n + sin g, g' = cos g - sin n), so
  the tilt is the angle between the module normal and the stock normal (vertical for the top sectors). Tilted modules
  are louvers: the pitch along g is the projected module width plus the module gap, so a line along n always meets a
  module; at small tilts the pitch grows until neighbouring modules do not overlap, which leaves a hole.
Every bar of a sector shares the sector axes, so in sector-local coordinates (rows of ``axes``: thickness, width,
length) every bar and absorber is an axis-aligned box: crossings use the core slab method there.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .. import geometry as G

RUN2_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_v10.txt"
ABSORBER_MATERIALS = ("G4_Al", "MildSteel", "G4_Pb", "G4_POLYETHYLENE")


@dataclass(frozen=True)
class Design:
    """One CRV design point. None = the stock value of the base counters file (per sector where stock varies)."""
    n_layers: int | None = None
    thickness: float | None = None        # counter thickness [mm]
    width: float | None = None            # counter width [mm]
    max_length: float = 0.0               # longest counter [mm]; 0 = stock lengths, else longer sectors are split
    segment_gap: float = 0.0              # gap between length segments [mm]
    absorber: float | None = None         # absorber thickness in every gap between layers [mm]
    absorber_material: str | None = None  # core materials name; None = stock
    tilt_deg: float = 0.0
    fiber_diameter: float | None = None   # mm; None = the stock fiber of each sector
    counters_per_module: int | None = None
    coincidence_layers: int | None = None  # k of the k-of-N layer rule; None = stock (N minus the stock spare layers)
    spare_layers: int | None = None        # or k = N - spare_layers in every sector (exclusive with coincidence_layers)
    pe_threshold: float | None = None      # PE per layer-end for a hit; None = the stock threshold of each sector

    def __post_init__(self):
        if self.n_layers is not None and not 1 <= self.n_layers <= 12:
            raise ValueError("n_layers must be 1..12")
        for k in ("thickness", "width", "absorber", "fiber_diameter"):
            v = getattr(self, k)
            if v is not None and not (v >= 0 and math.isfinite(v)):
                raise ValueError(f"{k} must be finite and >= 0")
        if self.thickness is not None and self.thickness <= 1.0 or self.width is not None and self.width <= 1.0:
            raise ValueError("thickness and width must exceed 1 mm")
        if not 0.0 <= self.tilt_deg < 80.0:
            raise ValueError("tilt_deg must be in [0, 80)")
        if self.max_length < 0 or self.segment_gap < 0:
            raise ValueError("max_length and segment_gap must be >= 0")
        if self.coincidence_layers is not None and not 1 <= self.coincidence_layers <= 12:
            raise ValueError("coincidence_layers must be 1..12")
        if None not in (self.n_layers, self.coincidence_layers) and self.coincidence_layers > self.n_layers:
            raise ValueError("coincidence_layers must not exceed n_layers")
        if self.spare_layers is not None and not 0 <= self.spare_layers <= 11:
            raise ValueError("spare_layers must be 0..11")
        if None not in (self.spare_layers, self.coincidence_layers):
            raise ValueError("give coincidence_layers or spare_layers, not both")
        if None not in (self.n_layers, self.spare_layers) and self.spare_layers >= self.n_layers:
            raise ValueError("spare_layers must be below n_layers")
        if self.pe_threshold is not None and not (self.pe_threshold > 0 and math.isfinite(self.pe_threshold)):
            raise ValueError("pe_threshold must be finite and > 0")
        if self.absorber_material is not None and self.absorber_material not in ABSORBER_MATERIALS:
            raise ValueError(f"absorber_material must be one of {ABSORBER_MATERIALS}")

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise KeyError(f"unknown design keys {bad}; known: {sorted(names)}")
        return cls(**d)

    def as_dict(self):
        return dataclasses.asdict(self)


@dataclass
class DesignGeometry:
    sector_names: list
    axes: np.ndarray          # [nsec, 3, 3] rows: thickness, width, length unit vectors (Mu2e)
    sector: np.ndarray        # [nbars]
    module: np.ndarray
    layer: np.ndarray
    counter: np.ndarray       # counter index along g within the sector's layer (module-major)
    segment: np.ndarray
    center: np.ndarray        # [nbars, 3] Mu2e, mm
    halflen: np.ndarray       # [nbars, 3] local (thickness, width, length) half lengths
    abs_sector: np.ndarray    # absorber slabs: sector, centre (Mu2e), local half lengths
    abs_center: np.ndarray
    abs_halflen: np.ndarray
    absorber_material: str
    n_layers: np.ndarray      # [nsec]
    fiber_diameter: np.ndarray   # [nsec] mm
    stock_fiber: np.ndarray      # [nsec] mm
    module_pitch: np.ndarray     # [nsec] along g, mm
    design: Design = field(default_factory=Design)

    @property
    def nbars(self):
        return len(self.sector)

    def local(self, x, s):
        """Mu2e points or vectors [..., 3] in sector s's local frame."""
        return np.asarray(x) @ self.axes[s].T


def stock_fiber_mm(yield_per_mev):
    """Stock fiber diameter of a sector from its scintillation yield (CRV-T 1.8 mm: 39794 /MeV; else 1.4 mm)."""
    return 1.8 if yield_per_mev >= 39000 else 1.4


def _rot(n, g, theta):
    c, s = math.cos(theta), math.sin(theta)
    return c * n + s * g, c * g - s * n


def _overlap(lo_a, hi_a, lo_b, hi_b, tol=1e-6):
    """Any box of a overlapping any box of b (axis-aligned, same frame)."""
    lo = np.maximum(lo_a[:, None, :], lo_b[None, :, :])
    hi = np.minimum(hi_a[:, None, :], hi_b[None, :, :])
    return bool(np.any(np.all(hi - lo > tol, axis=2)))


def build(design=Design(), counters=RUN2_COUNTERS, sector_yields=None):
    """DesignGeometry of ``design`` on the sectors of ``counters`` (an Offline-relative or absolute path, or a counters
    text, e.g. layout.counters_text). sector_yields: {sector: scintillation yield /MeV} (sets each sector's stock
    fiber); default 1.4 mm everywhere."""
    c = G.parse_simpleconfig(counters if "crs.sectorNames" in counters else open(G.find_counters(counters)).read())
    names = list(c["crs.sectorNames"])
    t_s, w_s = float(c["crs.scintillatorBarThickness"]), float(c["crs.scintillatorBarWidth"])
    g_small, g_large, g_mod = float(c["crs.gapSmall"]), float(c["crs.gapLarge"]), float(c["crs.gapBetweenModules"])
    n_glob = int(c["crs.nLayers"])
    gaps_stock = list(np.atleast_1d(c["crs.gapBetweenLayers"]).astype(float))
    off = float(c["crs.layerOffset"])
    mat = design.absorber_material or c.get("crs.absorberMaterialName", "G4_Al")
    t = design.thickness or t_s
    w = design.width or w_s
    theta = math.radians(design.tilt_deg)
    rows = {k: [] for k in ("sector", "module", "layer", "counter", "segment", "center", "halflen")}
    arows = {k: [] for k in ("sector", "center", "halflen")}
    axes, nl_out, fib, fib0, pitch_out = [], [], [], [], []
    for isec, name in enumerate(names):
        ldir = np.asarray(c[f"crs.layerDirection{name}"], float)
        gdir = np.asarray(c[f"crs.gapDirection{name}"], float)
        odir = np.asarray(c[f"crs.offsetDirection{name}"], float)
        ldim = [d for d in range(3) if ldir[d] == 0 and gdir[d] == 0][0]
        ell = np.zeros(3)
        ell[ldim] = 1.0
        counters_only = bool(c.get(f"crs.countersOnly{name}", False))
        n_stock = int(c.get(f"crs.nLayers{name}", n_glob))
        nl = design.n_layers or n_stock
        a = design.absorber if design.absorber is not None else (gaps_stock[0] if gaps_stock else 0.0)
        gaps = [0.0 if counters_only else (gaps_stock[j] if design.absorber is None and j < len(gaps_stock) else a)
                for j in range(nl - 1)]
        length = float(c[f"crs.scintillatorBarLength{name}"])
        nmod_s, ncnt_s = int(c[f"crs.nModules{name}"]), int(c[f"crs.nCountersPerModule{name}"])
        ncnt = design.counters_per_module or ncnt_s
        first = np.asarray(c[f"crs.firstCounter{name}"], float)
        y0 = (sector_yields or {}).get(name)
        f0 = stock_fiber_mm(y0) if y0 is not None else 1.4
        fib0.append(f0)
        fib.append(design.fiber_diameter or f0)
        # stock module pitch along g and the extent it covers (CosmicRayShieldMaker)
        pitch_stock = ncnt_s * w_s + (ncnt_s // 2 - 1) * g_large + (ncnt_s // 2) * g_small + g_mod
        extent = nmod_s * pitch_stock
        # counters of one module (module-local, before tilt): positions along g, layer offsets along n / off
        ic = np.arange(ncnt)
        cg = (ic // 2) * (w + g_large) + ((ic + 1) // 2) * (w + g_small)
        lay = np.zeros((nl, 3))
        for j in range(1, nl):
            lay[j] = lay[j - 1] + t * ldir + (0.0 if counters_only else gaps[j - 1]) * ldir + off * odir
        nseg = max(1, math.ceil(length / design.max_length - 1e-9)) if design.max_length > 0 else 1
        lseg = (length - (nseg - 1) * design.segment_gap) / nseg
        zseg = -length / 2 + lseg / 2 + np.arange(nseg) * (lseg + design.segment_gap)
        pts = (first[None, None, :] + lay[:, None, :] + cg[None, :, None] * gdir[None, None, :])   # [nl, ncnt, 3]
        mod_center = pts.reshape(-1, 3).mean(0)                 # tilt pivot: the centre of module 0's counters
        n_t, g_t = _rot(ldir, gdir, theta)
        A = np.stack([n_t, g_t, ell])
        axes.append(A)
        M = np.stack([ldir, gdir, ell])                          # untilted frame (orthonormal rows)
        half = np.array([t / 2, w / 2, lseg / 2])
        rel_t = (pts - mod_center).reshape(-1, 3) @ M.T          # (n, g, l) components, layer-major
        rel_rot = rel_t[:, 0:1] * n_t + rel_t[:, 1:2] * g_t + rel_t[:, 2:3] * ell
        # absorber slabs: between layers j and j+1, the extent of the layer's counters
        abs_rel = []
        if not counters_only:
            span = cg[-1] + w
            for j in range(nl - 1):
                if gaps[j] <= 0:
                    continue
                cj = first + lay[j] + (t / 2 + gaps[j] / 2) * ldir + 0.5 * (span - w) * gdir - mod_center
                cc = cj @ M.T
                abs_rel.append((cc[0] * n_t + cc[1] * g_t + cc[2] * ell, np.array([gaps[j] / 2, span / 2, length / 2])))
        # pitch: the projected module width plus the module gap (stock at no tilt, louvers that tile the stock plane
        # when tilted); grown until neighbouring modules do not overlap (small tilts leave a hole of that growth)
        pitch = (ncnt * w + (ncnt // 2 - 1) * g_large + (ncnt // 2) * g_small) * math.cos(theta) + g_mod
        loc0 = rel_rot @ A.T
        lo0, hi0 = loc0 - half, loc0 + half
        for _ in range(2000):
            sh = (pitch * gdir) @ A.T
            if not _overlap(lo0, hi0, lo0 + sh, hi0 + sh, tol=1e-6):
                break
            pitch *= 1.005
        else:
            raise ValueError(f"{name}: no non-overlapping module pitch found")
        pitch_out.append(pitch)
        nmod = max(1, math.ceil(extent / pitch - 1e-9))
        nl_out.append(nl)
        for im in range(nmod):
            mc = mod_center + im * pitch * gdir
            for k, z in enumerate(zseg):
                cen = mc + rel_rot + z * ell
                rows["center"].append(cen)
                rows["halflen"].append(np.broadcast_to(half, cen.shape))
                rows["sector"].append(np.full(len(cen), isec))
                rows["module"].append(np.full(len(cen), im))
                rows["layer"].append(np.repeat(np.arange(nl), ncnt))
                rows["counter"].append(np.tile(ic + im * ncnt, nl))
                rows["segment"].append(np.full(len(cen), k))
            for ac, ah in abs_rel:
                arows["sector"].append(isec)
                arows["center"].append(mc + ac)
                arows["halflen"].append(ah)
    cat = {k: np.concatenate(v) for k, v in rows.items()}
    na = len(arows["sector"])
    return DesignGeometry(
        names, np.array(axes), cat["sector"].astype(np.int64), cat["module"].astype(np.int64),
        cat["layer"].astype(np.int64), cat["counter"].astype(np.int64), cat["segment"].astype(np.int64),
        cat["center"].astype(float), cat["halflen"].astype(float), np.asarray(arows["sector"], np.int64),
        np.asarray(arows["center"], float).reshape(na, 3), np.asarray(arows["halflen"], float).reshape(na, 3), mat,
        np.asarray(nl_out, np.int64), np.asarray(fib, float), np.asarray(fib0, float), np.asarray(pitch_out, float),
        design)


def to_crv_geometry(dg):
    """The CRVGeometry of an untilted design (axis-aligned bars), for comparison with geometry.from_offline_config."""
    if dg.design.tilt_deg != 0.0:
        raise ValueError("tilted designs have no axis-aligned CRVGeometry")
    a = np.abs(np.rint(dg.axes)).astype(int)
    l2w = np.argmax(a, axis=2)[dg.sector]                      # Mu2e axis of (thickness, width, length)
    hl = np.zeros_like(dg.halflen)
    np.put_along_axis(hl, l2w, dg.halflen, axis=1)
    return G.CRVGeometry(dg.sector_names, dg.sector, dg.module, dg.layer, dg.counter, dg.center, hl, l2w)


def readout_sides(reflector):
    """Read-out ends of a counter: reflector 0 both, 1 the -l end (SiPMs 0, 2), -1 the +l end (stock SiPMs 1, 3)."""
    return {0: (0, 1), 1: (0,), -1: (1,)}[int(reflector)]


def cost_terms(dg, reflectors, density):
    """Cost-proxy terms: scintillator volume [m^3], channels, absorber mass [t] (density in g/cm^3), fiber length x
    diameter^2 [m mm^2]."""
    vol = float(np.prod(2 * dg.halflen, axis=1).sum()) * 1e-9
    nside = np.array([len(readout_sides(reflectors[n])) for n in dg.sector_names])
    channels = int((2 * nside[dg.sector]).sum())
    absmass = float(np.prod(2 * dg.abs_halflen, axis=1).sum()) * 1e-9 * density if len(dg.abs_halflen) else 0.0
    fiber = float((2 * 2 * dg.halflen[:, 2] * 1e-3 * dg.fiber_diameter[dg.sector] ** 2).sum())
    return dict(scint_m3=vol, channels=channels, absorber_t=absmass, fiber_m_mm2=fiber)
