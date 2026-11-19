"""Cosmic pre-filter: decide, before Geant4, which CRY primaries can reach a detector (g4fastsim plan step P6).

The stock extracted cosmic chain spends most of its Geant4 time on cosmics that never reach a detector: the
`PrimaryFilter` (Offline `DetectorStepFilter`: trk OR calo OR CRV steps) rejects about 95 % of the generated events
after they have been tracked through the hall and the overburden. This module predicts, from the CRY GenParticles
alone, whether an event is worth simulating, with straight-line transport and a mean-range check through the GDML
world. Everything is a parameter (decision D6): the loss of accepted events and the Geant4 time saved are measured as
functions of them (`cosmic_prefilter_scan`), and the operating point is chosen per campaign.

Decision for one primary (an event is simulated when any of its primaries is kept):

* **Envelopes.** Each detector term (`trk`, `calo`, `crv`, the three terms of `PrimaryFilter`) is a list of
  envelopes read from the job's own GDML by placed-volume name (`envelopes_from_gdml`; `EXTRACTED_SELECTION` for the
  extracted geometry: `TrackerMother`, `CaloDisk_<n>`, `CRSmother_CRV_<sector>`). A tube or polycone placed along a
  principal axis becomes a solid cylinder (rmin ignored); anything else becomes the global axis-aligned box of its
  whole subtree.
* **Geometry (margin).** For every primary and envelope, `required_margin` gives the smallest margin m (mm) for which
  the forward straight line meets the envelope grown by m (a box grown by m on every face, a cylinder by m in radius
  and half-length); m < 0 means the line passes at least |m| inside. It is exact for boxes (pairwise slab bounds) and
  a convex one-dimensional minimisation for cylinders, so a margin scan is a threshold on one number per term.
* **Range (muons).** For a muon whose line meets a term's envelope grown by the margin, the material column from the
  generator point to the entry point (`world.PieceSet.column`, plus air for the rest of the path) is converted to a
  concrete-equivalent length (each material scaled by its mean dE/dx relative to CONCRETE_MARS at `P_REF`), and the
  kinetic energy left at the entry is read from a CSDA range table of the full mean dE/dx (`eloss.dedx_restricted`
  without cut). The muon passes when that energy is at least `min_muon_ke_mev` (None: no range check). No energy-loss
  fluctuations, multiple scattering, radiative losses or decays: the margin and the energy threshold are the knobs that
  absorb them, and the scan measures what they miss.
* **Non-muons** (gamma, e+-, n, p, pi+-): `nonmuon="always"` keeps every event with a non-muon primary;
  `"threshold"` keeps a non-muon whose kinetic energy is at least `nonmuon_min_ke_mev` and, when `nonmuon_margin_mm`
  is set, whose line also meets an envelope grown by that margin.

Precision: `dtype=np.float32` or `np.float64` sets the floating type of the geometry and the range arithmetic (the
range table itself is built in float64 and read with linear interpolation). Inputs are cast to it.
"""
from dataclasses import asdict, dataclass, field
import re
import xml.etree.ElementTree as ET

import numpy as np

from . import eloss, materials
from . import world as world_mod

TERMS = ("trk", "calo", "crv")
EXTRACTED_SELECTION = {"trk": (r"TrackerMother",), "calo": (r"CaloDisk_\d+",), "crv": (r"CRSmother_CRV_\w+",)}
SUBSEGMENT_MM = 2000.0           # longest piece of a path handed to world.PieceSet.column (see Prefilter.column_eq)
T_MAX = 1.0e6                    # mm: longest straight path considered (the CRY world box is ~2.4e5 mm wide)
P_REF = 500.0                    # MeV/c: momentum at which each material's dE/dx is scaled to concrete
REF_MATERIAL = materials.CONCRETE_MARS
TK_TABLE_MIN = 1.0               # MeV: range table start (range linear to zero below, Bethe is not valid there)
TK_TABLE_MAX = 1.0e7
CYL_ITERATIONS = 64              # ternary-search steps of the cylinder margin (interval T_MAX * (2/3)**64 ~ 5e-6 mm)
CYL_EXACT_UP_TO = 1.0e4          # mm: cylinder margins are exact up to here; beyond, the bounding-box margin (a lower
                                 # bound) is kept, so margins above this value are not supported for cylinders
MUON = 13
NONMUON_POLICIES = ("always", "threshold")
MASS = {11: 0.51099895, 13: eloss.MMU, 22: 0.0, 211: 139.57039, 321: 493.677, 2112: 939.56542, 2212: 938.27209}


# ----------------------------------------------------------------------------------------------- envelopes
@dataclass(frozen=True)
class Envelope:
    """One detector envelope in the global frame. axis = -1: the box [lo, hi]; axis 0/1/2: a solid cylinder along
    that global axis (center, radius, half-length), with lo/hi its bounding box."""
    name: str
    term: str
    lo: tuple
    hi: tuple
    axis: int = -1
    center: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.0
    half: float = 0.0

    def __post_init__(self):
        if self.axis not in (-1, 0, 1, 2):
            raise ValueError("axis must be -1 (box) or 0, 1, 2 (cylinder)")
        if np.shape(self.lo) != (3,) or np.shape(self.hi) != (3,) or np.any(np.asarray(self.lo) > self.hi):
            raise ValueError(f"envelope {self.name}: invalid bounds")
        if self.axis >= 0 and not (self.radius > 0 and self.half > 0):
            raise ValueError(f"envelope {self.name}: cylinder needs radius > 0 and half > 0")

    @classmethod
    def box(cls, name, term, lo, hi):
        return cls(name, term, tuple(map(float, lo)), tuple(map(float, hi)))

    @classmethod
    def cylinder(cls, name, term, center, axis, radius, half):
        c = np.asarray(center, float)
        h = np.full(3, float(radius))
        h[axis] = float(half)
        return cls(name, term, tuple(c - h), tuple(c + h), int(axis), tuple(c), float(radius), float(half))


def _strip(n):
    return re.sub(r"0x[0-9a-f]+$", "", n)


def _mm(solid):
    unit = solid.get("lunit", "mm")
    if unit != "mm":
        raise ValueError(f"solid {solid.get('name')}: lunit {unit!r} (only mm is supported)")


def _tube_frame(solid):
    """(rmax, zmin, zmax) of a tube or polycone in its own frame, else None."""
    if solid.tag == "tube":
        _mm(solid)
        z = float(solid.get("z"))
        return float(solid.get("rmax")), -0.5 * z, 0.5 * z
    if solid.tag == "polycone":
        _mm(solid)
        planes = solid.findall("zplane")
        if not planes:
            return None
        zs = [float(p.get("z")) for p in planes]
        return max(float(p.get("rmax")) for p in planes), min(zs), max(zs)
    return None


def _corners(solid, solids, R, t):
    """Global corners of a solid's bounding box (box, xtru and booleans through world._pieces_of; tube, polycone)."""
    tube = _tube_frame(solid)
    if tube is not None:
        r, z0, z1 = tube
        loc = np.array([[x, y, z] for x in (-r, r) for y in (-r, r) for z in (z0, z1)])
        return loc @ R.T + t
    if solid.tag in ("box", "xtru"):
        _mm(solid)
    pieces = world_mod._pieces_of(solid, solids, R, t, "")
    if not pieces:
        return np.empty((0, 3))
    return np.vstack([np.vstack([q[2], q[3]]) for q in pieces])


def envelopes_from_gdml(gdml_path, selection=EXTRACTED_SELECTION):
    """Envelopes of the placed volumes whose names (pointer suffix stripped) fully match a pattern of `selection`
    ({term: (regex, ...)}). A matched volume is not descended into. A tube or polycone whose axis is a global
    principal axis becomes a cylinder; any other volume becomes the bounding box of its own solid and of every solid
    below it. Raises if a term matches nothing."""
    root = ET.parse(gdml_path).getroot()
    vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
    solids = {s.get("name"): s for s in root.find("solids")}
    pats = [(term, re.compile(p)) for term, ps in selection.items() for p in ps]
    out = []

    def solid_of(lv):
        return solids[vols[lv].find("solidref").get("ref")]

    def subtree_corners(lv, R, t, acc):
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            R2, t2 = _placement(pv, R, t)
            acc.append(_corners(solid_of(child), solids, R2, t2))
            subtree_corners(child, R2, t2, acc)

    def envelope(name, term, lv, R, t):
        s = solid_of(lv)
        tube = _tube_frame(s)
        if tube is not None:
            ax = R[:, 2]
            a = int(np.argmax(np.abs(ax)))
            if abs(abs(ax[a]) - 1.0) < 1e-9:
                r, z0, z1 = tube
                return Envelope.cylinder(name, term, R @ np.array([0.0, 0.0, 0.5 * (z0 + z1)]) + t, a, r,
                                         0.5 * (z1 - z0))
        acc = [_corners(s, solids, R, t)]
        subtree_corners(lv, R, t, acc)
        pts = np.vstack(acc)
        if not len(pts):
            raise ValueError(f"envelope {name}: no supported solid in its subtree")
        return Envelope.box(name, term, pts.min(0), pts.max(0))

    def walk(lv, R, t):
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            R2, t2 = _placement(pv, R, t)
            name = _strip(child)
            term = next((tm for tm, p in pats if p.fullmatch(name)), None)
            if term is not None:
                out.append(envelope(name, term, child, R2, t2))
            else:
                walk(child, R2, t2)

    walk(root.find("setup/world").get("ref"), np.eye(3), np.zeros(3))
    missing = [t for t in selection if not any(e.term == t for e in out)]
    if missing:
        raise ValueError(f"no envelope matched for term(s) {missing}")
    return out


def _placement(pv, R, t):
    p = pv.find("position")
    if p is not None and p.get("unit", "mm") != "mm":
        raise ValueError("only mm positions are supported")
    pos = np.array([float(p.get(a, 0)) for a in "xyz"]) if p is not None else np.zeros(3)
    return R @ world_mod._rot(pv.find("rotation")), R @ pos + t


# ----------------------------------------------------------------------------------------------- geometry
_EPS_DIR = 1e-12


def _box_bounds(o, d, lo, hi):
    """Slab data of a box: A, B [N, 3] (entry/exit parameters of each slab), U = 1/|d| [N, 3] (0 on axes with
    d = 0, where A = -inf, B = +inf) and G [N] (largest signed distance along those axes)."""
    flat = np.abs(d) <= _EPS_DIR
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(flat, 0.0, 1.0 / np.where(flat, 1.0, d))
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    A = np.where(flat, -np.inf, np.minimum(t1, t2))
    B = np.where(flat, np.inf, np.maximum(t1, t2))
    U = np.where(flat, 0.0, np.abs(inv))
    G = np.where(flat, np.maximum(lo - o, o - hi), -np.inf).max(1)
    return A, B, U, G


def box_required_margin(o, d, lo, hi, t_max=T_MAX):
    """Smallest m with which the segment o + t d, 0 <= t <= t_max, meets the box [lo - m, hi + m] (exact).

    The slab of axis i holds t in [A_i - m U_i, B_i + m U_i]; with the segment as a fourth interval [0, t_max]
    (U = 0), the intervals share a point iff every lower bound is below every upper bound, so m is the largest
    pairwise threshold (A_i - B_j) / (U_i + U_j), or the signed distance along an axis the line does not cross."""
    dt = np.result_type(o, d)
    lo = np.asarray(lo, dt)
    hi = np.asarray(hi, dt)
    A, B, U, G = _box_bounds(o, d, lo, hi)
    n = len(o)
    A = np.concatenate([A, np.zeros((n, 1), dt)], 1)
    B = np.concatenate([B, np.full((n, 1), t_max, dt)], 1)
    U = np.concatenate([U, np.zeros((n, 1), dt)], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pair = (A[:, :, None] - B[:, None, :]) / (U[:, :, None] + U[:, None, :])
    pair = np.where(np.isnan(pair), -np.inf, pair)
    return np.maximum(pair.reshape(n, -1).max(1), G).astype(dt)


def box_entry(o, d, lo, hi, m, t_max=T_MAX):
    """Entry parameter t >= 0 of the segment into the box grown by m (inf when it misses)."""
    dt = np.result_type(o, d)
    A, B, U, G = _box_bounds(o, d, np.asarray(lo, dt), np.asarray(hi, dt))
    m = np.asarray(m, dt)
    mm = m[:, None] if m.ndim else m          # one margin, or one per segment
    low = np.maximum((A - mm * U).max(1), 0)
    high = np.minimum((B + mm * U).min(1), t_max)
    return np.where((low <= high) & (G <= m), low, np.inf).astype(dt)


def _cyl_frame(o, d, env):
    dt = np.result_type(o, d)
    c = np.asarray(env.center, dt)
    a = env.axis
    tr = [k for k in range(3) if k != a]
    w = o - c
    return w[:, tr], d[:, tr], w[:, a], d[:, a]


def cylinder_required_margin(o, d, env, t_max=T_MAX):
    """Smallest m with which the segment meets the cylinder grown by m (radius + m, half-length + m):
    min over t in [0, t_max] of max(rho(t) - R, |z(t)| - H), a convex function, by ternary search."""
    dt = np.result_type(o, d)
    wT, dT, wa, da = _cyl_frame(o, d, env)
    R, H = dt.type(env.radius), dt.type(env.half)

    def f(t):
        q = wT + t[:, None] * dT
        return np.maximum(np.sqrt((q * q).sum(1)) - R, np.abs(wa + t * da) - H)

    a = np.zeros(len(o), dt)
    b = np.full(len(o), t_max, dt)
    for _ in range(CYL_ITERATIONS):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        left = f(m1) <= f(m2)
        b = np.where(left, m2, b)
        a = np.where(left, a, m1)
    return np.minimum(f(0.5 * (a + b)), np.minimum(f(np.zeros_like(a)), f(np.full_like(a, t_max)))).astype(dt)


def cylinder_entry(o, d, env, m, t_max=T_MAX):
    """Entry parameter t >= 0 into the cylinder grown by m (inf when it misses). The radial interval is taken
    around the transverse point of closest approach (no cancellation for distant generator points)."""
    dt = np.result_type(o, d)
    wT, dT, wa, da = _cyl_frame(o, d, env)
    m = np.asarray(m, dt)
    R, H = dt.type(env.radius) + m, dt.type(env.half) + m
    aa = (dT * dT).sum(1)
    flat = aa <= _EPS_DIR ** 2
    t0 = np.where(flat, 0, -(wT * dT).sum(1) / np.where(flat, 1, aa))
    q = wT + t0[:, None] * dT
    D2 = (q * q).sum(1)
    with np.errstate(invalid="ignore"):
        half = np.sqrt(np.maximum(R * R - D2, 0) / np.where(flat, 1, aa))
    r_lo = np.where(flat, -np.inf, t0 - half)
    r_hi = np.where(flat, np.inf, t0 + half)
    r_ok = D2 <= R * R
    zflat = np.abs(da) <= _EPS_DIR
    with np.errstate(divide="ignore", invalid="ignore"):
        z1 = (-H - wa) / np.where(zflat, 1, da)
        z2 = (H - wa) / np.where(zflat, 1, da)
    z_lo = np.where(zflat, -np.inf, np.minimum(z1, z2))
    z_hi = np.where(zflat, np.inf, np.maximum(z1, z2))
    z_ok = ~zflat | (np.abs(wa) <= H)
    low = np.maximum(np.maximum(r_lo, z_lo), 0)
    high = np.minimum(np.minimum(r_hi, z_hi), t_max)
    return np.where(r_ok & z_ok & (low <= high), low, np.inf).astype(dt)


def required_margin(o, d, env, t_max=T_MAX):
    if env.axis < 0:
        return box_required_margin(o, d, env.lo, env.hi, t_max)
    # the cylinder grown by m lies inside its bounding box grown by m, so the box margin is a lower bound: lines
    # farther than CYL_EXACT_UP_TO from the box keep it
    m = box_required_margin(o, d, env.lo, env.hi, t_max)
    near = m <= CYL_EXACT_UP_TO
    if near.any():
        m[near] = cylinder_required_margin(o[near], d[near], env, t_max)
    return m


def entry(o, d, env, m, t_max=T_MAX):
    if env.axis < 0:
        return box_entry(o, d, env.lo, env.hi, m, t_max)
    return cylinder_entry(o, d, env, m, t_max)


# ----------------------------------------------------------------------------------------------- range
class RangeTable:
    """CSDA range of a muon in REF_MATERIAL from the full mean dE/dx (eloss.dedx_restricted with no delta cut), and
    the concrete-equivalent length of a material column (each length scaled by dE/dx(P_REF) / dE/dx_ref(P_REF))."""

    def __init__(self, ref=REF_MATERIAL, p_ref=P_REF, n=4000, densities=None):
        self.ref = ref
        self.densities = dict(densities or {})
        tk = np.geomspace(TK_TABLE_MIN, TK_TABLE_MAX, n)
        p = np.sqrt((tk + eloss.MMU) ** 2 - eloss.MMU ** 2)
        inv = 1.0 / eloss.dedx_restricted(ref, p, 1e9)
        r = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(tk))])
        self.tk = np.concatenate([[0.0], tk])
        self.r = np.concatenate([[0.0], r + TK_TABLE_MIN * inv[0]])   # linear to zero below the table start
        self.p_ref = p_ref
        self._ref_dedx = float(eloss.dedx_restricted(ref, np.array([p_ref]), 1e9)[0])
        self._scale = {}
        self.unknown = set()

    def scale(self, name):
        """dE/dx of the named material relative to the reference at P_REF. A material without constants in
        mu2efastsim.materials is scaled by its density (`densities`, g/cm3, e.g. gdml_densities: dE/dx per g/cm2
        varies by about 10 % between concrete and iron), or counts as the reference when its density is unknown
        too (as transport.world_loss does); either way it is recorded in `unknown`."""
        if name not in self._scale:
            mat = materials.BY_NAME.get(name)
            if mat is None:
                self.unknown.add(name)
                rho = self.densities.get(name)
                self._scale[name] = rho / self.ref.rho if rho else 1.0
            else:
                self._scale[name] = float(eloss.dedx_restricted(mat, np.array([self.p_ref]), 1e9)[0]) / self._ref_dedx
        return self._scale[name]

    def range_of(self, tk):
        return np.interp(tk, self.tk, self.r)

    def ke_after(self, tk, length_eq):
        """Kinetic energy after a concrete-equivalent length; negative (-1) when the muon ranges out before."""
        left = self.range_of(tk) - length_eq
        return np.where(left > 0, np.interp(np.maximum(left, 0), self.r, self.tk), -1.0)


def gdml_densities(gdml_path):
    """{material name (pointer suffix stripped): density in g/cm3} of a GDML file."""
    out = {}
    for m in ET.parse(gdml_path).getroot().find("materials").findall("material"):
        dd = m.find("D")
        if dd is not None and dd.get("unit", "g/cm3") == "g/cm3":
            out[_strip(m.get("name"))] = float(dd.get("value"))
    return out


# ----------------------------------------------------------------------------------------------- decision
@dataclass(frozen=True)
class Params:
    """Pre-filter parameters (D6). See the module docstring for their meaning."""
    margin_mm: float = 500.0
    min_muon_ke_mev: float | None = 0.0
    nonmuon: str = "always"
    nonmuon_min_ke_mev: float = 0.0
    nonmuon_margin_mm: float | None = None
    terms: tuple = TERMS

    def __post_init__(self):
        if self.nonmuon not in NONMUON_POLICIES:
            raise ValueError(f"nonmuon must be one of {NONMUON_POLICIES}")
        if not self.terms or any(t not in TERMS for t in self.terms):
            raise ValueError(f"terms must be a non-empty subset of {TERMS}")
        for m in (self.margin_mm, self.nonmuon_margin_mm):
            if m is not None and not (np.isfinite(m) and m <= CYL_EXACT_UP_TO):
                raise ValueError(f"margins must be finite and at most {CYL_EXACT_UP_TO} mm")
        if self.min_muon_ke_mev is not None and not self.min_muon_ke_mev >= 0:
            raise ValueError("min_muon_ke_mev must be >= 0 or None")
        if not self.nonmuon_min_ke_mev >= 0:
            raise ValueError("nonmuon_min_ke_mev must be >= 0")

    def record(self):
        """Provenance entry (JSON-serialisable): what an output's metadata carries."""
        d = asdict(self)
        d["terms"] = list(self.terms)
        return d


def kinetic_energy(pdg, e):
    a = np.abs(np.asarray(pdg))
    m = np.zeros(len(a))
    known = np.zeros(len(a), bool)
    for code, mass in MASS.items():
        sel = a == code
        m[sel], known[sel] = mass, True
    if not known.all():
        raise ValueError(f"unknown CRY particle code(s): {sorted(set(a[~known].tolist()))}")
    return np.asarray(e, float) - m


@dataclass
class Features:
    """Per-primary quantities from which any parameter set is decided without new geometry.

    margin_star [N, 3]: required margin per term (TERMS order; +inf for a term without envelopes);
    tk_env [N, 3, M]: muon kinetic energy at the entry of each term's envelope grown by margins[k] (NaN where the line
    misses at that margin or the primary is not a muon; -1 when it ranges out before)."""
    event: np.ndarray
    pdg: np.ndarray
    tk: np.ndarray
    margin_star: np.ndarray
    margins: np.ndarray
    tk_env: np.ndarray
    dtype: str
    unknown_materials: list = field(default_factory=list)

    def save(self, path):
        np.savez_compressed(path, event=self.event, pdg=self.pdg, tk=self.tk, margin_star=self.margin_star,
                            margins=self.margins, tk_env=self.tk_env, dtype=np.array(self.dtype),
                            unknown_materials=np.array(self.unknown_materials, dtype=str))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["event"], z["pdg"], z["tk"], z["margin_star"], z["margins"], z["tk_env"], str(z["dtype"]),
                   [str(x) for x in z["unknown_materials"]])

    @classmethod
    def concat(cls, parts):
        if not parts:
            raise ValueError("no features to concatenate")
        m0 = parts[0].margins
        if any(not np.array_equal(p.margins, m0) or p.dtype != parts[0].dtype for p in parts):
            raise ValueError("features with different margin grids or dtypes")
        cat = lambda k: np.concatenate([getattr(p, k) for p in parts])   # noqa: E731
        unknown = sorted({u for p in parts for u in p.unknown_materials})
        return cls(cat("event"), cat("pdg"), cat("tk"), cat("margin_star"), m0, cat("tk_env"), parts[0].dtype, unknown)


class Prefilter:
    """The decision of the module docstring for one geometry: envelopes (`envelopes_from_gdml`) and, for the range
    check, the world's convex pieces (`world.load(gdml)`; None: no material, only air)."""

    def __init__(self, envelopes, world_pieces=None, dtype=np.float64, t_max=T_MAX):
        self.envelopes = list(envelopes)
        if not self.envelopes:
            raise ValueError("no envelopes")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError("dtype must be float32 or float64")
        self.t_max = t_max
        self.world = None
        if world_pieces is not None and len(world_pieces):
            ps = world_pieces if isinstance(world_pieces, world_mod.PieceSet) else world_mod.PieceSet(world_pieces)
            self.world = ps.build_grid() if getattr(ps, "grid", None) is None else ps
        self.range = RangeTable()

    @classmethod
    def from_gdml(cls, gdml_path, selection=EXTRACTED_SELECTION, dtype=np.float64, material=True):
        envs = envelopes_from_gdml(gdml_path, selection)
        pf = cls(envs, world_mod.load(gdml_path) if material else None, dtype)
        pf.range.densities.update(gdml_densities(gdml_path))
        return pf

    def _cast(self, primaries):
        o = np.column_stack([primaries["x"], primaries["y"], primaries["z"]]).astype(self.dtype)
        p = np.column_stack([primaries["px"], primaries["py"], primaries["pz"]]).astype(np.float64)
        pn = np.linalg.norm(p, axis=1)
        if np.any(pn <= 0):
            raise ValueError("primary with zero momentum")
        return o, (p / pn[:, None]).astype(self.dtype)

    def margin_star(self, o, d):
        out = np.full((len(o), len(TERMS)), np.inf, self.dtype)
        for env in self.envelopes:
            k = TERMS.index(env.term)
            out[:, k] = np.minimum(out[:, k], required_margin(o, d, env, self.t_max))
        return out

    def term_entry(self, o, d, term, m):
        t = np.full(len(o), np.inf, self.dtype)
        for env in self.envelopes:
            if env.term == term:
                t = np.minimum(t, entry(o, d, env, m, self.t_max))
        return t

    def column_eq(self, o, d, t):
        """Concrete-equivalent length [mm] of the path o -> o + t d (material pieces + air for the rest)."""
        t = np.asarray(t, float)
        solid = np.zeros(len(o))
        eq = np.zeros(len(o))
        if self.world is not None and len(o):
            # sub-segments of at most SUBSEGMENT_MM: the PieceSet grid search visits every cell of a segment's x-z
            # bounding box, which grows with the square of a long slanted path; path lengths are additive
            o64, d64 = np.asarray(o, float), np.asarray(d, float)
            n = np.maximum(np.ceil(t / SUBSEGMENT_MM).astype(np.int64), 1)
            row = np.repeat(np.arange(len(o)), n)
            k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            start = k * SUBSEGMENT_MM
            seg = np.clip(t[row] - start, 0, SUBSEGMENT_MM)
            for name, length in self.world.column(o64[row] + start[:, None] * d64[row], d64[row], seg).items():
                per = np.bincount(row, weights=length, minlength=len(o))
                solid += per
                eq += self.range.scale(name) * per
        air = np.clip(t - solid, 0, None)
        return (eq + self.range.scale(materials.AIR.name) * air).astype(self.dtype)

    def features(self, primaries, margins):
        """Features of every primary (a mapping of equal-length columns: event, pdg, x, y, z, px, py, pz, e) for the
        given margin grid. The range is computed only for muons whose line meets a term at the largest margin."""
        margins = np.asarray(sorted(set(float(m) for m in margins)), self.dtype)
        o, d = self._cast(primaries)
        pdg = np.asarray(primaries["pdg"], np.int32)
        tk = kinetic_energy(pdg, primaries["e"]).astype(self.dtype)
        ms = self.margin_star(o, d)
        tk_env = np.full((len(o), len(TERMS), len(margins)), np.nan, self.dtype)
        mu = np.abs(pdg) == MUON
        for k, term in enumerate(TERMS):
            for j, m in enumerate(margins):
                sel = np.flatnonzero(mu & (ms[:, k] <= m))
                if not len(sel):
                    continue
                t = self.term_entry(o[sel], d[sel], term, m)
                ok = np.isfinite(t)
                sel, t = sel[ok], t[ok]
                eq = self.column_eq(o[sel], d[sel], t)
                tk_env[sel, k, j] = self.range.ke_after(tk[sel].astype(float), eq.astype(float)).astype(self.dtype)
        return Features(np.asarray(primaries["event"], np.int64), pdg, tk, ms, margins, tk_env, self.dtype.name,
                        sorted(self.range.unknown))

    def decide(self, primaries, params):
        """(keep per primary, unique events, keep per event) for one parameter set."""
        f = self.features(primaries, [params.margin_mm] + ([params.nonmuon_margin_mm]
                                                           if params.nonmuon_margin_mm is not None else []))
        keep = decide_primaries(f, params)
        ev, kev = event_keep(f.event, keep)
        return keep, ev, kev


def decide_primaries(f, params):
    """Keep flag of every primary of a Features for one parameter set (the margin must be in the features' grid)."""
    j = np.flatnonzero(np.isclose(f.margins, params.margin_mm, rtol=0, atol=1e-6))
    if not len(j):
        raise ValueError(f"margin {params.margin_mm} not in the features' grid {f.margins.tolist()}")
    j = int(j[0])
    terms = [TERMS.index(t) for t in params.terms]
    mu = np.abs(f.pdg) == MUON
    geo = f.margin_star[:, terms] <= params.margin_mm
    if params.min_muon_ke_mev is None:
        reach = geo
    else:
        te = f.tk_env[:, terms, j]
        reach = geo & (np.nan_to_num(te, nan=-np.inf) >= params.min_muon_ke_mev)
    keep_mu = mu & reach.any(1)
    if params.nonmuon == "always":
        keep_nm = ~mu
    else:
        keep_nm = ~mu & (f.tk >= params.nonmuon_min_ke_mev)
        if params.nonmuon_margin_mm is not None:
            keep_nm &= (f.margin_star[:, terms] <= params.nonmuon_margin_mm).any(1)
    return keep_mu | keep_nm


def event_keep(event, keep):
    """(unique event keys, any(keep) per event)."""
    ev, inv = np.unique(np.asarray(event), return_inverse=True)
    out = np.zeros(len(ev), bool)
    np.logical_or.at(out, inv, np.asarray(keep, bool))
    return ev, out
