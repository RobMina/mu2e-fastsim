"""On-device optical-photon transport of the CRV lookup-table generator (JAX, float32 or float64).

A port of the table mode (-c) of CRVStandalone: per bin, photons are generated as WLSPrimaryGeneratorAction does and
tracked through the strip (counter.Counter) with the Geant4 10.7 optical physics that program uses (optics.py,
materials.py), and the photons detected by the photodetector at -z on the fiber at -y ("SiPM 0") are histogrammed as
WLSEventAction does. The bins come from a binning.Binning.

Execution model: a fixed set of lanes, one photon chain per lane (every WLS process re-emits at most one photon, so a
chain never branches). A step advances every live lane to its next interaction; finished lanes are refilled from a
queue of (bin, photon) indices. All randomness comes from a counter-based generator keyed by (seed, sim type) with the
counter built from (bin, photon index, step, draw), so a bin's photons do not depend on how the bins are split into
jobs.

Exact shortcut: a ray in the fiber core whose first hit on the core/cladding interface is a total internal reflection
keeps hitting it at the same angle (concentric polished cylinders), so it is moved to its Y11 absorption point or the
fiber end in one step along the reflected (polygonal helix) path, with the same path length, time, position, direction
and polarization as the bounce-by-bounce transport. The same holds for a ray confined to the inner cladding annulus.

Geant4 detail kept on purpose: the unified-model reflection probabilities of G4OpBoundaryProcess are members that a
surface without a property table does not reset. The fiber-end/air surfaces have no table, so their reflections use
the values of the last unified surface with a table that was hit: lobe after the TiO2 coating (probability 1), and
Lambertian after the fiber guide bar (all zero). Each lane carries this flag from photon to photon, like the process
object carries it from track to track.
"""
import math
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from . import materials as _materials  # noqa: F401  (registers the "current" model)
from . import optics as M
from .binning import Binning
from .counter import Counter
from .limits import DRAW_BITS, LEN_BINS, STEP_BITS  # noqa: F401  (re-exported)

jax.config.update("jax_enable_x64", True)

# regions
DEAD, SCINT, HOLE, CLAD2, CLAD1, CORE, GAP, EPOXY = range(8)
# materials (index into the stacked RINDEX / GROUPVEL tables)
PS, PMMA, FPE, AIR, EPX = range(5)
REG_MAT = np.array([AIR, PS, AIR, FPE, PMMA, PS, AIR, EPX], np.int32)
# surfaces a lane may sit on (to drop the zero-distance root of the surface it just crossed)
S_NONE, S_R1, S_R2, S_R3, S_HOLE, S_OUTER = 0, 1, 2, 3, 4, 7
# events
E_NONE, E_PSPPO, E_POPOP, E_Y11, E_TIO2, E_BLACK, E_DIEL, E_SI, E_REFL, E_LOST = range(10)

TWO_PI = 2.0 * math.pi
NT, NE = 150, 10                 # LookupBin::maxTimeDelays, maxFiberEmissions
K_FACET = 8                      # facet candidates drawn per boundary sub-iteration
MAX_SUB = 24                     # boundary sub-iterations per step (unfinished lanes are counted)
# draw offsets inside one step
R_PROC, R_SKIP, R_WLS, R_METAL, R_SUB = 0, 4, 8, 20, 32
SUB_STRIDE = 2 * K_FACET + 6


def _vec(x):
    return jnp.asarray(np.asarray(x, np.float64))


@dataclass(frozen=True)
class Config:
    counter: Counter
    binning: Binning
    simtype: int
    dtype: str = "float64"
    seed: int = 1
    lanes: int = 1 << 16
    refill_fraction: float = 0.125
    helix: bool = True              # exact helical shortcut for totally reflected fiber rays (False: bounce by bounce)
    rough_reflection: str = "stale"  # fiber/air reflections: "stale" (Geant4: inherited), "lobe" or "lambert" (tests)
    model: str = M.DEFAULT_MODEL    # name of a registered optical model (optics.get)
    max_steps: int = 1 << STEP_BITS  # chains are stopped (and counted as capped) at this many steps; <= 2**STEP_BITS

    def __post_init__(self):
        if not 2 < self.max_steps <= (1 << STEP_BITS):
            raise ValueError(f"max_steps must be in (2, 2**{STEP_BITS}]")


class Tables:
    """Device copies of the material tables (G4 vectors as sorted (energy, value) arrays)."""

    def __init__(self, dtype, model):
        f = jnp.float64 if dtype == "float64" else jnp.float32
        self.f = f
        m = model

        def vec(v):
            return (jnp.asarray(v.e, f), jnp.asarray(v.v, f))

        rind = [m.rindex_ps, m.rindex_pmma, m.rindex_fpe, M.RINDEX_AIR, m.rindex_epoxy]
        vgs = [m.vg_ps, m.vg_pmma, m.vg_fpe, M.VG_AIR, m.vg_epoxy]
        self.rindex = [vec(v) for v in rind]
        self.vg = [vec(v) for v in vgs]
        self.abs_psppo, self.abs_popop, self.abs_y11 = vec(m.abs_psppo), vec(m.abs_popop), vec(m.abs_y11)
        self.qy_psppo, self.qy_popop, self.qy_y11 = vec(m.qy_psppo), vec(m.qy_popop), vec(m.qy_y11)
        # emission integrals of PS+PPO (scintillation and WLS), POPOP and Y11
        self.cii = [vec(M.integral(m.em_ppo)), vec(M.integral(m.em_popop)), vec(M.integral(m.em_y11))]
        self.refl_tio2, self.eff_mppc = vec(m.refl_tio2), vec(m.eff_mppc)
        self.icdf = jnp.asarray(np.stack([M.facet_alpha_icdf(M.SIGMA_TIO2), M.facet_alpha_icdf(M.SIGMA_FIBER_END)]), f)
        self.nmax_ps = float(m.rindex_ps.v.max())


def g4_value(tab, x):
    """G4PhysicsVector::Value(x) with a fresh index (see materials.G4Vector.value)."""
    e, v = tab
    n = e.shape[0]
    i = jnp.searchsorted(e, x, side="left") - 1
    i = jnp.where((x > e[1]) | (x < e[0]), i, 0)
    i = jnp.where(x < e[1], 0, i)
    i = jnp.clip(jnp.where(x >= e[n - 2], n - 2, i), 0, n - 2)
    dl = e[i + 1] - e[i]
    b = jnp.where(dl > 0, jnp.clip((x - e[i]) / jnp.where(dl > 0, dl, 1), 0, 1), 1)
    y = v[i] + b * (v[i + 1] - v[i])
    y = jnp.where(x <= e[0], v[0], y)
    return jnp.where(x >= e[n - 1], v[n - 1], y)


def g4_energy(tab, a):
    """G4PhysicsOrderedFreeVector::GetEnergy(a)."""
    e, v = tab
    n = e.shape[0]
    i = jnp.clip(jnp.searchsorted(v, a, side="left") - 1, 0, n - 2)
    d = v[i + 1] - v[i]
    r = jnp.where(d > 0, e[i] + (a - v[i]) * (e[i + 1] - e[i]) / jnp.where(d > 0, d, 1), e[i])
    r = jnp.where(a <= v[0], e[0], r)
    return jnp.where(a >= v[n - 1], e[n - 1], r)


# ------------------------------------------------------------------------------------------------ random numbers
def _key(seed, simtype):
    """splitmix64 of (seed, simtype), forced odd: the key of the squares counter generator."""
    z = (int(seed) * 0x9E3779B97F4A7C15 + int(simtype) * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) | 1


def squares32(ctr, key):
    """Widynski's squares32 counter-based generator (uint64 counter -> uint32)."""
    key = jnp.uint64(key)
    x = ctr * key
    y = x
    z = y + key
    x = x * x + y
    x = (x >> jnp.uint64(32)) | (x << jnp.uint64(32))
    x = x * x + z
    x = (x >> jnp.uint64(32)) | (x << jnp.uint64(32))
    x = x * x + y
    x = (x >> jnp.uint64(32)) | (x << jnp.uint64(32))
    return ((x * x + z) >> jnp.uint64(32)).astype(jnp.uint32)


def uniforms(base, offsets, key, f):
    """Uniforms in (0, 1): base [N] uint64 counters, offsets a static tuple/range -> [N, len(offsets)]."""
    off = jnp.asarray(np.asarray(offsets, np.uint64))
    u = squares32(base[:, None] + off[None, :], key)
    if f == jnp.float64:
        return (u.astype(jnp.float64) + 0.5) * (1.0 / 4294967296.0)
    return ((u >> jnp.uint32(8)).astype(jnp.float32) + jnp.float32(0.5)) * jnp.float32(1.0 / 16777216.0)


# ------------------------------------------------------------------------------------------------ vector helpers
def dot(a, b):
    return jnp.sum(a * b, axis=-1)


def cross(a, b):
    return jnp.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                      a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def unit(a):
    n = jnp.sqrt(dot(a, a))
    return a / jnp.where(n > 0, n, 1)[..., None]


def rotate_uz(v, u):
    """CLHEP Hep3Vector::rotateUz: rotate v from the frame whose z axis is the unit vector u."""
    u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]
    px, py, pz = v[..., 0], v[..., 1], v[..., 2]
    up2 = u1 * u1 + u2 * u2
    up = jnp.sqrt(up2)
    ups = jnp.where(up > 0, up, 1)
    rx = (u1 * u3 * px - u2 * py) / ups + u1 * pz
    ry = (u2 * u3 * px + u1 * py) / ups + u2 * pz
    rz = -up * px + u3 * pz
    neg = u3 < 0
    rx = jnp.where(up2 > 0, rx, jnp.where(neg, -px, px))
    ry = jnp.where(up2 > 0, ry, py)
    rz = jnp.where(up2 > 0, rz, jnp.where(neg, -pz, pz))
    return jnp.stack([rx, ry, rz], -1)


def iso_dir_pol(u1, u2, u3):
    """Isotropic direction and random linear polarization exactly as G4Scintillation / G4OpWLS draw them."""
    cost = 1 - 2 * u1
    sint = jnp.sqrt((1 - cost) * (1 + cost))
    phi = TWO_PI * u2
    sp, cp = jnp.sin(phi), jnp.cos(phi)
    mom = jnp.stack([sint * cp, sint * sp, cost], -1)
    pol = jnp.stack([cost * cp, cost * sp, -sint], -1)
    perp = cross(mom, pol)
    phi = TWO_PI * u3
    pol = unit(jnp.cos(phi)[:, None] * pol + jnp.sin(phi)[:, None] * perp)
    return mom, pol


def facet_candidates(icdf_tab, which, u_a, u_p, mom, N):
    """GetFacetNormal of the unified model for K candidates per lane; returns the first valid (mom . f < 0) one.

    icdf_tab [2, n+1]: alpha at equally spaced CDF points; which [N] selects the surface (0 TiO2, 1 fiber end)."""
    n = icdf_tab.shape[1] - 1
    x = u_a * n
    i = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, n - 1)
    w = x - i
    tab = icdf_tab[which]                                       # [N, n+1]
    a0 = jnp.take_along_axis(tab, i, 1)
    a1 = jnp.take_along_axis(tab, i + 1, 1)
    alpha = a0 + w * (a1 - a0)
    phi = TWO_PI * u_p
    sa = jnp.sin(alpha)
    local = jnp.stack([sa * jnp.cos(phi), sa * jnp.sin(phi), jnp.cos(alpha)], -1)   # [N, K, 3]
    f = rotate_uz(local, N[:, None, :])
    ok = dot(mom[:, None, :], f) < 0
    first = jnp.argmax(ok, axis=1)
    any_ok = jnp.any(ok, axis=1)
    fsel = jnp.take_along_axis(f, first[:, None, None], 1)[:, 0, :]
    return fsel, any_ok


def lambertian(N, u1, u2):
    """G4LambertianRand(N): cosine-weighted direction around N."""
    ct = jnp.sqrt(u1)
    st = jnp.sqrt(1 - u1)
    phi = TWO_PI * u2
    return rotate_uz(jnp.stack([st * jnp.cos(phi), st * jnp.sin(phi), ct], -1), N)


# ------------------------------------------------------------------------------------------------ geometry
def circle_roots(rx, ry, dx, dy, R, on):
    """Ray (rx, ry) + t (dx, dy) against the circle of radius R: (t_low, t_high, valid); on: the lane sits on it."""
    a = dx * dx + dy * dy
    b = rx * dx + ry * dy
    c = jnp.where(on, 0, rx * rx + ry * ry - R * R)
    disc = b * b - a * c
    ok = (disc >= 0) & (a > 0)
    sq = jnp.sqrt(jnp.where(ok, disc, 0))
    q = -(b + jnp.where(b >= 0, sq, -sq))
    qs = jnp.where(q != 0, q, 1)
    as_ = jnp.where(a > 0, a, 1)
    t1, t2 = q / as_, jnp.where(q != 0, c / qs, 0)
    return jnp.minimum(t1, t2), jnp.maximum(t1, t2), ok


class Kernel:
    """Builds the jitted step loop for one counter, sim type and dtype."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        c = cfg.counter
        self.model = M.get(cfg.model)
        self.T = Tables(cfg.dtype, self.model)
        f = self.T.f
        self.f = f
        self.key = _key(cfg.seed, cfg.simtype)
        self.key_src = _key(cfg.seed, cfg.simtype + 16)
        self.eps = 1e-9 if cfg.dtype == "float64" else 2e-5
        self.c = c
        self.hx, self.hy, self.rc = c.hx, c.hy, c.rc
        self.hl = c.half_length
        self.zend = c.half_length + c.guide_bar                 # fiber end plane
        self.zdev = self.zend + c.air_gap                       # SiPM / reflector face
        # epoxy window: a box of half-length sipm_window placed (sipm_length - sipm_window)/2 off the SiPM centre
        # towards the fiber, so it starts inside the fiber end region and ends inside the silicon
        wc = self.zdev + c.sipm_length / 2 - (c.sipm_length - c.sipm_window) / 2
        self.zsi = wc + c.sipm_window              # epoxy -> silicon
        self.zback = wc - c.sipm_window            # window face behind the fiber end (the stock overlap)
        self.s2 = c.fiber_separation / 2
        self.r = (c.r_core, c.r_clad1, c.r_clad2)
        self.ax, self.ay = c.hole_rx, c.hole_ry
        self.hw = c.sipm_width / 2
        self.R_plus = self.model.refl_reflector if c.reflector == 1 else 0.0
        self.contact = self.model.fiber_end == "contact"
        self.rough = self.model.rough_fiber_air

    # --------------------------------------------------------------------------------------- per-lane properties
    def n_of(self, E):
        return jnp.stack([g4_value(t, E) for t in self.T.rindex], -1)          # [N, 5]

    def vg_of(self, E):
        return jnp.stack([g4_value(t, E) for t in self.T.vg], -1)

    # --------------------------------------------------------------------------------------- boundaries
    def boundaries(self, s):
        """Distance, event and parameters of the next boundary of every lane (straight-line transport)."""
        f = self.f
        p, d, reg, fib, surf = s["p"], s["d"], s["reg"], s["fib"], s["surf"]
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        inf = jnp.asarray(jnp.inf, f)
        eps = self.eps
        yc = jnp.where(fib == 0, -self.s2, self.s2).astype(f)
        rx, ry = x, y - yc
        r1, r2, r3 = self.r

        # circles of the lane's fiber
        lo1, hi1, ok1 = circle_roots(rx, ry, dx, dy, r1, surf == S_R1)
        lo2, hi2, ok2 = circle_roots(rx, ry, dx, dy, r2, surf == S_R2)
        lo3, hi3, ok3 = circle_roots(rx, ry, dx, dy, r3, surf == S_R3)

        def near(lo, hi, ok):     # entering from outside: the smaller root, if positive
            return jnp.where(ok & (lo > eps), lo, inf)

        def far(hi, ok):          # leaving from inside
            return jnp.where(ok & (hi > eps), hi, inf)

        # ellipses (both fibers, for the scintillator; the lane's own for the hole)
        def ellipse(yc_, on):
            X, Y, DX, DY = x / self.ax, (y - yc_) / self.ay, dx / self.ax, dy / self.ay
            lo, hi, ok = circle_roots(X, Y, DX, DY, 1.0, on)
            return lo, hi, ok

        e0lo, e0hi, e0ok = ellipse(-self.s2, (surf == S_HOLE) & (fib == 0))
        e1lo, e1hi, e1ok = ellipse(self.s2, (surf == S_HOLE) & (fib == 1))
        t_e0 = jnp.where(e0ok & (e0lo > eps), e0lo, inf)
        t_e1 = jnp.where(e1ok & (e1lo > eps), e1lo, inf)
        t_eown_far = jnp.where(fib == 0, jnp.where(e0ok & (e0hi > eps), e0hi, inf),
                               jnp.where(e1ok & (e1hi > eps), e1hi, inf))

        # outer rounded rectangle (scintillator exit)
        hx, hy, rc = self.hx, self.hy, self.rc
        tx = jnp.where(dx > 0, (hx - x) / jnp.where(dx != 0, dx, 1), jnp.where(dx < 0, (-hx - x) / jnp.where(dx != 0, dx, 1), inf))
        ty = jnp.where(dy > 0, (hy - y) / jnp.where(dy != 0, dy, 1), jnp.where(dy < 0, (-hy - y) / jnp.where(dy != 0, dy, 1), inf))
        tbox = jnp.minimum(tx, ty)
        qx, qy = x + tbox * dx, y + tbox * dy
        corner = (jnp.abs(qx) > hx - rc) & (jnp.abs(qy) > hy - rc)
        ccx, ccy = jnp.sign(qx) * (hx - rc), jnp.sign(qy) * (hy - rc)
        clo, chi, cok = circle_roots(x - ccx, y - ccy, dx, dy, rc,
                                     (surf == S_OUTER) & (jnp.abs((x - ccx) ** 2 + (y - ccy) ** 2 - rc * rc) < 1e-6))
        t_out = jnp.where(corner, jnp.where(cok & (chi > eps), chi, tbox), tbox)
        qx, qy = x + t_out * dx, y + t_out * dy
        n_face = jnp.where((tx <= ty)[:, None], jnp.stack([jnp.sign(dx), 0 * dx, 0 * dx], -1),
                           jnp.stack([0 * dy, jnp.sign(dy), 0 * dy], -1))
        n_corner = jnp.stack([(qx - ccx) / rc, (qy - ccy) / rc, 0 * qx], -1)
        n_out = jnp.where(corner[:, None], n_corner, n_face)

        # z planes
        dzs = jnp.where(dz != 0, dz, 1)
        t_zbar = jnp.where(dz > 0, (self.hl - z) / dzs, jnp.where(dz < 0, (-self.hl - z) / dzs, inf))
        t_zend = jnp.where(dz > 0, (self.zend - z) / dzs, jnp.where(dz < 0, (-self.zend - z) / dzs, inf))
        esgn = jnp.sign(z)                                  # which end (gap / epoxy lanes)
        outward = dz * esgn > 0
        t_gap = jnp.where(outward, (esgn * self.zdev - z) / dzs, (esgn * self.zend - z) / dzs)
        t_epx = jnp.where(outward, (esgn * self.zsi - z) / dzs, (esgn * self.zback - z) / dzs)
        t_gap = jnp.where(dz == 0, inf, t_gap)
        t_epx = jnp.where(dz == 0, inf, t_epx)
        # lateral faces of the gap / window (SiPM box): |x| or |y - yc| = sipm_width / 2
        hw = self.hw
        tlx = jnp.where(dx > 0, (hw - x) / jnp.where(dx != 0, dx, 1), jnp.where(dx < 0, (-hw - x) / jnp.where(dx != 0, dx, 1), inf))
        tly = jnp.where(dy > 0, (hw - ry) / jnp.where(dy != 0, dy, 1), jnp.where(dy < 0, (-hw - ry) / jnp.where(dy != 0, dy, 1), inf))
        t_lat = jnp.minimum(tlx, tly)
        n_lat = jnp.where((tlx <= tly)[:, None], jnp.stack([jnp.sign(dx), 0 * dx, 0 * dx], -1),
                          jnp.stack([0 * dy, jnp.sign(dy), 0 * dy], -1))

        zhat = jnp.stack([0 * z, 0 * z, jnp.sign(dz)], -1)

        # candidates: (mask, t, event, n1, n2, rough, next region, next surface, normal [N,3], next fiber)
        cands = []

        def add(mask, t, ev, n1, n2, rough, nreg, nsurf, nrm, nfib=None):
            cands.append((jnp.where(mask, t, inf), ev, n1, n2, rough, nreg, nsurf, nrm,
                          fib if nfib is None else nfib))

        def radial(R, t):
            return jnp.stack([(rx + t * dx) / R, (ry + t * dy) / R, 0 * t], -1)

        def ell_n(t, yc_):
            X, Y = (x + t * dx) / (self.ax * self.ax), (y + t * dy - yc_) / (self.ay * self.ay)
            return unit(jnp.stack([X, Y, 0 * X], -1))

        sc, ho, c2, c1, co, gp, ep = (reg == k for k in (SCINT, HOLE, CLAD2, CLAD1, CORE, GAP, EPOXY))
        zero = jnp.zeros_like(fib)
        t_n3, t_n2, t_n1 = near(lo3, hi3, ok3), near(lo2, hi2, ok2), near(lo1, hi1, ok1)
        t_f3, t_f2, t_f1 = far(hi3, ok3), far(hi2, ok2), far(hi1, ok1)
        # scintillator: TiO2 coating, bar ends (fiber guide bar: black), the two holes (polished PS -> air)
        add(sc, t_out, E_TIO2, PS, PS, False, SCINT, S_OUTER, n_out)
        add(sc, t_zbar, E_BLACK, PS, PS, False, DEAD, S_NONE, zhat)
        add(sc, t_e0, E_DIEL, PS, AIR, False, HOLE, S_HOLE, ell_n(t_e0, -self.s2), zero)
        add(sc, t_e1, E_DIEL, PS, AIR, False, HOLE, S_HOLE, ell_n(t_e1, self.s2), zero + 1)
        # hole (air): back into the scintillator (polished), onto the outer cladding (rough), bar ends (black)
        add(ho, t_eown_far, E_DIEL, AIR, PS, False, SCINT, S_HOLE, ell_n(t_eown_far, yc))
        rgh = self.rough
        add(ho, t_n3, E_DIEL, AIR, FPE, rgh, CLAD2, S_R3, radial(r3, t_n3))
        add(ho, t_zbar, E_BLACK, AIR, AIR, False, DEAD, S_NONE, zhat)
        # outer cladding: radially out in the bar region -> hole air (rough); inside the guide bar -> black
        inbar = jnp.abs(z + t_f3 * dz) < self.hl
        add(c2 & inbar, t_f3, E_DIEL, FPE, AIR, rgh, HOLE, S_R3, radial(r3, t_f3))
        add(c2 & ~inbar, t_f3, E_BLACK, FPE, FPE, False, DEAD, S_NONE, radial(r3, t_f3))
        add(c2, t_n2, E_DIEL, FPE, PMMA, False, CLAD1, S_R2, radial(r2, t_n2))
        # fiber end: "gap" model -> rough interface to the air gap; "contact" model -> the SiPM / reflector disc
        fz = jnp.sign(dz)
        end_dev_sipm = (fz < 0) | (self.c.reflector == 0)
        fe = (c2 | c1 | co) & self.contact
        add(fe & end_dev_sipm, t_zend, E_SI, PS, PS, False, DEAD, S_NONE, zhat)
        add(fe & ~end_dev_sipm, t_zend, E_REFL, PS, PS, False, DEAD, S_NONE, zhat)
        fg = ~jnp.asarray(self.contact)
        add(c2 & fg, t_zend, E_DIEL, FPE, AIR, True, GAP, S_NONE, zhat)
        # inner cladding
        add(c1, t_f2, E_DIEL, PMMA, FPE, False, CLAD2, S_R2, radial(r2, t_f2))
        add(c1, t_n1, E_DIEL, PMMA, PS, False, CORE, S_R1, radial(r1, t_n1))
        add(c1 & fg, t_zend, E_DIEL, PMMA, AIR, True, GAP, S_NONE, zhat)
        # core
        add(co, t_f1, E_DIEL, PS, PMMA, False, CLAD1, S_R1, radial(r1, t_f1))
        add(co & fg, t_zend, E_DIEL, PS, AIR, True, GAP, S_NONE, zhat)
        # gap: outward -> SiPM window (polished air -> epoxy) or reflector; inward -> fiber end (rough) or guide bar
        rr = jnp.sqrt((rx + t_gap * dx) ** 2 + (ry + t_gap * dy) ** 2)
        dev_sipm = (esgn < 0) | (self.c.reflector == 0)
        add(gp & outward & dev_sipm, t_gap, E_DIEL, AIR, EPX, False, EPOXY, S_NONE, zhat)
        add(gp & outward & ~dev_sipm, t_gap, E_REFL, AIR, AIR, False, GAP, S_NONE, zhat)
        add(gp & ~outward & (rr < r1), t_gap, E_DIEL, AIR, PS, True, CORE, S_NONE, zhat)
        add(gp & ~outward & (rr >= r1) & (rr < r2), t_gap, E_DIEL, AIR, PMMA, True, CLAD1, S_NONE, zhat)
        add(gp & ~outward & (rr >= r2) & (rr < r3), t_gap, E_DIEL, AIR, FPE, True, CLAD2, S_NONE, zhat)
        add(gp & ~outward & (rr >= r3), t_gap, E_BLACK, AIR, AIR, False, DEAD, S_NONE, zhat)
        add(gp, t_lat, E_LOST, AIR, AIR, False, DEAD, S_NONE, n_lat)
        # epoxy window: outward / lateral -> silicon; inward -> the fiber layer behind it (polished) or guide bar
        rr = jnp.sqrt((rx + t_epx * dx) ** 2 + (ry + t_epx * dy) ** 2)
        add(ep & outward, t_epx, E_SI, EPX, EPX, False, EPOXY, S_NONE, zhat)
        add(ep & ~outward & (rr < r1), t_epx, E_DIEL, EPX, PS, False, CORE, S_NONE, zhat)
        add(ep & ~outward & (rr >= r1) & (rr < r2), t_epx, E_DIEL, EPX, PMMA, False, CLAD1, S_NONE, zhat)
        add(ep & ~outward & (rr >= r2) & (rr < r3), t_epx, E_DIEL, EPX, FPE, False, CLAD2, S_NONE, zhat)
        add(ep & ~outward & (rr >= r3), t_epx, E_BLACK, EPX, EPX, False, DEAD, S_NONE, zhat)
        add(ep, t_lat, E_SI, EPX, EPX, False, EPOXY, S_NONE, n_lat)

        ts = jnp.stack([c_[0] for c_ in cands], 1)                         # [N, C]
        k = jnp.argmin(ts, axis=1)

        def static(i, dtype):
            return jnp.asarray(np.array([c_[i] for c_ in cands], dtype))[k]

        nrm = jnp.take_along_axis(jnp.stack([c_[7] for c_ in cands], 1), k[:, None, None], 1)[:, 0]
        nfib = jnp.take_along_axis(jnp.stack([jnp.broadcast_to(c_[8], fib.shape) for c_ in cands], 1),
                                   k[:, None], 1)[:, 0]
        # impact parameter of the ray about the lane's fiber axis (for the cladding-helix shortcut)
        lt = jnp.sqrt(dx * dx + dy * dy)
        bimp = jnp.abs(rx * dy - ry * dx) / jnp.where(lt > 0, lt, 1)
        return {"t": jnp.take_along_axis(ts, k[:, None], 1)[:, 0], "ev": static(1, np.int32),
                "n1": static(2, np.int32), "n2": static(3, np.int32), "rough": static(4, bool),
                "nreg": static(5, np.int32), "nsurf": static(6, np.int32), "nrm": nrm, "nfib": nfib.astype(jnp.int32),
                "core_far": t_f1, "c1_far": t_f2, "bimp": bimp}

    # --------------------------------------------------------------------------------------- helix shortcut
    def helix(self, p, d, pol, yc, R, s):
        """Move along a polygonal helix inside a circle of radius R (every hit a reflection) by path length s."""
        f = self.f
        rx, ry = p[:, 0], p[:, 1] - yc
        dtx, dty, dz = d[:, 0], d[:, 1], d[:, 2]
        lt = jnp.sqrt(dtx * dtx + dty * dty)
        lts = jnp.where(lt > 0, lt, 1)
        hx_, hy_ = dtx / lts, dty / lts
        D = s * lt
        proj = rx * hx_ + ry * hy_
        cr = rx * hy_ - ry * hx_                                   # signed impact parameter
        t0 = -proj + jnp.sqrt(jnp.maximum(R * R - (rx * rx + ry * ry) + proj * proj, 0))
        b = jnp.abs(cr)
        chord = 2 * jnp.sqrt(jnp.maximum(R * R - b * b, 0))
        gam = jnp.arccos(jnp.clip(b / R, -1, 1))
        th = 2 * gam * jnp.where(cr >= 0, 1, -1)
        Dp = D - t0
        before = (Dp <= 0) | (lt <= 0)
        j = jnp.where(before, 0, jnp.floor(Dp / jnp.where(chord > 0, chord, 1)) + 1)      # reflections
        rem = Dp - (j - 1) * chord
        B0x, B0y = rx + t0 * hx_, ry + t0 * hy_

        def rot(vx, vy, a):
            ca, sa = jnp.cos(a), jnp.sin(a)
            return ca * vx - sa * vy, sa * vx + ca * vy

        jm = jnp.mod((j - 1) * th, TWO_PI)
        jj = jnp.mod(j * th, TWO_PI)
        Px, Py = rot(B0x, B0y, jm)
        Dx, Dy = rot(hx_, hy_, jj)
        nx = jnp.where(before, rx + D * hx_, Px + rem * Dx)
        ny = jnp.where(before, ry + D * hy_, Py + rem * Dy)
        ndx = jnp.where(before, dtx, Dx * lt)
        ndy = jnp.where(before, dty, Dy * lt)
        # polarization: pol_j = (-1)^j Ref_{j-1} ... Ref_0 pol (mirrors about the radial normals n_k)
        jodd = jnp.mod(j, 2) == 1
        ex, ey, ez = pol[:, 0], pol[:, 1], pol[:, 2]
        ang_even = jnp.mod(j * th, TWO_PI)
        ang_odd = jm
        a = jnp.where(jodd, ang_odd, ang_even)
        qx, qy = rot(ex, ey, a)
        # last mirror (odd j): normal at B_{j-1}
        Bn = jnp.sqrt(Px * Px + Py * Py)
        nnx, nny = Px / jnp.where(Bn > 0, Bn, 1), Py / jnp.where(Bn > 0, Bn, 1)
        pn = qx * nnx + qy * nny
        qx = jnp.where(jodd, qx - 2 * pn * nnx, qx)
        qy = jnp.where(jodd, qy - 2 * pn * nny, qy)
        sgn = jnp.where(jodd, -1.0, 1.0).astype(f)
        npol = jnp.stack([jnp.where(before, ex, sgn * qx), jnp.where(before, ey, sgn * qy),
                          jnp.where(before, ez, sgn * ez)], -1)
        npos = jnp.stack([nx, ny + yc, p[:, 2] + s * dz], -1)
        nd = jnp.stack([ndx, ndy, dz], -1)
        return npos, nd, unit(npol)

    # --------------------------------------------------------------------------------------- sources
    def source(self, lb, box, pid, key_src):
        """New photons for local bins lb (WLSPrimaryGeneratorAction): state pieces, inside flag, failed flag."""
        f, T = self.f, self.T
        st = self.cfg.simtype
        base = pid.astype(jnp.uint64) << jnp.uint64(16)
        lo = {k: jnp.asarray(v[0], f)[lb] for k, v in box.items()}
        hi = {k: jnp.asarray(v[1], f)[lb] for k, v in box.items()}
        U = uniforms(base, range(16), key_src, f)
        bad = jnp.zeros(lb.shape, bool)
        if st in (0, 1):
            x = lo["x"] + U[:, 0] * (hi["x"] - lo["x"])
            y = lo["y"] + U[:, 1] * (hi["y"] - lo["y"])
            z = lo["z"] + U[:, 2] * (hi["z"] - lo["z"])
            inside = self.inside_scint(x, y, z)
            mom, pol = iso_dir_pol(U[:, 3], U[:, 4], U[:, 5])
            if st == 0:
                E = g4_energy(T.cii[0], U[:, 6] * T.cii[0][1][-1])
                md = self.model
                tr = 1.0 / (1.0 / md.scint_decay + 1.0 / md.scint_rise)   # decay x (1 - rise): sum of two exponentials
                t = -md.scint_decay * jnp.log(U[:, 7]) - tr * jnp.log(U[:, 8])
            else:
                bi = 1.0 / hi["beta"] + U[:, 6] * (1.0 / lo["beta"] - 1.0 / hi["beta"])
                E, _, bad = self.cerenkov_energy(bi, base, key_src)
                t = jnp.zeros_like(x)
            reg = jnp.full(lb.shape, SCINT, jnp.int32)
            fib = jnp.zeros(lb.shape, jnp.int32)
            p = jnp.stack([x, y, z], -1)
        else:
            bi = 1.0 / hi["beta"] + U[:, 0] * (1.0 / lo["beta"] - 1.0 / hi["beta"])
            E, cos_t, bad = self.cerenkov_energy(bi, base, key_src)
            sin_t = jnp.sqrt(jnp.maximum((1 - cos_t) * (1 + cos_t), 0))
            ph = TWO_PI * U[:, 1]
            mom_l = jnp.stack([sin_t * jnp.cos(ph), sin_t * jnp.sin(ph), cos_t], -1)
            pol_l = jnp.stack([cos_t * jnp.cos(ph), cos_t * jnp.sin(ph), -sin_t], -1)
            th = lo["theta"] + U[:, 2] * (hi["theta"] - lo["theta"])
            pp = lo["phi"] + U[:, 3] * (hi["phi"] - lo["phi"])
            part = jnp.stack([jnp.sin(th) * jnp.cos(pp), jnp.sin(th) * jnp.sin(pp), jnp.cos(th)], -1)
            mom = unit(rotate_uz(mom_l, part))
            pol = unit(rotate_uz(pol_l, part))
            r = jnp.sqrt((hi["r"] ** 2 - lo["r"] ** 2) * U[:, 4] + lo["r"] ** 2)
            z = lo["z"] + U[:, 5] * (hi["z"] - lo["z"])
            # stock: startPosition(minR, separation/2, minZ) + (r, 0, dz), i.e. x = minR + r (the radius is added to
            # the bin's lower edge), in the +y fiber; outer bins start outside the core, even in the hole air
            x = lo["r"] + r
            p = jnp.stack([x, jnp.full_like(r, self.s2), z], -1)
            t = jnp.zeros_like(r)
            r1, r2, r3 = self.r
            reg = jnp.where(x < r1, CORE, jnp.where(x < r2, CLAD1, jnp.where(x < r3, CLAD2, HOLE))).astype(jnp.int32)
            fib = jnp.ones(lb.shape, jnp.int32)
            inside = jnp.ones(lb.shape, bool)
        return dict(p=p, d=mom, e=pol, E=E, t=t, reg=reg, fib=fib), inside, bad

    def inside_scint(self, x, y, z):
        hx, hy, rc = self.hx, self.hy, self.rc
        ok = (jnp.abs(x) <= hx) & (jnp.abs(y) <= hy) & (jnp.abs(z) <= self.hl)
        cx, cy = jnp.abs(x) - (hx - rc), jnp.abs(y) - (hy - rc)
        ok &= ~((cx > 0) & (cy > 0) & (cx * cx + cy * cy > rc * rc))
        for yc in (-self.s2, self.s2):
            X, Y = x / self.ax, (y - yc) / self.ay
            ok &= X * X + Y * Y >= 1
        return ok

    def cerenkov_energy(self, bi, base, key_src):
        """G4Cerenkov energy sampling: E uniform over the RINDEX range, kept if U * maxSin2 <= sin2(E).

        Proposals outside {n(E) > 1/beta} are always rejected, so drawing E uniformly on the hull of that set (from
        the piecewise-linear RINDEX) and applying the same acceptance gives the same distribution with far fewer
        rejections. Returns E, cos(theta), failed (no acceptance within the candidate budget)."""
        f = self.f
        e, n = self.T.rindex[PS]
        maxcos = bi / self.T.nmax_ps
        maxsin2 = (1 - maxcos) * (1 + maxcos)
        # hull of the support: per RINDEX segment the part with n > 1/beta (linear crossing)
        e0, e1, n0, n1 = e[:-1][None, :], e[1:][None, :], n[:-1][None, :], n[1:][None, :]
        b = bi[:, None]
        dn = jnp.where(n1 != n0, n1 - n0, 1)
        ex = e0 + (b - n0) / dn * (e1 - e0)
        up0, up1 = n0 > b, n1 > b
        seg_lo = jnp.where(up0, e0, jnp.where(up1, ex, jnp.inf))
        seg_hi = jnp.where(up1, e1, jnp.where(up0, ex, -jnp.inf))
        elo = jnp.min(seg_lo, 1)
        ehi = jnp.max(seg_hi, 1)
        ok_range = ehi > elo
        elo = jnp.where(ok_range, elo, e[0])
        ehi = jnp.where(ok_range, ehi, e[-1])
        K = 64

        def body(c):
            it, E, done = c
            U = uniforms(base + (64 + it * 2 * K).astype(jnp.uint64), range(2 * K), key_src, f)
            Ec = elo[:, None] + U[:, :K] * (ehi - elo)[:, None]
            cos = bi[:, None] / g4_value((e, n), Ec)
            acc = U[:, K:] * maxsin2[:, None] <= (1 - cos) * (1 + cos)
            first = jnp.argmax(acc, 1)
            got = jnp.any(acc, 1) & ~done
            Esel = jnp.take_along_axis(Ec, first[:, None], 1)[:, 0]
            return it + 1, jnp.where(got, Esel, E), done | got

        _, E, done = lax.while_loop(lambda c: (c[0] < 400) & ~jnp.all(c[2]), body,
                                    (jnp.int32(0), jnp.full(bi.shape, e[0], f), jnp.zeros(bi.shape, bool)))
        return E, bi / g4_value((e, n), E), ~done

    # --------------------------------------------------------------------------------------- one step
    def step(self, s, aux):
        """Advance every live lane to its next interaction and apply it. Returns (state, aux, record mask)."""
        f, T, key = self.f, self.T, self.key
        N = s["reg"].shape[0]
        reg = s["reg"]
        alive = reg != DEAD
        base = (s["pid"].astype(jnp.uint64) << jnp.uint64(STEP_BITS + DRAW_BITS)) | \
            (s["nstep"].astype(jnp.uint64) << jnp.uint64(DRAW_BITS))
        U = uniforms(base, range(R_WLS), key, f)
        E, d, pol = s["E"], s["d"], s["e"]
        nE, vgE = self.n_of(E), self.vg_of(E)
        bd = self.boundaries(s)

        # process lengths (WLS absorption; Geant4's interaction-length bookkeeping is memoryless, so resampled)
        lam1 = jnp.where(reg == SCINT, g4_value(T.abs_psppo, E), jnp.inf)
        lam2 = jnp.where(reg == SCINT, g4_value(T.abs_popop, E), jnp.inf)
        lam3 = jnp.where(reg == CORE, g4_value(T.abs_y11, E), jnp.inf)
        t1, t2, t3 = (-jnp.log(U[:, i]) * lam for i, lam in enumerate((lam1, lam2, lam3)))
        tp = jnp.minimum(jnp.minimum(t1, t2), t3)
        evp = jnp.where(t1 <= jnp.minimum(t2, t3), E_PSPPO, jnp.where(t2 <= t3, E_POPOP, E_Y11))

        # helical shortcut (exact): the next transverse hit is a total internal reflection on a polished circle
        yc = jnp.where(s["fib"] == 0, -self.s2, self.s2).astype(f)

        def tir_at(R, tfar, n1, n2):
            tt = jnp.where(jnp.isfinite(tfar), tfar, 0)
            hx_ = s["p"][:, 0] + tt * d[:, 0]
            hy_ = s["p"][:, 1] - yc + tt * d[:, 1]
            c1 = jnp.abs(hx_ * d[:, 0] + hy_ * d[:, 1]) / R
            s1 = jnp.sqrt(jnp.maximum(1 - c1 * c1, 0))
            return (s1 * n1 / n2 >= 1) & jnp.isfinite(tfar)

        hel_core = (reg == CORE) & (bd["ev"] == E_DIEL) & (bd["nreg"] == CLAD1) & \
            tir_at(self.r[0], bd["core_far"], nE[:, PS], nE[:, PMMA])
        hel_c1 = (reg == CLAD1) & (bd["ev"] == E_DIEL) & (bd["nreg"] == CLAD2) & (bd["bimp"] > self.r[0]) & \
            tir_at(self.r[1], bd["c1_far"], nE[:, PMMA], nE[:, FPE])
        hel = (hel_core | hel_c1) & alive & self.cfg.helix
        dz = d[:, 2]
        zhat = jnp.stack([0 * dz, 0 * dz, jnp.sign(dz)], -1)
        t_endz = jnp.where(dz != 0, (jnp.sign(dz) * self.zend - s["p"][:, 2]) / jnp.where(dz != 0, dz, 1), jnp.inf)
        t_b = jnp.where(hel, t_endz, bd["t"])
        if self.contact:       # the helix ends on the SiPM / reflector disc
            ev_end = jnp.where((jnp.sign(dz) < 0) | (self.c.reflector == 0), E_SI, E_REFL)
        else:                  # the helix ends on the rough fiber-end / air interface
            ev_end = E_DIEL
        ev_b = jnp.where(hel, ev_end, bd["ev"])
        n1_b = jnp.where(hel, jnp.where(hel_core, PS, PMMA), bd["n1"])
        n2_b = jnp.where(hel, AIR, bd["n2"])
        rough_b = jnp.where(hel, True, bd["rough"])
        nreg_b = jnp.where(hel, GAP, bd["nreg"])
        nsurf_b = jnp.where(hel, S_NONE, bd["nsurf"])
        nrm_b = jnp.where(hel[:, None], zhat, bd["nrm"])

        use_proc = tp < t_b
        tstep = jnp.where(use_proc, tp, t_b)
        ev = jnp.where(alive, jnp.where(use_proc, evp, ev_b), E_NONE)
        stuck = alive & ~jnp.isfinite(tstep)
        ev = jnp.where(stuck, E_LOST, ev)
        tstep = jnp.where(alive & ~stuck, tstep, 0)

        # move (time at the group velocity of the traversed material)
        vg = jnp.take_along_axis(vgE, jnp.asarray(REG_MAT)[reg][:, None], 1)[:, 0]
        Rh = jnp.where(hel_core, self.r[0], self.r[1]).astype(f)
        hp, hd, he = self.helix(s["p"], d, pol, yc, Rh, tstep)
        p = jnp.where(hel[:, None], hp, s["p"] + tstep[:, None] * d)
        d = jnp.where(hel[:, None], hd, d)
        pol = jnp.where(hel[:, None], he, pol)
        t = s["t"] + tstep / vg
        reg_new = reg
        lobe = s["lobe"]
        nem = s["nem"]
        fib = s["fib"]
        surf = s["surf"]

        # ---- WLS absorption and re-emission (G4OpWLSY11 / PSPPO / POPOP)
        is_wls = (ev == E_PSPPO) | (ev == E_POPOP) | (ev == E_Y11)
        wi = jnp.where(ev == E_PSPPO, 0, jnp.where(ev == E_POPOP, 1, 2))
        Uw = uniforms(base, range(R_WLS, R_WLS + 8), key, f)
        qy = jnp.where(wi == 0, g4_value(T.qy_psppo, E),
                       jnp.where(wi == 1, g4_value(T.qy_popop, E), g4_value(T.qy_y11, E)))
        cE = jnp.where(wi == 0, g4_value(T.cii[0], E), jnp.where(wi == 1, g4_value(T.cii[1], E), g4_value(T.cii[2], E)))
        cM = jnp.where(wi == 0, T.cii[0][1][-1], jnp.where(wi == 1, T.cii[1][1][-1], T.cii[2][1][-1]))
        F = jnp.clip(cE / cM, 0, 1)
        fail = Uw[:, 1] < (1 - F) ** 100           # 100 draws above the primary energy
        a = Uw[:, 2] * cE
        Enew = jnp.where(wi == 0, g4_energy(T.cii[0], a), jnp.where(wi == 1, g4_energy(T.cii[1], a),
                                                                   g4_energy(T.cii[2], a)))
        wmom, wpol = iso_dir_pol(Uw[:, 3], Uw[:, 4], Uw[:, 5])
        md = self.model
        tau = jnp.where(wi == 0, md.tau_psppo, jnp.where(wi == 1, md.tau_popop, md.tau_y11)).astype(f)
        wls_kill = is_wls & ((Uw[:, 0] > qy) | fail)
        wls_emit = is_wls & ~wls_kill
        d = jnp.where(wls_emit[:, None], wmom, d)
        pol = jnp.where(wls_emit[:, None], wpol, pol)
        E2 = jnp.where(wls_emit, Enew, E)
        t = jnp.where(wls_emit, t - tau * jnp.log(Uw[:, 6]), t)
        nem = jnp.where(wls_emit, jnp.where(wi == 2, nem + 1, 0), nem)
        reg_new = jnp.where(wls_kill, DEAD, reg_new)
        surf = jnp.where(use_proc, S_NONE, surf)

        # ---- black (fiber guide bar and bar ends: reflectivity 0) and lost
        black = ev == E_BLACK
        reg_new = jnp.where(black | (ev == E_LOST), DEAD, reg_new)
        lobe = jnp.where(black, False, lobe)       # unified surface with a table but no lobe constants

        # ---- glisur metal surfaces (polish 1: specular): SiPM silicon and the reflector
        Um = uniforms(base, range(R_METAL, R_METAL + 4), key, f)
        nrm = nrm_b
        Ng = jnp.where((dot(d, nrm) > 0)[:, None], -nrm, nrm)           # Geant4's global normal: against the momentum
        is_si, is_rf = ev == E_SI, ev == E_REFL
        Rm = jnp.where(is_si, self.model.refl_mppc, self.R_plus).astype(f)
        absorb = (is_si | is_rf) & (Um[:, 0] > Rm)
        detected = is_si & absorb & (Um[:, 1] < g4_value(T.eff_mppc, E))
        refl_m = (is_si | is_rf) & ~absorb
        d = jnp.where(refl_m[:, None], unit(d - 2 * dot(d, Ng)[:, None] * Ng), d)
        pol = jnp.where(refl_m[:, None], unit(-pol + 2 * dot(pol, Ng)[:, None] * Ng), pol)
        reg_new = jnp.where(absorb, DEAD, reg_new)
        record = detected & (p[:, 2] < 0) & (fib == 0)

        # ---- TiO2 coating (unified, ground, sigma_alpha 1.5, lobe) and dielectric interfaces
        is_t = ev == E_TIO2
        t_abs = is_t & (Um[:, 2] > g4_value(T.refl_tio2, E))
        reg_new = jnp.where(t_abs, DEAD, reg_new)
        lobe = jnp.where(is_t, True, lobe)
        is_d = ev == E_DIEL
        n1 = jnp.take_along_axis(nE, n1_b[:, None], 1)[:, 0]
        n2 = jnp.take_along_axis(nE, n2_b[:, None], 1)[:, 0]
        rough = rough_b & is_d
        pend0 = ((is_t & ~t_abs) | is_d) & alive
        which = jnp.where(is_t, 0, 1)
        if self.cfg.rough_reflection == "lobe":
            lamb = jnp.zeros_like(rough)
        elif self.cfg.rough_reflection == "lambert":
            lamb = rough
        else:
            lamb = rough & ~lobe

        def sub(c):
            it, mom, pl, Ngl, a1, a2, inside, through, pend = c
            Us = uniforms(base + (R_SUB + it * SUB_STRIDE).astype(jnp.uint64), range(SUB_STRIDE), key, f)
            sw = through & is_d                                         # G4: swap sides after a transmission
            Ngl = jnp.where(sw[:, None], -Ngl, Ngl)
            a1, a2 = jnp.where(sw, a2, a1), jnp.where(sw, a1, a2)
            need_f = pend & (is_t | rough)
            fac, fok = facet_candidates(T.icdf, which, Us[:, :K_FACET], Us[:, K_FACET:2 * K_FACET], mom, Ngl)
            facet = jnp.where(need_f[:, None], fac, Ngl)
            go = pend & (fok | ~need_f)
            # TiO2: lobe reflection about the facet, repeated while it points into the surface
            nm_t = mom - 2 * dot(mom, facet)[:, None] * facet
            ne_t = -pl + 2 * dot(pl, facet)[:, None] * facet
            done_t = dot(nm_t, Ngl) >= 0
            # dielectric (G4OpBoundaryProcess::DielectricDielectric, one pass of its loop)
            EdotN = dot(pl, facet)
            c1 = -dot(mom, facet)
            obl = jnp.abs(c1) < 1 - 1e-9
            s1 = jnp.where(obl, jnp.sqrt(jnp.maximum(1 - c1 * c1, 0)), 0)
            s2 = jnp.where(obl, s1 * a1 / a2, 0)
            tir = s2 >= 1
            c2 = jnp.where(c1 > 0, 1, -1) * jnp.sqrt(jnp.maximum(1 - s2 * s2, 0))
            At = jnp.where(obl[:, None], unit(cross(mom, facet)), pl)
            E1perp = jnp.where(obl, dot(pl, At), 0)
            E1pl = pl - E1perp[:, None] * At
            E1parl = jnp.where(obl, jnp.sqrt(dot(E1pl, E1pl)), 1)
            sR = a1 * c1
            den1 = a1 * c1 + a2 * c2
            den2 = a2 * c1 + a1 * c2
            E2perp = 2 * sR * E1perp / jnp.where(den1 != 0, den1, 1)
            E2parl = 2 * sR * E1parl / jnp.where(den2 != 0, den2, 1)
            E2tot = E2perp * E2perp + E2parl * E2parl
            sT = a2 * c2 * E2tot
            Tc = jnp.where(c1 != 0, sT / jnp.where(sR != 0, sR, 1), 0)
            transmit = ~tir & (Us[:, 2 * K_FACET] < Tc)
            spec = mom - 2 * dot(mom, facet)[:, None] * facet
            lmb = lambertian(Ngl, Us[:, 2 * K_FACET + 1], Us[:, 2 * K_FACET + 2])
            nm_r = jnp.where(lamb[:, None], lmb, spec)
            fl = unit(lmb - mom)
            ne_lamb = -pl + 2 * dot(pl, fl)[:, None] * fl
            ne_tir = -pl + 2 * EdotN[:, None] * facet
            E2parl_r = a2 * E2parl / a1 - E1parl
            E2perp_r = E2perp - E1perp
            E2abs_r = jnp.sqrt(E2perp_r ** 2 + E2parl_r ** 2)
            Ap_r = unit(cross(spec, At))
            Er = jnp.where(E2abs_r > 0, E2abs_r, 1)
            ne_fr = jnp.where(obl[:, None], (E2parl_r / Er)[:, None] * Ap_r + (E2perp_r / Er)[:, None] * At,
                              jnp.where((a2 > a1)[:, None], -pl, pl))
            ne_r = jnp.where(lamb[:, None], ne_lamb, jnp.where(tir[:, None], ne_tir, ne_fr))
            alp = c1 - c2 * (a2 / a1)
            nm_tr = jnp.where(obl[:, None], unit(mom + alp[:, None] * facet), mom)
            Ap_t = unit(cross(nm_tr, At))
            E2abs = jnp.sqrt(E2tot)
            Et = jnp.where(E2abs > 0, E2abs, 1)
            ne_tr = jnp.where(obl[:, None], (E2parl / Et)[:, None] * Ap_t + (E2perp / Et)[:, None] * At, pl)
            nm_d = jnp.where(transmit[:, None], nm_tr, nm_r)
            ne_d = jnp.where(transmit[:, None], ne_tr, ne_r)
            done_d = jnp.where(transmit, dot(nm_d, Ngl) <= 0, dot(nm_d, Ngl) >= -1e-9)
            upd_t, upd_d = go & is_t, go & is_d
            mom = jnp.where(upd_t[:, None], unit(nm_t), jnp.where(upd_d[:, None], unit(nm_d), mom))
            pl = jnp.where(upd_t[:, None], unit(ne_t), jnp.where(upd_d[:, None], unit(ne_d), pl))
            inside = jnp.where(upd_d & transmit, ~inside, inside)
            through = jnp.where(upd_d, transmit, through & ~sw)
            pend = pend & ~(upd_t & done_t) & ~(upd_d & done_d)
            return it + 1, mom, pl, Ngl, a1, a2, inside, through, pend

        init = (jnp.int32(0), d, pol, Ng, n1, n2, jnp.zeros(N, bool), jnp.zeros(N, bool), pend0)
        _, mom_f, pol_f, _, _, _, inside_f, _, pend_f = lax.while_loop(
            lambda c: (c[0] < MAX_SUB) & jnp.any(c[8]), sub, init)
        d = jnp.where(pend0[:, None], mom_f, d)
        pol = jnp.where(pend0[:, None], pol_f, pol)
        crossed = is_d & inside_f
        reg_new = jnp.where(crossed, nreg_b, reg_new)
        fib = jnp.where(is_d, bd["nfib"], fib)
        surf = jnp.where(is_d, nsurf_b, jnp.where(is_t, S_OUTER, surf))
        surf = jnp.where(is_si | is_rf, S_NONE, surf)
        reg_new = jnp.where(stuck, DEAD, reg_new)
        long_ = alive & (s["nstep"] + 2 >= self.cfg.max_steps)    # step budget of a chain exhausted
        reg_new = jnp.where(long_, DEAD, reg_new)
        ns = dict(s, p=p, d=d, e=pol, E=E2, t=t, reg=reg_new.astype(jnp.int32), fib=fib.astype(jnp.int32),
                  surf=surf.astype(jnp.int32), nem=nem, lobe=lobe, nstep=s["nstep"] + 1)
        # chain length (steps of the photon and its re-emissions) of every chain ending here, and of the SiPM-0
        # arrivals: what a lower step budget would drop
        ended = alive & (reg_new == DEAD)
        lk = jnp.clip(63 - lax.clz(s["nstep"] + 1), 0, LEN_BINS - 1).astype(jnp.int32)
        aux = dict(aux, helix=aux["helix"] + jnp.sum(hel & ~use_proc), forced=aux["forced"] + jnp.sum(pend_f),
                   capped=aux["capped"] + jnp.sum(long_), steps=aux["steps"] + jnp.sum(alive),
                   stuck=aux["stuck"] + jnp.sum(stuck),
                   reg_steps=aux["reg_steps"] + jnp.zeros(8, jnp.int64).at[reg].add(alive.astype(jnp.int64)),
                   len_end=aux["len_end"].at[lk].add(ended.astype(jnp.int64)),
                   len_arr=aux["len_arr"].at[lk].add(record.astype(jnp.int64)))
        return ns, aux, record

    # --------------------------------------------------------------------------------------- loop
    def build(self, bins, nphot):
        """Jitted run(state, aux, acc, n_iter) -> (state, aux, acc) over the table bins `bins`, init() and done()."""
        cfg, f = self.cfg, self.f
        bins = np.asarray(bins, np.int64)
        nbins = len(bins)
        gbins = jnp.asarray(bins)
        box_np = cfg.binning.boxes(cfg.simtype, bins)
        box = {k: (np.asarray(v[0], np.float64), np.asarray(v[1], np.float64)) for k, v in box_np.items()}
        qend = nbins * nphot
        lanes = cfg.lanes
        thresh = max(1, int(cfg.refill_fraction * lanes))
        key_src = self.key_src

        def refill(c):
            s, aux, acc = c
            need = s["reg"] == DEAD
            q = aux["q"] + jnp.cumsum(need) - 1
            take = need & (q < qend)
            qn = jnp.where(take, q, 0)
            lb = qn // nphot
            pid = gbins[lb] * nphot + qn % nphot
            new, inside, bad = self.source(lb, box, pid, key_src)
            take_ok = take & inside & ~bad
            acc = dict(acc, gen=acc["gen"].at[lb].add((take & inside).astype(jnp.int32)))
            ns = dict(s)
            for kk in ("p", "d", "e"):
                ns[kk] = jnp.where(take_ok[:, None], new[kk].astype(f), s[kk])
            for kk in ("E", "t"):
                ns[kk] = jnp.where(take_ok, new[kk].astype(f), s[kk])
            ns["reg"] = jnp.where(take_ok, new["reg"], s["reg"]).astype(jnp.int32)
            ns["fib"] = jnp.where(take_ok, new["fib"], s["fib"]).astype(jnp.int32)
            ns["surf"] = jnp.where(take_ok, S_NONE, s["surf"]).astype(jnp.int32)
            ns["nem"] = jnp.where(take_ok, 0, s["nem"])
            ns["nstep"] = jnp.where(take_ok, 0, s["nstep"])
            ns["pid"] = jnp.where(take_ok, pid, s["pid"])
            ns["lbin"] = jnp.where(take_ok, lb, s["lbin"])
            aux = dict(aux, q=jnp.minimum(aux["q"] + jnp.sum(need), qend), badsrc=aux["badsrc"] + jnp.sum(take & bad))
            return ns, aux, acc

        def body(c):
            it, s, aux, acc = c
            ndead = jnp.sum(s["reg"] == DEAD)
            s, aux, acc = lax.cond(((ndead >= thresh) | (ndead == lanes)) & (aux["q"] < qend), refill,
                                   lambda c_: c_, (s, aux, acc))
            s2, aux, rec = self.step(s, aux)
            tns = jnp.maximum(jnp.floor(s2["t"] + 0.5), 0)
            tns = jnp.where(rec, jnp.minimum(tns, NT), NT).astype(jnp.int32)
            nem = jnp.where(rec, s2["nem"], NE)
            t_in, e_in = rec & (tns < NT), rec & (nem < NE)
            ok = t_in & e_in

            def hist(a):
                lb = jnp.where(rec, s["lbin"], 0)
                tc, ec = jnp.clip(tns, 0, NT - 1), jnp.clip(nem, 0, NE - 1)
                return dict(a, nall=a["nall"].at[lb].add(rec.astype(jnp.int32)),
                            hta=a["hta"].at[lb, tc].add(t_in.astype(jnp.int32)),
                            hea=a["hea"].at[lb, ec].add(e_in.astype(jnp.int32)),
                            ht=a["ht"].at[lb, tc].add(ok.astype(jnp.int32)),
                            he=a["he"].at[lb, ec].add(ok.astype(jnp.int32)))

            acc = lax.cond(jnp.any(rec), hist, lambda a: a, acc)
            return it + 1, s2, aux, acc

        @jax.jit
        def run(s, aux, acc, n_iter):
            def cnd(c):
                it, s_, aux_, _ = c
                return (it < n_iter) & ((aux_["q"] < qend) | jnp.any(s_["reg"] != DEAD))
            _, s, aux, acc = lax.while_loop(cnd, body, (jnp.int64(0), s, aux, acc))
            return s, aux, acc

        def init():
            z3 = jnp.zeros((lanes, 3), f)
            s = dict(p=z3, d=z3.at[:, 2].set(1), e=z3.at[:, 0].set(1), E=jnp.full(lanes, 2.5, f),
                     t=jnp.zeros(lanes, f), reg=jnp.zeros(lanes, jnp.int32), fib=jnp.zeros(lanes, jnp.int32),
                     surf=jnp.zeros(lanes, jnp.int32), nem=jnp.zeros(lanes, jnp.int32),
                     nstep=jnp.zeros(lanes, jnp.int64), pid=jnp.zeros(lanes, jnp.int64),
                     lbin=jnp.zeros(lanes, jnp.int64), lobe=jnp.ones(lanes, bool))
            aux = {k: jnp.int64(0) for k in ("q", "helix", "forced", "capped", "steps", "stuck", "badsrc")}
            aux["reg_steps"] = jnp.zeros(8, jnp.int64)
            aux["len_end"] = jnp.zeros(LEN_BINS, jnp.int64)
            aux["len_arr"] = jnp.zeros(LEN_BINS, jnp.int64)
            acc = dict(gen=jnp.zeros(nbins, jnp.int32), nall=jnp.zeros(nbins, jnp.int32),
                       ht=jnp.zeros((nbins, NT), jnp.int32), he=jnp.zeros((nbins, NE), jnp.int32),
                       hta=jnp.zeros((nbins, NT), jnp.int32), hea=jnp.zeros((nbins, NE), jnp.int32))
            return s, aux, acc

        def done(s, aux):
            return bool(aux["q"] >= qend) and not bool(jnp.any(s["reg"] != DEAD))

        return run, init, done
