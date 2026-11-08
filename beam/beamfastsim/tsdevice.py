"""Opt-in device (JAX) TS transport, joined to the DS transport and stopping in one process.

TS entrance states (x, p, t, pdg) -> TS transport -> DS entrance plane -> DS transport -> stopping-target stops,
with every stage on the device and the per-lane state resident between stages. Nothing imports this module from
the default chains; the NumPy route (`tstransport`) is unchanged.

The TS stage follows `tstransport` (same geometry tables, field bank, material caps, Urban loss, cumulative
Highland scattering, pion decay, fates) with these device simplifications, all inside one step:
  * the material-cap step shortening runs while any lane is over a cap, at most SHRINK_ROUNDS rounds (halving
    after 5, as tstransport); a lane still over a cap ends as `unconverged`, counted, never silently accepted;
  * at most MATERIAL_SLOTS materials (the largest columns of the step) are sampled per step;
  * the DS-plane crossing fraction comes from Newton iterations on the Hermite arc, then one RK4 step of that
    length snapped onto the plane, not bisection;
  * wall exits are detected on the quarter-step points of the cubic Hermite arc through the step's endpoints and
    directions, not extra RK4 probes; a lane that leaves the envelopes ends as `wall`, or `upstream` when the
    first outside point is upstream of the TS1 entrance plane on the production-solenoid side;
  * random numbers come from jax.random, so a seed is not the NumPy route's seed: compare statistically.
The DS stage is `transport.make_stepper` unchanged; DS-entrance muons keep their remaining proper lifetime.
Charged pions reaching the DS plane are counted, not transported (the DS stepper is muon-only).

CLI (one process; three passes with the shared warm-rate timer when --bench is given):
  python3 -m beamfastsim.tsdevice --reference reference.npz --gdml ts.gdml --field-config CFG \
      --offline-root ROOT --maps MAPS --ds-geom Offline/Mu2eG4/geom/geom_run1_a.txt --ds-map DSMap.header \
      --out chain.npz [--arm fast|stock-ds|saved-ts --saved fast.npz] [--pool LANES]
Several --reference files stream through one resident pass (stream_chain): the lane pool refills from the next
file, so compilation and the slow-lane tail are paid once per job.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import ehelix, eloss, eloss_jax

from . import geometry as dsgeometry, simpleconfig, transport as dst, tsphysics as phys
from .tstransport import FATES as TS_FATES

jax.config.update("jax_enable_x64", True)

FATES = TS_FATES + ("unconverged",)
LIVE, DS, RANGEOUT, MU_DECAY, WALL, UPSTREAM, PATH_LIMIT, FIELD_MISSING, UNCONVERGED = range(len(FATES))
SHRINK_ROUNDS = 64     # material-cap shortening rounds per step at most (tstransport: halving after 5)
MATERIAL_SLOTS = int(os.environ.get("TSDEV_MATERIAL_SLOTS", 2))  # materials sampled per step (largest first)
# Compile-size / launch-count trade-offs (no physics effect): TS steps per device loop iteration, unrolled loss
# sampling loops, smallest lane bucket after the inputs are exhausted (= pool: one compiled shape).
STEPS_PER_ITERATION = int(os.environ.get("TSDEV_STEPS_PER_ITERATION", 1))
UNROLL_LOSS = os.environ.get("TSDEV_UNROLL_LOSS", "1") == "1"
MIN_BUCKET = int(os.environ.get("TSDEV_MIN_BUCKET", 1024))
# Material-cap shortening cascade: per level, the lanes still over a cap gathered into at most this fraction of the
# batch (every lane as the fallback); the last level runs the remaining rounds. "" runs every round on every lane.
SHRINK_MARGIN_F32 = 1e-2
CROSS_FRACTION = float(os.environ.get("TSDEV_CROSS_FRACTION", 1 / 64))   # DS-plane lanes re-measured per step
SHRINK_LEVELS = tuple(float(v) for v in os.environ.get("TSDEV_SHRINK_LEVELS", "0.5,0.25,0.015625").split(",") if v)
PLANE_TOL_F32 = 1e-2   # mm; float32 rounding of Mu2e coordinates near z = 4179 mm is ~5e-4 mm


def set_dtype(dtype):
    """float64 (default) or float32 for every device array of the chain, TS and DS stages alike. float32 switches
    JAX x64 off for the process, so call it before building device tables or arrays (load_chain does)."""
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    jax.config.update("jax_enable_x64", dtype == "float64")
_INF = jnp.inf


# ---------------------------------------------------------------- geometry (tsgeometry.Geometry -> jnp)

def _ratio(num, den):
    ok = jnp.abs(den) > 1e-15
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), _INF)


def _mm(a, b):
    """a @ b with a contracted 3-vector axis, written elementwise so that XLA fuses it with its neighbours (a
    (N,3)@(3,3) matmul is a separate library call per use, which dominates the per-step latency)."""
    if np.ndim(b) == 1:
        return a[..., 0] * b[0] + a[..., 1] * b[1] + a[..., 2] * b[2]
    return a[..., 0, None] * b[0] + a[..., 1, None] * b[1] + a[..., 2, None] * b[2]


def _contains(s, p):
    """tsgeometry._Solid.contains on (K,3) jnp points."""
    if s.tag in ("subtraction", "intersection", "union"):
        a, b = _contains(s.a, p), _contains(s.b, _mm(p - s.t, s.R))
        return a & ~b if s.tag == "subtraction" else (a & b if s.tag == "intersection" else a | b)
    if s.tag in ("box", "xtru"):
        hit = jnp.zeros(p.shape[0], bool)
        for q in s.planes:
            hit = hit | jnp.all(_mm(p, q[:, :3].T) <= q[:, 3] + 1e-10, axis=1)
        return hit
    radial = jnp.hypot(p[:, 0], p[:, 1])
    r = jnp.hypot(radial - s.extent, p[:, 2]) if s.tag == "torus" else radial
    ok = ((r >= s.rmin - 1e-10) & (r <= s.rmax + 1e-10) if s.vacuum else (r >= s.rmin) & (r < s.rmax))
    if s.tag == "tube":
        ok = ok & (jnp.abs(p[:, 2]) <= s.extent + 1e-10)
    if s.width < 2 * np.pi - 1e-12:
        phi = jnp.mod(jnp.arctan2(p[:, 1], p[:, 0]) - s.phi, 2 * np.pi)
        ok = ok & ((phi <= s.width + 1e-12) | (phi >= 2 * np.pi - 1e-12))
    return ok


def _cuts(s, o, d):
    """tsgeometry._Solid.cuts: chord parameters of every surface root, (N, k)."""
    if s.tag in ("subtraction", "intersection", "union"):
        return jnp.concatenate([_cuts(s.a, o, d), _cuts(s.b, _mm(o - s.t, s.R), _mm(d, s.R))], axis=1)
    if s.tag in ("box", "xtru"):
        q = np.concatenate(s.planes)
        return _ratio(q[:, 3] - _mm(o, q[:, :3].T), _mm(d, q[:, :3].T))
    cuts = [_ratio(-s.extent - o[:, 2], d[:, 2]), _ratio(s.extent - o[:, 2], d[:, 2])]
    a = jnp.sum(d[:, :2] ** 2, axis=1)
    b = jnp.sum(o[:, :2] * d[:, :2], axis=1)
    for r in (s.rmin, s.rmax):
        disc = b * b - a * (jnp.sum(o[:, :2] ** 2, axis=1) - r * r)
        root = jnp.sqrt(jnp.maximum(disc, 0))
        cuts += [jnp.where(disc >= 0, _ratio(-b + sg * root, a), _INF) for sg in (-1, 1)]
    if s.width < 2 * np.pi - 1e-12:
        for phi in (s.phi, s.phi + s.width):
            n = np.array([-np.sin(phi), np.cos(phi), 0.0])
            cuts.append(_ratio(-_mm(o, n), _mm(d, n)))
    return jnp.stack(cuts, axis=1)


class DeviceGeometry:
    """Device twin of a tsgeometry.Geometry: inside(x) and columns(x0, x1) -> (N, M) mm per material."""

    def __init__(self, geometry):
        self.names = list(geometry.materials)
        self.materials = [geometry.materials[k] for k in self.names]
        self.x0 = np.array([m.x0_mm if hasattr(m, "x0_mm") else eloss.radiation_length_mm(m)
                            for m in self.materials])
        self.consts = [eloss_jax.consts(m) for m in self.materials]
        self._vacuum = list(geometry._vacuum)
        matter = [(s, R, t, self.names.index(key), *s.bounds()) for s, R, t, key in geometry._matter]
        plain = [m for m in matter if m[0].tag == "tube" and m[0].width >= 2 * np.pi - 1e-12]
        self._matter = [m for m in matter if not (m[0].tag == "tube" and m[0].width >= 2 * np.pi - 1e-12)]
        # plain full-phi tubes: one vectorised analytic interval computation (no sort)
        self._tubes = None
        if plain:
            R = np.stack([m[1] for m in plain])
            t = np.stack([m[2] for m in plain])
            self._tubes = dict(R=R, tR=np.einsum("sj,sjk->sk", t, R), material=np.array([m[3] for m in plain]),
                               rmin=jnp.asarray([m[0].rmin for m in plain]), rmax=jnp.asarray([m[0].rmax for m in plain]),
                               ez=jnp.asarray([m[0].extent for m in plain]),
                               )
        self.z_start, self.z_end = float(geometry.z_start), float(geometry.z_end)

    def inside(self, x):
        hit = jnp.zeros(x.shape[0], bool)
        for s, R, t in self._vacuum:
            hit = hit | _contains(s, _mm(x - t, R))
        return hit

    def _tube_columns(self, x0, delta, length):
        """Straight-chord lengths in every plain tube at once: |[0,1] & z-slab & r<=rmax| - |... & r<rmin|."""
        T = self._tubes
        R = T["R"]                                      # (S, 3, 3) numpy constants
        o = sum(x0[:, None, j, None] * R[None, :, j, :] for j in range(3)) - T["tR"][None]
        d = sum(delta[:, None, j, None] * R[None, :, j, :] for j in range(3))
        big = 1e300
        dz = d[..., 2]
        zs = jnp.abs(dz) > 1e-15
        za = jnp.where(zs, (-T["ez"] - o[..., 2]) / jnp.where(zs, dz, 1.0), -big)
        zb = jnp.where(zs, (T["ez"] - o[..., 2]) / jnp.where(zs, dz, 1.0), big)
        inz = jnp.abs(o[..., 2]) <= T["ez"]
        lo = jnp.where(zs, jnp.minimum(za, zb), jnp.where(inz, -big, big))
        hi = jnp.where(zs, jnp.maximum(za, zb), jnp.where(inz, big, -big))
        lo, hi = jnp.maximum(lo, 0.0), jnp.minimum(hi, 1.0)
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = o[..., 0] * d[..., 0] + o[..., 1] * d[..., 1]
        c0 = o[..., 0] ** 2 + o[..., 1] ** 2

        def disk(r):
            disc = b * b - a * (c0 - r * r)
            moving = a > 1e-30
            root = jnp.sqrt(jnp.maximum(disc, 0.0))
            s0 = jnp.where(moving, (-b - root) / jnp.where(moving, a, 1.0), jnp.where(c0 <= r * r, -big, big))
            s1 = jnp.where(moving, (-b + root) / jnp.where(moving, a, 1.0), jnp.where(c0 <= r * r, big, -big))
            empty = moving & (disc < 0)
            return jnp.where(empty, big, s0), jnp.where(empty, -big, s1)

        o0, o1 = disk(T["rmax"])
        i0, i1 = disk(T["rmin"])
        outer = jnp.clip(jnp.minimum(hi, o1) - jnp.maximum(lo, o0), 0.0, None)
        inner = jnp.clip(jnp.minimum(jnp.minimum(hi, o1), i1) - jnp.maximum(jnp.maximum(lo, o0), i0), 0.0, None)
        inner = jnp.where(T["rmin"] > 0, inner, 0.0)
        per = length[:, None] * (outer - inner)          # (N, S): one material per tube
        return jnp.stack([jnp.sum(jnp.where(T["material"][None] == m, per, 0.0), axis=1)
                          for m in range(len(self.names))], axis=1)

    @staticmethod
    def _generic_add(s, o, d, length):
        """Straight-chord length inside one generic solid (local frame). Every surface root splits the chord; each
        distinct root opens an interval that ends at the next larger root, and the interval's midpoint decides
        whether it is inside. No sort: the next root is a fused min over all roots."""
        n = o.shape[0]
        c = jnp.clip(jnp.concatenate([jnp.zeros((n, 1)), _cuts(s, o, d), jnp.ones((n, 1))], axis=1), 0, 1)
        k = c.shape[1]
        ci, cj = c[:, :, None], c[:, None, :]
        nxt = jnp.min(jnp.where(cj > ci, cj, 2.0), axis=2)
        earlier = np.arange(k)[None, :] < np.arange(k)[:, None]          # [i, j]: j < i
        first = ~jnp.any((cj == ci) & earlier[None], axis=2)
        valid = first & (nxt <= 1.0)
        mid = jnp.where(valid, 0.5 * (c + nxt), 0.0)
        p = o[:, None, :] + mid[:, :, None] * d[:, None, :]
        mask = _contains(s, p.reshape(-1, 3)).reshape(mid.shape)
        return length * jnp.sum(jnp.where(valid & mask, nxt - c, 0.0), axis=1)

    def columns(self, x0, x1):
        """(N, M) straight-chord material lengths: analytic for the plain tubes, sort-free intervals otherwise."""
        return self.columns_d(x0, x1 - x0)

    def columns_d(self, x0, delta):
        """columns() of the chord from x0 along the increment delta (exact in float32, where x1 - x0 is not)."""
        length = jnp.linalg.norm(delta, axis=1)
        out = jnp.zeros((x0.shape[0], len(self.names)))
        if self._tubes is not None:
            out = out + self._tube_columns(x0, delta, length)
        for s, R, t, m, lo, hi in self._matter:
            o, d = _mm(x0 - t, R), _mm(delta, R)
            cand = jnp.all((jnp.maximum(o, o + d) >= lo - 1e-9) & (jnp.minimum(o, o + d) <= hi + 1e-9), axis=1)
            out = out.at[:, m].add(jnp.where(cand, self._generic_add(s, o, d, length), 0.0))
        return out


# ---------------------------------------------------------------- field bank (tsfield.FieldBank -> jnp)

class DeviceField:
    """First-containing-map field bank on the device: one gather per corner from the concatenated grids."""

    def __init__(self, maps, scale=1.0):
        """maps: list of (lo, spacing, n, grid[nx,ny,nz,3], reflect_y)."""
        if not maps:
            raise ValueError("field bank needs at least one map")
        self.scale = float(scale)
        lo, sp, n, hi, base, refl, flats = [], [], [], [], [], [], []
        offset = 0
        for l, d, nn, g, r in maps:
            l, d, nn = np.asarray(l, float), np.asarray(d, float), np.asarray(nn, int)
            if r and l[1] != 0:
                raise ValueError("Y-reflected map must start at y=0")
            lo.append(l), sp.append(d), n.append(nn), hi.append(l + d * (nn - 1)), refl.append(bool(r))
            base.append(offset)
            flats.append(np.asarray(g, float).reshape(-1, 3))
            offset += flats[-1].shape[0]
        self.lo, self.sp, self.hi = (jnp.asarray(np.array(a)) for a in (lo, sp, hi))
        self.n = jnp.asarray(np.array(n))
        self.gmax = jnp.asarray(np.nextafter((np.array(n) - 1).astype(float), 0))
        self.base = jnp.asarray(np.array(base))
        self.reflect = jnp.asarray(np.array(refl))
        self.grid = jnp.asarray(np.concatenate(flats))
        self.max_field_t = float(abs(self.scale) * max(np.linalg.norm(np.max(np.abs(f), axis=0)) for f in flats))

    @classmethod
    def from_config(cls, config, offline_root, maps_dir):
        params = simpleconfig.read(str(config), str(offline_root))
        maps = []
        for name in params["bfield.innerMaps"] + params["bfield.outerMaps"]:
            header = Path(maps_dir) / Path(name).name
            lo, d, n, grid = ehelix.read_g4bl(str(header))
            maps.append((lo, d, n, grid, "extendY flip=By" in header.read_text()))
        return cls(maps, params["bfield.scaleFactor"])

    def evaluate(self, x):
        """B (N,3) [T] and covered (N,) with fixed first-containing precedence and the By flip below y=0."""
        pos = jnp.where(self.reflect[None, :, None],
                        jnp.stack([x[:, 0], jnp.abs(x[:, 1]), x[:, 2]], 1)[:, None, :], x[:, None, :])
        inmap = jnp.all((pos >= self.lo) & (pos <= self.hi), axis=2)           # (N, maps)
        covered = jnp.any(inmap, axis=1)
        k = jnp.argmax(inmap, axis=1)
        u = jnp.clip((pos[jnp.arange(x.shape[0]), k] - self.lo[k]) / self.sp[k], 0, self.gmax[k])
        i = jnp.floor(u).astype(jnp.int64)
        f = u - i
        n = self.n[k]
        b = jnp.zeros_like(x)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                         * (f[:, 2] if dz else 1 - f[:, 2]))
                    j = ((i[:, 0] + dx) * n[:, 1] + i[:, 1] + dy) * n[:, 2] + i[:, 2] + dz
                    b = b + w[:, None] * self.grid[self.base[k] + j]
        flip = self.reflect[k] & (x[:, 1] < 0)
        b = b.at[:, 1].multiply(jnp.where(flip, -1.0, 1.0))
        return self.scale * jnp.where(covered[:, None], b, 0.0), covered


# ---------------------------------------------------------------- TS stepper

class TSState(NamedTuple):
    x: object
    p: object
    t: object
    pdg: object
    fate: object
    path: object
    remaining: object
    decayed: object
    steps: object
    hist: object        # (N, M) cumulative x/X0 per material
    mpath: object
    waits: object       # steps the lane was deferred (a gather bucket was full); its physics is unchanged


class Settings(NamedTuple):
    step_mm: float = 2.5
    max_path_mm: float = 100000.0
    max_steps: int = 20000     # step cap, both dtypes (fate path_limit): bounds the slow-lane tail; on the Run 1A
                               # held-out cohort DS-reaching lanes need at most 13,480 steps
    material_step_mm: float = 0.25
    loss_fraction: float = 0.05


def _fdt():
    """The device float dtype: float64 with JAX x64 on, float32 with it off (set_dtype)."""
    return jnp.zeros(()).dtype


def _mass(pdg):
    return jnp.where(jnp.abs(pdg) == 211, phys.MPI, phys.MMU)


def _charge(pdg):
    return jnp.where(jnp.abs(pdg) == 211, jnp.sign(pdg), -jnp.sign(pdg)).astype(_fdt())


def _lifetime(pdg):
    return jnp.where(jnp.abs(pdg) == 211, phys.TAU_PI_NS, phys.TAU_MU_NS)


def _rk4(field, x, u, kq, h, b0, relative=False):
    """One RK4 step; returns (end point, or the increment when relative, unit direction, field coverage)."""
    h = h[:, None]

    def du(xx, uu):
        b, ok = field.evaluate(xx)
        return kq[:, None] * jnp.cross(uu, b), ok

    k1u = kq[:, None] * jnp.cross(u, b0)
    k2x = u + 0.5 * h * k1u
    k2u, ok2 = du(x + 0.5 * h * u, u + 0.5 * h * k1u)
    k3x = u + 0.5 * h * k2u
    k3u, ok3 = du(x + 0.5 * h * k2x, u + 0.5 * h * k2u)
    k4x = u + h * k3u
    k4u, ok4 = du(x + h * k3x, u + h * k3u)
    inc = h / 6 * (u + 2 * k2x + 2 * k3x + k4x)
    un = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
    return (inc if relative else x + inc), un / jnp.linalg.norm(un, axis=1, keepdims=True), ok2 & ok3 & ok4


def _hermite(x0, u0, x1, u1, h, f):
    f2, f3 = f * f, f * f * f
    return ((2 * f3 - 3 * f2 + 1) * x0 + (f3 - 2 * f2 + f) * h[:, None] * u0
            + (-2 * f3 + 3 * f2) * x1 + (f3 - f2) * h[:, None] * u1)


def _highland_var(p, x_prev, x_new, mass):
    p = jnp.maximum(p, 1e-3)
    beta2 = p * p / (p * p + mass * mass)

    def g(x):
        x = jnp.maximum(x, 0.0)
        return jnp.where(x > 0, x * (1 + 0.038 * jnp.log(jnp.maximum(x, 1e-30) / beta2)) ** 2, 0.0)

    return (13.6 ** 2 / (beta2 * p * p)) * jnp.maximum(g(x_new) - g(x_prev), 0.0)


def _first_round(field, geom, x0, u, kq, b0, live, h, xn, un, ok, cols, sc, it=0, margin=1e-6, relative=False):
    """One round of the material-cap shortening (the shrink loop body at round `it`). relative: xn holds the
    increment from x0 (float32)."""
    changed = live & (sc < 1)
    shrink = sc * (1 - margin)
    h = jnp.where(changed, h * jnp.where(it >= 5, jnp.minimum(0.5, shrink), shrink), h)
    xn2, un2, ok2 = _rk4(field, x0, u, kq, h, b0, relative)
    pick = changed[:, None]
    c2 = geom.columns_d(x0, xn2) if relative else geom.columns(x0, xn2)
    return (h, jnp.where(pick, xn2, xn), jnp.where(pick, un2, un), jnp.where(changed, ok2, ok),
            jnp.where(pick, c2, cols), None)


def make_ts_step(field, geom, s):
    """One TS step for every lane (finished lanes are inert): (key, TSState) -> TSState."""
    # float32 (x64 off): DS-plane endpoints are tested this far upstream of the closed envelope face; 0 in float64
    plane_tol = 0.0 if jax.config.jax_enable_x64 else PLANE_TOL_F32
    step_cap = s.max_steps
    # step shortening margin below a cap: 1e-6 in float64 (tstransport); float32 chords near |x| ~ 4 m carry ~5e-4 mm
    # of rounding, so a 1e-6 margin would leave lanes just over the cap round after round
    margin = 1e-6 if jax.config.jax_enable_x64 else SHRINK_MARGIN_F32
    # float32 carries the step end as the increment from the start (positions near |x| ~ 4 m resolve only ~5e-4 mm,
    # so x1 - x0 of a short step is mostly rounding); float64 keeps the end point, as tstransport
    rel = not jax.config.jax_enable_x64

    def cols_of(x0, e):
        return geom.columns_d(x0, e) if rel else geom.columns(x0, e)
    M = len(geom.names)
    x0_mm = jnp.asarray(geom.x0)
    const_tab = [jnp.asarray(np.array([getattr(c, f) for c in geom.consts], float))
                 for f in eloss_jax.MatConsts._fields]

    def dedx_all(pm, mass):
        return jnp.stack([eloss_jax._dedx(c, pm, 1e9, mass) for c in geom.consts], axis=1) if M else \
            jnp.zeros((pm.shape[0], 0))

    def step(key, st):
        k_mat, k_dec = jax.random.split(key)
        live = st.fate == LIVE
        mass = _mass(st.pdg)
        pm = jnp.linalg.norm(st.p, axis=1)
        pms = jnp.maximum(pm, 1e-12)
        kin = pm ** 2 / (jnp.hypot(pm, mass) + mass)
        fate = jnp.where(live & (kin <= phys.TK_STOP_MEV), RANGEOUT, st.fate)
        live = fate == LIVE
        u = st.p / pms[:, None]
        kq = ehelix.K * _charge(st.pdg) / pms
        decay_path = st.remaining * pm * phys.C_MM_NS / mass
        b0, valid = field.evaluate(st.x)
        fate = jnp.where(live & ~valid, FIELD_MISSING, fate)
        live = live & valid
        h = jnp.minimum(jnp.minimum(s.step_mm, s.max_path_mm - st.path), decay_path)
        h = jnp.minimum(h, 0.1 * pm / jnp.maximum(ehelix.K * jnp.linalg.norm(b0, axis=1), 1e-30))
        h = jnp.where(live, h, 0.0)
        dedx = dedx_all(pms, mass)

        def scale_of(cols):
            mean = jnp.sum(dedx * cols, axis=1)
            length = jnp.sum(cols, axis=1)
            sc = jnp.minimum(1.0, s.material_step_mm / jnp.maximum(length, 1e-30))
            return jnp.minimum(sc, s.loss_fraction * kin / jnp.maximum(mean, 1e-30))

        # material caps: shorten the step until no lane is over a cap; lanes still over a cap after SHRINK_ROUNDS end
        # as `unconverged` and are counted
        xn, un, ok = _rk4(field, st.x, u, kq, h, b0, rel)        # increment when rel (float32)
        cols = cols_of(st.x, xn)

        def scale_on(cols, dedx_, kin_):
            mean = jnp.sum(dedx_ * cols, axis=1)
            length = jnp.sum(cols, axis=1)
            sc = jnp.minimum(1.0, s.material_step_mm / jnp.maximum(length, 1e-30))
            return jnp.minimum(sc, s.loss_fraction * kin_ / jnp.maximum(mean, 1e-30))

        def shrink_loop(x0, u_, kq_, b0_, live_, dedx_, kin_, h_, xn_, un_, ok_, cols_, it0):
            """Shorten over-cap steps of the given lanes until none is over a cap (at most SHRINK_ROUNDS rounds,
            halving from round 5 on, as tstransport)."""
            def cond(c):
                h_, xn_, un_, ok_, cols_, it = c
                return (it < SHRINK_ROUNDS) & jnp.any(live_ & (scale_on(cols_, dedx_, kin_) < 1))

            def body(c):
                h_, xn_, un_, ok_, cols_, it = c
                sc = scale_on(cols_, dedx_, kin_)
                changed = live_ & (sc < 1)
                shrink = sc * (1 - margin)
                h_ = jnp.where(changed, h_ * jnp.where(it >= 5, jnp.minimum(0.5, shrink), shrink), h_)
                xn2, un2, ok2 = _rk4(field, x0, u_, kq_, h_, b0_, rel)
                pick = changed[:, None]
                return (h_, jnp.where(pick, xn2, xn_), jnp.where(pick, un2, un_), jnp.where(changed, ok2, ok_),
                        jnp.where(pick, cols_of(x0, xn2), cols_), it + 1)

            return lax.while_loop(cond, body, (h_, xn_, un_, ok_, cols_, it0))

        # Shortening cascade: each level gathers the lanes still over a cap into a bucket of at most a fixed
        # fraction of the batch (falling back to every lane when more are pending); the first levels run one round
        # each, the last runs the remaining rounds. Per-lane arithmetic is that of the every-lane loop.
        n_lanes = h.shape[0]

        def gathered(args, pending, frac, it0, rounds):
            """Run the round(s) on the pending lanes, gathered into a bucket of at most frac of the batch (no
            branch: a lane beyond the bucket is not processed and is reported in the returned mask)."""
            k2 = min(n_lanes, max(256, int(np.ceil(n_lanes * frac))))

            def run(x0, u_, kq_, b0_, live_, dedx_, kin_, h_, xn_, un_, ok_, cols_):
                if rounds == 1:
                    return _first_round(field, geom, x0, u_, kq_, b0_, live_, h_, xn_, un_, ok_, cols_,
                                        jnp.where(live_, scale_on(cols_, dedx_, kin_), 1.0), it0, margin, rel)[:5]
                return shrink_loop(x0, u_, kq_, b0_, live_, dedx_, kin_, h_, xn_, un_, ok_, cols_, it0)[:5]

            if k2 >= n_lanes:
                return run(st.x, u, kq, b0, pending, dedx, kin, *args), pending
            idx = jnp.nonzero(pending, size=k2, fill_value=n_lanes)[0]
            j = jnp.minimum(idx, n_lanes - 1)
            out = run(st.x[j], u[j], kq[j], b0[j], pending[j] & (idx < n_lanes), dedx[j], kin[j], *(a[j] for a in args))
            done = jnp.zeros(n_lanes, bool).at[idx].set(True, mode="drop")
            return tuple(a.at[idx].set(v, mode="drop") for a, v in zip(args, out)), done

        state = (h, xn, un, ok, cols)
        levels = SHRINK_LEVELS if SHRINK_LEVELS else (1.0,)
        deferred = jnp.zeros(n_lanes, bool)
        for level, frac in enumerate(levels):
            last = level == len(levels) - 1
            pending = live & ~deferred & (scale_on(state[4], dedx, kin) < 1)
            state, done = gathered(state, pending, frac, jnp.int32(level), SHRINK_ROUNDS - level if last else 1)
            deferred = deferred | (pending & ~done)
        h, xn, un, ok, cols = state
        if rel:                                   # float32: keep the increment e, and the end point xn
            e, xn = xn, st.x + xn
        unconverged = live & ~deferred & (scale_of(cols) < 1)

        # DS plane (z_end on the DS side, x < 0): Newton on the Hermite arc for the step fraction, then one RK4 step
        # of that length, snapped onto the plane
        cross = live & ~deferred & (st.x[:, 2] < geom.z_end) & (xn[:, 2] >= geom.z_end) & (xn[:, 0] < 0)
        z0, z1, hz0, hz1 = st.x[:, 2], xn[:, 2], h * u[:, 2], h * un[:, 2]
        fr = jnp.clip((geom.z_end - z0) / jnp.where(cross, z1 - z0, 1.0), 0.0, 1.0)
        for _ in range(3):
            f2, f3 = fr * fr, fr * fr * fr
            z = (2 * f3 - 3 * f2 + 1) * z0 + (f3 - 2 * f2 + fr) * hz0 + (-2 * f3 + 3 * f2) * z1 + (f3 - f2) * hz1
            dz = (6 * f2 - 6 * fr) * z0 + (3 * f2 - 4 * fr + 1) * hz0 + (6 * fr - 6 * f2) * z1 + (3 * f2 - 2 * fr) * hz1
            fr = jnp.clip(fr + (geom.z_end - z) / jnp.where(jnp.abs(dz) > 1e-12, dz, 1e-12), 0.0, 1.0)
        hc = jnp.where(cross, h * fr, h)
        xc, uc, okc = _rk4(field, st.x, u, kq, hc, b0, rel)
        pick = cross[:, None]
        if rel:
            ec = xc.at[:, 2].set(geom.z_end - st.x[:, 2])
            e = jnp.where(pick, ec, e)
            xc = st.x + xc
        xn = jnp.where(pick, xc.at[:, 2].set(geom.z_end), xn)
        un = jnp.where(pick, uc, un)
        ok = jnp.where(cross, okc, ok)
        h = hc
        kc = min(n_lanes, max(256, int(np.ceil(n_lanes * CROSS_FRACTION))))
        if kc >= n_lanes:
            cols = jnp.where(pick, cols_of(st.x, e if rel else xn), cols)
        else:                         # DS-plane lanes beyond the bucket wait a step (deferred)
            idx = jnp.nonzero(cross, size=kc, fill_value=n_lanes)[0]
            j = jnp.minimum(idx, n_lanes - 1)
            cols = cols.at[idx].set(cols_of(st.x[j], (e if rel else xn)[j]), mode="drop")
            deferred = deferred | (cross & ~jnp.zeros(n_lanes, bool).at[idx].set(True, mode="drop"))
        fate = jnp.where(live & ~deferred & ~ok, FIELD_MISSING, fate)
        fate = jnp.where(live & ok & unconverged, UNCONVERGED, fate)
        live = (fate == LIVE) & ~deferred          # deferred lanes keep their state and retry next step

        # wall exits on the Hermite arc
        first_out = jnp.full(h.shape, jnp.nan)
        out_pt = jnp.zeros_like(xn)
        for f in (0.25, 0.5, 0.75, 1.0):
            pt = xn.at[:, 2].add(-jnp.where(cross, plane_tol, 0.0)) if f == 1.0 else _hermite(st.x, u, xn, un, h, f)
            bad = ~geom.inside(pt) & jnp.isnan(first_out)
            first_out = jnp.where(bad, f, first_out)
            out_pt = jnp.where(bad[:, None], pt, out_pt)
        crossed_wall = live & ~jnp.isnan(first_out)
        upstream = crossed_wall & (out_pt[:, 2] < geom.z_start) & (out_pt[:, 0] > 0)
        moving = live & ~crossed_wall

        # material: the two largest columns of the step, largest first (a capped chord rarely spans more), each a
        # frozen-rate Urban loss and a cumulative Highland kick with per-lane material constants
        newp = un * pm[:, None]
        travelled = jnp.ones_like(h)
        stopped = jnp.zeros_like(live)
        hist, mpath = st.hist, st.mpath
        lanes = jnp.arange(h.shape[0])
        order = jnp.argsort(-cols, axis=1) if M else None
        mkeys = jax.random.split(k_mat, 2 * MATERIAL_SLOTS)
        for slot in range(min(M, MATERIAL_SLOTS)):
            m = order[:, slot]
            mm = cols[lanes, m]
            active = moving & (mm > 0) & ~stopped
            c = eloss_jax.MatConsts(*(tab[m] for tab in const_tab))
            pa = jnp.linalg.norm(newp, axis=1)
            pas = jnp.maximum(pa, 1e-12)
            tk = pa * (pa / (jnp.hypot(pa, mass) + mass))
            cold = tk <= phys.TK_STOP_MEV
            length = jnp.where(active, mm, 1.0)
            mean = eloss_jax._dedx(c, pas, 1e9, mass) * length
            loss = eloss_jax._sample(mkeys[2 * slot], c, pas, length, 1e9, mass, unroll=UNROLL_LOSS)
            loss = jnp.maximum(jnp.where(loss >= tk, mean, loss), 0.0)
            rem = jnp.maximum(tk - loss, 0.0)
            stop = cold | (rem <= phys.TK_STOP_MEV)
            frac = jnp.where(cold, 0.0, jnp.where(stop & (loss > 0), (tk - phys.TK_STOP_MEV) / jnp.where(
                loss > 0, loss, 1.0), 1.0))
            frac = jnp.clip(frac, 0.0, 1.0)
            newpm = jnp.sqrt(rem * (rem + 2 * mass))
            d = newp / pas[:, None]
            x0m = x0_mm[m]
            prev = hist[lanes, m]
            var = _highland_var(pas, prev, prev + length / x0m, mass)
            kick = jax.random.normal(mkeys[2 * slot + 1], d.shape, _fdt())
            kick = kick - jnp.sum(kick * d, axis=1)[:, None] * d
            d = d + jnp.sqrt(var)[:, None] * kick
            d = d / jnp.linalg.norm(d, axis=1)[:, None]
            after = jnp.where(stop[:, None], 0.0, d * newpm[:, None])
            newp = jnp.where(active[:, None], after, newp)
            travelled = jnp.where(active, jnp.minimum(travelled, frac), travelled)
            stopped = stopped | (active & stop)
            hist = hist.at[lanes, m].add(jnp.where(active, mm * frac / x0m, 0.0))
            mpath = mpath + jnp.where(active, mm * frac, 0.0)

        actual = h * travelled
        go = live
        x = jnp.where(go[:, None], st.x + ((e if rel else xn - st.x) * travelled[:, None]), st.x)
        p = jnp.where(go[:, None], newp, st.p)
        path = st.path + jnp.where(go, actual, 0.0)
        steps = st.steps + go.astype(jnp.int32)
        t = st.t + jnp.where(go, actual * jnp.hypot(pm, mass) / (pms * phys.C_MM_NS), 0.0)
        remaining = st.remaining - jnp.where(go, actual * mass / (pms * phys.C_MM_NS), 0.0)
        fate = jnp.where(go & stopped & moving, RANGEOUT, fate)
        _, end_ok = field.evaluate(x)
        fate = jnp.where(go & moving & ~stopped & ~end_ok, FIELD_MISSING, fate)
        running = go & (fate == LIVE)
        at_decay = running & (remaining <= 1e-9)
        mu = at_decay & (jnp.abs(st.pdg) == 13)
        fate = jnp.where(mu, MU_DECAY, fate)
        pion = at_decay & ~mu
        # pi -> mu nu, isotropic in the pion rest frame (tsphysics.decay_pions)
        kc, kp, kl = jax.random.split(k_dec, 3)
        costh = jax.random.uniform(kc, pm.shape, _fdt(), -1.0, 1.0)
        phi = jax.random.uniform(kp, pm.shape, _fdt(), 0.0, 2 * np.pi)
        sinth = jnp.sqrt(jnp.maximum(1 - costh * costh, 0.0))
        dirn = jnp.stack([sinth * jnp.cos(phi), sinth * jnp.sin(phi), costh], 1)
        pstar = (phys.MPI - phys.MMU) * (phys.MPI + phys.MMU) / (2 * phys.MPI)
        rest = pstar * dirn
        epi = jnp.hypot(jnp.linalg.norm(p, axis=1), phys.MPI)
        proj = jnp.sum(p * rest, axis=1) / (epi + phys.MPI)
        pmu = rest + ((jnp.hypot(pstar, phys.MMU) + proj) / phys.MPI)[:, None] * p
        p = jnp.where(pion[:, None], pmu, p)
        pdg = jnp.where(pion, -13 * jnp.sign(st.pdg), st.pdg)
        remaining = jnp.where(pion, jax.random.exponential(kl, pm.shape, _fdt()) * phys.TAU_MU_NS, remaining)
        decayed = st.decayed | pion
        running = go & (fate == LIVE)
        downstream = (running & cross & ~crossed_wall & (jnp.abs(x[:, 2] - geom.z_end) < max(1e-7, plane_tol))
                      & geom.inside(x.at[:, 2].add(-plane_tol)))
        fate = jnp.where(downstream, DS, fate)
        fate = jnp.where(running & upstream, UPSTREAM, fate)
        outside = running & (crossed_wall | ~geom.inside(x)) & ~upstream & ~downstream
        fate = jnp.where(outside, WALL, fate)
        fate = jnp.where((fate == LIVE) & go & ((path >= s.max_path_mm - 1e-8) | (steps + st.waits >= step_cap)), PATH_LIMIT,
                         fate)
        waits = st.waits + (deferred & (st.fate == LIVE)).astype(jnp.int32)
        return TSState(x, p, t, pdg, fate.astype(jnp.int8), path, remaining, decayed, steps, hist, mpath, waits)

    return step


# ---------------------------------------------------------------- resident chunked driver

def _next_pow2(n, floor):
    return max(floor, 1 << int(np.ceil(np.log2(max(n, 1)))))


def _tree(f, *trees):
    return jax.tree_util.tree_map(f, *trees)


@partial(jax.jit, donate_argnums=0)
def _scatter(result, rows, cur):
    return _tree(lambda r, c: r.at[rows].set(c, mode="drop"), result, cur)


@jax.jit
def _refill(cur, rows, full, live, nl, k, nxt):
    """Live lanes first, then the next k inputs (from full) into the following slots; returns the mask of slots
    left empty (inputs exhausted)."""
    order = jnp.argsort(~live, stable=True)
    cur, rows = _tree(lambda a: a[order], cur), rows[order]
    pos = jnp.arange(rows.shape[0])
    fill = (pos >= nl) & (pos < nl + k)
    src = jnp.clip(nxt + pos - nl, 0, full[0].shape[0] - 1)
    cur = _tree(lambda c, f: jnp.where(fill.reshape((-1,) + (1,) * (c.ndim - 1)), f[src], c), cur, full)
    return cur, jnp.where(fill, src, rows), pos >= nl + k


def run_resident(run_chunk, state, live_of, kill, key, max_chunks=100000, min_bucket=1024, pool=None,
                 strict=True):
    """Run a jitted chunk function (key, state) -> state over every lane of `state`, all on the device.

    A pool of `pool` lanes (default: all) runs at a time; after each chunk finished lanes are replaced by the next
    inputs, and once the inputs are exhausted the live lanes are compacted into power-of-two buckets. Only the live
    count crosses to the host. Returns the full-size final state and the number of lane-chunks run. After max_chunks
    with live lanes: raise when strict, else return them still live (the caller labels them)."""
    n = state[0].shape[0]
    size = n if pool is None or pool >= n else int(pool)
    result = _tree(jnp.copy, state)
    rows = jnp.arange(size)
    cur = _tree(lambda a: a[:size], state)
    nxt, lane_chunks = size, 0
    progress = os.environ.get("TSDEV_PROGRESS")
    t0 = time.perf_counter()
    for it in range(max_chunks):
        cur = run_chunk(jax.random.fold_in(key, it), cur)
        lane_chunks += rows.shape[0]
        result = _scatter(result, rows, cur)
        live = live_of(cur)
        nl = int(jnp.sum(live))
        if progress:
            print(f"chunk {it} lanes {rows.shape[0]} live {nl} next {nxt}/{n} t {time.perf_counter() - t0:.2f}s",
                  flush=True)
        if nxt < n:
            k = min(rows.shape[0] - nl, n - nxt)
            cur, rows, pad = _refill(cur, rows, state, live, jnp.int64(nl), jnp.int64(k), jnp.int64(nxt))
            nxt += k
            if nxt >= n:
                cur, rows = kill(cur, pad), jnp.where(pad, n, rows)
            continue
        if nl == 0:
            return result, lane_chunks
        m = _next_pow2(nl, min_bucket)
        if m < rows.shape[0]:
            order = jnp.argsort(~live, stable=True)[:m]
            cur = _tree(lambda a: a[order], cur)
            pad = jnp.arange(m) >= nl
            rows = jnp.where(pad, n, rows[order])        # out of range: dropped on scatter
            cur = kill(cur, pad)
    if strict:
        raise RuntimeError("chunk limit reached with live lanes")
    return result, lane_chunks


def make_ts_chunk(field, geom, s, chunk=256):
    """Jitted (key, TSState) -> TSState running up to `chunk` steps (STEPS_PER_ITERATION per loop iteration)."""
    step = make_ts_step(field, geom, s)
    iterations = max(1, chunk // STEPS_PER_ITERATION)

    @jax.jit
    def run(key, st):
        def cond(c):
            i, st = c
            return (i < iterations) & jnp.any(st.fate == LIVE)

        def body(c):
            i, st = c
            for j in range(STEPS_PER_ITERATION):
                st = step(jax.random.fold_in(key, i * STEPS_PER_ITERATION + j), st)
            return i + 1, st

        return lax.while_loop(cond, body, (jnp.int32(0), st))[1]

    return run


def ts_initial(x, p, t, pdg, n_materials, key):
    x, p, t, pdg = (jnp.asarray(a) for a in (x, p, t, pdg))
    n = x.shape[0]
    fd = _fdt()
    return TSState(x.astype(fd), p.astype(fd), t.astype(fd), pdg.astype(jnp.int64 if fd == jnp.float64 else jnp.int32),
                   jnp.zeros(n, jnp.int8), jnp.zeros(n), jax.random.exponential(key, (n,), fd)
                   * _lifetime(pdg), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32), jnp.zeros((n, n_materials)),
                   jnp.zeros(n), jnp.zeros(n, jnp.int32))


def transport_ts(ts_chunk, geom, x, p, t, pdg, key, min_bucket=None, pool=None):
    k0, k1 = jax.random.split(key)
    st = ts_initial(x, p, t, pdg, len(geom.names), k0)
    return run_resident(ts_chunk, st, lambda s: s.fate == LIVE,
                        lambda s, pad: s._replace(fate=jnp.where(pad, jnp.int8(PATH_LIMIT), s.fate)),
                        k1, min_bucket=MIN_BUCKET if min_bucket is None else min_bucket, pool=pool)[0]


# ---------------------------------------------------------------- DS stage and the joined chain

DS_CHUNK = 200


def make_ds_chunk(fm, ds_geom, prm, chunk=DS_CHUNK):
    dsgeometry.validate_foils(ds_geom, prm.ds)
    core = dst._build(ds_geom, prm, chunk)
    return lambda key, st: core(fm, key, st)[0]


def ds_initial(x, p3, t, q, tau_max):
    pm = jnp.linalg.norm(p3, axis=1)
    n = x.shape[0]
    return dst.State(x, p3 / jnp.maximum(pm, 1e-12)[:, None], pm, t, jnp.zeros(n), tau_max, q,
                     jnp.zeros(n, jnp.int32), jnp.full(n, -1, jnp.int32), jnp.zeros(n, jnp.int32))


def transport_ds(ds_chunk, prm, x, p3, t, q, tau_max, enter, key, min_bucket=256):
    """DS stage on the lanes with `enter`; other lanes are inert (fate 9) and keep foil -1."""
    st = ds_initial(x, p3, t, q, tau_max)
    st = st._replace(fate=jnp.where(enter, 0, 9).astype(jnp.int32))
    out, _ = run_resident(ds_chunk, st, lambda s: s.fate == 0,
                          lambda s, pad: s._replace(fate=jnp.where(pad, 9, s.fate)), key,
                          max_chunks=int(np.ceil(prm.max_steps / DS_CHUNK)), min_bucket=min_bucket, strict=False)
    return out._replace(fate=jnp.where(out.fate == 0, 9, out.fate))


class Chain(NamedTuple):
    ts_chunk: object
    ts_geom: object
    ds_chunk: object
    ds_prm: object
    pool: object = None


def build_chain(ts_field, ts_geom, ds_fm, ds_geom, settings=Settings(), ds_prm=dst.Params(), ts_chunk=256,
                pool=None):
    """Compile-ready chain from device field/geometry objects (DeviceField, DeviceGeometry, DS FieldMap on device,
    DS BeamGeometry). pool: TS lanes in flight (default: all inputs at once)."""
    return Chain(make_ts_chunk(ts_field, ts_geom, settings, ts_chunk), ts_geom,
                 make_ds_chunk(ds_fm, ds_geom, ds_prm), ds_prm, pool)


def run_chain(chain, x, p, t, pdg, seed=20261107):
    """TS entrance states -> stops, all on the device. Returns (TSState, DS State); the DS state is per input lane
    (fate 9 and foil -1 for lanes that did not enter the DS as muons). Target stops: DS fate == 1."""
    key = jax.random.PRNGKey(seed)
    k_ts, k_ds = jax.random.split(key)
    ts = transport_ts(chain.ts_chunk, chain.ts_geom, x, p, t, pdg, k_ts, pool=chain.pool)
    enter = (ts.fate == DS) & (jnp.abs(ts.pdg) == 13)
    ds = transport_ds(chain.ds_chunk, chain.ds_prm, ts.x, ts.p, ts.t, _charge(ts.pdg),
                      jnp.where(enter, ts.remaining, _INF), enter, k_ds)
    return ts, ds


def stream_chain(chain, batches, seed=20261107):
    """Many input batches (e.g. one per input file) through one resident pass, in order. The TS lane pool
    (chain.pool) refills from the next batch as lanes finish, so compilation and the slow-lane tail are paid once
    per job, not once per batch; the DS stage then runs once over every DS-entering muon. A lane's random numbers
    depend on its slot in the pool, so a batch streamed with others is a statistically equivalent, not identical,
    sample to the same batch run alone. Returns one (TSState, DS State) pair per batch."""
    sizes = [len(b["pdg"]) for b in batches]
    if not sizes or min(sizes) == 0:
        raise ValueError("stream_chain needs non-empty batches")
    joined = {k: np.concatenate([np.asarray(b[k]) for b in batches]) for k in ("x", "p", "t", "pdg")}
    ts, ds = run_chain(chain, joined["x"], joined["p"], joined["t"], joined["pdg"], seed)
    edges = np.cumsum([0] + sizes)
    return [(jax.tree_util.tree_map(lambda a, i=i: a[edges[i]:edges[i + 1]], ts),
             jax.tree_util.tree_map(lambda a, i=i: a[edges[i]:edges[i + 1]], ds)) for i in range(len(sizes))]


def run_ds_only(chain, x, p, t, pdg, enter, seed=20261107):
    """The DS stage alone from given DS-entrance states (e.g. stock), fresh muon lifetimes."""
    key = jax.random.PRNGKey(seed)
    k_tau, k_ds = jax.random.split(key)
    x, p, t, pdg = (jnp.asarray(a) for a in (x, p, t, pdg))
    tau = jax.random.exponential(k_tau, (x.shape[0],), _fdt()) * phys.TAU_MU_NS
    return transport_ds(chain.ds_chunk, chain.ds_prm, x, p, t, _charge(pdg), tau, jnp.asarray(enter), k_ds)


# ---------------------------------------------------------------- inputs, bench entry points, CLI

def load_chain(gdml, field_config, offline_root, maps, ds_geom_file, ds_offline_root, ds_map,
               settings=Settings(), ds_prm=dst.Params(), pool=None, dtype="float64"):
    from . import fieldmap
    from .tsgeometry import Geometry
    set_dtype(dtype)
    tsg = DeviceGeometry(Geometry(gdml))
    tsf = DeviceField.from_config(field_config, offline_root, maps)
    dsg = dsgeometry.from_offline(ds_geom_file, ds_offline_root)
    fm = fieldmap.to_device(fieldmap.load(ds_map))
    return build_chain(tsf, tsg, fm, dsg, settings, ds_prm, pool=pool)


def load_reference(path, limit=None):
    with np.load(path) as d:
        r = {k: d[k] for k in d.files}
    n = len(r["pdg"]) if limit is None else min(limit, len(r["pdg"]))
    return {k: (v[:n] if len(v) == len(r["pdg"]) else v) for k, v in r.items()}


def load_references(paths, limit=None):
    """Per-input-row arrays of several references, concatenated in order, with a `batch` index per row (the
    position of its file in paths). Arrays that are not per input row (exposure tables) are dropped."""
    refs = [load_reference(p, limit) for p in paths]
    keys = [k for k, v in refs[0].items() if len(v) == len(refs[0]["pdg"])
            and all(k in r and len(r[k]) == len(r["pdg"]) for r in refs)]
    out = {k: np.concatenate([r[k] for r in refs]) for k in keys}
    out["batch"] = np.concatenate([np.full(len(r["pdg"]), i) for i, r in enumerate(refs)])
    return out


def _env(name):
    v = os.environ.get(name)
    if not v:
        raise KeyError(f"set {name}")
    return v


def bench_prepare(n):
    """Shared-timer prepare: inputs, geometry and field loaded and on the device (reported as prepare_s).
    Paths from TSDEV_REFERENCE (one file, or several separated by ':' to stream them in one pass), TSDEV_GDML,
    TSDEV_FIELD_CONFIG, TSDEV_OFFLINE_ROOT, TSDEV_MAPS, TSDEV_DS_GEOM, TSDEV_DS_OFFLINE_ROOT, TSDEV_DS_MAP; TSDEV_POOL
    sets the lanes in flight. Inputs tile the (joined) references cyclically to n rows."""
    ref = load_references(_env("TSDEV_REFERENCE").split(":"))
    idx = np.arange(n) % len(ref["pdg"])
    chain = load_chain(_env("TSDEV_GDML"), _env("TSDEV_FIELD_CONFIG"), _env("TSDEV_OFFLINE_ROOT"),
                       _env("TSDEV_MAPS"), _env("TSDEV_DS_GEOM"), _env("TSDEV_DS_OFFLINE_ROOT"), _env("TSDEV_DS_MAP"),
                       pool=int(os.environ["TSDEV_POOL"]) if os.environ.get("TSDEV_POOL") else None,
                       dtype=os.environ.get("TSDEV_DTYPE", "float64"))
    data = tuple(jnp.asarray(ref[k][idx]) for k in ("x", "p", "t", "pdg"))
    return chain, data


def bench_chain(prepared, seed=20261107):
    """Shared-timer entry point: one full pass TS entrance -> stops; returns the device stop arrays."""
    chain, (x, p, t, pdg) = prepared
    ts, ds = run_chain(chain, x, p, t, pdg, seed)
    return ts.fate, ds.fate, ds.x, ds.t, ds.foil


def summarize(ts, ds):
    ts_fate, ds_fate, pdg = np.asarray(ts.fate), np.asarray(ds.fate), np.asarray(ts.pdg)
    return dict(n=int(len(ts_fate)),
                ts_fates={name: int(np.sum(ts_fate == i)) for i, name in enumerate(FATES)},
                ts_waits=int(np.sum(np.asarray(ts.waits))), ts_lanes_waited=int(np.sum(np.asarray(ts.waits) > 0)),
                ds_entering_muons=int(np.sum((ts_fate == DS) & (np.abs(pdg) == 13))),
                ds_pions=int(np.sum((ts_fate == DS) & (np.abs(pdg) == 211))),
                ds_fates={name: int(np.sum(ds_fate == i)) for i, name in dst.FATES.items()},
                stops=int(np.sum(ds_fate == 1)), stops_mu_minus=int(np.sum((ds_fate == 1) & (np.asarray(ts.pdg) == 13))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, nargs="+",
                    help="one or more references; several stream through one resident pass (column batch)")
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--field-config", required=True)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--ds-geom", required=True, help="Offline geometry file for the DS stage, e.g. geom_run1_a.txt")
    ap.add_argument("--ds-offline-root", default=None, help="default: --offline-root")
    ap.add_argument("--ds-map", required=True, help="DS field map .header")
    ap.add_argument("--arm", choices=("fast", "stock-ds", "saved-ts"), default="fast",
                    help="fast: device TS + DS; stock-ds: DS stage from the reference's stock DS-entrance states; "
                         "saved-ts: DS stage from a saved NumPy TS output (--saved)")
    ap.add_argument("--saved")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pool", type=int, help="TS lanes in flight (default: all inputs)")
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    chain = load_chain(a.gdml, a.field_config, a.offline_root, a.maps, a.ds_geom, a.ds_offline_root or a.offline_root,
                       a.ds_map, Settings(step_mm=a.step), pool=a.pool, dtype=a.dtype)
    ref = load_references(a.reference, a.limit)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = dict(id=ref["id"], shard=ref["shard"], pdg_in=ref["pdg"], batch=ref["batch"])
    if a.arm == "fast":
        ts, ds = run_chain(chain, ref["x"], ref["p"], ref["t"], ref["pdg"], a.seed)
        meta = summarize(ts, ds)
        out.update({f"ts_{k}": np.asarray(v) for k, v in ts._asdict().items() if k != "hist"})
        pdg_ds = np.asarray(ts.pdg)
    else:
        if a.arm == "stock-ds":
            src = {k: ref[f"stock_{k}"] for k in ("x", "p", "t", "pdg")}
            enter = ref["stock_reached"] & (np.abs(ref["stock_pdg"]) == 13)
        else:
            with np.load(a.saved) as d:
                if not np.array_equal(d["id"], ref["id"]):
                    raise ValueError("saved TS output rows differ from the reference")
                src = {k: d[k] for k in ("x", "p", "t", "pdg")}
                enter = (d["fate"] == DS) & (np.abs(d["pdg"]) == 13)
        x = np.where(enter[:, None], src["x"], np.array([-3904.0, 0.0, 4179.0]))
        p = np.where(enter[:, None], src["p"], np.array([0.0, 0.0, 50.0]))
        pdg = np.where(enter, src["pdg"], 13)
        ds = run_ds_only(chain, x, p, np.where(enter, src["t"], 0.0), pdg, enter, a.seed)
        pdg_ds = pdg
        meta = dict(n=int(len(enter)), ds_entering_muons=int(enter.sum()),
                    ds_fates={name: int(np.sum(np.asarray(ds.fate) == i)) for i, name in dst.FATES.items()},
                    stops=int(np.sum(np.asarray(ds.fate) == 1)))
    out.update({f"ds_{k}": np.asarray(v) for k, v in ds._asdict().items()})
    out["pdg_ds"] = np.asarray(pdg_ds)
    jax.block_until_ready(ds.fate)
    meta.update(arm=a.arm, seed=a.seed, step_mm=a.step, dtype=a.dtype, load_s=load_s, run_s=time.perf_counter() - t0,
                device=str(jax.devices()[0]), ds_geom=a.ds_geom)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
