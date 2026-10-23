"""Batched muon transport from the DS entrance to the stopping target and beyond, in JAX (CPU or GPU).

Each muon is a lane; all lanes take fixed RK4 steps of `ds` mm in the field map (Lorentz force on the unit direction,
du/ds = (0.2998 q / p) u x B, B in T, p in MeV/c) until they stop, decay, leave the region or hit the step cap.
Materials are thin layers crossed between two steps:
  * target foils: planes z = foil_z with an annulus [rin, rout]; path 2 ht / |u_z| split in slices, each with a
    G4 Urban sampled loss (mu2efastsim.eloss_jax) at the slice momentum, then a Highland kick in two planes;
    a muon whose kinetic energy drops below T_STOP stops in that foil;
  * IPA: cylinder r = ipa_r for z in [ipa_z0, ipa_z1], same treatment with its wall thickness;
  * TSdA and the outer proton absorber are perfect absorbers (5 cm and 2 cm of polyethylene);
  * decay in flight: proper lifetime sampled at the start (memoryless), proper time accumulated per step.
Fate codes: see FATES.
"""
from functools import partial
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eloss, eloss_jax

from . import fieldmap, geometry

jax.config.update("jax_enable_x64", True)

C_MM_NS = 299.792458
K_LORENTZ = 0.299792458      # (MeV/c) / (T mm)
TAU_MU = 2196.9811           # ns
T_STOP = 0.1                 # MeV: below this kinetic energy the muon is stopped where it is
FATES = {0: "alive", 1: "stop_foil", 2: "stop_ipa", 3: "absorbed_tsda", 4: "absorbed_opa", 5: "decay",
         6: "downstream", 7: "upstream", 8: "wall", 9: "step_cap"}


class Params(NamedTuple):
    ds: float = 5.0             # step [mm]
    n_slices: int = 4           # energy-loss slices per layer crossing
    tcut: float = 1.0           # delta-ray production threshold [MeV] (restricted loss; G4 Al cut is ~0.6 MeV)
    max_steps: int = 3000
    z_up: float = 4100.0        # leaves upstream below this z
    r_wall: float = 950.0       # DS inner cryostat radius [mm]
    msc: bool = True
    decay: bool = True


class State(NamedTuple):
    x: object       # (N, 3) mm
    u: object       # (N, 3) unit direction
    p: object       # (N,) MeV/c
    t: object       # (N,) ns
    tau: object     # (N,) proper time so far, ns
    tau_max: object  # (N,) sampled proper lifetime
    q: object       # (N,) charge
    fate: object    # (N,) int
    foil: object    # (N,) int: foil of the stop, -1 otherwise
    nfoil: object   # (N,) int: foil crossings


def _deriv(fm, x, u, kq):
    return u, kq[:, None] * jnp.cross(u, fieldmap.field(fm, x))


def _rk4(fm, x, u, kq, h):
    k1x, k1u = _deriv(fm, x, u, kq)
    k2x, k2u = _deriv(fm, x + 0.5 * h * k1x, u + 0.5 * h * k1u, kq)
    k3x, k3u = _deriv(fm, x + 0.5 * h * k2x, u + 0.5 * h * k2u, kq)
    k4x, k4u = _deriv(fm, x + h * k3x, u + h * k3u, kq)
    xn = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    un = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
    return xn, un / jnp.linalg.norm(un, axis=1, keepdims=True)


def _orthonormal(u):
    a = jnp.where(jnp.abs(u[:, 2:3]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]))
    e1 = jnp.cross(u, a)
    e1 = e1 / jnp.linalg.norm(e1, axis=1, keepdims=True)
    return e1, jnp.cross(u, e1)


def _layer(key, mat_c, x0_mm, p, u, length, active, n_slices, tcut, msc):
    """Energy loss in slices and one Highland kick for lanes crossing a layer of `length` mm; returns (p, u, stopped)."""
    kl, km = jax.random.split(key)
    dl = length / n_slices
    pk = p
    for s in range(n_slices):
        loss = eloss_jax._sample(jax.random.fold_in(kl, s), mat_c, pk, dl, tcut, jnp.asarray(eloss.MMU))
        tk = jnp.sqrt(pk * pk + eloss.MMU ** 2) - eloss.MMU - loss
        tk = jnp.maximum(tk, 0.0)
        pk = jnp.sqrt(tk * (tk + 2 * eloss.MMU))
    tk_out = jnp.sqrt(pk * pk + eloss.MMU ** 2) - eloss.MMU
    stopped = active & (tk_out < T_STOP)
    p_new = jnp.where(active, pk, p)
    if msc:
        pm = jnp.sqrt(jnp.maximum(p * p_new, 1.0))            # geometric mean momentum over the layer
        e = jnp.sqrt(pm * pm + eloss.MMU ** 2)
        beta = pm / e
        xx = jnp.maximum(length / x0_mm, 1e-12)
        th0 = 13.6 / (beta * pm) * jnp.sqrt(xx) * (1 + 0.038 * jnp.log(xx / beta ** 2))
        g = jax.random.normal(km, (p.shape[0], 2), jnp.float64) * th0[:, None]
        e1, e2 = _orthonormal(u)
        un = u + g[:, :1] * e1 + g[:, 1:] * e2
        un = un / jnp.linalg.norm(un, axis=1, keepdims=True)
        u = jnp.where((active & ~stopped)[:, None], un, u)
    return p_new, u, stopped


_CACHE = {}


def make_stepper(fm, geom, prm, chunk=None):
    """Return a jitted function (key, State) -> (State, steps) that runs at most `chunk` steps (default max_steps).

    Lanes that are done keep their state; call repeatedly (with compaction, see transport_all) to finish. The field
    map is a traced argument (not a compile-time constant) and the jitted core is cached per (geom, prm, chunk).
    """
    ck = (id(geom), prm, chunk)
    if ck not in _CACHE:
        _CACHE[ck] = _build(geom, prm, chunk)
    core = _CACHE[ck]
    return lambda key, st: core(fm, key, st)


def _build(geom, prm, chunk):
    al = eloss_jax.consts(geometry.STOPPING_TARGET_AL)
    pe = eloss_jax.consts(geometry.IPA_PE)
    x0_al = eloss.radiation_length_mm(geometry.STOPPING_TARGET_AL)
    x0_pe = eloss.radiation_length_mm(geometry.IPA_PE)
    fz = jnp.asarray(geom.foil_z)
    frin, frout, fht = jnp.asarray(geom.foil_rin), jnp.asarray(geom.foil_rout), jnp.asarray(geom.foil_ht)
    h = prm.ds
    nmax = chunk or prm.max_steps

    def step(carry):
        fm, key, st, i = carry
        key, kl = jax.random.split(key)
        alive = st.fate == 0
        kq = K_LORENTZ * st.q / jnp.maximum(st.p, 1e-3)
        xn, un = _rk4(fm, st.x, st.u, kq, h)
        xn = jnp.where(alive[:, None], xn, st.x)
        un = jnp.where(alive[:, None], un, st.u)
        z0, z1 = st.x[:, 2], xn[:, 2]
        # foil crossing: first foil plane between z0 and z1 (spacing >> step, so at most one)
        lo, hi = jnp.minimum(z0, z1), jnp.maximum(z0, z1)
        crossed = (fz[None, :] > lo[:, None]) & (fz[None, :] <= hi[:, None])
        k = jnp.argmax(crossed, axis=1).astype(jnp.int32)
        anyc = alive & jnp.any(crossed, axis=1)
        frac = jnp.where(anyc, (fz[k] - z0) / jnp.where(z1 != z0, z1 - z0, 1.0), 0.0)
        xc = st.x + frac[:, None] * (xn - st.x)
        rc = jnp.hypot(xc[:, 0] - geom.axis_x, xc[:, 1])
        hit = anyc & (rc >= frin[k]) & (rc <= frout[k])
        uz = jnp.maximum(jnp.abs(un[:, 2]), 0.02)
        # IPA: radial crossing of r = ipa_r inside its z range (exclusive with a foil hit in the same step)
        r0 = jnp.hypot(st.x[:, 0] - geom.axis_x, st.x[:, 1])
        r1 = jnp.hypot(xn[:, 0] - geom.axis_x, xn[:, 1])
        zin = (z1 >= geom.ipa_z0) & (z1 <= geom.ipa_z1)
        ipa = alive & ~hit & zin & ((r0 - geom.ipa_r) * (r1 - geom.ipa_r) < 0)
        ur = jnp.abs(((xn[:, 0] - geom.axis_x) * un[:, 0] + xn[:, 1] * un[:, 1]) / jnp.maximum(r1, 1e-6))
        # one merged layer per step: per-lane material, X0 and path length
        mc = eloss_jax.MatConsts(*(jnp.where(ipa, b, a) for a, b in zip(al, pe)))
        x0 = jnp.where(ipa, x0_pe, x0_al)
        length = jnp.where(ipa, geom.ipa_t / jnp.maximum(ur, 0.02), 2 * fht[k] / uz)
        act = hit | ipa
        p2, u2, stop_l = lax.cond(jnp.any(act),
                                  lambda: _layer(kl, mc, x0, st.p, un, length, act, prm.n_slices, prm.tcut, prm.msc),
                                  lambda: (st.p, un, jnp.zeros_like(act)))
        # time and proper time over the step at the pre-step speed
        e = jnp.sqrt(st.p ** 2 + eloss.MMU ** 2)
        dt = jnp.where(alive, h * e / (jnp.maximum(st.p, 1e-3) * C_MM_NS), 0.0)
        tau = st.tau + dt * eloss.MMU / e
        # fates, in priority order
        fate = st.fate
        fate = jnp.where(alive & stop_l & hit, jnp.int32(1), fate)
        fate = jnp.where((fate == 0) & alive & stop_l & ipa, 2, fate)
        in_tsda = (jnp.abs(z1 - geom.tsda_z) < geom.tsda_hl) & (r1 > geom.tsda_rin) & (r1 < geom.tsda_rout)
        fate = jnp.where((fate == 0) & alive & in_tsda, 3, fate)
        in_opa = (z1 > geom.opa_z0) & (z1 < geom.opa_z1) & (r1 > geom.opa_rin)
        fate = jnp.where((fate == 0) & alive & in_opa, 4, fate)
        if prm.decay:
            fate = jnp.where((fate == 0) & alive & (tau > st.tau_max), 5, fate)
        fate = jnp.where((fate == 0) & alive & (z1 > geom.ipa_z1), 6, fate)
        fate = jnp.where((fate == 0) & alive & (z1 < prm.z_up), 7, fate)
        fate = jnp.where((fate == 0) & alive & (r1 > prm.r_wall), 8, fate)
        xn = jnp.where((alive & (fate == 1))[:, None], xc, xn)
        foil = jnp.where(alive & (fate == 1), k, st.foil)
        st = State(xn, u2, p2, st.t + dt, tau, st.tau_max, st.q, fate, foil, st.nfoil + hit.astype(jnp.int32))
        return fm, key, st, i + 1

    def cond(carry):
        _, _, st, i = carry
        return (i < nmax) & jnp.any(st.fate == 0)

    @jax.jit
    def run(fm, key, st):
        _, key, st, n = lax.while_loop(cond, step, (fm, key, st, jnp.int32(0)))
        return st, n

    return run


def transport_all(fm, geom, prm, key, st, chunk=200, min_bucket=256):
    """Run to completion in chunks of `chunk` steps, compacting live lanes between chunks (padded to powers of 2
    so only a few shapes compile). Returns the final State (numpy) and the total number of lane-steps executed."""
    run = make_stepper(fm, geom, prm, chunk=chunk)
    final = {f: np.array(getattr(st, f)) for f in State._fields}
    idx = np.arange(len(final["p"]))
    cur = st
    done_steps, lane_steps, it = 0, 0, 0
    while len(idx) and done_steps < prm.max_steps:
        cur, n = run(jax.random.fold_in(key, it), cur)
        n = int(n)
        lane_steps += n * len(idx)
        done_steps += n
        it += 1
        host = {f: np.asarray(getattr(cur, f)) for f in State._fields}
        real = idx >= 0
        for f in State._fields:
            final[f][idx[real]] = host[f][real]
        live = real & (host["fate"] == 0)
        idx = idx[live]
        if not len(idx):
            break
        m = max(min_bucket, 1 << int(np.ceil(np.log2(len(idx)))))
        pad = m - len(idx)
        sel = np.concatenate([np.flatnonzero(live), np.full(pad, np.flatnonzero(live)[0])])
        cur = State(*(jnp.asarray(host[f][sel]) for f in State._fields))
        cur = cur._replace(fate=cur.fate.at[len(idx):].set(9))       # padding lanes are inert
        idx = np.concatenate([idx, np.full(pad, -1)])
    final["fate"] = np.where(final["fate"] == 0, 9, final["fate"])
    return State(**final), lane_steps


def initial_state(key, x, p3, t, q, prm):
    """State from start positions (N,3) mm, momenta (N,3) MeV/c, times (N,) ns and charges (N,)."""
    x, p3 = np.asarray(x, np.float64), np.asarray(p3, np.float64)
    p = np.linalg.norm(p3, axis=1)
    n = len(p)
    tau_max = jax.random.exponential(key, (n,), jnp.float64) * TAU_MU if prm.decay else jnp.full(n, jnp.inf)
    return State(jnp.asarray(x, jnp.float64), jnp.asarray(p3 / p[:, None]), jnp.asarray(p), jnp.asarray(t, jnp.float64),
                 jnp.zeros(n), tau_max, jnp.asarray(q, jnp.float64), jnp.zeros(n, jnp.int32),
                 jnp.full(n, -1, jnp.int32), jnp.zeros(n, jnp.int32))
