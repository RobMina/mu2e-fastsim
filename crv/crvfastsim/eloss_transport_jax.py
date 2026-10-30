"""Execution twin of mu2efastsim.eloss_jax for opt-in device transport.

Same Urban/Gauss/Gamma physics, float64, split(key, 9) layout, fold_in round
streams, K=32 Poisson cap and ten-round Gamma fallback as core. Only scheduling
changes: batch random rounds, fully unroll the small PMF/CDF recurrence, then
compare/count/reduce over the round axis. No nested 32-trip GPU loop remains.
Gamma proposals are evaluated in parallel and the first accepted round wins;
a scalar conditional skips that work if no lane uses Gamma.

The ion sum is a parallel reduction rather than a sequential sum, so results
are numerically paired, not promised bit-identical to core. Tests require
rtol=2e-12, atol=2e-13 MeV (also for loss-ledger/kinetic-energy state
updates). Near-stop momentum sensitivity needs separate transport validation;
there is no distributional approximation or relaxed precision. Bitwise replay within this implementation is retained.
Temporary storage is O(K * batch size). Material preparation and all shared
physics helpers/constants are owned by core. Optional `mean` matches newer core.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eloss_jax as ej
from mu2efastsim.eloss import (ME, MMU, TWOPI_MC2_RCL2, _A0, _BOHR_MIN,
                              _FW, _MINLOSS, _NMAXCONT, _RATE)

# Reuse, do not fork the material, stopping-power or fluctuation helpers.
MatConsts = ej.MatConsts
consts = ej.consts
stack_consts = ej.stack_consts
kinematics = ej.kinematics
dedx_restricted = ej.dedx_restricted
_dedx = ej._dedx
_gauss_clipped = ej._gauss_clipped
_ion_split = ej._ion_split
K_POISSON = ej.K_POISSON
GAMMA_ROUNDS = ej.GAMMA_ROUNDS


def _round_keys(key, rounds):
    """Vectorized fold_in, never split(key, rounds): retain reference substreams."""
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(
        jnp.arange(rounds, dtype=jnp.int32))


def _gamma_parallel(key, alpha, rounds=GAMMA_ROUNDS):
    """Reference bounded Marsaglia-Tsang, evaluating independent rounds at once."""
    kb, kr = jax.random.split(key)
    a1 = jnp.where(alpha < 1, alpha + 1, alpha)
    d = a1 - 1.0 / 3.0
    cc = 1.0 / jnp.sqrt(9 * d)
    if rounds:
        keys = jax.vmap(jax.random.split)(_round_keys(kr, rounds))
        x = jax.vmap(lambda k: jax.random.normal(k, alpha.shape, jnp.float64))(keys[:, 0])
        lu = jnp.log(jax.vmap(lambda k: jax.random.uniform(
            k, alpha.shape, jnp.float64, minval=1e-300))(keys[:, 1]))
        v = (1 + cc * x) ** 3
        ok = (v > 0) & (lu < 0.5 * x * x + d - d * v + d * jnp.log(jnp.where(v > 0, v, 1.0)))
        first = jnp.argmax(ok, axis=0)
        val = jnp.take_along_axis(d * v, first[None], axis=0)[0]
        val = jnp.where(jnp.any(ok, axis=0), val, d)
    else:
        val = d
    ub = jax.random.uniform(kb, alpha.shape, jnp.float64, minval=1e-300)
    return jnp.where(alpha < 1, val * ub ** (1 / jnp.where(alpha > 0, alpha, 1.0)), val)


def _poisson_cdfs(lams, K=K_POISSON):
    """CDF[j] = P(N <= j), j=0..K-1, with the reference arithmetic order.

    Do not use log-PMF, cumprod or reassociated prefix sums: an unrolled
    recurrence preserves near-boundary Poisson decisions without runtime loops.
    """
    if K == 0:
        return jnp.zeros((0,) + lams.shape, jnp.float64)
    pmf0 = jnp.exp(-lams)

    def cdf_step(carry, j):
        pmf, cdf = carry
        nxt = pmf * lams / (j + 1)
        return (nxt, cdf + nxt), cdf

    _, cdfs = lax.scan(cdf_step, (pmf0, pmf0), jnp.arange(K), unroll=K)
    return cdfs


def _poisson_ion(key, lams, uu, w3, w, K=K_POISSON):
    """Same capped inverse CDF as core; compare/draw/reduce rounds in parallel."""
    if K == 0:
        return jnp.zeros(lams.shape, jnp.int32), jnp.zeros(w3.shape, jnp.float64)
    above = uu[None] > _poisson_cdfs(lams, K)
    r = jax.vmap(lambda k: jax.random.uniform(k, w3.shape, jnp.float64))(_round_keys(key, K))
    ion = jnp.sum(jnp.where(above[:, 2], w3 / (1 - w * r), 0.0), axis=0)
    return jnp.sum(above, axis=0, dtype=jnp.int32), ion


@partial(jax.jit, static_argnames=("K", "gamma_rounds"))
def _sample(key, c, p, length, tcut, mass, K=K_POISSON, gamma_rounds=GAMMA_ROUNDS, mean=None):
    p = jnp.asarray(p, jnp.float64)
    p, length, tcut = jnp.broadcast_arrays(p, jnp.asarray(length, jnp.float64), jnp.asarray(tcut, jnp.float64))
    c = MatConsts(*(jnp.broadcast_to(jnp.asarray(v, jnp.float64), p.shape) for v in c))
    sh = p.shape
    ks = jax.random.split(key, 9)
    tkin, beta2, bg2, gam, tmax = kinematics(p, mass)
    cut = jnp.minimum(tcut, tmax)
    # mean: caller-supplied mean loss (e.g. range-integrated, as G4VEnergyLossProcess for long steps)
    mean = _dedx(c, p, tcut, mass) * length if mean is None else jnp.broadcast_to(jnp.asarray(mean, jnp.float64), sh)
    # Gaussian/Gamma regime: many collisions, cut close to tmax
    bohr = (mean >= _BOHR_MIN * cut) & (tmax <= 2 * cut)
    siga = jnp.sqrt(jnp.clip((tmax / beta2 - 0.5 * cut) * TWOPI_MC2_RCL2 * length * c.ne, 0, None))
    sn = mean / jnp.where(siga > 0, siga, 1)
    gs = bohr & (sn >= 2)
    gm = bohr & (sn < 2)
    x_gs = _gauss_clipped(jax.random.uniform(ks[0], sh, jnp.float64), mean, siga ** 2)
    neff = jnp.where(gm, sn * sn, 1.0)
    # The scalar conditional skips all Gamma work for ordinary transport batches.
    # Within a mixed batch the original full-shape RNG draws are retained.
    x_gm = lax.cond(jnp.any(gm),
                    lambda: mean * _gamma_parallel(ks[1], neff, gamma_rounds) / neff,
                    lambda: jnp.zeros(sh, jnp.float64))
    # Urban model for the rest (inputs made safe outside u so every lane stays finite)
    u = ~bohr & (mean >= _MINLOSS) & (cut > c.e0)
    tc = jnp.where(u, cut, 1.0)
    scaling = jnp.minimum(1 + 0.5e-3 / tc, 1.5)
    ml = jnp.where(u, mean, 0.0) / scaling
    w2 = jnp.where(tc > c.I, jnp.log(2 * ME * beta2 * gam ** 2) - beta2, 0.0)
    lnI, lne1, lne2 = jnp.log(c.I), jnp.log(c.e1), jnp.log(c.e2)
    ok = (tc > c.I) & (w2 > lnI)
    full = ok & (w2 > lne2)
    C = ml * (1 - _RATE) / jnp.where(w2 > lnI, w2 - lnI, 1)
    a1 = jnp.where(full, C * c.f1 * (w2 - lne1) / c.e1, jnp.where(ok, ml * (1 - _RATE) / c.e1, 0.0))
    a2 = jnp.where(full, C * c.f2 * (w2 - lne2) / c.e2, 0.0)
    small = ok & (a1 < _A0)
    fwnow = jnp.where(small, 0.5 + (_FW - 0.5) * jnp.sqrt(jnp.clip(a1, 0, None) / _A0), _FW)
    e1 = jnp.where(ok, c.e1 * fwnow, c.e1)
    a1 = jnp.where(ok, a1 / fwnow, a1)
    e2 = c.e2
    w1 = tc / c.e0
    lw1 = jnp.log(jnp.where(w1 > 1, w1, 2.0))
    a3 = _RATE * ml * (tc - c.e0) / (c.e0 * tc * lw1)
    a3 = jnp.where(a1 + a2 <= 0, a3 / _RATE, a3)
    # excitation: Gaussian (continuous) part above _NMAXCONT, Poisson below
    cont1, cont2 = a1 > _NMAXCONT, a2 > _NMAXCONT
    eav = jnp.where(cont1, a1 * e1, 0.0) + jnp.where(cont2, a2 * e2, 0.0)
    s2 = jnp.where(cont1, a1 * e1 * e1, 0.0) + jnp.where(cont2, a2 * e2 * e2, 0.0)
    lam1 = jnp.where(cont1 | (a1 <= 0), 0.0, a1)
    lam2 = jnp.where(cont2 | (a2 <= 0), 0.0, a2)
    # ionisation: continuous part above _NMAXCONT, the rest as Poisson count of 1/E^2 terms
    cont = a3 > _NMAXCONT
    alfa, alfa1, namean, p3 = _ion_split(a3, w1)
    eav3 = namean * c.e0 * alfa1
    s23 = c.e0 * c.e0 * namean * (alfa - alfa1 * alfa1)
    w3 = alfa * c.e0
    has = tc > w3
    w = jnp.where(has, (tc - w3) / tc, 0.0)
    lam3 = jnp.where(has, jnp.clip(p3, 0, None), 0.0)
    # Generate exactly the reference round substreams, but in one batched operation.
    uu = jax.random.uniform(ks[2], (3,) + sh, jnp.float64)
    kk, ion = _poisson_ion(ks[3], jnp.stack([lam1, lam2, lam3]), uu, w3, w, K)
    rx = jax.random.uniform(ks[4], (2,) + sh, jnp.float64)
    loss = jnp.where(kk[0] > 0, ((kk[0] + 1) - 2 * rx[0]) * e1, 0.0)
    loss = loss + jnp.where(kk[1] > 0, ((kk[1] + 1) - 2 * rx[1]) * e2, 0.0)
    loss = loss + jnp.where(s2 > 0, _gauss_clipped(jax.random.uniform(ks[5], sh, jnp.float64), eav, s2), 0.0)
    loss = loss + ion
    loss = loss + jnp.where(s23 > 0, _gauss_clipped(jax.random.uniform(ks[6], sh, jnp.float64), eav3,
                                                    jnp.where(s23 > 0, s23, 0.0)), 0.0)
    out = jnp.where(gs, x_gs, jnp.where(gm, x_gm, mean))
    return jnp.where(u, loss * scaling, out)


def sample_loss(key, mat, p, length, tcut, mass=MMU, K=K_POISSON, mean=None):
    """Sampled restricted energy loss [MeV] over `length` [mm] (arrays broadcast to p's shape), G4 Urban model.

    key: jax.random key; mat: Material or MatConsts. jit-able (close over a Material, or pass MatConsts as arrays).
    mean: optional mean loss [MeV] replacing dE/dx(p) * length (the fluctuation width still uses p, length, tcut).
    """
    return _sample(key, consts(mat), p, length, tcut, jnp.asarray(mass, jnp.float64), K=K, mean=mean)

