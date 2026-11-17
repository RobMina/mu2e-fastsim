"""Poisson draws in a fixed number of fused steps: the exact Poisson distribution without a per-lane while loop.

`jax.random.poisson` runs Knuth's product method (means below 10) and transformed rejection (PTRS, Hormann 1993)
as while loops over the whole array until its slowest lane is done, both on every lane. On a GPU every iteration
launches its own small kernels: thousands per W6 batch. `poisson` here draws the same distribution in elementwise
steps that XLA fuses:
- means below SMALL: inversion of the Poisson CDF from one uniform, over TERMS terms (the sum stops counting once
  the CDF passes the uniform);
- larger means: TRIALS independent PTRS trials at once; the first accepted trial is the draw.
A lane that the fixed steps do not settle, under lax.cond, continues exactly: a small-mean lane continues the same
CDF sum past TERMS terms (its draw is TERMS or more), a large-mean lane takes a fresh `jax.random.poisson` draw (PTRS
trials are independent, so the rejected ones do not bias it). With the defaults this is below 1e-15 of the small
lanes and about 1e-9 of the large ones. All arithmetic is float64.

The draws are other random numbers than jax.random.poisson's (same key, same distribution). METHOD = "jax" keeps
jax.random.poisson (read when a kernel is traced).
"""
import jax
import jax.numpy as jnp
from jax import lax

METHOD = "batched"      # "batched" (this module) or "jax" (jax.random.poisson)
SMALL = 10.0            # means below: CDF inversion; from here: PTRS trials (jax.random.poisson's own split)
TERMS = 48              # CDF terms of the inversion: P(X > 47 | mean < 10) < 1e-15
TRIALS = 8              # PTRS trials at once: P(no acceptance) < 1e-9 (acceptance per trial above 0.9)


def _inversion(u, lam, terms):
    """Inversion of the Poisson CDF over `terms` terms: the count of CDF values below u, and the last term, CDF and
    term index, to continue from (a lane is settled once its CDF reaches u)."""
    p = jnp.exp(-lam)
    cdf = p
    k = jnp.zeros(lam.shape, jnp.int64)
    for i in range(1, terms):
        k = k + (cdf < u).astype(jnp.int64)
        p = p * lam / i
        cdf = cdf + p
    return k, p, cdf, u <= cdf


def _continue_inversion(u, lam, k, p, cdf, start):
    """The same inversion past `start` terms, for the lanes whose CDF has not reached u: exact, stops where a term
    underflows (the CDF then stays below u only by rounding)."""
    def more(state):
        k, p, cdf, i = state
        return jnp.any((cdf < u) & (p > 0))

    def step(state):
        k, p, cdf, i = state
        live = (cdf < u) & (p > 0)
        k = k + live.astype(jnp.int64)
        p = jnp.where(live, p * lam / i, p)
        cdf = jnp.where(live, cdf + p, cdf)
        return k, p, cdf, i + 1.0
    k, *_ = lax.while_loop(more, step, (k, p, cdf, jnp.float64(start)))
    return k


def _ptrs(key, lam, trials):
    """First accepted of `trials` PTRS trials per lane (numpy's and jax.random.poisson's transformed rejection), and
    whether one was."""
    ku, kv = jax.random.split(key)
    shape = (trials,) + lam.shape
    U = jax.random.uniform(ku, shape, jnp.float64) - 0.5
    V = jax.random.uniform(kv, shape, jnp.float64)
    slam, loglam = jnp.sqrt(lam), jnp.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)
    us = 0.5 - jnp.abs(U)
    k = jnp.floor((2 * a / us + b) * U + lam + 0.43)
    quick = (us >= 0.07) & (V <= vr)
    reject = (k < 0) | ((us < 0.013) & (V > us))
    lhs = jnp.log(V) + jnp.log(invalpha) - jnp.log(a / (us * us) + b)
    rhs = -lam + k * loglam - lax.lgamma(k + 1)
    accept = quick | (~reject & (lhs <= rhs))
    first = jnp.argmax(accept, axis=0)
    found = jnp.any(accept, axis=0)
    draw = jnp.take_along_axis(k, first[None], axis=0)[0]
    return draw.astype(jnp.int64), found


def poisson(key, lam, *, terms=None, trials=None):
    """int64 Poisson draws with means `lam` (any shape, >= 0), exact distribution; see the module docstring."""
    if METHOD == "jax":
        return jax.random.poisson(key, lam, dtype=jnp.int64)
    if METHOD != "batched":
        raise ValueError("mu2efastsim.poisson.METHOD must be \"batched\" or \"jax\"")
    terms = TERMS if terms is None else int(terms)
    trials = TRIALS if trials is None else int(trials)
    lam = jnp.asarray(lam, jnp.float64)
    small = lam < SMALL
    k_u, k_r, k_f = jax.random.split(key, 3)
    u = jax.random.uniform(k_u, lam.shape, jnp.float64)
    lam_small = jnp.where(small, lam, 0.0)
    k_small, p, cdf, ok_small = _inversion(u, lam_small, terms)
    k_large, ok_large = _ptrs(k_r, jnp.where(small, SMALL, lam), trials)
    draw = jnp.where(small, k_small, k_large)
    settled = jnp.where(small, ok_small, ok_large)

    def finish():   # rare: continue the inversion (small means); fresh PTRS draws, independent of the rejected
        small_rest = _continue_inversion(u, lam_small, k_small, p, cdf, terms)   # trials (large means)
        large_rest = jax.random.poisson(k_f, jnp.where(small, SMALL, lam), dtype=jnp.int64)
        return jnp.where(settled, draw, jnp.where(small, small_rest, large_rest))
    return lax.cond(jnp.all(settled), lambda: draw, finish)
