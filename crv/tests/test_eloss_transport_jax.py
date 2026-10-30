"""Paired execution tests, not a new statistical validation of Urban physics.

Same-seed reference agreement: rtol=2e-12, atol=2e-13 MeV. The sole intended
numeric change is float64 ion-sum reduction order; RNG keys and bounded samplers
are unchanged. Derived loss-ledger/kinetic-energy state changes use the same
absolute/relative tolerance, not bitwise equality. Momentum near stopping is
ill-conditioned and is intentionally NOT certified here (no transport wiring).
Run these tests on UVA compute nodes; no experiment data required.
"""
import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax import lax

from mu2efastsim import eloss_jax as ref, materials as M
from crvfastsim import eloss_transport_jax as fast

RTOL = 2e-12
ATOL = 2e-13
GAS = M.Material("StrawGas", 0.001696, 158.62097145526,
                 {"Ar": 0.784057035890315, "O": 0.15700963928136, "C": 0.0589333248283256})


def _paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert a.dtype == b.dtype == np.float64
    assert np.isfinite(a).all() and (a >= 0).all()
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def _branch(mat, p, length, tcut):
    c = ref.consts(mat)
    _, b2, _, _, tmax = ref.kinematics(p)
    cut = jnp.minimum(tcut, tmax)
    mean = ref.dedx_restricted(c, p, tcut) * length
    bohr = (mean >= ref._BOHR_MIN * cut) & (tmax <= 2 * cut)
    sig = jnp.sqrt(jnp.maximum((tmax / b2 - 0.5 * cut) * ref.TWOPI_MC2_RCL2 * length * c.ne, 0))
    sn = mean / jnp.where(sig > 0, sig, 1)
    return np.asarray(jnp.where(bohr, jnp.where(sn >= 2, 0, 1), 2))


POINTS = [
    (M.CONCRETE_MARS, 200., 100., 1e9, 0),  # Gauss
    (M.MILD_STEEL, 1.85, 1e-4, 1e9, 1),   # actual Gamma, last micrometre of range
    (GAS, 3000., 1., 990e-6, 2),           # Urban with discrete zero spike
    (M.POLYSTYRENE, 300., 20., .0872, 2),
    (M.CONCRETE_MARS, 3000., 50., 1e9, 2),
]


@pytest.mark.parametrize("seed", [0, 23])
@pytest.mark.parametrize("mat,p0,length,tcut,branch", POINTS,
                         ids=["gauss", "gamma", "urban-gas", "urban-ps", "urban-concrete"])
def test_same_seed_all_branches(seed, mat, p0, length, tcut, branch):
    p = jnp.full((513,), p0)
    assert (_branch(mat, p, length, tcut) == branch).all()
    key = jax.random.key(seed)
    actual = fast.sample_loss(key, mat, p, length, tcut)
    expected = ref.sample_loss(key, mat, p, length, tcut)
    _paired(actual, expected)
    np.testing.assert_array_equal(actual, fast.sample_loss(key, mat, p, length, tcut))
    # Different keys must still give distinct full-shape draw streams.
    assert not np.array_equal(actual, fast.sample_loss(jax.random.key(seed + 1), mat, p, length, tcut))


def test_mixed_material_branches_and_zero_length_state():
    # Repeat mixed lanes to test full-shape RNG draws, not per-mask compaction.
    index = np.tile([0, 1, 2, 3, 0, 1], 47)
    c = ref.stack_consts([M.CONCRETE_MARS, M.MILD_STEEL, GAS, M.POLYSTYRENE], index)
    p = jnp.tile(jnp.array([200., 1.85, 3000., 300., 200., 1.85]), 47)
    length = jnp.tile(jnp.array([100., 1e-4, 1., 20., 0., 0.]), 47)
    cut = jnp.tile(jnp.array([1e9, 1e9, 990e-6, .0872, 1e9, 1e9]), 47)
    assert set(_branch(c, p, length, cut).tolist()) == {0, 1, 2}
    key = jax.random.key(101)
    actual = fast.sample_loss(key, c, p, length, cut)
    expected = ref.sample_loss(key, c, p, length, cut)
    _paired(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual)[np.asarray(length) == 0], 0.)
    tk = jnp.sqrt(p * p + ref.MMU ** 2) - ref.MMU
    # Explicit state tolerance for future energy ledgers; neither sampler mutates inputs.
    np.testing.assert_allclose(jnp.maximum(tk - actual, 0.), jnp.maximum(tk - expected, 0.),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(jnp.cumsum(actual), jnp.cumsum(expected), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shape", [(), (0,), (13,), (2, 7)])
def test_zero_lengths_and_empty_batches(shape):
    p = jnp.full(shape, 3000.)
    key = jax.random.key(3)
    actual = fast.sample_loss(key, M.POLYSTYRENE, p, 0., .0872)
    _paired(actual, ref.sample_loss(key, M.POLYSTYRENE, p, 0., .0872))
    np.testing.assert_array_equal(actual, np.zeros(shape))


def test_standalone_jit_broadcast_and_dynamic_constants():
    f = jax.jit(fast.sample_loss)
    c = ref.stack_consts([M.POLYSTYRENE, M.CONCRETE_MARS], np.array([0, 1, 0]))
    p = jnp.array([[300.], [3000.]])
    length = jnp.array([0., 20., 40.])
    key = jax.random.key(9)
    actual = f(key, c, p, length, .0872)
    _paired(actual, ref.sample_loss(key, c, p, length, .0872))
    assert actual.shape == (2, 3)
    # All-Material constants closed over are also standalone jit-able.
    g = jax.jit(lambda k, p, L: fast.sample_loss(k, M.MILD_STEEL, p, L, 1e9))
    _paired(g(key, jnp.full(71, 1.85), 1e-4),
            ref.sample_loss(key, M.MILD_STEEL, jnp.full(71, 1.85), 1e-4, 1e9))


@pytest.mark.parametrize("rounds", [0, 1, ref.GAMMA_ROUNDS])
def test_gamma_round_substreams_and_first_acceptance(rounds):
    # Includes alpha<1 boost, low acceptance and reference fallback when rounds=0.
    alpha = jnp.tile(jnp.array([.01, .3, 1., 3.5, 100.]), 101)
    key = jax.random.key(41)
    actual = jax.jit(lambda k, a: fast._gamma_parallel(k, a, rounds))(key, alpha)
    _paired(actual, ref._gamma_fixed(key, alpha, rounds))


def _sequential_poisson(key, lams, uu, w3, w, K):
    """Literal reference loop, exposing counts to catch CDF boundary/key changes."""
    def body(j, carry):
        pmf, cdf, count, ion = carry
        above = uu > cdf
        r = jax.random.uniform(jax.random.fold_in(key, j), w.shape, jnp.float64)
        ion = ion + jnp.where(above[2], w3 / (1 - w * r), 0.)
        pmf = pmf * lams / (j + 1)
        return pmf, cdf + pmf, count + above, ion
    p0 = jnp.exp(-lams)
    _, _, count, ion = lax.fori_loop(0, K, body,
        (p0, p0, jnp.zeros(lams.shape, jnp.int32), jnp.zeros(w.shape)))
    return count, ion


@pytest.mark.parametrize("K", [0, 1, ref.K_POISSON])
def test_poisson_cap_counts_boundaries_and_round_substreams(K):
    # Includes exact-CDF boundaries, zero means, worst-case mean 8, and U=1
    # (an artificial variate exercising the cap/fallback deterministically).
    lams = jnp.broadcast_to(jnp.array([0., 1e-12, .2, 1., 4., 8., 8.]), (3, 7))
    uu = jnp.broadcast_to(jnp.array([0., 1., .99, np.exp(-1.), .999, 1., .1]), (3, 7))
    w3, w = jnp.full(7, .001), jnp.linspace(0., .999, 7)
    key = jax.random.key(52)
    actual = jax.jit(lambda k, l, u, a, b: fast._poisson_ion(k, l, u, a, b, K))(key, lams, uu, w3, w)
    expected = _sequential_poisson(key, lams, uu, w3, w, K)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert np.asarray(actual[0]).max() <= K
    _paired(actual[1], expected[1])


def test_bounded_poisson_has_no_runtime_loop():
    # Compiler contract: no 32-round scan/while kernel launch train.
    # Inspect the recurrence alone: PRNG implementations can contain their own
    # fixed cipher loops, which are unrelated to the removed Poisson loop.
    lams = jnp.full((3, 8), 8.)
    ir = str(jax.jit(fast._poisson_cdfs).lower(lams).compiler_ir("stablehlo"))
    assert "stablehlo.while" not in ir
    assert fast.K_POISSON == ref.K_POISSON == 32
    assert fast.GAMMA_ROUNDS == ref.GAMMA_ROUNDS == 10


@pytest.mark.parametrize("K", [0, 1, ref.K_POISSON])
def test_public_cap_and_mass_are_preserved(K):
    p = jnp.geomspace(100., 3000., 257)
    key = jax.random.key(71)
    # Non-default mass ensures all kinematic helpers see the caller's mass.
    actual = fast.sample_loss(key, M.POLYSTYRENE, p, 20., .0872, mass=140., K=K)
    expected = ref.sample_loss(key, M.POLYSTYRENE, p, 20., .0872, mass=140., K=K)
    _paired(actual, expected)


def test_optional_supplied_mean_matches_newer_core():
    if "mean" not in inspect.signature(ref.sample_loss).parameters:
        pytest.skip("caller-supplied mean requires newer core")
    p = jnp.geomspace(50., 3000., 257)
    mean = ref.dedx_restricted(M.CONCRETE_MARS, p, 1e9) * 15.
    key = jax.random.key(13)
    f = jax.jit(lambda k, p, m: fast.sample_loss(k, M.CONCRETE_MARS, p, 10., 1e9, mean=m))
    _paired(f(key, p, mean), ref.sample_loss(key, M.CONCRETE_MARS, p, 10., 1e9, mean=mean))
