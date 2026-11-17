"""mu2efastsim.poisson: the batched Poisson draws follow the Poisson distribution (both methods, the rare
continuations included) and keep the shape, dtype and zero means of jax.random.poisson."""
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mu2efastsim import poisson as P  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)

N = 200_000


def _chi2_ok(draws, lam):
    """Pearson chi2 of the draws against the Poisson pmf, bins with at least 20 expected, tails merged."""
    draws = np.asarray(draws)
    sd = math.sqrt(max(lam, 1e-12))
    lo, hi = max(0, int(lam - 7 * sd) - 1), int(lam + 7 * sd) + 2
    ks = np.arange(lo, hi + 1)
    logp = np.array([k * math.log(lam) - lam - math.lgamma(k + 1) if lam > 0 else (0.0 if k == 0 else -np.inf)
                     for k in ks])
    expected = len(draws) * np.exp(logp)
    observed = np.array([np.sum(draws == k) for k in ks], float)
    assert np.all((draws >= lo) & (draws <= hi)), (lam, draws.min(), draws.max())
    keep = expected >= 20
    o = np.r_[observed[keep], observed[~keep].sum()]
    e = np.r_[expected[keep], expected[~keep].sum()]
    m = e > 0
    chi2 = float(np.sum((o[m] - e[m]) ** 2 / e[m]))
    ndf = int(m.sum()) - 1
    return chi2 <= ndf + 6 * math.sqrt(2 * max(ndf, 1)) + 10, (lam, chi2, ndf)


@pytest.mark.parametrize("lam", [0.3, 3.0, 9.999, 10.0, 10.001, 30.0, 300.0, 1e4, 1e6])
def test_batched_draws_are_poisson(lam):
    draws = jax.jit(lambda k: P.poisson(k, jnp.full(N, lam)))(jax.random.PRNGKey(11))
    assert draws.dtype == jnp.int64 and draws.shape == (N,)
    ok, info = _chi2_ok(draws, lam)
    assert ok, info
    assert abs(float(np.mean(draws)) - lam) < 5 * math.sqrt(lam / N), info


@pytest.mark.parametrize("terms,trials,lam", [(3, 8, 4.0), (6, 8, 9.0), (48, 1, 50.0), (48, 1, 1e4)])
def test_continuations_keep_the_distribution(terms, trials, lam):
    """Few CDF terms (most small lanes continue the inversion) or one PTRS trial (about a tenth of the large lanes
    take the fresh draws): still Poisson."""
    draws = jax.jit(lambda k: P.poisson(k, jnp.full(N, lam), terms=terms, trials=trials))(jax.random.PRNGKey(5))
    ok, info = _chi2_ok(draws, lam)
    assert ok, info


def test_mixed_means_shape_and_zero():
    lam = jnp.array([[0.0, 0.5, 12.0], [3e5, 0.0, 9.5]])
    draws = jax.jit(P.poisson)(jax.random.PRNGKey(3), lam)
    assert draws.shape == (2, 3) and draws.dtype == jnp.int64
    assert int(draws[0, 0]) == 0 and int(draws[1, 1]) == 0
    many = jax.jit(lambda k: P.poisson(k, jnp.tile(lam.ravel(), 50_000)))(jax.random.PRNGKey(4)).reshape(-1, 6)
    np.testing.assert_allclose(np.asarray(many).mean(0), np.asarray(lam).ravel(), rtol=0.02, atol=0.01)


def test_jax_method_is_jax_random_poisson(monkeypatch):
    monkeypatch.setattr(P, "METHOD", "jax")
    key, lam = jax.random.PRNGKey(9), jnp.array([0.5, 4.0, 40.0])
    np.testing.assert_array_equal(P.poisson(key, lam), jax.random.poisson(key, lam, dtype=jnp.int64))
    monkeypatch.setattr(P, "METHOD", "other")
    with pytest.raises(ValueError, match="METHOD"):
        P.poisson(key, lam)
