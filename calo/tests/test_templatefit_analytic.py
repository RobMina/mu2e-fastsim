"""Closed-form Newton gradient and Hessian of the template fit (opt-in hessian="analytic") against autodiff."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from calofastsim.pulse import PulseCache  # noqa: E402
from calofastsim.templatefit import FitConfig, _compiled, _grad_hess, make_evaluate, make_evaluate_slope  # noqa: E402


@pytest.fixture(scope="module")
def pulse():
    x = (np.arange(6200) + 0.5) * 0.05
    return PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)


def _objective(evaluate, x, yfull, mask):
    def objective(p, act):
        f = p[0] + jnp.sum(jnp.where(act[:, None], p[1::2][:, None] * evaluate(x[None, :] - p[2::2][:, None]), 0.0),
                           axis=0)
        r = jnp.where(mask, yfull - f, 0.0)
        return jnp.where(jnp.abs(p[0]) > 1e-5, jnp.sum(r * r) / jnp.where(p[0] == 0, 1.0, p[0]), 0.0)
    return objective


def test_grad_hess_equal_autodiff(pulse):
    rng = np.random.default_rng(1)
    L, K = 64, 4
    x = jnp.asarray(100.0 + (np.arange(L) + 0.5) * 5.0)
    mask = jnp.asarray(np.arange(L) < 50)
    yfull = jnp.where(mask, jnp.asarray(rng.normal(12, 2, L) + 300 * np.exp(-((np.arange(L) - 12) / 4.0) ** 2)), 0.0)
    obj = _objective(make_evaluate(pulse), x, yfull, mask)
    slope = make_evaluate_slope(pulse)
    for _ in range(20):
        p = jnp.asarray(np.r_[rng.uniform(5, 20), np.column_stack((rng.uniform(1, 400, K),
                                                                     rng.uniform(110, 300, K))).ravel()])
        act = jnp.asarray(rng.random(K) < 0.7)
        g, h = _grad_hess(slope, x, yfull, mask, p, act)
        np.testing.assert_allclose(g, jax.grad(obj)(p, act), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(h, jax.hessian(obj)(p, act), rtol=1e-10, atol=1e-9)


def _digis(rng, n, L):
    t0 = rng.uniform(400, 1500, n)
    idx = np.arange(L)
    y = rng.normal(12.0, 1.5, (n, L))
    for k in range(2):
        amp = rng.uniform(30, 1500, n) * (k == 0 or 1) * (rng.random(n) < (1.0 if k == 0 else 0.3))
        pos = rng.uniform(14, 30, n) + 12 * k
        y += amp[:, None] * np.clip((idx - pos[:, None]) / 4.0, 0, None) ** 2 * np.exp(-(idx - pos[:, None]) / 4.0)
    length = rng.integers(40, L + 1, n)
    y[idx[None, :] >= length[:, None]] = 0.0
    return t0, np.trunc(y), length


@pytest.mark.parametrize("dtype", [None, jnp.float32])
def test_fit_matches_autodiff(pulse, dtype):
    rng = np.random.default_rng(7)
    L, cfg = 96, FitConfig()
    t0, y, length = _digis(rng, 256, L)
    fdt = jnp.float64 if dtype is None else dtype
    x = jnp.asarray(t0, fdt)[:, None] + (jnp.arange(L) + 0.5) * cfg.digi_sampling
    mask = jnp.arange(L)[None, :] < jnp.asarray(length)[:, None]
    args = (x, jnp.asarray(y, fdt), mask, jnp.asarray(length))
    ref = jax.tree_util.tree_map(np.asarray, _compiled(pulse, cfg, L, dtype)(*args))
    ana = jax.tree_util.tree_map(np.asarray, _compiled(pulse, cfg, L, dtype, "analytic")(*args))
    assert ana["amp"].dtype == ref["amp"].dtype
    # roundoff can flip a damped step's accept test, so a few digis may take another path; most must agree
    same = (ref["valid"] == ana["valid"]).all(axis=1)
    v = ref["valid"] & ana["valid"]
    rel = np.abs(ana["amp"] - ref["amp"]) / np.maximum(np.abs(ref["amp"]), 1.0)
    dt = np.abs(ana["peak_time"] - ref["peak_time"])
    tol_a, tol_t, frac = (1e-6, 1e-5, 0.99) if dtype is None else (2e-3, 0.05, 0.97)
    assert same.mean() >= frac
    assert np.mean((rel[v] < tol_a) & (dt[v] < tol_t)) >= frac
