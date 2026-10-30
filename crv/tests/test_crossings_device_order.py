"""Ordered device crossing selection against stable full-sort reference."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from crvfastsim import crossings_jax as jc


@pytest.mark.parametrize("capacity", [1, 4, 8, 16])
def test_bounded_selection_keeps_stable_rank_and_overflow(capacity):
    rng = np.random.default_rng(17)
    lo = rng.uniform(-10., 10., (13, 3))
    hi = lo + rng.uniform(.1, 5., (13, 3))
    lo[:5] = [-1., -1., -1.]
    hi[:5] = [1., 1., 1.]
    o = rng.uniform(-20., 20., (41, 3))
    d = rng.normal(size=o.shape)
    o[:7] = [[0., 10., 0.], [0., 0., 0.], [1., 10., 0.], [-1., 10., 0.],
             [1., 1., 1.], [0., -10., 0.], [30., 30., 30.]]
    d[:7] = [[0., -1., 0.], [1., 0., 0.], [-0., -1., 0.], [0., -1., 0.],
             [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]]
    active = np.ones(len(o), bool)
    active[6::4] = False
    ti, to = jc.slab(jnp.asarray(o), jnp.asarray(d), jnp.asarray(lo), jnp.asarray(hi))
    ti = jnp.maximum(ti, 0.)
    hit = (to > ti) & jnp.asarray(active)[:, None]
    order = jnp.argsort(jnp.where(hit, ti, jnp.inf), axis=1, stable=True)[:, :min(capacity, len(lo))]
    valid = jnp.take_along_axis(hit, order, axis=1)
    entry = jnp.where(valid, jnp.take_along_axis(ti, order, axis=1), 0.)
    exit_ = jnp.where(valid, jnp.take_along_axis(to, order, axis=1), 0.)
    cand = jax.device_get(jc.crossings(o, d, lo, hi, capacity, active))
    for a, b in zip(cand[:4], (order, entry, exit_, valid)):
        np.testing.assert_array_equal(a, np.asarray(b))
    assert cand.overflow == bool(jnp.any(hit.sum(1) > capacity))
