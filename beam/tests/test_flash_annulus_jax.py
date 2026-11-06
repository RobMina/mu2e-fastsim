"""Compatibility routing and independent beam-photon check of the core primitive."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from beamfastsim import flash_annulus_jax as A
from beamfastsim.flash_exit import foil_exit
from mu2efastsim.geometry import annulus_jax as shared


def test_compatibility_names_are_the_core_implementation():
    for name in ('Crossing', 'crossing', 'safety'):
        assert getattr(A, name) is getattr(shared, name)


def test_core_straight_boundary_matches_beam_photon_ray():
    rng = np.random.default_rng(123)
    n = 1024
    theta = rng.uniform(0., 2 * np.pi, n)
    inner, outer = rng.uniform(0., 3., n), rng.uniform(4., 10., n)
    radius = rng.uniform(inner, outer)
    center, half = rng.uniform(5000., 6000., n), rng.uniform(.002, 1., n)
    x = np.column_stack((-3904. + radius * np.cos(theta), radius * np.sin(theta),
                         center + rng.uniform(-half, half)))
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    expected = foil_exit(x, u, center, half, inner, outer, -3904.)
    solve = jax.jit(lambda x, u: A.crossing(x, u, 1., 1., None, 1e6, center, half, inner, outer, -3904.))
    got = solve(jnp.asarray(x), jnp.asarray(u))
    assert np.all(got.supported) and expected['valid'].all()
    np.testing.assert_allclose(got.distance, expected['distance'], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(got.surface, expected['face'])
