"""Synthetic straight-ray geometry tests; not curved navigation/transport closure."""

import numpy as np
import pytest

from beamfastsim.flash_foil_boundary import (
    NONE, ZLO, ZHI, RIN, ROUT, first_transition, first_transition_jax, validate_inputs,
)


def trace(x, u, **kw):
    geometry = dict(rin=1., rout=3., zlo=-2., zhi=2., s_max=20.)
    geometry.update(kw)
    return first_transition(np.asarray(x, dtype=float), np.asarray(u, dtype=float), **geometry)


def assert_hit(result, distance, kind, atol=1.e-12):
    assert np.all(result["valid"])
    assert not np.any(result["unresolved"])
    np.testing.assert_allclose(result["distance"], distance, rtol=0, atol=atol)
    np.testing.assert_array_equal(result["kind"], np.broadcast_to(kind, np.shape(result["kind"])))


def assert_nohit(result):
    assert not np.any(result["valid"])
    assert not np.any(result["unresolved"])
    assert np.all(np.isinf(result["distance"]))
    assert np.all(result["kind"] == NONE)


def assert_unresolved(result):
    assert not np.any(result["valid"])
    assert np.all(result["unresolved"])
    assert np.all(np.isinf(result["distance"]))
    assert np.all(result["kind"] == NONE)


@pytest.mark.parametrize("x,u,ex,distance,kind", [
    ([2, 0, -4], [0, 0, 1], False, 2, ZLO),
    ([2, 0, 4], [0, 0, -1], False, 2, ZHI),
    ([2, 0, 0], [0, 0, -1], True, 2, ZLO),
    ([2, 0, 0], [0, 0, 1], True, 2, ZHI),
    ([5, 0, 0], [-1, 0, 0], False, 2, ROUT),
    ([-5, 0, 0], [1, 0, 0], False, 2, ROUT),
    ([2, 0, 0], [1, 0, 0], True, 1, ROUT),
    ([-2, 0, 0], [-1, 0, 0], True, 1, ROUT),
    ([0, 0, 0], [1, 0, 0], False, 1, RIN),
    ([0, 0, 0], [-1, 0, 0], False, 1, RIN),
    ([2, 0, 0], [-1, 0, 0], True, 1, RIN),
    ([-2, 0, 0], [1, 0, 0], True, 1, RIN),
])
def test_analytic_faces_rims_and_holes(x, u, ex, distance, kind):
    assert_hit(trace(x, u, exiting=ex), distance, kind)


@pytest.mark.parametrize("x,u,ex,kind", [
    ([3, 0, 0], [-1, 0, 0], False, ROUT),
    ([3, 0, 0], [1, 0, 0], True, ROUT),
    ([1, 0, 0], [1, 0, 0], False, RIN),
    ([1, 0, 0], [-1, 0, 0], True, RIN),
    ([2, 0, -2], [0, 0, 1], False, ZLO),
    ([2, 0, -2], [0, 0, -1], True, ZLO),
    ([2, 0, 2], [0, 0, -1], False, ZHI),
    ([2, 0, 2], [0, 0, 1], True, ZHI),
])
def test_exact_surface_starts(x, u, ex, kind):
    assert_hit(trace(x, u, exiting=ex, s_max=0), 0, kind)


def test_no_hole_zero_radius_and_no_axis_transition():
    assert_hit(trace([0, 0, 0], [1, 0, 0], rin=0, exiting=True), 3, ROUT)
    assert_hit(trace([-2, 0, 0], [1, 0, 0], rin=0, exiting=True), 5, ROUT)
    assert_hit(trace([0, 0, -4], [0, 0, 1], rin=0), 2, ZLO)


def test_both_cylinder_roots_and_reentry_across_hole():
    assert_hit(trace([-5, 0, 0], [1, 0, 0]), 2, ROUT)
    assert_hit(trace([-3, 0, 0], [1, 0, 0], exiting=False), 0, ROUT)
    assert_hit(trace([-2, 0, 0], [1, 0, 0], exiting=True), 1, RIN)
    assert_hit(trace([-1, 0, 0], [1, 0, 0], exiting=True), 0, RIN)
    assert_hit(trace([-.5, 0, 0], [1, 0, 0]), 1.5, RIN)
    assert_hit(trace([1, 0, 0], [1, 0, 0]), 0, RIN)
    assert_hit(trace([2, 0, 0], [1, 0, 0], exiting=True), 1, ROUT)


def test_z_interval_can_remove_first_radial_material_interval():
    u = np.array([1., 0., 1.]) / np.sqrt(2)
    assert_hit(trace([-5, 0, -5], u, zlo=0, zhi=10), 6 * np.sqrt(2), RIN)


def test_oblique_side_exit_before_face():
    u = np.array([1., 0., 1.]) / np.sqrt(2)
    assert_hit(trace([2, 0, 0], u, exiting=True), np.sqrt(2), ROUT)


def test_axis_offset_and_shared_thickness_different_target_radii():
    x = np.array([[12., 0., -4.], [14., 0., -4.], [11.5, 0., -4.]])
    result = trace(x, [0, 0, 1], rin=[1., 1., 2.], rout=[3., 3., 3.], axis_x=10.)
    np.testing.assert_array_equal(result["valid"], [True, False, False])
    assert not np.any(result["unresolved"])
    assert result["distance"][0] == 2
    assert_hit(trace([12, 0, 0], [1, 0, 0], axis_x=10., exiting=True), 1, ROUT)


def test_ray_by_target_broadcast():
    x = np.array([[2., 0., -4.], [4., 0., -4.]])[:, None, :]
    result = trace(x, [0, 0, 1], rin=[1., 3.], rout=[3., 5.])
    np.testing.assert_array_equal(result["valid"], [[True, False], [False, True]])
    assert result["distance"].shape == (2, 2)
    assert not np.any(result["unresolved"])


@pytest.mark.parametrize("x,u", [
    ([4, 0, 0], [0, 0, 1]),
    ([0, 0, -5], [0, 0, 1]),
    ([5, 0, 3], [-1, 0, 0]),
    ([5, 0, 0], [1, 0, 0]),
    ([4, 4, 0], [0, 1, 0]),
])
def test_certified_misses_parallel_and_away(x, u):
    assert_nohit(trace(x, u))


def test_smax_is_finite_nonnegative_and_inclusive():
    assert_hit(trace([5, 0, 0], [-1, 0, 0], s_max=2), 2, ROUT)
    assert_nohit(trace([5, 0, 0], [-1, 0, 0], s_max=1))
    assert_nohit(trace([5, 0, 0], [-1, 0, 0], s_max=0))
    assert_unresolved(trace([5, 0, 0], [-1, 0, 0], s_max=2 - 1.e-14))


def test_exact_and_near_tangencies_never_false_transitions():
    assert_unresolved(trace([-4, 3, 0], [1, 0, 0]))
    assert_unresolved(trace([-2, 1, 0], [1, 0, 0], exiting=True))
    assert_nohit(trace([-4, 3.01, 0], [1, 0, 0]))
    assert_unresolved(trace([-4, 3 - 1.e-15, 0], [1, 0, 0]))
    assert_hit(trace([-4, 2., 0], [1, 0, 0]), 4 - np.sqrt(5), ROUT)


def test_parallel_on_surface_and_corners_are_refused():
    assert_unresolved(trace([3, 0, -4], [0, 0, 1]))
    assert_unresolved(trace([1, 0, -4], [0, 0, 1]))
    assert_unresolved(trace([5, 0, 2], [-1, 0, 0]))
    u = np.array([-1., 0., 1.]) / np.sqrt(2)
    assert_unresolved(trace([5, 0, -4], u))
    assert_unresolved(trace([3, 0, -2], u, s_max=0))
    assert_unresolved(trace([5, 0, 0], u))
    assert_unresolved(trace([5, 0, 0], u, zhi=2 - 1.e-14))


def test_tolerance_does_not_silently_drop_short_material_interval():
    assert_unresolved(trace([-4, 2.99, 0], [1, 0, 0], tolerance=.6))
    assert_unresolved(trace([2, 0, -4], [0, 0, 1], zlo=0, zhi=.01, tolerance=.1))
    assert_unresolved(trace([-5, 0, 0], [1, 0, 0], rin=2.99, tolerance=.1))


def test_wrong_initial_material_claim_is_unresolved():
    assert_unresolved(trace([2, 0, 0], [1, 0, 0]))
    assert_unresolved(trace([5, 0, 0], [-1, 0, 0], exiting=True))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_and_stable_near_root(dtype):
    x = np.array([3 + .001, 0, 0], dtype=dtype)
    u = np.array([-1., 0, 0], dtype=dtype)
    result = first_transition(x, u, 1, 3, -2, 2, s_max=10)
    assert result["distance"].dtype == dtype
    assert_hit(result, float(x[0]) - 3, ROUT, atol=5 * np.finfo(dtype).eps)
    result = first_transition(np.array([-1.e4, 2., 0.], dtype=dtype),
                              np.array([1., 0., 0.], dtype=dtype), 1, 3, -2, 2, s_max=2.e4)
    assert_hit(result, 1.e4 - np.sqrt(5), ROUT, atol=.01 if dtype == np.float32 else 1.e-10)


@pytest.mark.parametrize("kw", [
    {"rin": -1}, {"rout": 1}, {"rin": 4}, {"zlo": 2}, {"zhi": -3},
    {"rin": np.nan}, {"rout": np.inf}, {"axis_x": np.inf},
    {"s_max": np.inf}, {"s_max": -1}, {"tolerance": -1},
    {"tolerance": np.nan}, {"exiting": 1}, {"s_max": [20, -1]},
])
def test_malformed_geometry(kw):
    with pytest.raises(ValueError):
        trace([5, 0, 0], [-1, 0, 0], **kw)


@pytest.mark.parametrize("x,u", [
    ([1, 2], [0, 0, 1]), ([1, 2, 3], [0, 1]),
    ([np.nan, 0, 0], [1, 0, 0]), ([0, 0, 0], [0, 0, 0]),
    ([0, 0, 0], [2, 0, 0]), ([0, 0, 0], [np.inf, 0, 0]),
])
def test_malformed_rays(x, u):
    with pytest.raises(ValueError):
        trace(x, u)


def test_host_validation_rejects_incompatible_shapes_and_complex():
    with pytest.raises(ValueError):
        trace(np.zeros((2, 3)), [1, 0, 0], rin=np.ones(3))
    with pytest.raises(ValueError):
        validate_inputs([1j, 0, 0], [1, 0, 0], 1, 3, -2, 2)


def test_jax_optional_jit_parity_and_device_dtype():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    x = np.array([[5, 0, 0], [2, 0, 0], [0, 0, 0], [-4, 3, 0], [5, 0, 3]], np.float32)
    u = np.array([[-1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0]], np.float32)
    exiting = np.array([False, True, False, False, False])
    validate_inputs(x, u, 1, 3, -2, 2, exiting=exiting)
    reference = first_transition(x, u, 1, 3, -2, 2, exiting=exiting)
    fn = jax.jit(first_transition_jax)
    result = fn(jnp.asarray(x), jnp.asarray(u), 1., 3., -2., 2., exiting=jnp.asarray(exiting))
    for key in reference:
        np.testing.assert_array_equal(np.asarray(result[key]), reference[key])
    assert result["distance"].dtype == jnp.float32
    try:
        from jax import enable_x64
    except ImportError:
        from jax.experimental import enable_x64
    with enable_x64():
        result = fn(jnp.array([5., 0., 0.], dtype=jnp.float64),
                    jnp.array([-1., 0., 0.], dtype=jnp.float64), 1., 3., -2., 2.)
        assert result["distance"].dtype == jnp.float64
        assert_hit(result, 2, ROUT)


def test_jax_malformed_numeric_lanes_refuse_without_host_exception():
    jax = pytest.importorskip("jax")
    result = jax.jit(first_transition_jax)(np.array([5., 0., 0.], np.float32),
                                         np.array([-2., 0., 0.], np.float32), 1., 3., -2., 2.)
    assert_unresolved(result)


def test_direction_underflow_is_not_mistaken_for_parallel():
    assert_unresolved(trace([0, 0, 0], [1.e-200, 0, 1.], rin=0, exiting=True))


def test_seeded_generic_rays_against_independent_midpoint_oracle():
    rng = np.random.default_rng(9027)
    x = rng.uniform(-7, 7, (2048, 3))
    u = rng.normal(size=x.shape)
    u /= np.linalg.norm(u, axis=-1)[:, None]
    radius = np.hypot(x[:, 0], x[:, 1])
    ex = (radius > 1) & (radius < 3) & (np.abs(x[:, 2]) < 2)
    result = trace(x, u, exiting=ex)
    a = np.sum(u[:, :2] ** 2, axis=-1)
    b = np.sum(x[:, :2] * u[:, :2], axis=-1)
    c = np.sum(x[:, :2] ** 2, axis=-1)
    candidates = [(-2 - x[:, 2]) / u[:, 2], (2 - x[:, 2]) / u[:, 2]]
    for r in (1., 3.):
        disc = b * b - a * (c - r * r)
        root = np.sqrt(np.maximum(disc, 0))
        candidates.extend((np.where(disc >= 0, (-b - root) / a, np.inf),
                           np.where(disc >= 0, (-b + root) / a, np.inf)))
    roots = np.sort(np.array(candidates).T, axis=-1)

    def membership(t):
        p = x[:, None, :] + t[..., None] * u[:, None, :]
        r = np.hypot(p[..., 0], p[..., 1])
        return (r > 1) & (r < 3) & (np.abs(p[..., 2]) < 2)

    crossing = membership(roots - 1.e-7) != membership(roots + 1.e-7)
    ref = np.min(np.where(crossing & (roots >= 0) & (roots <= 20), roots, np.inf), axis=-1)
    assert not np.any(result["unresolved"])
    np.testing.assert_array_equal(result["valid"], np.isfinite(ref))
    np.testing.assert_allclose(result["distance"], ref, rtol=0, atol=1.e-11)


def test_jax_mixed_surfaces_degeneracies_and_target_broadcast():
    jax = pytest.importorskip("jax")
    x = np.array([[3, 0, 0], [1, 0, 0], [2, 0, -4], [3, 0, -4],
                  [-4, 3 - 1.e-6, 0], [3, 0, -2], [0, 0, 0]], np.float32)
    u = np.array([[-1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 1],
                  [1, 0, 0], [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [1, 0, 0]], np.float32)
    ex = np.array([False, True, False, False, False, False, True])[:, None]
    kw = dict(rin=np.array([1, 0], np.float32), rout=3., zlo=-2., zhi=2., exiting=ex)
    ref = first_transition(x[:, None, :], u[:, None, :], **kw)
    result = jax.jit(first_transition_jax)(x[:, None, :], u[:, None, :], **kw)
    for key in ref:
        np.testing.assert_array_equal(np.asarray(result[key]), ref[key])
