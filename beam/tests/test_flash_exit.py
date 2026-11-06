"""Analytic checks of transparent-foil geometry, not physical flux validation."""
import numpy as np
import pytest

from beamfastsim.flash_exit import (
    INNER_HOLE, INVALID, OUTER_RIM, Z_MINUS, Z_PLUS, foil_exit,
    stock_time_selection,
)


def exit_at(p, d, **kwargs):
    params = dict(foil_z=0., foil_ht=2., foil_rin=1., foil_rout=5., axis_x=0.)
    params.update(kwargs)
    return foil_exit(p, d, **params)


def assert_exit(result, distance, position, face):
    assert np.all(result["valid"])
    np.testing.assert_allclose(result["distance"], distance, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(result["position"], position, rtol=2e-14, atol=1e-14)
    np.testing.assert_array_equal(result["face"], face)
    np.testing.assert_allclose(np.linalg.norm(result["direction"], axis=-1), 1.)


def test_slab_faces_half_thickness_and_direction_normalization():
    result = exit_at([[2., 0., 10.], [2., 0., 10.5]],
                     [[0., 0., 12.], [0., 0., -7.]], foil_z=10.)
    assert_exit(result, [2., 2.5], [[2., 0., 12.], [2., 0., 8.]], [Z_PLUS, Z_MINUS])
    np.testing.assert_array_equal(result["direction"], [[0., 0., 1.], [0., 0., -1.]])


def test_oblique_slab_distance_is_path_length():
    result = exit_at([2., 0., 0.], [3., 0., 4.], foil_ht=1.)
    assert_exit(result, 1.25, [2.75, 0., 1.], Z_PLUS)
    np.testing.assert_allclose(result["direction"], [0.6, 0., 0.8])


def test_rim_and_hole_are_first_outward_boundaries():
    result = exit_at([[2., 0., 0.], [2., 0., 0.], [0., 2., 0.]],
                     [[1., 0., 0.], [-1., 0., 0.], [0., -1., 0.]])
    assert_exit(result, [3., 1., 1.], [[5., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                [OUTER_RIM, INNER_HOLE, INNER_HOLE])


def test_nonzero_axis_and_broadcast_per_photon_parameters():
    axis = np.array([[-3904.], [7.]])
    z = np.array([[10.], [-20.]])
    p = np.zeros((2, 3, 3))
    p[..., 0], p[..., 2] = axis + 2., z
    d = np.broadcast_to([0., 0., 1.], p.shape)
    ht = np.array([0.1, 0.2, 0.3])
    result = foil_exit(p, d, z, ht, np.ones((2, 1)), [4., 5., 6.], axis)
    expected = p.copy()
    expected[..., 2] += ht
    assert_exit(result, np.broadcast_to(ht, (2, 3)), expected, np.full((2, 3), Z_PLUS))
    assert_exit(exit_at([-3902., 0., 0.], [-1., 0., 0.], axis_x=-3904.),
                1., [-3903., 0., 0.], INNER_HOLE)


@pytest.mark.parametrize("p,d,distance,end,face", [
    ([2., 0., 2.], [0., 0., 1.], 0., [2., 0., 2.], Z_PLUS),
    ([2., 0., 2.], [0., 0., -1.], 4., [2., 0., -2.], Z_MINUS),
    ([2., 0., -2.], [0., 0., -1.], 0., [2., 0., -2.], Z_MINUS),
    ([2., 0., -2.], [0., 0., 1.], 4., [2., 0., 2.], Z_PLUS),
    ([5., 0., 0.], [1., 0., 0.], 0., [5., 0., 0.], OUTER_RIM),
    ([5., 0., 0.], [-1., 0., 0.], 4., [1., 0., 0.], INNER_HOLE),
    ([1., 0., 0.], [-1., 0., 0.], 0., [1., 0., 0.], INNER_HOLE),
    ([1., 0., 0.], [1., 0., 0.], 4., [5., 0., 0.], OUTER_RIM),
    ([2., 0., 2.], [1., 0., 0.], 3., [5., 0., 2.], OUTER_RIM),
    ([5., 0., 0.], [0., 0., 1.], 2., [5., 0., 2.], Z_PLUS),
])
def test_exact_surface_outward_inward_and_surface_parallel(p, d, distance, end, face):
    assert_exit(exit_at(p, d), distance, end, face)


def test_tangency_does_not_enter_hole_but_leaves_outer_rim():
    root = np.sqrt(24.)
    result = exit_at([[1., 0., 0.], [1., -2., 0.], [5., 0., 0.]],
                     [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]])
    assert_exit(result, [root, root + 2., 0.],
                [[1., root, 0.], [1., root, 0.], [5., 0., 0.]], [OUTER_RIM] * 3)


def test_near_tangent_hole_is_still_an_exit():
    x = 1. - 1e-10
    y = -np.sqrt((1. - x) * (1. + x))
    result = exit_at([x, -2., 0.], [0., 1., 0.])
    assert result["valid"] and result["face"] == INNER_HOLE
    np.testing.assert_allclose(result["position"], [x, y, 0.], atol=1e-10)
    np.testing.assert_allclose(result["distance"], 2. + y, atol=1e-10)


def test_solid_disc_has_no_spurious_zero_radius_hole():
    assert_exit(exit_at([2., 0., 0.], [-1., 0., 0.], foil_rin=0.),
                7., [-5., 0., 0.], OUTER_RIM)
    assert_exit(exit_at([0., 0., 0.], [1., 0., 0.], foil_rin=0.),
                5., [5., 0., 0.], OUTER_RIM)


def test_corner_ties_prefer_z_face():
    assert_exit(exit_at([3., 0., 0.], [1., 0., 1.]),
                np.sqrt(8.), [5., 0., 2.], Z_PLUS)
    assert_exit(exit_at([5., 0., 2.], [1., 0., 1.]), 0., [5., 0., 2.], Z_PLUS)


def test_near_surface_roots_do_not_cancel():
    near_outer, near_inner = np.nextafter(5., 0.), np.nextafter(1., 2.)
    result = exit_at([[near_outer, 0., 0.], [near_inner, 0., 0.]],
                     [[1., 0., 0.], [-1., 0., 0.]])
    np.testing.assert_allclose(result["distance"], [5. - near_outer, near_inner - 1.],
                               rtol=2e-14, atol=0.)
    np.testing.assert_array_equal(result["face"], [OUTER_RIM, INNER_HOLE])


@pytest.mark.parametrize("p,d,params", [
    ([0., 0., 0.], [1., 0., 0.], {}),  # hole birth
    ([6., 0., 0.], [-1., 0., 0.], {}),  # outside, even if it later enters
    ([2., 0., 3.], [0., 0., -1.], {}),
    ([2., 0., np.nextafter(2., 3.)], [0., 0., -1.], {}),
    ([np.nan, 0., 0.], [1., 0., 0.], {}),
    ([2., np.inf, 0.], [1., 0., 0.], {}),
    ([2., 0., 0.], [0., 0., 0.], {}),
    ([2., 0., 0.], [np.nan, 0., 1.], {}),
    ([2., 0., 0.], [np.inf, 0., 1.], {}),
    ([2., 0., 0.], [1., 0., 0.], {"foil_ht": 0.}),
    ([2., 0., 0.], [1., 0., 0.], {"foil_ht": -1.}),
    ([2., 0., 0.], [1., 0., 0.], {"foil_ht": np.inf}),
    ([2., 0., 0.], [1., 0., 0.], {"foil_rin": -1.}),
    ([2., 0., 0.], [1., 0., 0.], {"foil_rin": 5.}),
    ([2., 0., 0.], [1., 0., 0.], {"foil_rout": np.inf}),
    ([2., 0., 0.], [1., 0., 0.], {"foil_z": np.nan}),
    ([2., 0., 0.], [1., 0., 0.], {"axis_x": np.inf}),
    ([1e308, 0., 0.], [1., 0., 0.], {"axis_x": -1e308}),
])
def test_invalid_lanes_have_explicit_mask_and_nan_payload(p, d, params):
    with np.errstate(all="raise"):
        result = exit_at(p, d, **params)
    assert not result["valid"]
    assert result["face"] == INVALID
    for name in ("distance", "position", "direction"):
        assert np.all(np.isnan(result[name]))


def test_invalid_lanes_do_not_poison_valid_lanes_or_modify_inputs():
    p = np.array([[2., 0., 0.], [np.nan, 0., 0.], [2., 0., 0.]])
    d = np.tile([0., 0., 1.], (3, 1))
    original = p.copy()
    with np.errstate(all="raise"):
        result = exit_at(p, d, foil_ht=[2., 2., -1.])
    np.testing.assert_array_equal(result["valid"], [True, False, False])
    assert result["distance"][0] == 2.
    np.testing.assert_array_equal(p, original)
    np.testing.assert_array_equal(d, np.tile([0., 0., 1.], (3, 1)))


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_finite_extreme_direction_scales(magnitude):
    assert_exit(exit_at([2., 0., 0.], [0., 0., magnitude]), 2., [2., 0., 2.], Z_PLUS)


def test_empty_batches_and_bad_shapes():
    result = exit_at(np.empty((0, 3)), np.empty((0, 3)))
    assert result["distance"].shape == (0,)
    assert result["position"].shape == (0, 3)
    with pytest.raises(ValueError, match="identical"):
        exit_at(np.zeros((2, 3)), np.ones(3))
    with pytest.raises(ValueError, match="identical"):
        exit_at(np.zeros((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        exit_at(np.zeros((2, 3)), np.ones((2, 3)), foil_ht=np.ones(3))


def test_stock_time_strict_boundaries_negative_fmod_and_late_times():
    t = np.array([0., np.nextafter(10., 0.), 10., 250., np.nextafter(250., np.inf),
                  1695., -100., -1500., 1695. * 1e9 + 100., 1695. * 1e9 + 251.,
                  np.nan, np.inf, -np.inf])
    expected = [True, True, False, False, True, True, True, True, False, True,
                False, False, False]
    np.testing.assert_array_equal(stock_time_selection(t), expected)
    assert not ((np.remainder(-1500., 1695.) > 250.) or
                (np.remainder(-1500., 1695.) < 10.))  # catches wrong folding
    late = np.finfo(float).max
    phase = np.fmod(late, 1695.)
    assert stock_time_selection(late) == ((phase > 250.) or (phase < 10.))


def test_stock_time_custom_period_and_shape():
    np.testing.assert_array_equal(stock_time_selection([[1001., 1010.], [1250., 1251.]], 1000.),
                                  [[True, False], [False, True]])
    assert stock_time_selection(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("period", [0., -1., np.nan, np.inf, [1695.]])
def test_stock_time_invalid_period(period):
    with pytest.raises(ValueError, match="period"):
        stock_time_selection(0., period)
