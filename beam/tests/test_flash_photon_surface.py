"""Pure analytic geometry fixtures; no transport/producer/physics validation."""
import numpy as np
import pytest
from beamfastsim.flash_photon_surface import (
    box_segment_crossings, OK, INVALID, AMBIGUOUS, XLO, XHI, YLO, YHI, ZLO, ZHI,
)


def cross(a, b, **kw):
    return box_segment_crossings(a, b, lo=kw.pop("lo", [-1.] * 3),
                                 hi=kw.pop("hi", [1.] * 3),
                                 tolerance=kw.pop("tolerance", 1e-10), **kw)


def refused(r, status):
    np.testing.assert_array_equal(r.status, status)
    assert not r.resolved.any()
    assert np.isnan(r.u).all() and np.isnan(r.facebitmask).all() and np.isnan(r.direction).all()


@pytest.mark.parametrize("axis,faces", [(0, (XLO, XHI)), (1, (YLO, YHI)), (2, (ZLO, ZHI))])
@pytest.mark.parametrize("reverse", [False, True])
def test_through_each_face_and_reverse(axis, faces, reverse):
    a, b = np.zeros((1, 3)), np.zeros((1, 3))
    a[:, axis], b[:, axis] = (-2, 2) if not reverse else (2, -2)
    r = cross(a, b)
    np.testing.assert_array_equal(r.status, [OK])
    np.testing.assert_allclose(r.u, [[.25, .75]])
    np.testing.assert_array_equal(r.facebitmask, [faces[::-1] if reverse else faces])
    np.testing.assert_array_equal(r.direction, [[-1, 1]])


def test_single_crossings_no_hit_and_duplicate_rows():
    a = [[0, 0, 0], [2, 0, 0], [0, 0, 0], [2, 3, 0], [0, 0, 0]]
    b = [[2, 0, 0], [0, 0, 0], [.5, .2, .1], [-2, 3, 0], [2, 0, 0]]
    r = cross(a, b)
    assert r.resolved.all()
    np.testing.assert_allclose(r.u[[0, 1, 4]], [[.5, np.nan]] * 3, equal_nan=True)
    np.testing.assert_array_equal(r.direction[[0, 1, 4], 0], [1, -1, 1])
    assert np.isnan(r.u[2:4]).all()


@pytest.mark.parametrize("a,b", [
    ([-1, 0, 0], [0, 0, 0]), ([0, 0, 0], [1, 0, 0]),
    ([-1 - 1e-11, 0, 0], [0, 0, 0]), ([-2, -2, 0], [2, 2, 0]),
    ([-2, -2, -2], [2, 2, 2]), ([-2, 0, 0], [0, 2, 0]),
    ([-2, 1, 0], [2, 1, 0]), ([-2, 1, 1], [2, 1, 1]),
    ([-2, 0, 0], [2, 1e-12, 0]), ([0, 0, 0], [1e-12, 0, 0]),
])
def test_boundary_corner_tangent_coplanar_nearparallel(a, b):
    refused(cross([a], [b]), [AMBIGUOUS])


@pytest.mark.parametrize("a,b", [([0, 0, 0], [0, 0, 0]),
    ([np.nan, 0, 0], [2, 0, 0]), ([0, 0, 0], [np.inf, 0, 0])])
def test_invalid(a, b):
    refused(cross([a], [b]), [INVALID])


def test_random_well_conditioned_against_six_plane_oracle():
    rng = np.random.default_rng(734)
    a, b = rng.uniform(-3, 3, (2, 4096, 3))
    r = cross(a, b)
    assert r.resolved.all()
    axes, bounds = np.repeat(np.arange(3), 2), np.tile([-1., 1.], 3)
    d = b - a
    t = (bounds - a[:, axes]) / d[:, axes]
    p = a[:, None, :] + t[:, :, None] * d[:, None, :]
    transverse = np.arange(3)[None, None, :] != axes[None, :, None]
    hits = (t > 0) & (t < 1) & np.all((np.abs(p) < 1) | ~transverse, axis=2)
    order = np.argsort(np.where(hits, t, np.inf), axis=1)[:, :2]
    expected = np.take_along_axis(np.where(hits, t, np.nan), order, axis=1)
    np.testing.assert_allclose(r.u, expected, atol=2e-15, equal_nan=True)
    face = np.broadcast_to(1 << np.arange(6), t.shape)
    np.testing.assert_allclose(r.facebitmask, np.take_along_axis(np.where(hits, face, np.nan), order, axis=1), equal_nan=True)
    outgoing = d[:, axes] * np.tile([-1, 1], 3) > 0
    np.testing.assert_allclose(r.direction, np.take_along_axis(np.where(hits, np.where(outgoing, 1, -1), np.nan), order, axis=1), equal_nan=True)


@pytest.mark.parametrize("scale", [1e-250, 1e-100, 1., 1e100, 1e250])
def test_extreme_resolved_scales(scale):
    r = cross(np.array([[-2., 0, 0]]) * scale, np.array([[2., 0, 0]]) * scale,
              lo=np.full(3, -scale), hi=np.full(3, scale), tolerance=scale * 1e-10)
    assert r.resolved.all()
    np.testing.assert_allclose(r.u, [[.25, .75]])


def test_precision_promotion_empty_and_overflow_refusal():
    a = np.array([[-2., 0, 0]], dtype=np.float32)
    b = -a
    old = a.copy()
    r = cross(a, b)
    assert r.u.dtype == r.facebitmask.dtype == r.direction.dtype == np.float64
    np.testing.assert_array_equal(a, old)
    np.testing.assert_allclose(r.u, [[.25, .75]])
    e = cross(np.empty((0, 3)), np.empty((0, 3)))
    assert e.u.shape == e.facebitmask.shape == e.direction.shape == (0, 2)
    assert e.status.shape == e.resolved.shape == (0,)
    refused(cross([[-1e308, 0, 0]], [[1e308, 0, 0]]), [AMBIGUOUS])
    refused(cross([[0, 0, 0]], [[2, 0, 0]], tolerance=1.), [AMBIGUOUS])
    refused(cross([[0, 0, 0]], [[1e307, 0, 0]], lo=[-1e308] * 3,
                  hi=[1e308] * 3), [AMBIGUOUS])
    refused(cross([[1e16, 0, 0]], [[1e16 + 4, 0, 0]],
                  lo=[1e16 - 2, -1, -1], hi=[1e16 + 2, 1, 1]), [AMBIGUOUS])


@pytest.mark.parametrize("kw", [{"tolerance": 0}, {"tolerance": -1}, {"tolerance": np.nan},
    {"tolerance": [1e-10]}, {"lo": [0, 0, 0], "hi": [0, 1, 1]},
    {"hi": [1, 1, np.inf]}, {"lo": [-1, -1]}, {"hi": ["1"] * 3}])
def test_bad_geometry(kw):
    with pytest.raises(ValueError):
        cross([[0, 0, 0]], [[2, 0, 0]], **kw)


def test_bad_schema_and_no_geometry_defaults():
    with pytest.raises(ValueError):
        cross([0, 0, 0], [2, 0, 0])
    with pytest.raises(ValueError):
        cross([[0, 0, 0]], [[2, 0, 0], [3, 0, 0]])
    with pytest.raises(TypeError):
        box_segment_crossings([[0, 0, 0]], [[2, 0, 0]])


def test_translated_anisotropic_physical_frame_and_corner_refusal():
    lo = np.array([-3980., -90., 5450.])
    hi = np.array([-3820., 90., 6300.])
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    a = np.broadcast_to(center, (6, 3)).copy()
    b = a.copy()
    axis = np.repeat(np.arange(3), 2)
    signs = np.tile([-1, 1], 3)
    b[np.arange(6), axis] += signs * 2 * half[axis]
    result = cross(a, b, lo=lo, hi=hi)
    assert result.resolved.all()
    np.testing.assert_allclose(result.u[:, 0], .5)
    np.testing.assert_array_equal(result.facebitmask[:, 0], 1 << np.arange(6))
    np.testing.assert_array_equal(result.direction[:, 0], np.ones(6))
    assert np.isnan(result.u[:, 1]).all()
    refused(cross([center - 2 * half], [center + 2 * half], lo=lo, hi=hi), [AMBIGUOUS])
