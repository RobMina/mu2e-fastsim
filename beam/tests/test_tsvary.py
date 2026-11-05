"""Model-bound regressions; dense quadrature checks are not interval certificates."""

import numpy as np
import pytest

from beamfastsim import tsarc, tsbound, tsfield, tslocate, tsnav, tsstep, tsvary as v


def linear_map(matrix=None, offset=(0, 0, .5), *, lo=(-8, -8, -8), hi=(8, 8, 8), mirror=False):
    matrix = np.zeros((3, 3)) if matrix is None else np.asarray(matrix)
    header = tsfield.GridHeader('synthetic.header', lo, [0, 0, 0], [2, 2, 2],
                                np.subtract(hi, lo), mirror, True)
    xyz = np.stack(np.meshgrid(*zip(lo, hi), indexing='ij'), axis=-1)
    return tsfield.FieldMap(header, xyz @ matrix.T + offset)


def enclosure(item=None, **kwargs):
    defaults = dict(x=[[0., 0, 0]], p3=[[tsstep.K*.8, 0, tsstep.K*.6]], pdg=13,
                    length=3., field_error_tesla=0., position_error_mm=0.)
    defaults.update(kwargs)
    return v.MapEnclosure(item=linear_map() if item is None else item, **defaults)


def geometry(mother, *obstacles):
    root = tslocate.Volume(mother, 'vacuum', 'mother', 'root', True)
    daughters = tuple(tslocate.Volume(s, 'metal', 'obstacle', 'root', False) for s in obstacles)
    return tslocate.Partition((tslocate.Cell(root, daughters),))


def quadratic_phase_path(s, charge, b0=.5, gradient=.3):
    nodes, weights = np.polynomial.legendre.leggauss(48)
    t = s[:, None] * (nodes + 1) / 2
    phase = -charge * (b0*t + gradient*.6*t*t/2)
    x = .8 * s/2 * (np.cos(phase) @ weights)
    y = .8 * s/2 * (np.sin(phase) @ weights)
    return np.column_stack((x, y, .6*s))


def test_all_edge_spatial_bound_and_exact_affine_slopes():
    matrix = np.array([[1., 2, 3], [-2, 1, 0], [0, 4, -1]])
    item = linear_map(matrix)
    b = tsbound.spatial(item, chunk_nodes=3)
    np.testing.assert_allclose(b.column_tesla_per_mm, np.linalg.norm(matrix, axis=0))
    np.testing.assert_allclose(b.lipschitz_tesla_per_mm, np.linalg.norm(matrix))
    assert b.edge_count == 12 and b.reflection_continuous
    assert not b.floating_point_certified and not b.native_query_certified
    rng = np.random.default_rng(391)
    x, y = rng.uniform(-8, 8, (2, 1000, 3))
    assert np.all(np.linalg.norm(item.field(x)-item.field(y), axis=1) <=
                  b.lipschitz_tesla_per_mm*np.linalg.norm(x-y, axis=1) + 1e-13)


def test_nonlinear_trilinear_spatial_bound_samples_all_cells():
    rng = np.random.default_rng(1937)
    header = tsfield.GridHeader('synthetic.header', [-2, -3, -4], [0, 0, 0], [4, 5, 6], [1, 2, 3])
    item = tsfield.FieldMap(header, rng.normal(size=(4, 5, 6, 3)))
    b = tsbound.spatial(item, chunk_nodes=7)
    assert b.edge_count == 3*5*6 + 4*4*6 + 4*5*5
    x = rng.uniform(item.lo, item.hi, (1000, 3))
    y = rng.uniform(item.lo, item.hi, (1000, 3))
    assert np.all(np.linalg.norm(item.field(x)-item.field(y), axis=1) <=
                  b.lipschitz_tesla_per_mm*np.linalg.norm(x-y, axis=1) + 1e-14)


def test_model_error_encloses_quadratic_phase_varying_field_both_charges_species():
    item = linear_map(np.diag([0, 0, .3]))
    e = enclosure(item, x=np.zeros((4, 3)), p3=np.tile([tsstep.K*.8, 0, tsstep.K*.6], (4, 1)),
                  pdg=[13, -13, 211, -211])
    n = 257
    s = np.tile(np.linspace(0, 3, n), 4)
    rows = np.repeat(np.arange(4), n)
    actual = np.concatenate([quadratic_phase_path(np.linspace(0, 3, n), q) for q in [-1, 1, 1, -1]])
    ref, _ = e.reference.state(rows, s)
    bound = e.model_error(rows, s)
    assert np.all(np.linalg.norm(actual-ref, axis=1) <= bound + 1e-14)
    np.testing.assert_allclose(bound, .3*s**3/6, atol=1e-15)
    assert np.linalg.norm(actual[n-1]-ref[n-1]) > .4
    assert not e.floating_point_certified and not e.native_query_certified


def test_both_separate_balls_enclose_dense_actual_varying_path():
    item = linear_map(np.diag([0, 0, .3]))
    e = enclosure(item)
    lo, hi = np.array([0., .25, 1., 2.]), np.array([3., .5, 1.5, 2.125])
    balls = e.balls(np.zeros(4, dtype=int), lo, hi)
    s = (lo[:, None] + (hi-lo)[:, None]*np.linspace(0, 1, 257)).ravel()
    actual = quadratic_phase_path(s, -1).reshape(4, 257, 3)
    assert np.all(np.linalg.norm(actual-balls.midpoint[:, None], axis=2) <= balls.midpoint_radius_mm[:, None]+1e-14)
    assert np.all(np.linalg.norm(actual-balls.chord_center[:, None], axis=2) <= balls.chord_radius_mm[:, None]+1e-14)
    assert np.any(np.linalg.norm(balls.midpoint-balls.chord_center, axis=1) > .01)
    assert balls.source_index.dtype.kind in 'iu'
    np.testing.assert_array_equal(e.model_error(balls.source_index, balls.s_hi), balls.endpoint_model_error_mm)
    for arr in (balls.midpoint, balls.s_lo, e.curvature, e.field_error_tesla):
        with pytest.raises(ValueError):
            arr.setflags(write=True)


def test_uniform_map_zero_model_error_and_declared_evaluation_error():
    e = enclosure(field_error_tesla=.04, position_error_mm=.03)
    s = np.array([0., 1, 2, 3])
    np.testing.assert_allclose(e.model_error(np.zeros(4, dtype=int), s), .02*s*s)
    balls = e.balls([0], [1.], [3.])
    np.testing.assert_allclose(balls.midpoint_radius_mm, 1+.08+.03)
    np.testing.assert_allclose(balls.endpoint_model_error_mm, [.18])
    zero = enclosure()
    np.testing.assert_array_equal(zero.model_error(np.zeros(4, dtype=int), s), 0.)


def test_zero_field_tiny_momentum_does_not_form_infinite_charge_over_momentum():
    e = enclosure(linear_map(offset=(0, 0, 0)), p3=[[1e-310, 0, 0]], length=1.)
    np.testing.assert_array_equal(e.model_error([0], [1.]), [0.])
    b = e.balls([0], [0.], [1.])
    np.testing.assert_array_equal(b.midpoint, [[.5, 0, 0]])


def test_large_model_error_uses_unit_speed_bound_not_small_angle_assumption():
    e = enclosure(field_error_tesla=1e100)
    np.testing.assert_allclose(e.model_error([0], [3.]), [6.])


def test_owner_queries_keep_model_uncertainty_and_material_identity():
    e = enclosure(linear_map(np.diag([0, 0, .3])))
    large = geometry(tsnav.Box([20, 20, 20]))
    result = e.owners([0], [0.], [3.], large, geometry_error_mm=0.)
    np.testing.assert_array_equal(result.status, [tslocate.VACUUM])
    np.testing.assert_array_equal(result.volume_index, [0])
    assert all(a.dtype.kind in 'iu' for a in (result.status, result.volume_index, result.ball_kind))
    point, _ = e.reference.state(np.array([0]), np.array([3.]))
    small = geometry(tsnav.Placed(tsnav.Box([.1, .1, .1]), point[0], np.eye(3)))
    pending = e.owners([0], [3.], [3.], small, geometry_error_mm=0.)
    np.testing.assert_array_equal(pending.status, [0])
    np.testing.assert_array_equal(pending.volume_index, [-1])
    np.testing.assert_array_equal(pending.ball_kind, [0])
    metal = geometry(tsnav.Box([30, 30, 30]), tsnav.Box([20, 20, 20]))
    material = e.owners([0], [0.], [3.], metal, geometry_error_mm=0.)
    np.testing.assert_array_equal(material.status, [tslocate.MATERIAL])
    np.testing.assert_array_equal(material.volume_index, [1])
    no = e.owners([0], [0.], [3.], large, geometry_error_mm=100.)
    np.testing.assert_array_equal(no.status, [0])


def test_reflected_seam_discontinuity_and_lobe_gap_fail_closed():
    bad = linear_map(offset=(0, 1, .5), lo=(-8, 0, -8), mirror=True)
    assert not tsbound.spatial(bad).reflection_continuous
    with pytest.raises(ValueError, match='discontinuous'):
        enclosure(bad)
    enclosure(bad, x=[[0, 4, 0]], length=1.)
    with pytest.raises(ValueError, match='discontinuous'):
        tsbound.require_boxes(bad, [[0, -1, 0]], [[0, 0, 0]], continuous=True)
    good = linear_map(np.diag([0, .2, 0]), lo=(-8, 0, -8), mirror=True)
    assert tsbound.spatial(good).reflection_continuous
    enclosure(good)
    gap = linear_map(lo=(-8, 1, -8), mirror=True)
    with pytest.raises(ValueError, match='wholly covered'):
        enclosure(gap)
    with pytest.raises(ValueError, match='wholly covered'):
        enclosure(length=9.)


def test_late_subdivision_does_not_reset_accumulated_model_error():
    e = enclosure(linear_map(np.diag([0, 0, .3])))
    b = e.balls([0, 0], [2., 2.], [2., 2.000001])
    np.testing.assert_allclose(b.midpoint_radius_mm[0], .4)
    assert b.midpoint_radius_mm[1] > .4
    np.testing.assert_allclose(b.chord_radius_mm[0], .4)
    c = enclosure(field_error_tesla=0., position_error_mm=0.)
    zero = c.balls([0], [2.], [2.])
    np.testing.assert_array_equal(zero.midpoint_radius_mm, 0.)


def test_discontinuous_seam_is_not_welded_for_tiny_nonzero_by():
    bad = linear_map(offset=(0, 1e-100, .5), lo=(-8, 0, -8), mirror=True)
    with pytest.raises(ValueError, match='discontinuous'):
        enclosure(bad)


def test_returned_actual_interval_owner_agrees_with_quadratic_phase_points():
    e = enclosure(linear_map(np.diag([0, 0, .3])))
    geo = geometry(tsnav.Box([10, 10, 10]),
                   tsnav.Placed(tsnav.Box([1, 1, 1]), [0, 0, 7], np.eye(3)))
    lo, hi = np.array([0., .5, 1., 2.]), np.array([.5, 1., 2., 3.])
    result = e.owners(np.zeros(4, dtype=int), lo, hi, geo, geometry_error_mm=0.)
    assert np.all(result.status == tslocate.VACUUM)
    s = (lo[:, None] + (hi-lo)[:, None]*np.linspace(0, 1, 129)).ravel()
    actual = quadratic_phase_path(s, -1)
    np.testing.assert_array_equal(geo.locate(actual).volume_index, np.repeat(result.volume_index, 129))


def test_map_and_caller_mutations_cannot_change_constructed_enclosure():
    item = linear_map(np.diag([0, 0, .3]))
    x = np.zeros((1, 3))
    e = enclosure(item, x=x)
    before = e.balls([0], [0.], [3.])
    x[:] = 100
    item.B.setflags(write=True)
    item.B[:] = 0
    after = e.balls([0], [0.], [3.])
    np.testing.assert_array_equal(before.midpoint, after.midpoint)
    np.testing.assert_array_equal(before.midpoint_radius_mm, after.midpoint_radius_mm)


@pytest.mark.parametrize('kwargs', [dict(field_error_tesla=-1), dict(position_error_mm=-1),
    dict(length=-1), dict(pdg=11), dict(field_error_tesla=np.nan), dict(length=[1, 2]),
    dict(x=[[1e16, 0, 0]], length=.1)])
def test_invalid_initial_contracts(kwargs):
    with pytest.raises(ValueError):
        enclosure(**kwargs)


def test_invalid_maps_queries_and_controls():
    with pytest.raises(ValueError, match='single FieldMap'):
        enclosure(item=lambda x: x)
    for chunk in (0, True, 1.5, 1000001):
        with pytest.raises(ValueError):
            enclosure(chunk_nodes=chunk)
    e = enclosure()
    for rows, s in (([0.], [1.]), ([-1], [1.]), ([1], [1.]), ([0], [-1.]), ([0], [4.]), ([0], [np.nan])):
        with pytest.raises(ValueError):
            e.model_error(rows, s)
    with pytest.raises(ValueError, match='ordered'):
        e.balls([0], [2.], [1.])
    with pytest.raises(ValueError, match='Partition'):
        e.owners([0], [0.], [1.], object(), geometry_error_mm=0.)


def test_empty_batch_and_reordered_heterogeneous_sources():
    e = enclosure(x=np.empty((0, 3)), p3=np.empty((0, 3)), length=np.empty(0))
    balls = e.balls(np.empty(0, dtype=int), np.empty(0), np.empty(0))
    assert balls.midpoint.shape == (0, 3)
    e = enclosure(linear_map(np.diag([0, 0, .3])), x=np.zeros((2, 3)),
                  p3=[[tsstep.K, 0, 0], [2*tsstep.K, 0, 0]], length=[1, 2],
                  field_error_tesla=[.02, .04], position_error_mm=[.01, .03])
    np.testing.assert_allclose(e.model_error([1, 0], [2., 1.]), [.24, .06])
    b = e.balls([1, 0], [0., 0.], [2., 1.])
    np.testing.assert_allclose(b.midpoint_radius_mm, [1+.035+.03, .5+.00875+.01])
