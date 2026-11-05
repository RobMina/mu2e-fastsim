"""Exact rational ideal-grid oracles; no native or transport certification."""

from fractions import Fraction as F
import itertools

import numpy as np
import pytest

from beamfastsim import tsfield, tsquery


def f(x):
    return F.from_float(float(x))


def item(*, reflect=False, lo=(0., 0., 0.), shift=(0., 0., 0.), spacing=(1., 1., 1.), nodes=None):
    if nodes is None:
        nodes = np.arange(4*4*4*3, dtype=np.float64).reshape(4, 4, 4, 3)/17
    head = tsfield.GridHeader('synthetic', lo, shift, nodes.shape[:3], spacing, reflect)
    return tsfield.FieldMap(head, nodes)


def exact(field, point):
    coordinates = [f(x) for x in point]
    sign = -1 if coordinates[1] < 0 and field.header.extend_y else 1
    if field.header.extend_y:
        coordinates[1] = abs(coordinates[1])
    q = [(x-f(o)+f(s))/f(d) for x, o, s, d in zip(coordinates, field.header.grid_origin, field.header.origin_shift, field.d)]
    index = [min(int(x//1), int(n)-2) for x, n in zip(q, field.n)]
    fraction = [x-i for x, i in zip(q, index)]
    result = [F(0), F(0), F(0)]
    for corner in itertools.product((0, 1), repeat=3):
        weight = F(1)
        for axis, bit in enumerate(corner):
            weight *= fraction[axis] if bit else 1-fraction[axis]
        node = tuple(a+b for a, b in zip(index, corner))
        for axis in range(3):
            result[axis] += weight*f(field.B[node][axis])
    result[1] *= sign
    return result


def contains(interval, row, vector):
    assert all(f(a) <= v <= f(b) for a, b, v in zip(interval.lo[row], interval.hi[row], vector))


@pytest.mark.parametrize('reflect', [False, True])
def test_nominal_rational_field_and_computed_error(reflect):
    field = item(reflect=reflect, lo=(-.3, 0., -.7), spacing=(.1, .2, .3))
    points = np.array([[-.17, .27, -.33], [-.21, .19, -.19]])
    if reflect:
        points[1, 1] *= -1
    receipt = tsquery.evaluate(field, points)
    for row, point in enumerate(points):
        vector = exact(field, point)
        contains(receipt.nominal_field, row, vector)
        residual2 = sum((v-f(a))**2 for v, a in zip(vector, receipt.value_tesla[row]))
        assert f(receipt.evaluation_error_tesla[row])**2 >= residual2
    assert receipt.query_arithmetic_outward and receipt.stored_grid_domain_checked
    assert not receipt.floating_point_certified and not receipt.native_query_certified
    with pytest.raises(ValueError):
        receipt.value_tesla.setflags(write=True)
    with pytest.raises(ValueError):
        receipt.cell_count.setflags(write=True)


def test_uncertain_box_all_cells_and_independent_rational_points():
    field = item()
    center = np.array([[1.5, 1.5, 1.5]])
    receipt = tsquery.evaluate(field, center, position_radius_mm=.75)
    assert receipt.cell_count[0, 1] == 27
    for point in itertools.product((.75, 1., 1.5, 2., 2.25), repeat=3):
        vector = exact(field, point)
        contains(receipt.region_field, 0, vector)
        assert f(receipt.region_error_tesla[0])**2 >= sum((v-f(a))**2 for v, a in zip(vector, receipt.value_tesla[0]))
    assert receipt.region_error_tesla[0] >= receipt.evaluation_error_tesla[0]
    assert np.all(receipt.position_box.lo <= center-.75)
    assert np.all(receipt.position_box.hi >= center+.75)


def test_reflection_crossing_retains_discontinuity_and_gap_refused():
    field = item(reflect=True, nodes=np.broadcast_to([1., 5., -2.], (4, 4, 4, 3)))
    receipt = tsquery.evaluate(field, np.array([[1.5, .125, 1.5]]), position_radius_mm=.25)
    for y in (-.1, 0., .3):
        contains(receipt.region_field, 0, exact(field, [1.5, y, 1.5]))
    assert receipt.region_field.lo[0, 1] <= -5 and receipt.region_field.hi[0, 1] >= 5
    with pytest.raises(ValueError, match='domain cover'):
        tsquery.evaluate(item(reflect=True, lo=(0., 1., 0.)), np.array([[1.5, .5, 1.5]]))


def test_nominal_rounded_upper_node_is_not_silently_clipped_to_exact_domain():
    field = item(spacing=(.1, .1, .1))
    point = field.hi[None, :]
    assert np.all(field.coverage(point))
    with pytest.raises(ValueError, match='domain cover'):
        tsquery.evaluate(field, point)
    with pytest.raises(ValueError, match='domain cover'):
        tsquery.evaluate(field, np.array([[.01, .1, .1]]), position_radius_mm=.02)


def test_work_cap_counts_whole_batch_and_region_not_just_nominal():
    points = np.array([[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]])
    with pytest.raises(ValueError, match='work cap'):
        tsquery.evaluate(item(), points, max_cells=3)
    with pytest.raises(ValueError, match='work cap'):
        tsquery.evaluate(item(), points[:1], position_radius_mm=.75, max_cells=27)
    assert tsquery.evaluate(item(), points[:1], position_radius_mm=.75, max_cells=28).cell_count.sum() == 28


def test_shift_cancellation_empty_rows_and_zero_map():
    field = item(lo=(1e10+.125, 1e10+.25, 1e10+.5), shift=(1e10,)*3)
    points = np.array([[1., 1., 1.]])
    receipt = tsquery.evaluate(field, points)
    contains(receipt.nominal_field, 0, exact(field, points[0]))
    empty = tsquery.evaluate(field, np.empty((0, 3)))
    assert empty.cell_count.shape == (0, 2) and empty.evaluation_error_tesla.shape == (0,)
    zero = item(nodes=np.zeros((4, 4, 4, 3)))
    receipt = tsquery.evaluate(zero, np.array([[1.5, 1.5, 1.5]]))
    assert np.all(receipt.nominal_field.lo <= 0) and np.all(receipt.nominal_field.hi >= 0)


@pytest.mark.parametrize('kwargs', [dict(max_cells=True), dict(max_cells=0), dict(max_cells=1000001),
    dict(position_radius_mm=-1.), dict(position_radius_mm=np.nan), dict(position_radius_mm=1)])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        tsquery.evaluate(item(), np.array([[1., 1., 1.]]), **kwargs)


@pytest.mark.parametrize('points', [np.array([1., 1., 1.]), np.array([[1, 1, 1]]), np.array([[np.inf, 1., 1.]])])
def test_invalid_points(points):
    with pytest.raises(ValueError):
        tsquery.evaluate(item(), points)


def test_all_clipped_cell_vertices_nonmonotone_signed_field():
    nodes = (((np.arange(192)*73) % 101)-50).astype(np.float64).reshape(4, 4, 4, 3)/7
    field = item(nodes=nodes)
    receipt = tsquery.evaluate(field, np.array([[1.5, 1.5, 1.5]]), position_radius_mm=.75)
    for point in itertools.product((.75, 1., 2., 2.25), repeat=3):
        contains(receipt.region_field, 0, exact(field, point))


@pytest.mark.parametrize('y', [-.0, 0., -.5, .5])
def test_reflected_zero_boundary_and_negative_nominal(y):
    field = item(reflect=True, nodes=np.broadcast_to([1., 5., -2.], (4, 4, 4, 3)))
    point = np.array([[1.5, y, 1.5]])
    receipt = tsquery.evaluate(field, point)
    contains(receipt.nominal_field, 0, exact(field, point[0]))


def test_mixed_radii_cell_faces_and_exact_endpoints():
    field = item(reflect=True)
    points = np.array([[0., 0., 0.], [3., 3., 3.], [1., -.5, 1.],
                       [np.nextafter(1., 0.), 1.5, 1.5], [np.nextafter(1., 2.), 1.5, 1.5]])
    receipt = tsquery.evaluate(field, points, position_radius_mm=np.array([0., 0., .125, 0., 0.]))
    for row, point in enumerate(points):
        contains(receipt.nominal_field, row, exact(field, point))


def test_subnormal_spacing_and_nodes_and_overflow_refusal():
    tiny = np.nextafter(0., 1.)
    field = item(spacing=(tiny,)*3, nodes=np.full((4, 4, 4, 3), tiny))
    point = np.full((1, 3), tiny)
    receipt = tsquery.evaluate(field, point)
    contains(receipt.nominal_field, 0, exact(field, point[0]))
    huge = item(nodes=np.full((4, 4, 4, 3), np.finfo(float).max))
    with pytest.raises(ValueError):
        tsquery.evaluate(huge, np.array([[1.5, 1.5, 1.5]]), position_radius_mm=.5)
