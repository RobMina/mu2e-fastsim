"""Rational stored-transform/CSG oracles, not native geometry validation."""

from fractions import Fraction as F
import itertools

import numpy as np
import pytest

from beamfastsim import tsgeomiv as g, tsinterval as iv, tslocate, tsnav, tssolid, tssurface


def f(v):
    return F.from_float(float(v))


def contains(bound, row, value):
    assert f(bound.lo[row]) <= value <= f(bound.hi[row])


def box(points, radius=0.):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return iv.inflate(iv.Interval(points), iv.Interval(np.float64(radius))) if radius else iv.Interval(points)


def pullback(points, origin, rotation):
    return [sum((f(points[k])-f(origin[k]))*f(rotation[k, j]) for k in range(3)) for j in range(3)]


def test_nested_noncommuting_stored_affine_box_rational_wholebox_vertices():
    r1 = np.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])
    r2 = np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
    shape = tsnav.Placed(tsnav.Placed(tsnav.Box([4., 3., 2.]), [1., -.5, .25], r2), [-2., .25, .5], r1)
    points = box([[0., 0., 0.], [2., -3., 1.]], .125)
    result = g.evaluate(shape, points)
    for row in range(2):
        for bits in itertools.product((0, 1), repeat=3):
            p = [points.hi[row, j] if b else points.lo[row, j] for j, b in enumerate(bits)]
            p = pullback(p, shape.origin, shape.rotation)
            p = [sum((p[k]-f(shape.solid.origin[k]))*f(shape.solid.rotation[k, j]) for k in range(3)) for j in range(3)]
            exact = min(f(a)-abs(v) for a, v in zip(shape.solid.solid.half_lengths, p))
            contains(result.clearance, row, exact)
    assert result.visited_nodes == 3 and result.stored_expression_arithmetic_outward
    assert not result.unit_lipschitz_certified and not result.native_query_certified


@pytest.mark.parametrize('shape,points,expected', [
    (tsnav.Box([1., 2., 3.]), [[0., 0., 0.], [2., 0., 0.]], [1., -1.]),
    (tsnav.Sphere(5.), [[3., 4., 0.], [0., 0., 0.]], [0., 5.]),
    (tsnav.Tube(5., 10., 1.), [[3., 4., 0.], [0., 0., 0.]], [0., -1.]),
    (tsnav.Torus(4., 1.), [[4., 0., 0.], [0., 0., 0.]], [1., -3.]),
    (tssolid.Cone(3., 3., 5., 1., 1.), [[0., 0., 0.], [2., 0., 0.]], [-1., 1.]),
    (tsnav.HalfSpace([1., 0., 0.], 2.), [[1., 0., 0.], [3., 0., 0.]], [1., -1.]),
])
def test_simple_primitives_exact_oracles(shape, points, expected):
    result = g.evaluate(shape, box(points)).clearance
    for row, value in enumerate(expected):
        contains(result, row, f(value))


def test_slope_cone_exact_squared_inequality_and_corner_interval():
    shape = tssolid.Cone(1., 3., 2.)
    result = g.evaluate(shape, box([[0., 0., 0.]])).clearance
    assert result.lo[0] > 0
    assert f(result.lo[0])**2 <= F(16, 5) <= f(result.hi[0])**2
    shell = tssolid.Cone(3., 5., 2., 1., 2.)
    values = g.evaluate(shell, box([[0., 0., 0.]], .125)).clearance
    assert values.hi[0] < 0


def test_csg_splitunion_exact_selected_points_and_nonzero_gap():
    a, b = tsnav.Box([2., 2., 2.]), tsnav.Sphere(1.)
    shapes = [tsnav.Intersection(a, b), tsnav.Union(a, b), tsnav.Difference(a, b),
              tsnav.SplitUnion(a, a, tsnav.HalfSpace([1., 0., 0.], 0.))]
    for shape in shapes:
        result = g.evaluate(shape, box([[0., 0., 0.], [3., 0., 0.]])).clearance
        for row, v in enumerate(shape.clearance(np.array([[0., 0., 0.], [3., 0., 0.]]))):
            contains(result, row, f(v))
    first = tsnav.Placed(tsnav.Box([1., 1., 1.]), [-1.125, 0., 0.], np.eye(3))
    second = tsnav.Placed(tsnav.Box([1., 1., 1.]), [1.125, 0., 0.], np.eye(3))
    gap = g.evaluate(tsnav.Union(first, second), box([[0., 0., 0.]], .0625)).clearance
    assert gap.hi[0] < 0


def test_sector_stored_normals_not_trigonometric_assumption():
    shape = tssolid.Sector(tsnav.Sphere(10.), .2, 4.)
    points = np.array([[1., 2., 0.], [-2., 1., 0.], [1., -1., 0.]])
    result = g.evaluate(shape, iv.Interval(points)).clearance
    for row, point in enumerate(points):
        sides = [sum(f(a)*f(b) for a, b in zip(point[:2], normal)) for normal in shape._normals]
        angular = max(sides)
        assert angular < 5
        contains(result, row, angular)


def test_polygon_exact_distances_concavity_winding_and_uncertain_sign():
    vertices = [[0., 0.], [4., 0.], [4., 1.], [2., 1.], [2., 3.], [0., 3.]]
    for data in (vertices, vertices[::-1]):
        shape = tssolid.Extrusion(data, 2.)
        points = box([[1., .5, 0.], [3., 2., 0.], [1., 1., 3.], [0., 0., 0.]])
        result = g.evaluate(shape, points).clearance
        for row, expected in enumerate((.5, -1., -1., 0.)):
            contains(result, row, f(expected))
        uncertain = g.evaluate(shape, box([[2., 2., 0.]], .125)).clearance
        assert uncertain.lo[0] < 0 < uncertain.hi[0]
        interior = g.evaluate(shape, box([[1., .5, 0.]], .125)).clearance
        assert interior.lo[0] > 0


def partition(solid):
    mother = tslocate.Volume(tsnav.Box([100., 100., 100.]), 'vac', 'mother', 'cell', True)
    leaf = tslocate.Volume(solid, 'metal', 'daughter', 'cell', False, instance=2, instance_path=(1, 2))
    return tslocate.Partition((tslocate.Cell(mother, (leaf,)),))


def test_complete_faces_guard_and_pullback_plane_difference_rational():
    r1 = np.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])
    r2 = np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
    shape = tsnav.Placed(tsnav.Placed(tsnav.Box([4., 3., 2.]), [1., -.5, .25], r2), [-2., .25, .5], r1)
    for face in tssurface.faces(partition(shape), 1):
        points = box(face.origin[None, :], .01)
        result = g.face_bounds(face, points)
        assert result.face is face and result.guard.solid is face.guard
        assert result.primitive_plane.lo[0] <= 0 <= result.primitive_plane.hi[0]
        assert result.plane_discrepancy_upper_mm[0] < 1e-10
        assert result.normal_difference_upper < 1e-12
        assert not result.native_query_certified
    tube = tsnav.Placed(tssolid.Sector(tsnav.Tube(5., .125, 1.), 0., 2*np.pi), [0., 0., 0.], np.eye(3))
    for face in tssurface.faces(partition(tube), 1):
        result = g.face_bounds(face, box([[2., 0., face.origin[2]-.01]], .001))
        assert result.guard.clearance.lo[0] > 0
        assert result.plane_discrepancy_upper_mm[0] < 1e-12


def test_not_exactly_orthogonal_matrix_discrepancy_is_not_zero():
    r = np.diag([1.+2e-13, 1., 1.])
    face = next(fa for fa in tssurface.faces(partition(tsnav.Placed(tsnav.Box([2., 2., 2.]), [0., 0., 0.], r)), 1) if fa.key == 'x-')
    result = g.face_bounds(face, box([[-2., 0., 0.]]))
    assert result.plane_discrepancy_upper_mm[0] > 1e-13
    assert result.plane_discrepancy_upper_mm[0] < 1e-10


@pytest.mark.parametrize('kwargs', [dict(max_nodes=0), dict(max_nodes=True), dict(max_depth=0), dict(max_depth=129)])
def test_bad_controls(kwargs):
    with pytest.raises(ValueError):
        g.evaluate(tsnav.Sphere(1.), box([[0., 0., 0.]]), **kwargs)


def test_unknown_work_caps_empty_and_unqualified_callables():
    shape = tsnav.Placed(tsnav.Sphere(1.), [0., 0., 0.], np.eye(3))
    with pytest.raises(ValueError, match='cap'):
        g.evaluate(shape, box([[0., 0., 0.]]), max_depth=1)
    with pytest.raises(ValueError, match='cap'):
        g.evaluate(shape, box([[0., 0., 0.]]), max_nodes=1)
    with pytest.raises(ValueError, match='unsupported'):
        g.evaluate(lambda p: np.ones(len(p)), box([[0., 0., 0.]]))
    with pytest.raises(ValueError):
        g.evaluate(shape, np.zeros((1, 3)))
    empty = g.evaluate(shape, box(np.empty((0, 3))))
    assert empty.clearance.lo.shape == (0,)
    with pytest.raises(ValueError):
        empty.clearance.lo.setflags(write=True)


@pytest.mark.parametrize('reverse', [False, True])
def test_slanted_polygon_exact_squared_distance_and_winding_box_samples(reverse):
    vertices = np.array([[0., 0.], [3., 1.], [2., 4.], [-1., 2.]])
    if reverse:
        vertices = vertices[::-1]
    shape = tssolid.Extrusion(vertices, 10.)
    points = box([[1., 1., 0.], [3., 3., 0.], [-1., 0., 0.]], .0625)
    result = g.evaluate(shape, points).clearance
    for row in range(3):
        for weights in itertools.product((F(0), F(1, 2), F(1)), repeat=2):
            p = [f(points.lo[row, j])*(1-w)+f(points.hi[row, j])*w for j, w in enumerate(weights)]
            distances, inside = [], False
            for aa, bb in zip(vertices, np.roll(vertices, -1, axis=0)):
                a, b = [f(v) for v in aa], [f(v) for v in bb]
                edge, rel = [y-x for x, y in zip(a, b)], [x-y for x, y in zip(p, a)]
                t = min(F(1), max(F(0), sum(v*w for v, w in zip(rel, edge))/sum(v*v for v in edge)))
                distances.append(sum((v-t*w)**2 for v, w in zip(rel, edge)))
                crosses = (a[1] > p[1]) != (b[1] > p[1])
                cross = edge[0]*rel[1]-edge[1]*rel[0]
                inside ^= crosses and ((cross > 0) == (edge[1] > 0))
            distance2 = min(distances)
            lo, hi = f(result.lo[row]), f(result.hi[row])
            if inside:
                assert hi >= 0 and hi**2 >= distance2
                assert lo <= 0 or lo**2 <= distance2
            else:
                assert lo <= 0 and lo**2 >= distance2
                assert hi >= 0 or hi**2 <= distance2


def test_affine_nonorthogonal_face_plane_exact_rational_discrepancy():
    scale = 1.+2e-13
    solid = tsnav.Placed(tsnav.Box([2., 2., 2.]), [.25, -.5, 1.], np.diag([scale, 1., 1.]))
    face = next(fa for fa in tssurface.faces(partition(solid), 1) if fa.key == 'x-')
    points = box([[-2., 0., 0.], [1., 0., 0.]], .125)
    result = g.face_bounds(face, points)
    offset = -f(solid.origin[0])+2/f(scale)
    stored_offset = -f(face.origin[0])
    exact_difference = abs(offset-stored_offset)
    for row in range(2):
        assert f(result.plane_discrepancy_upper_mm[row]) >= exact_difference > 0
        contains(result.primitive_plane, row, f(points.lo[row, 0])+offset)
        contains(result.primitive_plane, row, f(points.hi[row, 0])+offset)


def test_polygon_edge_work_cap_and_arithmetic_overflow_refused():
    shape = tssolid.Extrusion([[0., 0.], [2., 0.], [2., 2.], [0., 2.]], 1.)
    with pytest.raises(ValueError, match='cap'):
        g.evaluate(shape, box([[1., 1., 0.]]), max_nodes=4)
    assert g.evaluate(shape, box([[1., 1., 0.]]), max_nodes=5).visited_nodes == 5
    solid = tsnav.Placed(tsnav.Box([1., 1., 1.]), [-np.finfo(float).max, 0., 0.], np.eye(3))
    with pytest.raises(ValueError):
        g.evaluate(solid, box([[np.finfo(float).max, 0., 0.]]))


def test_sheared_face_plane_discrepancy_rational():
    shear = 2e-13
    rotation = np.array([[1., shear, 0.], [0., 1., 0.], [0., 0., 1.]])
    shape = tsnav.Placed(tsnav.Box([2., 2., 2.]), [0., 0., 0.], rotation)
    face = next(fa for fa in tssurface.faces(partition(shape), 1) if fa.key == 'x-')
    points = box([[-2., -1., 0.], [0., 1., 0.]], .125)
    result = g.face_bounds(face, points)
    unit2 = sum(f(v)**2 for v in face.inward_normal)
    lo, hi = f(np.nextafter(np.sqrt(float(unit2)), -np.inf)), f(np.nextafter(np.sqrt(float(unit2)), np.inf))
    assert lo*lo <= unit2 <= hi*hi
    for row in range(2):
        for bits in itertools.product((0, 1), repeat=3):
            p = [f(points.hi[row, j] if bit else points.lo[row, j]) for j, bit in enumerate(bits)]
            numerator = sum((v-f(o))*f(n) for v, o, n in zip(p, face.origin, face.inward_normal))
            stored_lo, stored_hi = sorted((numerator/lo, numerator/hi))
            primitive = p[0]+2
            upper = max(abs(primitive-stored_lo), abs(primitive-stored_hi))
            assert f(result.plane_discrepancy_upper_mm[row]) >= upper
            contains(result.primitive_plane, row, primitive)
    assert result.normal_difference_upper < 1e-12


def test_nested_face_plane_discrepancy_sampled_exact_rational_normalization():
    import math
    r1 = np.array([[.6, -.8, 0.], [.8, .6, 0.], [0., 0., 1.]])
    r2 = np.array([[1., 0., 0.], [0., .8, -.6], [0., .6, .8]])
    shape = tsnav.Placed(tsnav.Placed(tsnav.Box([4., 3., 2.]), [1., -.5, .25], r2), [-2., .25, .5], r1)
    matrix = [[sum(f(r1[i, k])*f(r2[k, j]) for k in range(3)) for j in range(3)] for i in range(3)]
    offset = [-f(v) for v in shape.origin]
    offset = [sum(offset[k]*f(r1[k, j]) for k in range(3))-f(shape.solid.origin[j]) for j in range(3)]
    offset = [sum(offset[k]*f(r2[k, j]) for k in range(3)) for j in range(3)]
    def root_bracket(square):
        scale = 2**200
        integer = math.isqrt(square.numerator*scale*scale//square.denominator)
        lo, hi = F(integer, scale), F(integer+1, scale)
        assert lo*lo <= square <= hi*hi
        return lo, hi
    for face in tssurface.faces(partition(shape), 1):
        axis, sign = 'xyz'.index(face.key[0]), 1 if face.key[1] == '-' else -1
        normal = [sign*matrix[j][axis] for j in range(3)]
        constant = sign*offset[axis]+f(shape.solid.solid.half_lengths[axis])
        a, b = root_bracket(sum(v*v for v in normal))
        c, d = root_bracket(sum(f(v)**2 for v in face.inward_normal))
        points = box(face.origin[None, :], .125)
        result = g.face_bounds(face, points)
        for bits in itertools.product((0, 1), repeat=3):
            p = [f(points.hi[0, j] if bit else points.lo[0, j]) for j, bit in enumerate(bits)]
            primitive = sum(v*n for v, n in zip(p, normal))+constant
            stored = sum((v-f(o))*f(n) for v, o, n in zip(p, face.origin, face.inward_normal))
            plo, phi = sorted((primitive/a, primitive/b))
            slo, shi = sorted((stored/c, stored/d))
            assert f(result.plane_discrepancy_upper_mm[0]) >= max(abs(plo-shi), abs(phi-slo))
            assert f(result.primitive_plane.lo[0]) <= plo <= phi <= f(result.primitive_plane.hi[0])
