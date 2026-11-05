"""Selected-cell point identities and unresolved cases, not native navigation."""

import numpy as np
import pytest

from beamfastsim import tslocate as l, tsnav as g


def volume(solid, *, material='vac', physical='mother', root='cell', vacuum=True):
    return l.Volume(solid, material, physical, root, vacuum)


def partition(*daughters):
    return l.Partition((l.Cell(volume(g.Box([5, 5, 5])), daughters),))


def test_vacuum_material_boundary_and_unknown_are_distinct():
    metal = volume(g.Tube(2, 3, 1), material='exact-metal0xab', physical='insert', vacuum=False)
    p = partition(metal)
    out = p.locate([[0, 0, 0], [1.5, 0, 0], [2, 0, 0], [5, 0, 0], [6, 0, 0]])
    np.testing.assert_array_equal(out.status, [l.VACUUM, l.MATERIAL, l.BOUNDARY, l.BOUNDARY, l.OUTSIDE_SCOPE])
    np.testing.assert_array_equal(out.volume_index, [0, 1, -1, -1, -1])
    np.testing.assert_array_equal(out.boundary_index, [-1, -1, 1, 0, -1])
    np.testing.assert_allclose(out.clearance, [1, .5, 0, 0, 0])
    assert p.volumes[1].material == 'exact-metal0xab'
    assert p.volumes[1].physical == 'insert'


def test_vacuum_leaf_faces_are_coalesced_and_clipped_by_mother():
    leaf = volume(g.Box([1, 1, 1]), physical='vd')
    out = partition(leaf).locate([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    np.testing.assert_array_equal(out.status, l.VACUUM)
    np.testing.assert_array_equal(out.volume_index, 0)
    np.testing.assert_allclose(out.clearance, [5, 4, 3])
    outside = volume(g.Placed(g.Sphere(1), [10, 0, 0], np.eye(3)), vacuum=False)
    np.testing.assert_array_equal(partition(outside).locate([[10, 0, 0]]).status, l.OUTSIDE_SCOPE)


def test_material_overlaps_and_boundary_intersections_never_choose_priority():
    a = volume(g.Box([2, 2, 2]), material='a', physical='a', vacuum=False)
    b = volume(g.Placed(g.Box([2, 2, 2]), [1, 0, 0], np.eye(3)),
               material='b', physical='b', vacuum=False)
    out = partition(a, b).locate([[0, 0, 0], [-2, 0, 0], [2, 0, 0], [-1.5, 0, 0]])
    np.testing.assert_array_equal(out.status, [l.OVERLAP, l.BOUNDARY, l.BOUNDARY, l.MATERIAL])
    np.testing.assert_array_equal(out.volume_index, [-1, -1, -1, 1])
    np.testing.assert_allclose(out.clearance, [0, 0, 0, .5])
    out = partition(a, a).locate([[2, 0, 0]])
    assert out.boundary_count[0] == 2
    assert out.boundary_index[0] == -1


def test_root_overlap_is_unresolved_even_with_same_vacuum_material():
    a = l.Cell(volume(g.Box([5, 5, 5])))
    b = l.Cell(volume(g.Placed(g.Box([5, 5, 5]), [1, 0, 0], np.eye(3)), root='other'))
    out = l.Partition((a, b)).locate([[0, 0, 0], [-4.5, 0, 0], [6, 0, 0]])
    np.testing.assert_array_equal(out.status, [l.OVERLAP, l.VACUUM, l.BOUNDARY])
    np.testing.assert_array_equal(out.root_count, [2, 1, 0])


def test_material_local_clearance_limits_distance_to_other_leaf():
    a = volume(g.Box([3, 3, 3]), material='a', vacuum=False)
    b = volume(g.Placed(g.Box([1, 1, 1]), [4, 0, 0], np.eye(3)), material='b', vacuum=False)
    out = partition(a, b).locate([[2.9, 0, 0], [3, 0, 0], [3.5, 0, 0]])
    np.testing.assert_array_equal(out.status, [l.MATERIAL, l.BOUNDARY, l.MATERIAL])
    np.testing.assert_allclose(out.clearance, [.1, 0, .5])


def test_empty_and_vector_batch_do_not_mutate_queries():
    p = partition()
    points = np.array([[0., 0, 0], [5, 0, 0], [6, 0, 0]])
    original = points.copy()
    result = p.locate(points)
    np.testing.assert_array_equal(points, original)
    assert result.status.shape == (3,)
    assert p.locate(np.empty((0, 3))).status.shape == (0,)
    for bad in ([0, 0, 0], [[np.nan, 0, 0]], [[True, False, False]], [[1j, 0, 0]]):
        with pytest.raises(ValueError):
            p.locate(bad)


@pytest.mark.parametrize('constructor', [
    lambda: volume(None), lambda: volume(g.Sphere(1), material=''),
    lambda: volume(g.Sphere(1), physical=1), lambda: volume(g.Sphere(1), vacuum=1),
    lambda: l.Cell(volume(g.Sphere(1), vacuum=False)),
    lambda: l.Cell(volume(g.Sphere(1)), (volume(g.Sphere(1), root='other'),)),
    lambda: l.Partition(()), lambda: l.Partition((None,)),
    lambda: l.Partition((l.Cell(volume(g.Sphere(1))), l.Cell(volume(g.Sphere(2))))),
])
def test_invalid_explicit_partition_contracts(constructor):
    with pytest.raises(ValueError):
        constructor()


def test_split_union_eliminates_only_exact_complementary_plane_seam():
    a, b = g.Box([2, 3, 4]), g.Sphere(3)
    h = g.HalfSpace([1, 0, 0], 0)
    joined = g.SplitUnion(a, b, h)
    ordinary = g.Union(g.Intersection(a, h), g.Intersection(b, g.HalfSpace([-1, 0, 0], 0)))
    assert ordinary.clearance([[0, 0, 0]])[0] == 0
    assert joined.clearance([[0, 0, 0]])[0] == 2
    rng = np.random.default_rng(812)
    x, y = rng.uniform(-6, 6, (2, 2000, 3))
    np.testing.assert_array_equal(joined.clearance(x) >= 0, ordinary.clearance(x) >= 0)
    assert np.all(np.abs(joined.clearance(x) - joined.clearance(y)) <= np.linalg.norm(x-y, axis=1) + 1e-13)
    with pytest.raises(ValueError, match='HalfSpace'):
        g.SplitUnion(a, b, a)


def test_split_union_does_not_fill_real_gap_or_general_internal_zero():
    h = g.HalfSpace([1, 0, 0], 0)
    a = g.Intersection(g.Box([3, 3, 3]), g.HalfSpace([1, 0, 0], -1))
    b = g.Intersection(g.Box([3, 3, 3]), g.HalfSpace([-1, 0, 0], -1))
    assert g.SplitUnion(a, b, h).clearance([[0, 0, 0]])[0] == -1
    a = g.Intersection(g.Box([3, 3, 3]), h)
    b = g.Intersection(g.Box([3, 3, 3]), g.HalfSpace([-1, 0, 0], 0))
    assert g.SplitUnion(a, b, h).clearance([[0, 0, 0]])[0] == 0


def test_driver_crosses_exact_split_without_nudge_but_stops_at_real_material():
    from beamfastsim import tsdriver as d
    body = g.Box([5, 5, 5])
    region = g.SplitUnion(body, body, g.HalfSpace([1, 0, 0], 0))
    out = d.propagate([[-1, 0, 0]], [[100, 0, 0]], 13, 2., np.zeros(3), region,
                      mode='no_decay', max_step=.25)
    assert out.status[0] == d.PATH_LIMIT
    np.testing.assert_allclose(out.x, [[1, 0, 0]])
    slab = g.Placed(g.Box([.001, 4, 4]), [.5, 0, 0], np.eye(3))
    region = g.Difference(region, slab)
    out = d.propagate([[-1, 0, 0]], [[100, 0, 0]], 13, 2., np.zeros(3), region,
                      mode='no_decay', max_step=.25, boundary_atol=1e-8)
    assert out.status[0] == d.BOUNDARY_PENDING
    assert out.x[0, 0] < .499
    assert .499 - out.x[0, 0] <= 1e-8


def test_other_root_clearance_and_boundary_guard_even_with_unique_owner():
    a = l.Cell(volume(g.Box([5, 5, 5])))
    b = l.Cell(volume(g.Placed(g.Box([1, 1, 1]), [5, 0, 0], np.eye(3)), root='other'))
    out = l.Partition((a, b)).locate([[3.9, 0, 0], [4, 0, 0]])
    np.testing.assert_array_equal(out.status, [l.VACUUM, l.BOUNDARY])
    np.testing.assert_array_equal(out.volume_index, [0, -1])
    np.testing.assert_allclose(out.clearance, [.1, 0])


@pytest.mark.parametrize('bad', [lambda p: np.zeros((len(p), 1)),
                                lambda p: np.full(len(p), np.nan),
                                lambda p: np.full(len(p), np.inf),
                                lambda p: np.full(len(p), True)])
def test_malformed_child_clearances_fail_closed(bad):
    with pytest.raises(ValueError):
        l.Partition((l.Cell(volume(bad)),)).locate([[0, 0, 0]])


def test_optional_native_instance_identity_is_immutable_and_checked():
    path = [1, 7, 21]
    v = l.Volume(g.Sphere(1), 'vac', None, 'root', True,
                 instance=21, logical='same-logical', instance_path=path)
    path[-1] = 99
    assert v.physical is None
    assert v.instance_path == (1, 7, 21)
    for kwargs in ({'instance': 0}, {'instance': True}, {'instance_path': [1]},
                   {'instance': 2, 'instance_path': [1]}, {'logical': ''}):
        with pytest.raises(ValueError):
            l.Volume(g.Sphere(1), 'vac', None, 'root', True, **kwargs)
