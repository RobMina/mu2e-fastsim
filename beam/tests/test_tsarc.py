"""Curved ownership enclosures against analytic uniform helices, not native G4."""

import json

import numpy as np
import pytest

from beamfastsim import tsarc as a, tslocate as l, tsnav as g, tsstep


def geometry(*obstacles, mother=None):
    mother = g.Box([10, 10, 10]) if mother is None else mother
    root = l.Volume(mother, 'vac', 'mother', 'root', True)
    daughters = tuple(l.Volume(v, f'metal{i}', f'leaf{i}', 'root', False)
                      for i, v in enumerate(obstacles))
    return l.Partition((l.Cell(root, daughters),))


def curve(x=None, p=None, pdg=13, field=(0, 0, 1), length=1., error=0.):
    x = [[0., 0, 0]] if x is None else x
    p = [[tsstep.K, 0, 0]] if p is None else p
    return a.UniformHelix(x, p, pdg, field, length, position_error_mm=error)


def run(c, geo=None, **kwargs):
    return a.partition(c, geometry() if geo is None else geo, geometry_error_mm=kwargs.pop('geometry_error_mm', 0.), **kwargs)


def assert_complete(result, lengths):
    for i, length in enumerate(lengths):
        use = result.source_index == i
        lo, hi = result.s_lo[use], result.s_hi[use]
        assert lo[0] == 0 and hi[-1] == length
        np.testing.assert_array_equal(hi[:-1], lo[1:])
        np.testing.assert_allclose(np.sum(result.width[use]), length, rtol=2e-15)
    unresolved = result.status != a.RESOLVED
    np.testing.assert_array_equal(result.volume_index[unresolved], -1)
    np.testing.assert_array_equal(result.ball_kind[unresolved], 0)
    assert np.all(result.clearance_margin[~unresolved] > 0)


def test_uniform_helix_charge_species_analytic_and_unwrapped_phase():
    c = curve(x=np.zeros((4, 3)), p=np.tile([tsstep.K, 0, 0], (4, 1)),
              pdg=[13, -13, 211, -211], length=2*np.pi)
    x, p = c.state(np.arange(4), np.full(4, np.pi / 2))
    np.testing.assert_allclose(x, [[1, 1, 0], [1, -1, 0], [1, -1, 0], [1, 1, 0]], atol=5e-16)
    np.testing.assert_allclose(p[:, 0], 0, atol=5e-16)
    np.testing.assert_allclose(p[:, 1], [tsstep.K, -tsstep.K, -tsstep.K, tsstep.K])
    x, _ = c.state(np.arange(4), np.full(4, 2*np.pi))
    np.testing.assert_allclose(x, 0, atol=1e-15)
    result = run(c)
    assert_complete(result, np.full(4, 2*np.pi))
    assert np.all(result.status == a.RESOLVED)


def test_two_distinct_centred_balls_enclose_dense_uniform_arc_samples():
    rng = np.random.default_rng(4812)
    n = 128
    c = curve(x=rng.uniform(-2, 2, (n, 3)), p=rng.normal(size=(n, 3)) * 2,
              pdg=rng.choice([13, -13, 211, -211], n), field=rng.normal(size=(n, 3)),
              length=rng.uniform(0, 8, n), error=0.)
    index = np.arange(n)
    middle, midpoint, rm, center, rc = a._enclosures(c, index, np.zeros(n), c.length)
    samples = 129
    rows = np.repeat(index, samples)
    s = (c.length[:, None] * np.linspace(0, 1, samples)).ravel()
    points, _ = c.state(rows, s)
    distance_m = np.linalg.norm(points - np.repeat(midpoint, samples, axis=0), axis=1)
    distance_c = np.linalg.norm(points - np.repeat(center, samples, axis=0), axis=1)
    assert np.all(distance_m <= np.repeat(rm, samples) + 2e-14)
    assert np.all(distance_c <= np.repeat(rc, samples) + 2e-14)
    assert np.any(np.linalg.norm(midpoint - center, axis=1) > .1)
    np.testing.assert_allclose(middle, c.length / 2)


def test_full_turn_same_vacuum_endpoints_does_not_skip_intervening_material():
    c = curve(length=2*np.pi)
    obstacle = g.Placed(g.Sphere(.1), [0, 2, 0], np.eye(3))
    geo = geometry(obstacle)
    result = run(c, geo, atol=1e-5)
    assert_complete(result, [2*np.pi])
    assert np.any(result.volume_index == 1)
    uncertain = result.status != a.RESOLVED
    assert uncertain.any()
    assert np.all(result.width[uncertain] <= 1e-5)
    expected = 4 * np.arcsin(.05)
    measured = np.sum(result.width[result.volume_index == 1])
    assert abs(measured - expected) <= np.sum(result.width[uncertain]) + 1e-14
    inside = geo.locate(c.state(np.array([0, 0]), np.array([0., 2*np.pi]))[0])
    np.testing.assert_array_equal(inside.status, l.VACUUM)


def test_resolved_intervals_have_consistent_dense_sample_owners():
    c = curve(length=4.)
    geo = geometry(g.Placed(g.Sphere(.25), [.9, .8, 0], np.eye(3)))
    result = run(c, geo, atol=1e-5)
    safe = result.status == a.RESOLVED
    n = 17
    s = (result.s_lo[safe, None] + result.width[safe, None] * np.linspace(0, 1, n)).ravel()
    points, _ = c.state(np.repeat(result.source_index[safe], n), s)
    locations = geo.locate(points)
    np.testing.assert_array_equal(locations.volume_index, np.repeat(result.volume_index[safe], n))


def test_tangent_contact_retained_even_when_owner_does_not_change():
    c = curve(length=2*np.pi)
    geo = geometry(g.Placed(g.Sphere(1), [0, 3, 0], np.eye(3)))
    result = run(c, geo, atol=1e-4)
    assert_complete(result, [2*np.pi])
    assert not np.any(result.volume_index == 1)
    use = result.status != a.RESOLVED
    assert np.any(use & (result.s_lo <= np.pi) & (result.s_hi >= np.pi))


def test_error_floors_reduce_certified_ownership_never_inflate_geometry():
    geo = geometry(mother=g.HalfSpace([0, 1, 0], 1.))
    c = curve(field=[0, 0, 0], length=.5)
    good = run(c, geo, max_depth=0)
    assert good.status[0] == a.RESOLVED
    bad = run(curve(field=[0, 0, 0], length=.5, error=2.), geo, max_depth=0)
    assert bad.status[0] == a.DEPTH_LIMIT and bad.volume_index[0] == -1
    bad = run(c, geo, geometry_error_mm=2., max_depth=0)
    assert bad.status[0] == a.DEPTH_LIMIT
    np.testing.assert_array_equal(c.position_error_mm, 0)


def test_zero_field_zero_path_parallel_field_and_mutation_contracts():
    original = np.array([[1., 2, 3]])
    c = curve(x=original, p=[[1, 0, 0]], field=[0, 0, 0], length=2.)
    original[:] = 99
    np.testing.assert_allclose(c.state(np.array([0]), np.array([2.]))[0], [[3, 2, 3]])
    assert_complete(run(c), [2.])
    parallel = curve(p=[[0, 0, 2]], length=2.)
    np.testing.assert_allclose(parallel.state(np.array([0]), np.array([2.]))[0], [[0, 0, 2]])
    zero = run(curve(length=0.))
    assert zero.status[0] == a.RESOLVED and zero.width[0] == 0
    for array in (c.x, c.direction, c.field, c.length, c.curvature, c.position_error_mm):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    empty = curve(x=np.empty((0, 3)), p=np.empty((0, 3)), length=np.empty(0))
    assert run(empty).status.size == 0


def test_boundary_following_outside_scope_and_resource_caps_stay_unresolved():
    geo = geometry(mother=g.HalfSpace([0, 0, 1], 0))
    result = run(curve(length=.01), geo, atol=.001)
    assert np.all(result.status == a.WIDTH_LIMIT)
    assert np.all(result.midpoint_status == l.BOUNDARY)
    assert_complete(result, [.01])
    with pytest.raises(ValueError, match='max_intervals'):
        run(curve(length=1.), geo, atol=1e-6, max_intervals=8)
    out = run(curve(x=[[0, 0, 1]], length=.1), geo, max_depth=0)
    assert out.status[0] == a.DEPTH_LIMIT
    assert out.midpoint_status[0] == l.OUTSIDE_SCOPE


@pytest.mark.parametrize('kwargs', [
    {'length': -1}, {'length': np.inf}, {'length': 65*np.pi}, {'pdg': 11},
    {'p': [[0, 0, 0]]}, {'p': [[np.inf, 0, 0]]}, {'x': [0, 0, 0]},
    {'field': lambda x: x}, {'field': [1e308, 1e308, 0]}, {'field': [1e-300, 0, 0]},
    {'error': -1}, {'error': np.nan},
])
def test_invalid_model_inputs_fail_closed(kwargs):
    with pytest.raises(ValueError):
        curve(**kwargs)


@pytest.mark.parametrize('kwargs', [{'atol': 0}, {'geometry_error_mm': -1},
                                   {'max_depth': 65}, {'max_depth': True},
                                   {'max_intervals': 1000001}, {'max_intervals': 0}])
def test_invalid_partition_controls(kwargs):
    with pytest.raises(ValueError):
        run(curve(), **kwargs)


def test_invalid_state_queries():
    c = curve()
    for source, s in (([0.], [0.]), ([-1], [0.]), ([1], [0.]), ([0], [-1.]), ([0], [2.])):
        with pytest.raises(ValueError):
            c.state(np.array(source), np.array(s))


def test_cli_retains_conditional_error_assumptions_and_output_alias_guard(tmp_path):
    from test_tsgeom import fixture
    path = fixture(tmp_path)
    out = tmp_path / 'arc.json'
    args = [str(path), '--root', 'Mother', '--vacuum-material', 'vac', '--arc',
            '110', '20', '30', '100', '0', '0', '13', '4', '--field', '0', '0', '1',
            '--position-error-mm', '0', '--geometry-error-mm', '0', '--vacuum-prefix', '--out', str(out)]
    a.main(args)
    receipt = json.loads(out.read_text())
    assert receipt['errors_are_caller_assumptions']
    assert not receipt['floating_point_certified']
    assert not receipt['native_query_certified']
    assert not receipt['variable_field_transport']
    assert not receipt['material_transport']
    assert receipt['per_arc']['resolved_material_mm'][0] > 0
    assert receipt['per_arc']['unresolved_mm'][0] > 0
    assert receipt['vacuum_prefix']['status'] == [a.BOUNDARY_PENDING]
    assert 0 < receipt['vacuum_prefix']['path'][0] < 1.01
    assert receipt['vacuum_prefix']['next_volume_index'] == [1]
    assert receipt['vacuum_prefix']['neighbor_is_not_crossing_certificate']
    saved = path.read_bytes()
    alias = tmp_path / 'alias.json'
    alias.symlink_to(path)
    with pytest.raises(SystemExit):
        a.main(args[:-1] + [str(alias)])
    assert path.read_bytes() == saved


def test_chord_only_ball_resolution_has_its_own_center_and_positive_margin():
    c = curve(length=1.5)
    middle, m, rm, center, rc = a._enclosures(c, np.array([0]), np.array([0.]), np.array([1.5]))
    separation = np.linalg.norm(m[0] - center[0])
    assert rc[0] < rm[0] + separation
    radius = (rc[0] + rm[0] + separation) / 2
    geo = geometry(mother=g.Placed(g.Sphere(radius), center[0], np.eye(3)))
    out = run(c, geo, max_depth=0)
    assert out.status[0] == a.RESOLVED
    assert out.ball_kind[0] == a.CHORD_BALL
    assert out.volume_index[0] == 0
    np.testing.assert_allclose(out.ball_radius, rc)
    np.testing.assert_allclose(out.clearance_margin, radius - rc)
    assert geo.locate(m).clearance[0] < rm[0]
    assert middle[0] == .75


def test_multiple_turns_nonzero_pitch_reentries_complete_and_no_phase_wrapping():
    p = tsstep.K * np.array([[np.sqrt(.99), 0., .1]])
    c = curve(p=p, length=6*np.pi)
    geo = geometry(g.Placed(g.Tube(.12, 4.), [0, 2, .9], np.eye(3)))
    out = run(c, geo, atol=1e-4)
    assert_complete(out, [6*np.pi])
    assert np.any(out.s_lo[out.volume_index == 1] > 4*np.pi)
    assert np.count_nonzero((out.status != a.RESOLVED) & (out.s_lo > 2*np.pi)) >= 4
    positions, _ = c.state(np.array([0, 0]), np.array([2*np.pi, 6*np.pi]))
    np.testing.assert_allclose(positions[:, 2], [.2*np.pi, .6*np.pi], atol=2e-15)
    assert np.all(out.depth > 0)


def test_heterogeneous_rows_retain_order_coverage_and_error_floor_stops():
    c = curve(x=np.array([[0., 0, 0], [0, 0, 0], [0, 0, 0]]),
              p=np.array([[tsstep.K, 0, 0], [0, 0, 100], [1, 0, 0]]),
              pdg=[13, -211, -13], field=[[0, 0, 1], [0, 0, 2], [0, 0, 0]],
              length=[2*np.pi, .2, 1.], error=[0, 0, 20])
    geo = geometry(g.Placed(g.Sphere(.1), [0, 2, 0], np.eye(3)))
    out = run(c, geo, max_depth=5, atol=1e-6)
    assert_complete(out, [2*np.pi, .2, 1.])
    assert np.all(out.status[out.source_index == 1] == a.RESOLVED)
    assert np.all(out.status[out.source_index == 2] == a.DEPTH_LIMIT)
    assert np.all(out.depth[out.source_index == 2] == 5)
    assert np.all(out.source_index[:-1] <= out.source_index[1:])
    with pytest.raises(ValueError, match='uniform field requires'):
        curve(field=1.)


def test_vacuum_prefix_stops_before_first_gap_and_exposes_only_geometric_neighbor():
    c = curve(length=2*np.pi)
    geo = geometry(g.Placed(g.Sphere(.1), [0, 2, 0], np.eye(3)))
    out = a.vacuum_prefix(c, geo, geometry_error_mm=0., atol=1e-5)
    assert out.status[0] == a.BOUNDARY_PENDING
    contact = np.pi - 2*np.arcsin(.05)
    assert out.path[0] < contact < out.pending_hi[0]
    assert out.pending_hi[0] - out.path[0] <= 2e-5
    assert out.left_volume_index[0] == 0 and out.next_volume_index[0] == 1
    np.testing.assert_allclose(out.x, c.state(np.array([0]), out.path)[0])
    assert geo.locate(out.x).status[0] == l.VACUUM


def test_prefix_tangent_same_owner_is_not_a_passage_certificate():
    c = curve(length=2*np.pi)
    geo = geometry(g.Placed(g.Sphere(1), [0, 3, 0], np.eye(3)))
    out = a.vacuum_prefix(c, geo, geometry_error_mm=0., atol=1e-4)
    assert out.status[0] == a.BOUNDARY_PENDING
    assert out.path[0] <= np.pi <= out.pending_hi[0]
    assert out.left_volume_index[0] == out.next_volume_index[0] == 0


def test_prefix_batch_limits_material_start_and_initial_error_floor():
    geo = geometry(g.Sphere(.25))
    c = curve(x=[[2, 0, 0], [0, 0, 0], [.25, 0, 0]], p=np.tile([1., 0, 0], (3, 1)),
              field=[0, 0, 0], length=[1, 1, 1])
    out = a.vacuum_prefix(c, geo, geometry_error_mm=0.)
    np.testing.assert_array_equal(out.status, [a.PATH_LIMIT, a.START_MATERIAL, a.START_UNRESOLVED])
    np.testing.assert_allclose(out.path, [1, 0, 0])
    np.testing.assert_array_equal(out.next_volume_index, -1)
    uncertain = a.vacuum_prefix(curve(x=[[0, 0, 0]], error=1.), geo,
                                geometry_error_mm=0., max_depth=0)
    assert uncertain.status[0] == a.START_UNRESOLVED
    empty = curve(x=np.empty((0, 3)), p=np.empty((0, 3)), length=np.empty(0))
    assert a.vacuum_prefix(empty, geo, geometry_error_mm=0.).status.size == 0


def test_prefix_multiple_blocked_lanes_do_not_merge_gaps_across_source_rows():
    geo = geometry(g.Placed(g.Box([.125, 5, 5]), [.5, 0, 0], np.eye(3)))
    c = curve(x=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], p=np.tile([1., 0, 0], (3, 1)),
              field=[0, 0, 0], length=[1., .375, .1])
    out = a.vacuum_prefix(c, geo, geometry_error_mm=0., atol=1e-6)
    np.testing.assert_array_equal(out.status, [a.BOUNDARY_PENDING, a.BOUNDARY_PENDING, a.PATH_LIMIT])
    assert out.path[0] < .375 < out.pending_hi[0]
    assert out.path[1] < .375 and out.pending_hi[1] == .375
    assert out.next_volume_index[0] == 1
    assert out.next_volume_index[1] == -1
    assert out.path[2] == .1
    capped = a.vacuum_prefix(c, geo, geometry_error_mm=0., max_depth=0)
    assert capped.path[0] == 0 and capped.left_volume_index[0] == -1


def test_prefix_flattened_owner_identity_includes_coalesced_vacuum_leaf():
    root = l.Volume(g.Box([10, 10, 10]), 'vac', 'mother', 'root', True)
    vd = l.Volume(g.Box([1, 1, 1]), 'vac', 'vd', 'root', True)
    slab = l.Volume(g.Placed(g.Box([.001, 5, 5]), [.5, 0, 0], np.eye(3)),
                    'metal', 'slab', 'root', False)
    geo = l.Partition((l.Cell(root, (vd, slab)),))
    out = a.vacuum_prefix(curve(p=[[1, 0, 0]], field=[0, 0, 0], length=1), geo,
                          geometry_error_mm=0.)
    assert out.status[0] == a.BOUNDARY_PENDING
    assert out.left_volume_index[0] == 0
    assert out.next_volume_index[0] == 2


def test_collapsed_spatial_progress_rejected_but_zero_and_closed_turns_supported():
    c = curve(x=[[1e16, 0, 0]], p=[[1, 0, 0]], field=[0, 0, 0], length=.1)
    np.testing.assert_array_equal(c.state(np.array([0]), np.array([0.]))[0], c.x)
    with pytest.raises(ValueError, match='displacement is not representable'):
        c.state(np.array([0]), np.array([.1]))
    with pytest.raises(ValueError, match='displacement is not representable'):
        a.vacuum_prefix(c, geometry(mother=g.Box([2e16, 2e16, 2e16])), geometry_error_mm=0.)
    closed = curve(x=[[10, 10, 10]], length=4*np.pi)
    positions, _ = closed.state(np.array([0, 0]), np.array([2*np.pi, 4*np.pi]))
    np.testing.assert_allclose(positions, [[10, 10, 10], [10, 10, 10]], atol=2e-15)
