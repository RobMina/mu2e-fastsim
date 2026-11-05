"""Restart/carry regressions for conditional ideal-map bounds, not native closure."""

import numpy as np
import pytest

from beamfastsim import tscarry as c, tsfield, tslocate, tsnav, tsstep, tsvary
from test_tsvary import linear_map, geometry


def context(item=None):
    return c.MapContext(linear_map() if item is None else item)


def state(ctx=None, **kwargs):
    defaults = dict(x=[[0., 0, 0]], direction=[[.8, 0, .6]], momentum=tsstep.K,
                    pdg=13, position_error_mm=0., direction_error=0.)
    defaults.update(kwargs)
    return c.State(context() if ctx is None else ctx, **defaults)


def segment(initial=None, length=1., **kwargs):
    defaults = dict(field_error_tesla=0., position_error_mm=0., direction_error=0.)
    defaults.update(kwargs)
    return c.Segment(state() if initial is None else initial, length, **defaults)


def free_geometry():
    return geometry(tsnav.Box([100, 100, 100]))


def advance(s, geo=None, **kwargs):
    return c.attempt(s, free_geometry() if geo is None else geo,
                     geometry_error_mm=kwargs.pop('geometry_error_mm', 0.), **kwargs)


def exact_varying_path(s, x0, direction, q):
    direction = np.asarray(direction)/np.linalg.norm(direction)
    nodes, weights = np.polynomial.legendre.leggauss(48)
    t = s[:, None]*(nodes+1)/2
    phase = -q*((.5+.3*x0[2])*t+.3*direction[2]*t*t/2)
    ux, uy, uz = direction
    x = x0[0]+s/2*((ux*np.cos(phase)-uy*np.sin(phase)) @ weights)
    y = x0[1]+s/2*((ux*np.sin(phase)+uy*np.cos(phase)) @ weights)
    z = x0[2]+uz*s
    theta = -q*((.5+.3*x0[2])*s+.3*uz*s*s/2)
    u = np.column_stack((ux*np.cos(theta)-uy*np.sin(theta),
                         ux*np.sin(theta)+uy*np.cos(theta), np.full(len(s), uz)))
    return np.column_stack((x, y, z)), u


def test_generalized_polynomial_bound_and_cap_include_initial_position():
    initial = state(context(linear_map(np.diag([0, 0, .3]))), position_error_mm=.2, direction_error=.1)
    seg = segment(initial, length=3., field_error_tesla=.04)
    s = np.array([0., .25, 1., 2., 3.])
    err = seg.errors(np.zeros(5, dtype=int), s)
    np.testing.assert_allclose(err.position_mm, np.minimum(.2+.1*s+.1*s*s/2+.3*s**3/6, .2+2*s))
    np.testing.assert_allclose(err.direction, np.minimum(2., .1+.1*s+.3*s*s/2))
    cap = segment(state(position_error_mm=.25, direction_error=2.), field_error_tesla=1e100)
    np.testing.assert_array_equal(cap.errors([0], [1.]).position_mm, [2.25])
    np.testing.assert_array_equal(cap.errors([0], [1.]).direction, [2.])


def test_zero_initial_error_matches_previous_single_segment_model():
    item = linear_map(np.diag([0, 0, .3]))
    seg = segment(state(context(item)), length=3., field_error_tesla=.02)
    old = tsvary.MapEnclosure([[0, 0, 0]], [[.8*tsstep.K, 0, .6*tsstep.K]], 13, item, 3.,
                             field_error_tesla=.02, position_error_mm=0.)
    s = np.linspace(0, 3, 129)
    rows = np.zeros(len(s), dtype=int)
    np.testing.assert_allclose(seg.errors(rows, s).position_mm, old.model_error(rows, s))
    balls, prior = seg.balls([0], [1.], [2.]), old.balls([0], [1.], [2.])
    np.testing.assert_allclose(balls.midpoint_radius_mm, prior.midpoint_radius_mm)
    np.testing.assert_allclose(balls.chord_radius_mm, prior.chord_radius_mm)


def test_nonzero_actual_initial_offsets_and_directions_both_charges():
    ctx = context(linear_map(np.diag([0, 0, .3])))
    initial = state(ctx, x=np.zeros((4, 3)), direction=np.tile([.8, 0, .6], (4, 1)),
                    pdg=[13, -13, 211, -211], position_error_mm=.2, direction_error=.1)
    seg = segment(initial, length=3.)
    s = np.linspace(0, 3, 257)
    x0, u0 = np.array([0, 0, .2]), np.array([.84, 0, .55])
    u0 = u0/np.linalg.norm(u0)
    assert np.linalg.norm(u0-np.array([.8, 0, .6])) < .1
    rows = np.repeat(np.arange(4), len(s))
    paths = np.tile(s, 4)
    actual = [exact_varying_path(s, x0, u0, q) for q in [-1, 1, 1, -1]]
    actual_x, actual_u = [np.concatenate(a) for a in zip(*actual)]
    nominal_x, nominal_u = seg.reference_state(rows, paths)
    nominal_u /= np.linalg.norm(nominal_u, axis=1)[:, None]
    err = seg.errors(rows, paths)
    assert np.all(np.linalg.norm(actual_x-nominal_x, axis=1) <= err.position_mm+2e-14)
    assert np.all(np.linalg.norm(actual_u-nominal_u, axis=1) <= err.direction+2e-14)
    balls = seg.balls(np.arange(4), np.zeros(4), np.full(4, 3.))
    assert np.all(np.linalg.norm(actual_x.reshape(4, -1, 3)-balls.midpoint[:, None], axis=2)
                  <= balls.midpoint_radius_mm[:, None]+2e-14)
    assert np.all(np.linalg.norm(actual_x.reshape(4, -1, 3)-balls.chord_center[:, None], axis=2)
                  <= balls.chord_radius_mm[:, None]+2e-14)


def test_many_segments_carry_actual_varying_path_both_errors_fixed_momentum():
    ctx = context(linear_map(np.diag([0, 0, .3])))
    initial = state(ctx, x=np.zeros((4, 3)), direction=np.tile([8., 0, 6.], (4, 1)),
                    pdg=[13, -13, 211, -211], position_error_mm=.02, direction_error=.03,
                    source_index=[101, 18, 390, 42])
    current = initial
    for step in range(12):
        seg = segment(current, length=.125, position_error_mm=1e-12, direction_error=2e-12)
        out = advance(seg)
        assert np.all(out.status == c.VACUUM_ADVANCED)
        now = out.state
        np.testing.assert_array_equal(now.momentum, initial.momentum)
        np.testing.assert_array_equal(now.pdg, initial.pdg)
        np.testing.assert_array_equal(now.source_index, initial.source_index)
        assert np.all(now.position_error_mm >= current.position_error_mm)
        assert np.all(now.direction_error >= current.direction_error)
        s = np.array([(step+1)*.125])
        actual = [exact_varying_path(s, [0, 0, .02], [.81, 0, .59], q) for q in [-1, 1, 1, -1]]
        x, u = [np.concatenate(a) for a in zip(*actual)]
        d = now.direction/np.linalg.norm(now.direction, axis=1)[:, None]
        assert np.all(np.linalg.norm(x-now.x, axis=1) <= now.position_error_mm+1e-13)
        assert np.all(np.linalg.norm(u-d, axis=1) <= now.direction_error+1e-13)
        current = now
    np.testing.assert_array_equal(current.path_mm, 1.5)
    assert np.all(current.position_error_mm > .02) and np.all(current.direction_error > .03)


def test_injected_endpoint_errors_are_included_in_the_following_segment(monkeypatch):
    ctx = context(linear_map(offset=(0, 0, 0)))
    initial = state(ctx, direction=[[1., 0, 0]], momentum=100.)
    from mu2efastsim import ehelix
    original = ehelix.helix

    def perturbed(*args):
        x, u = original(*args)
        x = x + np.array([0., .0625, 0])
        u = u + np.array([0., .125, 0])
        return x, u

    monkeypatch.setattr(ehelix, 'helix', perturbed)
    first = advance(segment(initial, position_error_mm=.0625, direction_error=.125)).state
    np.testing.assert_array_equal(first.position_error_mm, [.0625])
    np.testing.assert_array_equal(first.direction_error, [.125])
    next_segment = segment(first, position_error_mm=.0625, direction_error=.125)
    model = next_segment.errors([0], [1.])
    np.testing.assert_array_equal(model.position_mm, [.1875])
    result = advance(next_segment).state
    np.testing.assert_array_equal(result.position_error_mm, [.25])
    np.testing.assert_array_equal(result.direction_error, [.25])
    actual_x, actual_u = np.array([[2., 0, 0]]), np.array([[1., 0, 0]])
    normalized = result.direction/np.linalg.norm(result.direction, axis=1)[:, None]
    assert np.linalg.norm(actual_x-result.x) <= result.position_error_mm[0]
    assert np.linalg.norm(actual_u-normalized) <= result.direction_error[0]
    np.testing.assert_array_equal(result.momentum, initial.momentum)


def test_each_evaluation_error_is_added_at_every_accepted_handoff():
    current = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]],
                    position_error_mm=.125, direction_error=.0625)
    for i in range(4):
        out = advance(segment(current, length=.25, position_error_mm=.03125, direction_error=.015625))
        expected_x = current.position_error_mm + current.direction_error*.25 + .03125
        expected_u = current.direction_error + .015625
        np.testing.assert_array_equal(out.state.position_error_mm, expected_x)
        np.testing.assert_array_equal(out.state.direction_error, expected_u)
        current = out.state
    assert current.path_mm[0] == 1.


def test_zero_length_handoff_retains_initial_errors_adds_evaluation_declarations():
    initial = state(position_error_mm=.125, direction_error=.25, path_mm=4.)
    seg = segment(initial, length=0., position_error_mm=.0625, direction_error=.125)
    out = advance(seg)
    np.testing.assert_array_equal(out.state.position_error_mm, [.1875])
    np.testing.assert_array_equal(out.state.direction_error, [.375])
    np.testing.assert_array_equal(out.state.path_mm, [4.])
    np.testing.assert_array_equal(out.state.x, initial.x)
    for name in ('x', 'direction', 'momentum', 'source_index', 'pdg', 'position_error_mm', 'direction_error'):
        with pytest.raises(ValueError):
            getattr(out.state, name).setflags(write=True)


def test_zero_field_persists_direction_uncertainty_and_fixed_scalar_momentum():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[20, 0, 0]],
                    momentum=1e-310, position_error_mm=.125, direction_error=.25)
    seg = segment(initial, length=1.)
    err = seg.errors([0], [1.])
    np.testing.assert_array_equal(err.position_mm, [.375])
    np.testing.assert_array_equal(err.direction, [.25])
    out = advance(seg)
    np.testing.assert_array_equal(out.state.x, [[1, 0, 0]])
    np.testing.assert_array_equal(out.state.momentum, [1e-310])
    np.testing.assert_array_equal(out.state.position_error_mm, [.375])


def test_opposite_unit_directions_saturate_without_reset():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[3., 0, 0]], direction_error=2.)
    out = advance(segment(initial, length=1.))
    np.testing.assert_array_equal(out.state.direction_error, [2.])
    np.testing.assert_array_equal(out.state.position_error_mm, [2.])
    actual_x = np.array([[-1., 0, 0]])
    assert np.linalg.norm(actual_x-out.state.x) <= out.state.position_error_mm[0]


def test_transactional_vacuum_only_material_and_unresolved_roll_back_entire_rows():
    ctx = context(linear_map(offset=(0, 0, 0)))
    initial = state(ctx, x=[[0, 0, 0], [0, 2, 0], [0, 4, 0]], direction=np.tile([1., 0, 0], (3, 1)),
                    momentum=[10, 20, 30], pdg=[13, -13, -211], path_mm=[10, 20, 30],
                    position_error_mm=[0, .01, 0], direction_error=[0, .02, 0], source_index=[71, 7, 9])
    geo = geometry(tsnav.Box([20, 20, 20]),
        tsnav.Placed(tsnav.Box([2, .75, 1]), [0, 2, 0], np.eye(3)),
        tsnav.Placed(tsnav.Box([.05, .5, 1]), [.5, 4, 0], np.eye(3)))
    out = advance(segment(initial), geo)
    np.testing.assert_array_equal(out.status, [c.VACUUM_ADVANCED, c.MATERIAL_PENDING, c.UNRESOLVED])
    np.testing.assert_array_equal(out.accepted, [True, False, False])
    np.testing.assert_array_equal(out.owner_index, [0, 1, -1])
    for name in ('x', 'direction', 'momentum', 'pdg', 'position_error_mm', 'direction_error', 'path_mm', 'source_index'):
        np.testing.assert_array_equal(getattr(out.state, name)[1:], getattr(initial, name)[1:])
    np.testing.assert_array_equal(out.state.path_mm, [11, 20, 30])
    assert np.all(geo.locate(np.array([[0, 4, 0], [1, 4, 0]])).status == tslocate.VACUUM)
    assert out.status.dtype.kind in 'iu' and out.accepted.dtype.kind == 'b'


def test_pending_whole_interval_does_not_commit_later_vacuum():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]])
    geo = geometry(tsnav.Box([20, 20, 20]),
        tsnav.Placed(tsnav.Box([.05, 1, 1]), [.5, 0, 0], np.eye(3)))
    seg = segment(initial)
    assert geo.locate([[1., 0, 0]]).status[0] == tslocate.VACUUM
    out = advance(seg, geo)
    assert out.status[0] == c.UNRESOLVED and not out.accepted[0]
    np.testing.assert_array_equal(out.state.x, initial.x)
    np.testing.assert_array_equal(out.state.path_mm, initial.path_mm)


def test_late_subdivision_keeps_position_and_direction_error_floors():
    seg = segment(state(context(linear_map(np.diag([0, 0, .3]))),
                        position_error_mm=.1, direction_error=.2), length=3.)
    err = seg.errors([0], [2.])
    balls = seg.balls([0], [2.], [2.])
    np.testing.assert_array_equal(balls.midpoint_radius_mm, err.position_mm)
    assert balls.midpoint_radius_mm[0] > .1
    late = seg.balls([0], [2.], [2.000001])
    assert late.midpoint_radius_mm[0] > balls.midpoint_radius_mm[0]


def test_uncertain_reachable_box_rejects_nominally_covered_path_and_seams():
    ctx = context(linear_map())
    with pytest.raises(ValueError, match='wholly covered'):
        segment(state(ctx, x=[[7, 0, 0]], position_error_mm=.1), length=1.)
    bad = context(linear_map(offset=(0, 1e-100, .5), lo=(-8, 0, -8), mirror=True))
    with pytest.raises(ValueError, match='discontinuous'):
        segment(state(bad, x=[[0, 1, 0]], position_error_mm=.1), length=1.)
    gap = context(linear_map(lo=(-8, 1, -8), mirror=True))
    with pytest.raises(ValueError, match='wholly covered'):
        segment(state(gap, x=[[0, 1.5, 0]]), length=2.)


def test_snapshot_is_immutable_and_caller_map_mutation_cannot_change_it():
    item = linear_map(np.diag([0, 0, .3]))
    ctx = context(item)
    prior = segment(state(ctx)).errors([0], [1.])
    item.B.setflags(write=True)
    item.B[:] = 0
    np.testing.assert_array_equal(segment(state(ctx)).errors([0], [1.]).position_mm, prior.position_mm)
    for array in (ctx.item.B, ctx.item.header.n, ctx.item.header.grid_origin, ctx.item.header.d):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert len(ctx.fingerprint) == 64
    assert context(linear_map(np.diag([0, 0, .3]))).fingerprint == ctx.fingerprint
    assert context(item).fingerprint != ctx.fingerprint


def test_mutable_scalar_map_flags_cannot_escape_snapshot():
    item = linear_map(lo=(-8, 0, -8), mirror=True)
    reflected, origin = np.array(True), np.array(True)
    object.__setattr__(item.header, 'extend_y', reflected)
    object.__setattr__(item.header, 'explicit_origin', origin)
    ctx = context(item)
    initial = state(ctx, x=[[0, -4, 0]])
    seg = segment(initial)
    fingerprint = ctx.fingerprint
    reflected[...] = False
    origin[...] = False
    assert ctx.item.header.extend_y is True and ctx.item.header.explicit_origin is True
    assert ctx.fingerprint == fingerprint
    assert advance(seg).accepted[0]
    assert not item.coverage([[0, -4, 0]])[0]
    for value in (1, [True], 'true'):
        object.__setattr__(item.header, 'extend_y', value)
        with pytest.raises(ValueError, match='scalar bool'):
            context(item)


@pytest.mark.parametrize('kwargs', [dict(momentum=0), dict(momentum=np.inf), dict(pdg=11),
    dict(direction=[[0, 0, 0]]), dict(direction_error=-1), dict(direction_error=2.01),
    dict(position_error_mm=-1), dict(source_index=[1.]), dict(path_mm=-1)])
def test_invalid_initial_receipts(kwargs):
    with pytest.raises(ValueError):
        state(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(length=-1), dict(field_error_tesla=-1),
    dict(position_error_mm=-1), dict(direction_error=2.01), dict(direction_error=np.nan)])
def test_invalid_segment_contracts(kwargs):
    with pytest.raises(ValueError):
        segment(**kwargs)


def test_invalid_types_queries_and_empty_batches():
    with pytest.raises(ValueError, match='single FieldMap'):
        c.MapContext(lambda x: x)
    with pytest.raises(ValueError, match='MapContext'):
        state(ctx=linear_map())
    with pytest.raises(ValueError, match='State'):
        c.Segment(object(), 1, field_error_tesla=0, position_error_mm=0, direction_error=0)
    with pytest.raises(ValueError, match='unique'):
        state(x=np.zeros((2, 3)), direction=np.ones((2, 3)), source_index=[3, 3])
    seg = segment()
    for rows, s in (([0.], [0.]), ([-1], [0.]), ([1], [0.]), ([0], [-1.]), ([0], [2.])):
        with pytest.raises(ValueError):
            seg.errors(rows, s)
    with pytest.raises(ValueError, match='ordered'):
        seg.balls([0], [1.], [0.])
    empty = state(x=np.empty((0, 3)), direction=np.empty((0, 3)), momentum=np.empty(0))
    result = advance(segment(empty, length=np.empty(0)))
    assert result.state.x.shape == (0, 3) and len(result.status) == 0


def test_cumulative_path_spatial_and_phase_representability_fail_closed():
    ctx = context(linear_map(offset=(0, 0, 0)))
    with pytest.raises(ValueError, match='cumulative path'):
        advance(segment(state(ctx, path_mm=1e16), length=.1))
    with pytest.raises(ValueError, match='curvature/phase'):
        segment(state(momentum=.001), length=3.)
