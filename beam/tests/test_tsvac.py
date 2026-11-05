"""Adaptive conditional vacuum prefixes; numerical idealizations, not native tests."""

import numpy as np
import pytest

from beamfastsim import tsbound, tscarry, tslocate, tsnav, tsstep, tsvac as v
from test_tscarry import state, context, free_geometry, exact_varying_path
from test_tsvary import linear_map, geometry


def walk(initial=None, geo=None, length=1., **kwargs):
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], momentum=100.) if initial is None else initial
    controls = dict(field_error_tesla=0., position_error_mm=0., direction_error=0., geometry_error_mm=0.,
                    max_position_error_mm=10., max_direction_error=2., max_step=1., min_step=2**-20,
                    max_trials=128, max_steps=64)
    controls.update(kwargs)
    return v.walk(initial, free_geometry() if geo is None else geo, length, **controls)


def assert_ledger(out, initial, requested):
    ledger = out.ledger
    accepted = ledger.accepted
    n = len(initial.x)
    total = np.bincount(ledger.row_index[accepted], weights=ledger.attempted_mm[accepted], minlength=n)
    np.testing.assert_array_equal(total, out.progress_mm)
    np.testing.assert_array_equal(out.remaining_mm+out.progress_mm, np.broadcast_to(requested, (n,)))
    np.testing.assert_array_equal(out.state.path_mm, initial.path_mm+total)
    np.testing.assert_array_equal(out.state.source_index, initial.source_index)
    np.testing.assert_array_equal(out.state.momentum, initial.momentum)
    np.testing.assert_array_equal(out.state.pdg, initial.pdg)
    assert np.all(ledger.clearance_margin_mm[accepted] > 0)
    assert np.all(ledger.rejection_reason[accepted] == v.NONE)
    assert np.all(ledger.position_error_after_mm[accepted] >= ledger.position_error_before_mm[accepted])
    assert np.all(ledger.direction_error_after[accepted] >= ledger.direction_error_before[accepted])
    np.testing.assert_array_equal(ledger.position_error_after_mm[~accepted], ledger.position_error_before_mm[~accepted])
    np.testing.assert_array_equal(ledger.direction_error_after[~accepted], ledger.direction_error_before[~accepted])


def test_exact_requested_path_and_short_final_remainder_below_minimum():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], momentum=100., path_mm=4.)
    out = walk(initial, length=1.125, max_step=1., min_step=.25)
    np.testing.assert_array_equal(out.status, [v.PATH_LIMIT])
    np.testing.assert_array_equal(out.state.x, [[1.125, 0, 0]])
    np.testing.assert_array_equal(out.ledger.attempted_mm, [1., .125])
    assert_ledger(out, initial, 1.125)
    exact_last = walk(initial, length=2., max_step=1., max_steps=2)
    assert exact_last.status[0] == v.PATH_LIMIT
    exhausted = walk(initial, length=3., max_step=1., max_steps=2)
    assert exhausted.status[0] == v.STEP_LIMIT and exhausted.progress_mm[0] == 2
    assert exhausted.last_rejection_reason[0] == v.NONE


def test_zero_request_never_inflates_evaluation_errors_or_checks_transport():
    initial = state(x=[[20, 0, 0]], position_error_mm=.25, direction_error=.125)
    out = walk(initial, length=0., position_error_mm=.0625, direction_error=.0625)
    assert out.status[0] == v.PATH_LIMIT and out.trials[0] == 0
    np.testing.assert_array_equal(out.state.x, initial.x)
    np.testing.assert_array_equal(out.state.position_error_mm, initial.position_error_mm)
    np.testing.assert_array_equal(out.state.direction_error, initial.direction_error)
    assert len(out.ledger.row_index) == 0


def test_geometry_retry_cannot_skip_thin_barrier_with_vacuum_endpoint():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], momentum=100.)
    geo = geometry(tsnav.Box([20, 20, 20]),
        tsnav.Placed(tsnav.Box([.0625, 1, 1]), [.75, 0, 0], np.eye(3)))
    assert geo.locate([[0., 0, 0], [2., 0, 0]]).status.tolist() == [tslocate.VACUUM]*2
    out = walk(initial, geo, length=2., max_step=2., min_step=2**-12, max_trials=256)
    assert out.status[0] == v.MIN_STEP and out.last_rejection_reason[0] == v.GEOMETRY
    assert 0 < out.progress_mm[0] < .6875
    assert out.last_rejected_mm[0] == 2**-12
    assert np.any(~out.ledger.accepted) and np.any(out.ledger.accepted)
    assert np.all(out.ledger.owner_index[out.ledger.accepted] == 0)
    assert_ledger(out, initial, 2.)


def test_initial_map_geometry_and_budget_floors_return_no_trial():
    ctx = context(linear_map(offset=(0, 0, 0)))
    initial = state(ctx, x=[[9, 0, 0], [0, 0, 0], [0, 2, 0], [0, 4, 0]],
        direction=np.tile([1., 0, 0], (4, 1)), momentum=100., position_error_mm=[0, .5, .125, 0],
        direction_error=[0, 0, .25, 0], source_index=[90, 20, 14, 3])
    geo = geometry(tsnav.Box([20, 20, 20]), tsnav.Box([.25, .25, .25]))
    out = walk(initial, geo, length=[1, 1, 1, 0], max_position_error_mm=[1, 1, .0625, 1])
    np.testing.assert_array_equal(out.status, [v.INITIAL_MAP, v.INITIAL_GEOMETRY, v.INITIAL_BUDGET, v.PATH_LIMIT])
    np.testing.assert_array_equal(out.trials, 0)
    np.testing.assert_array_equal(out.state.x, initial.x)
    np.testing.assert_array_equal(out.state.position_error_mm, initial.position_error_mm)


def test_initial_material_identity_is_retained_without_a_transport_trial():
    initial = state(context(linear_map(offset=(0, 0, 0))), position_error_mm=.125)
    geo = geometry(tsnav.Box([10, 10, 10]), tsnav.Box([1, 1, 1]))
    out = walk(initial, geo, length=1.)
    assert out.status[0] == v.INITIAL_GEOMETRY and out.trials[0] == 0
    assert out.initial_geometry_status[0] == tslocate.MATERIAL
    assert out.initial_owner_index[0] == 1
    assert out.initial_clearance_margin_mm[0] == .875
    np.testing.assert_array_equal(out.state.x, initial.x)
    np.testing.assert_array_equal(out.state.position_error_mm, initial.position_error_mm)


def test_position_budget_rejection_rolls_back_all_fields_before_retry():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], momentum=100.,
                    direction_error=.25, position_error_mm=.125, path_mm=2.)
    out = walk(initial, length=1., max_step=1., min_step=.125, max_position_error_mm=.25)
    assert out.ledger.rejection_reason[0] == v.POSITION_BUDGET and not out.ledger.accepted[0]
    np.testing.assert_array_equal(out.ledger.path_before_mm[:2], [0., 0.])
    np.testing.assert_array_equal(out.ledger.position_error_before_mm[:2], [.125, .125])
    np.testing.assert_array_equal(out.ledger.attempted_mm[:2], [1., .5])
    assert out.status[0] == v.MIN_STEP and out.progress_mm[0] == .5
    np.testing.assert_array_equal(out.state.position_error_mm, [.25])
    np.testing.assert_array_equal(out.state.direction_error, [.25])
    assert_ledger(out, initial, 1.)


def test_direction_budget_refines_then_stops_without_false_completion():
    initial = state(context(linear_map(np.diag([0, 0, .3]))), momentum=tsstep.K)
    out = walk(initial, length=1., max_direction_error=.015625, min_step=2**-8, max_trials=256)
    assert v.DIRECTION_BUDGET in out.ledger.rejection_reason
    assert out.status[0] == v.MIN_STEP and out.remaining_mm[0] > 0
    assert out.state.direction_error[0] <= .015625
    assert_ledger(out, initial, 1.)


def test_map_preflight_refines_without_broad_exception_catch_or_seam_switch():
    initial = state(context(linear_map(offset=(0, 0, 0))), x=[[7, 0, 0]], direction=[[1., 0, 0]])
    out = walk(initial, length=2., max_step=2., min_step=.125)
    assert out.status[0] == v.MIN_STEP and out.last_rejection_reason[0] == v.MAP
    assert out.progress_mm[0] == 1.
    assert np.any(out.ledger.rejection_reason == v.MAP)
    assert out.state.context is initial.context
    assert_ledger(out, initial, 2.)
    seam = state(context(linear_map(offset=(0, 1, .5), lo=(-8, 0, -8), mirror=True)),
                 x=[[0, 1, 0]], direction=[[0, -1, 0]])
    stopped = walk(seam, length=2., max_step=2., min_step=.25)
    assert v.MAP in stopped.ledger.rejection_reason
    assert stopped.state.x[0, 1] > 0
    assert stopped.state.context is seam.context


def test_small_final_remainder_fails_once_then_stops_not_truncated():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], direction_error=.25)
    out = walk(initial, length=1.125, max_step=1., min_step=.25, max_position_error_mm=.25)
    assert out.progress_mm[0] == 1. and out.remaining_mm[0] == .125
    assert out.status[0] == v.MIN_STEP
    assert out.last_rejected_mm[0] == .125
    assert_ledger(out, initial, 1.125)


def test_mixed_sources_have_independent_trials_and_stable_row_tags():
    ctx = context(linear_map(offset=(0, 0, 0)))
    initial = state(ctx, x=np.zeros((3, 3)), direction=np.tile([1., 0, 0], (3, 1)),
        momentum=[10, 20, 30], pdg=[13, -13, -211], source_index=[91, 1, 32],
        direction_error=[0, .25, 0], position_error_mm=0.)
    out = walk(initial, length=[1.125, 1., 0.], min_step=.25, max_position_error_mm=[1., .125, 1.])
    np.testing.assert_array_equal(out.status, [v.PATH_LIMIT, v.MIN_STEP, v.PATH_LIMIT])
    np.testing.assert_array_equal(out.progress_mm, [1.125, .5, 0.])
    np.testing.assert_array_equal(out.state.source_index, [91, 1, 32])
    assert out.trials[0] != out.trials[1] and out.trials[2] == 0
    assert_ledger(out, initial, [1.125, 1., 0.])
    for array in (out.status, out.ledger.accepted, out.progress_mm, out.ledger.row_index):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_adaptive_actual_varying_paths_both_charges_and_carried_bounds():
    initial = state(context(linear_map(np.diag([0, 0, .3]))), x=np.zeros((4, 3)),
        direction=np.tile([.8, 0, .6], (4, 1)), momentum=tsstep.K, pdg=[13, -13, 211, -211],
        position_error_mm=.02, direction_error=.03)
    out = walk(initial, length=1.5, max_step=.5, min_step=2**-8,
               max_direction_error=.5, max_position_error_mm=.5)
    assert np.all(out.status == v.PATH_LIMIT)
    actual = [exact_varying_path(np.array([1.5]), [0, 0, .02], [.81, 0, .59], q) for q in [-1, 1, 1, -1]]
    x, u = [np.concatenate(a) for a in zip(*actual)]
    d = out.state.direction/np.linalg.norm(out.state.direction, axis=1)[:, None]
    assert np.all(np.linalg.norm(x-out.state.x, axis=1) <= out.state.position_error_mm+1e-13)
    assert np.all(np.linalg.norm(u-d, axis=1) <= out.state.direction_error+1e-13)
    assert_ledger(out, initial, 1.5)


def test_nonzero_evaluation_errors_only_accumulate_on_accepted_trials():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], momentum=100.)
    geo = geometry(tsnav.Box([10., .375, .375]))
    out = walk(initial, geo, length=2., max_step=1., min_step=2**-10,
               position_error_mm=2**-10, direction_error=2**-8, max_position_error_mm=1.)
    assert np.any(~out.ledger.accepted) and np.any(out.ledger.accepted)
    assert out.status[0] == v.PATH_LIMIT
    ex, eu = 0., 0.
    for row in range(len(out.ledger.row_index)):
        np.testing.assert_array_equal(out.ledger.position_error_before_mm[row], ex)
        np.testing.assert_array_equal(out.ledger.direction_error_before[row], eu)
        if out.ledger.accepted[row]:
            ex += eu*out.ledger.attempted_mm[row] + 2**-10
            eu = min(2., eu+2**-8)
        np.testing.assert_array_equal(out.ledger.position_error_after_mm[row], ex)
        np.testing.assert_array_equal(out.ledger.direction_error_after[row], eu)
    np.testing.assert_array_equal(out.state.position_error_mm, [ex])
    np.testing.assert_array_equal(out.state.direction_error, [eu])
    assert_ledger(out, initial, 2.)


def test_interleaved_map_geometry_budget_and_accepted_rows_remain_independent():
    initial = state(context(linear_map(offset=(0, 0, 0))),
        x=[[7, -4, 0], [0, 0, 0], [0, 4, 0], [0, -2, 0]],
        direction=np.tile([1., 0, 0], (4, 1)), momentum=[10, 20, 30, 40],
        pdg=[13, -13, 211, -211], source_index=[12, 200, 3, 90], direction_error=[0, 0, .25, 0])
    geo = geometry(tsnav.Box([20, 20, 20]),
        tsnav.Placed(tsnav.Box([.0625, .5, 1]), [.75, 0, 0], np.eye(3)))
    controls = dict(length=2., max_step=2., min_step=.125, max_position_error_mm=[1, 1, .125, 1])
    out = walk(initial, geo, **controls)
    np.testing.assert_array_equal(out.ledger.rejection_reason[:4], [v.MAP, v.GEOMETRY, v.POSITION_BUDGET, v.NONE])
    np.testing.assert_array_equal(out.state.source_index, [12, 200, 3, 90])
    for row in range(4):
        solo = v._subset(initial, np.array([row]))
        single = walk(solo, geo, length=2., max_step=2., min_step=.125,
                      max_position_error_mm=controls['max_position_error_mm'][row])
        np.testing.assert_array_equal(out.state.x[row], single.state.x[0])
        np.testing.assert_array_equal(out.progress_mm[row], single.progress_mm[0])
        np.testing.assert_array_equal(out.trials[row], single.trials[0])
        np.testing.assert_array_equal(out.status[row], single.status[0])
    assert_ledger(out, initial, 2.)


def test_lifetime_trial_limit_and_resource_preflight_are_explicit():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], direction_error=.25)
    out = walk(initial, length=1., max_position_error_mm=.125, max_trials=1)
    assert out.status[0] == v.TRIAL_LIMIT and out.last_rejection_reason[0] == v.POSITION_BUDGET
    assert out.progress_mm[0] == 0 and out.steps[0] == 0
    with pytest.raises(ValueError, match='ledger'):
        walk(initial, max_trials=128, max_records=127)


def test_binary_exact_path_requirement_rejects_lossy_accumulations():
    initial = state(context(linear_map(offset=(0, 0, 0))), direction=[[1., 0, 0]], path_mm=1e16)
    with pytest.raises(ValueError, match='representable|bookkeeping'):
        walk(initial, length=.125)
    with pytest.raises(ValueError, match='bookkeeping'):
        v._exact_add(np.array([.1]), np.array([.2]))
    np.testing.assert_array_equal(v._exact_add(np.array([.5]), np.array([.25])), [.75])
    with pytest.raises(ValueError, match='bookkeeping'):
        walk(length=1., max_step=.1)


def test_arbitrary_candidate_exception_is_not_swallowed_as_a_retry(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('injected unsupported numerical failure')
    monkeypatch.setattr(tscarry, 'attempt', fail)
    with pytest.raises(ValueError, match='injected unsupported'):
        walk()


def test_initial_evaluation_and_direction_budget_floors_do_not_advance():
    initial = state(context(linear_map(offset=(0, 0, 0))), position_error_mm=.125, direction_error=.25)
    out = walk(initial, length=1., position_error_mm=.125, max_position_error_mm=.125)
    assert out.status[0] == v.INITIAL_BUDGET and out.trials[0] == 0
    direction = walk(initial, length=1., direction_error=.125, max_direction_error=.25)
    assert direction.status[0] == v.INITIAL_BUDGET and direction.trials[0] == 0
    np.testing.assert_array_equal(direction.state.direction_error, initial.direction_error)


def test_phase_arithmetic_exceptions_abort_not_reclassified_as_geometry():
    initial = state(momentum=.0001)
    with pytest.raises(ValueError, match='curvature/phase'):
        walk(initial, length=1.)


def test_batched_box_predicate_never_relabels_numeric_errors_as_coverage():
    item = linear_map(lo=(-8, 0, -8), mirror=True, offset=(0, 1e-100, .5))
    flags = tsbound.box_coverage(item, [[0, 1, 0], [0, -1, 0], [9, 1, 0]],
                               [[1, 2, 1], [1, 1, 1], [10, 2, 1]], continuous=True)
    np.testing.assert_array_equal(flags, [True, False, False])
    with pytest.raises(ValueError):
        tsbound.box_coverage(item, [[np.nan, 0, 0]], [[1, 1, 1]])


@pytest.mark.parametrize('kwargs', [dict(min_step=0), dict(max_step=0), dict(min_step=2, max_step=1),
    dict(max_trials=True), dict(max_steps=0), dict(max_records=0), dict(direction_error=2.1),
    dict(max_direction_error=2.1), dict(max_position_error_mm=-1), dict(path_limit=-1)])
def test_invalid_controls(kwargs):
    with pytest.raises((ValueError, TypeError)):
        walk(**kwargs)


def test_empty_batch():
    initial = state(x=np.empty((0, 3)), direction=np.empty((0, 3)), momentum=np.empty(0))
    out = walk(initial, length=np.empty(0))
    assert len(out.status) == len(out.ledger.row_index) == 0
