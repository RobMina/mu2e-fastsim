"""Synthetic safety/lineage checks only; no stock-geometry physics certification."""

from dataclasses import fields

import numpy as np
import pytest

from beamfastsim import tsdecay, tsdriver as d, tsnav as g


ZERO = np.zeros(3)


def run(x=None, p=None, pdg=13, path=10., **kwargs):
    x = np.zeros((1, 3)) if x is None else np.asarray(x, float)
    p = np.tile([100., 0, 0], (len(x), 1)) if p is None else np.asarray(p, float)
    return d.propagate(x, p, pdg, path, kwargs.pop("field", ZERO),
                       kwargs.pop("region", g.Sphere(1000.)),
                       mode=kwargs.pop("mode", "no_decay"), **kwargs)


def test_straight_species_clocks_order_and_input_immutability():
    x = np.arange(12., dtype=float).reshape(4, 3)
    p = np.tile([30., 40, 0], (4, 1))
    copyx, copyp = x.copy(), p.copy()
    lengths = np.array([0., 1., 10., 100.])
    out = run(x, p, [13, -13, 211, -211], lengths, time=-2., max_step=3.)
    np.testing.assert_array_equal(out.status, d.PATH_LIMIT)
    np.testing.assert_allclose(out.x, x + lengths[:, None] * [0.6, 0.8, 0])
    np.testing.assert_allclose(out.path, lengths)
    mass = np.array([tsdecay.MMU, tsdecay.MMU, tsdecay.MPI, tsdecay.MPI])
    np.testing.assert_allclose(out.proper_time, lengths * mass / (50 * tsdecay.C_MM_NS))
    np.testing.assert_allclose(out.time, -2 + lengths * np.hypot(50, mass) / (50 * tsdecay.C_MM_NS))
    np.testing.assert_array_equal(out.root_index, np.arange(4))
    np.testing.assert_array_equal(out.generation, 0)
    np.testing.assert_array_equal(out.pion_decayed, False)
    assert np.isinf(out.remaining).all()
    assert np.isnan(out.pion_decay_x).all()
    np.testing.assert_array_equal(x, copyx)
    np.testing.assert_array_equal(p, copyp)


def test_uniform_field_convergence_and_charge():
    out = run(np.zeros((2, 3)), np.tile([100., 0, 0], (2, 1)), [13, -13], 100.,
              field=[0, 0, 1], max_step=0.5)
    theta = 0.299792458 * np.array([-1., 1.])
    expected_x = np.column_stack((100 * np.sin(theta) / theta,
                                  -100 * (1 - np.cos(theta)) / theta, [0, 0]))
    np.testing.assert_allclose(out.x, expected_x, atol=2e-10)
    np.testing.assert_array_equal(out.status, d.PATH_LIMIT)
    assert out.x[0, 1] > 0 > out.x[1, 1]


def test_boundary_pending_not_loss_and_not_chord_escape():
    out = run(path=100., region=g.Sphere(10.), max_step=100., boundary_atol=1e-7)
    assert out.status[0] == d.BOUNDARY_PENDING
    assert 0 < 10 - out.x[0, 0] <= 1e-7
    assert out.path[0] < 10
    assert "absorption" in d.LIMITATIONS
    assert "boundary_pending" in d.STATUS.values()
    assert all("lost" not in s and "absorbed" not in s for s in d.STATUS.values())


def test_starts_outside_and_zero_limits_not_invented_losses():
    out = run([[0, 0, 0], [11, 0, 0], [10, 0, 0], [9.9999999, 0, 0]],
              pdg=13, path=[0, 10, 10, 10], region=g.Sphere(10.), boundary_atol=1e-6)
    np.testing.assert_array_equal(out.status, [d.PATH_LIMIT, d.START_NOT_INTERIOR,
                                              d.START_NOT_INTERIOR, d.BOUNDARY_PENDING])
    np.testing.assert_array_equal(out.steps, 0)
    np.testing.assert_array_equal(out.path, 0.)


def test_iteration_cap_is_explicit_unresolved():
    out = run(path=100., max_step=1., max_steps=2)
    assert out.status[0] == d.ITERATION_LIMIT
    assert out.path[0] == 2
    assert out.steps[0] == 2


def test_pion_decay_exact_point_parent_ledger_and_daughter_clocks():
    distance = 7.5
    lifetime = tsdecay.proper_time(distance, 100.)
    out = run(pdg=211, path=distance, mode="conditional_muonic",
              rng=np.random.default_rng(30), proper_remaining=lifetime, max_step=10.)
    assert out.pion_decayed[0]
    assert out.pdg[0] == -13
    assert out.generation[0] == 1
    assert out.status[0] == d.PATH_LIMIT
    np.testing.assert_allclose(out.pion_decay_x, [[distance, 0, 0]], atol=2e-14)
    np.testing.assert_allclose(out.pion_decay_p3, [[100., 0, 0]])
    np.testing.assert_array_equal(out.proper_time, 0.)
    np.testing.assert_array_equal(out.daughter_birth_p3, out.p3)
    np.testing.assert_allclose(out.p3 + out.neutrino_p3, [[100., 0, 0]], atol=3e-14)
    np.testing.assert_allclose(np.hypot(np.linalg.norm(out.p3, axis=1), tsdecay.MMU)
                               + np.linalg.norm(out.neutrino_p3, axis=1), np.hypot(100., tsdecay.MPI))


def test_daughter_advances_under_own_mass_clock_and_identity():
    out = run(pdg=-211, path=10., mode="conditional_muonic",
              rng=np.random.default_rng(43), proper_remaining=0., max_step=1.)
    assert out.pdg[0] == 13
    assert out.generation[0] == 1
    np.testing.assert_array_equal(out.pion_decay_x, [[0, 0, 0]])
    p = np.linalg.norm(out.p3, axis=1)
    np.testing.assert_allclose(out.x, 10 * out.p3 / p[:, None])
    np.testing.assert_allclose(out.proper_time, 10 * tsdecay.MMU / (p * tsdecay.C_MM_NS))
    np.testing.assert_allclose(out.time, 10 * np.hypot(p, tsdecay.MMU) / (p * tsdecay.C_MM_NS))
    assert out.steps[0] == 11


def test_muon_decay_at_exact_distance_and_no_daughters():
    out = run(pdg=13, path=10., mode="conditional_muonic",
              rng=np.random.default_rng(4), proper_remaining=tsdecay.proper_time(3., 100., tsdecay.MMU))
    assert out.status[0] == d.MUON_DECAY
    assert out.path[0] == pytest.approx(3.)
    assert not out.pion_decayed[0]
    assert np.isnan(out.daughter_birth_p3).all()


def test_decay_rng_reproducibility_and_fixed_input_order():
    opts = dict(pdg=[211, -211, 13], path=2., mode="conditional_muonic", proper_remaining=0.)
    x = np.zeros((3, 3))
    a = run(x, rng=np.random.default_rng(19), **opts)
    b = run(x, rng=np.random.default_rng(19), **opts)
    for f in fields(a):
        if f.name == "mode":
            assert a.mode == b.mode
        else:
            np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name))
    np.testing.assert_array_equal(a.root_index, [0, 1, 2])
    np.testing.assert_array_equal(a.generation, [1, 1, 0])
    np.testing.assert_array_equal(a.status, [d.PATH_LIMIT, d.PATH_LIMIT, d.MUON_DECAY])


def test_all_clocks_sampled_once_no_step_resampling():
    rng = np.random.default_rng(10)
    expected_rng = np.random.default_rng(10)
    sampled = expected_rng.exponential([tsdecay.TAU_PI_NS, d.TAU_MU_NS])
    out = run(np.zeros((2, 3)), pdg=[211, 13], path=0.01,
              mode="conditional_muonic", rng=rng, max_step=0.001)
    np.testing.assert_allclose(out.remaining + out.proper_time, sampled)
    np.testing.assert_array_equal(rng.random(10), expected_rng.random(10))


@pytest.mark.parametrize("kwargs", [dict(mode="bad"), dict(max_step=0), dict(safety_fraction=0.6),
    dict(safety_fraction=0), dict(boundary_atol=0), dict(max_steps=True), dict(max_steps=1.5),
    dict(pdg=11), dict(path=-1), dict(time=np.nan), dict(mode="conditional_muonic"),
    dict(proper_remaining=1), dict(rng=np.random.default_rng(1))])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        run(**kwargs)


def test_invalid_geometry_and_lifetime_consume_no_rng():
    rng, untouched = np.random.default_rng(1), np.random.default_rng(1)
    with pytest.raises(ValueError):
        run(mode="conditional_muonic", rng=rng, proper_remaining=-1)
    np.testing.assert_array_equal(rng.random(5), untouched.random(5))
    class Bad:
        def clearance(self, x):
            return np.zeros((len(x), 1))
    with pytest.raises(ValueError, match="shape"):
        run(region=Bad())
    with pytest.raises(ValueError, match="region"):
        run(region=lambda x: np.ones(len(x)))


def test_field_coverage_and_clearance_contract_failure_propagate():
    class CoverageError(RuntimeError):
        pass
    def absent(x):
        raise CoverageError("no map")
    with pytest.raises(CoverageError):
        run(field=absent)
    class Liar:
        def clearance(self, x):
            return np.where(x[:, 0] == 0, 100., -1.)
    with pytest.raises(ValueError, match="contradicts"):
        run(region=Liar())


def test_empty_batch():
    out = run(np.empty((0, 3)), p=np.empty((0, 3)), pdg=[], path=[])
    assert out.x.shape == (0, 3)
    assert out.status.shape == (0,)


def test_endpoint_boundary_policy_precedes_decay_and_consumes_no_daughters():
    rng, untouched = np.random.default_rng(7), np.random.default_rng(7)
    out = run(pdg=211, path=1., region=g.HalfSpace([1, 0, 0], 1.5e-6),
              mode="conditional_muonic", rng=rng,
              proper_remaining=tsdecay.proper_time(0.6e-6, 100.), boundary_atol=1e-6)
    assert out.status[0] == d.BOUNDARY_PENDING
    assert not out.pion_decayed[0]
    assert out.generation[0] == 0
    np.testing.assert_array_equal(rng.random(5), untouched.random(5))


def test_large_origin_no_false_path_progress():
    with pytest.raises(ValueError, match="spatial advancement"):
        run([[1e16, 0, 0]], path=10., max_step=0.1, region=g.Sphere(1e18))


def test_invalid_initial_field_magnitude_does_not_consume_lifetime():
    rng, untouched = np.random.default_rng(2), np.random.default_rng(2)
    with pytest.raises(ValueError, match="initial field magnitude"):
        run(mode="conditional_muonic", rng=rng, field=[1.7e308, 1.7e308, 0])
    np.testing.assert_array_equal(rng.random(5), untouched.random(5))


@pytest.mark.parametrize("max_step", [100., 0.3])
def test_endpoint_decay_sweep_in_proper_clock_not_roundtrip_distance(max_step):
    p = np.geomspace(1., 1000., 50)
    length = np.geomspace(0.001, 20., 50)
    initial = np.column_stack((p, np.zeros((50, 2))))
    out = run(np.zeros((50, 3)), initial, pdg=13, path=length,
              mode="conditional_muonic", rng=np.random.default_rng(10),
              proper_remaining=tsdecay.proper_time(length, p, tsdecay.MMU), max_step=max_step)
    np.testing.assert_array_equal(out.status, d.MUON_DECAY)
    np.testing.assert_allclose(out.path, length, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(out.remaining, 0.)


def test_curve_hits_obstacle_that_straight_endpoint_chord_misses():
    radius = 100 / 0.299792458
    theta = 50 / radius
    center = np.array([radius * np.sin(theta), radius * (1 - np.cos(theta)), 0.])
    final = np.array([radius * np.sin(2 * theta), radius * (1 - np.cos(2 * theta)), 0.])
    closest = final * np.clip(np.dot(center, final) / np.dot(final, final), 0, 1)
    assert np.linalg.norm(center - closest) > 1.
    region = g.Difference(g.Sphere(1000.), g.Placed(g.Sphere(1.), center, np.eye(3)))
    out = run(path=100., field=[0, 0, 1.], region=region, max_step=1., boundary_atol=1e-6)
    assert out.status[0] == d.BOUNDARY_PENDING
    assert 48.9 < out.path[0] < 50
    assert 0 < region.clearance(out.x)[0] <= 1e-6


def test_unscheduled_decay_distance_overflow_is_never_evaluated():
    out = run(mode="conditional_muonic", rng=np.random.default_rng(3),
              proper_remaining=1e308, path=1.)
    assert out.status[0] == d.PATH_LIMIT
    assert out.path[0] == 1.
    assert out.remaining[0] > 1e307


def test_decay_candidate_clock_uses_angle_cap_before_scheduling():
    out = run(p=[[1e-308, 0, 0]], path=100., max_step=100., field=[0, 0, 1.],
              mode="conditional_muonic", rng=np.random.default_rng(3),
              proper_remaining=1., max_steps=1)
    assert out.status[0] == d.ITERATION_LIMIT
    assert 0 < out.path[0] < 1e-308
    assert 0 < out.proper_time[0] < 1.
    assert out.remaining[0] > 0.


def test_absolute_lab_clock_preserves_accumulated_sub_ulp_steps():
    start_time = 1e16
    path = 1000.
    out = run(path=path, max_step=1., time=start_time, region=g.Sphere(1e6))
    increment = path / tsdecay.C_MM_NS * np.hypot(1., tsdecay.MMU / 100.)
    assert increment > np.spacing(start_time)
    np.testing.assert_array_equal(out.time, start_time + increment)
