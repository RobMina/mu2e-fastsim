"""Frozen-layer opt-in sampler tests; no external references or production edits."""
import inspect

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from mu2efastsim import eloss, eloss_jax
from beamfastsim import geometry, layerstudy as ls
from beamfastsim.rangefluct import MeanLaw, delta_mean_per_mm, run_layer, sample_supplied_mean


def constant_law(cut=1.):
    energy = np.linspace(.1, 10., 1001)
    return MeanLaw(energy, np.full_like(energy, 2.), (energy - .1) / 2, cut)


def test_original_mean_reproduces_sampler_and_restores_function():
    original = eloss_jax._dedx
    source = inspect.getsource(eloss_jax._sample.__wrapped__)
    p = np.geomspace(1., 10000., 2048)
    length = np.geomspace(.0001, 10., len(p))
    cut = .03410575
    mean = np.asarray(eloss_jax.dedx_restricted(geometry.STOPPING_TARGET_AL, p, cut)) * length
    key = jax.random.key(55)
    expected = np.asarray(eloss_jax.sample_loss(key, geometry.STOPPING_TARGET_AL, p, length, cut))
    actual = sample_supplied_mean(key, p, length, cut, mean)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-13)
    assert eloss_jax._dedx is original
    assert inspect.getsource(eloss_jax._sample.__wrapped__) == source
    np.testing.assert_array_equal(actual, sample_supplied_mean(key, p, length, cut, mean))


def test_supplied_mean_changes_mean_not_bohr_variance():
    p, length, cut = np.full(65536, 10.), .025, 1.
    key = jax.random.key(123)
    one = sample_supplied_mean(key, p, length, cut, .2)
    two = sample_supplied_mean(key, p, length, cut, .4)
    np.testing.assert_allclose(two - one, .2, atol=1e-8)
    assert abs(one.mean() - .2) < 5 * one.std() / np.sqrt(len(one))
    assert abs(two.mean() - .4) < 5 * two.std() / np.sqrt(len(two))


def test_zero_path_mean_and_invalid_sampler_inputs():
    np.testing.assert_array_equal(sample_supplied_mean(jax.random.key(1), [2., 3.], 0., 1., 0.), [0., 0.])
    for p, length, cut, mean in [(1, 0, 1, .1), (1, 1, 0, 1), (-1, 1, 1, 1),
                                  (1, -1, 1, 1), (1, 1, 1, -1), (1, 1, 1, np.nan)]:
        with pytest.raises(ValueError):
            sample_supplied_mean(jax.random.key(1), p, length, cut, mean)


def test_constant_range_mean_strict_endpoint_and_forced_stop():
    law = constant_law()
    mean, stop = law.mean([1.1, 1.1, 1.1, .1], [.25, .5, .6, 0], "range")
    np.testing.assert_allclose(mean, [.5, 1., 1.1, 0.])
    np.testing.assert_array_equal(stop, [False, False, True, False])
    result = run_layer([1.1, 1.1, .1], [.25, .6, 0.], law, slices=1, replicas=2, fluctuate=False)
    np.testing.assert_allclose(result["exit_MeV"], [[.6, 0., .1]] * 2)
    np.testing.assert_array_equal(result["forced_range_stop"], [[False, True, False]] * 2)


def test_same_law_range_semigroup_and_deterministic_reference():
    table = ls.range_table(10., points=65537)
    power = eloss.dedx_restricted(geometry.STOPPING_TARGET_AL, ls.momentum(table[0]), 1.)
    law = MeanLaw(table[0], power, table[1], 1.)
    energy = np.array([.1, .3, 1., 1.5, 2., 2.2, 5., 9.])
    length = np.array([0., .2, .001, .1056, .2, .2, .2, .2])
    expected, stopped = ls.range_out(energy, length, table)
    for slices in (1, 4, 16, 32):
        result = run_layer(energy, length, law, slices=slices, replicas=1, fluctuate=False)
        np.testing.assert_allclose(result["exit_MeV"][0], expected, atol=3e-12)
        np.testing.assert_array_equal(result["stop"][0], stopped)


def test_range_exhaustion_bypasses_fluctuations_and_deltas():
    result = run_layer([.2, .3], [2., 2.], constant_law(.001), replicas=64, slices=1, deltas=True)
    assert result["forced_range_stop"].all()
    assert not result["exit_MeV"].any()
    assert not result["hard_count"].any()
    assert not result["continuous_cap_excess"].any()


def test_split_energy_ledger_and_repeatability():
    law = constant_law(.01)
    a = run_layer([2., 5., 9.], [.2, .1, 0.], law, deltas=True, replicas=256)
    b = run_layer([2., 5., 9.], [.2, .1, 0.], law, deltas=True, replicas=256)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert a["hard_count"].sum() > 0
    assert np.all(a["hard_applied"] <= a["hard_requested"])
    assert np.all(a["exit_MeV"] <= [2., 5., 9.])
    np.testing.assert_array_equal(a["exit_MeV"][:, 2], 9.)
    total = sum(a[k] for k in ("exit_MeV", "continuous_applied", "hard_applied", "threshold_residual"))
    np.testing.assert_allclose(total, np.broadcast_to([2., 5., 9.], total.shape), atol=1e-12)


def test_hard_first_moment_closes_fast_cut_difference():
    energy = np.geomspace(.1, 40., 200)
    p, cut = ls.momentum(energy), .03410575
    difference = eloss.dedx_restricted(geometry.STOPPING_TARGET_AL, p, 1e9)
    difference -= eloss.dedx_restricted(geometry.STOPPING_TARGET_AL, p, cut)
    np.testing.assert_allclose(delta_mean_per_mm(energy, cut), difference, atol=2e-14)
    assert not delta_mean_per_mm(np.array([.1, .5, 1.]), cut).any()


def test_hard_sample_moment_matches_analytic():
    energy, length, cut, n = 5., .2, .03410575, 200000
    idx, transfers = eloss.sample_deltas(np.random.default_rng(42), geometry.STOPPING_TARGET_AL,
                                        np.full(n, ls.momentum(energy)), np.full(n, length), cut)
    losses = np.bincount(idx, weights=transfers, minlength=n)
    expected = delta_mean_per_mm(energy, cut) * length
    assert abs(losses.mean() - expected) < 5 * losses.std(ddof=1) / np.sqrt(n)


@pytest.mark.parametrize("name,value", [("slices", 0), ("replicas", 0), ("seed", -1),
                                         ("slices", 1.5), ("replicas", True), ("mode", "bad")])
def test_invalid_run_controls(name, value):
    with pytest.raises(ValueError):
        run_layer([1.], [.1], constant_law(), **{name: value})


@pytest.mark.parametrize("energy,length", [([-.09], [.1]), ([11.], [.1]), ([1.], [-1.]),
                                            ([np.nan], [.1]), ([], []), ([1.], [1., 2.])])
def test_invalid_layer_inputs(energy, length):
    with pytest.raises(ValueError):
        run_layer(energy, length, constant_law())


def test_already_subthreshold_entries_are_absorbed_without_evaluating_law():
    result = run_layer([0., .03, .099, .1], [1., 1., 0., 0.], constant_law(.001), replicas=2, deltas=True)
    np.testing.assert_array_equal(result["exit_MeV"], [[0., 0., 0., .1]] * 2)
    np.testing.assert_allclose(result["threshold_residual"], [[0., .03, .099, 0.]] * 2)
    assert not result["hard_count"].any()
    assert not result["continuous_applied"].any()


def test_invalid_tables_and_out_of_domain():
    for e, s, r in [([.1, 1, 2], [1, 0, 1], [0, 1, 2]),
                     ([.1, 1, 2], [1, 1, 1], [0, 2, 1]),
                     ([.2, 1, 2], [1, 1, 1], [0, 1, 2])]:
        with pytest.raises(ValueError):
            MeanLaw(e, s, r, 1.)
    with pytest.raises(ValueError):
        constant_law().mean([.09], [.1], "range")


@pytest.mark.parametrize("energy", [.2, .3, .5, 1., 1.1, 2., 4.])
def test_deterministic_exact_threshold_arrival_is_not_lost_by_subtraction(energy):
    law = constant_law()
    length = float(np.interp(energy, law.energy, law.ranges))
    result = run_layer([energy], [length], law, slices=1, replicas=1, fluctuate=False)
    assert result["exit_MeV"][0, 0] == .1
    assert not result["stop"][0, 0]
    assert not result["forced_range_stop"][0, 0]


def test_controlled_caps_and_discarded_hard_requests(monkeypatch):
    import beamfastsim.rangefluct as rf
    monkeypatch.setattr(rf, "sample_supplied_mean", lambda key, p, length, cut, mean: np.array([[2., .7]]))
    monkeypatch.setattr(rf.eloss, "sample_deltas", lambda *args: (np.array([0, 1]), np.array([.2, .5])))
    result = run_layer([1., 1.], [.1, .1], constant_law(.01), mode="local", deltas=True, slices=1, replicas=1)
    np.testing.assert_allclose(result["continuous_applied"], [[1., .7]])
    np.testing.assert_allclose(result["continuous_cap_excess"], [[1., 0.]])
    np.testing.assert_allclose(result["hard_requested"], [[.2, .5]])
    np.testing.assert_allclose(result["hard_applied"], [[0., .3]])
    np.testing.assert_allclose(result["hard_cap_excess"], [[.2, .2]])
    np.testing.assert_array_equal(result["hard_count"], [[1, 1]])
    assert not result["exit_MeV"].any()


@pytest.mark.parametrize("p,length,cut,mean", [(1.85, .0001, 1., .00003),
                                               (300., .2, .03410575, .03),
                                               (300., .2, .03410575, .06)])
def test_changed_mean_gamma_and_urban_moments(p, length, cut, mean):
    values = sample_supplied_mean(jax.random.key(77), np.full(100000, p), length, cut, mean)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert abs(values.mean() - mean) < 6 * values.std(ddof=1) / np.sqrt(len(values)) + 1e-10
