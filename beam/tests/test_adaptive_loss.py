"""Standalone adaptive mean-loss diagnostics; no transport replay or external data."""
import numpy as np
import pytest

from beamfastsim.adaptive_loss import AdaptiveLossResult, adaptive_mean_loss
from beamfastsim import geometry, layerstudy
from mu2efastsim import eloss


FRACTIONS = (0.1, 0.05, 0.02, 0.01, 0.005)


@pytest.mark.parametrize("frac", FRACTIONS)
def test_constant_stopping_analytic_energy_and_consumption(frac):
    energy = np.array([0.05, 0.1, 1.0, 2.0, 3.0])
    length = np.array([10.0, 1.0, 1.0, 0.25, 0.0])
    initial, paths = energy.copy(), length.copy()
    result = adaptive_mean_loss(energy, length, frac, stopping=lambda t: np.full_like(t, 2.0))
    assert isinstance(result, AdaptiveLossResult)
    np.testing.assert_array_equal(energy, initial)
    np.testing.assert_array_equal(length, paths)
    np.testing.assert_allclose(result.energy, initial - 2 * result.consumed_length, atol=3e-14)
    assert result.stopped.tolist() == [True, True, True, False, False]
    np.testing.assert_allclose(result.energy[3:], [1.5, 3.0])
    np.testing.assert_allclose(result.consumed_length[[0, 3, 4]], [0.0, 0.25, 0.0])
    exact_stop_path = (initial[1:3] - 0.1) / 2
    assert np.all(result.consumed_length[1:3] > exact_stop_path)
    assert np.all(result.consumed_length[1:3] <= exact_stop_path + frac * 0.1 / (2 * (1 - frac)))
    assert np.all(result.max_requested_fraction <= frac * (1 + 1e-14))
    assert not result.capped.any()


def test_step_cap_is_unresolved_not_stopped_and_masks_finished_lanes():
    evaluated = []

    def stopping(t):
        evaluated.append(t.copy())
        return np.ones_like(t)

    result = adaptive_mean_loss([1.0, 0.1, 2.0, 0.01, 3.0], [10.0, 10.0, 0.1, 10.0, 0.0],
                                frac=0.1, max_steps=2, stopping=stopping)
    np.testing.assert_allclose(result.energy, [0.81, 0.09, 1.9, 0.01, 3.0])
    np.testing.assert_allclose(result.consumed_length, [0.19, 0.01, 0.1, 0.0, 0.0])
    np.testing.assert_array_equal(result.steps, [2, 1, 1, 0, 0])
    np.testing.assert_array_equal(result.capped, [True, False, False, False, False])
    np.testing.assert_array_equal(result.stopped, [False, True, False, True, False])
    np.testing.assert_allclose(evaluated[0], [1.0, 0.1, 2.0])
    np.testing.assert_allclose(evaluated[1], [0.9])
    assert len(evaluated) == 2


def test_exact_completion_on_cap_is_not_capped():
    result = adaptive_mean_loss([1.0, 0.1], [0.1, 10.0], frac=0.1, max_steps=1, stopping=lambda t: 1.0)
    assert not result.capped.any()
    np.testing.assert_array_equal(result.steps, [1, 1])
    np.testing.assert_array_equal(result.stopped, [False, True])


def test_zeros_and_strict_default_threshold():
    result = adaptive_mean_loss([0.0, 0.099, 0.1, 0.1, 1.0], [10.0, 10.0, 0.0, 1.0, 0.0],
                                stopping=lambda t: 1.0)
    assert result.stopped.tolist() == [True, True, False, True, False]
    assert result.steps.tolist() == [0, 0, 0, 1, 0]
    assert result.max_requested_fraction[[0, 1, 2, 4]].tolist() == [0.0] * 4
    assert result.energy[3] == pytest.approx(0.095)
    assert not result.capped.any()


def test_exact_threshold_endpoint_survives_then_crosses_without_asymptote():
    result = adaptive_mean_loss([0.25, 0.25], [0.125, 1.0], frac=0.5, threshold=0.125,
                                stopping=lambda t: 1.0)
    np.testing.assert_array_equal(result.energy, [0.125, 0.0625])
    np.testing.assert_array_equal(result.steps, [1, 2])
    np.testing.assert_array_equal(result.stopped, [False, True])
    np.testing.assert_array_equal(result.consumed_length, [0.125, 0.1875])
    assert not result.capped.any()


def test_empty_and_zero_stopping_power():
    def must_not_evaluate(t):
        raise AssertionError("finished lanes evaluated")

    for energies, lengths in (([], []), ([0.01, 1.0], [10.0, 0.0])):
        result = adaptive_mean_loss(energies, lengths, stopping=must_not_evaluate)
        for value in result:
            assert value.shape == (len(energies),)
        assert not result.steps.any()
        assert not result.capped.any()
    result = adaptive_mean_loss([1.0, 2.0], [100.0, 3.0], stopping=lambda t: 0.0)
    np.testing.assert_array_equal(result.energy, [1.0, 2.0])
    np.testing.assert_array_equal(result.consumed_length, [100.0, 3.0])
    np.testing.assert_array_equal(result.steps, [1, 1])
    assert not result.max_requested_fraction.any()
    assert not result.stopped.any()
    assert not result.capped.any()


def test_same_law_range_convergence_and_stop_flags():
    table = layerstudy.range_table(10.0, points=131073, threshold=0.1)
    energy = np.array([1.0, 2.0, 5.0, 1.0, 2.0, 5.0])
    ranges = np.interp(energy, *table)
    length = ranges * np.array([0.5, 0.5, 0.5, 1.2, 1.2, 1.2])
    reference, stopped = layerstudy.range_out(energy, length, table)
    errors, range_errors = [], []
    for frac in FRACTIONS:
        result = adaptive_mean_loss(energy, length, frac)
        np.testing.assert_array_equal(result.stopped, stopped)
        assert not result.capped.any()
        assert np.all(result.energy[~stopped] >= reference[~stopped])
        errors.append(np.max(result.energy[~stopped] - reference[~stopped]))
        range_errors.append(np.max(np.abs(result.consumed_length[stopped] - ranges[stopped])))
        assert np.all(result.energy <= energy)
        assert np.all(result.energy >= 0)
        assert np.all(result.max_requested_fraction <= frac * (1 + 1e-14))
    assert np.all(np.diff(errors) < 0)
    assert np.all(np.diff(range_errors) < 0)
    assert errors[-1] < 0.1 * errors[0]
    assert range_errors[-1] < 0.1 * range_errors[0]


def test_default_law_and_injected_interpolant_agree():
    grid = np.geomspace(0.1, 10.0, 131073)
    power = eloss.dedx_restricted(geometry.STOPPING_TARGET_AL, layerstudy.momentum(grid), 1.0)
    energy, length = np.array([1.0, 2.0, 5.0]), np.array([0.01, 0.05, 0.1])
    direct = adaptive_mean_loss(energy, length, frac=0.005)
    interpolated = adaptive_mean_loss(energy, length, frac=0.005, stopping=lambda t: np.interp(t, grid, power))
    np.testing.assert_allclose(direct.energy, interpolated.energy, rtol=1e-8)
    np.testing.assert_array_equal(direct.steps, interpolated.steps)
    repeated = adaptive_mean_loss(energy, length, frac=0.005)
    for first, second in zip(direct, repeated):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("frac", FRACTIONS)
def test_vectorized_fraction_bound_and_no_energy_gain(frac):
    energy = np.geomspace(0.1, 100.0, 257)
    length = np.linspace(0.0, 4.0, len(energy))
    result = adaptive_mean_loss(energy, length, frac, stopping=lambda t: 1.0 / t)
    assert np.all(result.energy >= 0)
    assert np.all(result.energy <= energy)
    assert np.all(result.consumed_length <= length)
    assert np.all(result.max_requested_fraction <= frac * (1 + 1e-14))
    assert not result.capped.any()
    for value in result:
        assert value.shape == energy.shape
    assert result.steps.dtype.kind in "iu"
    assert result.stopped.dtype == result.capped.dtype == np.dtype(bool)


@pytest.mark.parametrize("energy,length", [([np.nan], [1.0]), ([np.inf], [1.0]), ([-1.0], [1.0]),
                                           ([1.0], [np.nan]), ([1.0], [np.inf]), ([1.0], [-1.0]),
                                           (1.0, [1.0]), ([1.0], 1.0), ([[1.0]], [[1.0]]),
                                           ([1.0, 2.0], [1.0]), ([], [1.0])])
def test_invalid_arrays(energy, length):
    with pytest.raises(ValueError):
        adaptive_mean_loss(energy, length)


@pytest.mark.parametrize("keyword,value", [("frac", x) for x in [0, -0.1, 1, 2, np.nan, np.inf, True, [0.1]]]
                         + [("threshold", x) for x in [0, -1, np.nan, np.inf, True, [0.1]]]
                         + [("max_steps", x) for x in [0, -1, 1.5, True, np.nan, np.inf, [1]]]
                         + [("tcut", x) for x in [0, -1, np.nan, np.inf]] + [("stopping", 2)])
def test_invalid_controls(keyword, value):
    with pytest.raises(ValueError):
        adaptive_mean_loss([1.0], [1.0], **{keyword: value})


@pytest.mark.parametrize("power", [-1.0, np.nan, np.inf, [[1.0]], [1.0, 2.0]])
def test_invalid_stopping_power(power):
    with pytest.raises(ValueError, match="stopping power"):
        adaptive_mean_loss([1.0], [1.0], stopping=lambda t: power)


def test_nonrepresentable_step_fails_instead_of_claiming_a_stop():
    with pytest.raises(FloatingPointError, match="path progress"):
        adaptive_mean_loss([0.1], [1.0], frac=np.nextafter(0.0, 1.0), stopping=lambda t: 1.0)
