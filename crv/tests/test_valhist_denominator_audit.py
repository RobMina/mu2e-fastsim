"""Diagnostic propagation retains native boundary behavior and every block draw."""
import numpy as np
import pytest
from valhist.denominator_audit import propagate


def test_duplicate_identical_files_leave_normalization_fixed():
    result = propagate([[2, 3], [2, 3]], [10, 10], [1, 2], 20, 101, 5)
    assert result['nominal_scaled_denominator'] == [4, 6]
    assert np.allclose(result['effective_acceptance_sd'], 0, atol=1e-15)
    assert result['any_invalid_replicas'] == 0


def test_saturated_and_missing_bins_not_dropped():
    result = propagate([[1, 0, 2], [0, 0, 3]], [10, 10], [4, 1, 0], 20, 1000, 5)
    assert result['nominal_saturated'] == [True, False, False]
    assert result['nominal_missing_support'] == [False, True, False]
    assert result['nominal_effective_acceptance'] == [1, 0, 0]
    assert result['any_invalid_replicas'] == 1000
    assert result['zero_denominator_replicas_per_bin'][0] > 0
    assert result['effective_acceptance_sd'][0] > 0
    assert len(result['effective_acceptance_covariance']) == 3


def test_ratio_of_totals_not_average_rates():
    result = propagate([[1], [8]], [2, 8], [2], 100, 200, 1)
    assert result['nominal_scaled_denominator'] == [90]
    assert result == propagate([[1], [8]], [2, 8], [2], 100, 200, 1)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        propagate([[1], [2]], [0, 2], [1], 10, 5, 1)
