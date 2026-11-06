"""Paired occupancy includes empty events, all geometry bins and covariance."""
import awkward as ak
import numpy as np
import pytest

from mixing.compare import occupancy, paired_counts


def test_empty_events_and_noncontiguous_detector_mapping():
    channels = ak.Array([[0, 3, 3], [], [1]])
    result = occupancy(channels, np.array([2, 0, 3, 1]), 4)
    np.testing.assert_array_equal(result, [[0, 2, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]])


def test_identical_has_zero_paired_error():
    counts = np.array([[1, 0], [3, 0], [2, 0]])
    result = paired_counts(counts, counts)
    np.testing.assert_array_equal(result["difference"], [0, 0])
    np.testing.assert_array_equal(result["difference_error"], [0, 0])
    assert result["ratio"][0] == 1
    assert result["ratio_error"][0] == 0
    assert np.isnan(result["ratio"][1])


def test_covariance_and_constant_difference():
    stock = np.array([[1], [2], [3], [4]])
    fast = stock * 2
    result = paired_counts(stock, fast)
    assert result["ratio"][0] == 2
    assert result["ratio_error"][0] == 0
    assert result["difference_error"][0] == pytest.approx(np.std(stock, ddof=1) / 2)
    shifted = paired_counts(stock, stock + 1)
    assert np.isinf(shifted["chi2_diagonal"])
    assert shifted["zero_variance_nonzero_difference"][0]


def test_bad_occupancy_map_and_unpaired_shapes():
    with pytest.raises(ValueError):
        occupancy(ak.Array([[9]]), np.arange(4), 4)
    with pytest.raises(ValueError):
        paired_counts(np.zeros((2, 3)), np.zeros((3, 2)))
