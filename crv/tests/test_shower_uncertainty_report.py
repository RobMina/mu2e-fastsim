"""Sparse report guards do not hide deterministic residuals or zero reference populations."""
import numpy as np

from valhist.shower_uncertainty_report import positions, stderr, jsonable


def test_zero_variance_disagreement_is_counted():
    mean = np.array([0., 2., 1., 0.]); g4 = np.array([0., 1., 1., 0.])
    result = positions(mean, np.zeros((4, 4)), np.zeros((4, 4)), np.zeros(4), g4,
                       np.zeros((4, 4)), {"EX/y": [0, 4]})["EX/y"]
    assert result["paired_diagonal_score"] == {
        "value": None, "used_bins": 0, "zero_variance_nonzero_residual_bins": 1}
    assert result["flow_expected"] == result["flow_g4"] == 0


def test_jsonable_nested_numpy_and_covariance_sd():
    assert jsonable({"a": [np.int64(1), np.array([2.])]}) == {"a": [1, [2.]]}
    np.testing.assert_array_equal(stderr(np.diag([1., 4.])), [1., 2.])
