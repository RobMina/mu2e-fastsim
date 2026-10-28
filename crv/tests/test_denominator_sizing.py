"""Finite support sizing bounds are planning diagnostics, never an authorization."""
import numpy as np
import pytest
from scipy.stats import binom

from valhist.denominator_sizing import minimum_files, required_support


def test_required_support_controls_all_replicas_and_cells():
    m = required_support(44, 1000, .01)
    assert 44 * 1000 * np.exp(-m) <= .01
    assert 44 * 1000 * np.exp(-(m - 1)) > .01
    for n in (m, 2 * m, 1000):
        assert (1 - m / n) ** n <= np.exp(-m)


def test_file_sizing_returns_smallest_satisfying_integer():
    rates = np.array([.01, .02])
    result = minimum_files(rates, 10, 3, .05)
    n = result["files"]
    p = -np.expm1(-rates * 10)
    assert binom.cdf(2, n, p).sum() <= .05
    assert binom.cdf(2, n - 1, p).sum() > .05
    assert result["generated_events"] == n * 10


@pytest.mark.parametrize("rates", [[0], [-1], [np.nan]])
def test_missing_support_is_not_fitted_away(rates):
    with pytest.raises(ValueError):
        minimum_files(rates, 50000, 16)
