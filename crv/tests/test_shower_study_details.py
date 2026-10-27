"""Finite Monte Carlo tail fractions carry their own binomial uncertainty."""
import pytest

from valhist.shower_study_details import tail_interval


def test_two_tail_occurrences_interval_not_exact_probability():
    lo, hi = tail_interval(2, 10000)
    assert 0 < lo < .0002 < hi < .001


def test_zero_occurrences_do_not_mean_zero_probability():
    lo, hi = tail_interval(0, 10000)
    assert lo == 0 and .0003 < hi < .0004
    assert tail_interval(100, 100)[1] == 1
    with pytest.raises(ValueError):
        tail_interval(-1, 100)
