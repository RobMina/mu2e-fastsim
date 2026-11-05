"""Fixed conditional ideal-uniform diagnostic tests, not a production RNG validation."""
import numpy as np
import pytest

from calofastsim.propagation import propagation_time
from calofastsim.travel_stats import POLICY, diagnostic


def test_fixed_sampler_in_support_is_rejected_by_distribution_diagnostic():
    count = 20000
    report = diagnostic([0], [0], np.array([count]), np.zeros(count), np.array([0]), [[.5, 1]], [0, 1], 1)
    assert report["ideal_uniform_model_rejected"]
    assert [0, 0] in report["failed_stratum_threshold_pairs"]
    assert report["empty_strata"] == 1
    assert report["production_rng_validated"] is False


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_constant_first_last_and_midpoint_samplers_rejected(value):
    n = 20000
    report = diagnostic([0], [0], np.array([n]), np.full(n, value), np.array([0]),
                        [[.1, .3, .6, 1]], [0, 1, 2, 3], 1)
    assert report["ideal_uniform_model_rejected"]


def test_uniform_bin_sampling_is_not_inverse_CDF():
    n = 20000
    observed = np.tile(np.arange(4, dtype=np.float32), n // 4)
    report = diagnostic([0], [0], np.array([n]), observed, np.array([0]),
                        [[.1, .3, .6, 1]], [0, 1, 2, 3], 1)
    assert report["ideal_uniform_model_rejected"]


def test_balanced_quantiles_agree_without_claiming_rng_validation():
    n = 10000
    draws = (np.arange(n, dtype=np.float64) + .5) / n
    cdf, times = np.array([[.25, .25, 1]], np.float32), np.array([0, .25, 1], np.float32)
    arrivals = propagation_time(0, draws, cdf, times, 1)
    report = diagnostic([0], [0], np.array([n]), arrivals, np.array([1]), cdf, times, 1)
    assert not report["ideal_uniform_model_rejected"]
    assert report["informative_strata"] == 1
    assert not report["all_strata_informative"]
    assert report["counts_by_depth_end"] == [[0, n]]
    assert not report["stochastic_distribution_certified"]


def test_collapsed_arrival_threshold_probabilities_use_last_equal_CDF():
    cdf = np.array([[.25, .5, 1]], np.float32)
    report = diagnostic([2**24], [0], np.array([4]), np.full(4, 2**24), np.array([0]), cdf, [0, .25, 1], 1)
    assert report["expected_cumulative_counts_by_stratum_threshold"][0] == [4, 4, 4]
    assert report["observed_counts_by_stratum_bin"][0] == [4, 0, 0]
    assert report["underpowered_populated_strata"] == 1


def test_conditional_heterogeneous_arrival_collapse_is_not_pooled_naively():
    cdf = [[.25, .5, 1]]
    base = [0, 2**24]
    n = 4000
    first = np.tile(np.array([0, .25, 1, 1], np.float32), n // 4)
    observed = np.r_[first, np.full(n, 2**24, np.float32)]
    report = diagnostic(base, [0, 0], np.array([n, n]), observed, np.array([0, 0]), cdf, [0, .25, 1], 1)
    assert report["expected_cumulative_counts_by_stratum_threshold"][0] == [n * 1.25, n * 1.5, n * 2]
    assert not report["ideal_uniform_model_rejected"]


def test_all_empty_ledger_retained_not_informative():
    report = diagnostic([], [], np.array([], np.int64), [], np.array([], np.int64), [[1]], [0], 1)
    assert report["empty_strata"] == 2
    assert report["photoelectrons"] == 0
    assert not report["all_strata_informative"]
    assert report["model_diagnostic_status"] == "insufficient_coverage_or_power"
    assert not report["ideal_uniform_model_rejected"]


def test_prospective_policy_alpha_allowance_and_power_are_fixed():
    assert POLICY["family_alpha"] == .01
    assert POLICY["rounding_allowance"] == 2**-24 + 2**-50
    assert POLICY["informative_max_CDF_radius"] == .05
    assert not POLICY["seed_matching"]
    assert "independent_uniform" in POLICY["null_assumption"]


def test_non_supported_arrival_refused_before_statistics():
    with pytest.raises(ValueError, match="support"):
        diagnostic([0], [0], np.array([1]), [.5], np.array([0]), [[.5, 1]], [0, 1], 1)


def test_simultaneous_analytic_bound_covers_fixed_ideal_null_ensemble():
    from calofastsim.travel_stats import POLICY
    rng = np.random.default_rng(20261104)
    trials, n = 1000, 4000
    probabilities = np.array([.1, .2, .3, .4])
    counts = rng.multinomial(n, probabilities, size=trials)
    delta = np.abs(np.cumsum(counts, axis=1) / n - np.cumsum(probabilities)[None, :])
    radius = np.sqrt(np.log(2 * 8 / POLICY["family_alpha"]) / (2 * n)) + POLICY["rounding_allowance"]
    rejected = np.any(delta > radius, axis=1)
    assert np.count_nonzero(rejected) <= 20
