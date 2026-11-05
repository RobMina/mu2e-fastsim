"""Joint ideal marked-Poisson fixtures and fixed mutations, not production RNG certification."""
import numpy as np
import pytest
from scipy.special import pdtr
from scipy.stats import poisson

from calofastsim.combined_stats import POLICY, diagnostic
from calofastsim.poisson_stats import diagnostic as count_diagnostic
from calofastsim.travel_stats import diagnostic as travel_diagnostic


def mark_inputs(counts, rng, *, p=.5):
    order = np.flatnonzero(counts > 0)
    return order, (rng.random(int(counts.sum())) >= p).astype(np.float32)


def run(mean, counts, sipm, order, observed, *, base=None, cdf=None, times=None):
    return diagnostic(mean, counts, sipm, np.zeros(len(mean), np.float32) if base is None else base,
                      np.zeros(len(mean), np.float32), order, observed,
                      [[.5, 1]] if cdf is None else cdf, [0, 1] if times is None else times, 1)


def test_joint_fixed_policy_and_mid_table_threshold():
    assert POLICY["family_size"] == 16
    assert POLICY["family_alpha"] == .01
    assert POLICY["rounding_allowance"] == 2**-24 + 2**-50
    assert POLICY["informative_max_CDF_radius"] == .05
    report = run([2], np.array([2]), np.array([0]), np.array([0]), [0, 1])
    assert report["arrival_threshold_table_index"] == 1
    assert report["expected_indicator_counts"][0][0] == pytest.approx(1-pdtr(2, 2))
    assert not report["stock_sampler_law_validated"]


def test_first_means_generated_not_minimum_and_zeros_are_observations():
    means, counts, sipm = np.array([2., 2., 0.]), np.array([3, 0, 0]), np.array([0, 0, 1])
    report = run(means, counts, sipm, np.array([0]), [1, 0, 0], cdf=[[.25, .5, 1]], times=[0, .25, 1])
    assert report["observed_indicator_counts"][0][0] == 0
    assert report["observed_indicator_counts"][0][1] == 1
    assert report["counts_by_branch_end"] == [[2, 0], [0, 0]]
    assert report["zero_mean_rows"] == 1
    sorted_report = run(means, counts, sipm, np.array([0]), [0, 0, 1], cdf=[[.25, .5, 1]], times=[0, .25, 1])
    assert sorted_report["observed_indicator_counts"][0][0] == 1


def test_float_collapse_uses_last_equal_probability_before_thinning():
    report = run([2., 2.], np.array([1, 0]), np.array([0, 0]), np.array([0]), [2**24],
                 base=np.array([2**24, 0], np.float32), cdf=[[.25, .5, 1]], times=[0, .25, 1])
    expected = report["expected_indicator_counts"][0]
    assert expected[0] == pytest.approx((1-pdtr(2, 2)) * 1.5)
    assert expected[1] == pytest.approx(np.exp(-2) + np.exp(-1))
    assert report["observed_indicator_counts"][0][1] == 1


def test_tied_readout_order_has_no_effect_on_joint_membership():
    mean, counts, sipm = np.array([2., 2.]), np.array([3, 2]), np.array([0, 0])
    a = run(mean, counts, sipm, np.array([0, 1]), [1, 0, 0, 0, 1],
            cdf=[[.25, .5, 1]], times=[0, .25, 1])
    b = run(mean, counts, sipm, np.array([1, 0]), [0, 1, 1, 0, 0],
            cdf=[[.25, .5, 1]], times=[0, .25, 1])
    assert a == b


@pytest.mark.parametrize("value", [0, .25, 1])
def test_supported_constant_sampler_rejects_joint_law(value):
    n = 8000
    means = np.full(n, 2.)
    counts = poisson.ppf((np.arange(n)+.5)/n, 2).astype(np.int64)
    order = np.flatnonzero(counts)
    report = run(means, counts, np.zeros(n, np.int64), order, np.full(counts.sum(), value),
                 cdf=[[.25, .5, 1]], times=[0, .25, 1])
    assert report["ideal_joint_model_rejected"]


def test_count_dependent_first_mark_mutation_with_correct_marginals_is_detected():
    n = 12000
    means = np.full(n, 2.)
    counts = poisson.ppf((np.arange(n)+.5)/n, 2).astype(np.int64)
    order = np.flatnonzero(counts)
    offsets = np.cumsum(counts[order])-counts[order]
    original = np.full(counts.sum(), .25, np.float32)
    high_count = counts[order] > 2
    original[offsets[high_count]] = 1
    remaining = np.ones(len(original), bool)
    remaining[offsets] = False
    number_one_needed = len(original)//2-int(high_count.sum())
    available = np.flatnonzero(remaining)
    original[available[:number_one_needed]] = 1
    original[np.flatnonzero(original != 1)[:len(original)//4]] = 0
    sipm = np.zeros(n, np.int64)
    assert not count_diagnostic(means, counts, sipm)["ideal_Poisson_model_rejected"]
    assert not travel_diagnostic(np.zeros(len(order)), np.zeros(len(order)), counts[order], original,
                                 sipm[order], [[.25, .5, 1]], [0, .25, 1], 1)["ideal_uniform_model_rejected"]
    joint = run(means, counts, sipm, order, original, cdf=[[.25, .5, 1]], times=[0, .25, 1])
    assert joint["ideal_joint_model_rejected"]
    assert [0, 0] in joint["failed_stratum_contrast_pairs"]


def test_empty_and_underpowered_status_not_pass():
    result = run([], np.array([], np.int64), np.array([], np.int64), np.array([], np.int64), [])
    assert result["ledger_rows"] == 0
    assert result["model_diagnostic_status"] == "insufficient_coverage_or_power"
    result = run([0], np.array([0]), np.array([0]), np.array([], np.int64), [])
    assert result["zero_mean_rows"] == 1 and result["positive_mean_rows"] == 0


@pytest.mark.parametrize("order,observed,error", [([0, 0], [0, 1], "permutation"),
                                                ([1], [0], "permutation"),
                                                ([0], [.5], "support"),
                                                ([0], [np.nan], "nonfinite")])
def test_bad_join_or_arrival_not_repaired(order, observed, error):
    with pytest.raises(ValueError, match=error):
        run([1., 0.], np.array([1, 0]), np.array([0, 0]), np.array(order), observed)


def test_1000_replicates_call_actual_joint_diagnostic_under_fixed_ideal_null():
    rng = np.random.default_rng(43716)
    mean = np.repeat([.2, 2., 100., 130.], 250)
    sipm = np.repeat([0, 1, 0, 1], 250)
    rejected = 0
    for _ in range(1000):
        counts = rng.poisson(mean)
        order, arrivals = mark_inputs(counts, rng)
        report = run(mean, counts, sipm, order, arrivals, cdf=[[.25, .5, 1]], times=[0, .25, 1])
        rejected += report["ideal_joint_model_rejected"]
    assert rejected <= 25
