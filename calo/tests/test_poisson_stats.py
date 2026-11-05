"""Fixed ideal-Poisson diagnostic mutation/null fixtures; no production RNG claims."""
import numpy as np
import pytest
from scipy.special import pdtr
from scipy.stats import poisson

from calofastsim.poisson_stats import POLICY, diagnostic


def balanced():
    means = np.repeat([.2, 12., 100., 1000.], 10000)
    sipm = np.repeat([0, 1, 0, 1], 10000)
    quantile = np.tile((np.arange(10000) + .5) / 10000, 4)
    return means, poisson.ppf(quantile, means).astype(np.int64), sipm


def test_fixed_policy_balanced_quantiles_and_zero_mean_retention():
    means, counts, sipm = balanced()
    result = diagnostic(np.r_[means, 0], np.r_[counts, 0], np.r_[sipm, 0])
    assert result["model_diagnostic_status"] == "not_rejected"
    assert result["zero_mean_rows"] == 1
    assert result["informative_strata"] == 4
    assert POLICY["family_size"] == 20 and POLICY["family_alpha"] == .01
    assert not result["stock_sampler_law_validated"]


@pytest.mark.parametrize("mutation", ["rounding", "zero", "dispersion", "bias"])
def test_fixed_bad_samplers_reject(mutation):
    means, counts, sipm = balanced()
    if mutation == "rounding":
        counts = np.floor(means + .5).astype(np.int64)
    elif mutation == "zero":
        counts[:] = 0
    elif mutation == "dispersion":
        counts = np.where(np.arange(len(counts)) % 2, 0, 2 * np.floor(means + .5)).astype(np.int64)
    else:
        counts += 4
    assert diagnostic(means, counts, sipm)["model_diagnostic_status"] == "rejected"


def test_heterogeneous_cdf_not_cdf_at_average_mean():
    means = np.array([.01, 9., 0])
    result = diagnostic(means, [0, 0, 0], [0, 0, 1])
    assert np.isclose(result["expected_cumulative_counts"][0][0], np.exp(-.01) + np.exp(-9))
    assert not np.isclose(result["expected_cumulative_counts"][0][0], 2 * np.exp(-4.505))
    assert result["positive_mean_rows"] == 2
    assert result["counts_by_branch_end"] == [[2, 0], [0, 0]]
    assert result["model_diagnostic_status"] == "insufficient_coverage_or_power"


def test_empty_and_underpowered_not_passes():
    for mean, count, sipm in (([], [], []), ([0], [0], [0]), ([.1], [0], [0])):
        result = diagnostic(mean, np.array(count, np.int64), np.array(sipm, np.int64))
        assert result["model_diagnostic_status"] == "insufficient_coverage_or_power"


@pytest.mark.parametrize("mean,count,sipm", [([0], [1], [0]), ([np.nan], [0], [0]),
                                             ([1], [-1], [0]), ([1], [0.], [0]), ([1], [0], [-1])])
def test_invalid_input_refused(mean, count, sipm):
    with pytest.raises(ValueError):
        diagnostic(mean, count, sipm)


def test_1000_replicate_fixed_ideal_null_simultaneous_coverage():
    rng = np.random.default_rng(7261)
    means = np.repeat([.2, 12, 100, 1000.], 2000)
    sipm = np.repeat([0, 1, 0, 1], 2000)
    rejected = 0
    for _ in range(1000):
        counts = rng.poisson(means)
        rejected += diagnostic(means, counts, sipm)["ideal_Poisson_model_rejected"]
    assert rejected <= 25
