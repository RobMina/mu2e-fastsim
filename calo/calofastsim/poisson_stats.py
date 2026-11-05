"""Prospective independent ideal-Poisson CDF diagnostic, not the RandPoissonQ/engine law."""
from __future__ import annotations

import numpy as np
from scipy.special import pdtr


POLICY = {
    "version": 1,
    "purpose": "prospective_conditional_independent_ideal_Poisson_diagnostic_only",
    "strata": "positive_mean_source_branch_lt100_or_ge100_times_two_plus_SiPM_parity",
    "thresholds": ["0", "floor(max(0,mu-2*sqrt(mu)))", "floor(mu)",
                   "floor(mu+2*sqrt(mu))", "floor(mu+4*sqrt(mu))"],
    "family_alpha": 0.01,
    "family_size": 20,
    "correction": "two_sided_Hoeffding_Bonferroni_including_empty_and_redundant_pairs",
    "informative_max_CDF_radius": 0.05,
    "CDF_evaluation": "scipy.special.pdtr_float64_not_a_formal_numerical_error_certificate",
    "zero_mean_policy": "retained_in_ledger_and_checked_deterministically_not_discarded_as_missing_RO",
    "source_caveat": "RandPoissonQ_uses_skew_corrected_Gaussian_approximation_at_mu_ge100",
    "production_rng_validated": False,
    "stock_sampler_law_validated": False,
    "stochastic_response_certified": False,
    "same_seed_equivalence_claimed": False,
}


def diagnostic(mean, counts, sipm):
    """Condition on the complete input ledger, never on positive observed counts."""
    mean, counts, sipm = np.asarray(mean, np.float64), np.asarray(counts), np.asarray(sipm)
    if (mean.ndim != 1 or counts.shape != mean.shape or sipm.shape != mean.shape or
            not np.issubdtype(counts.dtype, np.integer) or not np.issubdtype(sipm.dtype, np.integer) or
            np.any(~np.isfinite(mean) | (mean < 0) | (mean > 2e9)) or np.any(counts < 0) or np.any(sipm < 0)):
        raise ValueError("finite bounded means and aligned nonnegative integer count/SiPM vectors required")
    if np.any(counts[mean == 0] != 0):
        raise ValueError("zero mean must have zero count")
    positive = mean > 0
    mu, observed, parity = mean[positive], counts[positive], sipm[positive] % 2
    strata = 2 * (mu >= 100).astype(np.int64) + parity
    thresholds = np.floor(np.column_stack((np.zeros(len(mu)), np.maximum(0, mu - 2 * np.sqrt(mu)),
                                           mu, mu + 2 * np.sqrt(mu), mu + 4 * np.sqrt(mu))))
    cdf = pdtr(thresholds, mu[:, None])
    if np.any(~np.isfinite(cdf) | (cdf < 0) | (cdf > 1)):
        raise ValueError("invalid numerically evaluated ideal Poisson CDF")
    expected = np.zeros((4, 5), np.float64)
    actual = np.zeros((4, 5), np.int64)
    np.add.at(expected, strata, cdf)
    np.add.at(actual, strata, observed[:, None] <= thresholds)
    totals = np.bincount(strata, minlength=4)
    populated = totals > 0
    radius = np.zeros(4)
    radius[populated] = np.sqrt(np.log(2 * POLICY["family_size"] / POLICY["family_alpha"]) / (2 * totals[populated]))
    difference = np.zeros((4, 5))
    difference[populated] = np.abs(actual[populated] - expected[populated]) / totals[populated, None]
    failed = populated[:, None] & (difference > radius[:, None])
    informative = populated & (radius <= POLICY["informative_max_CDF_radius"])
    return dict(policy=POLICY, ledger_rows=len(mean), zero_mean_rows=int(np.count_nonzero(~positive)),
                positive_mean_rows=len(mu), counts_by_branch_end=totals.reshape(2, 2).tolist(),
                observed_cumulative_counts=actual.tolist(), expected_cumulative_counts=expected.tolist(),
                CDF_radius_by_branch_end=radius.reshape(2, 2).tolist(),
                max_CDF_abs_difference_by_branch_end=difference.max(axis=1).reshape(2, 2).tolist(),
                maximum_CDF_abs_difference=float(difference.max(initial=0)),
                maximum_deviation_over_radius=float(np.max(difference[populated] / radius[populated, None], initial=0)),
                failed_stratum_threshold_pairs=np.argwhere(failed).tolist(),
                populated_strata=int(populated.sum()), informative_strata=int(informative.sum()),
                ideal_Poisson_model_rejected=bool(np.any(failed)),
                model_diagnostic_status=("rejected" if np.any(failed) else "not_rejected" if np.all(informative)
                                         else "insufficient_coverage_or_power"),
                interpretation="ideal_independent_Poisson_only_not_actual_RandPoissonQ_or_MixMax_law",
                production_rng_validated=False, stock_sampler_law_validated=False,
                stochastic_response_certified=False)
