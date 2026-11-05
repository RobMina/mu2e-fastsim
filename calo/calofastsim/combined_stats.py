"""Prospective joint count/arrival diagnostic under an explicitly ideal marked-Poisson model."""
from __future__ import annotations

import numpy as np
from scipy.special import pdtr

from .travel import arrival_support, check_support


POLICY = {
    "version": 1,
    "null_assumption": "independent_exact_Poisson_counts_and_independent_iid_ideal_uniform_double_travel_marks",
    "strata": "positive_mu_lt100_or_ge100_times_two_plus_SiPM_parity",
    "arrival_threshold": "float32(base+center[nTime//2]),last_equal_bin_CDF",
    "contrasts": ["N>floor(mu)_and_first_generated_PE_at_or_below_arrival_threshold",
                  "Y<=0", "Y<=floor(mu*p)", "Y<=floor(mu*p+2*sqrt(mu*p))"],
    "Y": "number_of_PE_at_or_below_input_conditional_arrival_threshold_including_zero_count_inputs",
    "family_size": 16,
    "family_alpha": 0.01,
    "rounding_allowance": float(2**-24 + 2**-50),
    "thinning_CDF_allowance": "stratum_average_min(1,mu*rounding_allowance)_by_Poisson_coupling",
    "informative_max_CDF_radius": 0.05,
    "CDF_evaluation": "float64_scipy.special.pdtr_not_formally_error_certified",
    "first_PE": "original_generated_vector_order_not_minimum_or_resorted_arrival",
    "production_rng_validated": False,
    "stock_sampler_law_validated": False,
    "stochastic_response_certified": False,
}


def diagnostic(mean, counts, sipm, base, depth, readout_order, observed, cdf, times, dz):
    """Use all inputs for joint marks/counts, rather than conditioning on positive readouts."""
    mean, counts, sipm = np.asarray(mean, np.float64), np.asarray(counts), np.asarray(sipm)
    order = np.asarray(readout_order)
    if (mean.ndim != 1 or counts.shape != mean.shape or sipm.shape != mean.shape or
            not np.issubdtype(counts.dtype, np.integer) or not np.issubdtype(sipm.dtype, np.integer) or
            np.any(~np.isfinite(mean) | (mean < 0) | (mean > 2e9)) or np.any(counts < 0) or np.any(sipm < 0) or
            order.ndim != 1 or not np.issubdtype(order.dtype, np.integer) or
            np.any((order < 0) | (order >= len(counts))) or
            not np.array_equal(np.sort(order), np.flatnonzero(counts > 0))):
        raise ValueError("complete bounded mean/count/SiPM ledger and exact positive-readout permutation required")
    if np.any(counts[mean == 0] != 0):
        raise ValueError("zero mean must have zero count")
    candidates, rows = arrival_support(base, depth, cdf, times, dz)
    if len(candidates) != len(mean) or np.any(~np.isfinite(candidates)):
        raise ValueError("aligned finite candidate arrival thresholds required")
    observed = np.asarray(observed, np.float32)
    positive_counts = counts[order]
    support = check_support(np.asarray(base)[order], np.asarray(depth)[order], positive_counts,
                            observed, cdf, times, dz)
    if not support["all_arrivals_in_conditional_float32_support"]:
        raise ValueError("conditional arrival support must pass before joint diagnostic")
    j = len(times) // 2
    threshold = candidates[:, j]
    last_equal = np.sum(candidates <= threshold[:, None], axis=1) - 1
    probability = np.asarray(cdf, np.float64)[rows, last_equal]
    pe_ledger = np.repeat(order, positive_counts)
    early = observed <= threshold[pe_ledger]
    thinned = np.bincount(pe_ledger, weights=early.astype(np.int64), minlength=len(mean)).astype(np.int64)
    first_early = np.zeros(len(mean), bool)
    offsets = np.cumsum(positive_counts) - positive_counts
    first_early[order] = observed[offsets] <= threshold[order]
    positive = mean > 0
    mu, p, n, y = mean[positive], probability[positive], counts[positive], thinned[positive]
    strata = 2 * (mu >= 100).astype(np.int64) + sipm[positive] % 2
    marked_mean = mu * p
    thresholds = np.floor(np.column_stack((np.zeros(len(mu)), marked_mean,
                                          marked_mean + 2 * np.sqrt(marked_mean))))
    expected_probabilities = np.column_stack(((1 - pdtr(np.floor(mu), mu)) * p,
                                              pdtr(thresholds, marked_mean[:, None])))
    indicators = np.column_stack(((n > np.floor(mu)) & first_early[positive], y[:, None] <= thresholds))
    if np.any(~np.isfinite(expected_probabilities) | (expected_probabilities < 0) | (expected_probabilities > 1)):
        raise ValueError("invalid numerically evaluated ideal joint probabilities")
    expected, actual = np.zeros((4, 4)), np.zeros((4, 4), np.int64)
    np.add.at(expected, strata, expected_probabilities)
    np.add.at(actual, strata, indicators)
    totals = np.bincount(strata, minlength=4)
    populated = totals > 0
    delta = POLICY["rounding_allowance"]
    correction = np.zeros((4, 4))
    np.add.at(correction, strata, np.column_stack((np.full(len(mu), delta),
                                                 np.repeat(np.minimum(1, mu * delta)[:, None], 3, axis=1))))
    radius = np.zeros((4, 4))
    radius[populated] = (np.sqrt(np.log(2 * POLICY["family_size"] / POLICY["family_alpha"]) /
                                      (2 * totals[populated, None])) + correction[populated] / totals[populated, None])
    difference = np.zeros((4, 4))
    difference[populated] = np.abs(actual[populated] - expected[populated]) / totals[populated, None]
    failed = populated[:, None] & (difference > radius)
    informative = populated & np.all(radius <= POLICY["informative_max_CDF_radius"], axis=1)
    return dict(policy=POLICY, ledger_rows=len(mean), positive_mean_rows=len(mu), zero_mean_rows=int((~positive).sum()),
                arrival_threshold_table_index=j, counts_by_branch_end=totals.reshape(2, 2).tolist(),
                observed_indicator_counts=actual.tolist(), expected_indicator_counts=expected.tolist(),
                simultaneous_radius_by_stratum_contrast=radius.tolist(),
                absolute_difference_by_stratum_contrast=difference.tolist(),
                maximum_absolute_difference=float(difference.max(initial=0)),
                maximum_deviation_over_radius=float(np.max(difference[populated] / radius[populated], initial=0)),
                failed_stratum_contrast_pairs=np.argwhere(failed).tolist(),
                populated_strata=int(populated.sum()), informative_strata=int(informative.sum()),
                ideal_joint_model_rejected=bool(np.any(failed)),
                model_diagnostic_status=("rejected" if np.any(failed) else "not_rejected" if np.all(informative)
                                         else "insufficient_coverage_or_power"),
                interpretation="ideal_marked_Poisson_only_not_actual_RandPoissonQ_or_MixMax_law",
                production_rng_validated=False, stock_sampler_law_validated=False,
                stochastic_response_certified=False)
