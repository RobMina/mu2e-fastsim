"""Fixed ideal-uniform conditional-CDF diagnostic, not a certificate for a PRNG stream."""
from __future__ import annotations

import numpy as np

from .travel import arrival_support, check_support


POLICY = {
    "version": 1,
    "purpose": "separate_post_support_ideal_uniform_CDF_diagnostic",
    "strata": "stock_depth_row_times_two_plus_SiPM_parity",
    "thresholds": "every_table_center_after_float32_addition_and_equal_arrival_collapse",
    "family_alpha": 0.01,
    "correction": "two_sided_Hoeffding_Bonferroni_over_all_depths_ends_thresholds",
    "rounding_allowance": float(2**-24 + 2**-50),
    "rounding_scope": "conservative_CDF_allowance_for_ideal_uniform_double_then_float32_not_actual_engine_law",
    "informative_max_CDF_radius": 0.05,
    "null_assumption": "conditional_independent_uniform_doubles_before_float32_narrowing",
    "seed_matching": False,
    "production_rng_validated": False,
    "stochastic_response_certified": False,
}


def diagnostic(base, depth, counts, observed, sipm, cdf, times, dz):
    """Fixed per-depth/end CDF contrasts with non-asymptotic conditional null bounds."""
    support = check_support(base, depth, counts, observed, cdf, times, dz)
    if not support["all_arrivals_in_conditional_float32_support"]:
        raise ValueError("conditional support must pass before distribution diagnostic")
    candidates, rows = arrival_support(base, depth, cdf, times, dz)
    if np.any(~np.isfinite(candidates)):
        raise ValueError("diagnostic requires finite candidate thresholds, including unreachable bins")
    counts, observed, sipm = np.asarray(counts), np.asarray(observed, np.float32), np.asarray(sipm)
    if (sipm.shape != counts.shape or not np.issubdtype(sipm.dtype, np.integer) or np.any(sipm < 0)):
        raise ValueError("nonnegative integer SiPM per readout required")
    ndepth, ntime = np.shape(cdf)
    strata = 2 * rows + sipm % 2
    pe_readout = np.repeat(np.arange(len(counts)), counts)
    lo = np.zeros(len(observed), np.int64)
    hi = np.full(len(observed), ntime, np.int64)
    for _ in range(ntime.bit_length()):
        mid = (lo + hi) // 2
        less = candidates[pe_readout, np.minimum(mid, ntime - 1)] < observed
        active = lo < hi
        lo = np.where(active & less, mid + 1, lo)
        hi = np.where(active & ~less, mid, hi)
    histogram = np.bincount(strata[pe_readout] * ntime + lo, minlength=2 * ndepth * ntime).reshape(2 * ndepth, ntime)
    observed_cumulative = np.cumsum(histogram, axis=1)
    last_equal = np.column_stack((np.diff(candidates, axis=1) != 0, np.ones(len(candidates), bool)))
    upper = np.minimum.accumulate(np.where(last_equal, np.arange(ntime), ntime)[:, ::-1], axis=1)[:, ::-1]
    conditional_cdf = np.asarray(cdf, np.float64)[rows[:, None], upper]
    expected_cumulative = np.zeros((2 * ndepth, ntime), np.float64)
    np.add.at(expected_cumulative, strata, conditional_cdf * counts[:, None])
    totals = np.bincount(strata, weights=counts, minlength=2 * ndepth).astype(np.int64)
    active_strata = totals > 0
    tests = 2 * ndepth * ntime
    radius = np.full(2 * ndepth, np.nan)
    radius[active_strata] = np.sqrt(np.log(2 * tests / POLICY["family_alpha"]) / (2 * totals[active_strata]))
    radius[active_strata] += POLICY["rounding_allowance"]
    difference = np.zeros_like(expected_cumulative)
    difference[active_strata] = np.abs(observed_cumulative[active_strata] - expected_cumulative[active_strata]) / totals[active_strata, None]
    failed = active_strata[:, None] & (difference > radius[:, None])
    informative = active_strata & (radius <= POLICY["informative_max_CDF_radius"])
    return {
        "policy": POLICY, "photoelectrons": len(observed), "family_size_including_empty_strata": tests,
        "populated_strata": int(np.count_nonzero(active_strata)),
        "empty_strata": int(np.count_nonzero(~active_strata)),
        "informative_strata": int(np.count_nonzero(informative)),
        "underpowered_populated_strata": int(np.count_nonzero(active_strata & ~informative)),
        "counts_by_depth_end": totals.reshape(ndepth, 2).tolist(),
        "max_CDF_abs_difference_by_depth_end": difference.max(axis=1).reshape(ndepth, 2).tolist(),
        "CDF_radius_by_depth_end": np.where(active_strata, radius, 0).reshape(ndepth, 2).tolist(),
        "observed_counts_by_stratum_bin": histogram.tolist(),
        "expected_cumulative_counts_by_stratum_threshold": expected_cumulative.tolist(),
        "minimum_populated_stratum_PE": int(np.min(totals[active_strata], initial=np.iinfo(np.int64).max)) if np.any(active_strata) else 0,
        "maximum_stratum_PE": int(np.max(totals, initial=0)),
        "maximum_CDF_abs_difference": float(np.max(difference, initial=0)),
        "maximum_deviation_over_simultaneous_radius": float(np.max(difference[active_strata] / radius[active_strata, None], initial=0)),
        "failed_stratum_threshold_pairs": np.argwhere(failed).tolist(),
        "ideal_uniform_model_rejected": bool(np.any(failed)),
        "all_strata_informative": bool(np.all(informative)),
        "model_diagnostic_status": ("rejected" if np.any(failed) else
                                    "not_rejected" if np.all(informative) else "insufficient_coverage_or_power"),
        "interpretation": "conditional_iid_ideal_uniform_model_diagnostic_only_not_production_rng_or_full_response_validation",
        "production_rng_validated": False, "stochastic_distribution_certified": False,
        "stochastic_response_certified": False,
    }
