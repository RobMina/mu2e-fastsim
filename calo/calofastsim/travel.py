"""Conditional float32 arrival support for supplied PE times, without RNG generation."""
from __future__ import annotations

import numpy as np

from .propagation import validate_table


def arrival_support(base, depth, cdf, times, dz):
    """Return per-readout stock stored-arrival candidates and depth rows.

    Only the small time-bin axis is expanded. Interior equal-CDF bins cannot
    be selected by lower_bound; the first bin remains reachable by supplied
    uniform zero. No independent RNG equivalence is claimed.
    """
    cdf, times, dz = validate_table(cdf, times, dz)
    base, depth = np.broadcast_arrays(np.asarray(base, dtype=np.float32), np.asarray(depth, dtype=np.float32))
    if base.ndim != 1 or np.any(~np.isfinite(base)) or np.any(~np.isfinite(depth)):
        raise ValueError("finite one-dimensional base times and depths required")
    with np.errstate(over="ignore", invalid="ignore"):
        quotient = np.trunc(depth / dz).astype(np.float64)
    if np.any(~np.isfinite(quotient) | (quotient < -(2**31)) | (quotient >= 2**31)):
        raise ValueError("stock depth quotient must fit signed int32")
    if np.any(np.diff(times) <= 0):
        raise ValueError("ordered distinct time centers required for arrival support search")
    rows = np.clip(quotient, 0, len(cdf) - 1).astype(np.int64)
    with np.errstate(over="ignore"):
        candidates = base[:, None] + times[None, :]
    reachable = np.column_stack((np.ones(len(cdf), dtype=bool), np.diff(cdf, axis=1) > 0))[rows]
    if np.any(reachable & ~np.isfinite(candidates)):
        raise ValueError("reachable stored arrival overflow")
    return candidates, rows


def check_support(base, depth, counts, observed, cdf, times, dz):
    """Vectorized binary search per PE; never infer a travel bin by subtraction."""
    candidates, rows = arrival_support(base, depth, cdf, times, dz)
    counts = np.asarray(counts)
    observed = np.asarray(observed, dtype=np.float32)
    if (counts.shape != (len(candidates),) or not np.issubdtype(counts.dtype, np.integer) or
            np.any(counts <= 0) or observed.ndim != 1 or np.sum(counts, dtype=np.int64) != len(observed)):
        raise ValueError("positive counts must reconcile with all observed PE times")
    if np.any(~np.isfinite(observed)):
        raise ValueError("nonfinite PE arrival")
    ro = np.repeat(np.arange(len(counts)), counts)
    lo = np.zeros(len(observed), dtype=np.int64)
    hi = np.full(len(observed), len(times), dtype=np.int64)
    for _ in range(len(times).bit_length()):
        mid = (lo + hi) // 2
        less = candidates[ro, np.minimum(mid, len(times) - 1)] < observed
        active = lo < hi
        lo = np.where(active & less, mid + 1, lo)
        hi = np.where(active & ~less, mid, hi)
    selected = candidates[ro, np.minimum(lo, len(times) - 1)]
    first = lo.copy()
    lo = np.zeros(len(observed), dtype=np.int64)
    hi = np.full(len(observed), len(times), dtype=np.int64)
    for _ in range(len(times).bit_length()):
        mid = (lo + hi) // 2
        less_equal = candidates[ro, np.minimum(mid, len(times) - 1)] <= observed
        active = lo < hi
        lo = np.where(active & less_equal, mid + 1, lo)
        hi = np.where(active & ~less_equal, mid, hi)
    reachable = np.column_stack((np.ones(len(cdf), dtype=bool), np.diff(cdf, axis=1) > 0))[rows]
    prefix = np.column_stack((np.zeros(len(rows), dtype=np.int64), np.cumsum(reachable, axis=1)))
    supported = ((first < len(times)) & (selected.view(np.uint32) == observed.view(np.uint32)) &
                 (prefix[ro, lo] > prefix[ro, first]))
    collapsed = np.count_nonzero(np.diff(candidates, axis=1) == 0, axis=1)
    result = {
        "photoelectrons": len(observed), "unsupported_arrival_bits": int(np.count_nonzero(~supported)),
        "all_arrivals_in_conditional_float32_support": bool(np.all(supported)),
        "readouts_with_collapsed_arrival_bins": int(np.count_nonzero(collapsed)),
        "adjacent_collapsed_bins": int(np.sum(collapsed)),
        "depth_rows_observed": np.unique(rows).tolist(),
        "zero_travel_bin_support_is_not_physical_zero_certification": True,
        "production_rng_validated": False, "stochastic_distribution_validated": False,
    }
    return result
