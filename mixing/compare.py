"""Paired per-event occupancy statistics for straw digis and CRV pulses."""
import numpy as np
import awkward as ak


def occupancy(channel_ids, channel_to_bin, n_bins):
    """Return integer event-by-bin counts, including empty events and empty bins.

    channel_to_bin must come from the sample's geometry (plane or CRV sector),
    not from a top-only selection. This counts digis/pulses, not distinct channels.
    """
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    channels = ak.to_numpy(ak.flatten(channel_ids))
    mapping = np.asarray(channel_to_bin)
    if mapping.ndim != 1:
        raise ValueError("channel_to_bin must be one-dimensional")
    if channels.dtype.kind not in "iu" or mapping.dtype.kind not in "iu":
        raise ValueError("channel IDs and geometry mapping must be integers")
    if np.any(channels < 0) or np.any(channels >= mapping.size):
        raise ValueError("channel outside geometry map")
    bins = mapping[channels]
    if np.any(bins < 0) or np.any(bins >= n_bins):
        raise ValueError("geometry bin outside requested range")
    event = np.repeat(np.arange(len(channel_ids)), ak.to_numpy(ak.num(channel_ids)))
    return np.bincount(event * n_bins + bins, minlength=len(channel_ids) * n_bins).reshape(-1, n_bins)


def paired_counts(stock, fast):
    """Difference, ratio and paired errors per bin; caller aligns the event IDs."""
    stock, fast = np.asarray(stock, dtype=float), np.asarray(fast, dtype=float)
    if stock.ndim != 2 or stock.shape != fast.shape or stock.shape[0] < 2:
        raise ValueError("two or more paired events with equal event-by-bin shapes required")
    if np.any(~np.isfinite(stock)) or np.any(~np.isfinite(fast)) or np.any(stock < 0) or np.any(fast < 0):
        raise ValueError("counts must be finite and nonnegative")
    n = stock.shape[0]
    reference, candidate = stock.mean(axis=0), fast.mean(axis=0)
    diff = fast - stock
    se = diff.std(axis=0, ddof=1) / np.sqrt(n)
    ratio = np.divide(candidate, reference, out=np.full_like(reference, np.nan), where=reference > 0)
    influence = fast - stock * ratio
    ratio_se = np.divide(influence.std(axis=0, ddof=1) / np.sqrt(n), reference,
                         out=np.full_like(reference, np.nan), where=reference > 0)
    difference = diff.mean(axis=0)
    pull = np.divide(difference, se, out=np.full_like(se, np.nan), where=se > 0)
    deterministic = (se == 0) & (difference != 0)
    pull[deterministic] = np.copysign(np.inf, difference[deterministic])
    active = se > 0
    chi2 = float(np.sum(pull[active] ** 2)) if not np.any(deterministic) else float("inf")
    return {"events": n, "stock": reference, "fast": candidate, "difference": difference,
            "difference_error": se, "ratio": ratio, "ratio_error": ratio_se, "pull": pull,
            "chi2_diagonal": chi2, "ndf_diagonal": int(active.sum()),
            "zero_variance_nonzero_difference": deterministic}
