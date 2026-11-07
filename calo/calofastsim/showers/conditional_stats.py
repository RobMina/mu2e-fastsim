"""Fixed-query conditional moments; no draws, fitted corrections, or data access.

The caller supplies original stored-row // 100 block labels, before applying either
selection.  All supplied blocks remain in the delete-block universe, even blocks
selected in neither arm.  These errors do not include model/training uncertainty.
"""
from __future__ import annotations

import numpy as np


_FIELDS = ("mean", "std", "covariance", "correlation")


def _prefix(y, x):
    """Vectorized running means and centered sums, with an empty leading row.

    The weighted online variance identity is evaluated using cumulative means;
    there is no Python iteration over observations.  Unlike E[y*y]-E[y]**2,
    the variance is a sum of nonnegative terms.  A local origin also protects
    suffixes after deleting an extreme block.
    """
    n = np.arange(1, len(y) + 1, dtype=float)
    dy, dx = y - y[0], x - x[0]
    my, mx = np.cumsum(dy) / n, np.cumsum(dx) / n
    ry = dy - np.r_[0., my[:-1]]
    rx = dx - np.r_[0., mx[:-1]]
    weight = (n - 1) / n
    return np.vstack((np.zeros(5), np.column_stack((
        my, mx,
        np.cumsum(weight * ry * ry), np.cumsum(weight * rx * rx),
        np.cumsum(weight * ry * rx)))))


def _statistics(n, summary, origin):
    valid = n > 0
    denominator = np.where(valid, n, 1)
    vy, vx = summary[..., 2] / denominator, summary[..., 3] / denominator
    covariance = summary[..., 4] / denominator
    std, xstd = np.sqrt(vy), np.sqrt(vx)
    correlation = np.full(np.shape(n), np.nan)
    np.divide(covariance, std, out=correlation, where=valid & (std > 0) & (xstd > 0))
    np.divide(correlation, xstd, out=correlation, where=xstd > 0)
    # Cauchy-Schwarz can be exceeded by floating-point roundoff only.
    correlation = np.clip(correlation, -1., 1.)
    return dict(mean=np.where(valid, origin + summary[..., 0], np.nan),
                std=np.where(valid, std, np.nan),
                covariance=np.where(valid, covariance, np.nan), correlation=correlation)


def _arm(y, x, inverse, mask, blocks):
    take = np.flatnonzero(mask)
    take = take[np.argsort(inverse[take], kind="stable")]
    n = len(take)
    if not n:
        return dict(n=0, **{key: None for key in _FIELDS}), {
            key: np.full(blocks, np.nan) for key in _FIELDS}
    y, x, labels = y[take], x[take], inverse[take]
    # Separate endpoint origins preserve small fluctuations after deleting an extreme block.
    oy, ox = y[0], x[0]
    origin_dy, origin_dx = y[-1] - oy, x[-1] - ox
    prefix, suffix = _prefix(y, x), _prefix(y[::-1], x[::-1])
    left_n = np.searchsorted(labels, np.arange(blocks), side="left")
    right_n = n - np.searchsorted(labels, np.arange(blocks), side="right")
    left, right = prefix[left_n], suffix[right_n]
    remaining = left_n + right_n
    denominator = np.maximum(remaining, 1)
    right_weight = right_n / denominator
    cross_weight = left_n * right_weight
    dy = origin_dy + (right[:, 0] - left[:, 0])
    dx = origin_dx + (right[:, 1] - left[:, 1])
    leave = np.column_stack((
        left[:, 0] + right_weight * dy,
        left[:, 1] + right_weight * dx,
        left[:, 2] + right[:, 2] + cross_weight * dy * dy,
        left[:, 3] + right[:, 3] + cross_weight * dx * dx,
        left[:, 4] + right[:, 4] + cross_weight * dy * dx))
    # Empty-side combinations use the other centered moments with prefix-relative means.
    only_right = left_n == 0
    leave[only_right] = right[only_right]
    leave[only_right, 0] += origin_dy
    leave[only_right, 1] += origin_dx
    full = _statistics(np.asarray(n), prefix[-1], oy)
    # Unshifted sums preserve residuals when positive and negative values cancel.
    full["mean"] = oy if np.all(y == oy) else np.mean(y)
    absent = remaining == n
    leave[absent] = prefix[-1]
    # A constant arm-specific offset cancels from the jackknife dispersion.
    deleted = _statistics(remaining, leave, 0.)
    return dict(n=n, **{key: _number(full[key]) for key in _FIELDS}), deleted


def _number(value):
    return float(value) if np.isfinite(value) else None


def _mask(value, n, name):
    if value is None:
        return np.ones(n, dtype=bool)
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype.kind != "b":
        raise ValueError(f"{name} must be a one-dimensional boolean mask of length {n}")
    return value


def moments(fast, stock, x, group, fast_mask=None, stock_mask=None):
    """Compare population moments against the SAME fixed per-query covariate x.

    Inputs are equally sized 1-D arrays. ``group`` contains integer labels such
    as original stored-row // 100 (not selected-row indices). Optional boolean
    masks select each arm separately, with its own denominator and selected-x
    mean. For jointly-selected results pass the same intersection mask twice;
    do not subset away the original block universe first. Zeros are observations.
    Nonfinite y/x values are rejected wherever that arm is selected, and ignored
    only where explicitly unselected. Nonfinite/malformed block labels fail.

    Returns::

        {"fast": {"n": int, "mean": ..., "std": ..., "covariance": ...,
                  "correlation": ...},
         "stock": {same fields},
         "difference": {"mean": ..., "std": ..., "covariance": ...,
                        "correlation": ...},
         "error": {same four fields}}

    Differences are fast minus stock. Covariance and std use population (1/n)
    normalization, not 1/(n-1). Empty statistics and degenerate correlations are
    None; a singleton has std=covariance=0. Errors use sqrt((B-1)/B sum_b
    (delta_without_b - average_delta_without_b)**2), with B the number of unique
    ORIGINAL supplied blocks. A missing full statistic, B<2, any empty deletion,
    or any undefined deleted statistic makes that statistic's error None.
    No Python event/block loops or stochastic operations are used.
    """
    fast, stock, x = (np.asarray(value, dtype=float) for value in (fast, stock, x))
    group = np.asarray(group)
    if fast.ndim != 1 or stock.shape != fast.shape or x.shape != fast.shape or group.shape != fast.shape:
        raise ValueError("fast, stock, x and group must be equally sized one-dimensional arrays")
    # Empty Python lists naturally have float dtype and are valid empty inputs.
    if len(group) and group.dtype.kind not in "iu":
        raise ValueError("group must contain integer original block labels")
    fm = _mask(fast_mask, len(fast), "fast_mask")
    sm = _mask(stock_mask, len(stock), "stock_mask")
    for values, selected, name in ((fast, fm, "fast"), (stock, sm, "stock")):
        if not np.all(np.isfinite(values[selected])) or not np.all(np.isfinite(x[selected])):
            raise ValueError(f"nonfinite {name} or x among selected queries")
    unique, inverse = np.unique(group, return_inverse=True)
    blocks = len(unique)
    f, fd = _arm(fast, x, inverse, fm, blocks)
    s, sd = _arm(stock, x, inverse, sm, blocks)
    difference, error = {}, {}
    for key in _FIELDS:
        difference[key] = None if f[key] is None or s[key] is None else f[key] - s[key]
        delta = fd[key] - sd[key]
        if difference[key] is None or blocks < 2 or not np.all(np.isfinite(delta)):
            error[key] = None
        else:
            # Center before taking a mean to avoid unnecessary cancellation.
            centered = delta - delta[0]
            centered -= np.mean(centered)
            error[key] = _number(np.sqrt((blocks - 1) / blocks * np.sum(centered ** 2)))
    return dict(fast=f, stock=s, difference=difference, error=error)
