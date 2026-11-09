"""Fractions with Jeffreys intervals (the analyses quote Jeffreys per sector because misses are few)."""
from __future__ import annotations

import numpy as np


def jeffreys(k: int, n: int, cl: float = 0.6827):
    """Central Jeffreys interval for k successes in n: Beta(k + 1/2, n - k + 1/2) quantiles (0/1 at the edges)."""
    from scipy.stats import beta
    if n <= 0:
        return (float("nan"), float("nan"))
    a = (1 - cl) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 0.5, n - k + 0.5))
    return lo, hi


def fraction(flags, weights=None, cl: float = 0.6827) -> dict:
    """k/n of a boolean array, with its Jeffreys interval (unweighted counts) and the weighted fraction if given."""
    f = np.asarray(flags, bool)
    k, n = int(f.sum()), int(f.size)
    lo, hi = jeffreys(k, n, cl)
    out = {"k": k, "n": n, "frac": k / n if n else float("nan"), "lo": lo, "hi": hi, "cl": cl}
    if weights is not None:
        w = np.asarray(weights, float)
        out["frac_weighted"] = float((w * f).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return out
