"""Calibrated parametric estimate of the tracker flash deposit from the foil geometry.

This is a ranking surrogate, not a transport. It describes the stock flash
objective (tracker ionising deposit per generated beam electron) as

    flash = c0 + c1 * sum_i t_i * P_i(sigma) * exp((z_i - z_ref) / lam)

where foil ``i`` has full thickness ``t_i`` [mm], hole radius ``rin_i``, outer
radius ``rout_i`` and centre ``z_i``; ``P_i`` is the fraction of a 2D Gaussian
beam footprint of width ``sigma`` inside the annulus, ``exp(-rin^2/2 sigma^2) -
exp(-rout^2/2 sigma^2)``. The sum is the beam-weighted aluminium thickness, with
material nearer the tracker weighted up by the e-folding length ``lam``. ``c0``
carries the contributions that do not scale with target material.

The constants are global: no per-design correction is fitted. Calibrate them
with :func:`calibrate` on stock design evaluations and keep the calibration
sample separate from any ranking test (:func:`kfold`). The supported domain is
coaxial, untilted aluminium annular foils read by
:func:`beamfastsim.flash_geometry.from_offline` with otherwise stock external
hardware; anything else (materials, absorber changes, field changes) is outside
it and must be reported, not mapped onto this formula.
"""
import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

from . import flash_geometry, simpleconfig

#: Stack centre of the stock 37-foil target [mm, Mu2e z].
Z_REF = 6271.0
#: Flash budget of the stock foil studies, per generated beam electron [MeV].
BUDGET = 7.506758e-06


@dataclass(frozen=True)
class FlashParam:
    """Global constants of the parametric estimate (MeV per generated electron)."""
    c0: float
    c1: float
    sigma: float
    lam: float | None = None
    z_ref: float = Z_REF

    def __call__(self, stacks, dtype=np.float64):
        return predict(stacks, self, dtype=dtype)


@dataclass(frozen=True)
class Stack:
    """Foil arrays of one design: centres, hole and outer radii, full thickness [mm]."""
    z: np.ndarray
    rin: np.ndarray
    rout: np.ndarray
    thickness: np.ndarray


def stack_from_config(c):
    """Foil arrays from a parsed SimpleConfig mapping, with the flash reader's checks."""
    z, rin, rout, ht = flash_geometry._foils(c)
    return Stack(z, rin, rout, 2 * ht)


def stack_from_offline(geom_file, offline_root):
    """Foil arrays of an overlay resolved against ``offline_root`` (includes followed)."""
    path = os.path.join(offline_root, geom_file)
    return stack_from_config(simpleconfig.read(path, offline_root))


def feature(stack, sigma, lam=None, z_ref=Z_REF, dtype=np.float64):
    """Beam-weighted foil thickness [mm] of one design."""
    if not sigma > 0:
        raise ValueError("sigma must be positive")
    if lam is not None and not lam > 0:
        raise ValueError("lam must be positive or None")
    z, rin, rout, t = (np.asarray(a, dtype=dtype) for a in (stack.z, stack.rin, stack.rout, stack.thickness))
    if not (z.shape == rin.shape == rout.shape == t.shape) or z.ndim != 1:
        raise ValueError("foil arrays must be one-dimensional and equal length")
    s2 = dtype(2) * dtype(sigma) ** 2
    frac = np.exp(-rin ** 2 / s2) - np.exp(-rout ** 2 / s2)
    w = np.ones_like(z) if lam is None else np.exp((z - dtype(z_ref)) / dtype(lam))
    return np.sum(t * frac * w, dtype=dtype)


def predict(stacks, p, dtype=np.float64):
    m = np.array([feature(s, p.sigma, p.lam, p.z_ref, dtype) for s in stacks], dtype=dtype)
    return dtype(p.c0) + dtype(p.c1) * m


def calibrate(stacks, y, sigmas=np.arange(16.0, 52.0, 2.0), lams=(None, 500.0, 700.0, 1000.0, 1400.0, 2000.0, 3000.0),
              z_ref=Z_REF, dtype=np.float64):
    """Grid over (sigma, lam), relative least squares for (c0, c1). Returns a FlashParam.

    ``dtype`` is the feature precision; the two-column solve is always float64.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) != len(stacks) or len(y) < 3 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("need at least three positive finite targets, one per stack")
    best = None
    for s in sigmas:
        for lam in lams:
            m = np.array([feature(st, s, lam, z_ref, dtype) for st in stacks], dtype=np.float64)
            a = np.stack([np.ones_like(m), m], axis=1) / y[:, None]
            c, *_ = np.linalg.lstsq(a, np.ones_like(y), rcond=None)
            res = float(np.sum((a @ c - 1) ** 2))
            if best is None or res < best[0]:
                best = (res, float(s), lam, c)
    _, s, lam, c = best
    return FlashParam(float(c[0]), float(c[1]), s, None if lam is None else float(lam), z_ref)


def spearman(a, b):
    """Spearman rank correlation with average ranks for ties."""
    def rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x), dtype=np.float64)
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return sums[inv] / counts[inv]
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2))
    return float(np.sum(ra * rb) / den) if den > 0 else float("nan")


def metrics(pred, y, budget=BUDGET):
    pred, y = np.asarray(pred, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rel = pred / y - 1
    return {"n": int(len(y)), "spearman": spearman(pred, y), "pearson": float(np.corrcoef(pred, y)[0, 1]),
            "rel_bias": float(np.mean(rel)), "rel_rms": float(np.std(rel)), "rel_max": float(np.max(np.abs(rel))),
            "budget_agreement": float(np.mean((pred <= budget) == (y <= budget))) if budget else None}


def kfold(stacks, y, k=5, seed=1, dtype=np.float64, **kw):
    """Out-of-fold predictions: each design is predicted by constants calibrated without it."""
    y = np.asarray(y, dtype=np.float64)
    if not 2 <= k <= len(y):
        raise ValueError("need 2 <= k <= number of designs")
    perm = np.random.default_rng(seed).permutation(len(y))
    out = np.empty(len(y), dtype=dtype)
    for fold in np.array_split(perm, k):
        train = np.setdiff1d(perm, fold)
        p = calibrate([stacks[i] for i in train], y[train], dtype=dtype, **kw)
        out[fold] = predict([stacks[i] for i in fold], p, dtype=dtype)
    return out


def _table(path):
    """TSV with columns ``name``, ``geom`` (overlay path) and ``flash`` (stock value, optional)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or not {"name", "geom"} <= set(rows[0]):
        raise ValueError("table needs name and geom columns")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flash_param", description=__doc__.splitlines()[0])
    ap.add_argument("--table", required=True, help="TSV: name, geom[, flash]")
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--params", help="JSON FlashParam to predict with; omitted = calibrate on the table")
    ap.add_argument("--kfold", type=int, default=5, help="Out-of-fold ranking test when calibrating (0 = off)")
    ap.add_argument("--float32", action="store_true", help="Compute the feature in float32")
    ap.add_argument("--budget", type=float, default=BUDGET)
    ap.add_argument("--out", required=True, help="Output JSON")
    a = ap.parse_args(argv)
    dtype = np.float32 if a.float32 else np.float64
    rows = _table(a.table)
    stacks = [stack_from_offline(r["geom"], a.offline_root) for r in rows]
    y = np.array([float(r["flash"]) for r in rows]) if all(r.get("flash") for r in rows) else None
    out = {"dtype": np.dtype(dtype).name, "designs": len(rows)}
    if a.params:
        with open(a.params) as f:
            p = FlashParam(**json.load(f))
    elif y is not None:
        p = calibrate(stacks, y, dtype=dtype)
        if a.kfold:
            out["kfold"] = metrics(kfold(stacks, y, k=a.kfold, dtype=dtype), y, a.budget)
    else:
        ap.error("--params is required when the table has no flash column")
    pred = predict(stacks, p, dtype=dtype)
    out["params"] = asdict(p)
    out["predictions"] = {r["name"]: float(v) for r, v in zip(rows, pred)}
    if y is not None:
        out["in_sample" if not a.params else "fixed_params"] = metrics(pred, y, a.budget)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    json.dump({k: v for k, v in out.items() if k != "predictions"}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
