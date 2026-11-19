"""Score one fastsim commit on the three metrics: validity, throughput multiplier and SU per event.

The scorer recomputes everything from binned histograms and bench.json reports, so a score never rests on a
self-reported statistic (throughput is self-reported by construction, and the device label says where it ran).

Layout (one directory per workflow; the reference side is never in this repository):

    submission/<W>/<dtype>/hists.json     the fast arm, dtype = float32 and float64 (both required)
    submission/<W>/<dtype>/bench.json     mu2efastsim.bench report of the same workflow and dtype
    reference/<W>/stock/hists.json        the stock arm on the same generated events
    reference/<W>/floor/hists.json        stock with another seed (optional: without it the threshold is 2)
    reference/<W>/reference.json          {"stock_rate_per_slot": events/s, "boundary": "...", "mode": ...,
                                           "yield_tolerance": 0.01, "min_count": 0, "su_per_hour": {device: SU}}
    reference/<W>/stock/features.npz      per-cluster feature vectors (optional: turns on the classifier score;
                                           then submission/<W>/<dtype>/features.npz is required)

hists.json: {"hists": {key: H}, "yields": {name: Y}}, where H is one of
- counts: {"edges": [...], "counts": [...]} or {"edges": [...], "shards": {shard_id: [...]}};
- values: {"edges": [...], "values": [...], "errors": [...]} (a table or a derived quantity with its own per-bin
  error, e.g. a light-yield table, a rate per POT; a flattened N-d table takes edges 0..n and may carry "shape");
- efficiency: {"edges": [...], "passed": [...], "total": [...]} (per-bin binomial, e.g. inefficiency per momentum bin);
and Y is a count, {"shards": {shard_id: count}}, or a measured value {"value": v, "error": e} (e.g. stops per POT).
hists.json may also hold "peaks": {name: H} (counts or shards): the scorer fits the peak position of H in each arm
(CAL1's Gaussian core, gauss_core: a Gaussian fitted to the bin counts with weights 1/n over the window
[mu - 1.5 s, mu + 2 s], 6 iterations from mu = the median of the entries above "start_above" and s = "start_width",
both from reference.json "peak_fit") and scores the ratio as a yield; its standard error is the paired bootstrap of
the ratio over the shared shards (>= 3; "n_boot" replicates, default 200, fixed "seed"), else the two fit errors in
quadrature. (A delete-one-shard jackknife overestimates it: the fit starts from a median and picks its window bin by
bin, so it is not smooth in the data.)

Statistics, per histogram (the same ones the repository's validation code uses):
- mode "same_events" (default; crvfastsim.validation.compare): both arms on the same generated events, no scaling,
  chi2 = sum (n1 - n2)^2 / (n1 + n2) over bins with n1 + n2 > 0, ndf = number of those bins;
- mode "shape": the candidate scaled to the reference total s = N1/N2, chi2 = sum (n1 - s n2)^2 / (n1 + s^2 n2),
  ndf = bins - 1;
- shards in both arms (trkfastsim.validation.chain_pair): d_sb = c_B,sb - c_A,sb over the S shared shards,
  chi2 = sum_b D_b^2 / var_b, var_b = S/(S-1) sum_s (d_sb - mean_b)^2, over bins with var_b > 0 and at least
  min_count entries in the reference (a correct arm then has <chi2/ndf> = (S-1)/(S-3), which the floor absorbs).
  With mode "shape" and shards, the paired shape test instead: d_sb = c_B,sb x (T_A/T_B) - c_A,sb (the candidate
  scaled to the reference total, so the yield is scored only by the yields), bins with var_b > 0 and at least
  min_count entries in both arms together, ndf = bins - 1, chi2/ndf reported x (S-3)/(S-1), p from the raw sum
  (the Run 1A DQMVAL comparison exactly).
- values: chi2 = sum (v1 - v2)^2 / (e1^2 + e2^2) over bins with a positive error sum, ndf = those bins;
- efficiency: e = passed / total per arm, var = e (1 - e) / total with e moved to 1/(2 total) at 0 and 1, chi2 = sum
  (e1 - e2)^2 / (var1 + var2) over bins with total > 0 in both arms.
Flag rule (DQMVAL): chi2/ndf > max(2, 3 x floor) and p < 1e-3, where the floor is the same statistic for the floor
arm against stock (threshold 2 without a floor). A yield passes when |ratio - 1| - 3 se <= yield_tolerance
(per-yield tolerances in "yield_tolerances": {name: tol}). Statistical power: a yield named in "max_se":
{name: se_max} passes only when also se <= se_max (B1: the CE peak, se <= tolerance / 3), so a small sample cannot
pass the yield test trivially; each such yield row reports power_ok, and the arm reports power_ok (all of them).

Scores per workflow and dtype: validity = (1 - flagged fraction) x (fraction of yields that pass); throughput
multiplier = bench warm_rate / stock_rate_per_slot (in the reference's rate_unit, default events/s; bench.json
may name its own unit, which must match); jobs ratio = stock_jobs_per_unit / bench jobs_per_unit when both are given
(grid jobs replaced per unit of work, e.g. per table); a task whose reference has "stock_rate_per_slot": null
is scored on validity only (no bench.json needed); SU/event = su_per_hour[device] / (3600 warm_rate), only with
--private (SU numbers are never published). A bench.json with "host": "mu2eg4" (a model inside Geant4) is scored
against its own "baseline_rate" (model off, same node and primaries) instead of the per-slot stock rate. With
"max_memory_mb_per_thread" in reference.json, a bench "memory_mb_per_thread" is checked (memory_ok; reported only).
Classifier score (classifier.py, after CaloChallenge): with reference/<W>/stock/features.npz and "classifier":
{"features": [...], "min_energy": ..., "energy_feature": "energy", "seed": ...} in reference.json, cls_score =
clip(2 (1 - AUC), 0, 1) of a fixed logistic regression on quadratic features, fast against stock; reported per arm
and as the mean over arms (not part of validity). A floor features.npz gives the floor AUC (stock against stock).

    python -m benchmarks.fastsim_score.score SUBMISSION_DIR REFERENCE_DIR [--private] [--out scores.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:  # package import (python -m benchmarks.fastsim_score.score) or flat (the Codabench scoring program)
    from . import classifier as C
except ImportError:  # pragma: no cover - exercised by the Codabench worker test
    import classifier as C

DTYPES = ("float32", "float64")
P_FLAG = 1e-3
PEAK_ITERATIONS = 6
PEAK_BOOTSTRAP = 200
PEAK_SEED = 20261118


class ScoreError(ValueError):
    pass


# ---------------------------------------------------------------- chi2 survival function (no scipy needed)

def chi2_sf(x: float, k: int) -> float:
    """P(chi2_k > x): the regularized upper incomplete gamma Q(k/2, x/2) (series below a+1, continued fraction above)."""
    if k <= 0 or not math.isfinite(x):
        return float("nan")
    if x <= 0:
        return 1.0
    a, z = 0.5 * k, 0.5 * x
    lg = math.lgamma(a)
    if z < a + 1:
        term = s = 1.0 / a
        n = a
        for _ in range(10000):
            n += 1
            term *= z / n
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-z + a * math.log(z) - lg))
    b, c, d = z + 1 - a, 1e300, 1.0 / (z + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - lg) * h)


# ---------------------------------------------------------------- statistics

def _arr(h: dict, key: str) -> np.ndarray:
    return np.asarray(h[key], dtype=float)


def _check_edges(ref: dict, cand: dict, name: str) -> None:
    e1, e2 = _arr(ref, "edges"), _arr(cand, "edges")
    if e1.shape != e2.shape or not np.allclose(e1, e2, rtol=1e-9, atol=0):
        raise ScoreError(f"{name}: binning differs between the arms")


def _bins(h: dict, name: str) -> np.ndarray:
    if "counts" in h:
        v = _arr(h, "counts")
    elif "shards" in h:
        v = np.sum([np.asarray(s, float) for s in h["shards"].values()], axis=0)
    else:
        raise ScoreError(f"{name}: needs counts or shards")
    if v.shape != (len(h["edges"]) - 1,):
        raise ScoreError(f"{name}: {v.size} bins for {len(h['edges']) - 1} edges")
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ScoreError(f"{name}: counts must be finite and >= 0")
    return v


def kind_of(h: dict, name: str = "") -> str:
    if "values" in h:
        return "values"
    if "passed" in h:
        return "efficiency"
    if "counts" in h or "shards" in h:
        return "counts"
    raise ScoreError(f"{name}: needs counts, shards, values+errors or passed+total")


def _vec(h: dict, key: str, name: str) -> np.ndarray:
    v = _arr(h, key)
    if v.shape != (len(h["edges"]) - 1,):
        raise ScoreError(f"{name}: {v.size} {key} for {len(h['edges']) - 1} bins")
    if not np.all(np.isfinite(v)):
        raise ScoreError(f"{name}: {key} must be finite")
    return v


def _finish(chi2: float, ndf: int, kind: str) -> dict:
    if ndf == 0:
        return {"chi2": None, "ndf": 0, "chi2ndf": None, "p": None, "kind": kind}
    return {"chi2": chi2, "ndf": ndf, "chi2ndf": chi2 / ndf, "p": chi2_sf(chi2, ndf), "kind": kind}


def _values_stat(ref: dict, cand: dict, name: str) -> dict:
    v1, v2 = _vec(ref, "values", name), _vec(cand, "values", name)
    e1, e2 = _vec(ref, "errors", name), _vec(cand, "errors", name)
    if np.any(e1 < 0) or np.any(e2 < 0):
        raise ScoreError(f"{name}: errors must be >= 0")
    den = e1 ** 2 + e2 ** 2
    use = den > 0
    return _finish(float(((v1 - v2)[use] ** 2 / den[use]).sum()), int(use.sum()), "values")


def _eff_var(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    ee = np.clip(e, 0.5 / np.maximum(n, 1), 1 - 0.5 / np.maximum(n, 1))  # no zero variance at e = 0 or 1
    return e, np.divide(ee * (1 - ee), n, out=np.zeros_like(k), where=n > 0)


def _eff_stat(ref: dict, cand: dict, name: str) -> dict:
    k1, n1 = _vec(ref, "passed", name), _vec(ref, "total", name)
    k2, n2 = _vec(cand, "passed", name), _vec(cand, "total", name)
    if np.any(k1 < 0) or np.any(k2 < 0) or np.any(k1 > n1) or np.any(k2 > n2):
        raise ScoreError(f"{name}: need 0 <= passed <= total")
    e1, w1 = _eff_var(k1, n1)
    e2, w2 = _eff_var(k2, n2)
    use = (n1 > 0) & (n2 > 0)
    return _finish(float(((e1 - e2)[use] ** 2 / (w1 + w2)[use]).sum()), int(use.sum()), "efficiency")


def statistic(ref: dict, cand: dict, mode: str = "same_events", min_count: float = 0, name: str = "") -> dict:
    """(chi2, ndf, chi2/ndf, p, kind) of the candidate against the reference histogram."""
    _check_edges(ref, cand, name)
    k1, k2 = kind_of(ref, name), kind_of(cand, name)
    if k1 != k2:
        raise ScoreError(f"{name}: reference is {k1}, submission is {k2}")
    if k1 == "values":
        return _values_stat(ref, cand, name)
    if k1 == "efficiency":
        return _eff_stat(ref, cand, name)
    n1, n2 = _bins(ref, name), _bins(cand, name)
    shared = sorted(set(ref.get("shards", {})) & set(cand.get("shards", {})))
    if len(shared) >= 3:
        a = np.array([ref["shards"][s] for s in shared], float)
        b = np.array([cand["shards"][s] for s in shared], float)
        S = len(shared)
        if mode == "shape":  # paired shape test: the candidate scaled to the reference total (the yield is separate)
            ta, tb = a.sum(), b.sum()
            if (ta > 0) != (tb > 0):  # one arm empty: always flagged
                return {"chi2": None, "ndf": 0, "chi2ndf": math.inf, "p": 0.0, "kind": "one_empty"}
            if ta <= 0:
                return _finish(0.0, 0, "empty")
            d = b * (ta / tb) - a
            var = S / (S - 1) * ((d - d.mean(axis=0)) ** 2).sum(axis=0)
            use = (var > 0) & ((a + b).sum(axis=0) >= max(min_count, 1e-300))
            ndf = int(use.sum()) - 1
            if ndf <= 0:
                return _finish(0.0, 0, f"paired_shape_{S}_shards")
            raw = float((d.sum(axis=0)[use] ** 2 / var[use]).sum())
            # <raw/ndf> = (S-1)/(S-3) for a correct arm: report the corrected chi2/ndf, take p from the raw sum
            return {"chi2": raw, "ndf": ndf, "chi2ndf": raw / ndf * (S - 3) / (S - 1), "p": chi2_sf(raw, ndf),
                    "kind": f"paired_shape_{S}_shards"}
        d = b - a
        var = S / (S - 1) * ((d - d.mean(axis=0)) ** 2).sum(axis=0)
        use = (var > 0) & (a.sum(axis=0) >= max(min_count, 1e-300))
        chi2, ndf, kind = float((d.sum(axis=0)[use] ** 2 / var[use]).sum()), int(use.sum()), f"paired_{S}_shards"
    elif mode == "shape":
        N1, N2 = n1.sum(), n2.sum()
        if N1 <= 0 or N2 <= 0:
            return {"chi2": None, "ndf": 0, "chi2ndf": None, "p": None, "kind": "empty"}
        s = N1 / N2
        den = n1 + s * s * n2
        use = (den > 0) & (n1 >= min_count)
        chi2, ndf, kind = float(((n1 - s * n2)[use] ** 2 / den[use]).sum()), max(int(use.sum()) - 1, 0), "shape"
    elif mode == "same_events":
        den = n1 + n2
        use = (den > 0) & (n1 >= min_count)
        chi2, ndf, kind = float(((n1 - n2)[use] ** 2 / den[use]).sum()), int(use.sum()), "same_events"
    else:
        raise ScoreError(f"unknown mode {mode!r}")
    return _finish(chi2, ndf, kind)


# ---------------------------------------------------------------- peak position (CAL1's Gaussian core, NumPy only)

def _gauss_lsq(c: np.ndarray, h: np.ndarray, p0) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares Gaussian a exp(-(x-m)^2 / 2 s^2) to counts h at c, weights 1/h (Levenberg-Marquardt)."""
    wt = 1.0 / h
    p = np.array(p0, dtype=float)

    def model(p):
        z = (c - p[1]) / p[2]
        g = np.exp(-0.5 * z * z)
        return p[0] * g, np.column_stack([g, p[0] * g * z / p[2], p[0] * g * z * z / p[2]])

    f, j = model(p)
    chi = float((wt * (h - f) ** 2).sum())
    lam = 1e-3
    for _ in range(500):
        jw = j.T * wt
        a, g = jw @ j, jw @ (h - f)
        try:
            step = np.linalg.solve(a + lam * np.diag(np.diag(a)) + 1e-300 * np.eye(3), g)
        except np.linalg.LinAlgError:
            break
        pn = p + step
        if pn[2] == 0:
            lam *= 10
            continue
        fn, jn = model(pn)
        chin = float((wt * (h - fn) ** 2).sum())
        if chin <= chi:
            done = np.all(np.abs(step) <= 1e-12 * np.maximum(np.abs(p), 1e-12)) or chi - chin <= 1e-14 * max(chi, 1)
            p, f, j, chi = pn, fn, jn, chin
            lam = max(lam * 0.3, 1e-12)
            if done:
                break
        else:
            lam *= 10
            if lam > 1e12:
                break
    jw = j.T * wt
    try:
        cov = np.linalg.inv(jw @ j)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    return p, cov


def gauss_core(edges, counts, start_above: float | None = None, start_width: float | None = None) -> dict:
    """Peak of a binned spectrum, CAL1's rule: a Gaussian fitted to the bins (weights 1/n) over the window
    [mu - 1.5 s, mu + 2 s] of filled bins, 6 iterations from mu = the median of the entries in bins centred above
    start_above (default: the lower edge) and s = start_width (default 4 % of the range). B1: 1 MeV bins 40-115 MeV,
    start above 75 MeV, start width 3 MeV. Returns the peak, the width and the peak's fit error."""
    e, h = np.asarray(edges, float), np.asarray(counts, float)
    c = 0.5 * (e[1:] + e[:-1])
    above = c > (e[0] if start_above is None else float(start_above))
    if h[above].sum() <= 0:
        raise ScoreError("peak: no entries above the start of the fit")
    cum = np.cumsum(h[above])
    mu = float(c[above][np.searchsorted(cum, 0.5 * cum[-1])])
    sig = 0.04 * (e[-1] - e[0]) if start_width is None else float(start_width)
    p = cov = None
    for _ in range(PEAK_ITERATIONS):
        w = (c > mu - 1.5 * sig) & (c < mu + 2.0 * sig) & (h > 0)
        if w.sum() < 4:
            raise ScoreError("peak: fewer than 4 filled bins in the fit window")
        p, cov = _gauss_lsq(c[w], h[w], [h[w].max(), mu, sig])
        if not (np.all(np.isfinite(p)) and e[0] < p[1] < e[-1] and p[2] != 0):
            raise ScoreError("peak: the Gaussian core fit failed")
        mu, sig = float(p[1]), float(abs(p[2]))
    return {"peak": mu, "width": sig, "peak_error": float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else None}


def peak_ratio(ref: dict, cand: dict, tol: float, name: str = "", fit: dict | None = None) -> dict:
    """Ratio of the fitted peak positions, candidate / reference, with the paired shard-bootstrap error.
    ``fit``: {"start_above", "start_width"} of gauss_core, "n_boot", "seed"."""
    _check_edges(ref, cand, name)
    if kind_of(ref, name) != "counts" or kind_of(cand, name) != "counts":
        raise ScoreError(f"peak {name}: needs counts or shards")
    edges = ref["edges"]
    fit = fit or {}
    core = lambda h: gauss_core(edges, h, fit.get("start_above"), fit.get("start_width"))  # noqa: E731
    try:
        r, c = core(_bins(ref, name)), core(_bins(cand, name))
    except ScoreError as e:
        raise ScoreError(f"peak {name}: {e}") from None
    ratio = c["peak"] / r["peak"]
    shared = sorted(set(ref.get("shards", {})) & set(cand.get("shards", {})))
    out = {"kind": "peak", "ratio": ratio, "peak_reference": r["peak"], "peak_submission": c["peak"],
           "width_reference": r["width"], "width_submission": c["width"]}
    if len(shared) >= 3:
        a = np.array([ref["shards"][s] for s in shared], float)
        b = np.array([cand["shards"][s] for s in shared], float)
        S, n_boot = len(shared), int(fit.get("n_boot", PEAK_BOOTSTRAP))
        rng = np.random.default_rng(int(fit.get("seed", PEAK_SEED)))
        reps = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)   # the same shards in both arms
            try:
                reps.append(core(w @ b)["peak"] / core(w @ a)["peak"])
            except ScoreError as e:
                raise ScoreError(f"peak {name}: bootstrap replicate: {e}") from None
        se = float(np.std(reps, ddof=1))
        out["se_method"] = f"bootstrap_{n_boot}_of_{S}_shards"
    else:
        if r["peak_error"] is None or c["peak_error"] is None:
            raise ScoreError(f"peak {name}: no fit error")
        se = abs(ratio) * math.hypot(r["peak_error"] / r["peak"], c["peak_error"] / c["peak"])
        out["se_method"] = "fit_errors"
    out.update(se=se, **{"pass": abs(ratio - 1) - 3 * se <= tol})
    return out


def _count(y, name: str) -> tuple[float, dict | None]:
    if isinstance(y, dict):
        sh = {k: float(v) for k, v in y["shards"].items()}
        return sum(sh.values()), sh
    if not isinstance(y, (int, float)) or y < 0:
        raise ScoreError(f"yield {name}: not a count")
    return float(y), None


def _measured(y) -> bool:
    return isinstance(y, dict) and "value" in y


def yield_ratio(ref, cand, tol: float, name: str = "") -> dict:
    if _measured(ref) or _measured(cand):
        if not (_measured(ref) and _measured(cand)):
            raise ScoreError(f"yield {name}: one arm is a measured value, the other a count")
        r, er, c, ec = (float(ref["value"]), float(ref.get("error", 0)), float(cand["value"]),
                        float(cand.get("error", 0)))
        if r <= 0:
            raise ScoreError(f"yield {name}: reference value is <= 0")
        ratio = c / r
        se = abs(ratio) * math.sqrt((er / r) ** 2 + ((ec / c) ** 2 if c else 0.0))
        return {"ratio": ratio, "se": se, "pass": abs(ratio - 1) - 3 * se <= tol}
    r, rs = _count(ref, name)
    c, cs = _count(cand, name)
    if r <= 0:
        raise ScoreError(f"yield {name}: reference count is 0")
    ratio = c / r
    shared = sorted(set(rs or {}) & set(cs or {}))
    if len(shared) >= 2:
        d = np.array([cs[s] - rs[s] for s in shared])
        S = len(shared)
        se = math.sqrt(S / (S - 1) * ((d - d.mean()) ** 2).sum()) / r
    else:
        se = ratio * math.sqrt(1 / r + 1 / max(c, 1))
    return {"ratio": ratio, "se": se, "pass": abs(ratio - 1) - 3 * se <= tol}


# ---------------------------------------------------------------- one workflow, one dtype

def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        raise ScoreError(f"missing {p.as_posix()}") from None
    except ValueError as e:
        raise ScoreError(f"{p.as_posix()}: not JSON ({e})") from None


def score_arm(cand: dict, stock: dict, floor: dict | None, refcfg: dict) -> dict:
    mode, minc = refcfg.get("mode", "same_events"), float(refcfg.get("min_count", 0))
    tol0 = float(refcfg.get("yield_tolerance", 0.01))
    tols = {k: float(v) for k, v in (refcfg.get("yield_tolerances") or {}).items()}
    max_se = {k: float(v) for k, v in (refcfg.get("max_se") or {}).items()}
    names = set(stock.get("yields", {})) | set(stock.get("peaks", {}))
    if set(max_se) - names or set(tols) - names:
        raise ScoreError(f"reference.json names yields the stock arm lacks: {sorted((set(max_se) | set(tols)) - names)}")
    rows, flagged = [], 0
    for key, h_ref in sorted(stock.get("hists", {}).items()):
        if key not in cand.get("hists", {}):
            raise ScoreError(f"histogram {key} missing from the submission")
        st = statistic(h_ref, cand["hists"][key], mode, minc, key)
        fl = statistic(h_ref, floor["hists"][key], mode, minc, key) if floor and key in floor.get("hists", {}) else None
        f0 = fl["chi2ndf"] if fl and fl["chi2ndf"] is not None else None
        thr = max(2.0, 3.0 * f0) if f0 is not None else 2.0
        flag = st["chi2ndf"] is not None and st["chi2ndf"] > thr and st["p"] < P_FLAG
        flagged += flag
        rows.append({"hist": key, **st, "floor": f0, "threshold": thr, "flag": bool(flag)})
    yields = []
    for name, y_ref in sorted(stock.get("yields", {}).items()):
        if name not in cand.get("yields", {}):
            raise ScoreError(f"yield {name} missing from the submission")
        yields.append({"yield": name, **yield_ratio(y_ref, cand["yields"][name], tols.get(name, tol0), name)})
    for name, h_ref in sorted(stock.get("peaks", {}).items()):
        if name not in cand.get("peaks", {}):
            raise ScoreError(f"peak {name} missing from the submission")
        yields.append({"yield": name, **peak_ratio(h_ref, cand["peaks"][name], tols.get(name, tol0), name,
                                                   (refcfg.get("peak_fit") or {}).get(name))})
    for y in yields:
        y["tolerance"] = tols.get(y["yield"], tol0)
        if y["yield"] in max_se:  # statistical power: the error must be small enough for the tolerance to bite
            y["pass_tolerance"] = bool(y["pass"])
            y["max_se"] = max_se[y["yield"]]
            y["power_ok"] = bool(y["se"] <= max_se[y["yield"]])
            y["pass"] = y["pass_tolerance"] and y["power_ok"]
    n = len(rows)
    if n == 0 and not yields:
        raise ScoreError("the reference has no histograms and no yields")
    ypass = sum(y["pass"] for y in yields) / len(yields) if yields else 1.0
    hpass = 1 - flagged / n if n else 1.0  # a yields-only task (e.g. per-path accept fractions) scores the yields
    out = {"n_hists": n, "n_flagged": flagged, "validity": hpass * ypass, "yield_pass": ypass,
           "hists": rows, "yields": yields}
    if max_se:
        out["power_ok"] = all(y["power_ok"] for y in yields if "power_ok" in y)
    return out


def _selected_features(path: Path, cfg: dict) -> np.ndarray:
    names = list(cfg["features"])
    efeat = cfg.get("energy_feature", "energy")
    try:
        x = C.load_features(path, names + ([efeat] if efeat not in names else []))
    except (OSError, ValueError) as e:
        raise ScoreError(f"{path.as_posix()}: {e}") from None
    keep = x[:, -1 if efeat not in names else names.index(efeat)] >= float(cfg.get("min_energy", -np.inf))
    return x[keep][:, :len(names)]


def classifier_arm(cand_path: Path, stock_x: np.ndarray, cfg: dict) -> dict:
    if not cand_path.is_file():
        raise ScoreError(f"missing {cand_path.as_posix()} (the reference scores per-cluster features)")
    try:
        return C.classifier_score(stock_x, _selected_features(cand_path, cfg), int(cfg.get("seed", C.DEFAULT_SEED)),
                                  float(cfg.get("l2", C.DEFAULT_L2)),
                                  int(cfg.get("max_per_class", C.DEFAULT_MAX_PER_CLASS)))
    except ValueError as e:
        raise ScoreError(f"classifier: {e}") from None


def score_one(d: Path, w: str, dt: str, stock: dict, floor: dict | None, cfg: dict, private: bool,
              stock_x: np.ndarray | None = None) -> dict:
    arm = score_arm(load_json(d / "hists.json"), stock, floor, cfg)
    res = dict(arm)
    if stock_x is not None:
        res["classifier"] = classifier_arm(d / "features.npz", stock_x, cfg["classifier"])
        res["cls_score"] = res["classifier"]["cls_score"]
    if cfg.get("stock_rate_per_slot") is None:  # a task with no stock rate at its boundary: validity only
        res.update(throughput_multiplier=None, throughput_scored=False)
        return res
    bench = load_json(d / "bench.json")
    if bench.get("dtype") not in (None, dt):
        raise ScoreError(f"{w}/{dt}: bench.json says dtype {bench.get('dtype')}")
    unit = cfg.get("rate_unit", "events/s")
    if bench.get("rate_unit") not in (None, unit):
        raise ScoreError(f"{w}/{dt}: bench.json rate is in {bench.get('rate_unit')}, the reference in {unit}")
    rate = float(bench.get("warm_rate") or 0)
    if rate <= 0:
        raise ScoreError(f"{w}/{dt}: bench.json has no positive warm_rate")
    if bench.get("host") == "mu2eg4":  # a model inside Geant4: model-on over model-off on the same node and primaries
        base = float(bench.get("baseline_rate") or 0)
        if base <= 0:
            raise ScoreError(f"{w}/{dt}: a mu2eg4 bench.json needs a positive baseline_rate (model off)")
        mult, basis = rate / base, "same node, model off"
    else:
        mult, basis = rate / float(cfg["stock_rate_per_slot"]), "stock rate per grid slot"
    res.update(throughput_scored=True, warm_rate=rate, rate_unit=unit, device=bench.get("device"), inputs=bench.get("inputs"),
               throughput_multiplier=mult, throughput_basis=basis, throughput_verified=False,
               boundary_match=(bench.get("boundary") in (None, cfg.get("boundary"))))
    for k in ("host", "edge_rule", "library_hash", "offline", "envset", "image"):
        if k in bench:
            res[k] = bench[k]
    if cfg.get("max_memory_mb_per_thread") is not None:
        mem = bench.get("memory_mb_per_thread")
        res["memory_mb_per_thread"] = mem
        res["memory_ok"] = None if mem is None else bool(float(mem) <= float(cfg["max_memory_mb_per_thread"]))
    if cfg.get("stock_jobs_per_unit") and bench.get("jobs_per_unit"):
        res["jobs_ratio"] = float(cfg["stock_jobs_per_unit"]) / float(bench["jobs_per_unit"])
    if private:
        su = (cfg.get("su_per_hour") or {}).get(bench.get("device"))
        res["su_per_event"] = None if su is None else float(su) / (3600.0 * rate)
    return res


def score(sub: Path, ref: Path, private: bool = False, allow_missing: bool = False) -> dict:
    """Score every workflow of the reference. allow_missing (coverage runs, never a leaderboard) records a missing
    or unscorable arm under "missing" instead of failing."""
    workflows = sorted(p.name for p in ref.iterdir() if (p / "reference.json").is_file())
    if not workflows:
        raise ScoreError(f"no <W>/reference.json under {ref.as_posix()}")
    out: dict = {"workflows": {}, "missing": {}}
    vals, mults, clss = [], [], []
    for w in workflows:
        cfg = load_json(ref / w / "reference.json")
        stock = load_json(ref / w / "stock" / "hists.json")
        floor = load_json(ref / w / "floor" / "hists.json") if (ref / w / "floor" / "hists.json").is_file() else None
        stock_x = None
        if (ref / w / "stock" / "features.npz").is_file():
            if not cfg.get("classifier", {}).get("features"):
                raise ScoreError(f"{w}: stock/features.npz without a classifier feature list in reference.json")
            stock_x = _selected_features(ref / w / "stock" / "features.npz", cfg["classifier"])
            if (ref / w / "floor" / "features.npz").is_file():
                fl = classifier_arm(ref / w / "floor" / "features.npz", stock_x, cfg["classifier"])
                out.setdefault("classifier_floor", {})[w] = {"auc": fl["auc"], "cls_score": fl["cls_score"]}
        out["workflows"][w] = {}
        for dt in DTYPES:
            d = sub / w / dt
            try:
                if not d.is_dir():
                    raise ScoreError(f"{w}: no {dt} arm (both dtypes are required)")
                res = score_one(d, w, dt, stock, floor, cfg, private, stock_x)
            except ScoreError as e:
                if not allow_missing:
                    raise
                out["missing"].setdefault(w, {})[dt] = str(e)
                continue
            out["workflows"][w][dt] = res
            vals.append(res["validity"])
            if res["throughput_multiplier"] is not None:
                mults.append(res["throughput_multiplier"])
            if "cls_score" in res:
                clss.append(res["cls_score"])
    if not vals:
        raise ScoreError("no arm could be scored")
    out["validity_score"] = float(np.mean(vals))
    out["min_validity"] = float(np.min(vals))
    out["throughput_geomean"] = float(np.exp(np.mean(np.log(mults)))) if mults else None
    out["cls_score"] = float(np.mean(clss)) if clss else None
    out["n_workflows"] = len(workflows)
    out["n_arms_scored"] = len(vals)
    if not out["missing"]:
        del out["missing"]
    return out


def leaderboard_row(res: dict) -> dict:
    """The flat keys a Codabench leaderboard shows: overall scores plus validity and multiplier per workflow/dtype."""
    row = {k: res[k] for k in ("validity_score", "min_validity", "throughput_geomean")}
    if res.get("cls_score") is not None:
        row["cls_score"] = res["cls_score"]
    for w, by in res["workflows"].items():
        for dt, r in by.items():
            tag = f"{w.lower()}_{'f32' if dt == 'float32' else 'f64'}"
            row[f"{tag}_validity"] = r["validity"]
            row[f"{tag}_multiplier"] = r["throughput_multiplier"]  # None when the task has no stock rate
            if "cls_score" in r:
                row[f"{tag}_cls"] = r["cls_score"]
            if "power_ok" in r:
                row[f"{tag}_power"] = float(r["power_ok"])
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("submission", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--private", action="store_true", help="add SU per event (never for a public leaderboard)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="coverage run: list missing or unscorable arms instead of failing (never for a leaderboard)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    try:
        res = score(a.submission, a.reference, a.private, a.allow_missing)
    except ScoreError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    text = json.dumps(res, indent=1)
    a.out.write_text(text + "\n") if a.out else print(text)
    for w, by in res["workflows"].items():
        for dt, r in by.items():
            extra = (f", cls {r['cls_score']:.3f}" if "cls_score" in r else "") + \
                    (f", power {'ok' if r['power_ok'] else 'FAIL'}" if "power_ok" in r else "")
            print(f"{w} {dt}: flagged {r['n_flagged']}/{r['n_hists']}, yields {r['yield_pass']:.2f}, "
                  f"validity {r['validity']:.3f}{extra}, " + (f"x{r['throughput_multiplier']:.1f} on {r['device']}"
                  if r["throughput_scored"] else "throughput not scored"), file=sys.stderr)
    for w, by in res.get("missing", {}).items():
        for dt, why in by.items():
            print(f"{w} {dt}: not scored ({why})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
