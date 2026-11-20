"""Censored per-crossing EX response calibration: one estimator family.

Two estimators share the parameters and the output format:

- ``profile_fit`` / ``fit(..., include_missing=False)``: the closed-form positive-only estimator (weighted least
  squares with the per-channel intercepts profiled out). It uses recorded pulses only, ignores the selection, and is
  the start point of the censored fit.
- ``fit(...)``: the censored marked likelihood. Each eligible track/channel crossing contributes either the joint
  density of the selected pulse's log PE and time, or, when no pulse passed the frozen selection, the probability of
  missing it.

Response model for one crossing at fibre distance ``x`` on channel ``c``, relative to the expected (nominal) log PE
``q0`` and time ``t0`` supplied per crossing::

    log PE      ~ Normal(mu_q, sigma_q),   mu_q = q0 + g[c] - alpha * x,
                  sigma_q^2 = sigma_0^2 + excess_noise * exp(-mu_q)       (photostatistics: wider at low PE)
    time | PE   ~ Normal(mu_t - walk * (log PE - log_pe_ref), sigma_t),   mu_t = t0 + o[c] + beta * x

A pulse is selected when PE >= threshold and |time - t0| <= half window. ``alpha`` (inverse attenuation length) and
``beta`` (inverse signal speed) are shared corrections; ``g`` and ``o`` are per-channel absolute log gains and time
offsets, reported as a mean (``log_scale``, ``time_zero_ns``) plus zero-mean channel terms. ``sigma_0``
(``sigma_log_pe``), ``excess_noise``, ``sigma_t`` and ``walk`` (time walk, ns per unit log PE) are fitted unless
fixed. The seen/silent terms are hepsimlib's censored crossing likelihood (``hepsimlib.likelihood.censored``): with walk
the miss probability is a bivariate normal orthant sum, evaluated exactly (Genz), with an analytic gradient.
The expected ``q0``/``t0`` columns can come from any forward model (the analytic nominal of ``ex_observations``
today, an expected-mode fastsim response later).

Uncertainties: the observed information (Hessian of the summed negative log likelihood at the optimum) is assembled
from analytic per-crossing gradients; its inverse is the covariance, transformed to the reported gauge (zero-mean
channel terms). The fit is float64; ``predict`` evaluates in float32 or float64 for replay.
"""

import argparse
import hashlib
import json

import numpy as np
from hepsimlib.likelihood.censored import CrossingPrediction, Selection, censored_nll, p_selected
from scipy.optimize import minimize

SCHEMA = "mu2e-crv-ex-response-v2"
GLOBAL = ("inv_attenuation_per_m", "slowness_ns_per_m", "log_sigma_log_pe", "log_sigma_time_ns",
          "walk_ns_per_log_pe", "excess_noise")
MAX_CORRELATION = 0.99                        # walk box: |corr(log PE, time)| at the start widths stays below this
# Row-level predictors (same order as the globals): mu_q, mu_t, log sigma_0, log sigma_t, walk, excess noise.
NG = 6


# ---------------------------------------------------------------------------------------------------------------
# Positive-only closed form (formerly charge_time_calib)

def _profile(index, nchan, distance, residual, weights):
    """Weighted least squares of residual = intercept[channel] + slope * distance, intercepts profiled."""
    weight_sum = np.bincount(index, weights=weights, minlength=nchan)
    if not np.isfinite(weight_sum).all() or np.any(weight_sum <= 0):
        raise ValueError("each fitted channel needs a recorded pulse with finite positive weight; "
                         "exclude unsupported channels using fit data only")
    xd = np.bincount(index, weights=weights * distance, minlength=nchan) / weight_sum
    yd = np.bincount(index, weights=weights * residual, minlength=nchan) / weight_sum
    x, y = distance - xd[index], residual - yd[index]
    denominator = np.dot(weights, x * x)
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps * np.dot(weights, distance * distance):
        raise ValueError("the shared slope requires within-channel distance variation")
    slope = np.dot(weights * x, y) / denominator
    intercepts = yd - slope * xd
    errors = y - slope * x
    rms = np.sqrt(np.average(errors**2, weights=weights))
    dof = max(len(index) - nchan - 1, 1)
    slope_err = np.sqrt(np.dot(weights, errors**2) / dof / denominator)
    return slope, intercepts, rms, slope_err


def profile_fit(channel, distance_mm, pe, expected_pe, time_ns, expected_time_ns, *, charge_weight=None,
                time_weight=None):
    """Positive-only shared slopes and per-channel gains/offsets from matched pulses (closed form).

    ``log(pe / expected_pe) = log_scale + log_gain[c] - delta_attenuation * d`` and
    ``time - expected_time = time_zero + time_offset[c] + delta_slowness * d``. Weights, when given, are relative
    inverse variances in log PE and ns. Select finite, unsaturated observations before calling. The slope errors use
    the residual scatter (weighted least squares with an estimated scale). No threshold or missed-pulse model.
    """
    channel = np.asarray(channel)
    if channel.ndim != 1 or channel.size == 0 or channel.dtype.kind not in "iu" or np.any(channel < 0):
        raise ValueError("channel must contain nonnegative integer IDs")

    def values(array, name, *, positive=False):
        out = np.asarray(array, dtype=float)
        if out.shape != channel.shape or not np.isfinite(out).all() or (positive and np.any(out <= 0)):
            raise ValueError(f"{name} must be finite and aligned with channel" + (" and positive" if positive else ""))
        return out

    distance = values(distance_mm, "distance_mm")
    if np.any(distance < 0):
        raise ValueError("distance_mm must be nonnegative")
    channels, index = np.unique(channel, return_inverse=True)
    log_pe = np.log(values(pe, "pe", positive=True)) - np.log(values(expected_pe, "expected_pe", positive=True))
    dt = values(time_ns, "time_ns") - values(expected_time_ns, "expected_time_ns")
    wc = np.ones(len(channel)) if charge_weight is None else values(charge_weight, "charge_weight", positive=True)
    wt = np.ones(len(channel)) if time_weight is None else values(time_weight, "time_weight", positive=True)
    q_slope, g, q_rms, q_err = _profile(index, len(channels), distance, log_pe, wc)
    t_slope, o, t_rms, t_err = _profile(index, len(channels), distance, dt, wt)
    return dict(channel=channels.tolist(), log_scale=float(g.mean()), log_gain=(g - g.mean()).tolist(),
                delta_attenuation_per_mm=float(-q_slope), time_zero_ns=float(o.mean()),
                time_offset_ns=(o - o.mean()).tolist(), delta_slowness_ns_per_mm=float(t_slope),
                rms_log_pe=float(q_rms), rms_time_ns=float(t_rms),
                errors=dict(delta_attenuation_per_mm=float(q_err), delta_slowness_ns_per_mm=float(t_err)))


# ---------------------------------------------------------------------------------------------------------------
# Design

def _design(rows, channel_ids):
    seen = np.asarray(rows["seen"])
    eligible = np.asarray(rows["eligible"])
    channels = np.asarray(rows["channel"])
    if (eligible.shape != channels.shape or seen.shape != channels.shape or channels.ndim != 1
            or seen.dtype.kind != "b" or eligible.dtype.kind != "b" or channels.dtype.kind not in "iu"
            or np.any(channels < 0) or np.any(seen & ~eligible)):
        raise ValueError("invalid crossing labels")
    if not eligible.any():
        raise ValueError("no eligible crossing observations")
    live = channels[eligible]
    if channel_ids is None:
        channel_ids, index = np.unique(live, return_inverse=True)
    else:
        channel_ids = np.asarray(channel_ids)
        if channel_ids.ndim != 1 or np.any(channel_ids[1:] <= channel_ids[:-1]):
            raise ValueError("candidate channels must be sorted and unique")
        index = np.searchsorted(channel_ids, live)
    if (index >= len(channel_ids)).any() or np.any(channel_ids[index] != live):
        raise ValueError("prediction contains channels without a fitted response")

    def col(key):
        x = np.asarray(rows[key], dtype=np.float64)
        if x.shape != channels.shape:
            raise ValueError(f"{key} must have one value per crossing")
        return x[eligible]

    distance, q0, t0 = col("distance_mm"), col("nominal_log_pe"), col("nominal_time_ns")
    threshold, window, pe, tm = (col(k) for k in ("threshold_pe", "time_half_width_ns", "pe", "time_ns"))
    seen = seen[eligible]
    if (not np.isfinite([distance, q0, t0, threshold, window]).all()
            or np.any(distance < 0) or np.any(threshold <= 0) or np.any(window <= 0)):
        raise ValueError("invalid distance, selection, or nominal response")
    if (not np.isfinite(pe[seen]).all() or not np.isfinite(tm[seen]).all()
            or np.any(pe[seen] < threshold[seen])
            or np.any(np.abs(tm[seen] - t0[seen]) > window[seen] + 4 * np.spacing(np.abs(t0[seen])))):
        raise ValueError("seen observations must have finite PE and time that pass the frozen selection")
    return channel_ids, dict(index=index, x=distance / 1000, q0=q0, t0=t0, lthr=np.log(threshold),
                             lo=t0 - window, hi=t0 + window, seen=seen,
                             lpe=np.log(np.where(seen, pe, 1.)), tm=np.where(seen, tm, t0))


def _eta(theta, d, nchan):
    g, o = theta[NG:NG + nchan], theta[NG + nchan:]
    n = len(d["x"])
    return np.column_stack([d["q0"] + g[d["index"]] - theta[0] * d["x"], d["t0"] + o[d["index"]] + theta[1] * d["x"],
                            *(np.full(n, v) for v in theta[2:NG])])


def _charge_width(mu_q, ls0, excess):
    """log sigma_q per crossing: sigma_q^2 = sigma_0^2 + excess / expected PE (photostatistics)."""
    with np.errstate(over="ignore"):
        v = np.exp(2 * ls0) + excess * np.exp(-mu_q)
    return 0.5 * np.log(v), v


def _prediction(mu_q, mu_t, lsq, lst, walk, ref):
    return CrossingPrediction(mu_logq=mu_q, mu_t=mu_t, log_sigma_q=lsq, log_sigma_t=lst, walk=walk, logq_ref=ref)


def _core(mu_q, mu_t, lsq, lst, walk, d, ref):
    """Per-crossing NLL and its gradient in (mu_q, mu_t, log sigma_q, log sigma_t, walk): hepsimlib's censored
    crossing likelihood (exact bivariate-normal miss probability, analytic gradient)."""
    return censored_nll(_prediction(mu_q, mu_t, lsq, lst, walk, ref), Selection(d["lthr"], d["lo"], d["hi"]),
                        d["seen"], d["lpe"], d["tm"], grad=True)


def _rows(eta, d, ref):
    """Per-crossing NLL terms and their gradient with respect to the row predictors (n, NG)."""
    mu_q, mu_t, ls0, lst, walk, excess = eta.T
    lsq, v = _charge_width(mu_q, ls0, excess)
    terms, g = _core(mu_q, mu_t, lsq, lst, walk, d, ref)
    e = np.exp(-mu_q) / (2 * v)
    grad = np.column_stack([g[:, 0] - g[:, 2] * excess * e, g[:, 1], g[:, 2] * np.exp(2 * ls0) / v, g[:, 3],
                            g[:, 4], g[:, 2] * e])
    return terms, grad


# Global k acts on row predictor k with coefficient COEF[k] (alpha: -x, beta: +x, the others: 1).
def _coef(x):
    one = np.ones_like(x)
    return np.column_stack([-x, x] + [one] * (NG - 2))


def _assemble_grad(geta, d, nchan):
    c = _coef(d["x"])
    return np.r_[(c * geta).sum(axis=0), np.bincount(d["index"], geta[:, 0], minlength=nchan),
                 np.bincount(d["index"], geta[:, 1], minlength=nchan)]


def _nll(theta, d, nchan, ref):
    terms, geta = _rows(_eta(theta, d, nchan), d, ref)
    return terms, _assemble_grad(geta, d, nchan)


def _row_hessian(eta, d, ref, free_eta):
    """Per-crossing Hessian in the row predictors by central differences of the analytic gradient (n, NG, NG)."""
    n = len(eta)
    st = np.exp(eta[0, 3])
    h = 1e-5 * np.array([1., st, 1., 1., st, 1.])
    hess = np.zeros((n, NG, NG))
    for k in np.flatnonzero(free_eta):
        step = np.zeros(NG)
        step[k] = h[k]
        gp = _rows(eta + step, d, ref)[1]
        gm = _rows(eta - step, d, ref)[1]
        hess[:, :, k] = (gp - gm) / (2 * h[k])
    hess = 0.5 * (hess + hess.transpose(0, 2, 1))
    return hess * np.outer(free_eta, free_eta)


def _hessian(theta, d, nchan, ref, free_global):
    """Full Hessian of the summed NLL in (globals, g, o), from the per-row predictor Hessians."""
    free_eta = np.r_[True, True, np.asarray(free_global[2:], bool)]
    hr = _row_hessian(_eta(theta, d, nchan), d, ref, free_eta)
    c, idx = _coef(d["x"]), d["index"]
    P = NG + 2 * nchan
    H = np.zeros((P, P))
    H[:NG, :NG] = np.einsum("ni,nj,nij->ij", c, c, hr)
    for i in range(NG):
        H[i, NG:NG + nchan] = np.bincount(idx, c[:, i] * hr[:, i, 0], minlength=nchan)
        H[i, NG + nchan:] = np.bincount(idx, c[:, i] * hr[:, i, 1], minlength=nchan)
    H[NG:, :NG] = H[:NG, NG:].T
    ch = np.arange(nchan)
    H[NG + ch, NG + ch] = np.bincount(idx, hr[:, 0, 0], minlength=nchan)
    H[NG + nchan + ch, NG + nchan + ch] = np.bincount(idx, hr[:, 1, 1], minlength=nchan)
    H[NG + ch, NG + nchan + ch] = H[NG + nchan + ch, NG + ch] = np.bincount(idx, hr[:, 0, 1], minlength=nchan)
    return H


def _gauge(nchan):
    """Linear map from (globals, g, o) to the reported (globals, log_scale, log_gain, time_zero, time_offset)."""
    P = NG + 2 * nchan
    T = np.zeros((P + 2, P))
    T[:NG, :NG] = np.eye(NG)
    mean = np.full(nchan, 1 / nchan)
    T[NG, NG:NG + nchan] = mean
    T[NG + 1:NG + 1 + nchan, NG:NG + nchan] = np.eye(nchan) - mean
    T[NG + 1 + nchan, NG + nchan:] = mean
    T[NG + 2 + nchan:, NG + nchan:] = np.eye(nchan) - mean
    return T


def _covariance(H, free):
    cov = np.zeros_like(H)
    Hf = H[np.ix_(free, free)]
    try:
        chol = np.linalg.cholesky(Hf)
    except np.linalg.LinAlgError:
        return None
    inv = np.linalg.inv(chol)
    cov[np.ix_(free, free)] = inv.T @ inv
    return cov


REPORTED = ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "sigma_log_pe", "sigma_time_ns",
            "walk_ns_per_log_pe", "excess_noise")


def _report(theta, cov, channels, ref, free_global, n_eligible, n_seen, include_missing):
    nchan = len(channels)
    alpha, beta, ls0, lst, walk, excess = theta[:NG]
    g, o = theta[NG:NG + nchan], theta[NG + nchan:]
    out = dict(schema=SCHEMA, candidate_only=True, channel=[int(c) for c in channels],
               sigma_log_pe=float(np.exp(ls0)), excess_noise=float(excess), sigma_time_ns=float(np.exp(lst)),
               walk_ns_per_log_pe=float(walk), log_pe_ref=float(ref),
               delta_attenuation_per_mm=float(alpha / 1000), delta_slowness_ns_per_mm=float(beta / 1000),
               log_scale=float(g.mean()), log_gain=(g - g.mean()).tolist(),
               time_zero_ns=float(o.mean()), time_offset_ns=(o - o.mean()).tolist(),
               fitted={k: bool(f) for k, f in zip(REPORTED[2:], free_global[2:])},
               eligible=int(n_eligible), seen=int(n_seen), include_missing=bool(include_missing))
    if cov is None:
        out["errors"] = None
        out["hessian_positive_definite"] = False
        return out
    T = _gauge(nchan)
    c = T @ cov @ T.T
    se = np.sqrt(np.clip(np.diag(c), 0, None))
    # Physical units of the globals (delta method for the log widths).
    scale = np.array([1e-3, 1e-3, np.exp(ls0), np.exp(lst), 1., 1.])
    se_g = se[:NG] * scale
    out["hessian_positive_definite"] = True
    out["errors"] = dict({k: float(v) for k, v in zip(REPORTED, se_g)},
                         log_scale=float(se[NG]), log_gain=se[NG + 1:NG + 1 + nchan].tolist(),
                         time_zero_ns=float(se[NG + 1 + nchan]), time_offset_ns=se[NG + 2 + nchan:].tolist())
    keep = list(range(NG)) + [NG, NG + 1 + nchan]
    s = np.r_[scale, 1., 1.]
    out["covariance_global"] = dict(names=list(REPORTED) + ["log_scale", "time_zero_ns"],
                                    matrix=(c[np.ix_(keep, keep)] * np.outer(s, s)).tolist())
    return out


def _positive_only_theta(d, nchan, weights=(None, None)):
    s = d["seen"]
    n = np.bincount(d["index"][s], minlength=nchan)
    if (n == 0).any():
        raise ValueError("each fitted channel needs a recorded pulse; exclude unsupported channels using fit data only")
    wc = np.ones(s.sum()) if weights[0] is None else weights[0]
    wt = np.ones(s.sum()) if weights[1] is None else weights[1]
    idx, x = d["index"][s], d["x"][s]
    qs, g, qrms, qerr = _profile(idx, nchan, x, d["lpe"][s] - d["q0"][s], wc)
    ts, o, trms, terr = _profile(idx, nchan, x, d["tm"][s] - d["t0"][s], wt)
    return np.r_[-qs, ts, np.log(qrms), np.log(trms), 0., 0., g, o], (qerr, terr)


def _weights(rows, key, eligible_seen):
    if key is None:
        return None
    w = np.asarray(rows[key], dtype=float) if isinstance(key, str) else np.asarray(key, dtype=float)
    w = w[eligible_seen]
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("weights must be finite and positive on the recorded pulses")
    return w


MAX_SCALED_GRADIENT = 1e-2   # convergence: largest |gradient| x (start curvature)^-1/2, in units of one sigma


def _optimize(theta, free, d, nchan, ref):
    """L-BFGS-B on the summed NLL in parameters scaled by their start curvature, with box limits on the widths,
    the excess noise and the walk (the walk box keeps corr(log PE, time) inside the bivariate rule's range)."""
    H0 = _hessian(theta, d, nchan, ref, free[:NG])
    diag = np.diag(H0)[free]
    scale = 1 / np.sqrt(np.where(np.isfinite(diag) & (diag > 0), diag, 1.))
    base = theta.copy()
    def full(u):
        t = base.copy()
        t[free] = base[free] + scale * u
        return t

    def loss(u):
        terms, grad = _nll(full(u), d, nchan, ref)
        return terms.sum(), grad[free] * scale

    sq_max = float(np.exp(_charge_width(_eta(theta, d, nchan)[:, 0], theta[2], theta[5])[0]).max())
    walk_max = 0.9 * MAX_CORRELATION / np.sqrt(1 - MAX_CORRELATION ** 2) * np.exp(theta[3]) / sq_max
    limits = {2: (-7., 3.), 3: (-7., 3.), 4: (-walk_max, walk_max), 5: (0., 20.)}
    bounds = [tuple((b - base[k]) / scale[i] for b in limits[k]) if k in limits else (None, None)
              for i, k in enumerate(np.flatnonzero(free))]
    result = minimize(loss, np.zeros(free.sum()), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-7, "maxcor": 30})
    theta = full(result.x)
    at_bound = [k for k in (4, 5) if free[k] and min(abs(theta[k] - lim) for lim in limits[k]) <= 1e-9 * max(1, abs(theta[k]))]
    return theta, dict(scale=np.r_[scale], free=free.copy(), message=result.message, nit=result.nit), at_bound


def fit(rows, *, sigma_log_pe=None, sigma_time_ns=None, excess_noise="fit", fit_widths=True, walk=True,
        include_missing=True, charge_weight=None, time_weight=None):
    """Return candidate corrections relative to the expected response; does not write conditions.

    ``include_missing=False`` is the positive-only closed form (``charge_weight``/``time_weight``: column names or
    arrays of relative inverse variances, applied to the recorded pulses). Otherwise the censored likelihood, started
    from the positive-only solution. Widths: ``sigma_q^2 = sigma_log_pe^2 + excess_noise / expected PE``;
    ``sigma_log_pe``/``sigma_time_ns`` are start values, fixed (and required) with ``fit_widths=False``.
    ``excess_noise`` is "fit" (default; 0 with fixed widths) or a fixed value (0: constant charge width, 1: Poisson
    photostatistics). ``walk`` fits the time-walk slope (False fixes it at zero).
    """
    channels, d = _design(rows, None)
    nchan = len(channels)
    elig = np.asarray(rows["eligible"])
    eligible_seen = elig & np.asarray(rows["seen"])
    weights = (_weights(rows, charge_weight, eligible_seen), _weights(rows, time_weight, eligible_seen))
    theta, (qerr, terr) = _positive_only_theta(d, nchan, weights)
    ref = float(np.median(d["lpe"][d["seen"]]))
    n_eligible, n_seen = len(d["x"]), int(d["seen"].sum())
    if not include_missing:
        out = _report(theta, None, channels, ref, np.zeros(NG, bool), n_eligible, n_seen, False)
        out["errors"] = dict(delta_attenuation_per_mm=float(qerr / 1000), delta_slowness_ns_per_mm=float(terr / 1000))
        out["hessian_positive_definite"] = None
        out["fitted"] = dict(sigma_log_pe="rms", sigma_time_ns="rms", walk_ns_per_log_pe=False, excess_noise=False)
        out["weighted"] = dict(charge=charge_weight is not None, time=time_weight is not None)
        return out
    if not fit_widths and (sigma_log_pe is None or sigma_time_ns is None):
        raise ValueError("fixed widths need sigma_log_pe and sigma_time_ns")
    fit_excess = fit_widths and isinstance(excess_noise, str)
    if isinstance(excess_noise, str):
        if excess_noise != "fit":
            raise ValueError('excess_noise must be "fit" or a number')
        theta[5] = 0.5 if fit_excess else 0.
    else:
        if not np.isfinite(excess_noise) or excess_noise < 0:
            raise ValueError("excess_noise must be finite and nonnegative")
        theta[5] = excess_noise
    if fit_excess:
        # Start the constant term from the pulse RMS less the start photostatistics term (median seen PE).
        theta[2] = 0.5 * np.log(max(np.exp(2 * theta[2]) - theta[5] * np.exp(-ref), 0.09 * np.exp(2 * theta[2])))
    for k, v in ((2, sigma_log_pe), (3, sigma_time_ns)):
        if v is not None:
            if not np.isfinite(v) or v <= 0:
                raise ValueError("response widths must be finite and positive")
            theta[k] = np.log(v)
    free = np.ones(len(theta), bool)
    free[2:4] = fit_widths
    free[4] = False
    free[5] = fit_excess
    if walk:
        # Two stages: without walk first (the start is far from the censored optimum), then with it.
        free[4] = False
        theta, _, _ = _optimize(theta, free, d, nchan, ref)
        free[4] = True
    theta, result, at_bound = _optimize(theta, free, d, nchan, ref)
    for k in at_bound:
        free[k] = False      # held at a bound: no curvature-based error for it
    terms, grad = _nll(theta, d, nchan, ref)
    gnorm = float(np.max(np.abs(grad[free] * result["scale"][free[result["free"]]]))) if free.any() else 0.
    if not np.isfinite(theta).all() or not np.isfinite(terms).all() or gnorm > MAX_SCALED_GRADIENT:
        raise RuntimeError(f"EX fit did not converge: {result['message']} (max scaled gradient {gnorm:.2g})")
    gfree = free[:NG].copy()
    H = _hessian(theta, d, nchan, ref, gfree)
    cov = _covariance(H, free)
    out = _report(theta, cov, channels, ref, gfree, n_eligible, n_seen, True)
    for k in at_bound:
        out["fitted"][REPORTED[k]] = "at_bound"
    out["optimizer"] = dict(iterations=int(result["nit"]), max_scaled_gradient=gnorm)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Prediction and diagnostics

def _parameters(cal):
    g = cal["log_scale"] + np.asarray(cal["log_gain"], dtype=float)
    o = cal["time_zero_ns"] + np.asarray(cal["time_offset_ns"], dtype=float)
    return np.r_[1000 * cal["delta_attenuation_per_mm"], 1000 * cal["delta_slowness_ns_per_mm"],
                 np.log(cal["sigma_log_pe"]), np.log(cal["sigma_time_ns"]), cal.get("walk_ns_per_log_pe", 0.),
                 cal.get("excess_noise", 0.), g, o]


def predict(cal, rows, dtype=np.float64):
    """Selection probabilities, mean log PE, its width, and the mean time at the mean log PE for the eligible rows."""
    dtype = np.dtype(dtype).type
    channels, d = _design(rows, np.asarray(cal["channel"]))
    eta = _eta(_parameters(cal), d, len(channels)).astype(dtype)
    mu_q, mu_t, ls0, lst, walk, excess = eta.T
    lsq, _ = _charge_width(mu_q, ls0, excess)
    ref = dtype(cal.get("log_pe_ref", 0.))
    c = mu_t - walk * (mu_q - ref)
    prob = p_selected(_prediction(mu_q, mu_t, lsq, lst, walk, ref), Selection(d["lthr"], d["lo"], d["hi"]))
    return dict(probability=prob.astype(dtype), log_pe=mu_q.astype(dtype), sigma_log_pe=np.exp(lsq).astype(dtype),
                time_ns=c.astype(dtype), walk_ns_per_log_pe=dtype(cal.get("walk_ns_per_log_pe", 0.)))


def diagnostics(cal, rows):
    channels, d = _design(rows, np.asarray(cal["channel"]))
    terms, _ = _nll(_parameters(cal), d, len(channels), cal.get("log_pe_ref", 0.))
    prob = predict(cal, rows)["probability"]
    return dict(nll_per_crossing=float(terms.mean()), detection_brier=float(np.mean((prob - d["seen"]) ** 2)))


def covariance(cal, rows):
    """Full gauge-fixed covariance (globals, log_scale, log_gain, time_zero, time_offset) at ``cal`` on ``rows``."""
    channels, d = _design(rows, np.asarray(cal["channel"]))
    nchan = len(channels)
    fitted = cal.get("fitted", {})
    gfree = np.array([True, True] + [fitted.get(k) is True for k in REPORTED[2:]])
    free = np.r_[gfree, np.ones(2 * nchan, bool)]
    H = _hessian(_parameters(cal), d, nchan, cal.get("log_pe_ref", 0.), gfree)
    cov = _covariance(H, free)
    if cov is None:
        raise RuntimeError("Hessian is not positive definite at the given calibration")
    T = _gauge(nchan)
    return T @ cov @ T.T


COVARIANCE_NAMES = ["inv_attenuation_per_m", "slowness_ns_per_m", "log_sigma_log_pe", "log_sigma_time_ns",
                    "walk_ns_per_log_pe", "excess_noise", "log_scale"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("observations")
    parser.add_argument("out", help="candidate calibration JSON, not an Offline conditions file")
    parser.add_argument("--sigma-log-pe", type=float, help="start value (fixed with --fix-widths)")
    parser.add_argument("--sigma-time-ns", type=float, help="start value (fixed with --fix-widths)")
    parser.add_argument("--fix-widths", action="store_true", help="hold the widths at the given values")
    parser.add_argument("--excess-noise", default="fit",
                        help='"fit" (default) or a fixed photostatistics factor (0: constant charge width)')
    parser.add_argument("--no-walk", action="store_true", help="fix the time-walk slope at zero")
    parser.add_argument("--positive-only", action="store_true", help="closed-form fit of recorded pulses only")
    parser.add_argument("--charge-weight", help="positive-only: column of relative inverse variances in log PE")
    parser.add_argument("--time-weight", help="positive-only: column of relative inverse variances in time")
    parser.add_argument("--covariance-out", help="write the full gauge-fixed covariance to this NPZ")
    parser.add_argument("--holdout", help="optional observation NPZ with separate runs for predictive diagnostics")
    args = parser.parse_args(argv)
    with np.load(args.observations, allow_pickle=False) as data:
        rows = {k: data[k] for k in data.files}
    excess = args.excess_noise if args.excess_noise == "fit" else float(args.excess_noise)
    cal = fit(rows, sigma_log_pe=args.sigma_log_pe, sigma_time_ns=args.sigma_time_ns, excess_noise=excess,
              fit_widths=not args.fix_widths, walk=not args.no_walk, include_missing=not args.positive_only,
              charge_weight=args.charge_weight, time_weight=args.time_weight)
    cal["fit_runs"] = np.unique(rows["run"][rows["eligible"]]).tolist()
    with open(args.observations, "rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        cal["observations_sha256"] = digest.hexdigest()
    if args.holdout:
        with np.load(args.holdout, allow_pickle=False) as data:
            hold = {k: data[k] for k in data.files}
        runs = np.unique(hold["run"][hold["eligible"]]).tolist()
        if set(runs) & set(cal["fit_runs"]):
            raise ValueError("held-out observations must come from separate runs")
        cal["holdout"] = dict(runs=runs, **diagnostics(cal, hold))
    if args.covariance_out:
        if args.positive_only:
            raise ValueError("--covariance-out needs the censored fit")
        np.savez_compressed(args.covariance_out, covariance=covariance(cal, rows),
                            names=np.array(COVARIANCE_NAMES + [f"log_gain_{c}" for c in cal["channel"]]
                                           + ["time_zero_ns"]
                                           + [f"time_offset_ns_{c}" for c in cal["channel"]]))
    with open(args.out, "w") as stream:
        json.dump(cal, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
