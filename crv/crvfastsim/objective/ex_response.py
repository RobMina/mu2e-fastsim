"""Fit an opt-in EX response surrogate on fixed track/channel crossings.

Preselection log PE and time residual are independent Gaussians conditional on
the crossing. A pulse is seen only when it passes both the PE and time cuts.
The model has exact CPU gradients, including the probability of missing a pulse.
"""

import argparse
import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def _design(rows, channel_ids, sigma_log_pe, sigma_time_ns, dtype):
    if not np.isfinite([sigma_log_pe, sigma_time_ns]).all() or min(sigma_log_pe, sigma_time_ns) <= 0:
        raise ValueError("fixed response widths must be finite and positive")
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
    zpe = (np.log(np.where(seen, pe, 1)) - q0) / sigma_log_pe
    ztime = (np.where(seen, tm, t0) - t0) / sigma_time_ns
    inputs = (distance / 1000, (np.log(threshold) - q0) / sigma_log_pe, window / sigma_time_ns, seen, zpe, ztime)
    return channel_ids, (index, *(v if v.dtype.kind == "b" else v.astype(dtype, copy=False) for v in inputs))


def _interval_logp(lo, hi):
    # Subtract normal tails on the side where CDFs are not both ~1.
    large = log_ndtr(np.where(lo > 0, -lo, hi))
    small = log_ndtr(np.where(lo > 0, -hi, lo))
    with np.errstate(divide="ignore"):
        return large + np.log(-np.expm1(small - large))


def _model(par, design):
    index, x, threshold, window, *_ = design
    q, t = par.reshape(2, -1)
    a = threshold - (q[0] * x + q[1:][index])
    mt = t[0] * x + t[1:][index]
    lo, hi = -window - mt, window - mt
    return a, lo, hi, mt


def _nll(par, design):
    index, x, threshold, _, seen, zpe, ztime = design
    a, lo, hi, mt = _model(par, design)
    qt = a - threshold + zpe
    tt = ztime - mt
    log_q_pass = log_ndtr(-a)
    log_t_fail = np.logaddexp(log_ndtr(lo), log_ndtr(-hi))
    log_miss = np.logaddexp(log_ndtr(a), log_q_pass + log_t_fail)
    terms = np.where(seen, 0.5 * (qt**2 + tt**2) + 2 * LOG_SQRT_2PI, -log_miss)
    qgrad = np.where(seen, -qt, np.exp(-0.5 * a**2 - LOG_SQRT_2PI + _interval_logp(lo, hi) - log_miss))
    tgrad = np.where(seen, -tt, np.exp(log_q_pass - 0.5 * lo**2 - LOG_SQRT_2PI - log_miss)
                     - np.exp(log_q_pass - 0.5 * hi**2 - LOG_SQRT_2PI - log_miss))
    nchan = par.size // 2 - 1
    grad = np.r_[np.dot(x, qgrad), np.bincount(index, qgrad, minlength=nchan),
                 np.dot(x, tgrad), np.bincount(index, tgrad, minlength=nchan)]
    return terms, grad


def _positive_only(design, nchan):
    index, x, _, _, seen, zpe, ztime = design
    index, x = index[seen], x[seen]
    n = np.bincount(index, minlength=nchan)
    if (n == 0).any():
        raise ValueError("each fitted channel needs a recorded pulse; exclude unsupported channels using fit data only")
    xm = np.bincount(index, x, minlength=nchan) / n
    dx = x - xm[index]
    var = np.dot(dx, dx)
    if var <= np.finfo(float).eps * np.dot(x, x):
        raise ValueError("shared slopes need within-channel fibre-distance variation")
    out = []
    for z in (zpe[seen], ztime[seen]):
        zm = np.bincount(index, z, minlength=nchan) / n
        slope = np.dot(dx, z - zm[index]) / var
        out.extend(np.r_[slope, zm - slope * xm])
    return np.asarray(out)


def fit(rows, *, sigma_log_pe, sigma_time_ns, include_missing=True):
    """Return candidate corrections relative to the nominal response; does not write conditions."""
    channels, design = _design(rows, None, sigma_log_pe, sigma_time_ns, np.float64)
    start = _positive_only(design, len(channels))
    if include_missing:
        def loss(par):
            terms, gradient = _nll(par, design)
            return terms.mean(), gradient / len(terms)

        result = minimize(loss, start, jac=True, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-8})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"EX fit did not converge: {result.message}")
        par = result.x
    else:
        par = start
    q, t = par.reshape(2, -1)
    dq, dt = q[1:] * sigma_log_pe, t[1:] * sigma_time_ns
    return dict(schema="mu2e-crv-ex-response-v1", candidate_only=True, channel=channels.tolist(),
                sigma_log_pe=float(sigma_log_pe), sigma_time_ns=float(sigma_time_ns),
                delta_attenuation_per_mm=float(-q[0] * sigma_log_pe / 1000),
                delta_slowness_ns_per_mm=float(t[0] * sigma_time_ns / 1000),
                log_scale=float(dq.mean()), log_gain=(dq - dq.mean()).tolist(),
                time_zero_ns=float(dt.mean()), time_offset_ns=(dt - dt.mean()).tolist(),
                eligible=int(len(design[0])), seen=int(design[4].sum()), include_missing=bool(include_missing))


def _parameters(cal, dtype):
    sq, st = cal["sigma_log_pe"], cal["sigma_time_ns"]
    q = np.r_[-1000 * cal["delta_attenuation_per_mm"], cal["log_scale"] + np.asarray(cal["log_gain"])] / sq
    t = np.r_[1000 * cal["delta_slowness_ns_per_mm"], cal["time_zero_ns"] + np.asarray(cal["time_offset_ns"])] / st
    return np.concatenate([q, t]).astype(dtype)


def _probability(a, lo, hi):
    return ndtr(-a) * np.exp(_interval_logp(lo, hi))


def predict(cal, rows, dtype=np.float64):
    """Return detection probabilities and preselection Gaussian means for eligible rows."""
    sq, st = cal["sigma_log_pe"], cal["sigma_time_ns"]
    _, design = _design(rows, np.asarray(cal["channel"]), sq, st, dtype)
    threshold, use = design[2], np.asarray(rows["eligible"])
    a, lo, hi, mt = _model(_parameters(cal, dtype), design)
    return dict(probability=_probability(a, lo, hi),
                log_pe=np.asarray(rows["nominal_log_pe"], dtype=dtype)[use] + (threshold - a) * sq,
                time_ns=np.asarray(rows["nominal_time_ns"], dtype=dtype)[use] + mt * st)


def diagnostics(cal, rows):
    sq, st = cal["sigma_log_pe"], cal["sigma_time_ns"]
    _, design = _design(rows, np.asarray(cal["channel"]), sq, st, np.float64)
    par = _parameters(cal, np.float64)
    terms, _ = _nll(par, design)
    a, lo, hi, _ = _model(par, design)
    prob = _probability(a, lo, hi)
    return dict(nll_per_crossing=float(terms.mean()),
                detection_brier=float(np.mean((prob - design[4])**2)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations")
    parser.add_argument("out", help="candidate calibration JSON, not an Offline conditions file")
    parser.add_argument("--sigma-log-pe", type=float, required=True)
    parser.add_argument("--sigma-time-ns", type=float, required=True)
    parser.add_argument("--positive-only", action="store_true", help="fit only pulses as an uncensored baseline")
    parser.add_argument("--holdout", help="optional observation NPZ with separate runs for predictive diagnostics")
    args = parser.parse_args(argv)
    with np.load(args.observations, allow_pickle=False) as rows:
        cal = fit(rows, sigma_log_pe=args.sigma_log_pe, sigma_time_ns=args.sigma_time_ns,
                  include_missing=not args.positive_only)
        cal["fit_runs"] = np.unique(rows["run"][rows["eligible"]]).tolist()
    with open(args.observations, "rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        cal["observations_sha256"] = digest.hexdigest()
    if args.holdout:
        with np.load(args.holdout, allow_pickle=False) as rows:
            runs = np.unique(rows["run"][rows["eligible"]]).tolist()
            if set(runs) & set(cal["fit_runs"]):
                raise ValueError("held-out observations must come from separate runs")
            cal["holdout"] = dict(runs=runs, **diagnostics(cal, rows))
    with open(args.out, "w") as stream:
        json.dump(cal, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
