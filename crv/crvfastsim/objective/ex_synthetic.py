"""Synthetic EX crossings for closure tests of the censored response fit (public seeds, no detector data).

``model="lognormal"`` draws from the fitted model itself (an optimizer closure; log-PE width
``sqrt(sigma_log_pe^2 + excess_noise / expected PE)``). ``model="poisson"`` is a model mismatch: a Poisson
photoelectron count on a log-normal light yield of width ``smear``, the threshold on the count, and the time walk on
the count's log (closest model parameters: ``sigma_log_pe = smear``, ``excess_noise = 1``). ``model="poisson_spe"``
adds a Gaussian single-photoelectron gain spread (``spe_width`` per photoelectron), so the charge is continuous as a
reconstructed PE is.
"""

import numpy as np

THRESHOLD_PE, HALF_WINDOW_NS = 9., 4.


def truth(rng, nchan=8, *, attenuation=2.4e-4, slowness=1.5e-3, sigma_log_pe=0.36, sigma_time_ns=2.0, walk=0.,
          excess_noise=0., log_pe_ref=3.0):
    gain, dt = rng.normal(0, 0.09, nchan), rng.normal(0, 0.4, nchan)
    return dict(channel=list(range(nchan)), log_scale=-0.16, log_gain=(gain - gain.mean()).tolist(),
                time_zero_ns=0.7, time_offset_ns=(dt - dt.mean()).tolist(), delta_attenuation_per_mm=attenuation,
                delta_slowness_ns_per_mm=slowness, sigma_log_pe=sigma_log_pe, sigma_time_ns=sigma_time_ns,
                walk_ns_per_log_pe=walk, excess_noise=excess_noise, log_pe_ref=log_pe_ref)


def draw(seed, run, n, truth, *, model="lognormal", smear=0.25, spe_width=0.15):
    """``n`` events x one crossing per channel. Returns the rows and the standardized (charge, time) noise."""
    rng = np.random.default_rng(seed)
    nc = len(truth["log_gain"])
    chan = np.tile(np.arange(nc), n)
    distance = rng.uniform(100, 1700, len(chan))
    path = rng.uniform(18, 30, len(chan))
    q0 = np.log(0.65 * path) - distance / 2200
    t0 = 120 + distance / 160
    threshold = np.full(len(chan), THRESHOLD_PE)
    window = np.full(len(chan), HALF_WINDOW_NS)
    mu_q = (q0 + truth["log_scale"] + np.asarray(truth["log_gain"])[chan]
            - distance * truth["delta_attenuation_per_mm"])
    mu_t = (t0 + truth["time_zero_ns"] + np.asarray(truth["time_offset_ns"])[chan]
            + distance * truth["delta_slowness_ns_per_mm"])
    zq = rng.standard_normal(len(chan))
    zt = rng.standard_normal(len(chan))
    walk, ref = truth.get("walk_ns_per_log_pe", 0.), truth.get("log_pe_ref", 0.)
    if model == "lognormal":
        sigma_q = np.sqrt(truth["sigma_log_pe"] ** 2 + truth.get("excess_noise", 0.) * np.exp(-mu_q))
        lpe = mu_q + sigma_q * zq
        pe = np.exp(lpe)
    elif model in ("poisson", "poisson_spe"):
        pe = rng.poisson(np.exp(mu_q + smear * zq)).astype(float)
        if model == "poisson_spe":
            # Charge in PE units: each photoelectron's gain fluctuates by spe_width (continuous, as reco PE).
            pe = np.maximum(pe + spe_width * np.sqrt(pe) * rng.standard_normal(len(pe)), 0.)
        lpe = np.log(np.maximum(pe, 1e-3))
    else:
        raise ValueError(f"unknown model {model}")
    t = mu_t - walk * (lpe - ref) + truth["sigma_time_ns"] * zt
    seen = (pe >= threshold) & (np.abs(t - t0) <= window)
    rows = dict(run=np.full(len(chan), run), subrun=np.repeat(np.arange(n) % 8, nc),
                event=np.repeat(np.arange(n), nc), channel=chan,
                nominal_log_pe=q0, nominal_time_ns=t0, distance_mm=distance, eligible=np.ones(len(chan), bool),
                threshold_pe=threshold, time_half_width_ns=window, seen=seen, pe=np.where(seen, pe, np.nan),
                time_ns=np.where(seen, t, np.nan))
    return rows, (zq, zt)


def errors(cal, truth):
    """Fit minus truth for the shared terms, and the gauge-fixed channel RMSE."""
    out = {k: cal[k] - truth[k] for k in ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "log_scale",
                                          "time_zero_ns", "sigma_log_pe", "sigma_time_ns")}
    # The fitted time origin is the time at the fit's reference log PE: move the truth to it.
    walk = truth.get("walk_ns_per_log_pe", 0.)
    out["time_zero_ns"] += walk * (cal.get("log_pe_ref", 0.) - truth.get("log_pe_ref", 0.))
    for k in ("walk_ns_per_log_pe", "excess_noise"):
        out[k] = cal.get(k, 0.) - truth.get(k, 0.)
    for k in ("log_gain", "time_offset_ns"):
        out[k + "_rmse"] = float(np.sqrt(np.mean((np.asarray(cal[k]) - truth[k]) ** 2)))
    return out


def pulls(cal, truth):
    """(fit - truth) / Hessian error for the shared terms, when the fit reported errors."""
    err = cal.get("errors") or {}
    dev = errors(cal, truth)
    return {k: dev[k] / err[k] for k in ("delta_attenuation_per_mm", "delta_slowness_ns_per_mm", "log_scale",
                                         "time_zero_ns", "sigma_log_pe", "sigma_time_ns", "walk_ns_per_log_pe",
                                         "excess_noise")
            if isinstance(err.get(k), float) and err[k] > 0}
