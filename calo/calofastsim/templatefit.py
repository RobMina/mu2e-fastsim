"""Stock calorimeter template fit (Offline CaloReco CaloTemplateWFProcessor + CaloTemplateWFUtil) in JAX.

Per digi: baseline from the first samples, primary peaks (local maxima of the baseline-subtracted waveform), the
sequential secondary-peak search on the residual, then a least-squares fit of ``b + sum_k A_k p(x - t_k)`` with the
source's objective ``sum (y - f)^2 / b`` and parameter limits ``[0, 1e6]``. The source minimiser (TMinuit MIGRAD)
is replaced by damped Newton steps on the same objective, so fitted values agree to tolerance, not bitwise.
Parabolic errors are ``sqrt(2 diag(H^-1))``. Components that fail the source's selection after a multi-peak fit
are removed and the fit repeated; amplitudes below 1 are dropped. The source's second pass for digis with
``chi2/ndf > chiThreshold`` (re-seeding with rising-edge peaks) is not implemented: such digis keep the first fit
and are flagged ``refit_skipped``.

All functions are vmapped over digis padded to a common length; JAX x64 must be enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from .pulse import N_STEPS, PulseCache


@dataclass(frozen=True)
class FitConfig:
    window_peak: int = 3
    min_peak_amplitude: float = 24.0
    num_noise_bins: int = 5
    min_dt_peaks: float = 20.0
    psd_threshold: float = 0.1
    chi_threshold: float = 1.0
    digi_sampling: float = 5.0
    max_peaks: int = 4
    iterations: int = 60


def _jnp():
    import jax.numpy as jnp
    return jnp


def make_evaluate(pulse: PulseCache):
    jnp = _jnp()
    vec = jnp.asarray(pulse.pulse_vec)
    step, delta, n = pulse.digi_step, pulse.delta_t, pulse.pulse_vec.size

    def evaluate(dt):
        t = dt + delta
        ibin = N_STEPS + jnp.trunc(t * N_STEPS / step / N_STEPS).astype(jnp.int64)
        ok = (ibin >= 0) & (ibin < n - 1)
        ib = jnp.clip(ibin, 0, n - 2)
        t0bin = (ib - N_STEPS) * step
        val = (vec[ib + 1] - vec[ib]) / step * (t - t0bin) + vec[ib]
        return jnp.where(ok, val, 0.0)

    return evaluate


def fit_digis(t0, samples, length, pulse: PulseCache, cfg: FitConfig = FitConfig()):
    """Fit padded digis. Returns per-digi, per-slot arrays ``amp, amp_err, time (t0 convention), time_err, valid``
    plus ``chi2``, ``ndf`` and ``refit_skipped``. ``time`` is ``fromPeakToT0(peak time)``, before the PBT offset."""
    import jax
    jnp = _jnp()
    evaluate = make_evaluate(pulse)
    K, W = cfg.max_peaks, cfg.window_peak
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    y = jnp.asarray(samples, dtype=jnp.float64)
    length = jnp.asarray(length, dtype=jnp.int64)
    L = y.shape[1]
    idx = jnp.arange(L)
    x = t0[:, None] + (idx + 0.5) * cfg.digi_sampling
    mask = idx[None, :] < length[:, None]

    def one(x, y, mask, n):
        return _fit_one(x, y, mask, n, evaluate, cfg, K, W, L)

    out = jax.jit(jax.vmap(one))(x, y, mask, length)
    out["time"] = out["peak_time"] - pulse.delta_t - 0.5 * pulse.digi_step * N_STEPS
    return {k: np.asarray(v) for k, v in out.items()}


def _window_sums(evaluate, x, y, x0, i0, i1, L):
    jnp = _jnp()
    idx = jnp.arange(L)
    w = (idx >= i0) & (idx <= i1)
    ff = jnp.where(w, evaluate(x - x0), 0.0)
    s1 = jnp.sum(ff * ff)
    s2 = jnp.sum(jnp.where(w, y, 0.0) * ff)
    norm = jnp.where(jnp.abs(s1) < 1e-6, 1e6, s2 / jnp.where(s1 == 0, 1.0, s1))
    return norm, ff, w


def _peak_norm(evaluate, x, y, x0, i0, i1, L):
    return _window_sums(evaluate, x, y, x0, i0, i1, L)[0]


def _sum_square(evaluate, x, y, x0, i0, i1, L):
    jnp = _jnp()
    a, ff, w = _window_sums(evaluate, x, y, x0, i0, i1, L)
    cc = jnp.where(w, a * ff - y, 0.0)
    return jnp.sum(cc * cc)


def _estimate_time(evaluate, x, y, ic, W, L):
    jnp = _jnp()
    xc = x[ic]
    y1 = _sum_square(evaluate, x, y, xc, ic - W, ic + W, L)
    y2 = _sum_square(evaluate, x, y, xc - 1, ic - W, ic + W, L)
    y3 = _sum_square(evaluate, x, y, xc - 2, ic - W, ic + W, L)
    a = (y1 - 2 * y2 + y3) / 2
    b = y1 - y2 - 2 * a * xc + a
    new = jnp.where(jnp.abs(a) > 1e-6, -b / 2.0 / jnp.where(a == 0, 1.0, a), 0.0)
    return jnp.where(jnp.abs(new - xc) < 15.0, new, xc)


def _local_max(yw, i, W, L):
    """``std::max_element(begin+i-W, begin+i+W+1) == begin+i``: strictly above earlier, not below later."""
    jnp = _jnp()
    ok = jnp.bool_(True)
    for d in range(1, W + 1):
        lo = jnp.clip(i - d, 0, L - 1)
        hi = jnp.clip(i + d, 0, L - 1)
        ok = ok & (yw[i] > yw[lo]) & (yw[i] >= yw[hi])
    return ok


def _model(evaluate, x, par, active):
    jnp = _jnp()
    amp, tim = par[1::2], par[2::2]
    pulses = amp[:, None] * evaluate(x[None, :] - tim[:, None])
    return par[0] + jnp.sum(jnp.where(active[:, None], pulses, 0.0), axis=0)


def _fit_one(x, y, mask, n, evaluate, cfg, K, W, L):
    import jax
    from jax import lax
    jnp = _jnp()
    amp_min = cfg.min_peak_amplitude
    noise = jnp.float32(jnp.sum(jnp.where(jnp.arange(L) < cfg.num_noise_bins, y, 0.0)) / cfg.num_noise_bins)
    noise = noise.astype(jnp.float64)
    yw = jnp.where(mask, y - noise, -1e30)

    # primary peaks (setPrimaryPeakPar1): independent local maxima of the baseline-subtracted waveform
    def primary(carry, i):
        par, active, k = carry
        inside = (i >= W) & (i + W < n)
        ok = inside & _local_max(yw, i, W, L)
        ok = ok & (yw[jnp.clip(i - 1, 0, L - 1)] >= amp_min) & (yw[i] >= amp_min) & (yw[jnp.clip(i + 1, 0, L - 1)] >= amp_min)
        ok = ok & (k < K)
        tx = _estimate_time(evaluate, x, yw, i, W, L)
        ty = _peak_norm(evaluate, x, yw, tx, i - W, i + W, L)
        slot = jnp.clip(k, 0, K - 1)
        par = jnp.where(ok, par.at[1 + 2 * slot].set(ty).at[2 + 2 * slot].set(tx), par)
        active = jnp.where(ok, active.at[slot].set(True), active)
        return (par, active, k + ok), None

    par0 = jnp.zeros(1 + 2 * K).at[0].set(noise)
    (par, active, k), _ = lax.scan(primary, (par0, jnp.zeros(K, bool), jnp.int64(0)), jnp.arange(L))

    # fallback: first sample above threshold with the next one at least 0.8 of it
    alt_ok = mask & (yw >= amp_min) & (jnp.roll(yw, -1) >= 0.8 * yw) & (jnp.arange(L) + 1 < n)
    alt_i = jnp.argmax(alt_ok)
    use_alt = (k == 0) & jnp.any(alt_ok)
    par = jnp.where(use_alt, par.at[1].set(yw[alt_i]).at[2].set(x[alt_i]), par)
    active = jnp.where(use_alt, active.at[0].set(True), active)
    k = jnp.where(use_alt, 1, k)
    has_peak = k > 0

    # secondary peaks (setSecondaryPeakPar): sequential search on the residual
    yfull = jnp.where(mask, y, 0.0)
    res0 = jnp.where(mask, yfull - _model(evaluate, x, par, active), -1e30)

    def secondary(carry, i):
        par, active, k, res = carry
        inside = (i >= W) & (i + W < n)
        ok = inside & _local_max(res, i, W, L)
        ok = ok & (res[jnp.clip(i - 1, 0, L - 1)] >= amp_min) & (res[i] >= amp_min) & (res[jnp.clip(i + 1, 0, L - 1)] >= amp_min)
        near = jnp.any(active & (jnp.abs(par[2::2] - x[i]) < cfg.min_dt_peaks))
        ok = ok & ~near & ~((yfull[i] > 0) & (res[i] / jnp.where(yfull[i] > 0, yfull[i], 1.0) < cfg.psd_threshold))
        ok = ok & (k < K) & has_peak
        tx = _estimate_time(evaluate, x, res, i, W, L)
        slot = jnp.clip(k, 0, K - 1)
        new_par = par.at[1 + 2 * slot].set(res[i]).at[2 + 2 * slot].set(tx)
        sub = res[i] * evaluate(x - tx)
        res = jnp.where(ok, jnp.where(mask, res - sub, res), res)
        par = jnp.where(ok, new_par, par)
        active = jnp.where(ok, active.at[slot].set(True), active)
        return (par, active, k + ok, res), None

    (par, active, k, _), _ = lax.scan(secondary, (par, active, k, res0), jnp.arange(L))

    def objective(p, act):
        f = _model(evaluate, x, p, act)
        r = jnp.where(mask, yfull - f, 0.0)
        return jnp.where(jnp.abs(p[0]) > 1e-5, jnp.sum(r * r) / jnp.where(p[0] == 0, 1.0, p[0]), 0.0)

    def minimise(p, act):
        free = jnp.concatenate([jnp.ones(1, bool), jnp.repeat(act, 2)])
        grad = jax.grad(objective)
        hess = jax.hessian(objective)

        def body(_, state):
            p, lam, fval = state
            g = jnp.where(free, grad(p, act), 0.0)
            h = hess(p, act)
            h = jnp.where(free[:, None] & free[None, :], h, 0.0) + jnp.diag(jnp.where(free, 0.0, 1.0))
            step = -jnp.linalg.solve(h + lam * jnp.diag(jnp.abs(jnp.diag(h)) + 1e-9), g)
            trial = jnp.clip(p + jnp.where(free, step, 0.0), 0.0, 1e6)
            ft = objective(trial, act)
            better = ft <= fval
            return (jnp.where(better, trial, p), jnp.where(better, lam * 0.3, lam * 10.0), jnp.where(better, ft, fval))

        p, _, fval = lax.fori_loop(0, cfg.iterations, body, (p, 1e-3, objective(p, act)))
        h = hess(p, act)
        h = jnp.where(free[:, None] & free[None, :], h, 0.0) + jnp.diag(jnp.where(free, 0.0, 1.0))
        cov = 2.0 * jnp.linalg.inv(h)
        err = jnp.sqrt(jnp.abs(jnp.diag(cov)))
        return p, err, fval

    p, err, fval = minimise(par, active)

    # component selection after a multi-peak fit (CaloTemplateWFUtil::selectComponent), then refit
    amp, tim, aerr = p[1::2], p[2::2], err[1::2]
    xfront, xback = x[0], x[jnp.clip(n - 1, 0, L - 1)]
    dt = jnp.abs(tim[:, None] - tim[None, :])
    beaten = jnp.any(active[None, :] & (dt < cfg.min_dt_peaks) & (amp[None, :] > amp[:, None])
                     & ~jnp.eye(K, dtype=bool), axis=1)
    keep = active & (amp >= amp_min) & (tim >= xfront) & (tim <= xback) & (aerr <= 1e3) & ~beaten
    multi = jnp.sum(active) > 1
    refit = multi & jnp.any(active & ~keep)
    act2 = jnp.where(multi, keep, active)
    p2 = jnp.where(jnp.concatenate([jnp.ones(1, bool), jnp.repeat(act2, 2)]), p, 0.0)
    p2, err2, fval2 = minimise(p2, act2)
    p = jnp.where(refit, p2, p)
    err = jnp.where(refit, err2, err)
    fval = jnp.where(refit, fval2, fval)
    active = jnp.where(refit, act2, active)

    amp, tim = p[1::2], p[2::2]
    valid = active & (amp >= 1.0) & has_peak
    npar = 1 + 2 * jnp.sum(valid)
    ndf = n - npar
    reported = valid & (tim <= xback)
    chi = jnp.where(ndf > 0, fval / jnp.where(ndf > 0, ndf, 1), 0.0)
    return {"amp": amp, "amp_err": jnp.sqrt(err[1::2] ** 2 + amp), "peak_time": tim, "time_err": err[2::2],
            "valid": reported, "chi2": fval, "ndf": ndf, "n_seeded": k,
            "refit_skipped": has_peak & (ndf > 0) & (chi > cfg.chi_threshold), "baseline": p[0]}
