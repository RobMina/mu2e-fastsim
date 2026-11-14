"""Stock calorimeter template fit (Offline CaloReco CaloTemplateWFProcessor + CaloTemplateWFUtil) in JAX.

Per digi: baseline from the first samples, primary peaks (local maxima of the baseline-subtracted waveform), the
sequential secondary-peak search on the residual, then a least-squares fit of ``b + sum_k A_k p(x - t_k)`` with the
source's objective ``sum (y - f)^2 / b`` and parameter limits ``[0, 1e6]``. The source minimiser (TMinuit MIGRAD)
is replaced by damped Newton steps on the same objective, so fitted values agree to tolerance, not bitwise. The
objective falls when the baseline grows, so weakly damped first steps can fall into a baseline-only solution;
a per-step trust region (baseline and amplitudes by half their value, times by 5 ns) and the initial damping
keep the iterates near the seed, as MIGRAD's line search does.
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
    iterations: int = 30
    damping: float = 100.0   # initial Levenberg-Marquardt damping: keeps early steps near the seed, as MIGRAD does


def _jnp():
    import jax.numpy as jnp
    return jnp


def make_evaluate(pulse: PulseCache, dtype=None):
    """``dtype`` None keeps the default (float64 under x64); e.g. ``jnp.float32`` evaluates in single precision."""
    jnp = _jnp()
    vec = jnp.asarray(pulse.pulse_vec) if dtype is None else jnp.asarray(pulse.pulse_vec, dtype)
    step, delta, n = pulse.digi_step, pulse.delta_t, pulse.pulse_vec.size

    def evaluate(dt):
        t = dt + delta
        ibin = N_STEPS + jnp.trunc(t * N_STEPS / step / N_STEPS).astype(jnp.int64)
        ok = (ibin >= 0) & (ibin < n - 1)
        ib = jnp.clip(ibin, 0, n - 2)
        t0bin = (ib - N_STEPS) * step if dtype is None else (ib - N_STEPS).astype(dtype) * jnp.asarray(step, dtype)
        val = (vec[ib + 1] - vec[ib]) / step * (t - t0bin) + vec[ib]
        return jnp.where(ok, val, 0.0)

    return evaluate


def make_evaluate_slope(pulse: PulseCache, dtype=None):
    """``evaluate`` of :func:`make_evaluate` plus its derivative in ``dt`` (the interpolation slope, zero outside the
    cache), as autodiff of ``evaluate`` gives it: the piecewise-linear cache has no second derivative."""
    jnp = _jnp()
    vec = jnp.asarray(pulse.pulse_vec) if dtype is None else jnp.asarray(pulse.pulse_vec, dtype)
    step, delta, n = pulse.digi_step, pulse.delta_t, pulse.pulse_vec.size

    def evaluate_slope(dt):
        t = dt + delta
        ibin = N_STEPS + jnp.trunc(t * N_STEPS / step / N_STEPS).astype(jnp.int64)
        ok = (ibin >= 0) & (ibin < n - 1)
        ib = jnp.clip(ibin, 0, n - 2)
        t0bin = (ib - N_STEPS) * step if dtype is None else (ib - N_STEPS).astype(dtype) * jnp.asarray(step, dtype)
        slope = (vec[ib + 1] - vec[ib]) / step
        val = slope * (t - t0bin) + vec[ib]
        return jnp.where(ok, val, 0.0), jnp.where(ok, slope, 0.0)

    return evaluate_slope


def fit_digis(t0, samples, length, pulse: PulseCache, cfg: FitConfig = FitConfig()):
    """Fit padded digis. Returns per-digi, per-slot arrays ``amp, amp_err, time (t0 convention), time_err, valid``
    plus ``chi2``, ``ndf`` and ``refit_skipped``. ``time`` is ``fromPeakToT0(peak time)``, before the PBT offset."""
    jnp = _jnp()
    n_digis = len(t0)
    rows = max(1024, 1 << int(np.ceil(np.log2(max(n_digis, 1)))))   # fixed shape buckets: no recompilation
    width = max(32, 32 * int(np.ceil(np.asarray(samples).shape[1] / 32)))
    pad_y = np.zeros((rows, width), np.float64)
    pad_y[:n_digis, :np.asarray(samples).shape[1]] = samples
    pad_l = np.zeros(rows, np.int64)
    pad_l[:n_digis] = length
    pad_t = np.zeros(rows, np.float64)
    pad_t[:n_digis] = t0
    t0 = jnp.asarray(pad_t)
    y = jnp.asarray(pad_y)
    length = jnp.asarray(pad_l)
    L = y.shape[1]
    idx = jnp.arange(L)
    x = t0[:, None] + (idx + 0.5) * cfg.digi_sampling
    mask = idx[None, :] < length[:, None]

    out = _compiled(pulse, cfg, L)(x, y, mask, length)
    out = {k: np.asarray(v)[:n_digis] for k, v in out.items()}
    out["time"] = out["peak_time"] - pulse.delta_t - 0.5 * pulse.digi_step * N_STEPS
    return out


_CACHE = {}


def _compiled(pulse, cfg, L, dtype=None, hessian=None):
    """The jitted, vmapped per-digi fit for one pulse cache and configuration, built once and reused.

    ``dtype`` None is the default double-precision fit; ``jnp.float32`` fits in single precision (opt-in).
    ``hessian`` None takes the Newton step's gradient and Hessian by autodiff (default); ``"analytic"`` (opt-in) uses
    their closed form, the same quantities up to roundoff at a fraction of the cost."""
    import jax
    if hessian not in (None, "analytic"):
        raise ValueError("hessian must be None or 'analytic'")
    key = (id(pulse), cfg, L) if dtype is None else (id(pulse), cfg, L, dtype)
    if hessian is not None:
        key = key + (hessian,)
    if key not in _CACHE:
        evaluate = make_evaluate(pulse, dtype)
        slope = make_evaluate_slope(pulse, dtype) if hessian == "analytic" else None
        K, W = cfg.max_peaks, cfg.window_peak

        def one(x, y, mask, n):
            return _fit_one(x, y, mask, n, evaluate, cfg, K, W, L, dtype, slope)

        _CACHE[key] = (jax.jit(jax.vmap(one)), pulse)
    return _CACHE[key][0]


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


def _grad_hess(evaluate_slope, x, yfull, mask, p, act):
    """Closed-form gradient and Hessian of the objective sum (y - f)^2 / b of :func:`_fit_one`, f = b + sum_k A_k
    p(x - t_k): the model's second derivatives are d2f/dA_k dt_k = -p'(x - t_k) (p'' = 0 for the linear cache)."""
    jnp = _jnp()
    b, amp, tim = p[0], p[1::2], p[2::2]
    e, s = evaluate_slope(x[None, :] - tim[:, None])
    a = act[:, None]
    f = b + jnp.sum(jnp.where(a, amp[:, None] * e, 0.0), axis=0)
    r = jnp.where(mask, yfull - f, 0.0)
    S = jnp.sum(r * r)
    m = mask.astype(r.dtype)
    ja = jnp.where(a, e, 0.0) * m
    jt = jnp.where(a, -amp[:, None] * s, 0.0) * m
    J = jnp.concatenate([m[None, :], jnp.stack([ja, jt], axis=1).reshape(-1, x.shape[0])], axis=0)   # (1+2K, L)
    gS = -2.0 * (J @ r)
    HS = 2.0 * (J @ J.T)
    cross = 2.0 * jnp.sum(jnp.where(a, s, 0.0) * r[None, :], axis=1)          # 2 sum_i r_i a_k p'_ki
    ia = 1 + 2 * jnp.arange(amp.shape[0])
    HS = HS.at[ia, ia + 1].add(cross).at[ia + 1, ia].add(cross)
    on = jnp.abs(b) > 1e-5
    bb = jnp.where(b == 0, 1.0, b)
    eb = jnp.zeros_like(p).at[0].set(1.0)
    g = gS / bb - eb * (S / (bb * bb))
    H = (HS / bb - (jnp.outer(eb, gS) + jnp.outer(gS, eb)) / (bb * bb) + 2.0 * jnp.outer(eb, eb) * S / (bb * bb * bb))
    return jnp.where(on, g, 0.0), jnp.where(on, H, 0.0)


def _fit_one(x, y, mask, n, evaluate, cfg, K, W, L, dtype=None, evaluate_slope=None):
    import jax
    from jax import lax
    jnp = _jnp()
    amp_min = cfg.min_peak_amplitude
    noise = jnp.float32(jnp.sum(jnp.where(jnp.arange(L) < cfg.num_noise_bins, y, 0.0)) / cfg.num_noise_bins)
    noise = noise.astype(jnp.float64 if dtype is None else dtype)
    yw = jnp.where(mask, y - noise, -1e30)

    pos = jnp.arange(L)

    def candidates(w):
        """Source peak test at every sample: local maximum in +-W, three samples above threshold, inside."""
        ok = (pos >= W) & (pos + W < n)
        pad = jnp.concatenate([jnp.full(W, -1e30, dtype=dtype), w, jnp.full(W, -1e30, dtype=dtype)])
        for d in range(1, W + 1):
            ok = ok & (w > pad[W - d:W - d + L]) & (w >= pad[W + d:W + d + L])
        return ok & (pad[W - 1:W - 1 + L] >= amp_min) & (w >= amp_min) & (pad[W + 1:W + 1 + L] >= amp_min)

    # primary peaks (setPrimaryPeakPar1): independent local maxima of the baseline-subtracted waveform
    prim = candidates(yw)
    (sel,) = jnp.nonzero(prim, size=K, fill_value=0)
    use = jnp.arange(K) < jnp.sum(prim)
    tx = jax.vmap(lambda i: _estimate_time(evaluate, x, yw, i, W, L))(sel)
    ty = jax.vmap(lambda i, t: _peak_norm(evaluate, x, yw, t, i - W, i + W, L))(sel, tx)
    par = jnp.zeros(1 + 2 * K, dtype=dtype).at[0].set(noise).at[1::2].set(jnp.where(use, ty, 0.0)).at[2::2].set(jnp.where(use, tx, 0.0))
    active = use
    k = jnp.sum(use).astype(jnp.int64)

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

    def secondary(state, _):
        par, active, k, res, last = state
        ok = candidates(res) & (pos > last)
        near = jnp.any(active[None, :] & (jnp.abs(par[2::2][None, :] - x[:, None]) < cfg.min_dt_peaks), axis=1)
        ok = ok & ~near & ~((yfull > 0) & (res / jnp.where(yfull > 0, yfull, 1.0) < cfg.psd_threshold))
        i = jnp.argmax(ok)
        add = jnp.any(ok) & (k < K) & has_peak
        tx = _estimate_time(evaluate, x, res, i, W, L)
        slot = jnp.clip(k, 0, K - 1)
        new_par = par.at[1 + 2 * slot].set(res[i]).at[2 + 2 * slot].set(tx)
        res = jnp.where(add, jnp.where(mask, res - res[i] * evaluate(x - tx), res), res)
        par = jnp.where(add, new_par, par)
        active = jnp.where(add, active.at[slot].set(True), active)
        return (par, active, k + add, res, jnp.where(add, i, last)), None

    # secondary peaks (setSecondaryPeakPar): each accepted peak is the first candidate after the previous one
    (par, active, k, _, _), _ = lax.scan(secondary, (par, active, k, res0, jnp.int64(-1)), None, length=K)

    def objective(p, act):
        f = _model(evaluate, x, p, act)
        r = jnp.where(mask, yfull - f, 0.0)
        return jnp.where(jnp.abs(p[0]) > 1e-5, jnp.sum(r * r) / jnp.where(p[0] == 0, 1.0, p[0]), 0.0)

    is_time = (jnp.arange(1 + 2 * K) % 2 == 0) & (jnp.arange(1 + 2 * K) > 0)

    def cap(p):
        """Trust region per step: baseline and amplitudes by half their value plus a floor, times by 5 ns."""
        return jnp.where(is_time, 5.0, 0.5 * jnp.abs(p) + jnp.where(jnp.arange(1 + 2 * K) == 0, 1.0, 10.0))

    def minimise(p, act):
        free = jnp.concatenate([jnp.ones(1, bool), jnp.repeat(act, 2)])
        if evaluate_slope is None:
            grad = jax.grad(objective)
            hess = jax.hessian(objective)

            def grad_hess(p, act):
                return grad(p, act), hess(p, act)
        else:
            def grad_hess(p, act):
                return _grad_hess(evaluate_slope, x, yfull, mask, p, act)

        def body(_, state):
            p, lam, fval = state
            g0, h = grad_hess(p, act)
            g = jnp.where(free, g0, 0.0)
            h = jnp.where(free[:, None] & free[None, :], h, 0.0) + jnp.diag(jnp.where(free, 0.0, 1.0))
            step = -jnp.linalg.solve(h + lam * jnp.diag(jnp.abs(jnp.diag(h)) + 1e-9), g)
            step = jnp.clip(step, -cap(p), cap(p))
            trial = jnp.clip(p + jnp.where(free, step, 0.0), 0.0, 1e6)
            ft = objective(trial, act)
            better = ft <= fval
            return (jnp.where(better, trial, p), jnp.where(better, lam * 0.3, lam * 10.0), jnp.where(better, ft, fval))

        lam0 = cfg.damping if dtype is None else jnp.asarray(cfg.damping, dtype)
        p, _, fval = lax.fori_loop(0, cfg.iterations, body, (p, lam0, objective(p, act)))
        h = grad_hess(p, act)[1]
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
