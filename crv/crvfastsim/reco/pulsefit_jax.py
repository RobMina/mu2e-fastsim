"""JAX twin of pulsefit: the same CRV pulse reconstruction as one jitted, fixed-shape function for GPU (and CPU).

Shapes are static so XLA compiles once per (W, L): waveforms are padded to L samples and batched to W rows; up to
P = RecoConfig.max_pulses rounds per waveform; the fit window is a fixed PW = samples_before + samples_after + 1 + G
slots with a mask (G covers plateaus; longer plateaus are flagged `window_overflow` and should go to the numpy path).
Every step mirrors pulsefit.py (which mirrors Offline MakeCrvRecoPulses), in float64:
  peak finder: a lax.scan over samples carrying (found, rising, peakStart, peakEnd) per waveform
  fit window: masked gathers instead of the stepwise monotonic walk
  fit: fixed-iteration Levenberg-Marquardt in Minuit's sine-mapped coordinates, then Gauss-Newton polish steps
  failed-fit rule, fallback values, rounded subtraction, failure propagation
Outputs are dense [W, P] arrays plus a count per waveform; `to_flat` turns them into pulsefit's flat dict on the host.

Only shape-setting settings are compile-time (STATIC_FIELDS: pulse rounds, fit-window sizes, LM iterations); every
physics setting (thresholds, beta and time limits, height ratios, failed-fit tolerance) is passed as data, so a new
value of a tuning knob reuses the compiled kernel.
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from .pulsefit import E, PERIOD, RecoConfig

jax.config.update("jax_enable_x64", True)

PLATEAU_SLOTS = 6   # extra fit-window slots for flat-topped (saturated) peaks
STATIC_FIELDS = ("max_pulses", "samples_before", "samples_after", "lm_iterations")
KNOBS = ("min_adc_difference", "default_beta", "min_beta", "max_beta", "max_time_difference", "min_pulse_height_ratio",
         "max_pulse_height_ratio", "le_time_factor", "pedestal_undershoot_threshold", "limit_tolerance", "knife_edge_adc")


def knobs_of(cfg):
    """The traced physics settings of a RecoConfig as float64 device scalars."""
    return {k: jnp.asarray(float(getattr(cfg, k)), jnp.float64) for k in KNOBS}


def _near_peak(y, valid, thr, edge):
    yv = jnp.where(valid, y, -1e9)
    pad = jnp.full((y.shape[0], 1), -1e9)
    left = jnp.concatenate([pad, yv[:, :-1]], 1)
    right = jnp.concatenate([yv[:, 1:], pad], 1)
    return (valid & (yv >= left) & (yv > right) & (jnp.abs(yv - thr) <= edge)).any(1)


def _gumbel(t, A, mu, b):
    z = (t - mu) / b
    return A * jnp.exp(-z - jnp.exp(-z))


def _peak(y, n, start, active, min_adc):
    """One FindNextPeak for all rows: y [W,L], n/start [W] int, active [W] bool."""
    W, L = y.shape

    def step(c, b):
        found, rising, ps, pe, done = c
        live = (~done) & (b > start) & (b < n) & (b >= 1)
        prev = y[:, jnp.maximum(b - 1, 0)]
        cur = y[:, b]
        up = live & (prev < cur)
        rising = rising | up
        ps = jnp.where(up, b, ps)
        pe = jnp.where(up, b, pe)
        flat = live & (prev == cur) & rising
        pe = jnp.where(flat, b, pe)
        down = live & (prev > cur) & rising
        ype = jnp.take_along_axis(y, pe[:, None], 1)[:, 0]
        hit = down & (ype > min_adc)
        found = found | hit
        done = done | hit
        rising = rising & ~(down & ~hit)
        return (found, rising, ps, pe, done), None

    z = jnp.zeros(W, bool)
    zi = jnp.zeros(W, jnp.int64)
    (found, _, ps, pe, _), _ = lax.scan(step, (z, z, zi, zi, ~active), jnp.arange(L))
    return found, ps, pe


def _window(y, n, ps, pe, found, sb, sa):
    """Fit window [fs, fe] of FindNextPeak, vectorised: monotonic run lengths before/after the peak, then trimming."""
    W, L = y.shape
    # before: i = ps-1 down to ps-sb while y[i] <= y[i+1]; also stops after i == 0
    k_b = jnp.arange(1, sb + 1)[None, :]
    ib = ps[:, None] - k_b
    okb = (ib >= 0) & (jnp.take_along_axis(y, jnp.clip(ib, 0, L - 1), 1) <=
                       jnp.take_along_axis(y, jnp.clip(ib + 1, 0, L - 1), 1))
    runb = jnp.cumprod(okb, axis=1).sum(1)
    fs = jnp.where(runb > 0, ps - runb, ps - 1)
    k_a = jnp.arange(1, sa + 1)[None, :]
    ia = pe[:, None] + k_a
    oka = (ia < n[:, None]) & (jnp.take_along_axis(y, jnp.clip(ia - 1, 0, L - 1), 1) >=
                               jnp.take_along_axis(y, jnp.clip(ia, 0, L - 1), 1))
    runa = jnp.cumprod(oka, axis=1).sum(1)
    fe = jnp.where(runa > 0, pe + runa, pe + 1)
    fs = jnp.where(found & (ps - fs > 1), fs + 1, fs)
    fe = jnp.where(found & (fe - pe > 1), fe - 1, fe)
    return fs, fe


def _to_ext(u, lo, hi):
    return lo + (hi - lo) * (jnp.sin(u) + 1.0) * 0.5


def _to_int(p, lo, hi):
    return jnp.arcsin(jnp.clip(2.0 * (p - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def _fit(t, y, w, p0, lo, hi, lm_iter, gn_iter):
    """Batched bounded Gumbel least squares (as pulsefit.fit_gumbel). t, y, w [K,PW]; p0, lo, hi [K,3]."""
    eye = jnp.eye(3)

    def rj(u):
        p = _to_ext(u, lo, hi)
        A, mu, b = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        z = (t - mu) / b
        ez = jnp.exp(-z)
        g = jnp.exp(-z - ez)
        f = A * g
        r = (y - f) * w
        dfdz = f * (ez - 1.0)
        dpdu = (hi - lo) * 0.5 * jnp.cos(u)
        J = jnp.stack([g * dpdu[:, 0:1], dfdz * (-1.0 / b) * dpdu[:, 1:2], dfdz * (-z / b) * dpdu[:, 2:3]], -1)
        return r, J * w[..., None]

    def solve(M, v):
        return jnp.linalg.solve(M, v[..., None])[..., 0]

    u0 = _to_int(p0, lo, hi)
    r0, J0 = rj(u0)
    c0 = (r0 * r0).sum(1)
    lam0 = jnp.full(t.shape[0], 1e-3)

    def lm(carry, _):
        u, r, J, chi2, lam = carry
        JTJ = jnp.einsum("kpi,kpj->kij", J, J)
        JTr = jnp.einsum("kpi,kp->ki", J, r)
        D = jnp.einsum("kii->ki", JTJ)
        du = solve(JTJ + lam[:, None, None] * (D[:, :, None] * eye + 1e-12 * eye), JTr)
        du = jnp.where(jnp.isfinite(du), du, 0.0)
        un = u + du
        rn, Jn = rj(un)
        cn = (rn * rn).sum(1)
        better = cn < chi2
        u = jnp.where(better[:, None], un, u)
        r = jnp.where(better[:, None], rn, r)
        J = jnp.where(better[:, None, None], Jn, J)
        chi2 = jnp.where(better, cn, chi2)
        lam = jnp.clip(jnp.where(better, lam * 0.3, lam * 10.0), 1e-9, 1e9)
        return (u, r, J, chi2, lam), None

    (u, r, J, chi2, _), _ = lax.scan(lm, (u0, r0, J0, c0, lam0), None, length=lm_iter)

    def gn(carry, _):
        u, r, J, chi2 = carry
        JTJ = jnp.einsum("kpi,kpj->kij", J, J)
        JTr = jnp.einsum("kpi,kp->ki", J, r)
        du = solve(JTJ + 1e-12 * eye, JTr)
        du = jnp.where(jnp.isfinite(du), du, 0.0)
        for s in (1.0, 0.5, 0.25, 0.125):
            un = u + s * du
            rn, Jn = rj(un)
            cn = (rn * rn).sum(1)
            better = cn < chi2
            u = jnp.where(better[:, None], un, u)
            r = jnp.where(better[:, None], rn, r)
            J = jnp.where(better[:, None, None], Jn, J)
            chi2 = jnp.where(better, cn, chi2)
            du = jnp.where(better[:, None], 0.0, du)
        return (u, r, J, chi2), None

    (u, r, J, chi2), _ = lax.scan(gn, (u, r, J, chi2), None, length=gn_iter)
    JTJ = jnp.einsum("kpi,kpj->kij", J, J)
    JTr = jnp.einsum("kpi,kp->ki", J, r)
    step = solve(JTJ + 1e-12 * eye, JTr)
    edm = 0.5 * jnp.einsum("ki,ki->k", JTr, step)
    eig = jnp.linalg.eigvalsh(JTJ)
    hess_ok = eig[:, 0] > 1e-12 * jnp.maximum(eig[:, -1], 1e-300)
    conv = jnp.isfinite(edm) & (edm <= 1e-3 * jnp.maximum(chi2, 1.0))
    return _to_ext(u, lo, hi), chi2, conv, hess_ok


def reconstruct_dense(adc, n, start_tdc, pedestal_db, calib_area, calib_height, cfg=RecoConfig()):
    """adc [W,L] (padded), n, start_tdc, pedestal_db, calib_area, calib_height [W]. Dense outputs [W, P].
    Recompiles only for a new (W, L) or a new STATIC_FIELDS value; the other settings of cfg are data."""
    return _reconstruct_dense(adc, n, start_tdc, pedestal_db, calib_area, calib_height, knobs_of(cfg),
                              P=cfg.max_pulses, sb=cfg.samples_before, sa=cfg.samples_after, lm=cfg.lm_iterations)


@partial(jax.jit, static_argnames=("P", "sb", "sa", "lm"))
def _reconstruct_dense(adc, n, start_tdc, pedestal_db, calib_area, calib_height, cfg, P, sb, sa, lm):
    W, L = adc.shape
    cfg = _Knobs(cfg)
    PW = sb + sa + 1 + PLATEAU_SLOTS
    adc = adc.astype(jnp.float64)
    under = pedestal_db - adc[:, 0] > cfg.pedestal_undershoot_threshold
    ped = jnp.where(under, adc[:, 0], pedestal_db)
    valid = jnp.arange(L)[None, :] < n[:, None]
    y0 = jnp.where(valid, adc - ped[:, None], 0.0)
    tgrid = (start_tdc[:, None] + jnp.arange(L)[None, :]) * PERIOD
    near0 = _near_peak(y0, valid, cfg.min_adc_difference, cfg.knife_edge_adc)

    def round_(carry, _):
        y, start, active, failed_before, near, overflow = carry
        found, ps, pe = _peak(y, n, start, active, cfg.min_adc_difference)
        fs, fe = _window(y, n, ps, pe, found, sb, sa)
        overflow = overflow | (found & (fe - fs + 1 > PW))
        yps = jnp.take_along_axis(y, ps[:, None], 1)[:, 0]
        t_ps = (start_tdc + ps) * PERIOD
        t_pe = (start_tdc + pe) * PERIOD
        t_peak = 0.5 * (t_ps + t_pe)
        p0 = jnp.stack([yps * E, t_peak, jnp.full(W, cfg.default_beta)], 1)
        lo = jnp.stack([yps * E * cfg.min_pulse_height_ratio, t_ps - cfg.max_time_difference,
                        jnp.full(W, cfg.min_beta)], 1)
        hi = jnp.stack([yps * E * cfg.max_pulse_height_ratio, t_pe + cfg.max_time_difference,
                        jnp.full(W, cfg.max_beta)], 1)
        hi = jnp.where(found[:, None], hi, lo + 1.0)        # keep idle rows finite
        idx = fs[:, None] + jnp.arange(PW)[None, :]
        mask = (idx <= fe[:, None]) & found[:, None]
        idxc = jnp.clip(idx, 0, L - 1)
        tt = jnp.take_along_axis(tgrid, idxc, 1)
        yy = jnp.take_along_axis(y, idxc, 1)
        par, chi2, conv, hok = _fit(tt, yy, mask.astype(jnp.float64), p0, lo, hi, lm, 8)
        npts = mask.sum(1)
        ndf = npts - 3
        span = hi - lo
        near_lim = (((par - lo) / span < cfg.limit_tolerance) | ((hi - par) / span < cfg.limit_tolerance)).any(1)
        failed = ~conv | ~hok | near_lim
        A, mu, b = par[:, 0], par[:, 1], par[:, 2]
        pe_ = jnp.where(failed, yps * E * cfg.default_beta, A * b) / calib_area
        tpulse = jnp.where(failed, t_peak, mu)
        height = jnp.where(failed, yps, A / E)
        beta = jnp.where(failed, cfg.default_beta, b)
        c2 = jnp.where(failed | (ndf <= 0), -1.0, chi2 / jnp.maximum(ndf, 1))
        failed_seq = failed | failed_before
        sub = jnp.where(valid & found[:, None],
                        jnp_round_half_away(_gumbel(tgrid, A[:, None], mu[:, None], b[:, None])), 0.0)
        y = y - sub
        near = near | (found & _near_peak(y, valid, cfg.min_adc_difference, cfg.knife_edge_adc))
        start = jnp.where(found, fe, start)
        rec = dict(found=found, pe=pe_, peph=height / calib_height, time=tpulse, height=height, beta=beta, chi2=c2,
                   letime=tpulse - cfg.le_time_factor * beta, failed=failed_seq, zerondf=ndf <= 0,
                   fit_A=A, fit_mu=mu, fit_b=b)
        return (y, start, found, jnp.where(found, failed_seq, failed_before), near, overflow), rec

    init = (y0, jnp.zeros(W, jnp.int64), jnp.ones(W, bool), jnp.zeros(W, bool), near0, jnp.zeros(W, bool))
    (y, _, _, _, near, overflow), recs = lax.scan(round_, init, None, length=P)
    out = {k: v.T for k, v in recs.items()}                  # [W, P]
    out["npulse"] = out["found"].sum(1)
    out["ped"] = ped
    out["pedfromdb"] = ~under
    out["wf_near_threshold"] = near
    out["window_overflow"] = overflow
    out["rounds_exhausted"] = out["found"][:, -1]   # a pulse in the last round: more may follow -> numpy path
    return out


class _Knobs:
    """Attribute access to the traced knob dict inside the kernel (cfg.min_beta etc., as in pulsefit.py)."""
    def __init__(self, d):
        self.__dict__.update(d)


def jnp_round_half_away(x):
    """std::round (jnp.round rounds half to even)."""
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


def to_flat(dense):
    """Dense [W,P] outputs -> pulsefit-style flat dict (host numpy), stock order (waveform, then sequence)."""
    d = {k: np.asarray(v) for k, v in dense.items()}
    W, P = d["found"].shape
    wf, seq = np.nonzero(d["found"])
    out = {k: d[k][wf, seq] for k in ("pe", "peph", "time", "height", "beta", "chi2", "letime", "failed", "zerondf",
                                      "fit_A", "fit_mu", "fit_b")}
    out.update(wf=wf, seq=seq, ped=d["ped"][wf], pedfromdb=d["pedfromdb"][wf],
               wf_near_threshold=d["wf_near_threshold"], window_overflow=d["window_overflow"],
               rounds_exhausted=d["rounds_exhausted"])
    return out
