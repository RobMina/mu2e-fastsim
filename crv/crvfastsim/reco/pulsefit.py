"""Batched CRV pulse reconstruction: a vectorised port of Offline CRVReco MakeCrvRecoPulses + CrvRecoPulsesFinder.

Stock algorithm (Offline v13_39_00, identical on main 2026-10-16), per SiPM waveform:
  1. y = ADC - pedestal (DB pedestal, or the first sample if it undershoots the DB value by > pedestalUndershootThreshold)
  2. FindNextPeak from `start`: rising edge, optional plateau, falling edge with y[peakEnd] > minADCdifference
  3. fit window: up to samplesBefore / samplesAfter monotonic samples around the peak, then one dropped on each side
     when more than one was added
  4. Gumbel fit A*exp(-(t-mu)/b - exp(-(t-mu)/b)) by Minuit2 (TGraph::Fit "NQSR", unit errors) with parameter limits
  5. failed fit (status != 0, invalid, or any parameter within 1 % of a limit) -> peak-sample fallback values
  6. subtract round(fit) from every sample, start = fitEnd, repeat; a failure flags every later pulse of the waveform

Here every step runs over all waveforms at once (arrays [W, L]); loops run over the short sample axis or over pulse
rounds, never over waveforms. The fit is a fixed-iteration Levenberg-Marquardt in Minuit's internal (sine-mapped)
coordinates, float64, so interior minima match Minuit2 to its tolerance; the failed-fit flag is the hard part
(validated by V-R2, docs/fastreco.md).
"""
from dataclasses import dataclass, replace

import numpy as np

PERIOD = 12.5          # CRVDigitizationPeriod [ns]
E = np.e


@dataclass(frozen=True)
class RecoConfig:
    """CrvRecoPulses settings of Offline CRVReco/fcl/prolog_v12.fcl."""
    min_adc_difference: float = 40.0
    default_beta: float = 19.0
    min_beta: float = 5.0
    max_beta: float = 40.0
    max_time_difference: float = 15.0
    min_pulse_height_ratio: float = 0.7
    max_pulse_height_ratio: float = 1.5
    le_time_factor: float = 0.985
    samples_before: int = 4
    samples_after: int = 3
    pedestal_undershoot_threshold: float = 6.0
    use_time_offset_db: bool = True
    time_offset_scale: float = 1.0
    ignore_channels: bool = True
    max_pulses: int = 8            # pulse rounds per waveform (stock: unbounded; 8 is never reached in the samples seen)
    lm_iterations: int = 40
    limit_tolerance: float = 0.01  # FailedFit's proximity to a limit, as a fraction of the limit range
    knife_edge_adc: float = 1.0    # V-R2 bookkeeping: a residual this close to minADCdifference marks the waveform


def near_threshold_peak(y, valid, thr, edge):
    """Per row: a local maximum (y[b] >= y[b-1], y[b] > y[b+1]) whose value is within `edge` of `thr`."""
    yv = np.where(valid, y, -1e9)
    left = np.concatenate([np.full((y.shape[0], 1), -1e9), yv[:, :-1]], 1)
    right = np.concatenate([yv[:, 1:], np.full((y.shape[0], 1), -1e9)], 1)
    peak = valid & (yv >= left) & (yv > right)
    return (peak & (np.abs(yv - thr) <= edge)).any(1)


def round_half_away(x):
    """std::round."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def gumbel(t, A, mu, b):
    z = (t - mu) / b
    return A * np.exp(-z - np.exp(-z))


# ---------------------------------------------------------------------------------------------------------------------
# step 2-3: peak finding and fit window, vectorised over waveforms

def find_next_peak(y, n, start, active, cfg):
    """Port of FindNextPeak for all waveforms. y [W,L] float, n [W] valid lengths, start [W] int, active [W] bool.

    Returns found [W], peak_start, peak_end, fit_start, fit_end [W] int.
    """
    W, L = y.shape
    rows = np.arange(W)
    found = np.zeros(W, bool)
    rising = np.zeros(W, bool)
    done = ~active
    ps = np.zeros(W, np.int64)
    pe = np.zeros(W, np.int64)
    for b in range(1, L):
        live = ~done & (b > start) & (b < n)
        if not live.any():
            continue
        prev, cur = y[:, b - 1], y[:, b]
        up = live & (prev < cur)
        rising |= up
        ps[up] = b
        pe[up] = b
        flat = live & (prev == cur) & rising
        pe[flat] = b
        down = live & (prev > cur) & rising
        hit = down & (y[rows, pe] > cfg.min_adc_difference)
        found |= hit
        done |= hit
        rising &= ~(down & ~hit)
    fs = ps - 1
    fe = pe + 1
    go = found.copy()
    for k in range(1, cfg.samples_before + 1):
        i = ps - k
        ok = go & (i >= 0) & (i >= ps - cfg.samples_before)
        ii = np.clip(i, 0, L - 2)
        cond = ok & (y[rows, ii] <= y[rows, ii + 1])
        fs[cond] = i[cond]
        go = cond & (i != 0)
    go = found.copy()
    for k in range(1, cfg.samples_after + 1):
        i = pe + k
        ok = go & (i < n)
        ii = np.clip(i, 1, L - 1)
        cond = ok & (y[rows, ii - 1] >= y[rows, ii])
        fe[cond] = i[cond]
        go = cond
    fs = np.where(found & (ps - fs > 1), fs + 1, fs)
    fe = np.where(found & (fe - pe > 1), fe - 1, fe)
    return found, ps, pe, fs, fe


# ---------------------------------------------------------------------------------------------------------------------
# step 4: bounded least squares, Minuit-style sine mapping, batched Levenberg-Marquardt

def _to_ext(u, lo, hi):
    return lo + (hi - lo) * (np.sin(u) + 1.0) * 0.5


def _to_int(p, lo, hi):
    return np.arcsin(np.clip(2.0 * (p - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def fit_gumbel(t, y, mask, p0, lo, hi, iterations=40):
    """Least-squares Gumbel fits for a batch. t, y, mask [K,P]; p0, lo, hi [K,3] (A, mu, beta).

    Returns params [K,3], chi2 [K], converged [K], hessian_ok [K].
    """
    K = t.shape[0]
    u = _to_int(p0, lo, hi)
    lam = np.full(K, 1e-3)
    w = mask.astype(float)

    def resid_jac(u):
        p = _to_ext(u, lo, hi)
        A, mu, b = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        z = (t - mu) / b
        ez = np.exp(-z)
        g = np.exp(-z - ez)
        f = A * g
        r = (y - f) * w
        dfdA = g
        dfdz = f * (ez - 1.0)            # d/dz of A*exp(-z-e^-z)
        dfdmu = dfdz * (-1.0 / b)
        dfdb = dfdz * (-z / b)
        dpdu = (hi - lo) * 0.5 * np.cos(u)   # [K,3]
        J = np.stack([dfdA * dpdu[:, 0:1], dfdmu * dpdu[:, 1:2], dfdb * dpdu[:, 2:3]], axis=-1) * w[..., None]
        return r, J

    r, J = resid_jac(u)
    chi2 = (r * r).sum(1)
    for _ in range(iterations):
        JTJ = np.einsum("kpi,kpj->kij", J, J)
        JTr = np.einsum("kpi,kp->ki", J, r)
        D = np.einsum("kii->ki", JTJ)
        Aug = JTJ + (lam[:, None, None] * (D[:, :, None] * np.eye(3) + 1e-12 * np.eye(3)))
        try:
            du = np.linalg.solve(Aug, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            du = np.zeros_like(u)
        un = u + du
        rn, Jn = resid_jac(un)
        chi2n = (rn * rn).sum(1)
        better = chi2n < chi2
        u = np.where(better[:, None], un, u)
        r = np.where(better[:, None], rn, r)
        J = np.where(better[:, None, None], Jn, J)
        chi2 = np.where(better, chi2n, chi2)
        lam = np.clip(np.where(better, lam * 0.3, lam * 10.0), 1e-9, 1e9)
    # polish: plain Gauss-Newton with step halving (LM's damping can stall a step short of the minimum)
    for _ in range(8):
        JTJ = np.einsum("kpi,kpj->kij", J, J)
        JTr = np.einsum("kpi,kp->ki", J, r)
        try:
            du = np.linalg.solve(JTJ + 1e-12 * np.eye(3), JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        du = np.where(np.isfinite(du), du, 0.0)
        for scale in (1.0, 0.5, 0.25, 0.125):
            un = u + scale * du
            rn, Jn = resid_jac(un)
            chi2n = (rn * rn).sum(1)
            better = chi2n < chi2
            u = np.where(better[:, None], un, u)
            r = np.where(better[:, None], rn, r)
            J = np.where(better[:, None, None], Jn, J)
            chi2 = np.where(better, chi2n, chi2)
            du = np.where(better[:, None], 0.0, du)
    # convergence: one more step would not lower chi2 by more than 1e-6 relative (Minuit2 EDM-like)
    JTJ = np.einsum("kpi,kpj->kij", J, J)
    JTr = np.einsum("kpi,kp->ki", J, r)
    try:
        step = np.linalg.solve(JTJ + 1e-12 * np.eye(3), JTr[..., None])[..., 0]
        edm = 0.5 * np.einsum("ki,ki->k", JTr, step)
        eig = np.linalg.eigvalsh(JTJ)
        hess_ok = eig[:, 0] > 1e-12 * np.maximum(eig[:, -1], 1e-300)
    except np.linalg.LinAlgError:
        edm = np.full(K, np.inf)
        hess_ok = np.zeros(K, bool)
    converged = np.isfinite(edm) & (edm <= 1e-3 * np.maximum(chi2, 1.0))
    return _to_ext(u, lo, hi), chi2, converged, hess_ok


# ---------------------------------------------------------------------------------------------------------------------
# steps 1-6 for a batch of waveforms

def reconstruct_full(adc, n, start_tdc, pedestal_db, calib_area, calib_height, cfg=RecoConfig()):
    """Full-length host recovery; advancing fit windows bound the pulse count by the sample count."""
    return reconstruct(adc, n, start_tdc, pedestal_db, calib_area, calib_height,
                       replace(cfg, max_pulses=max(cfg.max_pulses, int(np.max(n, initial=0)))))


def filter_channels(pulses, channels, status, cfg=RecoConfig()):
    """Remove bit-1 ignored-channel pulses while retaining original waveform indices and diagnostics."""
    if not cfg.ignore_channels:
        return pulses
    keep = (status[channels[pulses['wf']]] & 0b010) == 0
    return {k: v if k == 'wf_near_threshold' else v[keep] for k, v in pulses.items()}


def reconstruct(adc, n, start_tdc, pedestal_db, calib_area, calib_height, cfg=RecoConfig()):
    """Pulses of a batch of (already concatenated) waveforms.

    adc [W,L] int (padded), n [W] valid samples, start_tdc [W], pedestal_db / calib_area / calib_height [W].
    Returns a dict of flat pulse arrays (one entry per pulse) with 'wf' = waveform row and 'seq' = sequenceIndex,
    in stock order (by waveform, then sequence). Times are relative to TDC 0 (no TDC0time or DB time offset yet).
    """
    W, L = adc.shape
    if W == 0:
        out = {k: np.zeros(0) for k in ('pe', 'peph', 'time', 'height', 'beta', 'chi2', 'letime', 'ped',
                                       'fit_A', 'fit_mu', 'fit_b')}
        out.update({k: np.zeros(0, np.int64) for k in ('wf', 'seq')})
        out.update({k: np.zeros(0, bool) for k in ('failed', 'zerondf', 'pedfromdb', 'wf_near_threshold')})
        return out
    rows = np.arange(W)
    adc = adc.astype(float)
    ped = np.where(pedestal_db - adc[:, 0] > cfg.pedestal_undershoot_threshold, adc[:, 0], pedestal_db)
    ped_from_db = ~(pedestal_db - adc[:, 0] > cfg.pedestal_undershoot_threshold)
    y = adc - ped[:, None]
    y[np.arange(L)[None, :] >= n[:, None]] = 0.0
    tgrid = (start_tdc[:, None] + np.arange(L)[None, :]) * PERIOD
    start = np.zeros(W, np.int64)
    active = np.ones(W, bool)
    failed_before = np.zeros(W, bool)
    valid = np.arange(L)[None, :] < n[:, None]
    near_thr = near_threshold_peak(y, valid, cfg.min_adc_difference, cfg.knife_edge_adc)
    out = {k: [] for k in ("wf", "seq", "pe", "peph", "time", "height", "beta", "chi2", "letime", "failed", "zerondf",
                           "ped", "pedfromdb", "fit_A", "fit_mu", "fit_b")}
    for seq in range(cfg.max_pulses):
        found, ps, pe, fs, fe = find_next_peak(y, n, start, active, cfg)
        if not found.any():
            break
        k = np.flatnonzero(found)
        P = int((fe[k] - fs[k]).max()) + 1
        yps = y[k, ps[k]]
        t_ps = (start_tdc[k] + ps[k]) * PERIOD
        t_pe = (start_tdc[k] + pe[k]) * PERIOD
        t_peak = 0.5 * (t_ps + t_pe)
        p0 = np.stack([yps * E, t_peak, np.full(len(k), cfg.default_beta)], 1)
        lo = np.stack([yps * E * cfg.min_pulse_height_ratio, t_ps - cfg.max_time_difference,
                       np.full(len(k), cfg.min_beta)], 1)
        hi = np.stack([yps * E * cfg.max_pulse_height_ratio, t_pe + cfg.max_time_difference,
                       np.full(len(k), cfg.max_beta)], 1)
        idx = fs[k, None] + np.arange(P)[None, :]
        mask = idx <= fe[k, None]
        idxc = np.clip(idx, 0, L - 1)
        tt = tgrid[k[:, None], idxc]
        yy = y[k[:, None], idxc]
        par, chi2, conv, hok = fit_gumbel(tt, yy, mask, p0, lo, hi, cfg.lm_iterations)
        npts = mask.sum(1)
        ndf = npts - 3
        span = hi - lo
        near = ((par - lo) / span < cfg.limit_tolerance) | ((hi - par) / span < cfg.limit_tolerance)
        failed = ~conv | ~hok | near.any(1)
        A, mu, b = par[:, 0], par[:, 1], par[:, 2]
        pe_ = np.where(failed, yps * E * cfg.default_beta, A * b) / calib_area[k]
        tpulse = np.where(failed, t_peak, mu)
        height = np.where(failed, yps, A / E)
        beta = np.where(failed, cfg.default_beta, b)
        c2 = np.where(failed | (ndf <= 0), -1.0, chi2 / np.maximum(ndf, 1))
        failed_seq = failed | failed_before[k]
        for key, v in (("wf", k), ("seq", np.full(len(k), seq)), ("pe", pe_), ("peph", height / calib_height[k]),
                       ("time", tpulse), ("height", height), ("beta", beta), ("chi2", c2),
                       ("letime", tpulse - cfg.le_time_factor * beta), ("failed", failed_seq), ("zerondf", ndf <= 0),
                       ("ped", ped[k]), ("pedfromdb", ped_from_db[k]), ("fit_A", A), ("fit_mu", mu), ("fit_b", b)):
            out[key].append(v)
        failed_before[k] = failed_seq
        sub = round_half_away(gumbel(tgrid[k], A[:, None], mu[:, None], b[:, None]))
        sub[np.arange(L)[None, :] >= n[k, None]] = 0.0
        y[k] -= sub
        near_thr[k] |= near_threshold_peak(y[k], valid[k], cfg.min_adc_difference, cfg.knife_edge_adc)
        start[k] = fe[k]
        active = found
    bool_keys = ('failed', 'zerondf', 'pedfromdb')
    res = {key: (np.concatenate(v) if v else np.zeros(0, bool if key in bool_keys else
                                                     np.int64 if key in ('wf', 'seq') else float))
           for key, v in out.items()}
    order = np.lexsort((res["seq"], res["wf"])) if len(res["wf"]) else np.zeros(0, np.int64)
    res = {key: v[order] for key, v in res.items()}
    res["wf_near_threshold"] = near_thr          # per waveform, not per pulse
    return res


def concat_waveforms(key, bar, sipm, start_tdc, adc_flat, lens):
    """CrvRecoPulsesFinder's grouping: consecutive digis of one channel whose TDCs continue are one waveform.

    Inputs are per-digi arrays in collection order (key = event key), the ADC samples of all digis flattened and the
    sample count per digi. Returns group id per digi and, per group, the padded ADC matrix, valid lengths, start TDC
    and the first digi index.
    """
    nd = len(bar)
    lens = np.asarray(lens, dtype=np.int64)
    new = np.ones(nd, bool)
    if nd > 1:
        new[1:] = ~((key[1:] == key[:-1]) & (bar[1:] == bar[:-1]) & (sipm[1:] == sipm[:-1]) &
                    (start_tdc[1:] == start_tdc[:-1] + lens[:-1]))
    gid = np.cumsum(new) - 1
    G = int(gid[-1]) + 1 if nd else 0
    glen = np.bincount(gid, weights=lens, minlength=G).astype(np.int64)
    L = int(glen.max()) if G else 0
    first = np.flatnonzero(new)
    offset = np.zeros(nd, np.int64)
    if nd:
        csum = np.cumsum(lens) - lens
        offset = csum - csum[first][gid]
    mat = np.zeros((G, L), np.int64)
    within = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
    col = np.repeat(offset, lens) + within
    mat[np.repeat(gid, lens), col] = np.asarray(adc_flat, dtype=np.int64)
    return gid, mat, glen, start_tdc[first], first
