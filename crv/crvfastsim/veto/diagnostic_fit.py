"""Length-bucketed diagnostic pulse fitting; production and saved helpers remain unchanged."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..reco import pulsefit as P
from . import replay_waveform_control as W, diagnostic_lm as LM

CONTRACT = "diagnostic-fit-width-buckets-v1"


@dataclass(frozen=True)
class Limits:
    matrix_cells: int = 2_000_000
    sample_rows: int = 2_000_000
    max_length: int = 1024

    def validate(self):
        if any(type(v) is not int or not 0 < v <= 10**8 for v in vars(self).values()):
            raise ValueError("invalid diagnostic fit limits")


def fit_buckets(t, y, mask, p0, lo, hi, iterations=40):
    """Trim each fit to its own prefix length before unchanged vectorized least squares."""
    if t.ndim != 2 or y.shape != t.shape or mask.shape != t.shape or mask.dtype != bool:
        raise ValueError("invalid fit matrix")
    if t.size > 2_000_000 or t.shape[1] > 1024:
        raise ValueError("fit bucket allocation limit exceeded")
    count = mask.sum(axis=1)
    if np.any(count < 1) or not np.array_equal(mask, np.arange(t.shape[1])[None, :] < count[:, None]):
        raise ValueError("fit masks must be nonempty prefixes")
    if any(a.shape != (len(t), 3) for a in (p0, lo, hi)):
        raise ValueError("invalid fit parameters")
    par, chi2 = np.empty((len(t), 3)), np.empty(len(t))
    conv, hok = np.empty(len(t), bool), np.empty(len(t), bool)
    for width in np.unique(count):
        rows = np.flatnonzero(count == width)
        values = LM.fit_gumbel(t[rows, :width].copy(), y[rows, :width].copy(),
                              np.ones((len(rows), width), bool), p0[rows], lo[rows], hi[rows], iterations)
        par[rows], chi2[rows], conv[rows], hok[rows] = values
    return par, chi2, conv, hok


def reconstruct_matrix(adc, n, start_tdc, pedestal_db, calib_area, calib_height, cfg=P.RecoConfig(), *, limits=Limits()):
    """Full recovery with the production peak/subtraction rules and width-bucketed fits."""
    limits.validate()
    if cfg != P.RecoConfig():
        raise ValueError("diagnostic fit requires the default reconstruction configuration")
    adc, n = np.asarray(adc), np.asarray(n)
    if adc.dtype.kind not in "iu" or np.any(adc < 0) or np.any(adc > 4095):
        raise ValueError("diagnostic ADCs must be 12-bit integers")
    if adc.ndim != 2 or n.shape != (len(adc),) or n.dtype.kind not in "iu":
        raise ValueError("invalid waveform matrix/lengths")
    size, length = adc.shape
    if size * length > limits.matrix_cells or length > limits.max_length or int(n.sum()) > limits.sample_rows:
        raise ValueError("diagnostic fit allocation limit exceeded")
    if np.any(n < 1) or np.any(n > length):
        raise ValueError("invalid valid waveform length")
    arrays = [np.asarray(a) for a in (start_tdc, pedestal_db, calib_area, calib_height)]
    if any(a.shape != (size,) or not np.isfinite(a).all() for a in arrays) or not np.isfinite(adc).all():
        raise ValueError("nonfinite or malformed fit inputs")
    start_tdc, pedestal_db, calib_area, calib_height = arrays
    if start_tdc.dtype.kind not in "iu" or np.any(np.abs(start_tdc.astype(float)) > 2**40):
        raise ValueError("invalid waveform start TDC")
    if (np.any(calib_area < 1e-12) or np.any(calib_height < 1e-12)
            or np.any(calib_area > 1e12) or np.any(calib_height > 1e12)
            or np.any(pedestal_db < 0) or np.any(pedestal_db > 4095)):
        raise ValueError("pulse calibration outside diagnostic numeric domain")
    if not size:
        return P.reconstruct_full(adc, n, start_tdc, pedestal_db, calib_area, calib_height, cfg)
    adc = adc.astype(float)
    ped = np.where(pedestal_db - adc[:, 0] > cfg.pedestal_undershoot_threshold, adc[:, 0], pedestal_db)
    ped_from_db = ~(pedestal_db - adc[:, 0] > cfg.pedestal_undershoot_threshold)
    valid = np.arange(length)[None, :] < n[:, None]
    y = np.where(valid, adc - ped[:, None], 0.)
    tgrid = (start_tdc[:, None] + np.arange(length)[None, :]) * P.PERIOD
    start, active, failed_before = np.zeros(size, np.int64), np.ones(size, bool), np.zeros(size, bool)
    near_thr = P.near_threshold_peak(y, valid, cfg.min_adc_difference, cfg.knife_edge_adc)
    out = {k: [] for k in ("wf", "seq", "pe", "peph", "time", "height", "beta", "chi2", "letime", "failed", "zerondf",
                           "ped", "pedfromdb", "fit_A", "fit_mu", "fit_b")}
    for seq in range(max(cfg.max_pulses, int(n.max()))):
        found, ps, pe, fs, fe = P.find_next_peak(y, n, start, active, cfg)
        if not found.any():
            break
        k = np.flatnonzero(found)
        width = int((fe[k] - fs[k]).max()) + 1
        yps = y[k, ps[k]]
        t_ps, t_pe = (start_tdc[k] + ps[k]) * P.PERIOD, (start_tdc[k] + pe[k]) * P.PERIOD
        t_peak = .5 * (t_ps + t_pe)
        p0 = np.stack([yps * P.E, t_peak, np.full(len(k), cfg.default_beta)], 1)
        lo = np.stack([yps * P.E * cfg.min_pulse_height_ratio, t_ps - cfg.max_time_difference,
                       np.full(len(k), cfg.min_beta)], 1)
        hi = np.stack([yps * P.E * cfg.max_pulse_height_ratio, t_pe + cfg.max_time_difference,
                       np.full(len(k), cfg.max_beta)], 1)
        idx = fs[k, None] + np.arange(width)[None, :]
        mask = idx <= fe[k, None]
        idxc = np.clip(idx, 0, length - 1)
        par, chi2, conv, hok = fit_buckets(tgrid[k[:, None], idxc], y[k[:, None], idxc], mask,
                                         p0, lo, hi, cfg.lm_iterations)
        ndf = mask.sum(1) - 3
        span = hi - lo
        near = ((par - lo) / span < cfg.limit_tolerance) | ((hi - par) / span < cfg.limit_tolerance)
        failed = ~conv | ~hok | near.any(1)
        A, mu, b = par[:, 0], par[:, 1], par[:, 2]
        pe_ = np.where(failed, yps * P.E * cfg.default_beta, A * b) / calib_area[k]
        tpulse = np.where(failed, t_peak, mu)
        height = np.where(failed, yps, A / P.E)
        beta = np.where(failed, cfg.default_beta, b)
        c2 = np.where(failed | (ndf <= 0), -1., chi2 / np.maximum(ndf, 1))
        failed_seq = failed | failed_before[k]
        for key, value in (("wf", k), ("seq", np.full(len(k), seq)), ("pe", pe_), ("peph", height / calib_height[k]),
                           ("time", tpulse), ("height", height), ("beta", beta), ("chi2", c2),
                           ("letime", tpulse - cfg.le_time_factor * beta), ("failed", failed_seq), ("zerondf", ndf <= 0),
                           ("ped", ped[k]), ("pedfromdb", ped_from_db[k]), ("fit_A", A), ("fit_mu", mu), ("fit_b", b)):
            out[key].append(value)
        failed_before[k] = failed_seq
        sub = P.round_half_away(P.gumbel(tgrid[k], A[:, None], mu[:, None], b[:, None]))
        sub[~valid[k]] = 0.
        y[k] -= sub
        near_thr[k] |= P.near_threshold_peak(y[k], valid[k], cfg.min_adc_difference, cfg.knife_edge_adc)
        start[k], active = fe[k], found
    bools = ("failed", "zerondf", "pedfromdb")
    result = {key: np.concatenate(value) if value else np.zeros(0, bool if key in bools else
              np.int64 if key in ("wf", "seq") else float) for key, value in out.items()}
    order = np.lexsort((result["seq"], result["wf"]))
    result = {key: value[order] for key, value in result.items()}
    result["wf_near_threshold"] = near_thr
    return result


def reconstruct(digis, conditions, *, limits=Limits()):
    """Guard concatenation before allocation and restore the unchanged diagnostic pulse schema."""
    limits.validate()
    label, tdc, adc = (np.asarray(digis[k]) for k in ("chan", "starttdc", "adc"))
    if adc.ndim != 2 or label.shape != tdc.shape or label.shape != (len(adc),):
        raise ValueError("invalid digi arrays")
    if label.dtype.kind not in "iu" or tdc.dtype.kind not in "iu" or np.any(label < 0):
        raise ValueError("invalid digi identities")
    if np.any(label >= 2**20 * W.MULT) or np.any(np.abs(tdc.astype(float)) > 2**40):
        raise ValueError("digi identities outside helper domain")
    if (adc.shape[1] < 1 or adc.size > limits.sample_rows or adc.dtype.kind not in "iu"
            or np.any(adc < 0) or np.any(adc > 4095)):
        raise ValueError("digi sample limit or invalid 12-bit integer ADC")
    for name in ("pedestal", "pulsearea", "pulseheight", "timeoffset", "w"):
        values = conditions[name].to_numpy()
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all() or np.any(np.abs(values) > 1e12):
            raise ValueError("invalid diagnostic channel conditions")
    status = conditions.status.to_numpy()
    if status.dtype.kind not in "iu" or np.any(status < 0) or np.any(status > 2**31 - 1):
        raise ValueError("invalid diagnostic channel status")
    order = np.lexsort((tdc, label))
    label, tdc, adc = label[order], tdc[order], adc[order]
    if len(tdc) > 1 and np.any((label[1:] == label[:-1]) & (tdc[1:] == tdc[:-1])):
        raise ValueError("duplicate digi starts")
    lens = np.full(len(tdc), adc.shape[1], np.int64)
    new = np.r_[True, (label[1:] != label[:-1]) | (tdc[1:] != tdc[:-1] + lens[:-1])] if len(tdc) else np.zeros(0, bool)
    first = np.flatnonzero(new)
    lengths = np.diff(np.r_[first, len(tdc)]) * adc.shape[1]
    width = int(lengths.max(initial=0))
    if len(first) * width > limits.matrix_cells or width > limits.max_length:
        raise ValueError("concatenation allocation limit exceeded")
    ev, ch = label // W.MULT, label % W.MULT
    _, mat, n, start, first = P.concat_waveforms(ev, ch // 4, ch % 4, tdc, adc.ravel(), lens)
    c = conditions.set_index(["event", "chan"])
    if not c.index.is_unique:
        raise ValueError("duplicate frame/channel conditions")
    c = c.loc[pd.MultiIndex.from_arrays([ev[first], ch[first]])]
    cfg = P.RecoConfig()
    result = reconstruct_matrix(mat, n, start, c.pedestal.to_numpy(), c.pulsearea.to_numpy(), c.pulseheight.to_numpy(), limits=limits)
    near = result.pop("wf_near_threshold")
    pulses = pd.DataFrame(result)
    wi = pulses.wf.to_numpy(dtype=np.int64)
    pulses["event"], pulses["chan"] = ev[first][wi], ch[first][wi]
    pulses["time_relative"] = pulses.time + c.timeoffset.to_numpy()[wi]
    pulses["time_reco"] = pulses.time_relative + c.w.to_numpy()[wi]
    pulses["channel_ignored"] = (c.status.to_numpy()[wi].astype(int) & 2) != 0
    valid = np.arange(mat.shape[1])[None, :] < n[:, None]
    ped = np.where(c.pedestal.to_numpy() - (mat[:, 0] if len(n) else []) > cfg.pedestal_undershoot_threshold,
                   mat[:, 0] if len(n) else [], c.pedestal.to_numpy())
    ymax = np.max(np.where(valid, mat - ped[:, None], -np.inf), axis=1) if len(n) else np.zeros(0)
    waveforms = pd.DataFrame({"wf": np.arange(len(n)), "event": ev[first], "chan": ch[first], "starttdc": start,
        "n": n, "pedestal_used": ped, "max_adc_minus_pedestal": ymax,
        "peak_margin_adc": ymax - cfg.min_adc_difference, "near_threshold": near})
    rows = np.repeat(np.arange(len(n)), n)
    column = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    samples = pd.DataFrame({"wf": rows, "event": ev[first][rows], "chan": ch[first][rows],
                            "tdc": start[rows] + column, "adc": mat[rows, column]})
    return pulses, waveforms, samples
