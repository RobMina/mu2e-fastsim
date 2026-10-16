"""Batched CRV waveform synthesis and digitisation from SiPM charges: a vectorised port of Offline CRVResponse
CrvWaveformsGenerator (zero-suppressed path) + MakeCrvWaveforms + CrvDigitizer/MakeCrvDigis.

Stock algorithm per SiPM (Offline v13_39_00 = main 2026-10-16; settings CRVResponse/fcl/prolog_v12.fcl):
  1. charges (time, charge) are split into clusters wherever consecutive charges are more than
     singlePEWaveformMaxTime + 4 periods apart; a cluster is skipped if n_charges * max(1PE waveform) < minVoltage
  2. the cluster's waveform starts at the first FEB clock tick (TDC0 + k*12.5 ns) after first charge - 12.5 ns
  3. every charge adds charge/refCharge * singlePE(t_sample - t_charge) to the samples it reaches (1PE template in
     0.5 ns steps, stretched by 1.047, 100 ns long)
  4. Gaussian electronic noise (0.4 mV) on every sample
  5. zero suppression: a 12-sample digi starts at sample i if any of samples i-2..i+2 exceeds minVoltage (27.5 mV),
     the start lies inside the digitisation window, and the scan resumes after the digi
  6. ADC = int16(V * 2300 + 100 + 0.5) clipped to [0, 4095]; startTDC = lrint((start - TDC0) / 12.5)

TDC0 is per FEB and random within one period in the stock chain; here it is an input (taken from the reference digi MC
when validating, drawn when generating). All loops run over the short sample axis or the template; never over charges.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..geometry import find_counters
from .pulsefit import PERIOD

N_DIGI_PERIODS_GAP = 4


@dataclass(frozen=True)
class WaveformConfig:
    """CrvWaveforms + CrvDigi settings of Offline CRVResponse/fcl/prolog_v12.fcl (extracted: offspill window)."""
    single_pe_file: str = "Offline/CRVResponse/data/singlePEWaveform_v3.txt"
    single_pe_precision: float = 0.5
    single_pe_stretch: float = 1.047
    single_pe_max_time: float = 100.0
    single_pe_reference_charge: float = 2.652e-13
    min_voltage: float = 0.0275
    noise: float = 4.0e-4
    samples_zs: int = 12
    adc_conversion: float = 2300.0
    pedestal: int = 100
    min_adc: int = 0
    max_adc: int = 4095


@lru_cache(maxsize=4)
def single_pe_template(path, precision, stretch, max_time):
    """Port of MakeCrvWaveforms::LoadSinglePEWaveform: the 1PE template resampled to `precision` ns."""
    raw = np.loadtxt(find_counters(path))
    t, v = raw[:, 0] * stretch, raw[:, 1]
    out = [v[0]]
    idx = 1
    for i in range(1, len(t)):
        tt = idx * precision
        while t[i] >= tt and idx * precision < max_time:
            frac = (tt - t[i - 1]) / (t[i] - t[i - 1])
            out.append((v[i] - v[i - 1]) * frac + v[i - 1])
            idx += 1
            tt = idx * precision
        if idx * precision >= max_time:
            break
    return np.asarray(out)


def charge_clusters(chan, time, cfg):
    """Cluster id per charge (charges sorted by channel then time, as CrvSiPMCharges stores them)."""
    gap = cfg.single_pe_max_time + N_DIGI_PERIODS_GAP * PERIOD
    new = np.ones(len(time), bool)
    if len(time) > 1:
        new[1:] = (chan[1:] != chan[:-1]) | (time[1:] - time[:-1] > gap)
    return np.cumsum(new) - 1


def synthesize(chan, time, charge, tdc0_of_chan, cfg=WaveformConfig(), window=(0.0, 1e5), rng=None,
               max_charges=2_000_000):
    """Zero-suppressed digis from SiPM charges, in batches of whole charge clusters (memory ~ 10 x charges per batch).

    chan [Q] channel id per charge (any int label, charges grouped by chan and time-ordered within it), time [Q] ns,
    charge [Q] C, tdc0_of_chan: callable or array mapping chan -> TDC0 time [ns]. rng None = no noise.
    Returns dict: per-digi 'chan', 'start' (ns), 'starttdc', 'volt' [D,12], 'adc' [D,12]; and per-cluster 'cluster_*'.
    """
    cid = charge_clusters(chan, time, cfg)
    if len(cid) <= max_charges:
        return _synthesize(chan, time, charge, tdc0_of_chan, cfg, window, rng)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    cuts = [0]
    for s in starts[np.searchsorted(starts, np.arange(max_charges, len(cid), max_charges))]:
        if s > cuts[-1]:
            cuts.append(int(s))
    cuts.append(len(cid))
    parts = [_synthesize(chan[a:b], time[a:b], charge[a:b], tdc0_of_chan, cfg, window, rng)
             for a, b in zip(cuts[:-1], cuts[1:])]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _synthesize(chan, time, charge, tdc0_of_chan, cfg, window, rng):
    """One batch of synthesize.

    chan [Q] channel id per charge (any int label, charges grouped by chan and time-ordered within it), time [Q] ns,
    charge [Q] C, tdc0_of_chan: callable or array mapping chan -> TDC0 time [ns]. rng None = no noise.
    Returns dict: per-digi 'chan', 'start' (ns), 'starttdc', 'volt' [D,12], 'adc' [D,12]; and per-cluster 'cluster_*'.
    """
    tmpl = single_pe_template(cfg.single_pe_file, cfg.single_pe_precision, cfg.single_pe_stretch,
                              cfg.single_pe_max_time)
    vmax = tmpl.max()
    cid = charge_clusters(chan, time, cfg)
    C = int(cid[-1]) + 1 if len(cid) else 0
    first = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]]) if C else np.zeros(0, np.int64)
    last = np.r_[first[1:] - 1, len(cid) - 1] if C else np.zeros(0, np.int64)
    nq = np.bincount(cid, minlength=C)
    keep = nq * vmax >= cfg.min_voltage
    cchan = chan[first]
    tdc0 = tdc0_of_chan(cchan) if callable(tdc0_of_chan) else np.asarray(tdc0_of_chan)[cchan]
    t_first = time[first] - PERIOD
    t_start = np.ceil((t_first - tdc0) / PERIOD) * PERIOD + tdc0
    est = np.floor((time[last] - time[first] + cfg.single_pe_max_time) / PERIOD).astype(np.int64)
    # every charge reaches at most ceil(max_time / period) + 1 samples
    nreach = int(np.ceil(cfg.single_pe_max_time / PERIOD)) + 2
    q_ts = t_start[cid]
    j0 = np.maximum(np.ceil((time - q_ts) / PERIOD), 0).astype(np.int64)
    J = j0[:, None] + np.arange(nreach)[None, :]
    st = J * PERIOD + q_ts[:, None] - time[:, None]
    ti = np.rint(st / cfg.single_pe_precision).astype(np.int64)
    ok = (st >= 0) & (ti < len(tmpl)) & keep[cid][:, None]
    used_len = np.zeros(C, np.int64)
    if ok.any():
        np.maximum.at(used_len, np.repeat(cid, nreach)[ok.ravel()], J.ravel()[ok.ravel()] + 1)
    length = np.maximum(est, used_len)
    L = int(length.max()) + cfg.samples_zs if C else 0
    wave = np.zeros((C, L))
    vals = tmpl[np.clip(ti, 0, len(tmpl) - 1)] * (charge / cfg.single_pe_reference_charge)[:, None]
    np.add.at(wave, (np.repeat(cid, nreach)[ok.ravel()], J.ravel()[ok.ravel()]), vals.ravel()[ok.ravel()])
    valid = np.arange(L)[None, :] < length[:, None]
    if rng is not None:
        wave += rng.normal(0.0, cfg.noise, size=wave.shape) * valid
    # zero suppression scan over the sample axis
    above = (wave > cfg.min_voltage) & valid
    win = np.zeros_like(above)
    for d in range(-2, 3):
        sh = np.roll(above, -d, axis=1)
        if d > 0:
            sh[:, L - d:] = False
        elif d < 0:
            sh[:, :-d] = False
        win |= sh
    win &= valid & keep[:, None]
    next_i = np.zeros(C, np.int64)
    starts_r, starts_i = [], []
    for i in range(L):
        cand = win[:, i] & (next_i <= i)
        if not cand.any():
            continue
        tstart = t_start + i * PERIOD
        inwin = cand & (tstart >= window[0]) & (tstart <= window[1])
        r = np.flatnonzero(inwin)
        starts_r.append(r)
        starts_i.append(np.full(len(r), i))
        next_i[r] = i + cfg.samples_zs
    if starts_r:
        r = np.concatenate(starts_r)
        i0 = np.concatenate(starts_i)
    else:
        r = i0 = np.zeros(0, np.int64)
    cols = i0[:, None] + np.arange(cfg.samples_zs)[None, :]
    volt = np.where(cols < length[r, None], wave[r[:, None], np.minimum(cols, L - 1)], 0.0)
    start = t_start[r] + i0 * PERIOD
    adc = np.clip((volt * cfg.adc_conversion + cfg.pedestal + 0.5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    order = np.lexsort((start, cchan[r]))
    return {"chan": cchan[r][order], "start": start[order], "tdc0": tdc0[r][order],
            "starttdc": np.rint((start - tdc0[r]) / PERIOD).astype(np.int64)[order],
            "volt": volt[order], "adc": adc[order], "cluster_chan": cchan, "cluster_tstart": t_start,
            "cluster_keep": keep}
