"""Stock calorimeter waveform, noise and ADC digitisation (Offline CaloMC/src/CaloDigiMaker_module.cc).

Works on the active readouts only: one waveform row per (event, SiPM) that has photoelectrons. The waveform build,
noise addition, ADC conversion and peak search run in NumPy or JAX (``backend="jax"``; JAX needs x64 enabled for the
double-precision waveform). Variable-length digi lists are gathered on the host.

Exactness: with noise off the arithmetic follows the source types (float32 PE-time shift, double waveform, ``int``
truncation). Summation order of the double waveform may differ from the source at roundoff level, which can flip a
truncated ADC count when a sample lies within ~1e-12 of an integer. With noise on the source draws CLHEP uniforms;
here the noise offsets come from the caller's generator, so only distributions can agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pulse import N_STEPS, PulseCache


@dataclass(frozen=True)
class DigiConfig:
    digitization_start: float = 450.0
    digitization_end: float = 1795.0
    digi_sampling: float = 5.0
    buffer_digi: int = 16
    n_bits: int = 12
    min_peak_adc: int = 16
    n_bins_peak: int = 2
    min_noise_amp: float = 2.0
    time_from_protons_to_dr_marker: float = 200.0

    @property
    def start_time_buffer(self):
        return np.float32(self.digi_sampling * self.buffer_digi)

    @property
    def waveform_size(self):
        return int(np.float32((np.float32(self.digitization_end) - np.float32(self.digitization_start)
                               + self.start_time_buffer)) / np.float32(self.digi_sampling))

    @property
    def max_adc(self):
        return (1 << self.n_bits) - 1


def _xp(backend):
    if backend == "numpy":
        return np
    import jax.numpy as jnp
    return jnp


def _scatter_add(xp, target, index, values):
    if xp is np:
        out = target.copy()
        np.add.at(out, index, values)
        return out
    return target.at[index].add(values)


def sample_times(pe_time, pbtime, cfg: DigiConfig):
    """Float32 PE time in the digitizer frame, in the source's left-to-right order."""
    f = np.float32
    t = np.asarray(pe_time, dtype=f) + np.asarray(pbtime, dtype=f)
    t = t - f(cfg.digitization_start)
    t = t + f(cfg.time_from_protons_to_dr_marker)
    return (t + cfg.start_time_buffer).astype(f)


def build_waveforms(row, time32, scale, n_rows, pulse: PulseCache, cfg: DigiConfig, *, weight=None, backend="numpy"):
    """Return ``(n_rows, waveform_size)`` float64 waveforms (ADC units, no noise, no pedestal).

    ``row`` maps each PE (or PE group) to its waveform row, ``time32`` comes from :func:`sample_times`, ``scale`` is
    ADCPerMeV/pePerMeV of that row's SiPM (float32 division) and ``weight`` an optional PE multiplicity. PEs with a
    negative digitizer time are dropped (the source's unsigned conversion makes them a no-op).
    """
    xp = _xp(backend)
    size = cfg.waveform_size
    time32 = np.asarray(time32, dtype=np.float32)
    keep = time32 >= 0
    row = np.asarray(row)[keep]
    time32 = time32[keep]
    w = np.ones(time32.shape) if weight is None else np.asarray(weight, dtype=np.float64)[keep]
    start = np.trunc(time32 / np.float32(cfg.digi_sampling)).astype(np.int64)
    first = pulse.shift_bin(time32.astype(np.float64))
    k = np.arange(pulse.n_bin_shape)
    sample = start[:, None] + k
    ok = sample < size
    amp = xp.asarray(pulse.pulse_vec)[xp.asarray(first[:, None] + N_STEPS * k)]
    amp = amp * xp.asarray((np.asarray(scale, dtype=np.float64)[keep] * w)[:, None])
    flat = np.where(ok, row[:, None] * size + sample, n_rows * size)
    out = xp.zeros(n_rows * size + 1, dtype=xp.float64)
    out = _scatter_add(xp, out, xp.asarray(flat.ravel()), amp.ravel())
    return out[:-1].reshape(n_rows, size)


def noise_coverage(waveform, cfg: DigiConfig, backend="numpy"):
    """Mask of samples that receive noise: within ``bufferDigi`` before or inside a run above ``minNoiseAmp``."""
    xp = _xp(backend)
    above = waveform > cfg.min_noise_amp
    n = cfg.buffer_digi
    padded = xp.concatenate([above, xp.zeros((above.shape[0], n), dtype=bool)], axis=1)
    cover = above
    for d in range(1, n + 1):
        cover = cover | padded[:, d:d + above.shape[1]]
    return cover


def add_noise(waveform, noise_wf, offsets, cfg: DigiConfig, backend="numpy"):
    """Add noise segments to covered ranges; ``offsets(range_lengths) -> start indices`` draws the segment offsets.

    Each maximal covered range gets one contiguous segment of ``noise_wf`` starting at a random offset in
    ``[0, len(noise_wf) - length)``, as ``CaloNoiseUtil::noiseSegment``.
    """
    xp = _xp(backend)
    cover = np.asarray(noise_coverage(waveform, cfg, backend))
    prev = np.concatenate([np.zeros((cover.shape[0], 1), bool), cover[:, :-1]], axis=1)
    nxt = np.concatenate([cover[:, 1:], np.zeros((cover.shape[0], 1), bool)], axis=1)
    r_row, r_start = np.nonzero(cover & ~prev)
    _, r_last = np.nonzero(cover & ~nxt)
    length = r_last - r_start + 1
    start_noise = np.asarray(offsets(length), dtype=np.int64)
    pos = np.arange(cover.shape[1])
    anchor = np.maximum.accumulate(np.where(cover & ~prev, pos, -1), axis=1)
    rid = np.full(cover.shape, -1, dtype=np.int64)
    rid[r_row, r_start] = np.arange(r_row.size)
    rid = np.maximum.accumulate(rid, axis=1)
    idx = np.where(cover, start_noise[np.maximum(rid, 0)] + pos - anchor, 0)
    noise = np.asarray(noise_wf, dtype=np.float64)
    return waveform + xp.asarray(np.where(cover, noise[idx], 0.0))


def adc_waveform(waveform, pedestal, cfg: DigiConfig, backend="numpy"):
    """Pedestal-subtracted, truncated and clipped ADC samples used for the peak search."""
    xp = _xp(backend)
    v = waveform - pedestal
    return xp.where(waveform > pedestal, xp.minimum(cfg.max_adc, xp.trunc(v)), 0).astype(xp.int64)


def first_peaks(wf, cfg: DigiConfig, backend="numpy"):
    """Return ``(first, run_end)``: the source's accepted peak samples and the end of their above-threshold run.

    A sample is a peak when it is at least ``minPeakADC``, strictly above the ``nBinsPeak`` samples before it and not
    below the ones after it, inside ``[nBinsPeak + bufferDigi, size - nBinsPeak)``. Only the first peak of each run
    of samples at or above threshold is taken, because the source jumps to the end of the run after a peak.
    """
    xp = _xp(backend)
    n, size = wf.shape
    nb = cfg.n_bins_peak
    thr = wf >= cfg.min_peak_adc
    pad = xp.concatenate([xp.full((n, nb), -1, dtype=wf.dtype), wf, xp.full((n, nb), -1, dtype=wf.dtype)], axis=1)
    cand = thr
    for d in range(1, nb + 1):
        cand = cand & (wf > pad[:, nb - d:nb - d + size]) & (wf >= pad[:, nb + d:nb + d + size])
    pos = xp.arange(size)
    cand = cand & (pos >= nb + cfg.buffer_digi) & (pos + nb < size)
    prev = xp.concatenate([xp.zeros((n, 1), dtype=bool), thr[:, :-1]], axis=1)
    run_start = thr & ~prev
    csum = xp.cumsum(cand.astype(xp.int64), axis=1)
    base = xp.where(run_start, csum - cand.astype(xp.int64), 0)
    base = _cummax(xp, base)
    first = cand & ((csum - base) == 1)
    nxt_below = xp.where(~thr, pos, size)
    run_end = _cummin_reverse(xp, nxt_below)
    return first, run_end


def _cummax(xp, a):
    if xp is np:
        return np.maximum.accumulate(a, axis=1)
    from jax import lax
    return lax.cummax(a, axis=1)


def _cummin_reverse(xp, a):
    if xp is np:
        return np.minimum.accumulate(a[:, ::-1], axis=1)[:, ::-1]
    from jax import lax
    return lax.cummin(a, axis=1, reverse=True)


def extract_digis(waveform, wf, first, run_end, cfg: DigiConfig):
    """Gather digis on the host: merge peaks whose buffered start overlaps the previous stop, then cut samples.

    Returns a dict with ``row``, ``t0``, ``peakpos``, ``offsets`` and flat ``samples`` (``int(waveform)`` clipped
    to the ADC range, without pedestal subtraction, as stored by the source).
    """
    waveform = np.asarray(waveform)
    first = np.asarray(first)
    run_end = np.asarray(run_end)
    p_row, p_pos = np.nonzero(first)
    start = p_pos - cfg.buffer_digi
    stop = run_end[p_row, p_pos]
    new = np.ones(p_row.size, dtype=bool)
    new[1:] = (p_row[1:] != p_row[:-1]) | (start[1:] > stop[:-1])
    gid = np.cumsum(new) - 1
    g_first = np.flatnonzero(new)
    g_last = np.r_[g_first[1:] - 1, p_row.size - 1].astype(np.int64)
    row, g_start, g_stop = p_row[g_first], start[g_first], stop[g_last]
    f = np.float32
    t0 = np.trunc(g_start.astype(f) * f(cfg.digi_sampling) + f(cfg.digitization_start)
                  - f(cfg.time_from_protons_to_dr_marker) - cfg.start_time_buffer).astype(np.int64)
    length = g_stop - g_start
    offsets = np.r_[0, np.cumsum(length)].astype(np.int64)
    member = np.repeat(np.arange(row.size), length)
    local = np.arange(offsets[-1]) - offsets[:-1][member]
    samples = np.minimum(np.trunc(waveform[row[member], g_start[member] + local]), cfg.max_adc).astype(np.int64)
    ok = (t0[member].astype(f) + local.astype(f) * f(cfg.digi_sampling) + f(cfg.time_from_protons_to_dr_marker)
          >= f(cfg.digitization_start))
    peakpos = _peak_positions(samples, ok, member, local, offsets, row.size)
    del gid
    return {"row": row, "t0": t0, "peakpos": peakpos, "offsets": offsets, "samples": samples}


def _peak_positions(samples, ok, member, local, offsets, n):
    """Source rule: start at 0, move to each eligible sample not below the current peak (ties go later)."""
    neg = np.iinfo(np.int64).min
    best = np.full(n, neg, dtype=np.int64)
    np.maximum.at(best, member, np.where(ok, samples, neg))
    head = samples[offsets[:-1].clip(max=max(samples.size - 1, 0))] if samples.size else np.zeros(n, np.int64)
    hit = ok & (samples == best[member]) & (best[member] >= head[member])
    pos = np.full(n, -1, dtype=np.int64)
    np.maximum.at(pos, member, np.where(hit, local, -1))
    return np.maximum(pos, 0)


def digitize(row, pe_time, pbtime, scale, n_rows, pulse, cfg=DigiConfig(), *, weight=None, noise=None,
             backend="numpy"):
    """Full stock digitisation for the given PEs. ``noise=(noise_wf, pedestal, offsets)`` enables noise."""
    time32 = sample_times(pe_time, pbtime, cfg)
    wave = build_waveforms(row, time32, scale, n_rows, pulse, cfg, weight=weight, backend=backend)
    pedestal = 0.0
    if noise is not None:
        noise_wf, pedestal, offsets = noise
        wave = add_noise(wave, noise_wf, offsets, cfg, backend)
    wf = adc_waveform(wave, pedestal, cfg, backend)
    first, run_end = first_peaks(wf, cfg, backend)
    return extract_digis(wave, wf, first, run_end, cfg)


def noise_pedestal(noise_wf):
    """``CaloNoiseUtil::fillCache`` pedestal: truncated mean of the stored noise histogram."""
    noise = np.asarray(noise_wf, dtype=np.float32).astype(np.float64)
    return float(np.trunc(noise.sum() / noise.size))
