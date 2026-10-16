"""JAX twin of waveform.synthesize: CRV waveforms and zero-suppressed digis from SiPM charges, fixed shapes, one jit.

Host side (cheap, numpy): charges are grouped into the stock charge clusters (waveform.charge_clusters) and packed
into a [C, Q] matrix of (time, charge) per cluster, padded; cluster start times follow the stock rule (first FEB tick
after first charge - 12.5 ns). Device side (this module): every charge adds charge/refCharge * template(t_sample -
t_charge) to the L samples of its cluster (a dense [C, Q, L] gather from the 1PE template, summed over Q), optional
Gaussian noise, the minVoltage zero-suppression window, and ADC conversion. The digi split (12-sample digis starting
where the window opens, scan resumes after each digi) is a lax.scan over samples. Outputs are dense [C, L] samples,
ADCs and digi-start flags; `to_digis` turns them into waveform.synthesize's per-digi dict on the host.
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from .pulsefit import PERIOD
from .waveform import WaveformConfig, charge_clusters, single_pe_template

jax.config.update("jax_enable_x64", True)


def pack(chan, time, charge, tdc0_of_chan, cfg=WaveformConfig(), Q=None, L=None):
    """Host packing: clusters -> padded (t, q) matrices, start times and keep flags. Returns a dict of numpy arrays."""
    tmpl = single_pe_template(cfg.single_pe_file, cfg.single_pe_precision, cfg.single_pe_stretch,
                              cfg.single_pe_max_time)
    cid = charge_clusters(chan, time, cfg)
    C = int(cid[-1]) + 1 if len(cid) else 0
    first = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]]) if C else np.zeros(0, np.int64)
    nq = np.bincount(cid, minlength=C)
    pos = np.arange(len(cid)) - first[cid]
    Qm = int(nq.max()) if C else 1
    Q = Q or Qm
    if Qm > Q:
        raise ValueError(f"cluster with {Qm} charges > Q={Q}")
    cchan = chan[first]
    tdc0 = tdc0_of_chan(cchan) if callable(tdc0_of_chan) else np.asarray(tdc0_of_chan)[cchan]
    t_start = np.ceil((time[first] - PERIOD - tdc0) / PERIOD) * PERIOD + tdc0
    span = np.zeros(C)
    np.maximum.at(span, cid, time - time[first][cid])
    need = int(np.ceil((span.max() if C else 0) + cfg.single_pe_max_time) / PERIOD) + 2 + cfg.samples_zs if C else 1
    L = L or need
    T = np.zeros((C, Q))
    W = np.zeros((C, Q))
    T[cid, pos] = time
    W[cid, pos] = charge / cfg.single_pe_reference_charge
    keep = nq * tmpl.max() >= cfg.min_voltage
    est = np.floor((span + cfg.single_pe_max_time) / PERIOD).astype(np.int64)
    return dict(T=T, W=W, n=nq, t_start=t_start, tdc0=tdc0, keep=keep, chan=cchan, est=est, L=L, tmpl=tmpl)


@partial(jax.jit, static_argnames=("L", "zs", "precision", "noise"))
def synth_dense(T, W, n, t_start, keep, tmpl, key, L, zs, precision, noise, min_voltage, adc_conv, pedestal, adc_lo,
                adc_hi):
    """Dense waveforms, ADCs and digi starts for C clusters. T, W [C,Q]; n, t_start, keep [C]."""
    C, Q = T.shape
    qmask = jnp.arange(Q)[None, :] < n[:, None]
    ts = t_start[:, None] + jnp.arange(L)[None, :] * PERIOD                       # [C, L]
    dt = ts[:, None, :] - T[:, :, None]                                           # [C, Q, L]
    ti = jnp.rint(dt / precision).astype(jnp.int64)
    ok = (dt >= 0) & (ti < tmpl.shape[0]) & qmask[:, :, None]
    contrib = jnp.where(ok, tmpl[jnp.clip(ti, 0, tmpl.shape[0] - 1)] * W[:, :, None], 0.0)
    wave = contrib.sum(1)
    last = jnp.where(ok, jnp.arange(L)[None, None, :] + 1, 0).max((1, 2))
    if noise > 0:
        wave = wave + noise * jax.random.normal(key, wave.shape) * (jnp.arange(L)[None, :] < last[:, None])
    valid = jnp.arange(L)[None, :] < last[:, None]
    above = (wave > min_voltage) & valid
    pad = jnp.zeros((C, 2), bool)
    ap = jnp.concatenate([pad, above, pad], 1)
    win = ap[:, 0:L] | ap[:, 1:L + 1] | ap[:, 2:L + 2] | ap[:, 3:L + 3] | ap[:, 4:L + 4]
    win = win & valid & keep[:, None]

    def step(next_i, i):
        start = win[:, i] & (next_i <= i)
        return jnp.where(start, i + zs, next_i), start

    _, starts = lax.scan(step, jnp.zeros(C, jnp.int64), jnp.arange(L))
    starts = starts.T                                                              # [C, L]
    adc = jnp.clip((wave * adc_conv + pedestal + 0.5).astype(jnp.int16), adc_lo, adc_hi)
    return wave, adc, starts, last


def synthesize(chan, time, charge, tdc0_of_chan, cfg=WaveformConfig(), window=(0.0, 1e5), rng_key=None, Q=None,
               L=None):
    """Same contract as waveform.synthesize (noise from a JAX key; None = no noise)."""
    pk = pack(chan, time, charge, tdc0_of_chan, cfg, Q, L)
    key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
    wave, adc, starts, last = synth_dense(pk["T"], pk["W"], pk["n"], pk["t_start"], pk["keep"], pk["tmpl"], key,
                                          pk["L"], cfg.samples_zs, cfg.single_pe_precision,
                                          cfg.noise if rng_key is not None else 0.0, cfg.min_voltage,
                                          cfg.adc_conversion, cfg.pedestal, cfg.min_adc, cfg.max_adc)
    return to_digis(pk, np.asarray(wave), np.asarray(adc), np.asarray(starts), np.asarray(last), cfg, window)


def to_digis(pk, wave, adc, starts, last, cfg, window):
    """Dense device outputs -> per-digi dict (host), digis outside the digitisation window dropped as stock."""
    r, i0 = np.nonzero(starts)
    tstart = pk["t_start"][r] + i0 * PERIOD
    inwin = (tstart >= window[0]) & (tstart <= window[1])
    r, i0, tstart = r[inwin], i0[inwin], tstart[inwin]
    L = wave.shape[1]
    length = np.maximum(pk["est"], last)
    cols = i0[:, None] + np.arange(cfg.samples_zs)[None, :]
    ok = cols < np.minimum(length[r, None], L)
    volt = np.where(ok, wave[r[:, None], np.minimum(cols, L - 1)], 0.0)
    adcd = np.clip((volt * cfg.adc_conversion + cfg.pedestal + 0.5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    tdc0 = pk["tdc0"][r]
    order = np.lexsort((tstart, pk["chan"][r]))
    return {"chan": pk["chan"][r][order], "start": tstart[order], "tdc0": tdc0[order],
            "starttdc": np.rint((tstart - tdc0) / PERIOD).astype(np.int64)[order], "volt": volt[order],
            "adc": adcd[order]}
