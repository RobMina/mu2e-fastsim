"""On-device variant of :func:`calofastsim.digi.digitize`: waveform build, noise, ADC and peak search in one jitted
JAX function; only the variable-length digi gather stays on the host.

Same source rules as :mod:`calofastsim.digi`. Differences, by design:
- PEs are scattered one pulse sample at a time without first merging identical pulses (double sums in another
  order; with noise off the digis still match the host path and stock to the ADC count except at roundoff ties);
- noise offsets come from ``jax.random`` (uniform, as ``CaloNoiseUtil::noiseSegment``), so with noise on only
  distributions agree with the host path and with stock.
Shapes are padded to power-of-two buckets so warm calls reuse one compilation. JAX x64 is required.
"""
from __future__ import annotations

import numpy as np

from .digi import DigiConfig, extract_digis, noise_pedestal
from .pulse import N_STEPS, PulseCache

_CACHE = {}


def _bucket(n, floor):
    return max(floor, 1 << int(np.ceil(np.log2(max(n, 1)))))


def _kernel(pulse: PulseCache, cfg: DigiConfig, n_pe, n_rows, noise_size):
    import jax
    import jax.numpy as jnp
    from jax import lax

    from .digi import adc_waveform, first_peaks, noise_coverage

    size = cfg.waveform_size
    vec = jnp.asarray(pulse.pulse_vec)
    nshape = pulse.n_bin_shape
    f32 = jnp.float32

    def run(row, time32, weight, noise_wf, pedestal, key, add_noise):
        start = jnp.trunc(time32 / f32(cfg.digi_sampling)).astype(jnp.int64)
        keep = start >= 0
        q = jnp.trunc(time32.astype(jnp.float64) / pulse.digi_step).astype(jnp.int64)
        first = N_STEPS - jnp.fmod(q, N_STEPS)
        w = jnp.where(keep, weight, 0.0)
        flat0 = row * size + start

        def add_sample(k, out):
            sample = start + k
            ok = keep & (sample < size)
            idx = jnp.where(ok, flat0 + k, n_rows * size)
            return out.at[idx].add(jnp.where(ok, vec[jnp.clip(first + N_STEPS * k, 0, vec.size - 1)] * w, 0.0))

        wave = lax.fori_loop(0, nshape, add_sample, jnp.zeros(n_rows * size + 1, jnp.float64))
        wave = wave[:-1].reshape(n_rows, size)

        cover = noise_coverage(wave, cfg, backend="jax")
        pos = jnp.arange(size)
        prev = jnp.concatenate([jnp.zeros((n_rows, 1), bool), cover[:, :-1]], axis=1)
        nxt = jnp.concatenate([cover[:, 1:], jnp.zeros((n_rows, 1), bool)], axis=1)
        starts = cover & ~prev
        last = lax.cummin(jnp.where(cover & ~nxt, pos, size), axis=1, reverse=True)
        length = last - pos + 1
        u = jax.random.uniform(key, (n_rows, size), dtype=jnp.float64)
        offset = jnp.trunc(u * (noise_size - length)).astype(jnp.int64)
        anchor = lax.cummax(jnp.where(starts, pos, -1), axis=1)
        a = jnp.maximum(anchor, 0)
        idx = jnp.take_along_axis(offset, a, axis=1) + pos - a
        noisy = wave + jnp.where(cover, noise_wf[jnp.clip(idx, 0, noise_size - 1)], 0.0)
        wave = jnp.where(add_noise, noisy, wave)
        ped = jnp.where(add_noise, pedestal, 0.0)
        wf = adc_waveform(wave, ped, cfg, backend="jax")
        peaks, run_end = first_peaks(wf, cfg, backend="jax")
        stored = jnp.minimum(jnp.trunc(wave), cfg.max_adc).astype(jnp.int32)   # what the digi keeps
        return stored, peaks, run_end.astype(jnp.int32)

    return jax.jit(run)


def digitize_device(row, pe_time, pbtime, scale, n_rows, pulse: PulseCache, cfg: DigiConfig = DigiConfig(), *,
                    weight=None, noise_wf=None, key=None, marker=None):
    """Device digitisation of the given PEs (``noise_wf`` and a ``jax.random`` ``key`` enable noise).

    Returns the same dict as :func:`calofastsim.digi.digitize`.
    """
    import jax
    import jax.numpy as jnp
    from .digi import sample_times
    if not jax.config.x64_enabled:
        raise ValueError("the double-precision waveform requires caller-enabled JAX x64")
    time32 = sample_times(pe_time, pbtime, cfg, marker)
    n_pe = time32.size
    w = (np.ones(n_pe) if weight is None else np.asarray(weight, np.float64)) * np.asarray(scale, np.float64)
    pe_pad, row_pad = _bucket(n_pe, 1 << 16), _bucket(n_rows, 1 << 10)
    noise = np.zeros(1) if noise_wf is None else np.asarray(noise_wf, np.float64)
    ck = (id(pulse), cfg, pe_pad, row_pad, noise.size)
    if ck not in _CACHE:
        _CACHE[ck] = (_kernel(pulse, cfg, pe_pad, row_pad, noise.size), pulse)
    kernel = _CACHE[ck][0]
    r = np.full(pe_pad, row_pad - 1, np.int64)
    r[:n_pe] = row
    t = np.full(pe_pad, -1e6, np.float32)   # padding PEs start before the window and add nothing
    t[:n_pe] = time32
    ww = np.zeros(pe_pad)
    ww[:n_pe] = w
    add = noise_wf is not None
    key = jax.random.PRNGKey(0) if key is None else key
    stored, peaks, run_end = kernel(jnp.asarray(r), jnp.asarray(t), jnp.asarray(ww), jnp.asarray(noise),
                                    noise_pedestal(noise) if add else 0.0, key, add)
    stored = np.asarray(stored[:n_rows])
    peaks, run_end = np.asarray(peaks[:n_rows]), np.asarray(run_end[:n_rows])
    row_marker = None
    if marker is not None:
        row_marker = np.full(n_rows, cfg.time_from_protons_to_dr_marker, np.float32)
        row_marker[np.asarray(row)] = np.broadcast_to(np.asarray(marker, np.float32), np.shape(row))
    return extract_digis(stored, None, peaks, run_end, cfg, row_marker)
