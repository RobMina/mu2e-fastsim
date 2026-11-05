"""The vectorised digitizer against a literal sequential transcription of CaloDigiMaker/CaloPulseUtil."""
import math

import numpy as np
import pytest

from calofastsim.digi import DigiConfig, digitize, sample_times, build_waveforms, noise_pedestal
from calofastsim.pulse import N_STEPS, PulseCache, _th1_interpolate


def _shape():
    x = (np.arange(6200) + 0.5) * 0.05
    return (x / 20.0) ** 2 * np.exp(-x / 10.0)


@pytest.fixture(scope="module")
def pulse():
    return PulseCache.from_histogram(_shape(), 0.0, 310.0, 5.0)


def _reference_digis(ro_times, scale, pb, pulse, cfg, noise=None):
    """Literal per-readout transcription of fillROHits, addWFNoise, buildOutputDigi and extract."""
    size = cfg.waveform_size
    wave = [0.0] * size
    f = np.float32
    for t_pe in ro_times:
        t = f(f(f(f(f(t_pe) + f(pb)) - f(cfg.digitization_start)) + f(cfg.time_from_protons_to_dr_marker))
              + cfg.start_time_buffer)
        start = int(t / f(cfg.digi_sampling))   # C++ unsigned(): truncation toward zero
        if start < 0:
            continue
        shift = N_STEPS - int(math.fmod(int(float(t) / pulse.digi_step), N_STEPS))
        shape = [pulse.pulse_vec[shift + i * N_STEPS] for i in range(pulse.n_bin_shape)]
        for s in range(start, min(start + len(shape), size)):
            wave[s] += shape[s - start] * scale
    pedestal = 0.0
    if noise is not None:
        noise_wf, pedestal, draws = noise
        starts, stops, ts = [], [], 0
        while ts < size:
            if wave[ts] < cfg.min_noise_amp:
                ts += 1
                continue
            s0 = ts - cfg.buffer_digi if ts > cfg.buffer_digi else 0
            s1 = ts
            while s1 < size and wave[s1] > cfg.min_noise_amp:
                s1 += 1
            starts.append(s0)
            stops.append(s1)
            ts = s1 + 1
        keep = _merge(starts, stops, size)
        for (s0, s1), u in zip(keep, draws(np.array([b - a for a, b in keep]))):
            for i in range(s1 - s0):
                wave[s0 + i] += noise_wf[u + i]
    mx = cfg.max_adc
    wf = [min(mx, int(v - pedestal)) if v > pedestal else 0 for v in wave]
    starts, stops, ts = [], [], cfg.n_bins_peak + cfg.buffer_digi
    while ts + cfg.n_bins_peak < size:
        if wf[ts] < cfg.min_peak_adc:
            ts += 1
            continue
        imax = ts - cfg.n_bins_peak
        for i in range(ts - cfg.n_bins_peak, ts + cfg.n_bins_peak + 1):
            if wf[i] > wf[imax]:
                imax = i
        if ts != imax:
            ts += 1
            continue
        s0 = ts - cfg.buffer_digi if ts > cfg.buffer_digi else 0
        s1 = ts + 1
        while s1 < size and wf[s1] >= cfg.min_peak_adc:
            s1 += 1
        starts.append(s0)
        stops.append(s1)
        ts = s1 + 1
    digis = []
    for s0, s1 in _merge(starts, stops, size):
        t0 = int(f(f(f(f(s0) * f(cfg.digi_sampling) + f(cfg.digitization_start))
                     - f(cfg.time_from_protons_to_dr_marker)) - cfg.start_time_buffer))
        sample = [min(int(wave[i]), mx) for i in range(s0, s1)]
        peak = 0
        for i in range(len(sample)):
            if (f(t0) + f(i) * f(cfg.digi_sampling) + f(cfg.time_from_protons_to_dr_marker)
                    >= f(cfg.digitization_start) and sample[i] >= sample[peak]):
                peak = i
        digis.append((t0, peak, sample))
    return digis


def _merge(starts, stops, size):
    starts, stops = list(starts), list(stops)
    iprev, ic = 0, 1
    while ic < len(starts):
        if stops[iprev] >= starts[ic]:
            stops[iprev] = stops[ic]
            starts[ic] = stops[ic] = size + 1
        else:
            iprev = ic
        ic += 1
    return [(a, b) for a, b in zip(starts, stops) if a <= size]


def _event(rng, n_ro):
    ros = []
    for _ in range(n_ro):
        n_pulse = rng.integers(1, 4)
        times = []
        for _ in range(n_pulse):
            t = rng.uniform(260.0, 1700.0)
            times += [t] * int(rng.integers(1, 400))
        ros.append(np.array(times, dtype=np.float32))
    return ros


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_noise_off_matches_reference(pulse, backend):
    if backend == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(3)
    cfg = DigiConfig()
    ros = _event(rng, 40)
    pb, scale = -210.0, float(np.float32(16) / np.float32(30))
    row = np.concatenate([np.full(r.size, i) for i, r in enumerate(ros)])
    times = np.concatenate(ros)
    got = digitize(row, times, pb, np.full(times.size, scale), len(ros), pulse, cfg, backend=backend)
    n_match = 0
    for i, r in enumerate(ros):
        ref = _reference_digis(r, scale, pb, pulse, cfg)
        sel = np.flatnonzero(got["row"] == i)
        assert len(sel) == len(ref)
        for g, (t0, peak, sample) in zip(sel, ref):
            a, b = got["offsets"][g], got["offsets"][g + 1]
            assert got["t0"][g] == t0 and got["peakpos"][g] == peak
            assert np.array_equal(got["samples"][a:b], sample)
            n_match += 1
    assert n_match > 20


def test_noise_on_matches_reference_with_shared_draws(pulse):
    rng = np.random.default_rng(5)
    cfg = DigiConfig()
    noise_wf = rng.uniform(5.0, 20.0, 10000).astype(np.float32).astype(np.float64)
    ped = noise_pedestal(noise_wf)
    ros = _event(rng, 15)
    scale = float(np.float32(16) / np.float32(30))
    for i, r in enumerate(ros):
        seed = 100 + i

        def draws(length, seed=seed):
            g = np.random.default_rng(seed)
            return np.trunc(g.uniform(0.0, 1.0, len(length)) * (10000 - np.asarray(length))).astype(np.int64)

        ref = _reference_digis(r, scale, -200.0, pulse, cfg, noise=(noise_wf, ped, draws))
        got = digitize(np.zeros(r.size, np.int64), r, -200.0, np.full(r.size, scale), 1, pulse, cfg,
                       noise=(noise_wf, ped, draws))
        assert len(got["t0"]) == len(ref)
        for g, (t0, peak, sample) in enumerate(ref):
            a, b = got["offsets"][g], got["offsets"][g + 1]
            assert got["t0"][g] == t0 and got["peakpos"][g] == peak
            assert np.array_equal(got["samples"][a:b], sample)


def test_pulse_cache_shape(pulse):
    assert pulse.pulse_vec.size == 6200 + N_STEPS and pulse.n_bin_shape == 62
    assert np.all(pulse.pulse_vec[:N_STEPS] == 0) and math.isclose(pulse.pulse_vec.max(), 1.0, rel_tol=1e-7)
    assert pulse.delta_t > 0
    np.testing.assert_allclose(pulse.evaluate(0.0), pulse.pulse_vec[N_STEPS + int(pulse.delta_t / pulse.digi_step)])


def test_negative_digitizer_time(pulse):
    """t <= -digiSampling adds nothing; -digiSampling < t < 0 starts at sample 0, as the source's unsigned cast."""
    cfg = DigiConfig()
    w = build_waveforms(np.array([0]), np.array([-5.0], np.float32), np.array([1.0]), 1, pulse, cfg)
    assert not w.any()
    t = np.float32(-2.5)
    w = build_waveforms(np.array([0]), np.array([t]), np.array([1.0]), 1, pulse, cfg)
    shift = N_STEPS - int(math.fmod(int(float(t) / pulse.digi_step), N_STEPS))
    np.testing.assert_array_equal(w[0, :pulse.n_bin_shape], pulse.pulse_vec[shift + N_STEPS * np.arange(pulse.n_bin_shape)])


def test_weights_equal_repeated_pes(pulse):
    cfg = DigiConfig()
    times = np.array([300.0, 300.0, 300.0, 712.3, 712.3], np.float32)
    rep = build_waveforms(np.zeros(5, np.int64), times, np.full(5, 0.5), 1, pulse, cfg)
    grp = build_waveforms(np.zeros(2, np.int64), times[[0, 3]], np.full(2, 0.5), 1, pulse, cfg, weight=[3, 2])
    np.testing.assert_allclose(grp, rep, rtol=1e-15, atol=0)


def test_th1_interpolate_matches_linear_interpolation():
    rng = np.random.default_rng(2)
    contents = rng.uniform(0, 1, 50).astype(np.float32)
    centers = (np.arange(50) + 0.5) * 0.2
    x = rng.uniform(-1.0, 11.0, 500)
    np.testing.assert_allclose(_th1_interpolate(contents, 0.0, 10.0, x),
                               np.interp(x, centers, contents.astype(np.float64)), rtol=1e-12, atol=1e-12)


def test_offspill_waveform_size_and_x64_guard():
    assert DigiConfig().waveform_size == 285
    assert DigiConfig(event_length=100000.0).waveform_size == int((100000.0 - 450.0 + 80.0) / 5.0)
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="x64"):
            digitize(np.zeros(1, np.int64), np.array([500.0]), 0.0, np.ones(1), 1, None, backend="jax")
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_pedestal_is_sequential_truncated_mean():
    assert noise_pedestal(np.full(10000, 12.6)) == 12.0
