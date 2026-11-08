"""Zero-response digitisation preserves the output schema and unused noise draws."""
import numpy as np
import pytest

from calofastsim.digi import DigiConfig, add_noise, adc_waveform, digitize, extract_digis, first_peaks
from calofastsim.pulse import PulseCache


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        previous = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        try:
            yield request.param
        finally:
            jax.config.update("jax_enable_x64", previous)
    else:
        yield request.param


@pytest.fixture(scope="module")
def pulse():
    x = (np.arange(6200) + 0.5) * 0.05
    return PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)


def _assert_empty(digis):
    assert set(digis) == {"row", "t0", "peakpos", "offsets", "samples"}
    for name, values in digis.items():
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.int64
        assert values.shape == ((1,) if name == "offsets" else (0,))
    np.testing.assert_array_equal(digis["offsets"], [0])


@pytest.mark.pr_params("jax-covered-True")
@pytest.mark.parametrize("with_noise", [False, True])
@pytest.mark.parametrize("case", ["empty_rows", "no_pes", "zero_scale", "subthreshold", "covered", "late_pes"])
def test_digitize_no_accepted_peaks(pulse, backend, case, with_noise):
    n_rows = 0 if case == "empty_rows" else 2
    row = np.empty(0, np.int64) if case in ("empty_rows", "no_pes") else np.arange(n_rows)
    times = np.full(row.size, 3000.0 if case == "late_pes" else 700.0, np.float32)
    scale = np.full(row.size, {"zero_scale": 0.0, "covered": 8.0, "late_pes": 100.0}.get(case, 0.5))
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    draws = []

    def offsets(length):
        assert length.size > 0
        draws.append(length.copy())
        return np.trunc(rng.random(length.size) * (10000 - length)).astype(np.int64)

    noise = (np.full(10000, 12.0), 12.0, offsets) if with_noise else None
    result = digitize(row, times, -200.0, scale, n_rows, pulse, noise=noise,
                      marker=np.full(row.size, 200.0, np.float32), backend=backend)
    _assert_empty(result)
    if with_noise and case == "covered":
        assert len(draws) == 1 and draws[0].size == n_rows
    else:
        assert not draws
        assert rng.bit_generator.state == state


@pytest.mark.pr_params("jax-2-0.5-float32", "jax-2-0.5-float64")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_rows, amplitude", [(0, 0.0), (2, 0.0), (2, 0.5)])
def test_no_noise_coverage_preserves_waveform(backend, dtype, n_rows, amplitude):
    cfg = DigiConfig()
    waveform = np.full((n_rows, cfg.waveform_size), amplitude, dtype=dtype)
    if backend == "jax":
        import jax.numpy as jnp
        waveform = jnp.asarray(waveform)

    def offsets(length):
        pytest.fail("no covered ranges must not request noise offsets")

    result = add_noise(waveform, np.full(10000, 12.0), offsets, cfg, backend)
    assert result is waveform
    assert result.shape == (n_rows, cfg.waveform_size)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, np.full(result.shape, amplitude, dtype=dtype))


def test_all_peaks_outside_search_window(backend):
    cfg = DigiConfig()
    waveform = np.zeros((2, cfg.waveform_size), np.float64)
    waveform[0, cfg.buffer_digi] = 100.0
    waveform[1, -1] = 100.0
    if backend == "jax":
        import jax.numpy as jnp
        waveform = jnp.asarray(waveform)
    wf = adc_waveform(waveform, 0.0, cfg, backend)
    assert np.asarray(wf).max() > cfg.min_peak_adc
    first, run_end = first_peaks(wf, cfg, backend)
    assert not np.asarray(first).any()
    _assert_empty(extract_digis(waveform, wf, first, run_end, cfg))
