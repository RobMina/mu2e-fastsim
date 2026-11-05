"""Template fit on synthetic digis: single and double pulses, baseline, chi2 cut inputs."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from calofastsim.pulse import PulseCache  # noqa: E402
from calofastsim.templatefit import FitConfig, fit_digis  # noqa: E402


@pytest.fixture(scope="module")
def pulse():
    x = (np.arange(6200) + 0.5) * 0.05
    return PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)


def _digi(pulse, t0, length, peaks, base, rng, sigma=1.0):
    x = t0 + (np.arange(length) + 0.5) * 5.0
    y = base + sum(a * pulse.evaluate(x - t) for a, t in peaks)
    return np.round(y + rng.normal(0.0, sigma, length)).astype(np.int64)


def test_single_and_double_pulses(pulse):
    rng = np.random.default_rng(1)
    cases = [(500, 40, [(300.0, 620.0)]), (800, 50, [(1200.0, 905.0)]),
             (500, 60, [(400.0, 600.0), (250.0, 700.0)])]
    samples = np.zeros((len(cases), 64), np.int64)
    t0 = np.array([c[0] for c in cases])
    length = np.array([c[1] for c in cases])
    for i, (t, n, peaks) in enumerate(cases):
        samples[i, :n] = _digi(pulse, t, n, peaks, 12.0, rng)
    fit = fit_digis(t0, samples, length, pulse, FitConfig())
    for i, (_, _, peaks) in enumerate(cases):
        amp = fit["amp"][i][fit["valid"][i]]
        tim = fit["peak_time"][i][fit["valid"][i]]
        assert amp.size == len(peaks)
        order = np.argsort(tim)
        for (a, t), fa, ft in zip(sorted(peaks, key=lambda p: p[1]), amp[order], tim[order]):
            assert abs(fa - a) < 0.03 * a + 3.0
            assert abs(ft - t) < 0.3
        assert 10.0 < fit["baseline"][i] < 14.0
        assert fit["ndf"][i] == length[i] - 1 - 2 * len(peaks)
    np.testing.assert_allclose(fit["time"], fit["peak_time"] - pulse.delta_t - 2.5)


def test_flat_digi_has_no_peak(pulse):
    samples = np.full((1, 30), 12, np.int64)
    fit = fit_digis(np.array([500]), samples, np.array([30]), pulse)
    assert not fit["valid"].any()
