"""On-device digitisation against the host path: identical digis with noise off, same distributions with noise on."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from calofastsim.digi import DigiConfig, digitize, noise_pedestal  # noqa: E402
from calofastsim.digi_device import digitize_device  # noqa: E402
from calofastsim.pulse import PulseCache  # noqa: E402


@pytest.fixture(scope="module")
def pulse():
    x = (np.arange(6200) + 0.5) * 0.05
    return PulseCache.from_histogram((x / 20.0) ** 2 * np.exp(-x / 10.0), 0.0, 310.0, 5.0)


def _pes(rng, n_rows):
    rows, times, weights = [], [], []
    for r in range(n_rows):
        for _ in range(int(rng.integers(1, 4))):
            t = rng.uniform(240.0, 1700.0)
            n = int(rng.integers(1, 300))
            rows += [r] * n
            times += list(t + rng.exponential(3.0, n))
            weights += [1] * n
    return np.array(rows), np.array(times, np.float32), np.array(weights)


def test_noise_off_matches_host(pulse):
    rng = np.random.default_rng(4)
    row, t, w = _pes(rng, 60)
    scale = np.full(t.size, float(np.float32(16) / np.float32(30)))
    host = digitize(row, t, -210.0, scale, 60, pulse, DigiConfig(), weight=w, marker=200.0)
    dev = digitize_device(row, t, -210.0, scale, 60, pulse, DigiConfig(), weight=w, marker=200.0)
    for k in ("row", "t0", "peakpos", "offsets", "samples"):
        np.testing.assert_array_equal(dev[k], host[k])


def test_noise_on_distributions(pulse):
    rng = np.random.default_rng(5)
    noise = rng.normal(12.5, 2.0, 10000)
    row, t, w = _pes(rng, 400)
    scale = np.full(t.size, float(np.float32(16) / np.float32(30)))

    def offsets(length):
        return np.trunc(rng.random(len(length)) * (noise.size - np.asarray(length))).astype(np.int64)

    host = digitize(row, t, -200.0, scale, 400, pulse, DigiConfig(), weight=w,
                    noise=(noise, noise_pedestal(noise), offsets))
    dev = digitize_device(row, t, -200.0, scale, 400, pulse, DigiConfig(), weight=w, noise_wf=noise,
                          key=jax.random.PRNGKey(3))
    assert abs(dev["t0"].size - host["t0"].size) <= 0.03 * host["t0"].size
    peak_h = np.maximum.reduceat(host["samples"], host["offsets"][:-1])
    peak_d = np.maximum.reduceat(dev["samples"], dev["offsets"][:-1])
    assert abs(np.median(peak_d) - np.median(peak_h)) <= 0.02 * np.median(peak_h)
    assert abs(np.mean(np.diff(dev["offsets"])) - np.mean(np.diff(host["offsets"]))) < 1.0


def test_empty_input(pulse):
    out = digitize_device(np.zeros(0, np.int64), np.zeros(0, np.float32), 0.0, np.zeros(0), 1, pulse)
    assert out["t0"].size == 0 and list(out["offsets"]) == [0]
