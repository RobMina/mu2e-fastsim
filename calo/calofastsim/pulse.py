"""Stock calorimeter pulse cache (Offline Mu2eUtilities/src/CaloPulseUtil.cc), NumPy, exact.

The cache is the shape histogram re-binned to digiSampling/100 with TH1::Interpolate, stored as TH1F (float32),
scaled to unit maximum, and prefixed with 100 zeros. ``digitized_pulse`` reads every 100th fine value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_STEPS = 100


def _th1_interpolate(contents, xmin, xmax, x):
    """TH1::Interpolate for a fixed-bin histogram, vectorised over x (double arithmetic)."""
    contents = np.asarray(contents, dtype=np.float32).astype(np.float64)
    n = contents.size
    width = (xmax - xmin) / float(n)

    def center(b):
        return xmin + (b - 1) * width + 0.5 * width

    def content(b):
        return contents[np.clip(b, 1, n) - 1]

    x = np.asarray(x, dtype=np.float64)
    xbin = 1 + (n * (x - xmin) / (xmax - xmin)).astype(np.int64)
    low = xbin.copy()
    left = x <= center(xbin)
    low[left] = xbin[left] - 1
    x0, x1 = center(low), center(low + 1)
    y0, y1 = content(low), content(low + 1)
    value = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
    value = np.where(x <= center(1), contents[0], value)
    return np.where(x >= center(n), contents[-1], value)


@dataclass(frozen=True)
class PulseCache:
    pulse_vec: np.ndarray   # float64 copy of float32 fine contents, length nbins + N_STEPS
    digi_step: float        # fine step (ns)
    n_bin_shape: int        # samples in a digitized pulse
    delta_t: float          # peak time minus t0 of the digitized pulse

    @classmethod
    def from_histogram(cls, contents, xmin, xmax, digi_sampling):
        digi_step = digi_sampling / float(N_STEPS)
        span = xmax - xmin
        nbins = int(span / digi_step)
        centers = 0.0 + (np.arange(1, nbins + 1) - 1) * (span / nbins) + 0.5 * (span / nbins)
        fine = _th1_interpolate(contents, xmin, xmax, centers).astype(np.float32)
        scale = 1.0 / float(fine.max())
        fine = (fine.astype(np.float64) * scale).astype(np.float32)
        pulse_vec = np.concatenate([np.zeros(N_STEPS), fine.astype(np.float64)])
        n_bin_shape = nbins // N_STEPS
        delta_t = 0.0
        for i in range(1, n_bin_shape):
            if pulse_vec[(i + 1) * N_STEPS] < pulse_vec[i * N_STEPS]:
                break
            delta_t += N_STEPS * digi_step
        return cls(pulse_vec, digi_step, n_bin_shape, delta_t)

    def shift_bin(self, hit_time):
        """First fine index of ``digitizedPulse(hitTime)`` (C++ int truncation and %)."""
        q = np.trunc(np.asarray(hit_time, dtype=np.float64) / self.digi_step).astype(np.int64)
        return N_STEPS - np.fmod(q, N_STEPS)

    def digitized_pulse(self, hit_time):
        """Return ``(..., n_bin_shape)`` float64 digitized pulses for the given hit times."""
        idx = self.shift_bin(hit_time)[..., None] + N_STEPS * np.arange(self.n_bin_shape)
        return self.pulse_vec[idx]

    def evaluate(self, dt):
        """``CaloPulseUtil::evaluate``: linear interpolation of the fine cache at ``dt + delta_t``."""
        t = np.asarray(dt, dtype=np.float64) + self.delta_t
        ibin = N_STEPS + np.trunc(t * N_STEPS / self.digi_step / N_STEPS).astype(np.int64)
        ok = (ibin >= 0) & (ibin < self.pulse_vec.size - 1)
        ib = np.clip(ibin, 0, self.pulse_vec.size - 2)
        t0bin = (ib - N_STEPS) * self.digi_step
        value = (self.pulse_vec[ib + 1] - self.pulse_vec[ib]) / self.digi_step * (t - t0bin) + self.pulse_vec[ib]
        return np.where(ok, value, 0.0)

    def from_peak_to_t0(self, time_peak):
        return np.asarray(time_peak, dtype=np.float64) - self.delta_t - 0.5 * self.digi_step * N_STEPS
