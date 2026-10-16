"""Synthetic CRV waveforms with KPP-data-like statistics, for benchmarks where the reference tables are not available
(Polaris: the Mu2e-internal reference data stay off the public repo and off ALCF).

Matched to the KPP run 124155 reference (2026-10-16): 20 waveforms per event; 99.6 % have 12 samples and 0.4 % 24;
56 % carry one pulse and 44 % two (the stock reco on data); first-pulse PE 4.6 / 16.7 / 57 / 129 at the 10/50/90/99 %
quantiles; beta 12.4 / 15.0 / 18.2 ns; pedestal ~2046 ADC with 2 ADC noise; pulse-area calibration ~500 ADC*ns/PE.
Only shapes and rates matter for timing; the pulse content is drawn, not taken from data.
"""
import numpy as np

from .pulsefit import PERIOD, gumbel

WF_PER_EVENT = 20
CALIB_AREA = 500.0
PEDESTAL = 2046.0


def waveforms(n_events, seed=1, L=12):
    """adc [W,L], n, start_tdc, pedestal, calib_area, calib_height for n_events events (fixed L; 24-sample waveforms
    are folded into L=12 batches, which is what dominates the cost)."""
    rng = np.random.default_rng(seed)
    W = n_events * WF_PER_EVENT
    start = rng.integers(10, 8000, W)
    t = (start[:, None] + np.arange(L)[None, :]) * PERIOD
    pe = np.exp(rng.normal(np.log(16.7), 1.0, W))
    beta = rng.normal(15.0, 2.2, W).clip(8, 30)
    A = pe * CALIB_AREA / beta
    mu = (start + 3) * PERIOD + rng.uniform(-6, 6, W)
    y = gumbel(t, A[:, None], mu[:, None], beta[:, None])
    two = rng.random(W) < 0.44
    pe2 = np.exp(rng.normal(np.log(6.0), 0.8, W))
    b2 = rng.normal(15.0, 2.2, W).clip(8, 30)
    y += np.where(two[:, None], gumbel(t, (pe2 * CALIB_AREA / b2)[:, None], mu[:, None] + rng.uniform(25, 60, W)[:, None],
                                       b2[:, None]), 0.0)
    adc = np.clip(np.round(y + PEDESTAL + rng.normal(0, 2.0, (W, L))), 0, 4095).astype(np.int64)
    return (adc, np.full(W, L), start, np.full(W, PEDESTAL), np.full(W, CALIB_AREA), np.full(W, CALIB_AREA / 15.0))
