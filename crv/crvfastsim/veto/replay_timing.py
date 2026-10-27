"""Diagnostic-only Run1A onspill timing predicates, separate from production response."""
from __future__ import annotations

import numpy as np

PERIOD = 1695.0
DIGI_START = 200.0
DIGI_END = 1500.0
STEP_MARGIN = 100.0


def wrap_arrivals(time, origin=0.0, period=PERIOD):
    """Offline's corrected-fmod operation, including its floating-point operation order."""
    time, origin = np.broadcast_arrays(np.asarray(time, float), np.asarray(origin, float))
    if not np.isfinite(period) or period <= 0 or not np.isfinite(time).all() or not np.isfinite(origin).all():
        raise ValueError("finite times/origins and a positive finite period required")
    result = np.fmod(time - origin, period) + origin
    return np.where(result < origin, result + period, result)


def stock_step_mask(t0, t1, origin=0.0, period=PERIOD, start=DIGI_START, end=DIGI_END, margin=STEP_MARGIN):
    """Stock ZS endpoint heuristic; NaNs skip, infinities fail closed rather than emulate undefined optics."""
    t0, t1, origin = np.broadcast_arrays(np.asarray(t0, float), np.asarray(t1, float), np.asarray(origin, float))
    if np.isinf(t0).any() or np.isinf(t1).any() or not np.isfinite(origin).all():
        raise ValueError("infinite step endpoint or nonfinite origin")
    if not np.isfinite([start, end, margin]).all() or margin < 0 or start > end:
        raise ValueError("invalid readout interval")
    valid = np.isfinite(t0) & np.isfinite(t1)
    a = wrap_arrivals(np.where(valid, t0, origin), origin, period)
    b = wrap_arrivals(np.where(valid, t1, origin), origin, period)
    lo, hi = origin + start - margin, origin + end
    reject = ((a < lo) & (b < lo)) | ((a > hi) & (b > hi)) | ((a > hi) & (b < lo))
    return valid & ~reject


def arrival_arms(step, time, accepted_steps, period=PERIOD, end=DIGI_END):
    """Reuse one photon draw; apply wrapping and step eligibility before the SiPM upper cut."""
    step, time = np.asarray(step), np.asarray(time, float)
    accepted_steps = np.asarray(accepted_steps, bool)
    if step.shape != time.shape or step.ndim != 1 or accepted_steps.ndim != 1:
        raise ValueError("one-dimensional aligned photon arrays required")
    if not np.issubdtype(step.dtype, np.integer) or np.any((step < 0) | (step >= len(accepted_steps))):
        raise ValueError("photon step index out of range")
    if not np.isfinite(end):
        raise ValueError("nonfinite SiPM upper bound")
    folded = wrap_arrivals(time, period=period)
    return {
        "step_only": (time.copy(), time <= end),
        "arrival_only": (folded, folded <= end),
        "stock_timing": (folded, accepted_steps[step] & (folded <= end)),
    }


def digi_start_mask(start, origin=0.0, lo=DIGI_START, hi=DIGI_END):
    """Inclusive digi-start window; no truncation or rejection based on the last sample."""
    start, origin = np.broadcast_arrays(np.asarray(start, float), np.asarray(origin, float))
    if not np.isfinite(start).all() or not np.isfinite(origin).all() or not np.isfinite([lo, hi]).all() or lo > hi:
        raise ValueError("invalid digi window/times")
    return (start >= origin + lo) & (start <= origin + hi)


def boundary_table():
    """Deterministic edge probes in absolute stock coordinates at every retained bunch phase."""
    import pandas as pd

    w = -np.asarray([-224.46, -219.46, -214.46, -209.46, -204.46], np.float32).astype(float)
    edges = np.asarray([-PERIOD, 0., 100., 200., 1500., PERIOD, 2 * PERIOD])
    absolute = w[:, None] + edges
    times = np.stack([np.nextafter(absolute, -np.inf), absolute, np.nextafter(absolute, np.inf)], axis=-1)
    origin = np.broadcast_to(w[:, None, None], times.shape)
    folded = wrap_arrivals(times, origin)
    return pd.DataFrame({
        "w": origin.ravel(), "edge_relative": np.broadcast_to(edges[None, :, None], times.shape).ravel(),
        "epsilon_direction": np.broadcast_to(np.array([-1, 0, 1])[None, None, :], times.shape).ravel(),
        "raw_time": times.ravel(), "wrapped_absolute": folded.ravel(),
        "zero_length_step_kept": stock_step_mask(times, times, origin).ravel(),
        "sipm_upper_kept": (folded <= origin + DIGI_END).ravel(),
        "digi_start_kept": digi_start_mask(times, origin).ravel(),
    })
