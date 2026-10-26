"""Deterministic adaptive mean-loss diagnostic, without transport, straggling, MSC or RNG.

This integrates the existing restricted stopping law, not independent low-energy physics. Fractions
0.1, 0.05, 0.02, 0.01 and 0.005 are useful convergence probes, not production-default recommendations.
"""
from typing import NamedTuple

import numpy as np

from mu2efastsim import eloss
from . import geometry


class AdaptiveLossResult(NamedTuple):
    """Per-lane arrays; energies in MeV, consumed length in mm, fractions dimensionless.

    ``energy`` retains the actual Euler endpoint, including endpoints below the stopping threshold.
    ``max_requested_fraction`` is the maximum requested loss divided by pre-step energy (zero for
    untouched lanes). ``capped`` means remaining path and energy at/above threshold after max_steps;
    it is unresolved, never a stop. ``steps`` counts only updates actually taken by that lane.
    """

    energy: np.ndarray
    stopped: np.ndarray
    steps: np.ndarray
    consumed_length: np.ndarray
    max_requested_fraction: np.ndarray
    capped: np.ndarray


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def adaptive_mean_loss(energy, length, frac=0.05, *, threshold=0.1, max_steps=10000,
                       mat=geometry.STOPPING_TARGET_AL, tcut=1.0, stopping=None):
    """Integrate one-dimensional equal-shaped energy/path arrays by bounded explicit Euler steps.

    Each active lane uses h = min(remaining_path, frac*T/S(T)), followed by T -= h*S(T).
    Using T rather than T-threshold allows a finite crossing of the strict T < threshold stop.
    The crossing step is consumed in full; no threshold interpolation or zero-energy clamp is made.
    Already-stopped and zero-path lanes are not evaluated. Empty arrays are supported; inputs are
    copied, never modified. No broadcasting between energy and length is performed.

    By default S is eloss.dedx_restricted(mat, sqrt(T*(T+2*MMU)), tcut) in MeV/mm. An optional
    ``stopping(T)`` callable instead receives only active kinetic energies and returns their stopping
    powers, either an equal-shaped array or a scalar. It must be deterministic, finite and nonnegative;
    zero stopping power consumes all remaining path without energy loss. A same-law interpolant can
    be injected this way, but its domain/accuracy remains the caller's responsibility.

    ``frac`` must lie strictly between zero and one; threshold and tcut must be finite and positive;
    max_steps must be a positive integer. Step-capped lanes are explicitly marked unresolved. If a
    positive step cannot be represented or accumulated, FloatingPointError prevents silent nonprogress.
    Fraction bounds hold to floating-point rounding, not an exact-real-arithmetic guarantee.
    """
    energy = np.array(energy, dtype=float, copy=True)
    length = np.array(length, dtype=float, copy=True)
    if energy.ndim != 1 or length.ndim != 1 or energy.shape != length.shape:
        raise ValueError("energy and length must be equal-shaped one-dimensional arrays")
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(length)) or np.any(energy < 0) or np.any(length < 0):
        raise ValueError("energy and length must be finite and nonnegative")
    frac = _positive_scalar(frac, "frac")
    if frac >= 1:
        raise ValueError("frac must be strictly between zero and one")
    threshold = _positive_scalar(threshold, "threshold")
    tcut = _positive_scalar(tcut, "tcut")
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if stopping is not None and not callable(stopping):
        raise ValueError("stopping must be callable")

    steps = np.zeros(energy.shape, dtype=np.int64)
    consumed = np.zeros_like(energy)
    maximum = np.zeros_like(energy)
    active = (energy >= threshold) & (length > 0)
    for _ in range(max_steps):
        if not np.any(active):
            break
        current = energy[active]
        if stopping is None:
            power = eloss.dedx_restricted(mat, np.sqrt(current * (current + 2 * eloss.MMU)), tcut)
        else:
            power = stopping(current)
        power = np.asarray(power, dtype=float)
        if power.ndim == 0:
            power = np.full_like(current, power)
        if power.shape != current.shape or np.any(~np.isfinite(power)) or np.any(power < 0):
            raise ValueError("stopping power must be finite, nonnegative and scalar or match active energies")
        remaining = length[active] - consumed[active]
        with np.errstate(divide="ignore", over="ignore"):
            limit = np.divide(frac * current, power, out=np.full_like(current, np.inf), where=power > 0)
        h = np.minimum(remaining, limit)
        updated_length = np.minimum(length[active], consumed[active] + h)
        updated_length = np.where(h == remaining, length[active], updated_length)
        if np.any(h <= 0) or np.any(updated_length <= consumed[active]):
            raise FloatingPointError("adaptive loss step cannot make representable path progress")
        requested = h * power
        energy[active] = current - requested
        consumed[active] = updated_length
        maximum[active] = np.maximum(maximum[active], requested / current)
        steps[active] += 1
        active = (energy >= threshold) & (consumed < length)
    return AdaptiveLossResult(energy, energy < threshold, steps, consumed, maximum, active.copy())
