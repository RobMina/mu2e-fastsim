"""Opt-in frozen-layer range-mean/fluctuation diagnostic; production/core sources remain untouched.

A process-local trace hook supplies the mean to the existing JAX Urban sampler without changing its physical
length, entrance kinematics, cut or variance formula. Use only in a standalone, single-threaded interpreter.
The range endpoint is relative to 0.1 MeV, not a validated physical zero-energy range. G4-law arms change only
the mean law; the fluctuation kernel and optional analytic delta spectrum are still fast-model approximations.
"""
from dataclasses import dataclass
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from mu2efastsim import eloss, eloss_jax
from . import geometry
from .layerstudy import momentum


THRESHOLD = .1


@jax.jit
def _supplied_mean_kernel(key, p, length, cut, mean):
    """Trace the unjitted sampler body once with a temporary mean-density hook."""
    safe_length = jnp.where(length > 0, length, 1.)
    with patch.object(eloss_jax, "_dedx", lambda *args: mean / safe_length):
        return eloss_jax._sample.__wrapped__(key, eloss_jax.consts(geometry.STOPPING_TARGET_AL),
                                            p, length, cut, jnp.asarray(eloss.MMU))


def sample_supplied_mean(key, p, length, cut, mean):
    """Sample an externally supplied nonnegative mean, retaining the actual length and entrance momentum.

    Validation is deliberately host-side; this diagnostic API is not a production JIT entry point. The only
    rounding introduced by the trace hook is mean / length * length. Zero-length lanes require zero mean.
    """
    p, length, cut, mean = np.broadcast_arrays(*[np.asarray(v, float) for v in (p, length, cut, mean)])
    if any(np.any(~np.isfinite(v)) for v in (p, length, cut, mean)):
        raise ValueError("finite sampler inputs required")
    if np.any(p < 0) or np.any(length < 0) or np.any(cut <= 0) or np.any(mean < 0):
        raise ValueError("nonnegative momentum/path/mean and positive cut required")
    if np.any((length == 0) & (mean != 0)):
        raise ValueError("zero path requires zero mean")
    return np.asarray(_supplied_mean_kernel(key, p, length, cut, mean))


@dataclass(frozen=True)
class MeanLaw:
    """Dense checked mean-power and threshold-relative range arrays, in MeV and mm."""
    energy: np.ndarray
    power: np.ndarray
    ranges: np.ndarray
    cut: float

    def __post_init__(self):
        e, s, r = [np.array(v, float, copy=True) for v in (self.energy, self.power, self.ranges)]
        if e.ndim != 1 or len(e) < 3 or s.shape != e.shape or r.shape != e.shape:
            raise ValueError("equal one-dimensional law arrays with at least three points required")
        if any(np.any(~np.isfinite(v)) for v in (e, s, r)) or np.any(s <= 0):
            raise ValueError("finite arrays and positive stopping power required")
        if e[0] != THRESHOLD or r[0] != 0 or np.any(np.diff(e) <= 0) or np.any(np.diff(r) <= 0):
            raise ValueError("strictly increasing threshold-relative law required")
        if not np.isfinite(self.cut) or self.cut <= 0:
            raise ValueError("positive finite cut required")
        for name, value in (("energy", e), ("power", s), ("ranges", r)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def mean(self, energy, length, mode):
        """Return (supplied mean, forced stop); exact threshold arrival is not a strict stop."""
        energy, length = np.broadcast_arrays(np.asarray(energy, float), np.asarray(length, float))
        if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(length)) or np.any(length < 0):
            raise ValueError("finite energy/path and nonnegative path required")
        if np.any(energy < THRESHOLD) or np.any(energy > self.energy[-1]):
            raise ValueError("energy outside mean-law domain")
        if mode == "local":
            return np.interp(energy, self.energy, self.power) * length, np.zeros(energy.shape, bool)
        if mode != "range":
            raise ValueError("mode must be local or range")
        residual = np.interp(energy, self.energy, self.ranges) - length
        forced = residual < 0
        outgoing = np.interp(np.maximum(residual, 0), self.ranges, self.energy)
        mean = np.where(forced, energy, np.maximum(energy - outgoing, 0))
        return np.where(length == 0, 0., mean), forced


def delta_mean_per_mm(energy, cut):
    """First moment of the core spin-1/2 hard-electron spectrum, not a G4 tabulation."""
    energy = np.asarray(energy, float)
    _, beta2, _, _, tmax = eloss.kinematics(momentum(energy))
    upper = np.maximum(tmax, cut)
    integral = np.log(upper / cut) - beta2 * (upper - cut) / tmax
    integral += (upper ** 2 - cut ** 2) / (4 * (energy + eloss.MMU) ** 2)
    return eloss.TWOPI_MC2_RCL2 * geometry.STOPPING_TARGET_AL.electron_density_per_mm3 / beta2 * integral


def run_layer(energy, length, law, *, mode="range", deltas=False, slices=4, replicas=128, seed=271828,
              fluctuate=True):
    """Vectorized response replicas on frozen paths, without MSC, time, trajectory or electron transport.

    Each slice freezes the hard rate/spectrum at entrance, samples continuous loss first, then subtracts the
    aggregate hard energy with an explicit remaining-energy cap. A range-forced stop bypasses both samplers.
    Counts and cap budgets are saved; this ordering is not G4's post-step interaction scheduler. Below-threshold
    lanes are absorbed immediately; no unvalidated below-threshold stopping-power calls are made.
    """
    energy, length = np.asarray(energy, float), np.asarray(length, float)
    if energy.ndim != 1 or energy.shape != length.shape or len(energy) == 0:
        raise ValueError("nonempty equal one-dimensional entry/path arrays required")
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(length)) or np.any(energy < 0) or np.any(length < 0):
        raise ValueError("finite nonnegative entries and paths required")
    if energy.max() > law.energy[-1]:
        raise ValueError("entry outside mean-law domain")
    for name, value in (("slices", slices), ("replicas", replicas), ("seed", seed)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if name == "seed" else 1):
            raise ValueError(f"invalid {name}")
    if mode not in ("local", "range") or (deltas and not fluctuate):
        raise ValueError("invalid mode or stochastic deltas with fluctuations disabled")
    shape = (replicas, len(energy))
    current = np.broadcast_to(energy, shape).copy()
    path = np.broadcast_to(length / slices, shape)
    totals = {k: np.zeros(shape) for k in ("continuous_applied", "hard_requested", "hard_applied",
                                          "continuous_cap_excess", "hard_cap_excess", "threshold_residual")}
    forced_any = np.zeros(shape, bool)
    hard_count = np.zeros(shape, np.int64)
    key = jax.random.key(seed)
    for step in range(slices):
        active = (current >= THRESHOLD) & (path > 0)
        safe_energy = np.maximum(current, THRESHOLD)
        step_path = np.where(active, path, 0.)
        mean, forced = law.mean(safe_energy, step_path, mode)
        forced &= active
        forced_any |= forced
        sample_path = np.where(forced, 0., step_path)
        sampled = sample_supplied_mean(jax.random.fold_in(key, step), momentum(safe_energy), sample_path,
                                       law.cut, np.where(forced, 0., mean)) if fluctuate else mean
        requested = np.where(forced, current, np.where(active, sampled, 0.))
        if np.any(~np.isfinite(requested)) or np.any(requested < 0):
            raise FloatingPointError("invalid sampled continuous loss")
        continuous = np.minimum(current, requested)
        remaining = current - continuous
        if not fluctuate and mode == "range":
            residual_range = np.interp(safe_energy, law.energy, law.ranges) - step_path
            exact_endpoint = np.interp(np.maximum(residual_range, 0.), law.ranges, law.energy)
            remaining = np.where(active & ~forced, exact_endpoint, remaining)
            continuous = current - remaining
        totals["continuous_applied"] += continuous
        totals["continuous_cap_excess"] += np.maximum(requested - continuous, 0.)
        hard = np.zeros(shape)
        if deltas:
            rng = np.random.default_rng(np.random.SeedSequence([seed, step, 314159]))
            indices, transfers = eloss.sample_deltas(rng, geometry.STOPPING_TARGET_AL,
                                                    momentum(safe_energy).ravel(), sample_path.ravel(), law.cut)
            hard = np.bincount(indices, weights=transfers, minlength=current.size).reshape(shape)
            hard_count += np.bincount(indices, minlength=current.size).reshape(shape)
        applied = np.minimum(remaining, hard)
        totals["hard_requested"] += hard
        totals["hard_applied"] += applied
        totals["hard_cap_excess"] += hard - applied
        current = remaining - applied
        residual = np.where(current < THRESHOLD, current, 0.)
        totals["threshold_residual"] += residual
        current = np.where(current < THRESHOLD, 0., current)
    accounted = current + totals["continuous_applied"] + totals["hard_applied"] + totals["threshold_residual"]
    if not np.allclose(accounted, energy[None, :], rtol=1e-12, atol=1e-12):
        raise FloatingPointError("layer energy ledger does not close")
    return dict(exit_MeV=current, stop=current < THRESHOLD, forced_range_stop=forced_any,
                hard_count=hard_count, **totals)
