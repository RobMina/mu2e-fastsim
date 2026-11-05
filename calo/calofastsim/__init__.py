"""Deterministic calorimeter primitives, not a stock simulation replacement.

NumPy is the default backend. Import ``calofastsim.response_jax`` explicitly for
optional JAX support; importing this package does not import JAX.
"""

from .response import (
    corrected_energy,
    fold_step_time,
    nonnegative_lround,
    photoelectron_mean,
    weighted_reco_digis,
)

__all__ = [
    "corrected_energy",
    "fold_step_time",
    "nonnegative_lround",
    "photoelectron_mean",
    "weighted_reco_digis",
]
