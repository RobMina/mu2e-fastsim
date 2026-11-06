"""Compatibility imports for the shared, restricted annular geometry primitive.

The implementation lives in mu2efastsim.geometry.annulus_jax. It remains a
float64 straight/axial-field development solver, not source-navigator support
for finite charged foils. Callers must handle unsupported/unresolved results.
"""
from mu2efastsim.geometry.annulus_jax import Crossing, crossing, safety

__all__ = ["Crossing", "crossing", "safety"]
