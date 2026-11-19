"""Bins of a photon lookup table: the variables each photon source samples and the row-major bin numbering.

A sim type is one photon source. Its bin is a box in the variables it samples (``SAMPLED``): position for
scintillation, position and particle speed for Cerenkov light in the scintillator, and z, fiber radius, azimuth,
polar angle and speed for Cerenkov light in the fiber. A ``Binning`` holds the edges of every variable and, per sim
type, the order of its variables (slowest first); the bin number is the row-major index in that order.
"""
from dataclasses import dataclass

import numpy as np

SIM_TYPES = {0: "scintillation in scintillator", 1: "Cerenkov in scintillator", 2: "Cerenkov in fiber"}
SAMPLED = {0: frozenset("x y z".split()), 1: frozenset("x y z beta".split()),
           2: frozenset("z r phi theta beta".split())}


@dataclass(frozen=True, eq=False)
class Binning:
    """``edges``: {variable: increasing edges}; ``order``: {sim type: variables, slowest first}.

    Variables: x (>= 0, the strip is mirror symmetric in x), y, z (mm, strip frame); beta (particle speed / c);
    theta, phi (rad, photon direction relative to the fiber axis and the x axis); r (mm, distance from the fiber
    axis)."""
    edges: dict
    order: dict

    def __post_init__(self):
        for st, names in self.order.items():
            if set(names) != SAMPLED[st]:
                raise ValueError(f"sim type {st}: the order must name exactly {sorted(SAMPLED[st])}")
        for k, e in self.edges.items():
            e = np.asarray(e, np.float64)
            if e.ndim != 1 or len(e) < 2 or not np.all(np.diff(e) > 0):
                raise ValueError(f"edges of {k!r} must be strictly increasing, at least two")

    def shape(self, simtype):
        return tuple(len(self.edges[k]) - 1 for k in self.order[simtype])

    def n_bins(self, simtype):
        return int(np.prod(self.shape(simtype)))

    def boxes(self, simtype, bins):
        """Per variable, the (lo, hi) edges of every bin in ``bins`` (the ranges the photon source samples)."""
        b = np.asarray(bins, np.int64)
        idx = np.unravel_index(b, self.shape(simtype))
        return {k: (np.asarray(self.edges[k])[i], np.asarray(self.edges[k])[i + 1])
                for k, i in zip(self.order[simtype], idx)}
