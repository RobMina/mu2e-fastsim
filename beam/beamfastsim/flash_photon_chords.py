"""Bounded pure S1 photon-chord diagnostics, not physics acceptance.

The caller must explicitly bind the photon product/PDG and event-qualified,
new-stage particle identities. This module neither verifies nor infers them.
There is no geometry, time, tolerance, cut, transport, filtering, deduplication,
coordinate nudge, or overall validity/completeness certificate here. Storage and
work are O(N), using NumPy operations without per-row Python loops.
"""

from dataclasses import dataclass

import numpy as np


NONFINITE_POSITION = 1
NONFINITE_LENGTH = 2
NONPOSITIVE_LENGTH = 4
NONFINITE_MOMENTUM = 8
ZERO_CHORD = 16
ZERO_MOMENTUM = 32
NUMERIC_OVERFLOW = 64


@dataclass(frozen=True)
class PhotonChordDiagnostics:
    """Aligned (N,) arrays; floats are float64 and reason is a uint8 bitmask.

    chord_minus_length is signed chord_length minus the stored StepPoint length,
    including finite zero/negative lengths (which receive NONPOSITIVE_LENGTH).
    length_storage_ulp is np.spacing evaluated in float32 for positive finite
    stored lengths only. It is storage information, never a correction or cut.
    At float32's maximum, spacing overflows: this metric is NaN and the row has
    NUMERIC_OVERFLOW, even though other metrics can remain available.

    direction_cosine is the dot product of unit chord and unit pre-momentum;
    transverse_sine is the norm of their cross product. Backward chords retain
    negative cosine. Neither quantity is clipped or compared to a threshold.
    Zero chords retain their zero length/residual, but have no direction.
    Bad momentum does not erase chord/residual/ULP; bad length does not erase
    chord/direction. NaN means a particular metric is unavailable, not no row.

    Reason bits describe input/numerical conditions, not rejection physics.
    NONFINITE_* describes original input values; NUMERIC_OVERFLOW also covers
    finite inputs unavailable after float64 conversion/subtraction, and
    normalization/product underflow that would silently lose nonzero terms.
    Integer vector components must lie in the inclusive exact range
    [-2**53, +2**53]; even exactly representable integers outside that range
    are conservatively unavailable. Wider floats must round-trip through
    float64 unchanged. Conversion uncertainty never establishes a zero vector;
    position and momentum availability are independent.
    A zero reason is not a photon, straightness, producer or physics certificate.
    Frozen fields do not make the separately owned NumPy arrays immutable.
    """

    chord_length: np.ndarray
    chord_minus_length: np.ndarray
    length_storage_ulp: np.ndarray
    direction_cosine: np.ndarray
    transverse_sine: np.ndarray
    reason: np.ndarray


def _vector_array(value):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError("positions and momentum require real numeric (N,3) arrays")
    return array


def _float64_vector(value):
    """Convert without silently accepting uncertain integer/wider-float geometry."""
    converted = value.astype(np.float64, copy=False)
    if value.dtype.kind in "iu":
        # Compare in the original integer dtype, never after float promotion.
        lost = value > 2**53
        if value.dtype.kind == "i":
            lost |= value < -(2**53)
    elif value.dtype.itemsize > np.dtype(np.float64).itemsize:
        lost = np.isfinite(value) & (converted.astype(value.dtype) != value)
    else:
        lost = np.zeros(value.shape, dtype=bool)
    return converted, np.any(lost, axis=1)


def _unit_vectors(vector, available):
    """Scale before normalization, including subnormal or very large vectors."""
    scale = np.max(np.abs(vector), axis=1)
    nonzero = available & (scale > 0)
    scaled = np.zeros_like(vector)
    np.divide(vector, scale[:, None], out=scaled, where=nonzero[:, None])
    lost = nonzero & np.any((vector != 0) & (scaled == 0), axis=1)
    norm = np.hypot.reduce(scaled, axis=1)
    unit = np.full_like(vector, np.nan)
    np.divide(scaled, norm[:, None], out=unit, where=(nonzero & ~lost)[:, None])
    lost |= nonzero & np.any((scaled != 0) & (unit == 0), axis=1)
    unit[lost] = np.nan
    return unit, lost


def diagnose_photon_chords(pre, post, pre_momentum, length):
    """Diagnose aligned raw rows without selecting or interpreting photons.

    Positions/momentum must be matching real numeric (N,3) arrays; computations
    promote them to float64 under the finite exact-range/round-trip policy
    documented on PhotonChordDiagnostics. length must be a one-dimensional float32
    array of N stored StepPoint lengths, not widened doubles claimed as floats.
    Nonfinite data are diagnosed row-wise; malformed schemas raise ValueError.
    Inputs are never mutated; duplicate and empty rows retain their alignment.
    Unrepresentable individual metrics become NaN with NUMERIC_OVERFLOW rather
    than infinities or false finite results. All numeric warnings are suppressed.
    """
    a, b, momentum = map(_vector_array, (pre, post, pre_momentum))
    stored = np.asarray(length)
    if b.shape != a.shape or momentum.shape != a.shape:
        raise ValueError("positions and momentum must have matching (N,3) shapes")
    if (stored.ndim != 1 or stored.shape != (len(a),)
            or stored.dtype.kind != "f" or stored.dtype.itemsize != 4):
        raise ValueError("length requires an aligned one-dimensional float32 array")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        return _diagnose(a, b, momentum, stored)


def _diagnose(a, b, momentum, stored):
    n = len(a)
    finite_position = np.all(np.isfinite(a) & np.isfinite(b), axis=1)
    finite_momentum = np.all(np.isfinite(momentum), axis=1)
    finite_length = np.isfinite(stored)
    nonpositive = finite_length & (stored <= 0)
    a, a_cast_lost = _float64_vector(a)
    b, b_cast_lost = _float64_vector(b)
    momentum, momentum_cast_lost = _float64_vector(momentum)
    position_cast_lost = a_cast_lost | b_cast_lost
    delta = b - a
    chord_available = finite_position & ~position_cast_lost & np.all(np.isfinite(delta), axis=1)
    momentum_available = (finite_momentum & ~momentum_cast_lost
                          & np.all(np.isfinite(momentum), axis=1))
    raw_chord = np.hypot.reduce(delta, axis=1)
    chord_length = np.where(chord_available & np.isfinite(raw_chord), raw_chord, np.nan)
    zero_chord = chord_available & np.all(delta == 0, axis=1)
    zero_momentum = momentum_available & np.all(momentum == 0, axis=1)
    residual = chord_length - stored.astype(np.float64)
    residual_available = np.isfinite(chord_length) & finite_length
    chord_minus_length = np.where(residual_available & np.isfinite(residual), residual, np.nan)
    positive_length = finite_length & (stored > 0)
    spacing = np.full(n, np.nan, dtype=np.float32)
    np.spacing(stored, out=spacing, where=positive_length)
    length_storage_ulp = np.where(positive_length & np.isfinite(spacing), spacing, np.nan).astype(np.float64)
    chord_unit, chord_lost = _unit_vectors(delta, chord_available)
    momentum_unit, momentum_lost = _unit_vectors(momentum, momentum_available)
    direction_available = (chord_available & momentum_available & ~zero_chord
                           & ~zero_momentum & ~chord_lost & ~momentum_lost)
    dot_terms = chord_unit * momentum_unit
    dot_lost = np.any((chord_unit != 0) & (momentum_unit != 0) & (dot_terms == 0), axis=1)
    cosine = np.sum(dot_terms, axis=1)
    # Explicit cyclic products expose underflow hidden inside np.cross.
    left = chord_unit[:, [1, 2, 0]] * momentum_unit[:, [2, 0, 1]]
    right = chord_unit[:, [2, 0, 1]] * momentum_unit[:, [1, 2, 0]]
    cross_lost = np.any(((chord_unit[:, [1, 2, 0]] != 0)
                         & (momentum_unit[:, [2, 0, 1]] != 0) & (left == 0))
                        | ((chord_unit[:, [2, 0, 1]] != 0)
                           & (momentum_unit[:, [1, 2, 0]] != 0) & (right == 0)), axis=1)
    sine = np.hypot.reduce(left - right, axis=1)
    direction_cosine = np.where(direction_available & ~dot_lost & np.isfinite(cosine), cosine, np.nan)
    transverse_sine = np.where(direction_available & ~cross_lost & np.isfinite(sine), sine, np.nan)
    numeric = ((finite_position & ~chord_available)
               | (finite_momentum & ~momentum_available)
               | (chord_available & ~np.isfinite(raw_chord))
               | (residual_available & ~np.isfinite(residual))
               | (positive_length & ~np.isfinite(spacing))
               | chord_lost | momentum_lost
               | (direction_available & (dot_lost | cross_lost
                                          | ~np.isfinite(cosine) | ~np.isfinite(sine))))
    reason = ((~finite_position).astype(np.uint8) * NONFINITE_POSITION
              | (~finite_length).astype(np.uint8) * NONFINITE_LENGTH
              | nonpositive.astype(np.uint8) * NONPOSITIVE_LENGTH
              | (~finite_momentum).astype(np.uint8) * NONFINITE_MOMENTUM
              | zero_chord.astype(np.uint8) * ZERO_CHORD
              | zero_momentum.astype(np.uint8) * ZERO_MOMENTUM
              | numeric.astype(np.uint8) * NUMERIC_OVERFLOW)
    return PhotonChordDiagnostics(chord_length, chord_minus_length, length_storage_ulp,
                                  direction_cosine, transverse_sine, reason)
