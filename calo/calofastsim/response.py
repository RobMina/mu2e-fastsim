"""Vectorised deterministic pieces transcribed from Offline calorimeter code.

Sources: CaloMC/src/CaloShowerROMaker_module.cc (LRUCorrection and
makeReadoutHits), CaloMC/src/CaloPhotonPropagation.cc, and
CaloReco/src/CaloHitMaker_module.cc (makeCaloHits). Data product headers fix the
float32 input/storage types. There is no photon sampling, propagation table,
waveform fitting, or time/crystal grouping here. All condition and timing values
are supplied by the caller; these primitives introduce no calibration constants.
Synthetic source-conformance checks are not validation against stock execution.
"""

import operator

import numpy as np


__all__ = [
    "corrected_energy", "photoelectron_mean", "nonnegative_lround",
    "fold_step_time", "weighted_reco_digis",
]


def corrected_energy(energy_g4, energy_birks, z_local, crystal_length, lru, *,
                     birks=True, longitudinal=True):
    """Return broadcast float32 energy, with the source's operation order.

    Select Birks or G4 energy, then multiply by
    ``lru * (z_local / crystal_length - 0.5f) + 1.0f`` if enabled.
    All five numeric inputs broadcast together, including disabled inputs.
    Crystal lengths must be finite and positive; no position or energy clipping
    is performed. Switches can be scalar or broadcastable boolean arrays.
    """
    g4, quenched, z, length, slope = np.broadcast_arrays(
        *[np.asarray(x, dtype=np.float32) for x in
          (energy_g4, energy_birks, z_local, crystal_length, lru)]
    )
    if np.any(~np.isfinite(length) | (length <= 0)):
        raise ValueError("crystal_length must be finite and positive")
    energy = np.where(birks, quenched, g4)
    normalized = z / length
    centered = normalized - np.float32(0.5)
    scaled = slope * centered
    factor = scaled + np.float32(1.0)
    return np.where(longitudinal, energy * factor, energy).astype(np.float32)


def photoelectron_mean(corrected_energy, yields):
    """Return nonnegative float32 means on a trailing two-SiPM axis.

    ``yields`` must have shape ``(..., 2)``; its leading dimensions broadcast
    with the energy shape. For example energy ``(N,)`` and yields ``(2,)`` or
    ``(N, 2)`` give ``(N, 2)``. Negative products are clamped, not absolute-valued.
    """
    energy = np.asarray(corrected_energy, dtype=np.float32)
    pe_yields = np.asarray(yields, dtype=np.float32)
    if pe_yields.ndim == 0 or pe_yields.shape[-1] != 2:
        raise ValueError("yields must have a trailing axis of length 2")
    product = energy[..., None] * pe_yields
    return np.maximum(np.float32(0.0), product)


def nonnegative_lround(mean):
    """Round finite nonnegative float32 PE means to int32, halves upward.

    This is ``int(std::lround(mean))`` on its defined, nonnegative int32 domain,
    not NumPy's ties-to-even rounding. Reject NaN, infinity, negatives, and
    rounded counts outside int32 before converting. Empty arrays are supported.
    Host validation here can also preflight inputs for the JAX counterpart.
    """
    mean = np.asarray(mean, dtype=np.float32)
    if np.any(~np.isfinite(mean) | (mean < 0)):
        raise ValueError("mean must be finite and nonnegative")
    base = np.floor(mean.astype(np.float64))
    rounded = base + ((mean.astype(np.float64) - base) >= 0.5)
    if np.any(rounded > np.iinfo(np.int32).max):
        raise ValueError("rounded count exceeds int32 range")
    return rounded.astype(np.int32)


def fold_step_time(time, *, pbtime, mbtime, event_length, onspill,
                   digitization_start, digitization_end, digitization_buffer,
                   time_from_protons_to_dr_marker):
    """Return ``(folded_time_float64, accepted_bool)`` with source fmod folding.

    ``mbtime`` is the nominal DR period, ``event_length`` the event-window
    length, and ``onspill`` selects the former as folding period. Configuration
    times, pbtime and mbtime are float32; step time and event_length are float64.
    Start/end are corrected in float32, in left-to-right source order. Then::

        length = mbtime if onspill else event_length
        max_hit_time = max(mbtime, corrected_end)
        hit = fmod(time + pbtime, length) - pbtime
        if hit < max_hit_time - length: hit += length
        accepted = hit >= corrected_start

    There is deliberately NO upper-time rejection and NO extra positive-modulo
    correction. The returned time precedes float32 readout storage. All inputs
    broadcast, must be finite, and both candidate periods must be positive.
    """
    time = np.asarray(time, dtype=np.float64)
    event_length = np.asarray(event_length, dtype=np.float64)
    pb, mb, start, end, buffer, marker = (
        np.asarray(x, dtype=np.float32) for x in
        (pbtime, mbtime, digitization_start, digitization_end,
         digitization_buffer, time_from_protons_to_dr_marker)
    )
    if any(np.any(~np.isfinite(x)) for x in
           (time, event_length, pb, mb, start, end, buffer, marker)):
        raise ValueError("timing inputs must be finite")
    if np.any(event_length <= 0) or np.any(mb <= 0):
        raise ValueError("mbtime and event_length must be positive")
    corrected_start = ((start - pb) - marker) - buffer
    corrected_end = (end - pb) - marker
    length = np.where(onspill, mb.astype(np.float64), event_length)
    max_hit_time = np.maximum(mb.astype(np.float64), corrected_end.astype(np.float64))
    hit = np.fmod(time + pb.astype(np.float64), length) - pb.astype(np.float64)
    hit = np.where(hit < max_hit_time - length, hit + length, hit)
    hit, threshold = np.broadcast_arrays(hit, corrected_start)
    return hit, hit >= threshold


def weighted_reco_digis(values, errors, groups, n_groups):
    """Return grouped ``(value, error)`` float32 arrays for GIVEN membership.

    Values/errors broadcast to ``(N, ...)`` and are first stored as float32,
    matching CaloRecoDigi. ``groups`` is a length-N integer array in
    ``[0, n_groups)``. For each trailing component separately, accumulate
    ``w = 1.0 / double(error) / double(error)`` and ``w * double(value)`` in
    float64, then store ``sum(w*v)/sum(w)`` and ``1/sqrt(sum(w))`` as float32.
    Only finite values and strictly positive finite float32 errors are allowed.
    Empty groups give (NaN, +inf); no hit selection or grouping is inferred.
    """
    n_groups = operator.index(n_groups)
    if n_groups < 0:
        raise ValueError("n_groups must be nonnegative")
    values, errors = np.broadcast_arrays(
        np.asarray(values, dtype=np.float32), np.asarray(errors, dtype=np.float32)
    )
    groups = np.asarray(groups)
    if values.ndim == 0 or groups.ndim != 1 or groups.shape[0] != values.shape[0]:
        raise ValueError("groups must be 1D with one entry per first-axis value")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("groups must have integer dtype")
    if np.any((groups < 0) | (groups >= n_groups)):
        raise ValueError("group index outside [0, n_groups)")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite")
    if np.any(~np.isfinite(errors) | (errors <= 0)):
        raise ValueError("errors must be finite and strictly positive")
    error64 = errors.astype(np.float64)
    weights = (np.float64(1.0) / error64) / error64
    shape = (n_groups,) + values.shape[1:]
    total = np.zeros(shape, dtype=np.float64)
    weighted = np.zeros(shape, dtype=np.float64)
    np.add.at(total, groups, weights)
    np.add.at(weighted, groups, weights * values.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = (weighted / total).astype(np.float32)
        uncertainty = (np.float64(1.0) / np.sqrt(total)).astype(np.float32)
    return result, uncertainty
