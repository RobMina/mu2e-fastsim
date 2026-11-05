"""Optional JAX counterpart of :mod:`calofastsim.response`.

Array values stay on device and every primitive can be jitted. Shape checks are
static; NUMERIC validity is a HOST responsibility: preflight with the NumPy
backend outside jit. Invalid means, lengths, timing, errors or group IDs have no
promised device behavior. See response.py for the shared contracts and sources.

The caller must enable JAX x64 before tracing fold_step_time or
weighted_reco_digis; these operations need source double precision. This module
never changes JAX configuration. Group count must be static under jit. Parallel
scatter reductions may differ from sequential C++ accumulation at roundoff level.
XLA may also contract float32 arithmetic despite explicit source-order barriers;
bitwise NumPy equivalence is not promised for corrected_energy under jit.
These are deterministic primitives, not stock simulation execution validation.
"""

import operator

import jax
import jax.numpy as jnp
from jax import lax


__all__ = [
    "corrected_energy", "photoelectron_mean", "nonnegative_lround",
    "fold_step_time", "weighted_reco_digis",
]


def _require_x64():
    if not jax.config.x64_enabled:
        raise ValueError("source double arithmetic requires caller-enabled JAX x64")


def _stored(value):
    """Expose a source intermediate without promising compiler non-contraction."""
    return lax.optimization_barrier(value)


def corrected_energy(energy_g4, energy_birks, z_local, crystal_length, lru, *,
                     birks=True, longitudinal=True):
    """Float32 energy; signature/broadcasting match the NumPy backend.

    Positive finite lengths must be validated on the host before jit. Boolean
    switches may themselves be dynamic arrays. No position/energy clipping.
    """
    g4, quenched, z, length, slope = jnp.broadcast_arrays(
        *[jnp.asarray(x, dtype=jnp.float32) for x in
          (energy_g4, energy_birks, z_local, crystal_length, lru)]
    )
    energy = jnp.where(birks, quenched, g4)
    normalized = _stored(z / length)
    centered = _stored(normalized - jnp.float32(0.5))
    scaled = _stored(slope * centered)
    factor = _stored(scaled + jnp.float32(1.0))
    corrected = _stored(energy * factor)
    return jnp.where(longitudinal, corrected, energy)


def photoelectron_mean(corrected_energy, yields):
    """Float32 nonnegative means on a trailing axis of exactly two SiPMs."""
    energy = jnp.asarray(corrected_energy, dtype=jnp.float32)
    pe_yields = jnp.asarray(yields, dtype=jnp.float32)
    if pe_yields.ndim == 0 or pe_yields.shape[-1] != 2:
        raise ValueError("yields must have a trailing axis of length 2")
    return jnp.maximum(jnp.float32(0.0), energy[..., None] * pe_yields)


def nonnegative_lround(mean):
    """Int32 half-up PE counts, with NumPy's validity guards required on HOST.

    Means must be finite, nonnegative and round to a representable int32 count.
    Splitting integer/fractional parts avoids both ties-to-even and rounding a
    float32 value just below a half upward during an intermediate ``mean+0.5``.
    """
    mean = jnp.asarray(mean, dtype=jnp.float32)
    base = jnp.floor(mean)
    increment = (mean - base) >= jnp.float32(0.5)
    return base.astype(jnp.int32) + increment.astype(jnp.int32)


def fold_step_time(time, *, pbtime, mbtime, event_length, onspill,
                   digitization_start, digitization_end, digitization_buffer,
                   time_from_protons_to_dr_marker):
    """Return (float64 folded time, accepted mask), requiring caller x64.

    Uses fmod (not remainder), one conditional period shift, lower-only rejection
    and explicit supplied timing values, exactly as documented by the NumPy API.
    Host must validate finite times and positive mbtime/event_length.
    """
    _require_x64()
    time = jnp.asarray(time, dtype=jnp.float64)
    event_length = jnp.asarray(event_length, dtype=jnp.float64)
    pb, mb, start, end, buffer, marker = (
        jnp.asarray(x, dtype=jnp.float32) for x in
        (pbtime, mbtime, digitization_start, digitization_end,
         digitization_buffer, time_from_protons_to_dr_marker)
    )
    corrected_start = _stored(_stored(_stored(start - pb) - marker) - buffer)
    corrected_end = _stored(_stored(end - pb) - marker)
    pb64 = pb.astype(jnp.float64)
    length = jnp.where(onspill, mb.astype(jnp.float64), event_length)
    max_hit_time = jnp.maximum(mb.astype(jnp.float64), corrected_end.astype(jnp.float64))
    hit = _stored(jnp.fmod(_stored(time + pb64), length) - pb64)
    hit = jnp.where(hit < _stored(max_hit_time - length), _stored(hit + length), hit)
    hit, threshold = jnp.broadcast_arrays(hit, corrected_start)
    return hit, hit >= threshold


def weighted_reco_digis(values, errors, groups, n_groups):
    """Given membership, return float32 (weighted value, error); require x64.

    Numeric validation is host-only: finite float32 values, positive finite
    float32 errors, integer group IDs in [0, n_groups). Shapes match NumPy.
    ``n_groups`` is a nonnegative static integer (use static_argnames for jit).
    Empty groups return (NaN, +inf). This does not discover or select groups.
    """
    _require_x64()
    n_groups = operator.index(n_groups)
    if n_groups < 0:
        raise ValueError("n_groups must be nonnegative")
    values, errors = jnp.broadcast_arrays(
        jnp.asarray(values, dtype=jnp.float32), jnp.asarray(errors, dtype=jnp.float32)
    )
    groups = jnp.asarray(groups)
    if values.ndim == 0 or groups.ndim != 1 or groups.shape[0] != values.shape[0]:
        raise ValueError("groups must be 1D with one entry per first-axis value")
    if not jnp.issubdtype(groups.dtype, jnp.integer):
        raise ValueError("groups must have integer dtype")
    error64 = errors.astype(jnp.float64)
    weights = _stored(_stored(jnp.float64(1.0) / error64) / error64)
    shape = (n_groups,) + values.shape[1:]
    total = jnp.zeros(shape, dtype=jnp.float64).at[groups].add(weights)
    products = _stored(weights * values.astype(jnp.float64))
    weighted = jnp.zeros(shape, dtype=jnp.float64).at[groups].add(products)
    result = (weighted / total).astype(jnp.float32)
    uncertainty = (jnp.float64(1.0) / jnp.sqrt(total)).astype(jnp.float32)
    return result, uncertainty
