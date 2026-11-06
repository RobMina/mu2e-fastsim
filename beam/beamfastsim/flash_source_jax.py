"""Bounded, device-resident bremsstrahlung source records for a slab crossing.

This source diagnostic transports only the supplied charged primaries. It does not
transport photons, delta rays or annihilation photons, attenuate the source, or
implement target-region geometry. Records are births, never an exit photon flux.
The explicit capacity and transport-cap flags must be checked before using records.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def slab_source(key, tables, material_index, state, *, capacity=32, max_steps=10000,
                field=None, positron=False, msc=True):
    """Record brems births in a fixed per-parent buffer, without changing primary draws.

    Input and output position/direction use the slab frame (mm, MeV). The returned
    state and total emission counts include emissions beyond capacity: records are
    not complete when overflow is nonzero. Unfinished lanes at max_steps are marked
    capped. No host callbacks, event loops or host-side lane compaction are used.
    Requires the optional lane_step_photons core interface.
    """
    from mu2efastsim import eslab_jax as sj
    capacity = _positive_integer(capacity, "capacity")
    max_steps = _positive_integer(max_steps, "max_steps")
    if not hasattr(sj, "lane_step_photons"):
        raise RuntimeError("Core lane_step_photons is required for explicit photon births")
    n = state["T"].shape[0]
    if n < 1:
        raise ValueError("At least one parent lane is required")
    mi = np.asarray(material_index)
    if mi.shape != (n,) or mi.dtype.kind not in "iu" or np.any(mi < 0) or np.any(mi >= len(tables.length)):
        raise ValueError("material_index must contain one valid layer index per parent")
    mi = jnp.asarray(mi, jnp.int32)
    dt = state["T"].dtype
    if field is not None and np.shape(field) != (n, 3):
        raise ValueError("field must have shape (parents, 3)")

    @jax.jit
    def run(key, state):
        record = dict(energy=jnp.zeros((n, capacity), dt), position=jnp.zeros((n, capacity, 3), dt),
                      direction=jnp.zeros((n, capacity, 3), dt), path=jnp.zeros((n, capacity), dt),
                      valid=jnp.zeros((n, capacity), bool))
        count = jnp.zeros(n, jnp.int32)

        def cond(carry):
            i, st, _, _ = carry
            return (i < max_steps) & jnp.any(~st["stopped"] & ~st["exited"])

        def body(carry):
            i, st, rec, count = carry
            st, emitted = sj.lane_step_photons(key, tables, mi, st, B=field, positron=positron, msc=msc)
            write = emitted.valid & (count < capacity)
            row = jnp.arange(n)
            slot = jnp.minimum(count, capacity - 1)
            updated = {}
            for name in ("energy", "position", "direction", "path", "valid"):
                value = getattr(emitted, name)
                use = write[:, None] if value.ndim == 2 else write
                value = jnp.where(use, value, rec[name][row, slot])
                updated[name] = rec[name].at[row, slot].set(value)
            return i + 1, st, updated, count + emitted.valid

        steps, final, records, counts = jax.lax.while_loop(cond, body, (jnp.int32(0), state, record, count))
        return dict(state=final, records=records, count=counts,
                    overflow=jnp.maximum(counts - capacity, 0),
                    capped=~final["stopped"] & ~final["exited"], steps=steps)

    return run(key, state)


def validate_complete(result):
    """Fail closed if any emitted birth was dropped or a charged primary is unfinished."""
    overflow, capped = np.asarray(result["overflow"]), np.asarray(result["capped"])
    if np.any(overflow) or np.any(capped):
        raise ValueError(f"Incomplete source: dropped births={int(overflow.sum())}, capped parents={int(capped.sum())}")
    return result
