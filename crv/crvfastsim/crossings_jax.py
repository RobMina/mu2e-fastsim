"""Bounded, device-resident ray/box crossings, ordered by entry distance per track.

No dense crossing array is copied to the host. The caller checks overflow before
publishing tables; capacity is a resource limit, never a physics cut.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Crossings(NamedTuple):
    bar: object
    entry: object
    exit: object
    valid: object
    overflow: object


def slab(origin, direction, lo, hi):
    """Dense slab parameters on the device; same parallel-ray convention as core."""
    inv = 1.0 / jnp.where(direction == 0., 1e-300, direction)
    a = (lo[None] - origin[:, None]) * inv[:, None]
    b = (hi[None] - origin[:, None]) * inv[:, None]
    return jnp.minimum(a, b).max(-1), jnp.maximum(a, b).min(-1)


@partial(jax.jit, static_argnames=("capacity",))
def crossings(origin, direction, lo, hi, capacity=64, active=None):
    """Padded [track, capacity] crossings; ties retain ascending bar order."""
    ti, to = slab(origin, direction, lo, hi)
    ti = jnp.maximum(ti, 0.)
    hit = to > ti
    if active is not None:
        hit &= active[:, None]
    _, bar = jax.lax.top_k(-jnp.where(hit, ti, jnp.inf), min(capacity, lo.shape[0]))
    entry = jnp.take_along_axis(ti, bar, axis=1)
    exit_ = jnp.take_along_axis(to, bar, axis=1)
    valid = jnp.take_along_axis(hit, bar, axis=1)
    return Crossings(bar, jnp.where(valid, entry, 0.), jnp.where(valid, exit_, 0.), valid,
                     jnp.any(hit.sum(1) > capacity))
