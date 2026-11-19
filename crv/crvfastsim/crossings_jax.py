"""Bounded, device-resident ray/box crossings, ordered by entry distance per track.

No dense crossing array is copied to the host. The caller checks overflow before
publishing tables; capacity is a resource limit, never a physics cut.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mu2efastsim import precision as dprec

dprec.register(__name__)   # JAX creators/samplers here follow MU2EFS_DEVICE_FLOAT

jax.config.update("jax_enable_x64", True)


class Crossings(NamedTuple):
    bar: object
    entry: object
    exit: object
    valid: object
    overflow: object


def _slab(origin, direction, lo, hi):
    inv = (1.0 / jnp.where(direction == 0., dprec.TINY, direction))[:, None]
    a = (lo - origin[:, None]) * inv
    b = (hi - origin[:, None]) * inv
    return jnp.minimum(a, b).max(-1), jnp.maximum(a, b).min(-1)


def slab(origin, direction, lo, hi):
    """Dense slab parameters on the device; same parallel-ray convention as core."""
    return _slab(origin, direction, lo[None], hi[None])


@partial(dprec.jit, static_argnames=("capacity",))
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


class ModuleIndex(NamedTuple):
    lo: object
    hi: object
    bars: object


def module_index(lo, hi, sector, module):
    """Conservative boxes and padded global bar IDs for each (sector, module)."""
    _, group = np.unique(np.column_stack((sector, module)), axis=0, return_inverse=True)
    group = group.reshape(-1)
    counts = np.bincount(group)
    mlo = np.full((len(counts), 3), np.inf)
    mhi = np.full_like(mlo, -np.inf)
    np.minimum.at(mlo, group, lo)
    np.maximum.at(mhi, group, hi)
    bars = np.full((len(counts), max(counts)), -1, np.int32)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    bars[np.repeat(np.arange(len(counts)), counts), np.arange(len(group)) - start] = np.argsort(group, kind="stable")
    return ModuleIndex(jnp.asarray(mlo), jnp.asarray(mhi), jnp.asarray(bars))


@partial(dprec.jit, static_argnames=("capacity", "module_slots"))
def crossings_by_module(origin, direction, lo, hi, index, capacity=64, active=None, module_slots=2):
    """Use exact bar slabs in candidate modules; a crowded batch takes the dense route."""
    if module_slots < 1:
        raise ValueError("indexed crossings need at least one module slot")
    width = min(capacity, lo.shape[0])
    slots = min(module_slots, index.bars.shape[0])
    if slots * index.bars.shape[1] >= lo.shape[0]:
        return crossings(origin, direction, lo, hi, capacity, active)

    ti, to = slab(origin, direction, index.lo, index.hi)
    hit = to > jnp.maximum(ti, 0.)
    if active is not None:
        hit &= active[:, None]

    def sparse():
        present, modules = jax.lax.top_k(hit.astype(jnp.int32), slots)
        bar = jnp.where(present[..., None] != 0, index.bars[modules], -1).reshape(len(origin), -1)
        safe = jnp.maximum(bar, 0)
        start, end = _slab(origin, direction, lo[safe], hi[safe])
        start = jnp.maximum(start, 0.)
        valid = (bar >= 0) & (end > start)
        overflow = jnp.any(valid.sum(1) > capacity)
        pad = max(0, width - bar.shape[1])
        # A tie must use the global bar ID, not the position in the gathered modules.
        start, bar, end = jax.lax.sort(
            (jnp.pad(jnp.where(valid, start, jnp.inf), ((0, 0), (0, pad)), constant_values=jnp.inf),
             jnp.pad(safe.astype(jnp.int32), ((0, 0), (0, pad))),
             jnp.pad(end, ((0, 0), (0, pad)))),
            num_keys=2,
        )
        start, bar, end = start[:, :width], bar[:, :width], end[:, :width]
        keep = start != jnp.inf
        return Crossings(bar, jnp.where(keep, start, 0.), jnp.where(keep, end, 0.), keep, overflow)

    return jax.lax.cond(jnp.any(hit.sum(1) > slots),
                        lambda: crossings(origin, direction, lo, hi, capacity, active), sparse)
