"""Jittable stock inverse CDF with host-validated tables and supplied uniforms."""
import jax.numpy as jnp


def propagation_time(depth, uniforms, cdf, times, dz):
    """Mirror propagation.py; host must validate tables and numeric inputs first.

    Static CDF shape bounds a vectorized binary search, not a per-photon loop.
    Validate the truncated float32 depth/dz quotient fits signed int32 on the host.
    No global JAX config mutation, random generation or table approximation.
    """
    cdf = jnp.asarray(cdf, dtype=jnp.float32)
    times = jnp.asarray(times, dtype=jnp.float32)
    depth, u = jnp.broadcast_arrays(jnp.asarray(depth, dtype=jnp.float32),
                                    jnp.asarray(uniforms, dtype=jnp.float32))
    row = jnp.clip(jnp.trunc(depth / jnp.float32(dz)), 0, cdf.shape[0] - 1).astype(jnp.int32)
    lo = jnp.zeros(row.shape, dtype=jnp.int32)
    hi = jnp.full(row.shape, cdf.shape[1], dtype=jnp.int32)
    for _ in range(cdf.shape[1].bit_length()):
        mid = (lo + hi) // 2
        less = cdf[row, jnp.minimum(mid, cdf.shape[1] - 1)] < u
        active = lo < hi
        lo = jnp.where(active & less, mid + 1, lo)
        hi = jnp.where(active & ~less, mid, hi)
    return times[jnp.minimum(lo, times.shape[0] - 1)]
