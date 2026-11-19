"""Stock C++ arithmetic on the device: float64 for the double steps in either precision of the chain, and products
that the compiler cannot fuse into the add that follows them.

XLA's CPU backend contracts a * b + c into one fused multiply-add, with no flag to turn that off; the stock build
rounds the product first. `g` ORs a product's bits with a zero the compiler cannot see through (min(|ints|, 0) of
an input), so the product is rounded on its own. The zero is computed once per trace, outside any loop: an opaque
zero made inside a loop body gets folded.
"""
from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp

F32 = jnp.float32
F64 = jnp.float64
_ZERO = []                           # (int32, int64) zeros of the current trace, opaque to the compiler


@contextlib.contextmanager
def strict(zsrc):
    """Trace stock-exact code: float64 on (the stock double steps, in either precision of the chain) and the product
    guard armed. `zsrc` is any int array of the inputs."""
    with jax.enable_x64(True):
        z = jnp.minimum(jnp.min(jnp.abs(jnp.ravel(jnp.asarray(zsrc)).astype(jnp.int32))), 0)
        _ZERO.append((z, z.astype(jnp.int64)))
        try:
            yield
        finally:
            _ZERO.pop()


def g(p):
    """p, rounded as it is: a value the compiler cannot fuse into a following add."""
    z32, z64 = _ZERO[-1]
    p = jnp.asarray(p)
    if p.dtype == F64:
        return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(p, jnp.int64) | z64, F64)
    return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(p, jnp.int32) | z32, F32)


def m(*xs):
    """Left-to-right product, rounded after every factor as in C++, and never fused into a following add."""
    p = xs[0]
    for x in xs[1:]:
        p = p * x
    return g(jnp.asarray(p))
