"""Traceable local-leg to unique-foil query adapter; no source integration.

A query returns (distance, unresolved), requiring a future transactional core API.
The existing distance-only hook cannot consume unresolved status safely.
No-hit is positive infinity only when resolved. The closure binds a validated
host table and a retained proper-rotation entry frame, never layer type alone.
entry_surface is an explicit consumed EXACT surface receipt at frame.local_x;
it is used only on a leg starting bytewise equal in value to that local point.
Approximate curved endpoints are not receipts and are never projected/snapped.
The closure is built for the supplied gathered rows; rebuild after any gather.
"""
import numpy as np

from . import flash_foil_frames as F
from . import flash_foil_helix as H


def make_query(table, foil_id, frame, *, entry_surface=0, tolerance=None, max_intervals=512):
    """Build a JAX callback for fixed gathered foil lanes, retaining explicit refusals.

    Frames/table identity and entry_surface arrays must correspond to these rows.
    No finite geometry search runs during construction. The callback transforms
    each actual straight/helix leg and its local field into the global foil frame.
    Its answer concerns one finite annulus only; nonfoil boundaries still belong
    to the caller/core. A refused lane must be marked invalid, not silently retried.
    """
    if isinstance(max_intervals, (bool, np.bool_)) or not isinstance(max_intervals, (int, np.integer)) or max_intervals < 1:
        raise ValueError("max_intervals must be a positive static integer")
    token = np.asarray(entry_surface)
    if token.dtype.kind not in "iu" or np.any(token < 0) or np.any(token > 4):
        raise ValueError("entry_surface must be integer surface kinds")
    F.gather(table, foil_id)
    required = ("basis", "anchor", "local_x", "foil_id", "valid")
    if any(name not in frame for name in required):
        raise ValueError("Incomplete retained entry frame")
    if tolerance is not None:
        tol = np.asarray(tolerance)
        if tol.dtype.kind not in "iuf" or not np.all(np.isfinite(tol)) or np.any(tol < 0):
            raise ValueError("tolerance must be finite and nonnegative")
    def query(xx, uu, pm, q, B, smax):
        return leg_query_jax(table, foil_id, frame, xx, uu, pm, q, B, smax,
                             entry_surface=token, tolerance=tolerance, max_intervals=max_intervals)
    return query


def leg_query_jax(table, foil_id, frame, xx, uu, pm, q, B, smax, *, entry_surface=0,
                  tolerance=None, max_intervals=512):
    """Pure JAX leg query, including dynamic gathered foil IDs/frames.

    The table and static search budget are host-validated by the caller. Gather
    all frame leaves and foil IDs together; this function checks their matching
    IDs and preserves sentinel refusals. Numeric surface receipts remain exact.
    """
    import jax.numpy as jnp
    row = F.gather_jax(table, foil_id)
    local_x, local_u = jnp.asarray(xx), jnp.asarray(uu)
    dtype = local_x.dtype
    xyz = F.to_global_jax(frame, local_x, point=True)
    direction = F.to_global_jax(frame, local_u)
    local_B = jnp.zeros_like(local_x) if B is None else jnp.broadcast_to(jnp.asarray(B, dtype), local_x.shape)
    field = F.to_global_jax(frame, local_B)
    start = jnp.all(local_x == jnp.asarray(frame["local_x"], dtype), axis=-1)
    after = jnp.where(start, jnp.asarray(entry_surface), 0)
    ri, ro, z, ht, ax = [jnp.asarray(row[name], dtype) for name in
                        ("foil_rin", "foil_rout", "foil_z", "foil_ht", "axis_x")]
    result = H.first_transition_jax(xyz, direction, pm, q, field, ri, ro, z-ht, z+ht,
                                     axis_x=ax, s_max=smax, exiting=True, after_surface=after,
                                     tolerance=tolerance, max_intervals=max_intervals)
    identity = jnp.asarray(row["foil_id"]) == jnp.asarray(frame["foil_id"])
    good = jnp.asarray(row["valid"]) & jnp.asarray(frame["valid"]) & identity
    unresolved = result["unresolved"] | ~good
    return jnp.where(result["valid"] & ~unresolved, result["distance"], jnp.inf), unresolved
