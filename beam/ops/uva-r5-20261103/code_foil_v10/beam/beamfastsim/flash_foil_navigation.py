"""Standalone foil-candidate competition, not enabled in a transport navigator.

Each foil keeps its own index even when slab material/thickness types coincide.
Candidate queries are broadcast over lanes x foils, never endpoint-only radial
membership tests. Any unresolved foil query refuses the whole lane because this
interface has no certified lower bound on that unknown intersection. A tie or
near tie between foils also refuses instead of silently picking array order.
The caller must still compete the answer against nonfoil boundaries. Approximate
curved endpoints are not automatically exact-surface continuation receipts.
"""
import numpy as np

from . import flash_foil_helix as H


def _indices(value, nfoil, name, shape):
    a = np.asarray(value)
    if a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer foil indices")
    np.broadcast_shapes(shape, a.shape)
    if np.any(a < -1) or np.any(a >= nfoil):
        raise ValueError(f"{name} outside foil indices or -1 sentinel")


def validate_inputs(x, u, p, q, B, foil_z, foil_ht, foil_rin, foil_rout, *,
                    axis_x=0., s_max=1.e6, current_foil=-1, consumed_foil=-1,
                    after_surface=0, tolerance=None, max_intervals=512):
    """Host geometry/identity checks before NumPy use or JAX tracing."""
    arrays = [np.asarray(v) for v in (foil_z, foil_ht, foil_rin, foil_rout)]
    if any(a.ndim != 1 or not a.size or a.dtype.kind not in "iuf" or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Foil geometry must be finite nonempty numeric vectors")
    z, ht, ri, ro = arrays
    if any(a.shape != z.shape for a in arrays) or np.any(ht <= 0):
        raise ValueError("Foil shapes or half thicknesses invalid")
    if np.any(np.diff(z) <= 0) or np.any(z[1:]-ht[1:] <= z[:-1]+ht[:-1]):
        raise ValueError("Foils must be sorted with disjoint z intervals")
    x, u, B = (np.asarray(v) for v in (x, u, B))
    if any(a.ndim < 1 or a.shape[-1] != 3 for a in (x, u, B)):
        raise ValueError("x, u and B require final xyz dimension")
    lane_shape = np.broadcast_shapes(x.shape[:-1], u.shape[:-1], B.shape[:-1],
                                      np.shape(p), np.shape(q), np.shape(s_max),
                                      np.shape(current_foil), np.shape(consumed_foil), np.shape(after_surface),
                                      np.shape(axis_x), np.shape(0 if tolerance is None else tolerance))
    _indices(current_foil, z.size, "current_foil", lane_shape)
    _indices(consumed_foil, z.size, "consumed_foil", lane_shape)
    token = np.asarray(after_surface)
    if token.dtype.kind not in "iu" or np.any(token < 0) or np.any(token > 4):
        raise ValueError("after_surface must be integer surface kinds")
    if np.any((token != 0) != (np.asarray(consumed_foil) >= 0)):
        raise ValueError("Consumed foil and surface must be supplied together")
    H.validate_inputs(np.broadcast_to(x, lane_shape+(3,))[..., None, :],
                      np.broadcast_to(u, lane_shape+(3,))[..., None, :],
                      np.broadcast_to(p, lane_shape)[..., None], np.broadcast_to(q, lane_shape)[..., None],
                      np.broadcast_to(B, lane_shape+(3,))[..., None, :], ri, ro, z-ht, z+ht,
                      axis_x=np.broadcast_to(axis_x, lane_shape)[..., None],
                      s_max=np.broadcast_to(s_max, lane_shape)[..., None],
                      exiting=np.arange(z.size) == np.broadcast_to(current_foil, lane_shape)[..., None],
                      tolerance=np.broadcast_to(0 if tolerance is None else tolerance, lane_shape)[..., None],
                      max_intervals=max_intervals)


def first_foil(x, u, p, q, B, foil_z, foil_ht, foil_rin, foil_rout, *, axis_x=0.,
               s_max=1.e6, current_foil=-1, consumed_foil=-1, after_surface=0,
               tolerance=None, max_intervals=512):
    """Find first face/rim/hole candidate with unique foil identity; not transport."""
    validate_inputs(x, u, p, q, B, foil_z, foil_ht, foil_rin, foil_rout,
                    axis_x=axis_x, s_max=s_max, current_foil=current_foil,
                    consumed_foil=consumed_foil, after_surface=after_surface,
                    tolerance=tolerance, max_intervals=max_intervals)
    return _first(np, x, u, p, q, B, foil_z, foil_ht, foil_rin, foil_rout, axis_x,
                  s_max, current_foil, consumed_foil, after_surface, tolerance, max_intervals)


def first_foil_jax(x, u, p, q, B, foil_z, foil_ht, foil_rin, foil_rout, *, axis_x=0.,
                   s_max=1.e6, current_foil=-1, consumed_foil=-1, after_surface=0,
                   tolerance=None, max_intervals=512):
    """Optional JIT primitive; validate host inputs first; budget is static."""
    import jax.numpy as jnp
    return _first(jnp, x, u, p, q, B, foil_z, foil_ht, foil_rin, foil_rout, axis_x,
                  s_max, current_foil, consumed_foil, after_surface, tolerance, max_intervals)


def select_first(distance, kind, valid, unresolved, *, tolerance=0.):
    """Select foil axis last, refusing unresolved competitors or numerical ties."""
    distance, kind, valid, unresolved = map(np.asarray, (distance, kind, valid, unresolved))
    if (distance.ndim < 1 or not distance.shape[-1] or distance.dtype.kind != "f" or
            any(a.shape != distance.shape for a in (kind, valid, unresolved)) or kind.dtype.kind not in "iu" or
            valid.dtype.kind != "b" or unresolved.dtype.kind != "b"):
        raise ValueError("Candidate arrays require matching nonempty foil axis and numeric/bool dtypes")
    tol = np.asarray(tolerance)
    if tol.dtype.kind not in "iuf" or not np.all(np.isfinite(tol)) or np.any(tol < 0):
        raise ValueError("Candidate tolerance must be finite and nonnegative")
    np.broadcast_shapes(distance.shape[:-1], tol.shape)
    return _select(np, distance, kind, valid, unresolved, tolerance)


def _select(xp, distance, kind, valid, unresolved, tolerance):
    distance, kind = xp.asarray(distance), xp.asarray(kind)
    valid, unresolved = xp.asarray(valid, bool), xp.asarray(unresolved, bool)
    inf = xp.asarray(xp.inf, distance.dtype)
    malformed = ((valid & (~xp.isfinite(distance) | (distance < 0) | (kind < 1) | (kind > 4))) |
                 (valid & unresolved) | (~valid & ((distance != inf) | (kind != 0))))
    eligible = xp.where(valid, distance, inf)
    first = xp.min(eligible, axis=-1)
    found = xp.isfinite(first)
    eps = xp.finfo(distance.dtype).eps
    guard = xp.maximum(xp.asarray(tolerance, distance.dtype), 64*eps*xp.maximum(1, xp.where(found, first, 0)))
    # A named tie does not grant permission to choose either foil by index order.
    near = valid & (xp.abs(distance-xp.where(found, first, 0)[..., None]) <= guard[..., None])
    bad = xp.any(unresolved | malformed, axis=-1) | (found & (xp.sum(near, axis=-1) != 1))
    idx = xp.argmin(eligible, axis=-1).astype(xp.int32)
    chosen_kind = xp.take_along_axis(kind, idx[..., None], axis=-1)[..., 0]
    good = found & ~bad
    return dict(distance=xp.where(good, first, inf), foil_id=xp.where(good, idx, -1).astype(xp.int32),
                kind=xp.where(good, chosen_kind, 0).astype(xp.int32), valid=good, unresolved=bad)


def _first(xp, x, u, p, q, B, z, ht, ri, ro, ax, cap, current, consumed, token, tol, budget):
    x, u, B = [xp.asarray(a) for a in (x, u, B)]
    dtype = xp.result_type(x.dtype, u.dtype, xp.float32)
    x, u, B = [a.astype(dtype) for a in (x, u, B)]
    values = xp.broadcast_arrays(*[a[..., i] for a in (x, u, B) for i in range(3)],
                                 *[xp.asarray(a) for a in (p, q, cap, current, consumed, token, ax,
                                                           0 if tol is None else tol)])
    x, u, B = [xp.stack(values[i:i+3], axis=-1) for i in (0, 3, 6)]
    p, q, cap, current, consumed, token, ax, tol = values[9:]
    z, ht, ri, ro = [xp.asarray(a, dtype) for a in (z, ht, ri, ro)]
    ids = xp.arange(z.size, dtype=xp.int32)
    own = current[..., None] == ids
    tokens = xp.where(consumed[..., None] == ids, token[..., None], 0).astype(xp.int32)
    fn = H.first_transition if xp is np else H.first_transition_jax
    candidates = fn(x[..., None, :], u[..., None, :], p[..., None], q[..., None], B[..., None, :],
                    ri, ro, z-ht, z+ht, axis_x=ax[..., None], s_max=cap[..., None], exiting=own,
                    after_surface=tokens, tolerance=tol[..., None], max_intervals=budget)
    low_current = xp.zeros_like(current, bool) if current.dtype.kind == "u" else current < -1
    low_consumed = xp.zeros_like(consumed, bool) if consumed.dtype.kind == "u" else consumed < -1
    bad_id = (low_current | (current >= z.size) | low_consumed | (consumed >= z.size) |
              (token < 0) | (token > 4) | ((token != 0) != (consumed >= 0)))
    bad_id = bad_id | (current != xp.floor(current)) | (consumed != xp.floor(consumed)) | (token != xp.floor(token))
    malformed_geometry = (xp.any(~xp.isfinite(z+ht+ri+ro)) | xp.any(ht <= 0) | xp.any(ri < 0) |
                          xp.any(ro <= ri) | xp.any(z[1:]-ht[1:] <= z[:-1]+ht[:-1]))
    bad_id = bad_id | malformed_geometry
    result = _select(xp, candidates["distance"], candidates["kind"], candidates["valid"],
                     candidates["unresolved"], 0 if tol is None else tol)
    result["unresolved"] = result["unresolved"] | bad_id
    result["valid"] = result["valid"] & ~result["unresolved"]
    result["distance"] = xp.where(result["valid"], result["distance"], xp.inf)
    result["foil_id"] = xp.where(result["valid"], result["foil_id"], -1).astype(xp.int32)
    result["kind"] = xp.where(result["valid"], result["kind"], 0).astype(xp.int32)
    return result
