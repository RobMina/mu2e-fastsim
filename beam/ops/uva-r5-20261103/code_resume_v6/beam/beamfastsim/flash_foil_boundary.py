"""Finite-annulus straight-ray prerequisite, not foil transport or S1 closure.

The solid is zlo <= z <= zhi and rin <= hypot(x-axis_x, y) <= rout.
Only x(s) = x + s*u is represented; a helix/MSC trajectory is NOT a ray.
``exiting=False`` requires a vacuum start, ``True`` a material start. At an
exact surface start the *backward* ray limit defines that initial state, so
inward entry and outward exit both occur at distance zero. Tangencies do not
change material. Coincident/ill-conditioned surfaces are refused, not nudged.

Results are broadcast lane arrays: distance (inf unless valid), kind (integer
NONE/ZLO/ZHI/RIN/ROUT), valid, unresolved. No hit has both flags false;
unresolved is not a certified no-hit. The finite nonnegative s_max is inclusive.
Coincident corner candidates have no unique surface kind and are unresolved. Positive material or
vacuum intervals narrower than tolerance are unresolved, not discarded.

Coordinates/directions select float32 or float64; geometry is cast to that
dtype. All leading dimensions broadcast, including per-target geometry. Use
x[..., None, :] for a ray-by-target outer product. tolerance is an absolute
length, augmented by floating-point error bounds (default: error bounds only).
NumPy performs host validation; JAX is optional and imported only on use.
Call validate_inputs on host values before tracing first_transition_jax; the
latter is a pure array primitive, with malformed numeric lanes also refused.
This conservative numerical guard is not a proof for arbitrary floating-point
conditioning and supplies no curved navigation or transport validation.
"""

import numpy as np

NONE, ZLO, ZHI, RIN, ROUT = range(5)
KIND_NAMES = ("none", "zlo", "zhi", "rin", "rout")


def validate_inputs(x, u, rin, rout, zlo, zhi, *, axis_x=0.0,
                    s_max=1.0e6, exiting=False, tolerance=None):
    """Validate finite host inputs, broadcasting and unit directions; return None."""
    x, u = np.asarray(x), np.asarray(u)
    if x.ndim < 1 or u.ndim < 1 or x.shape[-1] != 3 or u.shape[-1] != 3:
        raise ValueError("x and u must have final dimension 3")
    values = [x, u, rin, rout, zlo, zhi, axis_x, s_max]
    if tolerance is not None:
        values.append(tolerance)
    if any(np.asarray(v).dtype.kind not in "iuf" for v in values):
        raise ValueError("coordinates and geometry must be real numeric arrays")
    dtype = np.result_type(x.dtype, u.dtype, np.float32)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("only float32 and float64 rays are supported")
    with np.errstate(over="ignore", invalid="ignore"):
        arrays = [np.asarray(v, dtype=dtype) for v in values]
    if any(not np.all(np.isfinite(v)) for v in arrays):
        raise ValueError("inputs must be finite in the ray dtype")
    xx, uu, ri, ro, zl, zh, ax, sm = arrays[:8]
    tol = np.asarray(0 if tolerance is None else tolerance, dtype=dtype)
    ex = np.asarray(exiting)
    if ex.dtype.kind != "b":
        raise ValueError("exiting must be boolean")
    np.broadcast_shapes(xx.shape[:-1], uu.shape[:-1], ri.shape, ro.shape,
                        zl.shape, zh.shape, ax.shape, sm.shape, tol.shape, ex.shape)
    if np.any(ri < 0) or np.any(ro <= ri) or np.any(zl >= zh):
        raise ValueError("require 0 <= rin < rout and zlo < zhi")
    if np.any(sm < 0) or np.any(tol < 0):
        raise ValueError("s_max and tolerance must be nonnegative")
    norm = np.sqrt(np.sum(uu * uu, axis=-1))
    if np.any(np.abs(norm - 1) > 32 * np.finfo(dtype).eps):
        raise ValueError("u must be a unit straight-ray direction")


def first_transition(x, u, rin, rout, zlo, zhi, *, axis_x=0.0,
                     s_max=1.0e6, exiting=False, tolerance=None):
    """Validated NumPy first entry/exit; see module contract for refusal semantics."""
    validate_inputs(x, u, rin, rout, zlo, zhi, axis_x=axis_x,
                    s_max=s_max, exiting=exiting, tolerance=tolerance)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return _transition(np, x, u, rin, rout, zlo, zhi, axis_x,
                           s_max, exiting, tolerance)


def first_transition_jax(x, u, rin, rout, zlo, zhi, *, axis_x=0.0,
                         s_max=1.0e6, exiting=False, tolerance=None):
    """JIT-compatible optional JAX primitive; validate host inputs before tracing."""
    import jax.numpy as jnp
    return _transition(jnp, x, u, rin, rout, zlo, zhi, axis_x,
                       s_max, exiting, tolerance)


def _transition(xp, x, u, rin, rout, zlo, zhi, axis_x, s_max, exiting, tolerance):
    x, u = xp.asarray(x), xp.asarray(u)
    dtype = xp.result_type(x.dtype, u.dtype, xp.float32)
    x, u = x.astype(dtype), u.astype(dtype)
    p, y, z, dx, dy, dz, ri, ro, zl, zh, ax, cap, ex, tol = xp.broadcast_arrays(
        x[..., 0], x[..., 1], x[..., 2], u[..., 0], u[..., 1], u[..., 2],
        xp.asarray(rin, dtype), xp.asarray(rout, dtype), xp.asarray(zlo, dtype),
        xp.asarray(zhi, dtype), xp.asarray(axis_x, dtype), xp.asarray(s_max, dtype),
        xp.asarray(exiting, bool), xp.asarray(0 if tolerance is None else tolerance, dtype))
    eps, inf = xp.finfo(dtype).eps, xp.asarray(xp.inf, dtype)
    finite = xp.ones_like(p, dtype=bool)
    for v in (p, y, z, dx, dy, dz, ri, ro, zl, zh, ax, cap, tol):
        finite = finite & xp.isfinite(v)
    norm2 = dx * dx + dy * dy + dz * dz
    malformed = (~finite | (ri < 0) | (ro <= ri) | (zl >= zh) | (cap < 0)
                 | (tol < 0) | (xp.abs(norm2 - 1) > 64 * eps))
    scale = xp.maximum(1, xp.maximum(xp.abs(p), xp.abs(ax)))
    for v in (y, z, ro, zl, zh):
        scale = xp.maximum(scale, xp.abs(v))
    base_tol = xp.maximum(tol, 32 * eps * scale)
    p = p - ax
    a = dx * dx + dy * dy
    b = p * dx + y * dy
    cross = p * dy - y * dx
    radius = xp.hypot(p, y)
    transverse = a > 0
    safe_a = xp.where(transverse, a, 1)
    vertex = -b / safe_a
    zp = dz != 0
    safe_dz = xp.where(zp, dz, 1)
    tzl, tzh = (zl - z) / safe_dz, (zh - z) / safe_dz
    zin = (z >= zl) & (z <= zh)
    za = xp.where(zp, xp.minimum(tzl, tzh), xp.where(zin, -inf, inf))
    zb = xp.where(zp, xp.maximum(tzl, tzh), xp.where(zin, inf, -inf))
    tzl, tzh = xp.where(zp, tzl, inf), xp.where(zp, tzh, inf)
    numeric_bad = ~xp.isfinite(a + b + cross + radius + p)
    numeric_bad = numeric_bad | ((a == 0) & ((dx != 0) | (dy != 0)))

    def cylinder(r, enabled):
        rr = a * r * r
        cc = cross * cross
        disc = rr - cc
        err = 32 * eps * (xp.abs(rr) + cc + (xp.abs(p * dy) + xp.abs(y * dx)) ** 2)
        near = transverse & (xp.abs(disc) <= err) & enabled
        real = transverse & (disc >= 0) & enabled
        root = xp.sqrt(xp.maximum(disc, 0))
        q = -b - xp.where(b >= 0, root, -root)
        c = (radius - r) * (radius + r)
        r1 = q / safe_a
        r2 = c / xp.where(q != 0, q, 1)
        r2 = xp.where(q != 0, r2, vertex)
        lo, hi = xp.minimum(r1, r2), xp.maximum(r1, r2)
        static_in = (~transverse) & (radius <= r) & enabled
        lo = xp.where(real, lo, xp.where(static_in, -inf, inf))
        hi = xp.where(real, hi, xp.where(static_in, inf, -inf))
        # A near-zero discriminant is not certified as an empty chord.
        collapsed = real & (disc > 0) & (hi <= lo)
        suspect = near | collapsed
        danger = suspect & (hi >= -base_tol) & (lo <= cap + base_tol)
        danger = danger & (hi >= za - base_tol) & (lo <= zb + base_tol)
        danger = danger | (near & ~real & (vertex >= -base_tol)
                           & (vertex <= cap + base_tol) & (vertex >= za - base_tol)
                           & (vertex <= zb + base_tol))
        parallel = (~transverse) & enabled & (xp.abs(radius - r) <= base_tol)
        danger = danger | (parallel & (zb >= 0) & (za <= cap))
        bad = enabled & ~xp.isfinite(rr + cc + disc + err + c)
        bad = bad | (real & (~xp.isfinite(r1) | ~xp.isfinite(r2)))
        danger_at = xp.maximum(xp.minimum(vertex, lo), 0)
        uncertainty = err / (safe_a * xp.where(root > 0, root, 1))
        uncertainty = xp.maximum(base_tol, uncertainty + 32 * eps * (xp.abs(vertex) + root / safe_a))
        return lo, hi, xp.where(danger, xp.minimum(danger_at, cap), inf), bad, uncertainty

    oa, ob, outer_bad_at, outer_numeric, outer_tol = cylinder(ro, xp.ones_like(ex))
    ia, ib, inner_bad_at, inner_numeric, inner_tol = cylinder(ri, ri > 0)
    numeric_bad = numeric_bad | outer_numeric | inner_numeric
    roots = xp.stack((tzl, tzh, ia, ib, oa, ob), axis=-1)
    kinds = xp.broadcast_to(xp.asarray((ZLO, ZHI, RIN, RIN, ROUT, ROUT), xp.int32), roots.shape)
    plane_tol = base_tol / xp.abs(safe_dz)
    uncertainty = xp.stack((plane_tol, plane_tol, inner_tol, inner_tol, outer_tol, outer_tol), axis=-1)
    # Infinite endpoints describe intervals, not candidate transitions.
    order = xp.argsort(roots, axis=-1)
    roots = xp.take_along_axis(roots, order, axis=-1)
    kinds = xp.take_along_axis(kinds, order, axis=-1)
    uncertainty = xp.take_along_axis(uncertainty, order, axis=-1)

    def material(t, after):
        def in_interval(lo, hi):
            lo, hi = lo[..., None], hi[..., None]
            if after:
                return (t >= lo) & (t < hi)
            return (t > lo) & (t <= hi)
        return in_interval(za, zb) & in_interval(oa, ob) & ~in_interval(ia, ib)

    before, after = material(roots, False), material(roots, True)
    crossing = (before != after) & xp.isfinite(roots)
    eligible = crossing & (roots >= 0) & (roots <= cap[..., None])
    ts = xp.where(eligible, roots, inf)
    first = xp.min(ts, axis=-1)
    index = xp.argmin(ts, axis=-1)[..., None]
    kind = xp.take_along_axis(kinds, index, axis=-1)[..., 0]
    actual_exit = xp.take_along_axis(before, index, axis=-1)[..., 0]
    hit = xp.isfinite(first)
    rt = xp.maximum(uncertainty, 32 * eps * xp.abs(roots))
    finite_roots = xp.isfinite(roots)
    # Nonzero near-origin roots cannot safely be snapped to surface starts.
    near_zero = crossing & (roots != 0) & (xp.abs(roots) <= rt)
    near_cap = crossing & (roots != cap[..., None]) & (xp.abs(roots - cap[..., None]) <= rt)
    boundary_at = xp.minimum(xp.maximum(roots, 0), cap[..., None])
    bad_at = xp.min(xp.where(near_zero | near_cap, boundary_at, inf), axis=-1)
    gap = roots[..., 1:] - roots[..., :-1]
    gap_tol = xp.maximum(rt[..., 1:], rt[..., :-1])
    relevant = crossing[..., 1:] | crossing[..., :-1] | after[..., :-1]
    relevant = relevant | (kinds[..., 1:] != kinds[..., :-1])
    short = (gap > 0) & (gap <= gap_tol) & relevant
    short = short & finite_roots[..., :-1] & finite_roots[..., 1:]
    short = short & (roots[..., 1:] >= 0) & (roots[..., :-1] <= cap[..., None])
    short_at = xp.min(xp.where(short, xp.maximum(roots[..., :-1], 0), inf), axis=-1)
    # Distinct surfaces at an exact corner cannot supply one crossing kind.
    corner = (gap == 0) & finite_roots[..., :-1] & finite_roots[..., 1:]
    corner = corner & (kinds[..., :-1] != kinds[..., 1:])
    corner_at = xp.min(xp.where(corner & (roots[..., :-1] >= 0), roots[..., :-1], inf), axis=-1)
    plane_parallel = (~zp) & ((xp.abs(z - zl) <= base_tol) | (xp.abs(z - zh) <= base_tol))
    plane_at = xp.where(plane_parallel & (ob >= 0) & (oa <= cap), xp.maximum(oa, 0), inf)
    for v in (short_at, corner_at, outer_bad_at, inner_bad_at, plane_at):
        bad_at = xp.minimum(bad_at, v)
    start_inside = material(xp.zeros_like(p)[..., None], False)[..., 0]
    unresolved = malformed | numeric_bad | (start_inside != ex)
    unresolved = unresolved | ((bad_at <= cap) & (bad_at <= first))
    unresolved = unresolved | (hit & (actual_exit != ex))
    valid = hit & ~unresolved
    return {"distance": xp.where(valid, first, inf),
            "kind": xp.where(valid, kind, NONE).astype(xp.int32),
            "valid": valid, "unresolved": unresolved}
