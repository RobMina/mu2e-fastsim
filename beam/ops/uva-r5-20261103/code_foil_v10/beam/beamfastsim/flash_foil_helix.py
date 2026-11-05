"""Standalone constant-vector-B finite-annulus navigation experiment, NOT transport.

Units/convention match eslab_jax: mm, MeV/c, T, du/ds = .299792458*q/p
(u cross B). B may be tilted; q is -1, 0, or +1. Directions must be unit.
Leading dimensions broadcast, including geometry, p, q, and B. The result has
ray-compatible distance/kind/valid/unresolved arrays. Only valid distances are
finite. No-hit means the entire inclusive [0,s_max] was certified clear of a
transition; unresolved is NEVER a no-hit. The backward limit owns a surface
start, and a transverse real transition at zero is returned without a nudge.
Optional integer-array after_surface=NONE retains that default. A non-NONE
kind names one already-consumed EXACT transverse start root; exiting then owns
the FORWARD side. All other clearances must be resolved and positive. Wrong
kinds, corners, tangents and approximate (even very close) receipts refuse.
Curved departure is certified by derivative-variation and other-face clearance
bounds over a positive interval before searching the remaining full cap. Only
the consumed root is excluded; later same-surface returns remain candidates.
An uncertifiable departure, including a zero or sub-resolution curved cap,
refuses rather than reporting a no-hit. No position/path nudge is used.

Integration gate: bisection endpoints generally have nonzero surface residual.
They are NOT exact receipts for this API. No implicit snapping/projection is
provided; a numerical receipt protocol needs separate design and validation.

Algorithm/contract: a chronological depth-first dyadic search covers the path.
For each interval of length h, |x''| <= A = |k| |u_perp|. Plane clearance has
second derivative bounded by A; cylindrical signed-distance clearance has bound
A + |u|^2/r_min, where r_min is a Lipschitz lower bound on transverse radius.
Thus endpoint chords enclose each clearance to M*h*h/8. An interval can be
skipped only if one constraint is strictly negative throughout (vacuum), or all
are strictly positive (material). This catches same-sign endpoint excursions.
An isolated crossing additionally requires a derivative of fixed nonzero sign
and every other constraint strictly positive throughout the bracket. Bisection
then returns an approximate distance, not an exact symbolic intersection.

Tolerance is absolute length (default zero), augmented by a floating-point guard
of 128*eps times coordinate/path/phase scales. It is a refusal/resolution scale,
NOT a material skip or a promise to resolve features smaller than it. Returned
curved distances have bisection bracket width at most that scale; intervals that
cannot be certified before that scale, or before max_intervals (default 512),
are refused. Tangent/parallel surface starts, near surfaces, corners, excessive
phase/conditioning and wrong initial ownership are refused. Some geometrically
harmless tangencies/nearby noncrossing surfaces are deliberately refused too.
The bounds prove coverage in real arithmetic; the guards are conservative
numerical engineering, NOT outward-rounded interval arithmetic or a proof for
arbitrary floating-point/libm conditioning. No finite sampling is called exact.
Zero force delegates to the existing conservative ray prerequisite. No navigator
or physics core uses this module. NumPy validates on host; for JAX validate host
inputs before tracing. max_intervals is a static positive integer under jit.
"""

import numpy as np

from . import flash_foil_boundary as ray
from .flash_foil_boundary import NONE, ZLO, ZHI, RIN, ROUT, KIND_NAMES

K_FIELD = 0.299792458


def validate_inputs(x, u, p, q, B, rin, rout, zlo, zhi, *, axis_x=0.,
                    s_max=1.e6, exiting=False, tolerance=None, max_intervals=512, after_surface=NONE):
    """Validate host arrays and broadcasting before NumPy use or JAX tracing."""
    ray.validate_inputs(x, u, rin, rout, zlo, zhi, axis_x=axis_x,
                        s_max=s_max, exiting=exiting, tolerance=tolerance, after_surface=after_surface)
    if isinstance(max_intervals, (bool, np.bool_)) or not isinstance(max_intervals, (int, np.integer)) or max_intervals < 1:
        raise ValueError("max_intervals must be a positive static integer")
    b = np.asarray(B)
    if b.ndim < 1 or b.shape[-1] != 3:
        raise ValueError("B must have final dimension 3")
    dtype = np.result_type(np.asarray(x).dtype, np.asarray(u).dtype, np.float32)
    vals = [np.asarray(v) for v in (p, q, B)]
    if any(v.dtype.kind not in "iuf" for v in vals):
        raise ValueError("p, q and B must be real numeric")
    with np.errstate(over="ignore", invalid="ignore"):
        pp, qq, bb = [v.astype(dtype) for v in vals]
    if not all(np.all(np.isfinite(v)) for v in (pp, qq, bb)) or np.any(pp <= 0):
        raise ValueError("require finite B, q and strictly positive finite p")
    if np.any((qq != -1) & (qq != 0) & (qq != 1)):
        raise ValueError("q must be -1, 0 or +1")
    np.broadcast_shapes(np.shape(x)[:-1], np.shape(u)[:-1], pp.shape, qq.shape,
                        bb.shape[:-1], np.shape(rin), np.shape(rout), np.shape(zlo),
                        np.shape(zhi), np.shape(axis_x), np.shape(s_max),
                        np.shape(exiting), np.shape(0 if tolerance is None else tolerance), np.shape(after_surface))


def first_transition(x, u, p, q, B, rin, rout, zlo, zhi, *, axis_x=0.,
                     s_max=1.e6, exiting=False, tolerance=None, max_intervals=512, after_surface=NONE):
    """Validated NumPy first transition; see module for approximation/refusal."""
    validate_inputs(x, u, p, q, B, rin, rout, zlo, zhi, axis_x=axis_x,
                    s_max=s_max, exiting=exiting, tolerance=tolerance,
                    max_intervals=max_intervals, after_surface=after_surface)
    with np.errstate(all="ignore"):
        return _transition(np, x, u, p, q, B, rin, rout, zlo, zhi,
                           axis_x, s_max, exiting, tolerance, max_intervals, after_surface)


def first_transition_jax(x, u, p, q, B, rin, rout, zlo, zhi, *, axis_x=0.,
                         s_max=1.e6, exiting=False, tolerance=None, max_intervals=512, after_surface=NONE):
    """Optional traceable primitive; max_intervals is static; validate on host."""
    import jax.numpy as jnp
    return _transition(jnp, x, u, p, q, B, rin, rout, zlo, zhi,
                       axis_x, s_max, exiting, tolerance, max_intervals, after_surface)


def _transition(xp, x, u, p, q, B, rin, rout, zlo, zhi, ax, cap, ex, tol, budget, after_surface=NONE):
    dtype = xp.result_type(xp.asarray(x).dtype, xp.asarray(u).dtype, xp.float32)
    x, u, B = [xp.asarray(v, dtype) for v in (x, u, B)]
    scalars = [p, q, rin, rout, zlo, zhi, ax, cap, ex, 0 if tol is None else tol]
    arrays = xp.broadcast_arrays(*[v[..., i] for v in (x, u, B) for i in range(3)],
                                 *[xp.asarray(v, dtype) for v in scalars], xp.asarray(after_surface))
    x, u, B = [xp.stack(arrays[i:i+3], -1) for i in (0, 3, 6)]
    p, q, ri, ro, zl, zh, ax, cap, ex, tol, token = arrays[9:]
    continuing = token != NONE
    ex = ex.astype(bool)
    eps, inf = xp.finfo(dtype).eps, xp.asarray(xp.inf, dtype)
    n = xp.sqrt(xp.sum(u*u, -1))
    bm = xp.sqrt(xp.sum(B*B, -1))
    bh = B / xp.where(bm > 0, bm, 1)[..., None]
    up = xp.sum(u*bh, -1)[..., None]*bh
    ut = u-up
    w = xp.cross(ut, bh)
    k = xp.asarray(K_FIELD, dtype)*q*bm/xp.where(p > 0, p, 1)
    force_zero = (q == 0) | xp.all(B == 0, -1)
    A = xp.abs(k)*xp.sqrt(xp.sum(ut*ut, -1))
    scale = xp.maximum(1, xp.maximum(xp.max(xp.abs(x), -1), xp.abs(ax)))
    for v in (ro, zl, zh):
        scale = xp.maximum(scale, xp.abs(v))
    guard = 128*eps*(scale + cap*n)*(1 + xp.abs(k*cap))
    resolution = xp.maximum(tol, guard)
    malformed = ((p <= 0) | ((q != -1) & (q != 0) & (q != 1))
                 | (xp.abs(n-1) > 32*eps) | (ri < 0) | (ro <= ri)
                 | (zl >= zh) | (cap < 0) | (tol < 0))
    malformed = malformed | (token < NONE) | (token > ROUT) | (token.dtype.kind not in "iu")
    for v in arrays:
        malformed = malformed | ~xp.isfinite(v)
    malformed = malformed | ~xp.isfinite(A+guard+bm) | ((bm == 0) & xp.any(B != 0, -1))
    malformed = malformed | ((k == 0) & ~force_zero) | (xp.abs(k*cap)*eps > .001)

    def trajectory(s):
        t = k*s
        small = xp.abs(t) < .01
        safe = xp.where(small, 1, t)
        t2 = xp.where(small, t*t, 0)
        sinc = xp.where(small, 1-t2/6+t2*t2/120-t2*t2*t2/5040, xp.sin(safe)/safe)
        cosc = xp.where(small, t*(.5-t2/24+t2*t2/720-t2*t2*t2/40320),
                        2*xp.sin(safe/2)**2/safe)
        pos = x+s[..., None]*(up+sinc[..., None]*ut+cosc[..., None]*w)
        vel = up+xp.cos(t)[..., None]*ut+xp.sin(t)[..., None]*w
        return pos, vel

    def clearance(s):
        pos, vel = trajectory(s)
        rx, yy, zz = pos[..., 0]-ax, pos[..., 1], pos[..., 2]
        r = xp.hypot(rx, yy)
        rd = (rx*vel[..., 0]+yy*vel[..., 1])/xp.where(r > 0, r, 1)
        g = xp.stack((zz-zl, zh-zz, xp.where(ri > 0, r-ri, inf), ro-r), -1)
        d = xp.stack((vel[..., 2], -vel[..., 2], xp.where(ri > 0, rd, 0), -rd), -1)
        return g, d, r

    zero = xp.zeros_like(cap)
    g0, d0, _ = clearance(zero)
    exact = g0 == 0
    near = xp.abs(g0) <= resolution[..., None]
    count = xp.sum(exact, -1)
    regular = (count == 1) & xp.all(exact | (g0 > resolution[..., None]), -1)
    regular = regular & xp.all(~exact | (xp.abs(d0) > 128*eps*n[..., None]), -1)
    backward = xp.all(xp.where(exact, d0 < 0, g0 > 0), -1)
    named = xp.arange(1, 5) == token[..., None]
    receipt = regular & xp.all(~named | exact, -1)
    forward = xp.all(xp.where(exact, d0 > 0, g0 > 0), -1)
    ownership = xp.where(continuing, forward, backward)
    start_bad = (ownership != ex) | (xp.any(near, -1) & ~regular) | malformed
    start_bad = start_bad | (continuing & ~receipt)
    start_hit = regular & ~start_bad & ~continuing
    active = ~start_bad & ~start_hit & ~force_zero
    kind0 = xp.argmax(exact, -1).astype(xp.int32)+1
    distance = xp.where(start_hit, zero, inf)
    kind = xp.where(start_hit, kind0, NONE)
    unresolved = start_bad
    depth_limit = 60
    stack = xp.zeros(cap.shape+(depth_limit,), dtype=dtype)
    depth = xp.zeros_like(cap, dtype=xp.int32)
    slots = xp.arange(depth_limit)
    state = (zero, cap, stack, depth, active, distance, kind, unresolved)

    def step(state):
        a, b, stack, depth, active, distance, kind, unresolved = state
        h, mid = b-a, a+(b-a)/2
        ga, _, _ = clearance(a)
        gb, _, _ = clearance(b)
        _, dm, rm = clearance(mid)
        rmin = rm-n*h/2-guard
        radial_M = A+n*n/xp.where(rmin > 0, rmin, 1)
        radial_err = xp.where(rmin > 0, radial_M*h*h/8, n*h)
        err = xp.stack((A*h*h/8, A*h*h/8, radial_err, radial_err), -1)
        lower = xp.minimum(ga, gb)-err-resolution[..., None]
        upper = xp.maximum(ga, gb)+err+resolution[..., None]
        clear = xp.any(upper < 0, -1) | xp.all(lower > 0, -1)
        variation = xp.stack((A*h/2, A*h/2, radial_M*h/2, radial_M*h/2), -1)
        monotone = xp.abs(dm) > variation+128*eps*(n+A*h)[..., None]
        monotone = monotone & xp.stack((xp.ones_like(active), xp.ones_like(active), rmin > 0, rmin > 0), -1)
        endpoint_a = (xp.abs(ga) > guard[..., None]) | (ga == 0)
        endpoint_b = (xp.abs(gb) > guard[..., None]) | (gb == 0)
        bracket = ((ga <= 0) & (gb >= 0)) | ((ga >= 0) & (gb <= 0))
        # Chronological departure certificate, not an epsilon step. Monotonicity
        # uses the same derivative-variation bound as an isolated crossing.
        # Together with resolved other constraints it proves that (0,b] has
        # exactly the claimed forward state and no unconsumed root. A positive,
        # resolved endpoint is required; short/zero caps cannot fabricate no-hit.
        departure = continuing & (a == 0)
        departing = xp.all(~named | (monotone & (dm*d0 > 0)), -1)
        departing = departing & xp.all(named | (lower > 0), -1)
        departing = departing & xp.all(~named | (xp.abs(gb) > resolution[..., None]), -1)
        departing = departing & (b > 0)
        clear = xp.where(departure, departing, clear)
        candidates = bracket & monotone & endpoint_a & endpoint_b
        candidates = candidates & ~(departure[..., None] & named)
        others = xp.sum(lower <= 0, -1) == 1
        hit = active & others & (xp.sum(candidates, -1) == 1)
        idx = xp.argmax(candidates, -1).astype(xp.int32)
        # Bisection is run only after a unique monotone bracket is certified.
        lo, hi = a, b
        fa = xp.take_along_axis(ga, idx[..., None], -1)[..., 0]

        def bisect(_, pair):
            l, r = pair
            m = l+(r-l)/2
            gm = clearance(m)[0]
            fm = xp.take_along_axis(gm, idx[..., None], -1)[..., 0]
            same = (fm > 0) == (fa > 0)
            return xp.where(same, m, l), xp.where(same, r, m)

        if xp is np:
            for j in range(56):
                lo, hi = bisect(j, (lo, hi))
        else:
            from jax import lax
            lo, hi = lax.fori_loop(0, 56, bisect, (lo, hi))
        fb = xp.take_along_axis(gb, idx[..., None], -1)[..., 0]
        root = xp.where(fa == 0, a, xp.where(fb == 0, b, lo+(hi-lo)/2))
        hit = hit & ((hi-lo) <= resolution) & xp.isfinite(root)
        distance = xp.where(hit, root, distance)
        kind = xp.where(hit, idx+1, kind)
        advance = active & clear & ~hit
        done = advance & (depth == 0)
        split = active & ~clear & ~hit
        fail = split & ((h <= resolution) | (mid <= a) | (mid >= b) | (depth >= depth_limit))
        split = split & ~fail
        unresolved = unresolved | fail
        pop = xp.sum(xp.where(slots == (depth-1)[..., None], stack, 0), -1)
        stack = xp.where(split[..., None] & (slots == depth[..., None]), b[..., None], stack)
        a = xp.where(advance, b, a)
        b = xp.where(split, mid, xp.where(advance & ~done, pop, b))
        depth = depth+split.astype(xp.int32)-(advance & ~done).astype(xp.int32)
        active = active & ~hit & ~done & ~fail
        return a, b, stack, depth, active, distance, kind, unresolved

    if xp is np:
        for _ in range(budget):
            if not np.any(state[4]):
                break
            state = step(state)
    else:
        from jax import lax
        def condition(pair):
            return (pair[0] < budget) & xp.any(pair[1][4])
        _, state = lax.while_loop(condition, lambda pair: (pair[0]+1, step(pair[1])), (0, state))
    _, _, _, _, active, distance, kind, unresolved = state
    unresolved = unresolved | active
    valid = xp.isfinite(distance) & ~unresolved
    curved = dict(distance=xp.where(valid, distance, inf), kind=xp.where(valid, kind, NONE).astype(xp.int32),
                  valid=valid, unresolved=unresolved)
    straight = ray._transition(xp, x, u, ri, ro, zl, zh, ax, cap, ex, tol, token)
    return {key: xp.where(force_zero & ~malformed, straight[key], curved[key]) for key in curved}
