"""S1 spatial segment/closed-box diagnostic, not a physics certificate.

Caller freezes explicit axis-aligned lo/hi in the endpoints' coordinate frame and
supplies a positive absolute geometry tolerance. No inferred detector geometry.
Birth != exit: all six faces include incoming and outgoing crossings; repeated
rows are never deduplicated (crossing flux is not a count of unique photons).
No native crossing flags, post-step times, history, response frontier, CLI or
transport are supplied. Validate producer endpoint semantics, straight-chord/path
length agreement and pre-momentum validity separately before physics use.

Float64 engineering guards are scale dependent, not exact predicates. No input
coordinates are nudged. Endpoint boundary contact, tangent/coplanar, edge/corner
and numerically unresolved rows are refused wholesale, not silently counted.
"""

from dataclasses import dataclass
import numpy as np

XLO, XHI, YLO, YHI, ZLO, ZHI = (1, 2, 4, 8, 16, 32)
OK, INVALID, AMBIGUOUS = 0, 1, 2


@dataclass(frozen=True)
class BoxCrossings:
    """Aligned (N,2) float64 u/facebitmask/direction, ordered in increasing u.

    Direction is -1 incoming, +1 outgoing. Unused slots and ALL refused-row
    slots are NaN (including facebitmask/direction). status is (N,) uint8;
    resolved is (N,) bool, true for OK rows, including guard-resolved no-hit
    rows. Thus resolved is NOT a crossing/physics acceptance flag.
    A crossing mask is resolved[:, None] & np.isfinite(u).
    """
    u: np.ndarray
    facebitmask: np.ndarray
    direction: np.ndarray
    status: np.ndarray
    resolved: np.ndarray


def _real64(value):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf":
        raise ValueError("require real numeric arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        return a.astype(np.float64, copy=False)


def box_segment_crossings(pre, post, *, lo, hi, tolerance):
    """Intersect (N,3) segments p(u)=pre+u*(post-pre) with a closed box.

    Bounds must be finite (3,) arrays with lo < hi; tolerance a finite positive
    scalar length. Bad schemas/geometry raise ValueError. Nonfinite/zero-length
    rows are INVALID; numerical/boundary ambiguity is AMBIGUOUS. Only strictly
    interior transverse roots tolparam < u < 1-tolparam survive; tolparam uses
    the length tolerance plus float64 roundoff guards. Exact parallel components
    away from face planes are allowed; tiny nonzero components are refused.
    No per-segment Python loops; storage and work are O(N).
    """
    a, b, low, high, tol = map(_real64, (pre, post, lo, hi, tolerance))
    if a.ndim != 2 or a.shape[1:] != (3,) or b.shape != a.shape:
        raise ValueError("pre and post must have matching (N,3) shapes")
    if (low.shape != (3,) or high.shape != (3,) or tol.ndim != 0
            or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
            or not np.all(low < high) or not np.isfinite(tol) or tol <= 0):
        raise ValueError("require finite ordered (3,) bounds and positive tolerance")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        return _crossings(a, b, low, high, tol)


def _crossings(a, b, low, high, tol):
    n, eps = len(a), 64 * np.finfo(np.float64).eps
    d = b - a
    span = np.max(np.abs(d), axis=1)
    invalid = (~np.all(np.isfinite(a) & np.isfinite(b), axis=1) | (span == 0))
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)),
                       np.maximum(np.abs(low), np.abs(high)))
    guard = np.maximum(tol, eps * scale)
    numeric = ~np.all(np.isfinite(d), axis=1) | ~np.all(np.isfinite(high - low))
    ambiguous = numeric | np.any(high - low <= 2 * guard, axis=1)
    # Endpoint contact with the finite closed surface, not merely its planes.
    for p in (a, b):
        near = (np.abs(p - low) <= guard) | (np.abs(p - high) <= guard)
        ambiguous |= np.all((p >= low - guard) & (p <= high + guard), axis=1) & np.any(near, axis=1)
    moving = d != 0
    near_parallel = moving & (np.abs(d) <= np.maximum(guard, eps * span[:, None]))
    coplanar = ~moving & ((np.abs(a - low) <= guard) | (np.abs(a - high) <= guard))
    ambiguous |= np.any(near_parallel | coplanar, axis=1)
    safe_d = np.where(moving, d, 1.)
    t0, t1 = (low - a) / safe_d, (high - a) / safe_d
    ambiguous |= np.any(moving & (~np.isfinite(t0) | ~np.isfinite(t1)), axis=1)
    inside = (a > low + guard) & (a < high - guard)
    left = np.where(moving, np.minimum(t0, t1), np.where(inside, -np.inf, np.inf))
    right = np.where(moving, np.maximum(t0, t1), np.where(inside, np.inf, -np.inf))
    entry, exit = np.max(left, axis=1), np.min(right, axis=1)
    param_tol = np.maximum(eps, np.max(np.where(moving, guard / np.abs(safe_d), 0.), axis=1))
    ambiguous |= param_tol >= .5
    roots = np.stack((entry, exit), axis=1)
    axes = np.stack((np.argmax(left, axis=1), np.argmin(right, axis=1)), axis=1)
    overlap = (entry <= 1 + param_tol) & (exit >= -param_tol)
    ambiguous |= overlap & (np.abs(exit - entry) <= 2 * param_tol)
    candidate = (roots > param_tol[:, None]) & (roots < 1 - param_tol[:, None]) & (entry <= exit)[:, None]
    # A root near either segment endpoint or another face cannot be owned safely.
    ambiguous |= overlap & np.any((np.abs(roots) <= param_tol[:, None]) |
                                  (np.abs(roots - 1) <= param_tol[:, None]), axis=1)
    points = a[:, None, :] + np.clip(roots, 0, 1)[:, :, None] * d[:, None, :]
    near_face = ((points <= low + guard[:, None, :]) |
                 (points >= high - guard[:, None, :]))
    other_axis = np.arange(3)[None, None, :] != axes[:, :, None]
    ambiguous |= np.any(candidate & np.any(near_face & other_axis, axis=2), axis=1)
    status = np.where(invalid, INVALID, np.where(ambiguous, AMBIGUOUS, OK)).astype(np.uint8)
    resolved = status == OK
    hit = candidate & resolved[:, None]
    sign = np.take_along_axis(d, axes, axis=1) > 0
    high_face = sign ^ np.array([True, False])
    faces = (1 << (2 * axes + high_face)).astype(np.float64)
    order = np.argsort(np.where(hit, roots, np.inf), axis=1, kind="stable")
    u = np.take_along_axis(np.where(hit, roots, np.nan), order, axis=1)
    face = np.take_along_axis(np.where(hit, faces, np.nan), order, axis=1)
    direction = np.take_along_axis(np.where(hit, [-1., 1.], np.nan), order, axis=1)
    return BoxCrossings(u, face, direction, status, resolved)
