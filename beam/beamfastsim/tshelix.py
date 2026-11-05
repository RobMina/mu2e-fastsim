"""Outward uniform-field helix sets without platform sin/cos assumptions.

The ideal model uses exact stored binary64 x, p, B, K and arclength, and the
mathematical normalization of the stored nonzero raw direction. For Wv=v cross
(q*K*s*B/p), exp(W) rotates this direction. An 18th-order interval series and
analytic tail bounds enclose endpoint position and direction for |W|<=1.
Only restricted IEEE basic-op/sqrt arithmetic is assumed. No varying-map,
geometry, native, material or existing transport-default certificate follows.
"""

from dataclasses import dataclass

import numpy as np

from . import tsinterval as iv, tsstep

ORDER = 18
MAX_PHASE = 1.


def _fixed(value):
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _column(a):
    return iv.Interval(a.lo[:, None], a.hi[:, None])


def _mul(a, b):
    result = iv.mul(a, b)
    zero = ((a.lo == 0) & (a.hi == 0)) | ((b.lo == 0) & (b.hi == 0))
    return iv.Interval(np.where(zero, 0., result.lo), np.where(zero, 0., result.hi))


def _cross(a, b):
    values = []
    for j, k in ((1, 2), (2, 0), (0, 1)):
        aj, ak = iv.Interval(a.lo[:, j], a.hi[:, j]), iv.Interval(a.lo[:, k], a.hi[:, k])
        bj, bk = iv.Interval(b.lo[:, j], b.hi[:, j]), iv.Interval(b.lo[:, k], b.hi[:, k])
        values.append(iv.sub(_mul(aj, bk), _mul(ak, bj)))
    return iv.Interval(np.stack([v.lo for v in values], axis=1), np.stack([v.hi for v in values], axis=1))


def _rows(value, n, name, *, positive=False):
    a = iv.Interval(value).lo
    if a.shape not in ((), (n,)) or np.any(a < 0) or (positive and np.any(a <= 0)):
        raise ValueError(f'{name} requires nonnegative binary64 scalar or row values; momentum positive')
    return np.broadcast_to(a, (n,))


@dataclass(frozen=True, eq=False)
class Helix:
    position: iv.Interval
    direction: iv.Interval
    x_center_mm: np.ndarray
    direction_vector: np.ndarray
    position_radius_mm: np.ndarray
    direction_radius: np.ndarray
    phase_upper: np.ndarray
    direction_tail: np.ndarray
    position_tail_mm: np.ndarray
    momentum: np.ndarray
    pdg: np.ndarray
    length_mm: np.ndarray
    evaluation_arithmetic_outward: bool = True
    transcendental_library_assumed: bool = False
    floating_point_certified: bool = False
    native_query_certified: bool = False
    restart_state: bool = False
    material_transport: bool = False


@dataclass(frozen=True, eq=False)
class Difference:
    position_mm: np.ndarray
    direction: np.ndarray


def difference(receipt, x, direction):
    """Errors versus supplied values; direction center means mathematical unit raw.

    Supplied values need not be accurately evaluated or produced by this module.
    Their discrepancies are enclosed, not assumed small. No source-State binding
    or permission to replace an existing Segment error declaration is implied.
    """
    if type(receipt) is not Helix:
        raise ValueError('explicit Helix receipt required')
    x, direction = iv.Interval(x), iv.Interval(direction)
    if x.lo.shape != receipt.position.lo.shape or direction.lo.shape != x.lo.shape:
        raise ValueError('matching binary64 (N,3) evaluation arrays required')
    dx = iv.norm(iv.sub(receipt.position, x)).hi
    du = iv.norm(iv.sub(receipt.direction, iv.normalize(direction))).hi
    return Difference(_fixed(dx), _fixed(np.minimum(2., du)))


def evaluate(x, direction, momentum, pdg, field, length):
    """Enclose one ideal uniform helix endpoint per row; no state advancement.

    Scalar momentum is exact and independent of direction magnitude. The phase
    norm upper bound must be <=1 rad; no libm argument reduction is used. Centers
    are stored values, not exact unit vectors. direction_radius bounds the ideal
    direction versus mathematical normalization of direction_vector. Tail values
    exclude interval dependency/roundoff, which remain in the full radii.
    """
    x, raw, field = iv.Interval(x), iv.Interval(direction), iv.Interval(field)
    if x.lo.ndim != 2 or x.lo.shape[1] != 3 or raw.lo.shape != x.lo.shape or field.lo.shape != x.lo.shape:
        raise ValueError('matching explicit binary64 (N,3) x/direction/field required')
    n = len(x.lo)
    p, s = _rows(momentum, n, 'momentum', positive=True), _rows(length, n, 'length')
    ids = np.asarray(pdg)
    if ids.shape not in ((), (n,)) or ids.dtype.kind not in 'iu' or np.any(~np.isin(ids, (13, -13, 211, -211))):
        raise ValueError('integer mu+/mu-/pi+/pi- PDG scalar or rows required')
    ids = np.broadcast_to(ids, (n,))
    q = np.where((ids == 13) | (ids == -211), -1., 1.)
    u = iv.normalize(raw)
    numerator = _mul(_mul(field, iv.Interval(np.float64(tsstep.K))), iv.Interval(s[:, None]))
    w = iv.div(numerator, iv.Interval(p[:, None]))
    zero = (s == 0) | np.all(field.lo == 0, axis=1)
    w = iv.Interval(np.where(zero[:, None], 0., w.lo), np.where(zero[:, None], 0., w.hi))
    w = _mul(w, iv.Interval(q[:, None]))
    theta = iv.norm(w).hi
    if np.any(theta > MAX_PHASE):
        raise ValueError('uniform helix phase enclosure exceeds 1 rad')
    term, total, integral = u, u, u
    for order in range(1, ORDER+1):
        term = iv.div(_cross(term, w), iv.Interval(np.float64(order)))
        total = iv.add(total, term)
        integral = iv.add(integral, iv.div(term, iv.Interval(np.float64(order+1))))
    tail = iv.Interval(np.full(n, 3., dtype=np.float64))
    for order in range(1, ORDER+2):
        tail = iv.div(_mul(tail, iv.Interval(theta)), iv.Interval(np.float64(order)))
    tail = iv.Interval(np.zeros(n), np.where(zero, 0., np.maximum(0., tail.hi)))
    position_tail = iv.div(_mul(tail, iv.Interval(s)), iv.Interval(np.float64(ORDER+2)))
    position_tail = iv.Interval(np.zeros(n), np.where(zero, 0., np.maximum(0., position_tail.hi)))
    endpoint = iv.inflate(iv.add(x, _mul(integral, iv.Interval(s[:, None]))), _column(position_tail))
    endpoint = iv.Interval(np.where((s == 0)[:, None], x.lo, endpoint.lo), np.where((s == 0)[:, None], x.hi, endpoint.hi))
    rotated = iv.inflate(total, _column(tail))
    rotated = iv.Interval(np.where(zero[:, None], u.lo, rotated.lo), np.where(zero[:, None], u.hi, rotated.hi))
    xc, uc = endpoint.lo/2+endpoint.hi/2, rotated.lo/2+rotated.hi/2
    dx = iv.norm(iv.sub(endpoint, iv.Interval(xc))).hi
    du = iv.norm(iv.sub(rotated, iv.normalize(iv.Interval(uc)))).hi
    return Helix(endpoint, rotated, *(_fixed(a) for a in
        (xc, uc, dx, np.minimum(2., du), theta, tail.hi, position_tail.hi, p, ids, s)))
