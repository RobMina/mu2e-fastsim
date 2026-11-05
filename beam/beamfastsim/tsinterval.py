"""Restricted outward binary64 arithmetic for finite interval endpoints.

Each elementary operation is enclosed by adjacent representable floats. This
requires IEEE-754 binary64 correctly rounded add/subtract/multiply/divide/sqrt
with gradual underflow, not reassociated fast-math or flush-to-zero. Overflow
or a zero-containing divisor fails closed. No transcendental, geometry, field,
trajectory or native certificate follows from these local arithmetic bounds.
Inputs are already binary64 values interpreted exactly; decimal conversion and
uncertain source data must be enclosed separately.
"""

from dataclasses import dataclass

import numpy as np


def _fixed(value):
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _values(value):
    a = np.asarray(value)
    if a.dtype != np.dtype('float64') or np.any(~np.isfinite(a)):
        raise ValueError('explicit finite binary64 endpoints required')
    return a


def _rounded(value, toward):
    if np.any(~np.isfinite(value)):
        raise ValueError('interval arithmetic overflow or nonfinite result')
    with np.errstate(under='ignore', over='ignore', invalid='ignore'):
        result = np.nextafter(value, toward)
    if np.any(~np.isfinite(result)):
        raise ValueError('outward interval endpoint is not finite')
    return result


@dataclass(frozen=True, eq=False, init=False)
class Interval:
    lo: np.ndarray
    hi: np.ndarray

    def __init__(self, lo, hi=None):
        lo = _values(lo)
        hi = lo if hi is None else _values(hi)
        lo, hi = np.broadcast_arrays(lo, hi)
        if np.any(lo > hi):
            raise ValueError('ordered interval endpoints required')
        object.__setattr__(self, 'lo', _fixed(lo))
        object.__setattr__(self, 'hi', _fixed(hi))


def _require(value):
    if type(value) is not Interval:
        raise ValueError('explicit Interval required')
    return value


def add(a, b):
    a, b = _require(a), _require(b)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        lo, hi = a.lo+b.lo, a.hi+b.hi
    return Interval(_rounded(lo, -np.inf), _rounded(hi, np.inf))


def sub(a, b):
    a, b = _require(a), _require(b)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        lo, hi = a.lo-b.hi, a.hi-b.lo
    return Interval(_rounded(lo, -np.inf), _rounded(hi, np.inf))


def mul(a, b):
    a, b = _require(a), _require(b)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        terms = np.stack(np.broadcast_arrays(a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi))
    if np.any(~np.isfinite(terms)):
        raise ValueError('interval product outside finite range')
    return Interval(_rounded(np.min(terms, axis=0), -np.inf),
                    _rounded(np.max(terms, axis=0), np.inf))


def div(a, b):
    a, b = _require(a), _require(b)
    if np.any((b.lo <= 0) & (b.hi >= 0)):
        raise ValueError('interval divisor contains zero')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        terms = np.stack(np.broadcast_arrays(a.lo/b.lo, a.lo/b.hi, a.hi/b.lo, a.hi/b.hi))
    if np.any(~np.isfinite(terms)):
        raise ValueError('interval quotient outside finite range')
    return Interval(_rounded(np.min(terms, axis=0), -np.inf),
                    _rounded(np.max(terms, axis=0), np.inf))


def square(a):
    a = _require(a)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        left, right = a.lo*a.lo, a.hi*a.hi
    lo = np.where((a.lo <= 0) & (a.hi >= 0), 0., np.minimum(left, right))
    return Interval(np.maximum(0., _rounded(lo, -np.inf)),
                    _rounded(np.maximum(left, right), np.inf))


def sqrt(a):
    a = _require(a)
    if np.any(a.lo < 0):
        raise ValueError('sqrt requires nonnegative interval')
    with np.errstate(under='ignore', invalid='ignore'):
        lo, hi = np.sqrt(a.lo), np.sqrt(a.hi)
    return Interval(np.maximum(0., _rounded(lo, -np.inf)), _rounded(hi, np.inf))


def sum_last(a):
    a = _require(a)
    if a.lo.ndim == 0 or a.lo.shape[-1] == 0:
        raise ValueError('nonempty last axis required')
    result = Interval(np.zeros(a.lo.shape[:-1], dtype=np.float64))
    for column in range(a.lo.shape[-1]):
        result = add(result, Interval(a.lo[..., column], a.hi[..., column]))
    return result


def dot(a, b):
    return sum_last(mul(a, b))


def norm(a):
    """Scaled Euclidean interval norm, avoiding raw square overflow/underflow."""
    a = _require(a)
    if a.lo.ndim == 0 or a.lo.shape[-1] == 0:
        raise ValueError('nonempty vector axis required')
    scale = np.max(np.maximum(np.abs(a.lo), np.abs(a.hi)), axis=-1)
    safe = np.where(scale == 0, 1., scale)
    scaled = div(a, Interval(safe[..., None]))
    squares = square(scaled)
    total = sum_last(squares)
    total = Interval(np.maximum(0., total.lo), total.hi)
    result = mul(sqrt(total), Interval(safe))
    zero = scale == 0
    return Interval(np.where(zero, 0., np.maximum(0., result.lo)),
                    np.where(zero, 0., result.hi))


def normalize(a):
    """Enclose mathematical normalization; no rounded unit-vector assertion."""
    a = _require(a)
    length = norm(a)
    return div(a, Interval(length.lo[..., None], length.hi[..., None]))


def inflate(a, radius):
    a, radius = _require(a), _require(radius)
    if np.any(radius.lo < 0):
        raise ValueError('nonnegative inflation radius required')
    return Interval(sub(a, radius).lo, add(a, radius).hi)
