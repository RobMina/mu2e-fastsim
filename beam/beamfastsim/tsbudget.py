"""Outward whole-map bounds and carried ideal-Lorentz error polynomials.

Opt-in prerequisites only; no defaults or existing carry implementation changed.
Every node/edge is scanned, not track samples. The ideal map uses exact stored
binary64 nodes and positive grid spacing. Interpolation, grid-domain endpoints,
reflections, native selection, reference helix and geometry are NOT certified.
Arithmetic inherits tsinterval's IEEE basic-op/sqrt/gradual-underflow assumptions.
"""

from dataclasses import dataclass
import hashlib

import numpy as np

from . import tsfield, tsinterval as iv, tsstep


@dataclass(frozen=True)
class MapBounds:
    magnitude_tesla: float
    lipschitz_tesla_per_mm: float
    column_tesla_per_mm: tuple
    node_count: int
    edge_count: int
    reflection_continuous: bool
    fingerprint: str
    bound_arithmetic_outward: bool = True
    floating_point_certified: bool = False
    native_query_certified: bool = False


def scan(item, *, chunk_nodes=65536):
    """All-node convex-hull norm and all-edge Frobenius Jacobian upper bounds.

    Fingerprint identifies exact stored grid/header bytes, not a native runtime.
    Input map must not be concurrently mutated. Continuity of reflected lobes
    is characterized separately; coverage/continuous-domain checks still needed.
    """
    if type(item) is not tsfield.FieldMap:
        raise ValueError('explicit FieldMap required')
    if type(chunk_nodes) is not int or not 1 <= chunk_nodes <= 1000000:
        raise ValueError('chunk_nodes requires integer in [1,1000000]')
    digest = hashlib.sha256()
    for array in (item.header.grid_origin, item.header.origin_shift, item.n, item.d):
        digest.update(array.tobytes())
    node_count = int(np.prod(item.n))
    maximum, reflected_plane_nonzero = 0., False
    for start in range(0, node_count, chunk_nodes):
        index = np.unravel_index(np.arange(start, min(node_count, start+chunk_nodes)), item.n)
        block = item.B[index]
        digest.update(block.tobytes())
        reflected_plane_nonzero |= bool(np.any((index[1] == 0) & (block[:, 1] != 0)))
        maximum = max(maximum, float(np.max(iv.norm(iv.Interval(block)).hi)))
    columns, count = [], 0
    for axis in range(3):
        shape = item.n.copy()
        shape[axis] -= 1
        size = int(np.prod(shape))
        upper = 0.
        for start in range(0, size, chunk_nodes):
            index = np.unravel_index(np.arange(start, min(size, start+chunk_nodes)), shape)
            other = tuple(v+1 if j == axis else v for j, v in enumerate(index))
            left, right = item.B[index], item.B[other]
            delta = iv.sub(iv.Interval(right), iv.Interval(left))
            norm = iv.norm(delta)
            slope = iv.div(norm, iv.Interval(np.float64(item.d[axis]))).hi
            slope = np.where(np.all(left == right, axis=1), 0., slope)
            upper = max(upper, float(np.max(slope)))
        columns.append(upper)
        count += size
    if any(columns):
        gradient = float(iv.norm(iv.Interval(np.asarray(columns, dtype=np.float64))).hi)
    else:
        gradient = 0.
    digest.update(bytes((bool(item.header.extend_y), bool(item.header.explicit_origin))))
    continuous = not (item.header.extend_y and item.lo[1] == 0 and reflected_plane_nonzero)
    return MapBounds(maximum, gradient, tuple(columns), node_count, count, continuous, digest.hexdigest())


def _nonnegative(value, name):
    array = np.asarray(value)
    if array.dtype != np.dtype('float64') or np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f'{name} requires finite nonnegative binary64 values')
    return array


def _nn(value):
    return iv.Interval(np.maximum(0., value.lo), value.hi)


def _add(a, b):
    return _nn(iv.add(a, b))


def _mul(a, b):
    result = _nn(iv.mul(a, b))
    zero = (a.hi == 0) | (b.hi == 0)
    return iv.Interval(np.where(zero, 0., result.lo), np.where(zero, 0., result.hi))


def _div(a, b):
    result = _nn(iv.div(a, b))
    return iv.Interval(np.where(a.hi == 0, 0., result.lo), np.where(a.hi == 0, 0., result.hi))


@dataclass(frozen=True)
class Errors:
    position_mm: np.ndarray
    direction: np.ndarray
    curvature_per_mm: np.ndarray
    model_arithmetic_outward: bool = True
    floating_point_certified: bool = False


def errors(bounds, momentum, length, *, position_error_mm, direction_error, field_error_tesla):
    """Upper bounds on Ex/Eu for an ideal single-map segment, not evaluated x/u.

    Model: exact positive scalar p, exact binary64 K from tsstep, unit speed,
    one continuous covered map domain. Initial ex/eu and field-evaluation error
    remain caller assumptions. Length/error arrays broadcast to one row axis.
    No clock, loss, scattering, root, helix or domain coverage check is supplied.
    """
    if type(bounds) is not MapBounds:
        raise ValueError('explicit outward MapBounds required')
    arrays = [_nonnegative(value, name) for value, name in (
        (momentum, 'momentum'), (length, 'length'), (position_error_mm, 'position_error_mm'),
        (direction_error, 'direction_error'), (field_error_tesla, 'field_error_tesla'))]
    p, s, ex, eu, eb = np.broadcast_arrays(*arrays)
    if p.ndim != 1 or np.any(p <= 0) or np.any(eu > 2):
        raise ValueError('one-dimensional rows, positive momentum and direction_error<=2 required')
    p, s, ex, eu, eb = [iv.Interval(a) for a in (p, s, ex, eu, eb)]
    k, two, six = [iv.Interval(np.float64(a)) for a in (tsstep.K, 2., 6.)]
    g = iv.Interval(np.float64(bounds.lipschitz_tesla_per_mm))
    b = iv.Interval(np.float64(bounds.magnitude_tesla))
    ag, ae = _div(_mul(k, g), p), _div(_mul(k, eb), p)
    rate = _add(_mul(ag, ex), ae)
    s2, s3 = _mul(s, s), _mul(_mul(s, s), s)
    dx = _add(_add(ex, _mul(eu, s)),
        _add(_div(_mul(rate, s2), two), _div(_mul(ag, s3), six)))
    du = _add(eu, _add(_mul(rate, s), _div(_mul(ag, s2), two)))
    cap = _add(ex, _mul(two, s))
    position = np.minimum(dx.hi, cap.hi)
    direction = np.minimum(2., du.hi)
    position = np.where(s.hi == 0, ex.hi, position)
    direction = np.where(s.hi == 0, eu.hi, direction)
    curvature = _div(_mul(k, b), p).hi
    return Errors(*[iv._fixed(a) for a in (position, direction, curvature)])
