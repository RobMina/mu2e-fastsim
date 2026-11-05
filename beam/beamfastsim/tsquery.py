"""Outward field-query bounds for exact stored binary64 grid metadata/nodes.

The ideal grid origin is the mathematical difference of stored origin and shift,
not the rounded GridHeader.lo. Query boxes must prove coverage in this grid;
uncertain boundary arithmetic fails closed. All intersected cells are enclosed,
including reflected By on both sides of zero. No native selector/cache, decimal
header conversion, helix, geometry or transport certificate follows. The source
map must not mutate during evaluation. Arithmetic inherits tsinterval assumptions.
"""

from dataclasses import dataclass
from fractions import Fraction
import itertools

import numpy as np

from . import tsfield, tsinterval as iv


def _fixed(value):
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class Query:
    value_tesla: np.ndarray
    nominal_field: iv.Interval
    region_field: iv.Interval
    evaluation_error_tesla: np.ndarray
    region_error_tesla: np.ndarray
    position_box: iv.Interval
    cell_count: np.ndarray
    query_arithmetic_outward: bool = True
    stored_grid_domain_checked: bool = True
    floating_point_certified: bool = False
    native_query_certified: bool = False
    material_transport: bool = False


def _coordinates(item, box):
    lo, hi = box.lo.copy(), box.hi.copy()
    sign_lo, sign_hi = np.ones(lo.shape[0]), np.ones(lo.shape[0])
    if item.header.extend_y:
        left, right = lo[:, 1].copy(), hi[:, 1].copy()
        lo[:, 1] = np.where((left <= 0) & (right >= 0), 0., np.minimum(np.abs(left), np.abs(right)))
        hi[:, 1] = np.maximum(np.abs(left), np.abs(right))
        sign_lo = np.where(left < 0, -1., 1.)
        sign_hi = np.where(right < 0, -1., 1.)
    domain_lo, domain_hi = [], []
    for axis in range(3):
        exact_lo = Fraction(float(item.header.grid_origin[axis]))-Fraction(float(item.header.origin_shift[axis]))
        exact_hi = exact_lo+int(item.n[axis]-1)*Fraction(float(item.d[axis]))
        left, right = float(exact_lo), float(exact_hi)
        if not np.isfinite(left) or not np.isfinite(right):
            raise ValueError('stored-grid domain endpoints outside finite range')
        domain_lo.append(np.nextafter(left, np.inf) if Fraction(left) < exact_lo else left)
        domain_hi.append(np.nextafter(right, -np.inf) if Fraction(right) > exact_hi else right)
    if np.any((lo < domain_lo) | (hi > domain_hi)):
        raise ValueError('whole query box lacks a proved stored-grid domain cover')
    origin = iv.sub(iv.Interval(item.header.grid_origin), iv.Interval(item.header.origin_shift))
    q = iv.div(iv.sub(iv.Interval(lo, hi), origin), iv.Interval(item.d))
    q = iv.Interval(np.maximum(0., q.lo), np.minimum(item.n-1, q.hi))
    return q, iv.Interval(sign_lo, sign_hi)


def _enclose(item, q, signs, max_cells):
    first = np.minimum(np.floor(q.lo).astype(np.int64), item.n-2)
    last = np.minimum(np.floor(q.hi).astype(np.int64), item.n-2)
    widths = last-first+1
    if np.any(widths > max_cells):
        raise ValueError('query cell work cap exceeded')
    counts = np.ones(q.lo.shape[0], dtype=np.int64)
    for axis in range(3):
        if np.any(counts > max_cells//widths[:, axis]):
            raise ValueError('query cell work cap exceeded')
        counts *= widths[:, axis]
    total = int(np.sum(counts, dtype=object))
    if total > max_cells:
        raise ValueError('query cell work cap exceeded')
    rows = np.repeat(np.arange(len(counts)), counts)
    starts = np.cumsum(counts)-counts
    flat = np.arange(total)-np.repeat(starts, counts)
    sizes = widths[rows]
    offset = np.stack((flat//(sizes[:, 1]*sizes[:, 2]), (flat//sizes[:, 2]) % sizes[:, 1], flat % sizes[:, 2]), axis=1)
    cells = first[rows]+offset
    clipped_lo = np.maximum(q.lo[rows], cells)
    clipped_hi = np.minimum(q.hi[rows], cells+1)
    local = iv.sub(iv.Interval(clipped_lo, clipped_hi), iv.Interval(cells.astype(np.float64)))
    local = iv.Interval(np.maximum(0., local.lo), np.minimum(1., local.hi))
    result = iv.Interval(np.zeros((total, 3), dtype=np.float64))
    for corner in itertools.product((0, 1), repeat=3):
        weight = iv.Interval(np.ones(total, dtype=np.float64))
        for axis in range(3):
            factor = iv.Interval(local.lo[:, axis], local.hi[:, axis])
            if not corner[axis]:
                factor = iv.sub(iv.Interval(1.), factor)
            weight = iv.mul(weight, factor)
        nodes = cells+np.asarray(corner)
        data = item.B[nodes[:, 0], nodes[:, 1], nodes[:, 2]]
        result = iv.add(result, iv.mul(iv.Interval(weight.lo[:, None], weight.hi[:, None]), iv.Interval(data)))
    by = iv.mul(iv.Interval(result.lo[:, 1], result.hi[:, 1]), iv.Interval(signs.lo[rows], signs.hi[rows]))
    lo, hi = result.lo.copy(), result.hi.copy()
    lo[:, 1], hi[:, 1] = by.lo, by.hi
    lower, upper = np.full(q.lo.shape, np.inf), np.full(q.lo.shape, -np.inf)
    np.minimum.at(lower, rows, lo)
    np.maximum.at(upper, rows, hi)
    return iv.Interval(lower, upper), counts


def evaluate(item, points, *, position_radius_mm=0., max_cells=65536):
    """Bound ideal nominal and uncertain-box fields relative to item.field(points).

    The scalar row radius defines a Cartesian box (also covering its Euclidean
    ball). evaluation_error excludes position uncertainty; region_error includes
    it and must not be added as another independent interpolation error. The cap
    counts nominal and region cells together, over the entire batch.
    """
    if not isinstance(item, tsfield.FieldMap):
        raise TypeError('explicit FieldMap required')
    if type(item.header.extend_y) not in (bool, np.bool_):
        raise ValueError('scalar reflection flag required')
    if type(max_cells) is not int or not 1 <= max_cells <= 1000000:
        raise ValueError('integer cell work cap in [1, 1000000] required')
    if np.any(item.n > 2**52):
        raise ValueError('grid indices exceed exactly represented integer range')
    points = iv.Interval(points).lo
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('explicit binary64 points with shape (N, 3) required')
    radius = iv.Interval(position_radius_mm).lo
    if radius.ndim > 1 or np.any(radius < 0):
        raise ValueError('nonnegative scalar or row binary64 radius required')
    radius = np.broadcast_to(radius, points.shape[:1])
    expanded = iv.inflate(iv.Interval(points), iv.Interval(radius[:, None]))
    box = iv.Interval(np.where(radius[:, None] == 0, points, expanded.lo),
                      np.where(radius[:, None] == 0, points, expanded.hi))
    n = len(points)
    both = iv.Interval(np.concatenate((points, box.lo)), np.concatenate((points, box.hi)))
    q, signs = _coordinates(item, both)
    field, counts = _enclose(item, q, signs, max_cells)
    nominal, region = iv.Interval(field.lo[:n], field.hi[:n]), iv.Interval(field.lo[n:], field.hi[n:])
    value = item.field(points)
    error = iv.norm(iv.sub(nominal, iv.Interval(value))).hi
    region_error = iv.norm(iv.sub(region, iv.Interval(value))).hi
    return Query(_fixed(value), nominal, region, _fixed(error), _fixed(region_error), box,
                 _fixed(np.stack((counts[:n], counts[n:]), axis=1)))
