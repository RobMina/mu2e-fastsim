"""Whole-map convex-hull magnitude bounds for trilinear interpolation.

This is an exact-arithmetic bound for the supplied NumPy map, not native field
cache equivalence or an interval-arithmetic certificate. Every trilinear value
is a convex combination of its eight nodes, so its norm is at most the largest
node norm. Y reflection flips By without changing that bound. Map seams/history,
ODE position errors and physical geometry navigation are not addressed here.
"""

from dataclasses import dataclass

import numpy as np

from . import tsfield, tsnav


@dataclass(frozen=True)
class Bound:
    magnitude_tesla: float
    node_count: int
    floating_point_certified: bool = False
    native_query_certified: bool = False


def magnitude(item, *, chunk_nodes=65536):
    """Scan all nodes in bounded chunks; no track-point sampling or fitted bound."""
    if type(item) is not tsfield.FieldMap:
        raise ValueError('explicit FieldMap required')
    if type(chunk_nodes) is not int or not 1 <= chunk_nodes <= 1000000:
        raise ValueError('chunk_nodes requires integer in [1,1000000]')
    nodes = item.B.reshape(-1, 3)
    upper = 0.
    for start in range(0, len(nodes), chunk_nodes):
        norms = np.hypot.reduce(nodes[start:start + chunk_nodes], axis=1)
        if not np.all(np.isfinite(norms)):
            raise ValueError('field magnitude outside supported finite range')
        upper = max(upper, float(np.max(norms)))
    return Bound(upper, len(nodes))


def box_coverage(item, lo, hi, *, continuous=False):
    """Per-row ENTIRE closed-box coverage, optionally excluding reflected seams.

    Only coverage/continuity failures become False; malformed numerical inputs
    still raise. No endpoint sampling, selector history or field-bank union.
    """
    if type(item) is not tsfield.FieldMap:
        raise ValueError('explicit FieldMap required')
    lo, hi = tsnav._real(lo, 'lo'), tsnav._real(hi, 'hi')
    if lo.ndim != 2 or lo.shape[1] != 3 or hi.shape != lo.shape or np.any(lo > hi):
        raise ValueError('finite ordered (N,3) box corners required')
    if type(continuous) is not bool:
        raise ValueError('continuous requires bool')
    covered = np.zeros(len(lo), dtype=bool)
    for a, b in item.header.boxes:
        covered |= np.all(lo >= a, axis=1) & np.all(hi <= b, axis=1)
    if item.header.extend_y and item.lo[1] == 0:
        lower = item.lo.copy()
        lower[1] = -item.hi[1]
        covered = np.all(lo >= lower, axis=1) & np.all(hi <= item.hi, axis=1)
        crosses = (lo[:, 1] < 0) & (hi[:, 1] >= 0)
        if continuous and np.any(crosses) and np.any(item.B[:, 0, :, 1] != 0):
            covered &= ~crosses
    return covered


def require_boxes(item, lo, hi, *, continuous=False):
    """Require whole-box coverage; no map selection or discontinuity welding."""
    covered = box_coverage(item, lo, hi, continuous=continuous)
    if not np.all(covered):
        without_continuity = box_coverage(item, lo, hi, continuous=False)
        if continuous and np.any(without_continuity & ~covered):
            raise ValueError('nonzero By on reflected zero-plane is discontinuous')
        raise ValueError('requested box is not wholly covered by this map')


def covered_box(item, lo, hi):
    """Require an entire closed (3,) AABB, then return the whole-map norm bound."""
    lo, hi = tsnav._real(lo, 'lo'), tsnav._real(hi, 'hi')
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError('finite ordered (3,) box corners required')
    require_boxes(item, lo[None], hi[None])
    return magnitude(item)


@dataclass(frozen=True)
class SpatialBound:
    lipschitz_tesla_per_mm: float
    column_tesla_per_mm: tuple
    edge_count: int
    reflection_continuous: bool
    floating_point_certified: bool = False
    native_query_certified: bool = False


def spatial(item, *, chunk_nodes=65536):
    """Scan all grid edges for a whole-map, within-lobe spatial Lipschitz bound.

    Each Jacobian column is a convex combination of parallel edge slopes.
    Column norm maxima give an operator bound via their root-sum-of-squares.
    Trilinear continuity extends that bound across grid cells. Across a reflected
    zero-plane it applies ONLY if reflection_continuous is true; a gap remains
    uncovered irrespective of this flag. Floating arithmetic is not certified.
    """
    if type(item) is not tsfield.FieldMap:
        raise ValueError('explicit FieldMap required')
    if type(chunk_nodes) is not int or not 1 <= chunk_nodes <= 1000000:
        raise ValueError('chunk_nodes requires integer in [1,1000000]')
    columns, count = [], 0
    for axis in range(3):
        shape = item.n.copy()
        shape[axis] -= 1
        size = int(np.prod(shape))
        upper = 0.
        for start in range(0, size, chunk_nodes):
            index = np.unravel_index(np.arange(start, min(size, start + chunk_nodes)), shape)
            other = tuple(v + 1 if j == axis else v for j, v in enumerate(index))
            with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
                delta = item.B[other] - item.B[index]
                norm = np.hypot.reduce(delta, axis=1)
                slope = norm / item.d[axis]
            if np.any(~np.isfinite(slope)):
                raise ValueError('field slope outside supported finite range')
            if np.any((np.any(delta != 0, axis=1)) & (slope == 0)):
                raise ValueError('nonzero field slope underflows')
            upper = max(upper, float(np.max(slope)))
        columns.append(upper)
        count += size
    value = float(np.hypot.reduce(columns))
    if not np.isfinite(value):
        raise ValueError('field Lipschitz bound outside supported finite range')
    continuous = not (item.header.extend_y and item.lo[1] == 0 and
                      np.any(item.B[:, 0, :, 1] != 0))
    return SpatialBound(value, tuple(columns), count, continuous)
