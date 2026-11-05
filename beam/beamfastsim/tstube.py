"""Whole-interval ideal single-map path sets, not a navigator or restart State.

An immutable stored map, a proved whole reachable-box domain cover and its
all-edge gradient bound support the carried Lorentz model error. A qualified
uniform-helix set at an interior parameter plus unit-speed/curvature reach
covers every parameter in [lo,hi]. Direct interval geometry can query the
resulting Cartesian set without an unproved 1-Lipschitz assumption. All initial
uncertainties remain explicit assumptions; stored/native representation, decay,
material response and full physical transport remain outside the contract.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import tsbudget, tsfield, tsgeomiv, tshelix, tsinterval as iv, tsquery, tsstep


def _fixed(a):
    return iv._fixed(np.asarray(a))


def _column(a):
    return iv.Interval(a.lo[:, None], a.hi[:, None])


def _nnadd(a, b):
    result = iv.add(a, b)
    zero = (a.hi == 0) & (b.hi == 0)
    return iv.Interval(np.where(zero, 0., np.maximum(0., result.lo)), np.where(zero, 0., result.hi))


@dataclass(frozen=True, eq=False, init=False)
class Context:
    item: tsfield.FieldMap
    bounds: tsbudget.MapBounds

    def __init__(self, item, *, chunk_nodes=65536):
        if type(item) is not tsfield.FieldMap:
            raise ValueError('explicit FieldMap required')
        old = item.header
        flags = []
        for v in (old.extend_y, old.explicit_origin):
            raw = np.asarray(v)
            if raw.shape != () or raw.dtype.kind != 'b':
                raise ValueError('scalar map reflection/origin flags required')
            flags.append(bool(raw))
        header = tsfield.GridHeader(old.path, old.grid_origin, old.origin_shift, old.n, old.d, *flags)
        for key in ('grid_origin', 'origin_shift', 'n', 'd'):
            object.__setattr__(header, key, _fixed(getattr(header, key)))
        frozen = tsfield.FieldMap(header, item.B, item.byte_order)
        object.__setattr__(frozen, 'B', _fixed(frozen.B))
        object.__setattr__(self, 'item', frozen)
        object.__setattr__(self, 'bounds', tsbudget.scan(frozen, chunk_nodes=chunk_nodes))


@dataclass(frozen=True, eq=False)
class Tube:
    context: Context
    initial_x_mm: np.ndarray
    initial_direction_raw: np.ndarray
    momentum: np.ndarray
    pdg: np.ndarray
    initial_position_error_mm: np.ndarray
    initial_direction_error: np.ndarray
    lo_mm: np.ndarray
    hi_mm: np.ndarray
    midpoint_mm: np.ndarray
    query: tsquery.Query
    reference: tshelix.Helix
    endpoint_reference: tshelix.Helix
    model: tsbudget.Errors
    position: iv.Interval
    direction: iv.Interval
    endpoint_position: iv.Interval
    endpoint_direction: iv.Interval
    path_arithmetic_outward: bool = True
    whole_interval: bool = True
    initial_errors_are_assumptions: bool = True
    floating_point_certified: bool = False
    native_query_certified: bool = False
    restart_state: bool = False
    material_transport: bool = False


def enclose(context, x, direction, momentum, pdg, lo, hi, *,
            position_error_mm, direction_error, max_cells=65536):
    """Bound every ideal vacuum Lorentz state in [lo,hi] from ORIGINAL inputs.

    Whole initial-reach box x +/- (ex+hi) must be covered continuously by one
    stored map. A reflected zero seam must have EXACTLY zero stored By on every
    seam node; a gap or discontinuity fails closed, never tolerance-welds. Input
    scalar p and true particle charge equal the supplied exact model values.
    position_error_mm bounds Euclidean initial position discrepancy; direction_error
    bounds chord distance of the true UNIT direction from mathematical normalization
    of stored raw direction, not an angle or raw-vector error. No uncertain momentum
    magnitude, path advancement, material classification or restart is supported.
    """
    if type(context) is not Context:
        raise ValueError('immutable tube Context required')
    x, raw = iv.Interval(x).lo, iv.Interval(direction).lo
    if x.ndim != 2 or x.shape[1] != 3 or raw.shape != x.shape:
        raise ValueError('binary64 x/direction shape (N,3) required')
    n = len(x)
    p = tshelix._rows(momentum, n, 'momentum', positive=True)
    left, right = tshelix._rows(lo, n, 'lo'), tshelix._rows(hi, n, 'hi')
    ex, eu = tshelix._rows(position_error_mm, n, 'position_error_mm'), tshelix._rows(direction_error, n, 'direction_error')
    if np.any(left > right) or np.any(eu > 2):
        raise ValueError('ordered path interval and direction_error<=2 required')
    reach = _nnadd(iv.Interval(ex), iv.Interval(right))
    query = tsquery.evaluate(context.item, x, position_radius_mm=reach.hi, max_cells=max_cells)
    if context.item.header.extend_y:
        seam = (query.position_box.lo[:, 1] <= 0) & (query.position_box.hi[:, 1] >= 0)
        if np.any(seam):
            h = context.item.header
            y0 = Fraction(float(h.grid_origin[1]))-Fraction(float(h.origin_shift[1]))
            if y0 != 0 or not context.bounds.reflection_continuous:
                raise ValueError('whole reachable box crosses an unproved continuous reflected seam')
    middle = left+(right-left)/2
    if np.any((middle < left) | (middle > right)):
        raise ValueError('midpoint outside supplied interval')
    dt = np.maximum(iv.sub(iv.Interval(middle), iv.Interval(left)).hi,
                    iv.sub(iv.Interval(right), iv.Interval(middle)).hi)
    dt = np.where(left == right, 0., np.maximum(0., dt))
    reference = tshelix.evaluate(x, raw, p, pdg, query.value_tesla, middle)
    endpoint = tshelix.evaluate(x, raw, p, pdg, query.value_tesla, right)
    model = tsbudget.errors(context.bounds, p, right, position_error_mm=ex,
                            direction_error=eu, field_error_tesla=query.evaluation_error_tesla)
    position_radius = _nnadd(iv.Interval(dt), iv.Interval(model.position_mm))
    position = iv.inflate(reference.position, _column(position_radius))
    frozen_curvature = iv.div(tshelix._mul(iv.norm(iv.Interval(query.value_tesla)), iv.Interval(np.float64(tsstep.K))), iv.Interval(p))
    turn = tshelix._mul(frozen_curvature, iv.Interval(dt))
    turn = iv.Interval(np.zeros(n), np.minimum(2., np.maximum(0., turn.hi)))
    direction_radius = _nnadd(turn, iv.Interval(model.direction))
    direction = iv.inflate(reference.direction, _column(direction_radius))
    direction = iv.Interval(np.maximum(-1., direction.lo), np.minimum(1., direction.hi))
    endpoint_position = iv.inflate(endpoint.position, iv.Interval(model.position_mm[:, None]))
    endpoint_direction = iv.inflate(endpoint.direction, iv.Interval(model.direction[:, None]))
    endpoint_direction = iv.Interval(np.maximum(-1., endpoint_direction.lo), np.minimum(1., endpoint_direction.hi))
    return Tube(context, *(_fixed(v) for v in (x, raw, p, endpoint.pdg, ex, eu, left, right, middle)),
                query, reference, endpoint, model, position, direction, endpoint_position, endpoint_direction)


def guard(tube, face, *, max_nodes=10000, max_depth=64):
    """Direct stored finite-face guard/plane bounds on the WHOLE incoming set.

    This is hypothetical vacuum extension only, not physical motion through a
    material face. Guard positivity neither proves a crossing nor advances state.
    Intended/native geometry and extraction provenance remain caller contracts.
    """
    if type(tube) is not Tube:
        raise ValueError('explicit whole-interval Tube required')
    return tsgeomiv.face_bounds(face, tube.position, max_nodes=max_nodes, max_depth=max_depth)
