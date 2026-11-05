"""Outward evaluation of selected stored-solid expressions and face guards.

Stored dimensions, origins, matrices and half-space/sector normals are exact
binary64 inputs. Placed is the affine pullback (p-origin)@rotation; no exactly
orthogonal matrix or 1-Lipschitz assumption is made. Cone/polygon derived data
are recomputed outward from defining dimensions/vertices, not assumed exact.
Decimal GDML conversion, trigonometric matrix construction, intended/native
geometry and world coverage remain separate, unqualified representation issues.
No defaults, ownership, contact or material-transport behavior is changed.
"""

from dataclasses import dataclass

import numpy as np

from . import tsinterval as iv, tsnav, tssolid, tssurface


def _scalar(value):
    return iv.Interval(np.float64(value))


def _component(p, axis):
    return iv.Interval(p.lo[..., axis], p.hi[..., axis])


def _abs(a):
    return iv.Interval(np.where((a.lo <= 0) & (a.hi >= 0), 0., np.minimum(np.abs(a.lo), np.abs(a.hi))),
                       np.maximum(np.abs(a.lo), np.abs(a.hi)))


def _neg(a):
    return iv.Interval(-a.hi, -a.lo)


def _min(*values):
    return iv.Interval(np.minimum.reduce([a.lo for a in values]), np.minimum.reduce([a.hi for a in values]))


def _max(*values):
    return iv.Interval(np.maximum.reduce([a.lo for a in values]), np.maximum.reduce([a.hi for a in values]))


def _stack(values):
    return iv.Interval(np.stack([a.lo for a in values], axis=-1), np.stack([a.hi for a in values], axis=-1))


def _matmul(points, matrix):
    return _stack([iv.dot(points, iv.Interval(matrix[:, j])) for j in range(3)])


def _cross2(a, b):
    return iv.sub(iv.mul(_component(a, 0), _component(b, 1)), iv.mul(_component(a, 1), _component(b, 0)))


def _cone_side(a, b, height, radial, z, *, inner=False):
    middle = iv.div(iv.add(_scalar(a), _scalar(b)), _scalar(2.))
    slope = iv.div(iv.div(iv.sub(_scalar(b), _scalar(a)), _scalar(2.)), _scalar(height))
    norm = iv.norm(_stack([iv.Interval(np.ones(z.lo.shape)),
                          iv.Interval(np.broadcast_to(slope.lo, z.lo.shape), np.broadcast_to(slope.hi, z.lo.shape))]))
    residual = iv.add(iv.sub(middle, radial), iv.mul(slope, z))
    return iv.div(_neg(residual) if inner else residual, norm)


@dataclass(frozen=True, eq=False)
class Bounds:
    solid: object
    position_box: iv.Interval
    clearance: iv.Interval
    visited_nodes: int
    stored_expression_arithmetic_outward: bool = True
    unit_lipschitz_certified: bool = False
    floating_point_certified: bool = False
    native_query_certified: bool = False
    material_transport: bool = False


class _Evaluator:
    def __init__(self, max_nodes, max_depth):
        if type(max_nodes) is not int or not 1 <= max_nodes <= 100000:
            raise ValueError('integer max_nodes in [1,100000] required')
        if type(max_depth) is not int or not 1 <= max_depth <= 128:
            raise ValueError('integer max_depth in [1,128] required')
        self.max_nodes, self.max_depth, self.count = max_nodes, max_depth, 0

    def query(self, solid, p, depth=1):
        self.count += 1
        if self.count > self.max_nodes or depth > self.max_depth:
            raise ValueError('stored geometry work/depth cap exceeded')
        child = lambda s, points=p: self.query(s, points, depth+1)
        kind = type(solid)
        if kind is tsnav.Placed:
            return child(solid.solid, _matmul(iv.sub(p, iv.Interval(solid.origin)), solid.rotation))
        if kind is tsnav.Intersection:
            return _min(*(child(s) for s in solid.solids))
        if kind is tsnav.Union:
            return _max(*(child(s) for s in solid.solids))
        if kind is tsnav.Difference:
            return _min(child(solid.first), _neg(child(solid.second)))
        if kind is tsnav.SplitUnion:
            a, b, h = child(solid.first), child(solid.second), child(solid.plane)
            return _max(_min(a, h), _min(b, _neg(h)), _min(a, b))
        if kind is tssolid.Sector:
            base = child(solid.solid)
            if solid.deltaphi == 2*np.pi:
                return base
            xy = iv.Interval(p.lo[:, :2], p.hi[:, :2])
            sides = [iv.dot(xy, iv.Interval(normal)) for normal in solid._normals]
            angular = _min(*sides) if solid.deltaphi <= np.pi else _max(*sides)
            return _min(base, angular)
        if kind is tsnav.Sphere:
            return iv.sub(_scalar(solid.radius), iv.norm(p))
        if kind is tsnav.HalfSpace:
            return iv.sub(_scalar(solid.offset), iv.dot(p, iv.Interval(solid.normal)))
        if kind in (tsnav.Box, tsnav.Tube, tssurface._Remainder):
            remainder = kind is tssurface._Remainder
            shape = solid.primitive if remainder else solid
            if type(shape) is tsnav.Box:
                values = []
                for axis in range(3):
                    coordinate = _component(p, axis)
                    constraint = iv.mul(_scalar(solid.sign), coordinate) if remainder and axis == solid.axis else _abs(coordinate)
                    values.append(iv.sub(_scalar(shape.half_lengths[axis]), constraint))
                return _min(*values)
            if type(shape) is not tsnav.Tube:
                raise ValueError('unsupported material face remainder primitive')
            radial = iv.norm(iv.Interval(p.lo[:, :2], p.hi[:, :2]))
            z = _component(p, 2)
            cap = iv.mul(_scalar(solid.sign), z) if remainder else _abs(z)
            values = [iv.sub(_scalar(shape.rmax), radial), iv.sub(_scalar(shape.half_length), cap)]
            if shape.rmin > 0:
                values.append(iv.sub(radial, _scalar(shape.rmin)))
            return _min(*values)
        if kind is tsnav.Torus:
            radial = iv.norm(iv.Interval(p.lo[:, :2], p.hi[:, :2]))
            ring = iv.norm(_stack([iv.sub(radial, _scalar(solid.major_radius)), _component(p, 2)]))
            return iv.sub(_scalar(solid.minor_radius), ring)
        if kind is tssolid.Cone:
            radial = iv.norm(iv.Interval(p.lo[:, :2], p.hi[:, :2]))
            z = _component(p, 2)
            values = [_cone_side(solid.rmax1, solid.rmax2, solid.half_length, radial, z),
                      iv.sub(_scalar(solid.half_length), _abs(z))]
            if solid.rmin1 or solid.rmin2:
                values.append(_cone_side(solid.rmin1, solid.rmin2, solid.half_length, radial, z, inner=True))
            return _min(*values)
        if kind is tssolid.Extrusion:
            return self.polygon(solid, p)
        raise ValueError('unsupported stored solid type; arbitrary clearance callables refused')

    def polygon(self, solid, p):
        xy = iv.Interval(p.lo[:, :2], p.hi[:, :2])
        distances, inside, ambiguous = [], np.zeros(len(p.lo), dtype=bool), np.zeros(len(p.lo), dtype=bool)
        if len(solid.vertices) > self.max_nodes-self.count:
            raise ValueError('stored geometry polygon work cap exceeded')
        self.count += len(solid.vertices)
        for a, b in zip(solid.vertices, np.roll(solid.vertices, -1, axis=0)):
            edge = iv.sub(iv.Interval(b), iv.Interval(a))
            relative = iv.sub(xy, iv.Interval(a))
            square = iv.sum_last(iv.square(edge))
            t = iv.div(iv.dot(relative, edge), square)
            t = iv.Interval(np.clip(t.lo, 0., 1.), np.clip(t.hi, 0., 1.))
            closest = iv.mul(iv.Interval(t.lo[:, None], t.hi[:, None]), edge)
            distances.append(iv.norm(iv.sub(relative, closest)))
            y = _component(xy, 1)
            above_a, below_a = a[1] > y.hi, a[1] <= y.lo
            above_b, below_b = b[1] > y.hi, b[1] <= y.lo
            crosses = (above_a & below_b) | (above_b & below_a)
            never = (above_a & above_b) | (below_a & below_b)
            cross = _cross2(edge, relative)
            hit = cross.lo > 0 if b[1] > a[1] else cross.hi < 0
            miss = cross.hi <= 0 if b[1] > a[1] else cross.lo >= 0
            inside ^= crosses & hit
            ambiguous |= ~(never | (crosses & (hit | miss)))
        distance = _min(*distances)
        signed = iv.Interval(np.where(ambiguous | ~inside, -distance.hi, distance.lo),
                             np.where(ambiguous | inside, distance.hi, -distance.lo))
        return _min(signed, iv.sub(_scalar(solid.half_length), _abs(_component(p, 2))))


def evaluate(solid, points, *, max_nodes=10000, max_depth=64):
    """Enclose the exact stored scalar expression on a whole Cartesian box.

    Positive lower/negative upper proves a stored-expression sign only. No
    Euclidean-ball clearance or unit-Lipschitz guarantee is supplied. Unknown
    solids and unsupported arithmetic fail closed. Source must remain unchanged.
    """
    if type(points) is not iv.Interval or points.lo.ndim != 2 or points.lo.shape[1] != 3:
        raise ValueError('explicit Interval points with shape (N,3) required')
    evaluator = _Evaluator(max_nodes, max_depth)
    result = evaluator.query(solid, points)
    return Bounds(solid, points, result, evaluator.count)


@dataclass(frozen=True, eq=False)
class FaceBounds:
    face: tssurface.Face
    position_box: iv.Interval
    primitive_plane: iv.Interval
    guard: Bounds
    plane_discrepancy_upper_mm: np.ndarray
    normal_difference_upper: float
    stored_transform_arithmetic_outward: bool = True
    floating_point_certified: bool = False
    native_query_certified: bool = False
    material_transport: bool = False


def face_bounds(face, points, *, max_nodes=10000, max_depth=64):
    """Bound extracted plane versus actual affine-pullback primitive face locally.

    The primitive plane uses the exact stored nested transforms, NOT an asserted
    exactly orthogonal forward product. Its covector is mathematically normalized
    for physical signed-distance units. Plane discrepancy is uniform on this box
    only; intended GDML/native geometry and unmodeled siblings remain unqualified.
    Extraction provenance/guard consistency is a caller precondition, not checked
    by the Face type check; manually assembled/replaced faces are unsupported.
    """
    if type(face) is not tssurface.Face:
        raise ValueError('explicit extracted Face required')
    guard = evaluate(face.guard, points, max_nodes=max_nodes, max_depth=max_depth)
    shape = face.volume.solid
    columns = [iv.Interval(np.eye(3)[:, j]) for j in range(3)]
    offset = iv.Interval(np.zeros(3))
    count = 0
    while type(shape) in (tsnav.Placed, tssolid.Sector):
        count += 1
        if count > max_depth:
            raise ValueError('face transform depth cap exceeded')
        if type(shape) is tssolid.Sector:
            if shape.deltaphi != 2*np.pi:
                raise ValueError('partial sector is not a complete primitive face')
            shape = shape.solid
            continue
        old = columns
        columns = [iv.sum_last(iv.mul(_stack(old), iv.Interval(shape.rotation[:, j]))) for j in range(3)]
        offset = _matmul(iv.sub(offset, iv.Interval(shape.origin)), shape.rotation)
        shape = shape.solid
    if type(shape) not in (tsnav.Box, tsnav.Tube) or face.key not in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+'):
        raise ValueError('unsupported complete primitive face')
    axis, sign = 'xyz'.index(face.key[0]), 1. if face.key[1] == '-' else -1.
    if type(shape) is tsnav.Tube and axis != 2:
        raise ValueError('only tube endcaps supported')
    half = shape.half_lengths[axis] if type(shape) is tsnav.Box else shape.half_length
    normal = iv.mul(columns[axis], _scalar(sign))
    constant = iv.add(iv.mul(_component(offset, axis), _scalar(sign)), _scalar(half))
    norm = iv.norm(normal)
    unit = iv.div(normal, norm)
    constant = iv.div(constant, norm)
    primitive = iv.add(iv.dot(points, unit), constant)
    stored_unit = iv.normalize(iv.Interval(face.inward_normal))
    stored_constant = _neg(iv.dot(iv.Interval(face.origin), stored_unit))
    delta_normal = iv.sub(unit, stored_unit)
    delta_constant = iv.sub(constant, stored_constant)
    discrepancy = _abs(iv.add(iv.dot(points, delta_normal), delta_constant)).hi
    frozen = np.frombuffer(discrepancy.tobytes(), dtype=discrepancy.dtype).reshape(discrepancy.shape)
    return FaceBounds(face, points, primitive, guard, frozen, float(iv.norm(delta_normal).hi))
