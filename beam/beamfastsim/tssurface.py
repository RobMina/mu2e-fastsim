"""Conditional first-entry intervals for isolated planar material faces.

Only whole placed boxes and full tubes have supported faces. Guards exclude
rims, opposite faces, mothers and competing selected volumes over the WHOLE
incoming vacuum extension. Strict endpoint signs plus a positive derivative
bound establish one first contact within the segment, never material transport.
Plane/root arithmetic is outward under restricted binary64 contracts, but
trajectory/geometry/representation declarations remain assumptions; zero is an
idealization. No full floating-point or native certificate is issued.
"""

from dataclasses import dataclass

import numpy as np

from . import tsarc, tscarry, tsinterval as iv, tslocate, tsnav, tsplane, tssolid

GUARD_PENDING = 1
SIGN_PENDING = 2
TRANSVERSE_PENDING = 3
ISOLATED = 4
UNCERTAINTY_LIMIT = 5
ITERATION_LIMIT = 6
NUMERICAL_LIMIT = 7
STATUS = {GUARD_PENDING: 'other_surface_or_scope_unresolved', SIGN_PENDING: 'endpoint_signs_unresolved',
          TRANSVERSE_PENDING: 'transversality_unresolved', ISOLATED: 'conditional_entry_interval_isolated',
          UNCERTAINTY_LIMIT: 'conditional_entry_interval_uncertainty_limit',
          ITERATION_LIMIT: 'conditional_entry_interval_iteration_limit',
          NUMERICAL_LIMIT: 'conditional_entry_interval_numerical_limit'}


@dataclass(frozen=True)
class _Remainder(tsnav._Solid):
    primitive: object
    axis: int
    sign: int

    def _clearance(self, p):
        if type(self.primitive) is tsnav.Box:
            distances = self.primitive.half_lengths - np.abs(p)
            distances[:, self.axis] = self.primitive.half_lengths[self.axis] - self.sign*p[:, self.axis]
            return np.min(distances, axis=1)
        shape = self.primitive
        radial = np.hypot(p[:, 0], p[:, 1])
        clearance = np.minimum(shape.rmax-radial, shape.half_length-self.sign*p[:, 2])
        return np.minimum(clearance, radial-shape.rmin) if shape.rmin > 0 else clearance


@dataclass(frozen=True, eq=False)
class Face:
    partition: tslocate.Partition
    volume_index: int
    mother_index: int
    key: str
    origin: np.ndarray
    inward_normal: np.ndarray
    guard: object
    floating_point_certified: bool = False
    native_query_certified: bool = False

    @property
    def volume(self):
        return self.partition.volumes[self.volume_index]

    def signed(self, points):
        p = tsnav._real(points, 'points')
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('points require (N,3) shape')
        result = (p-self.origin) @ self.inward_normal
        if np.any(~np.isfinite(result)):
            raise ValueError('nonfinite plane residual')
        return result


def faces(partition, volume_index):
    """Extract complete primitive faces, never an arbitrary Boolean operand face.

    Face identity includes the exact target Volume and its complete placement
    path. Positive guard clearance means the target's OTHER constraints are
    interior, its mother is interior, and every competing material/root is out.
    Completeness excludes unmodeled world siblings only by caller assumption.
    """
    if type(partition) is not tslocate.Partition or type(volume_index) is not int:
        raise ValueError('explicit Partition and integer volume index required')
    if not 0 <= volume_index < len(partition.volumes) or partition.volumes[volume_index].vacuum:
        raise ValueError('selected material leaf required')
    target = partition.volumes[volume_index]
    solid, placements = target.solid, []
    matrix, origin = np.eye(3), np.zeros(3)
    while True:
        if type(solid) is tsnav.Placed:
            placements.append(solid)
            origin = origin + matrix @ solid.origin
            matrix = matrix @ solid.rotation
            solid = solid.solid
        elif type(solid) is tssolid.Sector and solid.deltaphi == 2*np.pi:
            solid = solid.solid
        else:
            break
    if type(solid) not in (tsnav.Box, tsnav.Tube):
        raise ValueError('only complete box faces and full tube endcaps supported')
    offset, mother_index, cell = 0, None, None
    for candidate in partition.cells:
        if offset < volume_index < offset+1+len(candidate.daughters):
            mother_index, cell = offset, candidate
            break
        offset += 1+len(candidate.daughters)
    if cell is None:
        raise ValueError('selected material must be a direct leaf')
    constraints = [cell.mother.solid]
    for i, volume in enumerate(partition.volumes):
        if i != volume_index and not volume.vacuum and volume.root == target.root:
            constraints.append(tsnav.Difference(cell.mother.solid, volume.solid))
    constraints.extend(tsnav.Difference(cell.mother.solid, other.mother.solid)
                       for other in partition.cells if other is not cell)
    result = []
    for axis in (range(3) if type(solid) is tsnav.Box else (2,)):
        half = solid.half_lengths[axis] if type(solid) is tsnav.Box else solid.half_length
        for sign in (1, -1):
            remainder = _Remainder(solid, axis, sign)
            for placement in reversed(placements):
                remainder = tsnav.Placed(remainder, placement.origin, placement.rotation)
            local = np.zeros(3)
            local[axis] = -sign*half
            normal = sign*matrix[:, axis]
            normal = normal/np.hypot.reduce(normal)
            center = origin+matrix @ local
            result.append(Face(partition, volume_index, mother_index,
                f'{"xyz"[axis]}{"-" if sign == 1 else "+"}',
                tsnav._fixed(center, (3,), 'face origin'),
                tsnav._fixed(normal, (3,), 'face inward normal'),
                tsnav.Intersection(remainder, *constraints)))
    return tuple(result)


@dataclass(frozen=True)
class Entry:
    face: Face
    segment: tscarry.Segment
    status: np.ndarray
    unique_entry: np.ndarray
    source_index: np.ndarray
    lo_mm: np.ndarray
    hi_mm: np.ndarray
    derivative_lower: np.ndarray
    guard_margin_mm: np.ndarray
    iterations: np.ndarray
    plane_error_mm: np.ndarray
    projection_error: np.ndarray
    geometry_error_mm: np.ndarray
    path_atol_mm: float
    max_iterations: int
    floating_point_certified: bool = False
    native_query_certified: bool = False
    material_transport: bool = False
    plane_root_arithmetic_outward: bool = True


def entry(segment, face, *, plane_error_mm, projection_error, geometry_error_mm,
          path_atol=1e-3, max_iterations=64):
    """Bound first contact of an incoming VACUUM extension, without advancement.

    Required plane_error_mm bounds normal/origin representation relative to
    the intended face uniformly on the whole reachable tube; projection_error
    bounds its direction projection discrepancy. Stored-plane arithmetic and
    root operations are outward under tsinterval contracts. geometry_error_mm
    bounds guard evaluation. Geometry/trajectory assumptions remain unproved.
    Segment evaluation declarations cover normalized direction and all positions.
    The hypothetical far endpoint is never a state after real material contact.
    Exit handling requires a material-dynamics enclosure and is NOT implemented.
    """
    if type(segment) is not tscarry.Segment or type(face) is not Face:
        raise ValueError('explicit carried Segment and extracted Face required')
    if type(max_iterations) is not int or not 0 <= max_iterations <= 4096:
        raise ValueError('max_iterations requires integer in [0,4096]')
    atol = tsnav._scalar(path_atol, 'path_atol', positive=True)
    n = len(segment.length)
    rows = np.arange(n)
    pe = tsarc._lanes(plane_error_mm, (n,), 'plane_error_mm', nonnegative=True)
    qe = tsarc._lanes(projection_error, (n,), 'projection_error', nonnegative=True)
    ge = tsarc._lanes(geometry_error_mm, (n,), 'geometry_error_mm', nonnegative=True)
    if np.any(segment.length <= 0):
        raise ValueError('strictly positive incoming segment length required')
    balls = segment.balls(rows, np.zeros(n), segment.length)
    midpoint_margin = iv.sub(iv.sub(iv.Interval(tsnav._query(face.guard, balls.midpoint)),
        iv.Interval(balls.midpoint_radius_mm)), iv.Interval(ge)).lo
    chord_margin = iv.sub(iv.sub(iv.Interval(tsnav._query(face.guard, balls.chord_center)),
        iv.Interval(balls.chord_radius_mm)), iv.Interval(ge)).lo
    margin = np.maximum(midpoint_margin, chord_margin)
    midpoint = segment.length/2
    error = segment.errors(rows, midpoint)
    _, u = segment.reference_state(rows, midpoint)
    projection = tsplane.projection(iv.Interval(u), iv.Interval(face.inward_normal))
    width = np.maximum(midpoint, iv.sub(iv.Interval(segment.length), iv.Interval(midpoint)).hi)
    direction_error = iv.add(iv.Interval(error.direction), iv.Interval(segment.direction_error))
    turn = iv.mul(iv.Interval(segment.curvature), iv.Interval(width))
    lower = iv.sub(iv.sub(iv.sub(projection, direction_error), turn), iv.Interval(qe)).lo

    def residual(indices, s):
        x, _ = segment.reference_state(indices, s)
        model = segment.errors(indices, s)
        value = tsplane.signed(iv.Interval(x), iv.Interval(face.origin), iv.Interval(face.inward_normal))
        uncertainty = iv.add(iv.add(iv.Interval(model.position_mm),
            iv.Interval(segment.position_error_mm[indices])), iv.Interval(pe[indices]))
        uncertainty = iv.Interval(np.maximum(0., uncertainty.lo), uncertainty.hi)
        return iv.inflate(value, uncertainty)

    f0 = residual(rows, np.zeros(n))
    f1 = residual(rows, segment.length)
    if np.any(~np.isfinite(margin)) or np.any(~np.isfinite(lower)):
        raise ValueError('nonfinite entry prerequisite')
    status = np.full(n, GUARD_PENDING, dtype=np.int8)
    guarded = margin > 0
    status[guarded] = TRANSVERSE_PENDING
    transverse = guarded & (lower > 0)
    status[transverse] = SIGN_PENDING
    unique = transverse & (f0.hi < 0) & (f1.lo > 0)
    status[unique] = ITERATION_LIMIT
    lo, hi = np.zeros(n), np.where(unique, segment.length, 0.)
    iterations = np.zeros(n, dtype=np.int64)
    active = unique.copy()
    for _ in range(max_iterations):
        complete = active & (iv.sub(iv.Interval(hi), iv.Interval(lo)).hi <= atol)
        status[complete], active[complete] = ISOLATED, False
        use = np.flatnonzero(active)
        if not len(use):
            break
        mid = lo[use]+(hi[use]-lo[use])/2
        representable = (mid > lo[use]) & (mid < hi[use])
        stop = use[~representable]
        status[stop], active[stop] = NUMERICAL_LIMIT, False
        use, mid = use[representable], mid[representable]
        if not len(use):
            continue
        f = residual(use, mid)
        iterations[use] += 1
        before, after = f.hi < 0, f.lo > 0
        lo[use[before]] = mid[before]
        hi[use[after]] = mid[after]
        uncertain = ~(before | after)
        idx, center = use[uncertain], mid[uncertain]
        if len(idx):
            absolute = np.maximum(np.abs(f.lo[uncertain]), np.abs(f.hi[uncertain]))
            radius = iv.div(iv.Interval(absolute), iv.Interval(lower[idx])).hi
            left = np.maximum(lo[idx], iv.sub(iv.Interval(center), iv.Interval(radius)).lo)
            right = np.minimum(hi[idx], iv.add(iv.Interval(center), iv.Interval(radius)).hi)
            if np.any(left > right):
                raise ValueError('contradictory conditional root interval')
            collapsed = (left == right) & (radius > 0)
            keep = idx[~collapsed]
            lo[keep], hi[keep] = left[~collapsed], right[~collapsed]
            status[idx] = UNCERTAINTY_LIMIT
            status[idx[collapsed]] = NUMERICAL_LIMIT
            active[idx] = False
    width = iv.sub(iv.Interval(hi), iv.Interval(lo)).hi
    status[unique & (width <= atol) & (status != NUMERICAL_LIMIT)] = ISOLATED
    return Entry(face, segment, *[tscarry._fixed(a) for a in (status, unique, segment.initial.source_index,
        lo, hi, lower, margin, iterations, pe, qe, ge)], atol, max_iterations)
