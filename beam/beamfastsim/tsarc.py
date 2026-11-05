"""Uniform-field curved-path ownership enclosures, not map/native transport.

For a unit-speed curve with ||r''|| <= k over length L, its deviation from the
endpoint chord is <= k*L**2/8. A chord-centred enclosing ball and a separately
queried arc-midpoint ball provide conditional exact-arithmetic containment.
Declared evaluation/geometry errors are caller assumptions, NOT certificates.
Core helix evaluation, floating geometry and radius arithmetic are not rigorous
interval arithmetic. No variable-field RK4 error, material physics, boundary
root isolation or native geometry agreement is established by these diagnostics.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import tslocate, tsnav, tsstep

RESOLVED = 1
WIDTH_LIMIT = 2
DEPTH_LIMIT = 3
MIDPOINT_BALL = 1
CHORD_BALL = 2
MAX_PHASE = 64 * np.pi
STATUS = {RESOLVED: 'resolved_interval', WIDTH_LIMIT: 'unresolved_width_limit',
          DEPTH_LIMIT: 'unresolved_depth_limit'}


def _lanes(value, shape, name, *, nonnegative=False):
    raw = tsnav._real(value, name)
    try:
        array = np.broadcast_to(raw, shape)
    except ValueError as exc:
        raise ValueError(f'{name} must broadcast to {shape}') from exc
    if nonnegative and np.any(array < 0):
        raise ValueError(f'{name} must be nonnegative')
    return tsnav._fixed(array, shape, name)


@dataclass(frozen=True, eq=False, init=False)
class UniformHelix:
    """Immutable, per-row uniform field model using matched core ehelix.helix.

    position_error_mm explicitly declares an error bound for EVERY returned
    position relative to the ideal unit-speed curve, including initial state and
    arithmetic. Zero requests a numerical idealization, not a proven zero error.
    Fields are constant per row; field-map callables are deliberately rejected.
    The supported unwrapped phase range is |K*B*length/p| <= 64*pi.
    """

    x: np.ndarray
    direction: np.ndarray
    momentum: np.ndarray
    charge: np.ndarray
    pdg: np.ndarray
    field: np.ndarray
    length: np.ndarray
    curvature: np.ndarray
    position_error_mm: np.ndarray

    def __init__(self, x, p3, pdg, field, length, *, position_error_mm):
        x, p3 = tsnav._real(x, 'x'), tsnav._real(p3, 'p3')
        if x.ndim != 2 or x.shape[1] != 3 or p3.shape != x.shape:
            raise ValueError('x and p3 require matching (N,3) shapes')
        n = len(x)
        pdg = _lanes(pdg, (n,), 'pdg')
        if np.any(~np.isin(pdg, (13, -13, 211, -211))):
            raise ValueError('pdg must be mu+/mu-/pi+/pi-')
        q = np.where((pdg == 13) | (pdg == -211), -1., 1.)
        raw_field = tsnav._real(field, 'uniform field')
        if raw_field.shape not in ((3,), (n, 3)):
            raise ValueError('uniform field requires (3,) or (N,3) shape')
        field = _lanes(raw_field, (n, 3), 'uniform field')
        length = _lanes(length, (n,), 'length', nonnegative=True)
        error = _lanes(position_error_mm, (n,), 'position_error_mm', nonnegative=True)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            p = np.hypot.reduce(p3, axis=1)
            bm = np.linalg.norm(field, axis=1)
            k = tsstep.K * bm / p
            phase = k * length
        if np.any(~np.isfinite(p) | (p <= 0)) or np.any(~np.isfinite(bm)):
            raise ValueError('momentum/field magnitude outside finite supported range')
        if np.any(~np.isfinite(k) | ~np.isfinite(phase)) or np.any(phase > MAX_PHASE):
            raise ValueError('curvature/phase outside supported uniform helix domain')
        if np.any((bm > 0) & (k == 0)):
            raise ValueError('nonzero field curvature underflows')
        if np.any((np.any(field != 0, axis=1)) & (bm == 0)):
            raise ValueError('nonzero field magnitude underflows in core helix')
        for name, value in dict(x=x, direction=p3 / p[:, None], momentum=p,
                                charge=q, pdg=pdg, field=field, length=length,
                                curvature=k, position_error_mm=error).items():
            object.__setattr__(self, name, tsnav._fixed(value, np.shape(value), name))

    def state(self, source, arclength):
        """Evaluate absolute unwrapped arc lengths from the ORIGINAL input state."""
        raw = np.asarray(source)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu':
            raise ValueError('source requires one-dimensional integer row indices')
        if np.any(raw >= len(self.x)) or np.any(raw < 0):
            raise ValueError('source index out of range')
        source = raw.astype(np.int64)
        s = tsnav._real(arclength, 'arclength')
        if s.shape != source.shape or np.any(s < 0) or np.any(s > self.length[source]):
            raise ValueError('arclength requires in-range value per source row')
        from mu2efastsim.ehelix import helix
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            x, u = helix(self.x[source], self.direction[source], self.momentum[source],
                         self.charge[source], self.field[source], s)
            p3 = u * self.momentum[source, None]
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p3)):
            raise ValueError('nonfinite uniform helix state')
        stalled = (s > 0) & (self.curvature[source] * s <= np.pi / 2) & np.all(x == self.x[source], axis=1)
        if np.any(stalled):
            raise ValueError('positive uniform helix displacement is not representable')
        return x, p3


@dataclass(frozen=True)
class Result:
    source_index: np.ndarray
    s_lo: np.ndarray
    s_hi: np.ndarray
    depth: np.ndarray
    status: np.ndarray
    midpoint_status: np.ndarray
    volume_index: np.ndarray
    boundary_index: np.ndarray
    width: np.ndarray
    ball_kind: np.ndarray
    ball_radius: np.ndarray
    clearance_margin: np.ndarray


def _enclosures(curve, source, lo, hi):
    """Different centres require separate geometry queries; never mix radii."""
    width = hi - lo
    middle = lo + width / 2
    a, _ = curve.state(source, lo)
    b, _ = curve.state(source, hi)
    m, _ = curve.state(source, middle)
    center = a / 2 + b / 2
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        chord_radius = np.maximum(np.hypot.reduce(a - center, axis=1),
                                   np.hypot.reduce(b - center, axis=1))
        sagitta = (curve.curvature[source] * width / 8) * width
        chord_radius = chord_radius + sagitta + curve.position_error_mm[source]
        midpoint_radius = np.maximum(middle - lo, hi - middle) + curve.position_error_mm[source]
    if not all(np.all(np.isfinite(v)) for v in (width, center, chord_radius, midpoint_radius)):
        raise ValueError('nonfinite curved enclosure arithmetic')
    if np.any((width > 0) & (curve.curvature[source] > 0) & (sagitta == 0)):
        raise ValueError('nonzero curved enclosure term underflows')
    return middle, m, midpoint_radius, center, chord_radius


def partition(curve, geometry, *, geometry_error_mm, atol=1e-6, max_depth=64,
              max_intervals=1000000):
    """Complete ordered arc-length cover, conditional on all declared bounds.

    Strictly enclosed intervals retain exact selected owner indices. Width/depth
    stops remain unresolved with no owner; they do not imply physical crossings.
    atol is unresolved arc length, never an epsilon or geometry expansion. No
    frozen-field approximation is silently substituted for a varying-field path.
    """
    if type(curve) is not UniformHelix or not isinstance(geometry, tslocate.Partition):
        raise ValueError('explicit UniformHelix and selected-cell Partition required')
    error = tsnav._scalar(geometry_error_mm, 'geometry_error_mm', nonnegative=True)
    atol = tsnav._scalar(atol, 'atol', positive=True)
    for value, name, low, high in ((max_depth, 'max_depth', 0, 64),
                                   (max_intervals, 'max_intervals', 1, 1000000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name} requires integer in [{low},{high}]')
    if len(curve.x) > max_intervals:
        raise ValueError('max_intervals exceeded')
    source = np.arange(len(curve.x))
    lo, hi = np.zeros(len(source)), curve.length.copy()
    depth = np.zeros(len(source), dtype=np.int64)
    parts, count = [], 0
    while len(source):
        middle, midpoint, rm, center, rc = _enclosures(curve, source, lo, hi)
        lm, lc = geometry.locate(midpoint), geometry.locate(center)
        margin_m = lm.clearance - error - rm
        margin_c = lc.clearance - error - rc
        safe_m = np.isin(lm.status, (tslocate.VACUUM, tslocate.MATERIAL)) & (margin_m > 0)
        safe_c = np.isin(lc.status, (tslocate.VACUUM, tslocate.MATERIAL)) & (margin_c > 0)
        if np.any(safe_m & safe_c & (lm.volume_index != lc.volume_index)):
            raise ValueError('contradictory ownership enclosure guarantees')
        safe = safe_m | safe_c
        use_chord = safe_c & (~safe_m | (margin_c > margin_m))
        kind = np.where(safe, np.where(use_chord, CHORD_BALL, MIDPOINT_BALL), 0)
        radius = np.where(safe, np.where(use_chord, rc, rm), 0.)
        margin = np.where(safe, np.where(use_chord, margin_c, margin_m), 0.)
        owner = np.where(safe, np.where(use_chord, lc.volume_index, lm.volume_index), -1)
        small = hi - lo <= atol
        capped = depth >= max_depth
        done = safe | small | capped
        if np.any(done):
            status = np.where(safe, RESOLVED, np.where(small, WIDTH_LIMIT, DEPTH_LIMIT))
            parts.append(tuple(v[done] for v in (source, lo, hi, depth, status, lm.status,
                              owner, lm.boundary_index, hi - lo, kind, radius, margin)))
            count += int(np.count_nonzero(done))
        todo = ~done
        if not np.any(todo):
            break
        if count + 2 * np.count_nonzero(todo) > max_intervals:
            raise ValueError('max_intervals exceeded; no partial curved cover returned')
        if np.any((middle[todo] <= lo[todo]) | (middle[todo] >= hi[todo])):
            raise ValueError('arc-length subdivision is not representable')
        source, depth = np.repeat(source[todo], 2), np.repeat(depth[todo] + 1, 2)
        lo, hi = (np.column_stack((lo[todo], middle[todo])).ravel(),
                  np.column_stack((middle[todo], hi[todo])).ravel())
    if not parts:
        integer = np.empty(0, dtype=np.int64)
        real = np.empty(0)
        return Result(integer, real, real.copy(), integer.copy(), integer.copy(),
                      integer.copy(), integer.copy(), integer.copy(), real.copy(),
                      integer.copy(), real.copy(), real.copy())
    columns = [np.concatenate(c) for c in zip(*parts)]
    order = np.lexsort((columns[1], columns[0]))
    return Result(*(c[order] for c in columns))


PATH_LIMIT = 1
BOUNDARY_PENDING = 2
START_MATERIAL = 3
START_UNRESOLVED = 4
PREFIX_STATUS = {PATH_LIMIT: 'path_limit', BOUNDARY_PENDING: 'boundary_pending',
                 START_MATERIAL: 'start_material', START_UNRESOLVED: 'start_unresolved'}


@dataclass(frozen=True)
class VacuumPrefix:
    status: np.ndarray
    path: np.ndarray
    x: np.ndarray
    p3: np.ndarray
    pending_hi: np.ndarray
    left_volume_index: np.ndarray
    next_volume_index: np.ndarray


def vacuum_prefix(curve, geometry, *, geometry_error_mm, atol=1e-6, max_depth=64,
                  max_intervals=1000000):
    """Stop before the first unresolved/non-vacuum arc interval; never jump it.

    pending_hi is the end of the first contiguous unresolved block, not a root
    or proof of a unique boundary. next_volume_index is an adjacent resolved
    geometric owner only, never a certified crossed-to volume. Even equal left
    and next owners can hide tangency or multiple passages inside the gap.
    No material interval is transported and no decay/loss/clock is computed.
    """
    cover = partition(curve, geometry, geometry_error_mm=geometry_error_mm,
                      atol=atol, max_depth=max_depth, max_intervals=max_intervals)
    n = len(curve.x)
    initial = geometry.locate(curve.x)
    status = np.full(n, START_UNRESOLVED, dtype=np.int8)
    start_free = (initial.status == tslocate.VACUUM) & (
        initial.clearance > geometry_error_mm + curve.position_error_mm)
    status[start_free] = PATH_LIMIT
    start_matter = (initial.status == tslocate.MATERIAL) & (
        initial.clearance > geometry_error_mm + curve.position_error_mm)
    status[start_matter] = START_MATERIAL
    path, pending_hi = curve.length.copy(), curve.length.copy()
    left, next_owner = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    index = np.arange(len(cover.status))
    resolved = cover.status == RESOLVED
    vacuum = np.array([v.vacuum for v in geometry.volumes], dtype=bool)
    free = np.zeros(len(index), dtype=bool)
    free[resolved] = vacuum[cover.volume_index[resolved]]
    first = np.full(n, len(index), dtype=np.int64)
    np.minimum.at(first, cover.source_index[~free], index[~free])
    blocked = (first < len(index)) & start_free
    rows = np.flatnonzero(blocked)
    chosen = first[blocked]
    if len(rows):
        status[rows] = BOUNDARY_PENDING
        path[rows] = cover.s_lo[chosen]
        unresolved = ~resolved
        group_start = np.r_[True, (cover.source_index[1:] != cover.source_index[:-1])
                            | ~unresolved[:-1] | ~unresolved[1:]]
        groups = np.cumsum(group_start) - 1
        group_last = np.full(int(groups[-1]) + 1, -1, dtype=np.int64)
        np.maximum.at(group_last, groups, index)
        last = np.where(unresolved[chosen], group_last[groups[chosen]], chosen)
        pending_hi[rows] = cover.s_hi[last]
        prior = chosen - 1
        has_prior = (prior >= 0) & (cover.source_index[np.maximum(prior, 0)] == rows)
        prior_safe = has_prior & resolved[np.maximum(prior, 0)]
        left[rows[prior_safe]] = cover.volume_index[prior[prior_safe]]
        after = last + 1
        has_after = (after < len(index)) & (cover.source_index[np.minimum(after, len(index)-1)] == rows)
        after_safe = has_after & resolved[np.minimum(after, len(index)-1)]
        next_owner[rows[after_safe]] = cover.volume_index[after[after_safe]]
    path[~start_free] = 0
    pending_hi[~start_free] = 0
    left[~blocked & start_free] = initial.volume_index[~blocked & start_free]
    x, p3 = curve.state(np.arange(n), path)
    return VacuumPrefix(status, path, x, p3, pending_hi, left, next_owner)


def main(argv=None):
    from . import tsgeom
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--root', action='append', required=True)
    parser.add_argument('--vacuum-material', action='append', required=True)
    parser.add_argument('--arc', type=float, nargs=8, action='append', required=True,
                        help='x y z px py pz pdg path_mm; uniform-field diagnostic only')
    parser.add_argument('--field', type=float, nargs=3, required=True)
    parser.add_argument('--position-error-mm', type=float, required=True)
    parser.add_argument('--geometry-error-mm', type=float, required=True)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--max-depth', type=int, default=64)
    parser.add_argument('--max-intervals', type=int, default=1000000)
    parser.add_argument('--vacuum-prefix', action='store_true',
                        help='also report the leading resolved vacuum portion; never cross pending intervals')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = Path(args.input).resolve(strict=True)
        if args.out.resolve() == source or (args.out.exists() and args.out.samefile(source)):
            raise ValueError('output must not alias GDML input')
        scene = tsgeom.load(source, args.root, vacuum_materials=args.vacuum_material)
        rows = np.asarray(args.arc)
        curve = UniformHelix(rows[:, :3], rows[:, 3:6], rows[:, 6], args.field, rows[:, 7],
                             position_error_mm=args.position_error_mm)
        result = partition(curve, scene.partition, geometry_error_mm=args.geometry_error_mm,
                           atol=args.atol, max_depth=args.max_depth, max_intervals=args.max_intervals)
        resolved = result.status == RESOLVED
        vacuum = np.array([v.vacuum for v in scene.partition.volumes], dtype=bool)
        is_vacuum = np.zeros(len(result.status), dtype=bool)
        is_vacuum[resolved] = vacuum[result.volume_index[resolved]]
        summary = dict(interval_count=np.bincount(result.source_index, minlength=len(rows)).tolist())
        for name, mask in (('resolved_vacuum_mm', resolved & is_vacuum),
                           ('resolved_material_mm', resolved & ~is_vacuum),
                           ('unresolved_mm', ~resolved)):
            summary[name] = np.bincount(result.source_index, weights=np.where(mask, result.width, 0),
                                        minlength=len(rows)).tolist()
        receipt = dict(schema='ts-selected-uniform-arc-v1', geometry=scene.receipt,
                       input_arcs=rows.tolist(), uniform_field_tesla=args.field,
                       position_error_mm=args.position_error_mm, geometry_error_mm=args.geometry_error_mm,
                       errors_are_caller_assumptions=True, floating_point_certified=False,
                       native_query_certified=False, variable_field_transport=False, material_transport=False,
                       per_arc=summary, status_labels=STATUS,
                       ball_labels={0: 'unresolved', MIDPOINT_BALL: 'arc_midpoint', CHORD_BALL: 'endpoint_chord'},
                       atol_mm=args.atol,
                       **{name: getattr(result, name).tolist() for name in result.__dataclass_fields__})
        if args.vacuum_prefix:
            prefix = vacuum_prefix(curve, scene.partition, geometry_error_mm=args.geometry_error_mm,
                                   atol=args.atol, max_depth=args.max_depth, max_intervals=args.max_intervals)
            receipt['vacuum_prefix'] = dict(
                status_labels=PREFIX_STATUS, neighbor_is_not_crossing_certificate=True,
                **{name: getattr(prefix, name).tolist() for name in prefix.__dataclass_fields__})
        text = json.dumps(receipt, indent=2, allow_nan=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
        print(text)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
