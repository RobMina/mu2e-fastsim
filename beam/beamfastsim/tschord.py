"""Bounded straight-chord material partitions, NOT curved-path/native navigation.

Each resolved interval fits strictly in a midpoint ownership-clearance ball.
Adaptive dyadic bisection leaves surface/overlap/out-of-scope intervals unresolved;
no epsilon step or endpoint-only assumption skips an intervening thin obstacle.
Floating-point geometry and RK error remain uncertified. Never apply this to a
curved trajectory without a separately established enclosing-path error bound.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import tslocate, tsnav

RESOLVED = 1
BOUNDARY_PENDING = 2
DEPTH_LIMIT = 3


@dataclass(frozen=True)
class Result:
    source_index: np.ndarray
    fraction_lo: np.ndarray
    fraction_hi: np.ndarray
    depth: np.ndarray
    status: np.ndarray
    midpoint_status: np.ndarray
    volume_index: np.ndarray
    boundary_index: np.ndarray
    width: np.ndarray


def partition(start, end, geometry, *, assume_straight=False, atol=1e-6,
              max_depth=64, max_intervals=1000000):
    """Partition each input segment into a complete ordered interval cover.

    atol bounds unresolved chord length, not a geometry enlargement. At a depth
    cap intervals are DEPTH_LIMIT, never resolved; resource overflow raises before
    producing a partial result. Source identity is input row index, not TrackID.
    """
    if assume_straight is not True:
        raise ValueError("explicit assume_straight=True contract required")
    a, b = tsnav._real(start, "start"), tsnav._real(end, "end")
    if a.ndim != 2 or a.shape[1] != 3 or b.shape != a.shape:
        raise ValueError("start/end require matching (N,3) shapes")
    if not isinstance(geometry, tslocate.Partition):
        raise ValueError("explicit selected-cell Partition required")
    atol = tsnav._scalar(atol, "atol", positive=True)
    for value, name, minimum in ((max_depth, "max_depth", 0),
                                  (max_intervals, "max_intervals", 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} requires integer >= {minimum}")
    if max_depth > 64:
        raise ValueError("max_depth exceeds bounded dyadic precision")
    if len(a) > max_intervals:
        raise ValueError("max_intervals exceeded")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = b - a
        length = np.hypot.reduce(delta, axis=1)
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(length)):
        raise ValueError("nonfinite chord displacement/length")
    if not np.array_equal(a + delta, b) or not np.array_equal(b - delta, a):
        raise ValueError("chord displacement does not preserve both supplied endpoints")
    source, lo, hi = np.arange(len(a)), np.zeros(len(a)), np.ones(len(a))
    depth = np.zeros(len(a), dtype=np.int64)
    parts, count = [], 0
    while len(source):
        middle = lo + (hi - lo) / 2
        points = a[source] + middle[:, None] * delta[source]
        location = geometry.locate(points)
        width = length[source] * (hi - lo)
        left = a[source] + lo[:, None] * delta[source]
        right = a[source] + hi[:, None] * delta[source]
        moving = length[source] > 0
        if np.any(moving & ((width == 0) | np.all(points == left, axis=1)
                            | np.all(points == right, axis=1))):
            raise ValueError("chord subdivision is not spatially representable")
        radius = np.maximum(width / 2, np.maximum(np.hypot.reduce(points - left, axis=1),
                                                  np.hypot.reduce(points - right, axis=1)))
        local = (location.status == tslocate.VACUUM) | (location.status == tslocate.MATERIAL)
        safe = local & (location.clearance > radius)
        small = width <= atol
        capped = depth >= max_depth
        done = safe | small | capped
        if np.any(done):
            status = np.where(safe, RESOLVED, np.where(small, BOUNDARY_PENDING, DEPTH_LIMIT))
            owner = np.where(safe, location.volume_index, -1)
            parts.append((source[done], lo[done], hi[done], depth[done], status[done],
                          location.status[done], owner[done], location.boundary_index[done], width[done]))
            count += int(np.count_nonzero(done))
        todo = ~done
        if not np.any(todo):
            break
        if count + 2 * np.count_nonzero(todo) > max_intervals:
            raise ValueError("max_intervals exceeded; no partial chord partition returned")
        if np.any((middle[todo] <= lo[todo]) | (middle[todo] >= hi[todo])):
            raise ValueError("dyadic subdivision is not representable")
        source, depth = np.repeat(source[todo], 2), np.repeat(depth[todo] + 1, 2)
        lo, hi = (np.column_stack((lo[todo], middle[todo])).ravel(),
                  np.column_stack((middle[todo], hi[todo])).ravel())
    if not parts:
        empty = np.empty(0, dtype=np.int64)
        return Result(empty, np.empty(0), np.empty(0), empty.copy(), empty.copy(),
                      empty.copy(), empty.copy(), empty.copy(), np.empty(0))
    columns = [np.concatenate(c) for c in zip(*parts)]
    order = np.lexsort((columns[1], columns[0]))
    return Result(*(c[order] for c in columns))


def main(argv=None):
    from . import tsgeom
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--root', action='append', required=True)
    parser.add_argument('--vacuum-material', action='append', required=True)
    parser.add_argument('--chord', type=float, nargs=6, action='append', required=True,
                        help='world-frame start x y z and end x y z, in mm')
    parser.add_argument('--assume-straight', action='store_true', required=True)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--max-depth', type=int, default=64)
    parser.add_argument('--max-intervals', type=int, default=1000000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = Path(args.input).resolve(strict=True)
        if args.out.resolve() == source or (args.out.exists() and args.out.samefile(source)):
            raise ValueError('output must not alias GDML input')
        scene = tsgeom.load(source, args.root, vacuum_materials=args.vacuum_material)
        endpoints = np.asarray(args.chord)
        result = partition(endpoints[:, :3], endpoints[:, 3:], scene.partition,
                           assume_straight=args.assume_straight, atol=args.atol,
                           max_depth=args.max_depth, max_intervals=args.max_intervals)
        resolved = result.status == RESOLVED
        vacuum = np.array([v.vacuum for v in scene.partition.volumes], dtype=bool)
        is_vacuum = np.zeros(len(result.status), dtype=bool)
        is_vacuum[resolved] = vacuum[result.volume_index[resolved]]
        summaries = dict(
            interval_count=np.bincount(result.source_index, minlength=len(endpoints)).tolist(),
            resolved_vacuum_mm=np.bincount(result.source_index, weights=np.where(resolved & is_vacuum, result.width, 0),
                                           minlength=len(endpoints)).tolist(),
            resolved_material_mm=np.bincount(result.source_index, weights=np.where(resolved & ~is_vacuum, result.width, 0),
                                             minlength=len(endpoints)).tolist(),
            unresolved_mm=np.bincount(result.source_index, weights=np.where(~resolved, result.width, 0),
                                      minlength=len(endpoints)).tolist())
        receipt = dict(schema='ts-selected-straight-chord-v1', geometry=scene.receipt,
                       per_chord=summaries,
                       endpoints_world_mm=endpoints.tolist(), atol_mm=args.atol,
                       native_query_certified=False, curved_path_certified=False,
                       material_transport=False,
                       source_index=result.source_index.tolist(),
                       fraction_lo=result.fraction_lo.tolist(), fraction_hi=result.fraction_hi.tolist(),
                       depth=result.depth.tolist(), status=result.status.tolist(),
                       status_labels={RESOLVED: 'resolved_interval', BOUNDARY_PENDING: 'boundary_pending',
                                      DEPTH_LIMIT: 'depth_limit'},
                       midpoint_status=result.midpoint_status.tolist(),
                       midpoint_status_labels=tslocate.STATUS,
                       volume_index=result.volume_index.tolist(), boundary_index=result.boundary_index.tolist(),
                       width_mm=result.width.tolist())
        text = json.dumps(receipt, indent=2, allow_nan=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
        print(text)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
