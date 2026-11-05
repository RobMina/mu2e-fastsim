"""Opt-in numerical diagnostic for shared closed inner-map faces, edges and corners.

This compares loaded NumPy references, not bank selection or transport. It does
not change FieldMapBank's rejection of touching inner maps. Coordinates are Mu2e
mm and fields are tesla. Common-refinement node differences bound componentwise
interpolant differences in exact arithmetic on each continuous multilinear piece;
the numerical result is approximate, not bit-exact or a G4 runtime certificate.
A reflected By discontinuity at y=0 is explicitly flagged: node sampling alone
cannot certify the one-sided limit there.

No stock endpoint equivalence is claimed: BFGridMap source upper-endpoint corner
indexing is uncharacterized. No inclusive population or transport validity is
established. Run only on explicitly supplied maps, for example::

    python -m beamfastsim.tsfaces --inner A.header --inner B.header \\
        --atol 1e-12 --max-points 1000000 --out faces.json
"""

import argparse
import hashlib
import itertools
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys

import numpy as np

from .tsfield import FieldMap, audit_map, load


DEFAULT_MAX_POINTS = 1_000_000
_METRICS = ("max_abs_component_tesla", "max_vector_difference_tesla",
            "rms_component_difference_tesla")


def _parameters(atol, max_points):
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, Real):
        raise ValueError("atol must be finite and nonnegative")
    try:
        atol = float(atol)
    except (OverflowError, ValueError) as exc:
        raise ValueError("atol must be finite and nonnegative") from exc
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if (isinstance(max_points, (bool, np.bool_)) or not isinstance(max_points, Integral)
            or max_points < 1):
        raise ValueError("max_points must be a positive integer")
    return float(atol), int(max_points)


def _intersections(left, right):
    """Use exact coverage boxes, never the envelope of a reflected Y gap."""
    found = []
    for (alo, ahi), (blo, bhi) in itertools.product(left.header.boxes, right.header.boxes):
        lo, hi = np.maximum(alo, blo), np.minimum(ahi, bhi)
        if np.all(lo <= hi) and not any(np.array_equal(lo, oldlo) and np.array_equal(hi, oldhi)
                                       for oldlo, oldhi in found):
            found.append((lo, hi))
    # Merge Y intervals only when X/Z agree, retaining mirror gaps and unique zero seams.
    merged = []
    for lo, hi in sorted(found, key=lambda bounds: bounds[0][1]):
        if (merged and np.array_equal(lo[[0, 2]], merged[-1][0][[0, 2]])
                and np.array_equal(hi[[0, 2]], merged[-1][1][[0, 2]])
                and lo[1] <= merged[-1][1][1]):
            merged[-1][1][1] = max(merged[-1][1][1], hi[1])
        else:
            merged.append((lo.copy(), hi.copy()))
    return merged


def _knots(item, axis):
    values = item.lo[axis] + np.arange(int(item.n[axis]), dtype=np.float64) * item.d[axis]
    values[-1] = item.hi[axis]
    if axis == 1 and item.header.extend_y:
        values = np.concatenate((values, -values))
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinate knots must be finite")
    return values


def _grid(left, right, lo, hi):
    axes = []
    for axis in range(3):
        if lo[axis] == hi[axis]:
            values = np.array([lo[axis]])
        else:
            values = np.concatenate(([lo[axis], hi[axis]],
                                     _knots(left, axis), _knots(right, axis)))
            values = np.unique(values[(values >= lo[axis]) & (values <= hi[axis])])
        axes.append(values)
    return axes


def _seam_discontinuous(item, lo, hi):
    """Conservatively flag a negative-Y piece meeting a nonzero mirrored By seam."""
    return bool(item.header.extend_y and item.lo[1] == 0 and lo[1] < 0 <= hi[1]
                and np.any(item.B[:, 0, :, 1] != 0))


def _statistics(difference):
    if not np.all(np.isfinite(difference)):
        raise ValueError("field differences must be finite")
    maximum = float(np.max(np.abs(difference)))
    # Scaling avoids overflow from squaring otherwise finite differences.
    scaled = difference / maximum if maximum else difference
    with np.errstate(over="ignore", invalid="ignore"):
        vector = maximum * float(np.max(np.linalg.norm(scaled, axis=1)))
        rms = maximum * float(np.sqrt(np.mean(scaled * scaled)))
    if not math.isfinite(vector) or not math.isfinite(rms):
        raise ValueError("field difference statistics must be finite")
    return dict(zip(_METRICS, (maximum, vector, rms)))


def _pair(left, right, sources, atol, max_points):
    intersections = _intersections(left, right)
    row = {"names": [str(left.header.path.resolve()), str(right.header.path.resolve())],
           "sources": sources, "kind": "disjoint", "supported": True, "gate": True,
           "samples": 0, "intersection_closed": True, "intersections_mm": [],
           "history_independent_within_atol": None,
           **dict.fromkeys(_METRICS)}
    if not intersections:
        return row
    kinds = []
    for lo, hi in intersections:
        zeros = int(np.count_nonzero(lo == hi))
        kind = ("volume_overlap", "face", "edge", "corner")[zeros]
        kinds.append(kind)
        row["intersections_mm"].append({"lo": lo.tolist(), "hi": hi.tolist(),
                                        "closed": True, "kind": kind})
    # A volume overlap is unsupported, even if other reflected pieces only touch.
    if "volume_overlap" in kinds:
        row.update(kind="volume_overlap", supported=False, gate=False,
                   history_independent_within_atol=False,
                   reason="positive-volume overlap is not sampled by this diagnostic")
        return row
    row["kind"] = kinds[0] if len(set(kinds)) == 1 else "mixed_boundary"
    grids = [_grid(left, right, lo, hi) for lo, hi in intersections]
    counts = [math.prod(axis.size for axis in axes) for axes in grids]
    requested = sum(counts)
    if requested > max_points:
        raise ValueError(f"shared-boundary grid requires {requested} points, exceeds max_points={max_points}; "
                         "no truncation is performed")
    # At most four small coverage-box loops; all sample evaluation is vectorized.
    points = np.concatenate([np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
                             for axes in grids], axis=0)
    # Unique grid axes and disjoint intersections need no per-point deduplication.
    if not np.all(np.isfinite(points)):
        raise ValueError("comparison points must be finite")
    a, b = left.field(points), right.field(points)
    if a.shape != points.shape or b.shape != points.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("sampled fields must be finite arrays with shape (samples, 3)")
    with np.errstate(over="ignore", invalid="ignore"):
        stats = _statistics(a - b)
    seam = any(_seam_discontinuous(item, lo, hi)
               for item in (left, right) for lo, hi in intersections)
    agreement = stats["max_abs_component_tesla"] <= atol
    row.update(stats)
    row.update(samples=int(points.shape[0]),
               sampled_agreement_within_atol=agreement,
               history_independent_within_atol=agreement if not seam else None,
               gate=agreement and not seam, reflection_seam_discontinuous=seam,
               componentwise_interpolant_bound_applicable=not seam)
    if seam:
        row["warning"] = "nonzero reflected By at y=0: shared nodes do not bound the one-sided limit; gate fails closed"
    return row


def audit(maps, atol=0.0, *, max_points=DEFAULT_MAX_POINTS):
    """Return a JSON-serializable shared-boundary receipt (no bank is constructed).

    ``maps`` is a nonempty sequence of loaded FieldMap objects with distinct
    resolved header paths. ``max_points`` limits the total refinement nodes per
    map pair before mesh allocation, not per axis; oversized grids raise
    ValueError, never truncate. ``history_independent_within_atol`` reports
    max_abs_component_tesla <= atol on the sampled closed intersection, not a
    bank-policy override or runtime certificate. Source header/binary files must
    remain available for audit_map hashing; hashes are observations, not release
    approval and do not attest that a manually constructed FieldMap matches disk.
    """
    atol, max_points = _parameters(atol, max_points)
    maps = tuple(maps)
    if not maps:
        raise ValueError("maps must be nonempty")
    if not all(isinstance(item, FieldMap) for item in maps):
        raise TypeError("maps must contain loaded FieldMap objects")
    names = [str(item.header.path.resolve()) for item in maps]
    if len(set(names)) != len(names):
        raise ValueError("map names (resolved header paths) must be distinct")
    for item in maps:
        if not np.all(np.isfinite(item.B)):
            raise ValueError("field arrays must be finite")
    sources = [audit_map(item) for item in maps]
    for source, item in zip(sources, maps):
        source["disk_hash_scope"] = "On-disk observations, not a proof of correspondence to the evaluated array"
        source["evaluated_array"] = dict(shape=list(item.B.shape), dtype="<f8", order="C",
                                         sha256=hashlib.sha256(np.asarray(item.B, dtype="<f8").tobytes(order="C")).hexdigest())
        source["evaluated_grid"] = dict(lo=item.lo.tolist(), hi=item.hi.tolist(), n=item.n.tolist(),
                                        d=item.d.tolist(), extend_y=item.header.extend_y)
    pairs = [_pair(maps[i], maps[j], [sources[i], sources[j]], atol, max_points)
             for i, j in itertools.combinations(range(len(maps)), 2)]
    return {"schema": "beamfastsim.tsfaces.audit.v1", "backend": "numpy-cpu-reference",
            "diagnostic": "approximate common-refinement shared-boundary comparison",
            "atol_tesla": atol, "max_points": max_points, "maps": sources, "pairs": pairs,
            "gate": all(pair["gate"] for pair in pairs),
            "transport_validated": False, "stock_endpoint_equivalence": False,
            "inclusive_population_validated": False, "bank_policy_changed": False,
            "limitations": [
                "Common-node agreement bounds componentwise multilinear differences in exact arithmetic "
                "on continuous pieces; floating-point comparisons are approximate, not bit-exact.",
                "Not a G4 runtime certificate; BFGridMap source upper-endpoint corner indexing is uncharacterized.",
                "No inclusive population or transport validity; touching inner maps remain rejected by FieldMapBank."]}


def main(argv=None):
    """Write a receipt and print one compact summary per pair; failed gates exit 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", action="append", required=True, metavar="HEADER")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Validate options before reading potentially large maps.
        _parameters(args.atol, args.max_points)
        inputs = [Path(p) for p in args.inner] + [Path(p).with_suffix(".bin") for p in args.inner]
        for path in inputs:
            if args.out.resolve() == path.resolve() or (args.out.exists() and args.out.samefile(path)):
                raise ValueError("--out must not overwrite a map header or binary")
        receipt = audit([load(path) for path in args.inner], args.atol, max_points=args.max_points)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for pair in receipt["pairs"]:
            print(json.dumps({key: pair[key] for key in
                              ("names", "kind", "samples", "max_abs_component_tesla", "gate")},
                             allow_nan=False, separators=(",", ":")))
        return 0 if receipt["gate"] else 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"tsfaces: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
