"""Opt-in NumPy map selection for explicitly independent query streams.

Each lane represents a caller-specified independent stream, NOT Geant4's global
field-wrapper cache, whose history spans field queries rather than particles.
The selection rule mirrors BFCacheManager: retain a covering previous inner map;
otherwise search inner maps, then outer maps, in configured order. Previous outer
maps have no priority. Shared closed faces/edges/corners are allowed without any
claim that the fields there agree; positive-volume inner overlap is rejected.

``B, previous = selector.field(points, previous)`` returns fresh next-state
indices. The caller must carry those indices with its lanes through compaction,
reordering and subsequent calls. There are no implicit inactive lanes: query only
the active subset and manage unqueried state explicitly. Initialize/reset with
int64 -1. Bare indices cannot detect state from another selector: previous MUST
belong to this same selector with the same configured map order. No hidden state
or RNG is used. FieldMapBank and transport defaults are unchanged.

Interpolation is delegated to tsfield, including its clamped exact endpoints and
extendY flip=By handling. No stock BFGridMap endpoint/binary equivalence, field
continuity, inclusive-population validity or transport equivalence is claimed.
"""

import hashlib
import itertools

import numpy as np

from .tsfield import CoverageError, FieldMap, _positions, audit_map


SELECTION_CONTRACT = (
    "if prior inner covers point retain it; else configured first inner; "
    "else configured first outer; uncovered fail with CoverageError "
    "(return_coverage=True gives NaNs, selected_map=-1, covered=False)"
)


def _selector_audit_map(item):
    source = audit_map(item)
    source["disk_hash_scope"] = "On-disk observations, not a proof of correspondence to the evaluated array"
    source["evaluated_array"] = dict(shape=list(item.B.shape), dtype="<f8", order="C",
                                     sha256=hashlib.sha256(np.asarray(item.B, dtype="<f8").tobytes(order="C")).hexdigest())
    source["evaluated_grid"] = dict(lo=item.lo.tolist(), hi=item.hi.tolist(), n=item.n.tolist(),
                                    d=item.d.tolist(), extend_y=item.header.extend_y)
    return source


class MapSelector:
    """Ordered maps with explicit, caller-owned per-stream state.

    ``previous`` must be an ndarray of dtype int64 and shape points.shape[:-1].
    -1 resets inner history; 0..ninner-1 identify inner maps and
    ninner..ninner+nouter-1 identify outer maps. All entries, even on uncovered
    lanes, must be valid. State from another selector/bank is not supported and
    cannot be detected from bare indices. An all--1 query is deterministic and
    config-first. No broadcasting of state or implicit inactive-lane mask occurs.
    """

    def __init__(self, inner=(), outer=()):
        self.inner = tuple(inner)
        self.outer = tuple(outer)
        if not all(isinstance(item, FieldMap) for item in self.inner + self.outer):
            raise TypeError("selector entries must be loaded FieldMap objects")
        for left, right in itertools.combinations(self.inner, 2):
            for (alo, ahi), (blo, bhi) in itertools.product(left.header.boxes,
                                                          right.header.boxes):
                # Exact reflected boxes retain Y gaps; strict bounds admit shared faces only.
                if np.all(np.maximum(alo, blo) < np.minimum(ahi, bhi)):
                    raise ValueError("positive-volume overlapping inner coverage: "
                                     f"{left.header.path} and {right.header.path}")

    def field(self, points, previous, *, return_coverage=False):
        """Return (B, selected_map), or (B, selected_map, covered).

        Outputs have shapes points.shape, points.shape[:-1], and (if requested)
        points.shape[:-1], respectively. selected_map is a fresh int64 ndarray,
        suitable as next-call previous, with -1 for uncovered queries. Default
        uncovered behavior is CoverageError; opt-in coverage mode returns NaNs.
        Neither input is mutated. Empty batches and a single shape-(3,) point
        (with scalar shape-() ndarray state) are supported.
        """
        points = _positions(points)
        batch_shape = points.shape[:-1]
        if not isinstance(previous, np.ndarray) or previous.dtype != np.dtype(np.int64):
            raise ValueError("previous must be an int64 ndarray (not floats or bools)")
        if previous.shape != batch_shape:
            raise ValueError("previous must have the same batch shape as points")
        maps = self.inner + self.outer
        if np.any(previous < -1) or np.any(previous >= len(maps)):
            raise ValueError("previous indices must be -1 or within this selector's map range")

        flat = points.reshape(-1, 3)
        prior = previous.reshape(-1)
        selected = np.full(prior.shape, -1, dtype=np.int64)
        for index, item in enumerate(self.inner):
            lanes = prior == index
            selected[lanes] = np.where(item.coverage(flat[lanes]), index, -1)
        for index, item in enumerate(maps):
            lanes = selected < 0
            selected[lanes] = np.where(item.coverage(flat[lanes]), index, -1)

        covered = (selected >= 0).reshape(batch_shape)
        if not return_coverage and not np.all(covered):
            raise CoverageError(covered)
        values = np.full(flat.shape, np.nan, dtype=np.float64)
        for index, item in enumerate(maps):
            lanes = selected == index
            values[lanes] = item.field(flat[lanes])
        result = values.reshape(points.shape), selected.reshape(batch_shape)
        return result + (covered,) if return_coverage else result

    def audit(self):
        """Return a JSON-ready receipt with source hashes and literal semantics.

        Source files must remain available for tsfield.audit_map to hash them.
        This is a source/selection receipt, not runtime or transport validation.
        """
        return {
            "schema": "beamfastsim.tsselect.audit.v1",
            "backend": "numpy-cpu-reference",
            "selection_contract": SELECTION_CONTRACT,
            "scope": "explicitly independent caller-specified query streams; not Geant4 global cache",
            "state_contract": (
                "caller-owned int64 ndarray, same batch shape; -1 reset/no inner, "
                "inner indices then outer indices; same selector/bank and configured order only; "
                "bare indices cannot detect another bank; caller carries state through "
                "compaction/reordering and queries only active subsets"
            ),
            "inner": [_selector_audit_map(item) for item in self.inner],
            "outer": [_selector_audit_map(item) for item in self.outer],
            "transport_validated": False,
            "inclusive_population_validated": False,
            "stock_endpoint_equivalence": False,
            "bank_policy_changed": False,
            "limitations": [
                "Shared closed boundaries are allowed; fields need not be identical there.",
                "Positive-volume inner overlap is rejected using exact mirrored coverage boxes.",
                "Interpolation uses tsfield clamped endpoints, not certified stock BFGridMap "
                "endpoint/binary equivalence.",
                "Independent-stream state does not reproduce Geant4's global query-cache history.",
            ],
        }
