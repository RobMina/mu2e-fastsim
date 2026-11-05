"""Host-only unchanged-trajectory continuation, never material transport restart.

``continue_same_trajectory(receipts, *, expected, absolute_s_max)`` accepts the
ordinary emitter's flat C-order tuple and a mandatory independently owned
ReceiptBinding. It returns a frozen FollowResult with per-lane string status,
immutable four-array absolute ``result``, masks, next/relative isolation arrays,
and flat ``next_evidence`` (NOT GeometricReceipts). Only NEXT_HIT and CLEAR_TO_CAP
are accepted; missing, unsupported and invalid lanes are never clear. NaNs in
isolation arrays mean no supported next enclosure. No continuation chaining is
provided, even on the unchanged trajectory. transport_restart_valid is False.

The original binding is rerun once and each receipt is compared byte-for-byte
with its corresponding ordinary lane. Search uses absolute path coordinates on
the original anchored x/u/p/q/B, from the verified isolation upper endpoint b.
The original cap remains the conditioning/phase/resolution bound. The requested
absolute cap must be exactly representable in the solver dtype and lie in
[b, original cap]. No arbitrary public start/frontier or budget override exists.

Isolation/relative bounds are guarded engineering evidence, not outward-rounded
interval proofs. For old [L,U] and next [L',U'], the relative enclosure is
[L'-U,U'-L], not refined bisection width or representative subtraction. None of
these diagnostics certifies a material restart at an approximate point.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import flash_foil_helix as helix
from . import flash_foil_receipt as receipt

NEXT_HIT = "NEXT_HIT"
CLEAR_TO_CAP = "CLEAR_TO_CAP"
UNRESOLVED = "UNRESOLVED"
INVALID_RECEIPT = "INVALID_RECEIPT"
INVALID_CAP = "INVALID_CAP"
UNSUPPORTED_HIT = "UNSUPPORTED_HIT"


@dataclass(frozen=True, eq=False)
class FollowResult:
    """Immutable host diagnostics; next_evidence cannot authorize another query."""
    status: np.ndarray
    result: object
    accepted: np.ndarray
    next_hit: np.ndarray
    clear_to_cap: np.ndarray
    unresolved: np.ndarray
    next_isolation: np.ndarray
    relative_isolation: np.ndarray
    next_evidence: tuple
    transport_restart_valid: bool = False

    @property
    def distance(self):
        return self.result["distance"]

    @property
    def kind(self):
        return self.result["kind"]

    @property
    def valid(self):
        return self.result["valid"]


def _capture(dtype, rin, captures):
    """Separate window observer; the ordinary emitter intentionally stays unchanged."""
    def observer(**s):
        shape = s["hit"].shape
        for lane in np.ndindex(shape):
            if not s["hit"][lane] or s["continuing"][lane] or s["force_zero"][lane]:
                continue
            j = int(s["idx"][lane])
            ga, gb = s["ga"][lane], s["gb"][lane]
            if (abs(ga[j]) <= s["guard"][lane] or
                    abs(gb[j]) <= s["guard"][lane] or
                    (ga[j] > 0) == (gb[j] > 0)):
                continue
            enabled = np.array([True, True, np.broadcast_to(
                np.asarray(rin, dtype=dtype), shape)[lane] > 0, True])
            other = enabled.copy()
            other[j] = False
            dm = s["dm"][lane][j]
            variation = s["variation"][lane][j]
            margin = abs(dm)-variation-128*s["eps"]*(s["n"][lane]+s["A"][lane]*s["h"][lane])
            if (margin <= 0 or not np.all(s["lower"][lane][other] > 0) or
                    not np.all(np.isfinite(ga[enabled])) or
                    not np.all(np.isfinite(gb[enabled]))):
                continue
            evidence = dict(
                isolation=np.asarray([s["a"][lane], s["b"][lane]], dtype=dtype),
                refined=np.asarray([s["lo"][lane], s["hi"][lane]], dtype=dtype),
                distance=s["root"][lane], kind=np.asarray(j+1, np.int32),
                guard=s["guard"][lane], resolution=s["resolution"][lane],
                derivative_mid=dm, derivative_variation=variation,
                derivative_margin=np.asarray(margin, dtype=dtype),
                endpoint_clearance=np.asarray([ga[j], gb[j]], dtype=dtype),
                other_clearance_lower=np.where(other, s["lower"][lane], 0),
                other_clearance_enabled=other,
                forward_material=np.asarray(gb[j] > 0))
            if all(np.all(np.isfinite(a)) for a in evidence.values()):
                captures[lane] = MappingProxyType({k: receipt._frozen(a)
                                                 for k, a in evidence.items()})
    return observer


def _caps(value, shape, dtype):
    """Validate supplied element types and exact values before numeric coercion."""
    cast = np.zeros(shape, dtype=dtype)
    good = np.zeros(shape, bool)
    try:
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "iuf":
                return cast, good
            a = np.broadcast_to(value, shape)
        else:
            a = np.broadcast_to(np.asarray(value, dtype=object), shape)
    except (ValueError, TypeError, OverflowError):
        return cast, good
    for lane in np.ndindex(shape):
        v = a[lane]
        if isinstance(v, (bool, np.bool_)):
            continue
        try:
            if isinstance(v, (int, np.integer)):
                ratio = (int(v), 1)
            elif isinstance(v, (float, np.floating)) and np.isfinite(v):
                ratio = v.as_integer_ratio()
            else:
                continue
            if v < 0:
                continue
            with np.errstate(all="ignore"):
                c = np.asarray(v, dtype=dtype)[()]
            if np.isfinite(c) and ratio == c.as_integer_ratio():
                cast[lane], good[lane] = c, True
        except (ValueError, TypeError, OverflowError):
            continue
    return cast, good


def continue_same_trajectory(receipts, *, expected, absolute_s_max):
    """Verify initial receipts once, then query the same absolute anchored orbit.

    Invalid expected bindings raise ValueError. Malformed receipt collections or
    caps refuse per lane. A valid geometric hit lacking strict next isolation is
    UNSUPPORTED_HIT, never clear. ``result`` preserves such hit diagnostics, but
    accepted/next_hit are false. On refused input lanes result is explicitly
    invalid and unresolved; there is no speculative window outcome to consume.
    """
    binding = receipt._validated_binding(expected)
    ordinary, fresh = receipt._emit(binding)
    shape = ordinary["valid"].shape
    v = binding.inputs
    dtype = np.result_type(v["x"].dtype, v["u"].dtype, np.float32)
    status = np.full(shape, INVALID_RECEIPT, dtype="<U18")
    start = np.zeros(shape, dtype=dtype)
    previous = np.full(shape+(2,), np.nan, dtype=dtype)
    verified = np.zeros(shape, bool)
    if type(receipts) is tuple and len(receipts) == len(fresh):
        for lane, supplied, correct in zip(np.ndindex(shape), receipts, fresh):
            try:
                good = (correct is not None and type(supplied) is receipt.GeometricReceipt
                        and receipt._same(supplied, correct))
            except (TypeError, ValueError, IndexError, AttributeError, RecursionError):
                good = False
            if good:
                verified[lane] = True
                previous[lane] = correct.evidence["isolation"]
                start[lane] = previous[lane][1]
    cap, cap_good = _caps(absolute_s_max, shape, dtype)
    original_cap = np.broadcast_to(np.asarray(v["s_max"], dtype=dtype), shape)
    cap_good &= (cap >= start) & (cap <= original_cap)
    status[verified & ~cap_good] = INVALID_CAP
    enabled = verified & cap_good
    status[enabled] = UNRESOLVED
    distance = np.full(shape, np.inf, dtype=dtype)
    kind = np.full(shape, helix.NONE, np.int32)
    valid = np.zeros(shape, bool)
    unresolved = np.ones(shape, bool)
    zero_window = enabled & (cap == start)
    for lane, correct in zip(np.ndindex(shape), fresh):
        if zero_window[lane]:
            e = correct.evidence
            if abs(e["endpoint_clearance"][1]) > e["resolution"]:
                status[lane] = CLEAR_TO_CAP
                unresolved[lane] = False
    captures = {}
    search = enabled & (cap > start)
    if np.any(search):
        with np.errstate(all="ignore"):
            result = helix._transition(
                np, v["x"], v["u"], v["p"], v["q"], v["B"], v["rin"],
                v["rout"], v["zlo"], v["zhi"], v["axis_x"], v["s_max"],
                v["exiting"], v["tolerance"], binding.max_intervals, v["after_surface"],
                _observer=_capture(dtype, v["rin"], captures),
                _search_window=(start, np.where(search, cap, start), search))
        for lane in np.ndindex(shape):
            if not search[lane]:
                continue
            distance[lane], kind[lane] = result["distance"][lane], result["kind"][lane]
            valid[lane], unresolved[lane] = result["valid"][lane], result["unresolved"][lane]
            if valid[lane]:
                status[lane] = NEXT_HIT if lane in captures else UNSUPPORTED_HIT
            elif not unresolved[lane]:
                status[lane] = CLEAR_TO_CAP
    next_isolation = np.full(shape+(2,), np.nan, dtype=dtype)
    relative = np.full_like(next_isolation, np.nan)
    for lane, e in captures.items():
        if status[lane] == NEXT_HIT:
            next_isolation[lane] = e["isolation"]
            relative[lane] = [e["isolation"][0]-previous[lane][1],
                              e["isolation"][1]-previous[lane][0]]
    hit, clear = status == NEXT_HIT, status == CLEAR_TO_CAP
    freeze = receipt._frozen
    return FollowResult(
        status=freeze(status),
        result=MappingProxyType({k: freeze(a) for k, a in dict(
            distance=distance, kind=kind, valid=valid, unresolved=unresolved).items()}),
        accepted=freeze(hit | clear), next_hit=freeze(hit), clear_to_cap=freeze(clear),
        unresolved=freeze(unresolved), next_isolation=freeze(next_isolation),
        relative_isolation=freeze(relative),
        next_evidence=tuple(captures.get(lane) if hit[lane] else None
                            for lane in np.ndindex(shape)))
