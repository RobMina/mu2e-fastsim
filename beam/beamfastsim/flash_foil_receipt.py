"""NumPy-only solver-owned geometric evidence; NOT a transport restart protocol.

``bind_inputs`` retains exact supplied numeric values/dtypes/shapes (not just a
hash). ``first_transition_with_receipts`` returns the ordinary result plus a
flat C-order tuple of receipts/None, one per broadcast lane. Unsupported lanes
keep their ordinary solver result; receipt refusal is not a solver no-hit.

An isolation interval [a,b] records the chronological search's guarded unique
monotone crossing evidence. It is conservative engineering evidence, NOT an
outward-rounded floating-point enclosure proof. The refined bisection interval
and representative distance are approximate numerical diagnostics: hi-lo is NOT
an FP-certified error bound. Neither endpoint is made exact, snapped or nudged.
Every receipt has transport_restart_valid=False. No continuation consumer exists.

Verification reruns the chronological solver against an EXTERNALLY supplied
expected binding and compares every bound input and evidence byte. It does not
trust a self-supplied digest or certify the caller's geometry/frame provenance.
Optional frame_binding is an explicit finite numeric (4,4) matrix, retained
opaquely, never guessed or applied. None explicitly means no frame binding.
This is same-runtime reproducibility checking, not a portable libm certificate.
"""

from dataclasses import dataclass, fields
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from . import flash_foil_helix as helix

MAX_RECEIPT_INTERVALS = 4096
_SCHEMA = "flash-foil-geometric-v1-guarded"
_INPUTS = ("x", "u", "p", "q", "B", "rin", "rout", "zlo", "zhi",
           "axis_x", "s_max", "exiting", "tolerance", "after_surface")


def _frozen(value):
    """Copy onto immutable bytes; unlike setflags(False), cannot be re-enabled."""
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class ReceiptBinding:
    """Use bind_inputs; numeric foil_id is identity, NOT a thickness/material ID.

    A repeated foil_id in a batch must have identical represented geometry.
    Global uniqueness/provenance belongs to the external owner of the ID table.
    """
    inputs: Mapping
    foil_id: np.ndarray
    frame_binding: object
    max_intervals: int


@dataclass(frozen=True, eq=False)
class GeometricReceipt:
    binding: ReceiptBinding
    lane_index: tuple
    evidence: Mapping
    schema: str = _SCHEMA
    transport_restart_valid: bool = False


def bind_inputs(x, u, p, q, B, rin, rout, zlo, zhi, *, foil_id,
                frame_binding=None, axis_x=0., s_max=1.e6, exiting=False,
                tolerance=None, max_intervals=512, after_surface=helix.NONE):
    """Validate and copy a host binding; receipts support float32/float64 only.

    max_intervals is a static integer in [1,4096] for bounded host verification.
    foil_id is nonnegative int64-compatible and broadcasts to the solver lanes.
    after_surface is validated but non-NONE lanes never emit receipts.
    """
    helix.validate_inputs(x, u, p, q, B, rin, rout, zlo, zhi, axis_x=axis_x,
                          s_max=s_max, exiting=exiting, tolerance=tolerance,
                          max_intervals=max_intervals, after_surface=after_surface)
    if max_intervals > MAX_RECEIPT_INTERVALS:
        raise ValueError("receipt max_intervals must be <= 4096")
    values = dict(x=x, u=u, p=p, q=q, B=B, rin=rin, rout=rout, zlo=zlo,
                  zhi=zhi, axis_x=axis_x, s_max=s_max, exiting=exiting,
                  tolerance=tolerance, after_surface=after_surface)
    # Retain pre-cast values so cast-invisible changes cannot silently rebind evidence.
    if any(np.asarray(v).dtype.kind not in "biuf" or
           not np.all(np.isfinite(np.asarray(v)))
           for v in values.values() if v is not None):
        raise ValueError("receipt inputs must be finite real numeric values")
    shape = np.broadcast_shapes(np.shape(x)[:-1], np.shape(u)[:-1], np.shape(B)[:-1],
                                *[np.shape(values[k]) for k in _INPUTS
                                  if k not in ("x", "u", "B") and values[k] is not None])
    ids = np.asarray(foil_id)
    if (ids.dtype.kind not in "iu" or np.any(ids < 0) or
            np.any(ids > np.iinfo(np.int64).max)):
        raise ValueError("foil_id must contain nonnegative int64-compatible integers")
    try:
        ids_b = np.broadcast_to(ids, shape)
    except ValueError as exc:
        raise ValueError("foil_id must broadcast to existing solver lanes") from exc
    # Repeated foil IDs may repeat trajectories, but must retain identical geometry.
    _, inverse = np.unique(ids_b.ravel(), return_inverse=True)
    order = np.argsort(inverse)
    repeated = np.diff(inverse[order]) == 0
    for name in ("rin", "rout", "zlo", "zhi", "axis_x"):
        column = np.broadcast_to(np.asarray(values[name]), shape).ravel()[order]
        if np.any(repeated & (column[1:] != column[:-1])):
            raise ValueError("same foil_id has different geometry")
    if frame_binding is not None:
        frame = np.asarray(frame_binding)
        if (frame.shape != (4, 4) or frame.dtype.kind not in "iuf" or
                not np.all(np.isfinite(frame))):
            raise ValueError("frame_binding must be a finite real numeric (4,4) matrix")
        frame_binding = _frozen(frame)
    return ReceiptBinding(MappingProxyType({k: None if v is None else _frozen(v)
                                           for k, v in values.items()}),
                          _frozen(ids), frame_binding, int(max_intervals))


def _validated_binding(binding):
    if (type(binding) is not ReceiptBinding or not isinstance(binding.inputs, Mapping)
            or set(binding.inputs) != set(_INPUTS)):
        raise ValueError("expected a complete ReceiptBinding from bind_inputs")
    return bind_inputs(**binding.inputs, foil_id=binding.foil_id,
                       frame_binding=binding.frame_binding,
                       max_intervals=binding.max_intervals)


def _emit(binding):
    v = binding.inputs
    captures = {}
    # Geometry is used only by the observer, never by existing decisions.
    dtype = np.result_type(v["x"].dtype, v["u"].dtype, np.float32)

    def observer(**s):
        for lane in zip(*np.nonzero(np.atleast_1d(s["hit"]))):
            lane = () if s["hit"].ndim == 0 else lane
            if not s["hit"][lane] or s["continuing"][lane] or s["force_zero"][lane]:
                continue
            j = int(s["idx"][lane])
            ga, gb = s["ga"][lane], s["gb"][lane]
            # Exact isolation endpoints lack this receipt's strict-sign evidence.
            if (ga[j] == 0 or gb[j] == 0 or
                    abs(ga[j]) <= s["guard"][lane] or
                    abs(gb[j]) <= s["guard"][lane] or
                    (ga[j] > 0) == (gb[j] > 0)):
                continue
            enabled = np.asarray([True, True, bool(np.broadcast_to(
                np.asarray(v["rin"], dtype=dtype), s["hit"].shape)[lane] > 0), True])
            if not np.all(np.isfinite(ga[enabled])) or not np.all(np.isfinite(gb[enabled])):
                continue
            other = enabled.copy()
            other[j] = False
            margin = (abs(s["dm"][lane][j])-s["variation"][lane][j]
                      -128*s["eps"]*(s["n"][lane]+s["A"][lane]*s["h"][lane]))
            evidence = dict(
                isolation=np.asarray([s["a"][lane], s["b"][lane]], dtype=dtype),
                refined=np.asarray([s["lo"][lane], s["hi"][lane]], dtype=dtype),
                distance=s["root"][lane], kind=np.asarray(j+1, np.int32),
                guard=s["guard"][lane], resolution=s["resolution"][lane],
                derivative_margin=np.asarray(margin, dtype=dtype),
                derivative_mid=s["dm"][lane][j],
                derivative_variation=s["variation"][lane][j],
                endpoint_clearance=np.asarray([ga[j], gb[j]], dtype=dtype),
                other_clearance_lower=np.where(other, s["lower"][lane], 0),
                other_clearance_enabled=other,
                forward_material=np.asarray(not bool(np.broadcast_to(v["exiting"],
                                                    s["hit"].shape)[lane])))
            if (margin <= 0 or not np.all(s["lower"][lane][other] > 0) or
                    not all(np.all(np.isfinite(a)) for a in evidence.values())):
                continue
            captures[lane] = MappingProxyType({k: _frozen(a) for k, a in evidence.items()})

    with np.errstate(all="ignore"):
        result = helix._transition(np, v["x"], v["u"], v["p"], v["q"], v["B"],
                                   v["rin"], v["rout"], v["zlo"], v["zhi"],
                                   v["axis_x"], v["s_max"], v["exiting"], v["tolerance"],
                                   binding.max_intervals, v["after_surface"], _observer=observer)
    receipts = tuple(GeometricReceipt(binding, lane, captures[lane])
                     if lane in captures and result["valid"][lane] else None
                     for lane in np.ndindex(result["valid"].shape))
    return result, receipts


def first_transition_with_receipts(x, u, p, q, B, rin, rout, zlo, zhi, **kwargs):
    """Opt-in host query. Required foil_id; options exactly as bind_inputs.

    Returns (legacy_result, flat_C_order_receipts). None means emission refused,
    including zero force, no-hit, unresolved, exact endpoint and after_surface.
    No extra key or dtype is added to legacy_result, even for supported hits.
    """
    return _emit(bind_inputs(x, u, p, q, B, rin, rout, zlo, zhi, **kwargs))


def _same(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if type(a) is not type(b):
        return False
    if type(a) in (ReceiptBinding, GeometricReceipt):
        return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))
    if type(a) is tuple:
        return len(a) == len(b) and all(_same(aa, bb) for aa, bb in zip(a, b))
    if type(a) in (str, int, bool, type(None)):
        return a == b
    return False


def verify_receipt(receipt, *, expected):
    """Return bool after a full host rerun against independently owned inputs.

    An invalid expected binding raises ValueError. Malformed/altered receipts
    fail closed. Success authorizes geometric evidence only, NEVER restart.
    The caller must obtain expected independently, not from receipt.binding.
    """
    binding = _validated_binding(expected)
    try:
        if type(receipt) is not GeometricReceipt or not _same(receipt.binding, binding):
            return False
        _, fresh = _emit(binding)
        return any(r is not None and _same(receipt, r) for r in fresh)
    except (TypeError, ValueError, IndexError, AttributeError):
        return False
