"""Conditional magnet-only, single-map uncertainty carry; not native transport.

For initial position/direction errors ex,eu, fixed scalar momentum p and frozen
field error eB, set a=K/p, G=map spatial Lipschitz bound. Skew Lorentz rotation
does not amplify direction differences, giving
Eu(s) <= min(2, eu+a*((G*ex+eB)*s+G*s**2/2)),
Ex(s) <= min(ex+2*s, ex+eu*s+a*((G*ex+eB)*s**2/2+G*s**3/6)).
Endpoint reference-evaluation errors are ADDED before the next segment. Nominal
endpoint stitching need not be continuous; the true ideal path is continuous.
Every bound remains conditional on caller error assumptions and exact arithmetic;
no outward-rounded floating, RK4/native, decay or material certificate is issued.
"""

from dataclasses import dataclass
import hashlib

import numpy as np

from . import tsarc, tsbound, tsfield, tslocate, tsnav, tsstep, tsvary

VACUUM_ADVANCED = 1
MATERIAL_PENDING = 2
UNRESOLVED = 3
STATUS = {VACUUM_ADVANCED: 'conditional_vacuum_reference_advanced',
          MATERIAL_PENDING: 'material_pending_no_advance', UNRESOLVED: 'unresolved_no_advance'}


def _fixed(value):
    return tsvary._fixed_typed(value)


@dataclass(frozen=True, eq=False, init=False)
class MapContext:
    """One immutable NumPy map snapshot and its all-node/all-edge bounds.

    Every state and continuation remains bound to THIS context. No selector,
    trial/commit protocol or handoff to another map is implemented. Fingerprint
    identifies the evaluated grid/metadata, not a native runtime release.
    """

    item: tsfield.FieldMap
    magnitude: tsbound.Bound
    spatial: tsbound.SpatialBound
    fingerprint: str

    def __init__(self, item, *, chunk_nodes=65536):
        if type(item) is not tsfield.FieldMap:
            raise ValueError('explicit single FieldMap required')
        old = item.header
        flags = []
        for value in (old.extend_y, old.explicit_origin):
            flag = np.asarray(value)
            if flag.shape != () or flag.dtype.kind != 'b':
                raise ValueError('map reflection/origin flags require scalar bool')
            flags.append(bool(flag))
        header = tsfield.GridHeader(old.path, old.grid_origin, old.origin_shift, old.n, old.d, *flags)
        for key in ('grid_origin', 'origin_shift', 'n', 'd'):
            object.__setattr__(header, key, _fixed(getattr(header, key)))
        frozen = tsfield.FieldMap(header, item.B, item.byte_order)
        object.__setattr__(frozen, 'B', _fixed(frozen.B))
        digest = hashlib.sha256()
        for array in (header.grid_origin, header.origin_shift, header.n, header.d, frozen.B):
            digest.update(array.tobytes())
        digest.update(bytes((bool(header.extend_y),)))
        for key, value in dict(item=frozen, magnitude=tsbound.magnitude(frozen, chunk_nodes=chunk_nodes),
                               spatial=tsbound.spatial(frozen, chunk_nodes=chunk_nodes),
                               fingerprint=digest.hexdigest()).items():
            object.__setattr__(self, key, value)


@dataclass(frozen=True, eq=False, init=False)
class State:
    """Nominal/reference receipt with bounds on a continuous ideal Lorentz state.

    momentum is an independently supplied positive scalar per row, treated as
    EXACT for this ideal model and preserved bit-for-bit. It is never recomputed
    from a propagated vector. The ideal nominal unit direction is the mathematical
    normalization of the stored nonzero direction vector, not its raw length.
    position_error_mm/direction_error bound the true initial state relative to
    these nominal values. Source indices are row tags, NOT native particle IDs.
    Numerical evaluation/normalization and cumulative-path arithmetic are not
    certified. No clocks, momentum loss or material response are represented.
    """

    context: MapContext
    x: np.ndarray
    direction: np.ndarray
    momentum: np.ndarray
    pdg: np.ndarray
    position_error_mm: np.ndarray
    direction_error: np.ndarray
    source_index: np.ndarray
    path_mm: np.ndarray

    def __init__(self, context, x, direction, momentum, pdg, *, position_error_mm,
                 direction_error, source_index=None, path_mm=0.):
        if type(context) is not MapContext:
            raise ValueError('immutable MapContext required')
        x, direction = tsnav._real(x, 'x'), tsnav._real(direction, 'direction')
        if x.ndim != 2 or x.shape[1] != 3 or direction.shape != x.shape:
            raise ValueError('x and direction require matching (N,3) shapes')
        n = len(x)
        norm = np.hypot.reduce(direction, axis=1)
        if np.any(~np.isfinite(norm) | (norm <= 0)):
            raise ValueError('nonzero finite direction norm required')
        momentum = tsarc._lanes(momentum, (n,), 'momentum')
        if np.any(momentum <= 0):
            raise ValueError('strictly positive scalar momentum required')
        pdg = tsarc._lanes(pdg, (n,), 'pdg')
        if np.any(~np.isin(pdg, (13, -13, 211, -211))):
            raise ValueError('mu+/mu-/pi+/pi- required')
        ex = tsarc._lanes(position_error_mm, (n,), 'position_error_mm', nonnegative=True)
        eu = tsarc._lanes(direction_error, (n,), 'direction_error', nonnegative=True)
        if np.any(eu > 2):
            raise ValueError('direction_error must be in [0,2]')
        path = tsarc._lanes(path_mm, (n,), 'path_mm', nonnegative=True)
        source = np.arange(n, dtype=np.int64) if source_index is None else np.asarray(source_index)
        if source.shape != (n,) or source.dtype.kind not in 'iu' or np.any(source < 0):
            raise ValueError('nonnegative integer source_index per row required')
        if len(np.unique(source)) != n:
            raise ValueError('unique source_index row tags required')
        for key, value in dict(context=context, x=_fixed(x), direction=_fixed(direction),
                               momentum=momentum, pdg=_fixed(pdg.astype(np.int64)),
                               position_error_mm=ex, direction_error=eu,
                               source_index=_fixed(source), path_mm=path).items():
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class Errors:
    position_mm: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True, eq=False, init=False)
class Segment:
    """One frozen-field reference segment for a bound ideal varying-map path.

    field_error_tesla bounds B0 versus ideal map B at nominal initial x.
    position_error_mm bounds EVERY reference-position evaluation. direction_error
    bounds the mathematical normalization of EVERY stored evaluated direction
    versus the ideal reference direction, including normalization arithmetic.
    Required zero declarations are numerical idealizations, not measured errors.
    Initial carry bounds belong to State and are separate from evaluation errors.
    """

    initial: State
    length: np.ndarray
    field: np.ndarray
    curvature: np.ndarray
    field_error_tesla: np.ndarray
    position_error_mm: np.ndarray
    direction_error: np.ndarray
    floating_point_certified: bool = False
    native_query_certified: bool = False

    def __init__(self, initial, length, *, field_error_tesla, position_error_mm, direction_error):
        if type(initial) is not State:
            raise ValueError('explicit carried State required')
        n = len(initial.x)
        length = tsarc._lanes(length, (n,), 'length', nonnegative=True)
        eb = tsarc._lanes(field_error_tesla, (n,), 'field_error_tesla', nonnegative=True)
        ex = tsarc._lanes(position_error_mm, (n,), 'evaluation position_error_mm', nonnegative=True)
        eu = tsarc._lanes(direction_error, (n,), 'evaluation direction_error', nonnegative=True)
        if np.any(eu > 2):
            raise ValueError('evaluation direction_error must be in [0,2]')
        with np.errstate(over='ignore', invalid='ignore'):
            radius = initial.position_error_mm + length
            lo, hi = initial.x - radius[:, None], initial.x + radius[:, None]
        tsbound.require_boxes(initial.context.item, lo, hi, continuous=True)
        if np.any((radius[:, None] > 0) & ((lo == initial.x) | (hi == initial.x))):
            raise ValueError('uncertain reachable box displacement is not representable')
        field = initial.context.item.field(initial.x)
        with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
            bm = np.linalg.norm(field, axis=1)
            k = tsstep.K * bm / initial.momentum
            curvature = tsstep.K * initial.context.magnitude.magnitude_tesla / initial.momentum
            phase = k * length
        if not all(np.all(np.isfinite(a)) for a in (bm, k, curvature, phase)) or np.any(phase > tsarc.MAX_PHASE):
            raise ValueError('reference curvature/phase outside supported range')
        if np.any((np.any(field != 0, axis=1)) & (bm == 0)) or np.any((bm > 0) & (k == 0)) or np.any(
                (initial.context.magnitude.magnitude_tesla > 0) & (curvature == 0)):
            raise ValueError('nonzero field/curvature underflows')
        for key, value in dict(initial=initial, length=length, field=_fixed(field), curvature=_fixed(curvature),
                               field_error_tesla=eb, position_error_mm=ex, direction_error=eu,
                               floating_point_certified=False, native_query_certified=False).items():
            object.__setattr__(self, key, value)
        self.errors(np.arange(n), length)

    def _queries(self, source, arclength):
        raw = np.asarray(source)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw >= len(self.length)):
            raise ValueError('in-range integer row indices required')
        s = tsnav._real(arclength, 'arclength')
        if s.shape != raw.shape or np.any(s < 0) or np.any(s > self.length[raw]):
            raise ValueError('in-range arclength per source required')
        return raw.astype(np.int64), s

    def errors(self, source, arclength):
        """Model-only errors; evaluation declarations are added at the handoff."""
        rows, s = self._queries(source, arclength)
        state = self.initial
        ex, eu = state.position_error_mm[rows], state.direction_error[rows]
        g = state.context.spatial.lipschitz_tesla_per_mm
        with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
            ag = (tsstep.K * g) / state.momentum[rows]
            ae = (tsstep.K * self.field_error_tesla[rows]) / state.momentum[rows]
            coupling = ag * ex
            rate = coupling + ae
            turn = rate * s
            gradient_turn = (ag * s / 2) * s
            drift = eu * s
            coupling_position = (rate * s / 2) * s
            gradient_position = (ag * s / 6) * s * s
            dx = np.minimum(ex + drift + coupling_position + gradient_position, ex + 2*s)
            du = np.minimum(eu + turn + gradient_turn, 2.)
        if not all(np.all(np.isfinite(v)) for v in (ag, ae, rate, dx, du)):
            raise ValueError('carried model errors outside supported finite range')
        positive = s > 0
        if np.any((g > 0) & (ag == 0)) or np.any((self.field_error_tesla[rows] > 0) & (ae == 0)):
            raise ValueError('nonzero carried error rate underflows')
        if np.any((g > 0) & (ex > 0) & (coupling == 0)) or any(np.any(mask & (value == 0))
                for mask, value in ((positive & (rate > 0), coupling_position),
                                    (positive & (g > 0), gradient_position),
                                    (positive & (rate > 0), turn),
                                    (positive & (g > 0), gradient_turn),
                                    (positive & (eu > 0), drift))):
            raise ValueError('nonzero carried position term underflows')
        return Errors(_fixed(dx), _fixed(du))

    def reference_state(self, source, arclength):
        """Evaluated nominal x,direction; NOT an exact or RK4 propagated endpoint."""
        rows, s = self._queries(source, arclength)
        state = self.initial
        direction = state.direction[rows]
        direction = direction / np.hypot.reduce(direction, axis=1)[:, None]
        q = np.where((state.pdg[rows] == 13) | (state.pdg[rows] == -211), -1., 1.)
        from mu2efastsim.ehelix import helix
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            x, u = helix(state.x[rows], direction, state.momentum[rows], q, self.field[rows], s)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)) or np.any(np.hypot.reduce(u, axis=1) == 0):
            raise ValueError('nonfinite reference endpoint/direction')
        bm = np.hypot.reduce(self.field[rows], axis=1)
        phase = (tsstep.K * bm / state.momentum[rows]) * s
        if np.any((s > 0) & (phase <= np.pi/2) & np.all(x == state.x[rows], axis=1)):
            raise ValueError('positive reference displacement is not representable')
        return x, u

    def balls(self, source, s_lo, s_hi):
        """Actual ideal-path balls, retaining initial and absolute local-s errors."""
        rows, lo = self._queries(source, s_lo)
        _, hi = self._queries(source, s_hi)
        if np.any(lo > hi):
            raise ValueError('ordered arclength interval required')
        width = hi - lo
        middle = lo + width/2
        a, _ = self.reference_state(rows, lo)
        b, _ = self.reference_state(rows, hi)
        midpoint, _ = self.reference_state(rows, middle)
        center = a/2 + b/2
        em, ee = self.errors(rows, middle).position_mm, self.errors(rows, hi).position_mm
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            rm = np.maximum(middle-lo, hi-middle) + em + self.position_error_mm[rows]
            sagitta = (self.curvature[rows]*width/8)*width
            rc = np.maximum(np.hypot.reduce(a-center, axis=1), np.hypot.reduce(b-center, axis=1))
            rc = rc + sagitta + ee + self.position_error_mm[rows]
        if not all(np.all(np.isfinite(v)) for v in (center, rm, rc)):
            raise ValueError('nonfinite carried-path enclosure')
        if np.any((width > 0) & (self.curvature[rows] > 0) & (sagitta == 0)):
            raise ValueError('nonzero curved enclosure term underflows')
        return tsvary.Balls(*(_fixed(a) for a in (rows, lo, hi, midpoint, rm, center, rc, em, ee)))

    def owners(self, source, s_lo, s_hi, geometry, *, geometry_error_mm):
        """Use the shared separate-centre strict whole-ball ownership tests."""
        return tsvary.MapEnclosure.owners(self, source, s_lo, s_hi, geometry,
                                        geometry_error_mm=geometry_error_mm)


@dataclass(frozen=True)
class Attempt:
    state: State
    status: np.ndarray
    accepted: np.ndarray
    owner_index: np.ndarray
    clearance_margin_mm: np.ndarray


def attempt(segment, geometry, *, geometry_error_mm):
    """Transactionally carry only whole-interval conditional vacuum rows.

    Material/unresolved rows retain their entire initial nominal state, errors,
    and path; no partial advancement, later-vacuum jump or endpoint-only decision.
    Accepted rows add BOTH evaluation declarations to the model errors. This is
    an uncertain nominal reference receipt, not an actual endpoint/crossing.
    Fixed scalar momentum and source row tags are unchanged even on rollback.
    """
    if type(segment) is not Segment:
        raise ValueError('explicit carried Segment required')
    state = segment.initial
    rows = np.arange(len(state.x))
    owner = segment.owners(rows, np.zeros(len(rows)), segment.length, geometry,
                           geometry_error_mm=geometry_error_mm)
    accepted = owner.status == tslocate.VACUUM
    x, u = state.x.copy(), state.direction.copy()
    ex, eu, path = state.position_error_mm.copy(), state.direction_error.copy(), state.path_mm.copy()
    use = np.flatnonzero(accepted)
    if len(use):
        xe, ue = segment.reference_state(use, segment.length[use])
        model = segment.errors(use, segment.length[use])
        with np.errstate(over='ignore', invalid='ignore'):
            ex[use] = model.position_mm + segment.position_error_mm[use]
            eu[use] = np.minimum(2., model.direction + segment.direction_error[use])
            path[use] = path[use] + segment.length[use]
        if np.any((segment.length[use] > 0) & (path[use] == state.path_mm[use])):
            raise ValueError('cumulative path advancement is not representable')
        x[use], u[use] = xe, ue
    next_state = State(state.context, x, u, state.momentum, state.pdg, position_error_mm=ex,
                       direction_error=eu, source_index=state.source_index, path_mm=path)
    status = np.where(accepted, VACUUM_ADVANCED,
                      np.where(owner.status == tslocate.MATERIAL, MATERIAL_PENDING, UNRESOLVED))
    return Attempt(next_state, _fixed(status), _fixed(accepted), owner.volume_index, owner.clearance_margin_mm)
