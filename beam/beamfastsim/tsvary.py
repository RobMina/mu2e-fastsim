"""Conditional single-map varying-field path balls, not a transport integrator.

Compare the ideal unit-speed Lorentz trajectory r,u with a uniform-field helix
R,v sharing its exact initial state and momentum. Set a=|q|K/p and freeze B0 at
that state. If |B(r)-B0| <= G*s + eB, then w=u-v obeys
w'=a_signed*(w cross B(r) + v cross (B(r)-B0)). The skew first term does not
increase |w|. Thus |w(s)| <= a*(G*s**2/2 + eB*s), and integrating once gives
|r(s)-R(s)| <= a*(G*s**3/6 + eB*s**2/2). Also |r-R| <= 2*s.

This is a model-derived error bound, NOT an RK4 discrepancy or certified floating
arithmetic. Every reachable point must lie inside ONE continuous map domain.
Explicit field/helix evaluation errors remain caller assumptions. No native
cache/geometry agreement, material crossing, endpoint integration or clocks.
"""

from dataclasses import dataclass

import numpy as np

from . import tsarc, tsbound, tsfield, tslocate, tsnav, tsstep


def _fixed_typed(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class Balls:
    source_index: np.ndarray
    s_lo: np.ndarray
    s_hi: np.ndarray
    midpoint: np.ndarray
    midpoint_radius_mm: np.ndarray
    chord_center: np.ndarray
    chord_radius_mm: np.ndarray
    midpoint_model_error_mm: np.ndarray
    endpoint_model_error_mm: np.ndarray


@dataclass(frozen=True)
class Ownership:
    status: np.ndarray
    volume_index: np.ndarray
    ball_kind: np.ndarray
    clearance_margin_mm: np.ndarray


@dataclass(frozen=True, eq=False, init=False)
class MapEnclosure:
    """Immutable initial-state reference and conditional actual-map error bound.

    field_error_tesla bounds the frozen field evaluation error at each initial
    position relative to the ideal trilinear map. position_error_mm bounds EVERY
    helix reference evaluation, including input normalization/arithmetic effects.
    Neither may be inferred from RK step doubling. Zero is a numerical
    idealization. Bounds/radius arithmetic themselves lack interval certification.
    After construction no caller map/state mutation changes this object.
    """

    reference: tsarc.UniformHelix
    field_bound: tsbound.Bound
    spatial_bound: tsbound.SpatialBound
    field_error_tesla: np.ndarray
    curvature: np.ndarray
    floating_point_certified: bool = False
    native_query_certified: bool = False

    def __init__(self, x, p3, pdg, item, length, *, field_error_tesla,
                 position_error_mm, chunk_nodes=65536):
        if type(item) is not tsfield.FieldMap:
            raise ValueError('explicit single FieldMap required')
        x, p3 = tsnav._real(x, 'x'), tsnav._real(p3, 'p3')
        if x.ndim != 2 or x.shape[1] != 3 or p3.shape != x.shape:
            raise ValueError('x and p3 require matching (N,3) shapes')
        length = tsarc._lanes(length, (len(x),), 'length', nonnegative=True)
        error = tsarc._lanes(field_error_tesla, (len(x),), 'field_error_tesla', nonnegative=True)
        with np.errstate(over='ignore', invalid='ignore'):
            lo, hi = x - length[:, None], x + length[:, None]
        tsbound.require_boxes(item, lo, hi, continuous=True)
        if np.any((length[:, None] > 0) & ((lo == x) | (hi == x))):
            raise ValueError('reachable box displacement is not representable')
        fb = tsbound.magnitude(item, chunk_nodes=chunk_nodes)
        sb = tsbound.spatial(item, chunk_nodes=chunk_nodes)
        reference = tsarc.UniformHelix(x, p3, pdg, item.field(x), length,
                                      position_error_mm=position_error_mm)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            curvature = tsstep.K * fb.magnitude_tesla / reference.momentum
        if np.any(~np.isfinite(curvature)) or np.any((fb.magnitude_tesla > 0) & (curvature == 0)):
            raise ValueError('map curvature outside supported finite range')
        for name, value in dict(reference=reference, field_bound=fb, spatial_bound=sb,
                                field_error_tesla=error,
                                curvature=tsnav._fixed(curvature, curvature.shape, 'curvature'),
                                floating_point_certified=False, native_query_certified=False).items():
            object.__setattr__(self, name, value)
        self.model_error(np.arange(len(x)), length)

    def model_error(self, source, arclength):
        """Position deviation from the ideal frozen-field reference at absolute s."""
        source, s = self._queries(source, arclength)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            gradient = self.spatial_bound.lipschitz_tesla_per_mm
            ag = (tsstep.K * gradient) / self.reference.momentum[source]
            ae = (tsstep.K * self.field_error_tesla[source]) / self.reference.momentum[source]
            variation = (ag * s / 6) * s * s
            evaluation = (ae * s / 2) * s
            value = np.minimum(variation + evaluation, 2 * s)
        if np.any(~np.isfinite(value)):
            raise ValueError('model position bound outside supported finite range')
        if np.any((s > 0) & (gradient > 0) & (variation == 0)) or np.any(
                (s > 0) & (self.field_error_tesla[source] > 0) & (evaluation == 0)):
            raise ValueError('nonzero model position bound underflows')
        return value

    def _queries(self, source, arclength):
        raw = np.asarray(source)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu':
            raise ValueError('source requires one-dimensional integer row indices')
        if np.any(raw < 0) or np.any(raw >= len(self.reference.x)):
            raise ValueError('source index out of range')
        source = raw.astype(np.int64)
        s = tsnav._real(arclength, 'arclength')
        if s.shape != source.shape or np.any(s < 0) or np.any(s > self.reference.length[source]):
            raise ValueError('arclength requires in-range value per source row')
        return source, s

    def balls(self, source, s_lo, s_hi):
        """Two independently centred balls for the ACTUAL ideal varying-field path.

        Midpoint uses unit-speed reach from the uncertain true midpoint; chord
        uses the true-curve curvature plus both uncertain endpoint evaluations.
        Frozen-reference positions are never returned as actual propagated states.
        """
        source, lo = self._queries(source, s_lo)
        _, hi = self._queries(source, s_hi)
        if np.any(lo > hi):
            raise ValueError('ordered arclength interval required')
        width = hi - lo
        middle = lo + width / 2
        a, _ = self.reference.state(source, lo)
        b, _ = self.reference.state(source, hi)
        midpoint, _ = self.reference.state(source, middle)
        center = a / 2 + b / 2
        em, ee = self.model_error(source, middle), self.model_error(source, hi)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            rm = np.maximum(middle - lo, hi - middle) + em + self.reference.position_error_mm[source]
            sagitta = (self.curvature[source] * width / 8) * width
            rc = np.maximum(np.hypot.reduce(a - center, axis=1), np.hypot.reduce(b - center, axis=1))
            rc = rc + sagitta + ee + self.reference.position_error_mm[source]
        if not all(np.all(np.isfinite(v)) for v in (center, rm, rc)):
            raise ValueError('nonfinite varying-field enclosure arithmetic')
        if np.any((width > 0) & (self.curvature[source] > 0) & (sagitta == 0)):
            raise ValueError('nonzero curved enclosure term underflows')
        values = (source, lo, hi, midpoint, rm, center, rc, em, ee)
        return Balls(*(_fixed_typed(v) for v in values))

    def owners(self, source, s_lo, s_hi, geometry, *, geometry_error_mm):
        """Conditional whole-interval owners; zero status means UNRESOLVED, not lost.

        No subdivision, handoff, advancement or crossing is implied. Each ball
        queries geometry at its own centre. Unresolved outputs have owner -1.
        """
        if not isinstance(geometry, tslocate.Partition):
            raise ValueError('selected-cell Partition required')
        error = tsnav._scalar(geometry_error_mm, 'geometry_error_mm', nonnegative=True)
        balls = self.balls(source, s_lo, s_hi)
        lm, lc = geometry.locate(balls.midpoint), geometry.locate(balls.chord_center)
        mm = lm.clearance - error - balls.midpoint_radius_mm
        mc = lc.clearance - error - balls.chord_radius_mm
        sm = np.isin(lm.status, (tslocate.VACUUM, tslocate.MATERIAL)) & (mm > 0)
        sc = np.isin(lc.status, (tslocate.VACUUM, tslocate.MATERIAL)) & (mc > 0)
        if np.any(sm & sc & (lm.volume_index != lc.volume_index)):
            raise ValueError('contradictory ownership enclosure guarantees')
        use_chord = sc & (~sm | (mc > mm))
        safe = sm | sc
        status = np.where(safe, np.where(use_chord, lc.status, lm.status), 0)
        owner = np.where(safe, np.where(use_chord, lc.volume_index, lm.volume_index), -1)
        kind = np.where(safe, np.where(use_chord, tsarc.CHORD_BALL, tsarc.MIDPOINT_BALL), 0)
        margin = np.where(safe, np.where(use_chord, mc, mm), 0.)
        return Ownership(*(_fixed_typed(v) for v in (status, owner, kind, margin)))
