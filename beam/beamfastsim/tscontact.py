"""Conditional uncertain FIRST-contact sets, never material restart states.

For first contact s* in [lo,hi], evaluate the incoming reference at lo, not a
post-contact midpoint. Unit speed and precontact curvature imply
Rx=Ex(lo)+eval_x+(hi-lo), Ru=min(2,Eu(lo)+eval_u+kappa*(hi-lo)).
Direction center denotes the EXACT mathematical normalization of its stored raw
vector. No projected position, point-valued restart, material exit or clock is
created. Arithmetic here is outward under tsinterval's binary64 contracts;
upstream root, trajectory, field, geometry and evaluation bounds remain assumed.
"""

from dataclasses import dataclass

import numpy as np

from . import tscarry, tsinterval as iv, tsplane, tssurface


@dataclass(frozen=True)
class Contact:
    entry: tssurface.Entry
    row_index: np.ndarray
    source_index: np.ndarray
    entry_status: np.ndarray
    local_lo_mm: np.ndarray
    local_hi_mm: np.ndarray
    cumulative_path: iv.Interval
    x_center_mm: np.ndarray
    position_radius_mm: np.ndarray
    direction_vector: np.ndarray
    direction_radius: np.ndarray
    momentum: np.ndarray
    pdg: np.ndarray
    center_plane_residual: iv.Interval
    intended_plane_slab: iv.Interval
    local_arithmetic_outward: bool = True
    floating_point_certified: bool = False
    native_query_certified: bool = False
    restart_state: bool = False
    material_transport: bool = False

    @property
    def face(self):
        return self.entry.face


def contact(entry):
    """Preserve conditional first-entry sets for every unique row, even capped.

    Nonunique rows remain in the original entry receipt, not in the compact set.
    Each set is a ball INTERSECTED WITH the original conditional finite face,
    not the whole ball treated as material. If the intended face differs from
    the stored plane, intended_plane_slab bounds its stored-plane residual;
    do not impose zero stored residual or project off-face centers. Incoming
    direction is the one-sided preinteraction direction.
    The original segment and face bindings are retained; row tags are not full
    native particle identities. No error is reset or momentum recomputed.
    """
    if type(entry) is not tssurface.Entry or type(entry.segment) is not tscarry.Segment:
        raise ValueError('explicit segment-bound Entry required')
    segment = entry.segment
    initial = segment.initial
    n = len(segment.length)
    if any(np.shape(getattr(entry, key)) != (n,) for key in
           ('unique_entry', 'source_index', 'status', 'lo_mm', 'hi_mm')):
        raise ValueError('entry row shapes do not match its original segment')
    if not np.array_equal(entry.source_index, initial.source_index):
        raise ValueError('entry source tags do not match its original segment')
    rows = np.flatnonzero(entry.unique_entry)
    lo, hi = entry.lo_mm[rows], entry.hi_mm[rows]
    if (np.any(lo < 0) or np.any(lo > hi) or np.any(hi > segment.length[rows])
            or np.any(entry.derivative_lower[rows] <= 0) or np.any(entry.guard_margin_mm[rows] <= 0)
            or np.any(~np.isin(entry.status[rows], (tssurface.ISOLATED, tssurface.UNCERTAINTY_LIMIT,
                tssurface.ITERATION_LIMIT, tssurface.NUMERICAL_LIMIT)))):
        raise ValueError('invalid conditional unique entry receipt')
    x, direction = segment.reference_state(rows, lo)
    errors = segment.errors(rows, lo)
    width = iv.sub(iv.Interval(hi), iv.Interval(lo)).hi
    width = np.maximum(0., width)
    rx = iv.add(iv.add(iv.Interval(errors.position_mm),
        iv.Interval(segment.position_error_mm[rows])), iv.Interval(width)).hi
    turn = iv.mul(iv.Interval(segment.curvature[rows]), iv.Interval(width))
    ru = iv.add(iv.add(iv.Interval(errors.direction),
        iv.Interval(segment.direction_error[rows])), turn).hi
    ru = np.minimum(2., ru)
    if np.any(rx < 0) or np.any(ru < 0):
        raise ValueError('invalid contact radius')
    path = iv.add(iv.Interval(initial.path_mm[rows]), iv.Interval(lo, hi))
    path = iv.Interval(np.maximum(0., path.lo), path.hi)
    residual = tsplane.signed(iv.Interval(x), iv.Interval(entry.face.origin), iv.Interval(entry.face.inward_normal))
    pe = entry.plane_error_mm[rows]
    slab = iv.Interval(-pe, pe)
    values = (rows, initial.source_index[rows], entry.status[rows], lo, hi, x, rx, direction,
              ru, initial.momentum[rows], initial.pdg[rows])
    rows, tags, status, lo, hi, x, rx, direction, ru, momentum, pdg = [tscarry._fixed(a) for a in values]
    return Contact(entry, rows, tags, status, lo, hi, path, x, rx, direction, ru, momentum, pdg, residual, slab)
