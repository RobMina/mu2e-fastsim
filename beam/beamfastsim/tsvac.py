"""Bounded adaptive conditional vacuum reference prefixes, not physical transport.

Only whole-interval vacuum enclosures permit a nominal reference handoff. Trials
that fail map coverage, geometry, or error budgets halve FROM THE SAME prestate;
rejected trials never consume path or add evaluation errors. A minimum failed
step, lifetime trial count or accepted-step count stops a lane, not a particle
loss or crossing. Floating geometry and evaluation-error declarations remain
uncertified. No selector change, native tolerance, clock, decay or material step.
"""

from dataclasses import dataclass

import numpy as np

from . import tsarc, tsbound, tscarry, tslocate, tsnav

PATH_LIMIT = 1
INITIAL_GEOMETRY = 2
INITIAL_MAP = 3
INITIAL_BUDGET = 4
MIN_STEP = 5
TRIAL_LIMIT = 6
STEP_LIMIT = 7
STATUS = {PATH_LIMIT: 'requested_reference_path_complete', INITIAL_GEOMETRY: 'initial_geometry_unresolved_or_material',
          INITIAL_MAP: 'initial_map_uncertainty_uncovered', INITIAL_BUDGET: 'initial_error_budget',
          MIN_STEP: 'failed_trial_at_minimum_step', TRIAL_LIMIT: 'lifetime_trial_limit',
          STEP_LIMIT: 'accepted_step_limit'}
NONE = 0
MAP = 1
GEOMETRY = 2
MATERIAL = 3
POSITION_BUDGET = 4
DIRECTION_BUDGET = 5
REASON = {NONE: 'none_or_accepted', MAP: 'uncertain_reach_box_uncovered_or_discontinuous',
          GEOMETRY: 'whole_interval_geometry_unresolved', MATERIAL: 'whole_interval_material',
          POSITION_BUDGET: 'candidate_position_error_budget', DIRECTION_BUDGET: 'candidate_direction_error_budget'}


def _subset(state, rows):
    return tscarry.State(state.context, state.x[rows], state.direction[rows], state.momentum[rows], state.pdg[rows],
                        position_error_mm=state.position_error_mm[rows], direction_error=state.direction_error[rows],
                        source_index=state.source_index[rows], path_mm=state.path_mm[rows])


def _exact_add(a, b):
    """Fail if the exact sum of two finite binary floats is not representable."""
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        total = a + b
        virtual = total - a
        residual = (a - (total - virtual)) + (b - virtual)
    if np.any(~np.isfinite(total)) or np.any(residual != 0):
        raise ValueError('path bookkeeping requires exactly representable binary-float sums')
    return total


def _covered(state, length):
    with np.errstate(over='ignore', invalid='ignore'):
        radius = state.position_error_mm + length
        lo, hi = state.x - radius[:, None], state.x + radius[:, None]
    return tsbound.box_coverage(state.context.item, lo, hi, continuous=True)


@dataclass(frozen=True)
class Ledger:
    row_index: np.ndarray
    path_before_mm: np.ndarray
    attempted_mm: np.ndarray
    rejection_reason: np.ndarray
    accepted: np.ndarray
    owner_index: np.ndarray
    clearance_margin_mm: np.ndarray
    position_error_before_mm: np.ndarray
    direction_error_before: np.ndarray
    position_error_after_mm: np.ndarray
    direction_error_after: np.ndarray


@dataclass(frozen=True)
class Result:
    state: tscarry.State
    status: np.ndarray
    progress_mm: np.ndarray
    remaining_mm: np.ndarray
    trials: np.ndarray
    steps: np.ndarray
    last_rejection_reason: np.ndarray
    last_rejected_mm: np.ndarray
    initial_geometry_status: np.ndarray
    initial_owner_index: np.ndarray
    initial_clearance_margin_mm: np.ndarray
    ledger: Ledger
    floating_point_certified: bool = False
    native_query_certified: bool = False


def walk(initial, geometry, path_limit, *, field_error_tesla, position_error_mm, direction_error,
         geometry_error_mm, max_position_error_mm, max_direction_error,
         max_step=5., min_step=1e-6, max_trials=1024, max_steps=256, max_records=1000000):
    """Return only accepted conditional vacuum prefixes, in original row order.

    All lengths are mm and path_limit is ADDITIONAL requested reference length.
    Numerical evaluation declarations and error budgets are scalar or (N,).
    Retries halve, with min_step as a floor for a FAILED trial, not a ban on a
    shorter positive final remainder. No zero-length transport attempt is used.
    Initial geometry includes carried position error and evaluation/geometry
    declarations but does not commit a zero-length handoff or inflate errors.
    Zero request returns unchanged without a geometry/map transport check.

    max_trials is a LIFETIME per-lane attempt count, including accepted trials.
    max_steps counts only accepted handoffs. n*max_trials must fit max_records
    before work; no truncated ledger or silent partial success on resource error.
    Unsupported numerical/phase/arithmetic errors abort the batch, not a retry.

    Both relative and cumulative path sums must be exactly representable as
    binary floats (checked with error-free TwoSum). Otherwise the call fails,
    rather than discarding a positive remainder. No tolerance-based completion.
    A stop reason describes a failed trial/control limit, NOT a physical boundary
    location. Its attempted length is not a root bracket. Ledger records current
    accepted progress; rejected candidates retain prestate/errors/path entirely.
    """
    if type(initial) is not tscarry.State or not isinstance(geometry, tslocate.Partition):
        raise ValueError('explicit carried State and selected-cell Partition required')
    n = len(initial.x)
    limit = tsarc._lanes(path_limit, (n,), 'path_limit', nonnegative=True)
    eb = tsarc._lanes(field_error_tesla, (n,), 'field_error_tesla', nonnegative=True)
    ep = tsarc._lanes(position_error_mm, (n,), 'position_error_mm', nonnegative=True)
    eu = tsarc._lanes(direction_error, (n,), 'direction_error', nonnegative=True)
    bp = tsarc._lanes(max_position_error_mm, (n,), 'max_position_error_mm', nonnegative=True)
    bu = tsarc._lanes(max_direction_error, (n,), 'max_direction_error', nonnegative=True)
    if np.any(eu > 2) or np.any(bu > 2):
        raise ValueError('direction declaration and budget must be in [0,2]')
    error = tsnav._scalar(geometry_error_mm, 'geometry_error_mm', nonnegative=True)
    maximum = tsnav._scalar(max_step, 'max_step', positive=True)
    minimum = tsnav._scalar(min_step, 'min_step', positive=True)
    if minimum > maximum:
        raise ValueError('min_step must be <= max_step')
    for value, name, high in ((max_trials, 'max_trials', 100000), (max_steps, 'max_steps', 100000),
                              (max_records, 'max_records', 1000000)):
        if type(value) is not int or not 1 <= value <= high:
            raise ValueError(f'{name} requires integer in [1,{high}]')
    if n * max_trials > max_records:
        raise ValueError('worst-case ledger exceeds max_records; no partial result')
    status, trials, steps, reason = (np.zeros(n, dtype=np.int64) for _ in range(4))
    progress, last_h = np.zeros(n), np.zeros(n)
    current = initial
    initial_status = np.full(n, -1, dtype=np.int64)
    initial_owner = np.full(n, -1, dtype=np.int64)
    initial_margin = np.zeros(n)
    status[limit == 0] = PATH_LIMIT
    initial_rows = np.flatnonzero(status == 0)
    if len(initial_rows):
        start = _subset(initial, initial_rows)
        covered = _covered(start, np.zeros(len(initial_rows)))
        status[initial_rows[~covered]] = INITIAL_MAP
        loc = geometry.locate(start.x)
        margin = loc.clearance - start.position_error_mm - ep[initial_rows] - error
        owned = np.isin(loc.status, (tslocate.VACUUM, tslocate.MATERIAL)) & (margin > 0)
        initial_status[initial_rows] = np.where(owned, loc.status, 0)
        initial_owner[initial_rows] = np.where(owned, loc.volume_index, -1)
        initial_margin[initial_rows] = np.where(owned, margin, 0.)
        geometric = (loc.status == tslocate.VACUUM) & (margin > 0)
        status[initial_rows[(status[initial_rows] == 0) & ~geometric]] = INITIAL_GEOMETRY
        budget = (start.position_error_mm + ep[initial_rows] <= bp[initial_rows]) & (
                   np.minimum(2., start.direction_error + eu[initial_rows]) <= bu[initial_rows])
        status[initial_rows[(status[initial_rows] == 0) & ~budget]] = INITIAL_BUDGET
    trial_h = np.minimum(maximum, limit).copy()
    parts = []
    while np.any(status == 0):
        rows = np.flatnonzero(status == 0)
        state = _subset(current, rows)
        h = trial_h[rows]
        before_progress = progress[rows].copy()
        rejection = np.full(len(rows), MAP, dtype=np.int64)
        accepted = np.zeros(len(rows), dtype=bool)
        owner = np.full(len(rows), -1, dtype=np.int64)
        margin = np.zeros(len(rows))
        proposed_x, proposed_u = state.x.copy(), state.direction.copy()
        proposed_ex, proposed_eu = state.position_error_mm.copy(), state.direction_error.copy()
        proposed_path = state.path_mm.copy()
        inside = _covered(state, h)
        covered_rows = np.flatnonzero(inside)
        if len(covered_rows):
            segment = tscarry.Segment(_subset(state, covered_rows), h[covered_rows],
                        field_error_tesla=eb[rows[covered_rows]], position_error_mm=ep[rows[covered_rows]],
                        direction_error=eu[rows[covered_rows]])
            candidate = tscarry.attempt(segment, geometry, geometry_error_mm=error)
            owner[covered_rows], margin[covered_rows] = candidate.owner_index, candidate.clearance_margin_mm
            local = np.where(candidate.accepted, NONE,
                             np.where(candidate.status == tscarry.MATERIAL_PENDING, MATERIAL, GEOMETRY))
            exceeds_position = candidate.state.position_error_mm > bp[rows[covered_rows]]
            exceeds_direction = candidate.state.direction_error > bu[rows[covered_rows]]
            local[candidate.accepted & exceeds_position] = POSITION_BUDGET
            local[candidate.accepted & ~exceeds_position & exceeds_direction] = DIRECTION_BUDGET
            rejection[covered_rows] = local
            ok = covered_rows[local == NONE]
            take = np.flatnonzero(local == NONE)
            accepted[ok] = True
            proposed_x[ok], proposed_u[ok] = candidate.state.x[take], candidate.state.direction[take]
            proposed_ex[ok], proposed_eu[ok] = (candidate.state.position_error_mm[take],
                                              candidate.state.direction_error[take])
            proposed_path[ok] = _exact_add(state.path_mm[ok], h[ok])
        trials[rows] += 1
        accepted_rows = rows[accepted]
        if len(accepted_rows):
            progress[accepted_rows] = _exact_add(progress[accepted_rows], h[accepted])
            if np.any(progress[accepted_rows] <= before_progress[accepted]):
                raise ValueError('positive trial made no representable progress')
            remaining = _exact_add(limit[accepted_rows], -progress[accepted_rows])
            if np.any(remaining < 0):
                raise ValueError('accepted prefix exceeds requested path')
            x, u = current.x.copy(), current.direction.copy()
            ex, de, path = current.position_error_mm.copy(), current.direction_error.copy(), current.path_mm.copy()
            x[accepted_rows], u[accepted_rows] = proposed_x[accepted], proposed_u[accepted]
            ex[accepted_rows], de[accepted_rows], path[accepted_rows] = (proposed_ex[accepted], proposed_eu[accepted],
                                                                       proposed_path[accepted])
            current = tscarry.State(initial.context, x, u, initial.momentum, initial.pdg, position_error_mm=ex,
                                   direction_error=de, source_index=initial.source_index, path_mm=path)
            steps[accepted_rows] += 1
            status[accepted_rows[remaining == 0]] = PATH_LIMIT
            reason[accepted_rows], last_h[accepted_rows] = NONE, 0.
            trial_h[accepted_rows] = np.minimum(maximum, remaining)
        parts.append(tuple(tscarry._fixed(v) for v in (rows, before_progress, h, rejection, accepted, owner, margin,
                           state.position_error_mm, state.direction_error, proposed_ex, proposed_eu)))
        failed = ~accepted
        failed_rows = rows[failed]
        reason[failed_rows], last_h[failed_rows] = rejection[failed], h[failed]
        status[failed_rows[h[failed] <= minimum]] = MIN_STEP
        status[rows[(status[rows] == 0) & (trials[rows] >= max_trials)]] = TRIAL_LIMIT
        status[rows[(status[rows] == 0) & (steps[rows] >= max_steps)]] = STEP_LIMIT
        retry = failed & (status[rows] == 0)
        smaller = np.maximum(minimum, h[retry]/2)
        if np.any((smaller <= 0) | (smaller >= h[retry])):
            raise ValueError('positive retry length is not representably smaller')
        trial_h[rows[retry]] = smaller
    if parts:
        columns = [np.concatenate(items) for items in zip(*parts)]
    else:
        columns = [np.empty(0, dtype=dtype) for dtype in (np.int64, float, float, np.int64, bool, np.int64,
                                                        float, float, float, float, float)]
    ledger = Ledger(*(tscarry._fixed(v) for v in columns))
    remaining = _exact_add(limit, -progress)
    return Result(current, *(tscarry._fixed(v) for v in (status, progress, remaining, trials, steps, reason, last_h,
                                                       initial_status, initial_owner, initial_margin)), ledger)
