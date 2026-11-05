"""Opt-in safety-limited CPU vacuum lineage driver; not full PS/TS transport.

The caller supplies a COMPLETE open vacuum region through a pure positive-inside
1-Lipschitz ``clearance(x)`` function. Each committed arclength is smaller than
that clearance: a physical unit-speed curved path cannot leave its safety ball.
This avoids the false assumption that an empty chord or two vacuum endpoints
prove a curved path misses material. The supplied region is NOT certified here
against any native geometry, hierarchy, release, tolerances or omitted material.
RK4 accuracy is separate from the geometric arclength bound. Field stages and
numerical endpoints are also checked against the ball; no ODE error bound follows.

Boundary approach is terminal BOUNDARY_PENDING, NOT absorption, rejection or a
transmission result. Material transitions need a subsequent navigator/physics
handler. No boundary nudging, chord veto or guessed collimator loss is performed.
"""

from dataclasses import dataclass

import numpy as np

from . import tsdecay, tsstep


PATH_LIMIT = 1
BOUNDARY_PENDING = 2
START_NOT_INTERIOR = 3
MUON_DECAY = 4
ITERATION_LIMIT = 5
TAU_MU_NS = 2196.9811
STATUS = {PATH_LIMIT: "path_limit", BOUNDARY_PENDING: "boundary_pending",
          START_NOT_INTERIOR: "start_not_interior", MUON_DECAY: "muon_decay",
          ITERATION_LIMIT: "iteration_limit"}
LIMITATIONS = (
    "Safety-limited vacuum development driver, not full PS/TS transport or an "
    "efficiency estimator. Caller geometry completeness and pure 1-Lipschitz "
    "clearance are unverified contracts. RK4 has no rigorous ODE error bound. "
    "Boundary pending is unresolved, never an absorption/loss classification. "
    "No material crossings, ionization, scattering, hadronic interactions, "
    "native map-cache history, exit-surface passage census or POT normalization. "
    "Conditional_muonic mode assigns 100% pi -> mu nu, not physical branching; "
    "muon decay terminates the lineage without transporting its daughters. "
    "No muon polarization or stock/Run1A physics-validity claim."
)


@dataclass(frozen=True)
class Result:
    """One lineage per original input row; original ordering is never changed.

    (root_index, generation) is a LOCAL numerical identity, never a native art
    TrackID. generation is 0 for the original and 1 for its muonic daughter.
    External native keys must stay attached by root_index, not nearest matching.
    Pion parents are retained in the decay ledger; clocks reset for the daughter.
    path is total lineage length, proper_time is current-particle age [ns].
    NaNs in the pion ledger mean no represented pion decay; use pion_decayed.
    """

    x: np.ndarray
    p3: np.ndarray
    pdg: np.ndarray
    time: np.ndarray
    path: np.ndarray
    proper_time: np.ndarray
    remaining: np.ndarray
    root_index: np.ndarray
    generation: np.ndarray
    status: np.ndarray
    steps: np.ndarray
    pion_decayed: np.ndarray
    pion_decay_x: np.ndarray
    pion_decay_p3: np.ndarray
    pion_decay_time: np.ndarray
    daughter_birth_p3: np.ndarray
    neutrino_p3: np.ndarray
    mode: str


def _scalar(value, name, *, positive=True):
    a = tsstep._finite(value, name)
    if a.shape != () or (a <= 0 if positive else a < 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} scalar")
    return float(a)


def _clearance(region, points):
    value = tsstep._finite(region.clearance(points), "clearance")
    if value.shape != (len(points),):
        raise ValueError("clearance must return exactly shape (N,)")
    return value


def _accumulate(total, correction, lanes, increment):
    delta = increment - correction[lanes]
    updated = total[lanes] + delta
    correction[lanes] = (updated - total[lanes]) - delta
    total[lanes] = updated


def _guarded_field(evaluate, start, radius):
    def guarded(points):
        distance = np.hypot.reduce(points - start, axis=1)
        if np.any(~np.isfinite(distance) | (distance >= radius)):
            raise ValueError("RK field query left the conservative safety ball")
        return evaluate(points)
    return guarded


def propagate(x, p3, pdg, path_limit, field, region, *, mode, rng=None,
              proper_remaining=None, time=0, max_step=5.0, safety_fraction=0.5,
              boundary_atol=1e-6, max_steps=100000):
    """Advance a batch in caller-certified vacuum, stopping before boundaries.

    Inputs x/p3: (N,3); species: mu+/- or pi+/-. Lengths mm, momenta MeV/c,
    clocks ns. path_limit/time/proper_remaining are scalar or (N,). max_steps
    caps vector iterations, not events. No per-particle Python loop is used.
    Pure position-only field and region queries are required; exceptions propagate.

    mode MUST be 'no_decay' or 'conditional_muonic'. The latter samples complete
    pion/muon lifetimes unless supplied; a new muon always gets its own lifetime.
    rng MUST be a caller-owned numpy Generator, never a seed/global RNG. Invalid
    initial numeric/geometry inputs consume no random numbers. Field errors after
    initial validation may consume draws; calls are not RNG transactions.

    A lane within boundary_atol of the region's conservative zero set stops even
    if that zero is a CSG-internal seam. Such stops do not establish real material.
    Starts outside/on the boundary are START_NOT_INTERIOR, not lost particles.
    A decay at the path limit is processed before PATH_LIMIT. Initially zero
    limits stop without decay; zero remaining lifetime at positive limit decays
    at the input point if it is farther than boundary_atol from the boundary.
    Geometry boundary proximity takes priority over unresolved decay there.
    Decay endpoint scheduling uses proper clocks with an 8-epsilon relative
    roundoff allowance on the initial lifetime/current age, not a tuned mm cut.
    Lifetimes separated by less than that allowance are numerically unresolved.
    """
    if mode not in ("no_decay", "conditional_muonic"):
        raise ValueError("explicit mode must be no_decay or conditional_muonic")
    if not callable(getattr(region, "clearance", None)):
        raise ValueError("region must provide pure clearance(points)")
    max_step = _scalar(max_step, "max_step")
    fraction = _scalar(safety_fraction, "safety_fraction")
    tolerance = _scalar(boundary_atol, "boundary_atol")
    if fraction > 0.5:
        raise ValueError("safety_fraction must be <= 0.5")
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    x, u, p, q, limit, mass, t = tsstep._inputs(x, p3, pdg, path_limit, None, time)
    n = len(x)
    species = np.broadcast_to(np.asarray(pdg), (n,)).astype(np.int64, copy=True)
    x, momentum, t = x.copy(), u * p[:, None], t.copy()
    initial_time = t.copy()
    elapsed, time_correction = np.zeros(n), np.zeros(n)
    limit = limit.copy()
    clearance = _clearance(region, x)
    evaluate = tsstep._checked_field(field)
    with np.errstate(over="ignore"):
        initial_bmag = np.hypot.reduce(evaluate(x), axis=1)
    if np.any(~np.isfinite(initial_bmag)):
        raise ValueError("initial field magnitude is outside finite float64 range")
    decay = mode == "conditional_muonic"
    if decay:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("conditional_muonic requires a caller-owned numpy Generator")
        tsdecay._pion_inputs(momentum[np.abs(species) == 211], species[np.abs(species) == 211])
        if proper_remaining is None:
            remaining = rng.exponential(np.where(np.abs(species) == 211, tsdecay.TAU_PI_NS, TAU_MU_NS))
        else:
            remaining = tsstep._lanes(proper_remaining, "proper_remaining", n).copy()
            if np.any(remaining < 0):
                raise ValueError("proper_remaining must be nonnegative")
    else:
        if proper_remaining is not None or rng is not None:
            raise ValueError("no_decay accepts neither rng nor proper_remaining")
        remaining = np.full(n, np.inf)
    deadline = remaining.copy()
    path, age = np.zeros(n), np.zeros(n)
    path_correction, age_correction = np.zeros(n), np.zeros(n)
    generation, status, steps = (np.zeros(n, dtype=np.int64) for _ in range(3))
    decayed = np.zeros(n, dtype=bool)
    parent_x, parent_p, daughter_p, neutrino = (np.full((n, 3), np.nan) for _ in range(4))
    parent_t = np.full(n, np.nan)
    status[clearance <= 0] = START_NOT_INTERIOR
    status[(status == 0) & (limit == 0)] = PATH_LIMIT
    for _ in range(max_steps):
        live = np.flatnonzero(status == 0)
        if not len(live):
            break
        c = _clearance(region, x[live])
        pending = c <= tolerance
        status[live[pending]] = BOUNDARY_PENDING
        live, c = live[~pending], c[~pending]
        if not len(live):
            continue
        requested = np.minimum(np.minimum(max_step, fraction * c), limit[live] - path[live])
        guarded = _guarded_field(evaluate, x[live], c)
        if decay:
            prepared = tsstep._prepare(x[live], momentum[live], species[live], requested,
                                       guarded, None, t[live])
            requested = prepared[4]
            proposed_clock = tsdecay.proper_time(requested, p[live], mass[live])
            clock_slack = 8 * np.finfo(float).eps * np.maximum(deadline[live], age[live])
            scheduled = remaining[live] <= proposed_clock + clock_slack
            if np.any(scheduled):
                distance, _ = tsdecay.where_decay(requested[scheduled], p[live][scheduled],
                                                  remaining[live][scheduled], mass[live][scheduled])
                requested[scheduled] = np.minimum(requested[scheduled], distance)
        advanced = tsstep.step(x[live], momentum[live], species[live], requested,
                               guarded, time=0)
        displacement = np.hypot.reduce(advanced.x - x[live], axis=1)
        if np.any(~np.isfinite(displacement) | (displacement >= c)):
            raise ValueError("RK endpoint left the conservative safety ball")
        endpoint_clearance = _clearance(region, advanced.x)
        if np.any(endpoint_clearance <= 0):
            raise ValueError("endpoint contradicts the positive-clearance safety contract")
        if np.any((advanced.path > 0) & (displacement == 0)):
            raise ValueError("spatial advancement is not representable at this origin")
        if np.any((advanced.path > 0) & (path[live] + advanced.path <= path[live])):
            raise ValueError("path accumulation made no representable progress")
        x[live], momentum[live] = advanced.x, advanced.p3
        _accumulate(elapsed, time_correction, live, advanced.time)
        t[live] = initial_time[live] + elapsed[live]
        if not np.all(np.isfinite(t[live])):
            raise ValueError("accumulated lab time is outside finite float64 range")
        _accumulate(path, path_correction, live, advanced.path)
        _accumulate(age, age_correction, live, advanced.proper_time)
        steps[live] += 1
        status[live[endpoint_clearance <= tolerance]] = BOUNDARY_PENDING
        if decay:
            remaining[live] = np.maximum(0.0, deadline[live] - age[live])
            decay_here = scheduled & (advanced.path == requested) & (status[live] == 0)
            dying = live[decay_here]
            remaining[dying] = 0.0
            pion = dying[np.abs(species[dying]) == 211]
            muon = dying[np.abs(species[dying]) == 13]
            status[muon] = MUON_DECAY
            if len(pion):
                products = tsdecay.sample(rng, momentum[pion], species[pion])
                parent_x[pion], parent_p[pion], parent_t[pion] = x[pion], momentum[pion], t[pion]
                daughter_p[pion], neutrino[pion] = products.muon_p3, products.neutrino_p3
                momentum[pion], species[pion] = products.muon_p3, products.muon_pdg
                p[pion], mass[pion] = np.hypot.reduce(momentum[pion], axis=1), tsdecay.MMU
                age[pion], remaining[pion] = 0.0, rng.exponential(TAU_MU_NS, size=len(pion))
                deadline[pion], age_correction[pion] = remaining[pion], 0.0
                generation[pion], decayed[pion] = 1, True
        done = live[(status[live] == 0) & (path[live] >= limit[live])]
        status[done] = PATH_LIMIT
    status[status == 0] = ITERATION_LIMIT
    return Result(x, momentum, species, t, path, age, remaining, np.arange(n), generation,
                  status, steps, decayed, parent_x, parent_p, parent_t, daughter_p,
                  neutrino, mode)
