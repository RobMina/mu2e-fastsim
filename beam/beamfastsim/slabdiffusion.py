"""Opt-in, infinite-plane slab diagnostic; NOT production transport or G4 parity.

Numerical contract (MeV, mm, unit directions)
--------------------------------------------
The direction diffuses in *true path* s with generator Delta_S2/(2 lambda1(T)).
Each interior polygon vertex uses two independent tangent Gaussian components
of variance h/lambda1(T), followed by the spherical exponential map. There is
no effective-path inversion, Highland kick, field, decay, lateral foil edge,
or explicit hard secondary. This is a weak, finite-step diffusion model, not
Geant4's condensed-history angular distribution or single-scattering tail.

A leg first drifts along the current direction, with requested length
min(max_step, angular_fraction*lambda1(T), R(T)-R((1-energy_fraction)*T)).
The energy argument is bounded at the declared table floor (never extrapolated)
and the leg is clipped at the FIRST of the two faces it intersects (coincident
face/domain events own the face within 16 machine-epsilon times max(thickness,h)). Loss uses
T-R_inverse(R(T)-actual_leg), optionally passed as ``mean=`` to the companion
``eloss_jax.sample_loss``. At an interior, continuing vertex the direction is
rotated using actual_leg/lambda1(T_entry). There is no terminal kick: the exit
direction is the direction of the final polygon segment, and points outwards.
Thus the first leg is ballistic and boundary angular errors are finite-step
errors. Step-size convergence is required; this is not a Brownian bridge or
an exact continuous first-passage solver.

The floor is at least the common table lower bound. A positive energy at/below
it, or within 16 float64 eps * floor above it, is UNRESOLVED_FLOOR, NOT a
physical stop. This local roundoff allowance covers log/interpolation/exp and
subtraction at the domain edge, where distinct energies can map to identical
ranges and hence a zero step budget. Residual energy is retained verbatim:
no snapping to floor, fictitious loss, extra path, or physical stop is inferred. A sample >= available T is clipped
to T, recorded separately as RAWZERO (also NOT a certified physical stop).
Loss is assigned at a leg endpoint; no sub-leg stochastic stopping location is
inferred. A face reached on that leg remains in the ledger even for RAWZERO.
Otherwise face exits take precedence over floor: a known positive sub-floor
exit is retained verbatim, without a 0.1 MeV threshold. Cap/numerical endpoints
are unresolved and retain their residual energy. No endpoint is called a
physical stop by this module.

Inclusive CSDA means with a finite-cut Urban sampler are a diagnostic hybrid:
its width/branches still use entry momentum, actual length and tcut. It does
not consistently resolve restricted vs inclusive loss or delta-ray/hard-loss
tails, and the ledger is primary-energy debit, NOT local energy deposition.
Table material/charge/provenance must be chosen by the caller. In particular,
a mu+-derived lambda table shared with mu- is an assumption. Supplying tables
does not validate those physics choices. Production defaults are untouched.

``respond`` is a host-validated array API around a fixed-shape masked JAX loop.
Keys are fold_in(seed, row_id), replica_id, step, then channel (0 angular,
1 loss); they never depend on batch, compaction, or another lane's lifetime.
Explicit row_ids and replica_offset preserve streams across separate calls.
Bitwise replay is tested on the same CPU/backend, not promised across devices
or JAX versions. ``loss='none'`` and ``scatter=False`` are unit-test controls.
"""
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from mu2efastsim import eloss, eloss_jax


ACTIVE = 0
TRANSMITTED = 1
REFLECTED = 2
UNRESOLVED_FLOOR = 3
RAWZERO = 4
UNRESOLVED_CAP = 5
UNRESOLVED_NUMERICAL = 6
# One executable lane width even for partial/small host batches prevents XLA's
# shape-dependent scalar/vector arithmetic from perturbing bitwise replay.
_KERNEL_WIDTH = 256
# CPU vector-tail lowering can differ by a few ulps for the final position
# lanes even at fixed shape. Keep eight permanently inactive guard lanes so
# real rows never occupy that tail when host batches are repartitioned.
_EXEC_WIDTH = _KERNEL_WIDTH + 8
STATUS_NAMES = {TRANSMITTED: "transmitted", REFLECTED: "reflected",
                UNRESOLVED_FLOOR: "unresolved_floor", RAWZERO: "sampled_loss_rawzero",
                UNRESOLVED_CAP: "unresolved_cap", UNRESOLVED_NUMERICAL: "unresolved_numerical"}


class Tables(NamedTuple):
    """Natural-log nodes; same field layout as layerphys.Tables, without extrapolation."""
    loge: object
    logr: object
    lam_loge: object
    lam_log: object


class Response(NamedTuple):
    """NumPy arrays (replicas, rows), vectors (..., 3), and per-lane ledger arrays.

    raw_energy is residual primary kinetic energy at ANY endpoint, including
    unresolved ones; consult status before interpreting it as exit energy.
    Ledger sums: mean_loss, requested_loss (unclipped sample), applied_loss,
    clipped_loss (requested minus applied), angular_time (rotated h/lambda).
    Ledger flags: face_reached, floor_reached, rawzero, cap, numerical.
    """
    raw_energy: object
    status: object
    path: object
    position: object
    direction: object
    steps: object
    ledger: dict


class _Settings(NamedTuple):
    max_step: float
    angular_fraction: float
    energy_fraction: float
    max_steps: int
    floor: float
    ceiling: float
    scatter: bool
    loss: str
    tcut: float


def make_tables(energy, range_mm, msc_energy, lambda1_mm):
    """Validate explicit positive finite nodes; range and energies strictly increase."""
    arrays = [np.asarray(v, dtype=np.float64) for v in (energy, range_mm, msc_energy, lambda1_mm)]
    for v in arrays:
        if v.ndim != 1 or len(v) < 2 or not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("table nodes must be positive finite 1-D arrays with >=2 nodes")
    if arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[3].shape:
        raise ValueError("table energy/value shapes must match")
    if any(np.any(np.diff(arrays[i]) <= 0) for i in (0, 1, 2)):
        raise ValueError("range and energy nodes must strictly increase")
    return Tables(*(jnp.asarray(np.log(v)) for v in arrays))


def _range(tb, t):
    return jnp.exp(jnp.interp(jnp.log(t), tb.loge, tb.logr))


def _energy(tb, r):
    return jnp.exp(jnp.interp(jnp.log(r), tb.logr, tb.loge))


def _at_floor(t, floor):
    """Domain-edge roundoff only; never a generic zero-budget/nonfinite fallback.

    Keep the scale local to the floor, not the incident energy or unity. The
    raw residual and loss ledger are untouched even when t is just above it.
    """
    return t <= floor + 16 * jnp.finfo(jnp.float64).eps * floor


def _norm(v):
    return jnp.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2 + v[..., 2] ** 2)


def angular_step(key, direction, variance):
    """Jittable single-lane S2 exponential-map step; caller supplies unit u, variance>=0.

    ``jax.vmap(angular_step)`` vectorizes independent keyed lanes. For small
    variance q, E[P_l(u_new.u)] = 1-l(l+1)q/2+O(q^2), NOT an exact heat kernel.
    """
    u = jnp.asarray(direction, jnp.float64)
    axis = jnp.where(jnp.abs(u[2]) < 0.9, jnp.array([0., 0., 1.]), jnp.array([1., 0., 0.]))
    a = jnp.cross(u, axis)
    a = a / _norm(a)
    b = jnp.cross(u, a)
    g = jax.random.normal(key, (2,), dtype=jnp.float64) * jnp.sqrt(variance)
    tangent = g[0] * a + g[1] * b
    theta = jnp.sqrt(g[0] ** 2 + g[1] ** 2)
    v = jnp.cos(theta) * u + jnp.sinc(theta / jnp.pi) * tangent
    return v / _norm(v)


@partial(jax.jit, static_argnames=("cfg",))
def _run(energy, direction, thickness, rows, replicas, seed, tb, mc, cfg):
    n = energy.shape[0]
    keys = jax.vmap(lambda row, rep: jax.random.fold_in(
        jax.random.fold_in(jax.random.key(seed), row), rep))(rows, replicas)
    back_entry = direction[:, 2] < 0
    pos = jnp.zeros((n, 3), jnp.float64).at[:, 2].set(jnp.where(back_entry, thickness, 0.))
    status = jnp.where(_at_floor(energy, cfg.floor), UNRESOLVED_FLOOR, ACTIVE).astype(jnp.int32)
    zero = jnp.zeros(n, jnp.float64)
    # ledger columns: mean, requested, applied, clipped, angular_time, face, floor
    ledger = jnp.zeros((n, 7), jnp.float64).at[:, 6].set(_at_floor(energy, cfg.floor))
    state = (jnp.int32(0), energy, direction, pos, zero, status, jnp.zeros(n, jnp.int32), ledger)
    rf = _range(tb, cfg.floor)

    def cond(state):
        return (state[0] < cfg.max_steps) & jnp.any(state[5] == ACTIVE)

    def step(state):
        i, t, u, x, path, status, steps, ledger = state
        active = status == ACTIVE
        safe_t = jnp.clip(t, cfg.floor, cfg.ceiling)
        r = _range(tb, safe_t)
        lam = jnp.exp(jnp.interp(jnp.log(safe_t), tb.lam_loge, tb.lam_log))
        target = jnp.maximum((1 - cfg.energy_fraction) * safe_t, cfg.floor)
        budget = jnp.maximum(r - _range(tb, target), 0.)
        h = jnp.minimum(cfg.max_step, budget)
        if cfg.scatter:
            h = jnp.minimum(h, cfg.angular_fraction * lam)
        uz = u[:, 2]
        face_z = jnp.where(uz > 0, thickness, 0.)
        distance = jnp.where(uz != 0, (face_z - x[:, 2]) / jnp.where(uz != 0, uz, 1.), jnp.inf)
        distance = jnp.maximum(distance, 0.)
        # Own coincident face/domain events despite a few ulps of accumulated
        # range/position roundoff; the resulting leg ends exactly at the face.
        face_tol = 16 * jnp.finfo(jnp.float64).eps * jnp.maximum(thickness, h)
        face = distance <= h + face_tol
        dl = jnp.where(active, jnp.where(face, distance, h), 0.)
        bad = active & ((h <= 0) | ~jnp.isfinite(h))
        dl = jnp.where(bad, 0., dl)
        # Do not extrapolate the inverse range below rf, even by roundoff.
        mean = jnp.maximum(safe_t - _energy(tb, jnp.maximum(r - dl, rf)), 0.)
        mean = jnp.where(active & ~bad, mean, 0.)
        step_keys = jax.vmap(lambda k: jax.random.fold_in(k, i))(keys)
        if cfg.loss == "sample":
            loss_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(step_keys)
            p = jnp.sqrt(safe_t * (safe_t + 2 * eloss.MMU))
            requested = jax.vmap(lambda k, pp, ll, mm: eloss_jax.sample_loss(
                k, mc, pp, ll, cfg.tcut, mean=mm))(loss_keys, p, dl, mean)
        elif cfg.loss == "mean":
            requested = mean
        else:
            requested = jnp.zeros_like(mean)
            mean = jnp.zeros_like(mean)
        bad = bad | (active & (~jnp.isfinite(requested) | (requested < 0)))
        valid = active & ~bad
        dl = jnp.where(valid, dl, 0.)
        requested = jnp.where(valid, requested, 0.)
        mean = jnp.where(valid, mean, 0.)
        applied = jnp.minimum(requested, jnp.maximum(t, 0.))
        tout = t - applied
        rawzero = valid & (requested >= t)
        hit = valid & face
        # Exactly reaching the domain boundary in a mean step must not depend
        # on exp/log roundoff. A sampled undershoot outside the local floor
        # roundoff allowance continues; nonfinite samples still remain numerical.
        floor_hit = valid & _at_floor(tout, cfg.floor)
        if cfg.loss == "mean":
            floor_hit = floor_hit | (valid & (r - dl <= rf))
        nxt = x + dl[:, None] * u
        nxt = nxt.at[:, 2].set(jnp.where(hit, face_z, jnp.clip(nxt[:, 2], 0., thickness)))
        exiting_back = uz > 0
        exit_status = jnp.where(exiting_back == back_entry, REFLECTED, TRANSMITTED)
        new_status = jnp.where(bad, UNRESOLVED_NUMERICAL,
                              jnp.where(rawzero, RAWZERO, jnp.where(hit, exit_status,
                                        jnp.where(floor_hit, UNRESOLVED_FLOOR, ACTIVE))))
        continuing = valid & (new_status == ACTIVE)
        q = jnp.zeros_like(dl)
        if cfg.scatter:
            angle_keys = jax.vmap(lambda k: jax.random.fold_in(k, 0))(step_keys)
            q = jnp.where(continuing, dl / lam, 0.)
            rotated = jax.vmap(angular_step)(angle_keys, u, q)
            u = jnp.where(continuing[:, None], rotated, u)
        add = jnp.stack((mean, requested, applied, requested - applied, q,
                         hit.astype(jnp.float64), floor_hit.astype(jnp.float64)), axis=1)
        return (i + 1, jnp.where(active, tout, t), u, jnp.where(valid[:, None], nxt, x),
                path + dl, jnp.where(active, new_status, status), steps + valid.astype(jnp.int32), ledger + add)

    _, t, u, x, path, status, steps, ledger = jax.lax.while_loop(cond, step, state)
    status = jnp.where(status == ACTIVE, UNRESOLVED_CAP, status)
    return t, status, path, x, u, steps, ledger


def _integer(value, name, minimum=0, maximum=2**32 - 1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def respond(energy, direction, thickness, tables, *, material=None, tcut=1.0,
            replicas=1, seed=1, batch=256, row_ids=None, replica_offset=0,
            max_step=0.01, angular_fraction=0.02, energy_fraction=0.05,
            max_steps=10000, floor=None, scatter=True, loss="sample"):
    """Transport explicit entries through 0<=z<=thickness; return :class:`Response`.

    energy: (n,) MeV; direction: (n,3) finite unit vectors. Each starts at
    (0,0,0) for uz>=0, or (0,0,thickness) for uz<0. thickness is scalar or (n,).
    ``tables`` is make_tables(...) or compatible layerphys.Tables. Positive
    entries at/below the common table floor (including its documented local
    roundoff allowance) return unresolved immediately; above
    its upper bound are rejected. floor may raise, never lower, this boundary.
    A tangent entry is allowed; there is no finite lateral boundary.

    loss: 'sample' requires an explicit Material/MatConsts; 'mean' integrates
    the supplied range deterministically; 'none' freezes energy for diagnostics.
    row_ids: unique uint32-compatible identities (default arange(n)); replicas
    are indexed from replica_offset. batch only partitions host execution into
    chunks of at most 256 entries, padded to the same executable with 256 payload and 8 inactive guard lanes; this
    intentionally trades small-batch speed for exact shape-independent replay.
    max_steps is an exact nonnegative cap (zero is a useful unresolved control).
    The returned ledger has cumulative per-lane quantities, not step histories.
    """
    replicas = _integer(replicas, "replicas", 1)
    seed = _integer(seed, "seed")
    batch = _integer(batch, "batch", 1)
    replica_offset = _integer(replica_offset, "replica_offset")
    _integer(replica_offset + replicas - 1, "last replica id")
    max_steps = _integer(max_steps, "max_steps", maximum=2**31 - 1)
    if not isinstance(scatter, (bool, np.bool_)) or loss not in ("sample", "mean", "none"):
        raise ValueError("scatter must be boolean; loss must be sample, mean, or none")
    for name, value in (("max_step", max_step), ("angular_fraction", angular_fraction),
                        ("energy_fraction", energy_fraction), ("tcut", tcut)):
        if np.ndim(value) != 0 or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if energy_fraction >= 1:
        raise ValueError("energy_fraction must be below one")
    try:
        logs = [np.asarray(getattr(tables, field), float) for field in Tables._fields]
        if any(not np.all(np.isfinite(v)) for v in logs):
            raise ValueError("finite log tables required")
        with np.errstate(over="ignore", under="ignore"):
            tb = make_tables(*(np.exp(v) for v in logs))
    except (AttributeError, TypeError) as exc:
        raise ValueError("explicit log Tables required") from exc
    # Keep the caller's exact log nodes (avoid repeated exp/log perturbation).
    tb = Tables(*(jnp.asarray(v) for v in logs))
    lower = float(np.exp(max(logs[0][0], logs[2][0]) if scatter else logs[0][0]))
    upper = float(np.exp(min(logs[0][-1], logs[2][-1]) if scatter else logs[0][-1]))
    if floor is None:
        floor = lower
    if (np.ndim(floor) != 0 or not np.isfinite(floor) or floor <= 0
            or np.log(floor) < np.log(lower) - 4 * np.finfo(float).eps
            or floor >= upper):
        raise ValueError("floor must lie within the common table domain, below its upper edge")
    floor = max(float(floor), lower)  # permit only exp/log roundoff at the lower node
    energy = np.asarray(energy, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if energy.ndim != 1 or not np.all(np.isfinite(energy)) or np.any(energy <= 0) or np.any(energy > upper):
        raise ValueError("energy must be a positive finite 1-D array no higher than the table domain")
    n = len(energy)
    if direction.shape != (n, 3) or not np.all(np.isfinite(direction)):
        raise ValueError("direction must have finite shape (n,3)")
    norms = np.linalg.norm(direction, axis=1)
    if not np.all(np.abs(norms - 1.) <= 1e-10):
        raise ValueError("directions must be unit vectors")
    direction = direction / norms[:, None]
    thickness = np.asarray(thickness, dtype=np.float64)
    if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0):
        raise ValueError("thickness must be positive and finite")
    try:
        thickness = np.broadcast_to(thickness, (n,))
    except ValueError as exc:
        raise ValueError("thickness must be scalar or (n,)") from exc
    if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0):
        raise ValueError("thickness must be positive and finite")
    rows = np.arange(n, dtype=np.uint32) if row_ids is None else np.asarray(row_ids)
    if rows.shape != (n,) or (n and (rows.dtype.kind not in "iu" or np.any(rows < 0)
                                   or np.any(rows > 2**32 - 1) or len(np.unique(rows)) != n)):
        raise ValueError("row_ids must be unique uint32-compatible integers of shape (n,)")
    rows = rows.astype(np.uint32)
    mc = eloss_jax.MatConsts(*(0. for _ in eloss_jax.MatConsts._fields))
    if loss == "sample":
        if material is None:
            raise ValueError("sample loss requires an explicit material")
        mc = eloss_jax.consts(material)
        if any(np.ndim(v) != 0 or not np.isfinite(v) for v in mc):
            raise ValueError("material constants must be finite scalars")
        if any(getattr(mc, f) <= 0 for f in ("I", "ne", "e0", "e1", "e2")):
            raise ValueError("material I, density and excitation energies must be positive")
    cfg = _Settings(float(max_step), float(angular_fraction), float(energy_fraction), max_steps,
                    float(floor), upper, bool(scatter), loss, float(tcut))
    count = n * replicas
    arrays = [np.empty(count), np.empty(count, np.int32), np.empty(count), np.empty((count, 3)),
              np.empty((count, 3)), np.empty(count, np.int32), np.empty((count, 7))]
    for start in range(0, count, min(batch, _KERNEL_WIDTH)):
        flat = np.arange(start, min(start + min(batch, _KERNEL_WIDTH), count))
        index, rep = flat % n, flat // n + replica_offset
        m = len(flat)
        # Padded lanes are inactive from initialization, with safe finite data.
        ee = np.full(_EXEC_WIDTH, cfg.floor)
        uu = np.tile([0., 0., 1.], (_EXEC_WIDTH, 1))
        tt = np.ones(_EXEC_WIDTH)
        rr = np.zeros(_EXEC_WIDTH, np.uint32)
        pp = np.zeros(_EXEC_WIDTH, np.uint32)
        ee[:m], uu[:m], tt[:m], rr[:m], pp[:m] = energy[index], direction[index], thickness[index], rows[index], rep
        result = _run(*(jnp.asarray(v) for v in (ee, uu, tt, rr, pp)), seed, tb, mc, cfg)
        for target, value in zip(arrays, result):
            target[start:start + m] = np.asarray(value)[:m]
    shape = (replicas, n)
    e, st, path, x, u, steps, accum = arrays
    ledger = {name: accum[:, i].reshape(shape) for i, name in enumerate(
        ("mean_loss", "requested_loss", "applied_loss", "clipped_loss", "angular_time"))}
    ledger.update(face_reached=accum[:, 5].reshape(shape).astype(bool),
                  floor_reached=accum[:, 6].reshape(shape).astype(bool),
                  rawzero=(st == RAWZERO).reshape(shape), cap=(st == UNRESOLVED_CAP).reshape(shape),
                  numerical=(st == UNRESOLVED_NUMERICAL).reshape(shape))
    return Response(e.reshape(shape), st.reshape(shape), path.reshape(shape), x.reshape(shape + (3,)),
                    u.reshape(shape + (3,)), steps.reshape(shape), ledger)
