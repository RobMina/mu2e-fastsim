"""Opt-in, NumPy vacuum RK4 primitives for muons and charged pions.

Units: x/path mm, p3 MeV/c, B tesla, mass MeV/c**2, clocks ns. The core
``mu2efastsim.ehelix.rk4`` API at matched core pin c366e0c is imported lazily;
no core code, default beam transport or JAX code is changed/imported here.

A callable field MUST be a pure position -> field function, returning exactly
(N, 3). Exceptions (including map CoverageError) propagate unchanged. In
particular, DO NOT close over mutable MapSelector previous-map state: RK stages
and root trials are speculative, not committed queries. There is no selector
trial/commit/rollback protocol or selector receipt support in this primitive.
Purity cannot be checked for arbitrary Python callables; it is a caller contract.
"""

from dataclasses import dataclass

import numpy as np

from .tsdecay import C_MM_NS, MMU, MPI

K = 0.299792458
MAX_START_ANGLE = 0.1
LIMITATIONS = (
    "Vacuum numerical primitive only, not geometry navigation or full transport. "
    "No material-veto certificate, apertures, losses, scattering, decays, hadronic "
    "interactions, or Geant4/PS/TS physics-validity claim. The 0.1 rad cap uses "
    "only the start field, not a global error bound for varying fields. Caller "
    "step-halving studies are necessary. Plane results solve a single numerical "
    "RK4 segment under an explicit caller monotonicity contract, not an earliest "
    "physical crossing certificate. Pure fields only; stateful selectors and "
    "their trial protocols/receipts are unsupported."
)


@dataclass(frozen=True)
class StepResult:
    """Batched endpoint; time is absolute, proper_time is an INCREMENT [ns].

    path is the actual advanced length, possibly less than requested h because
    of the start-field angle cap. In vacuum the momentum magnitude is fixed.
    Inputs are never updated; callers own identity and accumulated proper clocks.
    """

    x: np.ndarray
    p3: np.ndarray
    time: np.ndarray
    proper_time: np.ndarray
    path: np.ndarray


@dataclass(frozen=True)
class PlaneStepResult:
    """state is the root where within_step, otherwise the full capped endpoint.

    end always holds the full capped StepResult (including xend = end.x).
    fraction is root path/end.path, NaN for no bracket or excluded direction.
    No hit means no numerical endpoint bracket, NOT no physical passage.
    """

    state: StepResult
    end: StepResult
    within_step: np.ndarray
    fraction: np.ndarray


def _finite(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real and finite")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be real and finite")
    return array


def _lanes(value, name, n):
    array = _finite(value, name)
    if array.shape not in ((), (n,)):
        raise ValueError(f"{name} must be scalar or shape (N,)")
    return np.broadcast_to(array, (n,))


def _inputs(x, p3, pdg, h, mass, time):
    x, p3 = _finite(x, "x"), _finite(p3, "p3")
    if x.ndim != 2 or x.shape[1] != 3 or p3.shape != x.shape:
        raise ValueError("x and p3 must have the same shape (N, 3)")
    n = len(x)
    pdg = _lanes(pdg, "pdg", n)
    if np.any(~np.isin(pdg, (13, -13, 211, -211))):
        raise ValueError("pdg must be +13, -13, +211 or -211")
    # PDG 13 is mu-; PDG -211 is pi-.
    q = np.where((pdg == 13) | (pdg == -211), -1.0, 1.0)
    m = np.where(np.abs(pdg) == 13, MMU, MPI) if mass is None else _lanes(mass, "mass", n)
    h, t = _lanes(h, "h", n), _lanes(time, "time", n)
    p = np.hypot.reduce(p3, axis=1)
    if np.any(~np.isfinite(p) | (p <= 0)):
        raise ValueError("momentum magnitude must be finite and strictly positive")
    if np.any(m <= 0):
        raise ValueError("mass must be strictly positive")
    if np.any(h < 0):
        raise ValueError("h must be nonnegative")
    return x, p3 / p[:, None], p, q, h, m, t


def _checked_field(field):
    if not callable(field):
        constant = _finite(field, "field")
        if constant.shape != (3,):
            raise ValueError("constant field must have shape (3,)")
        constant = constant.copy()

        def evaluate(x):
            return np.broadcast_to(constant, x.shape)
    else:
        def evaluate(x):
            value = _finite(field(x), "field result")
            if value.shape != x.shape:
                raise ValueError("callable field result must have exactly shape (N, 3)")
            return value
    return evaluate


def _prepare(x, p3, pdg, h, field, mass, time):
    x, u, p, q, h, m, t = _inputs(x, p3, pdg, h, mass, time)
    evaluate = _checked_field(field)
    bm = np.hypot.reduce(evaluate(x), axis=1)
    if np.any(~np.isfinite(bm)):
        raise ValueError("field magnitude is outside finite float64 range")
    # Overflow of a very large bound is harmless: min(h, inf) is finite.
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        bound = (MAX_START_ANGLE / K) * (p / bm)
    path = np.minimum(h, bound)
    if np.any((h > 0) & (path <= 0)):
        raise ValueError("angle-limited path is outside the supported float64 range")
    return x, u, p, q, path, evaluate, m, t


def _advance(x, u, p, q, path, field, mass, time):
    # Keep this dependency lazy: importing tsstep requires NumPy, not JAX/core.
    from mu2efastsim.ehelix import rk4

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xn, un = rk4(x, u, p, q, field, path)
        pn = un * p[:, None]
        from .tsdecay import proper_time

        proper = proper_time(path, p, mass)
        tn = time + np.hypot(path / C_MM_NS, proper)
    if not all(np.all(np.isfinite(a)) for a in (xn, pn, proper, tn)):
        raise ValueError("derived RK4 state or clock is outside finite float64 range")
    return StepResult(xn, pn, tn, proper, path.copy())


def step(x, p3, pdg, h, field, mass=None, time=0):
    """Advance one vacuum RK4 segment, capped at start-field angle <= 0.1 rad.

    x and p3 have identical shape (N,3). pdg, h, time and optional positive
    mass are scalar or (N,). Default masses: |PDG|=13 -> MMU, 211 -> MPI;
    explicit mass overrides kinematics, never charge. h is finite/nonnegative,
    p strictly positive even for h=0; time may be any finite absolute clock.
    A constant field has shape (3,); a PURE callable returns exactly x.shape.
    No global accuracy guarantee follows from the cap; study step halving.
    """
    return _advance(*_prepare(x, p3, pdg, h, field, mass, time))


def plane_step(x, p3, pdg, h, field, origin, normal, mass=None, time=0, *,
               assume_monotone=False, direction=0, atol=1e-9, maxiter=64):
    """Bracketed plane root in ONE capped segment; not a propagation driver.

    phi = dot(x-origin, normal), with a shared unit normal and origin, each
    shape (3,). direction=+1/-1 selects increasing/decreasing phi; 0 accepts
    both. Only endpoint sign brackets (or endpoints within atol) are searched.
    Every trial re-integrates the fraction from the PRE-step state with the
    SAME core RK4, never a frozen-field helix. atol is a positive mm residual.

    ``assume_monotone=True`` is REQUIRED: the caller asserts strict monotonicity
    of phi(RK4(f*h)), f in [0,1], and no multiple passages in this segment.
    Endpoint velocity checks can disprove this contract but cannot establish
    it. This is no proof of monotonicity or earliest root of the physical ODE.
    Starts on/within atol of the plane, zero paths, and tangent endpoint
    velocities are unsupported and raise, rather than certifying a departure
    or grazing hit. There is no multipassage scan, max-path driver, material
    navigation, selector-state protocol, or accuracy receipt.
    """
    if assume_monotone is not True:
        raise ValueError("explicit assume_monotone=True caller contract is required")
    if np.ndim(direction) != 0 or direction not in (-1, 0, 1):
        raise ValueError("direction must be -1, 0 or +1")
    atol = _finite(atol, "atol")
    if atol.shape != () or atol <= 0:
        raise ValueError("atol must be a strictly positive scalar")
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, (int, np.integer)) or maxiter <= 0:
        raise ValueError("maxiter must be a strictly positive integer")
    origin, normal = _finite(origin, "origin"), _finite(normal, "normal")
    if origin.shape != (3,) or normal.shape != (3,):
        raise ValueError("origin and normal must have shape (3,)")
    if not np.isclose(np.hypot.reduce(normal), 1.0, rtol=0, atol=1e-12):
        raise ValueError("normal must be a unit vector")
    args = _prepare(x, p3, pdg, h, field, mass, time)
    x, u, p, q, path, evaluate, m, t = args
    phi0 = (x - origin) @ normal
    if np.any(~np.isfinite(phi0)):
        raise ValueError("plane residual must be finite")
    if np.any(path == 0):
        raise ValueError("plane_step requires positive path in every lane")
    if np.any(np.abs(phi0) <= atol):
        raise ValueError("start on plane/departure is unsupported")
    end = _advance(*args)
    phi1 = (end.x - origin) @ normal
    v0, v1 = u @ normal, (end.p3 / p[:, None]) @ normal
    if np.any(~np.isfinite(phi1)):
        raise ValueError("plane residual must be finite")
    if np.any((np.abs(v0) <= 1e-14) | (np.abs(v1) <= 1e-14)):
        raise ValueError("tangent endpoint is unsupported")
    if np.any((np.sign(v0) != np.sign(v1)) | (np.sign(phi1 - phi0) != np.sign(v0))):
        raise ValueError("endpoint evidence contradicts monotonicity contract")
    hit = (np.sign(phi0) != np.sign(phi1)) | (np.abs(phi1) <= atol)
    if direction:
        hit &= np.sign(v0) == direction
    lo, hi = np.zeros_like(path), np.ones_like(path)
    fraction = np.ones_like(path)
    done = ~hit | (np.abs(phi1) <= atol)
    for _ in range(maxiter):
        active = ~done
        if not np.any(active):
            break
        trial_fraction = np.where(active, 0.5 * (lo + hi), fraction)
        trial = _advance(x, u, p, q, path * trial_fraction, evaluate, m, t)
        phi = (trial.x - origin) @ normal
        if np.any(~np.isfinite(phi)):
            raise ValueError("trial plane residual must be finite")
        converged = active & (np.abs(phi) <= atol)
        fraction = np.where(converged, trial_fraction, fraction)
        done |= converged
        same = np.sign(phi) == np.sign(phi0)
        lo = np.where(active & ~converged & same, trial_fraction, lo)
        hi = np.where(active & ~converged & ~same, trial_fraction, hi)
    if not np.all(done):
        raise RuntimeError("RK4 plane root did not converge to atol within maxiter")
    state = _advance(x, u, p, q, path * fraction, evaluate, m, t)
    return PlaneStepResult(state, end, hit, np.where(hit, fraction, np.nan))
