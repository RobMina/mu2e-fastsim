"""Opt-in NumPy pion/muon physics; momenta MeV/c, masses MeV/c², lengths mm, times ns.

No transport driver or geometry lives here: use mu2efastsim.ehelix for magnetic
stepping and mu2efastsim.materials for material constants. The caller owns decay
ordering, charge/PDG bookkeeping, proper-time integration and finite-energy step
limits. No pion nuclear interactions, capture, radiative losses or spin tracking
are modelled. Mass-substituted core muon ionisation is only an approximation for
spin-zero pions (the spin-zero corrections are missing).
"""
import numpy as np

from mu2efastsim import eloss, materials, transport

MPI = 139.57039
MMU = eloss.MMU
TAU_PI_NS = 26.033
TAU_MU_NS = 2196.9811
C_MM_NS = 299.792458
TK_STOP_MEV = 0.2
_TCUT_MEV = 1e9

__all__ = ["MPI", "MMU", "TAU_PI_NS", "TAU_MU_NS", "C_MM_NS", "TK_STOP_MEV",
           "mass_mev", "lifetime_ns", "sample_proper_lifetime", "decay_pions", "material_step"]


def _is_pion(pdg):
    pdg = np.asarray(pdg)
    if not np.all(np.isin(pdg, (-211, -13, 13, 211))):
        raise ValueError("pdg must contain only charged pions (+/-211) or muons (+/-13)")
    return np.abs(pdg) == 211


def mass_mev(pdg):
    """Invariant masses for scalar/array PDG IDs; other species raise ValueError.

    PDG +211 is pi+, -211 pi-, +13 mu-, -13 mu+. Charge signs must not be
    inferred uniformly from sign(pdg); daughter IDs are the driver's job.
    """
    return np.where(_is_pion(pdg), MPI, MMU)


def lifetime_ns(pdg):
    """Mean vacuum proper lifetimes [ns], same for both charges; shape follows pdg."""
    return np.where(_is_pion(pdg), TAU_PI_NS, TAU_MU_NS)


def sample_proper_lifetime(rng, pdg):
    """Independent exponential proper lifetimes [ns], shape follows scalar/array pdg.

    Draw once per particle and integrate d(proper time) = mass * ds / (p*c)
    along its flight. At fixed momentum, lab time = gamma * proper time and
    decay distance = (p/mass) * c * proper time. At rest, lab and proper time
    coincide; this sampler does not implement muon capture or decay daughters.
    """
    return rng.exponential(lifetime_ns(pdg))


def _momenta(pvec):
    pvec = np.asarray(pvec, dtype=float)
    if pvec.ndim != 2 or pvec.shape[1] != 3 or not np.all(np.isfinite(pvec)):
        raise ValueError("pvec must be finite with shape (N, 3)")
    return pvec


def _lanes(value, n, name):
    try:
        value = np.broadcast_to(np.asarray(value, dtype=float), (n,))
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or shape (N,)") from exc
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def decay_pions(rng, pvec):
    """Return (muon_pvec, neutrino_pvec), each (N, 3), from pi+- -> mu+- nu.

    Isotropic two-body emission in the pion rest frame with exact MPI/MMU and
    a massless neutrino; no muon polarisation is tracked. The Lorentz boost is
    nonsingular for a pion at rest. The driver maps +211 -> (-13, +14) and
    -211 -> (+13, -14); these are mu+/nu_mu and mu-/anti-nu_mu respectively.
    Input pvec is not modified; an empty batch is supported.
    """
    pvec = _momenta(pvec)
    costh = rng.uniform(-1.0, 1.0, len(pvec))
    phi = rng.uniform(0.0, 2 * np.pi, len(pvec))
    sinth = np.sqrt(np.maximum(1 - costh * costh, 0.0))
    direction = np.column_stack((sinth * np.cos(phi), sinth * np.sin(phi), costh))
    pstar = (MPI - MMU) * (MPI + MMU) / (2 * MPI)
    rest_mu = pstar * direction
    emu = np.hypot(pstar, MMU)
    epi = np.hypot(np.linalg.norm(pvec, axis=1), MPI)
    projection = np.sum(pvec * rest_mu, axis=1) / (epi + MPI)
    mu = rest_mu + ((emu + projection) / MPI)[:, None] * pvec
    nu = -rest_mu + ((pstar - projection) / MPI)[:, None] * pvec
    return mu, nu


def material_step(rng, mat, pvec, pdg, length_mm, thickness_x0_before=0):
    """One thin material step -> (pvec_after, fraction_travelled, stopped).

    mat is a positive-density core Material; pvec has shape (N, 3). pdg,
    length_mm and accumulated thickness_x0_before are scalar or (N,). Outputs
    have shapes (N, 3), (N,), (N,); stopped is boolean. Inputs are not mutated.

    Core restricted dE/dx and Urban fluctuations use per-lane masses with a
    1e9 MeV delta cut (effectively unrestricted here). An unphysical sampled
    loss >= kinetic energy is replaced by the mean, as in core slab_loss.
    The loss rate is frozen at entry: the CALLER must limit expected loss to
    a small fraction (e.g. 5%) of kinetic energy and check step convergence.
    This is not a thick-slab/range integrator, especially near a Bragg peak.

    Tracks at/below 0.2 MeV stop immediately with fraction zero. A step that
    reaches this threshold stops at fraction (T - 0.2)/loss, linearly estimated
    with the frozen rate; remaining kinetic energy is discarded, not tracked.
    Stops return zero momentum. Zero-length moving tracks remain unchanged.

    Survivors get a normalized Gaussian transverse-direction kick using core
    cumulative Highland variance at ENTRY momentum (small-angle model, no
    displacement or Rutherford tails). Add length_mm * fraction / X0 to the
    thickness history externally; X0 = eloss.radiation_length_mm(mat). These
    approximations do not describe pion nuclear interactions or capture.
    """
    pvec = _momenta(pvec)
    n = len(pvec)
    mass = np.broadcast_to(mass_mev(pdg), (n,))
    length = _lanes(length_mm, n, "length_mm")
    previous = _lanes(thickness_x0_before, n, "thickness_x0_before")
    if not isinstance(mat, materials.Material) or not np.isfinite(mat.rho) or mat.rho <= 0:
        raise ValueError("mat must be a positive-density mu2efastsim.materials.Material")
    p = np.linalg.norm(pvec, axis=1)
    energy = np.hypot(p, mass)
    kinetic = p * (p / (energy + mass))
    stopped = kinetic <= TK_STOP_MEV
    fraction = np.where(stopped, 0.0, 1.0)
    after = pvec.copy()
    after[stopped] = 0.0
    active = (length > 0) & ~stopped
    if not np.any(active):
        return after, fraction, stopped

    pa, ma, tk, step = p[active], mass[active], kinetic[active], length[active]
    mean = eloss.dedx_restricted(mat, pa, _TCUT_MEV, mass=ma) * step
    loss = eloss.sample_loss(rng, mat, pa, step, _TCUT_MEV, mass=ma)
    loss = np.maximum(np.where(loss >= tk, mean, loss), 0.0)
    remaining = np.maximum(tk - loss, 0.0)
    stop = remaining <= TK_STOP_MEV
    frac = np.ones_like(tk)
    np.divide(tk - TK_STOP_MEV, loss, out=frac, where=stop & (loss > 0))
    fraction[active] = np.clip(frac, 0.0, 1.0)
    stopped[active] = stop
    new_p = np.sqrt(remaining * (remaining + 2 * ma))
    after[active] = pvec[active] * (new_p / pa)[:, None]
    after[stopped] = 0.0

    moving = active & ~stopped
    if np.any(moving):
        direction = pvec[moving] / p[moving, None]
        x0 = mat.x0_mm if hasattr(mat, "x0_mm") else eloss.radiation_length_mm(mat)
        variance = transport.highland_var_increment(
            p[moving], previous[moving], previous[moving] + length[moving] / x0, mass=mass[moving])
        kick = rng.standard_normal(direction.shape)
        kick -= np.sum(kick * direction, axis=1)[:, None] * direction
        direction = direction + np.sqrt(variance)[:, None] * kick
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        after[moving] = direction * np.linalg.norm(after[moving], axis=1)[:, None]
    return after, fraction, stopped
