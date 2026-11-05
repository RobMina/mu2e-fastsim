"""Opt-in vectorized CPU charged-pion decay, conditional on pi -> mu nu only.

Units: momentum MeV/c, energy MeV, mass MeV/c**2, path mm, proper time ns.
Constants are PDG charged-pion/muon values: Particle Data Group, Review of
Particle Physics, Phys. Rev. D 110, 030001 (2024), charged-pion and muon listings
(https://pdg.lbl.gov/2024/). The speed of light is exact in these units.

A spin-zero pion has isotropic two-body rest-frame directions; the neutrino is
massless here. This is conditional on the dominant muonic branch, not a physical
branching-fraction sampler. Using it for every pion decay is a 100% muonic-mode
diagnostic: the rare pi -> e nu branch and other rare modes are ignored, not
assigned their physical branching fractions. There are no fitted scales.

The caller owns parent/daughter track identity, ancestry, proper-clock storage,
step subdivision and trajectory propagation. These functions do not update a
parent, allocate a track ID, or interpolate coordinates along a field step.
"""

from dataclasses import dataclass

import numpy as np


MPI = 139.57039
MMU = 105.6583755
C_MM_NS = 299.792458
TAU_PI_NS = 26.033
MAX_GAMMA = 1000.0

LIMITATIONS = (
    "Conditional pi -> mu nu branch only; treating every decay as this mode is a "
    "100% mode diagnostic, not an exact physical branching fraction. Massless "
    "neutrinos; rare pi -> e nu and other rare modes omitted. No muon polarization "
    "is tracked, although pion-decay muon polarization matters for subsequent "
    "muon decay. No muon decay, hadronic interactions, material effects, fields, "
    "geometry or full transport; no PS/TS physics-validity claim. The caller owns "
    "identity, ancestry, proper clocks and trajectory propagation."
)

_PSTAR = (MPI - MMU) * (MPI + MMU) / (2.0 * MPI)
_EMU_STAR = np.hypot(MMU, _PSTAR)


@dataclass(frozen=True)
class DecayProducts:
    """Daughter arrays: p3 has shape batch + (3,), E and PDG have shape batch.

    PDG signs: pi+ (211) -> mu+ (-13), nu_mu (14); charge conjugate for pi-.
    There are deliberately no parent IDs, weights or branching fractions here.
    """

    muon_p3: np.ndarray
    neutrino_p3: np.ndarray
    muon_E: np.ndarray
    neutrino_E: np.ndarray
    muon_pdg: np.ndarray
    neutrino_pdg: np.ndarray


def _finite(value, name):
    """Convert real array input to float64 and reject nonfinite entries."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real and finite")
    array = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be real and finite")
    return array


def _positive(value, name):
    """Validate a strictly positive finite real array."""
    array = _finite(value, name)
    if np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _nonnegative(value, name):
    """Validate a nonnegative finite real array."""
    array = _finite(value, name)
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _scaled_ratio(a, b, c, d):
    """Nonnegative a*b/(c*d), avoiding intermediate overflow for finite inputs."""
    a, b, c, d = np.broadcast_arrays(a, b, c, d)
    ma, ea = np.frexp(a)
    mb, eb = np.frexp(b)
    mc, ec = np.frexp(c)
    md, ed = np.frexp(d)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp((ma * mb) / (mc * md), ea + eb - ec - ed)
    if not np.all(np.isfinite(result)):
        raise ValueError("derived clock/distance is outside finite float64 range")
    return result


def sample_lifetime(rng, shape):
    """Draw pion proper lifetimes [ns], exponential with mean TAU_PI_NS.

    ``rng`` is a caller-owned numpy.random.Generator; ``shape`` follows NumPy's
    size convention (integer or tuple, including () and empty batches). This is
    a pion lifetime generator, not a generic-species or muon lifetime generator.
    No generator is created, reseeded, or shared globally.
    """
    return rng.exponential(TAU_PI_NS, size=shape)


def proper_time(path_mm, p_MeVc, mass=MPI):
    """Proper time [ns] for a constant-momentum path: path * mass / (p * c).

    Inputs broadcast. Path is finite and nonnegative; momentum magnitude and mass
    are finite and strictly positive. Zero momentum is rejected even for a zero
    path: a path parameter cannot advance a stopped particle's proper clock.
    Subdivide paths when momentum changes; this is not a material/field integral.
    """
    path = _nonnegative(path_mm, "path_mm")
    p = _positive(p_MeVc, "p_MeVc")
    m = _positive(mass, "mass")
    return _scaled_ratio(path, m, p, C_MM_NS)


def where_decay(pathstep, p, proper_remaining, mass=MPI):
    """Return (distance_mm, within_step) for a constant-momentum path segment.

    Distance is the physical path length p / mass * c * proper_remaining, not
    clipped to pathstep. The mask uses distance <= pathstep, including a decay
    at remaining proper time zero or exactly at the endpoint. All scalar inputs
    broadcast; paths and remaining times are finite/nonnegative, p and mass are
    finite/strictly positive. Stopped-particle time evolution needs a time-based
    caller. No position, direction or coordinate interpolation is performed.
    """
    step = _nonnegative(pathstep, "pathstep")
    momentum = _positive(p, "p")
    remaining = _nonnegative(proper_remaining, "proper_remaining")
    m = _positive(mass, "mass")
    step, momentum, remaining, m = np.broadcast_arrays(step, momentum, remaining, m)
    distance = _scaled_ratio(momentum, remaining, m, 1.0 / C_MM_NS)
    return distance, distance <= step


def _pion_inputs(pion_p3, pion_pdg):
    """Validate pion state and broadcast species with momentum's batch shape."""
    p3 = _finite(pion_p3, "pion_p3")
    if p3.ndim == 0 or p3.shape[-1] != 3:
        raise ValueError("pion_p3 must have shape (..., 3)")
    pdg = _finite(pion_pdg, "pion_pdg")
    if np.any((pdg != 211) & (pdg != -211)):
        raise ValueError("pion_pdg must be +211 or -211")
    with np.errstate(over="ignore"):
        gamma = np.hypot(np.hypot.reduce(p3, axis=-1), MPI) / MPI
    if np.any(~np.isfinite(gamma) | (gamma > MAX_GAMMA)):
        raise ValueError(f"supported pion boost is gamma <= {MAX_GAMMA}; extreme-boost cancellation is unvalidated")
    shape = np.broadcast_shapes(p3.shape[:-1], pdg.shape)
    return np.broadcast_to(p3, shape + (3,)), np.broadcast_to(pdg, shape)


def decay_muon(pion_p3, pion_pdg, cos_theta, phi):
    """Deterministic pi -> mu nu kinematics from supplied rest-frame angles.

    ``pion_p3`` has shape (..., 3); species and angles broadcast with its leading
    dimensions. cos_theta is in [-1, 1], phi is any finite angle in radians.
    Angles specify the muon direction relative to fixed Cartesian rest-frame
    axes (z polar axis), followed by a rotation-free boost along parent momentum.
    Isotropy requires uniform cos_theta and uniform phi modulo 2*pi, as in sample.
    Parent momentum zero is supported without constructing a parent unit vector.
    Gamma above 1000 is rejected: this beam-scale kernel does not certify extreme
    boost cancellation. Inputs are not mutated; no track identities are generated.
    """
    p3, pdg = _pion_inputs(pion_p3, pion_pdg)
    ct = _finite(cos_theta, "cos_theta")
    azimuth = _finite(phi, "phi")
    if np.any((ct < -1) | (ct > 1)):
        raise ValueError("cos_theta must be in [-1, 1]")
    shape = np.broadcast_shapes(p3.shape[:-1], ct.shape, azimuth.shape)
    p3 = np.broadcast_to(p3, shape + (3,))
    pdg, ct, azimuth = (np.broadcast_to(a, shape) for a in (pdg, ct, azimuth))
    st = np.sqrt((1.0 - ct) * (1.0 + ct))
    rest_p3 = _PSTAR * np.stack((st * np.cos(azimuth), st * np.sin(azimuth), ct), axis=-1)
    parent_p = np.hypot.reduce(p3, axis=-1)
    gamma = np.hypot(parent_p, MPI) / MPI
    u = p3 / MPI
    dot = np.sum(u * rest_p3, axis=-1)
    muon_p3 = rest_p3 + u * (_EMU_STAR + dot / (gamma + 1.0))[..., None]
    neutrino_p3 = -rest_p3 + u * (_PSTAR - dot / (gamma + 1.0))[..., None]
    muon_E = gamma * _EMU_STAR + dot
    neutrino_E = gamma * _PSTAR - dot
    positive = pdg == 211
    return DecayProducts(muon_p3, neutrino_p3, muon_E, neutrino_E,
                         np.where(positive, -13, 13), np.where(positive, 14, -14))


def sample(rng, pion_p3, pion_pdg):
    """Sample conditional daughters isotropically, without sampling a lifetime.

    After validating the pion state, consume one ``rng.random(batch + (2,))``
    draw: each C-order lane takes cos_theta from draw 0 and phi from draw 1.
    Lifetime sampling is separate so the caller can retain one proper clock
    across steps. No global RNG is used and malformed pion input consumes none.
    """
    p3, pdg = _pion_inputs(pion_p3, pion_pdg)
    draws = rng.random(p3.shape[:-1] + (2,))
    return decay_muon(p3, pdg, 2.0 * draws[..., 0] - 1.0, 2.0 * np.pi * draws[..., 1])
