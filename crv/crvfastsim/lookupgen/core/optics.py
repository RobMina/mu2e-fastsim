"""Optical-physics building blocks of the strip generator, with Geant4 10.7 semantics (NumPy; the device tables are
built from these).

- ``G4Vector`` is a G4MaterialPropertyVector (a G4PhysicsOrderedFreeVector filled by InsertValues): energies are sorted
  on insertion, a point going before an existing equal energy. ``value(E)`` is G4PhysicsVector::Value with a fresh
  index (edges clamp, bin std::lower_bound - 1, linear interpolation with the fraction clamped to [0, 1]);
  ``get_energy(v)`` is G4PhysicsOrderedFreeVector::GetEnergy (the inverse of a cumulative emission integral).
- ``group_velocity`` is G4MaterialPropertiesTable::CalculateGROUPVEL (optical photons move at the group velocity of
  the volume they traverse).
- ``facet_alpha_icdf`` is the facet-tilt distribution of G4OpBoundaryProcess::GetFacetNormal (unified model).
- ``cerenkov_photons_per_mm`` / ``cerenkov_map`` are G4CerenkovNew::GetAverageNumberOfPhotons for a unit charge.
- ``Model`` is one optical model (materials, surfaces, end readout); models are registered by name (``register``,
  ``get``). ``materials`` registers ``current``; experiment adapters register their own.

Standard library and NumPy only.
"""
import bisect
from dataclasses import dataclass

import numpy as np

C_LIGHT = 299.792458  # mm/ns


class G4Vector:
    """A G4PhysicsOrderedFreeVector: (energy, value) pairs inserted in order with std::lower_bound."""

    def __init__(self, energies, values):
        e, v = [], []
        for x, y in zip(energies, values):
            i = bisect.bisect_left(e, float(x))
            e.insert(i, float(x))
            v.insert(i, float(y))
        self.e = np.array(e, np.float64)
        self.v = np.array(v, np.float64)

    def __len__(self):
        return len(self.e)

    def bin_of(self, x):
        """G4PhysicsVector::FindBin(x, 0) for x strictly inside (edgeMin, edgeMax)."""
        e, n = self.e, len(self.e)
        x = np.asarray(x, np.float64)
        loc = np.minimum(np.searchsorted(e, x, side="left") - 1, n - 2)
        loc = np.where((x > e[1]) | (x < e[0]), loc, 0)   # idx 0 is kept when e[0] <= x <= e[1]
        loc = np.where(x < e[1], 0, loc)
        return np.where(x >= e[n - 2], n - 2, loc)

    def value(self, x):
        e, v = self.e, self.v
        x = np.asarray(x, np.float64)
        i = np.clip(self.bin_of(x), 0, len(e) - 2)
        dl = e[i + 1] - e[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(dl > 0, np.clip((x - e[i]) / np.where(dl > 0, dl, 1.0), 0.0, 1.0), 1.0)
        y = v[i] + b * (v[i + 1] - v[i])
        y = np.where(x <= e[0], v[0], y)
        return np.where(x >= e[-1], v[-1], y)

    def get_energy(self, a):
        """G4PhysicsOrderedFreeVector::GetEnergy: the inverse of a non-decreasing vector."""
        e, v, n = self.e, self.v, len(self.e)
        a = np.asarray(a, np.float64)
        i = np.minimum(np.searchsorted(v, a, side="left") - 1, n - 2)
        i = np.clip(i, 0, n - 2)
        d = v[i + 1] - v[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(d > 0, e[i] + (a - v[i]) * (e[i + 1] - e[i]) / np.where(d > 0, d, 1.0), e[i])
        r = np.where(a <= v[0], e[0], r)
        return np.where(a >= v[-1], e[-1], r)


def integral(vec):
    """Cumulative trapezoid integral as G4Scintillation / G4OpWLS build it (the emission integral)."""
    c = [0.0]
    for i in range(1, len(vec)):
        c.append(c[-1] + (vec.e[i] - vec.e[i - 1]) * (0.5 * (vec.v[i - 1] + vec.v[i])))
    out = G4Vector([], [])
    out.e, out.v = vec.e.copy(), np.array(c)
    return out


def group_velocity(rindex):
    """G4MaterialPropertiesTable::CalculateGROUPVEL (Geant4 10.7) of a RINDEX vector, in mm/ns."""
    E, n = rindex.e, rindex.v
    es, vs = [], []
    E0, n0, E1, n1 = E[0], n[0], E[1], n[1]
    vg = C_LIGHT / (n0 + (n1 - n0) / np.log(E1 / E0))
    if vg < 0 or vg > C_LIGHT / n0:
        vg = C_LIGHT / n0
    es.append(E0), vs.append(vg)
    for i in range(2, len(E)):
        vg = C_LIGHT / (0.5 * (n0 + n1) + (n1 - n0) / np.log(E1 / E0))
        if vg < 0 or vg > C_LIGHT / (0.5 * (n0 + n1)):
            vg = C_LIGHT / (0.5 * (n0 + n1))
        es.append(0.5 * (E0 + E1)), vs.append(vg)
        E0, n0, E1, n1 = E1, n1, E[i], n[i]
    vg = C_LIGHT / (n1 + (n1 - n0) / np.log(E1 / E0))
    if vg < 0 or vg > C_LIGHT / n1:
        vg = C_LIGHT / n1
    es.append(E1), vs.append(vg)
    return G4Vector(es, vs)


@dataclass(frozen=True)
class Model:
    """One optical model of a scintillator strip with wavelength-shifting fibers: material tables, surfaces and the
    end readout (lengths in mm, times in ns, energies in eV). ``materials.model`` builds one; ``register`` makes it
    available to the transport by name."""
    name: str
    rindex_ps: G4Vector
    rindex_pmma: G4Vector
    rindex_fpe: G4Vector
    rindex_epoxy: G4Vector
    abs_y11: G4Vector
    qy_y11: G4Vector
    em_y11: G4Vector
    tau_y11: float
    em_ppo: G4Vector
    qy_psppo: G4Vector
    abs_psppo: G4Vector
    tau_psppo: float
    em_popop: G4Vector
    qy_popop: G4Vector
    abs_popop: G4Vector
    tau_popop: float
    scint_decay: float
    scint_rise: float
    refl_tio2: G4Vector
    eff_mppc: G4Vector
    refl_mppc: float
    refl_reflector: float
    fiber_end: str            # "contact": SiPM / reflector cylinders on the fiber end; "gap": 1 um air, epoxy window
    rough_fiber_air: bool     # fiber/air interfaces are unified ground (sigma 0.02) surfaces (current model)

    @property
    def vg_ps(self):
        return group_velocity(self.rindex_ps)

    @property
    def vg_pmma(self):
        return group_velocity(self.rindex_pmma)

    @property
    def vg_fpe(self):
        return group_velocity(self.rindex_fpe)

    @property
    def vg_epoxy(self):
        return group_velocity(self.rindex_epoxy)

RINDEX_AIR = G4Vector((2.0, 15.75), (1.0, 1.0))
VG_AIR = group_velocity(RINDEX_AIR)
SIGMA_TIO2 = 1.5        # rad, unified/ground dielectric_metal, specular lobe 1
SIGMA_FIBER_END = 0.02  # rad, unified/ground dielectric_dielectric between the fiber layers and the air (current)

DEFAULT_MODEL = "current"
MODELS = {}


def register(model):
    """Make ``model`` available by its name (a second model of the same name must be identical)."""
    old = MODELS.get(model.name)
    if old is not None and old is not model and old != model:
        raise ValueError(f"optical model {model.name!r} is already registered with other content")
    MODELS[model.name] = model
    return model


def get(name):
    """The registered model ``name``."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown optical model {name!r}; registered: {sorted(MODELS)} (experiment adapters "
                         "register their own)") from None


def facet_alpha_icdf(sigma, n=8192):
    """Inverse CDF of the facet tilt alpha drawn by G4OpBoundaryProcess::GetFacetNormal (unified model).

    Geant4 draws alpha ~ N(0, sigma) and keeps it when U * f_max <= sin(alpha) and alpha < pi/2, with
    f_max = min(1, 4 sigma): the kept density is gauss(alpha) * min(1, sin(alpha) / f_max) on (0, pi/2). Returns the
    alpha values at n + 1 equally spaced CDF points (linear interpolation in between)."""
    fmax = min(1.0, 4.0 * sigma)
    a = np.linspace(0.0, min(np.pi / 2, 12.0 * sigma), 200001)
    pdf = np.exp(-0.5 * (a / sigma) ** 2) * np.minimum(1.0, np.sin(a) / fmax)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(a))])
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n + 1), cdf, a)


# --- Cerenkov photons per mm (G4CerenkovNew) ---------------------------------------
def _energy_intervals(rindex, beta_inv):
    """G4CerenkovNew::GetEnergyIntervals: the energy ranges where RINDEX > 1/beta."""
    E, n = rindex.e, rindex.v
    out, active, start = [], False, None
    if n[0] > beta_inv:
        active, start = True, E[0]
    for i in range(1, len(E)):
        f = (beta_inv - n[i - 1]) / (n[i] - n[i - 1]) if n[i] != n[i - 1] else 0.0
        if active and n[i] <= beta_inv:
            active = False
            out.append((start, E[i - 1] + (E[i] - E[i - 1]) * f))
        elif not active and n[i] > beta_inv:
            active, start = True, E[i - 1] + (E[i] - E[i - 1]) * f
    if active:
        out.append((start, E[-1]))
    return out


def cerenkov_photons_per_mm(beta, rindex):
    """G4CerenkovNew::GetAverageNumberOfPhotons for a unit charge (Geant4 internal units: MeV, mm)."""
    eV = 1.0e-6
    if beta <= 0:
        return 0.0
    bi = 1.0 / beta
    ev = G4Vector(rindex.e * eV, rindex.v)
    if ev.v[0] <= 1.0:
        return 0.0
    cai = [0.0]
    for i in range(1, len(ev)):
        cai.append(cai[-1] + (ev.e[i] - ev.e[i - 1]) * (0.5 * (1.0 / (ev.v[i - 1] * ev.v[i - 1]) + 1.0 / (ev.v[i] * ev.v[i]))))
    CAI = G4Vector([], [])
    CAI.e, CAI.v = ev.e.copy(), np.array(cai)
    nmin, nmax = ev.v.min(), ev.v.max()
    if nmax < bi:
        dp = ge = 0.0
    elif nmin > bi:
        dp, ge = ev.e[-1] - ev.e[0], CAI.v[-1]
    else:
        dp = ge = 0.0
        for a, b in _energy_intervals(ev, bi):
            dp += b - a
            ge += float(CAI.value(b)) - float(CAI.value(a))
    n = 369.81 / (eV * 10.0) * (dp - ge * bi * bi)
    return max(n, 0.0)


def cerenkov_map(rindex):
    """The (beta, photons/mm) map of the table header: beta = 1, 1/1.01, ... until no photons are made."""
    out, beta = [], 1.0
    while True:
        n = cerenkov_photons_per_mm(beta, rindex)
        out.append((beta, n))
        if n == 0:
            break
        beta /= 1.01
    return np.array(sorted(out))
