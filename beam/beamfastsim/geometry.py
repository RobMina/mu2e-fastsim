"""Passive geometry the beam muons see between the DS entrance and the IPA, read from the Offline geometry files.

Mu2e frame, mm. The DS axis is at x = -mu2e.solenoidOffset. Materials are mu2efastsim Material objects with the
Offline (ConstructMaterials.cc) composition and density.
"""
import os
from dataclasses import dataclass

import numpy as np

from mu2efastsim import materials as M

from . import simpleconfig

# Offline ConstructMaterials.cc: StoppingTarget_Al is G4_Al; IPAPolyethylene 0.954 g/cm3 (H 0.11, C 0.89);
# Polyethylene094 0.94 g/cm3 (H 0.14, C 0.86). I values: G4 for Al, ICRU polyethylene for the PE variants.
STOPPING_TARGET_AL = M.ALUMINIUM
IPA_PE = M.Material("IPAPolyethylene", 0.954, 57.4, {"C": 0.89, "H": 0.11})
PE094 = M.Material("Polyethylene094", 0.94, 57.4, {"C": 0.86, "H": 0.14})


@dataclass(frozen=True)
class BeamGeometry:
    axis_x: float           # DS axis x [mm]
    foil_z: np.ndarray      # foil centre z [mm], (nfoil,)
    foil_rin: np.ndarray    # hole radius [mm]
    foil_rout: np.ndarray   # outer radius [mm]
    foil_ht: np.ndarray     # half thickness [mm]
    tsda_z: float           # TSdA centre z, half length, inner/outer radius
    tsda_hl: float
    tsda_rin: float
    tsda_rout: float
    ipa_r: float            # inner proton absorber: radius, wall thickness, z range
    ipa_t: float
    ipa_z0: float
    ipa_z1: float
    opa_rin: float          # outer proton absorber inner radius (taken as a perfect absorber) and z range
    opa_z0: float
    opa_z1: float


def from_offline(geom_file, offline_root):
    """BeamGeometry from an Offline geometry file (e.g. Offline/Mu2eG4/geom/geom_common.txt) under offline_root."""
    path = geom_file if os.path.isabs(geom_file) else f"{offline_root}/{geom_file}"
    c = simpleconfig.read(path, offline_root)
    axis_x = -c["mu2e.solenoidOffset"]
    rout = np.array(c["stoppingTarget.radii"], float)
    n = len(rout)
    ht = np.array(c["stoppingTarget.halfThicknesses"], float)
    ht = np.concatenate([ht, np.full(n - len(ht), ht[-1])])
    zvar = np.array(c.get("stoppingTarget.zVars", [0.0] * n), float)
    zvar = np.concatenate([zvar, np.zeros(n - len(zvar))])
    dz, z0 = c["stoppingTarget.deltaZ"], c["stoppingTarget.z0InMu2e"]
    offset = z0 if n % 2 == 1 else z0 + dz / 2      # StoppingTargetMaker
    foil_z = offset + (np.arange(n) - n // 2) * dz + zvar
    rin = np.array(c["stoppingTarget.holeRadii"], float) if "stoppingTarget.holeRadii" in c else \
        np.full(n, c.get("stoppingTarget.holeRadius", 0.0))
    t_end = foil_z.max() + ht.max()
    ipa_z0 = t_end + c["protonabsorber.distFromTargetEnd"]
    return BeamGeometry(axis_x, foil_z, rin, rout, ht,
                        c["tsda.z0"], c["tsda.halfLength4"], c["tsda.rin"], c["tsda.r4"],
                        c["protonabsorber.OutRadius0"] - c["protonabsorber.thickness"],
                        c["protonabsorber.thickness"], ipa_z0, ipa_z0 + 2 * c["protonabsorber.halfLength"],
                        c["protonabsorber.outerPAInnerRadius0"],
                        c["protonabsorber.outerPAZCenter"] - c["protonabsorber.outerPAHalfLength"],
                        c["protonabsorber.outerPAZCenter"] + c["protonabsorber.outerPAHalfLength"])


def foil_index(geom, x, y, z, tol=1e-3):
    """Index of the foil containing each point, -1 if none (classifies G4 stop positions)."""
    r = np.hypot(x - geom.axis_x, y)
    k = np.argmin(np.abs(z[:, None] - geom.foil_z[None, :]), axis=1)
    ok = (np.abs(z - geom.foil_z[k]) <= geom.foil_ht[k] + tol) & (r >= geom.foil_rin[k] - tol) & \
         (r <= geom.foil_rout[k] + tol)
    return np.where(ok, k, -1)
