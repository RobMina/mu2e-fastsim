"""Flash-source geometry for the fixed known aluminium/material and DS-source setup.

This scoped reader supplies CEGeometry without changing the shared beam or CE
loaders. It checks represented coaxial, untilted foil arrays, not the complete
Offline configuration or SimpleConfig syntax. It is not parser certification or
an overlay allowlist: global material/source choices and omitted supports remain
assumptions of the fixed setup. Absorber shapes retain the CE approximations.
"""
import os

import numpy as np

from . import cechain as C
from . import geometry as G
from . import simpleconfig


def _vector(c, key, default=None, *, empty=False):
    a = np.asarray(c[key] if key in c else default, dtype=float)
    if a.ndim != 1 or (not empty and a.size == 0) or not np.all(np.isfinite(a)):
        raise ValueError(f"{key} must be a finite {'nonempty ' if not empty else ''}vector")
    return a


def _repeat(c, key, n, default=None):
    a = _vector(c, key, default)
    if a.size > n:
        raise ValueError(f"{key} has more entries than stoppingTarget.radii")
    return np.pad(a, (0, n - a.size), mode="edge")


def _scalar(c, key, default=None):
    v = np.asarray(c[key] if key in c else default, dtype=float)
    if v.ndim != 0 or not np.isfinite(v):
        raise ValueError(f"{key} must be a finite scalar")
    return float(v)


def _foils(c):
    if "stoppingTarget.materials" in c:
        raise ValueError("stoppingTarget.materials is rejected by Offline StoppingTargetMaker")
    if c.get("stoppingTarget.overrideMaterial", "") not in ("", "StoppingTarget_Al", "G4_Al"):
        raise ValueError("Unsupported stoppingTarget.overrideMaterial: the flash geometry assumes aluminium")
    if c.get("hasTarget", True) is not True:
        raise ValueError("Unsupported target type: hasTarget=false builds a dummy target")
    rout = _vector(c, "stoppingTarget.radii")
    n = rout.size
    ht = _repeat(c, "stoppingTarget.halfThicknesses", n)
    zvar = _repeat(c, "stoppingTarget.zVars", n, [0.0])
    for name in ("xVars", "yVars", "xCos", "yCos"):
        key = "stoppingTarget." + name
        if np.any(_repeat(c, key, n, [0.0]) != 0):
            raise ValueError(f"Unsupported {key}: the flash geometry requires coaxial, untilted foils")
    dz = _scalar(c, "stoppingTarget.deltaZ")
    z0 = _scalar(c, "stoppingTarget.z0InMu2e")
    if dz <= 0:
        raise ValueError("stoppingTarget.deltaZ must be positive")
    offset = z0 if n % 2 else z0 + dz / 2
    z = offset + (np.arange(n) - n // 2) * dz + zvar
    rin = _vector(c, "stoppingTarget.holeRadii", [], empty=True)
    if rin.size == 0:
        rin = np.full(n, _scalar(c, "stoppingTarget.holeRadius", 0.0))
    elif rin.size != n:
        raise ValueError("stoppingTarget.holeRadii must match stoppingTarget.radii length")
    if np.any(rin < 0) or np.any(rout <= rin) or np.any(ht <= 0):
        raise ValueError("foil radii require 0 <= rin < rout and positive half thickness")
    if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("foil centres must be finite and strictly increasing")
    if np.any(np.diff(z) <= ht[:-1] + ht[1:]):
        raise ValueError("foil thicknesses must not overlap in z")
    if np.any(z - ht < z[0] - 5.) or np.any(z + ht > z[-1] + 5.):
        raise ValueError("foil axial extent exceeds Offline target mother clearance")
    if not c.get("stoppingTarget.foilTarget_supportStructure_endAtOPA", False):
        mother_radius = 249.999
    else:
        r0 = _scalar(c, "protonabsorber.outerPAInnerRadius0", 436.)
        r1 = _scalar(c, "protonabsorber.outerPAInnerRadius1", 720.)
        centre = _scalar(c, "protonabsorber.outerPAZCenter", 6250.)
        half = _scalar(c, "protonabsorber.outerPAHalfLength", 2250.)
        wire = _scalar(c, "stoppingTarget.foilTarget_supportStructure_radius")
        if half <= 0 or wire < 0:
            raise ValueError("invalid OPA half length or target support-wire radius")
        side = 1 if r0 <= r1 else -1
        closest = np.clip(z0 - side * (n - 1) * dz / 2, centre - half, centre + half)
        mother_radius = r0 + (r1 - r0) * (closest - side * wire - (centre - half)) / (2 * half) - .011
    if np.any(rout > mother_radius):
        raise ValueError("foil radius exceeds supported Offline target mother cylinder")
    return z, rin, rout, ht


def from_offline(geom_file, offline_root, z_plane=8539.9, source_files=None):
    """Read the represented flash geometry; paths/includes follow SimpleConfig.

    StoppingTargetMaker repeats final array entries and encloses first/last foil
    centres with 5 mm clearance. MECOStyleProtonAbsorberMaker adds the VD thickness
    before the relative IPA distance, unless zStartInMu2e explicitly pins the IPA.
    """
    path = os.path.join(offline_root, geom_file)
    c = simpleconfig.read(path, offline_root, source_files=source_files)
    z, rin, rout, ht = _foils(c)
    distance = _scalar(c, "protonabsorber.distFromTargetEnd", 0.0)
    if not 0 <= distance <= 2500:
        raise ValueError("protonabsorber.distFromTargetEnd outside [0, 2500]; Offline would reset it")
    if "protonabsorber.zStartInMu2e" in c:
        ipa_z0 = _scalar(c, "protonabsorber.zStartInMu2e")
    else:
        vd = _scalar(c, "vd.halfLength") if c.get("hasVirtualDetector", True) else 0.0
        ipa_z0 = z[-1] + 5.0 + 2 * vd + distance
    tsda_hl = c["tsda.halfLength4"] if c.get("hasTSdA", True) else 0.0
    beam = G.BeamGeometry(
        -c["mu2e.solenoidOffset"], z, rin, rout, ht,
        c["tsda.z0"], tsda_hl, c["tsda.rin"], c["tsda.r4"],
        c["protonabsorber.OutRadius0"] - c["protonabsorber.thickness"],
        c["protonabsorber.thickness"], ipa_z0, ipa_z0 + 2 * c["protonabsorber.halfLength"],
        c["protonabsorber.outerPAInnerRadius0"],
        c["protonabsorber.outerPAZCenter"] - c["protonabsorber.outerPAHalfLength"],
        c["protonabsorber.outerPAZCenter"] + c["protonabsorber.outerPAHalfLength"])
    # BeamlineMaker places TS5 from the torus radius, independently of TSdA.
    ts5_start = _scalar(c, "ts.rTorus")
    ts5_end = ts5_start + 2 * c["ts.ts5.halfLength"]
    c5 = 0.5 * (ts5_start + ts5_end) + c["ts.coll5.sOffset"]
    rings = c.get("protonabsorber.ipa.buildSupports", False) and c.get("protonabsorber.ipa.nEndRings", 0) > 0
    r_ts = c["ts.ts5.rVac"] if "ts.ts5.rVac" in c else c["ts.ts5in.rOut"]
    return C.CEGeometry(
        beam, c["protonabsorber.outerPAInnerRadius0"], c["protonabsorber.outerPAInnerRadius1"],
        c["protonabsorber.outerPAThickness"], z_plane, z_up=ts5_start, z_ds=ts5_end,
        r_ts=r_ts, coll5_z0=c5 - c["ts.coll5.halfLength"],
        coll5_z1=c5 + c["ts.coll5.halfLength"], coll5_rin=c["ts.coll5.Radius1"],
        ring_t=c.get("protonabsorber.ipa.endRingRadialLength", 0.0) if rings else 0.0,
        ring_hl=c.get("protonabsorber.ipa.endRingHalfLength", 0.0))
