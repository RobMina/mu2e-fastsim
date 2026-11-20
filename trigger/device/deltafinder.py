"""DeltaFinder (TrigFlagPH) on the device: port of the stock DeltaFinder (Offline v13_41_01, CalPatRec), with its
arithmetic and orders.

DeltaFinder flags the panel hits of low-energy (delta) electrons as Background and clears EnergySelection on the
hits of proton candidates (EnergySelection is set on every other hit when flagProtonHits is on). It writes every hit
(writeFilteredComboHits 0), so only the flags change; TZClusterFinder, CalTimePeakFinder, RobustHelixFinder and
CalHelixFinder read them, TimeClusterFinder (TestFlag false) does not.

The geometry is the nominal tracker DeltaFinder reads (GeomHandle<Tracker>, not the aligned tracker the hits are made
with): straw 0 and 1 of each panel, shipped in `data/tracker_nominal_panels.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .helix import TRACKER_GEOM

DATA = Path(__file__).resolve().parent / "data"
NST, NFACE, NPF = 18, 4, 3                 # kNStations, kNFaces, kNPanelsPerFace
BKG_BIT = 6                                # StrawHitFlag::bkg
ESEL_BIT = 1                               # StrawHitFlag::energysel
FINDER_KEYS = ("timeBin", "maxDeltaEDep", "maxSeedEDep", "minProtonSeedEDep", "minProtonHitEDep", "minNSeeds",
               "minDeltaNHits", "maxEleHitEnergy", "maximumTime", "maxHitSeedDt", "maxChi2Seed", "maxChi2Par",
               "maxChi2Perp", "maxChi2SeedDelta", "rCore", "maxGap", "sigmaR", "maxDriftTime", "maxSeedDt",
               "maxHitDt", "maxDtDs", "maxDtDc", "flagProtonHits", "mergePC", "pickupProtonHits")


def settings(m):
    """DeltaFinder module table -> finder parameters. The port implements the trigger configuration: all hits
    written, no hit mask tested."""
    fp = m["finderParameters"]
    if int(m.get("writeFilteredComboHits", 0)) != 0 or int(m.get("writeStrawHits", 0)) != 0:
        raise ValueError("device DeltaFinder writes all combo hits (writeFilteredComboHits 0, writeStrawHits 0)")
    if bool(fp.get("testHitMask", False)) or list(m.get("bkgHitMask", [])) or list(fp.get("bkgHitMask", [])):
        raise ValueError("device DeltaFinder implements testHitMask false and an empty bkgHitMask")
    if int(fp.get("mergePC", 0)) != 0:
        raise ValueError("device DeltaFinder implements mergePC 0")
    out = {}
    for k in FINDER_KEYS:
        v = fp[k]
        out[k] = int(v) if k in ("minNSeeds", "minDeltaNHits", "maxGap", "flagProtonHits", "mergePC",
                                 "pickupProtonHits") else float(v)
    return out


def order_id(station, plane, panel):
    """ChannelID::orderID: (station, plane in station, panel) -> (station, z-ordered face, panel in face)."""
    face = panel % 2
    if station % 2 == 0:
        of = 1 - face if plane == 0 else face + 2
    else:
        of = face if plane == 0 else 3 - face
    return station, of, panel // 2


def plane_z(plane, geom=TRACKER_GEOM):
    """Plane::origin().z() of the nominal tracker (TrackerMaker: first plane z0 plus the plane's offset)."""
    z0 = -(float(geom["nstations"] - 1) / 2.0 * geom["planeSpacing"])
    st, k = plane // 2, plane % 2
    dz = st * geom["planeSpacing"] + (geom["planeHalfSeparation"] if k else -geom["planeHalfSeparation"])
    return z0 + dz


def geometry(path=DATA / "tracker_nominal_panels.json", cos="float"):
    """Data_t::InitGeometry: per (station, z-ordered face, panel) the wire direction (wx, wy, double), the outward
    normal (nx, ny: cos/sin of the float straw-0 azimuth; `cos` picks the float or double overload), the float z;
    the face z (double, from panel 0) and the float station z."""
    pan = np.asarray(json.loads(Path(path).read_text())["panels"], np.float64).reshape(36, 6, 6)
    wx, wy, nx, ny = (np.zeros((NST, NFACE, NPF)) for _ in range(4))
    z = np.zeros((NST, NFACE, NPF), np.float32)
    fz = np.zeros((NST, NFACE))
    for ipl in range(36):
        for ipn in range(6):
            x0, y0, z0, z1, dx, dy = pan[ipl, ipn]
            os_, of, op = order_id(ipl // 2, ipl % 2, ipn)
            wx[os_, of, op], wy[os_, of, op] = dx, dy
            phi = np.float32(math.atan2(y0, x0))                 # CLHEP Hep3Vector::phi() -> float
            if cos == "float":
                nx[os_, of, op] = float(np.float32(math.cos(float(phi))))
                ny[os_, of, op] = float(np.float32(math.sin(float(phi))))
            else:
                nx[os_, of, op], ny[os_, of, op] = math.cos(float(phi)), math.sin(float(phi))
            z[os_, of, op] = np.float32((z0 + z1) / 2.)
            if op == 0:
                fz[os_, of] = float(z[os_, of, op])
    sz = np.array([np.float32((plane_z(2 * s) + plane_z(2 * s + 1)) / 2) for s in range(NST)], np.float32)
    return {"wx": wx, "wy": wy, "nx": nx, "ny": ny, "z": z, "fz": fz, "stationZ": sz}
