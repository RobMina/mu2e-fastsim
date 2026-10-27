"""Per-muon G4 reference table from the ReadVirtualDetector tree of beam/fcl/g4ref_mubeam_ntpart.fcl.

Rows with creation code mu2ePrimary (56) are the DS-stage muons: their start is the resampled state at the DS entrance,
their end is where Geant4 stopped, captured, decayed or killed them. A stop at rest has |p_end| = 0.
"""
import numpy as np
import uproot

from . import geometry

MU2E_PRIMARY = 56
BRANCHES = ["run", "subrun", "evt", "trk", "pdg", "x", "y", "z", "px", "py", "pz", "time", "code",
            "xstop", "ystop", "zstop", "pxstop", "pystop", "pzstop", "pstop", "tstop", "codestop"]


def read(paths, geom):
    """Dict of numpy arrays, one entry per DS-stage muon, with derived fields at_rest, foil (-1 = not in a foil)."""
    cols = {b: [] for b in BRANCHES}
    for p in paths:
        with uproot.open(p) as f:
            a = f["mutree/ntpart"].arrays(BRANCHES, library="np")
        for b in BRANCHES:
            cols[b].append(a[b])
    a = {b: np.concatenate(v) for b, v in cols.items()}
    sel = a["code"] == MU2E_PRIMARY
    a = {b: v[sel] for b, v in a.items()}
    a["q"] = np.where(a["pdg"] == 13, -1.0, 1.0)
    a["at_rest"] = a["pstop"] < 1e-6
    a["foil"] = np.where(a["at_rest"], geometry.foil_index(geom, a["xstop"], a["ystop"], a["zstop"]), -1)
    return a


def save(a, path):
    np.savez_compressed(path, **a)


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def write_gunfile(g, path, charge=None):
    """G4beamline-format particle list (x y z px py pz t PDGid EventID TrackID ParentID Weight, Mu2e frame) of the
    reference starts, one muon per line, for EventGenerator fromG4BLFile with g4beamlineOrigin {0,0,0}.

    charge None keeps each muon's charge; +1 / -1 sets every muon to mu+ / mu-. Returns the number of lines."""
    if charge not in (None, 1, -1):
        raise ValueError("charge must be None, +1 or -1")
    pdg = np.asarray(g["pdg"], int) if charge is None else np.full(len(g["pdg"]), 13 if charge < 0 else -13)
    n = len(pdg)
    cols = [g[k] for k in ("x", "y", "z", "px", "py", "pz", "time")]
    with open(path, "w") as f:
        for i in range(n):
            f.write(" ".join(f"{float(c[i]):.17g}" for c in cols) + f" {pdg[i]} {i + 1} 1 0 1\n")
    return n
