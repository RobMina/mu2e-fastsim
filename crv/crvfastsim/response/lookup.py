"""Reader for the CRV photon lookup tables (format version 6, Offline CRVResponse/src/MakeCrvPhotons.cc).

Binary layout (native little-endian, as written by the C++ structs):
  LookupConstants (raw struct: 3 int32, 4 bytes padding, 10 float64 = 96 bytes)
  LookupCerenkov: two maps (size_t n, then n pairs of float64 beta, photons/mm): scintillator, fiber
  LookupBinDefinitions: 7 vectors (size_t n, n float64): x, y, z, beta, theta, phi, r bin edges
  bins of the three tables in order (scintillation in scintillator [x,y,z], Cerenkov in scintillator [x,y,z,beta],
  Cerenkov in fiber [beta,theta,phi,r,z]); each: uint32 bin number, float32 arrival probability, then size_t n +
  n uint8 time-delay weights (index = delay in ns), size_t n + n uint8 fiber-emission weights (index = emissions).

The tables are Mu2e-internal (see package doc). Arrays are kept compact: weights as uint8 padded to the longest
vector of each table, plus their per-bin sums (the stock code samples index i with probability w_i / sum w).
"""
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_DIR = "/cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0"
TABLES = ("scint", "ckscint", "ckfiber")


def lookup_dir():
    return os.environ.get("CRVFS_LOOKUP_DIR", DEFAULT_DIR)


@dataclass
class Table:
    prob: np.ndarray        # [N] float32 arrival probability at SiPM 0
    delays: np.ndarray      # [N, Dmax] uint8 weights
    delay_sum: np.ndarray   # [N] int32
    emis: np.ndarray        # [N, Emax] uint8 weights
    emis_sum: np.ndarray    # [N] int32


@dataclass
class LookupTable:
    name: str
    version: tuple
    reflector: int
    half_thickness: float
    half_width: float
    half_length: float
    fiber_separation: float
    hole_radius_x: float
    hole_radius_y: float
    fiber_radius: float
    corner_radius: float
    birks: float
    fiber_decay_time: float
    ck_scint: np.ndarray    # [n, 2] (beta, photons per mm), sorted by beta
    ck_fiber: np.ndarray
    edges: dict             # x, y, z, beta, theta, phi, r -> float64 arrays
    tables: dict            # scint, ckscint, ckfiber -> Table


class _Reader:
    def __init__(self, buf):
        self.b, self.o = buf, 0

    def take(self, dtype, n=1):
        dt = np.dtype(dtype)
        a = np.frombuffer(self.b, dtype=dt, count=n, offset=self.o)
        self.o += dt.itemsize * n
        return a

    def size_t(self):
        return int(self.take("<u8")[0])

    def vec(self, dtype):
        return self.take(dtype, self.size_t())


def _read_bins(r, n):
    """n LookupBins; vectorised over bins after one pass collecting offsets (variable-length vectors)."""
    prob = np.empty(n, np.float32)
    d_off, d_len, e_off, e_len = (np.empty(n, np.int64) for _ in range(4))
    b, o = r.b, r.o
    for i in range(n):
        num, p = np.frombuffer(b, "<u4", 1, o)[0], np.frombuffer(b, "<f4", 1, o + 4)[0]
        if num != i:
            raise ValueError(f"corrupt lookup table: bin {num} at position {i}")
        prob[i] = p
        o += 8
        nd = int(np.frombuffer(b, "<u8", 1, o)[0])
        d_off[i], d_len[i] = o + 8, nd
        o += 8 + nd
        ne = int(np.frombuffer(b, "<u8", 1, o)[0])
        e_off[i], e_len[i] = o + 8, ne
        o += 8 + ne
    r.o = o
    arr = np.frombuffer(b, np.uint8)

    def gather(off, ln):
        m = int(ln.max()) if n else 0
        out = np.zeros((n, max(m, 1)), np.uint8)
        idx = off[:, None] + np.arange(out.shape[1])[None, :]
        ok = np.arange(out.shape[1])[None, :] < ln[:, None]
        out[ok] = arr[idx[ok]]
        return out

    delays, emis = gather(d_off, d_len), gather(e_off, e_len)
    return Table(prob, delays, delays.sum(1, dtype=np.int32), emis, emis.sum(1, dtype=np.int32))


def load(name, directory=None):
    """Read LookupTable_<name> (e.g. '6000_0') from the lookup directory."""
    path = os.path.join(directory or lookup_dir(), f"LookupTable_{name}" if not name.startswith("LookupTable") else name)
    with open(path, "rb") as fh:
        buf = fh.read()
    r = _Reader(buf)
    v1, v2, refl = (int(x) for x in r.take("<i4", 3))
    r.take("<u1", 4)  # struct padding before the doubles
    c = r.take("<f8", 10)
    if v1 != 6:
        raise ValueError(f"{path}: lookup table version {v1}.{v2}, expected 6.x")

    def cmap():
        n = r.size_t()
        return r.take("<f8", 2 * n).reshape(n, 2).copy()

    ck_s, ck_f = cmap(), cmap()
    edges = {k: r.vec("<f8").copy() for k in ("x", "y", "z", "beta", "theta", "phi", "r")}
    nb = {k: len(v) - 1 for k, v in edges.items()}
    n_scint = nb["x"] * nb["y"] * nb["z"]
    n_cks = n_scint * nb["beta"]
    n_ckf = nb["beta"] * nb["theta"] * nb["phi"] * nb["r"] * nb["z"]
    tables = {"scint": _read_bins(r, n_scint), "ckscint": _read_bins(r, n_cks), "ckfiber": _read_bins(r, n_ckf)}
    if r.o != len(buf):
        raise ValueError(f"{path}: {len(buf) - r.o} trailing bytes")
    return LookupTable(os.path.basename(path), (v1, v2), refl, *(float(x) for x in c), ck_s, ck_f, edges, tables)
