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

Library pinning (mu2efastsim.library, core/docs/libraries.md "Pinning in run configurations"): the tables are pinned
by the run configurations, not by their names. A load with a pin (``load(name, pin=...)``, or any load inside
``with pinned(pin_map):``, as RunConfig.lookup_tables / photon_map and reco-r3 do with PIN_SET) checks the content
hash of the decoded arrays and refuses a mismatch or an asset missing from the set; any other load is
allowed (development tables, test doubles with production names) and records the content hash with the status
"unpinned". Either way the table's ``provenance`` says which, and the load is entered in
mu2efastsim.library.provenance.
"""
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mu2efastsim.library import provenance as LP
from mu2efastsim.library.hashing import content_hash
from mu2efastsim.library.store import LibraryError

DEFAULT_DIR = "/cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0"
TABLES = ("scint", "ckscint", "ckfiber")
PIN_FILE = Path(__file__).resolve().parent.parent / "configs" / "assets_crvconditions_v6_0.json"
PIN_SET = "crvconditions_v6_0"
_ACTIVE = threading.local()


@contextmanager
def pinned(pin_map):
    """Inside the block, every lookup-table and photon-map load is checked against pin_map ({asset: pin}, e.g.
    RunConfig.asset_pin_map()); None leaves the loads unpinned."""
    prev = getattr(_ACTIVE, "pins", None)
    _ACTIVE.pins = pin_map
    try:
        yield
    finally:
        _ACTIVE.pins = prev


def active_pin(asset):
    """The pin of asset ('LookupTable_6000_0', 'photonMap.root') in the active pin map; None outside pinned().
    An asset the active set does not list is refused."""
    pins = getattr(_ACTIVE, "pins", None)
    if pins is None:
        return None
    if asset not in pins:
        raise LibraryError(asset, [f"{asset} is not in the pinned asset set"])
    return pins[asset]


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
    provenance: dict = field(default=None, compare=False)   # library identity of the load (pinned or unpinned)


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


def load(name, directory=None, pin=None):
    """Read LookupTable_<name> (e.g. '6000_0') from the lookup directory. pin ({content_hash}, e.g. from pins()):
    refuse a table whose decoded content differs (LibraryError); without one the load is recorded as unpinned."""
    path = os.path.join(directory or lookup_dir(), f"LookupTable_{name}" if not name.startswith("LookupTable") else name)
    if pin is None:
        pin = active_pin(os.path.basename(path))
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
    L = LookupTable(os.path.basename(path), (v1, v2), refl, *(float(x) for x in c), ck_s, ck_f, edges, tables)
    L.provenance = _provenance(L, path, len(buf), pin)
    return L


def arrays(L):
    """The decoded content of a table as named arrays (what its content hash covers)."""
    out = {"version": np.array(L.version, np.int64), "reflector": np.int64(L.reflector),
           "constants": np.array([L.half_thickness, L.half_width, L.half_length, L.fiber_separation, L.hole_radius_x,
                                  L.hole_radius_y, L.fiber_radius, L.corner_radius, L.birks, L.fiber_decay_time]),
           "ck_scint": L.ck_scint, "ck_fiber": L.ck_fiber}
    out.update({f"edges_{k}": v for k, v in L.edges.items()})
    for t, T in L.tables.items():
        out.update({f"{t}_prob": T.prob, f"{t}_delays": T.delays, f"{t}_delay_sum": T.delay_sum,
                    f"{t}_emis": T.emis, f"{t}_emis_sum": T.emis_sum})
    return out


def pins(set_name=PIN_SET, path=PIN_FILE):
    """{asset name: {content_hash, size}} of a pinned asset set (configs/assets_<set>.json)."""
    d = json.loads(Path(path).read_text())
    if d.get("schema") != 1 or d.get("set") != set_name:
        raise LibraryError(path, [f"expected asset set {set_name!r} (schema 1)"])
    return d["assets"]


def write_pins(directory=None, tables=("2370_1", "3200_0", "6000_0"), set_name=PIN_SET, path=PIN_FILE,
               source=DEFAULT_DIR):
    """Write the pinned asset set of the tables and photonMap.root in ``directory`` (a copy of ``source``): per asset
    the content hash of the decoded arrays, the file size and the file sha256. Returns the set."""
    from mu2efastsim.library.hashing import file_sha256
    from . import sipm
    directory = directory or lookup_dir()
    assets = {}
    for t in tables:
        f = os.path.join(directory, f"LookupTable_{t}")
        assets[f"LookupTable_{t}"] = dict(content_hash=load(t, directory).provenance["content_hash"],
                                          size=os.path.getsize(f), file_sha256=file_sha256(f))
    f = os.path.join(directory, "photonMap.root")
    assets["photonMap.root"] = dict(content_hash=sipm.load_photon_map(f)[1]["content_hash"], size=os.path.getsize(f),
                                    file_sha256=file_sha256(f))
    d = dict(schema=1, set=set_name, source=source, assets=assets,
             description="CRV optical assets (Offline CRVConditions v6_0): photon lookup tables and the SiPM photon "
                         "map. content_hash: mu2efastsim.library.hashing.content_hash of the decoded arrays "
                         "(lookup.arrays, the photon map as 'photon_map'), checked by every pinned load.")
    Path(path).write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    return d


def _provenance(L, path, size, pin):
    h = content_hash(arrays(L))
    rec = dict(name=f"crv.lookup.{L.name.replace('LookupTable_', '')}", file=os.path.basename(path), size=size,
               content_hash=h, status="unpinned", pin_set=None)
    if pin is not None:
        if pin.get("content_hash") != h:
            raise LibraryError(path, [f"content hash {h[:16]}... differs from the pinned {str(pin.get('content_hash'))[:16]}... "
                                      f"of {pin.get('set', PIN_SET)}"])
        rec.update(status="pinned", pin_set=pin.get("set", PIN_SET))
    LP.record(rec)
    return rec
