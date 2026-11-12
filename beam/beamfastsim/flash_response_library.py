"""Flash photon response library: expected tracker ionising deposit per photon (core/docs/libraries.md section 8).

A ``binned`` library of the shared core (mu2efastsim.library). Photons start inside a source volume around the
stopping target (a cylinder about the DS axis) and are transported by the stock Mu2e simulation; the stock
``EdepAna`` analyzer records, per photon, its start position, momentum and the summed StrawGasStep ionising energy
(no time cut). This module tabulates those rows and looks the table up.

Axes (Mu2e frame, DS axis at x = -3904 mm): species (photon only), energy [MeV, log], start z and r [mm] in the
source volume, cos_theta (direction cosine to +z) and u_r (radial direction cosine about the DS axis). Outputs per
cell: mean deposit per photon [MeV], its variance per photon, the photon count n and the count with a nonzero
deposit. Domain: other species refused; energy below the lowest edge answers 0 (declared fallback, counted), above
the top edge refused; start positions outside the volume refused; direction cosines clipped into [-1, 1).

Contract: ``load``, ``lookup_numpy`` (host, float64), ``to_device`` / ``lookup`` (device, float32 or float64,
jit-able: refused rows answer NaN, the below-energy fallback 0), ``domain_counts``. ``build`` and the CLI make a
library from a campaign configuration (JSON) with ``mu2efastsim.library.store.save``.

Pinning: ``load(path, expect={"content_hash": ...})`` is a pinned load; ``load_registered(ident)`` resolves a
registry entry (``data/flash_response_registry.json``) and pins its content hash. Every load is entered in
``mu2efastsim.library.provenance`` as "pinned" or "unpinned".
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from mu2efastsim.library import Axis, DomainCounter, Grid, OutOfDomain, hashing, manifest, store
from mu2efastsim.library import provenance as LP
from mu2efastsim.library.registry import Registry, default_cache
from mu2efastsim.library.axes import BELOW, IN, domain_codes, domain_codes_jax, locate_jax

DS_X = -3904.0
PHOTON = 22
AXES = ("species", "energy", "z", "r", "cos_theta", "u_r")
OUTPUTS = ("mean", "variance", "n", "n_nonzero")
FALLBACK = "energy_below_lowest_edge"
_EPS = 1e-12
_COS_TOL = 1e-6   # float32 momentum rounding in the EdepAna tree
REGISTRY = Path(__file__).resolve().parent / "data" / "flash_response_registry.json"
CACHE_ENV = "MU2EFS_FLASH_CACHE"
DEFAULT_SPLIT = dict(
    training="this file: every job listed in source.samples",
    held_out="campaigns disjoint from this one (independent photon sources and stock design points); none of "
             "their events is in this file",
    rule="separate Geant4 campaigns with their own seeds; the even and odd job halves of this campaign, built as "
         "separate files, give the realization spread")


def make_axes(edges):
    """The six axes from ``edges`` {energy, z, r, cos_theta, u_r}; direction edges must span [-1, 1]."""
    for k in ("cos_theta", "u_r"):
        if edges[k][0] != -1.0 or edges[k][-1] != 1.0:
            raise ValueError(f"{k} edges must run from -1 to 1")
    return (Axis("species", "1", [-0.5, 0.5], frame="categorical: 0 = photon", array="species_edges",
                 scale="categorical", policy="refuse", extra={"labels": ["photon"], "pdg": [PHOTON]}),
            Axis("energy", "MeV", edges["energy"], frame="photon energy", array="energy_edges", scale="log",
                 policy="refuse", description="below the lowest edge: declared fallback 0 (counted)"),
            Axis("z", "mm", edges["z"], frame="Mu2e; photon start position in the source volume", array="z_edges",
                 scale="linear", policy="refuse"),
            Axis("r", "mm", edges["r"], frame="Mu2e; start radius about the DS axis x = -3904 mm", array="r_edges",
                 scale="irregular", policy="refuse"),
            Axis("cos_theta", "1", edges["cos_theta"], frame="direction cosine to +z (Mu2e)", array="cos_theta_edges",
                 scale="irregular", policy="clip"),
            Axis("u_r", "1", edges["u_r"], frame="radial direction cosine about the DS axis", array="u_r_edges",
                 scale="irregular", policy="clip"))


def keys_from_momentum(x, y, z, px, py, pz, x0=DS_X):
    """Lookup keys (z, r, cos_theta, u_r) from a Mu2e-frame start position and momentum (float64)."""
    x, y, z, px, py, pz = (np.asarray(a, dtype=np.float64) for a in (x, y, z, px, py, pz))
    dx = x - x0
    p = np.sqrt(px * px + py * py + pz * pz)
    r = np.hypot(dx, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_r = np.where(r > 0, (dx * px + y * py) / (r * p), 0.0)
    return dict(z=z, r=r, cos_theta=pz / p, u_r=u_r)


_EDEPANA = ("nprimaries", "primary_start_x", "primary_start_y", "primary_start_z", "primary_start_px",
            "primary_start_py", "primary_start_pz", "primary_start_e", "primary_start_pdg", "event_trk_edep")


def _rows_from_chunk(a, path):
    import awkward as ak
    if not np.all(ak.to_numpy(a["nprimaries"]) == 1):
        raise ValueError(f"{path}: every event must have exactly one primary")
    first = {k: ak.to_numpy(ak.fill_none(ak.firsts(a[k]), 0)) for k in _EDEPANA[1:-1]}
    if not np.all(first["primary_start_pdg"] == PHOTON):
        raise ValueError(f"{path}: non-photon primaries")
    rows = keys_from_momentum(first["primary_start_x"], first["primary_start_y"], first["primary_start_z"],
                              first["primary_start_px"], first["primary_start_py"], first["primary_start_pz"])
    rows["energy"] = first["primary_start_e"].astype(np.float64)
    rows["edep"] = ak.to_numpy(a["event_trk_edep"]).astype(np.float64)
    return rows


def iter_edepana(path, step_size=250_000):
    """Rows of an EdepAna tree (exactly one photon primary per event) in chunks of ``step_size`` events.

    Memory stays bounded by the chunk: the jagged primary branches are read as awkward arrays and reduced to their
    single entry, never expanded into Python objects.
    """
    import uproot
    with uproot.open(path) as f:
        for a in f["edep/tree"].iterate(list(_EDEPANA), step_size=step_size, library="ak"):
            yield _rows_from_chunk(a, path)


def read_edepana(path):
    """All rows of an EdepAna tree with exactly one photon primary: keys, energy and deposit (float64)."""
    chunks = list(iter_edepana(path))
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]} if chunks else {}


class Accumulator:
    """Per-cell sums over row batches (float64 sums, int64 counts)."""

    def __init__(self, axes):
        self.axes, self.grid = axes, Grid(axes)
        self.n = np.zeros(self.grid.size, np.int64)
        self.nnz = np.zeros(self.grid.size, np.int64)
        self.s1 = np.zeros(self.grid.size)
        self.s2 = np.zeros(self.grid.size)

    def add(self, rows, cos_range=(-1.0, 1.0)):
        """Add rows {energy, z, r, cos_theta, u_r, edep}; every row must be inside the table and ``cos_range``."""
        k = _clip_directions(rows)
        lo, hi = cos_range
        # EdepAna stores momenta in float32: allow that rounding, then keep each row inside its own class range so a
        # photon generated at cos(theta) <= hi is binned with its class, not in the neighbouring class's first bin
        if np.any((k["cos_theta"] < lo - _COS_TOL) | (k["cos_theta"] > hi + _COS_TOL)):
            raise ValueError(f"rows outside the sampled direction class [{lo}, {hi}]")
        top = hi if hi >= 1.0 else np.nextafter(hi, -np.inf)
        k["cos_theta"] = np.clip(k["cos_theta"], lo, min(top, np.nextafter(1.0, 0.0)))
        k["species"] = np.zeros(len(k["energy"]))
        for a in self.axes:
            if a.policy == "refuse" and np.any(domain_codes(a.edges, k[a.name]) != IN):
                raise ValueError(f"rows outside the table on axis {a.name!r}")
        cell = self.grid.cells(k)
        d = np.asarray(rows["edep"], dtype=np.float64)
        m = self.grid.size
        self.n += np.bincount(cell, minlength=m)
        self.nnz += np.bincount(cell, weights=(d > 0).astype(np.float64), minlength=m).astype(np.int64)
        self.s1 += np.bincount(cell, weights=d, minlength=m)
        self.s2 += np.bincount(cell, weights=d * d, minlength=m)

    def arrays(self):
        """Library arrays: edges, mean, variance (per photon, n - 1), n, n_nonzero. Empty cells: mean NaN."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.n > 0, self.s1 / np.maximum(self.n, 1), np.nan)
            var = np.where(self.n > 1, (self.s2 - self.n * mean * mean) / np.maximum(self.n - 1, 1), np.nan)
        var = np.where(np.isfinite(var), np.maximum(var, 0.0), var)
        out = {a.array: a.edges for a in self.axes}
        shape = self.grid.shape
        out.update(mean=mean.reshape(shape), variance=var.reshape(shape), n=self.n.reshape(shape),
                   n_nonzero=self.nnz.reshape(shape))
        return out


def _clip_directions(rows):
    k = {name: np.asarray(rows[name], dtype=np.float64) for name in ("energy", "z", "r", "cos_theta", "u_r")}
    for name in ("cos_theta", "u_r"):
        k[name] = np.clip(k[name], -1.0, np.nextafter(1.0, 0.0))
    return k


def manifest_fields(cfg, arrays, axes):
    """Complete manifest fields from a campaign configuration (see build)."""
    files = cfg["files"]
    return dict(
        name=cfg["name"], subsystem="flash", version=int(cfg["version"]), kind="binned",
        description="Expected tracker ionising deposit (StrawGasStep ionizing energy summed, no time cut) per "
                    "photon starting in the source volume around the stopping target, by species, energy, start "
                    "z and r, direction cosine to +z and radial direction cosine.",
        axes=[a.describe() for a in axes],
        outputs=[dict(name="mean", unit="MeV", dtype="float64",
                      description="mean deposit per photon; NaN for an empty cell or a refused row"),
                 dict(name="variance", unit="MeV2", dtype="float64", description="variance per photon (n - 1)"),
                 dict(name="n", unit="1", dtype="int64", description="photons in the cell"),
                 dict(name="n_nonzero", unit="1", dtype="int64", description="photons with a nonzero deposit")],
        source=dict(description=cfg["source_description"], classes=cfg["classes"],
                    samples=[dict(dataset=f.get("dataset", f["path"]), selection=_selection(cfg, f),
                                  events=int(f["events"])) for f in files],
                    **({"realization_files": cfg["realization_files"]} if cfg.get("realization_files") else {})),
        generator=cfg["generator"], release=cfg["release"],
        seeds={f"{f['class']}_{int(f['job']):04d}": int(f["seed"]) for f in files},
        geometry=cfg["geometry"], config_hash=hashing.digest(cfg["config_hashes"]),
        split=dict(cfg.get("split") or DEFAULT_SPLIT),
        dtype=dict(storage={k: str(v.dtype) for k, v in arrays.items()}, compute=["float32", "float64"],
                   notes="host lookup float64; device lookup in the caller's dtype"),
        rng=dict(host="none", device="none", streams={}, draws_per_row=0),
        domain=dict(coverage=cfg["coverage"],
                    fallbacks=[dict(name=FALLBACK, levels=["cell", "zero"],
                                    description=cfg.get("fallback_description",
                                                        "energy below the lowest edge answers 0, counted"))]),
        validation=dict(status="not validated", record=cfg.get("validation_record")),
        interpolation=dict(method="nearest", axes=list(AXES)),
        assumptions=[dict(name="azimuthal_symmetry", description="response independent of azimuth about the DS axis",
                          tested=False),
                     dict(name="foils_in_volume", description=cfg["geometry"]["tag"], tested=False)],
        unrecorded={})


def _selection(cfg, f):
    sha = cfg["classes"][f["class"]].get("gun_config_sha256")
    return f"direction class {f['class']}" + (f", gun configuration sha256 {sha}" if sha else "")


def build(cfg, out, reader=None):
    """Tabulate the campaign ``cfg`` and save it to ``out``; returns the manifest.

    ``cfg``: name, version, edges, classes {class: {cos_theta: [lo, hi], gun_config_sha256, ...}}, files [{path,
    dataset, class, job, seed, events}], source_description, generator, release, geometry {tag, hash},
    config_hashes, coverage; optional split {training, held_out, rule} (default DEFAULT_SPLIT),
    realization_files and fallback_description (why the below-edge fallback 0 is acceptable). ``dataset`` (default: the path) is the name written in the manifest: give a name with no
    user or site path for a file that is published.
    Each file must hold exactly its declared number of events. ``reader`` (path -> rows) replaces the default
    chunked EdepAna reader (iter_edepana), e.g. for tests.
    """
    axes = make_axes(cfg["edges"])
    acc = Accumulator(axes)
    for f in cfg["files"]:
        chunks = [reader(f["path"])] if reader is not None else iter_edepana(f["path"])
        rows_seen = 0
        for rows in chunks:
            rows_seen += len(rows["edep"])
            acc.add(rows, tuple(cfg["classes"][f["class"]]["cos_theta"]))
        if rows_seen != int(f["events"]):
            raise ValueError(f"{f['path']}: {rows_seen} rows, {f['events']} declared")
    arrays = acc.arrays()
    return store.save(out, arrays, manifest_fields(cfg, arrays, axes))


def load(path, expect=None, pin_set=None):
    """Load and check a flash response library. A load whose ``expect`` holds a ``content_hash`` is recorded as
    pinned (``pin_set``: what the pin came from), any other load as unpinned (mu2efastsim.library.provenance)."""
    lib = store.load(path, expect=dict({"subsystem": "flash", "kind": "binned"}, **(expect or {})))
    if [a["name"] for a in lib.manifest["axes"]] != list(AXES):
        raise store.LibraryError(path, [f"axes must be {AXES}"])
    pinned = bool(expect and expect.get("content_hash"))
    LP.record(dict(lib.provenance(), file=os.path.basename(str(path)), status="pinned" if pinned else "unpinned",
                   pin_set=(pin_set or "expect") if pinned else None))
    return lib


def registry(path=REGISTRY, cache=None):
    """The flash response registry (file, sha256, content hash and release asset URL per library)."""
    return Registry.read(path, cache=cache or default_cache(CACHE_ENV, "flash"))


def load_registered(ident, reg=None, download=True):
    """Load a registered library by id, ``name@version``, name or hash prefix; sha256 and content hash pinned."""
    reg = reg or registry()
    key = reg.find(ident)
    path = reg.fetch(key, download=download)
    return load(path, expect={"content_hash": reg.entries[key]["content_hash"]}, pin_set=f"registry:{key}")


def _axes_of(lib):
    return tuple(lib.axis(n) for n in AXES)


def _host_index(lib, energy, z, r, cos_theta, u_r, species, counter):
    axes = _axes_of(lib)
    keys = _clip_directions(dict(energy=energy, z=z, r=r, cos_theta=cos_theta, u_r=u_r))
    n = len(keys["energy"])
    sp = np.full(n, PHOTON) if species is None else np.asarray(species)
    codes = {a.name: domain_codes(a.edges, keys[a.name]) for a in axes[1:]}
    below = codes["energy"] == BELOW
    if counter is not None:
        counter.add_lookups(n)
        for a in axes[1:]:
            counter.add_codes(a.name, codes[a.name])
        counter.add_levels(FALLBACK, below.astype(np.int64))
    bad_species = np.flatnonzero(sp != PHOTON)
    if bad_species.size:
        if counter is not None:
            counter.add_refused(bad_species.size)
        raise OutOfDomain(lib.name, "species", bad_species)
    for name in ("energy", "z", "r"):
        refuse = (codes[name] != IN) & ~below if name == "energy" else codes[name] != IN
        if np.any(refuse):
            if counter is not None:
                counter.add_refused(int(refuse.sum()))
            raise OutOfDomain(lib.name, name, np.flatnonzero(refuse), codes[name][refuse])
    keys["species"] = np.zeros(n)
    keys["energy"] = np.where(below, axes[1].edges[0], keys["energy"])
    cell = Grid(axes).cells(keys)
    return cell, below


def lookup_numpy(lib, energy, z, r, cos_theta, u_r, species=None, counter=None):
    """Mean deposit per photon [MeV], float64; refuses out-of-domain rows (OutOfDomain) before answering."""
    cell, below = _host_index(lib, energy, z, r, cos_theta, u_r, species, counter)
    mean = np.asarray(lib.arrays["mean"], dtype=np.float64).ravel()[cell]
    return np.where(below, 0.0, mean)


def domain_counts(lib, energy, z, r, cos_theta, u_r, species=None):
    """DomainCounter of the rows (no exception: refusals are counted)."""
    c = DomainCounter(lib.name)
    try:
        _host_index(lib, energy, z, r, cos_theta, u_r, species, c)
    except OutOfDomain:
        pass
    return c


def to_device(lib, dtype):
    """Device copy: edges and mean in ``dtype`` (float32 or float64)."""
    import jax.numpy as jnp
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")
    axes = _axes_of(lib)
    return dict(dtype=np.dtype(dtype), shape=tuple(a.nbins for a in axes),
                edges={a.name: jnp.asarray(a.edges, dtype=dtype) for a in axes[1:]},
                mean=jnp.asarray(np.asarray(lib.arrays["mean"]).ravel(), dtype=dtype))


def lookup(dl, energy, z, r, cos_theta, u_r):
    """Device twin of lookup_numpy for photons (jit-able): refused rows NaN, below the energy table 0."""
    import jax.numpy as jnp
    dt = dl["dtype"]
    e, zz, rr = (jnp.asarray(v, dtype=dt) for v in (energy, z, r))
    one = jnp.nextafter(jnp.asarray(1, dt), jnp.asarray(0, dt))
    ct = jnp.clip(jnp.asarray(cos_theta, dtype=dt), -1, one)
    ur = jnp.clip(jnp.asarray(u_r, dtype=dt), -1, one)
    below = e < dl["edges"]["energy"][0]
    idx, ok = [], jnp.ones(e.shape, bool)
    for name, v in (("energy", jnp.where(below, dl["edges"]["energy"][0], e)), ("z", zz), ("r", rr),
                    ("cos_theta", ct), ("u_r", ur)):
        ed = dl["edges"][name]
        ok = ok & (domain_codes_jax(ed, v) == IN)
        idx.append(locate_jax(ed, v))
    cell = idx[0]
    for i, nb in zip(idx[1:], dl["shape"][2:]):
        cell = cell * nb + i
    val = dl["mean"][cell]
    return jnp.where(~ok, jnp.nan, jnp.where(below, jnp.asarray(0, dt), val))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flash_response_library", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="tabulate a campaign configuration (JSON) into a library .npz")
    b.add_argument("--config", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)
    m = build(cfg, a.out)
    lib = load(a.out, expect={"name": cfg["name"]})
    arr = lib.arrays
    n, nz = np.asarray(arr["n"]), np.asarray(arr["n_nonzero"])
    print(json.dumps(dict(name=lib.name, version=lib.version, content_hash=lib.content_hash,
                          complete=manifest.complete(m), shape=list(n.shape), cells=int(n.size),
                          filled=int((n > 0).sum()), photons=int(n.sum()), nonzero=int(nz.sum()),
                          cells_ge10_nonzero=int((nz >= 10).sum())), indent=1))


if __name__ == "__main__":
    main()
