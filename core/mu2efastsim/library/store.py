"""Library files: save and load with checks (docs/libraries.md, "Storage and distribution").

A library file is a NumPy ``.npz`` (``allow_pickle=False``) holding the library's arrays and its manifest as the
canonical JSON string ``__manifest__``. A library whose data file predates this scheme and is frozen (a shipped
file whose bytes are pinned elsewhere) carries the same manifest as a sidecar, ``<stem>.manifest.json`` next to it;
a file with both must have the same manifest in each.

:func:`load` refuses, with :class:`LibraryError` listing every problem, a file without a manifest, a manifest that
fails the schema, arrays that are missing, extra, or of another dtype or shape than declared, a content hash that
does not match the arrays, axis edges that disagree with the manifest, or any field of ``expect`` that differs. It
returns a :class:`Library`: the manifest, read-only arrays, the axes, and the file's sha256.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from . import manifest as M
from .axes import Axis, Grid
from .hashing import content_hash, file_sha256

MANIFEST_KEY = "__manifest__"


class LibraryError(ValueError):
    """A library file that fails its checks; ``problems`` lists every failure."""

    def __init__(self, path, problems):
        self.path, self.problems = str(path), list(problems)
        super().__init__(f"library {self.path}: " + "; ".join(self.problems))


def sidecar_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".manifest.json")


@dataclass(frozen=True)
class Library:
    """A loaded, checked library."""
    manifest: dict
    arrays: MappingProxyType
    path: str
    file_sha256: str

    @property
    def name(self):
        return self.manifest["name"]

    @property
    def version(self):
        return self.manifest["version"]

    @property
    def content_hash(self):
        return self.manifest["content_hash"]

    def axis(self, name):
        for d in self.manifest["axes"]:
            if d["name"] == name:
                return Axis.from_description(d, self.arrays[d["array"]])
        raise KeyError(f"{self.name}: no axis {name!r}")

    def grid(self, *names):
        """Grid of the named axes (all axes, in manifest order, by default)."""
        names = names or tuple(d["name"] for d in self.manifest["axes"])
        return Grid(tuple(self.axis(n) for n in names))

    def as_dict(self, writeable=False):
        """The arrays as a plain dict (copies when writeable, else the read-only arrays themselves)."""
        return {k: (np.array(v) if writeable else v) for k, v in self.arrays.items()}

    def provenance(self):
        """Short identity for an output's provenance record."""
        return dict(name=self.name, version=self.version, content_hash=self.content_hash,
                    file_sha256=self.file_sha256, config_hash=self.manifest.get("config_hash"),
                    validation=self.manifest["validation"].get("status"), complete=M.complete(self.manifest))


def describe_arrays(arrays):
    return M.array_specs(arrays)


def finalize(arrays, fields):
    """A checked manifest for ``arrays``: ``fields`` plus the arrays entry and the content hash."""
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    if MANIFEST_KEY in arrays:
        raise ValueError(f"{MANIFEST_KEY} is reserved")
    fields = dict(fields)
    fields["arrays"] = M.array_specs(arrays)
    fields["content_hash"] = content_hash(arrays)
    m = M.create(**fields)
    problems = _check(m, arrays)
    if problems:
        raise LibraryError("(new library)", problems)
    return m


def save(path, arrays, fields, *, sidecar=False, compressed=True):
    """Write ``arrays`` with their manifest (built from ``fields`` by :func:`finalize`) to ``path`` (an .npz).

    ``sidecar`` also writes ``<stem>.manifest.json``. Returns the manifest.
    """
    m = finalize(arrays, fields)
    out = {k: np.asarray(v) for k, v in arrays.items()}
    out[MANIFEST_KEY] = np.str_(M.dumps(m))
    path = Path(path)
    (np.savez_compressed if compressed else np.savez)(path, **out)
    if sidecar:
        M.write(sidecar_path(path), m)
    return m


def load_arrays(path):
    """The arrays of a library file without any check (to describe a file that has no manifest yet)."""
    with np.load(Path(path), allow_pickle=False) as f:
        return {k: f[k] for k in f.files if k != MANIFEST_KEY}


def write_sidecar(path, fields):
    """Sidecar manifest for an existing (frozen) library file: its arrays are read, hashed and described, the file
    itself is not touched. Returns the manifest."""
    path = Path(path)
    m = finalize(load_arrays(path), fields)
    M.write(sidecar_path(path), m)
    return m


def _check(m, arrays):
    problems = []
    declared, present = set(m["arrays"]), set(arrays)
    if declared - present:
        problems.append(f"missing arrays {sorted(declared - present)}")
    if present - declared:
        problems.append(f"undeclared arrays {sorted(present - declared)}")
    for k in sorted(declared & present):
        a, spec = arrays[k], m["arrays"][k]
        if a.dtype.str != spec["dtype"] or list(a.shape) != list(spec["shape"]):
            problems.append(f"array {k!r} is {a.dtype.str}{list(a.shape)}, manifest says {spec['dtype']}{spec['shape']}")
    if problems:
        return problems
    got = content_hash(arrays)
    if got != m["content_hash"]:
        problems.append(f"content hash {got[:16]}... differs from the manifest's {m['content_hash'][:16]}...")
    for d in m["axes"]:
        try:
            Axis.from_description(d, arrays[d["array"]])
        except (ValueError, KeyError) as exc:
            problems.append(str(exc))
    return problems


def verify(m, arrays, where="(library)", *, expect=None):
    """The load checks for a library in another container (ROOT trees, a CVMFS binary table): ``arrays`` are what
    its own loader returns, ``m`` its sidecar manifest. Raises LibraryError listing every problem."""
    try:
        M.validate(m)
    except M.ManifestError as exc:
        raise LibraryError(where, exc.problems) from None
    problems = _check(m, {k: np.asarray(v) for k, v in arrays.items()})
    problems += [f"{k} is {m.get(k)!r}, expected {v!r}" for k, v in (expect or {}).items() if m.get(k) != v]
    if problems:
        raise LibraryError(where, problems)
    return m


def load(path, *, expect=None, sidecar=None):
    """Load and check a library file.

    ``expect``: manifest fields that must have these values (e.g. ``{"name": ..., "version": 2}``, a pinned
    ``content_hash``, or ``{"legacy": {"schema": 1, ...}}``). ``sidecar``: path of the sidecar manifest (default
    ``<stem>.manifest.json`` when it exists).
    """
    path = Path(path)
    problems = []
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    embedded = arrays.pop(MANIFEST_KEY, None)
    side = Path(sidecar) if sidecar is not None else sidecar_path(path)
    manifests = []
    if embedded is not None:
        manifests.append(("embedded", json.loads(str(embedded))))
    if side.exists():
        manifests.append(("sidecar", json.loads(side.read_text())))
    elif sidecar is not None:
        raise LibraryError(path, [f"sidecar manifest {side} does not exist"])
    if not manifests:
        raise LibraryError(path, ["no manifest (neither an embedded __manifest__ nor a sidecar "
                                  f"{side.name}); a library without a manifest is refused"])
    if len(manifests) == 2 and M.dumps(manifests[0][1]) != M.dumps(manifests[1][1]):
        raise LibraryError(path, ["the embedded and sidecar manifests differ"])
    m = manifests[0][1]
    try:
        M.validate(m)
    except M.ManifestError as exc:
        raise LibraryError(path, exc.problems) from None
    problems += _check(m, arrays)
    for k, v in (expect or {}).items():
        if m.get(k) != v:
            problems.append(f"{k} is {m.get(k)!r}, expected {v!r}")
    if problems:
        raise LibraryError(path, problems)
    for a in arrays.values():
        a.flags.writeable = False
    return Library(manifest=m, arrays=MappingProxyType(arrays), path=str(path), file_sha256=file_sha256(path))
