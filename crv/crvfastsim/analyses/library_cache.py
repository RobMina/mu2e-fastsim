"""Content-bound, non-pickle cache of an unchanged fully attached host library."""
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from .context_audit import content_fingerprint

SCHEMA = 1


def _sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(spec, geom):
    import crvfastsim
    import mu2efastsim
    sources = {}
    for package in (crvfastsim, mu2efastsim):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json") and not path.name.startswith("._"):
                sources[package.__name__ + "/" + str(path.relative_to(root))] = _sha(path)
    assets = {str(index): _sha(path) for index, path in enumerate(spec.paths())}
    if spec.library_sha256 and assets["0"] != spec.library_sha256:
        raise ValueError("prepared library is not the pinned library")
    return {"schema": SCHEMA, "spec": {"name": spec.name, "options": spec.options,
            "library_sha256": spec.library_sha256, "has_topology": bool(spec.topology),
            "has_templates": bool(spec.templates)}, "assets": assets,
            "geometry": content_fingerprint(geom), "sources": sources,
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "uproot")},
            "python": list(sys.version_info[:2])}


class _Codec:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}

    def encode(self, value):
        if value is None or type(value) in (bool, int, float, str):
            return {"kind": "scalar", "value": value}
        if type(value) is np.ndarray:
            if value.dtype.kind == "O":
                content_fingerprint(value)
                return {"kind": "strings", "shape": value.shape, "value": value.tolist()}
            name = f"array_{len(self.files):05d}.npy"
            np.save(self.root / name, value, allow_pickle=False)
            self.files[name] = _sha(self.root / name)
            return {"kind": "array", "file": name}
        if isinstance(value, np.generic):
            return {"kind": "numpy_scalar", "value": self.encode(np.asarray(value))}
        if isinstance(value, pd.RangeIndex):
            return {"kind": "range", "start": value.start, "stop": value.stop,
                    "step": value.step, "name": self.encode(value.name)}
        if type(value) is pd.Index:
            return {"kind": "index", "value": self.encode(value.to_numpy(copy=False)),
                    "name": self.encode(value.name)}
        if type(value) is pd.DataFrame:
            if any(not isinstance(dtype, np.dtype) for dtype in value.dtypes):
                raise TypeError("prepared library cache supports primitive dataframe columns only")
            return {"kind": "frame", "index": self.encode(value.index), "columns": self.encode(value.columns),
                    "arrays": [self.encode(value.iloc[:, i].to_numpy(copy=False)) for i in range(len(value.columns))],
                    "attrs": self.encode(value.attrs), "duplicates": bool(value.flags.allows_duplicate_labels)}
        if type(value) is dict:
            return {"kind": "dict", "items": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if type(value) in (list, tuple):
            return {"kind": type(value).__name__, "items": [self.encode(v) for v in value]}
        raise TypeError(f"unsupported prepared library cache value: {type(value)!r}")

    def decode(self, record):
        kind = record["kind"]
        if kind == "scalar":
            return record["value"]
        if kind == "array":
            name = record["file"]
            if name not in self.files or Path(name).name != name:
                raise ValueError("invalid prepared library cache array")
            return np.asarray(np.load(self.root / name, allow_pickle=False, mmap_mode="r"))
        if kind == "strings":
            result = np.asarray(record["value"], dtype=object).reshape(record["shape"])
            content_fingerprint(result)
            return result
        if kind == "numpy_scalar":
            return self.decode(record["value"])[()]
        if kind == "range":
            return pd.RangeIndex(record["start"], record["stop"], record["step"], name=self.decode(record["name"]))
        if kind == "index":
            return pd.Index(self.decode(record["value"]), name=self.decode(record["name"]), copy=False)
        if kind == "frame":
            arrays = [self.decode(item) for item in record["arrays"]]
            frame = pd.DataFrame({i: array for i, array in enumerate(arrays)}, copy=False,
                                 index=self.decode(record["index"]))
            frame.columns = self.decode(record["columns"])
            frame.attrs = self.decode(record["attrs"])
            frame.flags.allows_duplicate_labels = record["duplicates"]
            return frame
        if kind == "dict":
            return {self.decode(k): self.decode(v) for k, v in record["items"]}
        if kind in ("list", "tuple"):
            items = [self.decode(v) for v in record["items"]]
            return tuple(items) if kind == "tuple" else items
        raise ValueError(f"invalid prepared library cache kind: {kind!r}")


def _read(root, expected):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["identity"] != expected:
        raise ValueError("prepared library cache identity mismatch")
    codec = _Codec(root)
    codec.files = manifest["files"]
    for name, digest in codec.files.items():
        if Path(name).name != name or _sha(root / name) != digest:
            raise ValueError("prepared library cache array changed")
    result = codec.decode(manifest["tree"])
    if content_fingerprint(result) != manifest["content_fingerprint"]:
        raise ValueError("prepared library cache content changed")
    return result


def load(spec, geom, directory, loader):
    """Reuse only a verified identical library, or create a new content-addressed entry."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = identity(spec, geom)
    key = hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False).encode()).hexdigest()
    root = directory / key
    if root.exists():
        return _read(root, expected)
    result = loader(spec, geom)
    if identity(spec, geom) != expected:
        raise ValueError("prepared library inputs changed during cache creation")
    temporary = Path(tempfile.mkdtemp(prefix=".creating_", dir=directory))
    try:
        codec = _Codec(temporary)
        tree = codec.encode(result)
        manifest = {"identity": expected, "tree": tree, "files": codec.files,
                    "content_fingerprint": content_fingerprint(result)}
        (temporary / "manifest.json").write_text(json.dumps(manifest, allow_nan=True))
        verified = _read(temporary, expected)
        if content_fingerprint(verified) != content_fingerprint(result):
            raise ValueError("prepared library cache round trip differs")
        try:
            temporary.rename(root)
        except OSError:
            if not root.is_dir():
                raise
            _read(root, expected)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
