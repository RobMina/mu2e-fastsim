"""Hashes of library files, manifests and array contents (docs/libraries.md, "Manifest").

Three hashes, each with one definition used by every library:

    file_sha256(path)       sha256 of the file's bytes: what a registry or a release asset pins
    content_hash(arrays)    sha256 over the arrays themselves (name, dtype, shape, bytes; names sorted): the identity of
                            the library's data, independent of compression and of the container format
    digest(obj)             sha256 of the canonical JSON of a mapping (a build configuration, a criterion), first
                            HASH_LEN hex digits: the short identifier of what a library was built from

Canonical JSON is ``json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)``, the form the tracker
response configurations and the CRV response configuration already use. Standard library and NumPy only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HASH_LEN = 16           # short identifiers (a build configuration's digest), as the tracker response registry uses
CHUNK = 1 << 20


def canonical_json(obj):
    """Canonical text of a JSON-compatible object: sorted keys, no spaces, NaN and infinity refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj, length=HASH_LEN):
    """sha256 of the canonical JSON of ``obj``, truncated to ``length`` hex digits (None: all 64)."""
    h = hashlib.sha256(canonical_json(obj).encode()).hexdigest()
    return h if length is None else h[:length]


def file_sha256(path):
    """sha256 of a file's bytes, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _array_bytes(name, a):
    a = np.asarray(a)
    if a.dtype.kind == "O":
        raise TypeError(f"array {name!r}: object arrays cannot be hashed or stored in a library")
    head = canonical_json({"name": name, "dtype": a.dtype.str, "shape": list(a.shape)}).encode()
    return len(head).to_bytes(8, "little") + head, np.ascontiguousarray(a).tobytes()


def array_hash(name, a):
    """sha256 of one array: its name, dtype (byte order included), shape and C-order bytes, length-prefixed."""
    h = hashlib.sha256()
    for part in _array_bytes(name, a):
        h.update(part)
    return h.hexdigest()


def content_hash(arrays, exclude=()):
    """sha256 over every array of ``arrays`` (a mapping name -> array), names sorted, ``exclude`` left out.

    Two libraries with the same content hash hold the same arrays bit for bit, whatever file they came from.
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        if name in exclude:
            continue
        for part in _array_bytes(name, arrays[name]):
            h.update(len(part).to_bytes(8, "little"))
            h.update(part)
    return h.hexdigest()


def is_hex(value, length=64):
    """True for a lower-case hex string of the given length."""
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)
