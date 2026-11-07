"""Explicit in-process source cache; content-checked, owned immutable array snapshots."""
from copy import deepcopy

import awkward as ak
import numpy as np

from .frames import FrameBatch


def _readonly(values):
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _snapshot(batch):
    collections = {}
    for name, rows in batch.collections.items():
        form, length, buffers = ak.to_buffers(ak.to_packed(rows))
        collections[name] = (form, length, {key: _readonly(buf) for key, buf in buffers.items()})
    return _readonly(batch.event_ids), collections


class FrameSourceCache:
    """Reuse parsed source pools only after hashing all source bytes on every request.

    The cache owns byte-backed read-only snapshots, not caller-provided mutable
    arrays. No disk cache or stale-path trust is involved. Modified files reload
    through the same complete validator. Reference checks at overlay/adapter are
    still performed; cached validation is not an unchecked bypass. Create one
    instance per process and pass it to load_manifest(source_cache=cache).
    """

    def __init__(self):
        self.__entries = {}
        self.hits = 0
        self.misses = 0

    def read(self, source):
        from .direct_frames import _read, _sha256
        key = tuple(sorted(source.items()))
        hashes = {"source_sha256": _sha256(source["path"])}
        if "event_log" in source:
            hashes["event_log_sha256"] = _sha256(source["event_log"])
        entry = self.__entries.get(key)
        if entry is not None and entry[1] == hashes:
            self.hits += 1
        else:
            batch, fingerprint = _read(source, hashes=hashes)
            entry = (_snapshot(batch), dict(hashes), deepcopy(fingerprint))
            self.__entries[key] = entry
            self.misses += 1
        (ids, collections), _, fingerprint = entry
        rows = {name: ak.from_buffers(deepcopy(form), length, {key: buf.view() for key, buf in buffers.items()})
                for name, (form, length, buffers) in collections.items()}
        return FrameBatch(ids.view(), rows), deepcopy(fingerprint)

    def clear(self):
        self.__entries.clear()
