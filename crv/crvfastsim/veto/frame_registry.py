"""Frozen source-qualified frame keys for legacy diagnostic kernels, never exposure weights."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import keyed_optics as K
from .replay import IDS

CONTRACT = "input-qualified-frame-registry-v1"
FRAME_KEYS = ("input_sha256", *IDS)
COLUMNS = (*FRAME_KEYS, "pbtime", "intensity", "frame_id")


def _normalize(events):
    out = events[[*FRAME_KEYS, "pbtime", "intensity"]].copy()
    if not len(out) or len(out) >= 2**20:
        raise ValueError("registry requires 1..2**20-1 frames")
    for digest in out.input_sha256.unique():
        K.Draws(digest, 0)
    for name in IDS:
        a = out[name].to_numpy()
        if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > np.iinfo(np.uint32).max):
            raise ValueError("frame identities must be uint32-range integers")
        out[name] = a.astype(np.int64)
    for name in ("pbtime", "intensity"):
        a = out[name].to_numpy(dtype=float)
        if not np.isfinite(a).all() or (name == "intensity" and np.any(a < 0)):
            raise ValueError("invalid registry exposure metadata")
        out[name] = a
    if out.duplicated(list(FRAME_KEYS)).any():
        raise ValueError("duplicate source-qualified frame")
    return out.sort_values(list(FRAME_KEYS)).reset_index(drop=True)


def freeze(events):
    out = _normalize(events)
    out["frame_id"] = np.arange(len(out), dtype=np.int64)
    return out[list(COLUMNS)]


def validate(registry):
    if list(registry) != list(COLUMNS):
        raise ValueError("unexpected registry columns")
    expected = freeze(registry)
    pd.testing.assert_frame_equal(registry.reset_index(drop=True), expected, check_exact=True)
    return expected


def digest(registry):
    r = validate(registry)
    h = hashlib.sha256(CONTRACT.encode())
    for name in COLUMNS:
        a = r[name].to_numpy()
        h.update(name.encode() + b"\0")
        if name == "input_sha256":
            h.update("".join(a).encode("ascii"))
        else:
            h.update(a.astype("<f8" if name in ("pbtime", "intensity") else "<i8").tobytes())
    return h.hexdigest()


def helper_seed(registry, response_seed):
    K.Draws("0" * 64, response_seed)
    payload = f"{CONTRACT}:{digest(registry)}:{int(response_seed)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def encode(frame, registry):
    """Replace physical event columns by immutable helper labels from the complete registry."""
    validate(registry)
    if "frame_id" in frame or "registry_sha256" in frame:
        raise ValueError("input already contains registry labels")
    for name in IDS:
        a = frame[name].to_numpy()
        if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > np.iinfo(np.uint32).max):
            raise ValueError("invalid source frame integer")
    location = pd.MultiIndex.from_frame(registry[list(FRAME_KEYS)]).get_indexer(pd.MultiIndex.from_frame(frame[list(FRAME_KEYS)]))
    if np.any(location < 0):
        raise ValueError("frame outside frozen registry")
    for name in ("pbtime", "intensity"):
        if name in frame and not np.array_equal(frame[name], registry[name].to_numpy()[location]):
            raise ValueError("changed frame exposure metadata")
    out = frame.drop(columns=list(FRAME_KEYS)).copy()
    out["event"] = registry.frame_id.to_numpy()[location]
    return out


def restore(frame, registry):
    """Restore source-qualified frame identity on every stage; keep frame_id for structural joins."""
    validate(registry)
    if any(name in frame for name in ("input_sha256", "run", "subrun", "frame_id", "registry_sha256")):
        raise ValueError("stage already contains physical or registry identities")
    a = frame.event.to_numpy()
    if a.dtype.kind not in "iu" and len(a):
        raise ValueError("noninteger stage label")
    if np.any(a < 0) or np.any(a >= len(registry)):
        raise ValueError("stage label outside registry")
    index = a.astype(np.int64)
    out = frame.drop(columns="event").copy()
    out["frame_id"] = index
    for name in FRAME_KEYS:
        out[name] = registry[name].to_numpy()[index]
    return out


def manifest(registry, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty distinct seed schedule required")
    return {"contract": CONTRACT, "registry_sha256": digest(registry), "frames": len(registry),
            "helper_seeds": {str(seed): helper_seed(registry, seed) for seed in seeds},
            "scope": "Frozen complete registry only; labels and seeds are not invariant to registry replacement."}
