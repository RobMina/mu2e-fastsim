"""Hash-bound numeric source checkpoints, separate from projected birth-summary states."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, arrays, binding, *, parent_sha256=None):
    """Write one new checkpoint pair; only the completed manifest authorizes reuse."""
    path = Path(path)
    states = path.with_suffix(".npz")
    if path == states or path.exists() or states.exists():
        raise ValueError("Refusing to overwrite a checkpoint")
    numeric = {name: np.asarray(value) for name, value in arrays.items()}
    if not numeric or any(v.dtype.kind not in "biuf" or not np.all(np.isfinite(v)) for v in numeric.values()):
        raise ValueError("Checkpoint arrays must be finite numeric values")
    encoded = json.dumps(binding, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with states.open("xb") as stream:
        np.savez_compressed(stream, **numeric)
    manifest = dict(schema="flash-source-checkpoint-v1", states=states.name, sha256=sha256(states),
                    binding=json.loads(encoded), parent_sha256=parent_sha256,
                    arrays={name: dict(dtype=value.dtype.str, shape=list(value.shape))
                            for name, value in numeric.items()})
    with path.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


def load(path, binding):
    """Reject foreign configuration, changed state bytes or malformed numeric columns."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "flash-source-checkpoint-v1":
        raise ValueError("Unknown checkpoint manifest schema")
    expected = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
    if manifest.get("binding") != expected:
        raise ValueError("Checkpoint source/settings/geometry/maps/code/runtime binding mismatch")
    name = manifest.get("states")
    if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".npz":
        raise ValueError("Checkpoint state must be a sibling NPZ basename")
    states = path.parent / name
    if sha256(states) != manifest.get("sha256"):
        raise ValueError("Checkpoint state artifact hash mismatch")
    with np.load(states, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    schema = manifest.get("arrays")
    if not isinstance(schema, dict) or set(schema) != set(arrays):
        raise ValueError("Checkpoint array schema mismatch")
    for name, value in arrays.items():
        if (value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)) or
                schema[name] != dict(dtype=value.dtype.str, shape=list(value.shape))):
            raise ValueError("Checkpoint array dtype/shape/finite-value mismatch")
    return arrays, manifest
