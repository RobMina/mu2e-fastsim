"""Numeric checkpoint manifests fail closed on foreign or corrupt transport state."""
import json

import numpy as np
import pytest

from beamfastsim import flash_checkpoint as C


def _save(tmp_path):
    path = tmp_path / "checkpoint.json"
    arrays = dict(counter=np.array([2, 4], np.int32), state=np.array([[1., 2., 3.]]), flag=np.array([True]))
    binding = dict(states="hash", settings=dict(seed=1, capacity=2), code=dict(kernel="codehash"))
    C.save(path, arrays, binding)
    return path, arrays, binding


def test_roundtrip_byte_exact_and_lineage(tmp_path):
    path, arrays, binding = _save(tmp_path)
    loaded, manifest = C.load(path, binding)
    assert all(value.tobytes() == loaded[name].tobytes() for name, value in arrays.items())
    child = C.save(tmp_path / "next.json", loaded, binding, parent_sha256=manifest["sha256"])
    assert child["parent_sha256"] == manifest["sha256"]


def test_no_overwrite_and_pickle_objects(tmp_path):
    path, arrays, binding = _save(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        C.save(path, arrays, binding)
    with pytest.raises(ValueError, match="numeric"):
        C.save(tmp_path / "object.json", dict(bad=np.array([object()])), binding)
    assert not (tmp_path / "object.npz").exists()


def test_foreign_binding_rejected(tmp_path):
    path, _, binding = _save(tmp_path)
    binding["settings"]["seed"] = 2
    with pytest.raises(ValueError, match="binding mismatch"):
        C.load(path, binding)


def test_corrupt_artifact_rejected(tmp_path):
    path, _, binding = _save(tmp_path)
    with path.with_suffix(".npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        C.load(path, binding)


@pytest.mark.parametrize("change", ["schema", "shape", "dtype", "path", "missing"])
def test_manifest_mutations_rejected(tmp_path, change):
    path, _, binding = _save(tmp_path)
    manifest = json.loads(path.read_text())
    if change == "schema":
        manifest["schema"] = "unknown"
    elif change == "shape":
        manifest["arrays"]["state"]["shape"] = [3]
    elif change == "dtype":
        manifest["arrays"]["counter"]["dtype"] = "<i8"
    elif change == "path":
        manifest["states"] = "../other.npz"
    else:
        manifest["arrays"].pop("flag")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        C.load(path, binding)
