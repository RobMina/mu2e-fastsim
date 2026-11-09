"""Prepared library caches are non-executable, exact and invalidated by every input."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import tier1b_assets
from crvfastsim.analyses import library_cache as C
from crvfastsim.analyses.context_audit import content_fingerprint


@pytest.fixture
def setup(tmp_path):
    library = tmp_path / "lib.root"
    library.write_bytes(b"source-library")
    side = tmp_path / "side.npz"
    side.write_bytes(b"side")
    Path(str(side) + ".json").write_text("{}")
    spec = tier1b_assets.Tier1bSpec("custom", str(library), topology=str(side))
    geom = SimpleNamespace(center=np.arange(12.).reshape(4, 3), labels=np.array(["EX", "M"], dtype=object))
    value = {"libdesc": pd.DataFrame({"xid": np.array([1, 0, 1], dtype=np.int64),
                                    "edep": np.array([.1, .2, .3], dtype=np.float32)}),
             "nested": {"array": np.arange(12).reshape(3, 4).T, "scalar": np.float32(.25),
                        "labels": np.array(["EX", None, "M"], dtype=object)},
             "levels": [(np.arange(5), np.array([], dtype=np.int32))], "flag": True}
    value["libdesc"].attrs["source"] = "reference"
    calls = []

    def loader(spec, geom):
        calls.append((spec, geom))
        return value

    return spec, geom, value, loader, calls, tmp_path / "cache"


def test_round_trip_and_loader_is_not_called_on_hit(setup):
    spec, geom, value, loader, calls, directory = setup
    expected = content_fingerprint(value)
    first = C.load(spec, geom, directory, loader)
    second = C.load(spec, geom, directory, loader)
    assert first is value and len(calls) == 1
    assert content_fingerprint(second) == expected
    assert second["libdesc"].edep.dtype == np.float32
    assert type(second["libdesc"].index) is pd.RangeIndex
    assert type(second["levels"]) is list and type(second["levels"][0]) is tuple
    assert not second["nested"]["array"].flags.writeable
    assert len(list(directory.iterdir())) == 1


@pytest.mark.parametrize("change", ["library", "side", "meta", "geometry", "options", "source"])
def test_all_dependencies_invalidate(setup, monkeypatch, change):
    spec, geom, value, loader, calls, directory = setup
    C.load(spec, geom, directory, loader)
    if change == "library":
        Path(spec.library).write_bytes(b"new library")
    elif change == "side":
        Path(spec.topology).write_bytes(b"new side")
    elif change == "meta":
        Path(spec.topology + ".json").write_text('{"new":true}')
    elif change == "geometry":
        geom.center[0, 0] = -4
    elif change == "options":
        spec = dataclasses.replace(spec, options={"hard": True})
    else:
        original = C.identity
        monkeypatch.setattr(C, "identity", lambda s, g: {**original(s, g), "test_source_changed": True})
    C.load(spec, geom, directory, loader)
    assert len(calls) == 2
    assert len(list(directory.iterdir())) == 2


def test_pin_corruption_and_manifest_tree_fail_closed(setup):
    spec, geom, value, loader, calls, directory = setup
    with pytest.raises(ValueError, match="pinned"):
        C.load(dataclasses.replace(spec, library_sha256="0" * 64), geom, directory, loader)
    C.load(spec, geom, directory, loader)
    root = next(directory.iterdir())
    file = next(root.glob("*.npy"))
    original = file.read_bytes()
    file.write_bytes(original[:-1] + b"x")
    with pytest.raises(ValueError, match="array changed"):
        C.load(spec, geom, directory, loader)
    file.write_bytes(original)
    manifest = root / "manifest.json"
    record = json.loads(manifest.read_text())
    record["content_fingerprint"] = "bad"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="content changed"):
        C.load(spec, geom, directory, loader)
    assert len(calls) == 1


def test_codec_duplicate_columns_and_named_index(tmp_path):
    value = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"], index=pd.Index([8, 9], name="row"))
    codec = C._Codec(tmp_path)
    restored = codec.decode(codec.encode(value))
    assert content_fingerprint(restored) == content_fingerprint(value)
    with pytest.raises(TypeError, match="unsupported"):
        codec.encode(object())


def test_opt_in_loader_wiring_and_default_unchanged(setup, monkeypatch):
    spec, geom, value, loader, calls, directory = setup
    monkeypatch.setattr(tier1b_assets, "_load_uncached", loader)
    monkeypatch.delenv("CRVFS_PREPARED_LIBRARY_CACHE", raising=False)
    assert tier1b_assets.load(spec, geom) is value
    monkeypatch.setenv("CRVFS_PREPARED_LIBRARY_CACHE", str(directory))
    tier1b_assets.load(spec, geom)
    assert content_fingerprint(tier1b_assets.load(spec, geom)) == content_fingerprint(value)
    assert len(calls) == 2
