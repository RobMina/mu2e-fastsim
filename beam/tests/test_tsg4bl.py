"""Synthetic ROOT contract tests only; no transport, production inputs or rate claims."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import uproot

from beamfastsim import tsg4bl as ref


def _rows(n=5, dtype=np.float64):
    data = {k: np.arange(n, dtype=dtype) + i for i, k in enumerate(ref.STATE)}
    data.update(EventID=np.arange(n, dtype=dtype), TrackID=np.ones(n, dtype=dtype),
                ParentID=np.zeros(n, dtype=dtype),
                PDGid=np.resize(np.array([13, -13, 211, -211, 11], dtype=dtype), n),
                Weight=np.ones(n, dtype=dtype))
    return data


def _write(path, data, tree="VirtualDetector/Coll_01_DetIn", update=False):
    types = {k: np.dtype((v.dtype, v.shape[1:])) if v.ndim > 1 else v.dtype for k, v in data.items()}
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        target = root.mktree(tree, types)
        if len(next(iter(data.values()))):
            target.extend(data)


def _read(path, **kwargs):
    kwargs.setdefault("planes", ["COL1in"])
    return ref.read([path], **kwargs)


def test_default_seven_planes_charges_stored_order_and_receipt(tmp_path):
    assert ref.PLANE_PATHS == {
        "COL1in": "VirtualDetector/Coll_01_DetIn",
        "COL1out": "VirtualDetector/Coll_01_DetOut",
        "COL3in": "VirtualDetector/Coll_03_DetIn",
        "COL3mid": "VirtualDetector/Coll_03_DetMid",
        "COL3out": "VirtualDetector/Coll_03_DetOut",
        "COL5in": "VirtualDetector/Coll_05_DetIn",
        "COL5out": "VirtualDetector/Coll_05_DetOut",
    }
    path = tmp_path / "seven.root"
    data = _rows()
    data["t"][:] = [9, 2, 1, 8, 3]
    data["Pz"][:] = [-5, 6, 0, -8, 2]
    for i, tree in enumerate(ref.PLANE_PATHS.values()):
        _write(path, data, tree=tree, update=bool(i))
    out, receipt = ref.read([path])
    assert set(out) == set(ref.COLUMNS)
    assert out["PDGid"].tolist() == [13, -13, 211, -211] * 7
    assert out["source_row"].tolist() == [0, 1, 2, 3] * 7
    assert out["iplane"].tolist() == np.repeat(np.arange(7), 4).tolist()
    assert out["ifile"].tolist() == [0] * 28
    assert out["t"].tolist() == [9, 2, 1, 8] * 7
    assert out["Pz"].tolist() == [-5, 6, 0, -8] * 7
    for name in ref.INTEGER + ("ifile", "iplane", "source_row"):
        assert out[name].dtype == np.int64
    for name in ref.STATE:
        assert out[name].dtype == np.float64
    assert receipt["identity"] == ["ifile", "EventID", "TrackID"]
    assert receipt["row_key"] == ["ifile", "EventID", "TrackID", "iplane", "source_row"]
    assert receipt["entries"] == 35 and receipt["selected_crossings"] == 28
    assert receipt["unique_selected_tracks"] == 4 and receipt["represented_eventIDs"] == 4
    assert receipt["recrossing_count"] == 0
    assert receipt["selected_species"]["-13"] == {"crossings": 7, "unique_tracks": 1}
    assert receipt["selected_weights"]["unitweights"]
    for gate in ("geometry", "cuts", "direction", "normalization"):
        assert receipt[f"{gate}_unverified"] is True
    assert "no G4BL-to-Mu2e transform" in receipt["coordinates"]
    assert "not POT exposure" in receipt["count_warning"]
    assert "parent PDG cannot be inferred" in receipt["parent_scope"]
    assert "gencount" not in receipt and "gencount" not in out
    assert "efficiency" not in out and "transmission" not in out
    assert json.loads(json.dumps(receipt, allow_nan=False))["schema"] == "ts-g4bl-crossings-v1"


def test_highest_cycle_full_inventory_and_hash(tmp_path):
    path = tmp_path / "cycles.root"
    _write(path, _rows(1))
    _write(path, _rows(5), update=True)
    _write(path, _rows(2), tree="Other/Unused", update=True)
    _write(path, _rows(0), tree="Other/Unused", update=True)
    with uproot.update(path) as root:
        root["description"] = "not a tree"
    out, receipt = _read(path)
    source = receipt["files"][0]
    plane = source["planes"][0]
    assert plane["selected_key"] == "VirtualDetector/Coll_01_DetIn;2"
    assert plane["cycle"] == 2 and plane["cycles"] == {"1": 1, "2": 5}
    assert plane["entries"] == 5 and plane["selected_crossings"] == 4
    inventory = source["tree_inventory"]
    assert set(inventory) == {"VirtualDetector/Coll_01_DetIn;1", "VirtualDetector/Coll_01_DetIn;2",
                              "Other/Unused;1", "Other/Unused;2"}
    assert inventory["Other/Unused;1"] == {"classname": "TTree", "num_entries": 2}
    assert inventory["Other/Unused;2"]["num_entries"] == 0
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source["bytes"] == path.stat().st_size
    assert source["path"] == str(path.resolve())
    assert out["EventID"].tolist() == [0, 1, 2, 3]


def test_recrossings_identical_rows_and_changing_weights_never_dedup(tmp_path):
    path = tmp_path / "recross.root"
    data = _rows(4)
    data["EventID"][:] = [7, 7, 7, 8]
    data["PDGid"][:] = 13
    data["t"][:] = [12, 1, 1, 8]
    data["Weight"][:] = [1, 0, 0, 2]
    for name in ref.BRANCHES:
        data[name][2] = data[name][1]  # Byte-identical crossing rows are still retained.
    _write(path, data)
    _write(path, data, tree=ref.PLANE_PATHS["COL1out"], update=True)
    out, receipt = ref.read([path], planes=["COL1in", "COL1out"])
    assert out["source_row"].tolist() == [0, 1, 2, 3] * 2
    assert out["Weight"].tolist() == [1, 0, 0, 2] * 2
    assert out["t"].tolist() == [12, 1, 1, 8] * 2
    assert receipt["selected_crossings"] == 8
    assert receipt["unique_selected_tracks"] == 2
    assert receipt["represented_eventIDs"] == 2
    assert receipt["recrossing_count"] == 4
    assert receipt["selected_tracks_with_varying_crossing_weights"] == 1
    weights = receipt["selected_weights"]
    assert weights == {"count": 8, "unitweights": False, "nonunit_count": 6,
                       "zero_count": 4, "min": 0.0, "max": 2.0}
    for plane in receipt["files"][0]["planes"]:
        assert plane["entries"] == 4 and plane["recrossing_count"] == 2
        assert plane["selected_tracks_with_varying_crossing_weights"] == 1


def test_cross_plane_weight_changes_are_reported_not_rejected(tmp_path):
    path = tmp_path / "plane_weights.root"
    data = _rows(1)
    _write(path, data)
    data["Weight"][:] = 0.25
    _write(path, data, tree=ref.PLANE_PATHS["COL1out"], update=True)
    out, receipt = ref.read([path], planes=["COL1in", "COL1out"])
    assert out["Weight"].tolist() == [1, 0.25]
    assert receipt["unique_selected_tracks"] == 1
    assert receipt["recrossing_count"] == 0
    assert receipt["selected_tracks_with_varying_crossing_weights"] == 1
    assert receipt["files"][0]["all_species_tracks_with_varying_crossing_weights"] == 1
    assert all(p["selected_tracks_with_varying_crossing_weights"] == 0
               for p in receipt["files"][0]["planes"])


@pytest.mark.parametrize("field", ["PDGid", "ParentID"])
@pytest.mark.parametrize("across_planes", [False, True])
@pytest.mark.parametrize("unselected", [False, True])
def test_all_species_identity_conflicts_fail(tmp_path, field, across_planes, unselected):
    path = tmp_path / "conflict.root"
    data = _rows(2)
    data["EventID"][:] = 3
    data["PDGid"][:] = 11 if unselected else 13
    if across_planes:
        _write(path, {k: v[:1] for k, v in data.items()})
        data[field][1] = (22 if unselected else -13) if field == "PDGid" else 9
        _write(path, {k: v[1:] for k, v in data.items()}, tree=ref.PLANE_PATHS["COL1out"], update=True)
        planes = ["COL1in", "COL1out"]
    else:
        data[field][1] = (22 if unselected else -13) if field == "PDGid" else 9
        _write(path, data)
        planes = ["COL1in"]
    with pytest.raises(ValueError, match=f"inconsistent {field}"):
        _read(path, planes=planes)


def test_file_scoped_event_track_identity_and_unresolved_parents(tmp_path):
    first, second = tmp_path / "a.root", tmp_path / "b.root"
    a = _rows(3)
    a["EventID"][:] = [0, 0, 1]
    a["TrackID"][:] = [1, 2, 1]
    a["PDGid"][:] = [13, -211, -13]
    a["ParentID"][:] = [99, 1, 0]
    _write(first, a)
    b = {k: v.copy() for k, v in a.items()}
    b["PDGid"][0] = 211  # Same identity in a different file is independent.
    b["ParentID"][0] = 88
    _write(second, b)
    out, receipt = ref.read([first, second], planes=["COL1in"])
    assert out["ifile"].tolist() == [0, 0, 0, 1, 1, 1]
    assert out["ParentID"].tolist() == [99, 1, 0, 88, 1, 0]
    assert receipt["unique_selected_tracks"] == 6
    assert receipt["represented_eventIDs"] == 4
    assert receipt["recrossing_count"] == 0
    assert not any("parent_pdg" in k.lower() for k in out)


@pytest.mark.parametrize("field", ref.INTEGER)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1.5])
def test_bad_integer_fields_before_species_filter(tmp_path, field, bad):
    data = _rows(1)
    data["PDGid"][:] = 11
    data[field][0] = bad
    path = tmp_path / "bad.root"
    _write(path, data)
    with pytest.raises(ValueError, match="nonfinite or nonintegral"):
        _read(path)


@pytest.mark.parametrize("field,value,message", [
    ("EventID", -1, "EventID must be nonnegative"),
    ("TrackID", 0, "TrackID must be positive"),
    ("TrackID", -1, "TrackID must be positive"),
    ("ParentID", -1, "ParentID must be nonnegative"),
    ("ParentID", 1, "self-parent"),
])
def test_identity_domains_before_selection(tmp_path, field, value, message):
    data = _rows(1)
    data["PDGid"][:] = 11
    data[field][0] = value
    path = tmp_path / "domain.root"
    _write(path, data)
    with pytest.raises(ValueError, match=message):
        _read(path)


@pytest.mark.parametrize("field", ["EventID", "TrackID", "ParentID"])
@pytest.mark.parametrize("dtype,bound", [(np.float32, 2 ** 24), (np.float64, 2 ** 53)])
@pytest.mark.parametrize("sign,delta", [(1, 0), (1, 2), (-1, 2)])
def test_float_integer_precision_boundaries(tmp_path, field, dtype, bound, sign, delta):
    data = _rows(1, dtype=dtype)
    data[field][0] = sign * (bound + delta)
    path = tmp_path / "precision.root"
    _write(path, data)
    with pytest.raises(ValueError, match="precision"):
        _read(path)


@pytest.mark.parametrize("dtype,bound", [(np.float32, 2 ** 24), (np.float64, 2 ** 53)])
def test_float_integer_just_below_limit_is_safe(tmp_path, dtype, bound):
    data = _rows(1, dtype=dtype)
    data["EventID"][0] = bound - 1
    data["TrackID"][0] = bound - 1
    data["ParentID"][0] = bound - 2
    path = tmp_path / "safe.root"
    _write(path, data)
    out, _ = _read(path)
    assert out["EventID"][0] == bound - 1
    assert out["TrackID"][0] == bound - 1
    assert out["ParentID"][0] == bound - 2


def test_int64_maximum_identity_preserved_exactly(tmp_path):
    data = _rows(1)
    for field in ref.INTEGER:
        data[field] = data[field].astype(np.int64)
    data["EventID"][0] = np.iinfo(np.int64).max
    data["TrackID"][0] = 2 ** 60 + 1
    data["ParentID"][0] = 2 ** 60 + 2
    path = tmp_path / "int64.root"
    _write(path, data)
    out, _ = _read(path)
    for field in ref.INTEGER:
        np.testing.assert_array_equal(out[field], data[field])


@pytest.mark.parametrize("dtype,value", [(np.uint64, 2 ** 63), (np.float64, 2.0 ** 63)])
def test_int64_overflow_rejected(tmp_path, dtype, value):
    data = _rows(1)
    data["EventID"] = np.array([value], dtype=dtype)
    path = tmp_path / "overflow.root"
    _write(path, data)
    with pytest.raises(ValueError, match="range|precision"):
        _read(path)


@pytest.mark.parametrize("field", ref.BRANCHES)
def test_scalar_shapes_required_even_for_unselected_species(tmp_path, field):
    data = _rows(2)
    data["PDGid"][:] = 11
    data[field] = np.ones((2, 2))
    path = tmp_path / "shape.root"
    _write(path, data)
    with pytest.raises(ValueError, match="one-dimensional|scalar per particle"):
        _read(path)


@pytest.mark.parametrize("field", ["EventID", "x", "Weight"])
def test_ragged_scalar_columns_rejected(tmp_path, field):
    import awkward as ak

    path = tmp_path / "ragged_scalar.root"
    data = _rows(2)
    data["PDGid"][:] = 11
    types = {k: v.dtype for k, v in data.items()}
    types[field] = "var * float64"
    data[field] = ak.Array([[1.0], [2.0, 3.0]])
    with uproot.recreate(path) as root:
        root.mktree(ref.PLANE_PATHS["COL1in"], types).extend(data)
    with pytest.raises(ValueError, match="one-dimensional|scalar per particle"):
        _read(path)


@pytest.mark.parametrize("field", ["EventID", "x", "Weight"])
def test_boolean_scalar_columns_rejected(tmp_path, field):
    data = _rows(1)
    data["PDGid"][:] = 11
    data[field] = np.ones(1, dtype=bool)
    path = tmp_path / "bool.root"
    _write(path, data)
    with pytest.raises(ValueError, match="one-dimensional|scalar per particle"):
        _read(path)


@pytest.mark.parametrize("field", ref.STATE)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_state_fields_finite_before_selection(tmp_path, field, bad):
    data = _rows(1)
    data["PDGid"][:] = 11
    data[field][0] = bad
    path = tmp_path / "nonfinite.root"
    _write(path, data)
    with pytest.raises(ValueError, match="finite numeric scalar"):
        _read(path)


def test_negative_weight_fails_for_unselected_species(tmp_path):
    data = _rows(1)
    data["Weight"][:] = -0.001
    data["PDGid"][:] = 11
    path = tmp_path / "negative.root"
    _write(path, data)
    with pytest.raises(ValueError, match="Weight must be nonnegative"):
        _read(path)


@pytest.mark.parametrize("n", [0, 2])
def test_empty_trees_and_no_selected_rows(tmp_path, n):
    data = _rows(n)
    data["PDGid"][:] = 11
    path = tmp_path / "empty.root"
    _write(path, data)
    out, receipt = _read(path)
    assert all(v.shape == (0,) and v.dtype.kind != "O" for v in out.values())
    assert receipt["entries"] == n and receipt["selected_crossings"] == 0
    assert receipt["unique_selected_tracks"] == 0 and receipt["represented_eventIDs"] == 0
    assert receipt["recrossing_count"] == 0
    assert receipt["selected_weights"]["min"] is None
    assert receipt["selected_weights"]["max"] is None
    assert receipt["files"][0]["planes"][0]["cycles"] == {"1": n}
    json.dumps(receipt, allow_nan=False)


def test_all_species_weight_ambiguity_reported_when_selection_empty(tmp_path):
    data = _rows(2)
    data["PDGid"][:] = 11
    data["EventID"][:] = 0
    data["Weight"][:] = [0, 7]
    path = tmp_path / "unselected_weights.root"
    _write(path, data)
    _, receipt = _read(path)
    source = receipt["files"][0]
    assert source["all_species_tracks_with_varying_crossing_weights"] == 1
    assert source["all_crossing_weights"]["min"] == 0
    assert source["all_crossing_weights"]["max"] == 7
    assert source["all_species_recrossing_count"] == 1
    assert source["selected_crossings"] == 0


@pytest.mark.parametrize("branch", ref.BRANCHES)
def test_missing_required_branches_fail_closed(tmp_path, branch):
    data = _rows(1)
    del data[branch]
    path = tmp_path / "missing.root"
    _write(path, data)
    with pytest.raises(ValueError, match="missing required branches"):
        _read(path)


@pytest.mark.parametrize("kwargs", [
    {"planes": []}, {"planes": [""]}, {"planes": [" "]}, {"planes": ["COL1in", "COL1in"]},
    {"planes": ["COL1in", "VirtualDetector/Coll_01_DetIn"]},
    {"planes": ["VirtualDetector/Coll_01_DetIn;1"]}, {"planes": ["/VirtualDetector/a"]},
    {"planes": ["VirtualDetector//a"]}, {"planes": ["VirtualDetector/../a"]},
    {"pdgs": []}, {"pdgs": [13, 13]}, {"pdgs": [11]}, {"pdgs": [13.5]},
])
def test_invalid_requests(tmp_path, kwargs):
    path = tmp_path / "request.root"
    _write(path, _rows(1))
    with pytest.raises(ValueError):
        _read(path, **kwargs)


def test_arbitrary_tree_path_z3712_and_required_planes(tmp_path):
    path = tmp_path / "z.root"
    _write(path, _rows(1), tree="VirtualDetector/Z3712")
    out, receipt = ref.read([path], planes=["VirtualDetector/Z3712"])
    assert out["PDGid"].tolist() == [13]
    assert receipt["plane_index"] == {"0": "VirtualDetector/Z3712"}
    with pytest.raises(ValueError, match="Missing tree"):
        ref.read([path])
    with pytest.raises(ValueError, match="Missing tree"):
        ref.read([path], planes=["VirtualDetector/Z3712", "VirtualDetector/Missing"])


def test_duplicate_paths_and_byte_identical_inputs_fail(tmp_path):
    path = tmp_path / "input.root"
    _write(path, _rows(1))
    copy = tmp_path / "copy.root"
    shutil.copyfile(path, copy)
    with pytest.raises(ValueError, match="Duplicate input path"):
        ref.read([path, path], planes=["COL1in"])
    with pytest.raises(ValueError, match="Byte-identical"):
        ref.read([path, copy], planes=["COL1in"])
    alias = tmp_path / "alias.root"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="Duplicate input path"):
        ref.read([path, alias], planes=["COL1in"])


def test_no_input_missing_input_and_remote_rejected_before_io(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="At least one"):
        ref.read([])
    with pytest.raises(ValueError, match="Missing or non-file"):
        ref.read([tmp_path / "missing.root"])
    with pytest.raises(ValueError, match="Missing or non-file"):
        ref.read([tmp_path])
    def forbidden(*args, **kwargs):
        raise AssertionError("No remote open/stat/hash permitted")
    monkeypatch.setattr(ref, "_source", forbidden)
    monkeypatch.setattr(uproot, "open", forbidden)
    for remote in ("root://server/file.root", "https://server/file.root", "s3://bucket/file.root"):
        with pytest.raises(ValueError, match="local/staged"):
            ref.read([remote])


def test_cli_npz_json_metadata_and_hashes(tmp_path):
    path, output = tmp_path / "input.root", tmp_path / "subdir" / "out.npz"
    _write(path, _rows(), tree="VirtualDetector/Z3712")
    ref.main([str(path), "--plane", "VirtualDetector/Z3712", "--pdg", "13",
              "--pdg=-211", "--out", str(output)])
    receipt = json.loads(output.with_suffix(".json").read_text())
    with np.load(output, allow_pickle=False) as saved:
        assert saved["PDGid"].tolist() == [13, -211]
        assert saved["source_row"].tolist() == [0, 3]
        metadata = json.loads(saved["metadata_json"].item())
        assert metadata["planes"] == ["VirtualDetector/Z3712"]
        assert metadata["pdgs"] == [13, -211]
        assert metadata["normalization_unverified"]
        assert all(saved[k].dtype.kind != "O" for k in saved.files)
    assert receipt["selected_crossings"] == 2
    assert receipt["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["files"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_default_planes_and_empty_selection(tmp_path):
    path, output = tmp_path / "input.root", tmp_path / "empty.npz"
    data = _rows(1)
    data["PDGid"][:] = 11
    for i, tree in enumerate(ref.PLANE_PATHS.values()):
        _write(path, data, tree=tree, update=bool(i))
    ref.main([str(path), "--out", str(output)])
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert receipt["planes"] == list(ref.PLANE_PATHS.values())
    assert receipt["selected_crossings"] == 0 and receipt["entries"] == 7
    with np.load(output, allow_pickle=False) as saved:
        assert saved["EventID"].shape == (0,)


@pytest.mark.parametrize("kind", ["npz_input", "json_input", "hardlink", "bad_suffix", "missing"])
def test_cli_failures_do_not_clobber_inputs(tmp_path, kind):
    path = tmp_path / ("input.json" if kind == "json_input" else "input.npz")
    _write(path, _rows(1))
    before = path.read_bytes()
    output = path.with_suffix(".npz") if kind in ("npz_input", "json_input") else tmp_path / "out.npz"
    if kind == "hardlink":
        os.link(path, output)
    elif kind == "bad_suffix":
        output = tmp_path / "bad.root"
    source = tmp_path / "missing.root" if kind == "missing" else path
    with pytest.raises(SystemExit) as exc:
        ref.main([str(source), "--plane", "COL1in", "--out", str(output)])
    assert exc.value.code != 0
    assert path.read_bytes() == before
    if kind in ("bad_suffix", "missing"):
        assert not output.exists()


def test_import_does_not_require_jax_or_uproot():
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")
    code = """
import importlib.abc
import sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('jax', 'jaxlib', 'uproot'):
            raise AssertionError('unexpected eager dependency: ' + fullname)
sys.meta_path.insert(0, BlockImports())
import beamfastsim.tsg4bl
assert 'jax' not in sys.modules
assert 'uproot' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
