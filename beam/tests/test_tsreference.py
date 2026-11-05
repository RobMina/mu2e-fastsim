"""Synthetic ROOT contract tests; no production inputs, transport or rate claims."""
import hashlib
import json

import numpy as np
import pytest
import uproot

from beamfastsim import tsreference as ref


def _rows(n=5, dtype=np.float64, width=8):
    a = {k: np.zeros(n, dtype=dtype) for k in ref.INTEGER + ref.BORN + ref.END + ref.OPTIONAL}
    a.update(run=np.full(n, 7, dtype=dtype), subrun=np.full(n, 3, dtype=dtype),
             evt=np.arange(n, dtype=dtype) + 1, trk=np.ones(n, dtype=dtype),
             parent_id=np.full(n, 10, dtype=dtype), parent_pdg=np.full(n, -211, dtype=dtype),
             pdg=np.resize(np.array([13, -13, 211, -211, 11], dtype=dtype), n),
             code=np.full(n, 56, dtype=dtype), codestop=np.full(n, 99, dtype=dtype))
    for j, k in enumerate(ref.BORN + ref.END):
        a[k] = np.arange(n, dtype=dtype) + j + 10
    a["isvd"] = np.ones((n, width), dtype=dtype)
    for j, k in enumerate(ref.VD[1:]):
        a[k] = np.arange(n * width, dtype=dtype).reshape(n, width) + j * 100
    return a


def _write(path, a, tree="mutree/ntpart", update=False):
    types = {k: np.dtype((v.dtype, v.shape[1:])) if v.ndim > 1 else v.dtype for k, v in a.items()}
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        root.mktree(tree, types).extend(a)


def test_highest_cycle_hash_selection_and_stored_states(tmp_path):
    path = tmp_path / "reference.root"
    _write(path, _rows(1))
    rows = _rows()
    rows["tvd"][0, :] = np.arange(8)[::-1]
    _write(path, rows, update=True)
    out, receipt = ref.read([path], vds=[1, 8])
    source = receipt["files"][0]
    assert source["selected_key"] == "mutree/ntpart;2"
    assert source["cycle"] == 2 and source["cycles"] == {"1": 1, "2": 5}
    assert source["rows"] == 5 and source["selected_rows"] == 4
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert source["bytes"] == path.stat().st_size
    assert out["pdg"].tolist() == [13, -13, 211, -211]
    assert out["q"].tolist() == [-1, 1, 1, -1]
    assert out["source_row"].tolist() == [0, 1, 2, 3]
    for k in ref.INTEGER + ref.BORN + ref.END + ref.OPTIONAL:
        np.testing.assert_array_equal(out[k], rows[k][:4])
    for k in ref.INTEGER:
        assert out[k].dtype == np.int64
    assert out["vd1_stored_first_time"][0] == 7
    assert out["vd8_stored_first_time"][0] == 0
    assert out["vd1_hit"].dtype == bool
    assert "vd2_hit" not in out
    assert receipt["retained_species_counts"] == {"13": 1, "-13": 1, "211": 1, "-211": 1}
    assert receipt["stored_vd_hit_counts"]["1"]["13"] == 1
    assert receipt["survivor_conditioned"] and receipt["normalisation_unverified"]
    assert "chronology" in receipt["vd_state"]
    assert "gencount" not in out and "gencount" not in receipt
    assert "efficiency" not in out and "pot" not in out


def test_file_event_track_identity_and_parent_ancestry(tmp_path):
    a = _rows(4)
    a["evt"][:] = [1, 1, 1, 1]
    a["trk"][:] = [10, 11, 10, 10]
    a["subrun"][:] = [3, 3, 4, 3]
    a["run"][:] = [7, 7, 7, 8]
    a["pdg"][:] = [-211, 13, 13, 13]
    a["parent_id"][:] = [0, 10, 0, 0]
    a["parent_pdg"][:] = [0, -211, 0, 0]
    first, second = tmp_path / "a.root", tmp_path / "b.root"
    _write(first, a)
    _write(second, a)
    out, receipt = ref.read([first, second])
    assert out["ifile"].tolist() == [0] * 4 + [1] * 4
    assert out["parent_id"].tolist() == [0, 10, 0, 0] * 2
    assert out["parent_pdg"].tolist() == [0, -211, 0, 0] * 2
    assert receipt["identity"] == ["ifile", "run", "subrun", "evt", "trk"]
    selected, _ = ref.read([first], pdgs=[13])
    assert selected["parent_id"][0] == 10
    assert -211 not in selected["pdg"]


@pytest.mark.parametrize("unselected", [False, True])
def test_duplicate_identity_is_error_not_deduplication(tmp_path, unselected):
    a = _rows(2)
    a["evt"][:] = 1
    if unselected:
        a["pdg"][:] = 11
    path = tmp_path / "duplicate.root"
    _write(path, a)
    with pytest.raises(ValueError, match="duplicate particle identity"):
        ref.read([path])


@pytest.mark.parametrize("field", ref.INTEGER)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1.5])
def test_nonfinite_nonintegral_ids_fail_before_selection(tmp_path, field, bad):
    a = _rows(1)
    a[field][0] = bad
    a["pdg"][0] = bad if field == "pdg" else 11
    path = tmp_path / "bad.root"
    _write(path, a)
    with pytest.raises(ValueError, match="nonfinite or nonintegral"):
        ref.read([path])


@pytest.mark.parametrize("field", ["run", "subrun", "evt", "trk", "parent_id"])
@pytest.mark.parametrize("value", [2 ** 24, 2 ** 24 + 1, 2 ** 24 + 2, -(2 ** 24 + 2)])
def test_float32_large_ids_rejected(tmp_path, field, value):
    a = _rows(1, dtype=np.float32)
    a[field][0] = value
    path = tmp_path / "precision.root"
    _write(path, a)
    with pytest.raises(ValueError, match="precision"):
        ref.read([path])


def test_float32_small_exact_ids_and_int64_large_ids(tmp_path):
    path = tmp_path / "exact.root"
    _write(path, _rows(1, dtype=np.float32))
    out, _ = ref.read([path])
    assert out["evt"].dtype == np.int64 and out["evt"].tolist() == [1]
    a = _rows(1)
    for k in ref.INTEGER:
        a[k] = a[k].astype(np.int64)
    a["evt"][0] = 2 ** 60 + 1
    a["parent_id"][0] = 2 ** 60 + 2
    _write(path, a)
    out, _ = ref.read([path])
    assert out["evt"][0] == 2 ** 60 + 1 and out["parent_id"][0] == 2 ** 60 + 2


@pytest.mark.parametrize("dtype,value", [(np.float64, 2.0 ** 63), (np.uint64, 2 ** 63)])
def test_out_of_int64_range_fails(tmp_path, dtype, value):
    a = _rows(1)
    a["evt"] = np.array([value], dtype=dtype)
    path = tmp_path / "overflow.root"
    _write(path, a)
    with pytest.raises(ValueError, match="range|precision"):
        ref.read([path])


@pytest.mark.parametrize("value", [2 ** 53, 2 ** 53 + 2, -(2 ** 53 + 2)])
def test_float64_precision_boundary_fails(tmp_path, value):
    a = _rows(1)
    a["evt"][0] = value
    path = tmp_path / "float64_precision.root"
    _write(path, a)
    with pytest.raises(ValueError, match="precision"):
        ref.read([path])


@pytest.mark.parametrize("branch", ["evt", "parent_id", "x", "xstop"])
def test_malformed_scalar_dimensions(tmp_path, branch):
    a = _rows(2)
    a[branch] = np.ones((2, 2))
    path = tmp_path / "scalar_dimensions.root"
    _write(path, a)
    with pytest.raises(ValueError, match="one-dimensional|scalar per particle"):
        ref.read([path])


def test_no_hit_rows_kept_and_missing_optional_state(tmp_path):
    a = _rows(4)
    a["isvd"][:] = 0
    for k in ref.VD[1:]:
        a[k][:] = np.nan
    for k in ref.OPTIONAL:
        del a[k]
    path = tmp_path / "nohit.root"
    _write(path, a)
    out, receipt = ref.read([path])
    assert len(out["evt"]) == 4
    for v in range(1, 9):
        assert not out[f"vd{v}_hit"].any()
        assert np.isnan(out[f"vd{v}_stored_first_x"]).all()
    assert not out["has_code"].any() and not out["has_codestop"].any()
    assert np.isnan(out["pstop"]).all()
    assert receipt["files"][0]["optional_branches"] == []


@pytest.mark.parametrize("n", [0, 2])
def test_empty_selection_and_empty_tree_have_stable_shapes(tmp_path, n):
    a = _rows(n)
    a["pdg"][:] = 11
    path = tmp_path / "empty.root"
    _write(path, a)
    out, receipt = ref.read([path])
    assert all(v.shape == (0,) and v.dtype.kind != "O" for v in out.values())
    assert receipt["selected_rows"] == 0 and receipt["rows"] == n
    assert out["evt"].dtype == np.int64 and out["vd8_hit"].dtype == bool


@pytest.mark.parametrize("branch,shape", [("xvd", (2, 7)), ("isvd", (2,)), ("tvd", (2, 8, 1))])
def test_malformed_vd_dimensions(tmp_path, branch, shape):
    a = _rows(2)
    a[branch] = np.zeros(shape)
    path = tmp_path / "dimensions.root"
    _write(path, a)
    with pytest.raises(ValueError, match="dimensions"):
        ref.read([path])


def test_ragged_vd_dimensions(tmp_path):
    import awkward as ak

    path = tmp_path / "ragged.root"
    a = _rows(2)
    with uproot.recreate(path) as root:
        types = {k: np.dtype((v.dtype, v.shape[1:])) if v.ndim > 1 else v.dtype for k, v in a.items()}
        types["xvd"] = "var * float64"
        a["xvd"] = ak.Array([[1.0] * 8, [2.0] * 7])
        root.mktree("mutree/ntpart", types).extend(a)
    with pytest.raises(ValueError, match="ragged|dimensions"):
        ref.read([path])


@pytest.mark.parametrize("n", [0, 2])
def test_rectangular_variable_length_vd_branches(tmp_path, n):
    import awkward as ak

    path = tmp_path / "variable.root"
    a = _rows(n)
    types = {k: v.dtype for k, v in a.items() if k not in ref.VD}
    types.update({k: "var * float64" for k in ref.VD})
    with uproot.recreate(path) as root:
        tree = root.mktree("mutree/ntpart", types)
        if n:
            tree.extend({k: ak.unflatten(v.reshape(-1), np.full(n, v.shape[1])) if k in ref.VD else v
                         for k, v in a.items()})
    out, receipt = ref.read([path])
    assert len(out["evt"]) == n and receipt["rows"] == n
    assert out["vd8_hit"].shape == (n,)


def test_optional_availability_stays_row_aligned_across_files(tmp_path):
    first, second = tmp_path / "present.root", tmp_path / "absent.root"
    a = _rows(1)
    _write(first, a)
    for k in ref.OPTIONAL:
        del a[k]
    _write(second, a)
    out, _ = ref.read([first, second])
    assert out["has_code"].tolist() == [True, False]
    assert out["code"].tolist() == [56, 0]
    assert out["has_pstop"].tolist() == [True, False]
    assert np.isnan(out["pstop"][1])


@pytest.mark.parametrize("status", [np.nan, 2, -1])
def test_invalid_hit_status(tmp_path, status):
    a = _rows(1)
    a["isvd"][0, 0] = status
    path = tmp_path / "status.root"
    _write(path, a)
    with pytest.raises(ValueError, match="binary hit status"):
        ref.read([path])


def test_nonfinite_hit_state_fails_but_unrequested_vd_is_not_used(tmp_path):
    a = _rows(1)
    a["xvd"][0, 7] = np.nan
    path = tmp_path / "state.root"
    _write(path, a)
    with pytest.raises(ValueError, match="nonfinite hit state"):
        ref.read([path])
    out, _ = ref.read([path], vds=[1])
    assert out["vd1_hit"].tolist() == [True]


@pytest.mark.parametrize("branch", ["parent_id", "parent_pdg", "x", "xstop", "isvd"])
def test_required_branches_fail_closed(tmp_path, branch):
    a = _rows(1)
    del a[branch]
    path = tmp_path / "missing.root"
    _write(path, a)
    with pytest.raises(ValueError, match="missing required branches"):
        ref.read([path])


@pytest.mark.parametrize("kwargs", [{"vds": [0]}, {"vds": [9]}, {"vds": [1, 1]},
                                     {"vds": []}, {"vds": [1.5]}, {"pdgs": [11]},
                                     {"pdgs": []}, {"pdgs": [13, 13]},
                                     {"tree": "mutree/ntpart;1"}])
def test_invalid_requests(tmp_path, kwargs):
    path = tmp_path / "input.root"
    _write(path, _rows(1))
    with pytest.raises(ValueError):
        ref.read([path], **kwargs)


def test_missing_tree_and_no_inputs(tmp_path):
    path = tmp_path / "input.root"
    _write(path, _rows(1))
    with pytest.raises(ValueError, match="Missing tree"):
        ref.read([path], tree="other/ntpart")
    with pytest.raises(ValueError, match="At least one"):
        ref.read([])


def test_cli_npz_and_json_receipt(tmp_path):
    source, output = tmp_path / "input.root", tmp_path / "out.npz"
    _write(source, _rows(), tree="upstream/ntpart")
    ref.main([str(source), "--tree", "upstream/ntpart", "--vd", "1", "--vd", "8",
              "--pdg", "13", "--pdg=-211", "--out", str(output)])
    receipt = json.loads(output.with_suffix(".json").read_text())
    with np.load(output, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata_json"].item())
        assert metadata["survivor_conditioned"] and metadata["normalisation_unverified"]
        assert metadata["vds"] == [1, 8]
        assert metadata["pdgs"] == [13, -211]
        assert saved["pdg"].tolist() == [13, -211]
        assert saved["parent_id"].tolist() == [10, 10]
        assert all(saved[k].dtype.kind != "O" for k in saved.files)
    assert receipt["selected_rows"] == 2
    assert receipt["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_cli_default_vds_and_empty_selection(tmp_path):
    path = tmp_path / "input.root"
    a = _rows(1)
    a["pdg"][:] = 11
    _write(path, a)
    output = tmp_path / "empty.npz"
    ref.main([str(path), "--out", str(output)])
    with np.load(output) as saved:
        assert saved["evt"].shape == (0,)
        assert "vd8_hit" in saved
    assert json.loads(output.with_suffix(".json").read_text())["selected_rows"] == 0
