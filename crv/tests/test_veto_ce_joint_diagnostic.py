"""Synthetic local fixtures only; production pins overridden solely in these tests."""
import ast
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crvfastsim.veto import ce_joint_diagnostic as J


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays():
    return dict(run=np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),
        subrun=np.array([2, 2, 2, 2, 2, 2], dtype=np.int64),
        event=np.array([1, 2, 3, 4, 5, 2**32 + 1], dtype=np.int64),
        source_index=np.array([0, 0, 0, 1, 1, 1], dtype=np.int16),
        measured_nprotons=np.zeros(6, dtype=np.int32),
        measured_pbtime=np.array([-240, -230, -220, -210, -200, -190], dtype=np.float32),
        measured_pbterr=np.full(6, 0.5, dtype=np.float32),
        nProtonTCs=np.array([0, 1, 2, 3, 5, 10], dtype=np.int32),
        nTrackerHits=np.array([0, 1, 32, 64, 4096, 4097], dtype=np.int32),
        nCaloHitsD0=np.array([0, 1, 32, 64, 128, 256], dtype=np.int32),
        nCaloHitsD1=np.array([0, 1, 32, 64, 128, 512], dtype=np.int32),
        caloEnergy=np.array([0, 1, 32, 64, 128, 1024], dtype=np.int32),
        caphriIsValid=np.ones(6, dtype=np.bool_))


def table_selected():
    return pa.table(dict(run=pa.array([1] * 5, pa.int64()), subrun=pa.array([2] * 5, pa.int64()),
        event=pa.array([1, 2, 3, 2**32 + 1, 999], pa.int64()),
        p_mid=pa.array([[85., 86., 200.], [100., 100., 100., 100.], [], [86., 199.], [99.]], pa.list_(pa.float64())),
        t_mid=pa.array([[13., -1., 15.], [0., 100., 1999., 2000.], [], [2500., 100.], [90.]], pa.list_(pa.float64())),
        coincs=pa.array([[b"never decode this"]] * 5)))


class Saved:
    def __init__(self, tmp, monkeypatch):
        self.mp = monkeypatch
        self.root = tmp
        for name in ("reports", "rows", "tracks", "ledger", "output"):
            (tmp / name).mkdir()
        self.sp = tmp / "reports" / "selection.json"
        self.rp = tmp / "rows" / "report.json"
        self.npz = tmp / "rows" / "rows.npz"
        self.selected = tmp / "tracks" / "tracks.parquet"
        self.ledger = tmp / "ledger" / "ledger.parquet"
        self.out = tmp / "output" / "joint.json"
        self.rows = arrays()
        self.selected_table = table_selected()
        self.ledger_arrays = {k: self.rows[k].copy() for k in J.IDS}
        self.ledger_arrays.update(nprotons=np.array([0., 1., 2., 3., 4., 5.]),
            pbtime=np.arange(6, dtype=np.float64), measured_nprotons=np.zeros(6, dtype=np.float64),
            measured_pbtime=self.rows["measured_pbtime"].astype(np.float64))
        self.report = dict(selection=dict(path=str(self.selected)), exposure_ledger=dict(path=str(self.ledger)),
            momentum_window=dict(pmin=85., pmax=200., strict=True, selected_events=4, selected_tracks=8),
            diagnostic=dict(ntuple_events=6, selected_events_matched=3, selected_tracks_matched=7,
                selected_events_not_in_supplied_ntuples=1, pbi_quantile_strata=[
                    dict(pbi_lo=0., pbi_hi=2., upper_inclusive=False, ntuple_events=2, selected_events=2, selected_tracks=5),
                    dict(pbi_lo=2., pbi_hi=5., upper_inclusive=True, ntuple_events=4, selected_events=1, selected_tracks=2)]),
            ntuples=[dict(path=f"/pnfs/inherited_{i}.root", uuid=f"inherited-{i}",
                         trees={"EventNtuple/ntuple": dict(entries=3)}) for i in range(2)])
        self.rows_report = dict(schema="ce-proxy-rows-v1", availability_report=dict(sha256=J.AVAILABILITY_SHA256),
            files=[dict(index=i, path=r["path"], uuid=r["uuid"], entries=3) for i, r in enumerate(self.report["ntuples"])])
        self.mp.setattr(J, "EXPECTED_ENTRIES", 6)
        self.repin()

    def repin(self):
        pq.write_table(self.selected_table, self.selected)
        pq.write_table(pa.table(self.ledger_arrays), self.ledger)
        np.savez(self.npz, **self.rows)
        self.report["selection"]["sha256"] = digest(self.selected)
        self.report["exposure_ledger"]["sha256"] = digest(self.ledger)
        self.mp.setattr(J, "SELECTED_SHA256", digest(self.selected))
        self.mp.setattr(J, "LEDGER_SHA256", digest(self.ledger))
        self.sp.write_text(json.dumps(self.report))
        self.mp.setattr(J, "SELECTION_SHA256", digest(self.sp))
        self.rows_report["selection_report"] = dict(sha256=digest(self.sp))
        self.rows_report["rows"] = dict(path=str(self.npz), sha256=digest(self.npz), entries=6,
            columns={k: dict(dtype=v.dtype.str, shape=list(v.shape), sha256=J.sha(v.tobytes())) for k, v in self.rows.items()})
        self.rp.write_text(json.dumps(self.rows_report))

    def run(self, **kwargs):
        return J.run(self.sp, self.rp, self.out, expected_selection_sha256=digest(self.sp),
                     expected_rows_sha256=digest(self.rp), **kwargs)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return Saved(tmp_path, monkeypatch)


def test_production_pins_and_cli():
    assert J.SELECTION_SHA256 == "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
    assert J.SELECTED_SHA256 == "7f60b88899f35bea7cf0a438aa18250ff0021ad229bc3c000bfd180fa386a2ba"
    assert J.LEDGER_SHA256 == "b9f94c948099cdeddb1e27634bb4f2f232619aa11d854ae5a96d4d5339e955cf"
    assert J.EXPECTED_ENTRIES == 91226
    with pytest.raises(SystemExit):
        J.main(["x", "y", "--out", "z"])
    with pytest.raises(ValueError, match="hardbound"):
        J.run("x", "y", "z", expected_selection_sha256="f" * 64, expected_rows_sha256="a" * 64)


def test_integration_accounting(saved):
    result = saved.run()
    assert json.loads(saved.out.read_bytes()) == result
    c = result["counts"]
    assert c["full_selected_rows"] == 5 and c["full_selected_tracks"] == 10
    assert c["exactly_85"] == c["exactly_200"] == 1
    assert c["selected_tracks_matched"] == 7 and c["selected_tracks_unmatched"] == 1
    assert c["selected_rows_unmatched_including_empty"] == 1
    assert c["retained_events_without_selected_row"] == 2
    assert c["retained_events_without_window_tracks"] == 3
    for grid in result["grids"].values():
        n, a, t = [np.array(grid[k]) for k in ("N_all_retained", "A_events_with_tracks", "T_tracks")]
        assert (n.sum(), a.sum(), t.sum()) == (6, 3, 7)
        assert np.array_equal(np.array(grid["selected_track_raw_time_histogram"]).sum(axis=-1), t)
        assert np.array_equal(np.array(grid["multiplicity_0_1_2_3_ge4"]).sum(axis=-1), n)
        assert np.array(grid["multiplicity_0_1_2_3_ge4"])[..., 4].sum() == 1
        assert grid["unsupported"]["joint_support"]["event_fraction"] == 1
        assert grid["unsupported"]["joint_support"]["track_fraction"] == 1
    assert [x["selected_tracks"] for x in result["sources"]] == [5, 2]
    assert result["caphriIsValid"] == dict(zero=0, one=6, constant=True)
    assert all(result[k] is None for k in ("F", "weight_model", "calibration", "finite_library_uncertainty"))
    assert result["matched_track_identity"]["count"] == 7


def test_no_coincs_decode(saved, monkeypatch):
    real = J.pq.ParquetFile
    seen = []
    class Guard:
        def __init__(self, *args, **kwargs):
            assert kwargs["thrift_string_size_limit"] == J.FOOTER_CAP
            self.inner = real(*args, **kwargs)
        def __getattr__(self, key):
            return getattr(self.inner, key)
        def iter_batches(self, *args, **kwargs):
            columns = kwargs["columns"]
            assert columns in [list(J.SELECTED_KEYS), list(J.LEDGER_KEYS)]
            assert "coincs" not in columns
            seen.append(columns)
            return self.inner.iter_batches(*args, **kwargs)
    monkeypatch.setattr(J.pq, "ParquetFile", Guard)
    saved.run()
    assert len(seen) == 2


def test_permutation_invariance(saved):
    first = saved.run()
    saved.out.unlink()
    order = np.array([5, 1, 3, 2, 0, 4])
    saved.rows = {k: v[order] for k, v in saved.rows.items()}
    saved.ledger_arrays = {k: v[::-1] for k, v in saved.ledger_arrays.items()}
    saved.selected_table = saved.selected_table.take(pa.array([4, 0, 3, 2, 1]))
    saved.repin()
    second = saved.run()
    for name in ("counts", "grids", "sources", "matched_track_identity"):
        assert first[name] == second[name]


@pytest.mark.parametrize("target", ["sp", "rp", "npz", "selected", "ledger"])
def test_file_pin_mutation(saved, target):
    path = getattr(saved, target)
    if target in ("sp", "rp"):
        pin_sp, pin_rp = digest(saved.sp), digest(saved.rp)
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError, match="SHA256"):
            J.run(saved.sp, saved.rp, saved.out, expected_selection_sha256=pin_sp, expected_rows_sha256=pin_rp)
    else:
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError, match="SHA256"):
            saved.run()
    assert not saved.out.exists()


@pytest.mark.parametrize("target", ["sp", "rp", "npz", "selected", "ledger", "code"])
def test_recheck_mutation(saved, monkeypatch, target):
    real = J.Evidence.recheck
    if target == "code":
        code = saved.root / "copy" / "pkg" / "subpkg" / "diagnostic.py"
        code.parent.mkdir(parents=True)
        code.write_bytes(Path(J.__file__).read_bytes())
        monkeypatch.setattr(J, "__file__", str(code))
    else:
        code = getattr(saved, target)
    def mutate(evidence):
        code.write_bytes(code.read_bytes() + b" ")
        real(evidence)
    monkeypatch.setattr(J.Evidence, "recheck", mutate)
    with pytest.raises(ValueError, match="changed"):
        saved.run()
    assert not saved.out.exists()


@pytest.mark.parametrize("directory", ["reports", "rows", "tracks", "ledger"])
def test_unsafe_output(saved, directory):
    saved.out = saved.root / directory / "new.json"
    with pytest.raises(ValueError, match="output under"):
        saved.run()
    assert not saved.out.exists()


def test_output_existing_and_symlink(saved):
    saved.out.write_bytes(b"keep")
    with pytest.raises(ValueError, match="new output"):
        saved.run()
    assert saved.out.read_bytes() == b"keep"
    saved.out.unlink()
    linked = saved.root / "linked"
    linked.symlink_to(saved.out.parent, target_is_directory=True)
    saved.out = linked / "new.json"
    with pytest.raises(ValueError, match="symlink"):
        saved.run()


@pytest.mark.parametrize("path", ["root://host/file", "https://host/f", "/pnfs/x", "/net/server/x", "/x/test.root", "//server/share"])
def test_nonlocal_paths(path):
    with pytest.raises(ValueError):
        J.local_path(path)


def test_input_symlink(saved):
    original = saved.npz.with_name("original.npz")
    saved.npz.rename(original)
    saved.npz.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        saved.run()


@pytest.mark.parametrize("field,value", [("schema", "other"), ("selection_report", {"sha256": "a" * 64}),
                                          ("availability_report", {"sha256": "b" * 64})])
def test_rows_report_bindings(saved, field, value):
    saved.rows_report[field] = value
    saved.rp.write_text(json.dumps(saved.rows_report))
    with pytest.raises(ValueError, match="differs"):
        saved.run()


def test_rows_outside_report_parent(saved):
    saved.rows_report["rows"]["path"] = str(saved.root / "elsewhere.npz")
    saved.rp.write_text(json.dumps(saved.rows_report))
    with pytest.raises(ValueError, match="report parent"):
        saved.run()


@pytest.mark.parametrize("kind", ["duplicate", "extra", "object", "shape", "member_cap", "total_cap", "header_size", "declared_dtype"])
def test_zip_rejections(saved, kind, monkeypatch):
    description = saved.rows_report["rows"]
    data = saved.npz.read_bytes()
    if kind in ("duplicate", "extra"):
        stream = io.BytesIO(data)
        with zipfile.ZipFile(stream, "a") as z:
            if kind == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate"):
                    z.writestr("run.npy", b"bad")
            else:
                z.writestr("extra.npy", b"bad")
        data = stream.getvalue()
    elif kind in ("object", "shape"):
        values = dict(saved.rows)
        values["run"] = np.array([object()] * 6) if kind == "object" else np.ones((6, 1))
        stream = io.BytesIO()
        np.savez(stream, **values)
        data = stream.getvalue()
    elif kind in ("member_cap", "total_cap"):
        monkeypatch.setattr(J, "ROWS_CAP", 10 if kind == "member_cap" else len(data) - 1)
    elif kind == "header_size":
        stream = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(stream, "w") as dest:
            for name in source.namelist():
                raw = source.read(name)
                if name == "run.npy":
                    raw = raw[:-1]
                dest.writestr(name, raw)
        data = stream.getvalue()
    else:
        description["columns"]["run"]["dtype"] = "<f8"
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        J.load_rows(data, description)


def test_rows_dtype_preserved(saved):
    actual = J.load_rows(saved.npz.read_bytes(), saved.rows_report["rows"])
    for name, expected in saved.rows.items():
        assert actual[name].dtype == expected.dtype
        assert actual[name].tobytes() == expected.tobytes()


@pytest.mark.parametrize("field,value", [("nProtonTCs", -1), ("caloEnergy", -1), ("measured_nprotons", -1),
    ("measured_pbterr", -0.5), ("measured_pbtime", np.nan), ("caphriIsValid", 2), ("source_index", -1), ("event", -1)])
def test_invalid_rows_domains(saved, field, value):
    if field == "caphriIsValid":
        saved.rows[field] = saved.rows[field].astype(np.int8)
    saved.rows[field][0] = value
    saved.repin()
    with pytest.raises(ValueError):
        saved.run()


@pytest.mark.parametrize("kind", ["selected", "ledger", "rows"])
def test_duplicate_full_ids_before_cut(saved, kind):
    if kind == "selected":
        # Empty event 3 becomes duplicate event 1; must reject even though empty.
        saved.selected_table = saved.selected_table.set_column(2, "event", pa.array([1, 2, 1, 2**32 + 1, 999]))
    else:
        (saved.ledger_arrays if kind == "ledger" else saved.rows)["event"][2] = 1
    saved.repin()
    with pytest.raises(ValueError, match="duplicate full"):
        saved.run()


def test_full_ids_beyond_32bit():
    keys = J.identities(dict(run=np.array([1, 1, 1]), subrun=np.array([2, 2, 3]),
                             event=np.array([1, 2**32 + 1, 1])))
    assert len(np.unique(keys)) == 3
    assert J.lookup(keys[::-1], keys).tolist() == [2, 1, 0]
    with pytest.raises(ValueError, match="signed-int64"):
        J.integer_array(np.array([2**63], dtype=np.uint64), "id")


@pytest.mark.parametrize("kind", ["null_list", "null_element", "misaligned", "nonfinite_time", "nonfinite_p", "too_many"])
def test_ragged_rejections(saved, kind):
    values = saved.selected_table["t_mid"].to_pylist()
    name = "t_mid"
    if kind == "null_list":
        values[0] = None
    elif kind == "null_element":
        values[0][0] = None
    elif kind == "misaligned":
        values[0].append(9.)
    elif kind == "nonfinite_time":
        values[0][0] = float("nan")  # outside momentum cut must still fail
    elif kind == "nonfinite_p":
        name = "p_mid"
        values = saved.selected_table[name].to_pylist()
        values[0][0] = float("inf")
    else:
        values[0] = [0.] * 65
    saved.selected_table = saved.selected_table.set_column(saved.selected_table.schema.get_field_index(name), name,
                                                          pa.array(values, pa.list_(pa.float64())))
    saved.repin()
    with pytest.raises(ValueError):
        saved.run()


@pytest.mark.parametrize("kind", ["footer_cap", "footer_outside", "selected_rows", "selected_tracks", "decoded", "file_cap"])
def test_parquet_caps(saved, monkeypatch, kind):
    data = saved.selected.read_bytes()
    if kind.startswith("footer"):
        length = J.FOOTER_CAP + 1 if kind == "footer_cap" else len(data)
        data = data[:-8] + struct.pack("<I", length) + b"PAR1"
    else:
        monkeypatch.setattr(J, {"selected_rows": "MAX_ROWS", "selected_tracks": "MAX_TRACKS",
                               "decoded": "DECODED_CAP", "file_cap": "SELECTED_CAP"}[kind], 1)
    with pytest.raises(ValueError):
        if kind == "file_cap":
            saved.run()
        else:
            J.read_selected(data)


@pytest.mark.parametrize("kind", ["ledger_key", "measured_pbtime", "measured_nprotons", "window", "matched", "stratum", "source"])
def test_join_and_inherited_mismatch(saved, kind):
    if kind == "ledger_key":
        saved.ledger_arrays["event"][0] = 888
    elif kind in ("measured_pbtime", "measured_nprotons"):
        saved.ledger_arrays[kind][0] += 1
    elif kind == "window":
        saved.report["momentum_window"]["selected_tracks"] += 1
    elif kind == "matched":
        saved.report["diagnostic"]["selected_events_matched"] += 1
    elif kind == "stratum":
        saved.report["diagnostic"]["pbi_quantile_strata"][0]["selected_tracks"] += 1
    else:
        saved.rows["source_index"][0] = 1
    saved.repin()
    with pytest.raises(ValueError):
        saved.run()


def test_strict_edges_and_flow():
    assert J.bins(np.array([-241, -240, -230, -190, -189]), J.PB_EDGES).tolist() == [0, 1, 2, 6, 6]
    assert J.bins(np.array([-1, 0, 100, 2000, 2001]), J.TIME_EDGES).tolist() == [0, 1, 2, 21, 21]
    assert J.bins(np.array([0, 2, 5, 6]), [0, 2, 5], True).tolist() == [1, 2, 2, 3]
    assert J.bins(np.array([0, 1, 3, 4, 5, 6, 9, 10]), J.TC_EDGES).tolist() == [1, 2, 4, 5, 5, 6, 6, 7]
    assert J.bins(np.array([0, 1, 32, 4096]), J.HIT_EDGES).tolist() == [1, 2, 3, 10]


def test_grid_zero_denominator_and_support():
    g = J.grid([np.array([0] * 100 + [1] * 20)], (3,),
               np.array([0] * 100 + [2] * 20), np.repeat(np.arange(100, 120), 2), np.zeros(40, dtype=int))
    assert g["event_selection_probability_A_over_N"] == [0., 1., None]
    assert g["tracks_per_event_T_over_N_not_probability"] == [0., 2., None]
    assert g["tracks_per_selected_event_T_over_A"] == [None, 2., None]
    assert g["support_N_ge100"] == [True, False, False]
    assert g["support_A_ge20_and_T_ge30"] == [False, True, False]


def test_no_selected_events(saved):
    saved.selected_table = saved.selected_table.slice(0, 0)
    saved.report["momentum_window"].update(selected_events=0, selected_tracks=0)
    saved.report["diagnostic"].update(selected_events_matched=0, selected_tracks_matched=0,
                                        selected_events_not_in_supplied_ntuples=0)
    for stratum in saved.report["diagnostic"]["pbi_quantile_strata"]:
        stratum.update(selected_events=0, selected_tracks=0)
    saved.repin()
    result = saved.run()
    assert result["counts"]["retained_events_without_window_tracks"] == 6
    assert result["counts"]["selected_tracks_matched"] == 0
    for grid in result["grids"].values():
        assert grid["unsupported"]["joint_support"]["track_fraction"] is None


def test_empty_grid():
    empty = np.array([], dtype=np.int64)
    g = J.grid([empty], (3,), empty, empty, empty)
    assert g["N_all_retained"] == [0, 0, 0]
    assert g["event_selection_probability_A_over_N"] == [None] * 3
    assert g["unsupported"]["joint_support"]["event_fraction"] is None


def test_large_integer_measurement_no_lossy_cast():
    with pytest.raises(ValueError, match="differ"):
        J.exact_measurement_equal(np.array([2**53 + 1], dtype=np.int64), np.array([float(2**53)]), "nprotons")
    J.exact_measurement_equal(np.array([2**53], dtype=np.int64), np.array([float(2**53)]), "nprotons")


def test_no_root_or_old_module_imports():
    source = Path(J.__file__).read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(x.name for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(any(s in name for s in ("uproot", "ROOT", "selection_exposure", "ce_proxy")) for name in imports)
    assert len(source.encode()) <= J.SOURCE_CAP


def test_duplicate_json_and_nonfinite():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'):
        with pytest.raises(ValueError):
            J.json_object(raw)


def test_nonordinal_source_indices(saved):
    saved.rows["source_index"] = np.where(saved.rows["source_index"] == 0, 8, 11).astype(np.int32)
    saved.rows_report["files"][0]["index"] = 8
    saved.rows_report["files"][1]["index"] = 11
    saved.rows_report["files"].reverse()
    saved.repin()
    result = saved.run()
    assert [s["source_index"] for s in result["sources"]] == [8, 11]
    assert [s["selected_tracks"] for s in result["sources"]] == [5, 2]


def test_awkward_parquet_extension_input(saved):
    import awkward as ak
    data = ak.from_arrow(saved.selected_table)
    ak.to_parquet(data, saved.selected)
    actual = J.read_selected(saved.selected.read_bytes())
    assert actual["counts"]["momentum_window_tracks"] == 8
    assert actual["counts"]["full_selected_rows"] == 5


@pytest.mark.parametrize("kind", ["prefix", "trailer", "local_extra", "central_extra"])
def test_zip_envelope_extras(saved, kind):
    data = saved.npz.read_bytes()
    if kind == "prefix":
        data = b"extra" + data
    elif kind == "trailer":
        data += b"trailer"
    elif kind == "local_extra":
        data = bytearray(data)
        nlen = struct.unpack_from("<H", data, 26)[0]
        data[30 + nlen] = 2
        data = bytes(data)
    else:
        stream = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(stream, "w") as dest:
            for info in source.infolist():
                info.extra = b"\xfe\xca\x00\x00"
                dest.writestr(info, source.read(info))
        data = stream.getvalue()
    with pytest.raises(ValueError):
        J.load_rows(data, saved.rows_report["rows"])


def test_relative_ledger_reference(saved):
    saved.report["exposure_ledger"]["path"] = "../ledger/ledger.parquet"
    saved.repin()
    assert saved.run()["counts"]["retained_events"] == 6


def test_one_batch_at_time_and_no_track_dedup(saved, monkeypatch):
    real = J.pq.ParquetFile
    class TinyBatches:
        def __init__(self, *args, **kwargs):
            self.inner = real(*args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.inner, name)
        def iter_batches(self, **kwargs):
            kwargs["batch_size"] = 1
            return self.inner.iter_batches(**kwargs)
    monkeypatch.setattr(J.pq, "ParquetFile", TinyBatches)
    result = saved.run()
    assert result["counts"]["selected_tracks_matched"] == 7
    assert result["counts"]["selected_events_matched"] == 3


@pytest.mark.parametrize("kind", ["size", "dtype", "fortran_shape", "pickle"])
def test_npy_header_rejected_before_allocation(saved, monkeypatch, kind):
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(saved.npz.read_bytes())) as original, zipfile.ZipFile(stream, "w") as archive:
        for info in original.infolist():
            raw = original.read(info)
            if info.filename == "run.npy":
                header = io.BytesIO()
                descriptor = dict(descr="<i8", fortran_order=False, shape=(6,))
                if kind == "size":
                    descriptor["shape"] = (2**60,)
                elif kind == "dtype":
                    descriptor["descr"] = [("x", "<i8")]
                elif kind == "fortran_shape":
                    descriptor.update(fortran_order=True, shape=(3, 2))
                else:
                    descriptor["descr"] = "|O"
                np.lib.format.write_array_header_1_0(header, descriptor)
                raw = header.getvalue()
            archive.writestr(info.filename, raw)
    def forbidden(*args, **kwargs):
        pytest.fail("must reject malicious first header before NumPy allocation")
    monkeypatch.setattr(J.np, "load", forbidden)
    with pytest.raises(ValueError):
        J.load_rows(stream.getvalue(), saved.rows_report["rows"])


def test_npz_uncompressed_total_bound(saved, monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(saved.npz.read_bytes())) as original, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info in original.infolist():
            archive.writestr(info.filename, original.read(info) + b"\x00" * 1000)
    compressed = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(compressed)) as z:
        total = sum(i.file_size for i in z.infolist())
    assert len(compressed) < total
    monkeypatch.setattr(J, "ROWS_CAP", total - 1)
    with pytest.raises(ValueError, match="uncompressed"):
        J.load_rows(compressed, saved.rows_report["rows"])


def test_output_code_directory_forbidden(saved):
    saved.out = Path(J.__file__).parents[2] / "never_created_joint_report.json"
    with pytest.raises(ValueError, match="output under"):
        saved.run()
    assert not saved.out.exists()


def test_missing_source_records_and_bad_index(saved):
    saved.rows_report.pop("files")
    saved.rp.write_text(json.dumps(saved.rows_report))
    with pytest.raises(ValueError, match="source file records"):
        saved.run()


def test_original_track_positions_hash(saved):
    result = saved.run()
    expected = np.array([(1, 2, 1, 1), (1, 2, 2, 0), (1, 2, 2, 1), (1, 2, 2, 2), (1, 2, 2, 3),
                         (1, 2, 2**32 + 1, 0), (1, 2, 2**32 + 1, 1)], dtype="<i8")
    assert result["matched_track_identity"]["sha256"] == J.sha(expected.tobytes())


def test_null_scalar_ledger(saved):
    values = {k: pa.array(v) for k, v in saved.ledger_arrays.items()}
    values["measured_pbtime"] = pa.array([None, 1., 2., 3., 4., 5.])
    target = io.BytesIO()
    pq.write_table(pa.table(values), target)
    with pytest.raises(ValueError, match="null"):
        J.read_ledger(target.getvalue())


def test_pre_read_report_and_source_caps(saved, monkeypatch):
    monkeypatch.setattr(J, "REPORT_CAP", 10)
    with pytest.raises(ValueError, match="byte cap"):
        saved.run()
    monkeypatch.setattr(J, "SOURCE_CAP", 10)
    with pytest.raises(ValueError, match="byte cap"):
        saved.run()


def test_all_raw_time_flow_counts(saved):
    result = saved.run()
    grid = result["grids"]["nProtonTCs__measured_pbtime"]
    hist = np.array(grid["selected_track_raw_time_histogram"]).sum(axis=(0, 1))
    assert hist[0] == 1  # -1 ns, never wrapped
    assert hist[1] == 1  # raw zero
    assert hist[2] == 2  # exact 100 ns
    assert hist[20] == 1  # 1999 ns
    assert hist[21] == 2  # exact 2000 plus 2500 ns
    assert hist.sum() == 7


def test_all_empty_selected_lists(saved):
    for name in ("t_mid", "p_mid"):
        saved.selected_table = saved.selected_table.set_column(saved.selected_table.schema.get_field_index(name), name,
            pa.array([[]] * 5, type=pa.list_(pa.float64())))
    saved.report["momentum_window"].update(selected_events=0, selected_tracks=0)
    saved.report["diagnostic"].update(selected_events_matched=0, selected_tracks_matched=0,
        selected_events_not_in_supplied_ntuples=0)
    for s in saved.report["diagnostic"]["pbi_quantile_strata"]:
        s.update(selected_events=0, selected_tracks=0)
    saved.repin()
    result = saved.run()
    assert result["counts"]["selected_rows_unmatched_including_empty"] == 1
    assert result["counts"]["selected_events_unmatched"] == 0
    assert result["counts"]["zero_window_rows"] == 5


def test_rejected_momentum_still_requires_finite_time(saved):
    table = saved.selected_table
    times = table["t_mid"].to_pylist()
    times[0][2] = float("inf")  # p=200 is rejected by strict cut, but input must be well-defined
    saved.selected_table = table.set_column(table.schema.get_field_index("t_mid"), "t_mid", pa.array(times))
    saved.repin()
    with pytest.raises(ValueError, match="nonfinite"):
        saved.run()


def test_exclusive_output_race(saved, monkeypatch):
    real = J.Evidence.recheck
    def competing(evidence):
        real(evidence)
        saved.out.write_bytes(b"other writer")
    monkeypatch.setattr(J.Evidence, "recheck", competing)
    with pytest.raises(ValueError, match="new output"):
        saved.run()
    assert saved.out.read_bytes() == b"other writer"


def test_npz_compressed_valid(saved):
    stream = io.BytesIO()
    np.savez_compressed(stream, **saved.rows)
    result = J.load_rows(stream.getvalue(), saved.rows_report["rows"])
    assert result["measured_pbtime"].tobytes() == saved.rows["measured_pbtime"].tobytes()


@pytest.mark.parametrize("varying", [False, True])
def test_stored_pbterr_observation_is_reported_not_assumed(saved, varying):
    if varying:
        saved.rows["measured_pbterr"][0] = 0.75
        saved.repin()
    result = saved.run()["measured_pbterr"]
    assert result["constant"] is (not varying)
    assert result["min"] == 0.5 and result["max"] == (0.75 if varying else 0.5)
    assert "not an independent" in result["interpretation"]
