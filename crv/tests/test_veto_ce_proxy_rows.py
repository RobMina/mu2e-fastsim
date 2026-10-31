"""Synthetic ROOT and in-memory XRootD only; small report pins are test-only."""
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
import uproot

from crvfastsim.veto import ce_proxy_audit as A
from crvfastsim.veto import ce_proxy_rows as R
from test_veto_ce_proxy_audit import FakeXRootD, FakeStatus, forbidden, pinned_path, version_hist

pytestmark = pytest.mark.usefixtures("dcache_site")


def data(index):
    return {"evtinfo/run": np.full(4, 1430, dtype=np.uint32),
        "evtinfo/subrun": np.full(4, index, dtype=np.int32),
        "evtinfo/event": np.arange(1, 5, dtype=np.int32),
        "evtinfo/nprotons": np.zeros(4, dtype=np.uint64),
        "evtinfo/pbtime": np.array([-1, -200, 0, 1], dtype=np.float32),
        "evtinfo/pbterr": np.full(4, .5, dtype=np.float32),
        **{"lumistream./lumistream." + name: np.array([-1, 0, 1, 2], dtype=np.int32) for name in
           ("nProtonTCs", "nTrackerHits", "nCaloHitsD0", "nCaloHitsD1", "caloEnergy")},
        "lumistream./lumistream.caphriIsValid": np.ones(4, dtype=np.bool_),
        "evtinfomc/nprotons": np.ones(4), "nprotons": np.ones(4),
        "tracks": np.ones((4, 3)), "CRV": np.ones((4, 2)), "coincs": np.ones((4, 2))}


class SavedRows:
    def __init__(self, base, monkeypatch):
        self.monkeypatch, self.base = monkeypatch, base
        self.client = FakeXRootD(monkeypatch)
        evidence = base / "evidence"
        evidence.mkdir()
        self.selected = base / "selected" / "never-open.parquet"
        self.selected.parent.mkdir()
        self.ledger = base / "ledger" / "never-open.parquet"
        self.ledger.parent.mkdir()
        self.selection = dict(selection=dict(path=str(self.selected), sha256="a" * 64),
                              exposure_ledger=dict(path=str(self.ledger)), ntuples=[])
        # Synthetic report made by the current audit source (the production report embeds the earlier revision).
        monkeypatch.setattr(R, "REPORT_AUDIT_SOURCE_SHA256", A.sha(Path(A.__file__).read_bytes()))
        self.availability = dict(schema=A.SCHEMA, source=dict(path=A.__file__, sha256=R.REPORT_AUDIT_SOURCE_SHA256,
            snapshot=Path(A.__file__).read_text()), total_entries=44, full_ids_unique=True, files=[])
        self.args = dict(selection_report=evidence / "selection.json", availability_report=evidence / "availability.json",
                         out_dir=base / "output")
        self.physical = {}
        for index in sorted(A.INDICES):
            path = pinned_path(index)
            self.physical[path] = evidence / f"synthetic_{index}.root"
            record, saved = self.make(index, data(index))
            self.selection["ntuples"].append(record)
            self.availability["files"].append(saved)
        monkeypatch.setattr(A, "EXPECTED_ENTRIES", 44)
        self.repin()
        for name in ("run", "inspect_file", "version", "summary", "aggregate", "BoundedFile"):
            monkeypatch.setattr(A, name, forbidden)

    def make(self, index, values):
        path = pinned_path(index)
        physical = self.physical[path]
        with uproot.recreate(physical) as root:
            tree = root.mktree("EventNtuple/ntuple", {k: np.dtype((v.dtype, v.shape[1:])) if v.ndim > 1 else v.dtype
                                                     for k, v in values.items()})
            tree.extend(values)
            root["EventNtuple/version"] = version_hist()
        with A.metadata_limits(), uproot.open(physical) as root:
            record = dict(path=path, uuid=str(root.file.uuid), entries=4, index=index)
            _, tree, cycles = A.tree_header(root, record)
            scalars = {}
            for name in A.KINDS:
                branch = tree[name]
                scalars[name] = dict(available=True, metadata=A.branch_metadata(branch, name, 4),
                                    consumed_array=A.array_digest(branch.array(library="np")))
        raw = physical.read_bytes()
        self.client.add(path, raw)
        saved = dict(**record, chosen_cycles=cycles, header_sha256=A.sha(raw[:100]), header_bytes=100,
                     header_reopened=True, scalars=scalars, version=dict(available=True, major_minor_patch=[6, 13, 2]))
        selection_record = dict(path=path, uuid=record["uuid"], trees={"EventNtuple/ntuple": {"entries": 4}})
        return selection_record, saved

    def rewrite(self, index, values):
        record, saved = self.make(index, values)
        position = sorted(A.INDICES).index(index)
        self.selection["ntuples"][position] = record
        self.availability["files"][position] = saved
        self.repin()

    def repin(self):
        selection_bytes = json.dumps(self.selection).encode()
        self.args["selection_report"].write_bytes(selection_bytes)
        selection_pin = A.sha(selection_bytes)
        self.monkeypatch.setattr(R, "SELECTION_SHA256", selection_pin)
        self.availability["selection_report"] = dict(path=str(self.args["selection_report"]), sha256=selection_pin)
        self.repin_availability()
        self.args["expected_selection_sha256"] = selection_pin

    def repin_availability(self):
        payload = json.dumps(self.availability).encode()
        self.args["availability_report"].write_bytes(payload)
        self.monkeypatch.setattr(R, "AVAILABILITY_SHA256", A.sha(payload))
        self.args["expected_availability_sha256"] = A.sha(payload)

    def run(self, **kwargs):
        return R.run(**(self.args | kwargs))

    def incomplete(self):
        assert not (self.args["out_dir"] / "report.json").exists()


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return SavedRows(tmp_path, monkeypatch)


def test_production_constants():
    assert R.SELECTION_SHA256 == "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
    assert R.AVAILABILITY_SHA256 == "2b59602667126391fe1b32bb6ce4051b83c88268f5f60c580922d1cfeef18325"
    assert R.REPORT_AUDIT_SOURCE_SHA256 == "3765b8114f4128d7bae438d3b943cc6f8b61c1c4f393e0963fb4ede6494cc776"
    assert R.AUDIT_SOURCE_SHA256 == "f776977470c9ed6c62ddb94f902c368cde76ccb8e5f5b64c74ade0bbc364abc7"
    assert A.sha(Path(A.__file__).read_bytes()) == R.AUDIT_SOURCE_SHA256
    assert A.EXPECTED_ENTRIES == 91226
    assert R.NPZ_CAP == 16 << 20 and R.REPORT_CAP == 2 << 20
    assert R.JSON_CAP == 8 << 20 and R.SOURCE_CAP == 64 << 10
    assert R.ROW_KEYS == ("run", "subrun", "event", "source_index", "measured_nprotons", "measured_pbtime",
        "measured_pbterr", "nProtonTCs", "nTrackerHits", "nCaloHitsD0", "nCaloHitsD1", "caloEnergy", "caphriIsValid")


def test_real_root_allowlist_digests_dtypes_and_exclusive_output(saved, monkeypatch):
    original_array = uproot.behaviors.TBranch.TBranch.array
    seen, meta, headers = [], [], []
    original_metadata, original_header = A.branch_metadata, A.tree_header
    def metadata(branch, name, entries):
        meta.append(name)
        return original_metadata(branch, name, entries)
    def header(root, record):
        headers.append(record["path"])
        return original_header(root, record)
    def array(self, *args, **kwargs):
        assert self.name in A.KINDS
        assert len(meta) == ((len(seen) // 12) + 1) * 12
        assert kwargs["library"] == "np" and kwargs["array_cache"] is None
        assert isinstance(kwargs["decompression_executor"], uproot.source.futures.TrivialExecutor)
        seen.append(self.name)
        return original_array(self, *args, **kwargs)
    monkeypatch.setattr(A, "branch_metadata", metadata)
    monkeypatch.setattr(A, "tree_header", header)
    monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", array)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", forbidden)
    for name in ("open", "fstat", "lseek", "listdir", "scandir"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("stat", "lstat"):
        original = getattr(os, name)
        def guard(path, *args, _original=original, **kwargs):
            assert not str(path).startswith("/dcache-test")
            assert path not in (saved.selected, saved.ledger)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(os, name, guard)
    result = saved.run()
    assert len(seen) == 132 and len(headers) == 22 and len(saved.client.instances) == 22
    assert result["schema"] == R.SCHEMA and result["total_entries"] == 44
    assert result["read_bytes"] == sum(result["per_file_read_bytes"].values()) > 0
    assert all(f["read_bytes"] > f["read_bytes_before_reopen"] for f in result["files"])
    assert all(f["version_inherited"]["major_minor_patch"] == [6, 13, 2] and not f["version_reread"] for f in result["files"])
    assert A.sha(result["source"]["snapshot"].encode()) == result["source"]["sha256"]
    assert all(result[k] is None for k in ("F", "weights", "weight_model", "calibration", "finite_library_uncertainty"))
    assert not any(result[k] for k in ("availability_reexecuted", "prior_audit_executed", "selection_executed", "mc_read", "full_file_hashes"))
    path = Path(result["rows"]["path"])
    assert A.sha(path.read_bytes()) == result["rows"]["sha256"]
    with zipfile.ZipFile(path) as archive:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in archive.infolist())
    with np.load(path, allow_pickle=False) as rows:
        assert tuple(rows.files) == R.ROW_KEYS
        assert rows["source_index"].dtype == np.dtype("int32")
        assert np.array_equal(rows["source_index"], np.repeat(sorted(A.INDICES), 4))
        for name in R.ROW_KEYS:
            assert rows[name].shape == (44,) and rows[name].dtype.kind in "biuf"
            assert A.array_digest(rows[name])["sha256"] == result["rows"]["columns"][name]["sha256"]
        for name in ("run", "subrun", "event"):
            assert rows[name].dtype == np.dtype("int64")
        for leaf, column in R.COLUMNS.items():
            if leaf not in A.IDS:
                assert rows[column].dtype == data(0)[leaf].dtype
                assert np.array_equal(rows[column][:4], data(0)[leaf])
    assert json.loads((saved.args["out_dir"] / "report.json").read_text()) == result
    with pytest.raises(ValueError, match="overwrite"):
        saved.run()


@pytest.mark.parametrize("target", ["selection_argument", "availability_argument", "selection_bytes", "availability_bytes", "audit_source"])
def test_pins_before_remote(saved, monkeypatch, target):
    if target.endswith("argument"):
        saved.args["expected_" + target.split("_")[0] + "_sha256"] = "f" * 64
    elif target.endswith("bytes"):
        saved.args[target.split("_")[0] + "_report"].write_text("{}")
    else:
        original = A.small_bytes
        monkeypatch.setattr(A, "small_bytes", lambda p, c: original(p, c) + (b" " if p == Path(A.__file__) else b""))
    monkeypatch.setattr(A, "XRootDBoundedFile", forbidden)
    with pytest.raises(ValueError): saved.run()
    saved.incomplete()


@pytest.mark.parametrize("mutation", ["schema", "selectionpin", "sourcepin", "snapshot", "total", "booltotal", "unique", "count",
    "path", "duplicate", "uuid", "entries", "index", "cycles", "boolcycle", "header", "headerbytes", "reopened", "scalars",
    "available", "metadata", "dtype", "shape", "digest", "order", "bytes", "metabytes", "metabaskets", "metaentries", "metaclass",
    "sourcepath", "selectionpath"])
def test_availability_schema_before_remote(saved, monkeypatch, mutation):
    old = saved.availability
    row = old["files"][0]
    scalar = row["scalars"]["evtinfo/run"]
    meta, dig = scalar["metadata"], scalar["consumed_array"]
    if mutation == "schema": old["schema"] = "other"
    if mutation == "selectionpin": old["selection_report"]["sha256"] = "f" * 64
    if mutation == "sourcepin": old["source"]["sha256"] = "f" * 64
    if mutation == "snapshot": old["source"]["snapshot"] += " "
    if mutation == "total": old["total_entries"] = 43
    if mutation == "booltotal": old["total_entries"] = True
    if mutation == "unique": old["full_ids_unique"] = False
    if mutation == "count": old["files"].pop()
    if mutation == "path": row["path"] = "/dcache-test/other.root"
    if mutation == "duplicate": old["files"][1] = row
    if mutation == "uuid": row["uuid"] = "a" * 32
    if mutation == "entries": row["entries"] = 3
    if mutation == "index": row["index"] = True
    if mutation == "cycles": row["chosen_cycles"] = {}
    if mutation == "boolcycle": row["chosen_cycles"]["ntuple"] = True
    if mutation == "header": row["header_sha256"] = "invalid"
    if mutation == "headerbytes": row["header_bytes"] = 99
    if mutation == "reopened": row["header_reopened"] = False
    if mutation == "scalars": row["scalars"].pop("evtinfo/event")
    if mutation == "available": scalar["available"] = False
    if mutation == "metadata": scalar.pop("metadata")
    if mutation == "dtype": dig["dtype"] = "O"
    if mutation == "shape": dig["shape"] = [4, 1]
    if mutation == "digest": dig["sha256"] = "invalid"
    if mutation == "order": dig["order"] = "F"
    if mutation == "bytes": dig["bytes"] += 1
    if mutation == "metabytes": meta["uncompressed_bytes"] = A.BRANCH_BYTES + 1
    if mutation == "metabaskets": meta["baskets"] = A.MAX_BASKETS + 1
    if mutation == "metaentries": meta["entries"] = True
    if mutation == "metaclass": meta["classname"] = "TTree"
    if mutation == "sourcepath": old["source"]["path"] = "/" + "x" * R.PATH_CAP
    if mutation == "selectionpath": old["selection_report"]["path"] = "../bad"
    saved.repin_availability()
    monkeypatch.setattr(A, "XRootDBoundedFile", forbidden)
    with pytest.raises((ValueError, TypeError)): saved.run()
    saved.incomplete()


@pytest.mark.parametrize("name", list(A.KINDS))
def test_every_consumed_array_digest_checked(saved, monkeypatch, name):
    original = uproot.behaviors.TBranch.TBranch.array
    def mutated(self, *args, **kwargs):
        array = original(self, *args, **kwargs)
        if self.name == name:
            array = array.copy()
            array[0] = not array[0] if array.dtype.kind == "b" else array[0] + 1
        return array
    monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", mutated)
    with pytest.raises(ValueError, match="consumed array differs"):
        saved.run()
    saved.incomplete()


@pytest.mark.parametrize("mutation", ["dtype", "shape", "metadata", "header", "cycles", "missing", "duplicate_names"])
def test_fresh_data_mismatch(saved, monkeypatch, mutation):
    if mutation in ("dtype", "shape"):
        original = uproot.behaviors.TBranch.TBranch.array
        def array(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            return value.astype(np.float64) if mutation == "dtype" else value.reshape(4, 1)
        monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", array)
    elif mutation == "metadata":
        original = A.branch_metadata
        def metadata(*args):
            result = original(*args)
            result["compressed_bytes"] += 1
            return result
        monkeypatch.setattr(A, "branch_metadata", metadata)
        monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", forbidden)
    elif mutation == "header":
        saved.availability["files"][0]["header_sha256"] = "f" * 64
        saved.repin_availability()
    elif mutation == "cycles":
        saved.availability["files"][0]["chosen_cycles"]["ntuple"] += 1
        saved.repin_availability()
    else:
        original = uproot.behaviors.TBranch.HasBranches.keys
        def keys(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if self.classname == "TTree":
                return [n for n in result if n != "evtinfo/nprotons"] if mutation == "missing" else result + result[:1]
            return result
        monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "keys", keys)
        monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", forbidden)
    with pytest.raises(ValueError): saved.run()
    saved.incomplete()


@pytest.mark.parametrize("mutation", ["within", "across", "negative", "overflow", "promotion"])
def test_identity_domain_uniqueness_and_no_dtype_promotion(saved, mutation):
    values = data(1 if mutation == "across" else 0)
    if mutation == "within": values["evtinfo/event"][-1] = 3
    if mutation == "across": values["evtinfo/subrun"][:] = 0
    if mutation == "negative": values["evtinfo/event"][0] = -1
    if mutation == "overflow": values["evtinfo/event"] = np.full(4, 2 ** 63, dtype=np.uint64)
    if mutation == "promotion": values["evtinfo/pbtime"] = values["evtinfo/pbtime"].astype(np.float64)
    saved.rewrite(1 if mutation == "across" else 0, values)
    with pytest.raises(ValueError, match="duplicate full IDs|full ID domain|promotion forbidden"):
        saved.run()
    saved.incomplete()


@pytest.mark.parametrize("target", ["selection", "availability", "audit_source", "rows_source", "reopen_header", "reopen_uuid", "reopen_entries", "reopen_cycles"])
def test_end_rechecks_and_header_reopens(saved, monkeypatch, target):
    counts = {}
    if target.startswith("reopen_"):
        if target == "reopen_header":
            original = A.XRootDBoundedFile.read
            def read(self, size=-1):
                value = original(self, size)
                if size == 100:
                    counts[self.path] = counts.get(self.path, 0) + 1
                    if counts[self.path] == 2: value = b"x" + value[1:]
                return value
            monkeypatch.setattr(A.XRootDBoundedFile, "read", read)
        else:
            original = A.tree_header
            def header(root, record):
                counts[record["path"]] = counts.get(record["path"], 0) + 1
                record = dict(record)
                second = counts[record["path"]] == 2
                if second and target == "reopen_uuid": record["uuid"] = "f" * 36
                if second and target == "reopen_entries": record["entries"] += 1
                directory, tree, cycles = original(root, record)
                if second and target == "reopen_cycles": cycles["ntuple"] += 1
                return directory, tree, cycles
            monkeypatch.setattr(A, "tree_header", header)
    else:
        wanted = {"selection": saved.args["selection_report"], "availability": saved.args["availability_report"],
                  "audit_source": Path(A.__file__), "rows_source": Path(R.__file__)}[target]
        original = A.small_bytes
        def read(path, cap):
            value = original(path, cap)
            counts[path] = counts.get(path, 0) + 1
            return value + b" " if path == wanted and counts[path] > 1 else value
        monkeypatch.setattr(A, "small_bytes", read)
    with pytest.raises(ValueError): saved.run()
    saved.incomplete()


@pytest.mark.parametrize("owner,cap,value", [(R, "JSON_CAP", 1), (R, "SOURCE_CAP", 1), (R, "REPORT_CAP", 1),
    (R, "NPZ_CAP", 1), (A, "FILE_CAP", 99), (A, "TOTAL_CAP", 99), (A, "READ_CAP", 99), (A, "MAX_KEYS", 1),
    (A, "MAX_BRANCHES", 1), (A, "MAX_BASKETS", 0), (A, "TOTAL_BASKETS", 1), (A, "BRANCH_BYTES", 1),
    (A, "PAYLOAD_CAP", 1), (A, "METADATA_CAP", 1)])
def test_caps(saved, monkeypatch, owner, cap, value):
    monkeypatch.setattr(owner, cap, value)
    with pytest.raises(ValueError): saved.run()
    saved.incomplete()


@pytest.mark.parametrize("where", ["evidence", "selected", "ledger", "code", "existing", "symlink", "pnfs", "traversal", "long"])
def test_output_exclusion_before_remote(saved, monkeypatch, where):
    if where == "evidence": out = saved.args["selection_report"].parent / "new"
    if where == "selected": out = saved.selected.parent / "new"
    if where == "ledger": out = saved.ledger.parent / "new"
    if where == "code": out = Path(R.__file__).parent / "never-created"
    if where == "existing":
        out = saved.args["out_dir"]
        out.mkdir()
    if where == "symlink":
        link = saved.base / "link"
        link.symlink_to(saved.base / "evidence", target_is_directory=True)
        out = link / "new"
    if where == "pnfs": out = Path("/mnt/pnfs/forbidden/new")
    if where == "traversal": out = saved.base / "evidence" / ".." / "new"
    if where == "long": out = "/" + "x" * R.PATH_CAP
    monkeypatch.setattr(A, "XRootDBoundedFile", forbidden)
    with pytest.raises(ValueError): saved.run(out_dir=out)
    saved.incomplete()


@pytest.mark.parametrize("target", ["selection_report", "availability_report"])
def test_report_symlinks(saved, target):
    linked = saved.base / (target + "-link")
    linked.symlink_to(saved.args[target])
    with pytest.raises(ValueError, match="symlink"):
        saved.run(**{target: linked})
    saved.incomplete()


@pytest.mark.parametrize("operation", ["open", "read", "close"])
def test_transport_failures_no_fallback(saved, operation):
    saved.client.status[operation] = FakeStatus(False, "intentional")
    with pytest.raises(OSError, match="XRootD"):
        saved.run()
    assert len(saved.client.instances) == 1
    saved.incomplete()


def test_short_read_no_retry(saved):
    saved.client.read_result = b""
    with pytest.raises(ValueError, match="short read"):
        saved.run()
    assert len(saved.client.instances) == 1
    saved.incomplete()


def test_report_written_last_and_recheck_after_npz(saved, monkeypatch):
    original = Path.open
    seen = []
    def opened(path, mode="r", *args, **kwargs):
        if mode == "xb":
            seen.append(path.name)
            if path.name == "rows.npz":
                saved.args["availability_report"].write_bytes(b"mutated")
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(ValueError, match="availability JSON changed"):
        saved.run()
    assert seen == ["rows.npz"] and (saved.args["out_dir"] / "rows.npz").exists()
    saved.incomplete()


def test_exclusive_npz_no_overwrite(saved, monkeypatch):
    original = Path.mkdir
    def mkdir(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == saved.args["out_dir"]: (path / "rows.npz").write_bytes(b"keep")
        return result
    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(FileExistsError): saved.run()
    assert (saved.args["out_dir"] / "rows.npz").read_bytes() == b"keep"
    saved.incomplete()


def test_cli_required_arguments_and_compact_summary(saved, capsys):
    with pytest.raises(SystemExit): R.main(["selection", "availability", "--out-dir", "out"])
    R.main([str(saved.args["selection_report"]), str(saved.args["availability_report"]),
        "--out-dir", str(saved.args["out_dir"]), "--expected-selection-sha256", saved.args["expected_selection_sha256"],
        "--expected-availability-sha256", saved.args["expected_availability_sha256"]])
    printed = capsys.readouterr().out
    assert "\n" not in printed.strip() and ": " not in printed
    assert json.loads(printed)["total_entries"] == 44


@pytest.mark.parametrize("target", ["selection_report", "availability_report"])
@pytest.mark.parametrize("payload", [b'{"duplicate":1,"duplicate":2}', b'{"bad":NaN}', b'[]'])
def test_json_rejects_duplicates_nonfinite_and_nonobjects(saved, monkeypatch, target, payload):
    saved.args[target].write_bytes(payload)
    prefix = target.split("_")[0]
    monkeypatch.setattr(R, prefix.upper() + "_SHA256", A.sha(payload))
    saved.args["expected_" + prefix + "_sha256"] = A.sha(payload)
    monkeypatch.setattr(A, "XRootDBoundedFile", forbidden)
    with pytest.raises(ValueError): saved.run()
    saved.incomplete()


@pytest.mark.parametrize("target", ["selected", "ledger"])
def test_long_inherited_paths_are_bounded_before_remote(saved, monkeypatch, target):
    reference = saved.selection["selection" if target == "selected" else "exposure_ledger"]
    reference["path"] = "/" + "x" * R.PATH_CAP
    saved.repin()
    monkeypatch.setattr(A, "XRootDBoundedFile", forbidden)
    with pytest.raises(ValueError, match="bounded pathname"):
        saved.run()
    saved.incomplete()


def test_serialized_npz_size_cap_includes_zip_headers(saved, monkeypatch):
    raw_size = 44 * (3 * 8 + 4 + 8 + 2 * 4 + 5 * 4 + 1)
    monkeypatch.setattr(R, "NPZ_CAP", raw_size)
    with pytest.raises(ValueError, match="NPZ output byte cap"):
        saved.run()
    assert not (saved.args["out_dir"] / "rows.npz").exists()
    saved.incomplete()


@pytest.mark.parametrize("cap", ["FILE_CAP", "TOTAL_CAP"])
def test_reopens_count_against_read_budgets(saved, monkeypatch, cap):
    original = A.XRootDBoundedFile.__init__
    counts = {}
    def initialized(self, path, budget):
        original(self, path, budget)
        counts[path] = counts.get(path, 0) + 1
        if counts[path] == 2:
            already = budget.total if cap == "TOTAL_CAP" else budget.files[path]
            monkeypatch.setattr(A, cap, already + 99)
    monkeypatch.setattr(A.XRootDBoundedFile, "__init__", initialized)
    with pytest.raises(ValueError, match="read cap"):
        saved.run()
    assert len(saved.client.instances) == 2
    saved.incomplete()


@pytest.mark.parametrize("operation", ["short", "write", "close"])
def test_report_failure_removes_incomplete_completion_file(saved, monkeypatch, operation):
    original = Path.open
    class FailedOutput:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def write(self, payload):
            self.stream.write(payload[:3])
            if operation == "short": return 3
            if operation == "write": raise OSError("intentional write failure")
            return len(payload)
        def __exit__(self, *exc):
            self.stream.close()
            if operation == "close": raise OSError("intentional close failure")
    def opened(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return FailedOutput(stream) if path.name == "report.json" and mode == "xb" else stream
    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises((OSError, ValueError)):
        saved.run()
    assert (saved.args["out_dir"] / "rows.npz").exists()
    saved.incomplete()


def test_successful_exclusive_write_order(saved, monkeypatch):
    original = Path.open
    writes = []
    def opened(path, mode="r", *args, **kwargs):
        if mode == "xb": writes.append(path.name)
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", opened)
    saved.run()
    assert writes == ["rows.npz", "report.json"]


def test_nonallowlisted_metadata_rejected_without_payload(saved, monkeypatch):
    original = A.branch_metadata
    seen = []
    def metadata(branch, name, entries):
        seen.append(name)
        if name == tuple(A.KINDS)[-1]: raise ValueError("last metadata rejection")
        return original(branch, name, entries)
    monkeypatch.setattr(A, "branch_metadata", metadata)
    monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", forbidden)
    with pytest.raises(ValueError, match="last metadata rejection"):
        saved.run()
    assert seen == list(A.KINDS)
    saved.incomplete()


def test_payload_digest_failure_restores_metadata_hooks(saved):
    hook = uproot.model.Model.hook_before_read
    saved.availability["files"][0]["scalars"]["evtinfo/event"]["consumed_array"]["sha256"] = "f" * 64
    saved.repin_availability()
    with pytest.raises(ValueError, match="consumed array differs"):
        saved.run()
    assert uproot.model.Model.hook_before_read is hook
    saved.incomplete()
