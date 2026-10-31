"""Synthetic real ROOT only; test-only pin/entry overrides never touch PNFS."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pytest
import uproot

from crvfastsim.veto import ce_proxy_audit as A

pytestmark = pytest.mark.usefixtures("dcache_site")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    pytest.fail("forbidden namespace/payload access")


def version_hist(parts=(6, 13, 2)):
    identify = uproot.writing.identify
    axis = identify.to_TAxis("xaxis", "", len(parts), 0., float(len(parts)))
    data = np.array([0, *parts, 0], dtype=np.int32)
    return identify.to_TH1x("version", "synthetic version", data, float(sum(parts)),
        float(sum(parts)), float(sum(parts)), 0., 0., np.zeros(len(data)), axis)


def root_file(path, index=0, *, optional=True, version=(6, 13, 2), jagged=False, duplicate=False):
    data = {"evtinfo/run": np.full(4, 1430, dtype=np.int32),
            "evtinfo/subrun": np.full(4, index, dtype=np.int32),
            "evtinfo/event": np.array([1, 2, 3, 3 if duplicate else 4], dtype=np.int32)}
    if optional:
        data |= {"evtinfo/nprotons": np.array([-1, 0, 2, -2], dtype=np.int32),
                 "evtinfo/pbtime": np.array([-1., 0., -2., np.nan]),
                 "evtinfo/pbterr": np.array([0., 1., np.inf, -1.]),
                 "lumistream./lumistream.nProtonTCs": np.array([-1, 0, 7, -3], dtype=np.int32),
                 "lumistream./lumistream.caphriIsValid": np.array([-1, 0, 1, 2], dtype=np.int32),
                 "lumistream./lumistream.caloEnergy": np.array([-1, 0, 2, -2], dtype=np.int32)}
    # Presence cannot authorize reading them, nor substitute aliases.
    data |= {"evtinfomc/nprotons": np.array([111., 222., 333., 444.]),
             "nprotons": np.ones(4), "evtinfo.nprotons": np.ones(4),
             "tracks": np.ones((4, 3)), "CRV": np.ones((4, 2))}
    with uproot.recreate(path) as root:
        tree = root.mktree("EventNtuple/ntuple", {k: (np.dtype((v.dtype, v.shape[1:])) if v.ndim > 1 else v.dtype) for k, v in data.items()})
        tree.extend(data)
        if version is not None:
            root["EventNtuple/version"] = version_hist(version)
    if jagged:
        import awkward as ak
        with uproot.update(path) as root:
            data["evtinfo/nprotons"] = ak.Array([[1], [], [2, 3], [0]])
            root.mktree("EventNtuple/ntuple", data)
    with uproot.open(path) as root:
        return str(root.file.uuid)


class Saved:
    def __init__(self, base, monkeypatch):
        self.monkeypatch = monkeypatch
        evidence = base / "evidence"
        evidence.mkdir()
        self.selected = base / "selected" / "tracks.parquet"
        self.selected.parent.mkdir()
        self.selected.write_bytes(b"not a parquet file; forbidden")
        self.report = {"selection": {"path": str(self.selected), "sha256": "a" * 64}, "ntuples": []}
        self.mapping = {}
        for index in sorted(A.INDICES):
            path = ("/dcache-test/mu2e/tape/phy-nts/nts/mu2e/" + A.CAMPAIGN +
                f"/root/ab/cd/nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_{index:08d}.root")
            physical = evidence / f"test_{index}.root"
            identity = root_file(physical, index)
            self.mapping[path] = physical
            self.report["ntuples"].append(dict(path=path, uuid=identity,
                trees={"EventNtuple/ntuple": {"entries": 4}}))
        self.args = dict(selection_report=evidence / "selection_window_v2.json", out=base / "audit.json")
        monkeypatch.setattr(A, "EXPECTED_ENTRIES", 44)
        self.repin()
        original_open = A.os.open
        def exact_open(path, flags, *args, **kwargs):
            # Only the exact recorded paths may reach the low-level root opener.
            if str(path).startswith("/dcache-test"):
                assert str(path) in self.mapping
                return original_open(self.mapping[str(path)], flags, *args, **kwargs)
            return original_open(path, flags, *args, **kwargs)
        monkeypatch.setattr(A.os, "open", exact_open)

    def repin(self):
        self.args["selection_report"].write_text(json.dumps(self.report))
        pin = digest(self.args["selection_report"])
        self.monkeypatch.setattr(A, "SELECTION_SHA256", pin)
        self.args["expected_selection_sha256"] = pin

    def rewrite(self, **kwargs):
        record = self.report["ntuples"][0]
        record["uuid"] = root_file(self.mapping[record["path"]], **kwargs)
        self.repin()

    def run(self, **kwargs):
        return A.run(**(self.args | kwargs))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return Saved(tmp_path, monkeypatch)


def test_constants_and_cli_pin():
    assert A.SELECTION_SHA256 == "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
    assert A.EXPECTED_ENTRIES == 91226
    with pytest.raises(SystemExit):
        A.main(["anything", "--out", "new"])
    with pytest.raises(ValueError, match="hardbound"):
        A.run("unopened", "unopened", expected_selection_sha256="f" * 64)


def test_real_root_audit_allowlist_no_namespace(saved, monkeypatch):
    original_array = uproot.behaviors.TBranch.TBranch.array
    seen = []
    def array(self, *args, **kwargs):
        assert self.name in A.KINDS
        assert kwargs["library"] == "np" and kwargs["array_cache"] is None
        assert isinstance(kwargs["decompression_executor"], uproot.source.futures.TrivialExecutor)
        seen.append(self.name)
        return original_array(self, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", array)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", forbidden)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(os, "fstat", forbidden)
    for method in ("stat", "lstat"):
        original = getattr(os, method)
        def guard(path, *args, _original=original, **kwargs):
            assert not str(path).startswith("/dcache-test")
            assert str(path) != str(saved.selected)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(os, method, guard)
    report = saved.run()
    assert len(report["files"]) == 11 and report["total_entries"] == 44
    assert len(seen) == 99  # 9 exact leaves/file; no payload on reopen
    assert report["full_ids_unique"] and report["read_bytes"] > 0
    assert report["read_bytes"] == sum(report["per_file_read_bytes"].values())
    for row in report["files"]:
        assert row["read_bytes"] > row["read_bytes_before_reopen"]
        assert row["chosen_cycles"] == {"directory": 1, "ntuple": 1}
        assert row["version"]["major_minor_patch"] == [6, 13, 2]
        assert row["version"]["values_with_flow"] == [0, 6, 13, 2, 0]
        scalar = row["scalars"]["evtinfo/nprotons"]
        assert scalar["counts"] == dict(entries=4, finite=4, nonfinite=0, sentinel_minus_one=1,
            zero=1, positive=1, other_negative=1, invalid=1, valid=2)
        assert scalar["valid_statistics"]["mean"] == 1.
        assert scalar["consumed_array"] == A.array_digest(np.array([-1, 0, 2, -2], dtype=np.int32))
    assert report["aggregate"]["evtinfo/nprotons"]["counts"]["zero"] == 11
    assert report["aggregate"]["evtinfo/nprotons"]["aggregate_quantiles"] is None
    assert report["F"] is report["weight_model"] is report["finite_library_uncertainty"] is None
    assert hashlib.sha256(report["source"]["snapshot"].encode()).hexdigest() == report["source"]["sha256"]
    assert not report["mc_read"] and not report["selection_executed"] and not report["full_file_hashes"]
    assert json.loads(saved.args["out"].read_text()) == report


@pytest.mark.parametrize("version", [None, (5, 0, 1)])
def test_optional_absence_and_differing_version(saved, version):
    saved.rewrite(optional=False, version=version)
    row = saved.run()["files"][0]
    assert not row["scalars"]["evtinfo/nprotons"]["available"]
    if version is None:
        assert not row["version"]["available"]
    else:
        assert row["version"]["major_minor_patch"] == list(version)
        assert not row["version"]["equals_reference_6_13_2"]


@pytest.mark.parametrize("mutation", ["wrongpin", "hash", "notdict", "count", "path", "index", "duplicate", "uuid", "tree", "entries", "boolentries", "total", "selection"])
def test_bad_pin_schema_paths_before_root(saved, monkeypatch, mutation):
    row = saved.report["ntuples"][0]
    if mutation == "wrongpin":
        saved.args["expected_selection_sha256"] = "f" * 64
    elif mutation == "hash":
        saved.args["selection_report"].write_text("{}")
    else:
        if mutation == "notdict": saved.report = []
        if mutation == "count": saved.report["ntuples"].pop()
        if mutation == "path": row["path"] = row["path"].replace("CeMLeadingLogMix1BB/", "Other/")
        if mutation == "index": row["path"] = row["path"].replace("00000000.root", "00000007.root")
        if mutation == "duplicate": saved.report["ntuples"][1] = row
        if mutation == "uuid": row["uuid"] = "bad"
        if mutation == "tree": row["trees"] = {"ntuple": {"entries": 4}}
        if mutation == "entries": row["trees"]["EventNtuple/ntuple"]["entries"] = 10001
        if mutation == "boolentries": row["trees"]["EventNtuple/ntuple"]["entries"] = True
        if mutation == "total": row["trees"]["EventNtuple/ntuple"]["entries"] = 3
        if mutation == "selection": saved.report["selection"]["path"] = "relative"
        saved.repin()
    monkeypatch.setattr(A, "BoundedFile", forbidden)
    with pytest.raises((ValueError, TypeError)):
        saved.run()
    assert not saved.args["out"].exists()


@pytest.mark.parametrize("mutation", ["uuid", "entries", "jagged", "duplicate"])
def test_root_identity_and_scalar_rejections(saved, mutation):
    row = saved.report["ntuples"][0]
    if mutation == "uuid":
        row["uuid"] = str(uuid.uuid4())
        saved.repin()
    elif mutation == "entries":
        row["trees"]["EventNtuple/ntuple"]["entries"] = 5
        saved.report["ntuples"][1]["trees"]["EventNtuple/ntuple"]["entries"] = 3
        saved.repin()
    else:
        saved.rewrite(**{mutation: True})
    with pytest.raises(ValueError):
        saved.run()
    assert not saved.args["out"].exists()


def test_duplicate_across_files(saved):
    row = saved.report["ntuples"][1]
    row["uuid"] = root_file(saved.mapping[row["path"]], index=0)
    saved.repin()
    with pytest.raises(ValueError, match="across files"):
        saved.run()


@pytest.mark.parametrize("cap,value", [("FILE_CAP", 99), ("TOTAL_CAP", 99), ("READ_CAP", 99),
    ("MAX_KEYS", 1), ("MAX_BRANCHES", 1), ("MAX_BASKETS", 0), ("TOTAL_BASKETS", 1),
    ("BRANCH_BYTES", 1), ("PAYLOAD_CAP", 1), ("VERSION_CAP", 1), ("VERSION_BINS", 2),
    ("METADATA_CAP", 1), ("SOURCE_CAP", 1), ("JSON_CAP", 1), ("OUTPUT_CAP", 1)])
def test_caps(saved, monkeypatch, cap, value):
    monkeypatch.setattr(A, cap, value)
    with pytest.raises(ValueError):
        saved.run()
    assert not saved.args["out"].exists()


def test_bounded_file_counts_reopens_and_rejects_unbounded(tmp_path, monkeypatch):
    path = tmp_path / "bytes"
    path.write_bytes(b"abcdefgh")
    budget = A.Budget()
    monkeypatch.setattr(A, "FILE_CAP", 10)
    monkeypatch.setattr(os, "stat", forbidden)
    monkeypatch.setattr(os, "fstat", forbidden)
    monkeypatch.setattr(os, "lstat", forbidden)
    with A.BoundedFile(str(path), budget) as stream:
        with pytest.raises(ValueError): stream.read()
        with pytest.raises(ValueError): stream.read(A.READ_CAP + 1)
        with pytest.raises(ValueError): stream.seek(0, os.SEEK_END)
        assert stream.read(4) == b"abcd" and stream.tell() == 4
        stream.seek(-2, os.SEEK_CUR)
        assert stream.read(2) == b"cd"
    with A.BoundedFile(str(path), budget) as stream:
        assert stream.read(4) == b"abcd"
        with pytest.raises(ValueError, match="per-file"): stream.read(1)
    assert budget.total == budget.files[str(path)] == 10


@pytest.mark.parametrize("where", ["evidence", "selected", "code", "existing", "symlink"])
def test_output_exclusion(saved, monkeypatch, where):
    if where == "evidence": out = saved.args["selection_report"].parent / "new.json"
    elif where == "selected": out = saved.selected.parent / "new.json"
    elif where == "code": out = Path(A.__file__).parent / "never_created.json"
    elif where == "existing":
        out = saved.args["out"]
        out.write_text("keep")
    else:
        directory = saved.args["out"].parent / "link"
        directory.symlink_to(saved.args["selection_report"].parent, target_is_directory=True)
        out = directory / "new.json"
    monkeypatch.setattr(A, "BoundedFile", forbidden)
    with pytest.raises(ValueError): saved.run(out=out)


def test_smallfile_symlink(saved):
    path = saved.args["selection_report"]
    link = path.parent / "linked.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        saved.run(selection_report=link)


@pytest.mark.parametrize("target", ["json", "source", "root_header", "root_uuid", "root_entries", "root_cycle"])
def test_changes_fail_without_output(saved, monkeypatch, target):
    if target in ("json", "source"):
        original = A.small_bytes
        seen = {}
        def changing(path, cap):
            data = original(path, cap)
            seen[path] = seen.get(path, 0) + 1
            wanted = saved.args["selection_report"] if target == "json" else Path(A.__file__)
            return data + b" " if path == wanted and seen[path] > 1 else data
        monkeypatch.setattr(A, "small_bytes", changing)
    elif target == "root_header":
        original = A.BoundedFile.read
        hits = {}
        def changing(self, size=-1):
            data = original(self, size)
            if size == 100:
                hits[self.path] = hits.get(self.path, 0) + 1
                if hits[self.path] == 2:
                    return b"x" + data[1:]
            return data
        monkeypatch.setattr(A.BoundedFile, "read", changing)
    else:
        original = A.tree_header
        hits = {}
        def changing(root, expected):
            hits[expected["path"]] = hits.get(expected["path"], 0) + 1
            if hits[expected["path"]] == 2 and target != "root_cycle":
                expected = dict(expected)
                if target == "root_uuid": expected["uuid"] = str(uuid.uuid4())
                else: expected["entries"] += 1
            directory, tree, cycles = original(root, expected)
            if hits[expected["path"]] == 2 and target == "root_cycle": cycles["ntuple"] += 1
            return directory, tree, cycles
        monkeypatch.setattr(A, "tree_header", changing)
    with pytest.raises(ValueError): saved.run()
    assert not saved.args["out"].exists()


def test_scalar_domains_and_nonfinite():
    counts = A.summary(np.array([-1., 0., -2., 2., np.nan, np.inf]), "energy")["counts"]
    assert counts["invalid"] == 3 and counts["sentinel_minus_one"] == 1 and counts["zero"] == 1
    assert A.summary(np.array([-1, 0, 1, 2]), "bool")["counts"]["invalid"] == 2
    times = A.summary(np.array([-1., 0., -2., np.nan]), "time")
    assert times["counts"]["valid"] == 3 and times["counts"]["invalid"] == 1
    assert times["counts"]["sentinel_minus_one"] == 0
    assert times["valid_statistics"]["min"] == -2.
    assert A.summary(np.array([-1., np.nan]), "time")["valid_statistics"]["min"] == -1.


def test_hook_restoration_on_failure(saved, monkeypatch):
    original = uproot.reading.ReadOnlyDirectory.hook_before_keys
    monkeypatch.setattr(A, "MAX_KEYS", 0)
    with pytest.raises(ValueError): saved.run()
    assert uproot.reading.ReadOnlyDirectory.hook_before_keys is original


@pytest.mark.parametrize("mutation", ["missing_id", "alias_id", "float_count", "int_time", "vector", "bool_bad_id"])
def test_exact_required_scalar_leaf_domains(saved, mutation):
    row = saved.report["ntuples"][0]
    data = {name: np.arange(4, dtype=np.int32) for name in A.IDS}
    if mutation == "missing_id": del data[A.IDS[0]]
    elif mutation == "alias_id": data["evtinfo.run"] = data.pop(A.IDS[0])
    elif mutation == "float_count": data["evtinfo/nprotons"] = np.ones(4, dtype=np.float32)
    elif mutation == "int_time": data["evtinfo/pbtime"] = np.ones(4, dtype=np.int32)
    elif mutation == "vector": data["evtinfo/nprotons"] = np.ones((4, 2), dtype=np.int32)
    elif mutation == "bool_bad_id": data[A.IDS[0]] = np.ones(4, dtype=np.bool_)
    with uproot.update(saved.mapping[row["path"]]) as root:
        root.mktree("EventNtuple/ntuple", data)
    with pytest.raises(ValueError): saved.run()
    assert not saved.args["out"].exists()


def test_highest_cycle_and_wrong_version_class_not_deserialized(saved, monkeypatch):
    row = saved.report["ntuples"][0]
    with uproot.update(saved.mapping[row["path"]]) as root:
        root.mktree("EventNtuple/ntuple", {
            A.IDS[0]: np.full(4, 1430, dtype=np.int32), A.IDS[1]: np.zeros(4, dtype=np.int32),
            A.IDS[2]: np.arange(4, dtype=np.int32)})
        root["EventNtuple/version"] = "not TH1I"
    original = uproot.reading.ReadOnlyKey.get
    def get(self):
        assert self.fClassName != "TObjString"
        return original(self)
    monkeypatch.setattr(uproot.reading.ReadOnlyKey, "get", get)
    result = saved.run()["files"][0]
    assert result["chosen_cycles"]["ntuple"] == 2
    assert result["version"]["cycle"] == 2
    assert not result["version"]["available"]


def test_version_key_cap_before_deserialization(saved, monkeypatch):
    monkeypatch.setattr(A, "VERSION_CAP", 1)
    original = uproot.reading.ReadOnlyKey.get
    def get(self):
        assert self.fName != "version", "oversize version must not deserialize"
        return original(self)
    monkeypatch.setattr(uproot.reading.ReadOnlyKey, "get", get)
    with pytest.raises(ValueError, match="version object"):
        saved.run()


def test_nonallowlisted_branch_admission_is_impossible():
    with pytest.raises(ValueError, match="not allowlisted"):
        A.branch_metadata(object(), "evtinfomc/nprotons", 4)


def test_invalid_id_domain(saved):
    row = saved.report["ntuples"][0]
    with uproot.update(saved.mapping[row["path"]]) as root:
        root.mktree("EventNtuple/ntuple", {name: np.array([-1, 0, 1, 2], dtype=np.int32) for name in A.IDS})
    with pytest.raises(ValueError, match="full ID domain"):
        saved.run()


def test_summaries_extremes_and_fractional_negative():
    finite = np.finfo(np.float64).max
    result = A.summary(np.array([finite, finite]), "energy")
    assert result["valid_statistics"]["mean"] == finite
    assert A.summary(np.array([-.5, -1., 0.]), "energy")["counts"]["invalid"] == 1
    assert A.summary(np.array([-2., -1., 0.]), "error")["counts"]["invalid"] == 2


def test_duplicate_json_keys_and_nonfinite():
    for data in (b'{"selection":{},"selection":{}}', b'{"anything":NaN}'):
        with pytest.raises(ValueError): A.bindings(data)


def test_output_excludes_saved_ledger_parent_without_reading_it(saved, monkeypatch):
    parent = saved.args["out"].parent / "old_ledger"
    parent.mkdir()
    saved.report["exposure_ledger"] = dict(path=str(parent / "never_open.parquet"), sha256="a" * 64)
    saved.repin()
    monkeypatch.setattr(A, "BoundedFile", forbidden)
    with pytest.raises(ValueError, match="outside"):
        saved.run(out=parent / "new.json")


def test_reopen_read_budget_counts_toward_total(tmp_path, monkeypatch):
    paths = [tmp_path / "a", tmp_path / "b"]
    for path in paths: path.write_bytes(b"12345")
    budget = A.Budget()
    monkeypatch.setattr(A, "TOTAL_CAP", 7)
    with A.BoundedFile(str(paths[0]), budget) as stream:
        assert stream.read(5) == b"12345"
    with A.BoundedFile(str(paths[1]), budget) as stream:
        assert stream.read(2) == b"12"
        with pytest.raises(ValueError, match="total read cap"): stream.read(1)
    assert budget.total == 7


# A direct XRootD client double: only File.open/read/close exist. All payloads
# are in-memory bytes from synthetic ROOT files, never a remote service.
class FakeStatus:
    def __init__(self, ok=True, text="OK"):
        self.ok, self.text = ok, text

    def __str__(self):
        return self.text


class FakeXRootD:
    def __init__(self, monkeypatch):
        import sys
        from types import ModuleType
        self.payloads, self.instances, self.calls = {}, [], []
        self.status = {name: FakeStatus() for name in ("open", "read", "close")}
        self.errors, self.read_result = {}, None
        owner = self

        class File:
            __slots__ = ("url",)

            def __init__(self):
                self.url = None
                owner.instances.append(self)

            def __getattr__(self, name):
                pytest.fail("unexpected XRootD API: " + name)

            def open(self, url, *, flags, mode, timeout):
                assert flags == 0 and mode == 0 and timeout == 30
                assert url in owner.payloads
                self.url = url
                owner.calls.append((self, "open", dict(url=url, flags=flags, mode=mode, timeout=timeout)))
                if "open" in owner.errors: raise owner.errors["open"]
                return owner.status["open"], None

            def read(self, *, offset, size, timeout):
                assert type(offset) is int and offset >= 0
                assert type(size) is int and 0 < size <= A.READ_CAP and timeout == 30
                owner.calls.append((self, "read", dict(offset=offset, size=size, timeout=timeout)))
                if "read" in owner.errors: raise owner.errors["read"]
                data = owner.payloads[self.url][offset:offset + size]
                if owner.read_result is not None: data = owner.read_result
                return owner.status["read"], data

            def close(self, *, timeout):
                assert timeout == 30
                owner.calls.append((self, "close", dict(timeout=timeout)))
                if "close" in owner.errors: raise owner.errors["close"]
                return owner.status["close"], None

        package, client = ModuleType("XRootD"), ModuleType("XRootD.client")
        package.__version__ = "5.8.4-test-double"
        package.client, client.File = client, File
        monkeypatch.setitem(sys.modules, "XRootD", package)
        monkeypatch.setitem(sys.modules, "XRootD.client", client)

    def add(self, path, data):
        url = "root://door.test//ns" + path[len("/dcache-test"):]
        self.payloads[url] = data
        return url


@pytest.fixture
def xclient(monkeypatch):
    return FakeXRootD(monkeypatch)


def pinned_path(index=0):
    return ("/dcache-test/mu2e/tape/phy-nts/nts/mu2e/" + A.CAMPAIGN +
        f"/root/ab/cd/nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_{index:08d}.root")


def test_xrootd_exact_calls_zero_read_lexical_seek_and_reopens(xclient, monkeypatch):
    path, budget = pinned_path(), A.Budget()
    url = xclient.add(path, b"abcdefgh")
    for name in ("open", "read", "close", "lseek", "stat", "fstat", "lstat", "listdir", "scandir"):
        monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(A, "FILE_CAP", 10)
    with A.XRootDBoundedFile(path, budget) as stream:
        assert stream.tell() == 0
        assert stream.read(0) == b"" and stream.read(np.int64(0)) == b""
        assert len(xclient.calls) == 1
        assert stream.read(np.int64(4)) == b"abcd" and stream.tell() == 4
        assert stream.seek(np.int64(-2), os.SEEK_CUR) == 2
        assert stream.read(2) == b"cd" and stream.tell() == 4
        assert stream.seek(1000) == stream.tell() == 1000  # no remote seek/stat
        assert stream.seek(0) == 0
    stream.close()  # idempotent; no repeated remote close
    with A.XRootDBoundedFile(path, budget) as stream:
        assert stream.tell() == 0
        assert stream.read(4) == b"abcd"
        assert stream.read(0) == b""  # even at the exhausted budget
        with pytest.raises(ValueError, match="per-file"): stream.read(1)
    assert budget.files == {path: 10} and budget.total == 10
    assert [(op, args) for _, op, args in xclient.calls] == [
        ("open", dict(url=url, flags=0, mode=0, timeout=30)),
        ("read", dict(offset=0, size=4, timeout=30)),
        ("read", dict(offset=2, size=2, timeout=30)), ("close", dict(timeout=30)),
        ("open", dict(url=url, flags=0, mode=0, timeout=30)),
        ("read", dict(offset=0, size=4, timeout=30)), ("close", dict(timeout=30))]
    for action in (lambda: stream.read(0), stream.tell, lambda: stream.seek(0)):
        with pytest.raises(ValueError, match="closed"): action()


@pytest.mark.parametrize("size", [-1, None, True, np.bool_(False), 1.0, "1", A.READ_CAP + 1])
def test_xrootd_rejects_nonbounded_reads(xclient, size):
    path, budget = pinned_path(), A.Budget()
    xclient.add(path, b"abc")
    with A.XRootDBoundedFile(path, budget) as stream:
        with pytest.raises(ValueError, match="bounded single"): stream.read(size)
    assert [op for _, op, _ in xclient.calls] == ["open", "close"]
    assert budget.total == 0


@pytest.mark.parametrize("offset,whence", [(0, os.SEEK_END), (0, 99), (-1, os.SEEK_SET),
    (-1, os.SEEK_CUR), (True, os.SEEK_SET), (1.0, os.SEEK_SET), (0, True), (0, 0.0)])
def test_xrootd_rejects_nonlexical_seek(xclient, offset, whence):
    path = pinned_path()
    xclient.add(path, b"abc")
    with A.XRootDBoundedFile(path, A.Budget()) as stream:
        with pytest.raises(ValueError): stream.seek(offset, whence)
        assert stream.tell() == 0
    assert [op for _, op, _ in xclient.calls] == ["open", "close"]


@pytest.mark.parametrize("mutation", ["url", "prefix", "campaign", "traversal", "query", "index", "relative"])
def test_xrootd_rejects_nonexact_paths_before_client(xclient, mutation):
    path = pinned_path()
    if mutation == "url": path = "root://door.test//ns" + path[len("/dcache-test"):]
    elif mutation == "prefix": path = path.replace("/dcache-test/mu2e/", "/dcache-test/mu2e-other/")
    elif mutation == "campaign": path = path.replace(A.CAMPAIGN, "Other/v1")
    elif mutation == "traversal": path = path.replace("/root/", "/root/../root/")
    elif mutation == "query": path += "?alternate=1"
    elif mutation == "index": path = pinned_path(7)
    elif mutation == "relative": path = path[1:]
    with pytest.raises(ValueError, match="exact PNFS"):
        A.XRootDBoundedFile(path, A.Budget())
    assert not xclient.instances and not xclient.calls


@pytest.mark.parametrize("operation", ["open", "read", "close"])
@pytest.mark.parametrize("failure", ["status", "exception"])
def test_xrootd_failures_close_with_timeout_no_retry(xclient, operation, failure):
    path, budget = pinned_path(), A.Budget()
    xclient.add(path, b"abcd")
    if failure == "status": xclient.status[operation] = FakeStatus(False, "SERVER_FAILURE_123")
    else: xclient.errors[operation] = OSError("SERVER_FAILURE_123")
    with pytest.raises(OSError, match="SERVER_FAILURE_123") as exc:
        with A.XRootDBoundedFile(path, budget) as stream:
            stream.read(4)
    if failure == "status":
        assert operation in str(exc.value) and "status=" in str(exc.value)
    assert len(xclient.instances) == 1
    expected = ["open", "close"] if operation == "open" else ["open", "read", "close"]
    assert [op for _, op, _ in xclient.calls] == expected
    assert xclient.calls[-1][2] == {"timeout": 30}
    if operation == "close": stream.close()  # even failed close is not retried
    assert len(xclient.calls) == len(expected)


@pytest.mark.parametrize("data,message", [(b"xy", "short read"), (b"", "short read"),
    (b"abcde", "exceeds requested"), ("abcd", "not bytes"), (bytearray(b"abcd"), "not bytes")])
def test_xrootd_actual_byte_accounting_and_payload_validation(xclient, data, message):
    path, budget = pinned_path(), A.Budget()
    xclient.add(path, b"abcdefgh")
    xclient.read_result = data
    with pytest.raises(ValueError, match=message) as exc:
        with A.XRootDBoundedFile(path, budget) as stream:
            stream.read(4)
    assert "status=OK" in str(exc.value)
    actual = len(data) if isinstance(data, bytes) else 0
    assert budget.total == budget.files[path] == actual
    assert stream.position == actual
    assert [op for _, op, _ in xclient.calls] == ["open", "read", "close"]


def test_xrootd_total_budget_across_exact_paths_and_reopen(xclient, monkeypatch):
    paths, budget = [pinned_path(i) for i in (0, 1)], A.Budget()
    for path in paths: xclient.add(path, b"abcdefgh")
    monkeypatch.setattr(A, "TOTAL_CAP", 7)
    with A.XRootDBoundedFile(paths[0], budget) as stream:
        assert stream.read(3) == b"abc"
    with A.XRootDBoundedFile(paths[0], budget) as stream:
        assert stream.read(2) == b"ab"
    with A.XRootDBoundedFile(paths[1], budget) as stream:
        assert stream.read(2) == b"ab"
        with pytest.raises(ValueError, match="total"): stream.read(1)
        assert stream.read(0) == b""
    assert budget.files == {paths[0]: 5, paths[1]: 2} and budget.total == 7
    assert len([1 for _, op, _ in xclient.calls if op == "read"]) == 3


def test_xrootd_synthetic_real_root_header_reopens_and_report(saved, xclient, monkeypatch):
    urls = {path: xclient.add(path, file.read_bytes()) for path, file in saved.mapping.items()}
    original_array = uproot.behaviors.TBranch.TBranch.array
    seen = []
    def array(self, *args, **kwargs):
        assert self.name in A.KINDS
        assert kwargs["library"] == "np" and kwargs["array_cache"] is None
        seen.append(self.name)
        return original_array(self, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.TBranch, "array", array)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", forbidden)
    monkeypatch.setattr(A, "BoundedFile", forbidden)  # no NFS open or fallback
    for name in ("open", "read", "close", "lseek", "listdir", "scandir", "fstat"):
        monkeypatch.setattr(os, name, forbidden)
    for name in ("stat", "lstat"):
        original = getattr(os, name)
        def guard(path, *args, _original=original, **kwargs):
            assert not str(path).startswith(("/dcache-test", "root://"))
            assert str(path) != str(saved.selected)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(os, name, guard)
    report = saved.run(transport="xrootd")
    assert report["transport"] == "xrootd" and report["resolved_urls"] == urls
    assert report["software"]["xrootd"] == "5.8.4-test-double"
    assert report["transport_details"] == dict(api="XRootD.client.File",
        per_call_timeout_seconds=dict(open=30, read=30, close=30), application_retries=0,
        automatic_fallback=False, short_read_policy="raise")
    assert len(xclient.instances) == 22 and len(seen) == 99
    assert len(report["files"]) == 11 and report["total_entries"] == 44
    assert report["full_ids_unique"] and not report["full_file_hashes"]
    assert not report["mc_read"] and not report["selection_executed"]
    for row in report["files"]:
        assert row["transport"] == "xrootd" and row["resolved_url"] == urls[row["path"]]
        assert row["read_bytes"] > row["read_bytes_before_reopen"]
        assert row["header_reopened"] and row["version"]["major_minor_patch"] == [6, 13, 2]
        assert row["scalars"]["lumistream./lumistream.caloEnergy"]["counts"]["sentinel_minus_one"] == 1
        assert row["scalars"]["evtinfo/pbtime"]["counts"]["sentinel_minus_one"] == 0
        instances = [inst for inst in xclient.instances if inst.url == row["resolved_url"]]
        assert len(instances) == 2
        read_sizes = []
        for inst in instances:
            calls = [(op, args) for item, op, args in xclient.calls if item is inst]
            assert calls[0] == ("open", dict(url=row["resolved_url"], flags=0, mode=0, timeout=30))
            assert calls[1] == ("read", dict(offset=0, size=100, timeout=30))
            assert calls[-1] == ("close", dict(timeout=30))
            assert sum(op == "close" for op, _ in calls) == 1
            read_sizes.append(sum(args["size"] for op, args in calls if op == "read"))
        assert row["read_bytes_before_reopen"] == read_sizes[0]
        assert row["read_bytes"] == sum(read_sizes) == report["per_file_read_bytes"][row["path"]]
    assert report["read_bytes"] == sum(report["per_file_read_bytes"].values())
    assert json.loads(saved.args["out"].read_text()) == report


@pytest.mark.parametrize("cap,value", [("FILE_CAP", 99), ("TOTAL_CAP", 99), ("READ_CAP", 99),
    ("MAX_KEYS", 1), ("MAX_BRANCHES", 1), ("MAX_BASKETS", 0), ("TOTAL_BASKETS", 1),
    ("BRANCH_BYTES", 1), ("PAYLOAD_CAP", 1), ("VERSION_CAP", 1), ("VERSION_BINS", 2),
    ("METADATA_CAP", 1), ("SOURCE_CAP", 1), ("JSON_CAP", 1), ("OUTPUT_CAP", 1)])
def test_xrootd_reuses_all_existing_caps(saved, xclient, monkeypatch, cap, value):
    for path, file in saved.mapping.items(): xclient.add(path, file.read_bytes())
    monkeypatch.setattr(A, "BoundedFile", forbidden)
    monkeypatch.setattr(A, cap, value)
    with pytest.raises(ValueError): saved.run(transport="xrootd")
    assert not saved.args["out"].exists()
    assert len([1 for _, op, _ in xclient.calls if op == "open"]) == len(
        [1 for _, op, _ in xclient.calls if op == "close"])


@pytest.mark.parametrize("mutation", ["pin", "hash", "path", "uuid", "entries"])
def test_xrootd_pinned_validation_before_import(saved, monkeypatch, mutation):
    import sys
    monkeypatch.setitem(sys.modules, "XRootD", None)
    if mutation == "pin": saved.args["expected_selection_sha256"] = "f" * 64
    elif mutation == "hash": saved.args["selection_report"].write_text("{}")
    else:
        row = saved.report["ntuples"][0]
        if mutation == "path": row["path"] = row["path"].replace(A.CAMPAIGN, "Other/v1")
        elif mutation == "uuid": row["uuid"] = "bad"
        elif mutation == "entries": row["trees"]["EventNtuple/ntuple"]["entries"] = 0
        saved.repin()
    with pytest.raises(ValueError): saved.run(transport="xrootd")
    assert not saved.args["out"].exists()


def test_nfs_default_does_not_import_xrootd(saved, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "XRootD", None)
    monkeypatch.setitem(sys.modules, "XRootD.client", None)
    report = saved.run()
    assert report["transport"] == "nfs" and report["resolved_urls"] == {}
    assert "xrootd" not in report["software"]
    assert report["transport_details"]["per_call_timeout_seconds"] is None
    assert all(row["transport"] == "nfs" and row["resolved_url"] is None for row in report["files"])


def test_xrootd_missing_dependency_no_fallback(saved, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "XRootD", None)
    monkeypatch.setattr(A, "BoundedFile", forbidden)
    with pytest.raises(ImportError): saved.run(transport="xrootd")
    assert not saved.args["out"].exists()


@pytest.mark.parametrize("transport", [None, "nfs", "xrootd", "unknown"])
def test_transport_cli_is_opt_in(monkeypatch, transport, capsys):
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return dict(schema=A.SCHEMA, total_entries=0, read_bytes=0)
    monkeypatch.setattr(A, "run", run)
    args = ["pinned.json", "--expected-selection-sha256", A.SELECTION_SHA256, "--out", "new.json"]
    if transport is not None: args += ["--transport", transport]
    if transport == "unknown":
        with pytest.raises(SystemExit): A.main(args)
        assert not calls
    else:
        A.main(args)
        assert calls == [dict(selection_report="pinned.json", expected_selection_sha256=A.SELECTION_SHA256,
            out="new.json", transport=transport or "nfs")]
        assert json.loads(capsys.readouterr().out)["schema"] == A.SCHEMA


def test_xrootd_failed_status_with_partial_payload_is_charged(xclient):
    path, budget = pinned_path(), A.Budget()
    xclient.add(path, b"abcdefgh")
    xclient.read_result = b"ab"
    xclient.status["read"] = FakeStatus(False, "PARTIAL_FAILURE")
    with pytest.raises(OSError, match="status=PARTIAL_FAILURE"):
        with A.XRootDBoundedFile(path, budget) as stream:
            stream.read(4)
    assert budget.total == budget.files[path] == stream.position == 2
    assert [op for _, op, _ in xclient.calls] == ["open", "read", "close"]


def test_xrootd_single_read_cap_boundary(xclient, monkeypatch):
    path, budget = pinned_path(), A.Budget()
    xclient.add(path, b"abcde")
    monkeypatch.setattr(A, "READ_CAP", 4)
    with A.XRootDBoundedFile(path, budget) as stream:
        assert stream.read(4) == b"abcd"
        with pytest.raises(ValueError, match="bounded single"): stream.read(5)
    assert budget.total == 4
    assert [op for _, op, _ in xclient.calls] == ["open", "read", "close"]


@pytest.mark.parametrize("operation", ["open", "read", "close", "short"])
def test_xrootd_audit_transport_errors_never_publish_or_fallback(saved, xclient, monkeypatch, operation):
    for path, file in saved.mapping.items(): xclient.add(path, file.read_bytes())
    monkeypatch.setattr(A, "BoundedFile", forbidden)
    if operation == "short":
        xclient.read_result = b"r"
    else:
        xclient.status[operation] = FakeStatus(False, "ERROR_" + operation)
    with pytest.raises((OSError, ValueError), match="short read" if operation == "short" else "ERROR_" + operation):
        saved.run(transport="xrootd")
    assert not saved.args["out"].exists()
    assert len(xclient.instances) == 1
    assert xclient.calls[-1][1:] == ("close", {"timeout": 30})
    assert sum(op == "close" for _, op, _ in xclient.calls) == 1


@pytest.mark.parametrize("mutation", ["header", "open_failure", "short"])
def test_xrootd_header_reopen_failure_closes_without_output(saved, xclient, monkeypatch, mutation):
    from XRootD import client
    for path, file in saved.mapping.items(): xclient.add(path, file.read_bytes())
    monkeypatch.setattr(A, "BoundedFile", forbidden)
    original_open = client.File.open
    def opening(self, url, **kwargs):
        if len(xclient.instances) == 2:
            if mutation == "header": xclient.payloads[url] = b"x" + xclient.payloads[url][1:]
            elif mutation == "open_failure": xclient.status["open"] = FakeStatus(False, "REOPEN_FAILURE")
            else: xclient.read_result = b"r"
        return original_open(self, url, **kwargs)
    monkeypatch.setattr(client.File, "open", opening)
    match = {"header": "ROOT header changed", "open_failure": "REOPEN_FAILURE", "short": "short read"}[mutation]
    with pytest.raises((OSError, ValueError), match=match): saved.run(transport="xrootd")
    assert not saved.args["out"].exists()
    assert len(xclient.instances) == 2
    assert xclient.instances[0].url == xclient.instances[1].url
    assert sum(op == "close" for _, op, _ in xclient.calls) == 2
    assert xclient.calls[-1][1:] == ("close", {"timeout": 30})


@pytest.mark.parametrize("operation", ["open", "read"])
def test_xrootd_close_failure_preserves_prior_error_context(xclient, operation):
    path = pinned_path()
    xclient.add(path, b"abcd")
    xclient.status[operation] = FakeStatus(False, "FIRST_FAILURE")
    xclient.status["close"] = FakeStatus(False, "CLOSE_FAILURE")
    with pytest.raises(OSError, match="status=CLOSE_FAILURE") as exc:
        with A.XRootDBoundedFile(path, A.Budget()) as stream:
            stream.read(4)
    assert "status=FIRST_FAILURE" in str(exc.value.__context__)
    assert sum(op == "close" for _, op, _ in xclient.calls) == 1


@pytest.mark.parametrize("source", ["package", "client", "metadata", "unavailable"])
def test_xrootd_version_best_effort(xclient, monkeypatch, source):
    import XRootD
    from XRootD import client
    from importlib import metadata
    if source != "package": monkeypatch.delattr(XRootD, "__version__")
    if source == "client": monkeypatch.setattr(client, "__version__", "5.8.4-client", raising=False)
    def version(name):
        assert name == "XRootD"
        if source == "unavailable": raise metadata.PackageNotFoundError(name)
        assert source == "metadata"
        return "5.8.4-distribution"
    monkeypatch.setattr(metadata, "version", version)
    assert A.xrootd_version() == {"package": "5.8.4-test-double", "client": "5.8.4-client",
        "metadata": "5.8.4-distribution", "unavailable": None}[source]


def test_unknown_transport_before_file_access(monkeypatch):
    monkeypatch.setattr(A, "small_bytes", forbidden)
    with pytest.raises(ValueError, match="unknown transport"):
        A.run("unopened", "unopened", expected_selection_sha256=A.SELECTION_SHA256, transport="http")


def test_metadata_rejection_context_is_bounded_and_preserves_guard(monkeypatch):
    class Branch:
        classname = "TBranch"
        members = dict(fName="x" * 1000, fEntries=4, fMaxBaskets=2, fWriteBasket=0,
                       fBasketSeek=np.zeros(2, dtype=np.int64), fBasketEntry=np.zeros(2, dtype=np.int64),
                       fBranches=[object()], fLeaves=[], fTotBytes=1, fZipBytes=0,
                       fBasketBytes=np.zeros(2, dtype=np.int32))

        def member(self, name):
            return self.members[name]

    monkeypatch.setattr(uproot.model.Model, "hook_before_postprocess", forbidden)
    with A.metadata_limits():
        with pytest.raises(ValueError, match="embedded baskets forbidden during metadata") as exc:
            uproot.model.Model.hook_before_postprocess(Branch())
    message = str(exc.value)
    assert "name=" + repr("x" * 256) + ", entries=4" in message
    assert "normal=0, boundary=0, written=0, children=1, leaves=0, total_bytes=1, zip_bytes=0" in message
    assert len(message) < 512
    assert uproot.model.Model.hook_before_postprocess is forbidden


def basket_envelope(*, tags=(), name=b"", size=None, lower=0, version=3, tobversion=1, bits=0):
    import struct
    body = struct.pack(">HHII", version, tobversion, 0, bits)
    body += uproot.serialization.string(name.decode("ascii"))
    body += struct.pack(">ii", len(tags) if size is None else size, lower)
    body += b"".join(struct.pack(">I", tag) for tag in tags)
    return struct.pack(">I", 0x40000000 | len(body)) + body


def resident_metadata(payload, start=73):
    from types import SimpleNamespace
    source = SimpleNamespace(chunk=forbidden, chunks=forbidden, file_path="synthetic-metadata")
    chunk = uproot.source.chunk.Chunk.wrap(source, payload, start=start)
    chunk.wait()
    cursor = uproot.source.cursor.Cursor(start, origin=start, refs={7: object()})
    return chunk, cursor


@pytest.mark.parametrize("tags,name", [((), b""), ((0,), b"baskets"), ((0, 0), b""),
                                       ((0,) * A.TOTAL_BASKETS, b"x" * 256)])
def test_serialized_empty_baskets_proof_is_read_only(tags, name, monkeypatch):
    payload = basket_envelope(tags=tags, name=name)
    chunk, cursor = resident_metadata(payload)
    refs = dict(cursor.refs)
    monkeypatch.setattr(uproot.deserialization, "read_object_any", forbidden)
    monkeypatch.setattr(uproot.models.TObjArray.Model_TObjArray, "read_members", forbidden)
    monkeypatch.setattr(chunk, "get", forbidden)
    assert A.empty_basket_array(chunk, cursor, max(2, len(tags))) == chunk.stop
    assert cursor.index == chunk.start and cursor.refs == refs


@pytest.mark.parametrize("kwargs", [dict(tags=(1,)), dict(tags=(7,)), dict(tags=(999,)),
    dict(tags=(0xffffffff,)), dict(tags=(0x80000007,)), dict(tags=(0x40000000, 0)),
    dict(tags=(0, 1)), dict(size=-1), dict(size=3), dict(lower=1), dict(lower=-1),
    dict(version=2), dict(version=0x4003), dict(tobversion=0x4001), dict(tobversion=2),
    dict(bits=uproot.const.kIsReferenced), dict(name=b"x" * 257), dict(tags=(0,), size=0)])
def test_serialized_basket_proof_rejects_unsupported_before_resolution(kwargs, monkeypatch):
    chunk, cursor = resident_metadata(basket_envelope(**kwargs))
    monkeypatch.setattr(uproot.deserialization, "read_object_any", forbidden)
    with pytest.raises(ValueError):
        A.empty_basket_array(chunk, cursor, 2)
    assert cursor.index == chunk.start


@pytest.mark.parametrize("length", range(25))
def test_serialized_basket_proof_rejects_every_truncation(length):
    chunk, cursor = resident_metadata(basket_envelope()[:length])
    with pytest.raises(ValueError, match="truncated|envelope"):
        A.empty_basket_array(chunk, cursor, 2)


@pytest.mark.parametrize("count", [0, 21, 0xffffffff, 0xc0000015, 0x40000000,
                                   0x40000014, 0x40000016, 0x40800000])
def test_serialized_basket_proof_rejects_bytecount_envelope(count):
    import struct
    chunk, cursor = resident_metadata(struct.pack(">I", count) + basket_envelope()[4:])
    with pytest.raises(ValueError, match="bytecount|envelope"):
        A.empty_basket_array(chunk, cursor, 2)


@pytest.mark.parametrize("bad", ["missing", "pending", "memmap", "outside", "short", "maximum"])
def test_serialized_basket_proof_requires_bounded_resident_bytes(bad):
    chunk, cursor = resident_metadata(basket_envelope())
    maximum = 2
    if bad == "missing": chunk = None
    if bad == "pending": chunk._raw_data = None
    if bad == "memmap": chunk._is_memmap = True
    if bad == "outside": cursor.move_to(chunk.start - 1)
    if bad == "short": chunk._raw_data = chunk._raw_data[:-1]
    if bad == "maximum": maximum = A.TOTAL_BASKETS + 1
    with pytest.raises(ValueError, match="resident|allocation"):
        A.empty_basket_array(chunk, cursor, maximum)


def synthetic_branch(*, children=1, entries=4, written=0, seeks=(0, 0), offsets=(0, 0),
                     sizes=(0, 0), total=0, zipped=0, maximum=2, leaves=1):
    from types import SimpleNamespace
    branch = uproot.models.TBranch.Model_TBranch_v13.empty()
    branch._members.update(fName="evtinfo", fEntries=entries, fWriteBasket=written,
                          fBasketSeek=np.array(seeks, dtype=np.int64), fBasketEntry=np.array(offsets, dtype=np.int64),
                          fBasketBytes=np.array(sizes, dtype=np.int32), fTotBytes=total, fZipBytes=zipped,
                          fMaxBaskets=maximum, fBranches=[SimpleNamespace(member=lambda name: "child")] * children,
                          fLeaves=[object()] * leaves)
    branch._parent = SimpleNamespace(members={})
    branch._file = SimpleNamespace(source=SimpleNamespace(chunk=forbidden), options={"minimal_ttree_metadata": True})
    return branch


def process_synthetic_branch(branch, payload=None):
    chunk, cursor = resident_metadata(basket_envelope() if payload is None else payload)
    branch._cursor_baskets = cursor.copy()
    branch.hook_before_postprocess(chunk=chunk, cursor=cursor, context={}, file=branch._file)
    branch.postprocess(chunk, cursor, {}, branch._file)
    return chunk, cursor


@pytest.mark.parametrize("leaves", [0, 1, 6])
@pytest.mark.parametrize("tags", [(), (0, 0)])
def test_structural_parent_and_derived_postprocess_without_lazy_baskets(leaves, tags, monkeypatch):
    branch = synthetic_branch(leaves=leaves)
    original_offsets = branch.member("fBasketEntry")
    monkeypatch.setattr(uproot.models.TObjArray.Model_TObjArrayOfTBaskets, "read_members", forbidden)
    with A.metadata_limits():
        chunk, cursor = process_synthetic_branch(branch, basket_envelope(tags=tags))
        assert branch.member("fEntries") == 4 and branch.member("fBasketEntry") is original_offsets
        assert branch.has_member("fBaskets") and branch.member("fBaskets") == []
        assert branch.embedded_baskets == [] and branch._embedded_baskets_lock is None
        derived = uproot.models.TBranch.Model_TBranchElement_v10.empty()
        derived._bases.append(branch)
        derived._file, derived._parent = branch._file, branch._parent
        derived._cursor_baskets = branch._cursor_baskets
        derived.hook_before_postprocess(chunk=chunk, cursor=cursor, context={}, file=branch._file)
        derived.postprocess(chunk, cursor, {}, branch._file)
        assert derived.has_member("fBaskets") and derived.embedded_baskets == []
        assert derived.num_baskets == 0 and derived.num_entries == 4


@pytest.mark.parametrize("changes", [dict(children=0), dict(total=1), dict(zipped=1),
    dict(total=-1), dict(zipped=-1), dict(written=1), dict(seeks=(0, 8)), dict(sizes=(0, 1)),
    dict(offsets=(1, 0)), dict(offsets=(0, 4)), dict(entries=-1), dict(entries=A.MAX_ENTRIES + 1),
    dict(children=A.MAX_BRANCHES + 1), dict(leaves=A.MAX_BRANCHES + 1), dict(maximum=0),
    dict(maximum=A.TOTAL_BASKETS + 1), dict(sizes=(0,)), dict(seeks=(0,)), dict(written=-1)])
def test_structural_parent_exception_is_narrow(changes):
    branch = synthetic_branch(**changes)
    with A.metadata_limits(), pytest.raises(ValueError):
        process_synthetic_branch(branch)
    assert not branch.has_member("fBaskets")


@pytest.mark.parametrize("children", [0, 1])
@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize("tag", [1, 7, 999, 0xffffffff, 0x80000007])
def test_nonnull_baskets_rejected_even_with_matching_coverage(children, matching, tag, monkeypatch):
    branch = synthetic_branch(children=children, entries=0 if matching else 4)
    monkeypatch.setattr(uproot.deserialization, "read_object_any", forbidden)
    with A.metadata_limits(), pytest.raises(ValueError, match="embedded baskets forbidden"):
        process_synthetic_branch(branch, basket_envelope(tags=(tag,)))
    assert not branch.has_member("fBaskets")


def test_basket_header_fallback_is_rejected_before_source_chunk():
    branch = synthetic_branch(written=1, seeks=(80, 0), offsets=(0,))
    with A.metadata_limits(), pytest.raises(ValueError, match="basket-header fallback"):
        process_synthetic_branch(branch)


def test_selected_leaf_still_requires_positive_normal_baskets():
    branch = synthetic_branch(children=0, entries=0)
    branch._interpretation = uproot.interpretation.numerical.AsDtype(np.dtype("i4"))
    with A.metadata_limits():
        process_synthetic_branch(branch)
        branch._interpretation = uproot.interpretation.numerical.AsDtype(np.dtype("i4"))
        with pytest.raises(ValueError, match="basket count/embedded basket cap"):
            A.branch_metadata(branch, "evtinfo/run", 0)


def serialized_branch(*, children=(), entries=4, tags=(), derived=False, malformed=None,
                      written=0, seeks=(0, 0), offsets=(0, 0), sizes=(0, 0),
                      base_version=13, element_version=10):
    import struct
    def envelope(body, version):
        return uproot.serialization.numbytes_version(len(body), version) + body
    def objects(items):
        body = b"\x00\x01" + struct.pack(">II", 0, 0) + b"\0" + struct.pack(">ii", len(items), 0)
        for classname, payload in items:
            body += b"\xff\xff\xff\xff" + classname.encode() + b"\0" + payload
        return envelope(body, 3)
    named = envelope(b"\x00\x01" + struct.pack(">II", 0, 0) + b"\x07evtinfo\0", 1)
    fill = envelope(struct.pack(">hh", 0, 0), 2)
    body = named + fill
    if base_version == 13:
        body += uproot.models.TBranch._tbranch13_format1.pack(0, 32000, 0, written, entries)
        body += b"@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00"
        body += uproot.models.TBranch._tbranch13_format2.pack(0, 2, 1, entries, 0, 0, 0)
    else:
        fmt = getattr(uproot.models.TBranch, f"_tbranch{base_version}_format1")
        values = [0, 32000, 0, written, entries, 0, 2, 1, entries, 0, 0]
        if base_version != 10:
            values.insert(9, 0)
        body += fmt.pack(*values)
    body += objects(children) + objects([])
    baskets = basket_envelope(tags=tags)
    if malformed == "no-bytecount": baskets = baskets[4:]
    if malformed == "truncated": baskets = baskets[:-1]
    body += baskets + b"\1" + struct.pack(">ii", *sizes)
    body += b"\1" + struct.pack(">qq", *offsets) + b"\1" + struct.pack(">qq", *seeks) + b"\0"
    base = envelope(body, base_version)
    if derived:
        fmt = getattr(uproot.models.TBranch, f"_tbranchelement{element_version}_format1")
        tail = b"\0\0\0" + fmt.pack(0, 1, 0, 0, 0, 0) + bytes(8)
        return "TBranchElement", envelope(base + tail, element_version)
    return "TBranch", base


def read_serialized_branch(classname, payload, *, minimal=True):
    from types import SimpleNamespace
    def class_named(name, version=None):
        cls = uproot.classes[name]
        if version is not None and hasattr(cls, "known_versions"):
            return cls.known_versions[version]
        return cls
    chunk, cursor = resident_metadata(payload)
    file = SimpleNamespace(options={"minimal_ttree_metadata": minimal}, class_named=class_named,
                           file_path="synthetic-metadata", source=chunk.source)
    parent = SimpleNamespace(members={})
    return class_named(classname).read(chunk, cursor, {}, file, file, parent)


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("tags", [(), (0, 0)])
def test_serialized_parent_base_and_derived_integration(derived, tags, monkeypatch):
    children = [serialized_branch(entries=8212, written=1, seeks=(80, 0), offsets=(0, 8212),
                                  sizes=(40, 0), derived=derived) for _ in range(6)]
    classname, payload = serialized_branch(children=children, entries=8212, tags=tags, derived=derived)
    monkeypatch.setattr(uproot.models.TObjArray.Model_TObjArrayOfTBaskets, "read_members", forbidden)
    monkeypatch.setattr(uproot.models.TBasket.Model_TBasket, "read_members", forbidden)
    with A.metadata_limits():
        branch = read_serialized_branch(classname, payload)
        assert branch.num_entries == 8212 and len(branch.branches) == 6
        assert branch.embedded_baskets == [] and branch.num_baskets == 0
        assert branch.member("fBasketEntry").tolist() == [0, 0]
        assert branch.member("fBaskets") == []
        assert all(child.embedded_baskets == [] for child in branch.branches)


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("bad", ["nonnull", "no-bytecount", "truncated", "nonminimal"])
def test_serialized_fBaskets_never_resolved_integration(derived, bad, monkeypatch):
    classname, payload = serialized_branch(entries=0, tags=(7,) if bad == "nonnull" else (),
                                            derived=derived, malformed=bad)
    original = uproot.deserialization.read_object_any
    def no_basket_resolution(chunk, cursor, context, *args, **kwargs):
        assert not any(getattr(x, "_cursor_baskets", None) is not None for x in context.get("breadcrumbs", ()))
        return original(chunk, cursor, context, *args, **kwargs)
    monkeypatch.setattr(uproot.deserialization, "read_object_any", no_basket_resolution)
    monkeypatch.setattr(uproot.models.TBasket.Model_TBasket, "read_members", forbidden)
    with A.metadata_limits(), pytest.raises(ValueError, match="basket|fBaskets"):
        read_serialized_branch(classname, payload, minimal=bad != "nonminimal")


def test_failed_skip_cannot_enter_full_TObjArray_read(monkeypatch):
    monkeypatch.setattr(uproot.source.cursor.Cursor, "skip_over", lambda *args: False)
    classname, payload = serialized_branch(entries=0)
    with A.metadata_limits(), pytest.raises(ValueError, match="skip fallback forbidden"):
        read_serialized_branch(classname, payload)


@pytest.mark.parametrize("context", [{}, {"read_basket": False}, {"basket_num": 0, "breadcrumbs": ()}])
def test_TBasket_metadata_tripwire_before_members(context, monkeypatch):
    chunk, cursor = resident_metadata(b"")
    context = dict(context)
    if "basket_num" in context:
        context["breadcrumbs"] = (synthetic_branch(),)
    monkeypatch.setattr(uproot.models.TBasket.Model_TBasket, "read_members", forbidden)
    with A.metadata_limits(), pytest.raises(ValueError, match="TBasket deserialization forbidden"):
        uproot.models.TBasket.Model_TBasket.read(chunk, cursor, context, None, None, None)


@pytest.mark.parametrize("failure", [False, True])
def test_metadata_hooks_restore_exactly_and_nest(failure):
    targets = [(uproot.model.Model, "hook_before_read"), (uproot.model.Model, "hook_before_postprocess"),
               (uproot.source.cursor.Cursor, "skip_over"),
               (uproot.reading.ReadOnlyDirectory, "hook_before_read_keys"),
               (uproot.reading.ReadOnlyDirectory, "hook_before_keys"),
               (uproot.reading.ReadOnlyKey, "get_uncompressed_chunk_cursor")]
    original = [getattr(cls, name) for cls, name in targets]
    try:
        with A.metadata_limits():
            outer = [getattr(cls, name) for cls, name in targets]
            with A.metadata_limits():
                assert all(getattr(cls, name) is not old for (cls, name), old in zip(targets, outer))
            assert all(getattr(cls, name) is old for (cls, name), old in zip(targets, outer))
            if failure:
                raise ValueError("synthetic failure")
    except ValueError as exc:
        assert failure and str(exc) == "synthetic failure"
    assert all(getattr(cls, name) is old for (cls, name), old in zip(targets, original))


@pytest.mark.parametrize("derived", [False, True])
@pytest.mark.parametrize("children", [False, True])
@pytest.mark.parametrize("tag", [0, 1, 7, 999, 0xffffffff, 0x80000007])
def test_positive_matching_coverage_never_authorizes_embedded_baskets(derived, children, tag, monkeypatch):
    child = serialized_branch(entries=0, derived=derived)
    classname, payload = serialized_branch(entries=4, written=1, seeks=(80, 0), offsets=(0, 4),
                                            sizes=(40, 0), tags=(tag,), derived=derived,
                                            children=[child] if children else [])
    monkeypatch.setattr(uproot.models.TBasket.Model_TBasket, "read_members", forbidden)
    with A.metadata_limits():
        if tag:
            with pytest.raises(ValueError, match="nonnull basket slot"):
                read_serialized_branch(classname, payload)
        else:
            branch = read_serialized_branch(classname, payload)
            assert branch.num_entries == 4 and branch.num_baskets == 1
            assert branch.member("fBasketEntry").tolist() == [0, 4]
            assert branch.member("fBaskets") == [] and branch.embedded_baskets == []


@pytest.mark.parametrize("changes", [dict(seeks=(0, 80)), dict(offsets=(1, 0)), dict(sizes=(0, 40)),
                                     dict(written=1)])
def test_serialized_structural_parent_adversarial_metadata(changes):
    child = serialized_branch(entries=0)
    classname, payload = serialized_branch(children=[child], **changes)
    with A.metadata_limits(), pytest.raises(ValueError, match="embedded baskets forbidden"):
        read_serialized_branch(classname, payload)


def test_serialized_positive_leaf_cannot_use_structural_exception():
    classname, payload = serialized_branch(entries=4)
    with A.metadata_limits(), pytest.raises(ValueError, match="embedded baskets forbidden"):
        read_serialized_branch(classname, payload)


@pytest.mark.parametrize("array", [np.zeros((1, 2), dtype=np.int64), np.zeros(2, dtype=float),
                                   np.zeros(3, dtype=np.int64)])
@pytest.mark.parametrize("member", ["fBasketSeek", "fBasketEntry", "fBasketBytes"])
def test_branch_basket_metadata_shape_dtype_and_size_fail_closed(member, array):
    branch = synthetic_branch()
    branch._members[member] = array
    with A.metadata_limits(), pytest.raises(ValueError, match="basket metadata bounds"):
        process_synthetic_branch(branch)


def test_postprocess_proof_cannot_be_replaced_by_a_predecoded_empty_member():
    branch = synthetic_branch(entries=0, children=0)
    branch._members["fBaskets"] = []
    chunk, cursor = resident_metadata(basket_envelope(tags=(7,)))
    branch._cursor_baskets = cursor
    with A.metadata_limits(), pytest.raises(ValueError, match="nonnull basket slot"):
        branch.hook_before_postprocess(chunk=chunk, cursor=cursor, context={}, file=branch._file)


def test_metadata_skip_guard_does_not_change_unrelated_skip():
    chunk, cursor = resident_metadata(basket_envelope())
    with A.metadata_limits():
        assert cursor.skip_over(chunk, {})
    assert cursor.index == chunk.stop


def test_missing_proof_cannot_admit_structural_parent():
    branch = synthetic_branch()
    with A.metadata_limits(), pytest.raises(ValueError, match="resident basket metadata required"):
        branch.hook_before_postprocess()
    assert not branch.has_member("fBaskets")


def test_unknown_TBranch_reader_is_rejected_before_unprotected_fBaskets():
    class Model_TBranch_v99(uproot.model.VersionedModel):
        read_members = forbidden
    chunk, cursor = resident_metadata(b"")
    with A.metadata_limits(), pytest.raises(ValueError, match="unsupported TBranch metadata reader"):
        Model_TBranch_v99.read(chunk, cursor, {}, None, None, None)


def test_lazy_TObjArrayOfTBaskets_tripwire_precedes_reference_resolution(monkeypatch):
    chunk, cursor = resident_metadata(basket_envelope(tags=(7,)))
    monkeypatch.setattr(uproot.deserialization, "read_object_any", forbidden)
    with A.metadata_limits(), pytest.raises(ValueError, match="lazy embedded basket deserialization forbidden"):
        uproot.models.TObjArray.Model_TObjArrayOfTBaskets.read(chunk, cursor, {}, None, None, None)


@pytest.mark.parametrize("base_version,element_version", [(10, None), (11, None), (12, None), (13, None),
                                                         (10, 8), (12, 9), (12, 10)])
@pytest.mark.parametrize("tag", [0, 7])
def test_supported_uproot_branch_readers_all_use_empty_proof(base_version, element_version, tag, monkeypatch):
    child = serialized_branch(entries=4, written=1, seeks=(80, 0), offsets=(0, 4), sizes=(40, 0),
                              base_version=base_version)
    classname, payload = serialized_branch(children=[child], tags=(tag,), base_version=base_version,
                                            derived=element_version is not None,
                                            element_version=element_version)
    monkeypatch.setattr(uproot.models.TBasket.Model_TBasket, "read_members", forbidden)
    with A.metadata_limits():
        if tag:
            with pytest.raises(ValueError, match="nonnull basket slot"):
                read_serialized_branch(classname, payload)
        else:
            branch = read_serialized_branch(classname, payload)
            assert branch.num_entries == 4 and branch.num_baskets == 0
            assert branch.embedded_baskets == [] and branch.branches[0].num_baskets == 1
