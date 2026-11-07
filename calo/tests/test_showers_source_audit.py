"""Synthetic bounded-audit contracts; no SAM, remote, art job, or real-file claim."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calofastsim import artsteps
from calofastsim.showers import source_audit as audit
from calofastsim.showers.export import EVENT


def catalog():
    roles, metadata = {}, {}
    for offset, role in enumerate(("training", "development", "confirmation")):
        roles[role] = []
        for i in range(10):
            subrun = offset * 10 + i
            name = f"dts.mu2e.CeEndpoint.MDC2025ac.001201_{subrun:08d}.art"
            fid = str(123000 + subrun)
            frozen = dict(did="mu2e:" + name, fid=fid, size=60, checksums={"enstore": "123"})
            roles[role].append(frozen)
            metadata[name] = dict(file_name=name, file_id=int(fid), file_size=60, checksum=["enstore:123"],
                                  event_count=3, parents=[dict(file_name=audit.PARENT_NAME, file_id=int(audit.PARENT_FID)),
                                                          dict(file_name="cnf.mu2e.config.fcl", file_id=999)],
                                  locations=[dict(full_path=f"enstore:/data/mu2e/test/{name} (nearline)")],
                                  **{"dh.sha256": "a" * 64, "rs.first_run": 1201, "rs.last_run": 1201,
                                     "rs.first_subrun": subrun, "rs.last_subrun": subrun})
    metadata[audit.PARENT_NAME] = dict(file_name=audit.PARENT_NAME, file_id=int(audit.PARENT_FID))
    return dict(roles=roles), metadata


def config():
    return ('TargetStopResampler: { fileNames: ["xroot://server/data/test/' + audit.PARENT_NAME + '"] '
            'module_type: "ResamplingMixer" mu2e: { writeEventIDs: true otherSetting: 0.25 } '
            'readMode: "sequential" wrapFiles: true }\n')


def printer(events):
    lines = []
    for index in range(len(events["event"])):
        lines.append(f"Begin processing the {index+1}th record. run: {events['run'][index]} "
                     f"subRun: {events['subrun'][index]} event: {events['event'][index]} at today")
        lines.append("EventIDSequence InputTag: label = 'TargetStopResampler', instance = '', process = 'Primary' : "
                     f"( run: 1430 subRun: 0 event: {9 + index % 2} )")
    lines.extend([f"TrigReport Events total = {len(events['event'])} passed = {len(events['event'])} failed = 0",
                  audit.source_event_check.COMPLETION])
    return "\n".join(lines) + "\n"


def mock_sam(monkeypatch, metadata):
    calls = []
    def run(argv, **kwargs):
        assert argv[:6] == ["samweb", "--socket-timeout", "30", "get-metadata", "--json", "--locations"]
        assert kwargs["timeout"] == 120 and kwargs["check"] is True
        assert "shell" not in kwargs
        name = argv[-1]
        calls.append(name)
        return SimpleNamespace(stdout=json.dumps(metadata[name]).encode())
    monkeypatch.setattr(audit.subprocess, "run", run)
    return calls


def inventory(tmp_path, monkeypatch):
    manifest, metadata = catalog()
    calls = mock_sam(monkeypatch, metadata)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    out = tmp_path / "metadata"
    result = audit.metadata_inventory(path, out)
    return result, out, calls


def bind_inputs(tmp_path, monkeypatch):
    inv, out, _ = inventory(tmp_path, monkeypatch)
    events = dict(run=np.array([1201] * 3), subrun=np.zeros(3, dtype=np.int64), event=np.array([33, 11, 22]))
    census, log, cfg = (tmp_path / name for name in ("census.npz", "printer.log", "config.fcl"))
    np.savez(census, **events)
    log.write_text(printer(events))
    cfg.write_text(config())
    return [out / "inventory.json", inv["rows"][0]["name"], census, log, cfg]


def test_metadata_complete_and_no_confirmation_or_event_access(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.uproot, "open", lambda *a, **k: pytest.fail("metadata opened art"))
    result, out, calls = inventory(tmp_path, monkeypatch)
    assert len(calls) == 21 and calls[0] == audit.PARENT_NAME
    assert all(name not in calls for name in [x["did"].split(":", 1)[1] for x in catalog()[0]["roles"]["confirmation"]])
    assert len(result["rows"]) == 20
    assert [row["ordinal"] for row in result["rows"]] == list(range(1, 11)) * 2
    assert [row["role"] for row in result["rows"]] == ["training"] * 10 + ["development"] * 10
    row = result["rows"][0]
    assert row["path"] == "/data/mu2e/test/" + row["name"]
    assert row["metadata_sha256"] == audit.file_digest(out / row["metadata_filename"])
    assert row["event_count"] == 3 and row["sha256"] == "a" * 64
    assert not result["source_independence_certified"]
    assert (out / "inventory.json").exists()


@pytest.mark.parametrize("defect", ["missing-parent", "extra-sim", "wrong-parent-fid", "wrong-parent-name",
    "duplicate-parent", "unknown-parent", "fid", "name", "did", "size", "checksum", "count-zero", "count-float",
    "sha", "run", "path-missing", "path-ambiguous", "path-name", "no-sha"])
def test_metadata_fail_closed(defect):
    manifest, metadata = catalog()
    frozen = audit.manifest_rows(manifest)[0]
    md = metadata[frozen["name"]]
    if defect == "missing-parent": md["parents"] = []
    elif defect == "extra-sim": md["parents"].append(dict(file_name="sim.other.art", file_id=888))
    elif defect == "wrong-parent-fid": md["parents"][0]["file_id"] = 888
    elif defect == "wrong-parent-name": md["parents"][0]["file_name"] = "sim.copy.art"
    elif defect == "duplicate-parent": md["parents"].append(deepcopy(md["parents"][0]))
    elif defect == "unknown-parent": md["parents"][1]["file_name"] = "dts.parent.art"
    elif defect == "fid": md["file_id"] += 1
    elif defect == "name": md["file_name"] = "wrong.art"
    elif defect == "did": md["did"] = "wrong:" + frozen["name"]
    elif defect == "size": md["file_size"] += 1
    elif defect == "checksum": md["checksum"] = ["enstore:124"]
    elif defect == "count-zero": md["event_count"] = 0
    elif defect == "count-float": md["event_count"] = 3.0
    elif defect == "sha": md["dh.sha256"] = "bad"
    elif defect == "run": md["rs.first_run"] += 1
    elif defect == "path-missing": md["locations"] = []
    elif defect == "path-ambiguous": md["locations"].append(dict(full_path="enstore:/data/other/" + frozen["name"]))
    elif defect == "path-name": md["locations"][0]["full_path"] = "enstore:/data/wrong.art"
    elif defect == "no-sha": del md["dh.sha256"]
    with pytest.raises(ValueError):
        audit.validate_metadata(md, frozen)


def test_partial_metadata_never_publishes_inventory(tmp_path, monkeypatch):
    manifest, metadata = catalog()
    name = manifest["roles"]["training"][1]["did"].split(":", 1)[1]
    metadata[name]["event_count"] = 0
    calls = mock_sam(monkeypatch, metadata)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    out = tmp_path / "partial"
    with pytest.raises(ValueError):
        audit.metadata_inventory(path, out)
    assert len(calls) == 3 and not (out / "inventory.json").exists()
    assert (out / (name + ".metadata.json")).exists()
    with pytest.raises(FileExistsError):
        audit.metadata_inventory(path, out)


@pytest.mark.parametrize("defect", ["missing-role", "short-role", "overlap", "confirmation-as-training"])
def test_manifest_explicit_roles(defect):
    manifest, _ = catalog()
    if defect == "missing-role": del manifest["roles"]["confirmation"]
    elif defect == "short-role": manifest["roles"]["training"].pop()
    elif defect == "overlap": manifest["roles"]["development"][0] = manifest["roles"]["training"][0]
    else: manifest["roles"]["training"] = manifest["roles"]["confirmation"]
    with pytest.raises(ValueError): audit.manifest_rows(manifest)


class Root:
    def __init__(self, ids, count=None, branch_count=None):
        ids = np.asarray(ids, dtype=np.uint32)
        # Genuine synthetic unsplit EventAuxiliary payload, one vectorized basket.
        prefix = bytes(12) + bytes([0]) + artsteps._ID_HEADERS
        width = len(prefix) + 12
        data = np.empty((len(ids), width), dtype=np.uint8)
        data[:, :len(prefix)] = np.frombuffer(prefix, np.uint8)
        data[:, len(prefix):] = ids.astype(">u4").view(np.uint8).reshape(-1, 12)
        basket = SimpleNamespace(data=data.tobytes(), byte_offsets=np.arange(len(ids) + 1) * width)
        self.branch = SimpleNamespace(num_baskets=1, basket=lambda i: basket,
                                      num_entries=len(ids) if branch_count is None else branch_count)
        self.num_entries = len(ids) if count is None else count
        self.accessed = []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def keys(self, **kwargs):
        assert kwargs == dict(recursive=False, cycle=True)
        return ["Events;1", "Events;3", "MetaData;1", "Events;2"]
    def __getitem__(self, key):
        self.accessed.append(key)
        if key == "Events;3": return self
        if key == "EventAuxiliary": return self.branch
        pytest.fail("unexpected branch/tree read " + key)


def census_inputs(tmp_path, monkeypatch, ids=None, **kwargs):
    path = tmp_path / "input.art"
    path.write_bytes(b"synthetic file identity")
    root = Root([[1201, 0, 33], [1201, 0, 11], [1201, 0, 22]] if ids is None else ids, **kwargs)
    def opening(p, **kw):
        assert p == path and kw == {"timeout": 30}
        return root
    monkeypatch.setattr(audit.uproot, "open", opening)
    return path, root


def test_census_real_decoder_highest_cycle_and_only_auxiliary(tmp_path, monkeypatch, capsys):
    path, root = census_inputs(tmp_path, monkeypatch)
    sha = audit.file_digest(path)
    out = tmp_path / "events.npz"
    audit.main(["census", "--art", str(path), "--expected-count", "3", "--sha256", sha, "--out", str(out)])
    assert root.accessed == ["Events;3", "EventAuxiliary"]
    with np.load(out, allow_pickle=False) as data:
        assert set(data.files) == set(EVENT)
        np.testing.assert_array_equal(data["event"], [33, 11, 22])
        assert all(data[key].dtype.kind in "iu" for key in EVENT)
    report = json.loads(capsys.readouterr().out)
    assert report["input_sha256"] == sha and report["events"] == 3
    with pytest.raises(FileExistsError):
        audit.main(["census", "--art", str(path), "--expected-count", "3", "--sha256", sha, "--out", str(out)])


@pytest.mark.parametrize("defect", ["duplicate", "count", "branch-count", "empty", "sha", "changed"])
def test_census_rejects_incomplete_or_wrong_identity(tmp_path, monkeypatch, defect):
    kwargs = {"count": 2} if defect == "count" else {"branch_count": 2} if defect == "branch-count" else {}
    ids = [[1201, 0, 33]] * 3 if defect == "duplicate" else None
    path, _ = census_inputs(tmp_path, monkeypatch, ids, **kwargs)
    sha = audit.file_digest(path)
    if defect == "changed":
        hashes = iter([sha, "b" * 64])
        monkeypatch.setattr(audit, "file_digest", lambda p: next(hashes))
    with pytest.raises(ValueError):
        audit.read_event_census(path, 0 if defect == "empty" else 3, "a" * 64 if defect == "sha" else sha)


@pytest.mark.parametrize("text", [config().replace("true", "false", 1), config().replace('readMode: "sequential"', ''),
    config().replace("wrapFiles: true", ""), config().replace("writeEventIDs: true", ""),
    config().replace('"ResamplingMixer"', '"Other"'), config().replace('fileNames: [', 'fileNames: ["xroot://other/file",'),
    config() + config(), config() + "OtherModule: {}", config().replace("writeEventIDs: true", "writeEventIDs: true writeEventIDs: true"),
    config().replace(audit.PARENT_NAME, "sim.copy.art"), config().replace("xroot://", "file://"),
    config().replace("mu2e: {", "mu2e: { fileNames: []"), config().replace("wrapFiles: true", "wrapFiles: @local::flag")])
def test_config_fail_closed(text):
    with pytest.raises(ValueError): audit.extract_parent_config(text)


def test_config_explicit_flags():
    result = audit.extract_parent_config(config())
    assert result["parent_name"] == audit.PARENT_NAME and result["write_event_ids"] is True


def test_bind_complete_vectorized_route_and_npz_hash(tmp_path, monkeypatch, capsys):
    args = bind_inputs(tmp_path, monkeypatch)
    expected, ledger, report = audit.bind(*args)
    assert set(expected) == set(audit.source_split.OUTPUT_KEY)
    assert report["events"] == report["source_rows"] == 3
    assert report["input_sha256"]["expected_events"] == hashlib.sha256(args[2].read_bytes()).hexdigest()
    np.testing.assert_array_equal(ledger["event"], [33, 11, 22])
    np.testing.assert_array_equal(ledger["source_event"], [9, 10, 9])
    assert np.all(ledger["parent_fid"] == audit.PARENT_FID)
    assert np.all(ledger["source_file_resolved"])
    assert report["ledger"]["unique_parent_events"] == 2
    assert not report["source_independence_certified"]
    out = tmp_path / "bound"
    cli = ["bind", "--inventory", str(args[0]), "--item", args[1], "--expected-events", str(args[2]),
           "--printer-log", str(args[3]), "--config", str(args[4]), "--out", str(out)]
    audit.main(cli)
    assert json.loads(capsys.readouterr().out)["events"] == 3
    assert {p.name for p in out.iterdir()} == {"expected.npz", "ledger.npz", "summary.json"}
    with pytest.raises(FileExistsError): audit.main(cli)


@pytest.mark.parametrize("defect", ["confirmation", "metadata-hash", "parent-hash", "parent-fid", "count", "run", "subrun",
    "order", "duplicate", "float", "extra-column", "prefix-log", "multi-source", "config", "changed-input"])
def test_bind_rejects_before_writing(tmp_path, monkeypatch, defect):
    args = bind_inputs(tmp_path, monkeypatch)
    inv = json.loads(args[0].read_text())
    if defect == "confirmation": args[1] = catalog()[0]["roles"]["confirmation"][0]["did"].split(":", 1)[1]
    elif defect == "metadata-hash": (args[0].parent / inv["rows"][0]["metadata_filename"]).write_text("{}")
    elif defect == "parent-hash": (args[0].parent / inv["original_parent"]["metadata_filename"]).write_text("{}")
    elif defect == "parent-fid":
        inv["original_parent"]["fid"] = "888"
        args[0].write_text(json.dumps(inv))
    elif defect in ("count", "run", "subrun", "order", "duplicate", "float", "extra-column"):
        with np.load(args[2]) as archive: events = dict(archive)
        if defect == "count": events = {key: value[:2] for key, value in events.items()}
        elif defect in ("run", "subrun"): events[defect][0] += 1
        elif defect == "order": events = {key: value[::-1] for key, value in events.items()}
        elif defect == "duplicate": events["event"][1] = events["event"][0]
        elif defect == "float": events["event"] = events["event"].astype(float)
        else: events["extra"] = np.ones(3)
        np.savez(args[2], **events)
    elif defect == "prefix-log": args[3].write_text(args[3].read_text().split("TrigReport")[0])
    elif defect == "multi-source": args[3].write_text(args[3].read_text().replace("event: 9 )", "event: 9 ) ( run: 1430 subRun: 0 event: 10 )"))
    elif defect == "config": args[4].write_text(config().replace("wrapFiles: true", "wrapFiles: false"))
    elif defect == "changed-input":
        digest = audit.file_digest
        seen = {}
        def changing(path):
            seen[path] = seen.get(path, 0) + 1
            return "c" * 64 if path == args[3] and seen[path] > 1 else digest(path)
        monkeypatch.setattr(audit, "file_digest", changing)
    out = tmp_path / "not-created"
    with pytest.raises(ValueError):
        audit.main(["bind", "--inventory", str(args[0]), "--item", args[1], "--expected-events", str(args[2]),
                    "--printer-log", str(args[3]), "--config", str(args[4]), "--out", str(out)])
    assert not out.exists()


def test_dangling_symlink_nooverwrite(tmp_path):
    out = tmp_path / "exists"
    out.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError): audit._new(out)


def test_duplicate_json_rejected():
    with pytest.raises(ValueError): audit._json('{"event_count": 3, "event_count": 4}')


@pytest.mark.parametrize("suffix", ["", "(nearline)", " (123@tape)", "/"])
def test_sam_location_directory_form(suffix):
    manifest, metadata = catalog()
    row = audit.manifest_rows(manifest)[0]
    md = metadata[row["name"]]
    md["locations"][0]["full_path"] = "enstore:/data/mu2e/tape/phy-sim/dts/a1/fe" + suffix
    assert audit.validate_metadata(md, row)["path"] == "/data/mu2e/tape/phy-sim/dts/a1/fe/" + row["name"]


def test_actual_config_dumper_header_and_no_second_file():
    header = "=============================================\nProcessing file: /data/mu2e/test.art\n# Read SQLiteDB from file, total size: 80.0 KiB.\n\n"
    assert audit.extract_parent_config(header + config())["parent_name"] == audit.PARENT_NAME
    with pytest.raises(ValueError):
        audit.extract_parent_config(header + config() + header + config())


@pytest.mark.parametrize("field", [None, "input_sha256", "events", "expected_count", "ordered_full_event_census", "output_sha256"])
def test_census_receipt_binds_exact_art_and_npz(tmp_path, monkeypatch, field):
    args = bind_inputs(tmp_path, monkeypatch)
    receipt = dict(input_sha256="a" * 64, events=3, expected_count=3,
                   ordered_full_event_census=True, output_sha256=audit.file_digest(args[2]))
    if field:
        receipt[field] = False if isinstance(receipt[field], bool) else 4 if isinstance(receipt[field], int) else "b" * 64
    path = tmp_path / "census.json"
    path.write_text(json.dumps(receipt))
    if field:
        with pytest.raises(ValueError, match="census receipt"):
            audit.bind(*args, census_receipt=path)
    else:
        assert audit.bind(*args, census_receipt=path)[2]["census_receipt_checked"]


def test_inventory_listing_excludes_confirmation(tmp_path, monkeypatch, capsys):
    _, out, _ = inventory(tmp_path, monkeypatch)
    audit.main(["inventory", "--inventory", str(out / "inventory.json")])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 20 and all(len(line.split("\t")) == 5 for line in lines)
    assert "confirmation" not in "\n".join(lines)


def test_rebind_uses_only_saved_proofs_and_preserves_inputs(tmp_path, monkeypatch):
    inv, directory, _ = inventory(tmp_path, monkeypatch)
    evidence = tmp_path / "saved"
    out = tmp_path / "rebound"
    calls = []
    monkeypatch.setattr(audit, "main", lambda argv: calls.append(argv))
    audit.rebind(directory / "inventory.json", evidence, out, tmp_path / "file0.log", tmp_path / "file0.cfg")
    assert len(calls) == 20
    for row, argv in zip(inv["rows"], calls):
        assert argv[0] == "bind" and argv[argv.index("--item") + 1] == row["name"]
        assert "--census-receipt" in argv
        assert str(out / row["name"] / "bound") == argv[-1]
    with pytest.raises(FileExistsError):
        audit.rebind(directory / "inventory.json", evidence, out, tmp_path / "file0.log", tmp_path / "file0.cfg")
