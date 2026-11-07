"""Synthetic training-only regression gates; no SAM, real physics files, art, or remote jobs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import uproot

from calofastsim.showers import training_audit as training


audit = training.source_audit
TAG = "TargetStopResampler::Primary"
BRANCH = "art::EventIDs_TargetStopResampler__Primary."
GEOMETRY = '''GeometryService: {
 inputFile: "Offline/Mu2eG4/geom/geom_run1_a.txt"
 bFieldFile: "Offline/Mu2eG4/geom/bfgeom_no_tsu_ps_v01.txt"
 allowReplacement: true
 overrides: { offset: [0, 1, -2.5] nested: { enabled: false label: "brace } {" } }
}'''
RECEIPTS = (
    "input.before.sha256", "input.after.sha256", "source_check.json", "source_config.txt",
    "geometry_config.json", "status.json", "export/export.log", "export/executed.fcl",
    "export/geometry.resolved.txt", "export/runtime.txt", "export/export.time",
)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value))
    return sha(path.read_bytes())


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    # Invented records omit development metadata files so out-of-scope access fails immediately.
    def forbidden(*args, **kwargs):
        pytest.fail("inventory must not query SAM or open any art/ROOT file")
    monkeypatch.setattr(audit.subprocess, "run", forbidden)
    rows = []
    for role, offset in (("training", 1), ("development", 0)):
        for i in range(10):
            subrun = 100 * i + offset
            name = f"dts.mu2e.CeEndpoint.MDC2025ac.001201_{subrun:08d}.art"
            frozen = dict(did="mu2e:" + name, name=name, fid=str(100000 + subrun),
                          size=60, checksums={"enstore": "123"}, role=role,
                          ordinal=i + 1, run=1201, subrun=subrun)
            metadata = dict(file_name=name, file_id=int(frozen["fid"]), file_size=60,
                            checksum=["enstore:123"], event_count=2822 if i == 0 else 2827,
                            parents=[dict(file_name=audit.PARENT_NAME, file_id=int(audit.PARENT_FID)),
                                     dict(file_name="cnf.mu2e.CeEndpoint.MDC2025ba.0.tar", file_id=107657994)],
                            locations=[dict(full_path="enstore:/synthetic/not-opened/" + name)],
                            **{"dh.sha256": "a" * 64, "rs.first_run": 1201, "rs.last_run": 1201,
                               "rs.first_subrun": subrun, "rs.last_subrun": subrun})
            row = audit.validate_metadata(metadata, frozen)
            row.update(metadata_filename=name + ".metadata.json", metadata_sha256=sha(json.dumps(metadata).encode()))
            if role == "training":
                save_json(tmp_path / row["metadata_filename"], metadata)
            rows.append(row)
    data = dict(rows=rows, manifest_sha256="b" * 64)
    path = tmp_path / "inventory.json"
    save_json(path, data)
    return SimpleNamespace(path=path, data=data, rows=rows[:10], forbidden=forbidden)


def test_inventory_exact_training_allocation_and_raw_metadata_scope(inventory, monkeypatch, capsys):
    monkeypatch.setattr(training.uproot, "open", inventory.forbidden)
    original = audit._metadata_blob
    accessed = []
    def metadata_blob(base, row):
        accessed.append(row["name"])
        assert row["role"] == "training"
        return original(base, row)
    monkeypatch.setattr(audit, "_metadata_blob", metadata_blob)
    rows = training.training_rows(inventory.path, sha(inventory.path.read_bytes()))
    assert rows == inventory.rows
    assert len(rows) == 10 and sum(row["event_count"] for row in rows) == 28265
    assert [row["subrun"] for row in rows] == list(range(1, 902, 100))
    assert accessed == [row["name"] for row in rows]
    training.main(["inventory", "--inventory", str(inventory.path), "--sha256", sha(inventory.path.read_bytes())])
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t") for line in lines] == [
        [str(row[key]) for key in ("name", "path", "event_count", "sha256")] for row in rows]


@pytest.mark.parametrize("defect", ["hash", "hash-format", "nine", "eleven", "order", "subrun",
    "total", "confirmation", "duplicate", "metadata-hash", "metadata-count", "metadata-path",
    "metadata-fid", "metadata-sha", "config-parent", "extra-config-parent", "traversal"])
def test_inventory_fail_closed(inventory, defect):
    data, row = inventory.data, inventory.rows[0]
    metadata_path = inventory.path.parent / row["metadata_filename"]
    if defect == "nine": data["rows"].remove(row)
    elif defect == "eleven": data["rows"].append(deepcopy(row))
    elif defect == "order": data["rows"][0], data["rows"][1] = data["rows"][1], data["rows"][0]
    elif defect == "subrun": row["subrun"] = 2
    elif defect == "total": row["event_count"] += 1
    elif defect == "confirmation": row["role"] = "confirmation"
    elif defect == "duplicate": data["rows"][10]["fid"] = row["fid"]
    elif defect == "traversal": row["metadata_filename"] = "../outside.json"
    elif defect.startswith("metadata-") or defect in ("config-parent", "extra-config-parent"):
        metadata = json.loads(metadata_path.read_text())
        if defect == "metadata-count": metadata["event_count"] += 1
        elif defect == "metadata-path": metadata["locations"][0]["full_path"] = "enstore:/other/" + row["name"]
        elif defect == "metadata-fid": metadata["file_id"] += 1
        elif defect == "metadata-sha": metadata["dh.sha256"] = "c" * 64
        elif defect == "config-parent": metadata["parents"][1]["file_id"] += 1
        elif defect == "extra-config-parent": metadata["parents"].append(dict(file_name="cnf.other.tar", file_id=999))
        else: metadata["ignored"] = True
        digest = save_json(metadata_path, metadata)
        if defect != "metadata-hash": row["metadata_sha256"] = digest
    digest = save_json(inventory.path, data)
    if defect == "hash": digest = "0" * 64
    if defect == "hash-format": digest = "NOT-A-SHA256"
    with pytest.raises(ValueError):
        training.training_rows(inventory.path, digest)


def test_geometry_entire_table_not_only_filenames(tmp_path, capsys):
    text = '==== saved config dump ====\nservices: { Other: { x: 9 }\n' + GEOMETRY + '\n}\n'
    parsed = training.geometry_config(text)
    assert parsed["overrides"]["nested"] == {"enabled": False, "label": "brace } {"}
    assert parsed["overrides"]["offset"] == ["0", "1", "-2.5"]
    candidate = GEOMETRY.replace('allowReplacement: true', '\n allowReplacement : true\n')
    result = training.check_geometry_config(text, candidate)
    assert result["exact_geometry_service"] is True
    assert result["geometry_service"] == parsed
    reference, saved = tmp_path / "reference.txt", tmp_path / "candidate.txt"
    reference.write_text(text)
    saved.write_text(candidate)
    training.main(["config", "--reference", str(reference), "--candidate", str(saved)])
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("candidate", [
    "", GEOMETRY + GEOMETRY, GEOMETRY[:-1],
    GEOMETRY.replace("enabled: false", "enabled: false enabled: false"),
    GEOMETRY.replace("allowReplacement: true", "allowReplacement true"),
    GEOMETRY.replace("enabled: false", "enabled: @local::flag"),
    GEOMETRY.replace("[0, 1, -2.5]", "[0 1 -2.5]"),
    GEOMETRY.replace("allowReplacement: true", "allowReplacement: false"),
    GEOMETRY.replace("enabled: false", "enabled: true"),
    GEOMETRY.replace("[0, 1, -2.5]", "[0, 1, -2.6]"),
    GEOMETRY.replace("allowReplacement: true", "extraOverride: true allowReplacement: true"),
])
def test_geometry_rejects_ambiguity_malformed_and_nested_overrides(candidate):
    with pytest.raises(ValueError):
        training.check_geometry_config(GEOMETRY, candidate)


@pytest.mark.parametrize("filename", ["geom_run1_a.txt", "bfgeom_no_tsu_ps_v01.txt"])
def test_geometry_equal_but_unapproved_filename_fails(filename):
    text = GEOMETRY.replace(filename, "other.txt")
    with pytest.raises(ValueError, match="unexpected geometry filenames"):
        training.check_geometry_config(text, text)


def test_geometry_commented_service_is_not_configuration():
    with pytest.raises(ValueError):
        training.check_geometry_config("# " + GEOMETRY, "# " + GEOMETRY)


def write_root(path, tables):
    with uproot.recreate(path) as root:
        for name, columns in tables.items():
            root.mktree("dump/" + name, columns)


def test_status_diagnostic_preserves_failed_events_and_cli_stops(tmp_path, capsys):
    root = tmp_path / "status.root"
    write_root(root, {"events": {"g4_status": np.array([0, 3, 0, -1, 3], dtype=np.int32)}})
    before = sha(root.read_bytes())
    expected = dict(events=5, status_values=[-1, 0, 3], status_counts=[1, 2, 2], failed_transport_events=3)
    assert training.status_census(root) == expected
    with pytest.raises(SystemExit, match="nonzero StatusG4"):
        training.main(["status", "--tables", str(root)])
    assert json.loads(capsys.readouterr().out) == expected
    assert sha(root.read_bytes()) == before


def test_status_expected_count_rejects_complete_looking_prefix(tmp_path, capsys):
    root = tmp_path / "status.root"
    write_root(root, {"events": {"g4_status": np.zeros(3, np.uint32)}})
    with pytest.raises(SystemExit, match="stored count"):
        training.main(["status", "--tables", str(root), "--expected-count", "4"])
    assert json.loads(capsys.readouterr().out)["events"] == 3


def test_status_all_zero_success(tmp_path, capsys):
    root = tmp_path / "status.root"
    write_root(root, {"events": {"g4_status": np.zeros(3, np.uint32)}})
    training.main(["status", "--tables", str(root)])
    assert json.loads(capsys.readouterr().out) == dict(
        events=3, status_values=[0], status_counts=[3], failed_transport_events=0)


@pytest.mark.parametrize("status", [np.array([], dtype=int), np.zeros(3, float),
    np.zeros(3, bool), np.zeros((2, 2), int), np.array(0), np.array(["0"])])
def test_status_invalid_shape_type_or_empty_fails(tmp_path, monkeypatch, status):
    root = tmp_path / "synthetic.root"
    write_root(root, {"events": {"g4_status": np.zeros(1, int)}})
    # Mock only the column payload for shapes not naturally stored by this TTree.
    monkeypatch.setattr(training.export, "columns", lambda *args: {"g4_status": status})
    with pytest.raises(ValueError, match="StatusG4"):
        training.status_census(root)


@pytest.fixture
def saved(inventory, tmp_path):
    row = inventory.rows[0]
    row["event_count"] = 3
    metadata_path = tmp_path / row["metadata_filename"]
    metadata = json.loads(metadata_path.read_text())
    metadata["event_count"] = 3
    row["metadata_sha256"] = save_json(metadata_path, metadata)
    parent_metadata = dict(file_name=audit.PARENT_NAME, file_id=int(audit.PARENT_FID))
    parent_hash = save_json(tmp_path / "parent.metadata.json", parent_metadata)
    inventory.data.update(original_parent=dict(name=audit.PARENT_NAME, fid=audit.PARENT_FID,
        did="mu2e:" + audit.PARENT_NAME, metadata_filename="parent.metadata.json", metadata_sha256=parent_hash),
        parent_metadata_sha256=parent_hash)
    save_json(inventory.path, inventory.data)
    evidence = tmp_path / "evidence"
    proof = evidence / row["name"]
    proof.mkdir(parents=True)
    ids = dict(run=np.full(3, 1201), subrun=np.ones(3, dtype=np.int64), event=np.array([33, 11, 22]))
    source_ids = dict(source_run=np.full(3, 1430), source_subrun=np.zeros(3, dtype=np.int64),
                      source_event=np.array([9, 10, 9]))
    np.savez(proof / "expected_events.npz", **ids)
    save_json(proof / "census.json", dict(events=3, expected_count=3, ordered_full_event_census=True,
        input_sha256=row["sha256"], output_sha256=sha((proof / "expected_events.npz").read_bytes())))
    config = ('TargetStopResampler: { fileNames: ["xroot://synthetic.invalid/' + audit.PARENT_NAME + '"] '
              'module_type: "ResamplingMixer" mu2e: { writeEventIDs: true } '
              'readMode: "sequential" wrapFiles: true }')
    (proof / "source_config.txt").write_text(config)
    lines = []
    for i, event in enumerate(ids["event"]):
        lines += [f"Begin processing the {i + 1}th record. run: 1201 subRun: 1 event: {event} at today",
                  "EventIDSequence InputTag: label = 'TargetStopResampler', instance = '', process = 'Primary' : "
                  f"( run: 1430 subRun: 0 event: {source_ids['source_event'][i]} )"]
    terminal = audit.source_event_check.COMPLETION
    (proof / "printer.log").write_text("\n".join(lines + ["TrigReport Events total = 3 passed = 3 failed = 0", audit.source_event_check.COMPLETION]) + "\n")
    data = dict(events=dict(ids, source_present=np.ones(3, bool), source_product_id=np.full(3, 81),
        source_count=np.ones(3, dtype=np.int64), source_tag=np.array([TAG] * 3),
        source_branch=np.array([BRANCH] * 3), g4_status=np.zeros(3, dtype=np.int32)),
        source_events=dict(ids, **source_ids, product_id=np.full(3, 81), index=np.zeros(3, dtype=np.int64),
                           source_valid=np.ones(3, bool)),
        products=dict(ids, product_id=np.full(3, 81), kind=np.full(3, -6), size=np.ones(3, dtype=np.int64),
                      present=np.ones(3, bool), branch_name=np.array([BRANCH] * 3)))
    extraction = tmp_path / "extraction"
    (extraction / "export").mkdir(parents=True)
    root = extraction / "export" / "calo_library_tables.root"
    write_root(root, data)
    for name in RECEIPTS:
        (extraction / name).write_text("synthetic receipt\n")
    for name in ("input.before.sha256", "input.after.sha256"):
        (extraction / name).write_text(row["sha256"] + "  " + row["path"] + "\n")
    (extraction / "export/export.log").write_text("export diagnostic\n" + terminal + "\n")
    (extraction / "source_config.txt").write_text(GEOMETRY)
    save_json(extraction / "geometry_config.json", training.check_geometry_config(GEOMETRY, GEOMETRY))
    save_json(extraction / "status.json", training.status_census(root))
    save_json(extraction / "source_check.json", dict(art=row["path"], tables=str(root),
        geometry=dict(disks=2, crystals_per_disk=[674, 674]),
        reconciliation=dict(events=3, all_fields_exact=True)))
    return SimpleNamespace(root=root, row=row, data=data, proof=proof, extraction=extraction,
        args=(root, inventory.path, row, evidence, extraction))


def test_validate_saved_complete_real_synthetic_binding(saved):
    result = training.validate_saved(*saved.args)
    assert result["events"] == 3
    assert result["fid"] == saved.row["fid"]
    assert result["source_proof"]["census_receipt_checked"] is True
    assert result["source_proof"]["ordered_full_census_exact"] is True
    assert result["source_proof"]["parent_fid"] == audit.PARENT_FID
    assert result["source_identity"]["source_tag"] == TAG
    assert result["source_identity"]["source_branch"] == BRANCH
    assert result["source_identity"]["source_independence_certified"] is False
    assert result["tables_sha256"] == sha(saved.root.read_bytes())
    assert result["evidence_sha256"] == {name: sha((saved.extraction / name).read_bytes()) for name in RECEIPTS}


@pytest.mark.parametrize("field", training.export.EVENT + ("source_run", "source_subrun", "source_event", "source_count"))
def test_validate_saved_all_source_ledger_fields_bound(saved, field):
    table = "source_events" if field in ("source_run", "source_subrun", "source_event") else "events"
    saved.data[table][field] = saved.data[table][field].copy()
    saved.data[table][field][-1] += 1
    write_root(saved.root, saved.data)
    with pytest.raises(ValueError):
        training.validate_saved(*saved.args)


@pytest.mark.parametrize("defect", ["truncated", "reordered", "missing-product", "wrong-process",
    "census-hash", "printer-source", "printer-failed", "source-config"])
def test_validate_saved_complete_independent_source_proof_required(saved, defect):
    if defect in ("truncated", "reordered"):
        indices = [0, 1] if defect == "truncated" else [2, 1, 0]
        saved.data = {table: {key: value[indices] for key, value in columns.items()}
                      for table, columns in saved.data.items()}
    elif defect == "missing-product": saved.data["products"]["present"][-1] = False
    elif defect == "wrong-process": saved.data["events"]["source_tag"] = np.array(["TargetStopResampler::Other"] * 3)
    elif defect == "census-hash":
        path = saved.proof / "census.json"
        receipt = json.loads(path.read_text())
        receipt["input_sha256"] = "0" * 64
        save_json(path, receipt)
    elif defect in ("printer-source", "printer-failed"):
        path = saved.proof / "printer.log"
        before, after = ("event: 9 )", "event: 99 )") if defect == "printer-source" else ("status 0", "status 1")
        path.write_text(path.read_text().replace(before, after))
    else:
        path = saved.proof / "source_config.txt"
        path.write_text(path.read_text().replace("writeEventIDs: true", "writeEventIDs: false"))
    write_root(saved.root, saved.data)
    with pytest.raises(ValueError):
        training.validate_saved(*saved.args)


@pytest.mark.parametrize("log", ["", "Art has completed and will exit with status 1\n",
    "Art has completed and will exit with status 0.\nsubsequent failure\n",
    "Art has completed and will exit with status 00.\n"])
def test_validate_saved_requires_terminal_art_success(saved, log):
    (saved.extraction / "export/export.log").write_text(log)
    with pytest.raises(ValueError, match="terminal successful art"):
        training.validate_saved(*saved.args)


@pytest.mark.parametrize("name", ["input.before.sha256", "input.after.sha256"])
def test_validate_saved_both_input_hashes_required(saved, name):
    (saved.extraction / name).write_text("f" * 64 + "  input.art\n")
    with pytest.raises(ValueError, match="art hash"):
        training.validate_saved(*saved.args)


@pytest.mark.parametrize("report", [dict(events=2, all_fields_exact=True), dict(events=3, all_fields_exact=False),
    dict(events=3, all_fields_exact=1), dict(events=3, all_fields_exact="true")])
def test_validate_saved_css_exact_complete_reconciliation_required(saved, report):
    save_json(saved.extraction / "source_check.json", dict(reconciliation=report))
    with pytest.raises(ValueError, match="CSS reconciliation"):
        training.validate_saved(*saved.args)


def test_validate_saved_failed_transport_never_becomes_zero(saved):
    saved.data["events"]["g4_status"][-1] = 3
    write_root(saved.root, saved.data)
    with pytest.raises(ValueError, match="failed transport census"):
        training.validate_saved(*saved.args)


def test_validate_saved_detects_root_mutation_during_validation(saved, monkeypatch):
    original = training.status_census
    def mutate(path):
        result = original(path)
        with Path(path).open("ab") as stream:
            stream.write(b"synthetic concurrent mutation")
        return result
    monkeypatch.setattr(training, "status_census", mutate)
    with pytest.raises(ValueError, match="tables changed"):
        training.validate_saved(*saved.args)


@pytest.mark.parametrize("name", RECEIPTS)
def test_validate_saved_missing_receipt_never_certifies(saved, name):
    (saved.extraction / name).unlink()
    with pytest.raises(FileNotFoundError):
        training.validate_saved(*saved.args)


def test_validate_saved_caller_row_must_match_bound_inventory(saved):
    saved.row["fid"] = "999999"
    with pytest.raises(ValueError):
        training.validate_saved(*saved.args)


def test_validate_saved_failed_geometry_receipt_rejected(saved):
    save_json(saved.extraction / "geometry_config.json", dict(exact_geometry_service=False))
    with pytest.raises(ValueError):
        training.validate_saved(*saved.args)


def test_training_runner_static_scope_and_fail_fast():
    # SOURCE-STATIC ONLY: this test never starts bash, mu2e, config_dumper, or a build.
    script = (Path(__file__).resolve().parents[1] / "art/export_training_coverage.sh").read_text()
    assert "set -eo pipefail" in script
    assert "CALO_EVENTS=-1" in script and "CALO_EXPORT_SOURCE_EVENTS=1" in script
    assert "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1" in script
    assert "training_audit inventory" in script and 'done < "$CALO_COVERAGE_OUT/inputs.tsv"' in script
    assert script.index("training_audit inventory") < script.index("config_dumper")
    assert script.index("training_audit config") < script.index('bash "$CALO_AREA/export_primary.sh"')
    assert script.index("training_audit status") < script.index("production_check --art")
    assert '--check-geometry' in script
    assert not any(command in script for command in ("muse build", "cmake", "ninja", "ssh ", "samweb "))


def test_validate_saved_css_receipt_from_other_input_is_not_reusable(saved):
    save_json(saved.extraction / "source_check.json", dict(
        art="/synthetic/unrelated.art", tables="/synthetic/unrelated.root",
        geometry="not checked here", reconciliation=dict(events=3, all_fields_exact=True)))
    with pytest.raises(ValueError):
        training.validate_saved(*saved.args)


def test_validate_saved_detects_css_receipt_mutation_during_validation(saved, monkeypatch):
    original = training.status_census
    def mutate(path):
        result = original(path)
        save_json(saved.extraction / "source_check.json", dict(
            reconciliation=dict(events=3, all_fields_exact=False)))
        return result
    monkeypatch.setattr(training, "status_census", mutate)
    with pytest.raises(ValueError):
        training.validate_saved(*saved.args)


def test_validate_saved_original_tables_accepts_saved_remote_path(saved):
    remote = "/synthetic/remote/export/calo_library_tables.root"
    path = saved.extraction / "source_check.json"
    receipt = json.loads(path.read_text())
    receipt["tables"] = remote
    save_json(path, receipt)
    with pytest.raises(ValueError, match="CSS reconciliation input identity"):
        training.validate_saved(*saved.args)
    result = training.validate_saved(*saved.args, original_tables=remote)
    assert result["raw_reconciliation"]["tables"] == remote
    assert result["tables_sha256"] == sha(saved.root.read_bytes())
    assert result["events"] == 3
    with pytest.raises(ValueError, match="CSS reconciliation input identity"):
        training.validate_saved(*saved.args, original_tables=remote + ".other")


@pytest.mark.parametrize("field,value", [
    ("events", 2), ("status_values", [1]), ("status_counts", [2]),
    ("failed_transport_events", 1),
    ("status_values", [0, 3]), ("status_counts", [2, 1]),
])
def test_validate_saved_wrong_status_receipt_rejected(saved, field, value):
    path = saved.extraction / "status.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    save_json(path, receipt)
    with pytest.raises(ValueError, match="saved status differs from complete census"):
        training.validate_saved(*saved.args)


def test_validate_saved_detects_status_receipt_mutation_during_validation(saved, monkeypatch):
    original = training.status_census
    def mutate(path):
        result = original(path)
        receipt = dict(result, failed_transport_events=1)
        save_json(saved.extraction / "status.json", receipt)
        return result
    monkeypatch.setattr(training, "status_census", mutate)
    with pytest.raises(ValueError, match="receipt changed during validation"):
        training.validate_saved(*saved.args)
