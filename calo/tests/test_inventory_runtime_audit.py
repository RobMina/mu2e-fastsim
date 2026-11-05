"""Fail-closed execution-evidence checks with synthetic files, never native execution."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

RECORDING = Path(__file__).resolve().parents[1] / "recording"
sys.path.insert(0, str(RECORDING))
spec = importlib.util.spec_from_file_location("inventory_runtime_audit", RECORDING / "audit_inventory.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def execution(tmp_path):
    directory = tmp_path / "fixture"
    directory.mkdir()
    build = tmp_path / "build.json"
    build.write_text("{}\n")
    geometry = tmp_path / "geometry.txt"
    geometry.write_text("geometry\n")
    field = tmp_path / "field.txt"
    field.write_text("field\n")
    template = audit.HERE / "zero_transport.fcl"
    (directory / "executed.fcl").write_text(template.read_text().replace("REVIEW_OUTPUT_PREFIX", str(directory / "native")))
    expanded = directory / "executed.expanded.fcl"
    expanded.write_text("synthetic expanded config\n")
    (directory / "geometry.log").write_text(
        f"Geometry file: {geometry}\nGeometry lines: 2  hash: 10\n"
        f"BField file: {field}\nBField lines: 3  hash: 20\n"
        "Mu2eG4RunAction BeginOfRunAction : G4Run: 1430\n"
        "Number of events processed : 0\nUser=0s Real=0s Sys=0s\n"
        "TrigReport Events total = 1 passed = 0 failed = 1\n"
        "MemReport  VmPeak = 1868.78 VmHWM = 1695.52\n"
        "Art has completed and will exit with status 0.\n")
    (directory / "native.inventory.run1430.pid12.worker-1.jsonl").write_text("not parsed by evidence-only helper\n")
    boundary_dir = directory / "native.pid12.worker-1"
    boundary_dir.mkdir()
    (boundary_dir / "boundary.jsonl").write_bytes(b"")
    receipt = dict(mode="geometry", transport_requested=False, all_commands_exit_zero=True,
                   finished_utc="2026-11-05T02:56:44+00:00", results=dict(expand=0, native_initialize=0),
                   build_receipt_sha256=audit.digest(build), template_sha256=audit.digest(template),
                   expanded_sha256=audit.digest(expanded))
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return directory, build


def test_execution_evidence_success_is_not_structural_acceptance(execution):
    inventory, evidence = audit.execution_evidence(*execution)
    assert inventory.name.endswith("pid12.worker-1.jsonl")
    assert evidence["g4_events"] == 0 and evidence["art_rejected"] == 1
    assert evidence["vm_hwm_decimal_mb"] == 1695.52
    assert evidence["geometry_provenance"]["geometry"]["native_expanded_hash"] == "10"
    assert "complete_structural_inventory" not in evidence


@pytest.mark.parametrize("key,value", [("transport_requested", True), ("all_commands_exit_zero", False),
                                       ("finished_utc", ""), ("results", {"expand": 0, "native_initialize": 1}),
                                       ("build_receipt_sha256", "0" * 64), ("mode", "manifest")])
def test_refuses_failed_or_wrong_job(execution, key, value):
    directory, build = execution
    path = directory / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt[key] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt required"):
        audit.execution_evidence(directory, build)


@pytest.mark.parametrize("old,new", [("processed : 0", "processed : 1"),
                                     ("total = 1 passed = 0 failed = 1", "total = 1 passed = 1 failed = 0"),
                                     ("exit with status 0.", "exit with status 1."),
                                     ("User=0s Real=0s Sys=0s", "User=1s Real=1s Sys=0s")])
def test_refuses_transport_or_unclosed_job(execution, old, new):
    directory, build = execution
    path = directory / "geometry.log"
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match="zero G4 processing"):
        audit.execution_evidence(directory, build)


@pytest.mark.parametrize("filename", ["executed.fcl", "executed.expanded.fcl"])
def test_refuses_configuration_drift(execution, filename):
    directory, build = execution
    with (directory / filename).open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="fixture differs|configuration changed"):
        audit.execution_evidence(directory, build)


def test_refuses_nonempty_boundary(execution):
    directory, build = execution
    (directory / "native.pid12.worker-1/boundary.jsonl").write_text("event\n")
    with pytest.raises(ValueError, match="ledger must be empty"):
        audit.execution_evidence(directory, build)


def test_refuses_duplicate_inventory(execution):
    directory, build = execution
    (directory / "native.inventory.run1430.pid13.worker-1.jsonl").write_text("duplicate\n")
    with pytest.raises(ValueError, match="exactly one regular inventory"):
        audit.execution_evidence(directory, build)


@pytest.mark.parametrize("key,value", [("candidate_roots", ["WrongRoot"]), ("max_rows", 999999), ("max_bytes", 99)])
def test_refuses_inventory_config_disagreement(execution, monkeypatch, key, value):
    directory, build = execution
    header = dict(g4_run=1430, max_rows=1000000, max_bytes=268435456,
                  candidate_roots=json.loads(audit.re.search(r"boundaryRecorderRoots\s*:\s*(\[[\s\S]*?\])",
                                                            (audit.HERE / "zero_transport.fcl").read_text())[1]))
    header[key] = value
    monkeypatch.setattr(audit, "check_environment", lambda *_: {})
    monkeypatch.setattr(audit.placement_inventory, "audit_file", lambda *_, **__: SimpleNamespace(accepted=True, header=header))
    with pytest.raises(ValueError, match="differ from the executed fixture"):
        audit.audit(directory, directory, build)
