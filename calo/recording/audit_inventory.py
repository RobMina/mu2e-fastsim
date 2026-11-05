"""Audit an existing closed zero-transport inventory; never launch art or transport."""
from pathlib import Path
import argparse
import datetime
import json
import re

from calofastsim import placement_inventory
from runtime_checks import HERE, check_environment, digest


def execution_evidence(directory, build_receipt):
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("mode") != "geometry" or receipt.get("transport_requested") is not False
            or receipt.get("all_commands_exit_zero") is not True or not receipt.get("finished_utc")
            or receipt.get("results") != {"expand": 0, "native_initialize": 0}
            or receipt.get("build_receipt_sha256") != digest(build_receipt)):
        raise ValueError("successful matching zero-transport job receipt required")
    template_path = HERE / "zero_transport.fcl"
    template = template_path.read_text()
    if receipt.get("template_sha256") != digest(template_path):
        raise ValueError("zero-transport template changed")
    expected = template.replace("REVIEW_OUTPUT_PREFIX", str(directory / "native"))
    if (directory / "executed.fcl").read_text() != expected:
        raise ValueError("executed fixture differs from the reviewed zero-transport template")
    expanded = directory / "executed.expanded.fcl"
    if digest(expanded) != receipt.get("expanded_sha256"):
        raise ValueError("expanded configuration changed")
    log_path = directory / "geometry.log"
    log = log_path.read_text()
    if (re.findall(r"Number of events processed\s*:\s*(\d+)", log) != ["0"]
            or re.findall(r"TrigReport Events total = (\d+) passed = (\d+) failed = (\d+)", log)
            != [("1", "0", "1")]
            or log.count("Art has completed and will exit with status 0.") != 1
            or "User=0s Real=0s Sys=0s" not in log):
        raise ValueError("closed job must report one rejected art event and zero G4 processing")
    runs = re.findall(r"Mu2eG4RunAction BeginOfRunAction : G4Run: (\d+)", log)
    if len(runs) != 1:
        raise ValueError("exactly one native run initialization required")
    files = list(directory.glob("native.inventory.*.jsonl"))
    if len(files) != 1 or files[0].is_symlink():
        raise ValueError("exactly one regular inventory required")
    inventory = files[0]
    match = re.fullmatch(r"native\.inventory\.run(\d+)\.pid(\d+)\.worker(-?\d+)\.jsonl", inventory.name)
    if match is None or match[1] != runs[0] or match[3] != "-1":
        raise ValueError("inventory filename/run/sequential-worker mismatch")
    boundary = directory / f"native.pid{match[2]}.worker{match[3]}" / "boundary.jsonl"
    if boundary.is_symlink() or boundary.stat().st_size != 0:
        raise ValueError("zero-transport boundary ledger must be empty")
    geometry = re.findall(r"^Geometry file: (.+)\nGeometry lines: (\d+)  hash: (\d+)$", log, re.M)
    field = re.findall(r"^BField file: (.+)\nBField lines: (\d+)  hash: (\d+)$", log, re.M)
    memory = re.findall(r"MemReport  VmPeak = ([\d.]+) VmHWM = ([\d.]+)", log)
    if len(geometry) != 1 or len(field) != 1 or len(memory) != 1:
        raise ValueError("native geometry/field/memory provenance missing")
    provenance = {}
    for label, values in (("geometry", geometry[0]), ("field", field[0])):
        path, lines, native_hash = values
        provenance[label] = {"path": path, "sha256": digest(Path(path)),
                             "native_expanded_lines": int(lines), "native_expanded_hash": native_hash}
    return inventory, {
        "runtime_receipt_sha256": digest(receipt_path), "build_receipt_sha256": digest(build_receipt),
        "expanded_fcl_sha256": digest(expanded), "executed_fcl_sha256": digest(directory / "executed.fcl"),
        "template_sha256": digest(template_path), "log_sha256": digest(log_path),
        "boundary_sha256": digest(boundary), "boundary_bytes": 0, "g4_run": int(runs[0]),
        "art_events": 1, "art_rejected": 1, "g4_events": 0,
        "vm_peak_decimal_mb": float(memory[0][0]), "vm_hwm_decimal_mb": float(memory[0][1]),
        "geometry_provenance": provenance,
    }


def audit(directory, overlay, build_receipt):
    build = check_environment(overlay, build_receipt)
    inventory, evidence = execution_evidence(directory, build_receipt)
    sha = digest(inventory)
    completion = placement_inventory.CompletionReceipt(sha, writer_closed=True, job_success=True)
    report = placement_inventory.audit_file(inventory, receipt=completion)
    if not report.accepted or report.header["g4_run"] != evidence["g4_run"]:
        raise ValueError("inventory structural acceptance/run check failed")
    template = (HERE / "zero_transport.fcl").read_text()
    root_list = re.search(r"boundaryRecorderRoots\s*:\s*(\[[\s\S]*?\])", template)
    if root_list is None or report.header["candidate_roots"] != json.loads(root_list[1]):
        raise ValueError("inventory roots differ from the executed fixture")
    for setting, key in (("boundaryRecorderMaxRows", "max_rows"), ("boundaryRecorderMaxBytes", "max_bytes")):
        value = re.search(setting + r"\s*:\s*(\d+)", template)
        if value is None or report.header[key] != int(value[1]):
            raise ValueError("inventory caps differ from the executed fixture")
    roots = []
    for name, node in zip(report.header["candidate_roots"], report.root_placements):
        roots.append({"name": name, "placement_id": node, "path": report.path(node),
                      "solid": report.solids[report.logical_volumes[report.placements.logical_volume_id[node]]["solid_id"]]})
    return {
        "schema": 1, "audited_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "inventory": str(inventory), "inventory_sha256": sha, "counts": report.counts,
        "completion_receipt": {"sha256": sha, "writer_closed": True, "job_success": True},
        "writer_close_evidence": "hash-pinned throwing fsync/checked-close implementation and successful native job; not inferred from footer",
        "execution": evidence, "source_manifest_sha256": build["source_manifest_sha256"],
        "overlay_receipt_sha256": build["overlay_receipt_sha256"],
        "binaries": {name: value["sha256"] for name, value in build["binaries"].items()},
        "reader_sha256": digest(Path(placement_inventory.__file__)), "auditor_sha256": digest(Path(__file__)),
        "header": report.header, "roots": roots,
        "complete_structural_inventory": True, "external_completion_verified": report.external_completion_verified,
        "native_configuration_initialized": True, "diagnostic_only": True,
        "native_geometry_certified": False, "containment_validated": False, "root_store_independently_verified": False,
        "material_composition_exported": False, "physics_validated": False, "library_eligible": False,
        "transport_requested": False,
        "provenance_limit": "geometry entry-file SHA and logged expanded native configuration hash/line count; no independent full include/material/navigation certificate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("audit output must be new")
    report = audit(args.fixture_dir, args.overlay, args.build_receipt)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
