"""Audit closed no-transport manifest fixtures and record bounded completion evidence."""
from pathlib import Path
import argparse
import datetime
import json
import re
from types import SimpleNamespace

import numpy as np
import uproot

from runtime_checks import digest, fixture_validator


def outcome(mode, code, log):
    expected = {
        "single": (0, ""), "strict": (0, ""), "empty": (0, ""),
        "strict_empty": (1, "---- MANIFEST_MULTIPLICITY BEGIN"),
        "missing": (1, "---- MANIFEST_MISSING BEGIN"),
        "generator_error": (1, "---- BADINPUT BEGIN"),
    }
    status, marker = expected[mode]
    if code != status or (marker and marker not in log):
        raise ValueError("fixture did not reach its intended success or callback failure")
    return code == 0


def audit(directory, artifact, source_log):
    validator = fixture_validator()
    validator.check_snapshot()
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    mode = receipt["fixture_mode"]
    fcl = directory / "executed.expanded.fcl"
    log_path = directory / "manifest.log"
    root_path = directory / "incident_manifest.root"
    if receipt["transport_requested"] or not receipt["generator_only"] or not receipt["source_unchanged"]:
        raise ValueError("fixture transport or immutable-source contract failed")
    if digest(artifact) != receipt["artifact_sha256"] or digest(source_log) != receipt["independent_source_log_sha256"]:
        raise ValueError("fixture source bytes changed")
    if digest(fcl) != receipt["expanded_sha256"]:
        raise ValueError("executed expanded FCL changed")
    expected_path = directory / "expected_event_ids.json"
    expected = validator.expected_event_ids(expected_path.read_text())
    triples = re.findall(r"Begin processing the .*? record\. run: (\d+) subRun: (\d+) event: (\d+)", source_log.read_text())
    independent = np.asarray(triples[:3], dtype=np.uint32)
    if not np.array_equal(expected, independent) or not np.array_equal(expected, receipt["expected_event_ids"]):
        raise ValueError("independent source EventIDs disagree")
    log = log_path.read_text()
    complete = outcome(mode, receipt["results"]["manifest_job"], log)
    if complete and (not receipt["all_commands_exit_zero"] or not receipt.get("finished_utc")):
        raise ValueError("successful closed job receipt missing")
    if complete:
        validator.audit(SimpleNamespace(manifest=root_path, artifact=artifact, expanded_fcl=fcl,
                                        expected_events=expected_path,
                                        snapshot_manifest=validator.HERE / "source_manifest.json"))
    with uproot.open(root_path) as root:
        inputs = validator.read_columns(root["inputEventManifest/inputEvents"])
        events = validator.read_columns(root["incidentManifest/events"])
        products = validator.read_columns(root["incidentManifest/products"])
        roots = validator.read_columns(root["incidentManifest/roots"])
        for label in ("inputEventManifest", "incidentManifest"):
            metadata = validator.read_metadata(root[label + "/metadata"])
            validator.verify_claims(metadata, artifact, validator.HERE / "source_manifest.json", fcl)
        if not complete:
            n_incident = 0 if mode == "generator_error" else 1
            if len(inputs["ordinal"]) != 1 or len(events["ordinal"]) != n_incident:
                raise ValueError("failure-prefix phase counts differ from expected first callback")
            for columns in (inputs, events, products):
                n = len(columns["ordinal"])
                ids = np.column_stack([columns[field] for field in validator.ID_FIELDS])
                if not np.array_equal(columns["ordinal"], np.arange(n)) or not np.array_equal(ids, expected[:n]):
                    raise ValueError("failure-prefix EventIDs or ordinal differ")
            if len(products["ordinal"]) != n_incident or len(roots["ordinal"]) != 0:
                raise ValueError("failure-prefix product/root counts differ")
            if n_incident:
                present = mode == "strict_empty"
                if not np.all(events["present"] == present) or not np.all(products["present"] == present):
                    raise ValueError("missing versus empty distinction failed")
                if np.any(events["n_roots"] != 0) or np.any(products["size"] != 0):
                    raise ValueError("failure-prefix collection count differs")
                if not present and (np.any(products["product_id"] != 0) or np.any(products["branch_name"] != "")):
                    raise ValueError("missing product incorrectly has a registry identity")
        counts = {"input_rows": len(inputs["ordinal"]), "incident_rows": len(events["ordinal"]),
                  "product_rows": len(products["ordinal"]), "root_rows": len(roots["ordinal"])}
    return {"schema": 1, "audited_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "fixture_mode": mode, "expected_outcome_observed": True, **counts,
            "runtime_receipt_sha256": digest(receipt_path), "root_sha256": digest(root_path),
            "log_sha256": digest(log_path), "expanded_fcl_sha256": digest(fcl),
            "source_sha256": digest(artifact), "source_log_sha256": digest(source_log),
            "build_receipt_sha256": receipt["build_receipt_sha256"],
            "auditor_sha256": digest(Path(__file__)),
            "complete_structural_manifest": complete,
            "failure_prefix_only": not complete,
            "generator_state_independently_replayed": False,
            "transport_requested": False, "physics_validated": False, "library_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--source-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("audit output must be new")
    report = audit(args.fixture_dir, args.artifact, args.source_log)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
