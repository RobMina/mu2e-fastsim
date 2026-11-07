"""Bind a retained pilot's source IDs through an audited original stop-pool copy."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re

import numpy as np
import uproot

from . import source_audit as audit, source_event_check as check, source_split as split
from .export import EVENT


def _npz(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        return dict(archive)


def _receipt(receipt, ids_blob, expected_sha, count):
    required = dict(input_sha256=expected_sha, events=count, expected_count=count,
                    ordered_full_event_census=True, output_sha256=audit._sha(ids_blob))
    if any(receipt.get(k) != v or type(receipt.get(k)) is not type(v) for k, v in required.items()):
        raise ValueError("pilot/copy census receipt does not bind input bytes and full census")


def _body(text):
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#")).strip()


def validate_evidence(blobs, proof):
    """Validate exact evidence bytes and actual ordered IDs, not sequential-row assumptions."""
    count = audit._positive(proof["events"], "pilot events")
    if count != 2000:
        raise ValueError("bounded pilot exposure must cover all 2000 events")
    parent = audit._json(blobs["parent_metadata"])
    audit.parent_identity(parent)
    if (parent.get("file_size") != proof["parent_size"] or
            audit._checksums(parent.get("checksum")) != proof["parent_checksums"]):
        raise ValueError("original parent metadata differs from frozen identity")
    ids = check._census(_npz(blobs["pilot_events"]))
    copy_ids = check._census(_npz(blobs["copy_events"]))
    for name, events, sha in (("pilot", ids, proof["pilot_sha256"]), ("copy", copy_ids, proof["copy_sha256"])):
        if len(events["event"]) != count:
            raise ValueError("incomplete pilot/copy census")
        audit._digest(sha)
        _receipt(audit._json(blobs[name + "_census"]), blobs[name + "_events"], sha, count)
    printed, printer = check.parse_printer(blobs["printer"].decode(), ids,
                                          source_tag="TargetStopResampler::CaloLibraryCE")
    if not np.array_equal(split._keys(copy_ids, EVENT), split._keys(
            {key: printed["source_" + key] for key in EVENT}, EVENT)):
        raise ValueError("actual pilot source IDs differ from complete copied-pool IDs")
    if blobs["copy_source"].decode().strip() != proof["original_source_uri"]:
        raise ValueError("copy source URI differs from frozen original parent")
    if Path(proof["original_source_uri"]).name != audit.PARENT_NAME:
        raise ValueError("copy source does not name canonical original parent")
    copy_config, _ = audit._config_table(_body(blobs["copy_config"].decode()))
    expected_copy = dict(process_name="CaloLibraryPool", source=dict(module_type="RootInput"),
                         outputs=dict(pool=dict(module_type="RootOutput", fileName="stop_pool.art",
                                                outputCommands=["keep *_*_*_*"])),
                         physics=dict(e1=["pool"], end_paths=["e1"]))
    if copy_config != expected_copy:
        raise ValueError("copy configuration is not the unfiltered ID-preserving RootOutput")
    log = blobs["copy_log"].decode()
    opened = re.findall(r'Opened input file "([^"]+)"', log)
    if opened != [proof["original_source_uri"]]:
        raise ValueError("copy log does not identify exactly one original input")
    if (log.count("TrigReport Events total = 2000 passed = 2000 failed = 0") != 1 or
            not log.rstrip().endswith(check.COMPLETION)):
        raise ValueError("copy did not complete all records successfully")
    if blobs["copy_manifest"].decode().split() != [proof["copy_sha256"], "stop_pool.art"]:
        raise ValueError("copied file hash receipt mismatch")
    input_lines = re.findall(r"(?m)^([0-9a-f]{64})[ \t]+([^\r\n]+)$", blobs["pilot_inputs"].decode())
    if len(input_lines) != 4 or len({path for _, path in input_lines}) != 4:
        raise ValueError("pilot input receipt is malformed")
    inputs = {path: sha for sha, path in input_lines}
    if (inputs.get(proof["pilot_pool_path"]) != proof["copy_sha256"] or
            inputs.get(proof["approved_config_path"]) != audit._sha(blobs["approved_config"])):
        raise ValueError("pilot inputs do not bind copied pool and approved configuration")
    approved, executed = blobs["approved_config"].decode(), blobs["executed_config"].decode()
    if _body(approved) != _body(executed):
        raise ValueError("executed pilot configuration differs from approved configuration")
    provenance = re.findall(r"(?m)^([0-9a-f]{64})[ \t]+([^\r\n]+)$", blobs["pilot_provenance"].decode())
    if dict((path, sha) for sha, path in provenance).get("executed.expanded.fcl") != audit._sha(blobs["executed_config"]):
        raise ValueError("pilot executed-config receipt mismatch")
    config_text = blobs["pilot_config"].decode()
    root_start = config_text.find("\nTargetStopResampler:")
    if root_start < 0 or config_text[:root_start].count("Processing file:") != 1:
        raise ValueError("unsupported pilot module config dump")
    root, counts = audit._config_table(config_text[root_start:])
    if set(root) != {"TargetStopResampler"} or any(counts.get(k) != 1 for k in (
            "TargetStopResampler", "fileNames", "module_type", "writeEventIDs", "readMode", "wrapFiles", "MaxEventsToSkip")):
        raise ValueError("ambiguous pilot resampler config")
    mod = root["TargetStopResampler"]
    if (mod.get("fileNames") != [proof["pilot_pool_path"]] or mod.get("module_type") != "ResamplingMixer" or
            mod.get("readMode") != "sequential" or mod.get("wrapFiles") is not True or
            mod.get("mu2e", {}).get("writeEventIDs") is not True or mod["mu2e"].get("MaxEventsToSkip") != "0"):
        raise ValueError("pilot mixer does not bind verified copied input")
    expected = dict(ids, output_fid=np.full(count, "sha256-" + proof["pilot_sha256"]))
    ledger = dict(expected, parent_fid=np.full(count, audit.PARENT_FID), **printed,
                  source_present=np.ones(count, bool), source_file_resolved=np.ones(count, bool),
                  source_count=np.ones(count, np.int64))
    _, census = split.validate_ledger(ledger, expected)
    return expected, ledger, dict(events=count, source_rows=count, role="pilot", item="legacy-unfiltered-ce-pilot",
        parent_fid=audit.PARENT_FID, ordered_full_census_exact=True, art_sha256=proof["pilot_sha256"],
        source_file_resolution="verified_original_copy", pilot_sha256=proof["pilot_sha256"],
        copy_sha256=proof["copy_sha256"], output_fid="sha256-" + proof["pilot_sha256"],
        output_identity_kind="retained artifact SHA256, not a SAM catalog FID", ledger=census,
        actual_source_sequence_equals_full_copy=True, printer=printer, source_independence_certified=False,
        scope="all exposed pilot events; original immediate stop-event identity, not muon/proton independence")


def validate(proof_path, base, old_tables, library):
    proof_blob = Path(proof_path).read_bytes()
    proof = audit._json(proof_blob)
    paths, blobs, hashes = {}, {}, {}
    for name, record in proof["evidence"].items():
        path = Path(base) / record["path"]
        blob = path.read_bytes()
        sha = audit._sha(blob)
        if sha != audit._digest(record["sha256"]):
            raise ValueError(f"frozen evidence hash mismatch: {name}")
        paths[name], blobs[name], hashes[name] = path, blob, sha
    expected, ledger, report = validate_evidence(blobs, proof)
    if audit.file_digest(old_tables) != proof["old_tables_sha256"] or audit.file_digest(library) != proof["library_sha256"]:
        raise ValueError("old pilot tables or frozen library hash mismatch")
    with uproot.open(old_tables) as root:
        old_ids = check._census(root["dump/events"].arrays(list(EVENT), library="np"))
    if not np.array_equal(split._keys(old_ids, EVENT), split._keys(expected, EVENT)):
        raise ValueError("old pilot table event rows differ from independently decoded pilot")
    with np.load(library, allow_pickle=False) as archive:
        rows = archive["entry_event_row"]
    if (rows.dtype.kind not in "iu" or rows.shape != (proof["donors"],) or
            np.any(rows < 0) or np.any(rows >= proof["split_row"]) or len(np.unique(rows)) != len(rows)):
        raise ValueError("invalid frozen donor-to-pilot event rows")
    for name, path in paths.items():
        if audit.file_digest(path) != hashes[name]:
            raise ValueError("evidence changed during pilot binding")
    if (audit.file_digest(old_tables) != proof["old_tables_sha256"] or
            audit.file_digest(library) != proof["library_sha256"] or audit.file_digest(proof_path) != audit._sha(proof_blob)):
        raise ValueError("pilot inputs changed during binding")
    report.update(input_sha256=hashes, proof_sha256=audit._sha(proof_blob),
                  old_tables_sha256=proof["old_tables_sha256"], library_sha256=proof["library_sha256"],
                  donor_events=len(rows), source_sha256=audit.file_digest(__file__),
                  old_event_rows_exact=True, donor_scope="frozen library identity subset, not a new selection")
    return expected, ledger, rows, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proof", "base", "old-tables", "library", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    audit._new(args.out)
    expected, ledger, donors, report = validate(args.proof, args.base, args.old_tables, args.library)
    text = audit._text(report)
    args.out.mkdir(parents=True)
    for name, table in (("expected", expected), ("ledger", ledger),
                        ("donor_expected", {k: v[donors] for k, v in expected.items()}),
                        ("donor_ledger", {k: v[donors] for k, v in ledger.items()})):
        with (args.out / (name + ".npz")).open("xb") as stream:
            np.savez(stream, **table)
    report["output_sha256"] = {name + ".npz": audit.file_digest(args.out / (name + ".npz"))
                               for name in ("expected", "ledger", "donor_expected", "donor_ledger")}
    text = audit._text(report)
    (args.out / "summary.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
