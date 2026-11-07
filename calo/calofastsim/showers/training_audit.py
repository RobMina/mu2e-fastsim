"""Bounded training export gates using saved parent proofs, never library construction."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import uproot

from . import export, source_audit, source_events


def training_rows(inventory_path, digest):
    """Select only the authenticated ten-file training allocation."""
    path = Path(inventory_path)
    blob = path.read_bytes()
    if source_audit._sha(blob) != source_audit._digest(digest):
        raise ValueError("inventory hash mismatch")
    inventory = source_audit._json(blob)
    rows = [row for row in inventory["rows"] if row["role"] == "training"]
    if len(rows) != 10 or [r["subrun"] for r in rows] != list(range(1, 902, 100)):
        raise ValueError("requires exactly the fixed ten training sequencers")
    if sum(r["event_count"] for r in rows) != 28265:
        raise ValueError("fixed training event count mismatch")
    for row in rows:
        source_audit._inventory_item(inventory, row["name"])
        _, raw = source_audit._metadata_blob(path.parent, row)
        metadata = source_audit._json(raw)
        checked = source_audit.validate_metadata(metadata, row)
        if any(row.get(k) != v for k, v in checked.items()):
            raise ValueError("inventory differs from saved metadata")
        config = [p for p in metadata["parents"] if p["file_name"].startswith("cnf.")]
        if config != [{"file_name": "cnf.mu2e.CeEndpoint.MDC2025ba.0.tar", "file_id": 107657994}]:
            raise ValueError("simulation configuration parent differs")
    return rows


def geometry_config(text):
    """Parse the unique canonical GeometryService table, including all overrides."""
    text = re.sub(r'"(?:[^"\\]|\\.)*"|#[^\n]*', lambda m: "" if m.group().startswith("#") else m.group(), text)
    found = list(re.finditer(r"\bGeometryService\s*:\s*\{", text))
    if len(found) != 1:
        raise ValueError("missing or ambiguous GeometryService")
    start = found[0].start()
    tail = text[found[0].end() - 1:]
    tokens = re.finditer(r'"(?:[^"\\]|\\.)*"|[{}]', tail)
    depth, end = 0, None
    for token in tokens:
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                end = found[0].end() - 1 + token.end()
                break
    if end is None:
        raise ValueError("incomplete GeometryService table")
    parsed, _ = source_audit._config_table(text[start:end])
    return parsed["GeometryService"]


def check_geometry_config(reference, candidate):
    left, right = geometry_config(reference), geometry_config(candidate)
    if left != right:
        raise ValueError("saved GeometryService differs from verified production reference")
    if (right.get("inputFile") != "Offline/Mu2eG4/geom/geom_run1_a.txt" or
            right.get("bFieldFile") != "Offline/Mu2eG4/geom/bfgeom_no_tsu_ps_v01.txt"):
        raise ValueError("unexpected geometry filenames")
    return dict(exact_geometry_service=True, geometry_service=right,
                scope="saved service settings; original release include closure is required at runtime")


def status_census(tables):
    with uproot.open(tables) as source:
        status = export.columns(source, "events", ("g4_status",))["g4_status"]
    if status.ndim != 1 or status.dtype.kind not in "iu" or not len(status):
        raise ValueError("invalid or empty StatusG4 census")
    values, counts = np.unique(status, return_counts=True)
    return dict(events=len(status), status_values=values.tolist(), status_counts=counts.tolist(),
                failed_transport_events=int(np.count_nonzero(status != 0)))


def validate_saved(tables, inventory, row, evidence, extraction, *, original_tables=None):
    """Bind a complete numeric export to previously audited, independently saved source IDs."""
    tables, evidence, extraction = Path(tables), Path(evidence), Path(extraction)
    inventory_blob = Path(inventory).read_bytes()
    canonical = source_audit._inventory_item(source_audit._json(inventory_blob), row["name"])
    if row != canonical or row["role"] != "training":
        raise ValueError("caller row differs from bound training inventory")
    receipts = ("input.before.sha256", "input.after.sha256", "source_check.json", "source_config.txt",
                "geometry_config.json", "status.json", "export/export.log", "export/executed.fcl",
                "export/geometry.resolved.txt", "export/runtime.txt", "export/export.time")
    blobs = {name: (extraction / name).read_bytes() for name in receipts}
    hashes = {name: source_audit._sha(blob) for name, blob in blobs.items()}
    proof = evidence / row["name"]
    expected, ledger, binding = source_audit.bind(
        inventory, row["name"], proof / "expected_events.npz", proof / "printer.log",
        proof / "source_config.txt", census_receipt=proof / "census.json")
    before = source_events.file_digest(tables)
    numeric, identity = source_events.read(tables, expected, source_tag="TargetStopResampler::Primary",
                                         source_branch="art::EventIDs_TargetStopResampler__Primary.")
    for field in export.EVENT + ("source_run", "source_subrun", "source_event", "source_count"):
        if not np.array_equal(numeric[field], ledger[field]):
            raise ValueError(f"numeric source ledger mismatch: {field}")
    log = blobs["export/export.log"].decode()
    if re.search(r"Art has completed and will exit with status 0\.\s*\Z", log) is None:
        raise ValueError("missing terminal successful art completion")
    for name in ("input.before.sha256", "input.after.sha256"):
        if blobs[name].decode().split() != [row["sha256"], row["path"]]:
            raise ValueError("training art hash differs from audited identity")
    raw = source_audit._json(blobs["source_check.json"])
    if (raw["reconciliation"]["events"] != row["event_count"] or
            raw["reconciliation"]["all_fields_exact"] is not True):
        raise ValueError("missing complete raw CSS reconciliation")
    if (raw.get("art") != row["path"] or raw.get("tables") != str(original_tables or tables) or
            not isinstance(raw.get("geometry"), dict) or raw["geometry"].get("disks") != 2 or
            raw["geometry"].get("crystals_per_disk") != [674, 674]):
        raise ValueError("raw CSS reconciliation input identity or geometry mismatch")
    geom = source_audit._json(blobs["geometry_config.json"])
    if (geom.get("exact_geometry_service") is not True or
            geom.get("geometry_service") != geometry_config(blobs["source_config.txt"].decode())):
        raise ValueError("geometry receipt differs from saved configuration")
    status = status_census(tables)
    if status["events"] != row["event_count"] or status["failed_transport_events"]:
        raise ValueError("incomplete or failed transport census; never convert failures to zeros")
    if source_events.file_digest(tables) != before:
        raise ValueError("numeric tables changed during validation")
    if source_audit._json(blobs["status.json"]) != status:
        raise ValueError("saved status differs from complete census")
    if Path(inventory).read_bytes() != inventory_blob or any(
            source_events.file_digest(extraction / name) != digest for name, digest in hashes.items()):
        raise ValueError("inventory or receipt changed during validation")
    return dict(item=row["name"], fid=row["fid"], events=row["event_count"], art_sha256=row["sha256"],
                tables_sha256=before, source_identity=identity, source_proof=binding, status=status,
                raw_reconciliation=raw, evidence_sha256=hashes,
                proof_prerequisite="raw check and extraction receipts require external authentication; no signature inferred")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("inventory")
    listing.add_argument("--inventory", type=Path, required=True)
    listing.add_argument("--sha256", required=True)
    config = commands.add_parser("config")
    config.add_argument("--reference", type=Path, required=True)
    config.add_argument("--candidate", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--tables", type=Path, required=True)
    status.add_argument("--expected-count", type=int)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        for row in training_rows(args.inventory, args.sha256):
            print("\t".join(str(row[k]) for k in ("name", "path", "event_count", "sha256")))
    elif args.command == "config":
        print(source_audit._text(check_geometry_config(args.reference.read_text(), args.candidate.read_text())), end="")
    else:
        report = status_census(args.tables)
        print(source_audit._text(report), end="")
        if args.expected_count is not None and report["events"] != args.expected_count:
            raise SystemExit("stored count differs from complete allocated file")
        if report["failed_transport_events"]:
            raise SystemExit("nonzero StatusG4: stop; preserve failures diagnostically, not as zeros")


if __name__ == "__main__":
    main()
