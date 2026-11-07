"""Census authenticated training exports without constructing or sampling a library."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from . import coverage, export, library, production_check, source_audit, source_events, training_audit


def extraction_manifest(path, root, digest):
    """Verify portable relative artifact hashes against an external frozen digest."""
    path, root = Path(path), Path(root)
    blob = path.read_bytes()
    if source_audit._sha(blob) != source_audit._digest(digest):
        raise ValueError("extraction manifest hash mismatch")
    result = {}
    for line in blob.decode().splitlines():
        sha, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) in result:
            raise ValueError("unsafe or duplicate extraction path")
        if source_events.file_digest(root / relative) != source_audit._digest(sha):
            raise ValueError("extracted artifact hash mismatch: " + name)
        result[str(relative)] = sha
    if not result:
        raise ValueError("empty extraction manifest")
    return result


def _require_artifacts(mapping, name, validation):
    wanted = dict(validation["evidence_sha256"])
    wanted["export/calo_library_tables.root"] = validation["tables_sha256"]
    for relative, sha in wanted.items():
        if mapping.get(str(Path(name) / relative)) != sha:
            raise ValueError("artifact missing from authenticated extraction manifest")


def merge(parts):
    """Join file-local scalar/step arrays while retaining complete stored-event offsets."""
    if not parts:
        raise ValueError("no training parts")
    entries, steps, totals, counts = [], [], [], {}
    event_offset = entry_offset = 0
    for e, s, c, total in parts:
        entries.append(dict(e, event_row=e["event_row"] + event_offset))
        steps.append(dict(s, entry=s["entry"] + entry_offset))
        totals.append(total)
        for key, value in c.items():
            if type(value) not in (int, np.int64, np.int32) or value < 0:
                raise ValueError("census must contain nonnegative integer counts")
            counts[key] = counts.get(key, 0) + int(value)
        event_offset += c["events"]
        entry_offset += len(e["energy"])
    if any(set(e) != set(entries[0]) for e in entries) or any(set(s) != set(steps[0]) for s in steps):
        raise ValueError("per-file array schema mismatch")
    return ({key: np.concatenate([e[key] for e in entries]) for key in entries[0]},
            {key: np.concatenate([s[key] for s in steps]) for key in steps[0]}, counts, np.concatenate(totals))


def _checked_json(path, digest):
    blob = Path(path).read_bytes()
    if source_audit._sha(blob) != source_audit._digest(digest):
        raise ValueError("authenticated JSON hash mismatch")
    return source_audit._json(blob)


def run(args):
    source_audit._new(args.out)
    started = time.monotonic()
    rows = training_audit.training_rows(args.inventory, args.inventory_sha256)
    extracted = extraction_manifest(args.extraction_manifest, args.tables, args.extraction_sha256)
    parents = _checked_json(args.parent_summary, args.parent_sha256)
    if source_events.file_digest(args.library) != args.library_sha256:
        raise ValueError("frozen library hash mismatch")
    reference_sha = source_events.file_digest(args.reference_config)
    reference = args.reference_config.read_text()
    pieces, reports, receipts, reference_geometry = [], [], [], None
    for row in rows:
        directory = args.tables / row["name"]
        tables = directory / "export/calo_library_tables.root"
        original = args.original_tables / row["name"] / "export/calo_library_tables.root"
        validation = training_audit.validate_saved(tables, args.inventory, row, args.evidence, directory,
                                                   original_tables=original)
        _require_artifacts(extracted, row["name"], validation)
        bound_path = args.bound / row["name"] / "bound/summary.json"
        suffix = "/" + row["name"] + "/bound/summary.json"
        matching = [v for k, v in parents["input_sha256"].items() if k.endswith(suffix)]
        if len(matching) != 1:
            raise ValueError("missing unique authenticated parent binding")
        bound = _checked_json(bound_path, matching[0])
        if (bound["input_sha256"] != validation["source_proof"]["input_sha256"] or
                bound["art_sha256"] != row["sha256"]):
            raise ValueError("saved source proof differs from completed parent audit")
        training_audit.check_geometry_config(reference, (directory / "source_config.txt").read_text())
        e, s, geometry, census, total = export.read(tables, input_mode="primary")
        if census != validation["raw_reconciliation"]["census"]:
            raise ValueError("raw reconciliation census differs from current tables")
        import uproot
        with uproot.open(tables) as source:
            checked_geometry = production_check.geometry_check(source)
        if checked_geometry != validation["raw_reconciliation"]["geometry"]:
            raise ValueError("geometry check differs from saved reconciliation")
        if reference_geometry is None:
            reference_geometry = geometry
        elif set(geometry) != set(reference_geometry) or any(
                not np.array_equal(value, reference_geometry[key]) for key, value in geometry.items()):
            raise ValueError("training geometry tables differ between files")
        arrays, report = coverage.census(e, s, census, total)
        reports.append(dict(item=row["name"], fid=row["fid"], **report))
        receipts.append(validation)
        pieces.append((e, s, census, total))
        if source_events.file_digest(tables) != validation["tables_sha256"]:
            raise ValueError("tables changed during coverage")
    entries, steps, census, total = merge(pieces)
    training_arrays, training_report = coverage.census(entries, steps, census, total)
    frozen, _ = library.load(args.library)
    legacy_arrays, legacy_report = coverage.frozen(frozen)
    comparison = coverage.compare(training_arrays, legacy_arrays)
    if source_events.file_digest(args.library) != args.library_sha256:
        raise ValueError("frozen library changed during coverage")
    extraction_manifest(args.extraction_manifest, args.tables, args.extraction_sha256)
    training_audit.training_rows(args.inventory, args.inventory_sha256)
    _checked_json(args.parent_summary, args.parent_sha256)
    if source_events.file_digest(args.reference_config) != reference_sha:
        raise ValueError("reference geometry configuration changed")
    report = dict(training=training_report, legacy_frozen=legacy_report, comparison=comparison,
                  per_file=reports, validation=receipts, elapsed_seconds=time.monotonic() - started,
                  source_independence_certified=False, candidate_library_constructed=False,
                  limitations=["Selected stored production, not inclusive generated acceptance.",
                               "Training immediate parents disjoint from development and legacy pilot; not proton independence.",
                               "Development/pilot has 165 shared parents including 40 frozen donors; no rows pruned.",
                               "Simulation incident-state conditioning is a coverage diagnostic, not a measured-variable correction.",
                               "No development-query weighting, reconstruction closure or confirmation test performed."],
                  input_sha256=dict(inventory=args.inventory_sha256, extraction_manifest=args.extraction_sha256,
                                    parent_summary=args.parent_sha256, frozen_library=args.library_sha256,
                                    reference_config=reference_sha),
                  source_sha256={m.__name__: source_events.file_digest(m.__file__) for m in
                                 (coverage, export, library, production_check, source_audit, source_events, training_audit)})
    report["source_sha256"][__name__] = source_events.file_digest(__file__)
    source_audit._text(report)
    args.out.mkdir(parents=True)
    for name, arrays in (("training_cells.npz", training_arrays), ("legacy_cells.npz", legacy_arrays)):
        with (args.out / name).open("xb") as stream:
            np.savez(stream, **arrays)
    report["output_sha256"] = {name: source_events.file_digest(args.out / name)
                               for name in ("training_cells.npz", "legacy_cells.npz")}
    source_audit._save_json(args.out / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "tables", "original-tables", "extraction-manifest", "evidence", "bound",
                 "parent-summary", "reference-config", "library", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("inventory", "extraction", "parent", "library"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(source_audit._text({key: report[key] for key in ("training", "legacy_frozen", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
