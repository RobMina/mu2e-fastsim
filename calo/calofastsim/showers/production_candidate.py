"""Pack the one authenticated production-only training candidate; never draw donors.

The caller supplies the externally trusted coverage-report SHA256. Hashes inside
an unanchored bundle are not authentication. Only saved training numeric exports
and proofs are read; no raw art, development, pilot or confirmation physics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import (coverage, export, library, production_check, source_audit, source_events,
               source_overlap_report, source_split, training_audit, training_coverage)

PARENT_FID = "102778799"
RECEIPTS = ("input.before.sha256", "input.after.sha256", "source_check.json", "source_config.txt",
            "geometry_config.json", "status.json", "export/export.log", "export/executed.fcl",
            "export/geometry.resolved.txt", "export/runtime.txt", "export/export.time")
PROOF_KEYS = {"inventory", "metadata", "parent_metadata", "expected_events", "printer_log", "config", "census_receipt"}
ARTIFACTS = dict(library="library.npz", donor_provenance="donor_provenance.npz",
                 expected="training_expected.npz", ledger="training_ledger.npz",
                 cells="training_cells.npz", metadata="metadata.json", summary="summary.json")


class Inputs(source_overlap_report.Inputs):
    """Pin bytes before parsing, and retain all pins for before/after checks."""

    def pin(self, path, digest):
        digest = source_audit._digest(digest)
        key = str(Path(path).absolute())
        if source_events.file_digest(path) != digest:
            raise ValueError("authenticated input hash mismatch: " + Path(path).name)
        if key in self.sha256 and self.sha256[key] != digest:
            raise ValueError("conflicting input pin")
        self.sha256[key] = digest

    def checked_json(self, path, digest):
        self.pin(path, digest)
        return self.json(path)


def _binding_pin(parent, item, filename):
    suffix = "/" + item + "/bound/" + filename
    matches = [value for key, value in parent["input_sha256"].items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError("missing unique authenticated parent binding: " + filename)
    return source_audit._digest(matches[0])


def _geometry(geometry, reference=None):
    if not geometry or any(np.asarray(v).ndim != 1 or np.asarray(v).dtype.kind not in "iuf" or
                           not np.all(np.isfinite(v)) for v in geometry.values()):
        raise ValueError("invalid numeric geometry")
    for prefix in ("disk", "crystal", "vd"):
        lengths = {len(v) for k, v in geometry.items() if k.startswith(prefix + "_")}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError("incomplete geometry table: " + prefix)
    disk = np.asarray(geometry["disk_disk"])
    if not np.array_equal(disk, [0, 1]):
        raise ValueError("geometry requires both ordered disks")
    rotation = np.stack([np.column_stack([geometry["disk_axis_" + a + "_" + b] for b in "xyz"])
                         for a in "xyz"], axis=2)
    if (not np.allclose(np.einsum("nji,njk->nik", rotation, rotation), np.eye(3), rtol=0, atol=1e-12)
            or not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-12)):
        raise ValueError("geometry disk axes must be right-handed orthonormal")
    if reference is not None and (set(geometry) != set(reference) or any(
            not np.array_equal(value, reference[key]) for key, value in geometry.items())):
        raise ValueError("training geometry tables differ")


def validate_packing(packed, entries, steps, geometry):
    """Exact state/scalar/count parity plus inverse-frame position/time parity."""
    n = len(entries["energy"])
    order = np.argsort(steps["entry"], kind="stable")
    rows = steps["entry"][order]
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))]
    if not np.array_equal(packed["offsets"], offsets):
        raise ValueError("packed offsets differ from whole-event counts")
    for key, value in entries.items():
        if not np.array_equal(packed["entry_" + key], value):
            raise ValueError("packed entry differs: " + key)
    for key, value in steps.items():
        if key not in ("entry", "position", "time") and not np.array_equal(packed["step_" + key], value[order]):
            raise ValueError("packed scalar step differs: " + key)
    for key, value in geometry.items():
        if not np.array_equal(packed["geometry_" + key], value):
            raise ValueError("packed geometry differs: " + key)
    position = entries["position"][rows] + np.einsum(
        "nij,nj->ni", library.frame(entries["direction"])[rows], packed["step_relative_position"])
    time = entries["time"][rows] + packed["step_relative_time"]
    if (not np.allclose(position, steps["position"][order], rtol=0, atol=1e-9) or
            not np.allclose(time, steps["time"][order], rtol=0, atol=1e-9)):
        raise ValueError("packed relative position/time roundtrip differs")
    for key in ("energy_g4", "energy_birks"):
        original = np.bincount(steps["entry"], weights=steps[key], minlength=n)
        restored = np.bincount(rows, weights=packed["step_" + key], minlength=n)
        if not np.array_equal(original, restored):
            raise ValueError("packed whole-event energy differs")
    # Also independently validates the fixed grid and zero-offset donors.
    coverage.frozen(packed)


def pack(entries, steps, geometry, fullledger, expected, *, file_ordinal, file_event_row):
    """Pure all-entry packer returning (library arrays, donor provenance).

    File coordinates are aligned to the COMPLETE stored-event ledger. Donors are
    never selected using CSS outcomes; non-front events stay only in the ledger.
    """
    coverage._check(entries, steps)
    _geometry(geometry)
    parents, stats = source_split.validate_ledger(fullledger, expected)
    if not np.array_equal(source_split._keys(fullledger, source_split.OUTPUT_KEY),
                          source_split._keys(expected, source_split.OUTPUT_KEY)):
        raise ValueError("full ledger order differs from expected census")
    if np.any(fullledger["parent_fid"] != PARENT_FID) or len(np.unique(parents)) != len(parents):
        raise ValueError("requires unique original-parent identities throughout training")
    event_row = np.asarray(entries["event_row"])
    if np.any(event_row >= stats["events"]) or np.any(np.diff(event_row.astype(np.int64)) <= 0):
        raise ValueError("entries must follow complete stored-event order")
    for value in (file_ordinal, file_event_row):
        a = np.asarray(value)
        if a.shape != (stats["events"],) or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("file coordinates must cover complete stored census")
    coordinates = np.rec.fromarrays([file_ordinal, file_event_row])
    if len(np.unique(coordinates)) != len(coordinates):
        raise ValueError("duplicate file/event coordinates")
    packed = library.build(entries, steps, development=np.ones(len(event_row), dtype=bool), binning=library.Binning())
    packed.update({"geometry_" + key: np.asarray(value).copy() for key, value in geometry.items()})
    validate_packing(packed, entries, steps, geometry)
    provenance = {key: np.asarray(fullledger[key])[event_row].copy()
                  for key in source_split.OUTPUT_KEY + source_split.PARENT_KEY}
    provenance.update(donor_row=np.arange(len(event_row), dtype=np.int64), global_event_row=event_row.copy(),
                      file_ordinal=np.asarray(file_ordinal)[event_row].copy(),
                      file_event_row=np.asarray(file_event_row)[event_row].copy())
    return packed, provenance


def _coverage_equal(arrays, report, prior):
    if report != prior["training"]:
        raise ValueError("aggregate training coverage differs from authenticated census")
    records = prior["comparison"]["cells"]
    if len(records) != 96:
        raise ValueError("coverage comparison must contain all 96 cells")
    for i, record in enumerate(records):
        for key, value in arrays.items():
            saved = record[key] if key in coverage._COORDS else record["training"][key]
            if value[i] != saved:
                raise ValueError("authenticated training cell differs: " + key)
        if record["training_counts"] != int(arrays["counts"][i]):
            raise ValueError("authenticated training cell counts differ")


def _authorized(report):
    if (report["stored_events"] != 28265 or report["front_entry_events"] != 24120 or
            report["front"]["empty"] != 259 or report["front"]["weak"] != 835 or
            report["front"]["css_rows"] != 1034261):
        raise ValueError("candidate differs from the one authorized fixed population")


def run(args):
    """Authenticate and validate completely before creating a new output directory."""
    out = Path(args.out)
    source_audit._new(out)
    inputs = Inputs()
    for path in (__file__, source_split.__file__, source_overlap_report.__file__):
        inputs.read(path)
    prior = inputs.checked_json(args.coverage_report, args.coverage_sha256)
    source_modules = (coverage, export, library, production_check, source_audit, source_events,
                      training_audit, training_coverage)
    for module in source_modules:
        key = module.__name__
        if module is training_coverage and key not in prior["source_sha256"]:
            key = "__main__"  # Coverage CLI records its own executing module name.
        inputs.pin(module.__file__, prior["source_sha256"][key])
    inventory = inputs.checked_json(args.inventory, prior["input_sha256"]["inventory"])
    parent = inputs.checked_json(args.parent_summary, prior["input_sha256"]["parent_summary"])
    original = inventory["original_parent"]
    if original["fid"] != PARENT_FID or original["metadata_sha256"] != inventory["parent_metadata_sha256"]:
        raise ValueError("inventory original parent mismatch")
    for record in [original] + [r for r in inventory["rows"] if r["role"] == "training"]:
        filename = record["metadata_filename"]
        if Path(filename).name != filename:
            raise ValueError("unsafe metadata basename")
        inputs.pin(Path(args.inventory).parent / filename, record["metadata_sha256"])
    rows = training_audit.training_rows(args.inventory, prior["input_sha256"]["inventory"])
    if ([v["item"] for v in prior["validation"]] != [r["name"] for r in rows] or
            [v["item"] for v in prior["per_file"]] != [r["name"] for r in rows]):
        raise ValueError("coverage requires the exact ordered training allocation")
    pieces, files, ordinals, local_rows, reports = [], [], [], [], []
    reference_geometry = None
    for row, validation, old_report in zip(rows, prior["validation"], prior["per_file"]):
        source_overlap_report._require(validation, item=row["name"], fid=row["fid"],
                                       events=row["event_count"], art_sha256=row["sha256"])
        directory = Path(args.tables) / row["name"]
        table = directory / "export/calo_library_tables.root"
        inputs.pin(table, validation["tables_sha256"])
        evidence = validation["evidence_sha256"]
        if set(evidence) != set(RECEIPTS):
            raise ValueError("partial or unexpected saved extraction proof")
        for name, digest in evidence.items():
            inputs.pin(directory / name, digest)
        bound_dir = Path(args.bound) / row["name"] / "bound"
        for name in ("summary.json", "expected.npz", "ledger.npz"):
            inputs.pin(bound_dir / name, _binding_pin(parent, row["name"], name))
        file = source_overlap_report._bound(bound_dir, inputs, role="training", item=row["name"],
                                           fid=row["fid"], parent_fid=PARENT_FID, count=row["event_count"])
        binding = file["summary"]
        hashes = binding["input_sha256"]
        if set(hashes) != PROOF_KEYS or hashes != validation["source_proof"]["input_sha256"]:
            raise ValueError("partial or differing saved source proof")
        source_overlap_report._require(binding, ordinal=row["ordinal"], art_sha256=row["sha256"],
                                       census_receipt_checked=True, manifest_sha256=inventory["manifest_sha256"])
        source_overlap_report._require(hashes, inventory=prior["input_sha256"]["inventory"],
                                       metadata=row["metadata_sha256"], parent_metadata=inventory["parent_metadata_sha256"])
        if any(np.any(file["expected"][field] != row[field]) for field in ("run", "subrun")):
            raise ValueError("full stored EventID differs from inventory run/subrun")
        numeric, identity = source_events.read(table, file["expected"], source_tag="TargetStopResampler::Primary",
                                               source_branch="art::EventIDs_TargetStopResampler__Primary.")
        for field in export.EVENT + ("source_run", "source_subrun", "source_event", "source_count"):
            if not np.array_equal(numeric[field], file["ledger"][field]):
                raise ValueError("numeric source ledger mismatch: " + field)
        if identity != validation["source_identity"]:
            raise ValueError("numeric source identity report differs")
        e, s, geometry, census, total = export.read(table, input_mode="primary")
        if census != validation["raw_reconciliation"]["census"]:
            raise ValueError("raw reconciliation census differs")
        if validation["status"]["failed_transport_events"] != 0 or validation["status"]["events"] != row["event_count"]:
            raise ValueError("failed/incomplete transport census")
        _geometry(geometry, reference_geometry)
        if reference_geometry is None:
            reference_geometry = geometry
        _, report = coverage.census(e, s, census, total)
        report = dict(item=row["name"], fid=row["fid"], **report)
        if report != old_report:
            raise ValueError("per-file training coverage differs")
        inputs.pin(table, validation["tables_sha256"])
        pieces.append((e, s, census, total))
        files.append(file)
        reports.append(report)
        ordinals.append(np.full(row["event_count"], row["ordinal"], dtype=np.int64))
        local_rows.append(np.arange(row["event_count"], dtype=np.int64))
    entries, steps, census, total = training_coverage.merge(pieces)
    arrays, report = coverage.census(entries, steps, census, total)
    _coverage_equal(arrays, report, prior)
    _authorized(report)
    expected = source_overlap_report._concat(files, "expected", source_split.OUTPUT_KEY)
    ledger = source_overlap_report._concat(files, "ledger", source_overlap_report.LEDGER_FIELDS)
    packed, provenance = pack(entries, steps, reference_geometry, ledger, expected,
                              file_ordinal=np.concatenate(ordinals), file_event_row=np.concatenate(local_rows))
    packed_arrays, _ = coverage.frozen(packed)
    if any(not np.array_equal(packed_arrays[key], value) for key, value in arrays.items()):
        raise ValueError("packed training cells differ")
    modules = source_modules + (source_split, source_overlap_report)
    source_hashes = {module.__name__: source_audit._sha(inputs.read(module.__file__)) for module in modules}
    source_hashes["calofastsim.showers.production_candidate"] = source_audit._sha(inputs.read(__file__))
    metadata = dict(schema_version=1, candidate="production-only-all-training-front", artifacts=ARTIFACTS,
                    training=report, file_order=[dict(item=r["name"], ordinal=r["ordinal"], fid=r["fid"],
                                                     events=r["event_count"]) for r in rows],
                    selection="all first-forward direct-CE entries; inventory file order then stored event row; no outcome cuts",
                    coverage_sha256=args.coverage_sha256, parent_summary_sha256=prior["input_sha256"]["parent_summary"],
                    source_sha256=source_hashes, candidate_library_constructed=True, sampled=False,
                    legacy_donors_included=False, source_independence_certified=False,
                    limitations=["Selected stored production, not inclusive generated acceptance or proton independence.",
                                 "Incident simulation bins are diagnostics, not measured-variable correction models.",
                                 "Existing development/pilot overlap is unchanged; no pruning or reassignment.",
                                 "No draws, reconstruction, new physics, confirmation or source-control operation performed."])
    result = dict(metadata, per_file=reports, validation=prior["validation"], input_sha256=dict(inputs.sha256),
                  packing_scalars_counts_exact=True, packing_roundtrip_verified=True,
                  position_time_atol=1e-9, unique_training_parents=True,
                  training_only=True, confirmation_event_contents_opened_by_this_operation=False)
    source_audit._text(result)
    inputs.unchanged()
    out.mkdir(parents=True)
    with (out / ARTIFACTS["library"]).open("xb") as stream:
        library.save(stream, packed, metadata)
    saved, saved_metadata = library.load(out / ARTIFACTS["library"])
    if saved_metadata != metadata or set(saved) != set(packed) or any(
            not np.array_equal(saved[key], value) for key, value in packed.items()):
        raise ValueError("saved library roundtrip differs")
    for key, table in (("donor_provenance", provenance), ("expected", expected), ("ledger", ledger), ("cells", arrays)):
        source_overlap_report._save_npz(out / ARTIFACTS[key], table)
    source_audit._save_json(out / ARTIFACTS["metadata"], metadata)
    result["output_sha256"] = {name: source_events.file_digest(out / name)
                               for key, name in ARTIFACTS.items() if key != "summary"}
    inputs.unchanged()
    source_audit._save_json(out / ARTIFACTS["summary"], result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("coverage-report", "tables", "inventory", "bound", "parent-summary", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--coverage-sha256", required=True, help="Externally trusted frozen coverage SHA256")
    report = run(parser.parse_args(argv))
    print(source_audit._text({key: report[key] for key in ("artifacts", "training", "output_sha256")}), end="")


if __name__ == "__main__":
    main()
