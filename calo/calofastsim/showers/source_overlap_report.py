"""Final, no-pruning immediate-source overlap audit of 20 production files + pilot.

This consumes authenticated source_audit bindings, not art files. Mixer/module
proof and authentication of the pilot original copy are external prerequisites.
The development inventory ordinal 1 is the previously exported numeric file0.
Pilot summary contract: role=pilot, item, events=source_rows=2000, parent_fid,
art_sha256, output_fid='sha256-'+art_sha256, ordered_full_census_exact=true,
source_file_resolution='verified_original_copy', and input_sha256 digest mapping.
A checksum identity is an artifact identity, NOT a SAM catalog FID.
"""
from __future__ import annotations

import argparse
import io
from itertools import combinations
from pathlib import Path

import numpy as np

from . import source_audit as audit, source_event_check, source_events, source_split
from .export import EVENT

ROLES = ("training", "development", "pilot")
LEDGER_FIELDS = source_split.OUTPUT_KEY + source_split.PARENT_KEY + (
    "source_count", "source_present", "source_file_resolved")
SCOPE = "Immediate original-file source EventID intersection only; provenance authentication is an external prerequisite."


class Inputs:
    """Hash the exact parsed bytes and reject changes before publication."""

    def __init__(self):
        self.sha256 = {}

    def read(self, path):
        path = Path(path).absolute()
        blob = path.read_bytes()
        sha = audit._sha(blob)
        key = str(path)
        if key in self.sha256 and self.sha256[key] != sha:
            raise ValueError(f"input changed during audit: {path}")
        self.sha256[key] = sha
        return blob

    def json(self, path):
        value = audit._json(self.read(path))
        if not isinstance(value, dict):
            raise ValueError("JSON input must be an object")
        return value

    def npz(self, path, fields=None):
        with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("duplicate NPZ columns")
            if fields is not None and set(archive.files) != set(fields):
                raise ValueError("NPZ column schema mismatch")
            return dict(archive)

    def unchanged(self):
        for name, sha in self.sha256.items():
            if source_events.file_digest(name) != sha:
                raise ValueError(f"input changed during audit: {name}")


def _require(mapping, **wanted):
    for key, value in wanted.items():
        if mapping.get(key) != value or type(mapping.get(key)) is not type(value):
            raise ValueError(f"binding {key} mismatch")


def _proof_hashes(summary):
    hashes = summary.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("binding needs input_sha256 proof hashes")
    for value in hashes.values():
        audit._digest(value)
    if summary.get("source_independence_certified", False) is not False:
        raise ValueError("binding must not certify source independence")
    return hashes


def _inventory(path, inputs):
    inv = inputs.json(path)
    rows = inv.get("rows")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("inventory must contain exactly twenty production files")
    audit._inventory_item(inv, rows[0]["name"])
    _require(inv, schema_version=1)
    parent = inv["original_parent"]
    _require(parent, name=audit.PARENT_NAME, fid=audit.PARENT_FID,
             did="mu2e:" + audit.PARENT_NAME)
    if parent["metadata_sha256"] != inv["parent_metadata_sha256"]:
        raise ValueError("inventory parent metadata hash mismatch")
    for record in [parent] + rows:
        filename = record["metadata_filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("metadata filename must be a local basename")
        blob = inputs.read(Path(path).parent / filename)
        if audit._sha(blob) != audit._digest(record["metadata_sha256"]):
            raise ValueError("raw metadata hash mismatch")
        metadata = audit._json(blob)
        if record is parent:
            audit.parent_identity(metadata)
        else:
            frozen = {field: record[field] for field in (
                "did", "fid", "name", "size", "checksums", "role", "ordinal", "run", "subrun")}
            checked = audit.validate_metadata(metadata, frozen)
            if any(record.get(key) != value or type(record.get(key)) is not type(value)
                   for key, value in checked.items()):
                raise ValueError("inventory disagrees with saved metadata")
    return inv


def _bound(directory, inputs, *, role, item, fid, parent_fid, count):
    directory = Path(directory)
    summary = inputs.json(directory / "summary.json")
    _require(summary, role=role, item=item, events=count, source_rows=count,
             parent_fid=parent_fid, ordered_full_census_exact=True)
    _proof_hashes(summary)
    expected = inputs.npz(directory / "expected.npz", source_split.OUTPUT_KEY)
    ledger = inputs.npz(directory / "ledger.npz", LEDGER_FIELDS)
    for name in ("expected.npz", "ledger.npz"):
        declared = summary.get("output_sha256", {}).get(name)
        actual = inputs.sha256[str((directory / name).absolute())]
        if audit._digest(declared) != actual:
            raise ValueError("bound output hash mismatch: " + name)
    parents, stats = source_split.validate_ledger(ledger, expected)
    if stats["events"] != count:
        raise ValueError("bound census count mismatch")
    if (np.any(ledger["output_fid"] != fid) or np.any(expected["output_fid"] != fid)
            or np.any(ledger["parent_fid"] != parent_fid)):
        raise ValueError("bound output/parent FID mismatch")
    if not np.array_equal(source_split._keys(ledger, source_split.OUTPUT_KEY),
                          source_split._keys(expected, source_split.OUTPUT_KEY)):
        raise ValueError("bound full EventID order mismatch")
    unique, multiplicity = np.unique(parents, return_counts=True)
    return dict(role=role, item=item, output_fid=fid, expected=expected, ledger=ledger,
                summary=summary, stats=stats, unique=unique, multiplicity=multiplicity)


def _concat(files, key, fields):
    # Loops are over the fixed file allocation and schema, never events.
    return {field: np.concatenate([file[key][field] for file in files]) for field in fields}


def _file0_parity(path, file, inputs):
    numeric = inputs.npz(path)
    fields = EVENT + source_event_check.SOURCE_IDS
    a, b = source_split._keys(numeric, fields), source_split._keys(file["ledger"], fields)
    if not np.array_equal(a, b):
        raise ValueError("file0 numeric source IDs/full EventID ordered reuse parity mismatch")
    counts = source_split._columns(numeric, ("source_count",))[0]
    if counts.shape != (len(a),) or np.any(counts != 1):
        raise ValueError("file0 ambiguous source counts")
    return dict(item=file["item"], events=len(a), full_eventid_order_exact=True,
                source_ids_exact=True, art_reread=False,
                scope="Numeric-source reuse parity; FIDs supplied only by authenticated bound inventory")


def _save_npz(path, table):
    with Path(path).open("xb") as stream:
        np.savez(stream, **table)


def _families(shared):
    return {field: shared[field] for field in source_split.PARENT_KEY}


def report(inventory, production, pilot, file0_sources, out):
    """Validate all inputs, then publish new combined tables and exact intersections."""
    out = Path(out)
    audit._new(out)
    inputs = Inputs()
    inv = _inventory(inventory, inputs)
    inventory_sha = inputs.sha256[str(Path(inventory).absolute())]
    files = []
    for row in inv["rows"]:
        file = _bound(Path(production) / row["name"] / "bound", inputs,
                      role=row["role"], item=row["name"], fid=row["fid"],
                      parent_fid=row["parent_fid"], count=row["event_count"])
        summary = file["summary"]
        _require(summary, ordinal=row["ordinal"], manifest_sha256=inv["manifest_sha256"],
                 art_sha256=row["sha256"], census_receipt_checked=True)
        hashes = summary["input_sha256"]
        _require(hashes, inventory=inventory_sha, metadata=row["metadata_sha256"],
                 parent_metadata=inv["parent_metadata_sha256"])
        for key in ("expected_events", "printer_log", "config", "census_receipt"):
            audit._digest(hashes.get(key))
        if any(np.any(file["expected"][field] != row[field]) for field in ("run", "subrun")):
            raise ValueError("bound census run/subrun differs from inventory")
        file["ordinal"] = row["ordinal"]
        files.append(file)
    pilot_summary = inputs.json(Path(pilot) / "summary.json")
    pilot_sha = audit._digest(pilot_summary.get("art_sha256"))
    pilot_fid = "sha256-" + pilot_sha
    parent_fid = audit._fid(pilot_summary.get("parent_fid"))
    if parent_fid != inv["original_parent"]["fid"]:
        raise ValueError("pilot original parent FID differs from authenticated inventory")
    _require(pilot_summary, output_fid=pilot_fid, source_file_resolution="verified_original_copy")
    item = pilot_summary.get("item")
    if not isinstance(item, str) or not item:
        raise ValueError("pilot item must identify the legacy artifact")
    files.append(_bound(pilot, inputs, role="pilot", item=item, fid=pilot_fid,
                        parent_fid=parent_fid, count=2000))
    file0 = next(file for file in files if file["role"] == "development" and file["ordinal"] == 1)
    parity = _file0_parity(file0_sources, file0, inputs)
    combined, per_role = {}, {}
    for role in ROLES:
        selected = [file for file in files if file["role"] == role]
        expected = _concat(selected, "expected", source_split.OUTPUT_KEY)
        ledger = _concat(selected, "ledger", LEDGER_FIELDS)
        _, stats = source_split.validate_ledger(ledger, expected)
        combined[role] = (expected, ledger)
        per_role[role] = dict(stats, files=len(selected), complete_expected_events=len(expected["event"]),
                              source_independence_certified=False, no_pruning=True)
    role_pairs, family_arrays = {}, {}
    for left, right in combinations(ROLES, 2):
        ea, a = combined[left]
        eb, b = combined[right]
        result = source_split.compare_ledgers(a, b, ea, eb)
        name = left + "--" + right
        shared = np.intersect1d(source_split._keys(a, source_split.PARENT_KEY),
                                source_split._keys(b, source_split.PARENT_KEY))
        filename = "shared/" + name + ".npz"
        family_arrays[filename] = _families(shared)
        role_pairs[name] = dict(result, shared_event_families_npz=filename)
    collided = []
    for (i, a), (j, b) in combinations(enumerate(files), 2):
        shared, ia, ib = np.intersect1d(a["unique"], b["unique"], return_indices=True)
        if not len(shared):
            continue  # Valid measured zero, not a missing file or missing comparison.
        filename = f"shared/file-{i:02d}--file-{j:02d}.npz"
        family_arrays[filename] = _families(shared)
        collided.append(dict(left=a["item"], right=b["item"], left_role=a["role"], right_role=b["role"],
                             shared_parent_events=len(shared),
                             left_events_with_shared_parent=int(a["multiplicity"][ia].sum()),
                             right_events_with_shared_parent=int(b["multiplicity"][ib].sum()),
                             shared_event_families_npz=filename))
    per_file = [dict(item=file["item"], role=file["role"], output_fid=file["output_fid"],
                     complete_expected_events=len(file["expected"]["event"]), **file["stats"])
                for file in files]
    sources = {}
    for module in (source_split, audit, source_event_check, source_events):
        sources[module.__name__] = audit._sha(inputs.read(module.__file__))
    sources[__name__] = audit._sha(inputs.read(__file__))
    total = sum(role["complete_expected_events"] for role in per_role.values())
    result = dict(schema_version=1, manifest_sha256=inv["manifest_sha256"],
                  production_files=20, production_expected_events=total - 2000,
                  pilot_reference_events=2000, all_complete_expected_events=total,
                  roles=per_role, per_file=per_file, role_pairs=role_pairs,
                  cross_file_pairs_checked=len(files) * (len(files) - 1) // 2,
                  cross_file_pairs_with_parent_overlap=collided,
                  cross_file_pairs_without_parent_overlap=210 - len(collided),
                  zero_overlap_is_valid=True, file0_reuse_parity=parity,
                  input_sha256=inputs.sha256, source_sha256=sources,
                  external_prerequisites="Upstream module validation and authenticated full census/source-file binding; not independently repeated here",
                  no_pruning=True, source_independence_certified=False,
                  confirmation_event_contents_opened_by_this_operation=False, scope=SCOPE,
                  pilot_output_identity_kind="SHA256 artifact identity; not a SAM catalog FID")
    inputs.unchanged()
    out.mkdir(parents=True)  # No exist_ok: includes concurrent-writer protection.
    (out / "shared").mkdir()
    outputs = {}
    for role in ROLES:
        directory = out / role
        directory.mkdir()
        for name, table in zip(("expected.npz", "ledger.npz"), combined[role]):
            _save_npz(directory / name, table)
            outputs[f"{role}/{name}"] = source_events.file_digest(directory / name)
        audit._save_json(directory / "summary.json", per_role[role])
        outputs[f"{role}/summary.json"] = source_events.file_digest(directory / "summary.json")
    for name, table in family_arrays.items():
        _save_npz(out / name, table)
        outputs[name] = source_events.file_digest(out / name)
    # If any input changed while writing, no successful final summary is published.
    inputs.unchanged()
    result["output_sha256"] = outputs
    audit._save_json(out / "summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "production", "pilot", "file0-sources", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(args.inventory, args.production, args.pilot, args.file0_sources, args.out)
    print(audit._text(result), end="")


if __name__ == "__main__":
    main()
