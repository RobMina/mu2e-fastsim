"""Bounded saved-metadata inventory; no exposure normalization or physics audit.

The root contains selection_window_v2.json, provenance_final.json and
sam_<sequencer>.txt for the twelve exact source basenames declared by
provenance_final.json. The filename sequencer is checked against dh.sequencer;
it is not interpreted as an event identity. Only the explicitly bound exposure
ledger's Parquet footer is decoded. Selected files and ntuples are never opened.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import pyarrow.parquet as pq

from . import refit_saved_coverage as C

SCHEMA = "saved-exposure-metadata-inventory-v1"
JSON_CAP = 8 * 1024**2
SAM_CAP = 1024**2
TOTAL_CAP = 16 * 1024**2
LEDGER_CAP = 32 * 1024**2
OUTPUT_CAP = 1024**2
MAX_ROWS, MAX_COLUMNS, MAX_GROUPS = 100000, 32, 4096
MAX_UNCOMPRESSED = 64 * 1024**2
MAX_PARENTS = 5000
SOURCE_COUNT = 12
# Reference values are documentation, not defaults or an alternative to caller pins.
RETAINED_SELECTION_SHA256 = "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
RETAINED_PROVENANCE_SHA256 = "3ffe019bf7f72ee20cc14a136df9a11a5ccca157c282bf095b8ea4483f3c35e1"
SCOPE = ("Saved catalog observations and exposure-ledger metadata only. No selected-track, "
         "ntuple or ROOT access, prior-audit execution, table decoding, new selection, "
         "pairwise parent-overlap analysis, weight model or physics conclusion.")
NOT_ESTABLISHED = "not established by this inventory"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "invalid SHA256 pin")
    return value


def local_path(value):
    require(isinstance(value, (str, Path)) and bool(str(value)) and "://" not in str(value),
            "local pathname required")
    path = Path(os.path.abspath(value))  # lexical only: never resolve selected provenance
    require("pnfs" not in path.parts and not any(p.endswith(".root") for p in path.parts),
            "PNFS/ROOT access is outside this inventory")
    return path


def no_symlinks(path):
    for part in [*reversed(path.parents), path]:
        require(not stat.S_ISLNK(part.lstat().st_mode), "symlink path is not admitted")


def regular_size(path, cap, *, strict=False):
    no_symlinks(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "regular file required")
    require(info.st_size < cap if strict else info.st_size <= cap, "file byte cap exceeded before read/hash")
    return info.st_size


def outside(out, excluded):
    require(not os.path.lexists(out), "choose a new report; overwrite forbidden")
    no_symlinks(out.parent)
    require(out.parent.is_dir(), "existing output directory required")
    for directory in excluded:
        require(not out.is_relative_to(directory), "output must be outside input/code/selection parents")


def code_paths():
    # Bind the inventory and used pure helpers, without generation/admission calls.
    return list(dict.fromkeys(Path(p).absolute() for p in
                (__file__, C.__file__, C.N.__file__, C.SR.__file__, C.M.S.I.__file__)))


def source_paths(root, provenance):
    sources = provenance.get("sources")
    require(type(sources) is dict and len(sources) == SOURCE_COUNT, "exactly twelve sources required")
    paths = {}
    for name, record in sources.items():
        require(type(name) is str and type(record) is dict, "invalid source record")
        match = re.fullmatch(r"dts\.mu2e\.NeutralsFlashCat\.MDC2025ad\.[0-9]+_([0-9]+)\.art", name)
        require(match is not None, "exact source basename required")
        sequencer = unsigned(match.group(1), "source filename sequencer")
        path = root / f"sam_{sequencer}.txt"
        require(path not in [p for p, _ in paths.values()], "ambiguous source sequencer")
        sam = record.get("sam")
        require(type(sam) is dict and sam.get("file_name") == name, "saved SAM filename binding differs")
        digest(sam.get("metadata_sha256"))
        digest(sam.get("parents_sha256"))
        require(type(sam.get("parent_count")) is int and 0 < sam["parent_count"] <= MAX_PARENTS,
                "invalid saved parent count")
        paths[name] = (path, sequencer)
    return paths


def unsigned(text, field, *, positive=False):
    require(type(text) is str and len(text) <= 19 and re.fullmatch(r"[0-9]+", text) is not None,
            f"invalid integer domain: {field}")
    value = int(text)
    require(int(positive) <= value <= 2**63 - 1, f"invalid integer domain: {field}")
    return value


def parse_sam(payload):
    """Parse (possibly right-aligned) field headers, then their continuations.

    Non-Parents field values are preserved as strings, including all continuation
    lines. No key whitelist hides catalog fields; duplicate headers fail closed.
    Parents are exact basenames, sorted only to bind the inherited compact hash.
    """
    require(len(payload) < SAM_CAP, "SAM byte cap exceeded before decode")
    fields, current = {}, None
    header = re.compile(r"^(\s*)([A-Za-z][A-Za-z0-9_. -]*):\s*(.*)$")
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        match = header.fullmatch(line)
        if match and current is not None and match[2].strip() in ("adler32", "enstore") and current == "Checksum":
            fields[current].append(line.strip())
            continue
        if match:
            key = match[2].strip()
            require(key not in fields, "duplicate SAM field: " + key)
            current = key
            fields[key] = [match[3].strip()] if match[3].strip() else []
        else:
            require(current is not None and line[0].isspace(), "malformed SAM continuation")
            fields[current].append(line.strip())
        if current == "Parents":
            require(len(fields[current]) <= MAX_PARENTS, "SAM parent count cap exceeded")
    parents = fields.pop("Parents", [])
    require(parents and len(set(parents)) == len(parents), "missing or duplicate SAM parents")
    require(all(re.fullmatch(r"[A-Za-z0-9_.-]+\.art", p) for p in parents),
            "parent must be an exact art basename")
    scalar = {key: "\n".join(lines) for key, lines in fields.items()}
    parent_sha = hashlib.sha256(("\n".join(sorted(parents)) + "\n").encode()).hexdigest()
    return scalar, len(parents), parent_sha


def sam_inventory(path, sequencer, name, inherited, evidence):
    evidence.bound(path, inherited["metadata_sha256"])
    payload = C.bounded_bytes(path, SAM_CAP - 1)
    require(hashlib.sha256(payload).hexdigest() == inherited["metadata_sha256"], "consumed SAM bytes differ from pin")
    fields, count, parent_sha = parse_sam(payload)
    require(fields.get("File Name") == name, "SAM File Name differs from exact source filename")
    require(fields.get("dh.sequencer") == name.removesuffix(".art").split(".")[-1],
            "SAM dh.sequencer differs from filename")
    events = unsigned(fields.get("Event Count"), "Event Count")
    if "event_count" in inherited:
        require(type(inherited["event_count"]) is int and events == inherited["event_count"], "SAM Event Count differs")
    for old, field in (("file_id", "File Id"), ("source_sha256", "dh.sha256")):
        if old in inherited:
            require(fields.get(field) == inherited[old], "SAM compact binding differs: " + field)
    require(count == inherited["parent_count"], "SAM parent_count differs")
    require(parent_sha == inherited["parents_sha256"], "SAM parents_sha256 differs")
    gencount = unsigned(fields["dh.gencount"], "dh.gencount", positive=True) if "dh.gencount" in fields else None
    return dict(path=str(path), sha256=inherited["metadata_sha256"], sequencer=sequencer,
                scalar_fields=fields, parent_count=count, parents_sha256=parent_sha,
                inherited_parent_binding_matches=True,
                gencount_catalog_observation=gencount,
                interpretation="Catalog observation only; no normalization, alias sum, reuse weight or independent-shower count.")


def ledger_metadata(path):
    """Decode only bounded Parquet footer metadata; never a table or row group."""
    regular_size(path, LEDGER_CAP)
    with pq.ParquetFile(path, thrift_string_size_limit=OUTPUT_CAP,
                        thrift_container_size_limit=100000, pre_buffer=False) as parquet:
        meta = parquet.metadata
    require(0 <= meta.num_rows <= MAX_ROWS, "ledger metadata row cap exceeded")
    require(0 <= meta.num_columns <= MAX_COLUMNS, "ledger metadata column cap exceeded")
    require(0 <= meta.num_row_groups <= MAX_GROUPS, "ledger metadata row-group cap exceeded")
    uncompressed = 0
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        require(0 <= group.total_byte_size <= MAX_UNCOMPRESSED, "ledger metadata uncompressed cap exceeded")
        require(group.num_columns == meta.num_columns, "ledger metadata column count differs")
        column_bytes = 0
        for j in range(group.num_columns):
            amount = group.column(j).total_uncompressed_size
            require(0 <= amount <= MAX_UNCOMPRESSED, "ledger metadata uncompressed cap exceeded")
            column_bytes += amount
        uncompressed += max(group.total_byte_size, column_bytes)
        require(uncompressed <= MAX_UNCOMPRESSED, "ledger metadata uncompressed cap exceeded")
    schema = meta.schema.to_arrow_schema()
    names = schema.names
    return dict(nrows=meta.num_rows, ncols=meta.num_columns, row_groups=meta.num_row_groups,
                declared_uncompressed_bytes=uncompressed,
                parquet_schema=str(meta.schema), arrow_schema=str(schema),
                arrow_schema_serialized_base64=base64.b64encode(schema.serialize().to_pybytes()).decode("ascii"),
                fields=[dict(name=f.name, type=str(f.type), nullable=f.nullable) for f in schema],
                per_row_source_identity=dict(available_schema_field_names=names,
                    interpretation="Schema names only; field values, identity completeness and uniqueness " + NOT_ESTABLISHED),
                table_decoded=False, counts_recomputed=False)


def run(root, out, *, expected_provenance_sha256, expected_selection_sha256):
    """Inventory two externally pinned JSONs, twelve SAM texts and one ledger footer."""
    digest(expected_provenance_sha256)
    digest(expected_selection_sha256)
    root, out = local_path(root), local_path(out)
    no_symlinks(root)
    require(root.is_dir(), "input root must be a directory")
    codes = code_paths()
    excluded = [root, *(p.parent for p in codes)]
    outside(out, excluded)
    pp, sp = root / "provenance_final.json", root / "selection_window_v2.json"
    sizes = {p: regular_size(p, JSON_CAP, strict=True) for p in (pp, sp)}
    require(sum(sizes.values()) <= TOTAL_CAP, "metadata total byte cap exceeded before hash")
    # Derive fixed names from bounded JSON before admitting all metadata sizes.
    preflight = C.bounded_json(pp)
    paths = source_paths(root, preflight)
    sizes.update({p: regular_size(p, SAM_CAP, strict=True) for p, _ in paths.values()})
    require(sum(sizes.values()) <= TOTAL_CAP, "metadata total byte cap exceeded before hash")
    evidence = C.Evidence()
    snapshots, code = {}, {}
    for path in codes:
        require(path not in sizes, "source code must not alias consumed metadata")
        payload = C.bounded_bytes(path)
        code[str(path)] = hashlib.sha256(payload).hexdigest()
        snapshots[str(path)] = payload.decode("utf-8")
        evidence.bound(path, code[str(path)])
    provenance = evidence.json(pp, expected_provenance_sha256)
    require(source_paths(root, provenance) == paths, "provenance changed after size preflight")
    selection = evidence.json(sp, expected_selection_sha256)
    selected = selection.get("selection")
    ledger = selection.get("exposure_ledger")
    require(type(selected) is dict and type(ledger) is dict, "saved selection and exposure_ledger bindings required")
    require(type(selected.get("path")) is str and Path(selected["path"]).is_absolute(),
            "absolute selected provenance path required")
    digest(selected.get("sha256"))
    # Lexical normalization only: the selected path is never stat'ed or resolved.
    selected_path = Path(os.path.abspath(selected["path"]))
    excluded.append(selected_path.parent)
    outside(out, excluded)
    require(type(ledger.get("path")) is str and "://" not in ledger["path"],
            "explicit local ledger path required")
    lp = local_path(ledger["path"] if Path(ledger["path"]).is_absolute() else root / ledger["path"])
    require(lp.is_relative_to(root) and lp.suffix == ".parquet", "ledger must be an explicit local Parquet within root")
    require(selected_path not in {*sizes, lp}, "selected-track path must not be consumed")
    regular_size(lp, LEDGER_CAP)
    evidence.bound(lp, digest(ledger.get("sha256")))
    rows = {name: sam_inventory(path, seq, name, provenance["sources"][name]["sam"], evidence)
            for name, (path, seq) in sorted(paths.items())}
    metadata = ledger_metadata(lp)
    result = dict(schema=SCHEMA, scope=SCOPE, inputs=dict(evidence.bindings),
                  code_sha256=code, code_snapshots=snapshots,
                  sources=rows, metadata_bytes_preflight=sum(sizes.values()),
                  provenance_report=dict(path=str(pp), sha256=expected_provenance_sha256),
                  original_selection_report=dict(path=str(sp), sha256=expected_selection_sha256,
                      status="inherited saved metadata, not independently recomputed or validated",
                      saved_fields=selection),
                  inherited_diagnostic=dict(status="inherited; not recomputed",
                      value=selection.get("diagnostic")),
                  inherited_measured_field_checks=dict(status="inherited; not validated measured PBI",
                      value=selection.get("measured_field_checks")),
                  selected_track_provenance=dict(path=selected["path"], sha256=selected["sha256"],
                      interpretation="Inherited provenance only; not opened, stat'ed or hashed"),
                  exposure_ledger=dict(path=str(lp), sha256=ledger["sha256"], **metadata),
                  interpretation=dict(validated_measured_pbi=NOT_ESTABLISHED,
                      deeper_parent_metadata=NOT_ESTABLISHED, intended_reuse_declaration=NOT_ESTABLISHED,
                      gencount_semantics_and_additivity=NOT_ESTABLISHED,
                      independent_shower_count=NOT_ESTABLISHED,
                      scope="These are limits of this inventory, not assertions of absence beyond consumed metadata. "
                            "Inspect the complete scalar field maps and inherited reports; without documented "
                            "semantics, catalog counts cannot be combined or summed over aliases."),
                  selected_tracks_read=False, selected_tracks_rehashed=False,
                  prior_audit_executed=False, physics_executed=False, parent_overlap_recomputed=False,
                  weight_model=None, F=None, finite_library_uncertainty=None,
                  output_cap_bytes=OUTPUT_CAP,
                  publication="exclusive JSON; no atomic-completion or crash-durability claim")
    # Stat caps and rehash the complete evidence set immediately before publication.
    for p in sizes:
        regular_size(p, JSON_CAP if p in (pp, sp) else SAM_CAP, strict=True)
    require(sum(p.stat().st_size for p in sizes) <= TOTAL_CAP, "metadata total byte cap changed late")
    regular_size(lp, LEDGER_CAP)
    for path in codes:
        require(C.file_sha(path) == code[str(path)], "inventory source changed late")
    evidence.recheck()
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(len(payload) < OUTPUT_CAP, "inventory output byte cap exceeded")
    outside(out, excluded)
    evidence.recheck()
    with out.open("xb") as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    for name in ("expected-provenance-sha256", "expected-selection-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "scope", "interpretation")}, indent=2))


if __name__ == "__main__":
    main()
