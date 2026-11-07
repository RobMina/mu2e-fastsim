"""Bounded, provenance-only training/development audit; never reads confirmation.

metadata queries SAM only. census reads only EventAuxiliary (and hashes the full
input). bind proves the complete stock-printer route against that independent
census and a strict single-module config_dumper dump. No independence claim.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

import numpy as np
import uproot

from .. import artsteps
from . import source_event_check, source_events, source_split
from .export import EVENT

PARENT_NAME = "sim.mu2e.MuminusStopsCat.MDC2025ac.001430_00000000.art"
PARENT_FID = "102778799"
ROLES = ("training", "development")
TIMEOUT = 120
file_digest = source_events.file_digest


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _new(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)


def _json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=invalid)


def _positive(value, label):
    if type(value) is not int or value <= 0 or value > np.iinfo(np.int64).max:
        raise ValueError(f"{label} must be a positive signed-range integer")
    return value


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256")
    return value


def _fid(value):
    if type(value) not in (str, int) or re.fullmatch(r"[1-9][0-9]*", str(value)) is None:
        raise ValueError("expected canonical numeric SAM FID")
    return str(value)


def _checksums(value):
    if isinstance(value, list):
        pairs = [item.split(":", 1) for item in value if isinstance(item, str)]
        if len(pairs) != len(value) or any(len(item) != 2 for item in pairs):
            raise ValueError("invalid SAM checksums")
        if len({item[0] for item in pairs}) != len(pairs):
            raise ValueError("duplicate checksum algorithm")
        value = dict(pairs)
    patterns = {"adler32": r"[0-9a-fA-F]{8}", "enstore": r"[0-9]+",
                "md5": r"[0-9a-fA-F]{32}", "sha256": r"[0-9a-fA-F]{64}"}
    if not isinstance(value, dict) or not value or any(
            key not in patterns or not isinstance(v, str) or not re.fullmatch(patterns[key], v)
            for key, v in value.items()):
        raise ValueError("invalid or missing checksums")
    return value


def _identity(metadata, name, fid):
    if metadata.get("file_name") != name or _fid(metadata.get("file_id")) != fid:
        raise ValueError("SAM filename/FID mismatch")
    # SAM name/FID binds the fixed mu2e DID even if no separate DID field is emitted.
    did = "mu2e:" + name
    for key in ("did", "rucio_did"):
        if key in metadata and metadata[key] != did:
            raise ValueError("SAM DID mismatch")
    for key in ("namespace", "dh.namespace"):
        if key in metadata and metadata[key] != "mu2e":
            raise ValueError("SAM namespace mismatch")


def manifest_rows(manifest):
    """Select exactly the two explicitly authorized ten-file roles, not all roles."""
    roles = manifest["roles"]
    if set(roles) != set(source_split.ROLES):
        raise ValueError("expected explicit training/development/confirmation roles")
    rows = []
    for role in ROLES:
        if not isinstance(roles[role], list) or len(roles[role]) != 10:
            raise ValueError("bounded audit requires exactly ten files per authorized role")
        for ordinal, frozen in enumerate(roles[role], 1):
            did = frozen["did"]
            if not isinstance(did, str) or not did.startswith("mu2e:"):
                raise ValueError("expected frozen mu2e DID")
            name = did.split(":", 1)[1]
            match = re.fullmatch(r"dts\.mu2e\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.(\d+)_(\d+)\.art", name)
            if match is None:
                raise ValueError("unrecognized production filename")
            run, subrun = map(int, match.groups())
            rows.append(dict(did=did, fid=_fid(frozen["fid"]), name=name,
                             size=_positive(frozen["size"], "size"), checksums=_checksums(frozen["checksums"]),
                             role=role, ordinal=ordinal, run=run, subrun=subrun))
    # Confirmation identities are checked for disjointness, never queried/opened.
    all_rows = rows + roles["confirmation"]
    for field in ("did", "fid"):
        values = [str(row[field]) for row in all_rows]
        if len(values) != len(set(values)):
            raise ValueError("duplicate/overlapping role identities")
    return rows


def parent_identity(metadata):
    _identity(metadata, PARENT_NAME, PARENT_FID)
    return dict(name=PARENT_NAME, fid=PARENT_FID, did="mu2e:" + PARENT_NAME)


def _parent(metadata):
    parents = metadata.get("parents")
    if not isinstance(parents, list) or not parents:
        raise ValueError("missing parent metadata")
    names, fids = [], []
    for parent in parents:
        name = parent.get("file_name")
        if not isinstance(name, str) or not name.startswith(("sim.", "cnf.")):
            raise ValueError("unsupported parent; only sim and cnf parents are permitted")
        names.append(name)
        fids.append(_fid(parent.get("file_id")))
    if len(set(names)) != len(names) or len(set(fids)) != len(fids):
        raise ValueError("duplicate or ambiguous parent")
    simulation = [i for i, name in enumerate(names) if name.startswith("sim.")]
    if len(simulation) != 1 or names[simulation[0]] != PARENT_NAME or fids[simulation[0]] != PARENT_FID:
        raise ValueError("expected exactly one original simulation parent filename/FID")
    return PARENT_FID


def _pnfs_path(metadata, name):
    paths = set()
    for location in metadata.get("locations", []):
        full = location.get("full_path")
        if not isinstance(full, str):
            raise ValueError("location missing full_path")
        if full.startswith("enstore:"):
            clean = re.sub(r"\s*\([^()\r\n]*\)$", "", full)
            if re.fullmatch(r"enstore:/[^\s()]+", clean) is None:
                raise ValueError("invalid enstore PNFS path")
            path = clean[len("enstore:"):].rstrip("/")
            if ".." in Path(path).parts:
                raise ValueError("PNFS path filename mismatch")
            if Path(path).name != name:
                if Path(path).suffix:
                    raise ValueError("PNFS path filename mismatch")
                path += "/" + name
            paths.add(path)
    if len(paths) != 1:
        raise ValueError("missing or ambiguous PNFS path")
    return paths.pop()


def validate_metadata(metadata, frozen):
    """Match SAM identity to a frozen role row; no file-content access."""
    _identity(metadata, frozen["name"], frozen["fid"])
    if frozen["did"] != "mu2e:" + frozen["name"]:
        raise ValueError("frozen DID mismatch")
    size = _positive(metadata.get("file_size"), "catalog size")
    if size != frozen["size"]:
        raise ValueError("catalog size mismatch")
    checksums = _checksums(metadata.get("checksum", metadata.get("checksums")))
    if checksums != frozen["checksums"]:
        raise ValueError("catalog checksum mismatch")
    if "checksum" in metadata and "checksums" in metadata and _checksums(metadata["checksums"]) != checksums:
        raise ValueError("ambiguous catalog checksums")
    for field in ("run", "subrun"):
        for end in ("first", "last"):
            key = f"rs.{end}_{field}"
            if key in metadata and (type(metadata[key]) is not int or metadata[key] != frozen[field]):
                raise ValueError("catalog run/subrun mismatch")
    sha = _digest(metadata.get("dh.sha256"))
    if "sha256" in checksums and checksums["sha256"] != sha:
        raise ValueError("catalog SHA256 mismatch")
    return dict(frozen, path=_pnfs_path(metadata, frozen["name"]),
                event_count=_positive(metadata.get("event_count"), "catalog event_count"),
                sha256=sha, parent_fid=_parent(metadata), parent_name=PARENT_NAME)


def _source_hashes():
    return {module.__name__: file_digest(module.__file__) for module in
            (artsteps, source_event_check, source_events, source_split)} | {__name__: file_digest(__file__)}


def _text(report):
    return json.dumps(report, indent=2, allow_nan=False) + "\n"


def _save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(_text(value))


def _sam(name, out):
    # Bound SAM subprocess/socket time; no shell, namespace probes or art reads.
    result = subprocess.run(["samweb", "--socket-timeout", "30", "get-metadata", "--json", "--locations", name],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT, check=True)
    with Path(out).open("xb") as stream:
        stream.write(result.stdout)
    return _json(result.stdout), _sha(result.stdout)


def metadata_inventory(manifest_path, out):
    out, manifest_path = Path(out), Path(manifest_path)
    _new(out)
    blob = manifest_path.read_bytes()
    rows = manifest_rows(_json(blob))
    out.mkdir(parents=True)
    metadata, parent_hash = _sam(PARENT_NAME, out / "parent.metadata.json")
    parent = dict(parent_identity(metadata), metadata_filename="parent.metadata.json", metadata_sha256=parent_hash)
    audited = []
    for row in rows:
        filename = row["name"] + ".metadata.json"
        metadata, sha = _sam(row["name"], out / filename)
        audited.append(dict(validate_metadata(metadata, row), metadata_filename=filename, metadata_sha256=sha))
    if file_digest(manifest_path) != _sha(blob):
        raise ValueError("manifest changed during metadata audit")
    inventory = dict(schema_version=1, manifest_sha256=_sha(blob), parent_metadata_sha256=parent_hash,
                     original_parent=parent, rows=audited, source_sha256=_source_hashes(),
                     confirmation_event_contents_opened_by_this_operation=False,
                     source_independence_certified=False)
    _save_json(out / "inventory.json", inventory)
    return inventory


def read_event_census(path, expected_count, sha256):
    """Full ordered IDs from the highest Events cycle; never read physics branches."""
    n, sha = _positive(expected_count, "expected_count"), _digest(sha256)
    before = file_digest(path)
    if before != sha:
        raise ValueError("input SHA256 differs from catalog")
    with uproot.open(path, timeout=30) as root:
        cycles = [int(match.group(1)) for key in root.keys(recursive=False, cycle=True)
                  if (match := re.fullmatch(r"Events;([1-9][0-9]*)", key))]
        if not cycles:
            raise ValueError("missing Events tree")
        tree = root[f"Events;{max(cycles)}"]
        if tree.num_entries != n:
            raise ValueError("highest Events cycle count differs from catalog")
        branch = tree["EventAuxiliary"]
        if branch.num_entries != n:
            raise ValueError("EventAuxiliary count differs from catalog")
        ids = artsteps.decode_event_ids(branch)
    if ids.shape != (n, 3) or ids.dtype.kind not in "iu":
        raise ValueError("decoded census shape/integer count mismatch")
    result = {field: ids[:, index] for index, field in enumerate(EVENT)}
    source_event_check._census(result)
    if file_digest(path) != before:
        raise ValueError("art input changed during census")
    return result


# Canonical config_dumper grammar only; includes, references and unknown syntax fail.
_TOKEN = re.compile(r'\s*("(?:[^"\\\r\n]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"|[A-Za-z_][A-Za-z0-9_]*|[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?|[{}\[\]:,])')


def _config_table(text):
    if len(text) > 1024 * 1024:
        raise ValueError("config dump too large")
    tokens, at = [], 0
    while at < len(text.rstrip()):
        match = _TOKEN.match(text, at)
        if match is None:
            raise ValueError("unsupported canonical config dump syntax")
        tokens.append(match.group(1))
        at = match.end()
        if len(tokens) > 20000:
            raise ValueError("config dump too many tokens")
    position, counts = 0, {}

    def table(end=None, depth=0):
        nonlocal position
        if depth > 24:
            raise ValueError("config nesting too deep")
        result = {}
        while position < len(tokens) and tokens[position] != end:
            key = tokens[position]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None or key in result:
                raise ValueError("invalid/duplicate config key")
            counts[key] = counts.get(key, 0) + 1
            position += 1
            if position >= len(tokens) or tokens[position] != ":":
                raise ValueError("missing config colon")
            position += 1
            result[key] = value(depth + 1)
        if end is not None:
            if position >= len(tokens):
                raise ValueError("incomplete config table")
            position += 1
        return result

    def value(depth):
        nonlocal position
        if position >= len(tokens) or depth > 24:
            raise ValueError("incomplete/deep config value")
        token = tokens[position]
        position += 1
        if token == "{":
            return table("}", depth)
        if token == "[":
            items = []
            while position < len(tokens) and tokens[position] != "]":
                items.append(value(depth + 1))
                if position < len(tokens) and tokens[position] == ",":
                    position += 1
                elif position >= len(tokens) or tokens[position] != "]":
                    raise ValueError("unsupported config list separator")
            if position >= len(tokens):
                raise ValueError("incomplete config list")
            position += 1
            return items
        if token.startswith('"'):
            return json.loads(token)
        if token in ("true", "false"):
            return token == "true"
        if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?", token):
            return token  # Unused numeric settings are not interpreted.
        raise ValueError("unsupported unquoted config value")

    result = table()
    return result, counts


def extract_parent_config(text):
    """Accept only the explicit stock single-module, single-file mixer proof."""
    if text.startswith("=============================================\nProcessing file: "):
        prefix = re.match(r"\A=+\nProcessing file: [^\r\n]+\n# Read SQLiteDB from file, total size: [^\r\n]+\n\n", text)
        if prefix is None:
            raise ValueError("unsupported config_dumper header")
        text = text[prefix.end():]
    root, counts = _config_table(text)
    if set(root) != {"TargetStopResampler"}:
        raise ValueError("expected one TargetStopResampler root and no other module configs")
    for field in ("TargetStopResampler", "fileNames", "module_type", "writeEventIDs", "readMode", "wrapFiles"):
        if counts.get(field) != 1:
            raise ValueError(f"missing/ambiguous config {field}")
    module = root["TargetStopResampler"]
    if not isinstance(module, dict) or not isinstance(module.get("mu2e"), dict):
        raise ValueError("expected module and mu2e tables")
    if (module.get("module_type") != "ResamplingMixer" or module["mu2e"].get("writeEventIDs") is not True
            or module.get("readMode") != "sequential" or module.get("wrapFiles") is not True):
        raise ValueError("required explicit mixer flags differ or are missing")
    names = module.get("fileNames")
    if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
        raise ValueError("expected exactly one configured source file")
    uri = urlsplit(names[0])
    if (uri.scheme != "xroot" or not uri.netloc or uri.query or uri.fragment
            or Path(uri.path).name != PARENT_NAME or ".." in Path(uri.path).parts):
        raise ValueError("configured source is not the original parent xroot URI")
    return dict(parent_name=PARENT_NAME, source_file=names[0], read_mode="sequential", wrap_files=True,
                write_event_ids=True, module_type="ResamplingMixer")


def _metadata_blob(base, record):
    filename = record["metadata_filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("metadata filename must be a local basename")
    path = base / filename
    blob = path.read_bytes()
    if _sha(blob) != _digest(record["metadata_sha256"]):
        raise ValueError("raw metadata hash mismatch")
    return path, blob


def _inventory_item(inventory, item):
    rows = inventory["rows"]
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("inventory must contain all twenty authorized files")
    grouped = {role: [row for row in rows if row["role"] == role] for role in ROLES}
    frozen = manifest_rows(dict(roles=dict(grouped, confirmation=[])))
    for row, normalized in zip(rows, frozen):
        if any(row.get(key) != value for key, value in normalized.items()):
            raise ValueError("inventory role/order/identity mismatch")
    selected = [row for row in rows if row["name"] == item]
    if len(selected) != 1:
        raise ValueError("item absent/ambiguous or confirmation is not authorized")
    _digest(inventory["manifest_sha256"])
    return selected[0]


def bind(inventory_path, item, expected_path, printer_path, config_path, *, census_receipt=None):
    """Validate everything in memory before creating any output."""
    paths = dict(inventory=Path(inventory_path), expected_events=Path(expected_path),
                 printer_log=Path(printer_path), config=Path(config_path))
    before = {key: file_digest(path) for key, path in paths.items()}
    blobs = {key: path.read_bytes() for key, path in paths.items()}
    if any(_sha(blobs[key]) != value for key, value in before.items()):
        raise ValueError("input changed before parsing")
    inventory = _json(blobs["inventory"])
    row = _inventory_item(inventory, item)
    parent = inventory["original_parent"]
    if any(parent.get(key) != value for key, value in dict(name=PARENT_NAME, fid=PARENT_FID, did="mu2e:" + PARENT_NAME).items()):
        raise ValueError("inventory original parent identity mismatch")
    if parent["metadata_sha256"] != inventory["parent_metadata_sha256"]:
        raise ValueError("inventory parent metadata hash mismatch")
    for key, record in (("metadata", row), ("parent_metadata", parent)):
        path, blob = _metadata_blob(paths["inventory"].parent, record)
        paths[key], before[key] = path, _sha(blob)
        metadata = _json(blob)
        if key == "parent_metadata":
            parent_identity(metadata)
        else:
            frozen = {field: row[field] for field in ("did", "fid", "name", "size", "checksums", "role", "ordinal", "run", "subrun")}
            checked = validate_metadata(metadata, frozen)
            if any(row.get(field) != value for field, value in checked.items()):
                raise ValueError("inventory disagrees with raw metadata")
    with np.load(io.BytesIO(blobs["expected_events"]), allow_pickle=False) as archive:
        if set(archive.files) != set(EVENT) or len(archive.files) != len(EVENT):
            raise ValueError("expected census requires exclusively EVENT integer columns")
        events = source_event_check._census(dict(archive))
    n = len(events["event"])
    if n != row["event_count"] or any(np.any(events[field] != row[field]) for field in ("run", "subrun")):
        raise ValueError("independent census count/run/subrun differs from catalog")
    if census_receipt is not None:
        path = Path(census_receipt)
        blob = path.read_bytes()
        receipt = _json(blob)
        required = dict(input_sha256=row["sha256"], events=n, expected_count=n,
                        ordered_full_event_census=True, output_sha256=before["expected_events"])
        if any(receipt.get(key) != value or type(receipt.get(key)) is not type(value)
               for key, value in required.items()):
            raise ValueError("independent census receipt does not bind catalog input and expected NPZ")
        paths["census_receipt"], before["census_receipt"] = path, _sha(blob)
    config = extract_parent_config(blobs["config"].decode("utf-8"))
    source, printer = source_event_check.parse_printer(blobs["printer_log"].decode("utf-8"), events,
                                                       source_tag=source_event_check.SOURCE_TAG)
    expected = dict(events, output_fid=np.full(n, row["fid"]))
    ledger = dict(expected, parent_fid=np.full(n, parent["fid"]), **source,
                  source_count=np.ones(n, dtype=np.int64), source_present=np.ones(n, bool),
                  source_file_resolved=np.ones(n, bool))
    _, counts = source_split.validate_ledger(ledger, expected)
    for key, path in paths.items():
        if file_digest(path) != before[key]:
            raise ValueError(f"{key} changed during binding")
    report = dict(item=item, role=row["role"], ordinal=row["ordinal"], events=n, source_rows=n,
                  ordered_full_census_exact=True, input_sha256=before, art_sha256=row["sha256"],
                  manifest_sha256=inventory["manifest_sha256"], parent_fid=parent["fid"], parent_name=parent["name"],
                  config=config, printer=printer, ledger=counts, source_sha256=_source_hashes(),
                  data_source="stock EventIDSequencePrinter; independently checked complete ordered census",
                  census_receipt_checked=census_receipt is not None,
                  source_independence_certified=False,
                  scope="immediate original-file source EventIDs only; no stopped-muon identity or independence inference")
    return expected, ledger, report


def rebind(inventory_path, evidence, out, file0_log, file0_config):
    """Revalidate saved metadata/census/printer evidence without rereading art."""
    inventory_path, evidence, out = Path(inventory_path), Path(evidence), Path(out)
    _new(out)
    inventory = _json(inventory_path.read_bytes())
    _inventory_item(inventory, inventory["rows"][0]["name"])
    out.mkdir(parents=True)
    for row in inventory["rows"]:
        src = evidence / row["name"]
        if row["subrun"] == 0:
            log, config = Path(file0_log), Path(file0_config)
        else:
            log, config = src / "printer.log", src / "source_config.txt"
        main(["bind", "--inventory", str(inventory_path), "--item", row["name"],
              "--expected-events", str(src / "expected_events.npz"), "--census-receipt", str(src / "census.json"),
              "--printer-log", str(log), "--config", str(config), "--out", str(out / row["name"] / "bound")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("rebind")
    for name in ("inventory", "evidence", "file0-log", "file0-config", "out"):
        replay.add_argument("--" + name, type=Path, required=True)
    metadata = commands.add_parser("metadata")
    metadata.add_argument("--manifest", type=Path, required=True)
    census = commands.add_parser("census")
    census.add_argument("--art", type=Path, required=True)
    census.add_argument("--expected-count", type=int, required=True)
    census.add_argument("--sha256", required=True)
    binding = commands.add_parser("bind")
    binding.add_argument("--inventory", type=Path, required=True)
    binding.add_argument("--item", required=True)
    binding.add_argument("--census-receipt", type=Path)
    listing = commands.add_parser("inventory")
    listing.add_argument("--inventory", type=Path, required=True)
    configuration = commands.add_parser("config")
    configuration.add_argument("--config", type=Path, required=True)
    for name in ("expected-events", "printer-log", "config"):
        binding.add_argument("--" + name, type=Path, required=True)
    for command in (metadata, census, binding):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "rebind":
        rebind(args.inventory, args.evidence, args.out, args.file0_log, args.file0_config)
        return
    if args.command == "config":
        print(_text(extract_parent_config(args.config.read_text())), end="")
        return
    if args.command == "inventory":
        inventory = _json(args.inventory.read_bytes())
        rows = inventory["rows"]
        for row in rows:
            _inventory_item(inventory, row["name"])
            _, blob = _metadata_blob(args.inventory.parent, row)
            checked = validate_metadata(_json(blob), row)
            if any(row.get(key) != value for key, value in checked.items()):
                raise ValueError("inventory does not match saved metadata")
        for row in rows:
            print("\t".join(str(row[k]) for k in ("role", "name", "path", "event_count", "sha256")))
        return
    _new(args.out)
    if args.command == "metadata":
        inventory = metadata_inventory(args.manifest, args.out)
        report = dict(files=len(inventory["rows"]), inventory=str(args.out / "inventory.json"),
                      manifest_sha256=inventory["manifest_sha256"], source_sha256=inventory["source_sha256"],
                      source_independence_certified=False)
    elif args.command == "census":
        if args.out.suffix != ".npz":
            raise ValueError("census output must have .npz suffix")
        events = read_event_census(args.art, args.expected_count, args.sha256)
        report = dict(events=len(events["event"]), expected_count=args.expected_count,
                      ordered_full_event_census=True, input_sha256=args.sha256, source_sha256=_source_hashes())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            np.savez(stream, **events)
        report["output_sha256"] = file_digest(args.out)
    else:
        expected, ledger, report = bind(args.inventory, args.item, args.expected_events, args.printer_log,
                                       args.config, census_receipt=args.census_receipt)
        args.out.mkdir(parents=True)
        for name, table in (("expected.npz", expected), ("ledger.npz", ledger)):
            with (args.out / name).open("xb") as stream:
                np.savez(stream, **table)
        report["output_sha256"] = {name: file_digest(args.out / name) for name in ("expected.npz", "ledger.npz")}
        _save_json(args.out / "summary.json", report)
    print(_text(report), end="")


if __name__ == "__main__":
    main()
