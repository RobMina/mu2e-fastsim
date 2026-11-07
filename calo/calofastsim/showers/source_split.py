"""Freeze catalog-only file roles and check explicit upstream-event ledgers."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

ROLES = ("training", "development", "confirmation")
OUTPUT_KEY = ("output_fid", "run", "subrun", "event")
PARENT_KEY = ("parent_fid", "source_run", "source_subrun", "source_event")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog_rows(receipt, query):
    """Require contiguous, complete, mutually consistent pages for one query."""
    pages = []
    for record in receipt["records"]:
        arguments = record["arguments"]
        base = {k: v for k, v in arguments.items() if k not in ("offset", "limit")}
        if record["tool"] == "query_dataset_files" and base == query:
            pages.append(record)
    if not pages:
        raise ValueError("no matching catalog query")
    pages.sort(key=lambda r: r["arguments"].get("offset", 0))
    total = pages[0]["result"]["total_matches"]
    rows = []
    for page in pages:
        result, arguments = page["result"], page["arguments"]
        counts = (result["total_matches"], result["returned"], arguments.get("offset", 0), arguments["limit"])
        if any(type(v) is not int or v < 0 for v in counts) or arguments["limit"] < 1:
            raise ValueError("catalog pagination requires nonnegative integer counts")
        if result["dataset_did"] != query["dataset_did"] or result["total_matches"] != total:
            raise ValueError("catalog page dataset/count changed")
        if arguments.get("offset", 0) != len(rows):
            raise ValueError("catalog page gap or overlap")
        if result["returned"] != len(result["results"]) or result["returned"] > arguments["limit"]:
            raise ValueError("catalog page size mismatch")
        effective = result["applied_filters"]
        for key, value in dict(arguments, offset=arguments.get("offset", 0)).items():
            if key != "dataset_did" and (effective.get(key) != value or type(effective.get(key)) is not type(value)):
                raise ValueError("catalog effective filters disagree with request")
        if result["warnings"]:
            raise ValueError("catalog warnings require review before freezing")
        rows.extend(result["results"])
    if len(rows) != total:
        raise ValueError("incomplete catalog inventory")
    dids = [row["did"] for row in rows]
    fids = [row["fid"] for row in rows]
    if len(set(dids)) != len(dids) or len(set(fids)) != len(fids):
        raise ValueError("duplicate catalog file identity")
    return rows


def freeze(plan, receipt):
    """Validate explicit filename-only roles, never certify source independence."""
    if set(plan["roles"]) != set(ROLES):
        raise ValueError("expected training/development/confirmation roles")
    if plan["catalog_query"]["dataset_did"] != plan["dataset_did"]:
        raise ValueError("plan dataset mismatch")
    rows = catalog_rows(receipt, plan["catalog_query"])
    files = {row["did"]: row for row in rows}
    selected = sum((plan["roles"][role] for role in ROLES), [])
    if not all(plan["roles"][role] for role in ROLES):
        raise ValueError("empty file role")
    if len(set(selected)) != len(selected):
        raise ValueError("file role overlap or duplicate")
    missing = sorted(set(selected) - files.keys())
    if missing:
        raise ValueError(f"required files absent: {missing}")
    namespace, dataset_name = plan["dataset_did"].split(":", 1)
    parts = dataset_name.split(".")
    if len(parts) != 5:
        raise ValueError("invalid dataset naming convention")
    prefix, suffix = ".".join(parts[:4]) + ".", "." + parts[-1]
    for row in rows:
        name = row["name"]
        if (row["did"] != namespace + ":" + name or not name.startswith(prefix) or
                not name.endswith(suffix) or row["namespace"] != namespace):
            raise ValueError("catalog file/dataset identity mismatch")
        sequence = name[len(prefix):-len(suffix)]
        fields = sequence.split("_")
        if len(fields) != 2 or not all(f.isascii() and f.isdigit() for f in fields):
            raise ValueError("invalid file sequencer")
        run, subrun = map(int, fields)
        md = row["metadata"]
        numeric = {"rs.first_run": run, "rs.last_run": run, "rs.first_subrun": subrun, "rs.last_subrun": subrun}
        if (md.get("dh.dataset") != dataset_name or md.get("fn.sequencer") != sequence or
                any(type(md.get(k)) is not int or md[k] != v for k, v in numeric.items())):
            raise ValueError("catalog metadata disagrees with file identity")
        query = plan["catalog_query"]
        for key, value in (("run", run), ("subrun", subrun)):
            lower, upper = query.get(key + "_min"), query.get(key + "_max")
            if (lower is not None and value < lower) or (upper is not None and value > upper):
                raise ValueError("catalog row outside requested run/subrun range")
        if (not isinstance(row["fid"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", row["fid"]) or
                type(row["size"]) is not int or row["size"] <= 0):
            raise ValueError("invalid catalog FID/size")
        patterns = {"adler32": r"[0-9a-fA-F]{8}", "enstore": r"[0-9]+", "md5": r"[0-9a-fA-F]{32}",
                    "sha256": r"[0-9a-fA-F]{64}"}
        if not row["checksums"] or any(k not in patterns or not isinstance(v, str) or
                not re.fullmatch(patterns[k], v) for k, v in row["checksums"].items()):
            raise ValueError("missing or invalid catalog checksum")
    roles = {role: [{k: files[did][k] for k in ("did", "fid", "size", "checksums")}
                    for did in plan["roles"][role]] for role in ROLES}
    return dict(schema_version=1, state="candidate_file_roles_only", dataset_did=plan["dataset_did"],
                selection_rule=plan["selection_rule"], supersedes=plan.get("supersedes"),
                roles=roles, inventory_files=len(rows), unallocated_files=len(rows) - len(selected),
                file_roles_disjoint=True, upstream_event_overlap="not_checked",
                source_independence_certified=False, confirmation_event_contents_opened_by_this_operation=False,
                execution_authorized=False,
                scope="selected production only; catalog counts are not donor counts or availability")


def _columns(table, names):
    values = []
    for name in names:
        value = np.asarray(table[name])
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional column")
        if name.endswith("fid"):
            if value.dtype.kind != "U":
                raise ValueError(f"{name} requires nonempty canonical catalog FID strings")
            if value.size:
                normalized = np.char.replace(np.char.replace(value, "-", ""), "_", "")
                if np.any(~np.char.isalnum(normalized)) or np.any(np.char.encode(value, "ascii", "replace") != np.char.encode(value, "utf-8")):
                    raise ValueError(f"{name} requires nonempty canonical catalog FID strings")
        else:
            if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value > np.iinfo(np.int64).max):
                raise ValueError(f"{name} requires nonnegative signed-range integer identities")
            value = value.astype(np.int64, copy=False)
        values.append(value)
    if len({len(v) for v in values}) != 1:
        raise ValueError("ledger column lengths differ")
    return values


def _keys(table, names):
    return np.rec.fromarrays(_columns(table, names), names=names)


def validate_ledger(table, expected_output):
    """Check one mapped immediate source event per independently enumerated output event.

    FIDs must be resolved upstream by a verified exporter/manifest; this function
    cannot authenticate that provenance or infer an original proton/muon identity.
    """
    output = _keys(table, OUTPUT_KEY)
    expected = _keys(expected_output, OUTPUT_KEY)
    parent = _keys(table, PARENT_KEY)
    if len(output) != len(parent):
        raise ValueError("output/parent lengths differ")
    if len(np.unique(output)) != len(output) or len(np.unique(expected)) != len(expected):
        raise ValueError("duplicate output event identity")
    if not np.array_equal(np.sort(output), np.sort(expected)):
        raise ValueError("ledger does not cover the full expected output event census")
    count = _columns(table, ("source_count",))[0]
    present, resolved = np.asarray(table["source_present"]), np.asarray(table["source_file_resolved"])
    for flag in (present, resolved):
        if flag.dtype.kind != "b" or flag.shape != output.shape:
            raise ValueError("provenance flags must be event-aligned boolean columns")
    if count.shape != output.shape or np.any(count != 1) or not np.all(present & resolved):
        raise ValueError("missing, multiple, or unresolved source-event provenance")
    if not len(output):
        raise ValueError("empty ledger cannot certify a split")
    unique, multiplicity = np.unique(parent, return_counts=True)
    return parent, dict(events=len(output), unique_parent_events=len(unique),
                        reused_parent_events=int(np.sum(multiplicity > 1)),
                        repeated_output_events=int(np.sum(multiplicity - 1)),
                        max_parent_reuse=int(np.max(multiplicity)))


def compare_ledgers(left, right, expected_left, expected_right):
    """Report actual upstream-event intersection; never silently drop collisions."""
    a, sa = validate_ledger(left, expected_left)
    b, sb = validate_ledger(right, expected_right)
    if len(np.intersect1d(_keys(left, OUTPUT_KEY), _keys(right, OUTPUT_KEY))):
        raise ValueError("output event identity overlaps between split ledgers")
    shared_files = np.intersect1d(left["output_fid"], right["output_fid"])
    shared = np.intersect1d(a, b)
    return dict(left=sa, right=sb, shared_output_files=len(shared_files), shared_output_events=0,
                shared_parent_events=len(shared),
                left_events_with_shared_parent=int(np.sum(np.isin(a, shared))),
                right_events_with_shared_parent=int(np.sum(np.isin(b, shared))),
                parent_events_disjoint=not len(shared),
                source_independence_certified=False,
                scope="immediate stop-event identity intersection only, conditional on verified provenance")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    manifest = sub.add_parser("freeze")
    manifest.add_argument("--plan", type=Path, required=True)
    manifest.add_argument("--catalog", type=Path, required=True)
    overlap = sub.add_parser("overlap")
    for name in ("left", "right", "expected-left", "expected-right"):
        overlap.add_argument("--" + name, type=Path, required=True)
    for command in (manifest, overlap):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.mode == "freeze":
        blobs = {"plan": args.plan.read_bytes(), "catalog": args.catalog.read_bytes()}
        report = freeze(json.loads(blobs["plan"]), json.loads(blobs["catalog"]))
    else:
        paths = (args.left, args.right, args.expected_left, args.expected_right)
        blobs = {name: path.read_bytes() for name, path in zip(
            ("left", "right", "expected_left", "expected_right"), paths)}
        data = []
        for content in blobs.values():
            with np.load(io.BytesIO(content), allow_pickle=False) as archive:
                data.append(dict(archive))
        report = compare_ledgers(*data)
    inputs = {name: hashlib.sha256(content).hexdigest() for name, content in blobs.items()}
    report.update(input_sha256=inputs, source_sha256=digest(__file__))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(text)
    print(json.dumps({"out": str(args.out), "mode": args.mode, "source_sha256": report["source_sha256"]}))


if __name__ == "__main__":
    main()
