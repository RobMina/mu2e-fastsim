"""Hash-pinned independent stock source comparison, not full extraction certification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .flash_reference_export import (
    REQUIRED, ReferenceExport, SchemaError, _counts, _keys, _lookup,
    _positive_integer, audit_export, read_export,
)

EVENT = ("run", "subrun", "event")
SOURCE_ROLE = "BeamFilter:Beam"
STATE_FIELDS = (*EVENT, "particle_key", "volume_copy", "pdg", "product_id",
                "x", "p", "time_ns", "input_row", "event_headers", "event_hit_counts")


def _digest(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _read_states(path, expected_hash):
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError("states_sha256 must be an explicit 64-digit SHA256 from the prior certificate")
    with Path(path).open("rb") as stream:
        actual = _digest(stream)
        if actual != expected_hash.lower():
            raise ValueError("states_sha256 mismatch")
        loaded = np.load(stream, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("states must be an NPZ archive")
        with loaded as archive:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError("duplicate NPZ fields")
            missing = set(STATE_FIELDS) - set(archive.files)
            if missing:
                raise ValueError(f"missing NPZ fields: {sorted(missing)}")
            states = {name: archive[name] for name in STATE_FIELDS}
        if _digest(stream) != actual:
            raise ValueError("states changed during comparison read")
    return states, actual


def _validate_states(s):
    headers = s["event_headers"]
    if headers.ndim != 1 or headers.dtype.names != EVENT:
        raise ValueError("event_headers must be a structured run/subrun/event vector")
    n = len(s["particle_key"])
    for name in (*EVENT, "particle_key", "volume_copy", "pdg", "input_row", "event_hit_counts"):
        a = s[name]
        length = len(headers) if name == "event_hit_counts" else n
        if a.shape != (length,) or a.dtype.kind not in "iu":
            raise ValueError(f"{name}: expected aligned integer vector")
        if name != "pdg" and np.any(a < 0):
            raise ValueError(f"{name}: negative identity/count")
    if np.any(s["pdg"] < np.iinfo(np.int32).min) or np.any(s["pdg"] > np.iinfo(np.int32).max):
        raise ValueError("pdg: outside persistent int32 range")
    for name in EVENT:
        a = headers[name]
        if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > np.iinfo(np.uint32).max):
            raise ValueError(f"event_headers.{name}: invalid EventID")
        if np.any(s[name] > np.iinfo(np.uint32).max):
            raise ValueError(f"{name}: invalid EventID")
    if len(np.unique(_keys({k: headers[k] for k in EVENT}, EVENT))) != len(headers):
        raise ValueError("duplicate event_headers")
    counts = s["event_hit_counts"]
    if np.any(counts > n) or int(np.sum(counts, dtype=np.uint64)) != n:
        raise ValueError("event_hit_counts total differs from rows")
    for name in EVENT:
        if not np.array_equal(s[name], np.repeat(headers[name], counts.astype(np.intp))):
            raise ValueError("per-hit EventID/order differs from event_headers/counts")
    if not np.array_equal(s["input_row"], np.arange(n)):
        raise ValueError("input_row must preserve complete original row order")
    for name, shape, dtype in (("x", (n, 3), "float32"), ("p", (n, 3), "float32"),
                               ("time_ns", (n,), "float64")):
        a = s[name]
        if a.shape != shape or a.dtype != np.dtype(dtype) or not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: expected finite {dtype} array of shape {shape}")
    ids = s["product_id"]
    if ids.shape != (n,) or ids.dtype.kind not in "US":
        raise ValueError("product_id: expected aligned non-object string vector")
    if not np.all(np.char.str_len(ids.astype(str)) > 0):
        raise ValueError("product_id: empty string")
    return n


def _validate_export(export):
    for table, contract in REQUIRED.items():
        if table not in export.tables:
            raise SchemaError(f"missing table {table}")
        columns = export[table]
        length = None
        for name, dtype in contract.items():
            if name not in columns:
                raise SchemaError(f"{table}: missing {name}")
            a = columns[name]
            if not isinstance(a, np.ndarray) or a.ndim != 1:
                raise SchemaError(f"{table}.{name}: expected scalar vector")
            if length is None:
                length = len(a)
            if len(a) != length:
                raise SchemaError(f"{table}: misaligned columns")
            if dtype == "string":
                if a.dtype.kind not in "UOS" or not all(isinstance(v, str) for v in a):
                    raise SchemaError(f"{table}.{name}: expected strings")
            elif a.dtype != np.dtype(dtype):
                raise SchemaError(f"{table}.{name}: expected {dtype}")


def check_source(value, states, *, states_sha256, directory="export",
                 expected_events=1000, expected_rows=1130):
    """Compare measured fields only; the caller supplies a trusted prior NPZ digest.

    The small-fixture event/row overrides are API-only. Printed decimal ProductIDs
    are compared numerically without narrowing; colon formats fail unsupported,
    never falling back to event/key-only aliases or a guessed ProductID encoding.
    """
    _positive_integer(expected_events, "expected_events")
    _positive_integer(expected_rows, "expected_rows")
    s, digest = _read_states(states, states_sha256)
    n = _validate_states(s)
    export = value if isinstance(value, ReferenceExport) else read_export(value, directory)
    _validate_export(export)
    audit = audit_export(export, expected_events=expected_events, require_status=False)
    failures = []

    def check(ok, code):
        if not bool(ok):
            failures.append(code)

    report = {
        "schema": "flash-reference-source-check-v1", "passed": False,
        "physics_completeness": False, "full_persistent_extraction_certified": False,
        "states_sha256": digest, "source_role": SOURCE_ROLE,
        "expected_events": int(expected_events), "expected_rows": int(expected_rows),
        "failures": failures, "export_audit": audit,
        "scope": "Independent hash-pinned stock source agreement for measured fields only; no full persistent extraction or physics certificate.",
        "measured_fields": ["ordered EventID/counts including empty events", "ordered particle ProductID/key and volume copy", "joined PDG", "pre-position/momentum converted to float32 bytes", "double global time bytes"],
        "not_covered": ["post-state", "proper time", "deposits", "length", "end process", "full particle/generator extraction", "runtime/build verification", "source-to-primary correspondence", "geometry/crossings"],
    }
    check(audit["table_structure_valid"], "invalid_table_structure")
    check(not any(f["code"].endswith(("_unavailable", "_null")) for f in audit["failures"]),
          "invalid_reference_state")
    if failures:
        return report
    ev, products = export["events"], export["products"]
    check(len(ev["run"]) == len(s["event_headers"]) == expected_events, "event_count")
    check(n == expected_rows, "stock_row_count")
    for name in EVENT:
        check(np.array_equal(ev[name], s["event_headers"][name]), "ordered_event_" + name)
    selected = ((products["scope"] == 0) & (products["kind"] == 3)
                & (products["role"] == SOURCE_ROLE))
    provenance = (products["label"] == "BeamFilter") & (products["instance"] == "Beam")
    check(np.all(provenance[selected]), "source_provenance")
    pi = _lookup(_keys(ev, EVENT), _keys(products, EVENT))
    check(np.all(_counts(pi[selected], len(ev["run"])) == 1), "source_product_selection")
    if failures:
        return report
    source_products = {name: a[selected] for name, a in products.items()}
    order = np.argsort(pi[selected], kind="stable")
    counts = source_products["size"][order]
    check(np.array_equal(counts, s["event_hit_counts"]), "ordered_event_hit_counts")
    steps = export["steps"]
    keep = _lookup(_keys(source_products, (*EVENT, "product")), _keys(steps, (*EVENT, "product"))) >= 0
    rows = {name: a[keep] for name, a in steps.items()}
    check(len(rows["run"]) == n == expected_rows, "source_row_count")
    if len(rows["run"]) != n:
        return report
    for name in EVENT:
        check(np.array_equal(rows[name], s[name]), "ordered_row_" + name)
    indices = np.arange(n, dtype=np.uint64) - np.repeat(np.cumsum(counts) - counts, counts.astype(np.intp))
    check(np.array_equal(rows["index"], indices), "source_collection_order")
    for field, stock in (("particle_key", "particle_key"), ("volume", "volume_copy")):
        check(np.array_equal(rows[field], s[stock].astype(np.uint64)), "source_" + stock)
    printed = s["product_id"].astype(str)
    decimal = np.char.isdecimal(printed) & (np.char.str_len(printed) > 0)
    ascii_digits = np.char.isnumeric(printed) & (np.char.encode(printed, "ascii", errors="replace").astype(str) == printed)
    check(np.all(decimal & ascii_digits), "unsupported_printed_product_id")
    if np.all(decimal & ascii_digits):
        normalized = np.char.lstrip(printed, "0")
        normalized = np.where(normalized == "", "0", normalized)
        check(np.array_equal(normalized, rows["particle_product"].astype(str)), "source_product_id")
    particles = export["particles"]
    found = _lookup(_keys(particles, (*EVENT, "product", "key")),
                    _keys(rows, (*EVENT, "particle_product", "particle_key")))
    check(np.all(rows["particle_state"] == 1) and np.all(found >= 0), "source_particle_resolution")
    if np.all(found >= 0):
        check(np.array_equal(particles["pdg"][found], s["pdg"].astype(np.int32)), "source_pdg")
    for name, fields in (("x", ("x", "y", "z")), ("p", ("px", "py", "pz"))):
        raw = np.column_stack([rows[k] for k in fields])
        with np.errstate(over="ignore", invalid="ignore"):
            converted = raw.astype(np.float32)
        check(np.all(np.isfinite(raw)) and np.all(np.isfinite(converted)), "nonfinite_source_" + name)
        check(converted.tobytes() == s[name].tobytes(), "source_" + name + "_float32_bytes")
    check(np.all(np.isfinite(rows["time"])), "nonfinite_source_time")
    check(rows["time"].tobytes() == s["time_ns"].tobytes(), "source_time_float64_bytes")
    pdg, multiplicity = np.unique(s["pdg"], return_counts=True)
    subruns = np.unique(_keys({k: s["event_headers"][k] for k in EVENT}, ("run", "subrun")))
    report["source_summary"] = {
        "events": len(ev["run"]), "rows": n, "empty_events": int(np.count_nonzero(counts == 0)),
        "subruns": [{"run": int(v["f0"]), "subrun": int(v["f1"])} for v in subruns],
        "pdg_counts": {str(int(k)): int(v) for k, v in zip(pdg, multiplicity)},
        "electron": int(np.count_nonzero(s["pdg"] == 11)),
        "positron": int(np.count_nonzero(s["pdg"] == -11)),
        "mu_minus": int(np.count_nonzero(s["pdg"] == 13)),
        "mu_plus": int(np.count_nonzero(s["pdg"] == -13)),
        "time_ns_min": float(s["time_ns"].min()) if n else None,
        "time_ns_max": float(s["time_ns"].max()) if n else None,
        "time_dtype": "float64", "vector_destination_dtype": "float32",
    }
    report["passed"] = not failures
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "states", "states-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if Path(args.out).resolve() in (Path(args.input).resolve(), Path(args.states).resolve()):
        parser.error("output must be distinct from inputs")
    try:
        report = check_source(args.input, args.states, states_sha256=args.states_sha256)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        report = {"passed": False, "physics_completeness": False,
                  "full_persistent_extraction_certified": False, "failures": [str(exc)]}
    with Path(args.out).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
