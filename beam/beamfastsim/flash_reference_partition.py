"""Exact ordered export partition validation, not a physics certificate."""
from __future__ import annotations

import argparse
import json
from os import PathLike
from pathlib import Path
from typing import Sequence

import numpy as np

from .flash_reference_compare import _compare_column
from .flash_reference_export import (
    REQUIRED, ReferenceExport, SchemaError, _keys, _lookup, _positive_integer,
    audit_export, read_export,
)

ExportInput = str | PathLike[str] | ReferenceExport
_EVENT = ("run", "subrun", "event")
_SUBRUN = ("run", "subrun")
_EVENT_TABLES = ("events", "steps", "particles", "daughters", "generators", "statuses")


def _validate(export: ReferenceExport) -> None:
    for table, contract in REQUIRED.items():
        if table not in export.tables:
            raise SchemaError(f"missing table {table}")
        length = None
        for column, dtype in contract.items():
            if column not in export[table]:
                raise SchemaError(f"{table}: missing {column}")
            a = export[table][column]
            if not isinstance(a, np.ndarray) or a.ndim != 1:
                raise SchemaError(f"{table}.{column}: expected scalar vector")
            if length is None:
                length = len(a)
            if len(a) != length:
                raise SchemaError(f"{table}: misaligned columns")
            if dtype == "string":
                if a.dtype.kind not in "UOS" or not all(isinstance(v, (str, bytes)) for v in a):
                    raise SchemaError(f"{table}.{column}: expected strings")
            elif a.dtype != np.dtype(dtype):
                raise SchemaError(f"{table}.{column}: expected {dtype}")


def check_partition(full: ExportInput, first: ExportInput, remainder: ExportInput, *,
                    first_events: int = 10, total_events: int = 1000,
                    directory: str = "export") -> dict:
    """Check complete payloads against ordered full slices, without renumbering.

    Every input receives the structural/reference audit with require_status=False.
    Nonzero StatusG4 remains visible but is not a partition failure. Shared SubRun
    records are repeated in both parts and must match the full SubRun payload.
    No tolerance, provenance aliases, event sorting or survivor selection is used.
    """
    _positive_integer(first_events, "first_events")
    _positive_integer(total_events, "total_events")
    if not first_events < total_events <= 2**64:
        raise ValueError("require 0 < first_events < total_events <= 2**64")
    failures: list[dict] = []
    report = {
        "schema": "flash-reference-partition-v1", "passed": False,
        "physics_completeness": False, "first_events": int(first_events),
        "total_events": int(total_events), "failures": failures, "audits": {},
        "scope": (
            "Exact REQUIRED exported payload and ordered event assignment partition only; "
            "internal structure and reference checks, not a physics audit. Shared SubRun "
            "payloads are repeated consistently. Extra fields and ROOT serialization are "
            "excluded; no runtime, independent extraction or physics certificate."
        ),
    }
    exports = {}
    counts = {"full": total_events, "first": first_events, "remainder": total_events - first_events}
    bases = {"full": 0, "first": 0, "remainder": first_events}
    for side, value in (("full", full), ("first", first), ("remainder", remainder)):
        try:
            export = value if isinstance(value, ReferenceExport) else read_export(value, directory)
            _validate(export)
            exports[side] = export
            audit = audit_export(export, expected_events=counts[side], require_status=False)
            report["audits"][side] = audit
            if not audit["table_structure_valid"]:
                failures.append({"side": side, "code": "invalid_table_structure"})
            if any(f["code"].endswith(("_unavailable", "_null")) for f in audit["failures"]):
                failures.append({"side": side, "code": "invalid_reference_state"})
            if not np.array_equal(export["events"]["ordinal"],
                                  np.arange(counts[side], dtype=np.uint64) + np.uint64(bases[side])):
                failures.append({"side": side, "code": "ordinal_assignment"})
            for table in ("requirements", "completion"):
                if not np.all(export[table]["ordinal_base"] == bases[side]):
                    failures.append({"side": side, "table": table, "code": "ordinal_base"})
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            failures.append({"side": side, "code": "read_or_schema_error", "message": str(exc)})
    if failures:
        return report
    reference = exports["full"]
    first_keys = _keys(exports["first"]["events"], _EVENT)
    rest_keys = _keys(exports["remainder"]["events"], _EVENT)
    if np.any(_lookup(first_keys, rest_keys) >= 0):
        failures.append({"code": "partition_overlap"})
    assigned = np.concatenate((first_keys, rest_keys))
    if not np.array_equal(assigned, _keys(reference["events"], _EVENT)):
        failures.append({"code": "ordered_event_assignment"})
    for side, start, stop in (("first", 0, first_events), ("remainder", first_events, total_events)):
        part = exports[side]
        event_slice = {k: a[start:stop] for k, a in reference["events"].items()}
        event_keys = _keys(event_slice, _EVENT)
        subrun_keys = np.unique(_keys(event_slice, _SUBRUN))
        for table, contract in REQUIRED.items():
            rows = reference[table]
            if table in _EVENT_TABLES:
                keep = _lookup(event_keys, _keys(rows, _EVENT)) >= 0
            elif table == "products":
                event_keep = _lookup(event_keys, _keys(rows, _EVENT)) >= 0
                subrun_keep = _lookup(subrun_keys, _keys(rows, _SUBRUN)) >= 0
                keep = ((rows["scope"] == 0) & event_keep) | ((rows["scope"] == 1) & subrun_keep)
            elif table in ("stages", "volumes"):
                keep = _lookup(subrun_keys, _keys(rows, _SUBRUN)) >= 0
            elif table == "completion":
                continue
            else:
                keep = slice(None)
            for column, dtype in contract.items():
                if table == "requirements" and column in ("expected_events", "ordinal_base"):
                    continue
                differences = _compare_column(table, column, rows[column][keep],
                                              part[table][column], string=dtype == "string")
                failures.extend({**difference, "side": side} for difference in differences)
    report["passed"] = not failures
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Create a new JSON report exclusively; mismatches and I/O failures exit 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full", "first", "remainder", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--directory", default="export")
    args = parser.parse_args(argv)
    report = check_partition(args.full, args.first, args.remainder, directory=args.directory)
    try:
        with Path(args.out).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        parser.exit(2, f"{parser.prog}: cannot create report: {exc}\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
