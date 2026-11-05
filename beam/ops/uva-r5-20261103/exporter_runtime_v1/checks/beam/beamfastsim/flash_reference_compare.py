"""Exact ordered REQUIRED export payload comparison, not a physics audit.

Numeric columns compare dtype, shape and C-order bytes, including signed zero
and NaN payload bits. Strings compare shape and exact logical values, never
incidental NumPy Unicode widths or object-array addresses. Unicode and object
arrays of str are interchangeable (ROOT std::string has no NumPy width). Byte
arrays compare exact bytes only with other byte arrays; no decoding is attempted. Extra tables/columns and ROOT serialization details are
outside this scope. Identical payloads may both be invalid reference exports.
"""
from __future__ import annotations

import argparse
import json
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

from .flash_reference_export import REQUIRED, ReferenceExport, SchemaError, read_export

ExportInput = str | PathLike[str] | ReferenceExport
Side = Literal["left", "right"]


class Difference(TypedDict, total=False):
    """A required table/column mismatch or an unreadable input."""

    code: str
    table: str
    column: str
    side: Side
    left_dtype: str
    right_dtype: str
    left_shape: list[int]
    right_shape: list[int]
    message: str


class ComparisonReport(TypedDict):
    """JSON-compatible payload result; validity is deliberately not assessed."""

    identical: bool
    differences: list[Difference]
    physics_completeness: Literal[False]
    validity: Literal["not_assessed"]
    scope: str


def _report(differences: list[Difference]) -> ComparisonReport:
    return {
        "identical": not differences,
        "differences": differences,
        "physics_completeness": False,
        "validity": "not_assessed",
        "scope": (
            "Exact ordered REQUIRED table/column payloads only; extra fields and ROOT "
            "serialization are excluded. Identical payloads may both be invalid. "
            "No physics audit, runtime or independent extraction certificate."
        ),
    }


def _compare_column(
    table: str,
    column: str,
    left: NDArray[Any],
    right: NDArray[Any],
    *,
    string: bool,
) -> list[Difference]:
    differences: list[Difference] = []
    if not string and left.dtype != right.dtype:
        differences.append({
            "code": "dtype", "table": table, "column": column,
            "left_dtype": left.dtype.str, "right_dtype": right.dtype.str,
        })
    if left.shape != right.shape:
        differences.append({
            "code": "shape", "table": table, "column": column,
            "left_shape": list(left.shape), "right_shape": list(right.shape),
        })
    if differences:
        return differences
    if string:
        for side, values in (("left", left), ("right", right)):
            if values.dtype.kind not in "UOS":
                return [{"code": "unsupported_string_dtype", "table": table, "column": column, "side": side}]
            if values.dtype.kind == "O" and not all(isinstance(v, str) for v in values.flat):
                return [{"code": "unsupported_string_values", "table": table, "column": column, "side": side}]
        # Refuse mixed byte/Unicode representations rather than guessing an encoding.
        if (left.dtype.kind == "S") != (right.dtype.kind == "S"):
            return [{"code": "string_representation", "table": table, "column": column}]
        equal = bool(np.array_equal(left, right))
        code = "string_values"
    else:
        if left.dtype.kind not in "biufc":
            return [{"code": "unsupported_numeric_dtype", "table": table, "column": column}]
        equal = left.tobytes(order="C") == right.tobytes(order="C")
        code = "numeric_bytes"
    return [] if equal else [{"code": code, "table": table, "column": column}]


def compare_exports(
    left: ExportInput,
    right: ExportInput,
    *,
    directory: str = "export",
) -> ComparisonReport:
    """Compare all required payloads in stored row order, without a validity gate.

    Paths use the standard highest-cycle TTree loader. In-memory exports retain
    their supplied numeric dtypes; even a noncanonical dtype can compare identically.
    String widths/storage are incidental, but logical values remain exact;
    byte strings are never decoded or compared against Unicode/object strings.
    Every required table/column must be present on both sides. Missing or
    unreadable inputs produce structured differences rather than an identical
    result. No event sorting, filtering, tolerance or physics audit is applied.
    """
    differences: list[Difference] = []
    exports: list[ReferenceExport] = []
    sides: tuple[Side, Side] = ("left", "right")
    for side, value in zip(sides, (left, right)):
        try:
            exports.append(value if isinstance(value, ReferenceExport) else read_export(value, directory))
        except (OSError, SchemaError, ValueError, KeyError) as exc:
            differences.append({"code": "read_error", "side": side, "message": str(exc)})
    if differences:
        return _report(differences)
    a, b = exports
    for table, contract in REQUIRED.items():
        missing_table = False
        for side, export in zip(sides, exports):
            if table not in export.tables:
                differences.append({"code": "missing_table", "table": table, "side": side})
                missing_table = True
        if missing_table:
            continue
        for column, dtype in contract.items():
            missing_column = False
            for side, export in zip(sides, exports):
                if column not in export[table]:
                    differences.append({"code": "missing_column", "table": table, "column": column, "side": side})
                    missing_column = True
            if missing_column:
                continue
            differences.extend(_compare_column(table, column, a[table][column], b[table][column], string=dtype == "string"))
    return _report(differences)


def main(argv: Sequence[str] | None = None) -> int:
    """Write a new JSON report; return 2 for differences or I/O failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--directory", default="export")
    parser.add_argument("--out", required=True, help="New JSON file; never overwrite an existing file")
    args = parser.parse_args(argv)
    report = compare_exports(args.left, args.right, directory=args.directory)
    try:
        with Path(args.out).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        parser.exit(2, f"{parser.prog}: cannot create report: {exc}\n")
    return 0 if report["identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
