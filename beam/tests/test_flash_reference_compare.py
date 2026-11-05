"""Whole-export comparisons test payload identity, never extraction fidelity."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import uproot

from beamfastsim.flash_reference_compare import compare_exports, main
from beamfastsim.flash_reference_export import REQUIRED, ReferenceExport, audit_export, read_export


def _payload() -> ReferenceExport:
    return ReferenceExport({
        table: {
            column: np.array(["α reference", "β reference"], dtype=object)
            if dtype == "string" else np.zeros(2, dtype=dtype)
            for column, dtype in contract.items()
        }
        for table, contract in REQUIRED.items()
    })


def _write(path: Path, export: ReferenceExport, directory: str = "export") -> Path:
    with uproot.recreate(path) as root:
        for table, contract in REQUIRED.items():
            name = f"{directory}/{table}" if directory else table
            root.mktree(name, contract)
            root[name].extend(export[table])
    return path


def _codes(report: dict) -> set[tuple[str, str, str]]:
    return {(d["code"], d.get("table", ""), d.get("column", "")) for d in report["differences"]}


def test_identical_invalid_is_explicitly_not_a_physics_certificate() -> None:
    left, right = _payload(), _payload()
    left["statuses"]["status"][:] = right["statuses"]["status"][:] = 9
    assert not audit_export(left)["physics_checks_passed"]
    report = compare_exports(left, right)
    assert report["identical"]
    assert report["differences"] == []
    assert report["physics_completeness"] is False
    assert report["validity"] == "not_assessed"
    assert "both be invalid" in report["scope"]
    assert "No physics audit" in report["scope"]
    assert "no physics" in report["scope"].lower()
    assert "independent extraction certificate" in report["scope"]
    assert "physics_checks_passed" not in report


@pytest.mark.parametrize("table,column,dtype", [
    (table, column, dtype) for table, contract in REQUIRED.items() for column, dtype in contract.items()
])
def test_every_required_column_is_compared(table: str, column: str, dtype: str) -> None:
    left, right = _payload(), _payload()
    right[table][column][1] = "mutated" if dtype == "string" else 1
    report = compare_exports(left, right)
    assert not report["identical"]
    code = "string_values" if dtype == "string" else "numeric_bytes"
    assert _codes(report) == {(code, table, column)}


@pytest.mark.parametrize("table,column", [("steps", "px"), ("particles", "px")])
def test_signed_zero_is_not_equal(table: str, column: str) -> None:
    left, right = _payload(), _payload()
    right[table][column][0] = -0.0
    assert np.array_equal(left[table][column], right[table][column])
    assert _codes(compare_exports(left, right)) == {("numeric_bytes", table, column)}


@pytest.mark.parametrize("table,column,unsigned,bits", [
    ("steps", "time", "uint64", 0x7FF8000000000001),
    ("particles", "time", "uint32", 0x7FC00001),
])
def test_nan_payload_bits_are_exact(table: str, column: str, unsigned: str, bits: int) -> None:
    left, right = _payload(), _payload()
    left[table][column].view(unsigned)[0] = bits
    right[table][column].view(unsigned)[0] = bits
    assert compare_exports(left, right)["identical"]
    right[table][column].view(unsigned)[0] = bits + 1
    assert np.isnan(left[table][column][0]) and np.isnan(right[table][column][0])
    assert _codes(compare_exports(left, right)) == {("numeric_bytes", table, column)}


def test_single_float_bit_is_not_tolerated() -> None:
    left, right = _payload(), _payload()
    left["steps"]["time"][:] = right["steps"]["time"][:] = 1.0
    right["steps"]["time"].view("uint64")[1] ^= np.uint64(1)
    assert np.allclose(left["steps"]["time"], right["steps"]["time"])
    assert not compare_exports(left, right)["identical"]


@pytest.mark.parametrize("dtype", ["float32", ">f8", "int64"])
def test_dtype_is_exact(dtype: str) -> None:
    left, right = _payload(), _payload()
    right["steps"]["time"] = right["steps"]["time"].astype(dtype)
    assert _codes(compare_exports(left, right)) == {("dtype", "steps", "time")}


def test_identical_noncanonical_dtype_is_not_a_validity_audit() -> None:
    left, right = _payload(), _payload()
    left["steps"]["time"] = left["steps"]["time"].astype("float32")
    right["steps"]["time"] = right["steps"]["time"].astype("float32")
    assert compare_exports(left, right)["identical"]
    assert compare_exports(left, right)["validity"] == "not_assessed"


@pytest.mark.parametrize("shape", [(1, 2), (1,), (0,)])
def test_shape_is_exact(shape: tuple[int, ...]) -> None:
    left, right = _payload(), _payload()
    right["events"]["ordinal"] = np.zeros(shape, dtype="uint64")
    assert _codes(compare_exports(left, right)) == {("shape", "events", "ordinal")}


@pytest.mark.parametrize("table", list(REQUIRED))
def test_row_order_is_never_sorted(table: str) -> None:
    left, right = _payload(), _payload()
    column = next(iter(REQUIRED[table]))
    dtype = REQUIRED[table][column]
    values = np.array(["first", "second"], dtype=object) if dtype == "string" else np.array([10, 20], dtype=dtype)
    left[table][column] = values
    right[table][column] = values[::-1].copy()
    assert not compare_exports(left, right)["identical"]


def test_strings_compare_values_not_object_addresses() -> None:
    left, right = _payload(), _payload()
    right["products"]["branch"] = np.array([s.encode().decode() for s in left["products"]["branch"]], dtype=object)
    assert left["products"]["branch"][0] is not right["products"]["branch"][0]
    assert compare_exports(left, right)["identical"]
    right["products"]["branch"][1] = "β reference "
    assert _codes(compare_exports(left, right)) == {("string_values", "products", "branch")}


@pytest.mark.parametrize("left_dtype,right_dtype", [("U48", "U47"), (object, "U47"), ("U47", object)])
def test_string_storage_is_not_root_string_type(left_dtype, right_dtype) -> None:
    # ROOT std::string has no NumPy width; uproot sizes Unicode from values.
    # A full array's subset can retain U48 while a separately read slice is U47.
    left, right = _payload(), _payload()
    values = ["α" * 47, "second"]
    left["volumes"]["name"] = np.array(values + ["β" * 48], dtype=left_dtype)[:2]
    right["volumes"]["name"] = np.array(values, dtype=right_dtype)
    assert compare_exports(left, right)["identical"]


@pytest.mark.parametrize("before,after", [
    ("α", "β"), ("α", "α0"), ("α", "α "),
    ("α\0β", "αβ"), ("α", "α\0"), ("", "\0"),
])
def test_string_values_are_exact_including_nul(before, after) -> None:
    left, right = _payload(), _payload()
    # Object strings preserve even trailing NUL, unlike NumPy U padding.
    left["volumes"]["name"] = np.array([before, ""], dtype=object)
    right["volumes"]["name"] = np.array([after, ""], dtype=object)
    assert _codes(compare_exports(left, right)) == {("string_values", "volumes", "name")}


def test_unicode_embedded_nul_is_not_stripped() -> None:
    left, right = _payload(), _payload()
    left["volumes"]["name"] = np.array(["α\0β", ""], dtype="U48")
    right["volumes"]["name"] = np.array(["αβ", ""], dtype="U47")
    assert _codes(compare_exports(left, right)) == {("string_values", "volumes", "name")}
    right["volumes"]["name"] = np.array(["α\0β", ""], dtype=object)
    assert compare_exports(left, right)["identical"]


@pytest.mark.parametrize("values", [[], ["", ""]])
def test_empty_strings_and_arrays_ignore_unicode_width(values) -> None:
    left, right = _payload(), _payload()
    left["volumes"]["name"] = np.array(values, dtype="U48")
    right["volumes"]["name"] = np.array(values, dtype="U47")
    assert compare_exports(left, right)["identical"]
    right["volumes"]["name"] = np.array(values, dtype=object)
    assert compare_exports(left, right)["identical"]


@pytest.mark.parametrize("bad", [1, None, b"first", ["first"], {"first": 1}])
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_invalid_object_string_elements_are_rejected(bad, side) -> None:
    left, right = _payload(), _payload()
    for name, export in (("left", left), ("right", right)):
        if side in (name, "both"):
            export["volumes"]["name"][0] = bad
    assert _codes(compare_exports(left, right)) == {("unsupported_string_values", "volumes", "name")}


@pytest.mark.parametrize("dtype", ["int32", "float64", "bool"])
@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_nonstring_dtype_is_rejected(dtype, side) -> None:
    left, right = _payload(), _payload()
    for name, export in (("left", left), ("right", right)):
        if side in (name, "both"):
            export["volumes"]["name"] = np.zeros(2, dtype=dtype)
    assert _codes(compare_exports(left, right)) == {("unsupported_string_dtype", "volumes", "name")}


def test_bytes_are_exact_without_speculative_decoding() -> None:
    left, right = _payload(), _payload()
    left["volumes"]["name"] = np.array([b"\xff\0x", b""], dtype="S48")
    right["volumes"]["name"] = np.array([b"\xff\0x", b""], dtype="S47")
    assert compare_exports(left, right)["identical"]
    right["volumes"]["name"][0] = b"\xffx"
    assert _codes(compare_exports(left, right)) == {("string_values", "volumes", "name")}
    left["volumes"]["name"] = np.array(["first", ""], dtype="S48")
    right["volumes"]["name"] = np.array(["first", ""], dtype="U47")
    assert _codes(compare_exports(left, right)) == {("string_representation", "volumes", "name")}


def test_noncontiguous_storage_is_not_payload() -> None:
    left, right = _payload(), _payload()
    right["steps"]["time"] = np.zeros(4, dtype="float64")[::2]
    assert not right["steps"]["time"].flags.c_contiguous
    assert compare_exports(left, right)["identical"]


def test_missing_required_fields_report_both_sides() -> None:
    left, right = _payload(), _payload()
    del left.tables["volumes"]
    del right.tables["volumes"]
    del right["products"]["process"]
    report = compare_exports(left, right)
    assert not report["identical"]
    assert {d["side"] for d in report["differences"] if d["code"] == "missing_table"} == {"left", "right"}
    assert ("missing_column", "products", "process") in _codes(report)


def test_empty_tables_and_excluded_extra_fields() -> None:
    left, right = _payload(), _payload()
    for export in (left, right):
        for columns in export.tables.values():
            for column in columns:
                columns[column] = columns[column][:0]
    right.tables["extra"] = {"value": np.array([42])}
    right["steps"]["extra"] = np.array([99])
    assert compare_exports(left, right)["identical"]


def test_paths_mixed_inputs_directory_and_metadata(tmp_path: Path) -> None:
    left, right = _payload(), _payload()
    a = _write(tmp_path / "a.root", left, "reference")
    b = _write(tmp_path / "b.root", right, "reference")
    assert compare_exports(a, b, directory="reference")["identical"]
    assert compare_exports(a, read_export(b, "reference"), directory="reference")["identical"]
    right["completion"]["complete"][0] = 1
    b = _write(tmp_path / "b.root", right, "reference")
    assert _codes(compare_exports(a, b, directory="reference")) == {("numeric_bytes", "completion", "complete")}
    report = compare_exports(a, b)
    assert not report["identical"]
    assert {d["side"] for d in report["differences"]} == {"left", "right"}


def test_cli_success_failure_and_exclusive_create(tmp_path: Path) -> None:
    left, right = _payload(), _payload()
    a = _write(tmp_path / "a.root", left)
    b = _write(tmp_path / "b.root", right)
    out = tmp_path / "report.json"
    args = ["--left", str(a), "--right", str(b), "--out", str(out)]
    assert main(args) == 0
    original = out.read_bytes()
    assert json.loads(original)["identical"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert out.read_bytes() == original
    right["statuses"]["status"][0] = 1
    _write(b, right)
    assert main(args[:-1] + [str(tmp_path / "different.json")]) == 2
    assert main(["--left", str(tmp_path / "missing.root"), "--right", str(b), "--out", str(tmp_path / "unreadable.json")]) == 2
    assert json.loads((tmp_path / "unreadable.json").read_text())["differences"][0]["code"] == "read_error"


def test_cli_module_exit_status(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.root", _payload())
    out = tmp_path / "module.json"
    command = [sys.executable, "-B", "-m", "beamfastsim.flash_reference_compare", "--left", str(a), "--right", str(a), "--out", str(out)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["physics_completeness"] is False
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 2
    assert "cannot create report" in result.stderr
