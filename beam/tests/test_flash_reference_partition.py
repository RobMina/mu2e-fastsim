"""Partition payload tests are synthetic, not independent art extraction evidence."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from beamfastsim.flash_reference_export import REQUIRED, ReferenceExport, audit_export, read_export
from beamfastsim.flash_reference_partition import check_partition, main
from test_flash_reference_export import _fixture, _write

_EVENT_TABLES = ("events", "steps", "particles", "daughters", "generators", "statuses")


def _full(count=2, *, separate_subruns=False):
    assert count % 2 == 0
    tables = _fixture()
    for table in (*_EVENT_TABLES, "products"):
        rows = tables[table]
        keep = rows["scope"] == 0 if table == "products" else slice(None)
        source = {k: a[keep] for k, a in rows.items()}
        length = len(source["run"])
        expanded = {k: np.tile(a, count // 2) for k, a in source.items()}
        offsets = np.repeat(np.arange(count // 2, dtype=np.uint64) * 2, length)
        expanded["ordinal"] += offsets
        expanded["event"] += offsets.astype(np.uint32)
        if separate_subruns:
            expanded["run"][:] = 1 + expanded["ordinal"] // 4
            expanded["subrun"][:] = 2 + expanded["ordinal"] // 2
        tables[table] = expanded
    subruns = np.unique(np.stack((tables["events"]["run"], tables["events"]["subrun"]), axis=1), axis=0)
    original = _fixture()
    for table in ("products", "stages", "volumes"):
        rows = original[table]
        keep = rows["scope"] == 1 if table == "products" else slice(None)
        source = {k: a[keep] for k, a in rows.items()}
        expanded = {k: np.tile(a, len(subruns)) for k, a in source.items()}
        expanded["run"][:] = np.repeat(subruns[:, 0], len(source["run"]))
        expanded["subrun"][:] = np.repeat(subruns[:, 1], len(source["run"]))
        tables[table] = ({k: np.concatenate((tables[table][k], a)) for k, a in expanded.items()}
                         if table == "products" else expanded)
    tables["requirements"]["expected_events"][:] = count
    tables["completion"]["events"][:] = count
    tables["completion"]["subrun_products"][:] = len(subruns)
    return ReferenceExport(tables)


def _part(full, start, stop):
    tables = deepcopy(full.tables)
    events = full["events"]
    selected = (events["ordinal"] >= start) & (events["ordinal"] < stop)
    subruns = np.stack((events["run"][selected], events["subrun"][selected]), axis=1)
    for table in (*_EVENT_TABLES, "products", "stages", "volumes"):
        rows = full[table]
        event_keep = (rows["ordinal"] >= start) & (rows["ordinal"] < stop)
        pair = np.stack((rows["run"], rows["subrun"]), axis=1)
        subrun_keep = np.any(np.all(pair[:, None, :] == subruns[None, :, :], axis=2), axis=1)
        if table == "products":
            keep = ((rows["scope"] == 0) & event_keep) | ((rows["scope"] == 1) & subrun_keep)
        else:
            keep = subrun_keep if table in ("stages", "volumes") else event_keep
        tables[table] = {k: a[keep].copy() for k, a in rows.items()}
    for table in ("requirements", "completion"):
        tables[table]["ordinal_base"][:] = start
    tables["requirements"]["expected_events"][:] = stop - start
    tables["completion"]["events"][:] = stop - start
    tables["completion"]["subrun_products"][:] = np.count_nonzero(tables["products"]["scope"] == 1)
    return ReferenceExport(tables)


def _triple(count=2, first=1, **kwargs):
    full = _full(count, **kwargs)
    return full, _part(full, 0, first), _part(full, first, count)


def _check(triple, **kwargs):
    return check_partition(*triple, first_events=1, total_events=2, **kwargs)


def _codes(report):
    return {f["code"] for f in report["failures"]}


def test_valid_shared_subrun_empty_event_and_report_scope():
    triple = _triple()
    report = _check(triple)
    assert report["passed"], report["failures"]
    assert report["physics_completeness"] is False
    assert "payload" in report["scope"] and "assignment" in report["scope"]
    assert "not a physics audit" in report["scope"]
    assert set(report["audits"]) == {"full", "first", "remainder"}
    assert all(a["table_structure_valid"] for a in report["audits"].values())
    assert len(triple[2]["steps"]["run"]) == 0
    assert len(triple[1]["volumes"]["run"]) == len(triple[2]["volumes"]["run"]) == 2


def test_default_thousand_events_and_multirun_subrun_restriction():
    triple = _triple(1000, 10, separate_subruns=True)
    assert check_partition(*triple)["passed"]
    assert len(triple[1]["stages"]["run"]) < len(triple[0]["stages"]["run"])
    assert not check_partition(triple[0], triple[1], triple[0])["passed"]


@pytest.mark.parametrize("side", [0, 1, 2])
def test_each_input_is_audited(side):
    triple = _triple()
    triple[side]["events"]["nsteps"][0] += 1
    report = _check(triple)
    assert "invalid_table_structure" in _codes(report)
    assert set(report["audits"]) == {"full", "first", "remainder"}


@pytest.mark.parametrize("table,column,dtype", [
    (table, column, dtype) for table, contract in REQUIRED.items()
    for column, dtype in contract.items()
])
def test_every_required_column_is_checked(table, column, dtype):
    triple = _triple()
    target = triple[1][table][column]
    target[0] = "changed" if dtype == "string" else target[0] + 1
    assert not _check(triple)["passed"], (table, column)


def test_overlap_and_wrong_skip_even_with_valid_individual_audits():
    full, first, remainder = _triple(4, 2)
    duplicate = _part(full, 0, 2)
    for table in (*_EVENT_TABLES, "products"):
        rows = duplicate[table]
        keep = rows["scope"] == 0 if table == "products" else slice(None)
        rows["ordinal"][keep] += 2
    duplicate["requirements"]["ordinal_base"][:] = 2
    duplicate["completion"]["ordinal_base"][:] = 2
    assert audit_export(duplicate, require_status=False)["table_structure_valid"]
    report = check_partition(full, first, duplicate, first_events=2, total_events=4)
    assert {"partition_overlap", "ordered_event_assignment"} <= _codes(report)
    for table in (*_EVENT_TABLES, "products"):
        rows = remainder[table]
        keep = rows["scope"] == 0 if table == "products" else slice(None)
        rows["ordinal"][keep] -= 1
    remainder["completion"]["ordinal_base"][:] = 1
    remainder["requirements"]["ordinal_base"][:] = 1
    assert audit_export(remainder, require_status=False)["table_structure_valid"]
    assert "ordinal_assignment" in _codes(check_partition(full, first, remainder, first_events=2, total_events=4))


@pytest.mark.parametrize("table,column,unsigned", [
    ("steps", "time", "uint64"), ("particles", "px", "uint32"),
    ("generators", "time", "uint64"), ("statuses", "cpu", "uint32"),
])
def test_source_payload_single_bits_and_signed_zero(table, column, unsigned):
    triple = _triple()
    triple[1][table][column].view(unsigned)[0] ^= np.array(1, dtype=unsigned)
    report = _check(triple)
    assert "numeric_bytes" in _codes(report)
    triple = _triple()
    triple[0][table][column][0] = 0.0
    triple[1][table][column][0] = -0.0
    assert "numeric_bytes" in _codes(_check(triple))


def test_nan_payload_bits_exact():
    triple = _triple()
    for export in triple[:2]:
        export["steps"]["time"].view("uint64")[0] = 0x7FF8000000000001
    assert _check(triple)["passed"]
    triple[1]["steps"]["time"].view("uint64")[0] += np.uint64(1)
    assert "numeric_bytes" in _codes(_check(triple))


@pytest.mark.parametrize("side", [0, 1, 2])
@pytest.mark.parametrize("state", [0, 2, 3])
def test_each_input_rejects_null_unavailable_references(side, state):
    triple = _triple(4, 2)
    triple[side]["steps"]["particle_state"][0] = state
    triple[side]["events"]["ptr_failures"][0] = int(state >= 2)
    report = check_partition(*triple, first_events=2, total_events=4)
    assert "invalid_reference_state" in _codes(report)


@pytest.mark.parametrize("state", [0, 2, 3])
def test_identical_payloads_still_require_valid_references(state):
    triple = _triple()
    for export in triple[:2]:
        export["steps"]["particle_state"][0] = state
        export["events"]["ptr_failures"][0] = int(state >= 2)
    assert "invalid_reference_state" in _codes(_check(triple))


def test_valid_but_changed_ptr_and_retained_null_sentinel():
    triple = _triple()
    triple[1]["steps"]["particle_product"][0] = 11
    assert audit_export(triple[1], require_status=False)["table_structure_valid"]
    assert not _check(triple)["passed"]
    triple = _triple()
    triple[1]["particles"]["parent_key"][0] = np.iinfo(np.uint64).max
    assert audit_export(triple[1], require_status=False)["table_structure_valid"]
    assert not _check(triple)["passed"]


def test_failed_status_is_retained_not_a_physics_gate():
    triple = _triple()
    for export in (triple[0], triple[1]):
        export["statuses"]["status"][0] = 9
    report = _check(triple)
    assert report["passed"]
    assert not report["audits"]["full"]["physics_checks_passed"]
    assert report["physics_completeness"] is False


def test_full_may_have_all_event_collections_empty_and_no_status():
    full = _full()
    for table in ("steps", "particles", "daughters", "generators", "statuses"):
        full.tables[table] = {k: a[:0] for k, a in full[table].items()}
        full["events"]["n" + table][:] = 0
    products = full["products"]
    keep = products["kind"] != 4
    full.tables["products"] = {k: a[keep] for k, a in products.items()}
    full["products"]["size"][full["products"]["scope"] == 0] = 0
    full["events"]["nproducts"] -= 1
    requirements = full["requirements"]
    full.tables["requirements"] = {k: a[requirements["kind"] != 4] for k, a in requirements.items()}
    assert _check((full, _part(full, 0, 1), _part(full, 1, 2)))["passed"]


def test_subrun_payload_not_event_sliced_and_empty_volume_stages():
    triple = _triple()
    for export in triple:
        export.tables["volumes"] = {k: a[:0] for k, a in export["volumes"].items()}
        export["stages"]["size"][:] = 0
    assert _check(triple)["passed"]
    triple[2]["stages"]["stage"][1] = 9
    assert not _check(triple)["passed"]


@pytest.mark.parametrize("table", ["schema", "storage", "requirements", "products", "volumes"])
def test_stored_order_is_not_sorted(table):
    triple = _triple()
    if table == "schema":
        triple[1][table]["product_id_bytes"][:] = 4
    else:
        triple[1].tables[table] = {k: a[::-1].copy() for k, a in triple[1][table].items()}
    assert not _check(triple)["passed"]


@pytest.mark.parametrize("mutation", ["dtype", "missing", "shape", "alignment", "missing_table"])
def test_bad_in_memory_schema_is_reported(mutation):
    triple = _triple()
    rows = triple[1]["steps"]
    if mutation == "dtype":
        rows["time"] = rows["time"].astype("float32")
    elif mutation == "missing":
        del rows["time"]
    elif mutation == "shape":
        rows["time"] = rows["time"].reshape(1, -1)
    elif mutation == "alignment":
        rows["time"] = rows["time"][:0]
    else:
        del triple[1].tables["steps"]
    assert "read_or_schema_error" in _codes(_check(triple))


@pytest.mark.parametrize("first,total", [(0, 2), (2, 2), (3, 2), (True, 2), (1, 2.5), (1, 2**64 + 1)])
def test_invalid_configuration(first, total):
    with pytest.raises(ValueError):
        check_partition(*_triple(), first_events=first, total_events=total)


def test_paths_mixed_inputs_and_highest_cycle(tmp_path):
    triple = _triple()
    path = _write(tmp_path / "full.root", triple[0].tables, cycles=True)
    first = _write(tmp_path / "first.root", triple[1].tables)
    rest = _write(tmp_path / "rest.root", triple[2].tables)
    assert _check((path, read_export(first), rest))["passed"]
    assert not _check((path, first, rest), directory="missing")["passed"]
    assert "read_or_schema_error" in _codes(_check((tmp_path / "absent.root", first, rest)))


def test_cli_default_success_failure_and_exclusive_create(tmp_path):
    triple = _triple(1000, 10)
    paths = [_write(tmp_path / (name + ".root"), export.tables)
             for name, export in zip(("full", "first", "remainder"), triple)]
    out = tmp_path / "report.json"
    args = ["--full", str(paths[0]), "--first", str(paths[1]), "--remainder", str(paths[2]), "--out", str(out)]
    assert main(args) == 0
    original = out.read_bytes()
    assert json.loads(original)["passed"]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert out.read_bytes() == original
    triple[1]["steps"]["time"][0] += 1
    _write(paths[1], triple[1].tables)
    assert main(args[:-1] + [str(tmp_path / "bad.json")]) == 2
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-B", "-m", "beamfastsim.flash_reference_partition",
                             *args[:-1], str(tmp_path / "module.json")],
                            capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 2, result.stderr
    assert json.loads((tmp_path / "module.json").read_text())["physics_completeness"] is False


def test_later_subrun_longer_volume_name_does_not_change_partition(tmp_path):
    full = _full(20, separate_subruns=True)
    full["volumes"]["name"][:] = "α" * 47
    # The longer name belongs only to the remainder's final SubRun.
    full["volumes"]["name"][-1] = "β" * 48
    triple = (full, _part(full, 0, 10), _part(full, 10, 20))
    paths = [_write(tmp_path / (name + ".root"), export.tables)
             for name, export in zip(("full", "first", "remainder"), triple)]
    loaded = tuple(read_export(path) for path in paths)
    assert loaded[0]["volumes"]["name"].dtype == np.dtype("U48")
    assert loaded[1]["volumes"]["name"].dtype == np.dtype("U47")
    assert loaded[2]["volumes"]["name"].dtype == np.dtype("U48")
    report = check_partition(*loaded, first_events=10, total_events=20)
    assert report["passed"], report["failures"]
