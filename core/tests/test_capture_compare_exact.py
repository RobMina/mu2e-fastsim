"""Exact saved-capture regressions; synthetic records do not certify transport physics."""
import json

import numpy as np
import pytest
import uproot

from mu2efastsim import capture, capture_compare
from test_capture import BIG, tables  # noqa: F401 -- reuse the shared synthetic capture fixture


def clone(tables):
    return {name: {field: value.copy() for field, value in table.items()} for name, table in tables.items()}


def test_identical_and_reordered(tables):
    candidate = clone(tables)
    for name in capture.TABLES:
        candidate[name] = {field: value[::-1].copy() for field, value in candidate[name].items()}
    for other in (tables, candidate):
        report = capture_compare.compare_exact(tables, other, 2)
        assert report["capture_fields_identical"] and report["diagnostic_only"]
        assert not report["bank_admissible"]
        assert not report["geometry_equivalence_established"]
        assert not report["statistical_equivalence_established"]
        assert report["excluded_fields"] == ["events.cpuTime", "events.realTime"]
        assert len(report["audits"]) == 2
        assert "metrics" not in report


def test_single_event(tables):
    one = {name: {field: value[table["event"] == BIG + 1] for field, value in table.items()}
           for name, table in tables.items()}
    assert capture_compare.compare_exact(one, one, 1)["capture_fields_identical"]
    with pytest.raises(ValueError, match="at least two"):
        capture_compare.compare(one, one, 1)


@pytest.mark.parametrize("identity", capture.EVENT)
def test_full_signed_int64_join(tables, identity):
    # Adjacent IDs that collide as float64, differing in just one triplet component.
    for table in tables.values():
        for field in capture.EVENT:
            table[field][:] = table[field] - 1 if field == identity else BIG
    for field in capture.EVENT:
        tables["events"]["record" + field.capitalize()][:] = tables["events"][field]
    assert np.unique(tables["events"][identity].astype(float)).size == 1
    candidate = clone(tables)
    for name in capture.TABLES:
        candidate[name] = {field: value[::-1].copy() for field, value in candidate[name].items()}
    assert capture_compare.compare_exact(tables, candidate, 2)["capture_fields_identical"]
    # A valid capture with a different triplet must not match merely by row position.
    for table in candidate.values():
        table[identity] += 2
    candidate["events"]["record" + identity.capitalize()] += 2
    capture.audit(candidate, 2)
    with pytest.raises(ValueError, match="unmatched events identity"):
        capture_compare.compare_exact(tables, candidate, 2)


@pytest.mark.parametrize("field", ["cpuTime", "realTime"])
def test_only_timing_values_excluded(tables, field):
    candidate = clone(tables)
    candidate["events"][field] += 100
    assert capture_compare.compare_exact(tables, candidate, 2)["capture_fields_identical"]
    for bad in (np.nan, np.inf, -np.inf):
        candidate["events"][field][0] = bad
        for reference, other in ((tables, candidate), (candidate, tables)):
            with pytest.raises(ValueError, match="nonfinite " + field):
                capture_compare.compare_exact(reference, other, 2)


@pytest.mark.parametrize("name,field,index", [
    ("events", "translation", (0, 0)), ("events", "rotation", (0, 0)),
    ("events", "halfSize", (0, 0)), ("volumes", "translation", (0, 0)),
    ("volumes", "rotation", (0, 0)), ("volumes", "density", 0),
    ("tracks", "begin_energy", 0), ("classifications", "state_mass", 0),
    ("steps", "deposit", 3), ("steps", "nonIonizingDeposit", 3),
    ("steps", "length", 1), ("steps", "post_lz", 1),
])
def test_one_ulp_is_not_a_tolerance(tables, name, field, index):
    candidate = clone(tables)
    value = candidate[name][field][index]
    candidate[name][field][index] = np.nextafter(value, np.inf)
    capture.audit(candidate, 2)  # The audit tolerance must not become comparison tolerance.
    with pytest.raises(ValueError, match=rf"{name}\.{field}: exact field mismatch"):
        capture_compare.compare_exact(tables, candidate, 2)


@pytest.mark.parametrize("name,field,value", [
    ("events", "geometryTag", "different"), ("events", "physicsTag", "different"),
    ("tracks", "creator", "different"), ("tracks", "stopProcess", "different"),
    ("classifications", "status", 1), ("steps", "trackStatus", 1),
    ("steps", "pre_process", "different"), ("volumes", "copy", BIG + 3),
    ("volumes", "logical", "different"),
])
def test_other_audit_valid_fields_are_not_ignored(tables, name, field, value):
    candidate = clone(tables)
    candidate[name][field][0] = value
    capture.audit(candidate, 2)
    with pytest.raises(ValueError, match=rf"{name}\.{field}: exact field mismatch"):
        capture_compare.compare_exact(tables, candidate, 2)


@pytest.mark.parametrize("side", ["reference", "candidate"])
@pytest.mark.parametrize("fault", ["missing_table", "missing_field", "duplicate", "count", "status", "identity"])
def test_both_inputs_audited(tables, side, fault):
    bad = clone(tables)
    if fault == "missing_table":
        del bad["steps"]
    elif fault == "missing_field":
        del bad["steps"]["deposit"]
    elif fault == "duplicate":
        bad["steps"]["index"][1] = 0
    elif fault == "count":
        bad["events"]["nSteps"][0] += 1
    elif fault == "status":
        bad["events"]["complete"][0] = False
    else:
        bad["tracks"]["event"] = bad["tracks"]["event"].astype(float)
    with pytest.raises(ValueError):
        capture_compare.compare_exact(bad if side == "reference" else tables,
                                      bad if side == "candidate" else tables, 2)


@pytest.mark.parametrize("fault", ["value", "missing", "dtype", "shape", "rows", "scalar"])
def test_future_fields_checked(tables, fault):
    for name in capture.TABLES:
        tables[name]["future"] = np.zeros((len(tables[name]["event"]), 2), dtype=np.int64)
    candidate = clone(tables)
    assert capture_compare.compare_exact(tables, candidate, 2)["capture_fields_identical"]
    for name in capture.TABLES:
        changed = clone(candidate)
        if fault == "value":
            changed[name]["future"][0, 0] = 1
        elif fault == "missing":
            del changed[name]["future"]
        elif fault == "dtype":
            changed[name]["future"] = changed[name]["future"].astype(float)
        elif fault == "shape":
            changed[name]["future"] = changed[name]["future"][:, :1]
        elif fault == "rows":
            changed[name]["future"] = changed[name]["future"][:1]
        else:
            changed[name]["future"] = np.array(0)
        with pytest.raises(ValueError, match=name):
            capture_compare.compare_exact(tables, changed, 2)


@pytest.mark.parametrize("field", ["cpuTime", "realTime"])
def test_dtypes_not_excluded(tables, field):
    candidate = clone(tables)
    candidate["events"][field] = candidate["events"][field].astype(np.float32)
    with pytest.raises(ValueError, match="dtype mismatch"):
        capture_compare.compare_exact(tables, candidate, 2)


def test_string_storage_width_is_not_a_dtype_difference(tables):
    candidate = clone(tables)
    candidate["events"]["geometryTag"] = candidate["events"]["geometryTag"].astype("U40")
    report = capture_compare.compare_exact(tables, candidate, 2)
    assert report["capture_fields_identical"] and report["process_class_equated"] == {}
    candidate["events"]["geometryTag"][:] = "other-box"
    with pytest.raises(ValueError, match=r"events\.geometryTag: exact field mismatch"):
        capture_compare.compare_exact(tables, candidate, 2)



@pytest.mark.parametrize("expected", [0, 1, 3])
def test_expected_event_census(tables, expected):
    with pytest.raises(ValueError, match="expected event"):
        capture_compare.compare_exact(tables, tables, expected)


def test_exclusions_are_only_event_timing_values(tables):
    candidate = clone(tables)
    candidate["events"]["cpuTime"] = candidate["events"]["cpuTime"].reshape(-1, 1)
    with pytest.raises(ValueError, match="cpuTime: shape mismatch"):
        capture_compare.compare_exact(tables, candidate, 2)
    tables["tracks"]["cpuTime"] = np.zeros(len(tables["tracks"]["event"]))
    candidate = clone(tables)
    candidate["tracks"]["cpuTime"][0] = 1
    with pytest.raises(ValueError, match="tracks.cpuTime: exact field mismatch"):
        capture_compare.compare_exact(tables, candidate, 2)


def write_root(path, tables):
    with uproot.recreate(path) as root:
        for name, table in tables.items():
            root.mktree("dump/" + name, table)


def test_cli_highest_cycles_and_failure_preserves_output(tmp_path, tables, capsys):
    reference, candidate, output = (tmp_path / name for name in ("reference.root", "candidate.root", "exact.json"))
    write_root(reference, tables)
    write_root(candidate, tables)
    argv = [str(reference), str(candidate), "--exact", "--expected-events", "2", "--output", str(output)]
    capture_compare.main(argv)
    assert json.loads(output.read_text())["capture_fields_identical"]
    assert json.loads(capsys.readouterr().out)["excluded_fields"] == ["events.cpuTime", "events.realTime"]
    saved = output.read_bytes()
    bad = clone(tables)
    bad["events"]["geometryTag"][0] = "newest cycle differs"
    with uproot.update(candidate) as root:
        root.mktree("dump/events", bad["events"])
    with pytest.raises(ValueError, match="geometryTag"):
        capture_compare.main(argv)
    assert output.read_bytes() == saved
    output.unlink()
    with pytest.raises(ValueError, match="geometryTag"):
        capture_compare.main(argv)
    assert not output.exists()
    with uproot.update(candidate) as root:
        root.mktree("dump/events", tables["events"])
    capture_compare.main(argv)
    assert json.loads(output.read_text())["capture_fields_identical"]


@pytest.mark.parametrize("which", ["reference", "candidate"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_cli_cannot_clobber_either_input(tmp_path, tables, which, alias):
    reference, candidate = tmp_path / "reference.root", tmp_path / "candidate.root"
    write_root(reference, tables)
    write_root(candidate, tables)
    source = reference if which == "reference" else candidate
    output = source if alias == "direct" else tmp_path / "alias.root"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        output.hardlink_to(source)
    saved = source.read_bytes()
    with pytest.raises(ValueError, match="output must differ"):
        capture_compare.main([str(reference), str(candidate), "--exact", "--expected-events", "2", "--output", str(output)])
    assert source.read_bytes() == saved


def widen(tables, extra):
    """Store every string field as fixed-width unicode, padded by extra characters."""
    out = clone(tables)
    for table in out.values():
        for field, value in table.items():
            if value.dtype == object:
                table[field] = value.astype(f"<U{max([len(v) for v in value] + [1]) + extra}")
    return out


def transport(tables, name):
    """Set every process-name field except creator to one transport process name.

    Track ends get kinetic energy, so a Transportation stop is not a zero-energy natural boundary stop.
    """
    out = clone(tables)
    for table in out.values():
        for field in table:
            if capture.is_process_field(field) and field != "creator":
                table[field][:] = name
    t, s = out["tracks"], out["steps"]
    for row in range(len(t["track"])):
        steps = np.flatnonzero((s["event"] == t["event"][row]) & (s["track"] == t["track"][row]))
        last = steps[np.argmax(s["number"][steps])]
        t["end_kinetic"][row] = s["post_kinetic"][last] = 1.
    return out


def test_string_width_is_not_a_difference(tables):
    narrow, wide = widen(tables, 0), widen(tables, 7)
    for reference, other in ((narrow, wide), (wide, narrow), (tables, wide), (wide, tables)):
        report = capture_compare.compare_exact(reference, other, 2)
        assert report["capture_fields_identical"] and report["capture_fields_identical_by_process_class"]
        assert report["process_class_equated"] == {}


def test_transport_variants_are_one_process_class(tables):
    plain, coupled = transport(tables, "Transportation"), transport(tables, "CoupledTransportation")
    fields = {f"{name}.{field}" for name, table in plain.items() for field in table
              if capture.is_process_field(field) and field != "creator"}
    assert {"tracks.stopProcess", "tracks.end_process", "steps.post_process", "classifications.state_process"} <= fields
    for reference, other in ((plain, coupled), (coupled, plain), (widen(plain, 0), widen(coupled, 3))):
        report = capture_compare.compare_exact(reference, other, 2)
        assert not report["capture_fields_identical"] and report["capture_fields_identical_by_process_class"]
        assert report["process_classes"] == {"CoupledTransportation": "Transportation"}
        assert set(report["process_class_equated"]) == fields
        assert report["process_class_equated"]["steps.post_process"] == len(plain["steps"]["event"])
    assert capture_compare.compare_exact(coupled, coupled, 2)["capture_fields_identical"]


def test_transport_class_applies_to_process_fields_only(tables):
    tables["events"]["geometryTag"][:] = "Transportation"
    candidate = clone(tables)
    candidate["events"]["geometryTag"][:] = "CoupledTransportation"
    capture.audit(candidate, 2)
    with pytest.raises(ValueError, match=r"events\.geometryTag: exact field mismatch"):
        capture_compare.compare_exact(tables, candidate, 2)


@pytest.mark.parametrize("value", ["eIoni", "Transportation ", "CoupledTransportation2"])
def test_other_process_names_still_differ(tables, value):
    reference, candidate = widen(transport(tables, "Transportation"), 0), widen(transport(tables, value), 2)
    capture.audit(candidate, 2)
    with pytest.raises(ValueError, match="exact field mismatch"):
        capture_compare.compare_exact(reference, candidate, 2)


def test_string_storage_kind_still_checked(tables):
    candidate = clone(tables)
    candidate["events"]["geometryTag"] = np.full(2, 7, dtype=np.int32)
    with pytest.raises(ValueError):
        capture_compare.compare_exact(tables, candidate, 2)
