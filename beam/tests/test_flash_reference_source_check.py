"""Synthetic independent measured-state tests, not an installed art certificate."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from beamfastsim.flash_reference_export import ReferenceExport, SchemaError
from beamfastsim import flash_reference_source_check as checker
from test_flash_reference_export import _fixture, _write


def _sample():
    t = _fixture()
    p = t["products"]
    selected = p["kind"] == 3
    p["role"][selected] = "BeamFilter:Beam"
    p["label"][selected] = "BeamFilter"
    p["instance"][selected] = "Beam"
    t["requirements"]["tag"][2] = "BeamFilter:Beam"
    t["particles"]["pdg"][:] = [13, 22, -13]
    steps = t["steps"]
    steps["volume"][:] = [123, 124]
    steps["x"][:] = [1.0 + 2**-40, 2.0]
    steps["time"][:] = [10331490143436802., 10331490143436804.]
    s = {k: steps[k].astype(np.int64) for k in checker.EVENT}
    s.update(particle_key=steps["particle_key"].astype(np.int64),
             volume_copy=steps["volume"].astype(np.int64), pdg=np.array([13, -13]),
             product_id=np.array(["10", "11"]), input_row=np.arange(2),
             x=np.column_stack([steps[k] for k in ("x", "y", "z")]).astype(np.float32),
             p=np.column_stack([steps[k] for k in ("px", "py", "pz")]).astype(np.float32),
             time_ns=steps["time"].copy(), event_hit_counts=np.array([2, 0]))
    s["event_headers"] = np.empty(2, dtype=[(k, "<i8") for k in checker.EVENT])
    for k in checker.EVENT:
        s["event_headers"][k] = t["events"][k]
    return t, s


def _save(tmp_path, states):
    path = tmp_path / "source_states.npz"
    np.savez_compressed(path, **states)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _check(tmp_path, t, s):
    path, digest = _save(tmp_path, s)
    return checker.check_source(ReferenceExport(t), path, states_sha256=digest,
                                expected_events=2, expected_rows=2)


def test_measured_agreement_and_honest_scope(tmp_path):
    t, s = _sample()
    report = _check(tmp_path, t, s)
    assert report["passed"], report
    assert report["full_persistent_extraction_certified"] is False
    assert report["physics_completeness"] is False
    assert "post-state" in report["not_covered"]
    assert "deposits" in report["not_covered"]
    assert report["source_summary"]["empty_events"] == 1
    assert report["source_summary"]["electron"] == 0
    assert report["source_summary"]["positron"] == 0
    assert report["source_summary"]["mu_minus"] == 1
    assert report["source_summary"]["mu_plus"] == 1
    assert report["source_summary"]["time_ns_min"] == 10331490143436802.
    assert report["source_summary"]["subruns"] == [{"run": 1, "subrun": 2}]


@pytest.mark.parametrize("field,value,code", [
    ("x", 1.25, "source_x_float32_bytes"),
    ("px", 0.5, "source_p_float32_bytes"),
    ("time", 10331490143436806., "source_time_float64_bytes"),
    ("volume", 125, "source_volume_copy"),
    ("particle_product", 11, "source_product_id"),
    ("particle_key", 8, "source_particle_key"),
])
def test_measured_mutations_fail(tmp_path, field, value, code):
    t, s = _sample()
    t["steps"][field][0] = value
    report = _check(tmp_path, t, s)
    assert not report["passed"]
    assert code in report["failures"]


def test_destination_conversion_not_double_or_tolerance(tmp_path):
    t, s = _sample()
    t["steps"]["x"][0] = 1 + 2**-39
    assert _check(tmp_path, t, s)["passed"]
    s["x"][0, 0] = np.nextafter(s["x"][0, 0], np.float32(2))
    assert not _check(tmp_path, t, s)["passed"]


@pytest.mark.parametrize("field,stock", [("px", "p"), ("y", "x"), ("time", "time_ns")])
def test_signed_zero_bytes(tmp_path, field, stock):
    t, s = _sample()
    t["steps"][field][0] = -0.
    if stock == "time_ns":
        s[stock][0] = 0.
    assert not _check(tmp_path, t, s)["passed"]


@pytest.mark.parametrize("field", ["post_x", "proper_time", "total_edep", "length", "end_process"])
def test_unmeasured_fields_do_not_claim_comparison(tmp_path, field):
    t, s = _sample()
    t["steps"][field][0] = 42
    report = _check(tmp_path, t, s)
    assert report["passed"]
    assert not report["full_persistent_extraction_certified"]


@pytest.mark.parametrize("state", [0, 2, 3, 4])
def test_invalid_ptr_states_fail_closed(tmp_path, state):
    t, s = _sample()
    t["steps"]["particle_state"][0] = state
    assert not _check(tmp_path, t, s)["passed"]


def test_full_event_product_key_pdg_join(tmp_path):
    t, s = _sample()
    assert t["steps"]["particle_key"][0] == t["steps"]["particle_key"][1]
    assert _check(tmp_path, t, s)["passed"]
    t["particles"]["pdg"][2] = 13
    assert "source_pdg" in _check(tmp_path, t, s)["failures"]


@pytest.mark.parametrize("ids,passed", [(["00010", "011"], True), (["1:10", "1:11"], False),
                                        (["11", "10"], False), (["10 ", "11"], False)])
def test_product_ids_never_alias(tmp_path, ids, passed):
    t, s = _sample()
    s["product_id"] = np.array(ids)
    assert _check(tmp_path, t, s)["passed"] is passed


@pytest.mark.parametrize("mutation", ["missing", "object", "float_time", "shape", "nan", "counts",
                                      "row", "header_duplicate", "header_float", "negative", "float_id"])
def test_malformed_npz_rejected(tmp_path, mutation):
    t, s = _sample()
    if mutation == "missing":
        del s["pdg"]
    elif mutation == "object":
        s["product_id"] = s["product_id"].astype(object)
    elif mutation == "float_time":
        s["time_ns"] = s["time_ns"].astype(np.float32)
    elif mutation == "shape":
        s["x"] = s["x"].ravel()
    elif mutation == "nan":
        s["p"][0, 0] = np.nan
    elif mutation == "counts":
        s["event_hit_counts"][:] = [1, 1]
    elif mutation == "row":
        s["input_row"][:] = [1, 0]
    elif mutation == "header_duplicate":
        s["event_headers"][1] = s["event_headers"][0]
    elif mutation == "header_float":
        s["event_headers"] = s["event_headers"].astype([(k, "f8") for k in checker.EVENT])
    elif mutation == "negative":
        s["particle_key"][0] = -1
    else:
        s["particle_key"] = s["particle_key"].astype(float)
    with pytest.raises((ValueError, TypeError)):
        _check(tmp_path, t, s)


def test_hash_required_and_checked_before_numpy_load(tmp_path, monkeypatch):
    t, s = _sample()
    path, digest = _save(tmp_path, s)
    with pytest.raises(TypeError):
        checker.check_source(ReferenceExport(t), path)
    def forbidden(*args, **kwargs):
        raise AssertionError("must check trusted digest before loading")
    monkeypatch.setattr(np, "load", forbidden)
    for bad in (None, "", "a" * 64):
        with pytest.raises(ValueError, match="states_sha256"):
            checker.check_source(ReferenceExport(t), path, states_sha256=bad)


@pytest.mark.parametrize("mutation", ["role", "label", "instance", "event_order", "row_order", "index", "ledger"])
def test_selection_and_order_fail_closed(tmp_path, mutation):
    t, s = _sample()
    if mutation in ("role", "label", "instance"):
        t["products"][mutation][3] = "wrong"
    elif mutation == "event_order":
        t["events"] = {k: a[::-1] for k, a in t["events"].items()}
    elif mutation == "row_order":
        t["steps"] = {k: a[::-1] for k, a in t["steps"].items()}
    elif mutation == "index":
        t["steps"]["index"][1] = 0
    else:
        t["events"]["nsteps"][0] = 0
    assert not _check(tmp_path, t, s)["passed"]


def test_in_memory_schema_and_default_sample_size(tmp_path):
    t, s = _sample()
    path, digest = _save(tmp_path, s)
    assert not checker.check_source(ReferenceExport(t), path, states_sha256=digest)["passed"]
    t["steps"]["time"] = t["steps"]["time"].astype(np.float32)
    with pytest.raises(SchemaError):
        _check(tmp_path, t, s)


def test_subruns_and_empty_event_ids_are_not_ignored(tmp_path):
    t, s = _sample()
    s["event_headers"]["subrun"][1] = 3
    assert "ordered_event_subrun" in _check(tmp_path, t, s)["failures"]


def test_root_input_and_exclusive_cli(tmp_path, monkeypatch):
    t, s = _sample()
    root = _write(tmp_path / "export.root", t)
    path, digest = _save(tmp_path, s)
    report = checker.check_source(root, path, states_sha256=digest, expected_events=2, expected_rows=2)
    assert report["passed"]
    original = checker.check_source
    monkeypatch.setattr(checker, "check_source", lambda *a, **kw: original(*a, **kw, expected_events=2, expected_rows=2))
    out = tmp_path / "report.json"
    args = ["--input", str(root), "--states", str(path), "--states-sha256", digest, "--out", str(out)]
    assert checker.main(args) == 0
    assert json.loads(out.read_text())["passed"]
    with pytest.raises(FileExistsError):
        checker.main(args)
    result = subprocess.run([sys.executable, "-m", "beamfastsim.flash_reference_source_check", *args[:-1],
                             str(tmp_path / "default.json")], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    bad_out = tmp_path / "bad.json"
    args[-1] = str(bad_out)
    args[5] = "0" * 64
    assert checker.main(args) == 2
    assert not json.loads(bad_out.read_text())["passed"]


def test_repeated_event_product_keys_across_subruns(tmp_path):
    t, s = _sample()
    second, _ = _sample()
    for table in ("events", "products", "steps", "particles", "daughters", "generators", "statuses", "stages", "volumes"):
        for field in t[table]:
            extra = second[table][field].copy()
            if field == "subrun":
                extra[:] = 9
            if field == "ordinal" and table not in ("stages", "volumes"):
                mask = second[table]["scope"] == 0 if table == "products" else np.ones(len(extra), dtype=bool)
                extra[mask] += 2
            t[table][field] = np.concatenate([t[table][field], extra])
    t["events"]["ordinal"][:] = np.arange(4)
    t["completion"]["events"][:] = 4
    t["completion"]["subrun_products"][:] = 2
    t["requirements"]["expected_events"][:] = 4
    s = {k: np.concatenate([a, a]) for k, a in s.items()}
    s["subrun"][2:] = 9
    s["event_headers"]["subrun"][2:] = 9
    s["input_row"] = np.arange(4)
    path, digest = _save(tmp_path, s)
    report = checker.check_source(ReferenceExport(t), path, states_sha256=digest,
                                  expected_events=4, expected_rows=4)
    assert report["passed"], report
    t["particles"]["pdg"][3] = 22
    report = checker.check_source(ReferenceExport(t), path, states_sha256=digest,
                                  expected_events=4, expected_rows=4)
    assert "source_pdg" in report["failures"]


def test_production_denominator_1000_events_1130_rows(tmp_path):
    base, _ = _sample()
    counts = np.zeros(1000, dtype=np.int64)
    counts[:565] = 2
    populated = np.flatnonzero(counts)
    t = {name: {k: a.copy() for k, a in columns.items()} for name, columns in base.items()}
    event_indices = {
        "events": np.arange(1000), "products": np.arange(1000),
        "statuses": np.arange(1000), "steps": populated, "particles": populated,
        "daughters": populated, "generators": populated,
    }
    for table, owners in event_indices.items():
        if table == "products":
            pieces = []
            for active, events in ((True, populated), (False, np.flatnonzero(counts == 0))):
                mask = (base[table]["event"] == (3 if active else 4)) & (base[table]["scope"] == 0)
                piece = {k: np.tile(a[mask], len(events)) for k, a in base[table].items()}
                for k, values in (("event", events + 1), ("ordinal", events)):
                    piece[k] = np.repeat(values, np.count_nonzero(mask)).astype(base[table][k].dtype)
                pieces.append(piece)
            t[table] = {k: np.concatenate([piece[k] for piece in pieces] + [a[base[table]["scope"] == 1]])
                        for k, a in base[table].items()}
            continue
        mask = np.ones(len(base[table]["run"]), dtype=bool)
        if table in ("events", "statuses"):
            mask[:] = False
            mask[0] = True
        size = np.count_nonzero(mask)
        t[table] = {k: np.tile(a[mask], len(owners)) for k, a in base[table].items()}
        t[table]["event"] = np.repeat(owners + 1, size).astype(np.uint32)
        t[table]["ordinal"] = np.repeat(owners, size).astype(np.uint64)
    ev = t["events"]
    for field, nonempty, empty in (("nproducts", 5, 4), ("nparticles", 3, 0),
                                    ("ngenerators", 1, 0), ("nsteps", 2, 0), ("ndaughters", 1, 0)):
        ev[field][:] = np.where(counts > 0, nonempty, empty)
    t["completion"]["events"][:] = 1000
    t["requirements"]["expected_events"][:] = 1000
    steps = t["steps"]
    s = {k: steps[k].astype(np.int64) for k in checker.EVENT}
    s.update(particle_key=steps["particle_key"].astype(np.int64), volume_copy=steps["volume"].astype(np.int64),
             pdg=np.tile([13, -13], 565), product_id=np.tile(["10", "11"], 565),
             input_row=np.arange(1130), event_hit_counts=counts,
             x=np.column_stack([steps[k] for k in ("x", "y", "z")]).astype(np.float32),
             p=np.column_stack([steps[k] for k in ("px", "py", "pz")]).astype(np.float32),
             time_ns=steps["time"].copy())
    s["event_headers"] = np.empty(1000, dtype=[(k, "i8") for k in checker.EVENT])
    for k in checker.EVENT:
        s["event_headers"][k] = ev[k]
    path, digest = _save(tmp_path, s)
    report = checker.check_source(ReferenceExport(t), path, states_sha256=digest)
    assert report["passed"], report
    assert report["source_summary"]["rows"] == 1130
    assert report["source_summary"]["empty_events"] == 435


def test_non_npz_and_large_unsigned_identities(tmp_path):
    t, s = _sample()
    path = tmp_path / "not_archive.npy"
    np.save(path, np.arange(2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="NPZ"):
        checker.check_source(ReferenceExport(t), path, states_sha256=digest)
    t["storage"]["persistent_bytes"][t["storage"]["group"] == "step_volume"] = 8
    t["steps"]["volume"][0] = 2**63
    s["volume_copy"][0] = 2**63 - 1
    assert "source_volume_copy" in _check(tmp_path, t, s)["failures"]
