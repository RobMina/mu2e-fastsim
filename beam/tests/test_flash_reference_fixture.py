"""Analytic synthetic ROOT tests only; no C++ build, art job or transport validation."""
import json

import numpy as np
import pytest
import uproot

from beamfastsim.flash_reference_export import REQUIRED, SchemaError, read_export
from beamfastsim.flash_reference_fixture import check_fixture, main
from test_flash_reference_export import _fixture as _audit_fixture, _table, _write


# These constants and rows follow FlashReferenceFixture_module.cc, not the checker.
_X = 1.0 + 2.0**-40
_TIME = 10331490143436802.0
_ENERGY = np.sqrt(16.078125 + 0.51099895**2)
_PID = {"input": 10, "continued": 11, "generator": 20, "source": 30,
        "photon": 31, "tracker": 32, "status": 40, "volumes": 50}
_TYPES = {1: "mu2e::SimParticleCollection", 2: "mu2e::GenParticleCollection",
          3: "mu2e::StepPointMCCollection", 4: "mu2e::StatusG4",
          5: "mu2e::PhysicalVolumeInfoMultiCollection"}
_IDENTITY = {"run", "subrun", "event", "ordinal", "product", "index"}
_STEP_FIELDS = tuple(field for field in REQUIRED["steps"] if field not in _IDENTITY)


def _fixture(*, restart_events=False, fail_status=False, unavailable_state=None):
    # Reuse only the audited schema/storage declarations, never its physics rows.
    template = _audit_fixture()
    tables = {name: template[name] for name in ("schema", "storage")}
    rows = {name: [] for name in REQUIRED if name not in tables}

    def emit(table, identity=None, **values):
        rows[table].append({**(identity or {}), **values})

    for kind, tag in [(1, "fixture:input"), (1, "fixture:continued"), (2, "fixture"),
                      (3, "fixture:source"), (3, "fixture:photon"), (3, "fixture:tracker"),
                      (4, "fixture"), (5, "fixture")]:
        emit("requirements", kind=kind, tag=tag, expected_events=4, ordinal_base=0)
    emit("completion", complete=1, events=4, ordinal_base=0, subrun_products=2)

    def product(identity, name, kind, size):
        instance = name if name in ("input", "continued", "source", "photon", "tracker") else ""
        role = "discovered" if kind in (1, 2) else "fixture" + (":" + instance if instance else "")
        emit("products", identity, product=_PID[name], kind=kind, scope=int(kind == 5),
             size=size, role=role, branch=f"product_fixture_{instance}_flashReferenceFixture.",
             friendly=_TYPES[kind], label="fixture", instance=instance,
             process="flashReferenceFixture", type=_TYPES[kind], produced_type=_TYPES[kind])

    for ordinal in range(4):
        event = ordinal % 2 + 1 if restart_events else ordinal + 1
        identity = dict(run=1, subrun=ordinal // 2 + 1, event=event, ordinal=ordinal)
        filled = bool(event % 2)
        emit("events", identity, nproducts=7, nparticles=5 if filled else 0,
             ngenerators=int(filled), nsteps=4 if filled else 0,
             ndaughters=3 if filled else 0, nstatuses=1,
             ptr_failures=int(filled and unavailable_state is not None))
        for name, kind, size in [("input", 1, int(filled)), ("continued", 1, 4 * filled),
                                 ("generator", 2, int(filled)), ("source", 3, 2 * filled),
                                 ("photon", 3, int(filled)), ("tracker", 3, int(filled)),
                                 ("status", 4, 1)]:
            product(identity, name, kind, size)
        emit("statuses", identity, product=_PID["status"], status=int(fail_status),
             ntracks=3 if filled else 0, overflow=0, killed_step=0,
             killed_field=int(fail_status), cpu=.0625, real=.125)
        if not filled:
            continue
        emit("generators", identity, product=_PID["generator"], pdg=11, gen_id=1,
             x=_X, y=-0.0, z=4179., px=.125, py=-.25, pz=4., energy=_ENERGY,
             time=_TIME, proper_time=.125)
        for collection, key, index in [("input", 17, 0), ("continued", 17, 0),
                                        ("continued", 101, 1), ("continued", 102, 2),
                                        ("continued", 103, 3)]:
            copied, photon = key == 17, key == 102
            parent_key = 0 if copied else (101 if photon else 17)
            ndaughters = 2 if collection == "continued" and copied else int(key == 101)
            emit("particles", identity, product=_PID[collection], index=index, key=key,
                 id=key, stage=0 if copied else 2, ndaughters=ndaughters,
                 pdg=22 if photon else 11, creation=16 if photon else 56,
                 stopping=16 if key == 101 else 49,
                 parent_state=0 if copied else 1,
                 parent_product=0 if copied else _PID["continued"],
                 parent_key=np.iinfo(np.uint64).max if copied else parent_key,
                 generator_state=int(copied), generator_product=_PID["generator"] if copied else 0,
                 generator_key=0 if copied else np.iinfo(np.uint64).max,
                 x=_X, y=-0.0, z=4179., end_x=_X,
                 end_y=0.0 if key in (101, 102) else -0.0,
                 end_z=4179. + (1 if key == 101 else 2 if photon else 0),
                 px=0. if photon else .125, py=0. if photon else -.25,
                 pz=.5 if photon else 4., energy=.5 if photon else _ENERGY,
                 end_px=0. if photon else .125, end_py=0. if photon else -.25,
                 end_pz=.5 if photon else 4., end_energy=.5 if photon else _ENERGY,
                 time=_TIME + (2 if key == 103 else 0), proper_time=0. if photon else .125,
                 end_time=_TIME + (2 if key == 103 else 0), end_proper_time=0. if photon else .25,
                 start_volume=7 if copied else 11, end_volume=7 if copied else 11,
                 start_g4status=0, end_g4status=2, end_defined=1, pre_last_ke=-1.,
                 end_ke=3.5 if copied else .5 if photon else 3.,
                 nsteps=2 if copied else 3 if key == 101 else 1,
                 track_length=.75 if key == 101 else 2. if photon else .5,
                 ion_excitation=.0625 if photon else 0., ion_float_level=3 if photon else 0)
        for owner, daughters in [(17, (103, 101)), (101, (102,))]:
            for index, daughter in enumerate(daughters):
                emit("daughters", identity, product=_PID["continued"], index=index,
                     owner_key=owner, daughter_state=1,
                     daughter_product=_PID["continued"], daughter_key=daughter)
        for role, count in [("source", 2), ("photon", 1), ("tracker", 1)]:
            for index in range(count):
                photon, tracker = role == "photon", role == "tracker"
                emit("steps", identity, product=_PID[role], index=index,
                     particle_state=unavailable_state if tracker and unavailable_state is not None else 1,
                     particle_product=_PID["input"] if role == "source" else _PID["continued"],
                     particle_key=17 if role == "source" else 102 if photon else 999 if unavailable_state else 101,
                     volume=123 if tracker else 9, x=_X, y=-0.0, z=4179.,
                     post_x=_X + .25 if photon else _X,
                     post_y=-.5 if photon else -0.0 if tracker else 0.0,
                     post_z=4181. if photon else 4179. if tracker else 4179.5,
                     px=0. if photon else .125, py=0. if photon else -.25,
                     pz=.5 if photon else 4., post_px=0. if photon else .125,
                     post_py=.25 if photon else -.25, post_pz=.25 if photon else 4.,
                     time=_TIME + (2 * index if role == "source" else 0),
                     proper_time=0. if photon else .25 if tracker else .125,
                     total_edep=.125 if photon else .25 if tracker else 0.,
                     nonionizing_edep=.03125 if photon else .125 if tracker else 0.,
                     visible_edep=.015625 if photon else .0625 if tracker else 0.,
                     length=2.125 if photon else .5, end_process=12 if photon else 17 if tracker else 49)
    for subrun in (1, 2):
        identity = dict(run=1, subrun=subrun, event=0, ordinal=0)
        product(identity, "volumes", 5, 3)
        for stage, size in enumerate((1, 0, 1)):
            emit("stages", identity, product=_PID["volumes"], index=stage, stage=stage, size=size)
        for stage, index, name, material in [(0, 7, "upstream", "Vacuum"), (2, 11, "foil", "Aluminum")]:
            emit("volumes", identity, product=_PID["volumes"], stage=stage, index=index,
                 copy=9, name=name, material=material)
    for name, records in rows.items():
        tables[name] = _table(name, len(records))
        for index, record in enumerate(records):
            for field, value in record.items():
                tables[name][field][index] = value
    return tables


@pytest.mark.parametrize("restart_events", [False, True])
def test_analytic_cpp_fixture_passes(tmp_path, restart_events):
    path = _write(tmp_path / "fixture.root", _fixture(restart_events=restart_events))
    report = check_fixture(path)
    assert report["fixture_values_valid"], report
    assert report["failures"] == []
    assert report["audit"]["audit_passed"], report["audit"]["failures"]
    assert report["audit"]["rows"] == dict(schema=1, storage=8, requirements=8, completion=1,
        events=4, products=30, particles=10, generators=2, steps=8, statuses=4,
        daughters=6, stages=6, volumes=4)
    assert report["audit"]["physics_completeness"] is False
    assert "not physics or arbitrary-art" in report["scope"]
    exported = read_export(path)
    with uproot.open(path) as root:
        assert all(root["export/" + name].classname == "TTree" for name in REQUIRED)
    for name, contract in REQUIRED.items():
        assert set(exported[name]) == set(contract)
        for field, dtype in contract.items():
            if dtype != "string":
                assert exported[name][field].dtype == np.dtype(dtype)
    source = exported["steps"]["product"] == _PID["source"]
    np.testing.assert_array_equal(exported["steps"]["time"][source], [_TIME, _TIME + 2] * 2)
    assert np.all(np.signbit(exported["steps"]["y"]))
    assert np.all(exported["particles"]["time"] == np.float32(_TIME))
    assert float(np.float32(_TIME)) != _TIME
    assert _X != float(np.float32(_X))
    assert _ENERGY != float(np.float32(_ENERGY))


@pytest.mark.parametrize("role", ["source", "photon", "tracker"])
@pytest.mark.parametrize("field", _STEP_FIELDS)
def test_mutating_every_step_value_is_rejected(tmp_path, role, field):
    tables = _fixture()
    values = tables["steps"][field]
    index = np.flatnonzero(tables["steps"]["product"] == _PID[role])[0]
    old = values[index]
    values[index] = np.nextafter(old, np.array(np.inf, dtype=values.dtype)) if values.dtype.kind == "f" else old + 1
    report = check_fixture(_write(tmp_path / "mutated.root", tables))
    assert not report["fixture_values_valid"]
    assert role + "_" + field in report["failures"]


@pytest.mark.parametrize("role,field", [("source", "y"), ("source", "post_y"),
                                       ("tracker", "post_y"), ("photon", "px")])
def test_signed_zero_is_not_numeric_equality(tmp_path, role, field):
    tables = _fixture()
    index = np.flatnonzero(tables["steps"]["product"] == _PID[role])[0]
    tables["steps"][field][index] *= -1
    report = check_fixture(_write(tmp_path / "zero.root", tables))
    assert report["audit"]["audit_passed"]
    assert role + "_" + field in report["failures"]


@pytest.mark.parametrize("change", ["collapse", "reverse"])
def test_both_source_hits_keep_double_times_and_order(tmp_path, change):
    tables = _fixture()
    indices = np.flatnonzero(tables["steps"]["product"] == _PID["source"])
    if change == "collapse":
        tables["steps"]["time"][indices] = _TIME
    else:
        tables["steps"]["time"][indices] = [_TIME + 2, _TIME] * 2
    report = check_fixture(_write(tmp_path / "source.root", tables))
    assert report["audit"]["audit_passed"]
    assert "source_time" in report["failures"]


@pytest.mark.parametrize("table,field,code", [
    ("particles", "time", "particle_float_times"),
    ("particles", "x", "particle_double_position"),
    ("particles", "energy", "particle_float_energy"),
    ("generators", "time", "generator_double_time"),
    ("generators", "proper_time", "generator_proper_time"),
    ("generators", "energy", "generator_energy"),
    ("particles", "ion_excitation", "ion_excitation"),
    ("particles", "ion_float_level", "ion_level"),
])
def test_particle_and_generator_precision_mutations(tmp_path, table, field, code):
    tables = _fixture()
    values = tables[table][field]
    index = np.flatnonzero(tables[table]["pdg"] == 22)[0] if field.startswith("ion_") else 0
    old = values[index]
    values[index] = np.nextafter(old, np.array(np.inf, dtype=values.dtype)) if values.dtype.kind == "f" else old + 1
    report = check_fixture(_write(tmp_path / "precision.root", tables))
    assert report["audit"]["audit_passed"]
    assert code in report["failures"]
    assert not report["fixture_values_valid"]


def test_daughter_order_is_not_a_set(tmp_path):
    tables = _fixture()
    tables["daughters"]["daughter_key"][:2] = [101, 103]
    report = check_fixture(_write(tmp_path / "daughters.root", tables))
    assert report["audit"]["audit_passed"]
    assert report["failures"] == ["daughter_order"]


@pytest.mark.parametrize("fail_status,state", [(True, None), (False, 2), (False, 3), (True, 2)])
def test_injected_failures_are_retained_and_require_explicit_expectation(tmp_path, fail_status, state):
    path = _write(tmp_path / "injected.root", _fixture(fail_status=fail_status, unavailable_state=state))
    report = check_fixture(path, fail_status=fail_status, unavailable_ptr=state is not None)
    assert report["fixture_values_valid"], report
    assert report["audit"]["table_structure_valid"]
    assert not report["audit"]["physics_checks_passed"]
    assert report["audit"]["events"] == 4
    assert report["audit"]["rows"]["steps"] == 8
    codes = {failure["code"] for failure in report["audit"]["failures"]}
    expected_codes = {"status_status", "status_killed_field"} if fail_status else set()
    if state is not None:
        expected_codes.add("steps_particle_unavailable")
    assert codes == expected_codes
    unexpected = check_fixture(path)
    assert not unexpected["fixture_values_valid"]
    assert "physics_expected_result" in unexpected["failures"]


@pytest.mark.parametrize("option", ["fail_status", "unavailable_ptr"])
def test_expected_failure_flag_does_not_accept_clean_fixture(tmp_path, option):
    report = check_fixture(_write(tmp_path / "clean.root", _fixture()), **{option: True})
    assert not report["fixture_values_valid"]
    assert "physics_expected_result" in report["failures"]


def test_invalid_ledger_cannot_pass_analytic_values(tmp_path):
    tables = _fixture()
    tables["events"]["nsteps"][0] = 3
    report = check_fixture(_write(tmp_path / "ledger.root", tables))
    assert not report["fixture_values_valid"]
    assert "table_structure" in report["failures"]
    assert "nonempty_step_count" in report["failures"]


@pytest.mark.parametrize("table,field,dtype", [("steps", "time", "float32"),
                                               ("particles", "time", "float64"),
                                               ("particles", "x", "float32"),
                                               ("generators", "energy", "float32")])
def test_checker_requires_persistent_precision_schema(tmp_path, table, field, dtype):
    path = _write(tmp_path / "dtype.root", _fixture(), override=(table, field, dtype))
    with pytest.raises(SchemaError, match=table + r"\." + field):
        check_fixture(path)


@pytest.mark.parametrize("invalid", [False, True])
def test_cli_json_exit_status_and_exclusive_output(tmp_path, invalid):
    tables = _fixture()
    if invalid:
        tables["steps"]["length"][0] = 1.
    path = _write(tmp_path / "cli.root", tables)
    out = tmp_path / "report.json"
    argv = ["--input", str(path), "--directory", "export", "--out", str(out)]
    assert main(argv) == (2 if invalid else 0)
    report = json.loads(out.read_text())
    assert report["fixture_values_valid"] is not invalid
    assert report["audit"]["table_structure_valid"]
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        main(argv)
    assert out.read_bytes() == before


def test_cli_failure_options_and_custom_directory(tmp_path):
    path = _write(tmp_path / "custom.root", _fixture(fail_status=True, unavailable_state=3))
    with uproot.open(path) as root:
        tables = {name: root["export/" + name].arrays(library="ak", how=dict) for name in REQUIRED}
    custom = tmp_path / "nested.root"
    with uproot.recreate(custom) as root:
        for name, columns in tables.items():
            root.mktree("nested/raw/" + name, REQUIRED[name])
            root["nested/raw/" + name].extend(columns)
    out = tmp_path / "injected.json"
    assert main(["--input", str(custom), "--directory", "nested/raw", "--out", str(out),
                 "--fail-status", "--unavailable-ptr"]) == 0
    report = json.loads(out.read_text())
    assert report["fixture_values_valid"]
    assert not report["audit"]["physics_checks_passed"]


@pytest.mark.parametrize("role", ["source", "photon", "tracker"])
@pytest.mark.parametrize("field", sorted(_IDENTITY))
def test_mutating_every_step_identity_is_rejected(tmp_path, role, field):
    tables = _fixture()
    index = np.flatnonzero(tables["steps"]["product"] == _PID[role])[0]
    tables["steps"][field][index] += 1
    report = check_fixture(_write(tmp_path / "identity.root", tables))
    assert not report["fixture_values_valid"]
    assert "table_structure" in report["failures"]
    assert not report["audit"]["table_structure_valid"]


def test_continued_product_selection_uses_original_process(tmp_path):
    tables = _fixture()
    continued = tables["products"]["product"] == _PID["continued"]
    tables["products"]["process"][continued] = "readerProcess"
    report = check_fixture(_write(tmp_path / "process.root", tables))
    assert report["audit"]["table_structure_valid"]
    assert not report["fixture_values_valid"]
    assert "physics_expected_result" in report["failures"]
    assert "particle_product_selection" in {failure["code"] for failure in report["audit"]["failures"]}


@pytest.mark.parametrize("field", ["status", "ntracks", "overflow", "killed_step", "killed_field", "cpu", "real"])
@pytest.mark.parametrize("fail_status", [False, True])
@pytest.mark.parametrize("index", [0, 1], ids=["filled", "empty"])
def test_every_status_value_is_exact_even_with_expected_failure(tmp_path, field, fail_status, index):
    tables = _fixture(fail_status=fail_status)
    values = tables["statuses"][field]
    old = values[index]
    values[index] = np.nextafter(old, np.array(np.inf, dtype=values.dtype)) if values.dtype.kind == "f" else old + 1
    report = check_fixture(_write(tmp_path / "status.root", tables), fail_status=fail_status)
    assert report["audit"]["table_structure_valid"]
    assert not report["fixture_values_valid"]
    assert "status_" + field in report["failures"]


@pytest.mark.parametrize("fail_status,state", [(False, None), (True, None), (False, 2), (True, 3)])
def test_unrelated_nan_cpu_cannot_satisfy_expected_failure_flags(tmp_path, fail_status, state):
    tables = _fixture(fail_status=fail_status, unavailable_state=state)
    tables["statuses"]["cpu"][0] = np.nan
    # A generic failure must not masquerade as the deliberately requested injected status failure.
    report = check_fixture(_write(tmp_path / "nan-cpu.root", tables),
                           fail_status=True, unavailable_ptr=state is not None)
    assert report["audit"]["table_structure_valid"]
    assert not report["audit"]["physics_checks_passed"]
    assert not report["fixture_values_valid"]
    assert "physics_expected_result" not in report["failures"]
    assert "physics_failure_codes" in report["failures"]
    assert "status_cpu" in report["failures"]
    if not fail_status:
        assert "status_status" in report["failures"]
        assert "status_killed_field" in report["failures"]


@pytest.mark.parametrize("state", [2, 3])
@pytest.mark.parametrize("missing", ["status", "killed_field", "both"])
def test_combined_flags_require_both_status_failure_fields(tmp_path, state, missing):
    tables = _fixture(fail_status=True, unavailable_state=state)
    for field in ("status", "killed_field") if missing == "both" else (missing,):
        tables["statuses"][field][:] = 0
    report = check_fixture(_write(tmp_path / "missing-status.root", tables),
                           fail_status=True, unavailable_ptr=True)
    assert report["audit"]["table_structure_valid"]
    assert not report["audit"]["physics_checks_passed"]
    assert "physics_expected_result" not in report["failures"]
    assert not report["fixture_values_valid"]
    assert "physics_failure_codes" in report["failures"]
    for field in ("status", "killed_field") if missing == "both" else (missing,):
        assert "status_" + field in report["failures"]
