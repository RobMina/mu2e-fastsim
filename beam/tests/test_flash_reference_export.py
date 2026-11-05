"""Synthetic typed ROOT tables test internal audits, not an art extraction run."""
import json

import numpy as np
import pytest
import uproot

from beamfastsim.flash_reference_export import (
    REQUIRED, ReferenceExport, SchemaError, audit_export, compare_source_exports, lookup_volumes,
    main, read_export,
)


def _table(name, count):
    return {key: np.full(count, "", dtype=object) if dtype == "string" else np.zeros(count, dtype=dtype)
            for key, dtype in REQUIRED[name].items()}


def _fixture():
    sizes = {"requirements": 5, "storage": 8, "schema": 1, "completion": 1, "events": 2, "products": 10,
             "steps": 2, "particles": 3, "daughters": 1, "generators": 1,
             "statuses": 2, "stages": 2, "volumes": 2}
    tables = {name: _table(name, sizes[name]) for name in REQUIRED}
    requirements = tables["requirements"]
    requirements["kind"][:] = [1, 2, 3, 4, 5]
    requirements["tag"][:] = ["new", "generator", "photon", "status", "volumes"]
    requirements["expected_events"][:] = 2
    storage = tables["storage"]
    storage["group"][:] = ["positions;step_vectors;step_times;generator_states",
        "particle_momenta_times_energies_length_excitation;step_deposits_length;status_times",
        "particle_ion_float_level", "status_overflow;particle_end_defined",
        "product_id", "ptr_key", "map_key;particle_id", "step_volume"]
    storage["persistent"][:] = ["double", "float", "short", "bool", "art::ProductID::value_type",
        "art::Ptr::key_type", "cet::map_vector_key", "StepPointMC::VolumeId_type"]
    storage["export"][:] = ["Float64", "Float32", "Int32", "UInt32", "UInt64", "UInt64", "UInt64", "UInt64"]
    storage["persistent_bytes"][:] = [8, 4, 2, 1, 8, 8, 4, 4]
    storage["export_bytes"][:] = [8, 4, 4, 4, 8, 8, 8, 8]
    tables["schema"]["version"][:] = 1
    tables["schema"]["product_id_bytes"][:] = 8
    tables["schema"]["ptr_key_bytes"][:] = 8
    tables["completion"]["complete"][:] = 1
    tables["completion"]["events"][:] = 2
    tables["completion"]["subrun_products"][:] = 1
    for name in ("events", "products", "steps", "particles", "daughters", "generators", "statuses", "stages", "volumes"):
        tables[name]["run"][:] = 1
        tables[name]["subrun"][:] = 2
        tables[name]["event"][:] = 3
    ev = tables["events"]
    ev["event"][:] = [3, 4]
    ev["ordinal"][:] = [0, 1]
    for field, values in {"nproducts": [5, 4], "nparticles": [3, 0], "ngenerators": [1, 0],
                          "nsteps": [2, 0], "ndaughters": [1, 0], "nstatuses": [1, 1]}.items():
        ev[field][:] = values
    products = tables["products"]
    products["event"][:] = [3, 3, 3, 3, 3, 4, 4, 4, 0, 4]
    products["ordinal"][:] = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    products["product"][:] = [10, 11, 20, 30, 40, 10, 30, 40, 50, 20]
    products["kind"][:] = [1, 1, 2, 3, 4, 1, 3, 4, 5, 2]
    products["scope"][8] = 1
    products["size"][:] = [2, 1, 1, 2, 1, 0, 0, 1, 2, 0]
    types = {1: "mu2e::SimParticleCollection", 2: "mu2e::GenParticleCollection", 3: "mu2e::StepPointMCCollection", 4: "mu2e::StatusG4", 5: "mu2e::PhysicalVolumeInfoMultiCollection"}
    products["type"][:] = [types[kind] for kind in products["kind"]]
    # Runtime C++ spelling is preserved, not forced to match the declared typedef.
    products["produced_type"][:] = ["runtime::Type" + str(kind) for kind in products["kind"]]
    for field in ("role", "branch", "friendly", "label", "process"):
        products[field][:] = ["new", "input", "generator", "photon", "status", "new", "photon", "status", "volumes", "generator"]
    products["process"][:] = "PROCESS"
    particles = tables["particles"]
    particles["product"][:] = [10, 10, 11]
    particles["index"][:] = [0, 1, 0]
    particles["key"][:] = [7, 8, 7]
    particles["id"][:] = [7, 8, 7]
    particles["stage"][:] = [1, 1, 0]
    particles["ndaughters"][:] = [1, 0, 0]
    particles["parent_state"][1] = 1
    particles["parent_product"][1] = 10
    particles["parent_key"][1] = 7
    particles["generator_state"][0] = 1
    particles["generator_product"][0] = 20
    steps = tables["steps"]
    steps["product"][:] = 30
    steps["index"][:] = [0, 1]
    steps["particle_state"][:] = 1
    steps["particle_product"][:] = [10, 11]
    steps["particle_key"][:] = 7
    steps["time"][:] = [1e12 + 0.25, 1e12 + 0.5]
    steps["px"][:] = [0.0, -0.0]
    daughter = tables["daughters"]
    daughter["product"][:] = 10
    daughter["owner_key"][:] = 7
    daughter["daughter_state"][:] = 1
    daughter["daughter_product"][:] = 10
    daughter["daughter_key"][:] = 8
    tables["generators"]["product"][:] = 20
    tables["statuses"]["product"][:] = 40
    tables["statuses"]["event"][:] = [3, 4]
    tables["statuses"]["ordinal"][:] = [0, 1]
    for name in ("stages", "volumes"):
        tables[name]["event"][:] = 0
        tables[name]["product"][:] = 50
        tables[name]["stage"][:] = [0, 1]
        tables[name]["index"][:] = [0, 1] if name == "stages" else [7, 900]
    tables["stages"]["size"][:] = 1
    tables["volumes"]["copy"][:] = 99
    tables["volumes"]["name"][:] = ["first", "second"]
    tables["volumes"]["material"][:] = "material"
    return tables


def _write(path, tables, override=None, omit=None, cycles=False):
    with uproot.recreate(path) as root:
        for name, columns in tables.items():
            contract = REQUIRED[name].copy()
            if override and override[0] == name:
                contract[override[1]] = override[2]
                columns = {key: values.astype(override[2]) if key == override[1] else values
                           for key, values in columns.items()}
            if omit and omit[0] == name:
                contract.pop(omit[1])
                columns = {key: value for key, value in columns.items() if key != omit[1]}
            root.mktree("export/" + name, contract)
            if len(next(iter(columns.values()))):
                root["export/" + name].extend(columns)
            if cycles and name == "events":
                root.mktree("export/events", contract)
                root["export/events"].extend(columns)
    return path


def _codes(report):
    return {failure["code"] for failure in report["failures"]}


def test_valid_empty_products_repeated_keys_and_copies(tmp_path):
    path = _write(tmp_path / "valid.root", _fixture())
    export = read_export(path)
    report = audit_export(export, expected_events=2, photon_role="photon", photon_cap=3,
                          particle_role="new", particle_cap=3)
    assert report["audit_passed"], report["failures"]
    assert report["table_structure_valid"]
    assert report["physics_checks_passed"]
    assert report["physics_completeness"] is False
    assert "no runtime or extraction certificate" in report["scope"]
    assert export["steps"]["time"][0] == 1e12 + 0.25
    assert np.signbit(export["steps"]["px"][1])
    assert export["products"]["instance"][0] == ""
    assert export["volumes"]["material"][0] == "material"
    indices = lookup_volumes(export, run=1, subrun=2, product=50,
                             stage=[0, 1, 1], index=[7, 900, 99])
    np.testing.assert_array_equal(indices, [0, 1, -1])


def test_highest_tree_cycle(tmp_path):
    assert audit_export(_write(tmp_path / "cycles.root", _fixture(), cycles=True))["audit_passed"]


@pytest.mark.parametrize("table,branch,dtype", [
    ("steps", "time", "float32"), ("particles", "time", "float64"),
    ("particles", "key", "uint32"), ("events", "event", "float64"),
    ("statuses", "killed_field", "uint32"), ("products", "size", "int64"),
    ("steps", "total_edep", "float64"), ("schema", "version", "uint64"),
])
def test_exact_dtype_contract(tmp_path, table, branch, dtype):
    path = _write(tmp_path / "dtype.root", _fixture(), override=(table, branch, dtype))
    with pytest.raises(SchemaError, match=table + r"\." + branch):
        read_export(path)


@pytest.mark.parametrize("table,branch", [("statuses", "killed_field"), ("products", "role"), ("particles", "ion_excitation"), ("steps", "post_px")])
def test_required_branches(tmp_path, table, branch):
    path = _write(tmp_path / "missing.root", _fixture(), omit=(table, branch))
    with pytest.raises(SchemaError, match="missing branches"):
        read_export(path)


@pytest.mark.parametrize("table,field,index,value,code", [
    ("events", "event", 1, 3, "duplicate_event_id"),
    ("events", "ordinal", 1, 0, "duplicate_source_ordinal"),
    ("events", "nsteps", 0, 1, "event_steps_count"),
    ("products", "product", 1, 10, "duplicate_product_row"),
    ("particles", "key", 1, 7, "duplicate_particle_key"),
    ("particles", "id", 1, 7, "duplicate_particle_id"),
    ("particles", "event", 0, 999, "particles_outside_event_ledger"),
    ("products", "event", 0, 999, "product_outside_event_ledger"),
    ("steps", "ordinal", 0, 99, "steps_ordinal"),
    ("steps", "index", 1, 3, "steps_collection_index_order"),
    ("products", "size", 6, 1, "steps_declared_size"),
    ("products", "size", 4, 0, "status_product_size"),
    ("steps", "particle_product", 0, 999, "steps_particle_unresolved"),
    ("steps", "particle_key", 0, 999, "steps_particle_unresolved"),
    ("particles", "generator_product", 0, 10, "particles_generator_unresolved"),
    ("particles", "parent_product", 1, 999, "particles_parent_unresolved"),
    ("steps", "particle_state", 0, 4, "steps_particle_state"),
    ("daughters", "owner_key", 0, 999, "daughter_owner_unresolved"),
    ("particles", "ndaughters", 0, 0, "particle_daughter_count"),
    ("daughters", "index", 0, 1, "daughter_index_order"),
    ("daughters", "daughter_product", 0, 999, "daughters_daughter_unresolved"),
    ("stages", "size", 1, 0, "volume_stage_size"),
    ("stages", "stage", 1, 0, "duplicate_volume_stage"),
    ("volumes", "stage", 1, 2, "volume_stage_unresolved"),
    ("volumes", "product", 0, 999, "volumes_undeclared_volume_product"),
    ("completion", "events", 0, 1, "completion_event_count"),
    ("completion", "complete", 0, 0, "completion_marker"),
])
def test_structure_failures_reported(tmp_path, table, field, index, value, code):
    tables = _fixture()
    tables[table][field][index] = value
    report = audit_export(_write(tmp_path / "bad.root", tables))
    assert not report["table_structure_valid"]
    assert code in _codes(report)
    assert not report["physics_completeness"]


@pytest.mark.parametrize("field", ["status", "overflow", "killed_step", "killed_field"])
def test_status_failures_keep_denominator(tmp_path, field):
    tables = _fixture()
    tables["statuses"][field][0] = 1
    report = audit_export(_write(tmp_path / "status.root", tables))
    assert report["table_structure_valid"]
    assert not report["physics_checks_passed"]
    assert "status_" + field in _codes(report)
    assert report["events"] == 2
    assert "no survivor renormalization" in report["denominator_policy"]


@pytest.mark.parametrize("state", [2, 3])
def test_unavailable_not_null(tmp_path, state):
    tables = _fixture()
    tables["steps"]["particle_state"][0] = state
    tables["events"]["ptr_failures"][0] = 1
    report = audit_export(_write(tmp_path / "ptr.root", tables))
    assert report["table_structure_valid"]
    assert not report["physics_checks_passed"]
    assert "steps_particle_unavailable" in _codes(report)


@pytest.mark.parametrize("state", [0, 2, 3])
def test_source_equality_refuses_identical_invalid_references(tmp_path, state):
    tables = _fixture()
    tables["steps"]["particle_state"][0] = state
    tables["events"]["ptr_failures"][0] = int(state >= 2)
    export = read_export(_write(tmp_path / "bad_source.root", tables))
    comparison = compare_source_exports(export, export, source_role="photon")
    assert not comparison["equal"]
    assert "invalid_reference_state" in comparison["failures"]


def test_ptr_resolution_is_event_qualified(tmp_path):
    tables = _fixture()
    for field in ("event", "ordinal"):
        tables["steps"][field][0] = tables["events"][field][1]
    tables["steps"]["index"][:] = 0
    tables["events"]["nsteps"][:] = 1
    tables["products"]["size"][[3, 6]] = 1
    report = audit_export(_write(tmp_path / "event_ptr.root", tables))
    assert "steps_particle_unresolved" in _codes(report)


@pytest.mark.parametrize("kind,options", [
    ("photon", {"photon_role": "photon", "photon_cap": 2}),
    ("particle", {"particle_role": "new", "particle_cap": 2}),
])
def test_caps_fail_at_equality(tmp_path, kind, options):
    report = audit_export(_write(tmp_path / "cap.root", _fixture()), **options)
    assert report["table_structure_valid"]
    assert kind + "_cap_headroom" in _codes(report)
    assert not report["audit_passed"]


def test_explicit_product_selection(tmp_path):
    path = _write(tmp_path / "cap.root", _fixture())
    report = audit_export(path, particle_tag="new::PROCESS", particle_cap=3,
                          photon_tag="photon::PROCESS", photon_cap=3)
    assert report["audit_passed"], report["failures"]
    assert "particle_product_selection" in _codes(audit_export(path, particle_role="missing"))
    with pytest.raises(ValueError, match="explicit role or tag"):
        audit_export(path, particle_cap=3)


def test_required_status_even_with_self_consistent_zero_counts(tmp_path):
    tables = _fixture()
    tables["statuses"] = _table("statuses", 0)
    tables["events"]["nstatuses"][:] = 0
    tables["events"]["nproducts"] -= 1
    keep_requirements = tables["requirements"]["kind"] != 4
    tables["requirements"] = {field: a[keep_requirements] for field, a in tables["requirements"].items()}
    keep = tables["products"]["kind"] != 4
    tables["products"] = {field: a[keep] for field, a in tables["products"].items()}
    path = _write(tmp_path / "source.root", tables)
    report = audit_export(path)
    assert report["table_structure_valid"]
    assert "required_status_missing" in _codes(report)
    assert audit_export(path, require_status=False)["audit_passed"]


def test_expected_events(tmp_path):
    report = audit_export(_write(tmp_path / "expected.root", _fixture()), expected_events=3)
    assert "expected_event_count" in _codes(report)


@pytest.mark.parametrize("change,code", [("signed_zero", "source_rows_px"), ("process", "source_provenance_process"), ("product", "source_provenance_product"), ("time", "source_rows_time")])
def test_source_exact_comparison(tmp_path, change, code):
    a, b = _fixture(), _fixture()
    if change == "signed_zero":
        b["steps"]["px"][1] = 0.0
    elif change == "process":
        b["products"]["process"][b["products"]["kind"] == 3] = "OTHER"
    elif change == "product":
        b["products"]["product"][b["products"]["kind"] == 3] = 31
        b["steps"]["product"][:] = 31
    else:
        b["steps"]["time"][0] = np.nextafter(b["steps"]["time"][0], np.inf)
    left = _write(tmp_path / "a.root", a)
    right = _write(tmp_path / "b.root", b)
    assert compare_source_exports(left, left, source_role="photon")["equal"]
    comparison = compare_source_exports(left, right, source_role="photon")
    assert not comparison["equal"]
    assert code in comparison["failures"]


def test_cli_json_exit_and_exclusive_output(tmp_path):
    path = _write(tmp_path / "input.root", _fixture())
    out = tmp_path / "audit.json"
    assert main(["--input", str(path), "--out", str(out), "--expected-events", "2"]) == 0
    assert json.loads(out.read_text())["physics_completeness"] is False
    with pytest.raises(FileExistsError):
        main(["--input", str(path), "--out", str(out)])
    bad = tmp_path / "bad.json"
    assert main(["--input", str(path), "--out", str(bad), "--photon-role", "photon", "--photon-cap", "2"]) == 2
    missing = _write(tmp_path / "missing.root", _fixture(), omit=("steps", "time"))
    assert main(["--input", str(missing), "--out", str(tmp_path / "missing.json")]) == 2


def test_collection_order_not_just_contiguous_set(tmp_path):
    tables = _fixture()
    tables["steps"]["index"][:] = [1, 0]
    report = audit_export(_write(tmp_path / "order.root", tables))
    assert "steps_collection_index_order" in _codes(report)


def test_missing_tree_and_non_ttree(tmp_path):
    tables = _fixture()
    del tables["volumes"]
    path = _write(tmp_path / "missing_tree.root", tables)
    with pytest.raises(SchemaError, match="missing TTree volumes"):
        read_export(path)
    with uproot.update(path) as root:
        root["export/volumes"] = {"run": np.array([1], dtype=np.uint32)}
    with pytest.raises(SchemaError, match="expected TTree"):
        read_export(path)


def test_metadata_must_be_strings(tmp_path):
    tables = _fixture()
    tables["products"]["process"] = np.zeros(10, dtype=np.uint32)
    path = _write(tmp_path / "string.root", tables, override=("products", "process", "uint32"))
    with pytest.raises(SchemaError, match="expected string"):
        read_export(path)


def test_missing_configured_collection_cannot_be_hidden_in_counts(tmp_path):
    tables = _fixture()
    keep = ~((tables["products"]["kind"] == 3) & (tables["products"]["event"] == 4))
    tables["products"] = {field: a[keep] for field, a in tables["products"].items()}
    tables["events"]["nproducts"][1] -= 1
    report = audit_export(_write(tmp_path / "missing_empty.root", tables))
    assert "required_product_role_coverage_3" in _codes(report)


def test_repeated_keys_across_subruns(tmp_path):
    tables = _fixture()
    tables["events"]["subrun"][1] = 9
    tables["products"]["subrun"][tables["products"]["event"] == 4] = 9
    tables["statuses"]["subrun"][1] = 9
    for name in ("products", "stages", "volumes"):
        for field, values in tables[name].items():
            extra = values[8:9].copy() if name == "products" else values.copy()
            if field == "subrun":
                extra[:] = 9
            tables[name][field] = np.concatenate([values, extra])
    tables["completion"]["subrun_products"][:] = 2
    report = audit_export(_write(tmp_path / "subrun.root", tables))
    assert report["audit_passed"], report["failures"]


def _add_empty_product(tables, source_row):
    products = tables["products"]
    for name, values in products.items():
        extra = values[source_row:source_row + 1].copy()
        if name == "event":
            extra[:] = 4
        elif name == "ordinal":
            extra[:] = 1
        elif name == "size":
            extra[:] = 0
        products[name] = np.concatenate([values, extra])
    tables["events"]["nproducts"][1] += 1


def test_daughter_order_exact_source_particle_comparison(tmp_path):
    a, b = _fixture(), _fixture()
    _add_empty_product(a, 1)
    _add_empty_product(b, 1)
    b["particles"]["ion_excitation"][2] = np.nextafter(np.float32(0), np.float32(1))
    left = _write(tmp_path / "source_a.root", a)
    right = _write(tmp_path / "source_b.root", b)
    report = compare_source_exports(left, right, source_role="input", source_kind=1)
    assert not report["equal"]
    assert "source_rows_ion_excitation" in report["failures"]


def test_generator_source_comparison(tmp_path):
    tables = _fixture()
    path = _write(tmp_path / "gen.root", tables)
    assert compare_source_exports(path, path, source_role="generator", source_kind=2)["equal"]


def test_source_selection_fails_even_if_both_missing(tmp_path):
    path = _write(tmp_path / "source.root", _fixture())
    report = compare_source_exports(path, path, source_role="absent")
    assert not report["equal"]
    assert "source_product_selection" in report["failures"]


def test_daughter_source_bytes_and_order(tmp_path):
    a, b = _fixture(), _fixture()
    for tables in (a, b):
        _add_empty_product(tables, 0)
        keep = ~((tables["products"]["event"] == 4) & (tables["products"]["product"] == 10))
        keep[-1] = True
        tables["products"] = {field: values[keep] for field, values in tables["products"].items()}
        tables["events"]["nproducts"][1] -= 1
    b["daughters"]["daughter_key"][0] = 7
    left = _write(tmp_path / "daughter_a.root", a)
    right = _write(tmp_path / "daughter_b.root", b)
    report = compare_source_exports(left, right, source_role="new", source_kind=1)
    assert not report["equal"]
    assert "source_daughters_daughter_key" in report["failures"]


def test_duplicate_volume_key(tmp_path):
    tables = _fixture()
    for name, values in tables["volumes"].items():
        tables["volumes"][name] = np.concatenate([values, values[:1]])
    tables["stages"]["size"][0] = 2
    path = _write(tmp_path / "duplicate_volume.root", tables)
    assert "duplicate_volume_key" in _codes(audit_export(path))
    with pytest.raises(ValueError, match="duplicate volume identities"):
        lookup_volumes(path, run=1, subrun=2, product=50, stage=0, index=7)


@pytest.mark.parametrize("table,branch,dtype", [
    ("requirements", "kind", "uint64"),
    ("requirements", "expected_events", "uint32"),
    ("requirements", "ordinal_base", "int64"),
    ("storage", "persistent_bytes", "uint64"),
    ("storage", "export_bytes", "int32"),
    ("storage", "persistent", "uint32"),
    ("products", "produced_type", "uint32"),
])
def test_new_metadata_exact_schema(tmp_path, table, branch, dtype):
    tables = _fixture()
    if dtype != "string" and REQUIRED[table][branch] == "string":
        tables[table][branch] = np.zeros(len(tables[table][branch]), dtype=dtype)
    path = _write(tmp_path / "metadata.root", tables, override=(table, branch, dtype))
    with pytest.raises(SchemaError, match=table + r"\." + branch):
        read_export(path)


@pytest.mark.parametrize("table", ["requirements", "storage"])
def test_new_metadata_tables_mandatory(tmp_path, table):
    tables = _fixture()
    del tables[table]
    with pytest.raises(SchemaError, match="missing TTree " + table):
        read_export(_write(tmp_path / "missing.root", tables))


@pytest.mark.parametrize("field,index,value,code", [
    ("expected_events", 0, 0, "requirement_expected_events_positive"),
    ("expected_events", 0, 3, "requirement_event_count"),
    ("expected_events", 0, 3, "requirement_expected_events_consistent"),
    ("ordinal_base", 0, 1, "requirement_ordinal_base_consistent"),
    ("ordinal_base", 0, 1, "requirement_ordinal_base"),
    ("kind", 0, 99, "requirement_kind"),
    ("tag", 0, "", "requirement_tag"),
    ("tag", 0, ":instance", "requirement_tag"),
    ("tag", 0, "new:::PROCESS", "requirement_tag"),
])
def test_requirement_metadata_failures(field, index, value, code):
    tables = _fixture()
    tables["requirements"][field][index] = value
    report = audit_export(ReferenceExport(tables))
    assert not report["table_structure_valid"]
    assert code in _codes(report)


def test_requirements_all_positive_and_consistent_do_not_allow_wrong_count():
    tables = _fixture()
    tables["requirements"]["expected_events"][:] = 3
    report = audit_export(ReferenceExport(tables))
    assert "requirement_event_count" in _codes(report)


def test_required_source_kind_declarations_mandatory():
    tables = _fixture()
    keep = tables["requirements"]["kind"] != 1
    tables["requirements"] = {field: a[keep] for field, a in tables["requirements"].items()}
    assert "required_requirement_kinds" in _codes(audit_export(ReferenceExport(tables)))
    tables["requirements"] = _table("requirements", 0)
    assert "requirements_missing" in _codes(audit_export(ReferenceExport(tables)))


def test_duplicate_requirement():
    tables = _fixture()
    for field, values in tables["requirements"].items():
        tables["requirements"][field] = np.concatenate([values, values[:1]])
    assert "duplicate_requirement" in _codes(audit_export(ReferenceExport(tables)))


@pytest.mark.parametrize("tag", ["new", "new:", "new::", "new::PROCESS"])
def test_particle_requirements_use_label_not_discovered_role(tag):
    tables = _fixture()
    tables["products"]["role"][tables["products"]["kind"] == 1] = "discovered_simparticles"
    tables["requirements"]["tag"][0] = tag
    report = audit_export(ReferenceExport(tables))
    assert report["audit_passed"], report["failures"]


@pytest.mark.parametrize("tag", ["new:missing", "new::OTHER", "missing"])
def test_requirement_inputtag_mismatch(tag):
    tables = _fixture()
    tables["requirements"]["tag"][0] = tag
    assert "required_product_tag_coverage_1" in _codes(audit_export(ReferenceExport(tables)))


def test_requirement_instance_match():
    tables = _fixture()
    tables["products"]["instance"][tables["products"]["label"] == "new"] = "source"
    tables["requirements"]["tag"][0] = "new:source"
    assert audit_export(ReferenceExport(tables))["audit_passed"]


@pytest.mark.parametrize("kind", [1, 2, 3, 4, 5])
def test_required_product_missing_with_independently_consistent_ledgers(kind):
    tables = _fixture()
    # Remove the empty event-4 collection, or all SubRun geometry rows. No
    # collection-row or event-count contradiction should mask missing coverage.
    products = tables["products"]
    remove = (products["kind"] == kind) & (products["event"] == (0 if kind == 5 else 4))
    if kind == 4:
        keep = tables["statuses"]["event"] != 4
        tables["statuses"] = {field: a[keep] for field, a in tables["statuses"].items()}
        tables["events"]["nstatuses"][1] = 0
    if kind == 5:
        tables["stages"] = _table("stages", 0)
        tables["volumes"] = _table("volumes", 0)
        tables["completion"]["subrun_products"][:] = 0
    else:
        tables["events"]["nproducts"][1] -= 1
    tables["products"] = {field: a[~remove] for field, a in products.items()}
    report = audit_export(ReferenceExport(tables), require_status=False)
    assert f"required_product_tag_coverage_{kind}" in _codes(report)
    assert not report["audit_passed"]


def test_requirement_missing_vs_independent_configuration_remains_uncertified():
    tables = _fixture()
    # An unrecorded independent requirement cannot be inferred from these tables.
    report = audit_export(ReferenceExport(tables))
    assert report["audit_passed"]
    assert not report["physics_completeness"]
    assert "incomplete requirements" in report["requirements_scope"]
    assert "independent job configuration and required-product completeness" in report["external_checks_required"]


def test_nonzero_raw_null_optional_ptr_payload_is_preserved(tmp_path):
    tables = _fixture()
    for prefix in ("parent", "generator"):
        null = tables["particles"][prefix + "_state"] == 0
        tables["particles"][prefix + "_product"][null] = 999
        tables["particles"][prefix + "_key"][null] = np.iinfo(np.uint64).max
    export = read_export(_write(tmp_path / "raw_null.root", tables))
    assert audit_export(export)["audit_passed"]
    assert export["particles"]["parent_product"][0] == 999
    assert export["particles"]["parent_key"][0] == np.iinfo(np.uint64).max


@pytest.mark.parametrize("table,prefix", [("steps", "particle"), ("daughters", "daughter")])
def test_null_required_ptr_is_physics_failure(tmp_path, table, prefix):
    tables = _fixture()
    tables[table][prefix + "_state"][0] = 0
    tables[table][prefix + "_product"][0] = 999
    tables[table][prefix + "_key"][0] = np.iinfo(np.uint64).max
    report = audit_export(_write(tmp_path / "null_required.root", tables))
    assert report["table_structure_valid"], report["failures"]
    assert not report["physics_checks_passed"]
    assert table + "_" + prefix + "_null" in _codes(report)
    assert not report["audit_passed"]


def test_produced_runtime_type_nonempty_and_not_typedef():
    tables = _fixture()
    assert np.all(tables["products"]["type"] != tables["products"]["produced_type"])
    assert audit_export(ReferenceExport(tables))["audit_passed"]
    tables["products"]["produced_type"][0] = ""
    assert "empty_provenance_produced_type" in _codes(audit_export(ReferenceExport(tables)))


def test_source_comparison_preserves_exact_produced_type():
    a, b = _fixture(), _fixture()
    b["products"]["produced_type"][b["products"]["kind"] == 3] = "another::RuntimeType"
    result = compare_source_exports(ReferenceExport(a), ReferenceExport(b), source_role="photon")
    assert not result["equal"]
    assert "source_provenance_produced_type" in result["failures"]


@pytest.mark.parametrize("field,index,value,code", [
    ("persistent", 0, "float", "storage_persistent_positions;step_vectors;step_times;generator_states"),
    ("persistent_bytes", 1, 8, "storage_persistent_bytes_particle_momenta_times_energies_length_excitation;step_deposits_length;status_times"),
    ("export", 4, "Float64", "storage_export_product_id"),
    ("export_bytes", 5, 4, "storage_export_bytes_ptr_key"),
    ("persistent_bytes", 4, 4, "storage_schema_product_id"),
    ("persistent_bytes", 5, 4, "storage_schema_ptr_key"),
    ("persistent", 6, "unsigned", "storage_persistent_map_key;particle_id"),
    ("persistent_bytes", 6, 3, "storage_persistent_bytes_map_key;particle_id"),
    ("persistent_bytes", 7, 0, "storage_persistent_bytes_step_volume"),
    ("group", 7, "unknown", "storage_groups"),
    ("group", 7, "ptr_key", "duplicate_storage_group"),
])
def test_storage_mapping_failures(field, index, value, code):
    tables = _fixture()
    tables["storage"][field][index] = value
    report = audit_export(ReferenceExport(tables))
    assert not report["table_structure_valid"]
    assert code in _codes(report)


@pytest.mark.parametrize("index", range(8))
def test_every_storage_group_mandatory(index):
    tables = _fixture()
    tables["storage"] = {field: np.delete(a, index) for field, a in tables["storage"].items()}
    assert "storage_groups" in _codes(audit_export(ReferenceExport(tables)))


@pytest.mark.parametrize("parameter", ["expected_events", "photon_cap", "particle_cap"])
@pytest.mark.parametrize("bad", [True, False, np.bool_(True), 0, -1, 1.5, 2.0, "2"])
def test_event_and_cap_parameters_strict_positive_integers(parameter, bad):
    with pytest.raises(ValueError, match="positive integer"):
        audit_export(ReferenceExport(_fixture()), **{parameter: bad})


def test_numpy_integer_parameters_valid():
    report = audit_export(ReferenceExport(_fixture()), expected_events=np.int64(2),
                          photon_role="photon", photon_cap=np.uint64(3))
    assert report["audit_passed"]


@pytest.mark.parametrize("field,bad", [
    ("run", True), ("run", [1.0]), ("subrun", -1), ("stage", 2**32),
    ("run", 2**32), ("subrun", "2"), ("product", 2**64),
    ("product", -1), ("index", 7.5), ("index", np.inf),
    ("stage", np.nan), ("index", np.array([7], dtype=object)),
])
def test_volume_lookup_rejects_lossy_conversion(field, bad):
    options = dict(run=1, subrun=2, product=50, stage=0, index=7)
    options[field] = bad
    with pytest.raises(ValueError, match=field + " must contain integers"):
        lookup_volumes(ReferenceExport(_fixture()), **options)


def test_volume_lookup_accepts_in_range_integer_arrays_and_broadcasting():
    indices = lookup_volumes(ReferenceExport(_fixture()), run=np.uint32(1), subrun=2,
                             product=np.uint64(50), stage=np.array([[0], [1]], dtype=np.int16),
                             index=np.array([7, 900], dtype=np.uint64))
    np.testing.assert_array_equal(indices, [[0, -1], [-1, 1]])


@pytest.mark.parametrize("table,branch", [
    ("requirements", "tag"), ("requirements", "expected_events"),
    ("requirements", "ordinal_base"), ("storage", "group"),
    ("storage", "export_bytes"), ("products", "produced_type"),
])
def test_new_metadata_branches_mandatory(tmp_path, table, branch):
    with pytest.raises(SchemaError, match="missing branches"):
        read_export(_write(tmp_path / "missing.root", _fixture(), omit=(table, branch)))


def test_nonzero_ordinal_base_consistent_across_requirements_and_completion():
    tables = _fixture()
    tables["completion"]["ordinal_base"][:] = 123
    tables["requirements"]["ordinal_base"][:] = 123
    for name in ("events", "steps", "particles", "daughters", "generators", "statuses"):
        tables[name]["ordinal"] += 123
    tables["products"]["ordinal"][tables["products"]["scope"] == 0] += 123
    report = audit_export(ReferenceExport(tables))
    assert report["audit_passed"], report["failures"]
    tables["requirements"]["ordinal_base"][:] = 124
    assert "requirement_ordinal_base" in _codes(audit_export(ReferenceExport(tables)))


def test_source_ordinal_overflow_rejected():
    tables = _fixture()
    tables["completion"]["ordinal_base"][:] = np.iinfo(np.uint64).max
    tables["requirements"]["ordinal_base"][:] = np.iinfo(np.uint64).max
    assert "source_ordinal_overflow" in _codes(audit_export(ReferenceExport(tables)))


@pytest.mark.parametrize("group,table,field,value", [
    ("product_id", "steps", "particle_product", 256),
    ("ptr_key", "particles", "parent_key", 256),
    ("map_key;particle_id", "volumes", "index", 256),
    ("step_volume", "steps", "volume", 256),
])
def test_key_storage_width_must_represent_raw_values(group, table, field, value):
    tables = _fixture()
    selected = tables["storage"]["group"] == group
    tables["storage"]["persistent_bytes"][selected] = 1
    tables[table][field][0] = value
    assert "storage_value_range_" + group in _codes(audit_export(ReferenceExport(tables)))


def test_duplicate_key_storage_row_not_hidden_by_identical_metadata():
    tables = _fixture()
    for field, values in tables["storage"].items():
        tables["storage"][field] = np.concatenate([values, values[6:7]])
    report = audit_export(ReferenceExport(tables))
    assert "duplicate_storage_group" in _codes(report)
    assert not report["table_structure_valid"]


def test_processless_discovered_particle_requirement_allows_multiple_histories():
    tables = _fixture()
    _add_empty_product(tables, 5)
    tables["products"]["process"][-1] = "OLD_PROCESS"
    tables["products"]["product"][-1] = 99
    report = audit_export(ReferenceExport(tables))
    assert report["audit_passed"], report["failures"]
    assert not report["physics_completeness"]


def test_explicit_requirement_rejects_multiple_same_process_products():
    tables = _fixture()
    _add_empty_product(tables, 5)
    tables["products"]["product"][-1] = 99
    tables["requirements"]["tag"][0] = "new::PROCESS"
    assert "required_product_tag_coverage_1" in _codes(audit_export(ReferenceExport(tables)))
