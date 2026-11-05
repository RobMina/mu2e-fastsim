"""Numeric export reconciliation on synthetic complete and broken products."""
import numpy as np
import pytest

from calofastsim.export_audit import audit


def rows(fields, values):
    return {name: np.asarray(value) for name, value in zip(fields.split(), values)}


def fixture():
    events = rows("run subrun event g4_status showers_present transport_particles_present generated_present unresolved_links n_particles n_generated n_showers n_steps",
                  [[1], [0], [1], [0], [True], [True], [True], [0], [1], [1], [0], [0]])
    products = rows("run subrun event product_id kind size present",
                    [[1, 1, 1, 1, 1], [0] * 5, [1] * 5, [7, 8, 9, 10, 11], [-1, -2, -3, 0, -4], [1, 1, 0, 0, 0], [True] * 5])
    particles = rows("run subrun event sim_product_id sim_key parent_valid parent_product_id parent_key generator_valid generator_product_id generator_key",
                     [[1], [0], [1], [7], [1], [False], [0], [0], [False], [0], [0]])
    generated = rows("run subrun event product_id index parent_valid parent_product_id parent_key",
                     [[1], [0], [1], [8], [0], [True], [7], [1]])
    empty = {name: np.array([], dtype=np.int64) for name in ("run", "subrun", "event", "product_id", "index")}
    sim_empty = {**empty, "sim_valid": np.array([], dtype=bool),
                 "sim_product_id": np.array([], dtype=np.int64), "sim_key": np.array([], dtype=np.int64)}
    return dict(events=events, products=products, particles=particles, generated=generated,
                showers={**sim_empty}, steps={**sim_empty}, genparticles={**empty})


def test_valid_export_is_not_incident_or_zero_certification():
    result = audit(fixture(), expected_events=1, expected_step_products=1)
    assert result["numeric_export_reconciled"] is True
    assert result["status_zero_parent_forest_acyclic"] is True
    assert result["complete_incident_census"] is False
    assert result["zero_response_certified"] is False


@pytest.mark.parametrize("change", ["missing_event", "unknown_event", "wrong_count", "registry_count", "unresolved", "missing_showers", "broken_link", "duplicate_particle"])
def test_corrupt_export_refused(change):
    tables = fixture()
    if change == "missing_event":
        tables["events"] = {name: array[:0] for name, array in tables["events"].items()}
    elif change == "unknown_event":
        tables["generated"]["event"][0] = 2
    elif change == "wrong_count":
        tables["events"]["n_showers"][0] = 1
    elif change == "registry_count":
        tables["products"]["size"][1] = 2
    elif change == "unresolved":
        tables["events"]["unresolved_links"][0] = 1
    elif change == "missing_showers":
        tables["events"]["showers_present"][0] = False
    elif change == "broken_link":
        tables["generated"]["parent_product_id"][0] = 99
    else:
        tables["particles"] = {name: np.repeat(array, 2) for name, array in tables["particles"].items()}
        tables["events"]["n_particles"][0] = 2
        tables["products"]["size"][0] = 2
    with pytest.raises(ValueError):
        audit(tables, expected_events=1, expected_step_products=1)


def test_retained_failure_not_validated_as_zero():
    tables = fixture()
    tables["events"]["g4_status"][0] = 10
    tables["events"]["showers_present"][0] = False
    tables["products"]["present"][2] = False
    tables["products"]["product_id"][2] = 0
    result = audit(tables, expected_events=1, expected_step_products=1)
    assert result["status_counts"] == {"10": 1}
    assert result["zero_response_certified"] is False


def test_resolved_self_parent_is_still_a_cycle():
    tables = fixture()
    tables["particles"]["parent_valid"][0] = True
    tables["particles"]["parent_product_id"][0] = 7
    tables["particles"]["parent_key"][0] = 1
    with pytest.raises(ValueError, match="cyclic"):
        audit(tables, expected_events=1, expected_step_products=1)


@pytest.mark.parametrize("change", ["missing_step_registry", "absent_step", "extra_kind", "duplicate_kind", "empty_shower_absent_registry", "noncontiguous_index"])
def test_missing_empty_products_and_bad_indices_refused(change):
    tables = fixture()
    if change == "missing_step_registry":
        tables["products"] = {name: array[[0, 1, 2, 4]] for name, array in tables["products"].items()}
    elif change == "absent_step":
        tables["products"]["present"][3] = False
        tables["products"]["product_id"][3] = 0
    elif change == "extra_kind":
        tables["products"]["kind"][3] = 1
    elif change == "duplicate_kind":
        tables["products"] = {name: np.append(array, array[3]) for name, array in tables["products"].items()}
        tables["products"]["product_id"][-1] = 12
    elif change == "empty_shower_absent_registry":
        tables["products"] = {name: array[[0, 1, 3, 4]] for name, array in tables["products"].items()}
    else:
        tables["generated"]["index"][0] = 1
    with pytest.raises(ValueError):
        audit(tables, expected_events=1, expected_step_products=1)


def test_configured_step_product_count_cannot_be_inferred_from_export():
    with pytest.raises(ValueError, match="configured raw-step"):
        audit(fixture(), expected_events=1, expected_step_products=2)
    with pytest.raises(ValueError, match="positive configured"):
        audit(fixture(), expected_events=1, expected_step_products=0)


def test_generator_links_are_to_genparticles_not_stageparticles():
    tables = fixture()
    tables["particles"]["generator_valid"][0] = True
    tables["particles"]["generator_product_id"][0] = 8
    with pytest.raises(ValueError, match="generator identity"):
        audit(tables, expected_events=1, expected_step_products=1)
    tables["particles"]["generator_product_id"][0] = 11
    tables["products"]["size"][4] = 1
    tables["genparticles"] = rows("run subrun event product_id index", [[1], [0], [1], [11], [0]])
    assert audit(tables, expected_events=1, expected_step_products=1)["numeric_export_reconciled"] is True
