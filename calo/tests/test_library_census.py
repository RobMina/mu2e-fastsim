"""Multistage parent reseating and response zeros cannot imply incident completeness."""
import numpy as np
import pytest

from test_controls import load_script


def fixture():
    events = dict(run=np.array([1]), subrun=np.array([0]), event=np.array([1]),
                  g4_status=np.array([0]), n_showers=np.array([0]), n_steps=np.array([1]))
    registry = dict(run=np.array([1]), subrun=np.array([0]), event=np.array([1]),
                    product_id=np.array([7]), kind=np.array([-1]), size=np.array([3]))
    branch = np.array(["mu2e::SimParticlemv_g4run__CaloLibraryCE."])
    particles = dict(run=np.ones(4, dtype=int), subrun=np.zeros(4, dtype=int), event=np.ones(4, dtype=int),
                     sim_product_id=np.array([3, 7, 7, 7]), sim_key=np.array([5, 5, 6, 7]),
                     parent_valid=np.array([False, False, True, True]),
                     parent_product_id=np.array([0, 0, 7, 7]), parent_key=np.array([0, 0, 5, 6]),
                     pdg=np.array([13, 13, 11, -11]))
    generated = dict(run=np.array([1]), subrun=np.array([0]), event=np.array([1]), product_id=np.array([8]),
                     index=np.array([0]), pdg=np.array([11]), parent_valid=np.array([True]),
                     parent_product_id=np.array([3]), parent_key=np.array([5]))
    steps = dict(run=np.array([1]), subrun=np.array([0]), event=np.array([1]),
                 sim_product_id=np.array([7]), sim_key=np.array([6]), kind=np.array([0]), volume=np.array([73]))
    showers = {name: np.array([], dtype=int) for name in ("run", "subrun", "event", "sim_product_id", "sim_key", "crystal")}
    showers["energy_g4"] = np.array([], dtype=np.float32)
    particles["creation"] = np.full(4, 42, dtype=int)
    generated["creation"] = np.array([42])
    for target, source in (("start_x", "position_x"), ("start_y", "position_y"), ("start_z", "position_z"),
                           ("start_px", "momentum_px"), ("start_py", "momentum_py"), ("start_pz", "momentum_pz"),
                           ("start_e", "momentum_e"), ("start_time", "time")):
        particles[target] = np.zeros(4)
        generated[source] = np.zeros(1)
    return [events, registry, branch, particles, generated, steps, showers]


def test_reseated_parent_is_not_key_only_cross_product_matching():
    result = load_script("census_library").summarize(*fixture())
    assert result["transport_roots_by_generated_parent_identity"] == 1
    assert result["empty_shower_events_with_direct_root_boundary_rows"] == 1
    assert result["zero_response_certified"] is False
    assert result["complete_incident_census"] is False


@pytest.mark.parametrize("change", ["failure", "missing_source_parent", "missing_reseated_parent", "two_roots", "wrong_event_registry", "wrong_generated_pdg"])
def test_incomplete_or_ambiguous_root_selection_refused(change):
    args = fixture()
    if change == "failure":
        args[0]["g4_status"][0] = 1
    elif change == "missing_source_parent":
        args[4]["parent_product_id"][0] = 99
    elif change == "missing_reseated_parent":
        args[3]["sim_key"][1] = 9
    elif change == "two_roots":
        args[3]["pdg"][3] = 11
        args[3]["parent_key"][3] = 5
    elif change == "wrong_event_registry":
        args[1]["event"][0] = 2
    else:
        args[4]["pdg"][0] = -11
    with pytest.raises(ValueError):
        load_script("census_library").summarize(*args)


def test_shower_positive_secondary_boundary_gap_retained_with_exact_ancestry():
    args = fixture()
    args[0]["n_showers"][0] = 1
    args[5]["sim_key"][0] = 7
    args[6] = {name: np.array([value]) for name, value in
               dict(run=1, subrun=0, event=1, sim_product_id=7, sim_key=7, crystal=0, energy_g4=2.).items()}
    result = load_script("census_library").summarize(*args)
    assert result["shower_events_without_direct_root_boundary_rows"] == 1
    gap = result["shower_positive_boundary_gaps"]
    assert gap["event_columns"]["event"] == [1]
    assert gap["root_particle_columns"]["sim_key"] == [6]
    assert gap["shower_ancestor_particle_columns"]["sim_key"] == [7]
    assert gap["shower_ancestor_particle_columns"]["has_exact_boundary_row"] == [True]
    assert gap["summary"]["events_without_any_boundary_row"] == 0
    assert gap["summary"]["shower_ancestors_with_exact_boundary_row"] == 1
    assert gap["boundary_step_columns"]["particle_pdg"] == [-11]
    assert gap["boundary_step_columns"]["descends_from_generated_root"] == [True]
    assert result["complete_incident_census"] is False


def test_empty_boundary_gap_is_well_formed():
    report = load_script("census_library").summarize(*fixture())["shower_positive_boundary_gaps"]
    assert report["event_columns"]["event"] == []
    assert report["boundary_step_columns"]["particle_pdg"] == []


def test_upstream_boundary_gap_is_not_labelled_generated_root_descendant():
    args = fixture()
    args[0]["n_showers"][0] = 1
    args[5]["sim_key"][0] = 5
    result = load_script("census_library").summarize(*args)
    assert result["shower_positive_boundary_gaps"]["boundary_step_columns"]["descends_from_generated_root"] == [False]


def test_shower_gap_without_any_boundary_stays_explicit():
    args = fixture()
    args[0]["n_showers"][0] = 1
    args[5]["volume"][0] = 13
    report = load_script("census_library").summarize(*args)["shower_positive_boundary_gaps"]
    assert report["event_columns"]["event"] == [1]
    assert report["summary"]["events_without_any_boundary_row"] == 1
    assert report["summary"]["boundary_rows"] == 0
