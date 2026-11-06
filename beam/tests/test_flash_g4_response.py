"""Exclusive raw response, zero exposures, and explicit residual accounting."""
import json
import numpy as np
import pytest

from beamfastsim.flash_g4_frontier import classify, FRONTIER, COVERED
from beamfastsim.flash_g4_response import attribute, OWNED, RESIDUAL, UNKNOWN
from test_flash_g4_frontier import chain


@pytest.fixture
def case(chain):
    t, p, i = chain
    a = t["particles"]
    pid = a["product"][0]
    p["recorded_steps"] = np.ones(2, int)
    p["report_json"] = np.asarray(json.dumps(dict.fromkeys((
        "new_photons_without_volume", "photons_without_stage", "photons_after_destination_stage",
        "particles_without_event", "steps_without_event", "particles_without_product",
        "steps_without_product", "unresolved_photon_step_pointers"), 0)))
    t["events"] = dict(run=np.array([1, 1], np.uint64), subrun=np.array([7, 7], np.uint64),
                       event=np.array([1, 2], np.uint64), ptr_failures=np.zeros(2, np.uint64))
    t["products"] = dict(run=np.ones(8, np.uint64), subrun=np.full(8, 7, np.uint64),
                         event=np.repeat([1, 2], 4).astype(np.uint64),
                         product=pid + np.tile(np.arange(4, dtype=np.uint64), 2),
                         scope=np.zeros(8, int), kind=np.tile([1, 3, 3, 4], 2),
                         label=np.full(8, "g4run"), instance=np.tile(["", "photonSteps", "tracker", ""], 2),
                         size=np.array([3, 0, 2, 1, 0, 0, 0, 1], np.uint64))
    t["steps"] = dict(run=np.ones(2, np.uint64), subrun=np.full(2, 7, np.uint64),
                      event=np.ones(2, np.uint64), product=np.full(2, pid + np.uint64(2), np.uint64),
                      index=np.arange(2, dtype=np.uint64), particle_state=np.ones(2, np.uint32),
                      particle_product=np.full(2, pid, np.uint64), particle_key=np.array([30, 20], np.uint64),
                      time=np.array([30., 30.]), total_edep=np.array([3., 2.], np.float32),
                      nonionizing_edep=np.array([1., 0.], np.float32))
    t["statuses"] = {k: v.copy() for k, v in t["events"].items() if k != "ptr_failures"}
    t["statuses"].update(product=np.full(2, pid + np.uint64(3), np.uint64),
                         status=np.zeros(2, int), overflow=np.zeros(2, int),
                         killed_step=np.zeros(2, int), killed_field=np.zeros(2, int))
    t["completion"] = dict(complete=np.array([1]), events=np.array([2]), subrun_products=np.array([1]))
    return t, p, i


def run(case):
    t, p, i = case
    return attribute(t, p, i, classify(t, p, i))


def test_nested_exposures_count_deposits_once(case):
    out = run(case)
    np.testing.assert_array_equal(out["tracker_owner_photon_row"], [0, 0])
    np.testing.assert_array_equal(out["photon_assigned_ionizing_mev"], [4., 0.])
    np.testing.assert_array_equal(out["photon_response_known"], [True, False])
    np.testing.assert_array_equal(out["event_tracker_ionizing_mev"], [4., 0.])
    np.testing.assert_array_equal(out["event_owned_ionizing_mev"], [4., 0.])
    assert not out["event_unknown_ionizing_mev"].any()


def test_early_branch_not_owned_by_late_exiting_ancestor(case):
    case[0]["particles"]["time"][1] = 5.
    out = run(case)
    np.testing.assert_array_equal(out["tracker_ownership_status"], [OWNED, RESIDUAL])
    np.testing.assert_array_equal(out["photon_assigned_ionizing_mev"], [0., 2.])
    assert out["photon_response_known"].all()
    np.testing.assert_array_equal(out["event_residual_ionizing_mev"], [2., 0.])


@pytest.mark.parametrize("time", [5., 10., 11.])
def test_same_track_step_not_definitely_after_crossing_is_unknown(case, time):
    case[0]["steps"]["particle_key"][0] = 10
    case[0]["steps"]["time"][0] = time
    out = run(case)
    assert out["tracker_ownership_status"][0] == UNKNOWN
    assert not out["photon_response_known"].any()
    assert out["event_unknown_ionizing_mev"][0] == 2.


def test_unknown_step_or_ancestor_preserved_in_residual_ledger(case):
    case[0]["steps"]["particle_key"][0] = 999
    out = run(case)
    np.testing.assert_array_equal(out["tracker_ownership_status"], [UNKNOWN, OWNED])
    assert not out["photon_response_known"].any()
    total = out["event_owned_ionizing_mev"] + out["event_unknown_ionizing_mev"] + out["event_residual_ionizing_mev"]
    np.testing.assert_array_equal(total, out["event_tracker_ionizing_mev"])


def test_genuine_zero_and_missing_product_are_different(case):
    t, p, i = case
    t["steps"] = {k: v[:0] for k, v in t["steps"].items()}
    t["products"]["size"][2] = 0
    out = run(case)
    assert out["photon_response_known"][0] and out["photon_assigned_ionizing_mev"][0] == 0
    t["products"]["instance"][2] = "absent"
    out = run(case)
    assert not out["photon_response_known"].any()
    assert not out["event_tracker_product_complete"][0]


@pytest.mark.parametrize("field", ["status", "overflow", "killed_step", "killed_field"])
def test_failed_transport_does_not_certify_zero_response(case, field):
    case[0]["statuses"][field][0] = 1
    out = run(case)
    assert not out["photon_response_known"].any()
    assert not out["event_recording_checks_passed"][0]
    assert out["event_tracker_ionizing_mev"][0] == 4.


def test_ptr_failure_missing_completion_and_size_mismatch(case):
    case[0]["events"]["ptr_failures"][0] = 1
    assert not run(case)["photon_response_known"].any()
    case[0]["events"]["ptr_failures"][0] = 0
    case[0]["completion"]["complete"][0] = 0
    assert not run(case)["photon_response_known"].any()
    case[0]["completion"]["complete"][0] = 1
    case[0]["products"]["size"][1] = 1_000_000
    assert not run(case)["photon_response_known"].any()


def test_unresolved_birth_records_quarantine_event_exposure(case):
    case[1]["recorded_steps"][1] = 0
    out = run(case)
    assert not out["photon_response_known"].any()
    assert not out["event_exposure_checks_passed"][0]


def test_invalid_edep_fails_instead_of_disappearing(case):
    case[0]["steps"]["total_edep"][0] = np.nan
    with pytest.raises(ValueError, match="ionization"):
        run(case)


@pytest.mark.parametrize("name", ["new_photons_without_volume", "photons_without_stage"])
def test_missing_exposure_metadata_never_becomes_definite_residual(case, name):
    report = json.loads(str(case[1]["report_json"]))
    report[name] = 1
    case[1]["report_json"] = np.asarray(json.dumps(report))
    out = run(case)
    assert np.all(out["tracker_ownership_status"] == UNKNOWN)
    assert not out["photon_response_known"].any()
    assert out["event_unknown_ionizing_mev"][0] == 4.


def test_legitimate_nonfoil_photons_do_not_poison_exposure(case):
    report = json.loads(str(case[1]["report_json"]))
    report["new_photons_not_mapped_to_foil"] = 1000
    case[1]["report_json"] = np.asarray(json.dumps(report))
    assert run(case)["photon_response_known"][0]


def test_root_response_cli_preserves_empty_events(tmp_path, monkeypatch):
    import uproot
    from test_flash_g4_photons import sample
    from beamfastsim import flash_geometry
    from beamfastsim.flash_g4_response import main

    tables, geom = sample.__wrapped__()
    n = len(tables["particles"]["key"])
    tables["particles"].update(time=np.zeros(n, np.float32), end_time=np.full(n, 200., np.float32),
                               end_defined=np.ones(n, np.uint32))
    ns = len(tables["steps"]["index"])
    tables["steps"].update(total_edep=np.zeros(ns, np.float32), nonionizing_edep=np.zeros(ns, np.float32))
    tables["statuses"] = {k: np.array([], np.uint64) for k in
                          ("run", "subrun", "event", "product", "status", "overflow", "killed_step", "killed_field")}
    tables["completion"] = dict(complete=np.array([1]), events=np.array([4]), subrun_products=np.array([1]))
    path, out = tmp_path / "input.root", tmp_path / "response.npz"
    with uproot.recreate(path) as root:
        for name, columns in tables.items():
            root.mktree("export/" + name, columns)
    monkeypatch.setattr(flash_geometry, "from_offline", lambda *args: geom)
    args = ["--input", str(path), "--out", str(out), "--geom", "unused",
            "--offline-root", "unused", "--events", "4"]
    assert main(args) == 0
    with np.load(out, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["event_event"], [1, 2, 3, 4])
        assert not saved["photon_response_known"].any()
        assert len(saved["key"]) == 7
        report = json.loads(str(saved["response_report_json"]))
        assert report["failed_recording_events"] == 4
        assert not report["detector_response_valid_S2"]
        assert report["max_event_accounting_error_mev"] == 0.
    assert main(args[:-1] + ["5"]) == 1
