"""Ancestry birth census tests; nearest photon is not interface ownership."""
import json

import numpy as np
import pytest

from beamfastsim.flash_response_origins import trace, summarize, read_tables, main


@pytest.fixture
def case():
    pid = np.uint64(2**63 + 11)
    e = dict(run=np.array([1, 1], np.uint64), subrun=np.array([7, 7], np.uint64),
             event=np.array([5, 6], np.uint64), ordinal=np.array([1000, 1001], np.uint64))
    a = dict(run=np.ones(6, np.uint64), subrun=np.full(6, 7, np.uint64), event=np.full(6, 5, np.uint64),
             product=np.full(6, pid, np.uint64), key=np.array([10, 20, 30, 40, 50, 60], np.uint64),
             pdg=np.array([2212, 11, 11, 22, 11, 11]), creation=np.array([56, 16, 56, 16, 13, 56]),
             stage=np.array([0, 0, 1, 1, 1, 1], np.uint32), start_volume=np.array([0, 0, 1, 2, 3, 1], np.uint32),
             parent_state=np.array([0, 1, 1, 1, 1, 1], np.uint32), parent_product=np.full(6, pid, np.uint64),
             parent_key=np.array([0, 10, 20, 30, 40, 20], np.uint64),
             energy=np.array([8000., 50., 49., 5., 3., 48.], np.float32),
             x=np.zeros(6), y=np.zeros(6), z=np.arange(6).astype(float), time=np.arange(6).astype(np.float32))
    s = dict(run=np.ones(2, np.uint64), subrun=np.full(2, 7, np.uint64), event=np.full(2, 5, np.uint64),
             product=np.full(2, pid + np.uint64(1), np.uint64), index=np.array([0, 1], np.uint64),
             particle_state=np.ones(2, np.uint32), particle_product=np.full(2, pid, np.uint64),
             particle_key=np.array([50, 60], np.uint64), time=np.array([10., 10.]),
             total_edep=np.array([.25, 1.], np.float32), nonionizing_edep=np.zeros(2, np.float32))
    p = dict(run=np.ones(3, np.uint64), subrun=np.full(3, 7, np.uint64), event=np.array([5, 5, 0], np.uint64),
             product=pid + np.arange(3, dtype=np.uint64), kind=np.array([1, 3, 5]), scope=np.array([0, 0, 1]),
             label=np.array(["g4run"]*3), instance=np.array(["", "tracker", ""]), size=np.array([6, 2, 2], np.uint64))
    v = dict(run=np.ones(4, np.uint64), subrun=np.full(4, 7, np.uint64), product=np.full(4, pid+np.uint64(2), np.uint64),
             stage=np.array([0, 1, 1, 1], np.uint32), index=np.array([0, 1, 2, 3], np.uint64),
             name=np.array(["ProductionTarget", "TS3Vacuum", "TargetFoil_00", "TrackerGas"]),
             material=np.array(["G4_W", "Vacuum", "G4_Al", "Gas"]))
    r = {"event_" + k: v.copy() for k, v in e.items()}
    r.update(tracker_step_row=np.array([0, 1]), tracker_ownership_status=np.array([1, 0]),
             tracker_ionizing_mev=np.array([.25, 1.]), event_owned_ionizing_mev=np.array([.25, 0.]),
             event_residual_ionizing_mev=np.array([1., 0.]), event_unknown_ionizing_mev=np.array([0., 0.]))
    return dict(events=e, particles=a, steps=s, products=p, volumes=v), r


def test_direct_charged_and_target_photon_distinguished(case):
    t, r = case
    out = trace(t)
    np.testing.assert_array_equal(out["nearest_new_photon_particle_row"], [3, -1])
    np.testing.assert_array_equal(out["stage_entry_particle_row"], [2, 5])
    np.testing.assert_array_equal(out["leaf_is_stage_entry"], [False, True])
    np.testing.assert_array_equal(out["root_pdg"], [2212, 2212])
    assert out["nearest_new_photon_birth_volume"][0] == "TargetFoil_00"
    assert out["stage_entry_birth_volume"].tolist() == ["TS3Vacuum", "TS3Vacuum"]
    assert not out["ancestry_unresolved"].any()
    report = summarize(t, r, out)
    assert report["input_events"] == 2
    assert report["top_residual_events"]["event"] == [5]
    assert report["top_residual_events"]["largest_fraction"] == 1.
    assert report["top_residual_events"]["sum_mev"] == 1.


def test_pre_stage_photon_not_new_stage(case):
    t, _ = case
    t["particles"]["pdg"][1] = 22
    out = trace(t)
    assert out["nearest_photon_particle_row"][1] == 1
    assert out["nearest_new_photon_particle_row"][1] == -1
    assert out["nearest_photon_birth_volume"][1] == "ProductionTarget"


@pytest.mark.parametrize("state", [1, 2, 3])
def test_missing_ancestry_not_reported_as_root(case, state):
    t, _ = case
    t["particles"]["parent_state"][3] = state
    t["particles"]["parent_key"][3] = 999
    out = trace(t)
    assert out["ancestry_unresolved"][0]
    assert not out["root_present"][0]


def test_cycle_and_depth_limit(case):
    t, _ = case
    out = trace(t, max_depth=2)
    assert out["ancestry_unresolved"].all()
    t["particles"]["parent_state"][0] = 1
    t["particles"]["parent_key"][0] = 50
    out = trace(t, max_depth=8)
    assert out["ancestry_unresolved"].all()


def test_full_identity_volume_stage_and_subrun_join(case):
    t, _ = case
    assert t["particles"]["product"][0] > 2**63
    t["volumes"]["subrun"][2] = 8
    out = trace(t)
    assert out["nearest_new_photon_birth_volume"][0] == "<unresolved>"
    assert not out["nearest_new_photon_birth_known"][0]
    t["particles"]["event"][3] = 6
    out = trace(t)
    assert out["ancestry_unresolved"][0]


def test_duplicate_particles_raise(case):
    t, _ = case
    t["particles"]["key"][1] = t["particles"]["key"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        trace(t)


def test_empty_tracker_does_not_drop_event_ledger(case):
    t, r = case
    t["steps"] = {k: v[:0] for k, v in t["steps"].items()}
    r["tracker_ownership_status"] = r["tracker_ownership_status"][:0]
    r["tracker_ionizing_mev"] = r["tracker_ionizing_mev"][:0]
    for k in ("owned", "residual", "unknown"):
        r["event_"+k+"_ionizing_mev"][:] = 0
    out = trace(t)
    assert out["trace_hit_row"].size == 0
    assert summarize(t, r, out)["input_events"] == 2


def test_root_response_validation_and_cli(case, tmp_path):
    import uproot

    t, r = case
    path = tmp_path / "input.root"
    response = tmp_path / "response.npz"
    out = tmp_path / "origins.json"
    with uproot.recreate(path) as root:
        for name, table in t.items():
            root.mktree("export/"+name, table)
    np.savez_compressed(response, **r)
    assert main(["--input", str(path), "--response", str(response), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["tracker_steps"] == 2
    assert out.with_suffix(".npz").exists()
    r["event_event"][1] = 99
    with pytest.raises(ValueError, match="ledger"):
        read_tables(path, r)
    r["event_event"][1] = 6
    r["tracker_ionizing_mev"][0] = 5.
    with pytest.raises(ValueError, match="energies"):
        read_tables(path, r)


def test_earliest_photon_and_foil_ancestry_not_only_nearest(case):
    t, _ = case
    t["particles"]["pdg"][2] = 22
    out = trace(t)
    assert out["nearest_new_photon_particle_row"][0] == 3
    assert out["earliest_new_photon_particle_row"][0] == 2
    np.testing.assert_array_equal(out["has_new_foil_photon_ancestor"], [True, False])
    np.testing.assert_array_equal(out["has_new_foil_born_ancestor"], [True, False])


def test_absence_unknown_when_stage_or_volume_history_missing(case):
    t, _ = case
    t["volumes"]["subrun"][2] = 8
    out = trace(t)
    assert not out["has_new_foil_photon_ancestor"][0]
    assert out["ancestry_metadata_unknown"][0]
    assert not out["foil_ancestry_complete"][0]
    assert out["foil_ancestry_complete"][1]
    t["products"]["size"][-1] = 0
    out = trace(t)
    assert out["ancestry_metadata_unknown"].all()
    assert not out["foil_ancestry_complete"].any()


def test_absence_unknown_for_missing_parent_even_with_known_volumes(case):
    t, _ = case
    t["particles"]["parent_key"][5] = 999
    out = trace(t)
    assert not out["has_new_foil_photon_ancestor"][1]
    assert not out["foil_ancestry_complete"][1]
