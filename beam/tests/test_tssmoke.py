"""Raw single-stage reference bookkeeping and opt-in configuration controls."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

from beamfastsim import tssmoke


def particles():
    return {"run": np.ones(5, dtype="i8"), "subrun": np.ones(5, dtype="i8"),
            "evt": np.array([1, 1, 1, 1, 2]), "trk": np.array([1, 2, 3, 4, 1]),
            "parent_id": np.array([-1, 1, 2, 1, -1]),
            "parent_pdg": np.array([0, 2212, -211, 2212, 0]),
            "pdg": np.array([2212, -211, 13, -13, 2212])}


def hits(tracks, pdgs, times, pz=None):
    n = len(tracks)
    a = {k: np.zeros(n, dtype="f4") for k in ("x", "y", "z", "px", "py")}
    a.update(run=np.ones(n, dtype="f4"), subrun=np.ones(n, dtype="f4"),
             evt=np.ones(n, dtype="f4"), trk=np.array(tracks, dtype="f4"),
             pdg=np.array(pdgs, dtype="f4"), sid=np.ones(n, dtype="f4"),
             time=np.array(times, dtype="f4"),
             pz=np.ones(n, dtype="f4") if pz is None else np.array(pz, dtype="f4"))
    return a


def test_loss_empty_event_decay_and_repeated_forward_records():
    report = tssmoke.census(particles(), hits([2, 4, 2], [-211, -13, -211], [10, 9, 20]),
                           hits([3], [13], [50]))
    assert report["primary_proton_events"] == 2
    assert report["coll1_mu_pi_forward_rows"] == 3
    assert report["coll1_mu_pi_forward_tracks"] == 2
    assert report["tracks_with_repeated_forward_records"] == 1
    assert report["source_tracks_without_ds_mu_pi_descendant"] == 1
    assert report["source_tracks_with_ds_mu_pi_descendant"] == 1
    assert report["events_without_forward_mu_pi_in_ntpart"] == 1
    assert report["ds_mu_pi_linked_to_forward_source"] == 1
    assert report["source_species"]["-211"] == {"inputs": 1, "with_ds_mu_pi": 1}


def test_no_source_or_endpoint_and_backward_hit():
    report = tssmoke.census(particles(), hits([2], [-211], [10], [-1]), hits([], [], []))
    assert report["coll1_mu_pi_all_direction_rows"] == 1
    assert report["coll1_mu_pi_forward_tracks"] == 0
    assert report["events_without_forward_mu_pi_in_ntpart"] == 2


def test_missing_identity_rejected():
    with pytest.raises(ValueError, match="hit lacks"):
        tssmoke.census(particles(), hits([99], [13], [10]), hits([], [], []))
    p = particles()
    p["parent_id"][2] = 99
    with pytest.raises(ValueError, match="parent key"):
        tssmoke.census(p, hits([], [], []), hits([], [], []))


def test_unrelated_nuclear_pdg_rounding_does_not_block_mu_pi_census():
    p = particles()
    p["pdg"][3] = 1000010020
    report = tssmoke.census(p, hits([2, 4], [-211, 1000010020], [10, 9]), hits([3], [13], [50]))
    assert report["coll1_mu_pi_forward_tracks"] == 1
    assert report["ds_mu_pi_linked_to_forward_source"] == 1


def test_unmatched_ds_and_noncausal_time_reported():
    report = tssmoke.census(particles(), hits([2], [-211], [10]), hits([3, 4], [13, -13], [5, 20]))
    assert report["linked_ds_rows_preceding_source"] == 1
    assert report["ds_mu_pi_without_forward_source"] == 1


def test_cycle_rejected():
    p = particles()
    p["parent_id"][1:3] = [3, 2]
    with pytest.raises(ValueError, match="cyclic"):
        tssmoke.census(p, hits([], [], []), hits([3], [13], [20]))


@pytest.mark.parametrize("compact", [False, True])
def test_highest_cycle_and_finished_art_count(tmp_path, compact):
    art = tmp_path / "sim.test.raw.smoke.000001_00000001.art"
    ntuple = tmp_path / "inspect.root"
    with uproot.recreate(art) as f:
        f.mktree("Events", {"event": np.array([1])})
    with uproot.update(art) as f:
        f.mktree("Events", {"event": np.array([1, 2])})
    with uproot.recreate(ntuple) as f:
        f.mktree("tsinspect/ntpart", particles())
        f.mktree("tsinspect/ntvd", hits([2, 4], [-211, -13], [10, 9]))
        f.mktree("tsinspect/nttvd", hits([3], [13], [50]))
    report = tssmoke.inspect(art, ntuple, 2, compact=compact)
    assert report["checks_passed"]
    assert report["finished_art_events"] == 2
    assert report["art_files"][0]["tree"] == "Events;2"
    assert not tssmoke.inspect(art, ntuple, 3, compact=compact)["checks_passed"]


def test_raw_configuration_keeps_unfiltered_stock_products():
    root = Path(tssmoke.__file__).parent
    fcl = (root / "tsrun1a_raw.fcl").read_text()
    assert '#include "Production/JobConfig/beam/POT.fcl"' in fcl
    assert "[generate, genCounter, g4run]" in fcl
    assert "SelectEvents" not in fcl
    assert '"keep *_g4run_*_*"' in fcl
    assert '"g4run:Beam"' in fcl and '"g4run:virtualdetector"' in fcl
    assert "geom_run1_a.txt" in fcl and "bfgeom_no_ds_v01.txt" in fcl
    assert "saveAllPDG: false" in fcl
    assert "savePDG: [mu_minus, mu_plus, pi_minus, pi_plus, proton]" in fcl
    assert "navigateToParent: false" in fcl
    assert "tvdDropPDG" not in fcl
    assert "Mu2eG4CommonCut" not in fcl and "Mu2eG4SteppingOnlyCut" not in fcl
    subprocess.run(["bash", "-n", root / "tsrun1a_smoke.sbatch"], check=True)


def test_compact_omitted_parent_before_source_and_all_species_beam():
    p = particles()
    p["parent_id"][1] = 99
    p["parent_pdg"][1] = 2112
    report = tssmoke.census(p, hits([2], [-211], [10]),
                            hits([3, 88], [13, 11], [50, 40]), compact=True)
    assert report["omitted_parent_links"] == 1
    assert report["beam_rows_without_particle_identity"] == 1
    assert report["ds_mu_pi_ancestry_incomplete"] == 0
    assert report["ds_mu_pi_linked_to_forward_source"] == 1
    assert report["primary_proton_events"] == 2
    assert report["events_without_forward_mu_pi_in_ntpart"] == 1


def test_compact_missing_selected_parent_or_hit_is_error():
    p = particles()
    p["parent_id"][2] = 99
    with pytest.raises(ValueError, match="selected positive parent"):
        tssmoke.census(p, hits([], [], []), hits([], [], []), compact=True)
    with pytest.raises(ValueError, match="hit lacks"):
        tssmoke.census(particles(), hits([], [], []), hits([99], [13], [40]), compact=True)
    p = particles()
    p["parent_pdg"][2] = 211
    with pytest.raises(ValueError, match="parent/ntpart"):
        tssmoke.census(p, hits([], [], []), hits([], [], []), compact=True)


def test_compact_unresolved_endpoint_is_not_a_source_loss():
    p = particles()
    p["parent_id"][2] = 99
    p["parent_pdg"][2] = 2112
    report = tssmoke.census(p, hits([2], [-211], [10]), hits([3], [13], [50]), compact=True)
    assert report["ds_mu_pi_ancestry_incomplete"] == 1
    assert report["ds_mu_pi_without_forward_source"] == 0
    assert report["source_tracks_without_ds_mu_pi_descendant"] is None
    assert report["source_tracks_without_linked_ds_mu_pi_descendant"] == 1


def test_compact_no_endpoints_still_establishes_losses():
    p = particles()
    p["parent_id"][1] = 99
    p["parent_pdg"][1] = 2112
    report = tssmoke.census(p, hits([2], [-211], [10]), hits([], [], []), compact=True)
    assert report["source_tracks_without_ds_mu_pi_descendant"] == 1


def test_compact_cycle_rejected():
    p = particles()
    p["parent_id"][1:3] = [3, 2]
    p["parent_pdg"][1:3] = [13, -211]
    with pytest.raises(ValueError, match="cyclic"):
        tssmoke.census(p, hits([], [], []), hits([3], [13], [20]), compact=True)


def test_selected_record_comparison_detects_changed_momentum_and_missing_beam(tmp_path):
    baseline, reduced = tmp_path / "full.root", tmp_path / "reduced.root"
    p = particles()
    p["pdg"][3] = 11
    vd, beam = hits([2, 4], [-211, 11], [10, 20]), hits([3, 4], [13, 11], [50, 60])
    with uproot.recreate(baseline) as f:
        for name, table in (("ntpart", p), ("ntvd", vd), ("nttvd", beam)):
            f.mktree(name, table)
    with uproot.recreate(reduced) as f:
        f.mktree("ntpart", tssmoke.subset(p, p["pdg"] != 11))
        f.mktree("ntvd", tssmoke.subset(vd, vd["pdg"] != 11))
        f.mktree("nttvd", beam)
    assert tssmoke.compare_diagnostics(baseline, reduced)["checks_passed"]
    with uproot.update(reduced) as f:
        changed = tssmoke.subset(vd, vd["pdg"] != 11)
        changed["pz"][0] += 1
        f.mktree("ntvd", changed)
        f.mktree("nttvd", tssmoke.subset(beam, beam["pdg"] != 11))
    report = tssmoke.compare_diagnostics(baseline, reduced)
    assert not report["checks_passed"]
    assert report["trees"]["ntvd"]["changed_scalar_columns"] == ["pz"]
    assert not report["trees"]["nttvd"]["exact_selected_scalar_match"]
