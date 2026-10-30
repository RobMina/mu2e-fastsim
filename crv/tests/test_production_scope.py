"""Population diagnostics must not silently become a matched selection or generator-association repair."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.analyses import production_scope as S


def sample_tables():
    parts = pd.DataFrame({"run": [1, 1, 1], "subrun": [2, 2, 2], "event": [3, 3, 4],
                          "key": [8, 19, 8], "parent": [-1, 8, -1], "pdg": [13, 11, -13], "isprim": [1, 0, 1]})
    steps = pd.DataFrame({"run": [1] * 4, "subrun": [2] * 4, "event": [3, 3, 3, 4],
                          "bar": [0, 0, 0, 1], "sp": [8, 8, 19, 8], "len": [10., 15., 100., 20.],
                          "edep": [1., 2., 100., 3.], "t0": [2., 1., 1., 3.],
                          "px0": [3., 0., 0., 0.], "py0": [-4., -10., -10., -20.], "pz0": [0.] * 4})
    geom = type("Geometry", (), {"sector": np.array([0, 1])})()
    return parts, steps, geom


def sample_log():
    return (" ###############  PrintModule Run/Subrun/Event         1        2        3\n"
            "ProductPrint mu2e::StepPointMCs_cosmicFilter_crvStage1_Primary\n"
            "StepPointMCCollection has 2 steps\n"
            "ind parent vol eDep noIonEDep Position P time properTime endProc\n"
            " 0 8 80 0.0 0.0 1.0 2.0 3.0 10.0 4.0 0.0 mu2eKillerVolume\n"
            " 1 19 81 0.0 0.0 1.0 2.0 3.0 5.0 3.0 0.0 mu2eKillerVolume\n"
            " ###############  PrintModule Run/Subrun/Event         1        2        4\n"
            "ProductPrint mu2e::StepPointMCs_cosmicFilter_crvStage1_Primary\n"
            "StepPointMCCollection has 1 steps\n"
            " 0 19 82 0.0 0.0 1.0 2.0 3.0 4.0 8.0 0.0 mu2eKillerVolume\n"
            "TrigReport Events total = 2 passed = 2 failed = 0\n")


def test_crossings_use_primary_identity_not_genidx_or_nearest_state():
    parts, steps, geom = sample_tables()
    parts["genidx"] = -100
    x = S.crossing_rows(steps, parts, geom)
    assert x["len"].tolist() == [25., 20.]
    assert x.edep.tolist() == [3., 3.]
    assert x.p.tolist() == [10., 20.]
    assert x.abs_uy.tolist() == [1., 1.]
    assert x.sp.tolist() == [8, 8]


def test_crossing_identity_duplicate_and_bad_momentum_refused():
    parts, steps, geom = sample_tables()
    with pytest.raises(ValueError, match="duplicate"):
        S.crossing_rows(steps, pd.concat([parts, parts]), geom)
    steps.loc[0, ["px0", "py0", "pz0"]] = 0.
    with pytest.raises(ValueError, match="momentum"):
        S.crossing_rows(steps, parts, geom)


def test_spectra_pool_adds_disjoint_files():
    parts, steps, geom = sample_tables()
    x = S.crossing_rows(steps, parts, geom)
    both = S.spectra(x)
    got = S.pool_spectra(S.spectra(x.iloc[:1]), S.spectra(x.iloc[1:]))
    assert got == both
    assert both["variables"]["len"]["sum"] == 45.
    empty = S.spectra(x.iloc[:0])
    assert empty["crossings"] == 0 and empty["variables"]["p"]["mean"] is None


def test_stock_event_labelled_stage_log(tmp_path):
    path = tmp_path / "scope.log"
    path.write_text(sample_log())
    census = pd.DataFrame({"run": [1, 1], "subrun": [2, 2], "event": [3, 4]})
    h = S.parse_stage_log(path, census)
    assert h.sp.tolist() == [8, 19, 19]
    assert h.event.tolist() == [3, 3, 4]
    assert h.volume.tolist() == [80, 81, 82]


@pytest.mark.parametrize("change", ["truncated", "missing_row", "wrong_census", "duplicate", "noncontiguous", "wrong_product"])
def test_scope_log_fails_closed(tmp_path, change):
    log = sample_log()
    census = pd.DataFrame({"run": [1, 1], "subrun": [2, 2], "event": [3, 4]})
    if change == "truncated":
        log = log.replace("TrigReport Events total = 2 passed = 2 failed = 0", "")
    elif change == "missing_row":
        log = log.replace(" 1 19 81 0.0 0.0 1.0 2.0 3.0 5.0 3.0 0.0 mu2eKillerVolume\n", "")
    elif change == "wrong_census":
        census.loc[1, "event"] = 5
    elif change == "duplicate":
        log = log.replace(" 1 19 81", " 0 19 81")
    elif change == "noncontiguous":
        log = log.replace(" 1 19 81", " 2 19 81")
    else:
        log = log.replace("cosmicFilter_crvStage1_Primary", "cosmicFilter_CRV_Primary")
    path = tmp_path / "scope.log"
    path.write_text(log)
    with pytest.raises(ValueError):
        S.parse_stage_log(path, census)


def test_ds_entry_attribution_is_by_same_event_and_sim_key(tmp_path):
    parts, steps, geom = sample_tables()
    parts = pd.concat([parts, pd.DataFrame({"run": [1], "subrun": [2], "event": [4], "key": [19],
                                           "parent": [8], "pdg": [11], "isprim": [0]})], ignore_index=True)
    path = tmp_path / "scope.log"
    path.write_text(sample_log())
    census = parts[S.KEY].drop_duplicates()
    h = S.parse_stage_log(path, census)
    got = S.stage_population(h, parts, steps, geom, {"first": [0], "second": [1]})
    assert got["events_with_stage_steps"] == 2
    assert got["events_with_primary_muon_stage_steps"] == 1
    assert got["events_without_direct_primary_muon_entry"] == 1
    assert got["without_direct_but_with_primary_muon_ancestor"] == 1
    assert got["without_primary_muon_ancestor_in_stage_particles"] == 0
    assert got["groups"]["first"]["direct_primary_muon_enters_ds"] == 1
    assert got["groups"]["second"]["no_direct_entry_for_earliest_primary"] == 1
    h.loc[0, "sp"] = 200
    with pytest.raises(ValueError, match="missing retained"):
        S.stage_population(h, parts, steps, geom, {})


def test_ancestry_pointer_doubling_and_guards():
    p = pd.DataFrame({"run": [1] * 5, "subrun": [2] * 5, "event": [3] * 5,
                      "key": [10, 20, 30, 40, 50], "parent": [-1, 10, 20, 30, 40],
                      "pdg": [13, 11, 22, 11, 22], "isprim": [1, 0, 0, 0, 0]})
    roots = S.primary_roots(p)
    assert roots.root_pdg.tolist() == [13] * 5
    with pytest.raises(ValueError, match="missing"):
        S.primary_roots(p.iloc[1:])
    p.loc[0, "parent"] = 50
    with pytest.raises(ValueError, match="cyclic|nonconvergent"):
        S.primary_roots(p)


def test_two_primary_muons_sharing_bar_keep_particle_scope(tmp_path):
    parts, steps, geom = sample_tables()
    parts = pd.concat([parts, pd.DataFrame({"run": [1], "subrun": [2], "event": [3], "key": [30],
                                           "parent": [-1], "pdg": [-13], "isprim": [1]})], ignore_index=True)
    steps = pd.concat([steps, steps.iloc[[0]].assign(sp=30, t0=.5)], ignore_index=True)
    hits = pd.DataFrame({"run": [1], "subrun": [2], "event": [3], "sp": [8], "p": [10.], "t": [4.]})
    got = S.stage_population(hits, parts, steps, geom, {"first": [0]})["groups"]["first"]
    assert got["crossings"] == 1 and got["particle_bar_crossings"] == 2
    assert got["particle_bar_direct_ds_entry"] == 1 and got["particle_bar_no_direct_ds_entry"] == 1


def test_unmatched_comparison_never_claims_efficiency():
    parts, steps, geom = sample_tables()
    x = S.crossing_rows(steps, parts, geom)
    got = S.compare({"sector": S.spectra(x)}, {"sector": S.spectra(x)})
    assert got["selection_matched"] is False
    assert got["exact_reference_ds_entry_selection"]["supported"] is False
    assert got["groups"]["sector"]["variables"]["p"]["binned_total_variation"] == 0.


def test_summary_labels_unmatched_populations(tmp_path):
    parts, steps, geom = sample_tables()
    x = S.crossing_rows(steps, parts, geom)
    result = {"comparison": S.compare({"group:test": S.spectra(x)}, {"group:test": S.spectra(x)})}
    path = tmp_path / "summary.md"
    S.write_summary(result, path)
    assert "Unmatched populations" in path.read_text()
    assert "not separated" in path.read_text()


def test_scope_fcl_no_transport_and_only_exact_stage_product():
    text = (Path(__file__).parents[1] / "art/CrvFastSim/fcl/inspect_run2_s1_scope.fcl").read_text()
    assert '"cosmicFilter:crvStage1:Primary"' in text
    assert '"compressPV::Primary"' in text
    assert "Mu2eG4" not in text and "GeometryService" not in text
