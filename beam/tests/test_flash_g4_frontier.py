"""Causal birth-foil frontier tests with integer identities and empty exposures."""
import numpy as np
import pytest

from beamfastsim.flash_g4_frontier import classify, NO_EXIT, FRONTIER, COVERED, UNRESOLVED


@pytest.fixture
def chain():
    product = np.uint64(2**63 + 9)
    a = dict(run=np.ones(3, np.uint64), subrun=np.full(3, 7, np.uint64),
             event=np.ones(3, np.uint64), product=np.full(3, product, np.uint64),
             key=np.array([10, 20, 30], np.uint64), pdg=np.array([22, 11, 22]),
             parent_state=np.array([0, 1, 1], np.uint32),
             parent_product=np.full(3, product, np.uint64), parent_key=np.array([0, 10, 20], np.uint64),
             time=np.array([0., 12., 20.], np.float32))
    p = {k: a[k][[0, 2]] for k in ("run", "subrun", "event", "product", "key")}
    p.update(exit=np.ones(2, bool), ambiguous=np.zeros(2, bool),
             birth_inside_geometry=np.ones(2, bool), recorded_invalid_segments=np.zeros(2, int))
    intervals = dict(lower_ns=np.array([10., 21.]), upper_ns=np.array([11., 22.]), valid=np.ones(2, bool))
    return {"particles": a}, p, intervals


def test_after_branch_birth_suppression_preserves_both_exposure_rows(chain):
    out = classify(*chain)
    np.testing.assert_array_equal(out["frontier_status"], [FRONTIER, COVERED])
    np.testing.assert_array_equal(out["covering_photon_row"], [-1, 0])
    assert len(out["frontier_status"]) == 2


def test_immediate_child_time_not_final_descendant_time(chain):
    chain[0]["particles"]["time"][1] = 5.
    out = classify(*chain)
    np.testing.assert_array_equal(out["frontier_status"], [FRONTIER, FRONTIER])
    assert not out["frontier_time_overlap"].any()


@pytest.mark.parametrize("birth", [10., 10.5, 11.])
def test_time_overlap_and_equality_not_guessed(chain, birth):
    chain[0]["particles"]["time"][1] = birth
    out = classify(*chain)
    assert out["frontier_status"][1] == UNRESOLVED
    assert out["frontier_time_overlap"][1]
    assert out["covering_photon_row"][1] == -1


def test_unknown_spatial_ancestor_not_assumed_absorbed(chain):
    chain[1]["exit"][0] = False
    out = classify(*chain)
    np.testing.assert_array_equal(out["frontier_status"], [NO_EXIT, UNRESOLVED])
    assert out["frontier_uncertain_ancestor"][1]


def test_invalid_own_timing_geometry_or_segment_is_unknown(chain):
    for key in ("valid", "birth_inside_geometry", "recorded_invalid_segments", "ambiguous"):
        tables, p, intervals = chain
        cp, ci = {k: v.copy() for k, v in p.items()}, {k: v.copy() for k, v in intervals.items()}
        if key == "valid":
            ci[key][0] = False
        else:
            cp[key][0] = key in ("ambiguous", "recorded_invalid_segments")
        out = classify(tables, cp, ci)
        np.testing.assert_array_equal(out["frontier_status"], [UNRESOLVED, UNRESOLVED])


@pytest.mark.parametrize("state", [1, 2, 3])
def test_missing_parent_not_a_root(chain, state):
    a = chain[0]["particles"]
    a["parent_state"][0] = state
    a["parent_key"][0] = 999
    out = classify(*chain)
    assert np.all(out["frontier_status"] == UNRESOLVED)
    assert out["frontier_missing_parent"].all()


def test_cycle_and_depth_limit_quarantine(chain):
    out = classify(*chain, max_depth=1)
    assert out["frontier_status"][0] == FRONTIER
    assert out["frontier_depth_exhausted"][1]
    a = chain[0]["particles"]
    a["parent_state"][0] = 1
    a["parent_key"][0] = 30
    out = classify(*chain, max_depth=6)
    assert out["frontier_depth_exhausted"].all()
    assert np.all(out["frontier_status"] == UNRESOLVED)


def test_rootmost_eligible_ancestor_not_nearest(chain):
    tables, p, intervals = chain
    a = tables["particles"]
    a["pdg"][1] = 22
    p = {k: a[k].copy() for k in ("run", "subrun", "event", "product", "key")}
    p.update(exit=np.ones(3, bool), ambiguous=np.zeros(3, bool),
             birth_inside_geometry=np.ones(3, bool), recorded_invalid_segments=np.zeros(3, int))
    intervals = dict(lower_ns=np.array([10., 13., 21.]), upper_ns=np.array([11., 14., 22.]), valid=np.ones(3, bool))
    out = classify(tables, p, intervals)
    np.testing.assert_array_equal(out["frontier_status"], [FRONTIER, COVERED, COVERED])
    np.testing.assert_array_equal(out["covering_photon_row"], [-1, 0, 0])


def test_uint64_product_collisions_and_non_photon_intermediates(chain):
    a = chain[0]["particles"]
    a["key"][1] = 10
    a["product"][1] += np.uint64(1)
    a["parent_product"][2] = a["product"][1]
    a["parent_key"][2] = 10
    out = classify(*chain)
    np.testing.assert_array_equal(out["frontier_status"], [FRONTIER, COVERED])
    a["event"][1] = 2
    out = classify(*chain)
    assert out["frontier_missing_parent"][1]


def test_empty_births_duplicate_and_missing_identities(chain):
    t, p, i = chain
    empty = classify(t, {k: v[:0] for k, v in p.items()}, {k: v[:0] for k, v in i.items()})
    assert empty["frontier_status"].size == 0
    p["key"][1] = p["key"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        classify(t, p, i)
    p["key"][1] = 999
    with pytest.raises(ValueError, match="missing"):
        classify(t, p, i)


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_invalid_depth_fails(chain, depth):
    with pytest.raises(ValueError, match="max_depth"):
        classify(*chain, max_depth=depth)


def test_root_cli_keeps_empty_event_ledger(tmp_path, monkeypatch):
    import json
    import uproot
    from test_flash_g4_photons import sample
    from beamfastsim import flash_geometry
    from beamfastsim.flash_g4_frontier import main

    tables, geom = sample.__wrapped__()
    n = len(tables["particles"]["key"])
    tables["particles"].update(time=np.zeros(n, np.float32), end_time=np.full(n, 200., np.float32),
                               end_defined=np.ones(n, np.uint32))
    path, out = tmp_path / "input.root", tmp_path / "frontier.npz"
    with uproot.recreate(path) as root:
        for name, columns in tables.items():
            root.mktree("export/" + name, columns)
    monkeypatch.setattr(flash_geometry, "from_offline", lambda *args: geom)
    args = ["--input", str(path), "--out", str(out), "--geom", "unused",
            "--offline-root", "unused", "--events", "4"]
    assert main(args) == 0
    with np.load(out, allow_pickle=False) as saved:
        assert len(saved["key"]) == 7
        np.testing.assert_array_equal(saved["event_event"], [1, 2, 3, 4])
        assert saved["event_frontier"].shape == (4,)
        assert saved["event_unresolved"][-1] == 0
        report = json.loads(str(saved["frontier_report_json"]))
        assert not report["detector_response_valid_S2"]
    assert main(args[:-1] + ["5"]) == 1
