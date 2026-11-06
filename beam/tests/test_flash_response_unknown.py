"""Saved-only UNKNOWN evidence: strict times, exact unsigned identities, no relabels."""
import copy
import json
import sys

import numpy as np
import pytest

from beamfastsim.flash_g4_timing import float32_interval
from beamfastsim.flash_response_origins import trace
from beamfastsim.flash_response_unknown import diagnose, main, EXTRACTION_CHECKS
from test_flash_response_origins import case as origins_case


@pytest.fixture
def case():
    tables, response = origins_case.__wrapped__()
    a, s = tables["particles"], tables["steps"]
    a["pdg"][2] = 22
    # Exercise both PRODUCT and particle keys beyond signed int64.
    offset = np.uint64(2**63 + 123)
    a["key"] += offset
    a["parent_key"][a["parent_state"] == 1] += offset
    s["particle_key"] += offset
    response["tracker_ownership_status"] = np.array([2, 0], np.int8)
    response["event_owned_ionizing_mev"][:] = 0.
    response["event_unknown_ionizing_mev"][:] = [.25, 0.]
    response["event_tracker_ionizing_mev"] = np.array([1.25, 0.])
    response["input_event_count"] = np.asarray(2)
    for name in ("run", "subrun", "event", "product", "key"):
        response[name] = a[name][[2, 3]].copy()
    response.update(exit=np.ones(2, bool), birth_inside_geometry=np.ones(2, bool),
        ambiguous=np.zeros(2, bool), recorded_invalid_segments=np.zeros(2, int),
        crossing_lower_ns=np.array([3.1, 3.5]), crossing_upper_ns=np.array([3.2, 4.5]),
        crossing_valid=np.ones(2, bool), frontier_status=np.array([1, 3]),
        frontier_missing_parent=np.zeros(2, bool), frontier_time_overlap=np.array([False, True]),
        foil_index=np.array([0, 1]), first_step_product=np.full(2, 2**63 + 50, np.uint64),
        first_stepindex=np.array([np.iinfo(np.uint64).max, 9], np.uint64),
        crossing_upper_source=np.array([2, 1]),
        report_json=np.asarray(json.dumps(dict.fromkeys(EXTRACTION_CHECKS, 0))))
    origins = {**trace(tables), **{"particle_" + k: v.copy() for k, v in a.items()},
               **{"tracker_" + k: v.copy() for k, v in s.items()},
               "tracker_ownership_status": response["tracker_ownership_status"].copy(),
               "tracker_ionizing_mev": response["tracker_ionizing_mev"].copy()}
    return origins, response


def test_immediate_child_not_leaf_and_complete_denominator(case):
    o, r = case
    before_o, before_r = copy.deepcopy(o), copy.deepcopy(r)
    result = diagnose(o, r, response_max_depth=128)
    assert result["input_events"] == 2
    assert result["unknown_steps"] == result["unknown_events"] == 1
    assert result["unknown_ionizing_mev"] == .25
    assert result["energy_per_input_event_mev"]["unknown_ionizing_mev"] == .125
    assert result["events"]["unknown_ionizing_mev"] == [.25, 0.]
    assert result["events"]["ordinal"] == [1000, 1001]
    u, c = result["unknown_steps_table"], result["candidates"]
    assert u["original_status"] == [2]
    assert u["observed_root_reached"] == [True]
    assert u["candidate_unknown_trigger"] == [True]
    assert u["candidate_count"] == [2]
    assert c["photon_row"] == [1, 0]
    assert c["relation"] == ["overlap", "excluded"]
    assert c["immediate_child_particle_row"] == [4, 3]
    assert c["child_time"] == [4., 3.]
    assert c["child_birth_upper_ns"][1] < c["crossing_lower_ns"][1]
    assert result["observed_chain"]["particle_row"] == [4, 3, 2, 1, 0]
    assert c["first_stepindex"][1] == 2**64 - 1
    assert c["product"][0] == int(r["product"][0]) > 2**63
    assert c["key"][0] == int(r["key"][1]) > 2**63
    assert c["child_key"][0] == int(o["particle_key"][4]) > 2**63
    assert not result["validation"]["independent_tracker_identity_complete"]
    json.dumps(result, allow_nan=False)
    for name in o:
        np.testing.assert_array_equal(o[name], before_o[name])
    for name in r:
        np.testing.assert_array_equal(r[name], before_r[name])


@pytest.mark.parametrize("bound,relation", [(0., "eligible"), (3., "overlap"), (4., "excluded")])
def test_strict_descendant_bounds(case, bound, relation):
    o, r = case
    r["crossing_lower_ns"][0] = r["crossing_upper_ns"][0] = bound
    c = diagnose(o, r, response_max_depth=128)["candidates"]
    assert c["relation"][1] == relation
    assert c["triggers_original_unknown"][1] == (relation == "overlap")


@pytest.mark.parametrize("edge", ["lower", "upper"])
def test_equal_birth_interval_edges_overlap_not_guessed(case, edge):
    o, r = case
    lo, hi = float32_interval(o["particle_time"])
    x = lo[3] if edge == "lower" else hi[3]
    r["crossing_lower_ns"][0] = r["crossing_upper_ns"][0] = x
    assert diagnose(o, r)["candidates"]["relation"][1] == "overlap"
    # One float64 ULP is significant: no tolerance or time rounding.
    if edge == "lower":
        r["crossing_upper_ns"][0] = r["crossing_lower_ns"][0] = np.nextafter(x, -np.inf)
        assert diagnose(o, r)["candidates"]["relation"][1] == "eligible"
    else:
        r["crossing_upper_ns"][0] = r["crossing_lower_ns"][0] = np.nextafter(x, np.inf)
        assert diagnose(o, r)["candidates"]["relation"][1] == "excluded"


def test_asymmetric_power_of_two_birth_bounds(case):
    c = diagnose(*case)["candidates"]
    assert 4. - c["child_birth_lower_ns"][0] != c["child_birth_upper_ns"][0] - 4.


@pytest.mark.parametrize("time,relation,trigger", [(3., "excluded", True), (4., "overlap", True),
                                                     (4.5, "overlap", True), (5., "eligible", False)])
def test_same_track_pre_time_and_original_unknown_rule(case, time, relation, trigger):
    o, r = case
    # Make the tracked leaf a saved foil photon. Saved chain is still valid.
    o["particle_pdg"][4] = 22
    r["key"][1] = o["particle_key"][4]
    o["tracker_time"][0] = time
    c = diagnose(o, r)["candidates"]
    assert c["same_track"] == [True, False]
    assert c["relation"][0] == relation
    assert c["triggers_original_unknown"][0] == trigger
    assert c["child_key"][0] is None
    assert c["child_birth_lower_ns"][0] is None
    assert c["step_pre_time_ns"][0] == time


@pytest.mark.parametrize("field,value", [("exit", False), ("birth_inside_geometry", False),
    ("ambiguous", True), ("recorded_invalid_segments", 1), ("crossing_valid", False),
    ("crossing_lower_ns", np.nan), ("crossing_upper_ns", np.inf), ("crossing_upper_ns", -1.)])
def test_invalid_candidate_evidence_is_retained(case, field, value):
    o, r = case
    r[field][1] = value
    result = diagnose(o, r, response_max_depth=128)
    assert result["candidates"]["relation"][0] == "invalid"
    assert result["candidates"]["triggers_original_unknown"][0]
    assert result["unknown_steps_table"]["original_status"] == [2]
    json.dumps(result, allow_nan=False)


def test_non_float32_birth_not_rounded_and_raw_value_preserved(case):
    o, r = case
    o["particle_time"] = o["particle_time"].astype(float)
    o["particle_time"][4] = np.nextafter(4., 5.)
    c = diagnose(o, r)["candidates"]
    assert c["relation"][0] == "invalid"
    assert c["child_birth_lower_ns"][0] is None
    assert c["child_time"][0] == np.nextafter(4., 5.)


def test_nonfinite_birth_and_step_are_json_safe(case):
    o, r = case
    o["particle_time"][4] = np.nan
    o["tracker_time"][0] = np.inf
    result = diagnose(o, r)
    assert result["unknown_steps_table"]["nonfinite_step_time"] == [True]
    assert result["candidates"]["child_time"][0] is None
    assert result["candidates"]["original_walk_initially_active"] == [False, False]
    assert not any(result["candidates"]["triggers_original_unknown"])
    json.dumps(result, allow_nan=False)


def test_rootmost_eligible_frontier_gate_without_reclassification(case):
    o, r = case
    r["crossing_lower_ns"][:] = 0.
    r["crossing_upper_ns"][:] = 1.
    r["frontier_status"][:] = [3, 1]
    result = diagnose(o, r)
    assert result["candidates"]["relation"] == ["eligible", "eligible"]
    assert result["unknown_steps_table"]["rootmost_observed_eligible_photon_row"] == [0]
    assert result["unknown_steps_table"]["rootmost_observed_frontier_blocked"] == [True]
    assert result["unknown_steps_table"]["original_status"] == [2]
    r["frontier_status"][:] = 1
    result = diagnose(o, r)
    assert result["unknown_steps_table"]["unknown_without_observed_explanation"] == [True]
    assert result["unknown_steps_table"]["original_status"] == [2]


def test_global_extraction_unknown_evidence(case):
    o, r = case
    r["report_json"] = np.asarray(json.dumps({"photons_without_stage": 1}))
    result = diagnose(o, r)
    assert not result["extraction_checks_passed"]
    assert result["extraction_checks"]["photons_without_stage"] == 1
    assert result["extraction_checks"]["particles_without_event"] is None


@pytest.mark.parametrize("mode", ["parent_missing", "parent_invalid", "truncated", "cycle", "leaf_missing"])
def test_missing_depth_and_observed_chain_metadata(case, mode):
    o, r = case
    keep = np.ones(len(o["trace_depth"]), bool)
    if mode == "leaf_missing":
        o["tracker_particle_key"][0] = 999
        keep = o["trace_hit_row"] != 0
    elif mode == "cycle":
        o["particle_parent_state"][0] = 1
        o["particle_parent_key"][0] = o["particle_key"][4]
    else:
        keep &= (o["trace_hit_row"] != 0) | (o["trace_depth"] <= 1)
        if mode == "parent_missing":
            o["particle_parent_key"][3] = 999
        elif mode == "parent_invalid":
            o["particle_parent_state"][3] = 2
    for name in ("trace_hit_row", "trace_particle_row", "trace_depth"):
        o[name] = o[name][keep]
    result = diagnose(o, r)
    u = result["unknown_steps_table"]
    assert u["observed_root_reached"] == [False]
    assert u["missing_pointer"] == [mode in ("parent_missing", "parent_invalid", "leaf_missing")]
    assert u["saved_trace_truncated_or_depth_exhausted"] == [mode in ("truncated", "cycle")]
    assert result["response_depth_limit"] is None


@pytest.mark.parametrize("name", ["particle_product", "particle_event"])
def test_full_identity_not_key_only(case, name):
    o, r = case
    # Move ancestor photon to another product/event, updating observed parent pointers
    # or event identities consistently. It must not match the old response photon key.
    if name == "particle_product":
        o[name][3] += np.uint64(1)
        o["particle_parent_product"][4] = o[name][3]
        with pytest.raises(ValueError, match="response photon full identities"):
            diagnose(o, r)
    else:
        o[name][3] = 6
        with pytest.raises(ValueError, match="identities"):
            diagnose(o, r)


@pytest.mark.parametrize("kind", ["particles", "photons", "tracker", "events", "trace", "raw_rows"])
def test_duplicate_identities_fail_even_with_no_unknown(case, kind):
    o, r = case
    if kind == "particles":
        o["particle_key"][1] = o["particle_key"][0]
    elif kind == "photons":
        r["key"][1] = r["key"][0]
    elif kind == "tracker":
        o["tracker_index"][1] = o["tracker_index"][0]
    elif kind == "events":
        r["event_event"][1] = r["event_event"][0]
    elif kind == "trace":
        for name in ("trace_hit_row", "trace_particle_row", "trace_depth"):
            o[name] = np.r_[o[name], o[name][:1]]
    else:
        r["tracker_step_row"][1] = r["tracker_step_row"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        diagnose(o, r)


@pytest.mark.parametrize("field", ["tracker_ownership_status", "tracker_ionizing_mev", "tracker_total_edep",
                                   "event_unknown_ionizing_mev", "event_tracker_ionizing_mev"])
def test_original_energy_status_and_event_alignment_required(case, field):
    o, r = case
    target = o if field in o else r
    target[field][0] += 1
    with pytest.raises(ValueError, match="disagrees"):
        diagnose(o, r)


def test_independent_identity_columns_if_available_are_verified(case):
    o, r = case
    r["tracker_index"] = o["tracker_index"].copy()
    assert diagnose(o, r)["validation"]["independently_verified_tracker_fields"] == ["index"]
    r["tracker_index"][0] += 10
    with pytest.raises(ValueError, match="identity index"):
        diagnose(o, r)


def test_trace_bad_indices_shape_and_missing_rows_fail(case):
    o, r = case
    original = o["trace_particle_row"].copy()
    o["trace_particle_row"][0] = 10000
    with pytest.raises(ValueError, match="row indices"):
        diagnose(o, r)
    o["trace_particle_row"] = original
    o["trace_depth"] = o["trace_depth"][:-1]
    with pytest.raises(ValueError, match="equal length"):
        diagnose(o, r)


def test_no_unknown_and_empty_tracker_clean(case):
    o, r = case
    r["tracker_ownership_status"][0] = o["tracker_ownership_status"][0] = 0
    r["event_residual_ionizing_mev"][0] = 1.25
    r["event_unknown_ionizing_mev"][0] = 0.
    result = diagnose(o, r)
    assert result["input_events"] == 2 and result["unknown_steps"] == 0
    assert result["unknown_steps_table"]["original_status"] == []
    assert result["candidates"]["relation"] == []
    assert result["observed_chain"]["depth"] == []
    for name in list(o):
        if name.startswith("tracker_") or name.startswith("trace_") or not name.startswith("particle_"):
            o[name] = o[name][:0]
    for name in list(r):
        if name.startswith("tracker_"):
            r[name] = r[name][:0]
        elif name.startswith("event_") and name.endswith("ionizing_mev"):
            r[name][:] = 0.
    result = diagnose(o, r)
    assert result["tracker_steps"] == 0 and result["input_events"] == 2
    json.dumps(result, allow_nan=False)


def test_zero_energy_unknown_is_not_dropped(case):
    o, r = case
    o["tracker_total_edep"][0] = o["tracker_ionizing_mev"][0] = r["tracker_ionizing_mev"][0] = 0.
    r["event_unknown_ionizing_mev"][0] = 0.
    r["event_tracker_ionizing_mev"][0] = 1.
    result = diagnose(o, r)
    assert result["unknown_steps"] == result["unknown_events"] == 1
    assert result["unknown_ionizing_mev"] == 0.


@pytest.mark.parametrize("scenario", ["overlap", "same_track", "frontier", "invalid", "extraction", "resolved"])
@pytest.mark.parametrize("limit", [1, 2, 3, 128])
def test_evidence_matches_actual_response_attribute(scenario, limit):
    from beamfastsim.flash_g4_frontier import classify
    from beamfastsim.flash_g4_response import attribute
    from test_flash_g4_frontier import chain
    from test_flash_g4_response import case as response_case

    t, p, intervals = response_case.__wrapped__(chain.__wrapped__())
    if scenario == "overlap":
        t["particles"]["time"][1] = 10.
    elif scenario == "same_track":
        t["steps"]["time"][0] = 21.
    elif scenario == "invalid":
        p["ambiguous"][0] = True
    elif scenario == "extraction":
        report = json.loads(str(p["report_json"]))
        report["new_photons_without_volume"] = 1
        p["report_json"] = np.asarray(json.dumps(report))
    frontier = classify(t, p, intervals, max_depth=limit)
    if scenario == "frontier":
        frontier["frontier_status"][:] = 3
    r = attribute(t, p, intervals, frontier, max_depth=limit)
    r.update(p)
    r.update(frontier)
    r.update({"crossing_" + name: value for name, value in intervals.items()})
    r.update({"event_" + name: value for name, value in t["events"].items()})
    r["event_ordinal"] = np.arange(2, dtype=np.uint64)
    o = {**{"particle_" + name: value for name, value in t["particles"].items()},
         **{"tracker_" + name: value for name, value in t["steps"].items()},
         "tracker_ownership_status": r["tracker_ownership_status"],
         "tracker_ionizing_mev": r["tracker_ionizing_mev"],
         "trace_hit_row": np.array([0, 1, 0, 1, 0]),
         "trace_particle_row": np.array([2, 1, 1, 0, 0]),
         "trace_depth": np.array([0, 0, 1, 1, 2])}
    result = diagnose(o, r, response_max_depth=limit)
    assert all(result["unknown_steps_table"]["original_unknown_cause_verified"])
    assert result["unknown_steps"] == np.count_nonzero(r["tracker_ownership_status"] == 2)
    assert not any(result["unknown_steps_table"]["unknown_without_observed_explanation"])
    assert result["unknown_steps_table"]["original_status"] == [2] * result["unknown_steps"]
    if scenario == "same_track" and limit >= 3:
        assert result["candidates"]["relation"] == ["overlap", "eligible"]
    elif scenario == "resolved" and limit >= 3:
        assert result["unknown_steps"] == 0


@pytest.mark.parametrize("field", ["particle_product", "tracker_index", "particle_parent_key"])
def test_noninteger_identity_rejected(case, field):
    o, r = case
    o[field] = o[field].astype(np.float64)
    with pytest.raises(ValueError, match="identity"):
        diagnose(o, r)


def test_cli_saved_only_and_no_input_overwrite(case, tmp_path, monkeypatch):
    o, r = case
    op, rp, out = tmp_path / "origins.npz", tmp_path / "foil_response.npz", tmp_path / "unknown.json"
    np.savez_compressed(op, **o)
    np.savez_compressed(rp, **r)
    monkeypatch.setitem(sys.modules, "uproot", None)
    monkeypatch.setitem(sys.modules, "awkward", None)
    args = ["--origins", str(op), "--response", str(rp), "--out", str(out)]
    assert main(args) == 0
    report = json.loads(out.read_text())
    assert report["unknown_steps"] == 1 and report["input_events"] == 2
    assert report["candidates"]["first_stepindex"][1] == 2**64 - 1
    assert report["elapsed_seconds"] >= 0.
    assert report["response_depth_limit"] is None
    assert report["candidates"]["triggers_original_unknown"] == [None, False]
    assert main(args + ["--response-max-depth", "128"]) == 0
    report = json.loads(out.read_text())
    assert report["response_depth_limit"] == 128
    assert report["response_depth_limit_provenance"] == "user-supplied-original-config"
    assert report["candidates"]["triggers_original_unknown"] == [True, False]
    with pytest.raises(SystemExit):
        main(args[:-1] + [str(op)])


@pytest.mark.parametrize("limit", [None, 1, 2, 3, 128])
def test_original_candidate_depth_inclusive_and_unknown_is_conditional(case, limit):
    o, r = case
    r["crossing_lower_ns"][:] = r["crossing_upper_ns"][:] = 3.
    result = diagnose(o, r, response_max_depth=limit)
    c = result["candidates"]
    assert c["depth"] == [1, 2]
    assert c["relation"] == ["eligible", "overlap"]
    assert c["original_candidate_visited"] == ([None, None] if limit is None else [True, limit >= 2])
    assert c["triggers_original_unknown"] == [False, None if limit is None else limit >= 2]
    assert c["conditional_original_unknown_rule"] == [False, limit is None]
    assert result["response_depth_limit"] == limit
    assert result["response_depth_limit_provenance"] == (
        "unknown" if limit is None else "user-supplied-original-config")
    u = result["unknown_steps_table"]
    assert u["candidate_unknown_trigger"] == [limit is not None and limit >= 2]
    assert u["candidate_conditional_unknown_rule"] == [limit is None]
    assert u["original_depth_exhausted"] == [None if limit is None else limit <= 4]
    assert u["original_unknown_cause_verified"] == [limit is not None]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("limit", [None, 1, 128])
def test_same_track_visited_independently_of_original_depth(case, limit):
    o, r = case
    o["particle_pdg"][4] = 22
    r["key"][1] = o["particle_key"][4]
    o["tracker_time"][0] = 3.
    result = diagnose(o, r, response_max_depth=limit)
    assert result["candidates"]["original_candidate_visited"][0] is True
    assert result["candidates"]["triggers_original_unknown"][0] is True
    assert result["unknown_steps_table"]["original_unknown_cause_verified"] == [True]


@pytest.mark.parametrize("limit,trace_depth,exhausted", [
    (None, 1, None), (128, 1, None), (3, 1, None), (2, 1, True),
    (1, 1, True), (1, 0, True), (2, 0, None),
    (4, 4, True), (5, 4, False), (128, 4, False)])
def test_different_response_and_origins_depth_limits(case, limit, trace_depth, exhausted):
    o, r = case
    keep = (o["trace_hit_row"] != 0) | (o["trace_depth"] <= trace_depth)
    for name in ("trace_hit_row", "trace_particle_row", "trace_depth"):
        o[name] = o[name][keep]
    r["crossing_lower_ns"][:] = 100.
    r["crossing_upper_ns"][:] = 101.  # No overlapping or eligible candidates.
    result = diagnose(o, r, response_max_depth=limit)
    u = result["unknown_steps_table"]
    assert u["original_depth_exhausted"] == [exhausted]
    assert u["saved_trace_truncated_or_depth_exhausted"] == [trace_depth < 4]
    assert u["original_unknown_cause_verified"] == [exhausted is True]
    assert u["original_status"] == [2]


@pytest.mark.parametrize("limit,expected", [(None, None), (1, False), (2, True), (128, True)])
def test_observed_missing_parent_is_not_automatically_original_cause(case, limit, expected):
    o, r = case
    keep = (o["trace_hit_row"] != 0) | (o["trace_depth"] <= 1)
    for name in ("trace_hit_row", "trace_particle_row", "trace_depth"):
        o[name] = o[name][keep]
    o["particle_parent_key"][3] = 999
    r["crossing_lower_ns"][:] = 100.
    r["crossing_upper_ns"][:] = 101.
    u = diagnose(o, r, response_max_depth=limit)["unknown_steps_table"]
    assert u["missing_parent_pointer"] == [True]
    assert u["original_missing_pointer_trigger"] == [expected]
    assert u["original_depth_exhausted"] == [None if limit is None else limit == 1]


@pytest.mark.parametrize("field", ["run", "subrun", "event", "product", "key"])
def test_reverse_response_photon_join_requires_full_identity(case, field):
    o, r = case
    r[field][1] += np.uint64(1)
    # If event is changed, make that ledger event tracker-active too, without
    # changing its energy (use a zero-energy tracker step).
    if field == "event":
        o["tracker_event"][1] = r[field][1]
        o["tracker_total_edep"][1] = o["tracker_ionizing_mev"][1] = r["tracker_ionizing_mev"][1] = 0.
        r["event_residual_ionizing_mev"][0] = 0.
        r["event_tracker_ionizing_mev"][0] = .25
    with pytest.raises(ValueError, match="identit"):
        diagnose(o, r)


def test_nontracker_event_response_photon_need_not_be_in_compact_origins(case):
    o, r = case
    r["event"][1] = r["event_event"][1]
    r["key"][1] = np.iinfo(np.uint64).max
    result = diagnose(o, r, response_max_depth=128)
    assert result["candidates"]["photon_row"] == [0]
    assert result["validation"]["tracker_active_response_photons_in_origins"]
    assert result["input_events"] == 2


@pytest.mark.parametrize("limit", [0, -1, True, np.bool_(True), 1.5, "128"])
def test_invalid_original_depth_rejected(case, limit):
    with pytest.raises(ValueError, match="response_max_depth.*positive integer"):
        diagnose(*case, response_max_depth=limit)


@pytest.mark.parametrize("limit", ["0", "-1", "1.5"])
def test_cli_invalid_original_depth(case, tmp_path, limit):
    with pytest.raises(SystemExit):
        main(["--origins", "unused", "--response", "unused", "--out", str(tmp_path / "out"),
              "--response-max-depth", limit])


@pytest.mark.parametrize("limit", [1, 2, 3, 128])
@pytest.mark.parametrize("origins_rows", [1, 2, 3])
@pytest.mark.parametrize("terminal", ["root", "missing", "cycle"])
def test_saved_trace_depth_vs_actual_original_parent_walk(limit, origins_rows, terminal):
    from beamfastsim.flash_g4_response import attribute
    from test_flash_g4_frontier import chain
    from test_flash_g4_response import case as response_case

    t, p, intervals = response_case.__wrapped__(chain.__wrapped__())
    a = t["particles"]
    if terminal != "root":
        a["parent_state"][0] = 1
        a["parent_key"][0] = 999 if terminal == "missing" else a["key"][2]
    # Same-track is eligible; ancestors excluded; no other UNKNOWN cause.
    t["steps"]["time"][:] = 1000.
    intervals["lower_ns"][:] = 100.
    intervals["upper_ns"][:] = 101.
    frontier = dict(frontier_status=np.ones(2, int))
    r = attribute(t, p, intervals, frontier, max_depth=limit)
    r.update(p)
    r.update(frontier)
    r.update({"crossing_" + name: value for name, value in intervals.items()})
    r.update({"event_" + name: value for name, value in t["events"].items()})
    r["event_ordinal"] = np.arange(2, dtype=np.uint64)
    # Depths 0,1,2 for hit 0 and 0,1 for hit 1; an observed cycle
    # additionally has particle 2 at depth 2 for hit 1.
    h = np.array([0, 1, 0, 1, 0] + ([1] if terminal == "cycle" else []))
    ar = np.array([2, 1, 1, 0, 0] + ([2] if terminal == "cycle" else []))
    d = np.array([0, 0, 1, 1, 2] + ([2] if terminal == "cycle" else []))
    keep = d < origins_rows
    o = {**{"particle_" + name: value for name, value in a.items()},
         **{"tracker_" + name: value for name, value in t["steps"].items()},
         "tracker_ownership_status": r["tracker_ownership_status"],
         "tracker_ionizing_mev": r["tracker_ionizing_mev"],
         "trace_hit_row": h[keep], "trace_particle_row": ar[keep], "trace_depth": d[keep]}
    result = diagnose(o, r, response_max_depth=limit)
    u = result["unknown_steps_table"]
    for i, hit in enumerate(u["tracker_row"]):
        actual_depth = 2 - hit
        enough_edges = min(origins_rows, actual_depth) >= limit
        if terminal == "cycle":
            enough_edges = origins_rows >= limit
        complete = origins_rows > actual_depth and terminal != "cycle"
        expected_exhaustion = enough_edges if enough_edges or complete else None
        assert u["original_depth_exhausted"][i] is expected_exhaustion
        missing_proof = terminal == "missing" and complete and actual_depth < limit
        assert u["original_unknown_cause_verified"][i] == (enough_edges or missing_proof)
    assert not any(result["candidates"]["comparison_unknown_rule"])
    assert result["unknown_steps"] == np.count_nonzero(r["tracker_ownership_status"] == 2)


def test_large_float32_birth_bracket_is_precision_ambiguity_not_depth_failure(case):
    o, r = case
    birth = np.float32(1.e11)
    o["particle_time"][[3, 4]] = birth
    r["crossing_lower_ns"][:] = float(birth) + .05
    r["crossing_upper_ns"][:] = r["crossing_lower_ns"] + .0758
    result = diagnose(o, r, response_max_depth=128)
    c, u = result["candidates"], result["unknown_steps_table"]
    assert np.subtract(c["child_birth_upper_ns"], c["child_birth_lower_ns"]).tolist() == [8192., 8192.]
    assert c["relation"] == ["overlap", "overlap"]
    assert c["original_candidate_visited"] == [True, True]
    assert c["triggers_original_unknown"] == [True, True]
    assert u["observed_root_reached"] == [True]
    assert u["original_depth_exhausted"] == [False]
    assert u["original_unknown_cause_verified"] == [True]
    assert u["original_status"] == [2]
    unknown_limit = diagnose(o, r)
    assert unknown_limit["candidates"]["triggers_original_unknown"] == [None, None]
    assert unknown_limit["unknown_steps_table"]["original_unknown_cause_verified"] == [False]
