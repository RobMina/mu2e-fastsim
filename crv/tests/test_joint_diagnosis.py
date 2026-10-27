"""Synthetic measured arrays only: no response simulation, real inputs or objective rescoring."""
import copy
import json

import awkward as ak
import numpy as np
import pytest

from crvfastsim.objective import bars as G
from crvfastsim.objective import crvline as CL
from crvfastsim.objective import crvresponse as R
from crvfastsim.objective.joint import JointCapture, capture, summarize


def sample():
    c = ak.Array({
        "crvcoincs.sectorType": [[1, 2, 3, 4, 2, 3, 4], [1], [2], []],
        "crvcoincs.time": [[100., 101., 102., 103., 120., np.nan, 200.], [100.], [100.], []],
        "crvcoincs.PEs": [[100., 100., 100., 100., 200., 200., 200.], [100.], [100.], []],
        "crvcoincs.hitPosAndTimeCalculated": [[False] * 7, [False], [False], []],
        "crvcoincs.pos.fCoordinates.fX": [[0.] * 7, [0.], [0.], []],
        "crvcoincs.pos.fCoordinates.fY": [[CL.Y_REF] * 7, [CL.Y_REF], [CL.Y_REF], []],
        "crvcoincs.nLayers": [[3] * 7, [3], [3], []],
        "crvcoincs.nHits": [[12, 6, 6, 24, 1, 0, 0], [3], [1], []],
        "crvcoincs.PEsPerLayer[4]": [[[25., 25., 50., 0.]] * 7, [[25., 25., 50., 0.]],
                                   [[25., 25., 50., 0.]], []],
    })
    base = np.r_[np.arange(8) * 64, 512 + np.arange(8) * 8]
    bars = (base[:, None] + np.where(np.arange(16)[:, None] < 8, 16, 2) * np.arange(3)).ravel()
    slots, _, layers = G.bar_slot_body_layer(bars)
    bid = np.r_[bars, 256, 256]
    p = ak.Array({
        "crvpulses.barId": [bid, [0, 16, 32], [256], []],
        "crvpulses.PEs": [np.r_[np.full(48, 20.), 50., 10.], [20., 20., 20.], [10.], []],
        "crvpulses.crvHitIndex": [np.r_[slots, 1, 4], [0, 0, 0], [0], []],
        "crvpulses.sequenceIndex": [np.r_[np.zeros(48, int), 1, 0], [0, 0, 0], [0], []],
        "crvpulses.SiPMId": [np.arange(50) % 4, [0, 1, 0], [0], []],
        "crvpulses.time": [np.r_[100. + layers, 110., 120.], [100., 101., 102.], [100.], []],
        "crvpulses.pos.fCoordinates.fX": [np.zeros(50), [0., 0., 0.], [0.], []],
        "crvpulses.pos.fCoordinates.fY": [np.r_[CL.Y_REF + 30. * (layers - 1), CL.Y_REF, CL.Y_REF],
                                          [CL.Y_REF - 30., CL.Y_REF, CL.Y_REF + 30.], [CL.Y_REF], []],
        "crvpulses.pos.fCoordinates.fZ": [np.zeros(50), [0., 0., 0.], [0.], []],
    })
    keys = np.array([[7, 3, 2 ** 53 + 1], [7, 4, 2 ** 53 + 1], [8, 3, 2 ** 53 + 1], [8, 3, 9]], np.int64)
    return c, p, keys


def accumulated(c, p):
    state = R.new_state()
    with np.errstate(invalid="ignore"):
        result = R.accumulate_arrays(c, p, state, 100000.)
    return state, result


def assert_tree_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    else:
        np.testing.assert_array_equal(a, b)


def test_spread_histogram_exactly_matches_reference_and_all_bodies():
    c, p, keys = sample()
    state, o = accumulated(c, p)
    d = capture(c, p, o, keys)
    np.testing.assert_array_equal(np.unique(d["hit_body"]), np.arange(16))
    np.testing.assert_array_equal(d["event_group_count"][:4], [12, 6, 6, 24])
    assert d["hit_spread_member"].sum() == 48
    assert d["hit_sequence_index"].max() == 0
    for slot, name in enumerate(CL.SLOT_NAME):
        m = (d["event_slot"] == slot) & d["event_spread_eligible"]
        hist = np.histogram(d["event_group_rms"][m], R.E["tspread"])[0]
        np.testing.assert_array_equal(hist, state["h"]["tspread_" + name])
        hm = (d["hit_slot"] == slot) & d["hit_spread_member"]
        tc = d["hit_time_corrected"][hm]
        expected = np.sqrt(max(np.mean(tc ** 2) - np.mean(tc) ** 2, 0.))
        np.testing.assert_allclose(d["event_group_rms"][m], expected, rtol=0, atol=1e-10)
        np.testing.assert_allclose(d["hit_time_residual"][hm], tc - tc.mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(d["hit_time_residual"][hm].sum(), 0., atol=1e-10)
    assert not d["event_spread_eligible"][4:].any()
    assert np.isnan(d["event_group_rms"][4:]).all()
    for slot, name in enumerate(CL.SLOT_NAME):
        m = (d["hit_slot"] == slot) & d["hit_pe_along_eligible"]
        values = np.column_stack([d["hit_side"][m], d["hit_along"][m], d["hit_pe_path"][m]])
        hist = np.histogramdd(values, [np.arange(-0.5, 2., 1.), R.E["along"], R.E["pe"]])[0]
        np.testing.assert_array_equal(hist, state["h"]["pe_along_" + name])
        m = (d["hit_slot"] == slot) & d["hit_pe_path_eligible"]
        values = np.column_stack([d["hit_path"][m], d["hit_pe"][m]])
        hist = np.histogramdd(values, [R.E["path"], R.E["pe"]])[0]
        np.testing.assert_array_equal(hist, state["h"]["pe_path_" + name])
    m = d["hit_ftp_eligible"]
    feb = (d["hit_layer"][m] >= 2).astype(int) * 2 + d["hit_side"][m]
    hist = np.histogramdd(np.column_stack([d["hit_body"][m], feb, d["hit_time_residual"][m]]),
                          [np.arange(-0.5, CL.N_BODY), np.arange(-0.5, 4), R.FTP_EDGES])[0]
    np.testing.assert_array_equal(hist, state["ftp"])


def test_raw_candidates_timing_tail_and_full_identity():
    c, p, keys = sample()
    _, o = accumulated(c, p)
    d = capture(c, p, o, keys, source_index=19)
    np.testing.assert_array_equal(d["event_run"], np.repeat(keys[:, 0], 4))
    np.testing.assert_array_equal(d["event_subrun"], np.repeat(keys[:, 1], 4))
    np.testing.assert_array_equal(d["event_event"], np.repeat(keys[:, 2], 4))
    assert (d["hit_source_index"] == 19).all()
    np.testing.assert_array_equal(d["candidate_selected"][:7], [True] * 4 + [False] * 3)
    np.testing.assert_array_equal(d["candidate_timing_rejected"][:7], [False] * 4 + [True] * 3)
    assert d["candidate_dt_ex"][4] == CL.DT_MATCH
    assert d["candidate_dt_ex_nonfinite"][5]
    assert d["candidate_dt_ex_overflow"][6]
    assert not d["candidate_has_ex"][-1]
    assert not d["candidate_eligible"][-1]
    assert not any("residual" in k for k in d if k.startswith("candidate_"))
    assert d["candidate_body"][4] == 4
    assert not d["candidate_body_available"][5]
    np.testing.assert_array_equal(d["event_candidate_count"][:4], [1, 2, 2, 2])
    np.testing.assert_array_equal(d["event_timing_rejected_count"][:4], [0, 1, 1, 1])
    summary = summarize(d)
    assert summary["tables"]["candidate"]["all"]["rows"] == 9
    assert summary["tables"]["event"]["T1"]["counts"]["timing_rejected_count"] == 1
    json.dumps(summary, allow_nan=False)


def test_nonfinite_and_negative_pe_preserved_without_changing_spread_population():
    c, p, keys = sample()
    t = ak.to_numpy(ak.flatten(p["crvpulses.time"]))
    pe = ak.to_numpy(ak.flatten(p["crvpulses.PEs"]))
    t[0], pe[0], pe[1] = np.nan, -4., 250.
    p = ak.with_field(p, ak.unflatten(t, ak.num(p["crvpulses.time"])), "crvpulses.time")
    p = ak.with_field(p, ak.unflatten(pe, ak.num(p["crvpulses.PEs"])), "crvpulses.PEs")
    state, o = accumulated(c, p)
    d = capture(c, p, o, keys)
    assert d["event_group_count"][0] == 12
    assert d["event_group_rms_nonfinite"][0]
    assert d["event_spread_eligible"][0]
    assert not d["event_tspread_inrange"][0]
    assert d["hit_pe_raw"][0] == -4. and d["hit_pe"][0] == 0.
    assert d["hit_pe_raw_underflow"][0] and d["hit_pe_overflow"][1]
    assert d["hit_time_nonfinite"][0] and not d["hit_residual_valid"][0]
    assert d["hit_spread_member"][0]
    assert np.asarray(state["h"]["tspread_EX"]).sum() == 0


def test_single_valid_projection_is_enough_for_spread_and_range_boundaries():
    c, p, keys = sample()
    _, o = accumulated(c, p)
    o["okx"][0] = False
    d = capture(c, p, o, keys)
    np.testing.assert_array_equal(d["event_group_count"][:4], [0, 6, 0, 24])
    assert not d["hit_both_lines"].any()
    assert not d["hit_pe_along_inrange"].any()
    assert d["hit_path_inrange"][:48].all()
    assert d["hit_path"][0] == 1.
    o["z0"][0] = R.HALF["T1"]
    d = capture(c, p, o, keys)
    assert d["hit_along_inrange"][d["hit_slot"] == 1].all()
    assert not d["hit_has_along"][d["hit_slot"] == 3].any()
    assert d["hit_along_overflow"][d["hit_slot"] == 3].all()


def test_passive_owned_capture_no_fit_rng_or_mutation(monkeypatch):
    c, p, keys = sample()
    state, o = accumulated(c, p)
    before_o, before_state = copy.deepcopy(o), copy.deepcopy(state)
    before_c, before_p, before_keys = ak.copy(c), ak.copy(p), keys.copy()
    rng_before = np.random.get_state()
    def forbidden(*args, **kwargs):
        raise AssertionError("passive capture called a fitter, objective, or RNG")
    monkeypatch.setattr(CL, "fit_arrays", forbidden)
    monkeypatch.setattr(R, "accumulate_arrays", forbidden)
    monkeypatch.setattr(R, "objective", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    collector = JointCapture().add(c, p, o, keys)
    assert_tree_equal(before_o, o)
    assert_tree_equal(before_state, state)
    for actual, before in ((c, before_c), (p, before_p)):
        for field in actual.fields:
            np.testing.assert_array_equal(ak.to_numpy(ak.flatten(actual[field], axis=None)),
                                          ak.to_numpy(ak.flatten(before[field], axis=None)))
            np.testing.assert_array_equal(ak.to_numpy(ak.num(actual[field])), ak.to_numpy(ak.num(before[field])))
    np.testing.assert_array_equal(keys, before_keys)
    for a, b in zip(np.random.get_state(), rng_before):
        np.testing.assert_array_equal(a, b)
    o["hit_time"][:] = -999.
    keys[:] = 0
    d = collector.columns()
    assert (d["hit_time"] >= 100.).all()
    assert d["event_event"][0] == 2 ** 53 + 1


def test_chunk_identity_npz_roundtrip_and_refuse_overwrite(tmp_path):
    c, p, keys = sample()
    _, o = accumulated(c, p)
    collector = JointCapture().add(c, p, o, keys, source_index=2).add(c, p, o, keys, source_index=3)
    path = tmp_path / "joint.npz"
    summary = collector.write(path)
    assert summary["events"] == 8 and summary["chunks"] == 2
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(str(saved["summary_json"])) == summary == collector.summary()
        np.testing.assert_array_equal(saved["event_chunk_index"], np.repeat([0, 1], 16))
        np.testing.assert_array_equal(saved["event_source_index"], np.repeat([2, 3], 16))
        for name, values in collector.columns().items():
            np.testing.assert_array_equal(saved[name], values)
    with pytest.raises(FileExistsError):
        collector.write(path)


@pytest.mark.parametrize("selection", [slice(0, 0), slice(3, 4)])
def test_empty_arrays_and_no_hit_events(selection, tmp_path):
    c, p, keys = sample()
    c, p, keys = c[selection], p[selection], keys[selection]
    _, o = accumulated(c, p)
    collector = JointCapture().add(c, p, o, keys)
    d = collector.columns()
    assert len(d["hit_event"]) == len(d["candidate_event"]) == 0
    assert len(d["event_event"]) == 4 * len(keys)
    assert d["candidate_pe_layer"].shape == (0, 4)
    assert collector.write(tmp_path / "empty.npz")["events"] == len(keys)


@pytest.mark.parametrize("keys", [np.zeros((4, 2), int), np.zeros((4, 3), float),
                                  np.full((4, 3), -1), np.full((4, 3), 2 ** 63, np.uint64)])
def test_requires_full_integer_keys(keys):
    c, p, _ = sample()
    _, o = accumulated(c, p)
    with pytest.raises(ValueError, match="event.keys"):
        capture(c, p, o, keys)


def test_rejects_wrong_selected_row_mapping():
    c, p, keys = sample()
    _, o = accumulated(c, p)
    o["hit_body"] = o["hit_body"][::-1]
    with pytest.raises(ValueError, match="selected rows"):
        capture(c, p, o, keys)


def test_groups_do_not_merge_equal_event_numbers_across_full_keys_or_chunks():
    c, p, keys = sample()
    c = ak.concatenate([c[:1], c[:1]])
    p2 = ak.with_field(p[:1], p[:1]["crvpulses.time"] + 7., "crvpulses.time")
    p = ak.concatenate([p[:1], p2])
    keys = keys[:2]
    state, o = accumulated(c, p)
    whole = JointCapture().add(c, p, o, keys)
    d = whole.columns()
    np.testing.assert_array_equal(d["event_group_count"], [12, 6, 6, 24] * 2)
    np.testing.assert_allclose(d["event_group_mean"][4:] - d["event_group_mean"][:4], 7., atol=1e-12)
    split = JointCapture()
    for chunk in (slice(0, 1), slice(1, 2)):
        _, result = accumulated(c[chunk], p[chunk])
        split.add(c[chunk], p[chunk], result, keys[chunk])
    for name, values in split.columns().items():
        if name.endswith(("chunk_index", "event_index")):
            continue
        np.testing.assert_array_equal(values, d[name])
    for slot, name in enumerate(CL.SLOT_NAME):
        m = (d["event_slot"] == slot) & d["event_spread_eligible"]
        np.testing.assert_array_equal(np.histogram(d["event_group_rms"][m], R.E["tspread"])[0],
                                      state["h"]["tspread_" + name])
