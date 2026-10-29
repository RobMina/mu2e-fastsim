"""Common optical draws, exact identities, numerical order and bounded selection contracts."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import geometry
from crvfastsim.response import photons as P
from crvfastsim.veto import optical_timing_probe as probe
from crvfastsim.veto.replay import EVENT_TYPES, STEP_TYPES, sha256, write_replay
from crvfastsim.veto.replay_timing import wrap_arrivals


def sample():
    events = pd.DataFrame({"run": [1430] * 3, "subrun": [0] * 3, "event": [1, 2, 3],
                           "pbtime": [-224.46, -209.46, -204.46], "intensity": [1e7] * 3}).astype(EVENT_TYPES)
    steps = pd.DataFrame({k: np.zeros(6, dtype=v) for k, v in STEP_TYPES.items()})
    steps["run"], steps["event"] = np.uint32(1430), np.uint32(1)
    steps["sp"], steps["pdg"], steps["source_row"], steps["edep"] = 1, 13, np.arange(6) * 3, 0.1
    w = -float(np.float32(events.pbtime.iloc[0]))
    steps["t0"], steps["t1"] = w + np.array([1490., 1600., 50., 1690., np.nan, 1e16]), w + np.array([1490., 1600., 50., 1690., np.nan, 1e16])
    steps["px0"], steps["p1"], steps["len"] = 100., 100., .1
    steps["x0"], steps["x1"] = 2., 2.1
    geom = geometry.CRVGeometry(["T1", "T2"], np.array([0, 1]), np.zeros(2, int), np.zeros(2, int),
                                np.zeros(2, int), np.zeros((2, 3)), np.full((2, 3), 1000.), np.tile([0, 1, 2], (2, 1)))
    return steps, events, geom


def fake_optics(steps, pdg, geom, cfg, tables, deviation, rng):
    valid = np.flatnonzero(np.isfinite(steps.t0) & np.isfinite(steps.t1))
    si = np.repeat(valid, 2)
    delay = rng.integers(15, 16, len(si)).astype(float)
    return si, steps.bar.to_numpy()[si] * 4 + np.tile([0, 1], len(valid)), steps.t0.to_numpy()[si] + delay


def call(st, ev, geom, seed=21, status=None):
    return probe.common_photons(st, ev, geom, None, {}, np.zeros(8), np.zeros(8, int) if status is None else status, seed)


def test_four_arms_shared_photon_ids_all_channels_and_no_mutation(monkeypatch):
    monkeypatch.setattr(probe.photons, "make_photons", fake_optics)
    st, ev, geom = sample()
    old = st.copy(deep=True)
    q, rng = call(st, ev, geom)
    assert rng["same_photon_identities"] and rng["same_rng_consumption"]
    assert len(q) == 10
    assert q.source_step_row.tolist() == [0, 0, 3, 3, 6, 6, 9, 9, 15, 15]
    assert not q.duplicated(["source_step_row", "chan", "draw_seed", "ordinal"]).any()
    np.testing.assert_array_equal(q.rejection_only_kept, q.historical_kept & q.stock_step_kept)
    np.testing.assert_array_equal(q.stock_timing_kept, q.arrival_only_kept & q.stock_step_kept)
    assert q.rescued_by_arrival.sum() == 2
    assert q.rescued_survives_step.sum() == 0
    assert q[q.source_step_row == 0].historical_kept.sum() == 0
    assert not q[q.source_step_row == 3].stock_step_kept.any()
    frames, channels = probe.summaries(q, ev, geom)
    assert frames.photons.tolist() == [10, 0, 0]
    assert len(channels) == 2
    for key in probe.FLAGS:
        assert frames[key].sum() == channels[key].sum() == q[key].sum()
    pd.testing.assert_frame_equal(st, old)


def test_huge_times_use_raw_generator_addition_not_folded_algebra(monkeypatch):
    monkeypatch.setattr(probe.photons, "make_photons", fake_optics)
    st, ev, geom = sample()
    q, _ = call(st, ev, geom)
    extreme = q[q.source_step_row == 15]
    expected_raw = st.t0.iloc[5] + 15.
    assert np.all(extreme.time_raw_absolute == expected_raw)
    expected_wrapped = wrap_arrivals(expected_raw, extreme.origin.to_numpy())
    np.testing.assert_array_equal(extreme.time_raw_wrapped_absolute, expected_wrapped)
    assert extreme.raw_folded_circular_residual_ns.max() > 0


def test_masks_dont_remove_candidate_keys_and_both_ends_live_status(monkeypatch):
    monkeypatch.setattr(probe.photons, "make_photons", fake_optics)
    st, ev, geom = sample()
    q1, rng1 = call(st, ev, geom)
    status = np.zeros(8, int)
    status[1] = 1
    q2, rng2 = call(st, ev, geom, status=status)
    assert rng1 == rng2
    np.testing.assert_array_equal(q1.photon_id, q2.photon_id)
    np.testing.assert_array_equal(q1.time_raw_absolute, q2.time_raw_absolute)
    assert not q2[q2.chan == 1].live_channel.any()
    assert not q2[q2.chan == 1][[f"{arm}_kept" for arm in probe.ARMS]].to_numpy().any()


@pytest.mark.parametrize("which", ["identity", "consumption"])
def test_noncommon_rng_or_identity_fails(monkeypatch, which):
    calls = 0

    def changed(*args):
        nonlocal calls
        result = fake_optics(*args)
        calls += 1
        if calls == 2:
            if which == "identity":
                result[0][0] += 1
            else:
                args[-1].random()
        return result

    monkeypatch.setattr(probe.photons, "make_photons", changed)
    st, ev, geom = sample()
    with pytest.raises(ValueError, match="identities|consumption"):
        call(st, ev, geom)


@pytest.mark.parametrize("seed", [-1, 1.5, True, np.int64(1)])
def test_invalid_seed(seed):
    st, ev, geom = sample()
    with pytest.raises(ValueError, match="seed"):
        call(st, ev, geom, seed=seed)


@pytest.mark.parametrize("case", ["missing_event", "duplicate_event", "duplicate_row", "bar", "float_id", "infinite"])
def test_bad_step_or_event_contract(monkeypatch, case):
    monkeypatch.setattr(probe.photons, "make_photons", fake_optics)
    st, ev, geom = sample()
    if case == "missing_event":
        st["run"] = 1
    elif case == "duplicate_event":
        ev.loc[1, list(probe.IDS)] = ev.loc[0, list(probe.IDS)]
    elif case == "duplicate_row":
        st.loc[1, "source_row"] = st.source_row.iloc[0]
    elif case == "bar":
        st["bar"] = 3
    elif case == "float_id":
        st["source_row"] = st.source_row.astype(float)
    else:
        st.loc[0, "t0"] = np.inf
    with pytest.raises(ValueError):
        call(st, ev, geom)


def test_empty_optical_output_keeps_all_selected_frames(monkeypatch):
    monkeypatch.setattr(probe.photons, "make_photons", fake_optics)
    st, ev, geom = sample()
    q, _ = call(st.iloc[:0], ev, geom)
    frames, channels = probe.summaries(q, ev, geom)
    assert len(q) == len(channels) == 0
    assert len(frames) == 3 and (frames.photons == 0).all()


def test_real_production_optics_same_stream_synthetic_table():
    from test_propagation import _table

    st, ev, geom = sample()
    table = _table()
    cfg = P.ResponseConfig(sectors={k: P.SectorConfig("fake", 0, 39794.) for k in geom.sector_names})
    q, rng = probe.common_photons(st, ev, geom, cfg, {"fake": table}, np.zeros(8), np.zeros(8, int), 12)
    assert len(q) > 0 and rng["same_photon_identities"] and rng["same_rng_consumption"]
    q2, rng2 = probe.common_photons(st, ev, geom, cfg, {"fake": table}, np.zeros(8), np.zeros(8, int), 12)
    pd.testing.assert_frame_equal(q, q2)
    assert rng == rng2
    assert set(q.chan % 4) == {0, 1, 2, 3}


def test_select_root_rows_full_tuple_and_original_order(tmp_path):
    st, ev, _ = sample()
    st = st.iloc[:3].copy()
    st["run"] = np.array([1430, 1431, 1430], np.uint32)
    st["event"] = np.array([1, 1, 2], np.uint32)
    path = tmp_path / "test.root"
    write_replay(path, {"steps": st[list(STEP_TYPES)]})
    with uproot.open(path) as root:
        selected = probe.select_table(root["dump/steps"], ev.iloc[:1])
    assert len(selected) == 1 and selected.source_row.iloc[0] == 0


def test_particle_join_full_tuple_and_duplicate_refusal():
    st, ev, _ = sample()
    st = st.iloc[:2].copy()
    st = st.drop(columns="pdg")
    st.loc[1, "run"] = 1431
    pa = pd.DataFrame({"run": [1430, 1431], "subrun": [0, 0], "event": [1, 1], "key": [1, 1], "pdg": [13, 11]})
    joined = probe.join_particles(st, pa)
    assert joined.pdg.tolist() == [13, 11]
    with pytest.raises(ValueError, match="duplicate"):
        probe.join_particles(st, pd.concat([pa, pa]))
    with pytest.raises(ValueError, match="missing"):
        probe.join_particles(st, pa.iloc[:1])


def census(tmp_path):
    st, _, _ = sample()
    st = st.rename(columns={"source_row": "step_row"})
    st["fold_start_roundoff"] = [False] * 5 + [True]
    artifact = tmp_path / "input0_steps.parquet"
    st.to_parquet(artifact, index=False)
    meta = {"schema": "crv-step-timing-audit-v1", "counters": {"sha256": probe.COUNTERS_SHA},
            "inputs": [{"counts": {"fold_start_roundoff": 1}, "artifacts": {"steps": {
                "path": artifact.name, "sha256": sha256(artifact), "rows": len(st)}}}]}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(meta))
    verified = tmp_path / "verified.json"
    verified.write_text(json.dumps({"all_ledgers_reclassified_exactly": True, "pinned_geometry_verified": True,
                                   "summary_sha256": sha256(summary)}))
    return summary, verified


def test_selects_entire_frame_not_only_extreme_steps(tmp_path):
    _, verified = census(tmp_path)
    selected = probe.load_selection(tmp_path, verified)
    assert len(selected) == 1
    assert len(selected[0]["targets"]) == 1 and len(selected[0]["ledger"]) == 6


@pytest.mark.parametrize("case", ["unverified", "geometry", "summary", "artifact", "count"])
def test_selection_evidence_refusal(tmp_path, case):
    summary, verified = census(tmp_path)
    if case == "unverified":
        meta = json.loads(verified.read_text())
        meta["all_ledgers_reclassified_exactly"] = False
        verified.write_text(json.dumps(meta))
    elif case == "artifact":
        (tmp_path / "input0_steps.parquet").write_bytes(b"corrupted")
    elif case == "summary":
        summary.write_text("changed")
    else:
        meta = json.loads(summary.read_text())
        if case == "geometry":
            meta["counters"]["sha256"] = "0" * 64
        else:
            meta["inputs"][0]["counts"]["fold_start_roundoff"] += 1
        summary.write_text(json.dumps(meta))
        verification = json.loads(verified.read_text())
        verification["summary_sha256"] = sha256(summary)
        verified.write_text(json.dumps(verification))
    with pytest.raises(ValueError):
        probe.load_selection(tmp_path, verified)


def test_run_bad_assets_before_output(tmp_path):
    out = tmp_path / "out"
    for rel in probe.ASSETS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("wrong")
    with pytest.raises(ValueError, match="bound input"):
        probe.run(tmp_path, "none", "none", "none", tmp_path, tmp_path, out, 1)
    assert not out.exists()


def test_stock_ttree_dict_uses_event_count_not_branch_count(tmp_path):
    _, ev, _ = sample()
    stock = ev[list(probe.IDS)].copy()
    stock["pbtime"] = ev.pbtime.astype(np.float32)
    stock["spill"], stock["evlen"] = 1, 1700.
    stock["ndigi"], stock["npulse"], stock["ncluster"] = 0, 0, 0
    path = tmp_path / "stock.root"
    with uproot.recreate(path) as root:
        root.mktree("crvdump/events", {k: stock[k].dtype for k in stock})
        root["crvdump/events"].extend({k: stock[k].to_numpy() for k in stock})
    with uproot.open(path) as root:
        actual = root["crvdump/events"].arrays(library="np")
    assert isinstance(actual, dict) and len(actual) == 9
    assert probe.validate_stock_events(ev, actual)["events"] == 3
    actual["event"][1] = 42
    with pytest.raises(ValueError, match="event ID"):
        probe.validate_stock_events(ev, actual)


def test_source_bridge_byte_identical_body_only(tmp_path, monkeypatch):
    old, new = tmp_path / "old.cc", tmp_path / "new.cc"
    body = "  void CrvPhotonGenerator::produce(art::Event& event)\n{ corrected_fmod(); }\n"
    old.write_text("old yield sharing\n" + body)
    new.write_text("fixed yield sharing\n" + body)
    monkeypatch.setattr(probe, "STOCK_SHA", sha256(old))
    monkeypatch.setitem(probe.ASSETS, "CRVResponse/src/CrvPhotonGenerator_module.cc", sha256(new))
    assert probe.timing_source_bridge(old, new)["produce_body_byte_identical"] is True
    new.write_text(new.read_text().replace("corrected_fmod", "wrong"))
    monkeypatch.setitem(probe.ASSETS, "CRVResponse/src/CrvPhotonGenerator_module.cc", sha256(new))
    with pytest.raises(ValueError, match="processing differs"):
        probe.timing_source_bridge(old, new)
