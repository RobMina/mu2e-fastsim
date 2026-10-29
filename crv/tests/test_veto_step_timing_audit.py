"""Full-denominator timing census, exact historical fold and fail-closed evidence bindings."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import step_timing_audit as audit
from crvfastsim.veto.fastframes import fold_steps
from crvfastsim.veto.replay import EVENT_TYPES, STEP_TYPES, replay_tables, sha256, write_replay
from crvfastsim.veto.replay_timing import PERIOD, stock_step_mask


def sample():
    events = pd.DataFrame({"run": [1430] * 3, "subrun": [0] * 3, "event": [1, 2, 3],
                           "pbtime": [-224.46, -204.46, -219.46], "intensity": [1e7] * 3}).astype(EVENT_TYPES)
    origin = -float(np.float32(events.pbtime.iloc[0]))
    steps = pd.DataFrame({k: np.zeros(8, dtype=v) for k, v in STEP_TYPES.items()})
    steps["run"], steps["event"], steps["sp"], steps["edep"] = np.uint32(1430), np.uint32(1), 1, 1.
    steps["bar"] = np.arange(8, dtype=np.int64) % 2
    steps["t0"] = origin + np.array([50., 1600., 1600., 1695., -1695. + 500., 50., np.nan, 600.])
    steps["t1"] = origin + np.array([60., 1610., 1745., 1700., -1695. + 510., 3440., 100., 500.])
    geom = SimpleNamespace(nbars=2, sector=np.array([0, 1]), sector_names=["T1", "T2"])
    return steps, events, geom


def test_partition_all_frames_and_no_mutation():
    st, ev, geom = sample()
    old_st, old_ev = st.copy(deep=True), ev.copy(deep=True)
    rows, frames = audit.classify(st, ev, geom)
    assert rows.stock_keep.tolist() == [False, False, False, False, True, False, False, True]
    assert rows.reject_low.sum() == 3
    assert rows.reject_high.sum() == rows.reject_wrap.sum() == rows.nan_endpoint.sum() == 1
    assert rows.crosses_period.sum() == 2
    assert rows.multiperiod_duration.sum() == rows.negative_duration.sum() == 1
    assert frames.steps.tolist() == [8, 0, 0]
    assert frames.stock_keep.tolist() == [2, 0, 0]
    assert rows.step_row.tolist() == list(range(8))
    for key in audit.FLAGS:
        assert frames[key].sum() == rows[key].sum()
    report = audit.summarize(rows, frames)
    assert report["empty_frames"] == 2
    assert report["frames_with"]["historical_only_eligible"] == 1
    assert report["edep_MeV"]["stock_keep"] == 2.
    pd.testing.assert_frame_equal(st, old_st)
    pd.testing.assert_frame_equal(ev, old_ev)


def test_historical_fold_regression_both_endpoints_move_by_start_shift():
    st, ev, geom = sample()
    rows, _ = audit.classify(st, ev, geom)
    keys = (ev.subrun.to_numpy(dtype=np.int64) << 20) | ev.event.to_numpy(dtype=np.int64)
    expected, lost = fold_steps(st, keys, -ev.pbtime.to_numpy().astype(np.float32).astype(float))
    assert lost == 0
    np.testing.assert_array_equal(rows.historical_t0_relative, expected.t0)
    np.testing.assert_array_equal(rows.historical_t1_relative, expected.t1)
    assert rows.historical_t1_relative.iloc[2] == 1745.
    assert rows.stock_t1_absolute.iloc[2] - rows.origin.iloc[2] == 50.


def test_full_tuple_join_does_not_pack_or_drop_run():
    st, ev, geom = sample()
    ev["run"] = np.array([1430, 1431, 1430], np.uint32)
    ev["subrun"] = np.array([1, 1, 0], np.uint32)
    ev["event"] = np.array([1, 1, (1 << 20) + 1], np.uint32)
    st = st.iloc[:3].copy()
    st[list(audit.IDS)] = ev[list(audit.IDS)].to_numpy()
    st = st.astype({k: np.uint32 for k in audit.IDS})
    rows, frames = audit.classify(st, ev, geom)
    np.testing.assert_array_equal(rows.event_row, [0, 1, 2])
    np.testing.assert_array_equal(rows.pbtime, ev.pbtime)
    np.testing.assert_array_equal(frames.steps, [1, 1, 1])


@pytest.mark.parametrize("phase", audit.PBTIMES)
def test_exact_edges_and_float32_origin_contrast(phase):
    st, ev, geom = sample()
    ev["pbtime"] = phase
    w = -float(np.float32(phase))
    st = st.iloc[:6].copy()
    st["t0"] = [np.nextafter(w + 100., -np.inf), w + 100., np.nextafter(w + 100., np.inf),
                np.nextafter(w + 1500., -np.inf), w + 1500., np.nextafter(w + 1500., np.inf)]
    st["t1"] = st.t0
    rows, _ = audit.classify(st, ev, geom)
    np.testing.assert_array_equal(rows.stock_keep, [False, True, True, True, True, False])
    expected = stock_step_mask(st.t0, st.t1, w) != stock_step_mask(st.t0, st.t1, -phase)
    np.testing.assert_array_equal(rows.float32_origin_gate_flip, expected)
    assert rows.float32_origin_gate_flip.any()


@pytest.mark.parametrize("column,value,match", [
    ("run", 1, "absent"), ("event", -1, "identity"), ("bar", -1, "identity"),
    ("bar", 2, "identity"), ("t0", np.inf, "infinite"), ("t1", -np.inf, "infinite"),
    ("edep", np.nan, "energy"), ("edep", -1., "energy"),
])
def test_invalid_step_rejected(column, value, match):
    st, ev, geom = sample()
    st[column] = value
    with pytest.raises(ValueError, match=match):
        audit.classify(st, ev, geom)


@pytest.mark.parametrize("case", ["duplicate", "empty", "phase", "noninteger"])
def test_invalid_event_census(case):
    st, ev, geom = sample()
    if case == "duplicate":
        ev.loc[1, list(audit.IDS)] = ev.loc[0, list(audit.IDS)]
    elif case == "empty":
        ev = ev.iloc[:0]
    elif case == "phase":
        ev["pbtime"] = np.nan
    else:
        ev["event"] = ev.event.astype(float)
    with pytest.raises(ValueError):
        audit.classify(st, ev, geom)


def test_empty_steps_still_have_every_frame():
    st, ev, geom = sample()
    rows, frames = audit.classify(st.iloc[:0], ev, geom)
    report = audit.summarize(rows, frames)
    assert len(rows) == 0 and len(frames) == 3
    assert report["empty_frames"] == 3
    assert report["max_abs_fold_start_residual_ns"] is None
    assert not any(report["counts"].values())


def saved(tmp_path, monkeypatch):
    st, ev, geom = sample()
    st["pdg"] = 13
    path = tmp_path / "mixed.root"
    counts = write_replay(path, replay_tables(st, ev))
    meta = {"schema": "crv-mixed-replay-v1", "output_sha256": sha256(path), "raw_times": True,
            "events": len(ev), "entries": counts, "run": 1430, "subrun": 0, "first_event": 1}
    path.with_suffix(".json").write_text(json.dumps(meta))
    source, counters = tmp_path / "source.cc", tmp_path / "counters.txt"
    source.write_text("synthetic stock source fixture")
    counters.write_text("synthetic counters fixture")
    monkeypatch.setattr(audit, "STOCK_SHA", sha256(source))
    monkeypatch.setattr(audit, "COUNTERS_SHA", sha256(counters))
    monkeypatch.setattr(audit.geometry, "from_offline_config", lambda path: geom)
    return path, source, counters


def test_cli_persists_hashed_full_ledgers_and_no_overwrite(tmp_path, monkeypatch, capsys):
    path, source, counters = saved(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    audit.main([str(path), "--counters", str(counters), "--stock-source", str(source), "--out", str(out)])
    result = json.loads((out / "summary.json").read_text())
    assert result["timing_closure"] is False and result["F"] is None
    assert result["finite_library_uncertainty"] is None
    assert result["totals"]["frames"] == 3
    assert len(result["inputs"][0]["sector_phase"]) == 10
    for artifact in result["inputs"][0]["artifacts"].values():
        target = out / artifact["path"]
        assert sha256(target) == artifact["sha256"]
        assert len(pd.read_parquet(target)) == artifact["rows"]
    assert json.loads(capsys.readouterr().out)["empty_frames"] == 2
    with pytest.raises(ValueError, match="new output"):
        audit.run([path], counters, source, out)


@pytest.mark.parametrize("case", ["hash", "schema", "contract", "raw", "count", "identity"])
def test_input_contract_failures(tmp_path, monkeypatch, case):
    path, _, _ = saved(tmp_path, monkeypatch)
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text())
    changes = {"hash": {"output_sha256": "0" * 64}, "schema": {"schema": "crv-mixed-replay-v2"},
               "contract": {"exposure_contract": {}}, "raw": {"raw_times": False},
               "count": {"events": 4}, "identity": {"first_event": 2}}
    meta.update(changes[case])
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        audit.read_input(path)


def test_duplicate_payload_not_independent_sample(tmp_path, monkeypatch):
    path, source, counters = saved(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="repeated replay"):
        audit.run([path, path], counters, source, tmp_path / "audit")
    assert not (tmp_path / "audit" / "summary.json").exists()


def test_source_guard_before_output_creation(tmp_path, monkeypatch):
    path, source, counters = saved(tmp_path, monkeypatch)
    source.write_text("wrong source")
    with pytest.raises(ValueError, match="pinned"):
        audit.run([path], counters, source, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_empty_sector_phase_bins_recorded(tmp_path, monkeypatch):
    path, source, counters = saved(tmp_path, monkeypatch)
    result = audit.run([path], counters, source, tmp_path / "audit")
    groups = result["inputs"][0]["sector_phase"]
    assert sum(r["steps"] for r in groups) == result["totals"]["steps"]
    assert sum(r["steps"] == 0 for r in groups) == 8


def test_wrong_geometry_with_same_sector_names_rejected_before_outputs(tmp_path, monkeypatch):
    path, source, counters = saved(tmp_path, monkeypatch)
    counters.write_text("same T1/T2 names but different module count and sector boundary")
    with pytest.raises(ValueError, match="pinned Run1A"):
        audit.run([path], counters, source, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_verify_saved_census_exact_and_no_overwrite(tmp_path, monkeypatch):
    path, source, counters = saved(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    audit.run([path], counters, source, out)
    verified = tmp_path / "verified.json"
    result = audit.verify_saved(out, counters, source, verified)
    assert result["all_ledgers_reclassified_exactly"] is True
    assert result["pinned_geometry_verified"] is True
    assert result["checks"][0]["frames"] == 3
    assert len(result["checks"][0]["no_stock_eligible_step_frames"]) == 2
    with pytest.raises(ValueError, match="new verification"):
        audit.verify_saved(out, counters, source, verified)


@pytest.mark.parametrize("case", ["geometry", "artifact", "summary", "input", "totals", "sector_phase"])
def test_verify_saved_rejects_changes(tmp_path, monkeypatch, case):
    path, source, counters = saved(tmp_path, monkeypatch)
    out = tmp_path / "audit"
    audit.run([path], counters, source, out)
    if case == "geometry":
        counters.write_text("wrong geometry")
    elif case == "artifact":
        (out / "input0_steps.parquet").write_bytes(b"changed")
    elif case in ("summary", "totals", "sector_phase"):
        report_path = out / "summary.json"
        report = json.loads(report_path.read_text())
        if case == "summary":
            report["inputs"][0]["counts"]["stock_keep"] += 1
        elif case == "totals":
            report["totals"]["frames"] += 1
        else:
            report["inputs"][0]["sector_phase"][0]["steps"] += 1
        report_path.write_text(json.dumps(report))
    else:
        path.with_suffix(".json").write_text("changed")
    with pytest.raises(ValueError):
        audit.verify_saved(out, counters, source, tmp_path / "verified.json")
    assert not (tmp_path / "verified.json").exists()


def test_huge_finite_times_keep_measured_roundoff_not_zeroed():
    st, ev, geom = sample()
    st = st.iloc[:1].copy()
    st["t0"], st["t1"] = 1e16, 1e16
    rows, frames = audit.classify(st, ev, geom)
    assert rows.fold_start_roundoff.iloc[0]
    assert audit.summarize(rows, frames)["max_abs_fold_start_residual_ns"] > 0
