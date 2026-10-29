"""Bounded runner identity, timing-arm, provenance and saved verification checks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import keyed_optical_smoke as M
from crvfastsim.veto import keyed_optics as K
from test_veto_keyed_optics import fixture, SHA
from crvfastsim.veto.replay import EVENT_TYPES


def sample():
    st, pdg, geom, cfg, tables, deviation = fixture(n=12)
    st["pdg"] = pdg
    st["run"], st["subrun"], st["event"] = 1430, 0, 1
    st.loc[4:7, "event"] = 2
    st.loc[8:, "event"] = 3
    events = pd.DataFrame({"run": [1430]*4, "subrun": [0]*4, "event": [1, 2, 3, 4],
                           "pbtime": [-224.46, -219.46, -214.46, -209.46], "intensity": [1e7]*4}).astype(EVENT_TYPES)
    return st, events, geom, cfg, tables, deviation, np.zeros(4, int)


def make():
    return M.generate(*sample(), SHA, 3027000)


def test_real_synthetic_kernel_orchestration_and_all_frames():
    st, ev, geom, *rest = args = sample()
    q = M.generate(*args, SHA, 3027000)
    assert len(q) > 0
    assert not q.photon_id.duplicated().any()
    frames, channels = M.P.summaries(q, ev, geom)
    assert len(frames) == 4 and frames.photons.iloc[-1] == 0
    assert len(channels) > 0
    for name in M.P.FLAGS:
        assert frames[name].sum() == channels[name].sum() == q[name].sum()


def test_roots_subset_permutation_partition_and_seed_domain():
    q = make()
    np.testing.assert_array_equal(q.photon_id, M.roots(q))
    reverse = q.iloc[::-1]
    np.testing.assert_array_equal(reverse.photon_id, M.roots(reverse))
    np.testing.assert_array_equal(q.photon_id.iloc[::3], M.roots(q.iloc[::3]))
    other = q.assign(draw_seed=np.uint64(3027001))
    assert np.all(M.roots(q) != M.roots(other))
    assert M.roots(q.iloc[:0]).dtype == np.int64


def test_roots_collision_and_duplicate_fail(monkeypatch):
    q = make()
    with pytest.raises(ValueError, match="duplicate"):
        M.roots(pd.concat([q, q.iloc[:1]]))
    monkeypatch.setattr(M.S, "_mix", lambda x: np.zeros_like(np.asarray(x, dtype=np.uint64)))
    with pytest.raises(ValueError, match="collision"):
        M.roots(q)


@pytest.mark.parametrize("column", K.KEYS[2:])
def test_root_all_integer_domains(column):
    q = make()
    changed = q.copy()
    changed[column] += 1
    assert np.all(q.photon_id.to_numpy() != M.roots(changed))


def test_four_arms_keep_candidate_roots_and_correct_time_operations(monkeypatch):
    st, ev, geom, cfg, tables, dev, status = sample()
    w = -float(np.float32(ev.pbtime.iloc[0]))
    st["event"] = 1
    st["t0"] = w + np.resize([1490., 1600., 50., 1690.], len(st))
    st["t1"] = st.t0
    def optics(frame, *args, input_sha256, seed, **kw):
        q = frame[list(K.IDS)].astype(np.int64).copy()
        q["chan"], q["substep"], q["photon_type"], q["ordinal"] = 0, 0, 0, 0
        q["input_sha256"], q["draw_seed"], q["time"] = input_sha256, np.uint64(seed), frame.t0 + 15.
        return q[[*K.KEYS, "time"]]
    monkeypatch.setattr(K, "make_photons", optics)
    q = M.generate(st, ev, geom, cfg, tables, dev, status, SHA, 7)
    assert q.rescued_by_arrival.sum() == 3 and q.rescued_survives_step.sum() == 0
    assert q.raw_folded_upper_flip.sum() == 0
    assert q.historical_kept.sum() == 3 and q.arrival_only_kept.sum() == 6
    np.testing.assert_array_equal(q.rejection_only_kept, q.historical_kept & q.stock_step_kept)
    np.testing.assert_array_equal(q.stock_timing_kept, q.arrival_only_kept & q.stock_step_kept)
    dead = M.generate(st, ev, geom, cfg, tables, dev, np.ones(4, int), SHA, 7)
    np.testing.assert_array_equal(q.photon_id, dead.photon_id)
    assert not dead[list(f"{a}_kept" for a in M.P.ARMS)].to_numpy().any()


def test_no_step_and_nan_only_frames_retained():
    st, ev, geom, cfg, tabs, dev, status = sample()
    for frame in (st.iloc[:0], st.assign(t0=np.nan)):
        q = M.generate(frame, ev, geom, cfg, tabs, dev, status, SHA, 7)
        assert len(q) == 0
        f, c = M.P.summaries(q, ev, geom)
        assert len(f) == 4 and f.photons.sum() == 0 and len(c) == 0


@pytest.mark.parametrize("case", ["duplicate", "missing", "oversized"])
def test_bad_source_domain(case):
    st, ev, *rest = sample()
    if case == "duplicate":
        st.loc[1, "source_row"] = st.source_row.iloc[0]
    elif case == "missing":
        st["event"] = 99
    else:
        st = pd.concat([st]*1000, ignore_index=True)
    with pytest.raises(ValueError):
        M.generate(st, ev, *rest, SHA, 7)


def test_deterministic_new_frame_selection():
    frames = pd.DataFrame({"run": [1]*6, "subrun": [0]*6, "event": [1, 2, 31, 1124, 2505, 9],
                           "fold_start_roundoff": [0, 0, 0, 1, 0, 0], "stock_keep": [1, 1, 0, 0, 0, 1]})
    chosen = M.select_frames(frames.iloc[::-1])
    assert chosen.event.tolist() == [1, 31]
    with pytest.raises(ValueError, match="duplicate"):
        M.select_frames(pd.concat([frames, frames.iloc[:1]]))


def test_new_outputs_and_seed_schedule_fail_before_read(tmp_path):
    args = [tmp_path / "missing"]*6
    with pytest.raises(ValueError, match="new output"):
        M.run(*args, tmp_path, [1, 2])
    for seeds in ([1, 1], [1], [-1, 2]):
        with pytest.raises(ValueError):
            M.run(*args, tmp_path / "new", seeds)
    with pytest.raises(ValueError, match="new verification"):
        M.verify(tmp_path / "missing", tmp_path)


@pytest.mark.parametrize("mutation", ["time_raw_wrapped_relative", "photon_id", "stock_timing_kept", "event_row"])
def test_saved_reconstruction_detects_corruption(mutation):
    st, ev, geom, cfg, tabs, dev, status = sample()
    q = M.generate(st, ev, geom, cfg, tabs, dev, status, SHA, 7)
    expected = M.timing_table(q[list(K.KEYS)].assign(time=q.time_raw_absolute),
                              q[list(K.KEYS)].assign(time=q.time_historical_relative), st, ev, status, SHA, 7)
    changed = q.copy()
    if changed[mutation].dtype == bool:
        changed.loc[0, mutation] = not changed.loc[0, mutation]
    else:
        changed.loc[0, mutation] += 1
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(changed, expected, check_exact=True)


def test_census_selection_required_hashes_and_global_rows(tmp_path):
    from crvfastsim.veto.replay import sha256
    frames = pd.DataFrame({"run": [1, 1], "subrun": [0, 0], "event": [1, 31],
                           "fold_start_roundoff": [0, 0], "stock_keep": [1, 0]})
    steps = frames[list(M.IDS)].assign(step_row=[0, 1])
    arts = {name: M.save_table(tmp_path, f"input0_{name}", frame) for name, frame in (("steps", steps), ("frames", frames))}
    report = {"schema": "crv-step-timing-audit-v1", "counters": {"sha256": M.P.COUNTERS_SHA},
              "inputs": [{"sha256": SHA, "artifacts": arts}]}
    M._json(tmp_path / "summary.json", report)
    verified = tmp_path / "verified.json"
    M._json(verified, {"summary_sha256": sha256(tmp_path / "summary.json"),
                      "all_ledgers_reclassified_exactly": True, "pinned_geometry_verified": True})
    _, chosen, hashes = M.selections(tmp_path, verified)
    assert len(chosen) == 1 and chosen[0][1].event.tolist() == [1, 31]
    assert len(hashes) == 4
    steps["step_row"] = [0, 0]
    steps.to_parquet(tmp_path / "input0_steps.parquet", index=False)
    with pytest.raises(ValueError, match="changed bound input"):
        M.selections(tmp_path, verified)


def saved_fixture(tmp_path, monkeypatch):
    import uproot
    st, ev, geom, cfg, tabs, dev, status = sample()
    offline, audit, lut = (tmp_path / name for name in ("offline", "audit", "lut"))
    for p in (offline, audit, lut):
        p.mkdir()
    for p in (offline / "counters.txt", offline / "old.cc", lut / "LookupTable_6000_0", audit / "summary.json",
              tmp_path / "verified.json", tmp_path / "mixed.root", tmp_path / "mixed.json", tmp_path / "fast.json"):
        p.write_text(p.name)
    devpath = tmp_path / "dev.txt"
    np.savetxt(devpath, np.column_stack([np.arange(4), dev]), delimiter=",")
    stock = tmp_path / "stock.root"
    with uproot.recreate(stock) as root:
        root.mktree("crvdump/calib", {"channel": "int32", "status": "int32"})
        root["crvdump/calib"].extend({"channel": np.arange(4, dtype=np.int32), "status": status.astype(np.int32)})
    check_path = tmp_path / "check.json"
    check = {"fast_metadata_sha256": M.sha256(tmp_path / "fast.json"), "stock": str(stock), "stock_sha256": M.sha256(stock)}
    M._json(check_path, check)
    digest = M.sha256(tmp_path / "mixed.root")
    paired = tmp_path / "paired.json"
    M._json(paired, {"blocks": [{"input_sha256": digest, "check": str(check_path), "check_sha256": M.sha256(check_path)}]})
    counters_sha = M.sha256(offline / "counters.txt")
    oldsha = M.sha256(offline / "old.cc")
    monkeypatch.setattr(M.P, "ASSETS", {"counters.txt": counters_sha})
    monkeypatch.setattr(M.P, "COUNTERS_SHA", counters_sha)
    monkeypatch.setattr(M.P, "STOCK_SHA", oldsha)
    monkeypatch.setattr(M.P, "LUT_SHA", M.sha256(lut / "LookupTable_6000_0"))
    monkeypatch.setattr(M, "PHOTON_DEV_SHA", M.sha256(devpath))
    monkeypatch.setattr(M.P, "timing_source_bridge", lambda *a: {"test_bridge": True})
    monkeypatch.setattr(M.geometry, "from_offline_config", lambda p: geom)
    monkeypatch.setattr(M.photons, "load_tables", lambda *a: tabs)
    monkeypatch.setattr(M.P.fastframes, "run_config", lambda *a: (cfg, None, None, None))
    monkeypatch.setattr(M.P, "apply_policy", lambda *a: (cfg, {"test_contract": True}))
    record = {"path": str(tmp_path / "mixed.root"), "sha256": digest, "manifest_sha256": M.sha256(tmp_path / "mixed.json")}
    expected = st.rename(columns={"source_row": "step_row"})
    evidence = {str(p): M.sha256(p) for p in (audit / "summary.json", tmp_path / "verified.json")}
    monkeypatch.setattr(M, "selections", lambda *a: ({"stock_source": {"path": str(offline / "old.cc")}},
                                                   [(record, ev, expected)], evidence))
    paths = (tmp_path / "mixed.root", tmp_path / "mixed.json", check_path, tmp_path / "fast.json", stock, devpath)
    bindings = {str(p): M.sha256(p) for p in paths}
    monkeypatch.setattr(M, "prepare", lambda *a: (st, ev, status, dev, bindings))
    out = tmp_path / "smoke"
    report = M.run(audit, tmp_path / "verified.json", paired, devpath, offline, lut, out, [7, 8])
    return out, report


def test_end_to_end_saved_verify_no_optical_regeneration(tmp_path, monkeypatch):
    out, report = saved_fixture(tmp_path, monkeypatch)
    def forbidden(*a, **kw):
        pytest.fail("saved verifier must not generate photons")
    monkeypatch.setattr(K, "make_photons", forbidden)
    result = M.verify(out, tmp_path / "checked.json")
    assert result["saved_ledgers_exact"] and result["root_union_collision_free"]
    assert len(result["checks"]) == 2 and all(r["frames"] == 4 for r in result["checks"])
    assert all(r["photons"] > 0 for r in result["checks"])


@pytest.mark.parametrize("case", ["code", "asset", "evidence", "input_binding", "seed", "block", "snapshot", "artifact"])
def test_saved_required_hashes_and_domains(tmp_path, monkeypatch, case):
    out, report = saved_fixture(tmp_path, monkeypatch)
    if case == "code":
        report["code_sha256"].pop(next(iter(report["code_sha256"])))
    elif case == "asset":
        report["assets"].pop(next(iter(report["assets"])))
    elif case == "evidence":
        report["inputs"].pop(report["paths"]["verified"])
    elif case == "input_binding":
        report["blocks"][0]["input_bindings"].pop(report["paths"]["photon_dev"])
    elif case == "seed":
        report["blocks"][0]["replicas"].pop()
    elif case == "block":
        report["blocks"] = []
    elif case == "snapshot":
        report["code_snapshots"].pop(next(iter(report["code_snapshots"])))
    else:
        report["blocks"][0]["replicas"][0]["artifacts"]["photons"]["path"] = "wrong.parquet"
    (out / "summary.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        M.verify(out, tmp_path / "new_check.json")


def test_conflicting_bindings_never_override_pinned_hash(tmp_path):
    p = tmp_path / "asset"
    p.write_text("asset")
    valid = M.sha256(p)
    with pytest.raises(ValueError, match="conflicting"):
        M.check_bindings({str(p): valid}, {str(p): "0"*64})
    M.check_bindings({str(p): valid}, {str(p): valid})


def test_saved_verify_repeats_admission(tmp_path, monkeypatch):
    out, _ = saved_fixture(tmp_path, monkeypatch)
    def reject(*a, **kw):
        raise ValueError("fixed-stock response/input contract differs")
    monkeypatch.setattr(M, "prepare", reject)
    with pytest.raises(ValueError, match="response/input"):
        M.verify(out, tmp_path / "verified_again.json")
