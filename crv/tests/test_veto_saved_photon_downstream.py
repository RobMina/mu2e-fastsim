"""Four-arm orchestration, structural joins and fail-closed saved-stage verification."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto import saved_photon_downstream as D
from crvfastsim.response import sipm as SI
from crvfastsim.reco import waveform as WF


def calibration():
    return pd.DataFrame({"event": [1, 1], "chan": [0, 1], "roc": [1, 1], "feb": [2, 2],
                         "pedestal": [100.] * 2, "pulsearea": [1000.] * 2, "pulseheight": [20.] * 2,
                         "timeoffset": [0.] * 2, "status": [0] * 2, "w": [209.46] * 2})


def photons():
    n = 80
    q = pd.DataFrame({"event": np.ones(n, dtype=int), "chan": np.zeros(n, dtype=int), "photon_id": np.arange(n),
                      "time_historical_relative": np.full(n, 500.), "time_raw_wrapped_relative": np.full(n, 502.)})
    for arm in D.OP.ARMS:
        q[f"{arm}_kept"] = True
    return q


def test_structural_comparison_detects_change_addition_removal_no_nearest_time():
    keys, fields = D.STAGES["digis"]
    a = pd.DataFrame({"event": [1, 1], "chan": [0, 0], "starttdc": [30, 31], "sample": [0, 0], "adc": [110, 120]})
    b = a.copy()
    b.loc[0, "adc"] += 1
    b.loc[1, "starttdc"] += 1
    stats, diff = D.compare_rows(a, b, keys, fields)
    assert (stats["only_a"], stats["only_b"], stats["changed_common"]) == (1, 1, 1)
    assert stats["max_abs_delta_common"]["adc"] == 1
    assert not stats["exact"] and len(diff) == 3
    assert D.compare_rows(a, a.iloc[::-1], keys, fields)[0]["exact"]


def test_compare_rejects_duplicate_and_nonfinite():
    a = pd.DataFrame({"event": [1], "chan": [0], "x": [1.]})
    with pytest.raises(ValueError, match="duplicate"):
        D.compare_rows(pd.concat([a, a]), a, ["event", "chan"], ["x"])
    with pytest.raises(ValueError, match="nonfinite"):
        D.compare_rows(a.assign(x=np.nan), a, ["event", "chan"], ["x"])
    assert D.compare_rows(a.iloc[:0], a.iloc[:0], ["event", "chan"], ["x"])[0]["exact"]


def test_arm_driver_with_real_synthetic_signal_preserves_keys(monkeypatch):
    cfg = dataclasses.replace(SI.SiPMConfig(), p1=1., p2=.001, cross_talk=0.)
    wcfg = dataclasses.replace(WF.WaveformConfig(), noise=0.)
    t = np.arange(0., wcfg.single_pe_max_time, wcfg.single_pe_precision)
    monkeypatch.setattr(WF, "single_pe_template", lambda *args: .06 * np.exp(-.5 * ((t - 35.) / 14.) ** 2))
    q, cond = photons(), calibration()
    dark = D.SC.thermal_candidates(cond, cfg, 33)
    a, ac = D.run_arm(q, cond, np.ones((40, 40)), cfg, wcfg, 33, "historical", dark)
    b, bc = D.run_arm(q, cond, np.ones((40, 40)), cfg, wcfg, 33, "arrival_only", dark)
    assert ac["photons"] == bc["photons"] == len(q)
    assert ac["charges"] > 0 and ac["digis"] > 0 and ac["pulses"] > 0
    pd.testing.assert_frame_equal(a["candidates"].drop(columns="time"), b["candidates"].drop(columns="time"))
    keys, fields = D.STAGES["charges"]
    stats, _ = D.compare_rows(a["charges"], b["charges"], keys, fields)
    assert stats["changed_common"] > 0 and stats["max_abs_delta_common"]["time"] == 2.
    assert a["pulses"].starttdc.notna().all()
    assert set(a["digis"]["sample"]) == set(range(12))
    with pytest.raises(ValueError, match="unknown"):
        D.run_arm(q, cond, np.ones((40, 40)), cfg, wcfg, 33, "bad", dark)


def test_conditions_full_tuple_guard_and_quiet_union(tmp_path):
    path = tmp_path / "stock.root"
    c = calibration().drop(columns=["event", "w"]).rename(columns={"chan": "channel"})
    c = pd.concat([c, c.assign(channel=c.channel + 2)], ignore_index=True)
    with uproot.recreate(path) as root:
        root.mktree("crvdump/calib", {k: c[k].dtype for k in c})
        root["crvdump/calib"].extend({k: c[k].to_numpy() for k in c})
    ev = pd.DataFrame({"run": [1430], "subrun": [0], "event": [1], "pbtime": [-209.46]})
    q = pd.DataFrame({"run": [1430, 1430], "subrun": [0, 0], "event": [1, 1], "chan": [0, 1], "live_channel": [True, True]})
    result = D.conditions(path, ev, q, nbars=1)
    assert len(result) == 2
    assert np.all(result.w == -float(np.float32(-209.46)))
    with pytest.raises(ValueError, match="full event"):
        D.conditions(path, ev, q.assign(run=1431), nbars=1)
    with pytest.raises(ValueError, match="single-run"):
        D.conditions(path, pd.concat([ev, ev.assign(run=1431)]), q, nbars=1)
    with pytest.raises(ValueError, match="live-channel"):
        D.conditions(path, ev, q.assign(live_channel=False), nbars=1)


def test_no_overwrite_or_invalid_seed(tmp_path):
    with pytest.raises(ValueError, match="new output"):
        D.run("none", "none", "none", tmp_path)
    with pytest.raises(ValueError, match="seed"):
        D.run("none", "none", "none", tmp_path / "new", seed=True)


def saved(tmp_path):
    outputs, stages = {}, {}
    for arm in D.OP.ARMS:
        stages[arm] = {}
        for name, (keys, fields) in D.STAGES.items():
            table = pd.DataFrame({k: [1] for k in keys + fields})
            stages[arm][name] = table
            D.persist(table, tmp_path / f"input0_thermal0_{arm}_{name}.parquet", outputs)
    comp = D.comparisons(stages, tmp_path, "input0_thermal0", outputs)
    input_path = tmp_path / "optical.json"
    input_path.write_text("{}"); digest = D.sha256(input_path)
    for arm in D.OP.ARMS:
        for name, table in stages[arm].items():
            D.persist(table, tmp_path / f"input0_thermal1_{arm}_{name}.parquet", outputs)
    (tmp_path / "optical_verified.json").write_text("{}")
    outputs["optical_verified.json"] = D.sha256(tmp_path / "optical_verified.json")
    report = {"schema": "crv-saved-photon-downstream-v1", "F": None, "timing_closure": False,
              "input_summary": str(input_path), "input_summary_sha256": digest, "assets_sha256": {}, "code_sha256": {},
              "output_sha256": outputs, "blocks": [{"input_sha256": "a" * 64, "models": {
                  "input0_thermal0": {"comparisons": comp}, "input0_thermal1": {"comparisons": comp}}}]}
    (tmp_path / "summary.json").write_text(json.dumps(report))
    return report


def test_saved_verify_no_response_calls(tmp_path, monkeypatch):
    saved(tmp_path)
    monkeypatch.setattr(D, "run_arm", lambda *args: pytest.fail("must not run response"))
    result = D.verify(tmp_path, tmp_path / "verification")
    assert result["comparisons_reproduced_exactly"] is True
    assert result["regenerated_response"] is False
    with pytest.raises(ValueError, match="new verification"):
        D.verify(tmp_path, tmp_path / "verification")


@pytest.mark.parametrize("case", ["payload", "summary"])
def test_saved_changes_fail(tmp_path, case):
    report = saved(tmp_path)
    if case == "payload":
        (tmp_path / "input0_thermal0_historical_charges.parquet").write_bytes(b"bad")
    else:
        report["blocks"][0]["models"]["input0_thermal0"]["comparisons"]["historical_vs_stock_timing"]["charges"]["exact"] = False
        (tmp_path / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        D.verify(tmp_path, tmp_path / "verification")
    assert not (tmp_path / "verification" / "summary.json").exists()


@pytest.mark.parametrize("case", ["missing_stage_hash", "missing_verification", "empty_blocks", "empty_models", "missing_model"])
def test_fail_closed_manifest_coverage(tmp_path, case):
    r = saved(tmp_path)
    if case == "missing_stage_hash":
        del r["output_sha256"]["input0_thermal0_stock_timing_charges.parquet"]
    elif case == "missing_verification":
        del r["output_sha256"]["optical_verified.json"]
    elif case == "empty_blocks":
        r["blocks"] = []
    elif case == "empty_models":
        r["blocks"][0]["models"] = {}
    else:
        del r["blocks"][0]["models"]["input0_thermal1"]
    (tmp_path / "summary.json").write_text(json.dumps(r))
    with pytest.raises(ValueError):
        D.verify(tmp_path, tmp_path / "verification")


def test_all_four_arms_use_their_own_mask_and_time(monkeypatch):
    q = photons().iloc[:4].copy()
    q["time_historical_relative"] = [500., 510., 520., 530.]
    q["time_raw_wrapped_relative"] = [502., 512., 522., 532.]
    masks = {"historical": [True, True, False, False], "arrival_only": [False, True, True, False],
             "rejection_only": [False, True, False, False], "stock_timing": [False, False, True, False]}
    for name, values in masks.items():
        q[f"{name}_kept"] = values
    captured = []

    def capture(p, *args):
        captured.append(p.copy())
        raise RuntimeError("capture only")

    monkeypatch.setattr(D.SC, "optical_candidates", capture)
    for arm in D.OP.ARMS:
        with pytest.raises(RuntimeError, match="capture only"):
            D.run_arm(q, calibration(), np.ones((40, 40)), SI.SiPMConfig(), WF.WaveformConfig(), 10, arm, pd.DataFrame())
        selected = q[q[f"{arm}_kept"]]
        key = "time_historical_relative" if arm in ("historical", "rejection_only") else "time_raw_wrapped_relative"
        np.testing.assert_array_equal(captured[-1].photon_id, selected.photon_id)
        np.testing.assert_array_equal(captured[-1].time, selected[key])
