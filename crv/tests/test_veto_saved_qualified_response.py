"""Transactional orchestration on synthetic saved photons; no optical draws."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto import saved_qualified_response as S, bounded_response as R, frame_registry as F
from test_veto_qualified_response import domain, calibration, configs


def run_fixture(tmp_path, monkeypatch, *, invoke=True):
    r, q, _ = domain()
    _, wcfg = configs(monkeypatch)
    c = calibration()
    inputs = {}
    for name in ("plan", "smoke"):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "summary.json"
        path.write_text(json.dumps({"fixture": name}))
        inputs[str(path)] = S.sha256(path)
    pmap = tmp_path / "map.root"
    with uproot.recreate(pmap) as root:
        root["photonMap"] = np.ones((40, 40)), np.arange(41), np.arange(41)
    template = tmp_path / "template.txt"
    template.write_text("synthetic template bound independently of its patched reader")
    monkeypatch.setattr(R.D, "MAP_SHA", S.sha256(pmap))
    monkeypatch.setattr(R.D, "TEMPLATE_SHA", S.sha256(template))
    photons = {}
    for seed in S.SEEDS:
        value = q.assign(draw_seed=np.uint64(seed))
        value["photon_id"] = S.M.roots(value)
        photons[seed] = value
    verified = {"plan": {"registry_sha256": F.digest(r)}, "smoke": {"saved_ledgers_exact": True}}
    monkeypatch.setattr(S, "admission", lambda *a: (r, r, c, photons, inputs.copy(), verified))
    original = R.run_arm
    def bounded_small(q, registry, selected, calibration, **kwargs):
        return original(q, registry, selected, calibration, **{**kwargs, "nbars": 1})
    monkeypatch.setattr(R, "run_arm", bounded_small)
    monkeypatch.setattr(S.M.K, "make_photons", lambda *a, **k: pytest.fail("optical regeneration"))
    args = dict(expected_plan_sha256=inputs[str(tmp_path / "plan/summary.json")],
                expected_smoke_sha256=inputs[str(tmp_path / "smoke/summary.json")], photon_map=pmap, template=template)
    out = tmp_path / "result"
    summary = S.run(tmp_path / "plan", tmp_path / "smoke", out, **args) if invoke else None
    return out, summary, (r, c, photons, inputs, verified), args


def test_transactional_four_arm_two_seed_validation(monkeypatch, tmp_path):
    out, summary, _, args = run_fixture(tmp_path, monkeypatch)
    assert len(summary["records"]) == 8 and len(summary["structural_comparisons"]) == 6
    assert summary["frames"] == 4 and summary["all_tested_physical_partition_fields_exact"]
    assert summary["F"] is None and summary["execution_admitted"] is False
    assert (out / "run/complete.json").exists() and not (out / "run/failed.json").exists()
    for record in summary["records"]:
        child = out / record["path"]
        receipt = json.loads((child / "complete.json").read_text())
        assert len(receipt["artifacts"]) == len(R.STAGES)*3 + 1
        assert record["partition"]["frames"] == 4 and record["subset"]["frames"] == 2
        assert record["counts"]["live_channels"] == 16
    with pytest.raises(ValueError, match="new saved-response"):
        S.run(tmp_path / "plan", tmp_path / "smoke", out, **args)


@pytest.mark.parametrize("failure", ["response", "partition", "bytes", "changed_input"])
def test_failure_cannot_publish_complete_run(monkeypatch, tmp_path, failure):
    out, _, fixture, args = run_fixture(tmp_path, monkeypatch, invoke=False)
    if failure == "response":
        monkeypatch.setattr(R, "run_arm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected response failure")))
    elif failure == "partition":
        monkeypatch.setattr(R, "compare_partition", lambda *a: (_ for _ in ()).throw(AssertionError("injected partition failure")))
    elif failure == "bytes":
        monkeypatch.setattr(S, "BATCH_CAP", 5000)
    else:
        original = R.compare_partition
        def change(*a):
            (tmp_path / "smoke/summary.json").write_text("changed after admission")
            return original(*a)
        monkeypatch.setattr(R, "compare_partition", change)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        S.run(tmp_path / "plan", tmp_path / "smoke", out, **args)
    assert not (out / "run/complete.json").exists()
    assert (out / "run/failed.json").exists()


def test_saved_table_path_and_count_guard(tmp_path):
    table = pd.DataFrame({"x": [1]})
    path = tmp_path / "table.parquet"
    table.to_parquet(path)
    rec = {"path": path.name, "sha256": S.sha256(path), "rows": 1}
    pd.testing.assert_frame_equal(S._table(tmp_path, rec), table)
    for bad in ({**rec, "path": "../table.parquet"}, {**rec, "rows": 2}, {**rec, "sha256": "0"*64}):
        with pytest.raises(ValueError):
            S._table(tmp_path, bad)


def test_seed_union_roots_with_saved_draws():
    r, q, _ = domain()
    blocks = []
    for seed in S.SEEDS:
        part = q.assign(draw_seed=np.uint64(seed))
        part["photon_id"] = S.M.roots(part)
        blocks.append(part)
    union = pd.concat(blocks, ignore_index=True)
    np.testing.assert_array_equal(union.photon_id, S.M.roots(union))


def test_required_frozen_executed_helpers_are_bound():
    paths = {p.name for p in S.code_paths()}
    assert {"replay_waveform_control.py", "replay_sipm_control.py", "diagnostic_lm.py", "bounded_sipm.py"} <= paths
    assert S.sha256(R.W.W.__file__) == S.FROZEN_WAVEFORM_SHA


def test_preflight_never_executes_response(tmp_path, monkeypatch):
    _, _, fixture, args = run_fixture(tmp_path, monkeypatch, invoke=False)
    original = R.Q.conditions
    monkeypatch.setattr(R.Q, "conditions", lambda r, s, c, d: original(r, s, c, d, nbars=1))
    monkeypatch.setattr(R, "run_arm", lambda *a, **k: pytest.fail("preflight response execution"))
    output = tmp_path / "preflight.json"
    report = S.preflight(tmp_path / "plan", tmp_path / "smoke", output, **args)
    assert report["response_executed"] is False and report["live_channels"] == 16
    with pytest.raises(ValueError, match="new preflight"):
        S.preflight(tmp_path / "plan", tmp_path / "smoke", output, **args)
