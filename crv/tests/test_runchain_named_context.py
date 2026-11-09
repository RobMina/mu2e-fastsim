"""Synthetic R1 integration with prepared-context setup/reuse; no detector or physics closure claims."""
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from test_e2e_bench import B, R, pipeline
from test_prepared_context import prepared_pipeline


def ordinary(state, **kw):
    opts = {k: state[k] for k in ("inputs", "outdir", "library", "calib", "gdml")}
    return R.run(**{**opts, **kw})


@pytest.mark.parametrize("name", [None, *R.runconfig.names()])
@pytest.mark.parametrize("window", [None, "offspill", "onspill", (-10, 900)])
def test_resolver_preserves_default_physics_and_contract(name, window):
    calib = "calib.csv" if name and R.runconfig.get(name).conditions else "nominal"   # nominal refused there
    rc, cfg, gdml, contract, spec = R._resolve("lib", calib, None, "world", False, None, name, window)
    assert spec.name is None and spec.library == "lib" and spec.options == {}     # custom library: no placement
    old = R.runconfig.get(name, window if window or not name else "offspill")
    old_cfg = old.chain_config(jax=False) if old.name != R.runconfig.DEFAULT or window else R.chain.ChainConfig(jax=False)
    assert dataclasses.asdict(cfg) == dataclasses.asdict(old_cfg)
    assert rc == old and gdml == "world"
    assert list(cfg.window) == contract["effective_window_ns"]
    assert contract["generator_convention"]["nominal_offset_window_ns"] == [0, 100005]
    assert contract["stock_onspill_alignment"] == "not_implemented_event_pbtime_required"


def test_stats_emit_and_merge_persist_contract(pipeline, tmp_path):
    emit = tmp_path / "shard.json"
    result = ordinary(pipeline, emit=str(emit), window="onspill")
    shard = json.loads(emit.read_text())
    stats = json.loads((Path(pipeline["outdir"]) / "runchain_stats.json").read_text())
    for key in ("window", "time_window", "conditions"):
        assert result[key] == shard[key] == stats[key]
    assert result["window"] == [200, 1500]
    assert result["time_window"]["time_rebase_ns"] == 0
    assert result["generated_events"] == 3
    merged = R.merge([str(emit)], tmp_path / "merge")
    assert merged["time_window_status"] == "declared_compatible"
    assert merged["conditions"] == result["conditions"]
    assert merged["window"] == result["window"]


@pytest.mark.parametrize("change", [{"calib": "different.csv"}, {"deviation": "different.csv"}])
def test_prepared_rejects_mismatched_conditions_before_events(prepared_pipeline, change):
    state = prepared_pipeline
    context = R.prepare(library=state["library"], calib="nominal", gdml=state["gdml"])
    with pytest.raises(ValueError, match="does not match"):
        ordinary(state, prepared=context, **change)
    assert not state["calls"]


def test_prepared_metadata_is_sealed(prepared_pipeline):
    state = prepared_pipeline
    context = R.prepare(library=state["library"], calib="nominal", gdml=state["gdml"])
    context.payload["conditions_metadata"]["calib"] = "not-nominal.csv"
    with pytest.raises(ValueError, match="context mutated"):
        ordinary(state, prepared=context)
    assert not state["calls"]


@pytest.mark.parametrize("change,match", [("conditions", "conditions"), ("missing_conditions", "conditions"),
                                        ("window", "window"), ("legacy", "legacy")])
def test_merge_guards_before_counting(tmp_path, change, match):
    _, contract = R.resolve_runchain_window("run2_v10")
    first = {"run_config": "run2_v10", "window": [0, 100000], "time_window": contract,
             "conditions": {"sha256": "one"}}
    second = copy.deepcopy(first)
    if change == "conditions":
        second["conditions"]["sha256"] = "two"
    elif change == "missing_conditions":
        del second["conditions"]
    elif change == "legacy":
        del second["time_window"]
    else:
        second["window"] = [200, 1500]
    paths = []
    for i, record in enumerate([first, second]):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(record))
        paths.append(str(path))
    with pytest.raises(ValueError, match=match):
        R.merge(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()
