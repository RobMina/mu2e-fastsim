"""Synthetic R1 integration with agent13 setup/reuse; no detector or physics closure claims."""
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from crvfastsim import named_conditions
from test_e2e_bench import B, R, pipeline
from test_prepared_context import prepared_pipeline


def ordinary(state, **kw):
    opts = {k: state[k] for k in ("inputs", "outdir", "library", "calib", "gdml")}
    return R.run(**{**opts, **kw})


@pytest.mark.parametrize("name", [None, *R.runconfig.names()])
@pytest.mark.parametrize("window", [None, "offspill", "onspill", (-10, 900)])
def test_resolver_preserves_default_physics_and_contract(name, window):
    rc, cfg, gdml, contract = R._resolve("lib", "nominal", None, "world", False, None, name, window)
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


@pytest.mark.parametrize("window", [None, "onspill", (-10, 900)])
def test_manifest_validates_configured_not_effective_window(pipeline, monkeypatch, tmp_path, window):
    seen = []
    metadata = {"schema": "synthetic", "assets": {"calib.csv": {"sha256": "test-only"}}}

    def load(manifest, rc, geom):
        seen.append((manifest, rc, geom))
        return R.runconfig.nominal_conditions(geom), metadata

    monkeypatch.setattr(named_conditions, "load_conditions", load)
    monkeypatch.setattr(R.runconfig.RunConfig, "tagger_bar", lambda *a: None)
    monkeypatch.setattr(R.runconfig, "sector_type_names", lambda *a: {1: "R1"})
    emit = tmp_path / "manifest-shard.json"
    result = ordinary(pipeline, calib=None, run_config="run2_v10", window=window,
                      conditions_manifest="bundle/manifest.json", counters="override.txt", no_below_world=True,
                      emit=str(emit))
    assert len(seen) == 1
    manifest, rc, geom = seen[0]
    assert manifest == "bundle/manifest.json" and geom.nbars == 2
    assert rc.window == R.runconfig.get("run2_v10").window
    assert rc.counters == "override.txt"
    assert result["conditions"] == metadata == json.loads(emit.read_text())["conditions"]
    assert result["time_window"]["configured_window_ns"] == list(rc.window)
    assert result["window"] == ([0, 100000] if window is None else [200, 1500] if window == "onspill" else [-10, 900])


@pytest.mark.parametrize("opts", [{"calib": "nominal"}, {"calib": "calib.csv"},
                                  {"calib": None, "deviation": "photon.csv"}])
def test_manifest_csv_exclusion_precedes_loading(pipeline, monkeypatch, opts):
    monkeypatch.setattr(R, "_load_setup", lambda *a, **kw: pytest.fail("loaded before admission"))
    with pytest.raises(SystemExit, match="cannot be combined"):
        ordinary(pipeline, conditions_manifest="manifest.json", **opts)
    assert not pipeline["calls"]


def test_manifest_validation_failure_precedes_transport(pipeline, monkeypatch):
    def invalid(*args):
        raise ValueError("hash/size mismatch")
    monkeypatch.setattr(named_conditions, "load_conditions", invalid)
    with pytest.raises(ValueError, match="hash/size"):
        ordinary(pipeline, calib=None, conditions_manifest="manifest.json")
    assert not pipeline["calls"]


@pytest.mark.parametrize("entry", ["prepare", "run_prepared", "run_benchmark", "prepare_benchmark"])
def test_manifest_rejected_before_resolution_or_callbacks(monkeypatch, entry):
    monkeypatch.setattr(R, "_resolve", lambda *a, **kw: pytest.fail("resolved before manifest rejection"))
    options = {"conditions_manifest": "manifest.json"}
    if "benchmark" in entry:
        options["benchmark"] = object()
    if entry == "run_prepared":
        options["prepared"] = object()
    with pytest.raises(ValueError, match="not supported with prepared contexts or benchmarks"):
        if entry.startswith("prepare"):
            R.prepare(**options)
        else:
            R.run([], "unused", **options)


def test_benchmark_entry_refuses_manifest_before_output(pipeline):
    opts = {k: pipeline[k] for k in ("inputs", "outdir", "library", "gdml")}
    with pytest.raises(ValueError, match="unsupported benchmark options"):
        B.run(**opts, conditions_manifest="manifest.json")
    assert not Path(pipeline["outdir"]).exists()


@pytest.mark.parametrize("change", [{"calib": "different.csv"}, {"deviation": "different.csv"},
                                    {"conditions_manifest": "manifest.json", "calib": None}])
def test_prepared_rejects_mismatched_conditions_before_events(prepared_pipeline, change):
    state = prepared_pipeline
    context = R.prepare(library=state["library"], calib="nominal", gdml=state["gdml"])
    with pytest.raises(ValueError, match="does not match|not supported"):
        ordinary(state, prepared=context, **change)
    assert not state["calls"]


def test_prepared_metadata_is_sealed(prepared_pipeline):
    state = prepared_pipeline
    context = R.prepare(library=state["library"], calib="nominal", gdml=state["gdml"])
    context.payload["conditions_metadata"]["calib"] = "not-nominal.csv"
    with pytest.raises(ValueError, match="context mutated"):
        ordinary(state, prepared=context)
    assert not state["calls"]


@pytest.mark.parametrize("enabled,prefetch", [(False, False), (True, False), (True, True)])
def test_scatter_opt_in_forwarding_and_diagnostics(pipeline, monkeypatch, tmp_path, enabled, prefetch):
    old = R.tr.transport
    seen = []

    def transport(*args, **kw):
        seen.append(kw)
        if enabled:
            assert kw["below_scatter_in"] is True
            assert kw["scatter_in_step_mm"] == 25
            assert kw["scatter_in_envelope_mm"] == 1200
            assert kw["scatter_in_max_steps"] == 50
            kw["scatter_in_diagnostics"].update(candidates=2)
        else:
            assert not any(k.startswith("scatter_in") or k == "below_scatter_in" for k in kw)
        return old(*args, **kw)

    monkeypatch.setattr(R.tr, "transport", transport)
    second = tmp_path / "second.root"
    second.write_bytes(b"synthetic")
    emit = tmp_path / "scatter.json"
    result = ordinary(pipeline, inputs=[*pipeline["inputs"], str(second)], emit=str(emit), prefetch=prefetch,
                      below_scatter_in=enabled, scatter_in_step_mm=25, scatter_in_envelope_mm=1200,
                      scatter_in_max_steps=50)
    shard = json.loads(emit.read_text())
    assert len(seen) == 2
    if enabled:
        expected = {str(i): {"input": f, "candidates": 2} for i, f in enumerate([*pipeline["inputs"], str(second)])}
        assert result["scatter_in_diagnostics"] == shard["scatter_in_diagnostics"] == expected
    else:
        assert "scatter_in_diagnostics" not in result
        assert shard["scatter_in_diagnostics"] == {}


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
