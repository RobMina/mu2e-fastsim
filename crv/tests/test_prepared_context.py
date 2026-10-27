"""Prepared setup reuse preserves the existing orchestrator and fails closed on mutation."""
import copy
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_e2e_bench import B, R, detail, gen, invoke, pipeline


@pytest.fixture
def prepared_pipeline(pipeline, monkeypatch):
    from crvfastsim.reco import waveform as WF
    from functools import lru_cache

    @lru_cache(maxsize=4)
    def template(*args):
        return np.array([0., 1., 0.])

    monkeypatch.setattr(WF, "single_pe_template", template)
    monkeypatch.setattr(R.PH, "_TABLE_PROP", {})
    pipeline["load_counts"] = {}
    for obj, name in ((R.tr, "load_world"), (R.tier1b, "read_library"), (R.PH, "load_tables"), (R.SI, "photon_map")):
        original = getattr(obj, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            pipeline["load_counts"][_name] = pipeline["load_counts"].get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(obj, name, counted)
    return pipeline


def prepare(state, **kwargs):
    return R.prepare(library=state["library"], calib="nominal", gdml=state["gdml"], **kwargs)


def test_reuse_once_and_fresh_setup_outputs_exact(prepared_pipeline, tmp_path):
    state = prepared_pipeline
    baseline = invoke(state, repeat=3, seed=17)
    state["load_counts"].clear()
    cached = invoke(state, outdir=str(tmp_path / "cached"), repeat=3, seed=17, reuse_context=True)
    assert set(state["load_counts"].values()) == {1}
    assert cached["context_policy"] == "prepared_once" and cached["context_audits_passed"]
    assert cached["preparation_elapsed_s"] > 0
    assert cached["invocation_elapsed_before_report_s"] > cached["passes_elapsed_s"]
    for fresh, reused in zip(baseline["passes"], cached["passes"]):
        assert fresh["files"][0]["fingerprint"] == reused["files"][0]["fingerprint"]
        assert reused["runchain"]["timers"]["setup_s"] == 0
        assert reused["runchain"]["timers"]["context_audit_s"] > 0
        assert reused["generated_events_per_s"] == 3 / reused["elapsed_s"]
    assert len(set(state["calls"])) == 1


def test_reuse_prefetch_multiple_inputs_and_new_seed(prepared_pipeline, tmp_path, monkeypatch):
    state = prepared_pipeline
    first = state["inputs"][0]
    second = tmp_path / "second.root"
    second.write_bytes(b"second")
    monkeypatch.setattr(R.tables, "read", lambda path, tree: gen() if str(path) == first else gen(((2, 3, 3), (2, 3, 4))))

    def clusters(w, *args, timers, **kwargs):
        timers["pulses"] = 1
        return [detail(event=int(w["ev"][0]))], timers

    monkeypatch.setattr(R.chain, "clusters", clusters)
    a = invoke(state, inputs=[first, str(second)], reuse_context=True, prefetch=True, repeat=3)
    b = invoke(state, inputs=[first, str(second)], reuse_context=True, prefetch=True,
               repeat=3, seed=9, outdir=str(tmp_path / "other-seed"))
    assert a["repeat_fingerprints_identical"] and b["repeat_fingerprints_identical"]
    assert all(p["generated_events"] == 5 for p in a["passes"])
    assert state["calls"][:2] == state["calls"][2:4] == state["calls"][4:6]
    assert state["calls"][:2] != state["calls"][6:8]


@pytest.mark.parametrize("option", [{"window": "onspill"}, {"no_absorbers": True}, {"unbanded_world": True},
                                     {"no_below_world": True}, {"fused_response": True}])
def test_configuration_mismatch_rejected_before_events(prepared_pipeline, tmp_path, option):
    ctx = prepare(prepared_pipeline)
    with pytest.raises(ValueError, match="does not match"):
        R.run(prepared_pipeline["inputs"], str(tmp_path / "bad"), library=prepared_pipeline["library"],
              calib="nominal", gdml=prepared_pipeline["gdml"], prepared=ctx, **option)
    assert not prepared_pipeline["calls"]


def test_owned_config_and_readonly_arrays(prepared_pipeline):
    ctx = prepare(prepared_pipeline)
    assert ctx.payload["rc"].sectors is not R.runconfig.EXTRACTED_V04.sectors
    assert ctx.payload["cfg"].response.sectors["EX"] is not R.PH.EXTRACTED_V05["EX"]
    with pytest.raises(ValueError, match="read-only"):
        ctx.payload["cond"].status[0] = 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.signature = {}
    with pytest.raises(TypeError):
        ctx.payload["cond"] = None
    ctx.audit()


@pytest.mark.parametrize("mutate", [
    lambda c: setattr(c.payload["cond"], "status", np.ones(8, dtype=np.int64)),
    lambda c: setattr(c.payload["geom"], "lib_group", np.array(["changed"], dtype=object)),
    lambda c: setattr(c.payload["cfg"].response.sectors["EX"], "yield_per_mev", -1.),
    lambda c: c.payload["ccfg"].update(changed=[1, 2]),
])
def test_mutable_structure_drift_detected(prepared_pipeline, mutate):
    ctx = prepare(prepared_pipeline)
    mutate(ctx)
    with pytest.raises(ValueError, match="context mutated"):
        ctx.audit()


def test_global_cache_changes_detected(prepared_pipeline):
    from crvfastsim.reco import waveform as WF
    ctx = prepare(prepared_pipeline)
    R.PH._TABLE_PROP[("foreign", 1.)] = (1., 2.)
    with pytest.raises(ValueError, match="propagation cache"):
        ctx.audit()
    R.PH._TABLE_PROP.clear()
    WF.single_pe_template.cache_clear()
    with pytest.raises(ValueError, match="template cache"):
        ctx.audit()


def test_mutation_during_execution_withholds_report(prepared_pipeline, monkeypatch):
    old = R.chain.charges

    def charges(st, pa, geom, cond, cfg, rng, **kw):
        result = old(st, pa, geom, cond, cfg, rng, **kw)
        cond.status = np.ones_like(cond.status)
        return result

    monkeypatch.setattr(R.chain, "charges", charges)
    with pytest.raises(ValueError, match="context mutated"):
        invoke(prepared_pipeline, reuse_context=True)
    assert not (Path(prepared_pipeline["outdir"]) / "benchmark.json").exists()


def test_fused_reuse_refused_at_admission(prepared_pipeline):
    with pytest.raises(ValueError, match="memory-dependent"):
        invoke(prepared_pipeline, reuse_context=True, fused_response=True)
    assert not Path(prepared_pipeline["outdir"]).exists()


def test_reuse_final_asset_hash_still_required(prepared_pipeline):
    prepared_pipeline["mutate"] = lambda out: prepared_pipeline["asset"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance changed"):
        invoke(prepared_pipeline, reuse_context=True)
    assert not (Path(prepared_pipeline["outdir"]) / "benchmark.json").exists()
