"""Synthetic stage-0 retention and admission tests; no detector assets or accelerator required."""
import copy
import dataclasses
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.analyses import e2e_bench as B
from crvfastsim.analyses import runchain as R


def gen(keys=((1, 2, 3), (1, 2, 4), (1, 2, 5))):
    return pd.DataFrame(keys, columns=R.tables.KEY, dtype=np.int64)


def detail(event=(2 << 20) | 3, sector=99):
    return {"cl_event": np.array([event]), "cl_type": np.array([sector]), "cl_pe": np.array([17.0]),
            "cl_start": np.array([1.0]), "cl_end": np.array([2.0]), "cl_slope": np.array([0.0]),
            "cl_avgtime": np.array([1.5]), "cl_poscalc": np.array([False]), "cl_pos": np.array([[1., 2., 3.]]),
            "mem_cl": np.array([0]), "mem_event": np.array([event]), "mem_idx": np.array([0]),
            "pulses": {"ev": np.array([event]), "idx": np.array([0]), "bar": np.array([1]),
                       "sipm": np.array([0]), "seq": np.array([0]), "pe": np.array([17.]), "time": np.array([1.5])}}


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    inp = tmp_path / "gen.root"
    inp.write_bytes(b"synthetic input, not public detector data")
    asset = tmp_path / "asset"
    asset.write_bytes(b"synthetic asset")
    source = tmp_path / "source.py"
    source.write_text("synthetic = True\n")
    state = {"calls": [], "gen": gen(), "detail": detail(), "asset": asset, "source": source, "mutate": None}
    monkeypatch.setattr(B, "source_paths", lambda: [str(source)])
    monkeypatch.setattr(B, "software_hardware", lambda: {"synthetic": True})
    monkeypatch.setattr(B, "accelerator", lambda requested: (nullcontext(), {"response": "synthetic CPU"}))

    def setup(self, *args):
        self.provenance.setdefault("assets_sha256", {str(asset): B.sha256(asset)})

    monkeypatch.setattr(B.Recorder, "setup", setup)
    monkeypatch.setattr(R.runconfig.RunConfig, "geometry", lambda self: SimpleNamespace(nbars=2))
    monkeypatch.setattr(R.runconfig.RunConfig, "counters_cfg", lambda self: {})
    monkeypatch.setattr(R.runconfig.RunConfig, "bar_cut", lambda self, geom: None)
    monkeypatch.setattr(R.tr, "msc_plane", lambda geom: 0)
    monkeypatch.setattr(R.tr, "load_world", lambda *a, **kw: (None, None))
    monkeypatch.setattr(R.tier1b, "read_library", lambda path: None)
    monkeypatch.setattr(R.PH, "load_tables", lambda cfg: {})
    monkeypatch.setattr(R.SI, "photon_map", lambda: None)
    monkeypatch.setattr(R.tables, "read", lambda *args: state["gen"].copy())

    def transport(g, geom, rng, **kw):
        state["calls"].append(float(rng.random()))
        return g.iloc[:1].copy(), g.iloc[:1].copy()

    monkeypatch.setattr(R.tr, "transport", transport)
    monkeypatch.setattr(R.tier1b, "sample", lambda st, pa, *a, **kw: (st, pa, None))

    def charges(st, pa, geom, cond, cfg, rng, **kw):
        kw["timers"].update(photons=2, charges=1)
        return {"chan": R.chain.event_key(st) * R.chain.CHAN_MULT, "time": np.array([1.]),
                "charge": np.array([1.]), "pe": np.array([1.]), "photon": np.array([0])}

    def digis(q, cond, cfg, rng, timers):
        timers["digis"] = 1
        return {"chan": q["chan"], "starttdc": np.array([0]), "adc": np.zeros((1, 12))}

    def clusters(w, cond, geom, ccfg, cfg, timers, **kw):
        timers["pulses"] = 1
        out = copy.deepcopy(state["detail"])
        if state["mutate"]:
            state["mutate"](out)
        if not kw.get("detail"):
            out = {k: v for k, v in out.items() if k.startswith("cl_") and k != "cl_pos"}
        return [out], timers

    monkeypatch.setattr(R.chain, "charges", charges)
    monkeypatch.setattr(R.chain, "digis", digis)
    monkeypatch.setattr(R.chain, "reco_waveforms", lambda d, c: {"ev": d["chan"] // R.chain.CHAN_MULT})
    monkeypatch.setattr(R.chain, "clusters", clusters)
    state.update(inputs=[str(inp)], outdir=str(tmp_path / "bench"), library=str(asset), calib="nominal", gdml=str(asset))
    return state


def invoke(state, **kw):
    opts = {k: state[k] for k in ("inputs", "outdir", "library", "calib", "gdml")}
    return B.run(**{**opts, **kw})


def test_repeat_retains_universe_raw_details_counts_and_identity(pipeline):
    result = invoke(pipeline, repeat=3, seed=62)
    assert [p["stage"] for p in result["passes"]] == ["cold", "warm", "warm"]
    assert len(set(pipeline["calls"])) == 1
    assert result["repeat_fingerprints_identical"] is True
    assert result["cry_generation_timed"] is False
    first = result["passes"][0]
    assert first["generated_events"] == 3
    assert first["generated_events_per_s"] == 3 / first["elapsed_s"]
    record = first["files"][0]
    assert record["counts"]["pulses"] == record["counts"]["clusters"] == record["counts"]["members"] == 1
    assert record["counts"]["photons"] == 2
    assert record["cluster_type_counts"] == {"99": 1, "1": 0, "2": 0, "3": 0, "4": 0}
    assert {"input_io_s", "transport_s", "tier1b_s", "waveform_pack_s", "reco_s"}.issubset(record["phase_wall_s"])
    with np.load(f'{pipeline["outdir"]}/pass_000/input_0000.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["generated_keys"], pipeline["gen"].to_numpy())
        np.testing.assert_array_equal(arrays["rng_seed_and_input_index"], [62, 0])
        np.testing.assert_array_equal(arrays["chunk_0000__cl_pos"], [[1., 2., 3.]])
        np.testing.assert_array_equal(arrays["chunk_0000__pulses__time"], [1.5])
        assert B.fingerprint(dict(arrays)) == record["fingerprint"]
    report = json.loads((Path(pipeline["outdir"]) / "benchmark.json").read_text())
    assert report["status"] == "complete"


def test_selection_keeps_full_input_and_actual_selected_universe(pipeline):
    invoke(pipeline, repeat=1, max_events=1)
    with np.load(f'{pipeline["outdir"]}/pass_000/input_0000.npz') as arrays:
        assert arrays["generated_keys"].shape == (1, 3)
        assert arrays["input_generated_keys"].shape == (3, 3)


def test_uninstrumented_runchain_keeps_seed_and_cluster_result(pipeline, tmp_path):
    result = invoke(pipeline, repeat=1)
    regular = R.run(pipeline["inputs"], str(tmp_path / "regular"), library=pipeline["library"],
                    calib="nominal", gdml=pipeline["gdml"])
    assert regular["clusters_per_generated"] == result["passes"][0]["runchain"]["clusters_per_generated"]
    assert pipeline["calls"][0] == pipeline["calls"][1]
    assert "detail_output_s" not in regular["timers"]
    assert "waveform_pack_s" not in regular["timers"]


@pytest.mark.parametrize("keys,match", [(((1, 2, 2 ** 20),), "20-bit"),
                                        (((1, 2, 3), (2, 2, 3)), "collision"),
                                        (((1, -1, 3),), "nonnegative"),
                                        (((1, 2 ** 44, 3),), "overflows")])
def test_reject_ambiguous_event_keys(keys, match):
    with pytest.raises(ValueError, match=match):
        B.event_keys(gen(keys))


def test_multi_primary_rows_are_one_event_but_disjoint_runs_are_safe():
    keys, packed = B.event_keys(gen(((1, 2, 3), (1, 2, 3), (2, 3, 3))))
    assert keys.shape == (2, 3)
    assert len(np.unique(packed)) == 2


def test_duplicate_event_across_inputs_rejected(tmp_path):
    recorder = B.Recorder(tmp_path, 1, {})
    recorder.check_input(0, "a", gen())
    with pytest.raises(ValueError, match="collision"):
        recorder.check_input(1, "b", gen(((7, 2, 3),)))


@pytest.mark.parametrize("change,match", [
    (lambda d: d.update(mem_idx=np.array([9])), "dangling member pulse"),
    (lambda d: d.update(mem_cl=np.array([2])), "dangling member cluster"),
    (lambda d: d.update(mem_event=np.array([(2 << 20) | 4])), "wrong event"),
    (lambda d: d.update(cl_event=np.array([999]), mem_event=np.array([999])), "dangling member pulse"),
    (lambda d: d["pulses"].update(ev=np.array([999])), "dangling member pulse"),
    (lambda d: d.update(cl_type=np.array([])), "truncated"),
    (lambda d: d["pulses"].update(time=np.array([])), "truncated"),
    (lambda d: d.update(mem_idx=np.array([], dtype=int)), "truncated"),
    (lambda d: d.update(cl_pos=np.zeros((1, 2))), "truncated"),
    (lambda d: d.update(input_truncated=True), "incomplete"),
    (lambda d: d.update(response_overflow_batches=1), "incomplete"),
    (lambda d: d["pulses"].update(idx=np.array([7])), "noncontiguous"),
])
def test_detail_fail_closed(change, match):
    d = detail()
    change(d)
    with pytest.raises(ValueError, match=match):
        B.detail_arrays(gen().to_numpy(), [d])


def test_output_outside_universe_rejected():
    with pytest.raises(ValueError, match="outside generated"):
        B.detail_arrays(gen().to_numpy(), [detail(event=999)])


def test_zero_output_events_and_empty_chunk_tables():
    arrays, counts, sectors = B.detail_arrays(gen().to_numpy(), [])
    assert counts["pulses"] == counts["clusters"] == 0
    assert arrays["generated_keys"].shape == (3, 3)
    assert arrays["events_with_pulses_keys"].shape == (0, 3)
    d = detail()
    d = {k: {pk: pv[:0] for pk, pv in v.items()} if k == "pulses" else v[:0] for k, v in d.items()}
    assert B.detail_arrays(gen().to_numpy(), [d])[1]["pulses"] == 0


def test_different_chunk_cluster_indices_stay_local():
    arrays, counts, sectors = B.detail_arrays(gen().to_numpy(), [detail(), detail(event=(2 << 20) | 4, sector=8)])
    assert counts["clusters"] == 2
    assert sectors == {"99": 1, "8": 1}
    assert arrays["chunk_0001__mem_cl"].tolist() == [0]


def test_duplicate_chunk_pulses_rejected():
    with pytest.raises(ValueError, match="across reconstruction chunks"):
        B.detail_arrays(gen().to_numpy(), [detail(), detail()])


def test_fingerprint_binds_dtype_shape_and_values():
    a = {"x": np.array([1, 2])}
    assert B.fingerprint(a) != B.fingerprint({"x": a["x"].reshape(1, 2)})
    assert B.fingerprint(a) != B.fingerprint({"x": a["x"].astype(float)})
    assert B.fingerprint(a) != B.fingerprint({"x": np.array([1, 3])})
    with pytest.raises(ValueError, match="object array"):
        B.fingerprint({"x": np.array([object()])})


@pytest.mark.parametrize("truncated", [False, True])
def test_digi_source_clusters_have_independent_row_count(pipeline, monkeypatch, truncated):
    original = R.chain.digis

    def with_source_clusters(*args, **kwargs):
        d = original(*args, **kwargs)
        d.update(cluster_chan=np.array([1, 2]), cluster_tstart=np.zeros(1 if truncated else 2),
                 cluster_keep=np.ones(2, dtype=bool))
        return d

    monkeypatch.setattr(R.chain, "digis", with_source_clusters)
    if truncated:
        with pytest.raises(ValueError, match="source-cluster"):
            invoke(pipeline)
    else:
        assert invoke(pipeline)["repeat_fingerprints_identical"]


def test_repeat_mismatch_withholds_completion(pipeline):
    pipeline["mutate"] = lambda d: d["cl_pe"].__setitem__(0, len(pipeline["calls"]))
    with pytest.raises(ValueError, match="fingerprint differs"):
        invoke(pipeline)
    assert not (Path(pipeline["outdir"]) / "benchmark.json").exists()


@pytest.mark.parametrize("category", ["asset", "source", "input"])
def test_provenance_drift_withholds_completion(pipeline, category):
    path = Path(pipeline["inputs"][0]) if category == "input" else pipeline[category]
    pipeline["mutate"] = lambda d: path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance changed"):
        invoke(pipeline, repeat=1)
    assert not (Path(pipeline["outdir"]) / "benchmark.json").exists()


@pytest.mark.parametrize("option", [{"outdir": None}, {"repeat": 0}, {"max_events": 0}, {"seed": -1},
                                    {"compare": ["stock.root"]}, {"emit": "external.json"}, {"unknown": True},
                                    {"inputs": []}, {"inputs": ["does-not-exist.root"]}])
def test_admission_rejections(pipeline, option):
    with pytest.raises(ValueError):
        invoke(pipeline, **option)


def test_existing_output_and_symlink_refused(pipeline, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        invoke(pipeline, outdir=str(target))
    link = tmp_path / "broken-link"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        invoke(pipeline, outdir=str(link))


def test_empty_input_file_refused(pipeline, tmp_path):
    path = tmp_path / "empty"
    path.touch()
    with pytest.raises(ValueError, match="empty"):
        invoke(pipeline, inputs=[str(path)])


def test_empty_gen_table_refused(pipeline):
    pipeline["gen"] = gen().iloc[:0]
    with pytest.raises(ValueError, match="empty gen"):
        invoke(pipeline)


def test_gpu_cpu_fallback_rejected(monkeypatch):
    import sys
    cpu = SimpleNamespace(platform="cpu")
    fake_jnp = SimpleNamespace(zeros=lambda n: SimpleNamespace(block_until_ready=lambda: SimpleNamespace(devices=lambda: [cpu])))
    fake_jax = SimpleNamespace(numpy=fake_jnp)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    with pytest.raises(ValueError, match="GPU requested"):
        B.accelerator(True)


def test_setup_hashes_all_resolved_assets(tmp_path, monkeypatch):
    from crvfastsim.response import lookup
    paths = {}
    for name in ("gdml", "library", "counters", "template", "calib", "deviation", "photonMap.root", "LookupTable_test"):
        paths[name] = tmp_path / name
        paths[name].write_text(name)
    monkeypatch.setattr(lookup, "lookup_dir", lambda: str(tmp_path))
    monkeypatch.setattr(B.geometry, "find_counters", lambda path: str(paths["template"]))
    rc = dataclasses.replace(R.runconfig.get(), counters=str(paths["counters"]))
    monkeypatch.setattr(R.runconfig.RunConfig, "counters_path", lambda self: str(paths["counters"]))
    cfg = dataclasses.replace(R.chain.ChainConfig(), response=dataclasses.replace(R.PH.ResponseConfig(),
                              sectors={"test": R.PH.SectorConfig("test", 0, 1.)}))
    provenance = {}
    recorder = B.Recorder(tmp_path, 1, provenance)
    spec = R.tier1b_assets.Tier1bSpec(None, str(paths["library"]))
    args = (rc, cfg, str(paths["gdml"]), spec, str(paths["calib"]), str(paths["deviation"]))
    recorder.setup(*args)
    assert set(provenance["assets_sha256"]) == {str(p) for p in paths.values()}
    paths["calib"].write_text("changed")
    with pytest.raises(ValueError, match="asset provenance changed"):
        recorder.setup(*args)


def test_setup_hashes_the_tier1b_sidecars(tmp_path, monkeypatch):
    from crvfastsim.response import lookup
    monkeypatch.setattr(lookup, "lookup_dir", lambda: str(tmp_path))
    monkeypatch.setattr(B.geometry, "find_counters", lambda path: str(tmp_path / "template"))
    monkeypatch.setattr(R.runconfig.RunConfig, "counters_path", lambda self: str(tmp_path / "counters"))
    spec = R.tier1b_assets.resolve(directory=tmp_path / "set")
    (tmp_path / "set").mkdir()
    for f in ["template", "counters", "gdml", "calib", "photonMap.root"] + [f"LookupTable_{s.table}" for s in
                                                                           R.PH.ResponseConfig().sectors.values()]:
        (tmp_path / f).write_text(f)
    for f in spec.paths():
        Path(f).write_text(Path(f).name)
    provenance = {}
    B.Recorder(tmp_path, 1, provenance).setup(R.runconfig.get(), R.chain.ChainConfig(), str(tmp_path / "gdml"), spec,
                                              str(tmp_path / "calib"), None)
    assert set(spec.paths()) <= set(provenance["assets_sha256"]) and len(spec.paths()) == 5
    assert provenance["effective_tier1b"]["name"] == "strict_v2" and provenance["effective_tier1b"]["options"]["nearmiss"]
    Path(spec.templates).write_text("changed")
    with pytest.raises(ValueError, match="asset provenance changed"):
        B.Recorder(tmp_path, 1, provenance).setup(R.runconfig.get(), R.chain.ChainConfig(), str(tmp_path / "gdml"),
                                                  spec, str(tmp_path / "calib"), None)


def test_prefetch_multiple_inputs_repeats_and_order(pipeline, tmp_path, monkeypatch):
    second = tmp_path / "second.root"
    second.write_bytes(b"second synthetic input")
    first = pipeline["inputs"][0]
    monkeypatch.setattr(R.tables, "read", lambda path, tree: gen() if str(path) == first else gen(((2, 3, 3), (2, 3, 4))))

    def clusters(w, cond, geom, ccfg, cfg, timers, **kw):
        timers["pulses"] = 1
        return [detail(event=int(w["ev"][0]))], timers

    monkeypatch.setattr(R.chain, "clusters", clusters)
    result = invoke(pipeline, inputs=[first, str(second)], repeat=2, prefetch=True)
    assert all(p["generated_events"] == 5 for p in result["passes"])
    assert result["repeat_fingerprints_identical"]
    assert result["passes"][0]["runchain"]["timers"]["pulses"] == 2
    for i in range(2):
        with np.load(f'{pipeline["outdir"]}/pass_000/input_{i:04d}.npz') as arrays:
            np.testing.assert_array_equal(arrays["rng_seed_and_input_index"], [1, i])


def test_prefetch_failure_has_no_completion(pipeline, tmp_path):
    second = tmp_path / "second.root"
    second.write_bytes(b"same universe in second input")
    with pytest.raises(ValueError, match="collision"):
        invoke(pipeline, inputs=pipeline["inputs"] + [str(second)], prefetch=True)
    assert not (Path(pipeline["outdir"]) / "benchmark.json").exists()


def test_count_mismatch_fails_closed(pipeline, monkeypatch):
    def clusters(w, cond, geom, ccfg, cfg, timers, **kw):
        timers["pulses"] = 2
        return [detail()], timers

    monkeypatch.setattr(R.chain, "clusters", clusters)
    with pytest.raises(ValueError, match="truncated pulses"):
        invoke(pipeline)


def test_retained_output_drift_fails_closed(pipeline):
    def mutate(out):
        if len(pipeline["calls"]) == 2:
            (Path(pipeline["outdir"]) / "pass_000" / "input_0000.npz").write_bytes(b"changed")

    pipeline["mutate"] = mutate
    with pytest.raises(ValueError, match="retained output changed"):
        invoke(pipeline)
    assert not (Path(pipeline["outdir"]) / "benchmark.json").exists()


def test_local_assets_context_restores_environment(monkeypatch):
    import os
    monkeypatch.delenv("CRVFS_NO_FETCH", raising=False)
    with pytest.raises(RuntimeError), B.local_assets_only():
        assert os.environ["CRVFS_NO_FETCH"] == "1"
        raise RuntimeError("synthetic failure")
    assert "CRVFS_NO_FETCH" not in os.environ


def test_module_cli_passes_runchain_options(monkeypatch):
    seen = []
    monkeypatch.setattr(B, "run", lambda **kwargs: seen.append(kwargs))
    B.main(["gen.root", "--outdir", "new", "--library", "lib.root", "--calib", "nominal", "--repeat", "4",
            "--prefetch", "--jax", "--fused-response", "--run-config", "run2_v10", "--seed", "17"])
    assert seen[0]["repeat"] == 4
    assert seen[0]["prefetch"] and seen[0]["jax"] and seen[0]["fused_response"]
    assert seen[0]["run_config"] == "run2_v10" and seen[0]["seed"] == 17
    with pytest.raises(SystemExit):
        B.main(["gen.root", "--library", "lib.root", "--calib", "nominal"])
