"""Synthetic saved-CSS replay tests; no production files, JAX execution or frozen adapters."""
from dataclasses import dataclass
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from calofastsim.showers import develop_reco as replay


def saved_steps():
    return dict(query_event_rows=np.array([2, 102, 999]), event_row=np.array([2, 102, 999]),
                crystal=np.array([1, -1, 0]), mapped=np.array([True, False, True]),
                energy_g4=np.array([11., 12., 13.]), energy_birks=np.array([10., 11., 12.]),
                time=np.array([500., 501., 502.]), local_position=np.arange(9.).reshape(3, 3))


def simulation():
    return dict(lru=np.array([.1, .2]), pe_per_mev=np.array([[30., 31.], [40., 41.]]))


def test_vectorized_mapping_keeps_source_rows_and_drops_unmapped():
    steps, query = replay.steps_for_backend(saved_steps(), simulation())
    np.testing.assert_array_equal(query, [2, 102, 999])
    np.testing.assert_array_equal(steps["event"], [2, 999])
    np.testing.assert_array_equal(steps["energy_g4"], [11, 13])
    np.testing.assert_array_equal(steps["z_local"], [2, 8])
    np.testing.assert_array_equal(steps["yields"], [[40, 41], [30, 31]])
    np.testing.assert_array_equal(steps["lru"], [.2, .1])


@pytest.mark.parametrize("key,value,error", [
    ("query_event_rows", np.array([2, 2, 999]), "unique"),
    ("query_event_rows", np.array([-1, 102, 999]), "development"),
    ("query_event_rows", np.array([2, 102, 1000]), "development"),
    ("query_event_rows", np.array([2., 102., 999.]), "integers"),
    ("event_row", np.array([2, 1000, 999]), "non-query"),
    ("crystal", np.array([-1, -1, 0]), "calibration"),
    ("crystal", np.array([2, -1, 0]), "calibration"),
    ("mapped", np.array([1, 0, 1]), "boolean"),
    ("energy_g4", np.ones(2), "shape"),
    ("local_position", np.zeros((3, 2)), "shape"),
    ("time", np.array([np.nan, 0, 1]), "nonfinite"),
])
def test_reject_invalid_canonical_arrays(key, value, error):
    source = saved_steps()
    source[key] = value
    with pytest.raises(ValueError, match=error):
        replay.steps_for_backend(source, simulation())


def test_queries_must_match_including_order():
    with pytest.raises(ValueError, match="identical"):
        replay.steps_for_backend(saved_steps(), simulation(), np.array([102, 2, 999]))


def test_empty_steps_and_empty_queries():
    source = {k: v[:0] for k, v in saved_steps().items()}
    steps, query = replay.steps_for_backend(source, simulation())
    assert query.size == steps["event"].size == 0 and steps["yields"].shape == (0, 2)
    source["query_event_rows"] = np.array([2, 102])
    steps, query = replay.steps_for_backend(source, simulation())
    assert len(query) == 2 and not len(steps["event"])
    source = saved_steps()
    source["mapped"][:] = False
    assert not len(replay.steps_for_backend(source, simulation())[0]["event"])


def prepared_inputs(tmp_path):
    conditions = dict(run=np.ones(2000, int), subrun=np.arange(2000) // 100,
                      event=np.arange(2000) % 100, spill=np.arange(2000) % 2)
    conditions.update({k: np.arange(2000, dtype=float) for k in ("pbtime", "mbtime", "marker", "event_length", "crystal_length")})
    path = tmp_path / "inputs.npz"
    np.savez(path, **{"condition_" + k: v for k, v in conditions.items()}, stock_cluster_energy=np.full(2000, -999.))
    return path


def stock_reader(calls):
    def read(path, ids):
        calls.append((str(path), ids.copy()))
        return dict(energy=np.full(2000, 22. if str(path).endswith("2.root") else 20.),
                    **{k: np.arange(2000, dtype=float) for k in ("time", "x", "y", "z")})
    return read


def test_inputs_join_both_full_2000_id_sets_before_selecting(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(replay, "stock_clusters", stock_reader(calls))
    timing, stocks = replay.development_inputs(prepared_inputs(tmp_path), "stock1.root", "stock2.root")
    assert [len(ids) for _, ids in calls] == [2000, 2000]
    assert calls[0][1].dtype.names == ("run", "subrun", "event")
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    np.testing.assert_array_equal(timing["pbtime"], np.arange(1000))
    np.testing.assert_array_equal(timing["onspill"], np.arange(1000) % 2 == 1)
    assert stocks[0]["energy"].shape == (1000,) and np.all(stocks[1]["energy"] == 22)


def test_stock_identity_failure_is_not_hidden_by_development_slice(tmp_path, monkeypatch):
    def missing(path, ids):
        assert len(ids) == 2000
        raise ValueError("stock ntuple does not contain every input event once")
    monkeypatch.setattr(replay, "stock_clusters", missing)
    with pytest.raises(ValueError, match="every input event"):
        replay.development_inputs(prepared_inputs(tmp_path), "stock1", "stock2")


@dataclass(frozen=True)
class FitConfig:
    iterations: int = 15


def backend(drift=None):
    draws, timings = [], []
    def run(steps, timing, adc, pulse, noise, geom, rng, *, table, adc2mev, fit_cfg=FitConfig()):
        draws.append(rng.random())
        timings.append(len(timing["pbtime"]))
        clusters = dict(event=np.array([2, 999]), energy=np.array([20., 30.]),
                        **{k: np.array([1., 2.]) for k in ("time", "x", "y", "z")})
        if drift and len(draws) > 1:
            clusters[drift][0] += 1
        return dict(clusters=clusters, seconds=dict(readout=.1, template_fit=.2), n_refit_skipped=2, n_nonfinite_reco=0)
    return SimpleNamespace(run_chain=run), SimpleNamespace(adc_per_pe=lambda *a: 1, adc_to_mev=lambda x: 1), draws, timings


def run_stub(chain, io, **kwargs):
    return replay.replay(chain, io, simulation(), None, None, None, None, {}, {"pbtime": np.zeros(1000)},
                         seed=20261108, repeat=3, synchronize=lambda: None, **kwargs)


def test_same_seed_repeats_time_each_call_and_keep_nan_empty_rows(monkeypatch):
    clock = iter([0., 5., 10., 12., 20., 23.])
    monkeypatch.setattr(replay.time, "perf_counter", lambda: next(clock))
    chain, io, draws, timings = backend()
    first, runs = run_stub(chain, io)
    assert timings == [1000] * 3 and draws[0] == draws[1] == draws[2]
    assert [r["reco_seconds"] for r in runs] == [5., 2., 3.]
    assert [r["phase"] for r in runs] == ["cold", "warm", "warm"]
    assert all(r["stage_seconds"]["template_fit"] == .2 for r in runs)
    assert first["energy"].shape == (1000,) and first["energy"][0] == 0 and np.isnan(first["time"][0])


@pytest.mark.parametrize("field", ["energy", "time", "x", "y", "z"])
def test_repeat_checks_all_leading_fields(field):
    chain, io, _, _ = backend(field)
    with pytest.raises(RuntimeError, match="E/t/xyz"):
        run_stub(chain, io)


def test_empty_energy_summary_and_small_sample_no_peak_fit():
    assert replay.energy_summary([0, np.nan])["quantiles_mev"] is None
    result = replay.energy_summary([0, 20, 80])
    assert result["quantiles_mev"][2] == 50 and result["fraction_below70_given_above10"] == .5
    assert "peak" not in result


def test_cli_default_arms_native_backend_conditional_reports(tmp_path, monkeypatch):
    chain, io, draws, timings = backend()
    source = tmp_path / "backend.py"
    source.write_text("# synthetic backend source\n")
    template = SimpleNamespace(fit_digis=lambda: None, _CACHE={}, __file__=str(source))
    chain.__package__, chain.__file__, chain.fit_digis = "test_backend", str(source), template.fit_digis
    monkeypatch.setitem(sys.modules, "test_backend.templatefit", template)
    clears, barriers = [], []
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(config=SimpleNamespace(update=lambda *a: None),
                        default_backend=lambda: "cpu", devices=lambda: ["stub-cpu"], clear_caches=lambda: clears.append(1),
                        effects_barrier=lambda: barriers.append(1)))
    for name in ("pulse", "noise", "propagation"):
        setattr(io, "load_" + name, lambda path: None)
    io.load_sim_crystal = lambda path: simulation()
    cluster = SimpleNamespace(Geometry=SimpleNamespace(load=lambda path: None))
    monkeypatch.setattr(replay, "load_chain", lambda path: (chain, io, cluster))
    stock_calls = []
    monkeypatch.setattr(replay, "stock_clusters", stock_reader(stock_calls))
    path = prepared_inputs(tmp_path)
    for arm in ("stock", "baseline", "crystal"):
        np.savez(tmp_path / (arm + "_steps.npz"), **saved_steps())
    out = tmp_path / "out"
    replay.main(["--steps", str(tmp_path), "--inputs", str(path), "--reco-code", str(tmp_path),
                 "--reco-inputs", str(tmp_path), "--out", str(out), "--device", "cpu", "--stock-ntuple", "1.root",
                 "--stock-ntuple2", "2.root", "--save-clusters"])
    result = json.loads((out / "numbers.json").read_text())
    assert list(result["arms"]) == ["stock", "baseline", "crystal"]
    assert result["seed"] == 20261108 and result["entries"] == 3 and result["source_rows"] == 1000
    assert result["backend"]["native_cache"] and result["backend"]["fit_config"]["iterations"] == 15
    assert result["stock2_vs_stock1"]["events"] == 3
    assert result["stock2_vs_stock1"]["energy_mean"]["difference"] == 2
    assert len(clears) == 3 and len(barriers) == 18 and timings == [1000] * 9
    assert len(set(draws)) == 1 and [len(ids) for _, ids in stock_calls] == [2000, 2000]
    for arm in result["arms"].values():
        assert len(arm["repeats"]) == 3 and arm["events"] == 3
        assert arm["timing"]["source_rows"] == 1000 and arm["timing"]["queried_rows"] == 3
        assert "generated_per_second" not in arm["timing"]
    with np.load(out / "clusters.npz") as saved:
        assert saved["event_row"].shape == saved["stock_energy"].shape == saved["stock1_energy"].shape == (1000,)
        assert saved["stock_energy"][0] == 0 and saved["stock1_energy"][0] == 20
        assert saved["stock2_energy"][0] == 22 and saved["query_event_rows"].shape == (3,)
