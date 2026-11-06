"""Persistent response wiring with real timing/chunking/ROOT I/O and mocked device kernels."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from mixing import crv_inputs, fast_detectors
from mixing.stream_backend import Backend


IDS = ("run", "subrun", "event")


def _crv_result(ev):
    from crvfastsim.chain import CHAN_MULT
    n = len(ev)
    pulses = {"ev": ev, "idx": np.arange(n), "bar": np.zeros(n, np.int64),
              "sipm": np.zeros(n, np.int64), "seq": np.zeros(n, np.int64),
              "pe": np.full(n, 12.), "time": np.full(n, 600.)}
    cluster = {name: np.empty(0, np.int64) for name in
               ("cl_event", "cl_type", "cl_poscalc", "mem_cl", "mem_idx", "mem_event")}
    cluster.update(cl_pe=np.empty(0), cl_avgtime=np.empty(0), cl_pos=np.empty((0, 3)), pulses=pulses)
    return {"D": {"chan": ev * CHAN_MULT, "starttdc": np.full(n, 40), "adc": np.ones((n, 12), np.int16)},
            "clusters": [cluster], "timers": {"synthetic_steps": n}}


def _tracker_result(seed, empty=False):
    n = 0 if empty else 2
    return {"ev": np.array([0, 2], np.int64)[:n], "sid": np.array([11, 12])[:n],
            "tdc": np.full((n, 2), seed % 65536), "tot": np.ones((n, 2), np.int64),
            "pmp": np.array([20, 30])[:n], "adc": np.ones((n, 3), np.int64)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    from crvfastsim import onspill, runconfig
    from crvfastsim.response import photons_jax, sipm
    from trkfastsim.digi import digicond, fastdigi

    devices = Mock(return_value=[SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(jax, "devices", devices)
    monkeypatch.setattr(jax.random, "key", lambda seed: ("key", seed))
    monkeypatch.setattr(jax.random, "fold_in", lambda key, first: (key, first))
    barrier = Mock()
    monkeypatch.setattr(jax, "effects_barrier", barrier, raising=False)
    keys = np.array([[2**40 + 1, 7, 2**53 + 1], [2**40 + 2, 7, 2**53 + 1],
                     [2**40 + 1, 8, 2**53 + 1], [2**40 + 1, 7, 2**53 + 2]], np.uint64)
    mc, reco, length = np.arange(4.) - 215., np.arange(4.) - 214.5, np.arange(4.) + 1675.
    args = SimpleNamespace(truth=tmp_path / "truth.root", stock=tmp_path / "stock.root",
                           photon_csv=tmp_path / "photons.csv", expanded=tmp_path / "expanded.fcl",
                           tracker_conditions=tmp_path / "tracker.npz", chunk=2, seed=137, tracker_seed=251)
    args.expanded.write_text("# synthetic explicit response config\n")
    for path, times, order in ((args.truth, mc, [2, 3, 0, 1]), (args.stock, reco, [1, 0, 3, 2])):
        with uproot.recreate(path) as root:
            root.mktree("crvdump/events", {**dict(zip(IDS, keys[order].T)),
                         "pbtime": times[order], "spill": np.ones(4, int), "evlen": length[order]})
    read_timing = Mock(wraps=crv_inputs.read_timing)
    monkeypatch.setattr(crv_inputs, "read_timing", read_timing)
    geom, counters, cond, tabs, pmap = [object() for _ in range(5)]
    cfg, tracker_cond = SimpleNamespace(response=object()), SimpleNamespace(nADCSamples=3)
    geometry, counters_cfg = Mock(return_value=geom), Mock(return_value=counters)
    mocks = {
        "runconfig": Mock(return_value=SimpleNamespace(geometry=geometry, counters_cfg=counters_cfg)),
        "geometry": geometry, "counters": counters_cfg, "read_timing": read_timing,
        "conditions": Mock(return_value=cond), "config": Mock(return_value=cfg),
        "parse": Mock(return_value={"synthetic": True}), "tabs": Mock(return_value=tabs),
        "pmap": Mock(return_value=pmap), "tracker_conditions": Mock(return_value=tracker_cond),
    }
    monkeypatch.setattr(runconfig, "get", mocks["runconfig"])
    monkeypatch.setattr(crv_inputs, "load_conditions", mocks["conditions"])
    monkeypatch.setattr(crv_inputs, "build_config", mocks["config"])
    monkeypatch.setattr(digicond, "parse_fhicl_dump", mocks["parse"])
    monkeypatch.setattr(photons_jax, "load_device_tables", mocks["tabs"])
    monkeypatch.setattr(sipm, "photon_map", mocks["pmap"])
    monkeypatch.setattr(digicond.Conditions, "load", mocks["tracker_conditions"])
    state = SimpleNamespace(args=args, keys=keys, mc=mc, reco=reco, length=length, mocks=mocks,
                            source=SimpleNamespace(signal=SimpleNamespace(event_ids=keys)),
                            devices=devices, jax=jax, barrier=barrier, crv_calls=[], tracker_calls=[],
                            bad_tracker_keys=False, fail=None, empty=False)

    def crv_kernel(steps, pdg, actual_geom, actual_cond, actual_cfg, rng, *, timing,
                   counters_cfg, key, tabs, pmap, timers, detail):
        assert actual_geom is geom and actual_cond is cond and actual_cfg is cfg
        assert counters_cfg is counters and tabs is mocks["tabs"].return_value
        assert pmap is mocks["pmap"].return_value and detail is True and timers == {}
        state.crv_calls.append(SimpleNamespace(steps=steps, pdg=pdg, timing=timing, key=key,
                                               draw=rng.integers(0, 2**31, 4)))
        if state.fail == "crv":
            raise RuntimeError("synthetic CRV failure")
        return _crv_result(timing.lookup(steps[list(IDS)].to_numpy()))

    def tracker_kernel(steps, actual_cond, key, *, keys, chunk, stats, backend, event_timing):
        assert actual_cond is tracker_cond and chunk == 2 and backend == "jax"
        state.tracker_calls.append(SimpleNamespace(steps=steps, key=key, keys=keys, timing=event_timing))
        if state.fail == "tracker":
            raise RuntimeError("synthetic tracker failure")
        stats["synthetic_chunks"] = 2
        return keys[::-1] if state.bad_tracker_keys else keys.copy(), _tracker_result(key[1], state.empty), len(steps)

    monkeypatch.setattr(onspill, "run", crv_kernel)
    monkeypatch.setattr(fastdigi, "digitize_steps", tracker_kernel)
    state.crv_pass = Mock(wraps=fast_detectors.crv_pass)
    state.writer = Mock(wraps=fast_detectors.write_output)
    monkeypatch.setattr(fast_detectors, "crv_pass", state.crv_pass)
    monkeypatch.setattr(fast_detectors, "write_output", state.writer)
    state.forbidden = Mock(side_effect=AssertionError("inline comparisons or sampling writes are forbidden"))
    for name in ("compare_crv", "compare_reference", "comparison", "save_sampling"):
        monkeypatch.setattr(fast_detectors, name, state.forbidden)
    return state


def _batch(setup, number=0):
    rows = [3, 0, 2] if number == 0 else [2, 3, 0]
    keys = setup.keys[rows]
    return SimpleNamespace(keys=keys, tracker=pd.DataFrame(keys[[2, 0]], columns=IDS).assign(t=[3400., -2.]),
        crv=pd.DataFrame(keys[[2, 0, 2]], columns=IDS).assign(t0=[3400., -2., 251.], t1=[3401., -1., 252.]),
        pdg=np.array([13, 11, -13]), ordinal_start=2**63 + number * 3, batch_index=number)


def _directory(tmp_path, name="batch"):
    directory = tmp_path / name
    directory.mkdir()
    return directory


def test_reuses_setup_aligns_timing_and_separates_response_seeds_across_batches(setup, tmp_path):
    s = setup
    backend = Backend(s.args, s.source)
    assert backend.metadata["support"] == ["tracker_digis", "crv_reco"]
    assert backend.metadata["reconstruction"] == "not implemented"
    assert "host packing" in backend.metadata["residency"]
    assert backend.metadata["precision"]["tracker"] == "float64"
    json.dumps(backend.metadata, allow_nan=False)
    results = []
    for number in range(2):
        batch = _batch(s, number)
        directory = _directory(tmp_path, f"batch{number}")
        metrics = backend.run(batch, directory)
        results.append(metrics)
        json.dumps(metrics, allow_nan=False)
        assert metrics["events"] == 3 and metrics["response_s"] >= 0 and metrics["write_s"] > 0
        assert metrics["stages"]["tracker"]["steps"] == 2
        assert metrics["stages"]["crv"]["stages"]["synthetic_steps"] == 3
        assert {p.name for p in directory.iterdir()} == {"fast.root"}
        rows = [3, 0, 2] if number == 0 else [2, 3, 0]
        expected_seed = int(np.random.SeedSequence([137, number]).generate_state(1, dtype=np.uint32)[0])
        tracker_seed = int(np.random.SeedSequence([251, number]).generate_state(1, dtype=np.uint32)[0])
        assert metrics["response_seed"] == expected_seed and metrics["tracker_seed"] == tracker_seed
        tracker = s.tracker_calls[number]
        assert tracker.key == ("key", tracker_seed)
        np.testing.assert_array_equal(tracker.keys, batch.keys)
        pd.testing.assert_frame_equal(tracker.steps, batch.tracker)
        np.testing.assert_array_equal(tracker.timing["pbtime"], s.mc[rows])
        np.testing.assert_array_equal(tracker.timing["event_length"], s.length[rows])
        for offset, call in enumerate(s.crv_calls[number * 2:number * 2 + 2]):
            first = offset * 2
            assert call.key == (("key", expected_seed), first)
            expected_rng = np.random.default_rng(np.random.SeedSequence([expected_seed, first]))
            np.testing.assert_array_equal(call.draw, expected_rng.integers(0, 2**31, 4))
            np.testing.assert_array_equal(call.timing.keys, batch.keys[first:first + 2])
            for field, original in (("mc_pbtime", s.mc), ("reco_pbtime", s.reco), ("event_length", s.length)):
                np.testing.assert_array_equal(getattr(call.timing, field), original[rows][first:first + 2])
            selected = [1] if first == 0 else [0, 2]
            pd.testing.assert_frame_equal(call.steps, batch.crv.iloc[selected])
            np.testing.assert_array_equal(call.pdg, batch.pdg[selected])
        with uproot.open(directory / "fast.root") as root:
            expected_tables = {"crvfast/events", "crvfast/digis", "crvfast/pulses", "crvfast/clusters",
                               "trkdigi/events", "benchmark/events"}
            assert {name for name, kind in root.classnames(cycle=False).items() if kind == "TTree"} == expected_tables
            for name in expected_tables:
                assert f"{name};2" not in root.keys()
            for name in ("crvfast/events", "trkdigi/events", "benchmark/events"):
                actual = root[name].arrays(list(IDS), library="np")
                for i, field in enumerate(IDS):
                    assert actual[field].dtype == np.uint64
                    np.testing.assert_array_equal(actual[field], batch.keys[:, i])
            identity = root["benchmark/events"].arrays(library="np")
            assert identity["realization"].dtype == np.uint64
            np.testing.assert_array_equal(identity["realization"], np.arange(3, dtype=np.uint64) + np.uint64(batch.ordinal_start))
            events = root["crvfast/events"].arrays(library="np")
            np.testing.assert_array_equal(events["ndigis"], [1, 0, 2])
            np.testing.assert_array_equal(events["mc_pbtime"], s.mc[rows])
            np.testing.assert_array_equal(events["reco_pbtime"], s.reco[rows])
            digis = root["crvfast/digis"].arrays(list(IDS), library="np")
            np.testing.assert_array_equal(np.column_stack([digis[k] for k in IDS]), batch.keys[[0, 2, 2]])
            trk = root["trkdigi/events"].arrays(library="ak")
            np.testing.assert_array_equal(ak.to_numpy(ak.num(trk.d_sid)), [1, 0, 1])
            np.testing.assert_array_equal(ak.to_numpy(ak.flatten(trk.d_tdc0)), [tracker_seed % 65536] * 2)
            assert root["crvfast/clusters"].num_entries == 0
    assert results[0]["response_seed"] != results[1]["response_seed"]
    assert results[0]["tracker_seed"] != results[1]["tracker_seed"]
    for mock in s.mocks.values():
        assert mock.call_count == 1
    assert s.devices.call_count == 1 and s.writer.call_count == 2 and s.crv_pass.call_count == 2
    s.mocks["runconfig"].assert_called_once_with("run1a_v01")
    s.mocks["conditions"].assert_called_once_with(s.args.truth, s.args.photon_csv, backend.geom, backend.counters)
    s.mocks["config"].assert_called_once_with({"synthetic": True}, jax=True)
    s.mocks["tracker_conditions"].assert_called_once_with(s.args.tracker_conditions)
    s.forbidden.assert_not_called()
    assert s.barrier.call_count == 2
    backend.sync()
    assert s.barrier.call_count == 3


def test_gpu_required_before_asset_loading(setup):
    setup.devices.return_value = [SimpleNamespace(platform="cpu")]
    with pytest.raises(RuntimeError, match="requires a GPU"):
        Backend(setup.args, setup.source)
    for mock in setup.mocks.values():
        mock.assert_not_called()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "float"])
def test_invalid_source_timing_alignment_fails_setup(setup, mutation):
    keys = setup.keys.copy()
    if mutation == "missing":
        keys[0, 0] += np.uint64(1)
        keys[0, 1] += np.uint64(100)
    elif mutation == "duplicate":
        keys[0] = keys[1]
    else:
        keys = keys.astype(float)
    setup.source.signal.event_ids = keys
    with pytest.raises(ValueError):
        Backend(setup.args, setup.source)
    setup.mocks["runconfig"].assert_not_called()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "float", "tracker_step", "crv_step", "pdg", "ordinal", "batch_index"])
def test_invalid_batch_fails_before_response_or_output(setup, tmp_path, mutation):
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)
    if mutation == "missing":
        batch.keys[0, 1] += np.uint64(100)
    elif mutation == "duplicate":
        batch.keys[0] = batch.keys[1]
    elif mutation == "float":
        batch.keys = batch.keys.astype(float)
    elif mutation in ("tracker_step", "crv_step"):
        getattr(batch, mutation.split("_")[0]).loc[0, "subrun"] = 999
    elif mutation == "pdg":
        batch.pdg = batch.pdg[:-1]
    elif mutation == "ordinal":
        batch.ordinal_start = 2**64 - 1
    else:
        batch.batch_index = -1
    directory = _directory(tmp_path)
    with pytest.raises(ValueError):
        backend.run(batch, directory)
    setup.crv_pass.assert_not_called()
    assert not setup.tracker_calls and not list(directory.iterdir())


def test_tracker_identity_changes_are_rejected_before_write(setup, tmp_path):
    backend = Backend(setup.args, setup.source)
    setup.bad_tracker_keys = True
    directory = _directory(tmp_path)
    with pytest.raises(ValueError, match="tracker changed event identities"):
        backend.run(_batch(setup), directory)
    setup.writer.assert_not_called()
    assert not list(directory.iterdir())


@pytest.mark.parametrize("stage", ["tracker", "crv"])
def test_kernel_failures_propagate_without_success_output(setup, tmp_path, stage):
    backend = Backend(setup.args, setup.source)
    setup.fail = stage
    directory = _directory(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic"):
        backend.run(_batch(setup), directory)
    setup.writer.assert_not_called()
    assert not list(directory.iterdir())


def test_empty_detectors_preserve_census_and_sync_is_optional(setup, tmp_path, monkeypatch):
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)
    batch.tracker, batch.crv, batch.pdg = batch.tracker.iloc[:0], batch.crv.iloc[:0], batch.pdg[:0]
    setup.empty = True
    monkeypatch.delattr(setup.jax, "effects_barrier")
    backend.sync()
    directory = _directory(tmp_path)
    assert backend.run(batch, directory)["events"] == 3
    with uproot.open(directory / "fast.root") as root:
        assert root["crvfast/events"].num_entries == root["trkdigi/events"].num_entries == 3
        assert root["crvfast/digis"].num_entries == root["crvfast/pulses"].num_entries == 0
        np.testing.assert_array_equal(root["crvfast/events"]["ndigis"].array(library="np"), [0, 0, 0])


def test_mixed_integer_step_columns_never_float_or_mutate_input(setup, tmp_path):
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)
    for steps in (batch.tracker, batch.crv):
        steps["subrun"] = steps["subrun"].astype(np.int64)
    original_crv, original_tracker = batch.crv.copy(deep=True), batch.tracker.copy(deep=True)
    backend.run(batch, _directory(tmp_path))
    pd.testing.assert_frame_equal(batch.crv, original_crv)
    pd.testing.assert_frame_equal(batch.tracker, original_tracker)
    for call in setup.crv_calls:
        assert call.steps[list(IDS)].to_numpy().dtype == np.uint64
    assert setup.tracker_calls[0].steps[list(IDS)].to_numpy().dtype == np.uint64


def test_output_directory_is_parent_owned_and_existing_fast_root_is_protected(setup, tmp_path):
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)
    with pytest.raises(FileNotFoundError, match="already exist"):
        backend.run(batch, tmp_path / "absent")
    directory = _directory(tmp_path)
    (directory / "fast.root").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        backend.run(batch, directory)
    assert (directory / "fast.root").read_bytes() == b"existing"
    setup.crv_pass.assert_not_called()


def test_changed_signed_tracker_identity_above_float_precision_is_rejected(setup, tmp_path, monkeypatch):
    from trkfastsim.digi import fastdigi
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)

    def changed(steps, cond, key, **kwargs):
        returned = kwargs["keys"].astype(np.int64)
        returned[0, 2] += 1
        return returned, _tracker_result(key[1]), len(steps)

    monkeypatch.setattr(fastdigi, "digitize_steps", changed)
    with pytest.raises(ValueError, match="tracker changed event identities"):
        backend.run(batch, _directory(tmp_path))
    setup.writer.assert_not_called()


def test_same_batch_domain_is_reproducible_without_reinitializing(setup, tmp_path):
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)
    first = backend.run(batch, _directory(tmp_path, "first"))
    second = backend.run(batch, _directory(tmp_path, "second"))
    assert first["response_seed"] == second["response_seed"]
    assert first["tracker_seed"] == second["tracker_seed"]
    for left, right in zip(setup.crv_calls[:2], setup.crv_calls[2:]):
        assert left.key == right.key
        np.testing.assert_array_equal(left.draw, right.draw)
    with uproot.open(tmp_path / "first/fast.root") as a, uproot.open(tmp_path / "second/fast.root") as b:
        for name, kind in a.classnames(cycle=False).items():
            if kind == "TTree":
                assert ak.almost_equal(a[name].arrays(), b[name].arrays(), rtol=0, atol=0)
    for mock in setup.mocks.values():
        assert mock.call_count == 1


def test_partial_single_event_batch_keeps_max_uint64_realization(setup, tmp_path):
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)
    batch.keys = batch.keys[:1]
    batch.tracker, batch.crv, batch.pdg = batch.tracker.iloc[:0], batch.crv.iloc[:0], batch.pdg[:0]
    batch.ordinal_start = 2**64 - 1
    setup.empty = True
    directory = _directory(tmp_path)
    assert backend.run(batch, directory)["events"] == 1
    assert len(setup.crv_calls) == 1
    with uproot.open(directory / "fast.root") as root:
        np.testing.assert_array_equal(root["benchmark/events"]["realization"].array(library="np"),
                                      np.array([2**64 - 1], np.uint64))


def test_module_import_does_not_import_device_packages(monkeypatch):
    import builtins
    from pathlib import Path
    import runpy
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"jax", "crvfastsim", "trkfastsim", "mu2efastsim"}:
            raise AssertionError(f"device package imported at module import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    module = runpy.run_path(str(Path(__file__).parents[1] / "mixing/stream_backend.py"),
                           run_name="mixing._stream_backend_import_test")
    assert "Backend" in module


def test_full_uint64_ids_survive_signed_tracker_return(setup, tmp_path, monkeypatch):
    from trkfastsim.digi import fastdigi
    keys = setup.keys.copy()
    keys[:, 0] += np.uint64(2**63)
    setup.source.signal.event_ids = keys
    for path, times in ((setup.args.truth, setup.mc), (setup.args.stock, setup.reco)):
        with uproot.recreate(path) as root:
            root.mktree("crvdump/events", {**dict(zip(IDS, keys.T)), "pbtime": times,
                         "spill": np.ones(4, int), "evlen": setup.length})
    setup.keys = keys
    backend = Backend(setup.args, setup.source)
    batch = _batch(setup)
    original = fastdigi.digitize_steps

    def signed(*args, **kwargs):
        returned, digi, nsteps = original(*args, **kwargs)
        return returned.astype(np.int64), digi, nsteps

    monkeypatch.setattr(fastdigi, "digitize_steps", signed)
    directory = _directory(tmp_path)
    backend.run(batch, directory)
    with uproot.open(directory / "fast.root") as root:
        for name in ("crvfast/events", "trkdigi/events", "benchmark/events"):
            rows = root[name].arrays(list(IDS), library="np")
            np.testing.assert_array_equal(np.column_stack([rows[field] for field in IDS]), batch.keys)


@pytest.mark.parametrize("field,value", [("chunk", 0), ("seed", -1), ("tracker_seed", 1.5),
                                         ("tracker_conditions", None)])
def test_invalid_configuration_rejected_before_device_initialization(setup, field, value):
    setattr(setup.args, field, value)
    with pytest.raises(ValueError):
        Backend(setup.args, setup.source)
    setup.devices.assert_not_called()
