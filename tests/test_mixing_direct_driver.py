"""Direct-driver wiring with real chunking/writers, synthetic frames and no GPU/assets."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from mixing import crv_inputs, direct_frames, fast_detectors


IDS = ("run", "subrun", "event")


def _crv_result(ev, changed=False):
    """Small onspill.run result, including pulses and an empty cluster table."""
    from crvfastsim.chain import CHAN_MULT
    n = len(ev)
    pulses = {"ev": ev, "idx": np.arange(n), "bar": np.zeros(n, np.int64),
              "sipm": np.zeros(n, np.int64), "seq": np.zeros(n, np.int64),
              "pe": np.full(n, 12.), "time": np.full(n, 600.)}
    cluster = {name: np.empty(0, np.int64) for name in
               ("cl_event", "cl_type", "cl_poscalc", "mem_cl", "mem_idx", "mem_event")}
    cluster.update(cl_pe=np.empty(0), cl_avgtime=np.empty(0), cl_pos=np.empty((0, 3)), pulses=pulses)
    return {"D": {"chan": ev * CHAN_MULT, "starttdc": np.full(n, 40 + int(changed)),
                  "adc": np.ones((n, 12), np.int16)},
            "clusters": [cluster], "timers": {"synthetic_steps": n}}


def _tracker_result(changed=False):
    return {"ev": np.array([0, 2]), "sid": np.array([11, 12]),
            "tdc": np.full((2, 2), 100 + int(changed)), "tot": np.ones((2, 2), np.int64),
            "pmp": np.array([20, 30]), "adc": np.ones((2, 3), np.int64)}


@pytest.fixture
def driver(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    from crvfastsim import onspill, runconfig
    from crvfastsim.response import photons_jax, sipm
    from trkfastsim.digi import digicond, fastdigi

    # Even PRNG key creation is mocked: no JAX device operations in this fixture.
    devices = Mock(return_value=[SimpleNamespace(platform="gpu")])
    monkeypatch.setattr(jax, "devices", devices)
    monkeypatch.setattr(jax.random, "key", lambda seed: ("key", seed))
    monkeypatch.setattr(jax.random, "fold_in", lambda key, offset: (key, offset))
    keys = np.array([[2**40 + 1, 2**33 + 7, 2**53 + 1],
                     [2**40 + 2, 2**33 + 7, 2**53 + 1],
                     [2**40 + 1, 2**33 + 8, 2**53 + 3]], dtype=np.uint64)
    # Deliberately out of census order; event 1 has no steps in either detector.
    steps = pd.DataFrame(keys[[2, 0, 2]], columns=IDS).assign(t0=[3400., -2., 251.], t1=[3401., -1., 252.])
    tracker_steps = pd.DataFrame(keys[[2, 0]], columns=IDS).assign(t=[3400., -2.])
    pdg = np.array([13, 11, -13])
    mc = np.array([-210., -215., -220.])
    reco = np.array([-209.5, -216., -219.75])
    length = np.array([1675., 1700., 1690.])
    provenance = {"seed": 123, "timing_s": {"read_s": .01, "overlay_s": .02},
                  "selections": {"MuStopPileup": {"source_event_ids": keys[[2, 0]],
                                                  "donors": np.array([2, 0])}}}
    state = SimpleNamespace(keys=keys, steps=steps, tracker_steps=tracker_steps, pdg=pdg,
                            mc=mc, reco=reco, length=length, provenance=provenance,
                            fail_at=None, changed=None, bad_tracker_keys=False,
                            crv_calls=[], tracker_calls=[], devices=devices,
                            open_root=uproot.open, out=tmp_path / "out")
    state.manifest = tmp_path / "frames.json"
    state.truth = tmp_path / "truth.root"
    state.stock = tmp_path / "stock.root"
    state.expanded = tmp_path / "expanded.fcl"
    state.expanded.write_text("# synthetic explicit response configuration\n")
    state.tracker_conditions = tmp_path / "tracker.npz"
    photon_csv = tmp_path / "photon.csv"
    # Byte-only stand-ins allow real configuration hashing without loading any assets.
    state.manifest.write_text("{}\n")
    photon_csv.write_text("# synthetic photon calibration\n")
    state.tracker_conditions.write_bytes(b"synthetic tracker conditions")
    state.argv = ["--frames-manifest", str(state.manifest), "--expanded", str(state.expanded),
                  "--truth", str(state.truth), "--stock", str(state.stock),
                  "--photon-csv", str(photon_csv), "--out", str(state.out),
                  "--tracker-conditions", str(state.tracker_conditions),
                  "--seed", "137", "--tracker-seed", "251", "--chunk", "2",
                  "--repeats", "2", "--max-events", "3"]

    def receipt():
        assert (state.out / "sampling.npz").is_file()
        metadata = json.loads((state.out / "sampling.json").read_text())
        selection = metadata["selections"]["MuStopPileup"]
        with np.load(state.out / "sampling.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved[selection["source_event_ids"]["npz_key"]], keys[[2, 0]])
            np.testing.assert_array_equal(saved[selection["donors"]["npz_key"]], [2, 0])
        assert metadata["seed"] == 123  # Manifest sampling seed is not a response seed.

    state.receipt = receipt
    state.load_manifest = Mock(return_value=(SimpleNamespace(event_ids=keys), tracker_steps, steps, pdg, provenance))
    monkeypatch.setattr(direct_frames, "load_manifest", state.load_manifest)

    def read_timing(actual_keys, truth, stock):
        receipt()  # The selection receipt must precede even timing/asset setup.
        assert actual_keys is keys
        assert (truth, stock) == (state.truth, state.stock)
        return mc, reco, length

    state.read_timing = Mock(side_effect=read_timing)
    monkeypatch.setattr(crv_inputs, "read_timing", state.read_timing)
    forbidden = Mock(side_effect=AssertionError("direct frames used native input/ROOT reading"))
    state.forbidden = forbidden
    monkeypatch.setattr(crv_inputs, "read_inputs", forbidden)
    monkeypatch.setattr(crv_inputs, "check_tracker_inputs", forbidden)
    # All input data are explicit mocks. Keep the real writer, prohibit source ROOT reads.
    monkeypatch.setattr(uproot, "open", forbidden)

    geom, counters, cond, tabs, pmap = [object() for _ in range(5)]
    cfg = SimpleNamespace(response=object())
    rc = SimpleNamespace(geometry=lambda: geom, counters_cfg=lambda: counters)
    state.runconfig = Mock(return_value=rc)
    monkeypatch.setattr(runconfig, "get", state.runconfig)
    state.load_conditions = Mock(return_value=cond)
    monkeypatch.setattr(crv_inputs, "load_conditions", state.load_conditions)
    expanded = {"synthetic": True}
    state.parse_expanded = Mock(return_value=expanded)
    monkeypatch.setattr(digicond, "parse_fhicl_dump", state.parse_expanded)
    state.build_config = Mock(return_value=cfg)
    monkeypatch.setattr(crv_inputs, "build_config", state.build_config)
    monkeypatch.setattr(photons_jax, "load_device_tables", Mock(return_value=tabs))
    monkeypatch.setattr(sipm, "photon_map", Mock(return_value=pmap))
    tracker_cond = SimpleNamespace(nADCSamples=3)
    state.load_tracker = Mock(return_value=tracker_cond)
    monkeypatch.setattr(digicond.Conditions, "load", state.load_tracker)

    def crv_kernel(actual_steps, actual_pdg, actual_geom, actual_cond, actual_cfg, rng, *,
                   timing, counters_cfg, key, tabs, pmap, timers, detail):
        receipt()
        ordinal = len(state.crv_calls)
        first = (ordinal % 2) * 2
        last = min(first + 2, len(keys))
        selected = [1] if first == 0 else [0, 2]
        pd.testing.assert_frame_equal(actual_steps, steps.iloc[selected])
        np.testing.assert_array_equal(actual_pdg, pdg[selected])
        np.testing.assert_array_equal(timing.keys, keys[first:last])
        np.testing.assert_array_equal(timing.mc_pbtime, mc[first:last])
        np.testing.assert_array_equal(timing.reco_pbtime, reco[first:last])
        np.testing.assert_array_equal(timing.event_length, length[first:last])
        assert timing.period == 1695.
        assert actual_geom is geom and actual_cond is cond and actual_cfg is cfg
        assert counters_cfg is counters and tabs is state.tabs and pmap is state.pmap
        assert key == (("key", 137), first)
        expected_rng = np.random.default_rng(np.random.SeedSequence([137, first]))
        np.testing.assert_array_equal(rng.integers(0, 2**31, 4), expected_rng.integers(0, 2**31, 4))
        assert timers == {} and detail is True
        state.crv_calls.append(timing)
        if state.fail_at == "crv":
            raise RuntimeError("synthetic CRV response failure")
        return _crv_result(timing.lookup(actual_steps[list(IDS)].to_numpy()),
                           changed=state.changed == "crv" and ordinal >= 2)

    state.tabs, state.pmap = tabs, pmap
    monkeypatch.setattr(onspill, "run", crv_kernel)
    # A wrapping spy exercises the existing crv_pass, not a replacement implementation.
    state.crv_pass = Mock(wraps=fast_detectors.crv_pass)
    monkeypatch.setattr(fast_detectors, "crv_pass", state.crv_pass)

    def tracker_kernel(actual_steps, actual_cond, key, *, keys, chunk, stats, backend, event_timing):
        receipt()
        assert actual_steps is tracker_steps and actual_cond is tracker_cond
        assert keys is state.keys and chunk == 2 and backend == "jax"
        assert key == ("key", 251)
        assert set(event_timing) == {"pbtime", "event_length"}
        np.testing.assert_array_equal(event_timing["pbtime"], mc)
        np.testing.assert_array_equal(event_timing["event_length"], length)
        # The tracker consumes MC timing, not reconstructed timing.
        assert not np.array_equal(event_timing["pbtime"], reco)
        state.tracker_calls.append(event_timing)
        stats["synthetic_chunks"] = 2
        if state.fail_at == "tracker":
            raise RuntimeError("synthetic tracker response failure")
        out_keys = keys[::-1] if state.bad_tracker_keys else keys.copy()
        return out_keys, _tracker_result(state.changed == "tracker" and len(state.tracker_calls) > 1), len(actual_steps)

    monkeypatch.setattr(fastdigi, "digitize_steps", tracker_kernel)
    state.compare_crv = Mock(return_value={"synthetic": True})
    state.compare_reference = Mock(return_value=object())
    state.comparison = Mock(return_value={"synthetic": True})
    monkeypatch.setattr(fast_detectors, "compare_crv", state.compare_crv)
    monkeypatch.setattr(fast_detectors, "compare_reference", state.compare_reference)
    monkeypatch.setattr(fast_detectors, "comparison", state.comparison)
    return state


def _without(argv, option):
    index = argv.index(option)
    return argv[:index] + argv[index + 2:]


def test_direct_frames_end_to_end_shared_timing_ids_seeds_and_real_output(driver):
    d = driver
    fast_detectors.main(d.argv)
    d.load_manifest.assert_called_once_with(d.manifest, max_events=3)
    assert d.read_timing.call_count == 1
    assert d.crv_pass.call_count == 2 and len(d.crv_calls) == 4 and len(d.tracker_calls) == 2
    for call in d.crv_pass.call_args_list:
        assert call.args[0] is d.steps and call.args[1] is d.pdg
        assert call.args[-2:] == (137, 2)
        timing = call.args[2]
        np.testing.assert_array_equal(timing.keys, d.keys)
        np.testing.assert_array_equal(timing.mc_pbtime, d.mc)
        np.testing.assert_array_equal(timing.reco_pbtime, d.reco)
    assert d.crv_pass.call_args_list[0].args[2] is d.crv_pass.call_args_list[1].args[2]
    for event_timing in d.tracker_calls:
        assert event_timing["pbtime"] is timing.mc_pbtime
        assert event_timing["event_length"] is timing.event_length
    d.parse_expanded.assert_called_once_with(d.expanded.read_text())
    d.build_config.assert_called_once_with({"synthetic": True}, jax=True)
    d.runconfig.assert_called_once_with("run1a_v01")
    d.load_tracker.assert_called_once_with(d.tracker_conditions)
    d.forbidden.assert_not_called()
    # The only tracker comparison is statistical pairing against stock, not saved identity.
    assert d.compare_reference.call_count == 1
    call = d.compare_reference.call_args
    assert call.args[0] is d.keys and call.args[1] is None and call.args[2] == d.stock
    assert d.compare_crv.call_args.args[0] == d.stock
    assert d.compare_crv.call_args.args[1] is d.keys
    d.receipt()
    report = json.loads((d.out / "report.json").read_text())
    assert report["input_mode"] == "direct_frames" and report["events"] == 3
    assert report["seed"] == 137 and report["chunk"] == 2
    assert report["same_seed_repeats_identical"] is True
    assert report["tracker_saved_identical"] is None
    assert report["frame_preparation_s"] == d.provenance["timing_s"]
    assert "not exact stock donor replay" in report["comparison_scope"]
    assert len(report["passes"]) == 2
    with d.open_root(d.out / "fast.root") as root:
        events = root["crvfast/events"].arrays(library="np")
        tracker = root["trkdigi/events"].arrays(library="ak")
        for i, name in enumerate(IDS):
            np.testing.assert_array_equal(events[name], d.keys[:, i])
            values = ak.to_numpy(tracker[name])
            assert values.dtype == np.uint64  # Real digi_columns uses uint32; main must restore full IDs.
            np.testing.assert_array_equal(values, d.keys[:, i])
        np.testing.assert_array_equal(events["mc_pbtime"], d.mc)
        np.testing.assert_array_equal(events["reco_pbtime"], d.reco)
        np.testing.assert_array_equal(events["event_length"], d.length)
        np.testing.assert_array_equal(events["ndigis"], [1, 0, 2])
        np.testing.assert_array_equal(events["npulses"], [1, 0, 2])
        assert ak.to_list(tracker.d_sid) == [[11], [], [12]]
        digis = root["crvfast/digis"].arrays(library="np")
        np.testing.assert_array_equal(np.column_stack([digis[k] for k in IDS]), d.keys[[0, 2, 2]])
        assert root["crvfast/clusters"].num_entries == 0


@pytest.mark.parametrize("invalid", ["missing_expanded", "tracker_reference", "both_sources"])
def test_direct_cli_rejects_unsupported_sources_before_io(driver, invalid, capsys):
    d = driver
    argv = d.argv.copy()
    if invalid == "missing_expanded":
        argv = _without(argv, "--expanded")
    elif invalid == "tracker_reference":
        argv += ["--tracker-reference", str(d.stock)]
    else:
        argv += ["--input", str(d.truth)]
    with pytest.raises(SystemExit) as error:
        fast_detectors.main(argv)
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert ("not allowed with argument" if invalid == "both_sources" else
            "direct frames require --expanded and cannot claim saved tracker identity") in message
    d.devices.assert_not_called()
    d.load_manifest.assert_not_called()
    d.forbidden.assert_not_called()
    assert not d.out.exists()


@pytest.mark.parametrize("stage", ["crv", "tracker"])
def test_sampling_receipt_exists_before_response_failure(driver, stage):
    d = driver
    d.fail_at = stage
    with pytest.raises(RuntimeError, match="synthetic .* response failure"):
        fast_detectors.main(d.argv)
    d.receipt()
    assert len(d.crv_calls) == (1 if stage == "crv" else 2)
    assert len(d.tracker_calls) == (0 if stage == "crv" else 1)
    assert not (d.out / "fast.root").exists()
    assert not (d.out / "report.json").exists()
    d.forbidden.assert_not_called()


@pytest.mark.parametrize("detector", ["crv", "tracker"])
def test_same_seed_repeat_must_match_arrays_not_just_counts(driver, detector):
    d = driver
    d.changed = detector
    with pytest.raises(ValueError, match=f"same-seed {'CRV' if detector == 'crv' else 'tracker'} repeat changed output arrays"):
        fast_detectors.main(d.argv)
    assert d.crv_pass.call_count == 2 and len(d.crv_calls) == 4
    assert len(d.tracker_calls) == (1 if detector == "crv" else 2)
    assert not (d.out / "fast.root").exists()
    assert not (d.out / "report.json").exists()
    d.receipt()


def test_tracker_must_return_the_original_full_id_order(driver):
    d = driver
    d.bad_tracker_keys = True
    with pytest.raises(ValueError, match="tracker changed event identities"):
        fast_detectors.main(d.argv)
    assert len(d.tracker_calls) == 1
    assert not (d.out / "fast.root").exists()
    d.receipt()


def test_direct_crv_only_does_not_load_or_run_tracker(driver):
    d = driver
    fast_detectors.main(_without(d.argv, "--tracker-conditions"))
    assert d.crv_pass.call_count == 2 and len(d.crv_calls) == 4
    assert d.tracker_calls == []
    d.load_tracker.assert_not_called()
    d.compare_reference.assert_not_called()
    d.forbidden.assert_not_called()
    with d.open_root(d.out / "fast.root") as root:
        assert root["crvfast/events"].num_entries == 3
        assert "trkdigi/events" not in root
    report = json.loads((d.out / "report.json").read_text())
    assert report["input_mode"] == "direct_frames"
    assert "tracker_paired" not in report
    d.receipt()


def test_single_pass_throughput_skips_only_paired_diagnostics(driver):
    d = driver
    argv = _without(d.argv, "--repeats") + ["--repeats", "1", "--skip-comparisons"]
    fast_detectors.main(argv)
    d.load_manifest.assert_called_once_with(d.manifest, max_events=3)
    d.read_timing.assert_called_once()
    d.load_conditions.assert_called_once()
    d.build_config.assert_called_once()
    d.compare_crv.assert_not_called()
    d.compare_reference.assert_not_called()
    d.comparison.assert_not_called()
    assert len(d.crv_calls) == 2 and len(d.tracker_calls) == 1
    d.receipt()
    report = json.loads((d.out / "report.json").read_text())
    assert report["paired_comparisons_skipped"] is True
    assert report["crv_paired"] is None and "tracker_paired" not in report
    assert len(report["passes"]) == len(report["preparation"]["passes"]) == 1
    with d.open_root(d.out / "fast.root") as root:
        assert root["trkdigi/events"].num_entries == root["crvfast/events"].num_entries == 3


@pytest.mark.parametrize("extra", [["--repeats", "2"], ["--preparation-repeats", "2"],
                                    ["--sampling-reference", "saved"]])
def test_throughput_rejects_repeats_and_inline_reference(driver, extra):
    d = driver
    argv = _without(d.argv, "--repeats") + ["--repeats", "1", "--skip-comparisons"] + extra
    with pytest.raises(SystemExit) as error:
        fast_detectors.main(argv)
    assert error.value.code == 2
    d.devices.assert_not_called()
    d.load_manifest.assert_not_called()
