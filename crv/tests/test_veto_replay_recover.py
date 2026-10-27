"""Retained-only membership recovery, closure gates and C++ adapter contracts."""
import dataclasses
import json
import subprocess
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from crvfastsim.reco import coinc as C
from crvfastsim.veto import replay_recover as R


def geometry(n=4):
    layer = np.arange(n) % 4
    sector = np.zeros(n, dtype=int)
    sector[1088:] = 1
    pos = np.column_stack((np.zeros(n), layer * 10., np.zeros(n)))
    return C.SectorGeom(sector, layer, np.zeros(n, dtype=int), pos[:, 0], pos[:, 1], pos,
                        np.array([2, 2]), np.array([1, 1]), ["T1", "T2"],
                        np.array([-100., -100.]), np.array([100., 100.]),
                        np.array([True, True]), np.array([True, True]))


@pytest.fixture
def sample():
    events = ak.Array({"run": [1430] * 4, "subrun": [0] * 4, "event": [1, 2, 3, 4],
                       "pbtime": [-224.46, -214.46, -204.46, -219.46]})
    # Interleaved, non-channel/time-sorted events retain secondary pulse packets under the Run1A config.
    ev = np.array([1, 3, 1, 1, 3, 1, 1, 3, 1])
    chan = np.array([8, 9, 0, 4, 1, 0, 4, 5, 8])
    time = np.array([502., 701., 500., 501., 702., 600., 601., 700., 602.])
    pe = np.full(len(ev), 10.123456789)
    shift = -ak.to_numpy(events.pbtime).astype(np.float32).astype(float)[ev - 1]
    pulses = ak.Array({"ev": ev, "chan": chan, "time": time + shift, "pe": pe,
                       "seq": np.array([0, 0, 0, 0, 0, 1, 1, 0, 1])})
    return events, pulses, geometry(), np.zeros(16, dtype=int)


def saved_frames(clusters, events):
    ev = R._events(events)
    keys = (ak.to_numpy(clusters.subrun).astype(np.int64) << 20) | ak.to_numpy(clusters.event)
    counts = np.searchsorted(keys, ev["ev"], "right") - np.searchsorted(keys, ev["ev"], "left")
    return ak.zip({k: events[k] for k in R.IDS} | {"coincs": ak.zip({
        k: ak.unflatten(clusters[v], counts) for k, v in R.SUMMARY_FIELDS.items()})}, depth_limit=1)


def test_ordered_multipulse_and_empty_event_offsets(sample, monkeypatch):
    events, pulses, g, status = sample
    batch = R.prepare_pulses(pulses, events, 16)
    np.testing.assert_array_equal(batch.offsets, [0, 6, 6, 9, 9])
    np.testing.assert_array_equal(batch.pulses["idx"], [0, 1, 2, 3, 4, 5, 0, 1, 2])
    np.testing.assert_array_equal(batch.pulses["chan"], [8, 0, 4, 0, 4, 8, 9, 1, 5])
    np.testing.assert_array_equal(batch.finder_time, [502, 500, 501, 600, 601, 602, 701, 702, 700])
    original = C.find_event
    calls = []

    def spy(p, *args):
        assert all(isinstance(v, np.ndarray) for v in p.values())
        calls.append(p["time"].copy())
        return original(p, *args)

    monkeypatch.setattr(C, "find_event", spy)
    result = R.recover(batch, g, status)
    assert len(calls) == 2  # only nonempty events, never per-event awkward conversions
    assert ak.to_list(result.pulses) == [[0, 1, 2], [3, 4, 5], [0, 1, 2]]
    assert ak.to_list(result.event) == [1, 1, 3]
    assert ak.to_list(result.idx) == [0, 1, 0]
    assert ak.to_list(result.layers) == [[0, 1, 2]] * 3
    assert R.compare_saved(result, saved_frames(result, events))["passed"]


def test_time_shift_restores_all_times_and_empty_side_sentinel(sample):
    events, pulses, g, status = sample
    batch = R.prepare_pulses(pulses, events, 16)
    result = R.recover(batch, g, status)
    p = {k: batch.pulses[k][:6] for k in ("idx", "bar", "sipm", "pe", "seq")}
    p["time"] = batch.finder_time[:6]
    reference = C.find_event(p, g, status, R.finder_config())[0]
    shift = -np.float64(np.float32(events.pbtime[0]))
    for k in ("start", "end", "avgtime"):
        assert result[k][0] == reference[k] + shift
    assert result.sidetime0[0] == reference["sidetime"][0] + shift
    assert result.sidetime1[0] == 0
    assert result.sidetime0[2] == 0
    assert result.sidetime1[2] > 900


def test_both_side_times_shift_and_position_does_not():
    events = {"run": [1], "subrun": [0], "event": [1], "pbtime": [-224.46]}
    t = np.array([500., 501., 502., 500.5, 501.5, 502.5])
    pulses = {"ev": np.ones(6), "chan": [0, 4, 8, 1, 5, 9], "seq": np.zeros(6), "pe": np.full(6, 10.),
              "time": t - np.float64(np.float32(-224.46))}
    batch = R.prepare_pulses(pulses, events, 16)
    result = R.recover(batch, geometry(), np.zeros(16))
    no_shift = dataclasses.replace(batch, events=batch.events | {"pbtime": np.array([0.])})
    base = R.recover(no_shift, geometry(), np.zeros(16))
    assert len(result) == 1 and result.poscalc[0]
    for k in ("start", "end", "avgtime", "sidetime0", "sidetime1"):
        assert result[k][0] - base[k][0] == pytest.approx(-np.float64(np.float32(-224.46)))
    for k in ("x", "y", "z", "pe", "slope"):
        assert result[k][0] == base[k][0]


@pytest.mark.parametrize("field", ["time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType"])
def test_summary_closure_rejects_each_changed_property(sample, field):
    events, pulses, g, status = sample
    result = R.recover(R.prepare_pulses(pulses, events, 16), g, status)
    fast = saved_frames(result, events)
    fast = ak.with_field(fast, ak.with_field(fast.coincs, fast.coincs[field] + 1e-7, field), "coincs")
    with pytest.raises(ValueError, match="frame closure"):
        R.compare_saved(result, fast)


@pytest.mark.parametrize("mutation", ["count", "order", "idx", "ids", "nan"])
def test_summary_count_order_ids_finite_guards(sample, mutation):
    events, pulses, g, status = sample
    result = R.recover(R.prepare_pulses(pulses, events, 16), g, status)
    fast = saved_frames(result, events)
    if mutation == "count":
        result = result[:-1]
    elif mutation == "order":
        result = result[[1, 0, 2]]
    elif mutation == "idx":
        result = ak.with_field(result, result.idx + 1, "idx")
    elif mutation == "ids":
        result = ak.with_field(result, result.run + 1, "run")
    else:
        fast = ak.with_field(fast, ak.with_field(fast.coincs, fast.coincs.time * np.nan, "time"), "coincs")
    with pytest.raises(ValueError):
        R.compare_saved(result, fast)


def test_float32_only_pe_and_membership_report(sample):
    events, pulses, g, status = sample
    batch = R.prepare_pulses(pulses, events, 16)
    original = R.recover(batch, g, status)
    rounded = R.recover(batch, g, status, pe_float32=True)
    comparison = R.compare_precision(original, rounded)
    assert comparison["identical_memberships"]
    assert comparison["property_max_abs_differences"]["pe"] > 1e-9
    for k in ("start", "end"):
        assert comparison["property_max_abs_differences"][k] == 0
    assert not R.compare_saved(rounded, saved_frames(original, events), require=False)["passed"]
    np.testing.assert_array_equal(batch.pulses["pe"], np.full(9, 10.123456789))


def test_float32_threshold_membership_change_is_reported(sample):
    events, pulses, g, status = sample
    pulses = ak.with_field(pulses, np.full(len(pulses), 7.9999999), "pe")
    batch = R.prepare_pulses(pulses, events, 16)
    a = R.recover(batch, g, status)
    b = R.recover(batch, g, status, pe_float32=True)
    assert len(a) == 0 and len(b) == 3
    report = R.compare_precision(a, b)
    assert not report["identical_memberships"]
    assert report["property_max_abs_differences"]["pe"] is None


@pytest.mark.parametrize("empty", [False, True])
def test_exact_cpp_adapter_schema_and_disabled_field_intent(tmp_path, sample, empty):
    events, pulses, g, status = sample
    if empty:
        pulses = pulses[:0]
    batch = R.prepare_pulses(pulses, events, 16)
    path = tmp_path / "adapter.root"
    assert R.write_adapter(path, batch) == {"pulses": len(pulses), "digis": 0}
    with uproot.open(path) as f:
        p, d = f["crvdump/pulses"], f["crvdump/digis"]
        assert p.classname == d.classname == "TTree"
        assert d.num_entries == 0
        expected = {"uint32": "uint32_t", "int32": "int32_t", "float32": "float", "float64": "double",
                    "var * int32": "int32_t[]", "var * int16": "int16_t[]"}
        for tree, schema in ((p, R.PULSE_TYPES), (d, R.DIGI_TYPES)):
            for name, dtype in schema.items():
                assert tree[name].typename == expected[dtype], name
        assert p["wfidx"].member("fLeaves")[0].member("fLeafCount").member("fName") == "nwfidx"
        assert d["adc"].member("fLeaves")[0].member("fLeafCount").member("fName") == "nadc"
        a = p.arrays(library="ak")
        for k in (*R.IDS, "idx", "bar", "sipm", "seq", "time"):
            np.testing.assert_array_equal(ak.to_numpy(a[k]), batch.pulses[k])
        np.testing.assert_array_equal(ak.to_numpy(a.pe), batch.pulses["pe"].astype(np.float32))
        for k, value in R.ADAPTER_INTENT["placeholders"].items():
            assert bool(ak.all(ak.num(a[k]) == 0)) if k == "wfidx" else bool(ak.all(a[k] == value))
    assert "Diagnostic-only" in R.ADAPTER_INTENT["scope"]
    assert "leading-edge time selection" in R.ADAPTER_INTENT["required_disabled_features"]
    with pytest.raises(FileExistsError):
        R.write_adapter(path, batch)


def test_all_empty_typed_parquet_and_summary(sample, tmp_path):
    events, pulses, g, status = sample
    batch = R.prepare_pulses(pulses[:0], events, 16)
    result = R.recover(batch, g, status)
    assert set(ak.fields(result)) == set(R.CLUSTER_TYPES) | {"pulses", "layers"}
    ak.to_parquet(result, tmp_path / "empty.parquet")
    loaded = ak.from_parquet(tmp_path / "empty.parquet")
    assert "int32" in str(ak.type(loaded.pulses))
    assert R.compare_saved(loaded, saved_frames(result, events))["passed"]
    assert R.compare_precision(result, result)["identical_memberships"]


@pytest.mark.parametrize("field,value", [("chan", 16), ("chan", -1), ("chan", 1.5), ("seq", -1),
                                         ("seq", 2**31), ("ev", 17), ("ev", 2**52),
                                         ("time", np.nan), ("pe", np.inf)])
def test_invalid_pulse_domains(sample, field, value):
    events, pulses, _, _ = sample
    pulses = ak.with_field(pulses, np.full(len(pulses), value), field)
    with pytest.raises(ValueError):
        R.prepare_pulses(pulses, events, 16)


@pytest.mark.parametrize("field,value", [("event", [1, 1, 3, 4]), ("event", [1, 2, 3, 2**20]),
                                         ("run", [1, 2, 1, 1]), ("subrun", [-1, 0, 0, 0]),
                                         ("pbtime", [np.nan] * 4)])
def test_invalid_event_domains(sample, field, value):
    events, pulses, _, _ = sample
    with pytest.raises(ValueError):
        R.prepare_pulses(pulses, ak.with_field(events, value, field), 16)


def test_status_coverage_and_values(sample):
    events, pulses, g, status = sample
    batch = R.prepare_pulses(pulses, events, 16)
    for bad in (status[:-1], np.full(16, -1), np.full(16, 2**16)):
        with pytest.raises(ValueError):
            R.recover(batch, g, bad)


def test_pure_recovery_imports_no_response_or_fitting():
    # Fresh interpreter: unlike a sys.modules assertion this guards import execution.
    script = '''
import builtins
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if "response" in name or "pulsefit" in name or name in ("fastframes", "chain", "replay_fast"):
        raise AssertionError("forbidden response/fitting import: " + name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
from crvfastsim.veto import replay_recover as r
import numpy as np
p = {"ev": np.ones(3, dtype=int), "chan": np.array([8, 0, 4]), "seq": np.zeros(3, dtype=int),
     "time": np.array([700., 701., 702.]), "pe": np.full(3, 10.)}
b = r.prepare_pulses(p, {"run": [1], "subrun": [0], "event": [1], "pbtime": [-224.46]}, 12)
z = np.zeros(3, dtype=int)
pos = np.column_stack((z, np.arange(3) * 10., z))
g = r.C.SectorGeom(z, np.arange(3), z, pos[:, 0], pos[:, 1], pos,
                   np.array([2]), np.array([1]), ["T1"], np.array([-100.]), np.array([100.]),
                   np.array([True]), np.array([True]))
assert len(r.recover(b, g, np.zeros(12, dtype=int))) == 1
'''
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.fixture
def bound_inputs(tmp_path, sample, monkeypatch):
    """Real metadata verifier and ROOT/parquet inputs; only geometry lookup mocked."""
    from crvfastsim.veto.fastframes import run_config
    from crvfastsim.veto.replay_policy import INTENDED, TIMING, apply_policy, metadata_identity
    from crvfastsim.veto.replay_check import check_events
    from crvfastsim.veto.compareframes import compare
    events, pulses, _, _ = sample
    g, status = geometry(1280), np.zeros(5120, np.int32)
    batch = R.prepare_pulses(pulses, events)
    clusters = R.recover(batch, g, status)
    fast = saved_frames(clusters, events)
    paths = {k: tmp_path / name for k, name in {"fast": "fast.parquet", "input": "mixed.root",
             "check": "check.json", "stock": "stock.root", "pulses": "pulses.parquet",
             "counters": "counters.txt", "outdir": "new"}.items()}
    paths["counters"].write_text("synthetic retained geometry\n")
    monkeypatch.setattr(R.geometry, "from_offline_config", lambda p: None)
    monkeypatch.setattr(R.geometry, "parse_simpleconfig", lambda text: {})
    monkeypatch.setattr(C, "sector_geom", lambda *args: g)
    ak.to_parquet(fast, paths["fast"])
    ak.to_parquet(pulses, paths["pulses"])
    event_cols = {k: ak.to_numpy(events[k]) for k in ak.fields(events)}
    with uproot.create(paths["input"]) as f:
        f.mktree("dump/events", {k: "uint32" for k in R.IDS} | {"pbtime": "float64"})
        f["dump/events"].extend(event_cols)
    manifest = {"events": len(events), "output_sha256": R.sha256(paths["input"])}
    paths["input"].with_suffix(".json").write_text(json.dumps(manifest))
    stock_events = event_cols | {"pbtime": event_cols["pbtime"].astype(np.float32).astype(float),
                                "spill": np.ones(4, np.int32), "evlen": np.full(4, 1700.),
                                "ndigi": np.zeros(4, np.int32), "npulse": np.array([6, 0, 3, 0]),
                                "ncluster": np.array([2, 0, 1, 0])}
    with uproot.create(paths["stock"]) as f:
        f.mktree("crvdump/events", {k: v.dtype for k, v in stock_events.items()})
        f["crvdump/events"].extend(stock_events)
        f.mktree("crvdump/clusters", R.CLUSTER_TYPES | {"pulses": "var * int32", "layers": "var * int32"})
        f["crvdump/clusters"].extend({k: clusters[k] for k in ak.fields(clusters)})
        f.mktree("crvdump/calib", {"channel": "int32", "status": "int32"})
        f["crvdump/calib"].extend({"channel": np.arange(5120), "status": status})
        f.mktree("crvdump/pulses", {"idx": "int32"})
        f["crvdump/pulses"].extend({"idx": batch.pulses["idx"]})
        f.mktree("crvdump/digis", {"idx": "int32"})
    response, contract = apply_policy(run_config("run1a_v01")[0], INTENDED)
    meta = {"input_sha256": manifest["output_sha256"], "conditions_sha256": "a" * 64,
            "photon_dev_sha256": "b" * 64, "timing": TIMING, "response_contract": contract,
            "stock_shared_table_yields_diagnostic": False,
            "effective_yields": {k: v.yield_per_mev for k, v in response.sectors.items()},
            "pulse_file": str(paths["pulses"]), "pulse_sha256": R.sha256(paths["pulses"]),
            "frames": 4, "coincs": 3, "timers": {"pulses": 9}}
    meta_path = paths["fast"].with_suffix(".json")
    meta_path.write_text(json.dumps(meta))
    identity, digest, legacy = metadata_identity(meta)
    check = {"fast_metadata_sha256": R.sha256(meta_path), "input_sha256": manifest["output_sha256"],
             "conditions_sha256": meta["conditions_sha256"], "response_identity": identity,
             "response_identity_sha256": digest, "legacy_policy_recovery": legacy, "conditions_equal": True,
             "fast_sha256": R.sha256(paths["fast"]), "stock": str(paths["stock"]),
             "stock_sha256": R.sha256(paths["stock"]), "input_manifest_sha256": R.sha256(paths["input"].with_suffix(".json")),
             "entries_highest_cycle": {"events": 4, "pulses": 9, "digis": 0, "clusters": 3, "calib": 5120},
             "events_check": check_events(events, ak.Array(stock_events), 4), "comparison": compare(fast, fast)}
    paths["check"].write_text(json.dumps(check))
    return SimpleNamespace(**{k: str(v) for k, v in paths.items()}), check, meta


def test_end_to_end_verified_recovery_without_response_or_fitting(bound_inputs, monkeypatch):
    from crvfastsim import chain
    from crvfastsim.reco import pulsefit
    from crvfastsim.veto import fastframes

    def forbidden(*args, **kwargs):
        raise AssertionError("response or fitting executed")

    for module, names in ((chain, ("charges", "digis", "reco_waveforms")),
                          (pulsefit, ("reconstruct",)), (fastframes, ("_setup", "_clusters", "frames_from_steps"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    args, _, _ = bound_inputs
    result = R.run(args)
    from pathlib import Path
    out = Path(args.outdir)
    assert sorted(p.name for p in out.iterdir()) == ["fast_table.root", "recovered.parquet", "recovered_float32.parquet", "recovery.json"]
    assert result["frame_closure"]["passed"]
    assert result["float32_comparison"]["identical_memberships"]
    assert result["event_pulse_counts"] == [6, 0, 3, 0]
    for name, digest in result["output_sha256"].items():
        assert R.sha256(out / name) == digest
    for source in result["sources"].values():
        assert R.sha256(source["path"]) == source["sha256"]
    with pytest.raises(ValueError, match="new output directory"):
        R.run(args)


@pytest.mark.parametrize("mutation", ["fast", "stock", "input", "manifest", "pulses", "meta", "identity", "conditions",
                                      "pulse_count", "closure", "pbtime", "channels"])
def test_bound_input_fail_closed_before_export(bound_inputs, mutation):
    from pathlib import Path
    args, check, meta = bound_inputs
    if mutation in ("fast", "stock", "input", "pulses"):
        with Path(getattr(args, mutation)).open("ab") as f:
            f.write(b"changed")
    elif mutation == "manifest":
        Path(args.input).with_suffix(".json").write_text("{}")
    elif mutation == "meta":
        Path(args.fast).with_suffix(".json").write_text("{}")
    elif mutation == "identity":
        check["response_identity_sha256"] = "c" * 64
    elif mutation == "conditions":
        check["conditions_equal"] = False
    elif mutation == "pulse_count":
        meta["timers"]["pulses"] += 1
        Path(args.fast).with_suffix(".json").write_text(json.dumps(meta))
        check["fast_metadata_sha256"] = R.sha256(Path(args.fast).with_suffix(".json"))
    elif mutation == "closure":
        fast = ak.from_parquet(args.fast)
        fast = ak.with_field(fast, ak.with_field(fast.coincs, fast.coincs.time + .1, "time"), "coincs")
        ak.to_parquet(fast, args.fast)
        check["fast_sha256"] = R.sha256(args.fast)
    elif mutation == "pbtime":
        with uproot.open(args.stock) as f:
            events = f["crvdump/events"].arrays(library="np")
        events["pbtime"] += 1
        with uproot.update(args.stock) as f:
            f.mktree("crvdump/events", {k: v.dtype for k, v in events.items()})
            f["crvdump/events"].extend(events)
        check["stock_sha256"] = R.sha256(args.stock)
    else:
        with uproot.update(args.stock) as f:
            f.mktree("crvdump/calib", {"channel": "int32", "status": "int32"})
            f["crvdump/calib"].extend({"channel": np.zeros(5120, np.int32), "status": np.zeros(5120, np.int32)})
        check["stock_sha256"] = R.sha256(args.stock)
    Path(args.check).write_text(json.dumps(check))
    with pytest.raises(ValueError):
        R.run(args)
    assert not Path(args.outdir).exists()


def test_matches_independent_fastframes_summary_convention(sample):
    events, pulses, g, status = sample
    # Explicit original collection order and original (not reconstructed) times.
    p1 = {"idx": np.arange(6), "bar": np.array([2, 0, 1, 0, 1, 2]),
          "sipm": np.zeros(6, dtype=int), "seq": np.array([0, 0, 0, 1, 1, 1]),
          "time": np.array([502., 500., 501., 600., 601., 602.]), "pe": np.full(6, 10.123456789)}
    p3 = {"idx": np.arange(3), "bar": np.array([2, 0, 1]), "sipm": np.ones(3, dtype=int),
          "seq": np.zeros(3, dtype=int), "time": np.array([701., 702., 700.]), "pe": np.full(3, 10.123456789)}
    ref = C.find_event(p1, g, status, R.finder_config()) + C.find_event(p3, g, status, R.finder_config())
    # Exactly the fastframes._clusters summary projection and frames_from_steps +w.
    w = -ak.to_numpy(events.pbtime).astype(np.float32).astype(float)[[0, 0, 2]]
    co = {}
    for saved, field in R.SUMMARY_FIELDS.items():
        values = np.array([len(r["layers"]) if saved == "nLayers" else r[field] for r in ref])
        if saved in ("time", "timeStart", "timeEnd"):
            values = values + w
        co[saved] = ak.unflatten(values, [2, 0, 1, 0])
    fast = ak.zip({k: events[k] for k in R.IDS} | {"coincs": ak.zip(co)}, depth_limit=1)
    result = R.recover(R.prepare_pulses(pulses, events, 16), g, status)
    report = R.compare_saved(result, fast)
    assert report["passed"]
    assert max(report["max_abs_differences"].values()) < 1e-9


def test_full_event_identity_subrun_and_input_event_order(sample):
    events, pulses, g, status = sample
    # Repeated event number in distinct subruns; event table deliberately unsorted.
    events = ak.Array({"run": [1430] * 3, "subrun": [2, 0, 1], "event": [7, 7, 7],
                       "pbtime": [-204.46, -224.46, -214.46]})
    pulses = {"ev": np.repeat([(2 << 20) | 7, 7], 3), "chan": np.tile([8, 0, 4], 2),
              "seq": np.zeros(6), "time": np.array([700., 701., 702., 800., 801., 802.]),
              "pe": np.full(6, 10.)}
    batch = R.prepare_pulses(pulses, events, 16)
    result = R.recover(batch, g, status)
    assert ak.to_list(result.subrun) == [2, 0]
    assert ak.to_list(result.event) == [7, 7]
    assert ak.to_list(result.idx) == [0, 0]
    assert ak.to_list(result.pulses) == [[0, 1, 2], [0, 1, 2]]
    np.testing.assert_array_equal(batch.offsets, [0, 3, 6, 6])


def test_compensation_uses_retained_status_and_config(sample):
    events, pulses, g, status = sample
    batch = R.prepare_pulses(pulses, events, 16)
    original = R.recover(batch, g, status)
    status[[2, 6, 10]] = 1
    compensated = R.recover(batch, g, status)
    assert compensated.pe[0] == 2 * original.pe[0]
    assert compensated.pe[2] == original.pe[2]
    assert R.finder_config().ignore_secondary_pulses is False


def test_input_change_during_finding_fails_before_export(bound_inputs, monkeypatch):
    from pathlib import Path
    args, _, _ = bound_inputs
    original = R.recover

    def changing(*a, **kw):
        result = original(*a, **kw)
        Path(args.counters).write_text("changed during finder dispatch")
        return result

    monkeypatch.setattr(R, "recover", changing)
    with pytest.raises(ValueError, match="input changed during recovery: counters"):
        R.run(args)
    assert not Path(args.outdir).exists()


def test_summary_absolute_tolerance_not_pe_relative_tolerance(sample):
    events, pulses, g, status = sample
    result = R.recover(R.prepare_pulses(pulses, events, 16), g, status)
    fast = saved_frames(result, events)
    def shifted(delta):
        return ak.with_field(fast, ak.with_field(fast.coincs, fast.coincs.PEs + delta, "PEs"), "coincs")
    assert R.compare_saved(result, shifted(5e-10))["passed"]
    with pytest.raises(ValueError, match="frame closure"):
        R.compare_saved(result, shifted(2e-9))
