"""Combined resident physical response with explicit synthetic geometry/steps."""
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

jax = pytest.importorskip("jax")
pytest.importorskip("mu2efastsim.testing")
pytest.importorskip("trkfastsim")
pytest.importorskip("crvfastsim")

from test_mixing_device_tracker import helpers, cond
from mixing.device_integration import CAPACITY, source
from mixing.device_physics import _crv_pdg


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def template():
    from crvfastsim.geometry import find_counters
    from crvfastsim.reco.waveform import WaveformConfig

    # Use the public Offline template, not a synthetic replacement or a download.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CRVFS_NO_FETCH", "1")
        try:
            return Path(find_counters(WaveformConfig().single_pe_file))
        except FileNotFoundError:
            pytest.skip("public singlePEWaveform_v3.txt unavailable; set MU2E_SEARCH_PATH")


def test_truth_identity_join_sparse_keys_and_missing():
    cols = {
        "particles": {"mask": np.array([True, True, True, False]), "event": np.array([1, 0, 1, 0]),
                      "columns": {"key": np.array([2**53 + 7, 99, 99, 0]), "pdg": np.array([13, 22, 11, 0])}},
        "crv": {"mask": np.array([True, True, True, False]), "event": np.array([1, 0, 1, 0]),
                "columns": {"sp": np.array([99, 99, 2**53 + 7, 0])}}}
    actual, failed = jax.device_get(jax.jit(_crv_pdg)(jax.device_put(cols)))
    assert not failed
    np.testing.assert_array_equal(actual["columns"]["pdg"], [11, 22, 13, 0])
    cols["crv"]["columns"]["sp"][0] += 1
    _, failed = jax.device_get(jax.jit(_crv_pdg)(jax.device_put(cols)))
    assert failed


def test_combined_physical_stream_roundtrip(template, cond, tmp_path):
    from types import SimpleNamespace
    from trkfastsim.digi.digicond import Conditions
    from mixing.device_backend import Backend
    from mixing.stream_benchmark import run

    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    capacity = tmp_path / "capacities.json"
    capacity.write_text(json.dumps(CAPACITY))
    args = SimpleNamespace(out=tmp_path / "out", frames_manifest=None, device_platform="cpu",
                           device_kernel="mixing.device_integration:model", device_capacities=capacity,
                           synthetic=False, backend="mixing.device_backend:Backend", events=5, batch_events=2,
                           max_selected_rows=448, chunk=2, warmup_batches=1, seed=17, tracker_seed=19,
                           sampling_seed=23, require_stage=[], template=template)
    def source_factory(manifest, **kwargs):
        return source(c, **kwargs)
    report = run(args, source_factory, Backend)
    assert report["events"] == 5 and report["warmup_events"] == 2
    assert report["backend"]["model_kind"] == "physics"
    assert report["backend"]["model"]["synthetic_inputs"]
    assert report["backend"]["support"] == ["tracker_digis", "crv_reco"]
    totals = {"tracker": 0, "digis": 0, "pulses": 0, "clusters": 0}
    for path in args.out.glob("*/fast.root"):
        with uproot.open(path) as root:
            trk = root["trkdigi/events"].arrays(library="ak")
            assert {"d_tdc0", "d_tdc1", "d_tot0", "d_tot1", "d_flags", "d_nadc", "d_adc"} <= set(ak.fields(trk))
            assert ak.all(ak.num(trk.d_adc) == 15 * ak.num(trk.d_sid))
            totals["tracker"] += int(ak.sum(ak.num(trk.d_sid)))
            for name in ("digis", "pulses", "clusters"):
                totals[name] += root["crvfast/" + name].num_entries
    assert all(totals.values()), totals


def test_lifted_kernel_assets_are_explicit_operands():
    import jax.numpy as jnp
    from mixing.device_kernels import lift
    device_constant = jax.device_put(np.array([2., 3.]))
    @jax.jit
    def shared(x):
        return {"values": device_constant * x, "sum": (device_constant * x).sum()}
    lifted = lift(shared, jax.ShapeDtypeStruct((2,), np.float64))
    assert len(lifted.assets) == 1
    assets = jax.device_put(lifted.assets)
    values = jax.device_put(np.array([4., 5.]))
    program = jax.make_jaxpr(lifted.apply)(assets, values)
    assert len(program.consts) == 0
    with jax.transfer_guard("disallow"):
        out = jax.jit(lifted.apply)(assets, values)
        jax.block_until_ready(out)
    actual = jax.device_get(out)
    np.testing.assert_array_equal(actual["values"], [8., 15.])
    assert actual["sum"] == 23
    assert not lifted.assets[0].flags.writeable


def test_truth_pdg_invalid_float_and_unsigned_not_silently_truncated():
    for pdg in (np.array([13.5]), np.array([np.nan]), np.array([2**63 + 13], np.uint64)):
        cols = {"particles": {"mask": np.array([True]), "event": np.array([0]),
                              "columns": {"key": np.array([3]), "pdg": pdg}},
                "crv": {"mask": np.array([True]), "event": np.array([0]), "columns": {"sp": np.array([3])}}}
        _, failed = jax.device_get(jax.jit(_crv_pdg)(jax.device_put(cols)))
        assert failed


def test_occupancy_output_equals_full_tables(template, cond, tmp_path):
    """Same seeds and stream: on-device occupancy equals counts from the persisted full detector tables."""
    from types import SimpleNamespace
    from trkfastsim.digi.digicond import Conditions
    from mixing import device_occupancy as occ
    from mixing.device_backend import Backend
    from mixing.stream_benchmark import run

    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    capacity = tmp_path / "capacities.json"
    capacity.write_text(json.dumps(CAPACITY))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"sector_names": ["EX"], "bars_per_sector": [4], "geometry": "fixture"}))
    reports, outs = {}, {}
    for mode in ("root", "occupancy"):
        outs[mode] = tmp_path / mode
        args = SimpleNamespace(out=outs[mode], frames_manifest=None, device_platform="cpu",
                               device_kernel="mixing.device_integration:model", device_capacities=capacity,
                               synthetic=False, backend="mixing.device_backend:Backend", events=5, batch_events=2,
                               max_selected_rows=448, chunk=2, warmup_batches=1, seed=17, tracker_seed=19,
                               sampling_seed=23, require_stage=[], template=template,
                               device_output=mode, occupancy_geometry=geometry)
        reports[mode] = run(args, lambda manifest, **kwargs: source(c, **kwargs), Backend)
    assert reports["occupancy"]["backend"]["output_format"] == "occupancy"
    assert "occupancy" in reports["occupancy"]["boundary"] and "output_format" not in reports["root"]["backend"]
    paths = sorted(outs["root"].glob("*/fast.root"))
    assert len(paths) == 4
    for path in paths:
        with uproot.open(path) as root:
            trk = root["trkdigi/events"].arrays(library="ak")
            events = root["crvfast/events"].arrays(library="np")
        data = occ.read(outs["occupancy"] / path.parent.name / "occupancy.npz")
        ids = np.column_stack([events[k] for k in ("run", "subrun", "event")]).astype(np.uint64)
        np.testing.assert_array_equal(data["event_ids"], ids)
        planes = np.zeros((len(ids), 36), int)
        for row, sids in enumerate(ak.to_list(trk.d_sid)):
            np.add.at(planes[row], (np.asarray(sids, int) & 0xFC00) >> 10, 1)
        np.testing.assert_array_equal(data["planes"], planes)
        np.testing.assert_array_equal(data["sectors"][:, 0], events["npulses"])
        np.testing.assert_array_equal(data["crv_digis"], events["ndigis"])
        np.testing.assert_array_equal(data["crv_clusters"], events["nclusters"])
        sampling = (outs["occupancy"] / path.parent.name / "sampling.json").read_text()
        assert "source_event_ids" not in sampling and "omitted" in sampling
    assert not list(outs["occupancy"].glob("*/fast.root"))


def test_occupancy_output_rejected_without_device_backend(tmp_path):
    from types import SimpleNamespace
    from mixing.stream_benchmark import run

    class Host:
        metadata = {"name": "host", "support": ["tracker_digis", "crv_reco"], "residency": "host"}
        def __init__(self, args, source):
            pass
        def sync(self):
            pass

    class Pool:
        population, metadata = 4, {}
        def __init__(self, *args, **kwargs):
            pass

    args = SimpleNamespace(out=tmp_path / "out", frames_manifest=None, events=2, batch_events=2, max_selected_rows=8,
                           chunk=1, warmup_batches=0, seed=1, tracker_seed=1, sampling_seed=1, require_stage=[],
                           device_output="occupancy")
    with pytest.raises(ValueError, match="only by the device backend"):
        run(args, Pool, Host)


def test_capacity_usage_matches_tables_and_overflow_names_demand(template, cond, tmp_path):
    """Demand counters ride in the batch metrics; an overflow names its demand and capacity and writes nothing."""
    from types import SimpleNamespace
    from trkfastsim.digi.digicond import Conditions
    from mixing.device_backend import Backend
    from mixing.stream_benchmark import run

    c = Conditions(cond.s, cond.a)
    c.s["onspill"] = True
    capacity = tmp_path / "capacities.json"
    capacity.write_text(json.dumps(CAPACITY))
    args = SimpleNamespace(out=tmp_path / "out", frames_manifest=None, device_platform="cpu",
                           device_kernel="mixing.device_integration:model", device_capacities=capacity,
                           synthetic=False, backend="mixing.device_backend:Backend", events=4, batch_events=2,
                           max_selected_rows=448, chunk=2, warmup_batches=0, seed=17, tracker_seed=19,
                           sampling_seed=23, require_stage=[], template=template)
    run(args, lambda manifest, **kwargs: source(c, **kwargs), Backend)
    rows = [json.loads(line) for line in (args.out / "batches.jsonl").read_text().splitlines()]
    for row in rows:
        usage = row["backend"]["capacity_usage"]
        for name in ("rows/tracker", "max_donors", "response/tracker/drift_clusters", "response/tracker/straws",
                     "response/crv/photons", "response/crv/digis", "response/crv/reco_waveforms"):
            assert name in usage, name
        with uproot.open(args.out / row["directory"] / "fast.root") as root:
            assert usage["response/crv/digis"] == root["crvfast/digis"].num_entries
            sids = root["trkdigi/events"].arrays(["d_sid"], library="ak").d_sid
            assert usage["response/tracker/straws"] >= len(np.unique(ak.to_numpy(ak.flatten(sids))))
    small = json.loads(json.dumps(CAPACITY))
    small["response"]["crv"]["digis"] = 1
    capacity.write_text(json.dumps(small))
    args.out = tmp_path / "overflow"
    with pytest.raises(OverflowError, match=r"response/crv/crv_digis \(demand >= \d+, capacity 1\)"):
        run(args, lambda manifest, **kwargs: source(c, **kwargs), Backend)
