"""Real JAX integration, independent host overlay oracle and no callback/mocked GPU."""
import json
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

jax = pytest.importorskip("jax")

from mixing import device_backend as device
from mixing import device_synthetic as fake
from mixing.device_source import Layout, pack_source, sample_overlay
from mixing.frames import FrameBatch, overlay
from mixing.sampling import FrameSelection, STREAMS
from mixing.stream_benchmark import run


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def args(tmp_path):
    capacity = tmp_path / "capacity.json"
    capacity.write_text(json.dumps({"max_donors": 16, "rows": dict.fromkeys(("particles", "gen", "tracker", "crv"), 1024)}))
    return SimpleNamespace(out=tmp_path / "out", frames_manifest=None, device_platform="cpu",
                           device_kernel="mixing.device_synthetic:model", device_capacities=capacity,
                           synthetic=True, backend="mixing.device_backend:Backend", events=13, batch_events=4,
                           max_selected_rows=4096, chunk=4, warmup_batches=1, seed=17, tracker_seed=19,
                           sampling_seed=23, require_stage=[])


def test_real_compiled_integration_tail_wrap_and_output(args):
    report = run(args, fake.source, device.Backend)
    assert report["events"] == 13 and report["warmup_events"] == 4
    assert report["measured_batches"] == 4
    assert report["backend"]["model_kind"] == "synthetic"
    assert report["backend"]["support"] == ["synthetic_tracker", "synthetic_crv"]
    assert report["speedup"] is None
    rows = [json.loads(line) for line in (args.out / "batches.jsonl").read_text().splitlines()]
    assert [r["events"] for r in rows] == [4, 4, 4, 4, 1]
    previous = None
    for row in rows:
        assert row["backend"]["terminal_device_get_calls"] == 1
        folder = args.out / row["directory"]
        receipt = json.loads((folder / "sampling.json").read_text())
        with np.load(folder / "sampling.npz") as arrays:
            ids = arrays[receipt["signal_event_ids"]["npz_key"]]
            assert np.all(ids[:, 0] == np.uint64(2**63))
            indices = (np.arange(row["events"]) + row["ordinal_start"]) % 5
            np.testing.assert_array_equal(ids[:, 2], indices.astype(np.uint64) + np.uint64(2**53))
            current = {name: {field: arrays[selection[field]["npz_key"]].item()
                              for field in ("initial_cursor", "next_cursor")}
                       for name, selection in receipt["selections"].items()}
            if previous is not None:
                for name in STREAMS:
                    assert current[name]["initial_cursor"] == previous[name]["next_cursor"]
            previous = current
        with uproot.open(folder / "fast.root") as root:
            trk = root["trkdigi/events"].arrays(library="ak")
            assert ak.all(ak.num(trk.d_adc) == 15 * ak.num(trk.d_sid))
            assert root["crvfast/events"].num_entries == row["events"]


def test_device_overlay_matches_independent_existing_host_oracle():
    import jax.numpy as jnp
    src = fake.source(max_selected_rows=4096)
    layout = Layout(4, 16, dict.fromkeys(src.signal.collections, 1024), 4096)
    pools, law = pack_source(src, layout)
    pools = jax.device_put(pools)
    state = jax.device_put({"first": np.array(True), "cursors": np.zeros(4, np.int64)})
    execute = jax.jit(lambda state, ordinal, size: sample_overlay(pools, state, jax.random.PRNGKey(19), ordinal, size, layout=layout, law=law))
    for ordinal, size in ((0, 4), (4, 3), (7, 1)):
        batch, receipt, state, errors = execute(state, jnp.uint64(ordinal), jnp.int64(size))
        batch, receipt, errors = jax.device_get((batch, receipt, errors))
        assert not any(errors.values())
        idx = receipt["signal_indices"][:size]
        primary = FrameBatch(src.signal.event_ids[idx], {name: rows[idx] for name, rows in src.signal.collections.items()})
        backgrounds = []
        for name in STREAMS:
            s = receipt["selections"][name]
            selection = FrameSelection(s["counts"][:size], s["donors"][:size][s["mask"][:size]])
            backgrounds.append((src._pools[name], selection))
        expected = overlay(primary, backgrounds)
        np.testing.assert_array_equal(batch["keys"][:size], expected.event_ids)
        for name, rows in expected.collections.items():
            actual = batch["collections"][name]
            assert receipt["actual_rows"][name] == int(ak.sum(ak.num(rows)))
            np.testing.assert_array_equal(actual["event"][actual["mask"]], np.repeat(np.arange(size), ak.to_numpy(ak.num(rows))))
            for field in ak.fields(rows):
                np.testing.assert_array_equal(actual["columns"][field][actual["mask"]], ak.to_numpy(ak.flatten(rows[field])))


@pytest.mark.parametrize("failure", ["donors", "rows", "aggregate"])
def test_overflow_has_no_output_or_state_commit(args, tmp_path, failure):
    src = fake.source(max_selected_rows=args.max_selected_rows)
    if failure == "donors":
        src._rates = dict.fromkeys(STREAMS, 100.)
    elif failure == "rows":
        args.device_capacities.write_text(json.dumps({"max_donors": 16, "rows": dict.fromkeys(src.signal.collections, 1)}))
    else:
        args.max_selected_rows = 1
    backend = device.Backend(args, src)
    request = src.next_batch(4)
    state_before = jax.device_get(backend.state)
    with pytest.raises(OverflowError, match="device batch rejected"):
        backend.run(request, tmp_path)
    assert src._ordinal == src._batch_index == 0
    np.testing.assert_array_equal(jax.device_get(backend.state)["cursors"], state_before["cursors"])
    assert not (tmp_path / "fast.root").exists()
    assert not (tmp_path / "sampling.json").exists()


def test_all_empty_and_partial_tail_no_padding_output(args, tmp_path):
    src = fake.source(max_selected_rows=4096)
    src._rates = dict.fromkeys(STREAMS, 0.)
    backend = device.Backend(args, src)
    backend.run(src.next_batch(1), tmp_path)
    with uproot.open(tmp_path / "fast.root") as root:
        assert root["crvfast/events"].num_entries == 1
        for name in ("digis", "pulses", "clusters"):
            assert root["crvfast/" + name].num_entries == 0
        assert ak.sum(ak.num(root["trkdigi/events"]["d_sid"].array())) == 0


def test_repeated_seed_invocations_equal_every_output(args, tmp_path):
    run(args, fake.source, device.Backend)
    first = args.out
    args.out = tmp_path / "second"
    run(args, fake.source, device.Backend)
    from mixing.direct_regression import compare_outputs
    for path in first.glob("*/fast.root"):
        compare_outputs(path, args.out / path.parent.name / "fast.root")


@pytest.mark.parametrize("requested", ["tracker_digis", "crv_reco", "track_reco"])
def test_synthetic_cannot_advertise_physical_stages(args, requested):
    args.require_stage = [requested]
    with pytest.raises(ValueError, match="requested stages"):
        run(args, fake.source, device.Backend)
    assert not (args.out / "stream_report.json").exists()


def test_fake_requires_explicit_opt_in(args):
    args.synthetic = False
    with pytest.raises(ValueError, match="requires --synthetic"):
        run(args, fake.source, device.Backend)


def test_requires_x64_without_silent_id_truncation(args):
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="X64"):
            device.Backend(args, fake.source(max_selected_rows=4096))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_rejects_pure_host_callback_in_nested_jit():
    import jax.numpy as jnp
    @jax.jit
    def callback(x):
        return jax.pure_callback(lambda v: v + 1, jax.ShapeDtypeStruct((), jnp.float64), x)
    with pytest.raises(ValueError, match="host"):
        device._audit_program(jax.make_jaxpr(callback)(jnp.float64(1)))


def test_sampling_scalar_high_ordinal_not_promoted_to_float():
    import jax.numpy as jnp
    src = fake.source(max_selected_rows=4096)
    layout = Layout(4, 16, dict.fromkeys(src.signal.collections, 1024), 4096)
    pools, law = pack_source(src, layout)
    state = {"first": np.array(True), "cursors": np.zeros(4, np.int64)}
    ordinal = 2**53 + 1
    batch, _, _, _ = jax.jit(lambda p, s: sample_overlay(p, s, jax.random.PRNGKey(1), jnp.uint64(ordinal), jnp.int64(4), layout=layout, law=law))(jax.device_put(pools), jax.device_put(state))
    np.testing.assert_array_equal(jax.device_get(batch["keys"]), src.signal.event_ids[(np.arange(4) + ordinal) % 5])


def test_pack_mixed_unsigned_sparse_keys_are_lossless():
    src = fake.source(max_selected_rows=4096)
    parts = src.signal.collections["particles"]
    sparse = ak.values_astype(parts.key, np.uint64) + np.uint64(2**53)
    src.signal.collections["particles"] = ak.with_field(parts, sparse, "key")
    src.signal.collections["particles"] = ak.with_field(src.signal.collections["particles"],
        ak.where(parts.parent >= 0, parts.parent + 2**53, -1), "parent")
    for name in ("tracker", "crv"):
        rows = src.signal.collections[name]
        src.signal.collections[name] = ak.with_field(rows, ak.values_astype(rows.sp, np.uint64) + np.uint64(2**53), "sp")
    packed, _ = pack_source(src, Layout(4, 16, dict.fromkeys(src.signal.collections, 1024), 4096))
    assert packed["maxkey"][1] == 2**53 + 3
    assert packed["maxkey"][2] == 2**53 + 13
    assert packed["maxkey"].dtype == np.int64


def test_unsigned_synthetic_channels_compile(args, tmp_path):
    src = fake.source(max_selected_rows=4096)
    for pool in [src.signal, *src._pools.values()]:
        for name, field in (("tracker", "sid"), ("crv", "bar")):
            pool.collections[name] = ak.with_field(pool.collections[name],
                ak.values_astype(pool.collections[name][field], np.uint64), field)
    backend = device.Backend(args, src)
    assert backend.run(src.next_batch(4), tmp_path)["events"] == 4


from test_mixing_direct_frames import manifest as source_manifest


def test_real_root_source_factory_used_automatically(args, source_manifest):
    args.frames_manifest = source_manifest[0]
    args.device_capacities.write_text(json.dumps({"max_donors": 64,
        "rows": dict.fromkeys(("particles", "gen", "tracker", "crv"), 4096)}))
    args.max_selected_rows = 16000
    report = run(args)
    assert report["events"] == 13
    config = json.loads((args.out / "configuration.json").read_text())
    assert "source_fingerprints" in config["source"]
    assert "uploaded once" in config["source"]["residency"]
