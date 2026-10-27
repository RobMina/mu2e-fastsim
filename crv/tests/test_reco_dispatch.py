"""Opt-in dispatch parity: real kernels on synthetic inputs, no geometry/conditions assets."""
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from crvfastsim.reco import coinc as C, coinc_jax as J, chain_jax as CH, pulsefit as PF


def geometry():
    layer = np.repeat(np.arange(4), 2)
    x = np.tile([0., 40.], 4)
    y = layer * 20.
    return C.SectorGeom(np.zeros(8, int), layer, np.tile([0, 1], 4), x, y,
        np.column_stack([x, y, np.zeros(8)]), np.array([2]), np.array([3]), ["T2"],
        np.array([-1400.]), np.array([1400.]), np.array([True]), np.array([True]))


def pulses(count=5):
    # Deliberately sparse, nonidentity event/member indices. Two candidates compete for one slot.
    rows = np.array([(b, 0, 100., 10., 0) for b in (0, 2, 4, 6)] +
                    [(0, 1, 104., 10., 0), (0, 1, 102., 10., 0), (0, 3, 103., 10., 0)])
    return (np.repeat(1001 + np.arange(count) * 101, len(rows)),
            np.tile(np.arange(len(rows)) * 3 + 11, count),
            np.tile(rows[:, 0].astype(int), count), np.tile(rows[:, 1].astype(int), count),
            np.tile(rows[:, 2], count), np.tile(rows[:, 3], count),
            np.tile(rows[:, 4].astype(int), count))


def mode(monkeypatch, compact):
    if compact:
        monkeypatch.setenv("CRVFASTSIM_COMPACT_DISPATCH", "1")
    else:
        monkeypatch.delenv("CRVFASTSIM_COMPACT_DISPATCH", raising=False)


def assert_exact(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert_exact(a[key], b[key])
        else:
            assert a[key].dtype == b[key].dtype, key
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("attach", [False, True])
@pytest.mark.parametrize("with_pos", [False, True])
@pytest.mark.parametrize("tiers,expected", [
    (((16, 8, 4),), "native"),
    (((8, 1, 4), (16, 8, 4)), "promoted"),
    (((8, 1, 4),), "overflow"),
    (((4, 1, 4),), "oversize"),
])
def test_partial_batch_exact_members_properties_and_recovery(monkeypatch, attach, with_pos, tiers, expected):
    cols = pulses()
    cfg = replace(C.FinderConfig(), add_opposite_side_pulses=attach)
    outputs = []
    for compact in (False, True):
        mode(monkeypatch, compact)
        timers = {}
        out = J.finish(J.find_async(*cols, geometry(), np.zeros(32, int), cfg,
                                   tiers=tiers, split=False, timers=timers, with_pos=with_pos))
        outputs.append(out)
        members = (11, 14, 17, 20, 26, 29) if attach else (11, 14, 17, 20)
        assert J.cluster_keys(out) == {(int(e), 3, members) for e in np.unique(cols[0])}
        assert timers.get("promoted_H16", 0) == (5 if expected == "promoted" else 0)
        np.testing.assert_array_equal(out["fallback_events"],
                                      np.unique(cols[0]) if expected in ("overflow", "oversize") else [])
        if with_pos:
            assert out["cl_pos"].shape == (5, 3)
        assert len(out["cl_event"]) == 5
    assert_exact(*outputs)


@pytest.mark.parametrize("fallback", [False, True])
def test_split_and_whole_event_fallback_exact(monkeypatch, fallback):
    cols = list(pulses(1))
    # Same event: a small native block and an oversized block. Whole-event replacement must drop the native one.
    cols = [np.concatenate([v, np.tile(v, 3)]) for v in cols]
    cols[1] = np.arange(len(cols[0])) * 3 + 11
    cols[4][7:] += 1000
    outputs = []
    for compact in (False, True):
        mode(monkeypatch, compact)
        outputs.append(J.finish(J.find_async(*cols, geometry(), np.zeros(32, int),
            tiers=((16, 8, 8),), split=True, split_min=1, with_pos=True), fallback=fallback))
    assert_exact(*outputs)
    np.testing.assert_array_equal(outputs[0]["fallback_events"], [1001])
    assert len(outputs[0]["cl_event"]) == (2 if fallback else 1)
    assert len(J.cluster_keys(outputs[0])) == len(outputs[0]["cl_event"])


@pytest.mark.parametrize("with_pos", [False, True])
def test_empty_find_and_chain(monkeypatch, with_pos):
    outputs = []
    for compact in (False, True):
        mode(monkeypatch, compact)
        outputs.append(J.finish(J.find_async(*pulses(0), geometry(), np.zeros(32, int), with_pos=with_pos)))
        out, timers = CH.run({"ev": np.zeros(0, int)}, geometry(), np.zeros(32, int))
        assert out == [] and timers["events"] == timers["waveforms"] == 0
    assert_exact(*outputs)
    assert all(len(v) == 0 for v in outputs[0].values())


@pytest.mark.parametrize("batch", [1, 4, 6, 8, 16384])
def test_capacity_bounded_and_default_opt_in_only(monkeypatch, batch):
    for value in (None, "0", "true", "2", "1"):
        if value is None:
            monkeypatch.delenv("CRVFASTSIM_COMPACT_DISPATCH", raising=False)
        else:
            monkeypatch.setenv("CRVFASTSIM_COMPACT_DISPATCH", value)
        compact = J._compact_dispatch()
        assert compact == (value == "1")
        for n in range(1, min(batch, 17) + 1):
            cap = J._row_capacity(n, batch, compact)
            assert n <= cap <= batch
            if compact:
                assert cap == batch or cap & (cap - 1) == 0
                assert cap < 2 * n
            else:
                assert cap == batch


def test_real_rows_sliced_before_single_pytree_transfer_and_mode_pinned(monkeypatch):
    mode(monkeypatch, True)
    cols = pulses(3)
    handle = J.find_async(*cols, geometry(), np.zeros(32, int), tiers=((8, 1, 8), (16, 8, 8)),
                          split=False, with_pos=True)
    job = next(j for j in handle["jobs"] if not j.get("meta"))
    assert job["outs"][0][0]["root"].shape == (4, 8)
    # Promotions and collection must keep the dispatch mode even if environment changes while pending.
    mode(monkeypatch, False)
    seen = []
    original = jax.device_get
    def get(tree):
        assert isinstance(tree, list)
        for output in tree:
            assert set(output) == set(J.POS_KEYS)
            assert all(isinstance(v, jax.Array) and v.shape[0] == 3 for v in output.values())
            assert output["label"].shape[1] in (8, 16)  # hit/label axis must never shrink
        seen.append(tree)
        return original(tree)
    monkeypatch.setattr(jax, "device_get", get)
    out = J.finish(handle)
    assert len(seen) == 2  # one whole-pytree transfer for first tier, one for promotion
    assert len(out["cl_event"]) == 3 and not len(out["fallback_events"])
    assert seen[1][0]["root"].shape == (3, 16)


def test_oneapi_skip_and_four_layer_cap_unchanged(monkeypatch):
    mode(monkeypatch, True)
    monkeypatch.setattr(J, "is_oneapi", lambda: True)
    tiers = ((16, 6, 8), (32, 8, 8), (64, 48, 3))
    assert J.backend_tiers(tiers) == (tiers[0], tiers[2])
    P = next(iter(J.type_params(geometry()).values()))
    assert P.max_layers == 4 and J._kfor(P, 48) == J.K_MAX_4LAYER == 32
    assert J._row_capacity(3, 3, True) == 3  # no rounding above a non-power-of-two backend cap


def waveforms(count=5, length=12):
    bars = np.tile([0, 2, 4, 6, 0], count)
    sipm = np.tile([0, 0, 0, 0, 1], count)
    n = len(bars)
    t = np.arange(length) * PF.PERIOD
    peaks = np.array([4] if length == 12 else [4, 20])
    signal = PF.gumbel(t[None, :], 1000, peaks[:, None] * PF.PERIOD, 12).sum(0)
    return dict(ev=np.repeat(np.arange(count) * 101 + 1001, 5), bar=bars, sipm=sipm,
                adc=np.broadcast_to(np.rint(signal + 100).astype(int), (n, length)).copy(),
                n=np.full(n, length), start=np.zeros(n, int), ped=np.full(n, 100.),
                area=np.full(n, 400.), height=np.full(n, 11.), toff=sipm * 4.)


@pytest.mark.parametrize("length,rounds", [(12, 3), (12, 1), (36, 3)])
def test_pulse_partial_batch_preserves_row_indices_and_all_redo_flags(monkeypatch, length, rounds):
    W = waveforms(count=2, length=length)
    W["n"][1] = 12
    rows = np.array([7, 1, 9, 2, 5, 0, 6, 3, 8])  # not sorted; last row follows a full batch
    outputs = []
    for compact in (False, True):
        mode(monkeypatch, compact)
        pending = CH._pulse_dispatch(W, rows, replace(PF.RecoConfig(), max_pulses=rounds), 8, 12)
        assert pending[-1][0]["found"].shape == (8, rounds)
        assert set(pending[-1][0]) == set(CH.PULSE_KEYS)
        got = CH._pulse_collect(pending, W, rows, PF.RecoConfig())
        outputs.append(got)
        expected_redo = 8 if length > 12 else (9 if rounds == 1 else 0)
        assert got[-1] == expected_redo
        assert set(got[0]) == set(rows)
    for a, b in zip(*outputs):
        np.testing.assert_array_equal(a, b)


def combine(outputs):
    merged = {}
    offsets = np.cumsum([0] + [len(o["cl_event"]) for o in outputs])
    for key in outputs[0]:
        if key == "pulses":
            merged[key] = {k: np.concatenate([o[key][k] for o in outputs]) for k in outputs[0][key]}
        else:
            merged[key] = np.concatenate([o[key] + offsets[i] if key == "mem_cl" else o[key]
                                          for i, o in enumerate(outputs)])
    return merged


@pytest.mark.parametrize("attach", [False, True])
def test_chain_chunk_invariance_exact_outputs_and_member_links(monkeypatch, attach):
    # Keep this CPU regression small; production default tier caps are not changed.
    original = J.find_async
    def small_tiers(*args, **kwargs):
        return original(*args, **kwargs, tiers=((16, 8, 8),))
    monkeypatch.setattr(J, "find_async", small_tiers)
    W = waveforms()
    cfg = replace(C.FinderConfig(), add_opposite_side_pulses=attach)
    reference = None
    for compact in (False, True):
        mode(monkeypatch, compact)
        for chunks in (1, 3, 8):
            outputs, timers = CH.run(W, geometry(), np.zeros(32, int), n_chunks=chunks, batch=64,
                                     coinc_cfg=cfg, keep_pulses=True, with_pos=True)
            got = combine(outputs)
            assert timers["events"] == 5 and timers["pulses"] == 25
            assert J.cluster_keys(got) == {(int(e), 3, tuple(range(5 if attach else 4)))
                                          for e in np.unique(W["ev"])}
            assert got["cl_pos"].shape == (5, 3)
            if reference is None:
                reference = got
            else:
                assert_exact(reference, got)


@pytest.mark.parametrize("flag", ["rounds_exhausted", "window_overflow", "input_truncated"])
def test_pulse_transfer_retains_each_recovery_flag(monkeypatch, flag):
    W = waveforms(count=1)
    rows = np.array([4, 2, 0])
    outputs = []
    original = jax.device_get
    for compact in (False, True):
        mode(monkeypatch, compact)
        pending = CH._pulse_dispatch(W, rows, PF.RecoConfig(), 8, 12)
        d, n = pending[0]
        flags = np.zeros(8, bool)
        flags[2] = True  # final real row maps to original waveform row 0, not row 2
        flags[3:] = True  # padding flags must never reach the host recovery row map
        d[flag] = jax.numpy.asarray(flags)
        seen = []
        def get(tree):
            assert len(tree) == 1 and set(tree[0]) == set(CH.PULSE_KEYS)
            assert all(v.shape[0] == len(rows) for v in tree[0].values())
            seen.append(tree)
            return original(tree)
        monkeypatch.setattr(jax, "device_get", get)
        # Transfer choice is captured with dispatch, not read from the current environment.
        mode(monkeypatch, not compact)
        outputs.append(CH._pulse_collect(pending, W, rows, PF.RecoConfig()))
        assert len(seen) == int(compact)
        assert outputs[-1][-1] == 1
    for a, b in zip(*outputs):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("batch,expected", [(8, [8, 2]), (6, [6, 4]), (16384, [16])])
def test_launch_capacity_and_transfer_pytree_across_batches(monkeypatch, batch, expected):
    # Instrument shapes without asking the CPU to run a giant baseline production tier.
    cols = pulses(10)
    G = geometry()
    tables = J.hit_tables(G, np.zeros(32, int))
    P = J.type_params(G)[3]
    arr, _, _ = J.pack(*cols, G, tables, 3, H=16)
    seen = []
    def dense(*args, **kwargs):
        seen.append((args[0].shape, kwargs["K"]))
        # Sentinel row identity in every leaf; real membership/property arithmetic is tested above.
        return {k: args[1] for k in J.POS_KEYS}
    monkeypatch.setattr(J, "find_dense", dense)
    results = []
    original = jax.device_get
    for compact in (False, True):
        mode(monkeypatch, compact)
        seen.clear()
        pending = J._dispatch_tier(arr, P, 8, batch)
        assert [shape[0] for shape, _ in seen] == (expected if compact else [batch] * len(expected))
        assert all(shape[1] == 16 and K == 8 for shape, K in seen)
        transfers = []
        def get(tree):
            assert len(tree) == len(expected)
            assert all(set(d) == set(J.OUT_KEYS) for d in tree)  # don't transfer unrequested positions
            assert [d["label"].shape[0] for d in tree] == [n for _, n in pending]
            transfers.append(tree)
            return original(tree)
        monkeypatch.setattr(jax, "device_get", get)
        results.append(J._collect_tier(pending))
        assert len(transfers) == int(compact)
    assert_exact(*results)
    np.testing.assert_array_equal(results[1]["label"], arr["idx"])
