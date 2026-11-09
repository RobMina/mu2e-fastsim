"""Reco dispatch: real kernels on synthetic inputs, no geometry/conditions assets."""
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


def assert_exact(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert_exact(a[key], b[key])
        else:
            assert a[key].dtype == b[key].dtype, key
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.pr_params("tiers1-promoted-False-False")
@pytest.mark.parametrize("attach", [False, True])
@pytest.mark.parametrize("with_pos", [False, True])
@pytest.mark.parametrize("tiers,expected", [
    (((16, 8, 4),), "native"),
    (((8, 1, 4), (16, 8, 4)), "promoted"),
    (((8, 1, 4),), "overflow"),
    (((4, 1, 4),), "oversize"),
])
def test_partial_batch_exact_members_properties_and_recovery(attach, with_pos, tiers, expected):
    cols = pulses()
    cfg = replace(C.FinderConfig(), add_opposite_side_pulses=attach)
    timers = {}
    out = J.finish(J.find_async(*cols, geometry(), np.zeros(32, int), cfg,
                               tiers=tiers, split=False, timers=timers, with_pos=with_pos))
    members = (11, 14, 17, 20, 26, 29) if attach else (11, 14, 17, 20)
    assert J.cluster_keys(out) == {(int(e), 3, members) for e in np.unique(cols[0])}
    assert timers.get("promoted_H16", 0) == (5 if expected == "promoted" else 0)
    np.testing.assert_array_equal(out["fallback_events"],
                                  np.unique(cols[0]) if expected in ("overflow", "oversize") else [])
    if with_pos:
        assert out["cl_pos"].shape == (5, 3)
    assert len(out["cl_event"]) == 5


@pytest.mark.slow
@pytest.mark.parametrize("fallback", [False, True])
def test_split_and_whole_event_fallback_exact(fallback):
    cols = list(pulses(1))
    # Same event: a small native block and an oversized block. Whole-event replacement must drop the native one.
    cols = [np.concatenate([v, np.tile(v, 3)]) for v in cols]
    cols[1] = np.arange(len(cols[0])) * 3 + 11
    cols[4][7:] += 1000
    out = J.finish(J.find_async(*cols, geometry(), np.zeros(32, int),
        tiers=((16, 8, 8),), split=True, split_min=1, with_pos=True), fallback=fallback)
    np.testing.assert_array_equal(out["fallback_events"], [1001])
    assert len(out["cl_event"]) == (2 if fallback else 1)
    assert len(J.cluster_keys(out)) == len(out["cl_event"])


@pytest.mark.parametrize("with_pos", [False, True])
def test_empty_find_and_chain(with_pos):
    found = J.finish(J.find_async(*pulses(0), geometry(), np.zeros(32, int), with_pos=with_pos))
    out, timers = CH.run({"ev": np.zeros(0, int)}, geometry(), np.zeros(32, int))
    assert out == [] and timers["events"] == timers["waveforms"] == 0
    assert all(len(v) == 0 for v in found.values())


def test_oneapi_skip_and_four_layer_cap_unchanged(monkeypatch):
    monkeypatch.setattr(J, "is_oneapi", lambda: True)
    tiers = ((16, 6, 8), (32, 8, 8), (64, 48, 3))
    assert J.backend_tiers(tiers) == (tiers[0], tiers[2])
    P = next(iter(J.type_params(geometry()).values()))
    assert P.max_layers == 4 and J._kfor(P, 48) == J.K_MAX_4LAYER == 32


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


@pytest.mark.slow
@pytest.mark.parametrize("length,rounds", [(12, 3), (12, 1), (36, 3)])
def test_pulse_partial_batch_preserves_row_indices_and_all_redo_flags(length, rounds):
    W = waveforms(count=2, length=length)
    W["n"][1] = 12
    rows = np.array([7, 1, 9, 2, 5, 0, 6, 3, 8])  # not sorted; last row follows a full batch
    pending = CH._pulse_dispatch(W, rows, replace(PF.RecoConfig(), max_pulses=rounds), 8, 12)
    assert pending[-1][0]["found"].shape == (8, rounds)
    assert set(pending[-1][0]) == set(CH.PULSE_KEYS)
    got = CH._pulse_collect(pending, W, rows, PF.RecoConfig())
    assert got[-1] == (8 if length > 12 else (9 if rounds == 1 else 0))
    assert set(got[0]) == set(rows)


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


@pytest.mark.pr_params("True")
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


@pytest.mark.slow
@pytest.mark.parametrize("flag", ["rounds_exhausted", "window_overflow", "input_truncated"])
def test_pulse_transfer_retains_each_recovery_flag(flag):
    W = waveforms(count=1)
    rows = np.array([4, 2, 0])
    pending = CH._pulse_dispatch(W, rows, PF.RecoConfig(), 8, 12)
    d, n = pending[0]
    flags = np.zeros(8, bool)
    flags[2] = True  # final real row maps to original waveform row 0, not row 2
    flags[3:] = True  # padding flags must never reach the host recovery row map
    d[flag] = jax.numpy.asarray(flags)
    assert CH._pulse_collect(pending, W, rows, PF.RecoConfig())[-1] == 1


@pytest.mark.pr_params("16384-1")
@pytest.mark.parametrize("batch,expected", [(8, 2), (6, 2), (16384, 1)])
def test_launch_shape_stays_the_tier_batch(monkeypatch, batch, expected):
    # Instrument shapes without asking the CPU to run a giant production tier.
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
    out = J._collect_tier(J._dispatch_tier(arr, P, 8, batch))
    assert [shape[0] for shape, _ in seen] == [batch] * expected
    assert all(shape[1] == 16 and K == 8 for shape, K in seen)
    np.testing.assert_array_equal(out["label"], arr["idx"])
