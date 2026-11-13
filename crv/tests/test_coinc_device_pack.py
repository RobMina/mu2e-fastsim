"""Coincidence kernel inputs packed on the device (coinc_jax._pack_device) equal the host pack(): same arrays, so the
same clusters, members, promotions and fallbacks."""
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from crvfastsim import capacity
from crvfastsim.reco import coinc as C, coinc_jax as J
from mu2efastsim import precision as dprec

from test_reco_dispatch import geometry, pulses


def random_pulses(seed, events=40, mean=9):
    """Pulses on the 8-bar test sector: events sorted, collection order within, some secondary pulses."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(mean, events) + 1
    ev = np.repeat(np.sort(rng.choice(10_000, events, replace=False)), n)
    idx = np.arange(len(ev)) - np.searchsorted(ev, ev)
    bar = rng.integers(0, 8, len(ev))
    sipm = rng.integers(0, 4, len(ev))
    time = rng.uniform(0, 60, len(ev)) + 1000.0 * rng.integers(0, 3, len(ev))
    pe = rng.uniform(3, 60, len(ev))
    seq = (rng.random(len(ev)) < 0.15).astype(int)
    return ev, idx, bar, sipm, time, pe, seq


def host_padded(arr, batch):
    return {k: np.concatenate([v, np.zeros((batch - len(v),) + v.shape[1:], v.dtype)]) for k, v in arr.items()
            if k in J.DENSE_KEYS}


@pytest.mark.parametrize("seed", [0, 1])
def test_device_pack_equals_host_pack(seed):
    ev, idx, bar, sipm, time, pe, seq = random_pulses(seed)
    G, status, cfg = geometry(), np.zeros(32, int), C.FinderConfig()
    tables = J.hit_tables(G, status)
    H, batch = 32, 64
    sel = J._groups(3, J.type_params(G)[3], ev, idx, bar, sipm, time, pe, seq, G, tables, cfg, False, 0, {})["sel"]
    cols = (ev[sel], idx[sel], bar[sel], sipm[sel], time[sel], pe[sel], seq[sel])
    arr, tg, _ = J.pack(*cols, G, tables, 3, H=H)
    g = cols[0]
    gev, first, cnt = np.unique(g, return_index=True, return_counts=True)
    assert cnt.max() <= H and (seq[sel] == 0).all()
    inv = np.repeat(np.arange(len(gev)), cnt)
    col = np.arange(len(inv)) - first[inv]
    hit, rows, hidx = J._tier_hits(cols, inv, np.flatnonzero(cnt <= H))
    np.testing.assert_array_equal(gev[cnt <= H], tg)
    slot = np.full(4096, batch * H, np.int64)
    slot[:len(rows)] = rows * H + col[hidx]
    pad = lambda v: np.concatenate([v, np.zeros(4096 - len(v), v.dtype)])
    dev = J._pack_device(slot, *(pad(hit[k]) for k in ("idx", "bar", "sipm", "first", "time", "pe")),
                         np.int64(len(tg)), J._device_tables(G, tables), B=batch, H=H, single=bool(dprec.SINGLE))
    ref = host_padded(arr, batch)
    for k in J.DENSE_KEYS:
        got = np.asarray(dev[k])
        assert got.dtype == ref[k].dtype, k
        np.testing.assert_array_equal(got, ref[k], err_msg=k)


TIERS = ((4, 2, 4), (8, 3, 4), (16, 6, 2))   # small tiers: native rows, promotions, and events beyond the last tier


@pytest.mark.parametrize("seed,attach,split", [(0, False, False), (1, True, False), (2, True, True)])
def test_find_device_pack_exact(seed, attach, split):
    cols = random_pulses(seed, mean=7)
    G, status = geometry(), np.zeros(32, int)
    cfg = replace(C.FinderConfig(), add_opposite_side_pulses=attach)
    out, timers = {}, {}
    for dp in (False, True):
        timers[dp] = {}
        out[dp] = J.finish(J.find_async(*cols, G, status, cfg, tiers=TIERS, split=split, split_min=8,
                                        timers=timers[dp], with_pos=True, device_pack=dp))
    assert out[True].keys() == out[False].keys()
    for k in out[False]:
        assert out[True][k].dtype == out[False][k].dtype, k
        np.testing.assert_array_equal(out[True][k], out[False][k], err_msg=k)
    promoted = {k: v for k, v in timers[False].items() if k.startswith(("promoted_", "groups_"))}
    assert promoted == {k: v for k, v in timers[True].items() if k.startswith(("promoted_", "groups_"))}
    assert any(k.startswith("promoted_") for k in promoted) or len(out[False]["fallback_events"])


def test_find_device_pack_fixture_promotions():
    # the hand-made fixture of test_reco_dispatch: native, promoted, overflow and oversize rows
    for tiers in (((16, 8, 4),), ((8, 1, 4), (16, 8, 4)), ((8, 1, 4),), ((4, 1, 4),)):
        a = J.find(*pulses(), geometry(), np.zeros(32, int), tiers=tiers, split=False, device_pack=False)
        b = J.find(*pulses(), geometry(), np.zeros(32, int), tiers=tiers, split=False, device_pack=True)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_device_pack_hit_capacity():
    cols = random_pulses(3)
    G, status = geometry(), np.zeros(32, int)
    rec = capacity.Capacities(record=True)
    J.finish(J.find_async(*cols, G, status, tiers=TIERS, split=False, device_pack=True, caps=rec))
    used = rec.report()["usage"]
    assert used and all(k.startswith("coinc_hits/") for k in used)
    caps = capacity.Capacities(capacity.derive(used, 1.25))
    want = J.find(*cols, G, status, tiers=TIERS, split=False, device_pack=False)
    got = J.finish(J.find_async(*cols, G, status, tiers=TIERS, split=False, device_pack=True, caps=caps))
    assert not caps.report()["overflow"]
    for k in want:
        np.testing.assert_array_equal(got[k], want[k], err_msg=k)
