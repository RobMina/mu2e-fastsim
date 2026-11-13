"""Coincidence promotions of several chunks in shared kernel calls (coinc_jax.complete_all) give each chunk the same
outputs as its own finish(): same clusters, members, numbering, promotions and fallbacks."""
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from crvfastsim.reco import coinc as C, coinc_jax as J, chain_jax as CH

from test_coinc_device_pack import random_pulses, TIERS
from test_reco_dispatch import geometry, waveforms


def chunks(seed, n=3):
    cols = random_pulses(seed, events=60, mean=7)
    ue = np.unique(cols[0])
    edges = np.searchsorted(cols[0], ue[np.linspace(0, len(ue), n + 1).astype(int)[:-1]])
    edges = np.r_[edges, len(cols[0])]
    return [tuple(c[edges[i]:edges[i + 1]] for c in cols) for i in range(n)]


def assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype, k
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("seed,attach,split,with_pos", [(0, False, False, False), (1, True, False, True),
                                                       (2, True, True, True)])
def test_batched_promotions_equal_per_chunk_finish(seed, attach, split, with_pos):
    G, status = geometry(), np.zeros(32, int)
    cfg = replace(C.FinderConfig(), add_opposite_side_pulses=attach)
    tables, params = J.hit_tables(G, status, cfg), J.type_params(G, cfg)
    kw = dict(tiers=TIERS, split=split, split_min=8, with_pos=with_pos, tables=tables, params=params)
    parts = chunks(seed)
    t_one, t_all = {}, {}
    one = [J.finish(J.find_async(*c, G, status, cfg, timers=t_one, **kw)) for c in parts]
    many = J.complete_all([J.collect(J.find_async(*c, G, status, cfg, timers=t_all, **kw)) for c in parts])
    for a, b in zip(one, many):
        assert_same(a, b)
    assert {k: v for k, v in t_one.items() if k.startswith(("promoted_", "groups_"))} == \
        {k: v for k, v in t_all.items() if k.startswith(("promoted_", "groups_"))}
    assert any(k.startswith("promoted_") for k in t_one)


def test_mixed_handles_promote_one_by_one():
    # a host-packed handle cannot share device calls: every handle still gets its own finish() result
    G, status = geometry(), np.zeros(32, int)
    parts = chunks(3)
    ref = [J.find(*c, G, status, tiers=TIERS, split=False, device_pack=False) for c in parts]
    states = [J.collect(J.find_async(*c, G, status, tiers=TIERS, split=False, device_pack=(i != 1)))
              for i, c in enumerate(parts)]
    for a, b in zip(ref, J.complete_all(states)):
        assert_same(a, b)


@pytest.mark.parametrize("attach", [False, True])
def test_chain_run_deferred_promotions_exact(monkeypatch, attach):
    original = J.find_async
    monkeypatch.setattr(J, "find_async", lambda *a, **kw: original(*a, **kw, tiers=((4, 2, 8), (8, 4, 8), (16, 8, 8))))
    W = waveforms(count=12)
    cfg = replace(C.FinderConfig(), add_opposite_side_pulses=attach)
    got = {}
    for deferred in (False, True):
        monkeypatch.setattr(CH, "_deferrable", lambda h, d=deferred: d)
        got[deferred], _ = CH.run(W, geometry(), np.zeros(32, int), n_chunks=4, batch=64, coinc_cfg=cfg,
                                  keep_pulses=True, with_pos=True)
    assert len(got[True]) == len(got[False])
    for a, b in zip(got[False], got[True]):
        pa, pb = a.pop("pulses"), b.pop("pulses")
        assert_same(a, b)
        assert_same(pa, pb)
