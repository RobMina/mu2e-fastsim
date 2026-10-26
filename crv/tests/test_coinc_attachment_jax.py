"""Full-finder attachment regressions with explicit expected memberships and no external assets."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('jax')
from crvfastsim.reco import coinc as C
from crvfastsim.reco import coinc_jax as J


def geometry():
    layer = np.repeat(np.arange(4), 2)
    x = np.tile([0., 40.], 4)
    y = layer * 20.
    return C.SectorGeom(np.zeros(8, int), layer, np.tile([0, 1], 4), x, y,
        np.column_stack([x, y, np.zeros(8)]), np.array([2]), np.array([3]), ['T2'],
        np.array([-1400.]), np.array([1400.]), np.array([True]), np.array([True]))


def pulses(rows):
    a = np.asarray(rows)
    return dict(idx=np.arange(len(a)), bar=a[:, 0].astype(int), sipm=a[:, 1].astype(int),
                time=a[:, 2], pe=a[:, 3], seq=a[:, 4].astype(int))


def run(p, cfg=None, tiers=((32, 8, 2),), split=False, G=None):
    G = geometry() if G is None else G
    cfg = cfg or C.FinderConfig()
    status = np.zeros(32, int)
    ref = C.find_event(p, G, status, cfg)
    vec = C.find_event(p, G, status, cfg, vectorized=True)
    expected = {(0, r['sectortype'], tuple(sorted(r['pulses']))) for r in ref}
    assert expected == {(0, r['sectortype'], tuple(sorted(r['pulses']))) for r in vec}
    out = J.finish(J.find_async(np.zeros(len(p['idx']), int),
        *[p[k] for k in ('idx', 'bar', 'sipm', 'time', 'pe', 'seq')], G, status, cfg,
        tiers=tiers, split=split, split_min=1, with_pos=True))
    assert J.cluster_keys(out) == expected
    for i in range(len(out['cl_event'])):
        members = set(out['mem_idx'][out['mem_cl'] == i])
        r = next(r for r in ref if set(r['pulses']) == members)
        for name in ('pe', 'start', 'end', 'slope', 'avgtime', 'poscalc', 'pos'):
            np.testing.assert_allclose(out['cl_' + name][i], r[name], rtol=1e-12, atol=1e-10)
    return ref, out


def basic():
    return [(b, 0, 100., 10., 0) for b in (0, 2, 4, 6)] + [
        (0, 1, 104., 10., 0), (0, 1, 102., 10., 0), (0, 3, 103., 10., 0)]


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('tiers', [((16, 8, 2),), ((4, 1, 2),), ((8, 1, 2),), ((8, 1, 2), (16, 8, 2))])
def test_full_finder_attachment_and_fallback(split, tiers):
    ref, out = run(pulses(basic()), tiers=tiers, split=split)
    assert set(ref[0]['pulses']) == {0, 1, 2, 3, 5, 6}
    assert ref[0]['pe'] == 60.
    assert ref[0]['poscalc']
    assert bool(len(out['fallback_events'])) == (len(tiers) == 1 and tiers[0][0] <= 8)


def test_disabled_is_runtime_data():
    p = pulses(basic())
    a, _ = run(p)
    cache = J.find_dense._cache_size()
    b, _ = run(p, replace(C.FinderConfig(), add_opposite_side_pulses=False))
    assert J.find_dense._cache_size() == cache
    assert len(a[0]['pulses']) == 6 and len(b[0]['pulses']) == 4


@pytest.mark.parametrize('time,attached', [(130., False), (np.nextafter(130., 0.), True)])
def test_strict_window(time, attached):
    ref, _ = run(pulses(basic()[:4] + [(0, 1, time, 10., 0)]))
    assert (4 in ref[0]['pulses']) == attached


def test_candidate_tie_collection_order_and_no_secondary_attachment():
    p = pulses(basic()[:4] + [(0, 1, 104., 10., 0), (0, 1, 104., 12., 0), (0, 3, 101., 20., 1)])
    ref, _ = run(p, replace(C.FinderConfig(), ignore_secondary_pulses=False))
    assert set(ref[0]['pulses']) == {0, 1, 2, 3, 4}


def test_min_pe_is_applied_after_attachment():
    cfg = replace(C.FinderConfig(), cuts={'T2': replace(C.EXTRACTED_V04['T2'], min_cluster_pes=50.)})
    ref, _ = run(pulses(basic()), cfg)
    assert len(ref) == 1
    old, _ = run(pulses(basic()), replace(cfg, add_opposite_side_pulses=False))
    assert old == []


def test_same_counter_number_is_not_same_bar():
    ref, _ = run(pulses(basic()[:4] + [(1, 1, 101., 10., 0)]))
    assert set(ref[0]['pulses']) == {0, 1, 2, 3}


def test_stock_cluster_order_not_minimum_packed_root():
    rows = [(b, 1, 140., 10., 0) for b in (1, 2, 4, 6)]
    rows += [(b, 0, 100., 10., 0) for b in (0, 2, 4, 6)]
    rows += [(b, 0, 140., 10., 0) for b in (0, 2, 4, 6)]
    rows += [(0, 1, 120., 10., 0)]
    for split in (False, True):
        ref, _ = run(pulses(rows), split=split)
        assert len(ref) == 2
        assert set(ref[0]['pulses']) == {4, 5, 6, 7, 12}
        assert set(ref[1]['pulses']) == {0, 1, 2, 3, 8, 9, 10, 11}


def test_one_ended_geometry_does_not_attach():
    G = geometry()
    G.sipms_side1[:] = False
    ref, _ = run(pulses(basic()), G=G)
    assert set(ref[0]['pulses']) == {0, 1, 2, 3}


@pytest.mark.parametrize('secondary_anchor', [False, True])
def test_secondary_occupancy_and_anchor(secondary_anchor):
    rows = basic()[:4]
    if secondary_anchor:
        rows[0] = (0, 0, 100., 10., 1)
        rows += [(0, 1, 105., 10., 0)]
        expected = {0, 1, 2, 3}
    else:
        rows += [(b, 1, 100., 10., int(b == 0)) for b in (0, 2, 4, 6)]
        rows += [(0, 3, 115., 10., 0)]
        expected = set(range(8))
    ref, _ = run(pulses(rows), replace(C.FinderConfig(), ignore_secondary_pulses=False))
    assert set(ref[0]['pulses']) == expected


def test_no_cascade_from_attached_candidate():
    rows = basic()[:4] + [(0, 1, 125., 10., 0), (0, 2, 128., 10., 0)]
    ref, _ = run(pulses(rows))
    assert set(ref[0]['pulses']) == {0, 1, 2, 3, 4}


def test_negative_sequence_not_first_pulse():
    rows = basic()[:4] + [(0, 1, 105., 10., -1)]
    ref, _ = run(pulses(rows), replace(C.FinderConfig(), ignore_secondary_pulses=False))
    assert set(ref[0]['pulses']) == {0, 1, 2, 3}


@pytest.mark.parametrize('split', [False, True])
def test_multiple_batches_and_whole_event_fallback(split):
    G = geometry()
    status = np.zeros(32, int)
    ps = [pulses(basic()) for _ in range(5)]
    ps[3] = pulses(basic() + [(0, 1, 104. + i * .01, 10., 0) for i in range(12)])
    ev = np.concatenate([np.full(len(p['idx']), e) for e, p in enumerate(ps)])
    cols = [np.concatenate([p[k] for p in ps]) for k in ('idx', 'bar', 'sipm', 'time', 'pe', 'seq')]
    expected = {(e, r['sectortype'], tuple(sorted(r['pulses']))) for e, p in enumerate(ps)
                for r in C.find_event(p, G, status)}
    out = J.finish(J.find_async(ev, *cols, G, status, tiers=((16, 8, 2),), split=split,
                               split_min=1, with_pos=True))
    assert J.cluster_keys(out) == expected
    assert out['fallback_events'].tolist() == [3]
    assert out['cl_pos'].shape == (5, 3)
