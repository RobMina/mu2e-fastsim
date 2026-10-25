"""Synthetic regression contracts for the general chain and DQM table reco."""
from dataclasses import replace

import numpy as np
import pytest

from crvfastsim import chain
from crvfastsim.reco import coinc, pulsefit as PF


def conditions(n=8):
    z = np.zeros(n)
    return chain.Conditions(z + 100, z + 400, z + 11, np.arange(n) * 7.0 - 13,
                            np.arange(n, dtype=np.int64), z.astype(int), z.astype(int), z)


def waveforms(peaks=(4,), length=12, channels=(0,)):
    t = np.arange(length) * PF.PERIOD
    signal = PF.gumbel(t[None, :], 1000, np.asarray(peaks)[:, None] * PF.PERIOD, 12).sum(0)
    ch = np.asarray(channels, dtype=np.int64)
    return dict(ev=np.ones(len(ch), dtype=np.int64), chan=ch, bar=ch // 4, sipm=ch % 4,
                adc=np.broadcast_to(np.rint(signal + 100).astype(np.int64), (len(ch), length)).copy(),
                n=np.full(len(ch), length), start=np.zeros(len(ch), dtype=np.int64),
                ped=np.full(len(ch), 100.), area=np.full(len(ch), 400.), height=np.full(len(ch), 11.))


def numpy_pulses(W, max_pulses=128):
    return PF.reconstruct(*(W[k] for k in ('adc', 'n', 'start', 'ped', 'area', 'height')),
                          replace(PF.RecoConfig(), max_pulses=max_pulses))


@pytest.mark.parametrize('peaks,length', [((18,), 24), ((4, 20), 36), ((11,), 24),
                                          (tuple(range(4, 244, 24)), 252)])
def test_dispatch_retries_full_waveform(peaks, length):
    pytest.importorskip('jax')
    from crvfastsim.reco import chain_jax
    W = waveforms(peaks, length, (0, 1))
    rows = np.array([1, 0])
    ref = numpy_pulses(W)
    assert len(ref['wf']) == 2 * len(peaks)
    out = chain_jax._pulse_dispatch(W, rows, replace(PF.RecoConfig(), max_pulses=3), 4, 12)
    wf, seq, pe, tm, nredo = chain_jax._pulse_collect(out, W, rows, PF.RecoConfig())
    assert nredo == len(rows)
    np.testing.assert_array_equal(wf, rows[ref['wf']])
    np.testing.assert_array_equal(seq, ref['seq'])
    np.testing.assert_allclose(pe, ref['pe'], rtol=1e-12)
    np.testing.assert_allclose(tm, ref['time'], atol=1e-12)


def test_reco_waveforms_carries_offsets():
    cond = conditions()
    W = waveforms(channels=(3, 0, 5))
    D = dict(chan=W['ev'] * chain.CHAN_MULT + W['chan'], starttdc=W['start'], adc=W['adc'])
    grouped = chain.reco_waveforms(D, cond)
    np.testing.assert_array_equal(grouped['toff'], cond.timeoffset[grouped['chan']])


@pytest.mark.parametrize('jax', [False, True])
@pytest.mark.parametrize('explicit_offsets', [False, True])
def test_chain_conditions_before_coincidences(monkeypatch, jax, explicit_offsets):
    cond = conditions()
    W = waveforms(channels=tuple(range(8)))
    offsets = cond.timeoffset.copy()
    if explicit_offsets:
        W['toff'] = offsets = offsets + 31
    ref = numpy_pulses(W)
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    seen = []
    if jax:
        pytest.importorskip('jax')
        from crvfastsim.reco import chain_jax
        CJ = chain_jax.CJ
        monkeypatch.setattr(CJ, 'type_params', lambda *a: None)
        monkeypatch.setattr(CJ, 'hit_tables', lambda *a: None)
        def find(ev, idx, bar, sipm, tm, pe, seq, G, status, *a, **kw):
            seen.append((bar * 4 + sipm, tm, idx, status))
            return {'cl_event': np.zeros(0, dtype=np.int64)}
        monkeypatch.setattr(CJ, 'find_async', find)
        monkeypatch.setattr(CJ, 'finish', lambda x: x)
        dispatch = chain_jax._pulse_dispatch
        monkeypatch.setattr(chain_jax, '_pulse_dispatch',
                            lambda W, rows, cfg, batch, L: dispatch(W, rows, cfg, 16, L))
    else:
        def find(P, G, status, *a, **kw):
            seen.append((P['bar'] * 4 + P['sipm'], P['time'], P['idx'], status))
            return []
        monkeypatch.setattr(coinc, 'find_event', find)
    outs, timers = chain.clusters(W, cond, None, None, chain.ChainConfig(jax=jax), detail=True)
    assert len(seen) == 1
    ch, tm, idx, status = seen[0]
    live = np.array([0, 1, 4, 5])
    np.testing.assert_array_equal(ch, live)
    np.testing.assert_array_equal(idx, np.arange(len(live)))
    np.testing.assert_array_equal(status, cond.status)
    np.testing.assert_allclose(tm, ref['time'][live] + offsets[live], atol=1e-4)
    assert timers['pulses'] == len(live)
    np.testing.assert_array_equal(outs[0]['pulses']['bar'] * 4 + outs[0]['pulses']['sipm'], live)


@pytest.mark.parametrize('ignore', [False, True])
def test_table_reco_status_offsets_and_digi_links(monkeypatch, ignore):
    from valhist.fasttables import fast_reco
    cond = conditions()
    W = waveforms(channels=tuple(range(8)))
    D = {k: W[k] for k in ('ev', 'bar', 'sipm', 'adc')}
    D['starttdc'] = W['start']
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    monkeypatch.setattr(coinc, 'find_event', lambda *a, **kw: [])
    P, _ = fast_reco(D, cond, None, None, rcfg=replace(PF.RecoConfig(), ignore_channels=ignore))
    live = np.array([0, 1, 4, 5]) if ignore else np.arange(8)
    ref = numpy_pulses(W)
    np.testing.assert_array_equal(P['bar'] * 4 + P['sipm'], live)
    np.testing.assert_allclose(P['time'], ref['time'][live] + cond.timeoffset[live])
    np.testing.assert_allclose(P['letime'], ref['letime'][live] + cond.timeoffset[live])
    np.testing.assert_array_equal(np.asarray(P['wfidx'])[:, 0], live)


def test_numpy_chain_does_not_cap_at_eight(monkeypatch):
    cond = conditions()
    cond.status[:] = 0
    W = waveforms(tuple(range(4, 244, 24)), 252)
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    monkeypatch.setattr(coinc, 'find_event', lambda *a, **kw: [])
    outs, _ = chain.clusters(W, cond, None, None, chain.ChainConfig(), detail=True)
    assert len(outs[0]['pulses']['pe']) == 10


@pytest.mark.parametrize('jax', [False, True])
def test_empty_chain(monkeypatch, jax):
    if jax:
        pytest.importorskip('jax')
    cond = conditions()
    W = waveforms(channels=())
    W['adc'] = np.zeros((0, 0), dtype=np.int64)
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    outs, timers = chain.clusters(W, cond, None, None, chain.ChainConfig(jax=jax), detail=True)
    assert sum(len(o['cl_event']) for o in outs) == 0
    assert timers.get('pulses', 0) == 0


def test_cli_full_fallback_preserves_failed_flags():
    pytest.importorskip('jax')
    from crvfastsim.reco.cli import _run_jax
    W = waveforms(tuple(range(4, 244, 24)), 252)
    W['adc'][0, 4] = 4095
    ref = numpy_pulses(W)
    assert ref['failed'].any()
    out, _, nredo, _ = _run_jax({**W, 'mat': W['adc']}, PF.RecoConfig(), L=12, batch=4, rounds=3)
    assert nredo == 1
    for key in ('wf', 'seq', 'failed'):
        np.testing.assert_array_equal(out[key], ref[key])
    for key in ('pe', 'time'):
        np.testing.assert_allclose(out[key], ref[key])


@pytest.mark.parametrize('quiet', [False, True])
def test_mixed_long_and_short_recovery(quiet):
    pytest.importorskip('jax')
    from crvfastsim.reco import chain_jax
    W = waveforms((4, 20), 36, (0, 1, 2))
    W['n'][1] = 12
    if quiet:
        W['adc'][:] = 100
    cfg = PF.RecoConfig()
    rows = np.arange(3)
    ref = numpy_pulses(W)
    out = chain_jax._pulse_dispatch(W, rows, replace(cfg, max_pulses=3), 4, 12)
    wf, seq, pe, tm, nredo = chain_jax._pulse_collect(out, W, rows, cfg)
    assert nredo == 2
    np.testing.assert_array_equal(wf, ref['wf'])
    np.testing.assert_array_equal(seq, ref['seq'])
    np.testing.assert_allclose(pe, ref['pe'], rtol=1e-6)
    np.testing.assert_allclose(tm, ref['time'], atol=1e-4)


def test_conditions_real_coincidence_membership():
    pytest.importorskip('jax')
    from crvfastsim.reco import chain_jax, coinc_jax, synth
    from crvfastsim.reco.cli import _coinc_geom
    G = _coinc_geom()
    W = synth.chain_events(30, G, seed=11, noise_mean=0, shower_frac=0)
    W['toff'] = ((W['bar'] * 4 + W['sipm']) % 7 - 3) * 7.
    status = np.zeros(4 * len(G.sector), np.int64)
    status[::4] = 2
    ref = numpy_pulses(W)
    ref = PF.filter_channels(ref, W['bar'] * 4 + W['sipm'], status)
    row = ref['wf']
    ev = W['ev'][row]
    idx = np.arange(len(ev)) - np.searchsorted(ev, ev)
    tm = ref['time'] + W['toff'][row]
    expected = set()
    for e in np.unique(ev):
        sel = ev == e
        p = dict(idx=idx[sel], bar=W['bar'][row][sel], sipm=W['sipm'][row][sel],
                 pe=ref['pe'][sel], time=tm[sel], seq=ref['seq'][sel])
        expected.update((int(e), r['sectortype'], tuple(sorted(r['pulses'])))
                        for r in coinc.find_event(p, G, status))
    assert expected
    for chunks in (1, 3):
        out, _ = chain_jax.run(W, G, status, n_chunks=chunks, batch=512, keep_pulses=True)
        assert set().union(*(coinc_jax.cluster_keys(o) for o in out)) == expected


@pytest.mark.parametrize('jax', [False, True])
@pytest.mark.parametrize('ignored', [False, True])
def test_no_pulses_retains_event_population(jax, ignored):
    if jax:
        pytest.importorskip('jax')
    from crvfastsim.reco.cli import _coinc_geom
    from crvfastsim.reco import chain_jax
    G = _coinc_geom()
    W = waveforms(channels=(0, 1))
    if not ignored:
        W['adc'][:] = 100
    status = np.full(4 * len(G.sector), 2 if ignored else 0, dtype=np.int64)
    if jax:
        outs, timers = chain_jax.run(W, G, status, batch=4, keep_pulses=True)
        assert timers['events'] == 1
        assert sum(len(o['pulses']['pe']) for o in outs) == 0
    else:
        P = numpy_pulses(W)
        P = PF.filter_channels(P, W['chan'], status)
        assert P['wf'].dtype.kind in 'iu'
        assert len(P['pe']) == 0


@pytest.mark.parametrize('empty', [False, True])
def test_table_reco_full_length_and_empty(monkeypatch, empty):
    from valhist.fasttables import fast_reco
    cond = conditions()
    W = waveforms(tuple(range(4, 244, 24)), 252, () if empty else (0,))
    D = {k: W[k] for k in ('ev', 'bar', 'sipm', 'adc')}
    D['starttdc'] = W['start']
    if empty:
        D['adc'] = np.zeros((0, 0), dtype=np.int64)
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    monkeypatch.setattr(coinc, 'find_event', lambda *a, **kw: [])
    P, _ = fast_reco(D, cond, None, None)
    assert len(P['pe']) == (0 if empty else 10)
