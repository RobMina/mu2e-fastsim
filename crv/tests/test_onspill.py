"""Synthetic onspill timing contracts; no files, response assets or device execution."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import chain as CH, onspill as O
from crvfastsim.reco import coinc, pulsefit as PF


def timing(n=1, **kw):
    args = dict(keys=np.column_stack((np.ones(n, int), np.zeros(n, int), np.arange(n))),
                mc_pbtime=np.zeros(n), reco_pbtime=np.zeros(n), event_length=np.full(n, 1675.))
    return O.EventTiming(**{**args, **kw})


def conditions():
    z = np.zeros(8)
    return CH.Conditions(z + 100, z + 400, z + 11, z + 7, z.astype(int), z.astype(int), z.astype(int), z,
                         readout_mask=np.ones(8, bool))


def config(**kw):
    return CH.ChainConfig(window=(200, 1500), sipm=replace(CH.SI.SiPMConfig(), thermal_rate=1e-4), **kw)


def steps(t0=(300.,), t1=None, keys=None):
    n = len(t0)
    keys = np.tile([1, 0, 0], (n, 1)) if keys is None else keys
    frame = pd.DataFrame(keys, columns=['run', 'subrun', 'event'])
    frame['t0'], frame['t1'] = t0, t0 if t1 is None else t1
    frame['stream'] = np.arange(n) % 2
    frame.index = np.arange(n) * 3 + 77  # photon IDs must be positions, not DataFrame indices
    return frame


def response(st, tm=None, cfg=None, cond=None, **kw):
    return O.charges(st, np.full(len(st), 13, dtype=int), SimpleNamespace(nbars=2),
                     conditions() if cond is None else cond, config() if cfg is None else cfg,
                     np.random.default_rng(123), timing=timing() if tm is None else tm,
                     tabs={}, pmap=np.ones((40, 40)), **kw)


def spy(monkeypatch, arrivals=None):
    seen = {}
    def photons(st, pdg, geom, cfg, tabs, deviation, rng, *key):
        seen.update(steps=st.copy(), pdg=pdg.copy(), photon_key=key)
        if arrivals is not None:
            return tuple(np.asarray(x) for x in arrivals)
        return np.arange(len(st)), np.zeros(len(st), int), st.t0.to_numpy()
    def sipm(ch, t, ph, pmap, cfg, rng, *key, **kw):
        seen.update(chan=ch.copy(), time=t.copy(), photon=ph.copy(), sipm_key=key, **kw)
        return dict(chan=ch[::-1], time=t[::-1], photon=ph[::-1], pe=np.ones(len(ch)), charge=np.ones(len(ch)))
    monkeypatch.setattr(O.PH, 'make_photons', photons)
    monkeypatch.setattr(O.SI, 'simulate', sipm)
    return seen, photons, sipm


def test_exact_prefilter_boundaries_crossings_original_intervals(monkeypatch):
    seen, _, _ = spy(monkeypatch)
    # Below lower-margin, above upper, excluded crossing; equality at either bound retained.
    a = np.array([99., 100., 1500., 1501., 1600., 1600., 1690., -5., 1795., 99.])
    b = np.array([99., 100., 1500., 1501., 50., 100., 1805., 110., 1795., 100.])
    st = steps(a, b)
    original = st.copy(deep=True)
    Q = response(st)
    keep = np.array([1, 2, 5, 6, 7, 8, 9])
    pd.testing.assert_frame_equal(seen['steps'], st.iloc[keep])
    pd.testing.assert_frame_equal(st, original)
    np.testing.assert_array_equal(np.sort(Q['photon']), keep)
    assert seen['steps'].iloc[3].t0 == 1690 and seen['steps'].iloc[3].t1 == 1805
    assert seen['t_start'] == 100 and seen['t_end'] == 1500


def test_arrival_fold_negative_and_prestart_retained(monkeypatch):
    seen, _, _ = spy(monkeypatch, ([0] * 7, [0] * 7, [-1805., -110., -105., 1395., 1400., 1585., 3280.]))
    response(steps(), tm=timing(mc_pbtime=np.array([110.])))
    np.testing.assert_array_equal(seen['time'], [0, 0, 5, 1505, 1510, 0, 0])
    # No adapter lower photon cut, nor an upper pre-fold cut; SiPM gets the folded arrivals.
    assert len(seen['photon']) == 7


@pytest.mark.parametrize('length', [1675., 1700.])
def test_event_lengths_do_not_change_window(monkeypatch, length):
    seen, _, _ = spy(monkeypatch)
    response(steps(), tm=timing(event_length=np.array([length])))
    assert (seen['t_start'], seen['t_end']) == (100, 1500)


def test_same_shift_invariant_and_true_reco_separation(monkeypatch):
    spy(monkeypatch)
    st = steps([290., 301.], [310., 340.])
    tm = timing(mc_pbtime=np.array([10.]), reco_pbtime=np.array([80.]))
    a = response(st, tm=tm)
    b = response(st.assign(t0=st.t0 + 1699, t1=st.t1 + 1699),
                 tm=replace(tm, mc_pbtime=tm.mc_pbtime - 1699, reco_pbtime=np.array([-333.])))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    W = O.reco_timing(dict(ev=np.array([0, 0])), tm)
    np.testing.assert_array_equal(W['event_t0'], [-80, -80])


def test_full_large_ids_run_collisions_and_shared_streams(monkeypatch):
    seen, _, _ = spy(monkeypatch)
    keys = np.array([[2**61, 2**42, 2**40], [2**61 + 1, 2**42, 2**40], [3, 4, 5]], np.int64)
    tm = timing(3, keys=keys)
    st = steps([330., 310., 320., 300.], keys=keys[[1, 0, 1, 0]])
    pdg = np.array([13, -13, 11, 22])
    Q = O.charges(st, pdg, SimpleNamespace(nbars=2), conditions(), config(), np.random.default_rng(3),
                  timing=tm, tabs={}, pmap=np.ones((40, 40)))
    np.testing.assert_array_equal(seen['pdg'], pdg)
    np.testing.assert_array_equal(seen['chan'], np.array([1, 0, 1, 0]) * CH.CHAN_MULT)
    np.testing.assert_array_equal(Q['chan'], np.array([0, 0, 1, 1]) * CH.CHAN_MULT)
    np.testing.assert_array_equal(Q['time'], [300, 310, 320, 330])
    np.testing.assert_array_equal(Q['photon'], [3, 1, 2, 0])
    np.testing.assert_array_equal(tm.keys[Q['chan'] // CH.CHAN_MULT], keys[[0, 0, 1, 1]])
    assert 2 * CH.CHAN_MULT in seen['thermal_chans']  # zero-step third event


def test_uint64_full_ids_and_signed_lookup():
    keys = np.array([[2**64 - 1, 4, 5], [1, 0, 0]], dtype=np.uint64)
    tm = timing(2, keys=keys)
    np.testing.assert_array_equal(tm.lookup(keys[::-1]), [1, 0])
    np.testing.assert_array_equal(tm.lookup(np.array([[1, 0, 0]], np.int64)), [1])
    with pytest.raises(ValueError, match='nonnegative'):
        tm.lookup(np.array([[-1, 4, 5]], np.int64))
    with pytest.raises(ValueError, match='absent'):
        timing().lookup(keys[:1])


@pytest.mark.parametrize('t0,n_events', [((300.,), 2), ((), 2), ((), 0)])
@pytest.mark.parametrize('thermal_rate', [0., 1e-4])
def test_missing_readout_mask_rejected_before_response(monkeypatch, t0, n_events, thermal_rate):
    seen, _, _ = spy(monkeypatch)
    cond = replace(conditions(), readout_mask=None)
    cfg = config()
    cfg = replace(cfg, sipm=replace(cfg.sipm, thermal_rate=thermal_rate))
    with pytest.raises(ValueError, match='require cond.readout_mask'):
        response(steps(t0), tm=timing(n_events), cond=cond, cfg=cfg)
    assert seen == {}
    assert cond.readout_mask is None
    np.testing.assert_array_equal(cond.status, np.zeros(8, int))


@pytest.mark.parametrize('t0', [(300.,), ()])
@pytest.mark.parametrize('supplied_mask', [False, True])
def test_run_physical_sides_filter_optical_and_thermal_including_empty_event(monkeypatch, t0, supplied_mask):
    seen, _, _ = spy(monkeypatch, ([0] * 8, range(8), [300.] * 8))
    geom = SimpleNamespace(nbars=2, sector_names=['A', 'B'], sector=np.array([0, 1]))
    counters = {'crs.sectorNames': ['A', 'B'], 'crs.sipmsAtSide0A': True, 'crs.sipmsAtSide1A': False,
                'crs.sipmsAtSide0B': False, 'crs.sipmsAtSide1B': True}
    mask = np.array([True, False, True, False, False, True, False, True])
    cond = replace(conditions(), readout_mask=mask.copy() if supplied_mask else None)
    cond.status[:] = [0, 0, 2, 0, 0, 0, 0, 4]
    original_status = cond.status.copy()
    downstream = []
    def check_conditions(current):
        downstream.append(current)
        np.testing.assert_array_equal(current.readout_mask, mask)
        np.testing.assert_array_equal(current.status, original_status)
        assert (current is cond) == supplied_mask
    def digis(Q, current, *args, **kw):
        check_conditions(current)
        return {}
    def reco_waveforms(D, current):
        check_conditions(current)
        return dict(ev=np.empty(0, int))
    def clusters(W, current, *args, **kw):
        check_conditions(current)
        return [], kw['timers']
    monkeypatch.setattr(CH, 'digis', digis)
    monkeypatch.setattr(CH, 'reco_waveforms', reco_waveforms)
    monkeypatch.setattr(CH, 'clusters', clusters)
    st, tm = steps(t0), timing(2)
    out = O.run(st, np.full(len(st), 13, int), geom, cond, config(), np.random.default_rng(123),
                timing=tm, counters_cfg=counters, tabs={}, pmap=np.ones((40, 40)))
    live = np.array([0, 2, 5])
    np.testing.assert_array_equal(seen['chan'], live if len(st) else np.empty(0, int))
    np.testing.assert_array_equal(out['Q']['chan'], seen['chan'])
    np.testing.assert_array_equal(seen['thermal_chans'], np.concatenate((live, live + CH.CHAN_MULT)))
    assert ('steps' in seen) == bool(len(st))
    assert len(downstream) == 3 and all(current is downstream[0] for current in downstream)
    np.testing.assert_array_equal(out['event_keys'], tm.keys)
    np.testing.assert_array_equal(cond.status, original_status)
    if supplied_mask:
        np.testing.assert_array_equal(cond.readout_mask, mask)
    else:
        assert cond.readout_mask is None


@pytest.mark.parametrize('counters', [{}, {'crs.sipmsAtSide0A': True},
                                     {'crs.sipmsAtSide0A': True, 'crs.sipmsAtSide1A': 1}])
def test_run_missing_mask_never_guesses_side_presence(monkeypatch, counters):
    seen, _, _ = spy(monkeypatch)
    geom = SimpleNamespace(nbars=2, sector_names=['A'], sector=np.zeros(2, int))
    cond = replace(conditions(), readout_mask=None)
    with pytest.raises(ValueError, match='readout counters'):
        O.run(steps([]), np.empty(0, int), geom, cond, config(), np.random.default_rng(123),
              timing=timing(0), counters_cfg=counters)
    assert seen == {} and cond.readout_mask is None


def test_zero_step_thermal_readout_and_status(monkeypatch):
    seen, _, _ = spy(monkeypatch)
    cond = conditions()
    cond.status[:] = [0, 1, 2, 4, 0, 0, 0, 0]
    cond.readout_mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    Q = response(steps([]), tm=timing(2), cond=cond)
    assert 'steps' not in seen
    np.testing.assert_array_equal(seen['thermal_chans'], [0, 2, CH.CHAN_MULT, CH.CHAN_MULT + 2])
    assert len(seen['time']) == len(Q['time']) == 0
    assert Q['chan'].dtype.kind == Q['photon'].dtype.kind == 'i'


def test_photon_status_and_readout_mask(monkeypatch):
    seen, _, _ = spy(monkeypatch, ([0] * 8, range(8), [300] * 8))
    cond = conditions()
    cond.status[:] = np.arange(8)
    cond.readout_mask = np.array([0, 1, 1, 1, 1, 1, 1, 1], bool)
    response(steps(), cond=cond)
    np.testing.assert_array_equal(seen['chan'], [2])  # bit 1 is a reco-only veto


def test_real_numpy_thermal_zero_steps(monkeypatch):
    monkeypatch.setattr(O.PH, 'make_photons', lambda *a: pytest.fail('optics called for no steps'))
    cond = replace(conditions(), readout_mask=np.array([True, False, True, False, False, True, False, True]))
    cond.status[[2, 7]] = [2, 4]
    Q = response(steps([]), tm=timing(32), cond=cond)
    assert len(Q['time']) > 0
    assert np.all(np.isin(Q['chan'] % CH.CHAN_MULT, [0, 2, 5]))
    assert np.all(Q['photon'] == -1)
    assert np.all((Q['time'] >= 100) & (Q['time'] <= 1500))
    np.testing.assert_array_equal(np.lexsort((Q['time'], Q['chan'])), np.arange(len(Q['time'])))


def test_empty_population_no_assets_or_kernels(monkeypatch):
    for obj, name in ((O.PH, 'make_photons'), (O.SI, 'simulate'), (O.SI, 'photon_map')):
        monkeypatch.setattr(obj, name, lambda *a, **kw: pytest.fail('empty kernel or asset call'))
    Q = response(steps([]), tm=timing(0))
    assert all(len(v) == 0 for v in Q.values())
    with pytest.raises(ValueError, match='absent'):
        response(steps(), tm=timing(0))


def test_jax_forwarding_without_device(monkeypatch):
    from crvfastsim import response as package
    seen, photons, sipm = spy(monkeypatch)
    fakejax = SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda n: ('default', n), split=lambda key: ('optics', 'sipm')))
    monkeypatch.setitem(sys.modules, 'jax', fakejax)
    monkeypatch.setattr(package, 'photons_jax', SimpleNamespace(make_photons=photons), raising=False)
    monkeypatch.setattr(package, 'sipm_jax', SimpleNamespace(simulate=sipm), raising=False)
    response(steps(), tm=timing(2, mc_pbtime=np.array([1695., 0.])), cfg=config(jax=True))
    assert seen['photon_key'] == ('optics',) and seen['sipm_key'] == ('sipm',)
    np.testing.assert_array_equal(seen['time'], [300])
    assert seen['t_start'] == 100 and seen['t_end'] == 1500
    assert len(seen['thermal_chans']) == 16
    seen.clear()
    response(steps([]), tm=timing(2), cfg=config(jax=True))
    assert 'steps' not in seen and len(seen['thermal_chans']) == 16


@pytest.mark.parametrize('field,value', [
    ('keys', np.array([[1., 0., 0.]])), ('keys', np.array([[1, 0]])),
    ('keys', np.array([[1, 0, 0], [1, 0, 0]])), ('mc_pbtime', [np.nan]),
    ('reco_pbtime', [np.inf]), ('event_length', [np.inf]), ('event_length', [0]),
    ('event_length', [-1]), ('mc_pbtime', []), ('reco_pbtime', [[0]]),
    ('period', 0), ('period', np.inf), ('period', [1695])])
def test_invalid_timing(field, value):
    with pytest.raises(ValueError):
        timing(**{field: value})


@pytest.mark.parametrize('change', [dict(window=(0, 1e5)), dict(window=(200, 1800)), dict(window=(200, 199)),
                                      dict(window=(200, np.nan)), dict(fused_response=True), dict(background=object()),
                                      dict(feb_phase_width=np.nan)])
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        response(steps([]), cfg=replace(config(), **change))


@pytest.mark.parametrize('field,value', [('thermal_rate', -1), ('thermal_rate', np.nan), ('trap0_prob', .1),
                                        ('trap1_prob', .1), ('cross_talk', 1.1), ('p1', 2), ('p2', 0)])
def test_invalid_sipm(field, value):
    cfg = config()
    with pytest.raises(ValueError):
        response(steps([]), cfg=replace(cfg, sipm=replace(cfg.sipm, **{field: value})))


def test_event_window_not_silently_shortened_and_step_validation():
    with pytest.raises(ValueError, match='event_length'):
        response(steps([]), tm=timing(event_length=np.array([1499.])))
    with pytest.raises(ValueError, match='absent'):
        response(steps(keys=np.array([[2, 0, 0]])))
    with pytest.raises(ValueError, match='integer'):
        response(steps(keys=np.array([[1., 0., 0.]])))
    with pytest.raises(ValueError, match='infinite'):
        response(steps([np.inf]))
    with pytest.raises(ValueError, match='pdg'):
        O.charges(steps(), np.array([], int), SimpleNamespace(nbars=2), conditions(), config(),
                  np.random.default_rng(3), timing=timing())
    with pytest.raises(ValueError, match='dense'):
        O.reco_timing(dict(ev=np.array([1])), timing())


def test_stock_nan_endpoint_skip_preserves_original_row_ids(monkeypatch):
    seen, _, _ = spy(monkeypatch)
    timers = {}
    Q = response(steps([np.nan, 300., 400.], [300., np.nan, 400.]), timers=timers)
    np.testing.assert_array_equal(Q['photon'], [2])
    assert len(seen['steps']) == 1
    assert timers['nan_time_steps_skipped'] == 2


def test_default_contract_stays_offspill():
    from crvfastsim.response.contract import validate_sipm
    assert CH.ChainConfig().window == (0., 1e5)
    with pytest.raises(ValueError, match='thermal_rate'):
        validate_sipm(config().sipm)


def fake_chain_jax(monkeypatch):
    """Load host orchestration with fake device dependencies: never import or execute JAX."""
    from crvfastsim import reco
    cj = SimpleNamespace(type_params=lambda *a: None, hit_tables=lambda *a: None,
                         find_async=lambda *a, **kw: {}, finish=lambda x: x)
    for name, obj in [('coinc_jax', cj), ('pulsefit_jax', SimpleNamespace())]:
        monkeypatch.setattr(reco, name, obj, raising=False)
        monkeypatch.setitem(sys.modules, 'crvfastsim.reco.' + name, obj)
    name = 'crvfastsim.reco._onspill_chain_jax_test'
    spec = importlib.util.spec_from_file_location(name, Path(O.__file__).parent / 'reco' / 'chain_jax.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '_pulse_dispatch', lambda *a: None)
    monkeypatch.setattr(module, '_pulse_collect', lambda outs, W, rows, cfg:
                        (rows, np.zeros(len(rows), int), np.ones(len(rows)), np.full(len(rows), 300.), 0))
    monkeypatch.setattr(reco, 'chain_jax', module, raising=False)
    return module


@pytest.mark.parametrize('jax', [False, True])
@pytest.mark.parametrize('enabled,scale', [(False, 9.), (True, 0.), (True, 2.5)])
@pytest.mark.parametrize('correction', [False, True])
def test_bunch_correction_independent_of_db(monkeypatch, jax, enabled, scale, correction):
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    monkeypatch.setattr(coinc, 'find_event', lambda *a: [])
    if jax:
        fake_chain_jax(monkeypatch)
    else:
        monkeypatch.setattr(PF, 'reconstruct_full', lambda *a:
                            dict(wf=np.arange(2), seq=np.zeros(2, int), pe=np.ones(2), time=np.full(2, 300.)))
    W = dict(ev=np.arange(2), bar=np.zeros(2, int), sipm=np.zeros(2, int),
             adc=np.zeros((2, 12)), n=np.full(2, 12), start=np.zeros(2), ped=np.zeros(2),
             area=np.ones(2), height=np.ones(2), toff=np.array([7., -3.]))
    tm = timing(2, mc_pbtime=np.array([50., 80.]), reco_pbtime=np.array([100., 200.]))
    if correction:
        W = O.reco_timing(W, tm)
    rcfg = replace(PF.RecoConfig(), use_time_offset_db=enabled, time_offset_scale=scale)
    outs, _ = CH.clusters(W, conditions(), None, None, config(jax=jax, pulse_reco=rcfg), detail=True)
    expected = np.full(2, 300.) + (W['toff'] * scale if enabled else 0) - (tm.reco_pbtime if correction else 0)
    np.testing.assert_array_equal(outs[0]['pulses']['time'], expected)


def test_run_local_digis_external_mapping(monkeypatch):
    seen, _, _ = spy(monkeypatch)
    tm = timing(mc_pbtime=np.array([20.]), reco_pbtime=np.array([70.]))
    def digis(Q, cond, cfg, rng, **kw):
        np.testing.assert_array_equal(Q['time'], [320])
        assert cfg.window == (200, 1500) and cfg.feb_phase_width == 12.5
        return dict(chan=Q['chan'], starttdc=np.array([16]), adc=np.ones((1, 12), int))
    monkeypatch.setattr(CH, 'digis', digis)
    def clusters(W, *args, **kw):
        np.testing.assert_array_equal(W['event_t0'], [-70])
        np.testing.assert_array_equal(W['toff'], [7])
        return [], kw['timers']
    monkeypatch.setattr(CH, 'clusters', clusters)
    out = O.run(steps(), np.array([13]), SimpleNamespace(nbars=2), conditions(), config(),
                np.random.default_rng(1), timing=tm, counters_cfg={}, tabs={}, pmap=np.ones((40, 40)))
    assert set(out) == {'Q', 'D', 'W', 'clusters', 'timers', 'event_keys'}
    np.testing.assert_array_equal(out['event_keys'], tm.keys)
    assert out['timers']['charges'] == 1


def test_mixed_signed_unsigned_dataframe_keys_never_float(monkeypatch):
    seen, _, _ = spy(monkeypatch)
    keys = np.array([[2**64 - 1, 3, 2**54 + 1]], dtype=np.uint64)
    st = steps(keys=keys)
    st['subrun'] = st.subrun.astype(np.int64)
    st['event'] = st.event.astype(np.int64)
    assert st[['run', 'subrun', 'event']].to_numpy().dtype.kind == 'f'
    Q = response(st, tm=timing(keys=keys))
    np.testing.assert_array_equal(Q['chan'], [0])
    assert seen['steps'].run.to_numpy()[0] == keys[0, 0]


def test_run_empty_population(monkeypatch):
    from crvfastsim.reco import waveform
    monkeypatch.setattr(waveform, 'single_pe_template', lambda *a: np.ones(200))
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    geom = SimpleNamespace(nbars=2, sector_names=['A'], sector=np.zeros(2, int))
    counters = {'crs.sipmsAtSide0A': True, 'crs.sipmsAtSide1A': False}
    cond = replace(conditions(), readout_mask=None)
    out = O.run(steps([]), np.empty(0, int), geom, cond, config(),
                np.random.default_rng(1), timing=timing(0), counters_cfg=counters)
    assert cond.readout_mask is None
    assert len(out['Q']['chan']) == len(out['D']['chan']) == len(out['W']['ev']) == len(out['event_keys']) == 0
    assert sum(len(c['cl_event']) for c in out['clusters']) == 0
