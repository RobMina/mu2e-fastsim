"""Configuration wiring preserves the default chain and full-length pulse recovery."""
from dataclasses import replace

import numpy as np
import pytest

from crvfastsim import chain
from crvfastsim.reco import coinc, pulsefit as PF, waveform
from test_chain_reco_contract import conditions, waveforms


def phase_inputs():
    cond = conditions()
    cond.roc[:] = [1, 1, 1, 1, 2, 2, 2, 2]
    cond.feb[:] = [0, 0, 1, 1, 0, 0, 1, 1]
    labels = np.array([2, 1, 1, 1, 2, 1]) * chain.CHAN_MULT + np.array([0, 4, 1, 0, 7, 3])
    return cond, labels


def legacy_phase(labels, cond, rng):
    ev, ch = labels // chain.CHAN_MULT, labels % chain.CHAN_MULT
    keys = ev * 1_000_000 + cond.roc[ch] * 1000 + cond.feb[ch]
    unique, inv = np.unique(keys, return_inverse=True)
    return rng.random(len(unique))[inv.ravel()] * chain.PERIOD


def test_default_phase_is_seed_identical():
    cond, labels = phase_inputs()
    actual_rng, expected_rng = np.random.default_rng(941), np.random.default_rng(941)
    np.testing.assert_array_equal(chain.tdc0_per_feb(labels, cond, actual_rng),
                                  legacy_phase(labels, cond, expected_rng))
    np.testing.assert_array_equal(actual_rng.random(8), expected_rng.random(8))
    assert chain.ChainConfig().feb_phase_width == chain.PERIOD
    assert chain.ChainConfig().pulse_reco is None


@pytest.mark.parametrize('width', [0., 3.25, chain.PERIOD])
def test_phase_width_scales_existing_draws(width):
    cond, labels = phase_inputs()
    actual_rng, expected_rng = np.random.default_rng(941), np.random.default_rng(941)
    expected = legacy_phase(labels, cond, expected_rng) / chain.PERIOD * width
    actual = chain.tdc0_per_feb(labels, cond, actual_rng, width=width)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)
    assert np.all((actual >= 0) & (actual <= width))
    assert actual[2] == actual[3]
    np.testing.assert_array_equal(actual_rng.random(8), expected_rng.random(8))


@pytest.mark.parametrize('width', [-1., chain.PERIOD + .01, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('empty', [False, True])
def test_invalid_phase_width_fails_before_draw(width, empty):
    cond, labels = phase_inputs()
    rng = np.random.default_rng(941)
    with pytest.raises(ValueError, match='phase width'):
        chain.tdc0_per_feb(labels[:0] if empty else labels, cond, rng, width=width)
    np.testing.assert_array_equal(rng.random(8), np.random.default_rng(941).random(8))


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'jax_bucketed'])
def test_digis_passes_phase_width(monkeypatch, backend):
    cond, labels = phase_inputs()
    Q = dict(chan=labels, time=np.full(len(labels), 100.), charge=np.ones(len(labels)))
    seen = []
    def synth(chan, time, charge, lut, cfg, **kwargs):
        seen.append(lut(chan))
        return {'chan': np.zeros(0, np.int64)}
    if backend == 'numpy':
        monkeypatch.setattr(waveform, 'synthesize', synth)
    else:
        pytest.importorskip('jax')
        from crvfastsim.reco import waveform_jax, coinc_jax, cli
        monkeypatch.setattr(coinc_jax, 'is_oneapi', lambda: False)
        monkeypatch.setattr(waveform_jax, 'synthesize', synth)
        monkeypatch.setattr(cli, '_synth_jax_bucketed',
                            lambda mod, q, lut, cfg, **kw: synth(q['row'], q['time'], q['charge'], lut, cfg, **kw))
    cfg = chain.ChainConfig(jax=backend != 'numpy', unbucketed_synth=backend == 'jax', feb_phase_width=2.5)
    chain.digis(Q, cond, cfg, np.random.default_rng(941))
    expected = chain.tdc0_per_feb(labels, cond, np.random.default_rng(941), width=2.5)
    np.testing.assert_array_equal(seen[0], expected)


@pytest.mark.parametrize('jax', [False, True])
@pytest.mark.parametrize('length', [12, 36])
@pytest.mark.parametrize('policy', ['default', 'custom', 'no_offsets'])
def test_pulse_config_reaches_reconstruction_and_fallback(monkeypatch, jax, length, policy):
    cond = conditions()
    W = waveforms((4,) if length == 12 else (4, 20), length, channels=(0, 2))
    supplied = None if policy == 'default' else replace(
        PF.RecoConfig(), min_adc_difference=55., default_beta=17., max_pulses=1,
        ignore_channels=False, time_offset_scale=2., use_time_offset_db=policy != 'no_offsets')
    expected_cfg = supplied if supplied is not None else PF.RecoConfig()
    ref = PF.reconstruct_full(*(W[k] for k in ('adc', 'n', 'start', 'ped', 'area', 'height')), expected_cfg)
    ref = PF.filter_channels(ref, W['chan'], cond.status, expected_cfg)
    expected_time = ref['time'].copy()
    if expected_cfg.use_time_offset_db:
        expected_time += cond.timeoffset[W['chan'][ref['wf']]] * expected_cfg.time_offset_scale
    full_seen, kernel_seen = [], []
    full = PF.reconstruct_full
    def full_spy(adc, n, start, ped, area, height, cfg):
        full_seen.append(cfg)
        assert adc.shape[1] == length
        return full(adc, n, start, ped, area, height, cfg)
    monkeypatch.setattr(PF, 'reconstruct_full', full_spy)
    monkeypatch.setattr(coinc, 'sector_geom', lambda *a: None)
    if jax:
        pytest.importorskip('jax')
        from crvfastsim.reco import chain_jax
        CJ = chain_jax.CJ
        dense = chain_jax.PJ.reconstruct_dense
        def dense_spy(*args, cfg):
            kernel_seen.append(cfg)
            return dense(*args, cfg=cfg)
        monkeypatch.setattr(chain_jax.PJ, 'reconstruct_dense', dense_spy)
        dispatch = chain_jax._pulse_dispatch
        monkeypatch.setattr(chain_jax, '_pulse_dispatch',
                            lambda W, rows, cfg, batch, L: dispatch(W, rows, cfg, 4, L))
        monkeypatch.setattr(CJ, 'type_params', lambda *a: None)
        monkeypatch.setattr(CJ, 'hit_tables', lambda *a: None)
        monkeypatch.setattr(CJ, 'find_async', lambda *a, **kw: {'cl_event': np.zeros(0, np.int64)})
        monkeypatch.setattr(CJ, 'finish', lambda x: x)
    else:
        monkeypatch.setattr(coinc, 'find_event', lambda *a, **kw: [])
    cfg = chain.ChainConfig(jax=jax, rounds=2, pulse_reco=supplied)
    outs, timers = chain.clusters(W, cond, None, None, cfg, detail=True)
    P = outs[0]['pulses']
    np.testing.assert_array_equal(P['bar'] * 4 + P['sipm'], W['chan'][ref['wf']])
    np.testing.assert_array_equal(P['seq'], ref['seq'])
    np.testing.assert_allclose(P['pe'], ref['pe'], rtol=1e-6)
    np.testing.assert_allclose(P['time'], expected_time, atol=1e-4)
    assert timers['pulses'] == len(ref['wf']) > 0
    if not jax or length > 12:
        assert full_seen
        assert all(c == expected_cfg for c in full_seen)
        if supplied is not None:
            assert all(c is supplied for c in full_seen)
    if jax:
        assert kernel_seen == [replace(expected_cfg, max_pulses=2)]
        if length > 12:
            assert timers['pulse_numpy_redo'] == len(W['ev'])
    assert expected_cfg.max_pulses == (8 if policy == 'default' else 1)
