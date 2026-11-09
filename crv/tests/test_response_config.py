"""Versioned configuration defaults, strict admission, override precedence and runtime wiring."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from crvfastsim import chain, response_config as RC
from crvfastsim.response.photons import ResponseConfig
from crvfastsim.response.sipm import SiPMConfig
from crvfastsim.reco import r3, pulsefit, coinc, waveform


def test_nominal_preset_equals_existing_dataclass_defaults():
    config = RC.parse()
    out = RC.build(config)
    for key, cls in [('response', ResponseConfig), ('sipm', SiPMConfig), ('waveform', waveform.WaveformConfig),
                     ('pulse_reco', pulsefit.RecoConfig), ('finder', coinc.FinderConfig)]:
        assert asdict(getattr(out, key)) == asdict(cls())
    assert out.window == (0.0, 1e5) and out.feb_phase_width == 12.5 and out.noise
    assert set(RC.parse()['finder']['cuts']) == {'EX', 'T1', 'T2', *[f'M{i}' for i in range(1, 9)]}


def test_configuration_roundtrip_file_and_inline(tmp_path):
    config = {'schema': 1, 'sipm': {'overvoltage': 2.9}, 'response': {'atten_ratio': {'T10': 0.6}},
              'pulse_reco': {'min_adc_difference': 45.0}, 'finder': {'cuts': {'T1': {'pe_threshold': 10.0}}}}
    p = tmp_path / 'settings.json'
    p.write_text(json.dumps(config))
    assert RC.parse(p) == RC.parse(config) == RC.parse(p.read_text())
    out = RC.build(RC.parse(p))
    assert out.sipm.overvoltage == 2.9
    assert out.response.atten_ratio['T10'] == 0.6
    assert out.pulse_reco.min_adc_difference == 45
    assert out.finder.cuts['T1'].pe_threshold == 10
    assert RC.parse(RC.canonical(RC.parse(p))) == RC.parse(p)


def test_knob_overrides_match_existing_chain():
    knobs = dict(photon_yield_scale=1.1, time_constant=14, cross_talk=0.03, min_voltage=0.02,
                 wf_noise=0.0005, pe_stretch=1.05, background_scale=0.8, yield_TAG=30000,
                 speed_EX=1.03, atten_T1=0.6)
    cfg = RC.apply_knobs(RC.parse({'schema': 1, 'sipm': {'time_constant': 20}}), knobs)
    old, new = chain.apply_knobs(chain.ChainConfig(), knobs), RC.build(cfg)
    for key in ('response', 'sipm', 'waveform'):
        assert asdict(getattr(old, key)) == asdict(getattr(new, key))
    assert old.background_scale == new.background_scale
    assert cfg['sipm']['time_constant'] == 14
    assert set(knobs) <= set(chain.KNOBS)


@pytest.mark.parametrize('patch', [
    {'schema': 2}, {'schema': True}, {'sipm': {'p1': 0.5}}, {'schema': 1, 'bogus': 0},
    {'schema': 1, 'sipm': {'p1': '0.5'}}, {'schema': 1, 'sipm': {'p1': True}},
    {'schema': 1, 'sipm': {'capacitance': float('inf')}}, {'schema': 1, 'sipm': {'p2': 0}},
    {'schema': 1, 'sipm': {'cross_talk': 1.1}}, {'schema': 1, 'sipm': {'p1': .0001, 'cross_talk': .9}},
    {'schema': 1, 'sipm': {'thermal_rate': .01}}, {'schema': 1, 'sipm': {'trap0_prob': .01}},
    {'schema': 1, 'sipm': {'trap1_prob': .01}}, {'schema': 1, 'sipm': {'n_pixels_x': 41}},
    {'schema': 1, 'sipm': {'inactive': []}}, {'schema': 1, 'waveform': {'samples_zs': 24}},
    {'schema': 1, 'waveform': {'single_pe_precision': .1}}, {'schema': 1, 'timing': {'window': [0, 2e5]}},
    {'schema': 1, 'timing': {'feb_phase_width': 13}}, {'schema': 1, 'timing': {'phase_scope': 'unknown'}},
    {'schema': 1, 'pulse_reco': {'default_beta': 50}}, {'schema': 1, 'pulse_reco': {'samples_before': 3.5}},
    {'schema': 1, 'pulse_reco': {'ignore_channels': False}}, {'schema': 1, 'pulse_reco': {'lm_iterations': 1}},
    {'schema': 1, 'finder': {'cuts': {'EX': {'coincidence_layers': 5}}}},
    {'schema': 1, 'finder': {'cuts': {'EX': {'min_slope': 8}}}},
    {'schema': 1, 'response': {'sectors': {'EX': {'reflector': 1}}}},
    {'schema': 1, 'response': {'speed_ratio': {'unknown': 1}}},
    {'schema': 1, 'assets': {'unknown': '/x'}}, {'schema': 1, 'scenario': 'run2_v10'},
])
def test_invalid_or_unsupported_config_fails(patch):
    with pytest.raises(ValueError):
        RC.parse(patch)


@pytest.mark.parametrize('text', ['{"schema":1,"schema":1}', '{"schema":1,"sipm":{"p1":NaN}}'])
def test_nonstandard_json_fails(text):
    with pytest.raises(ValueError):
        RC.parse(text)


@pytest.mark.parametrize('knobs', [{'overvoltage': 3}, {'cross_talk': -1}, {'speed_EX': 'inf'},
                                   {'atten_T1': 'nan'}, {'time_constant': True}])
def test_invalid_knob_overrides_fail(knobs):
    with pytest.raises(ValueError):
        RC.apply_knobs(RC.parse(), knobs)


def test_identity_changes_with_effective_configuration():
    base = RC.parse()
    alt = RC.parse({'schema': 1, 'sipm': {'overvoltage': 2.5}})
    assert RC.record(base)['sha256'] != RC.record(alt)['sha256']
    assert RC.record(base)['sha256'] == RC.record(RC.parse(RC.canonical(base)))['sha256']
    assert RC.parse() == base


@pytest.mark.parametrize('patch', [
    {'finder': {'compensate_status_bits': [0.0, 1.0, 2.0]}},
    {'finder': {'compensate_status_bits': [False, True, 2]}},
    {'sipm': {'inactive': [[float(x), float(y)] for x in range(18, 22) for y in range(18, 22)]}},
    {'sipm': {'overvoltage': 1e-20, 'cross_talk': 0}},
    {'sipm': {'trap0_lifetime': 6}}, {'pulse_reco': {'le_time_factor': .99}},
])
def test_review_regressions_rejected(patch):
    for kit in (False, True):
        with pytest.raises(ValueError):
            RC.parse({'schema': 1, **patch}, kit=kit)


def test_kit_restricts_unstaged_assets():
    for patch in ({'assets': {'calib': '/not/staged.csv'}}, {'assets': {'lookup_dir': '/not/staged'}},
                  {'waveform': {'single_pe_file': '/not/staged.txt'}}):
        with pytest.raises(ValueError, match='asset'):
            RC.parse({'schema': 1, **patch}, kit=True)


def fake_r3(monkeypatch):
    from crvfastsim.response import photons, sipm
    from crvfastsim.objective import crvresponse as R
    seen = {}
    cond = chain.Conditions(*(np.ones(4) for _ in range(8)))
    monkeypatch.setattr(r3.geometry, 'find_counters', lambda p: __file__)
    monkeypatch.setattr(r3.geometry, 'from_offline_config', lambda p: None)
    monkeypatch.setattr(r3.geometry, 'parse_simpleconfig', lambda p: None)
    monkeypatch.setattr(chain.Conditions, 'from_csv', lambda *a: cond)
    monkeypatch.setattr(photons, 'load_tables', lambda *a, **kw:
                        {n: SimpleNamespace(half_length=L, version=(6, 0)) for n, L in
                         [('6000_0', 3000), ('2370_1', 1185), ('3200_0', 1600)]})
    monkeypatch.setattr(sipm, 'photon_map', lambda *a: np.ones((40, 40)))
    monkeypatch.setattr(RC, 'asset_manifest', lambda *a, **kw: {'fake': {'sha256': 'abc'}})
    monkeypatch.setattr(r3, 'load_cache', lambda *a: [dict(path='input', events=np.array([1]), ngen=2)])
    def batch(c, a, b, cond, cfg, geom, ccfg, tabs, pmap, rng, key, timers, obj, **kwargs):
        seen.update(cfg=cfg, cond=cond)
        R.add_empty_events(obj['state'], 1, 1e5)
        return 1, key
    monkeypatch.setattr(r3, 'run_batch', batch)
    return seen


def test_r3_file_values_overrides_and_resolved_record(monkeypatch, tmp_path):
    seen = fake_r3(monkeypatch)
    config = {'schema': 1, 'sipm': {'time_constant': 21, 'overvoltage': 2.5},
              'timing': {'feb_phase_width': 6}, 'assets': {'calib': 'in-file.csv'}, 'noise': False}
    out = r3.run(['input'], target=1, objective=True, response_config=config, v05=True,
                 knobs={'time_constant': 15}, emit=str(tmp_path / 'out.json'))
    assert seen['cfg'].sipm.time_constant == 15 and seen['cfg'].sipm.overvoltage == 2.5
    assert seen['cfg'].feb_phase_width == 6 and not seen['cfg'].noise
    assert np.all(seen['cond'].roc == 0) and np.all(seen['cond'].timeoffset == 0)
    resolved = out['resolved_config']
    assert resolved['values']['assets']['calib'] == 'in-file.csv'
    assert resolved['values']['timing']['phase_scope'] == 'per_event'
    assert resolved['assets']['fake']['sha256'] == 'abc'
    assert out['events_with_steps'] == 1 and out['events_without_steps'] == 1
    assert json.loads((tmp_path / 'out.json').read_text())['resolved_config'] == resolved


def test_r3_nominal_config_preserves_old_configuration(monkeypatch):
    seen = fake_r3(monkeypatch)
    out = r3.run(['input'], 'cal.csv', target=1, objective=True)
    old = chain.apply_knobs(chain.ChainConfig(), {})
    for key in ('response', 'sipm', 'waveform'):
        assert asdict(getattr(seen['cfg'], key)) == asdict(getattr(old, key))
    assert out['noise'] and np.all(seen['cond'].roc == 1)


def test_r3_invalid_config_refused_before_io(monkeypatch):
    monkeypatch.setattr(r3.geometry, 'find_counters', lambda *a: pytest.fail('IO before config validation'))
    with pytest.raises(ValueError):
        r3.run(['input'], 'cal.csv', response_config={'schema': 1, 'sipm': {'thermal_rate': 1}})


@pytest.mark.parametrize('weights', [np.zeros((40, 40)), np.ones((41, 41)), -np.ones((40, 40)),
                                    np.full((40, 40), np.nan)])
def test_photon_map_capability_enforced(weights):
    with pytest.raises(ValueError, match='photon_map'):
        RC.validate_loaded_assets(RC.parse(), {}, weights)


def test_lookup_geometry_capability_enforced():
    tables = {'6000_0': SimpleNamespace(half_length=1, version=(6, 0))}
    with pytest.raises(ValueError, match='lookup table'):
        RC.validate_loaded_assets(RC.parse(), tables, np.ones((40, 40)))


def test_cli_passes_configuration_without_masking_file_noise(monkeypatch):
    from crvfastsim.reco import cli
    monkeypatch.setattr(r3, 'run', lambda *a, **kw: kw)
    args = SimpleNamespace(merge=False, inputs=['input'], calib=None, deviation=None, target=1, jax=False,
                           seed=1, batch_events=1, emit=None, objective=True, no_noise=False, time_limit=None,
                           v05=False, knob=[], background=None, time_shift=None, response_config='config.json')
    out = cli.cmd_r3(args)
    assert out['response_config'] == 'config.json' and out['noise'] is None
