"""Synthetic-only tests: selection semantics, not measured or C++ emulator closure."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import jax_kalseed as selection
import replay

PATHS = ('tpr_TrkDe_80m70p_D0200', 'tpr_TrkDe_80m70p')
LABELS = ('TprTrkDe80m70pD0200KSFilter', 'TprTrkDe80m70pKSFilter')


@contextmanager
def enable_x64(enabled=True):
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', enabled)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.fixture(autouse=True)
def x64_context():
    with enable_x64():
        yield


def synthetic_config():
    """Synthetic expanded process with the two embedded CE selection tables' values."""
    return replay.parse_expanded('''
physics: {
  trigger_paths: [ "100:tpr_TrkDe_80m70p_D0200", "110:tpr_TrkDe_80m70p" ]
  tpr_TrkDe_80m70p_D0200: [ "TrigKSFDe", "TprTrkDe80m70pD0200KSFilter" ]
  tpr_TrkDe_80m70p: [ "TrigKSFDe", "TprTrkDe80m70pKSFilter" ]
  producers: { TrigKSFDe: { module_label: "TrigKSFDe" module_type: "LoopHelixFit" } }
  filters: {
    TprTrkDe80m70pD0200KSFilter: {
      module_type: "KalSeedFilter" module_label: "TprTrkDe80m70pD0200KSFilter"
      kalSeedCollections: [ "TrigKSFDe" ]
      KalSeedCuts: [
        { fitdirection: "downstream" fitparticle: 11 requireCaloCluster: false
          maxChi2DOF: 20 maxD0: 200 maxMomErr: 10 maxMomentum: 500 maxTanDip: 100
          minD0: -200 minFitCons: -1 minMomentum: 80 minNStrawHits: 15 minTanDip: 0 },
        { fitdirection: "downstream" fitparticle: -11 requireCaloCluster: false
          maxChi2DOF: 20 maxD0: 200 maxMomErr: 10 maxMomentum: 500 maxTanDip: 100
          minD0: -200 minFitCons: -1 minMomentum: 70 minNStrawHits: 15 minTanDip: 0 }
      ]
    }
    TprTrkDe80m70pKSFilter: {
      module_type: "KalSeedFilter" module_label: "TprTrkDe80m70pKSFilter"
      kalSeedCollections: [ "TrigKSFDe" ]
      KalSeedCuts: [
        { fitdirection: "downstream" fitparticle: 11 requireCaloCluster: false
          maxChi2DOF: 20 maxD0: 600 maxMomErr: 10 maxMomentum: 500 maxTanDip: 1000
          minD0: -600 minFitCons: -1 minMomentum: 80 minNStrawHits: 15 minTanDip: 0 },
        { fitdirection: "downstream" fitparticle: -11 requireCaloCluster: false
          maxChi2DOF: 20 maxD0: 600 maxMomErr: 10 maxMomentum: 500 maxTanDip: 1000
          minD0: -600 minFitCons: -1 minMomentum: 70 minNStrawHits: 15 minTanDip: 0 }
      ]
    }
  }
}
''')


def text(config):
    return 'Synthetic dumper preamble\nDigitize: ' + replay.render(config) + '\nPrimary: {}\n'


def compile_config(config=None, path=PATHS[0]):
    return selection.compile_selection(text(synthetic_config() if config is None else config), path)


def good_features(shape=(2, 3)):
    features = {name: np.ones(shape, dtype=np.bool_) for name in selection.BOOL_FEATURES}
    features.update({name: np.full(shape, value, dtype=np.float64) for name, value in {
        'pz': 40., 'momentum': 100., 'tan_dip': 1., 'd0': 0., 'momentum_error': 1.,
        'chi2_dof': 2., 'fit_consistency': 0.5,
    }.items()})
    features['fit_particle'] = np.full(shape, 11, dtype=np.int32)
    features['active_hits'] = np.full(shape, 15, dtype=np.int32)
    return features


def scalar_reference(features, contract):
    """Independent scalar short-circuit transcription; no kernel helper calls."""
    accepted = np.zeros(features['valid_track'].shape, dtype=bool)
    for event in range(accepted.shape[0]):
        for track in range(accepted.shape[1]):
            f = {name: value[event, track].item() for name, value in features.items()}
            if not (f['valid_track'] and f['status_has_SeedOK'] and f['has_t0_segment'] and f['supported_fit']):
                continue
            for c in contract['cuts']:
                if f['fit_particle'] != c['fitparticle'] or f['pz'] * 1.0 < 0.0:
                    continue
                if (f['active_hits'] >= c['minNStrawHits'] and
                    c['minMomentum'] < f['momentum'] < c['maxMomentum'] and
                    f['momentum_error'] < c['maxMomErr'] and f['chi2_dof'] < c['maxChi2DOF'] and
                    f['fit_consistency'] > c['minFitCons'] and
                    c['minTanDip'] < f['tan_dip'] < c['maxTanDip'] and
                    c['minD0'] < f['d0'] < c['maxD0']):
                    accepted[event, track] = True
                    break
    events = np.array([bool(contract['noFilter'] or sum(row) >= contract['minNTrks']) for row in accepted])
    return accepted, events


def assert_eager_jit(features, contract):
    selection.validate_features(features)
    expected = scalar_reference(features, contract)
    for actual in (selection.evaluate(features, contract),
                   jax.jit(lambda f: selection.evaluate(f, contract))(features)):
        for result, reference in zip(actual, expected):
            assert isinstance(result, jax.Array)
            np.testing.assert_array_equal(result, reference)
    return expected


@pytest.mark.parametrize('path,label,d0,tandip', list(zip(PATHS, LABELS, (200, 600), (100, 1000))))
def test_contract_thresholds_and_provenance(path, label, d0, tandip):
    config = synthetic_config()
    before = deepcopy(config)
    embedded = text(config)
    contract = selection.compile_selection(embedded, path)
    assert config == before
    assert json.loads(json.dumps(contract, allow_nan=False)) == contract
    assert contract['config_sha256'] == hashlib.sha256(embedded.encode()).hexdigest()
    assert contract['path'] == path and contract['filter'] == label
    assert contract['source']['sha256'] == '51b6dbfe6f5dcd115afdfb8b9ada7aa187b8610f717cdbbc1527e7b573fabd39'
    assert contract['source']['url'].endswith('/TrkFilters/src/KalSeedFilter_module.cc')
    assert 'prototype' in contract['status'] and 'no measured features' in contract['status']
    assert contract['collections'] == ['TrigKSFDe']
    assert set(contract['features']['fields']) == set(selection.FEATURES)
    assert contract['minNTrks'] == 1 and contract['noFilter'] is False
    assert contract['noFilter_explicit'] is False and contract['momSurface'] is None
    for row, pdg, mom in zip(contract['cuts'], (11, -11), (80, 70)):
        assert row == dict(fitparticle=pdg, fitdirection='downstream', requireCaloCluster=False,
                           maxChi2DOF=20., maxD0=float(d0), maxMomErr=10., maxMomentum=500.,
                           maxTanDip=float(tandip), minD0=-float(d0), minFitCons=-1.,
                           minMomentum=float(mom), minNStrawHits=15, minTanDip=0.,
                           minNStereo=0, minNPlanes=0, seedFitFlag=['SeedOK'])


def test_thresholds_come_from_supplied_config():
    config = synthetic_config()
    config['physics']['filters'][LABELS[0]]['KalSeedCuts'][0]['minMomentum'] = replay.Atom('91.125')
    contract = compile_config(config)
    assert contract['cuts'][0]['minMomentum'] == 91.125
    assert contract['config_sha256'] != compile_config()['config_sha256']


BOUNDARIES = (
    ('momentum', 'minMomentum', True), ('momentum', 'maxMomentum', False),
    ('tan_dip', 'minTanDip', True), ('tan_dip', 'maxTanDip', False),
    ('d0', 'minD0', True), ('d0', 'maxD0', False),
    ('momentum_error', 'maxMomErr', False), ('chi2_dof', 'maxChi2DOF', False),
    ('fit_consistency', 'minFitCons', True),
)


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('particle', (11, -11))
@pytest.mark.parametrize('feature,cut_name,lower', BOUNDARIES)
def test_exact_on_nextafter_strict_boundaries(path, particle, feature, cut_name, lower):
    contract = compile_config(path=path)
    c = next(row for row in contract['cuts'] if row['fitparticle'] == particle)
    bound = c[cut_name]
    features = good_features((3, 1))
    features['fit_particle'][:] = particle
    features[feature][:, 0] = [np.nextafter(bound, -np.inf), bound, np.nextafter(bound, np.inf)]
    tracks, _ = assert_eager_jit(features, contract)
    np.testing.assert_array_equal(tracks[:, 0], [False, False, True] if lower else [True, False, False])


def test_inclusive_hits_and_no_guessed_nonnegative_quality_cut():
    f = good_features((4, 1))
    f['active_hits'][:, 0] = [14, 15, 16, 15]
    f['momentum_error'][3, 0] = -1.
    f['chi2_dof'][3, 0] = -2.
    tracks, _ = assert_eager_jit(f, compile_config())
    np.testing.assert_array_equal(tracks[:, 0], [False, True, True, True])


def test_padding_status_t0_supported_pdg_and_direction():
    f = good_features((12, 2))
    f['valid_track'][:, 1] = False
    for event, name in enumerate(selection.BOOL_FEATURES):
        f[name][event, 0] = False
    f['fit_particle'][4, 0] = 13
    f['fit_particle'][5, 0] = -11
    f['momentum'][5, 0] = 75.
    f['pz'][6:, 0] = [-1., -0., 0., 1., np.nan, np.inf]
    tracks, events = assert_eager_jit(f, compile_config())
    np.testing.assert_array_equal(events, [False] * 5 + [True, False, True, True, True, True, True])
    assert not tracks[:, 1].any()


@pytest.mark.parametrize('feature', selection.FLOAT_FEATURES)
@pytest.mark.parametrize('value', (np.nan, np.inf, -np.inf))
def test_nonfinite_cpp_comparison_semantics(feature, value):
    f = good_features((1, 1))
    f[feature][0, 0] = value
    tracks, _ = assert_eager_jit(f, compile_config())
    expected = ((feature == 'pz' and not value < 0.) or
                (feature in ('momentum_error', 'chi2_dof') and value == -np.inf) or
                (feature == 'fit_consistency' and value == np.inf))
    assert tracks[0, 0] == expected


@pytest.mark.parametrize('shape', ((3, 0), (0, 3), (0, 0)))
def test_empty_dimensions(shape):
    assert_eager_jit(good_features(shape), compile_config())


def test_cut_or_counts_each_track_once():
    config = synthetic_config()
    module = config['physics']['filters'][LABELS[0]]
    module['KalSeedCuts'].append(deepcopy(module['KalSeedCuts'][0]))
    module['minNTrks'] = replay.Atom('2')
    f = good_features((2, 2))
    f['valid_track'][0, 1] = False
    tracks, events = assert_eager_jit(f, compile_config(config))
    np.testing.assert_array_equal(tracks, [[True, False], [True, True]])
    np.testing.assert_array_equal(events, [False, True])


@pytest.mark.parametrize('no_filter', ('true', 'false'))
def test_explicit_no_filter_and_min_tracks_zero(no_filter):
    config = synthetic_config()
    module = config['physics']['filters'][LABELS[0]]
    module['noFilter'] = replay.Atom(no_filter)
    contract = compile_config(config)
    assert contract['noFilter_explicit'] is True
    _, events = assert_eager_jit(good_features((2, 0)), contract)
    np.testing.assert_array_equal(events, [no_filter == 'true'] * 2)
    module['minNTrks'] = replay.Atom('0')
    _, events = assert_eager_jit(good_features((2, 0)), compile_config(config))
    assert events.all()


@pytest.mark.parametrize('field,value', [
    ('momSurface', 'TT_Front'), ('noInfo', replay.Atom('true')), ('unknown', replay.Atom('1')),
    ('minNTrks', replay.Atom('-1')), ('minNTrks', replay.Atom('1.5')),
    ('noFilter', replay.Atom('1')), ('module_type', 'Other'), ('module_label', 'Other'),
    ('kalSeedCollections', ['Other']), ('kalSeedCollections', ['TrigKSFDe', 'TrigKSFDe']),
    ('kalSeedCollections', ['TrigKSFDe::Digitize']),
])
def test_unsupported_module_configuration(field, value):
    config = synthetic_config()
    config['physics']['filters'][LABELS[0]][field] = value
    with pytest.raises(ValueError):
        compile_config(config)


@pytest.mark.parametrize('field,value', [
    ('fitparticle', replay.Atom('13')), ('fitdirection', 'upstream'),
    ('requireCaloCluster', replay.Atom('true')), ('minNStereo', replay.Atom('1')),
    ('minNPlanes', replay.Atom('1')), ('seedFitFlag', ['SeedOK', 'Other']),
    ('unknown', replay.Atom('1')), ('minMomentum', replay.Atom('nan')),
    ('minNStrawHits', replay.Atom('15.5')), ('minNStrawHits', replay.Atom('-1')),
])
def test_unsupported_cut_configuration(field, value):
    config = synthetic_config()
    config['physics']['filters'][LABELS[0]]['KalSeedCuts'][0][field] = value
    with pytest.raises(ValueError):
        compile_config(config)


@pytest.mark.parametrize('field', ('fitparticle', 'fitdirection', 'minMomentum', 'requireCaloCluster'))
def test_missing_required_cut_field(field):
    config = synthetic_config()
    del config['physics']['filters'][LABELS[0]]['KalSeedCuts'][0][field]
    with pytest.raises(ValueError, match='missing required'):
        compile_config(config)


def test_explicit_supported_defaults_and_debug_metadata():
    config = synthetic_config()
    module = config['physics']['filters'][LABELS[0]]
    module.update(noInfo=replay.Atom('false'), minNTrks=replay.Atom('1'), debugLevel=replay.Atom('4'))
    for row in module['KalSeedCuts']:
        row.update(minNStereo=replay.Atom('0'), minNPlanes=replay.Atom('0'), seedFitFlag=['SeedOK'])
    assert_eager_jit(good_features(), compile_config(config))


def test_path_and_config_binding_fail_closed():
    with pytest.raises(ValueError, match='unsupported path'):
        compile_config(path='cpr_Other')
    config = synthetic_config()
    config['physics']['trigger_paths'] = ['110:' + PATHS[1]]
    with pytest.raises(ValueError):
        compile_config(config)
    config = synthetic_config()
    config['physics'][PATHS[0]].reverse()
    with pytest.raises(ValueError):
        compile_config(config)
    config = synthetic_config()
    config['physics']['filters'][LABELS[0]]['KalSeedCuts'].pop()
    with pytest.raises(ValueError, match='both electron'):
        compile_config(config)
    with pytest.raises(ValueError):
        selection.compile_selection(text(synthetic_config()).replace('minD0: -200', 'minD0: -200 minD0: 0'), PATHS[0])


@pytest.mark.parametrize('name', selection.BOOL_FEATURES)
def test_boolean_dtypes_are_not_coerced(name):
    f = good_features()
    f[name] = f[name].astype(np.int32)
    with pytest.raises(ValueError, match='bool dtype'):
        selection.validate_features(f)


@pytest.mark.parametrize('name', selection.INT_FEATURES)
def test_integer_arrays_are_not_coerced(name):
    f = good_features()
    f[name] = f[name].astype(np.float64)
    with pytest.raises(ValueError, match='integer'):
        selection.validate_features(f)


def test_validation_unknown_truth_fields_shape_precision_and_counts():
    f = good_features()
    f['true_momentum'] = f['momentum'].copy()
    with pytest.raises(ValueError, match='exactly'):
        selection.validate_features(f)
    with pytest.raises(ValueError, match='exactly'):
        selection.evaluate(f, compile_config())
    for replacement in (np.ones(2), np.ones((3, 2)), np.ones((2, 3), dtype=np.float32)):
        f = good_features()
        f['momentum'] = replacement
        with pytest.raises(ValueError):
            selection.validate_features(f)
    f = good_features()
    del f['pz']
    with pytest.raises(ValueError):
        selection.validate_features(f)
    f = good_features()
    f['active_hits'][0, 0] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        selection.validate_features(f)
    f = good_features()
    f['fit_particle'] = f['fit_particle'].astype(np.uint32)
    with pytest.raises(ValueError, match='signed integer'):
        selection.validate_features(f)


def test_jax_arrays_validate_and_evaluate():
    f = {name: jnp.asarray(value) for name, value in good_features().items()}
    assert_eager_jit(f, compile_config())


def test_x64_is_required_and_not_mutated():
    contract = compile_config()
    with enable_x64(False):
        for call in (lambda: compile_config(), lambda: selection.validate_features(good_features()),
                     lambda: selection.evaluate(good_features(), contract)):
            with pytest.raises(ValueError, match='jax_enable_x64'):
                call()
            assert not jax.config.x64_enabled


def test_contract_cli_refuses_overwrite(tmp_path, monkeypatch):
    config = tmp_path / 'config.fcl'
    config.write_text(text(synthetic_config()))
    output = tmp_path / 'contract.json'
    monkeypatch.setattr(sys, 'argv', ['jax_kalseed', '--config', str(config), '--path', PATHS[0], '--output', str(output)])
    selection.main()
    assert json.loads(output.read_text())['path'] == PATHS[0]
    with pytest.raises(ValueError, match='overwrite'):
        selection.main()


def test_subnormal_direction_and_custom_subnormal_threshold():
    f = good_features((5, 1))
    tiny = np.nextafter(0., 1.)
    f['pz'][:, 0] = [-tiny, -0., 0., tiny, np.nan]
    _, events = assert_eager_jit(f, compile_config())
    np.testing.assert_array_equal(events, [False, True, True, True, True])
    config = synthetic_config()
    for row in config['physics']['filters'][LABELS[0]]['KalSeedCuts']:
        row['minTanDip'] = replay.Atom(repr(float(tiny)))
    f['pz'][:] = 1.
    f['tan_dip'][:, 0] = [-tiny, 0., tiny, np.nextafter(tiny, 1.), 1.]
    _, events = assert_eager_jit(f, compile_config(config))
    np.testing.assert_array_equal(events, [False, False, False, True, True])
