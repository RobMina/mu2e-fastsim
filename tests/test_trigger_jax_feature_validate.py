"""Synthetic vectorized-array checks only, not measured/C++ validation evidence."""
from copy import deepcopy
from pathlib import Path
import sys

import awkward as ak
import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import jax_feature_validate as validator
import jax_kalseed as selection
from test_trigger_jax_kalseed import synthetic_config, text

PATHS = tuple(selection.PATH_FILTERS)


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', previous)


def sample(counts=(2, 0, 1, 0)):
    counts = np.asarray(counts, dtype=np.int64)
    n = len(counts)
    total = int(np.sum(counts))
    floats = dict(pz=40., momentum=100., tan_dip=1., d0=0., momentum_error=1.,
                  chi2_dof=2., fit_consistency=0.5)
    features = {name: ak.unflatten(np.full(total, value, np.float64), counts)
                for name, value in floats.items()}
    features.update({name: ak.unflatten(np.ones(total, np.uint8), counts)
                     for name in selection.BOOL_FEATURES})
    features['fit_particle'] = ak.unflatten(np.full(total, 11, np.int32), counts)
    features['active_hits'] = ak.unflatten(np.full(total, 15, np.uint32), counts)
    tracks = ak.local_index(features['pz'], axis=1)
    config = text(synthetic_config())
    contracts = {path: selection.compile_selection(config, path) for path in PATHS}
    return dict(keys=np.column_stack((np.ones(n, int), np.ones(n, int), np.arange(1, n + 1))),
                features=features, track_index=tracks,
                info_present={path: np.ones(n, bool) for path in PATHS},
                accepted_keys={path: tracks for path in PATHS},
                states={path: np.where(counts > 0, 1, 2) for path in PATHS},
                indices={path: np.full(n, 2, np.uint16) for path in PATHS},
                sequences={path: ['TrigKSFDe', selection.PATH_FILTERS[path]] for path in PATHS},
                contracts=contracts, product_present=np.ones(n, bool))


def test_multitrack_empty_exact_and_metadata():
    data = sample()
    report = validator.validate_arrays(**data)
    assert report['arrays_equal'] is True
    assert report['measured_metadata_verified'] is False
    assert report['measured_parity_established'] is False
    assert report['comparison_process'] == 'TriggerReplay'
    assert 'not authenticated' in report['track_identity']
    for path in PATHS:
        row = report['paths'][path]
        assert row['reached'] == 4 and row['unreached'] == 0
        assert row['empty_reached'] == 2 and row['tracks_compared'] == 3
        assert row['actual_pass'] == 2 and row['actual_fail'] == 2
        assert row['track_identity_mismatches'] == row['full_path_mismatches'] == 0
        assert row['eager_jit_track_mismatches'] == 0


@pytest.mark.parametrize('counts', [(0, 0), ()])
def test_all_empty_and_zero_population(counts):
    report = validator.validate_arrays(**sample(counts))
    assert report['arrays_equal']
    assert report['events'] == len(counts)
    assert report['paths'][PATHS[0]]['tracks_compared'] == 0


def test_actual_reach_independent_per_path_and_missing_distinct_empty(monkeypatch):
    data = sample((1, 0, 0))
    a, b = PATHS
    data['states'][a] = np.array([2, 2, 2])
    data['indices'][a] = np.array([1, 2, 1])
    data['info_present'][a] = np.array([False, True, False])
    data['accepted_keys'][a] = ak.Array([[], [], []])
    data['states'][b] = np.array([1, 2, 2])
    data['indices'][b] = np.array([2, 2, 1])
    data['info_present'][b] = np.array([True, True, False])
    data['product_present'][2] = False
    original = selection.evaluate
    shapes = []

    def counted(features, contract):
        shapes.append(features['valid_track'].shape)
        return original(features, contract)

    monkeypatch.setattr(selection, 'evaluate', counted)
    report = validator.validate_arrays(**data)
    assert report['arrays_equal']
    assert shapes == [(1, 1), (1, 1), (2, 1), (2, 1)]
    assert report['paths'][a]['tracks_compared'] == 0
    assert report['paths'][a]['unreached'] == 2
    assert report['paths'][b]['tracks_compared'] == 1
    assert report['paths'][b]['unreached'] == 1


def test_no_reached_events_are_not_evaluated_as_cpp_failures():
    data = sample((1, 0))
    for path in PATHS:
        data['states'][path][:] = 2
        data['indices'][path][:] = 1
        data['info_present'][path][:] = False
        data['accepted_keys'][path] = ak.Array([[], []])
    report = validator.validate_arrays(**data)
    assert report['arrays_equal']
    assert report['paths'][PATHS[0]]['reached'] == 0
    assert report['paths'][PATHS[0]]['full_path_mismatches'] == 0


def test_same_count_wrong_identity_detected_eager_and_jit():
    data = sample((2,))
    data['features']['momentum'] = ak.Array([[100., 80.]])
    for path in PATHS:
        data['accepted_keys'][path] = ak.Array([[1]])
    report = validator.validate_arrays(**data)
    assert not report['arrays_equal']
    row = report['paths'][PATHS[0]]
    assert row['track_false_positives'] == row['track_false_negatives'] == 1
    assert row['track_identity_mismatches'] == 1
    assert row['track_count_mismatches'] == 0
    assert row['jit_track_mismatches'] == 2
    assert row['full_path_mismatches'] == 0


@pytest.mark.parametrize('cpp_pass', [False, True])
def test_filter_return_and_composed_path_mismatches(cpp_pass):
    data = sample((1,))
    for path in PATHS:
        data['states'][path][:] = 1 if cpp_pass else 2
    if cpp_pass:
        data['features']['momentum'] = ak.Array([[80.]])
    report = validator.validate_arrays(**data)
    row = report['paths'][PATHS[0]]
    assert not report['arrays_equal']
    assert row['filter_false_negatives' if cpp_pass else 'filter_false_positives'] == 1
    assert row['full_path_false_negatives' if cpp_pass else 'full_path_false_positives'] == 1
    assert row['jit_filter_mismatches'] == row['jit_full_path_mismatches'] == 1


def test_producer_suffix_pass_and_unsupported_downstream_failure():
    data = sample((1,))
    for path in PATHS:
        data['sequences'][path].append('MergeInfo')
        data['indices'][path][:] = 3
    assert validator.validate_arrays(**data)['arrays_equal']
    data['states'][PATHS[0]][:] = 2
    with pytest.raises(ValueError, match='downstream failure'):
        validator.validate_arrays(**data)


@pytest.mark.parametrize('state,index', [(0, 0), (3, 2), (4, 2), (1, 1), (2, 0), (2, 3)])
def test_endpoint_gate(state, index):
    data = sample((1,))
    data['states'][PATHS[0]][:] = state
    data['indices'][PATHS[0]][:] = index
    with pytest.raises(ValueError, match='endpoint|Ready/Exception'):
        validator.validate_arrays(**data)


@pytest.mark.parametrize('mutation,match', [
    ('duplicate_keys', 'unique'), ('zero_keys', 'positive integer'), ('float_keys', 'positive integer'),
    ('bad_keys_shape', 'positive integer'), ('feature_alignment', 'aligned'),
    ('index_gap', 'contiguous'), ('negative_index', 'nonnegative'), ('duplicate_track', 'contiguous'),
    ('bad_float', 'float64'), ('bad_bool_byte', 'strict uint8'), ('bad_bool_dtype', 'strict uint8'),
    ('invalid_track', 'valid_track=1'), ('negative_hits', 'nonnegative'), ('float_hits', 'integer'),
    ('unsigned_particle', 'signed'), ('missing_feature', 'exactly'), ('extra_feature', 'exactly'),
    ('presence_shape', 'shape'), ('presence_dtype', 'bool dtype'), ('missing_with_tracks', 'zero tracks'),
    ('reached_missing_empty', 'present collection'), ('info_reach_mismatch', 'presence differs'),
    ('accepted_duplicate', 'duplicate accepted'), ('accepted_negative', 'nonnegative'),
    ('accepted_outside', 'range'), ('accepted_float', 'integer'), ('accepted_alignment', 'malformed'),
    ('missing_leaf', 'malformed'), ('missing_list', 'malformed'),
])
def test_strict_array_gates(mutation, match):
    data = sample((2, 0))
    a = PATHS[0]
    if mutation == 'duplicate_keys': data['keys'][1] = data['keys'][0]
    elif mutation == 'zero_keys': data['keys'][0, 0] = 0
    elif mutation == 'float_keys': data['keys'] = data['keys'].astype(float)
    elif mutation == 'bad_keys_shape': data['keys'] = data['keys'][:, :2]
    elif mutation == 'feature_alignment': data['features']['pz'] = ak.Array([[40.], []])
    elif mutation == 'index_gap': data['track_index'] = ak.Array([[0, 2], []])
    elif mutation == 'negative_index': data['track_index'] = ak.Array([[0, -1], []])
    elif mutation == 'duplicate_track': data['track_index'] = ak.Array([[0, 0], []])
    elif mutation == 'bad_float': data['features']['pz'] = ak.values_astype(data['features']['pz'], np.float32)
    elif mutation == 'bad_bool_byte': data['features']['supported_fit'] = ak.values_astype(ak.Array([[1, 2], []]), np.uint8)
    elif mutation == 'bad_bool_dtype': data['features']['supported_fit'] = ak.Array([[1, 1], []])
    elif mutation == 'invalid_track': data['features']['valid_track'] = ak.values_astype(ak.Array([[1, 0], []]), np.uint8)
    elif mutation == 'negative_hits': data['features']['active_hits'] = ak.Array([[15, -1], []])
    elif mutation == 'float_hits': data['features']['active_hits'] = ak.Array([[15., 15.], []])
    elif mutation == 'unsigned_particle': data['features']['fit_particle'] = ak.values_astype(data['features']['fit_particle'], np.uint32)
    elif mutation == 'missing_feature': del data['features']['pz']
    elif mutation == 'extra_feature': data['features']['truth_p'] = data['features']['pz']
    elif mutation == 'presence_shape': data['product_present'] = np.ones((2, 1), bool)
    elif mutation == 'presence_dtype': data['product_present'] = np.ones(2, np.uint8)
    elif mutation == 'missing_with_tracks': data['product_present'][0] = False
    elif mutation == 'reached_missing_empty': data['product_present'][1] = False
    elif mutation == 'info_reach_mismatch': data['info_present'][a][0] = False
    elif mutation == 'accepted_duplicate': data['accepted_keys'][a] = ak.Array([[0, 0], []])
    elif mutation == 'accepted_negative': data['accepted_keys'][a] = ak.Array([[-1], []])
    elif mutation == 'accepted_outside': data['accepted_keys'][a] = ak.Array([[2], []])
    elif mutation == 'accepted_float': data['accepted_keys'][a] = ak.Array([[0.], []])
    elif mutation == 'accepted_alignment': data['accepted_keys'][a] = ak.Array([[0]])
    elif mutation == 'missing_leaf': data['features']['pz'] = ak.Array([[40., None], []])
    elif mutation == 'missing_list': data['features']['pz'] = ak.Array([[40., 40.], None])
    with pytest.raises(ValueError, match=match):
        validator.validate_arrays(**data)


@pytest.mark.parametrize('field,value', [('source', {}), ('config_sha256', 'short'),
                                        ('path', 'stock'), ('process', 'TriggerReplay'),
                                        ('collections', ['stock']), ('minNTrks', -1), ('noInfo', True)])
def test_contract_binding_rejected(field, value):
    data = sample((1,))
    data['contracts'][PATHS[0]][field] = value
    with pytest.raises(ValueError, match='contract'):
        validator.validate_arrays(**data)


def test_contract_full_config_consistency():
    data = sample((1,))
    data['contracts'][PATHS[1]]['config_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='same full configuration'):
        validator.validate_arrays(**data)


def test_min_tracks_no_filter_empty_and_nonfinite_semantics():
    data = sample((1, 0))
    config = synthetic_config()
    for path in PATHS:
        config['physics']['filters'][selection.PATH_FILTERS[path]]['minNTrks'] = selection.replay.Atom('0')
        data['contracts'][path] = selection.compile_selection(text(config), path)
    # Both contracts must bind the completed full config, not per-path intermediate bytes.
    data['contracts'] = {path: selection.compile_selection(text(config), path) for path in PATHS}
    for path in PATHS:
        data['states'][path][:] = 1
    data['features']['pz'] = ak.Array([[np.nan], []])
    assert validator.validate_arrays(**data)['arrays_equal']
    data['features']['momentum'] = ak.Array([[np.inf], []])
    for path in PATHS:
        data['accepted_keys'][path] = ak.Array([[], []])
    assert validator.validate_arrays(**data)['arrays_equal']


def test_no_filter_override_still_compares_selected_tracks():
    data = sample((1, 0))
    config = synthetic_config()
    for path in PATHS:
        config['physics']['filters'][selection.PATH_FILTERS[path]]['noFilter'] = selection.replay.Atom('true')
        data['states'][path][:] = 1
        data['accepted_keys'][path] = ak.Array([[], []])
    data['contracts'] = {path: selection.compile_selection(text(config), path) for path in PATHS}
    data['features']['momentum'] = ak.Array([[80.], []])
    assert validator.validate_arrays(**data)['arrays_equal']


def test_x64_required_and_input_not_mutated():
    data = sample((1,))
    before = deepcopy(data['contracts'])
    validator.validate_arrays(**data)
    assert data['contracts'] == before
    assert ak.to_numpy(ak.flatten(data['features']['valid_track'])).dtype == np.dtype('uint8')
    jax.config.update('jax_enable_x64', False)
    with pytest.raises(ValueError, match='x64'):
        validator.validate_arrays(**data)


@pytest.mark.parametrize('name,dtype,match', [('pz', np.float32, 'float64'),
                                            ('active_hits', np.float64, 'integer'),
                                            ('supported_fit', np.int32, 'strict uint8')])
def test_typed_empty_arrays_still_require_schema(name, dtype, match):
    data = sample((0, 0))
    data['features'][name] = ak.unflatten(np.empty(0, dtype=dtype), np.array([0, 0]))
    with pytest.raises(ValueError, match=match):
        validator.validate_arrays(**data)


def test_absent_info_cannot_have_accepted_tracks():
    data = sample((1,))
    a = PATHS[0]
    data['states'][a][:] = 2
    data['indices'][a][:] = 1
    data['info_present'][a][:] = False
    with pytest.raises(ValueError, match='absent TriggerInfo'):
        validator.validate_arrays(**data)


@pytest.mark.parametrize('sequence', [['TrigKSFDe', 'TrigKSFDe', selection.PATH_FILTERS[PATHS[0]]],
                                     [selection.PATH_FILTERS[PATHS[0]], 'TrigKSFDe'],
                                     ['TrigKSFDe', 'wrongFilter']])
def test_sequence_identity_gate(sequence):
    data = sample((1,))
    data['sequences'][PATHS[0]] = sequence
    with pytest.raises(ValueError, match='sequence|precede'):
        validator.validate_arrays(**data)


def test_upstream_failed_present_tracks_never_become_fullpath_false_positives():
    data = sample((2,))
    for path in PATHS:
        data['states'][path][:] = 2
        data['indices'][path][:] = 1
        data['info_present'][path][:] = False
        data['accepted_keys'][path] = ak.Array([[]])
    report = validator.validate_arrays(**data)
    assert report['arrays_equal']
    assert report['paths'][PATHS[0]]['full_path_false_positives'] == 0


def test_records_actual_comparison_backend_without_performance_claim():
    report = validator.validate_arrays(**sample())
    assert report['jax_runtime'] == {'version': jax.__version__, 'backend': jax.default_backend(),
                                     'x64_enabled': True}
