"""Synthetic saved-file orchestration only; no art, build or runtime attestation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock, call

import awkward as ak
import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import feature_validate as validator
import full_replay
import integrity
import path_state
import replay
from test_trigger_feature_replay import fast, install
from test_trigger_quiet_replay import saved, dump, log
from test_trigger_jax_kalseed import synthetic_config, text
from test_trigger_jax_feature_validate import sample

PATHS = tuple(validator.jax_kalseed.PATH_FILTERS)


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', previous)


def selection_case():
    config = synthetic_config()
    config['physics']['producers']['MergeInfo'] = {'module_type': 'MergeTriggerInfo'}
    for path in PATHS:
        config['physics'][path].append('MergeInfo')
    result = deepcopy(config)
    result['process_name'] = 'TriggerReplay'
    return text(config), result


def test_selection_contracts_compile_exact_config_and_keep_released_suffix():
    config, result = selection_case()
    before = deepcopy(result)
    contracts, sequences = validator.selection_contracts(config, result)
    for path in PATHS:
        assert contracts[path] == validator.jax_kalseed.compile_selection(config, path)
        assert sequences[path] == result['physics'][path]
        assert sequences[path][-1] == 'MergeInfo'
    assert result == before


@pytest.mark.parametrize('mutation', ['threshold', 'collection', 'fit-type', 'fit-filter',
    'missing-fit', 'duplicate-fit', 'duplicate-selection', 'missing-suffix',
    'downstream-filter', 'downstream-producer', 'unknown-suffix', 'process'])
def test_selection_contracts_reject_changed_execution(mutation):
    config, result = selection_case()
    physics = result['physics']
    label = validator.jax_kalseed.PATH_FILTERS[PATHS[0]]
    sequence = physics[PATHS[0]]
    if mutation == 'threshold':
        physics['filters'][label]['KalSeedCuts'][0]['minMomentum'] = replay.Atom('81')
    elif mutation == 'collection': physics['filters'][label]['kalSeedCollections'] = ['Other']
    elif mutation == 'fit-type': physics['producers']['TrigKSFDe']['module_type'] = 'OtherFit'
    elif mutation == 'fit-filter':
        physics['filters']['TrigKSFDe'] = physics['producers'].pop('TrigKSFDe')
    elif mutation == 'missing-fit': sequence.remove('TrigKSFDe')
    elif mutation == 'duplicate-fit': sequence.insert(0, 'TrigKSFDe')
    elif mutation == 'duplicate-selection': sequence.insert(1, label)
    elif mutation == 'missing-suffix': sequence.pop()
    elif mutation == 'downstream-filter': physics['filters']['MergeInfo'] = {'module_type': 'MergeTriggerInfo'}
    elif mutation == 'downstream-producer': physics['producers']['MergeInfo']['module_type'] = 'OtherProducer'
    elif mutation == 'unknown-suffix': sequence.append('Unknown')
    elif mutation == 'process': result['process_name'] = 'Digitize'
    with pytest.raises(ValueError):
        validator.selection_contracts(config, result)


def test_selection_contracts_reject_fit_after_selection():
    config, result = selection_case()
    sequence = result['physics'][PATHS[0]]
    sequence[:] = [sequence[1], sequence[0], sequence[2]]
    with pytest.raises(ValueError):
        validator.selection_contracts(config, result)


@pytest.fixture
def measured(fast, monkeypatch):
    """Reuse feature_replay's real preflight and synthetic ROOT seams, not its gates."""
    production = replay.production_config(fast.text, 'Digitize')
    physics = production['physics']
    selected = synthetic_config()['physics']
    physics['producers'].update(deepcopy(selected['producers']))
    physics['producers']['MergeInfo'] = {'module_type': 'MergeTriggerInfo'}
    physics['filters'].update(deepcopy(selected['filters']))
    for old, path in zip(('p0', 'p1'), PATHS):
        del physics[old]
        physics[path] = ['Digi', 'Cut', 'TrigKSFDe', validator.jax_kalseed.PATH_FILTERS[path], 'MergeInfo']
    renamed = {'p0': PATHS[0], 'p1': PATHS[1]}
    physics['trigger_paths'] = [entry.split(':')[0] + ':' + renamed.get(entry.split(':')[1], entry.split(':')[1])
                                if isinstance(entry, str) and ':' in entry else entry for entry in physics['trigger_paths']]
    fast.text = text(production)
    fast.config.write_text(fast.text)
    result, base_manifest = replay.prepare(production)
    result['source']['fileNames'] = list(map(str, fast.sources))
    fcl = full_replay.render_config(result)
    fast.manifest.update(base_manifest, config_sha256=hashlib.sha256(fast.text.encode()).hexdigest(),
                         replay_fcl_sha256=hashlib.sha256(fcl.encode()).hexdigest())
    fast.result = result
    (fast.baseline / 'replay.fcl').write_text(fcl)
    dump(fast.baseline / 'manifest.json', fast.manifest)
    baseline = fast.trees[str(fast.baseline_art)]
    stock_paths = replay.paths(production)
    replay_paths = replay.paths(result)
    baseline.stock[:] = 9
    baseline.stock[:, 0] = 5
    baseline.fast[:] = 5
    states = {PATHS[0]: np.array([2, 2, 1, 2, 1, 2], dtype='u2'),
              PATHS[1]: np.array([1, 2, 2, 2, 1, 2], dtype='u2')}
    indices = {PATHS[0]: np.array([1, 3, 4, 3, 4, 1], dtype='u2'),
               PATHS[1]: np.array([4, 3, 1, 3, 4, 1], dtype='u2')}
    for path in PATHS:
        baseline.stock[:, [name for _, name, _ in stock_paths].index(path)] = 21
        baseline.fast[:, [name for _, name, _ in replay_paths].index(path)] = states[path] | (indices[path] << 2)
    names = fast.manifest['paths']
    fast.comparison.update(config_sha256=fast.manifest['config_sha256'],
        replay_fcl_sha256=fast.manifest['replay_fcl_sha256'],
        validation_evidence=integrity.files([fast.baseline / 'manifest.json', fast.baseline / 'replay.fcl']),
        paths=path_state.compare_matrices(baseline.stock[:, 1:], baseline.fast, names,
            {name: seq for _, name, seq in stock_paths if name in names},
            {name: seq for _, name, seq in replay_paths}))
    dump(fast.comparison_path, fast.comparison)
    prepared, art, manifest = install(fast)
    _, result = validator.feature_replay.preflight(prepared, fast.text, 6,
        manifest['feature_export']['runtime']['TRIGGER_AREA'], manifest['feature_export']['runtime']['OFFLINE_IMAGE'],
        manifest['feature_export']['exporter_source'], manifest['feature_export']['exporter_sha256'])
    (art.parent / 'replay.expanded.fcl').write_text(full_replay.render_config(result))
    table = fast.tmp / 'features.root'
    table.write_bytes(b'synthetic table read at mocked I/O boundary')
    data = sample((1, 0, 1, 0, 1, 0))
    data['keys'] = fast.keys.copy()
    for path in PATHS:
        data['info_present'][path] = (states[path] == 1) | (indices[path] >= 3)
        data['accepted_keys'][path] = ak.Array([[0] if state == 1 else [] for state in states[path]])
    data['product_present'][-1] = False
    data['metadata'] = {'events': 6, 'schema_version': 1, 'runtime_admitted': False,
        'highest_cycles': {'events': 3, 'schema': 1}, 'raw_status_authenticated': False,
        'source_ordinal': np.arange(1, 7), 'product_id': ak.Array(['id'] * 6),
        'raw_status': ak.Array([[1]] * 6), 't0_segment_index': ak.Array([[0]] * 6),
        't0': ak.Array([[float('nan')]] * 6)}
    reader = Mock(return_value=data)
    monkeypatch.setattr(validator.feature_table, 'read', reader)
    observer = Mock(return_value={'runtime_observed': True, 'scope': 'synthetic pinned observation'})
    monkeypatch.setattr(validator.feature_runtime, 'observe_worker', observer)
    compare = Mock(wraps=validator.feature_replay.compare_decisions)
    monkeypatch.setattr(validator.feature_replay, 'compare_decisions', compare)
    pure = Mock(wraps=validator.jax_feature_validate.validate_arrays)
    monkeypatch.setattr(validator.jax_feature_validate, 'validate_arrays', pure)
    fast.prepared, fast.art, fast.table, fast.feature_manifest = prepared, art, table, manifest
    fast.data, fast.reader, fast.observer, fast.compare, fast.pure = data, reader, observer, compare, pure
    fast.kwargs = dict(config_text=fast.text, log_text=log(), bundle=fast.tmp / 'bundle',
        expected_library_sha256='a' * 64, expected_manifest_sha256='b' * 64)
    return fast


def run(case, **kwargs):
    return validator.validate(case.art, case.table, case.prepared, **(case.kwargs | kwargs))


def test_measured_orchestration_exact_decisions_actual_fast_reach_and_json(measured):
    report = run(measured)
    measured.compare.assert_called_once_with(measured.art, measured.baseline_art, measured.feature_manifest)
    assert report['decisions']['original_packed_words_equal']
    assert report['decisions']['replay_packed_words_equal']
    measured.reader.assert_called_once()
    np.testing.assert_array_equal(measured.reader.call_args.kwargs['expected_keys'], measured.keys)
    passed = measured.pure.call_args.kwargs
    for key in ('keys', 'features', 'track_index', 'info_present', 'accepted_keys', 'product_present'):
        assert passed[key] is measured.data[key]
    assert passed['states'][PATHS[0]].tolist() == [2, 2, 1, 2, 1, 2]
    assert passed['indices'][PATHS[1]].tolist() == [4, 3, 1, 3, 4, 1]
    assert report['selection']['arrays_equal'] and report['measured_arrays_equal']
    for path in PATHS:
        assert report['selection']['paths'][path]['reached'] == 4
        assert report['selection']['paths'][path]['unreached'] == 2
        assert report['selection']['paths'][path]['full_path_mismatches'] == 0
    observation = call(measured.kwargs['bundle'], measured.kwargs['log_text'],
        expected_library_sha256='a' * 64, expected_manifest_sha256='b' * 64)
    assert measured.observer.call_args_list == [observation, observation]
    assert report['runtime_observation'] == measured.observer.return_value
    assert not report['measured_metadata_verified'] and not report['measured_parity_established']
    assert report['feature_metadata']['collection_present_events'] == 5
    assert report['feature_metadata']['collection_absent_events'] == 1
    assert not set(('source_ordinal', 'product_id', 'raw_status', 't0_segment_index', 't0')) & set(report['feature_metadata'])
    assert json.loads(json.dumps(report, allow_nan=False))['events'] == 6


@pytest.mark.parametrize('section', ['process_name', 'source', 'physics', 'outputs', 'services'])
def test_executed_expanded_configuration_must_match_every_section(measured, section):
    path = measured.art.parent / 'replay.expanded.fcl'
    expanded = replay.parse_expanded(path.read_text())
    if section == 'process_name': expanded[section] = 'WrongProcess'
    else: expanded[section]['unexpected'] = 'changed'
    path.write_text(full_replay.render_config(expanded))
    with pytest.raises(ValueError, match='differs from admitted FCL'):
        run(measured)
    measured.compare.assert_not_called()
    measured.reader.assert_not_called()


@pytest.mark.parametrize('process', ['stock', 'fast'])
def test_full_saved_packed_word_gate_not_just_selected_acceptance(measured, process):
    # A non-BO path endpoint differs while its accepted boolean is unchanged.
    tree = measured.trees[str(measured.art)]
    getattr(tree, process)[0, -1] = 13
    with pytest.raises(ValueError, match='packed words differ'):
        run(measured)
    measured.compare.assert_called_once()
    measured.reader.assert_not_called()


def test_event_completion_count_is_not_table_population_guess(measured):
    with pytest.raises(ValueError, match='event count'):
        run(measured, log_text=log().replace('6', '5'))
    measured.reader.assert_not_called()


def test_runtime_booleans_cannot_replace_independent_pins(measured):
    with pytest.raises(TypeError):
        run(measured, runtime_ready=True)
    kwargs = measured.kwargs.copy()
    del kwargs['expected_library_sha256']
    with pytest.raises(TypeError):
        validator.validate(measured.art, measured.table, measured.prepared, **kwargs)
    measured.observer.side_effect = ValueError('pinned runtime receipt rejected')
    with pytest.raises(ValueError, match='pinned runtime'):
        run(measured)
    measured.compare.assert_not_called()
    measured.reader.assert_not_called()


def test_table_reach_must_follow_fast_not_stock_endpoints(measured):
    measured.data['info_present'][PATHS[0]][0] = True
    with pytest.raises(ValueError, match='actual TriggerReplay filter reach'):
        run(measured)


def test_changed_feature_evidence_during_comparison_is_rejected(measured):
    def changing(**kwargs):
        report = measured.pure._mock_wraps(**kwargs)
        measured.table.write_bytes(b'changed during validation')
        return report
    measured.pure.side_effect = changing
    with pytest.raises(ValueError):
        run(measured)


def test_runtime_admission_rechecked_after_array_comparison(measured):
    measured.observer.side_effect = [measured.observer.return_value, ValueError('deployment changed')]
    with pytest.raises(ValueError, match='deployment changed'):
        run(measured)
    measured.pure.assert_called_once()
    assert measured.observer.call_count == 2


@pytest.mark.parametrize('equal', [True, False])
def test_cli_preserves_report_but_rejects_selection_mismatches(tmp_path, monkeypatch, equal):
    config, log_file = tmp_path / 'config', tmp_path / 'log'
    config.write_text('config')
    log_file.write_text('log')
    output = tmp_path / 'comparison.json'
    report = {'events': 6, 'measured_arrays_equal': equal, 'measured_parity_established': False}
    monkeypatch.setattr(validator, 'validate', Mock(return_value=report))
    monkeypatch.setattr(sys, 'argv', ['feature_validate', '--art', 'a', '--table', 't',
        '--prepared', 'p', '--config', str(config), '--log', str(log_file), '--bundle', 'b',
        '--expected-library-sha256', 'a' * 64, '--expected-manifest-sha256', 'b' * 64,
        '--output', str(output)])
    if equal:
        validator.main()
    else:
        with pytest.raises(SystemExit) as error:
            validator.main()
        assert error.value.code == 1
    assert json.loads(output.read_text()) == report


def test_preflight_binds_original_full_configuration_bytes(measured):
    with pytest.raises(ValueError, match='configuration bytes differ'):
        run(measured, config_text=measured.text + '\n')
    measured.observer.assert_not_called()
    measured.reader.assert_not_called()


def test_optional_released_status_gate_is_separate_and_rechecked(measured, monkeypatch):
    source = measured.tmp / 'released-source'
    status = {'source_bound_status_consistent': True, 'measured_parity_established': False}
    check = Mock(return_value=status)
    monkeypatch.setattr(validator.feature_status, 'validate', check)
    report = run(measured, status_source_directory=source)
    assert check.call_args_list == [call(measured.data, source), call(measured.data, source)]
    assert report['released_source_status_check'] == status
    assert not report['measured_metadata_verified'] and not report['measured_parity_established']
    assert not report['feature_metadata']['raw_status_authenticated']


def test_no_implicit_released_status_admission(measured, monkeypatch):
    check = Mock(side_effect=AssertionError('source gate not requested'))
    monkeypatch.setattr(validator.feature_status, 'validate', check)
    assert run(measured)['released_source_status_check'] is None
    check.assert_not_called()


def test_released_status_contradiction_stops_before_jax(measured, monkeypatch):
    monkeypatch.setattr(validator.feature_status, 'validate',
                        Mock(side_effect=ValueError('raw status contradiction')))
    with pytest.raises(ValueError, match='raw status contradiction'):
        run(measured, status_source_directory=measured.tmp / 'source')
    measured.pure.assert_not_called()


def test_released_status_evidence_mutation_is_rejected(measured, monkeypatch):
    monkeypatch.setattr(validator.feature_status, 'validate', Mock(side_effect=[
        {'source_bound_status_consistent': True, 'sha256': 'a'},
        {'source_bound_status_consistent': True, 'sha256': 'b'}]))
    with pytest.raises(ValueError, match='status evidence changed'):
        run(measured, status_source_directory=measured.tmp / 'source')
    measured.pure.assert_called_once()
