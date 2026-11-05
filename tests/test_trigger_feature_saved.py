"""Synthetic saved-I/O seams with the real eager/JIT array comparison.

These tests establish no measured/runtime provenance; no art or build is run.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock, call

import awkward as ak
import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import feature_saved as validator
import feature_replay
import integrity
from test_trigger_feature_validate import measured, PATHS
from test_trigger_feature_replay import fast
from test_trigger_quiet_replay import saved, dump


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', previous)


@pytest.fixture
def local(measured, monkeypatch):
    measured.data['track_index'] = ak.values_astype(measured.data['track_index'], np.uint64)
    status = {'source_bound_status_consistent': True, 'counts': {'tracks': 3},
              'source_files': {'synthetic': 'mocked exact-source seam'}}
    measured.status = Mock(return_value=status)
    monkeypatch.setattr(validator.feature_status, 'validate', measured.status)
    measured.source_directory = measured.tmp / 'release-sources'
    # Fixture preparation is finished. None of these admission paths is allowed
    # in a saved-only check; original full postflight remains a separate operation.
    for owner, name in ((feature_replay, 'prepare'), (feature_replay, 'preflight'),
                        (validator.quiet, 'prepare'), (validator.quiet, '_source_keys'),
                        (validator.quiet.hybrid_history, 'verify')):
        monkeypatch.setattr(owner, name, Mock(side_effect=AssertionError('full admission forbidden')))
    return measured


def run(case, **kwargs):
    return validator.validate(case.art, case.table, case.prepared,
        **(dict(config=case.config, baseline_art=case.baseline_art,
                baseline_comparison=case.comparison_path,
                source_directory=case.source_directory) | kwargs))


def rebind_comparison(case):
    dump(case.comparison_path, case.comparison)
    manifest = validator.quiet._load(case.prepared / 'manifest.json')
    for row in manifest['feature_export']['baseline_evidence']:
        if row['path'] == str(case.comparison_path.resolve()):
            row.update(integrity.digest(case.comparison_path))
    dump(case.prepared / 'manifest.json', manifest)


def test_saved_actual_arrays_exact_both_processes_and_no_promotions(local):
    manifest_bytes = (local.prepared / 'manifest.json').read_bytes()
    report = run(local)
    assert report['measured_arrays_equal'] and report['nonvacuous_track_comparison']
    assert report['raw_packed_equal'] and report['original_packed_words_equal']
    assert report['replay_packed_words_equal'] and report['physics_paths'] == 52
    assert report['original_scheduled_paths'] == 53 and report['replay_scheduled_paths'] == 52
    assert report['counts'] == dict(events=6, tracks=3, zero_track_events=3,
                                   collection_present_events=5, collection_absent_events=1)
    for flag in ('measured_metadata_verified', 'measured_parity_established', 'runtime_admitted',
                 'source_history_reaudited', 'source_order_admitted', 'executed_configuration_verified',
                 'loaded_binary_authenticated', 'complete_parent_order_verified'):
        assert report[flag] is False
    assert local.status.call_args_list == [call(local.data, local.source_directory)] * 2
    np.testing.assert_array_equal(local.reader.call_args.kwargs['expected_keys'], local.keys)
    supplied = local.pure.call_args.kwargs
    assert supplied['states'][PATHS[0]].tolist() == [2, 2, 1, 2, 1, 2]
    assert supplied['indices'][PATHS[1]].tolist() == [4, 3, 1, 3, 4, 1]
    for path in PATHS:
        assert report['selection']['paths'][path]['tracks_compared'] == 2
    assert (local.prepared / 'manifest.json').read_bytes() == manifest_bytes
    assert json.loads(json.dumps(report, allow_nan=False))['events'] == 6
    local.observer.assert_not_called()
    assert 'raw_status' not in report['feature_metadata']


def test_byte_exact_copies_do_not_open_original_path_spellings(local):
    copies = local.tmp / 'copies'
    copies.mkdir()
    args = {}
    for field, source in [('config', local.config), ('baseline_art', local.baseline_art),
                          ('baseline_comparison', local.comparison_path)]:
        target = copies / source.name
        target.write_bytes(source.read_bytes())
        args[field] = target
    local.trees[str(args['baseline_art'])] = deepcopy(local.trees[str(local.baseline_art)])
    local.config.unlink()
    local.baseline_art.unlink()
    local.comparison_path.unlink()
    assert run(local, **args)['measured_arrays_equal']


@pytest.mark.parametrize('file', ['config', 'fcl', 'baseline_art', 'comparison'])
def test_copied_content_must_match_hashes(local, file):
    path = {'config': local.config, 'fcl': local.prepared / 'replay.fcl',
            'baseline_art': local.baseline_art, 'comparison': local.comparison_path}[file]
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='bytes|SHA256'):
        run(local)
    local.reader.assert_not_called()
    local.pure.assert_not_called()


@pytest.mark.parametrize('field', ['accepted_bits_crosschecked_with_log',
    'root_event_ids_independently_decoded', 'trigger_schedule_registry_verified',
    'full_production_history_certified', 'all_scheduled_paths_measured'])
def test_completed_baseline_flags_required(local, field):
    local.comparison[field] = False
    rebind_comparison(local)
    with pytest.raises(ValueError, match='completed baseline comparison evidence'):
        run(local)
    local.pure.assert_not_called()


@pytest.mark.parametrize('process', ['stock', 'fast'])
def test_all_packed_words_not_only_two_selections(local, process):
    matrix = getattr(local.trees[str(local.art)], process)
    matrix[0, -1] ^= 4
    with pytest.raises(ValueError, match='packed words differ'):
        run(local)
    local.reader.assert_not_called()


def test_original_nonphysics_column_is_also_compared(local):
    local.trees[str(local.art)].stock[0, 0] ^= 4
    with pytest.raises(ValueError, match='Digitize packed words differ'):
        run(local)


def test_saved_feature_event_order_independently_checked(local):
    local.trees[str(local.art)].keys = local.keys[::-1].copy()
    with pytest.raises(ValueError, match='IDs/order'):
        run(local)


@pytest.mark.parametrize('field', ['parameter_set_ids', 'schedule_registry', 'highest_cycle', 'paths'])
def test_prior_comparison_must_match_baseline_decode(local, field):
    local.comparison[field] = {} if field != 'highest_cycle' else 99
    rebind_comparison(local)
    with pytest.raises(ValueError, match='comparison|52 physics'):
        run(local)


@pytest.mark.parametrize('file', ['config', 'table', 'art', 'manifest', 'fcl', 'baseline_art', 'comparison'])
def test_evidence_mutation_during_array_comparison_rejected(local, monkeypatch, file):
    path = {'config': local.config, 'table': local.table, 'art': local.art,
            'manifest': local.prepared / 'manifest.json', 'fcl': local.prepared / 'replay.fcl',
            'baseline_art': local.baseline_art, 'comparison': local.comparison_path}[file]
    original = local.pure
    def mutate(**kwargs):
        report = original(**kwargs)
        path.write_bytes(path.read_bytes() + b'changed')
        return report
    monkeypatch.setattr(validator.jax_feature_validate, 'validate_arrays', mutate)
    with pytest.raises(ValueError, match='saved feature evidence'):
        run(local)


def test_source_gate_rechecked_and_cannot_promote(local):
    local.status.side_effect = [dict(source='first'), dict(source='changed')]
    with pytest.raises(ValueError, match='status evidence changed'):
        run(local)
    local.pure.assert_called_once()


def test_source_contradiction_before_jax(local):
    local.status.side_effect = ValueError('raw status contradiction')
    with pytest.raises(ValueError, match='raw status contradiction'):
        run(local)
    local.pure.assert_not_called()


def test_actual_array_mismatch_returned_not_silently_promoted(local):
    local.data['features']['momentum'] = local.data['features']['momentum'] * 0
    report = run(local)
    assert not report['measured_arrays_equal']
    assert not report['measured_metadata_verified'] and not report['measured_parity_established']
    assert report['selection']['paths'][PATHS[0]]['track_false_negatives'] == 2


def cli_args(local, output):
    return ['feature_saved', '--art', str(local.art), '--table', str(local.table),
            '--prepared', str(local.prepared), '--config', str(local.config),
            '--baseline-art', str(local.baseline_art), '--baseline-comparison', str(local.comparison_path),
            '--source-directory', str(local.source_directory), '--output', str(output)]


@pytest.mark.parametrize('equal', [True, False])
def test_cli_writes_real_report_before_mismatch_exit(local, monkeypatch, equal):
    if not equal:
        local.data['features']['momentum'] = local.data['features']['momentum'] * 0
    output = local.tmp / 'saved-report.json'
    monkeypatch.setattr(sys, 'argv', cli_args(local, output))
    if equal:
        validator.main()
    else:
        with pytest.raises(SystemExit) as error:
            validator.main()
        assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report['measured_arrays_equal'] is equal
    assert not report['measured_parity_established']


def test_cli_refuses_overwrite_before_reading_inputs(local, monkeypatch):
    output = local.tmp / 'saved-report.json'
    output.write_text('immutable previous report')
    check = Mock(side_effect=AssertionError('must not read'))
    monkeypatch.setattr(validator, 'validate', check)
    monkeypatch.setattr(sys, 'argv', cli_args(local, output))
    with pytest.raises(ValueError, match='overwrite'):
        validator.main()
    assert output.read_text() == 'immutable previous report'
    check.assert_not_called()


def test_x64_required_for_api(local):
    jax.config.update('jax_enable_x64', False)
    with pytest.raises(ValueError, match='JAX x64'):
        run(local)
    local.pure.assert_not_called()
