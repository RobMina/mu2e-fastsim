"""Local saved-art measured-array check, not full feature postflight admission.

Only supplied saved outputs and seven released source copies are read. No parent
history traversal, preparation rederivation, runtime inspection or art execution
is performed. Copied evidence is matched by bytes/SHA256, not old site paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import jax
import numpy as np

import event_keys
import feature_status
import feature_table
import feature_validate
import integrity
import jax_feature_validate
import path_state
import quiet_replay as quiet
import replay


def _same_content(saved, actual, label):
    """Match a copied file without dereferencing its original path spelling."""
    if (not isinstance(saved, dict) or type(saved.get('bytes')) is not int
            or saved['bytes'] != actual['bytes'] or saved.get('sha256') != actual['sha256']):
        raise ValueError(f'{label} bytes/SHA256 differ')


def _bound_copy(feature, field, actual):
    rows = [row for row in feature.get('baseline_evidence', [])
            if row.get('path') == feature.get(field)]
    if len(rows) != 1:
        raise ValueError(f'missing unique feature baseline evidence: {field}')
    _same_content(rows[0], actual, field)


def _saved_keys(art, expected):
    """Decode baseline ROOT identity independently of manifest/table/source IDs."""
    with quiet.uproot.open(art) as root:
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False)
                  if key.startswith('Events;')]
        if not cycles:
            raise ValueError('missing saved Events tree cycles')
        tree = root[f'Events;{max(cycles)}']
        if tree.classname != 'TTree' or tree.num_entries != expected:
            raise ValueError('highest saved Events count/type differs from manifest')
        keys = event_keys.read(tree)
    if len(keys) != expected:
        raise ValueError('decoded saved event count differs from manifest')
    return keys


def validate(art, table, prepared, *, config, baseline_art, baseline_comparison, source_directory):
    """Compare actual saved tables/decisions; return an unpromoted JSON report.

    ``config`` is the original full configuration file, and ``prepared`` holds
    byte-exact copies of the feature manifest and replay.fcl. Original paths in
    those files are evidence labels only; none are opened. Full postflight must
    still independently audit ordered source history, runtime and executed FCL.
    Array disagreement is returned, not raised, so the CLI saves it before exit 1.
    Evidence/identity/decision/status contradictions raise before array admission.
    """
    art, table, prepared, config, baseline_art, baseline_comparison = map(Path,
        (art, table, prepared, config, baseline_art, baseline_comparison))
    paths = [config, table, art, prepared / 'manifest.json', prepared / 'replay.fcl',
             baseline_art, baseline_comparison]
    evidence = integrity.files(paths)
    manifest = quiet._load(prepared / 'manifest.json')
    comparison = quiet._load(baseline_comparison)
    feature = manifest.get('feature_export', {})
    expected = manifest.get('expected_events')
    quiet._positive(expected)
    if (manifest.get('mode') != 'selection_feature_export'
            or manifest.get('full_production_history_certified') is not True
            or feature.get('baseline_kind') != 'fullhybrid'):
        raise ValueError('need saved feature preparation from a completed full-history hybrid')
    if manifest.get('config_sha256') != evidence[0]['sha256']:
        raise ValueError('original configuration bytes differ from manifest SHA256')
    if manifest.get('replay_fcl_sha256') != evidence[4]['sha256']:
        raise ValueError('feature replay FCL bytes differ from manifest SHA256')
    _same_content(comparison.get('saved_replay_file'), evidence[5], 'comparison saved baseline art')
    _bound_copy(feature, 'baseline_art', evidence[5])
    _bound_copy(feature, 'baseline_comparison', evidence[6])
    if comparison.get('events') != expected:
        raise ValueError('completed baseline comparison event count differs')
    for field in ('accepted_bits_crosschecked_with_log', 'root_event_ids_independently_decoded',
                  'trigger_schedule_registry_verified', 'full_production_history_certified',
                  'all_scheduled_paths_measured'):
        if comparison.get(field) is not True:
            raise ValueError(f'need completed baseline comparison evidence: {field}')
    if not re.fullmatch(r'[0-9a-f]{64}', comparison.get('decision_log_sha256', '')):
        raise ValueError('need bound completed baseline decision log SHA256')
    config_text = config.read_text()
    result = replay.parse_expanded((prepared / 'replay.fcl').read_text())
    schedules = quiet._schedule(config_text, result, manifest)
    if set(comparison.get('paths', {})) != set(manifest['paths']):
        raise ValueError('completed comparison must include all 52 physics paths')
    keys = _saved_keys(baseline_art, expected)
    baseline, baseline_metadata = quiet._read_art(baseline_art, keys, manifest, schedules)
    words, metadata = quiet._read_art(art, keys, manifest, schedules)
    for field in ('parameter_set_ids', 'schedule_registry', 'highest_cycle'):
        if comparison.get(field) != baseline_metadata[field]:
            raise ValueError('baseline ROOT schedule IDs/registry/cycle differ from completed comparison')
        if feature.get('baseline_metadata', {}).get(field) != baseline_metadata[field]:
            raise ValueError('baseline metadata differs from feature manifest')
    for field in ('parameter_set_ids', 'schedule_registry'):
        if metadata[field] != baseline_metadata[field]:
            raise ValueError('feature schedule IDs/registries differ from baseline')
    original, candidate = manifest['original_process'], manifest['replay_process']
    for process in (original, candidate):
        if not np.array_equal(words[process], baseline[process]):
            raise ValueError(f'feature {process} packed words differ from baseline')
    original_names = [name for _, name, _ in schedules[0]]
    rows = path_state.compare_matrices(
        baseline[original][:, [original_names.index(name) for name in manifest['paths']]],
        baseline[candidate], manifest['paths'],
        {name: sequence for _, name, sequence in schedules[0]},
        {name: sequence for _, name, sequence in schedules[1]})
    if comparison['paths'] != rows:
        raise ValueError('baseline packed states/endpoints differ from completed comparison')
    data = feature_table.read(table, expected_keys=keys)
    status_check = feature_status.validate(data, source_directory)
    if not jax.config.x64_enabled:
        raise ValueError('saved measured-array comparison requires JAX x64')
    contracts, sequences = feature_validate.selection_contracts(config_text, result)
    states, indices, _ = path_state.decode(words[candidate])
    names = [name for _, name, _ in schedules[1]]
    selection = jax_feature_validate.validate_arrays(
        keys=data['keys'], features=data['features'], track_index=data['track_index'],
        info_present=data['info_present'], accepted_keys=data['accepted_keys'],
        states={path: states[:, names.index(path)] for path in contracts},
        indices={path: indices[:, names.index(path)] for path in contracts},
        sequences=sequences, contracts=contracts, product_present=data['product_present'])
    counts, tracks = feature_table._ragged(data['track_index'], len(keys), 'track_index', 'uint64')
    nonvacuous = bool(tracks.size > 0 and all(
        row['tracks_compared'] > 0 for row in selection['paths'].values()))
    after_status = feature_status.validate(data, source_directory)
    if after_status != status_check:
        raise ValueError('released-source status evidence changed during validation')
    quiet.hybrid_history._same_files(evidence, integrity.files(paths), 'saved feature evidence')
    metadata_summary = {name: value for name, value in data['metadata'].items()
                        if name not in ('source_ordinal', 'product_id', 'raw_status', 't0_segment_index', 't0')}
    return {
        'events': len(keys), 'selection': selection,
        'counts': {'events': len(keys), 'tracks': int(tracks.size),
                   'zero_track_events': int(np.count_nonzero(counts == 0)),
                   'collection_present_events': int(np.count_nonzero(data['product_present'])),
                   'collection_absent_events': int(np.count_nonzero(~data['product_present']))},
        'nonvacuous_track_comparison': nonvacuous,
        'measured_arrays_equal': selection['arrays_equal'],
        'measured_metadata_verified': False, 'measured_parity_established': False,
        'original_packed_words_equal': True, 'replay_packed_words_equal': True,
        'raw_packed_equal': True, 'physics_paths': len(manifest['paths']),
        'original_scheduled_paths': len(schedules[0]), 'replay_scheduled_paths': len(schedules[1]),
        'baseline_paths': rows, 'baseline_decision_metadata': baseline_metadata,
        'decision_metadata': metadata, 'feature_metadata': metadata_summary,
        'released_source_status_check': status_check, 'evidence': evidence,
        'config_sha256': evidence[0]['sha256'],
        'root_event_ids_independently_decoded': True, 'trigger_schedule_registry_verified': True,
        'source_history_reaudited': False, 'source_order_admitted': False,
        'complete_parent_order_verified': False, 'runtime_admitted': False,
        'executed_configuration_verified': False, 'loaded_binary_authenticated': False,
        'physics_claim': False,
        'scope': 'local saved-art measured-array comparison only; not original full feature postflight',
        'limitations': [
            'saved feature IDs/order match independently decoded saved baseline ROOT, not a new source traversal',
            'source history/order, runtime and executed FCL NOT reaudited; full postflight remains separate',
            'prepared FCL byte binding is not proof of executed configuration or loaded binary',
            'source-bound status consistency is not independent feature extraction or ProductID authentication',
            'two final BO selections only; actual preceding fast reach retained, not emulated',
            'primary-only fast history, stock calorimeter and independent-confirmation limitations remain'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('art', 'table', 'prepared', 'config', 'baseline-art', 'baseline-comparison',
                 'source-directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse saved feature comparison overwrite')
    jax.config.update('jax_enable_x64', True)
    report = validate(args.art, args.table, args.prepared, config=args.config,
        baseline_art=args.baseline_art, baseline_comparison=args.baseline_comparison,
        source_directory=args.source_directory)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"saved arrays compared: {report['events']} events; arrays_equal={report['measured_arrays_equal']}; "
          'no runtime or measured-parity admission')
    if not report['measured_arrays_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
