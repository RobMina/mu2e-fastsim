"""Admit a separately measured feature run and compare the two BO selections.

Requires completed full-history hybrid evidence and exact saved packed decisions.
This module does not build, launch art, reconstruct tracks or emulate upstream
modules. It never promotes the pure-array comparator's provenance flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

import feature_replay
import feature_runtime
import feature_status
import feature_table
import full_replay
import injection_campaign
import integrity
import jax_feature_validate
import jax_kalseed
import path_state
import quiet_replay as quiet
import replay
import timing


def selection_contracts(config_text, result):
    """Authenticate shared input, exact selected filters and producer-only suffix."""
    contracts = {path: jax_kalseed.compile_selection(config_text, path)
                 for path in jax_kalseed.PATH_FILTERS}
    physics = result['physics']
    sequences = {name: sequence for _, name, sequence in replay.paths(result)}
    if replay.plain(result['process_name']) != 'TriggerReplay':
        raise ValueError('features require actual TriggerReplay process')
    producer = physics['producers'].get('TrigKSFDe', {})
    if replay.plain(producer.get('module_type')) != 'LoopHelixFit':
        raise ValueError('shared fit must be the released LoopHelixFit producer')
    original = replay.production_config(config_text, 'Digitize')
    for path, label in jax_kalseed.PATH_FILTERS.items():
        module = physics['filters'].get(label)
        if injection_campaign.semantic(module) != injection_campaign.semantic(original['physics']['filters'][label]):
            raise ValueError('actual selection module differs from compiled production contract')
        if replay.plain(module['kalSeedCollections']) != ['TrigKSFDe']:
            raise ValueError('selection does not consume the single shared replay collection')
        sequence = sequences[path]
        if sequence.count(label) != 1 or sequence.count('TrigKSFDe') != 1:
            raise ValueError('need unique shared fit and selection in each path')
        suffix = sequence[sequence.index(label) + 1:]
        if not suffix:
            raise ValueError('expected the released MergeTriggerInfo suffix')
        for name in suffix:
            definition = physics['producers'].get(name)
            if (name in physics['filters'] or not isinstance(definition, dict)
                    or replay.plain(definition.get('module_type')) != 'MergeTriggerInfo'):
                raise ValueError('unsupported downstream module after selection')
    return contracts, {path: sequences[path] for path in contracts}


def validate(art, table, prepared, *, config_text, log_text, bundle,
             expected_library_sha256, expected_manifest_sha256, status_source_directory=None):
    """Measured I/O and decisions, optionally with released-source status checks."""
    art, table, prepared = Path(art), Path(table), Path(prepared)
    manifest = quiet._load(prepared / 'manifest.json')
    feature = manifest['feature_export']
    runtime = feature['runtime']
    saved, result = feature_replay.preflight(prepared, config_text, manifest['expected_events'],
        runtime['TRIGGER_AREA'], runtime['OFFLINE_IMAGE'], feature['exporter_source'], feature['exporter_sha256'])
    runtime_observation = feature_runtime.observe_worker(bundle, log_text,
        expected_library_sha256=expected_library_sha256, expected_manifest_sha256=expected_manifest_sha256)
    evidence = integrity.files([art, table, prepared / 'manifest.json', prepared / 'replay.fcl',
                                art.parent / 'replay.expanded.fcl'])
    parsed = timing.parse_log(log_text)
    if parsed['events'] != manifest['expected_events']:
        raise ValueError('finished art event count differs from feature population')
    expanded = replay.parse_expanded((art.parent / 'replay.expanded.fcl').read_text())
    for key in ('process_name', 'source', 'physics', 'outputs', 'services'):
        if injection_campaign.semantic(expanded.get(key)) != injection_campaign.semantic(result[key]):
            raise ValueError(f'executed feature {key} differs from admitted FCL')
    decisions = feature_replay.compare_decisions(art, Path(feature['baseline_art']), saved)
    keys = quiet._source_keys(result, manifest['expected_events'])
    schedules = quiet._schedule(config_text, result, manifest)
    words, decision_metadata = quiet._read_art(art, keys, manifest, schedules)
    data = feature_table.read(table, expected_keys=keys)
    status_check = (feature_status.validate(data, status_source_directory)
                    if status_source_directory is not None else None)
    contracts, sequences = selection_contracts(config_text, result)
    states, indices, _ = path_state.decode(words[manifest['replay_process']])
    names = [name for _, name, _ in schedules[1]]
    report = jax_feature_validate.validate_arrays(keys=data['keys'], features=data['features'],
        track_index=data['track_index'], info_present=data['info_present'], accepted_keys=data['accepted_keys'],
        states={path: states[:, names.index(path)] for path in contracts},
        indices={path: indices[:, names.index(path)] for path in contracts}, sequences=sequences,
        contracts=contracts, product_present=data['product_present'])
    after, _ = feature_replay.preflight(prepared, config_text, manifest['expected_events'],
        runtime['TRIGGER_AREA'], runtime['OFFLINE_IMAGE'], feature['exporter_source'], feature['exporter_sha256'])
    quiet.hybrid_history._same_files(evidence, integrity.files([row['path'] for row in evidence]), 'measured feature evidence')
    if saved != after:
        raise ValueError('feature preparation changed during validation')
    feature_runtime.observe_worker(bundle, log_text, expected_library_sha256=expected_library_sha256,
                                   expected_manifest_sha256=expected_manifest_sha256)
    if status_check is not None and status_check != feature_status.validate(data, status_source_directory):
        raise ValueError('released-source status evidence changed during validation')
    metadata = {name: value for name, value in data['metadata'].items()
                if name not in ('source_ordinal', 'product_id', 'raw_status', 't0_segment_index', 't0')}
    metadata['collection_present_events'] = int(np.count_nonzero(data['product_present']))
    metadata['collection_absent_events'] = int(np.count_nonzero(~data['product_present']))
    return {'events': len(keys), 'selection': report, 'feature_metadata': metadata,
        'decisions': decisions, 'decision_metadata': decision_metadata, 'runtime_observation': runtime_observation,
        'released_source_status_check': status_check,
        'evidence': evidence, 'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
        'measured_arrays_equal': report['arrays_equal'],
        'measured_metadata_verified': False, 'measured_parity_established': False,
        'scope': 'measured-array comparison after file/decision gates; full runtime admission handled separately; no upstream reconstruction emulation or independent confirmation',
        'limitations': ['schema status bits and exact collection identity require source/runtime admission',
                        'two BO final selections only; actual preceding reach retained, not emulated',
                        'primary-only fast history and stock calorimeter limitations remain']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('art', 'table', 'prepared', 'config', 'log', 'bundle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-library-sha256', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--status-source-directory', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse measured feature comparison overwrite')
    jax.config.update('jax_enable_x64', True)
    report = validate(args.art, args.table, args.prepared, config_text=args.config.read_text(),
        log_text=args.log.read_text(), bundle=args.bundle, expected_library_sha256=args.expected_library_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        status_source_directory=args.status_source_directory)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"measured arrays compared: {report['events']} events; arrays_equal={report['measured_arrays_equal']}; no full emulator claim")
    if not report['measured_arrays_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
