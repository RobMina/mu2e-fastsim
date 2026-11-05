"""Hash-bound quiet timing companion to completed full-history stock or hybrid replay.

No art execution, baseline rerun, physics-closure or end-to-end speedup claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import uproot

import compare
import event_keys
import full_replay
import hybrid_history
import injection_campaign
import integrity
import path_state
import replay
import schedule_registry
import timing


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _positive(expected):
    if type(expected) is not int or expected <= 0:
        raise ValueError('expected events must be a positive integer')


def _runtime(area, image):
    if not all(isinstance(value, str) and value and not re.search(r'\s', value)
               for value in (area, image)):
        raise ValueError('explicit runtime area and image pins are required')
    return {'TRIGGER_AREA': area, 'OFFLINE_IMAGE': image}


def _schedule(config_text, result, manifest):
    original, candidate, names, bits = compare.validate_manifest(manifest)
    if original != 'Digitize' or candidate != 'TriggerReplay' or replay.plain(result['process_name']) != candidate:
        raise ValueError('need the exact Digitize/TriggerReplay baseline processes')
    original_paths = replay.paths(replay.production_config(config_text, original))
    candidate_paths = replay.paths(result)
    scheduled = [(int(bit), name) for bit, name, _ in original_paths
                 if name != 'DigitizePath' and not name.startswith('Triggerable')]
    if (len(names) != 52 or scheduled != list(zip(bits, names))
            or [(int(bit), name) for bit, name, _ in candidate_paths] != scheduled):
        raise ValueError('need exactly all 52 baseline physics paths in scheduled bit/name order')
    return original_paths, candidate_paths


def _variant(result):
    quiet = deepcopy(result)
    physics = quiet['physics']
    printer = physics.get('analyzers', {}).get('decisions', {})
    if (replay.plain(printer.get('module_type')) != 'PrintModule'
            or replay.plain(physics.get('DecisionOutput')) != ['decisions', 'DecisionProducts']
            or replay.plain(physics.get('end_paths')) != ['DecisionOutput']
            or 'DecisionProducts' not in quiet['outputs']):
        raise ValueError('baseline must have the verbose decisions PrintModule and DecisionProducts output')
    del physics['analyzers']['decisions']
    physics['DecisionOutput'] = ['DecisionProducts']
    # Only an existing known ArtReport category is changed; no logging destinations are introduced.
    destinations = quiet['services'].get('message', {}).get('destinations', {})
    for destination in destinations.values():
        if isinstance(destination, dict):
            category = destination.get('categories', {}).get('ArtReport')
            if isinstance(category, dict):
                category['limit'] = replay.Atom('0')
    return quiet


def _inventory(config_text, result):
    inventory = replay.inventory(replay.production_config(config_text, 'Digitize'))
    inventory['digitize_modules'] = []
    modules = {**result['physics']['producers'], **result['physics']['filters']}
    for row, (_, name, sequence) in zip(inventory['paths'], replay.paths(result)):
        if row['path'] != name:
            raise ValueError('inventory order differs from scheduled replay')
        row['sequence'] = sequence
        row['modules'] = {label: replay.plain(modules[label]) for label in sequence}
    timing.trigger_modules(inventory, [])
    return inventory


def _source_keys(result, expected):
    chunks = []
    for filename in result['source']['fileNames']:
        with uproot.open(filename) as root:
            values = event_keys.read(root['Events'])
        chunks.append(values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))])
    if not chunks:
        raise ValueError('empty source history')
    keys = np.concatenate(chunks)
    if len(keys) != expected or len(np.unique(keys, axis=0)) != expected:
        raise ValueError('source event count or uniqueness differs from complete history')
    return keys


def _read_art(art, keys, manifest, schedules):
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    words, ids, registry = {}, {}, {}
    with uproot.open(art, custom_classes=classes) as root:
        path_state.check_schema(root)
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False) if key.startswith('Events;')]
        if not cycles:
            raise ValueError('missing Events tree cycles')
        highest = max(cycles)
        tree = root[f'Events;{highest}']
        if tree.classname != 'TTree' or tree.num_entries != len(keys):
            raise ValueError('highest-cycle Events count/type differs from complete history')
        if not np.array_equal(event_keys.read(tree), keys):
            raise ValueError('saved ROOT IDs/order differ from independent parent traversal')
        for process, paths in zip((manifest['original_process'], manifest['replay_process']), schedules):
            matrix, pset = path_state.read_status(tree, process, len(paths))
            if matrix.shape != (len(keys), len(paths)):
                raise ValueError('saved status matrix shape differs from schedule/count')
            names = [name for _, name, _ in paths]
            physics = matrix[:, [names.index(name) for name in manifest['paths']]]
            states, _, _ = path_state.decode(physics)
            if np.any((states == 0) | (states == 3)):
                raise ValueError('Ready/Exception physics paths are not admitted')
            words[process], ids[process] = matrix, pset
            registry[process] = schedule_registry.verify(root, pset, paths)
    return words, {'parameter_set_ids': ids, 'schedule_registry': registry, 'highest_cycle': highest}


def _evidence(config_path, prepared, art, comparison):
    return integrity.files([config_path, prepared / 'manifest.json', prepared / 'replay.fcl', comparison, art])


def prepare(config_path, baseline_prepared, baseline_art, baseline_comparison, *,
            expected_events, runtime_area, runtime_image):
    """Return (quiet configuration, manifest); read evidence but write nothing."""
    _positive(expected_events)
    runtime = _runtime(runtime_area, runtime_image)
    config_path, baseline_prepared, baseline_art, baseline_comparison = map(
        lambda path: Path(path).resolve(), (config_path, baseline_prepared, baseline_art, baseline_comparison))
    evidence = _evidence(config_path, baseline_prepared, baseline_art, baseline_comparison)
    config_text = config_path.read_text()
    baseline = _load(baseline_prepared / 'manifest.json')
    comparison = _load(baseline_comparison)
    text = (baseline_prepared / 'replay.fcl').read_text()
    if (baseline.get('config_sha256') != _sha(config_text)
            or baseline.get('replay_fcl_sha256') != _sha(text)
            or baseline.get('expected_events') != expected_events
            or comparison.get('events') != expected_events):
        raise ValueError('baseline configuration/FCL hashes or complete event count differ')
    hybrid_history._same_files([comparison.get('saved_replay_file')],
        [integrity.digest(baseline_art)], 'comparison saved baseline art')
    for field in ('accepted_bits_crosschecked_with_log', 'root_event_ids_independently_decoded',
                  'trigger_schedule_registry_verified'):
        if comparison.get(field) is not True:
            raise ValueError(f'need completed verbose comparison evidence: {field}')
    if not re.fullmatch(r'[0-9a-f]{64}', comparison.get('decision_log_sha256', '')):
        raise ValueError('need bound completed verbose decision log evidence')
    result = replay.parse_expanded(text)
    schedules = _schedule(config_text, result, baseline)
    if set(comparison.get('paths', {})) != set(baseline['paths']):
        raise ValueError('comparison needs all 52 baseline paths')
    if baseline.get('full_production_history_certified') is True:
        kind = 'fullhybrid'
        for field in ('full_production_history_certified', 'all_scheduled_paths_measured'):
            if comparison.get(field) is not True:
                raise ValueError(f'need completed hybrid evidence: {field}')
        hybrid_history._same_files(comparison.get('validation_evidence'), integrity.files(
            [baseline_prepared / 'manifest.json', baseline_prepared / 'replay.fcl']),
            'hybrid comparison prepared evidence')
    else:
        kind = 'fullstock'
        for field in ('complete_first_pass_words_preserved', 'complete_parent_order_verified',
                      'all_menu_accepted_equal', 'all_menu_states_and_endpoints_equal'):
            if comparison.get(field) is not True:
                raise ValueError(f'need completed equal full-history stock baseline: {field}')
        if baseline.get('parent_order_verified') is not True:
            raise ValueError('stock baseline lacks complete parent history')
        if (comparison.get('config_sha256') != _sha(config_text)
                or comparison.get('replay_fcl_sha256') != _sha(text)):
            raise ValueError('stock comparison configuration/FCL evidence differs')
        if any(comparison.get(field) != 0 for field in
               ('accepted_mismatches', 'state_mismatches', 'wasrun_mismatches', 'mapped_endpoint_mismatches')):
            raise ValueError('stock baseline must be equal in accepted/state/wasrun/mapped endpoint')
    if baseline.get('mode') != 'decision_comparison':
        raise ValueError('need a completed verbose baseline, not a quiet companion')
    pinned = (baseline.get('injection_campaign') or {}).get('runtime')
    if pinned and any(pinned.get(key) != value for key, value in runtime.items()):
        raise ValueError('runtime pins differ from baseline injection runtime')
    sources = result['source']['fileNames']
    hybrid_history._same_files(baseline.get('input_integrity'), integrity.files(sources),
        'baseline ordered source')
    extra = baseline.get('reference_integrity', []) + baseline.get('injection_campaign_evidence', [])
    if extra:
        hybrid_history._same_files(extra, integrity.files([row['path'] for row in extra]),
            'baseline reference or campaign evidence')
    if kind == 'fullhybrid':
        hybrid_history.verify(baseline.get('production_history_receipt'), config_text,
            [row['path'] for row in baseline.get('reference_integrity', [])], expected_events)
    keys = _source_keys(result, expected_events)
    words, metadata = _read_art(baseline_art, keys, baseline, schedules)
    original_names = [name for _, name, _ in schedules[0]]
    rows = path_state.compare_matrices(
        words['Digitize'][:, [original_names.index(name) for name in baseline['paths']]],
        words['TriggerReplay'], baseline['paths'],
        {name: sequence for _, name, sequence in schedules[0]},
        {name: sequence for _, name, sequence in schedules[1]})
    if comparison['paths'] != rows:
        raise ValueError('baseline packed states/endpoints differ from completed comparison')
    if kind == 'fullstock' and any(row[field] for row in rows.values() for field in
            ('accepted_mismatches', 'state_mismatches', 'wasrun_mismatches', 'mapped_endpoint_mismatches')):
        raise ValueError('stock baseline decoded paths are not equal')
    if any(comparison.get(field) != value for field, value in metadata.items()):
        raise ValueError('baseline ROOT schedule IDs/registry/cycle differ from completed comparison')
    quiet = _variant(result)
    manifest = deepcopy(baseline)
    manifest.update(mode='quiet_timing_companion', replay_fcl_sha256=_sha(full_replay.render_config(quiet)))
    manifest['quiet_companion'] = {
        'version': 1, 'baseline_kind': kind, 'runtime': runtime,
        'config_path': str(config_path.resolve()), 'baseline_prepared': str(baseline_prepared.resolve()),
        'baseline_art': str(baseline_art.resolve()), 'baseline_comparison': str(baseline_comparison.resolve()),
        'baseline_evidence': evidence, 'baseline_metadata': metadata,
        'inventory': _inventory(config_text, quiet),
        'transformation': 'remove decisions PrintModule and its DecisionOutput entry; existing ArtReport limits only',
        'physics_claim': False}
    hybrid_history._same_files(evidence,
        _evidence(config_path, baseline_prepared, baseline_art, baseline_comparison), 'baseline evidence')
    hybrid_history._same_files(baseline['input_integrity'], integrity.files(sources), 'baseline ordered source')
    if extra:
        hybrid_history._same_files(extra, integrity.files([row['path'] for row in extra]),
            'baseline reference or campaign evidence')
    if kind == 'fullhybrid':
        hybrid_history.verify(baseline.get('production_history_receipt'), config_text,
            [row['path'] for row in baseline.get('reference_integrity', [])], expected_events)
    return quiet, manifest


def preflight(prepared, config_text, expected_events, runtime_area, runtime_image):
    """Re-admit the saved baseline and deterministically rederive the entire quiet preparation."""
    prepared = Path(prepared).resolve()
    _positive(expected_events)
    manifest = _load(prepared / 'manifest.json')
    quiet = manifest['quiet_companion']
    if config_text != Path(quiet['config_path']).read_text():
        raise ValueError('launch original configuration bytes differ')
    result, admitted = prepare(quiet['config_path'], quiet['baseline_prepared'], quiet['baseline_art'],
        quiet['baseline_comparison'], expected_events=expected_events,
        runtime_area=runtime_area, runtime_image=runtime_image)
    hybrid_history._same_files(quiet['baseline_evidence'],
        admitted['quiet_companion']['baseline_evidence'], 'baseline evidence')
    admitted['quiet_companion']['baseline_evidence'] = deepcopy(quiet['baseline_evidence'])
    if (manifest != admitted or (prepared / 'replay.fcl').read_text() != full_replay.render_config(result)
            or _load(prepared / 'replay_inventory.json') != admitted['quiet_companion']['inventory']):
        raise ValueError('quiet preparation differs from deterministic admitted variant')
    return manifest, result


def validate(art, prepared, config_text, log_text):
    """Validate saved quiet ROOT decisions and timing, never parse verbose event decisions."""
    art, prepared = Path(art), Path(prepared)
    saved = _load(prepared / 'manifest.json')
    runtime = saved['quiet_companion']['runtime']
    arguments = (prepared, config_text, saved['expected_events'], runtime['TRIGGER_AREA'], runtime['OFFLINE_IMAGE'])
    prepared_evidence = integrity.files([prepared / name for name in ('manifest.json', 'replay.fcl', 'replay_inventory.json')])
    manifest, result = preflight(*arguments)
    parsed = timing.parse_log(log_text)
    if parsed['events'] != manifest['expected_events']:
        raise ValueError('quiet completed timing event count differs')
    for label, value in (('image', runtime['OFFLINE_IMAGE']), ('MUSE_WORK_DIR', runtime['TRIGGER_AREA'])):
        if not re.search(r'(?<!\S)' + re.escape(label + '=' + value) + r'(?=\s|$)', log_text):
            raise ValueError('quiet execution lacks exact pinned runtime image/release area')
    expanded_path = art.parent / 'replay.expanded.fcl'
    execution_evidence = integrity.files([art, expanded_path])
    expanded = replay.parse_expanded(expanded_path.read_text())
    for key in ('process_name', 'source', 'physics', 'outputs', 'services'):
        if injection_campaign.semantic(expanded.get(key)) != injection_campaign.semantic(result[key]):
            raise ValueError(f'executed quiet {key} differs from admitted FCL')
    schedules = _schedule(config_text, result, manifest)
    keys = _source_keys(result, manifest['expected_events'])
    baseline_art = manifest['quiet_companion']['baseline_art']
    baseline_words, baseline_metadata = _read_art(baseline_art, keys, manifest, schedules)
    words, metadata = _read_art(art, keys, manifest, schedules)
    for field in ('parameter_set_ids', 'schedule_registry'):
        if metadata[field] != baseline_metadata[field]:
            raise ValueError('quiet schedule IDs/registry differ from baseline')
    if any(not np.array_equal(words[process], matrix) for process, matrix in baseline_words.items()):
        raise ValueError('quiet original/replay packed words differ from verbose baseline')
    report = timing.diagnose(log_text, manifest['quiet_companion']['inventory'])
    after, _ = preflight(*arguments)
    hybrid_history._same_files(execution_evidence, integrity.files([art, expanded_path]), 'quiet execution evidence')
    hybrid_history._same_files(prepared_evidence,
        integrity.files([prepared / name for name in ('manifest.json', 'replay.fcl', 'replay_inventory.json')]),
        'quiet validation evidence')
    if after != manifest:
        raise ValueError('quiet inputs/evidence changed during validation')
    return {'events': len(keys), 'paths': manifest['paths'], **metadata,
        'baseline_kind': manifest['quiet_companion']['baseline_kind'],
        'baseline_evidence': manifest['quiet_companion']['baseline_evidence'],
        'validation_evidence': prepared_evidence, 'execution_evidence': execution_evidence,
        'saved_replay_file': execution_evidence[0], 'timing_log_sha256': _sha(log_text),
        'original_packed_words_equal': True, 'replay_packed_words_equal': True, 'raw_packed_equal': True,
        'root_event_ids_independently_decoded': True, 'complete_parent_order_verified': True,
        'trigger_schedule_registry_verified': True, 'all_scheduled_paths_measured': True,
        'executed_configuration_verified': True, 'runtime_pins_verified': True,
        'timing': report, 'physics_claim': False,
        'caveats': ['same inputs/schedule support a same-hardware quiet timing measurement; the verbose baseline is not a speed benchmark',
                    'production hardware may differ; no cross-hardware throughput or speedup claim',
                    'retained stock calorimeter and primary-only hybrid limitations remain; not full-fast or JAX end-to-end timing',
                    'shared-module first-execution attribution is not independent path latency; no physics closure claim']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare')
    for name in ('baseline-prepared', 'baseline-art', 'baseline-comparison'):
        create.add_argument('--' + name, type=Path, required=True)
    check = commands.add_parser('preflight')
    measure = commands.add_parser('validate')
    for command in (create, check, measure):
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--prepared', type=Path, required=True)
        command.add_argument('--expected-events', type=int, required=True)
        command.add_argument('--runtime-area', required=True)
        command.add_argument('--runtime-image', required=True)
    for name in ('art', 'log', 'output'):
        measure.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    _positive(args.expected_events)
    if args.command == 'prepare':
        if args.prepared.exists():
            raise ValueError('refuse quiet preparation overwrite')
        result, manifest = prepare(args.config, args.baseline_prepared, args.baseline_art, args.baseline_comparison,
            expected_events=args.expected_events, runtime_area=args.runtime_area, runtime_image=args.runtime_image)
        args.prepared.mkdir(parents=True)
        (args.prepared / 'replay.fcl').write_text(full_replay.render_config(result))
        (args.prepared / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (args.prepared / 'replay_inventory.json').write_text(json.dumps(manifest['quiet_companion']['inventory'], indent=2) + '\n')
        print('prepared quiet companion: no art run or physics claim')
    else:
        if args.command == 'validate' and args.output.exists():
            raise ValueError('refuse quiet result overwrite')
        config_text = args.config.read_text()
        preflight(args.prepared, config_text, args.expected_events, args.runtime_area, args.runtime_image)
        if args.command == 'validate':
            report = validate(args.art, args.prepared, config_text, args.log.read_text())
            with args.output.open('x') as output:
                output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(f"quiet timing validated: {report['events']} events; exact baseline packed-word parity, not a speedup claim")
        else:
            print('quiet preflight passed: evidence/configuration admission only, no physics claim')


if __name__ == '__main__':
    main()
