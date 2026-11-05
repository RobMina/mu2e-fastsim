"""Prepare and measure all-path stock trigger migrations on admitted same-event replacement digis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

import injection_campaign

import compare
import event_keys
import hybrid_input
import hybrid_history
import integrity
import path_state
import replay


def prepare(config_text, references, candidates, admission, expected, history=None):
    if not admission.get('same_event_handoff_metadata_passed') or admission.get('events') != expected:
        raise ValueError('need passing same-event handoff admission for exact count')
    if admission['reference_files'] != integrity.files(references) or admission['candidate_files'] != integrity.files(candidates):
        raise ValueError('admission input bytes or order changed')
    digest = hashlib.sha256(config_text.encode()).hexdigest()
    if admission['config_sha256'] != digest:
        raise ValueError('admission uses different production configuration')
    complete = False
    if history is not None:
        hybrid_history.verify(history, config_text, references, expected)
        complete = True
    config = replay.production_config(config_text, 'Digitize')
    result, manifest = replay.prepare(config, straw_tag=admission['straw_digi_tag'], calo_tag=admission['calo_digi_tag'])
    result['source']['fileNames'] = [str(Path(path).resolve()) for path in candidates]
    manifest['original_selected_output_patterns'] = manifest.pop('stock_output_selection')
    manifest.update(expected_events=expected, config_sha256=digest, input_integrity=admission['candidate_files'],
        reference_integrity=admission['reference_files'], population='explicit admitted source population, not generated exposure',
        full_production_history_certified=complete, injected_physics_validated=False)
    if complete:
        manifest['population'] = 'complete original stored preselection-parent history, not generated exposure'
        manifest['production_history_receipt'] = history
    text = '\n'.join(f'{key}: {replay.render(value)}' for key, value in result.items()) + '\n'
    manifest['replay_fcl_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return text, manifest


def preflight(prepared, config_text):
    manifest = json.loads((prepared / 'manifest.json').read_text())
    text = (prepared / 'replay.fcl').read_text()
    if hashlib.sha256(text.encode()).hexdigest() != manifest['replay_fcl_sha256'] or hashlib.sha256(config_text.encode()).hexdigest() != manifest['config_sha256']:
        raise ValueError('prepared replay or original configuration bytes changed')
    result = replay.parse_expanded(text)
    if integrity.files(result['source']['fileNames']) != manifest['input_integrity']:
        raise ValueError('hybrid source bytes or order changed')
    references = [row['path'] for row in manifest['reference_integrity']]
    if integrity.files(references) != manifest['reference_integrity']:
        raise ValueError('stock reference bytes changed')
    if manifest['full_production_history_certified']:
        hybrid_history.verify(manifest['production_history_receipt'], config_text, references, manifest['expected_events'])
    evidence = manifest.get('injection_campaign_evidence', [])
    if evidence and integrity.files([row['path'] for row in evidence]) != evidence:
        raise ValueError('injection campaign evidence changed')
    return manifest, result


def validate(art, prepared, config_text, log_text):
    manifest, result = preflight(prepared, config_text)
    if manifest.get('injection_campaign'):
        runtime = manifest['injection_campaign']['runtime']
        if (f"image={runtime['OFFLINE_IMAGE']}" not in log_text
                or f"MUSE_WORK_DIR={runtime['TRIGGER_AREA']} " not in log_text):
            raise ValueError('hybrid execution lacks pinned production runtime')
        expanded = replay.parse_expanded((Path(art).parent / 'replay.expanded.fcl').read_text())
        for key in ('process_name', 'source', 'physics', 'outputs'):
            if injection_campaign.semantic(expanded[key]) != injection_campaign.semantic(result[key]):
                raise ValueError(f'executed hybrid {key} differs from admitted FCL')
        for key, service in result['services'].items():
            if injection_campaign.semantic(expanded['services'].get(key)) != injection_campaign.semantic(service):
                raise ValueError(f'executed hybrid service {key} differs from admitted FCL')
    keys = []
    for filename in result['source']['fileNames']:
        with uproot.open(filename) as root:
            values = event_keys.read(root['Events'])
            keys.append(values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))])
    keys = np.concatenate(keys)
    if len(keys) != manifest['expected_events'] or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('hybrid event count or uniqueness differs')
    original_paths = replay.paths(replay.production_config(config_text, manifest['original_process']))
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    originals, ids = [], []
    for filename in result['source']['fileNames']:
        with uproot.open(filename, custom_classes=classes) as root:
            values = event_keys.read(root['Events'])
            words, pset = path_state.read_status(root['Events'], manifest['original_process'], len(original_paths))
            originals.append(words[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))])
            ids.append(pset)
    with uproot.open(art, custom_classes=classes) as root:
        retained, retained_id = path_state.read_status(root['Events'], manifest['original_process'], len(original_paths))
    if len(set(ids)) != 1 or retained_id != ids[0] or not np.array_equal(retained, np.concatenate(originals)):
        raise ValueError('hybrid replay changed retained original TriggerResults')
    state_report = path_state.compare_file(art, config_text, (prepared / 'replay.fcl').read_text(), manifest,
        log_text, keys.tolist(), len(keys))
    decision_report = compare.compare(log_text, manifest, len(keys), keys.tolist())
    if any(row['replay_states']['Ready'] or row['replay_states']['Exception'] for row in state_report['paths'].values()):
        raise ValueError('hybrid has unrun or exceptional physics paths')
    state_report['decision_migrations'] = decision_report['paths']
    state_report['expected_fast_stock_equality'] = False
    state_report['all_scheduled_paths_measured'] = True
    state_report['scope'] = manifest['population'] + '; primary-only diagnostic, no physics closure or independent-shard uncertainty'
    state_report['full_production_history_certified'] = manifest['full_production_history_certified']
    state_report['injection_campaign'] = manifest.get('injection_campaign')
    state_report['validation_evidence'] = integrity.files([prepared / 'manifest.json', prepared / 'replay.fcl'])
    state_report['saved_replay_file'] = integrity.digest(art)
    state_report['decision_log_sha256'] = hashlib.sha256(log_text.encode()).hexdigest()
    return state_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('prepare')
    create.add_argument('--reference', nargs='+', type=Path, required=True)
    create.add_argument('--candidate', nargs='+', type=Path, required=True)
    create.add_argument('--admission', type=Path, required=True)
    create.add_argument('--expected-events', type=int, required=True)
    create.add_argument('--production-history', type=Path, help='passing full original regeneration/overlap receipt')
    for command in (create, commands.add_parser('preflight'), commands.add_parser('validate')):
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--prepared', type=Path, required=True)
    commands.choices['preflight'].add_argument('--expected-events', type=int)
    commands.choices['preflight'].add_argument('--runtime-area')
    commands.choices['preflight'].add_argument('--runtime-image')
    check = commands.choices['validate']
    check.add_argument('--art', type=Path, required=True)
    check.add_argument('--log', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = args.config.read_text()
    if args.command == 'prepare':
        history = json.loads(args.production_history.read_text()) if args.production_history else None
        text, manifest = prepare(config, args.reference, args.candidate, json.loads(args.admission.read_text()), args.expected_events, history)
        if len(manifest['paths']) != 52:
            raise ValueError('Run 1A hybrid requires all 52 scheduled physics paths')
        if args.prepared.exists():
            raise ValueError('refuse hybrid preparation overwrite')
        args.prepared.mkdir(parents=True)
        (args.prepared / 'replay.fcl').write_text(text)
        (args.prepared / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    elif args.command == 'preflight':
        manifest, _ = preflight(args.prepared, config)
        if args.expected_events is not None and manifest['expected_events'] != args.expected_events:
            raise ValueError('launch count differs from admitted complete history')
        if manifest.get('injection_campaign'):
            runtime = manifest['injection_campaign']['runtime']
            if args.runtime_area != runtime['TRIGGER_AREA'] or args.runtime_image != runtime['OFFLINE_IMAGE']:
                raise ValueError('hybrid runtime differs from admitted injection runtime')
    else:
        if args.output.exists():
            raise ValueError('refuse hybrid result overwrite')
        report = validate(args.art, args.prepared, config, args.log.read_text())
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f"events={report['events']} all paths measured; migrations={report['accepted_mismatches']}, not an equality gate")


if __name__ == '__main__':
    main()
