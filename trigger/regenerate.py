"""Prepare an opt-in complete-parent digitization control, preserving the production module set."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import compare
import replay
import source_audit


def verify_history(text, audit):
    if compare.COMPLETION.findall(text) != ['0']:
        raise ValueError('need exactly one successful original production completion')
    opened = re.findall(r'Opened input file "([^"\n]+)"', text)
    expected = [parent['url'] for parent in audit['parents']]
    if opened != expected:
        raise ValueError('production opened-parent order differs from embedded configuration')
    seeds = re.findall(r'^services\.SeedService\.baseSeed:\s*([0-9]+)\s*$', text, re.M)
    if seeds != [audit['seed_service'].get('baseSeed')]:
        raise ValueError('production base seed differs from embedded configuration')
    counts = re.findall(r'^TrigReport Events total = (\d+) passed = (\d+) failed = (\d+)\s*$', text, re.M)
    if len(counts) != 1:
        raise ValueError('need a unique production event summary')
    total, passed, failed = map(int, counts[0])
    if total <= 0 or passed + failed != total:
        raise ValueError('invalid production event summary')
    return {'opened_parent_order_verified': True, 'original_processed_events': total,
            'production_log_sha256': hashlib.sha256(text.encode()).hexdigest()}


def prepare(config_text, production_log, parent_dir, split_input_files=False):
    audit = source_audit.audit(config_text)
    history = verify_history(production_log, audit)
    original = replay.production_config(config_text, 'Digitize')
    modules = {**original['physics'].get('producers', {}), **original['physics'].get('filters', {})}
    if any(replay.plain(module['module_type']) == 'Mu2eG4' for module in modules.values()):
        raise ValueError('digitization control must not contain a Mu2eG4 module')
    if set(original) != {'outputs', 'physics', 'services', 'source', 'trigger_paths'}:
        raise ValueError('unexpected embedded top-level process fields')
    if set(original['outputs']) != {'Output'} or replay.plain(original['outputs']['Output']['module_type']) != 'RootOutput':
        raise ValueError('expected the single original RootOutput label')
    if replay.plain(original['services']['scheduler'].get('num_schedules')) != '1' or replay.plain(original['services']['scheduler'].get('num_threads')) != '1':
        raise ValueError('only original single-schedule/thread production supported')
    metadata_paths = replay.plain(original['trigger_paths']['trigger_paths'])
    physics_paths = [f'{bit if bit is not None else "0"}:{name}' for bit, name, _ in replay.paths(original)]
    if metadata_paths != physics_paths:
        raise ValueError('embedded path metadata differs from scheduled physics paths')
    result = deepcopy(original)
    del result['trigger_paths']
    result['process_name'] = 'Digitize'
    result['source']['fileNames'] = [str(Path(parent_dir) / parent['did'].split(':', 1)[1]) for parent in audit['parents']]
    output = result['outputs']['Output']
    del output['SelectEvents']
    output['fileName'] = 'unfiltered.art'
    if split_input_files:
        if 'fileProperties' in output:
            raise ValueError('cannot replace existing production file-switch policy')
        output['fileName'] = 'unfiltered_%#.art'
        output['fileProperties'] = {'maxInputFiles': replay.Atom('1'), 'granularity': 'InputFile'}
    if result['physics'] != original['physics'] or result['services'] != original['services']:
        raise AssertionError('production modules, paths and services must be unchanged')
    receipt = {**history, 'config_sha256': audit['config_sha256'], 'parents': audit['parents'],
        'local_parent_paths': result['source']['fileNames'], 'process': 'Digitize',
        'split_input_files': split_input_files, 'output_pattern': output['fileName'],
        'preserved_physics_and_services': True, 'original_output_selection': audit['stock_output_selection'],
        'changes': ['unwrap Digitize only and set original process name',
                    'omit reconciled framework-generated trigger_paths metadata table',
                    'replace parent URLs with staged identical-byte parent paths, preserving order',
                    'original Output private filename changed; remove only its SelectEvents restriction',
                    'optional InputFile output switching preserves original parent boundaries for second-pass ordering' if split_input_files else 'single output file; second-pass subrun order requires explicit handling'],
        'scope': 'preparation only; no payload, seed-map, finished-entry or trigger-closure certification'}
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--production-log', type=Path, required=True)
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-input-files', action='store_true', help='switch existing output per input file to preserve parent order')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse to overwrite an existing prepared regeneration')
    result, receipt = prepare(args.config.read_text(), args.production_log.read_text(), args.parent_dir, args.split_input_files)
    args.output.mkdir(parents=True)
    (args.output / 'regenerate.fcl').write_text('\n'.join(f'{key}: {replay.render(value)}' for key, value in result.items()) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f"prepared complete-parent digitization: {len(receipt['parents'])} parents; no simulation launched")


if __name__ == '__main__':
    main()
