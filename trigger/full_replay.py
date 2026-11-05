"""Prepare stock-trigger second pass over complete regenerated digis in original parent order."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import integrity
import regeneration_census
import replay


def prepare(config_text, regeneration_manifest, directory, expected, production_comparison):
    if not production_comparison.get('equal') or not production_comparison.get('all_reference_events_present'):
        raise ValueError('full replay needs complete passing original-production overlap validation')
    if production_comparison.get('regenerated_events') != expected or expected != regeneration_manifest['original_processed_events']:
        raise ValueError('full replay requires the complete original processed-event count')
    count = regeneration_census.census(regeneration_manifest, directory, expected)
    if not regeneration_manifest['split_input_files'] or not count['complete_parent_event_ids_verified']:
        raise ValueError('full-history replay requires parent-boundary files and exact source identity coverage')
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    if production_comparison.get('config_sha256') != config_hash:
        raise ValueError('production overlap report uses a different configuration')
    candidate_files = integrity.files([directory / row['filename'] for row in count['files']])
    if production_comparison.get('candidate_files') != candidate_files:
        raise ValueError('production overlap report uses different candidate bytes or order')
    if not production_comparison.get('reference_files') or production_comparison['reference_files'] != integrity.files([row['path'] for row in production_comparison['reference_files']]):
        raise ValueError('production overlap reference bytes changed')
    config = replay.production_config(config_text, 'Digitize')
    result, manifest = replay.prepare(config)
    result['source']['fileNames'] = [str((directory / row['filename']).resolve()) for row in count['files']]
    manifest['original_selected_output_patterns'] = manifest.pop('stock_output_selection')
    if hashlib.sha256(config_text.encode()).hexdigest() != regeneration_manifest['config_sha256']:
        raise ValueError('regeneration configuration hash differs from full replay source')
    manifest['config_sha256'] = hashlib.sha256(config_text.encode()).hexdigest()
    manifest['expected_events'] = expected
    manifest['population'] = 'complete original preselection parents, regenerated digis with verified event identities/order'
    manifest['original_production_processed_events'] = regeneration_manifest['original_processed_events']
    manifest['regeneration_config_sha256'] = regeneration_manifest['config_sha256']
    manifest['parent_order_verified'] = True
    manifest['input_integrity'] = candidate_files
    manifest['replay_fcl_sha256'] = hashlib.sha256(render_config(result).encode()).hexdigest()
    return result, manifest, count


def render_config(result):
    return '\n'.join(f'{key}: {replay.render(value)}' for key, value in result.items()) + '\n'


def preflight(prepared, directory, config_text, expected):
    manifest = json.loads((prepared / 'manifest.json').read_text())
    fcl_text = (prepared / 'replay.fcl').read_text()
    if hashlib.sha256(fcl_text.encode()).hexdigest() != manifest['replay_fcl_sha256']:
        raise ValueError('replay FCL bytes differ from prepared manifest')
    result = replay.parse_expanded(fcl_text)
    if manifest['expected_events'] != expected or hashlib.sha256(config_text.encode()).hexdigest() != manifest['config_sha256']:
        raise ValueError('launch count or configuration differs from prepared manifest')
    inputs = result['source']['fileNames']
    if integrity.files(inputs) != manifest['input_integrity']:
        raise ValueError('launch input bytes/order differ from prepared manifest')
    root = directory.resolve()
    if any(not Path(path).resolve().is_relative_to(root) for path in inputs):
        raise ValueError('replay input falls outside bound regenerated directory')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--regeneration-manifest', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--production-comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse to overwrite prepared full replay')
    result, manifest, count = prepare(args.config.read_text(), json.loads(args.regeneration_manifest.read_text()),
                                      args.directory, args.expected_events, json.loads(args.production_comparison.read_text()))
    args.output.mkdir(parents=True)
    (args.output / 'replay.fcl').write_text(render_config(result))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'regeneration-census.json').write_text(json.dumps(count, indent=2) + '\n')
    print(f"prepared full-history replay events={count['events']} files={len(count['files'])}")


if __name__ == '__main__':
    main()
