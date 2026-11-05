"""Hash-bound proof of complete stored production history, not generated exposure.

This certifies source coverage and an existing production-overlap report. It does
not certify replacement-digi physics or run a trigger/transport simulation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import integrity
import regeneration_census


_SCOPE = ('complete original stored preselection-parent history, not generated exposure; '
          'source coverage and production overlap only, not hybrid physics certification')
_COMPARISON_FLAGS = ('equal', 'all_reference_events_present', 'root_event_ids_verified',
                     'original_output_selection_event_set_equal')
_CENSUS_FLAGS = ('equal', 'root_event_ids_decoded', 'complete_parent_event_ids_verified',
                 'parent_order_verified')


def _positive_count(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{label} must be a positive integer event count')
    return value


def _require_flags(report, names, label):
    if not isinstance(report, dict) or any(report.get(name) is not True for name in names):
        raise ValueError(f'{label} requires explicitly True flags: {", ".join(names)}')


def _rows(rows, label):
    """Canonicalize both sides of stored-path comparisons (including mount aliases)."""
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{label} needs nonempty ordered file integrity rows')
    result = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
                or not isinstance(row['path'], str) or not row['path']
                or type(row['bytes']) is not int or row['bytes'] < 0
                or not isinstance(row['sha256'], str) or len(row['sha256']) != 64):
            raise ValueError(f'{label} contains invalid file integrity rows')
        result.append({**row, 'path': str(Path(row['path']).resolve())})
    return result


def _same_files(stored, actual, label):
    left, right = _rows(stored, label), _rows(actual, label)
    def equal(a, b):
        same_path = a['path'] == b['path'] or Path(a['path']).samefile(b['path'])
        return same_path and (a['bytes'], a['sha256']) == (b['bytes'], b['sha256'])
    if len(left) != len(right) or any(not equal(a, b) for a, b in zip(left, right)):
        first = next(((a, b) for a, b in zip(left, right) if not equal(a, b)), (len(left), len(right)))
        raise ValueError(f'{label} bytes, paths or order changed: first difference {first}')


def _load(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f'evidence {path} must contain a JSON object')
    return value, {'path': str(path), 'bytes': len(data),
                   'sha256': hashlib.sha256(data).hexdigest()}


def certify(config_text, regeneration_manifest_path, directory, comparison_path, expected) -> dict:
    """Recheck complete parent/source coverage and bind every evidence byte.

    ``directory`` contains the numbered stock regeneration outputs; the manifest
    must still name every original staged parent. Paths may use aliases only when
    Path.resolve() identifies the same local file. No payloads are written.
    """
    _positive_count(expected, 'expected')
    manifest, manifest_file = _load(regeneration_manifest_path)
    production, comparison_file = _load(comparison_path)
    directory = Path(directory).resolve()
    _require_flags(manifest, ('opened_parent_order_verified', 'split_input_files'),
                   'complete-history regeneration manifest')
    if _positive_count(manifest.get('original_processed_events'), 'original processed events') != expected:
        raise ValueError('full history requires the complete original processed-event count')
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    if manifest.get('config_sha256') != config_hash or production.get('config_sha256') != config_hash:
        raise ValueError('production history uses a different configuration')
    _require_flags(production, _COMPARISON_FLAGS, 'production overlap')
    if _positive_count(production.get('regenerated_events'), 'regenerated events') != expected:
        raise ValueError('production overlap count differs from complete history')

    parents, local_paths = manifest.get('parents'), manifest.get('local_parent_paths')
    if (not isinstance(parents, list) or not parents or not isinstance(local_paths, list)
            or len(parents) != len(local_paths)
            or any(not isinstance(path, str) or not path for path in local_paths)):
        raise ValueError('need all original staged parent paths in production order')
    parent_paths = [Path(path).resolve() for path in local_paths]
    if len(set(parent_paths)) != len(parent_paths):
        raise ValueError('duplicate original staged parent paths')
    parent_files = integrity.files(parent_paths)
    evidence = [manifest_file, comparison_file, *parent_files]

    reference_rows = _rows(production.get('reference_files'), 'original production reference')
    _same_files(reference_rows, integrity.files([row['path'] for row in reference_rows]),
                'original production reference')

    # Independently count every original parent to reject prefix-only census claims.
    source_files = []
    for path in parent_paths:
        with regeneration_census.uproot.open(path) as root:
            tree = root['Events']
            if tree.classname != 'TTree':
                raise ValueError('original parent Events is not an art TTree')
            entries = _positive_count(int(tree.num_entries), 'original parent Events')
            source_files.append({'path': str(path), 'events': entries})
    source_events = sum(row['events'] for row in source_files)
    if source_events != expected:
        raise ValueError('all original source events must be present, not a parent prefix')
    census_manifest = {**manifest, 'local_parent_paths': [str(path) for path in parent_paths]}
    count = regeneration_census.census(census_manifest, directory, expected)
    _require_flags(count, _CENSUS_FLAGS, 'complete parent census')
    if (type(count.get('events')) is not int or count['events'] != expected
            or type(count.get('expected_events')) is not int or count['expected_events'] != expected):
        raise ValueError('parent census event counts differ from complete history')
    outputs = count.get('files')
    if not isinstance(outputs, list) or len(outputs) != len(source_files):
        raise ValueError('parent census must contain every original parent boundary in order')
    candidate_paths = []
    for output, source in zip(outputs, source_files):
        if (type(output.get('events')) is not int or output['events'] != source['events']):
            raise ValueError('regenerated parent event count differs; prefixes are not full history')
        filename = output.get('filename')
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError('parent census output must be a local filename')
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('stock output falls outside the regenerated directory')
        candidate_paths.append(path)
    if len(set(candidate_paths)) != len(candidate_paths):
        raise ValueError('duplicate stock output paths')
    candidate_files = integrity.files(candidate_paths)
    _same_files(production.get('candidate_files'), candidate_files, 'ordered stock candidate')
    # Reject changes while the census was reading the same evidence.
    _same_files(evidence, integrity.files([row['path'] for row in evidence]), 'history evidence')
    _same_files(reference_rows, integrity.files([row['path'] for row in reference_rows]),
                'original production reference')
    count = deepcopy(count)
    count.update(original_source_events=source_events, original_parent_files=source_files)
    result = deepcopy(production)
    result.update(full_parent_history_verified=True, parent_census=count, evidence_files=evidence,
                  regeneration_manifest_path=manifest_file['path'],
                  production_comparison_path=comparison_file['path'],
                  regenerated_directory=str(directory), scope=_SCOPE)
    if 'scope' in production:
        result['production_comparison_scope'] = production['scope']
    return result


def _canonical_history(history):
    result = deepcopy(history)
    for name in ('candidate_files', 'reference_files', 'evidence_files'):
        result[name] = _rows(result.get(name), name)
    for name in ('regeneration_manifest_path', 'production_comparison_path', 'regenerated_directory'):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f'missing complete-history evidence location: {name}')
        result[name] = str(Path(result[name]).resolve())
    count = result.get('parent_census')
    if not isinstance(count, dict) or not isinstance(count.get('original_parent_files'), list):
        raise ValueError('missing complete parent census')
    for row in count['original_parent_files']:
        row['path'] = str(Path(row['path']).resolve())
    return result


def verify(history, config_text, references, expected) -> dict:
    """Independently re-certify a receipt for hybrid prepare/preflight.

    ``references`` are ordered stock input paths (or integrity rows), NOT the
    selected original-production references. All evidence remains required and
    readable: asserted booleans alone can never authorize a complete-history arm.
    Returns a freshly checked receipt; does not mutate the supplied one.
    """
    _positive_count(expected, 'expected')
    _require_flags(history, (*_COMPARISON_FLAGS, 'full_parent_history_verified'), 'complete history')
    saved = _canonical_history(history)
    _require_flags(saved['parent_census'], _CENSUS_FLAGS, 'saved complete parent census')
    for value in (saved.get('regenerated_events'), saved['parent_census'].get('events'),
                  saved['parent_census'].get('expected_events'),
                  saved['parent_census'].get('original_source_events')):
        if _positive_count(value, 'saved history event count') != expected:
            raise ValueError('saved history count differs from complete history')
    if saved.get('config_sha256') != hashlib.sha256(config_text.encode()).hexdigest():
        raise ValueError('saved history uses a different configuration')
    _same_files(saved['evidence_files'],
                integrity.files([row['path'] for row in saved['evidence_files']]), 'history evidence')
    if isinstance(references, (str, Path)):
        references = [references]
    else:
        references = list(references)
    if references and isinstance(references[0], dict):
        rows = _rows(references, 'hybrid stock reference')
        actual = integrity.files([row['path'] for row in rows])
        _same_files(rows, actual, 'hybrid stock reference')
    else:
        actual = integrity.files(references)
    _same_files(saved['candidate_files'], actual, 'ordered hybrid stock reference')
    fresh = certify(config_text, saved['regeneration_manifest_path'], saved['regenerated_directory'],
                    saved['production_comparison_path'], expected)
    fresh_canonical = _canonical_history(fresh)
    for name in ('candidate_files', 'reference_files', 'evidence_files'):
        _same_files(saved[name], fresh_canonical[name], name)
        fresh_canonical[name] = saved[name]
    if saved != fresh_canonical:
        raise ValueError('complete-history receipt differs from independently verified evidence')
    return fresh


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'regeneration-manifest', 'directory', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse history evidence overwrite')
    receipt = certify(args.config.read_text(), args.regeneration_manifest, args.directory,
                      args.comparison, args.expected_events)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f"verified complete stored history: {receipt['regenerated_events']} events, not generated exposure")


if __name__ == '__main__':
    main()
