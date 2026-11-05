"""Compare original and regenerated production TriggerResults plus serialized digi/timing bytes by event key."""
from __future__ import annotations

import argparse
import hashlib
import fnmatch
import json
import integrity
import regeneration_census
from pathlib import Path

import numpy as np
import uproot

import event_keys
import path_state
import replay
import serialized


def status_data(paths, names, classes):
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    keys, words, ids = [], [], []
    for path in paths:
        with uproot.open(path, custom_classes=classes) as root:
            path_state.check_schema(root)
            tree = root['Events']
            keys.append(event_keys.read(tree))
            matrix, pset = path_state.read_status(tree, 'Digitize', len(names))
            words.append(matrix)
            ids.append(pset)
    if len(set(ids)) != 1:
        raise ValueError('trigger schedule IDs change across files')
    keys = np.concatenate(keys)
    if np.unique(keys, axis=0).shape[0] != len(keys):
        raise ValueError('duplicate event identity across status files')
    return keys, np.concatenate(words), ids[0]


def compare(reference, candidate, config_text, expected_events, require_all_reference=False):
    config = replay.production_config(config_text, 'Digitize')
    schedule = replay.paths(config)
    names = [name for _, name, _ in schedule]
    sequences = {name: sequence for _, name, sequence in schedule}
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    keys_a, words_a, pset_a = status_data(reference, names, classes)
    keys_b, words_b, pset_b = status_data(candidate, names, classes)
    if len(keys_b) != expected_events or expected_events <= 0:
        raise ValueError('regenerated finished Events count differs from expected')
    dtype = np.dtype([('run', '<u8'), ('subrun', '<u8'), ('event', '<u8')])
    ids_a = np.ascontiguousarray(keys_a).view(dtype).ravel()
    ids_b = np.ascontiguousarray(keys_b).view(dtype).ravel()
    _, index_a, index_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    if len(index_a) == 0 or (require_all_reference and len(index_a) != len(keys_a)):
        raise ValueError('no overlap or incomplete original-event coverage')
    if pset_a != pset_b:
        raise ValueError('original/regenerated trigger schedule parameter-set IDs differ')
    selection = replay.plain(config['outputs']['Output']['SelectEvents'])
    columns = sorted({col for pattern in selection for col, name in enumerate(names) if fnmatch.fnmatchcase(name, pattern)})
    if not columns:
        raise ValueError('original output selector matches no configured paths')
    selected = np.any(path_state.decode(words_b)[2][:, columns], axis=1)
    selected_ids = np.sort(ids_b[selected])
    selection_equal = np.array_equal(selected_ids, np.sort(ids_a)) if require_all_reference else None
    rows = path_state.compare_matrices(words_a[index_a], words_b[index_b], names, sequences, sequences)
    products = serialized.compare_products(reference, candidate, len(index_a))
    mismatches = sum(row['packed_status_mismatches'] for row in rows.values())
    return {'reference_events': len(keys_a), 'regenerated_events': len(keys_b), 'overlap_events': len(index_a),
            'all_reference_events_present': len(index_a) == len(keys_a), 'parameter_set_id_equal': True,
            'config_sha256': hashlib.sha256(config_text.encode()).hexdigest(),
            'reference_files': integrity.files(reference), 'candidate_files': integrity.files(candidate),
            'root_event_ids_verified': True, 'paths': rows, 'packed_status_mismatches': mismatches,
            'products': products, 'original_output_selection_event_set_equal': selection_equal,
            'regenerated_original_selection_events': int(np.count_nonzero(selected)),
            'equal': mismatches == 0 and products['equal'] and selection_equal is not False,
            'scope': 'production regeneration on original-event overlap; not second-pass trigger replay or fast-digi validation',
            'caveat': 'matching trigger schedule IDs across files does not independently resolve them to embedded configuration'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--require-all-reference', action='store_true')
    parser.add_argument('--regeneration-manifest', type=Path, help='candidate is a split output directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate
    if args.regeneration_manifest:
        count = regeneration_census.census(json.loads(args.regeneration_manifest.read_text()), candidate, args.expected_events)
        candidate = [candidate / row['filename'] for row in count['files']]
    report = compare(args.reference, candidate, args.config.read_text(), args.expected_events, args.require_all_reference)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"regenerated={report['regenerated_events']} overlap={report['overlap_events']} packed_mismatches={report['packed_status_mismatches']} equal={report['equal']}")
    for name, row in report['products']['products'].items():
        print(f"{name} mismatches={row['mismatches']}")
    raise SystemExit(0 if report['equal'] else 1)


if __name__ == '__main__':
    main()
