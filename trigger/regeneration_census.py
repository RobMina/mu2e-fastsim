"""Verify finished regenerated output entries and parent-order event keys across file switches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import uproot

import event_keys


def census(manifest, directory, expected):
    if type(expected) is not int or expected <= 0:
        raise ValueError('expected event count must be positive')
    if manifest['split_input_files']:
        paths = list(directory.glob('unfiltered_*.art'))
        if not paths or not all(re.fullmatch(r'unfiltered_[0-9]+\.art', path.name) for path in paths):
            raise ValueError('missing or unexpected split output names')
        paths.sort(key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    else:
        paths = [directory / 'unfiltered.art']
    rows, all_keys = [], []
    if len(manifest['parents']) != len(manifest['local_parent_paths']):
        raise ValueError('parent identities and staged source counts differ')
    source_keys = []
    for parent_path in manifest['local_parent_paths']:
        with uproot.open(parent_path) as root:
            parent_keys = event_keys.read(root['Events'])
            order = np.lexsort((parent_keys[:, 2], parent_keys[:, 1], parent_keys[:, 0]))
            source_keys.append(parent_keys[order])
    for ordinal, path in enumerate(paths):
        with uproot.open(path) as root:
            tree = root['Events']
            if tree.classname != 'TTree':
                raise ValueError('regenerated Events is not a TTree')
            keys = event_keys.read(tree)
            if manifest['split_input_files'] and (ordinal >= len(source_keys) or
                    not np.array_equal(keys, source_keys[ordinal][:len(keys)])):
                raise ValueError('split outputs do not preserve original parent event IDs/run/subrun order')
            rows.append({'filename': path.name, 'events': tree.num_entries,
                         'first_event': keys[0].tolist(), 'last_event': keys[-1].tolist()})
            all_keys.append(keys)
    keys = np.concatenate(all_keys)
    if len(keys) != expected or np.unique(keys, axis=0).shape[0] != expected:
        raise ValueError('regenerated output event total differs or duplicate identities')
    source_keys = np.concatenate(source_keys)[:expected]
    if not np.array_equal(keys, source_keys):
        raise ValueError('regenerated event IDs/order differ from complete original parent stream')
    return {'events': len(keys), 'expected_events': expected, 'files': rows, 'equal': True,
            'root_event_ids_decoded': True, 'parent_prefix_event_ids_verified': True,
            'complete_parent_event_ids_verified': expected == manifest.get('original_processed_events'),
            'parent_order_verified': manifest['split_input_files']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = census(json.loads(args.manifest.read_text()), args.directory, args.expected_events)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"regenerated entries={report['events']} files={len(report['files'])} parent_order_verified={report['parent_order_verified']}")


if __name__ == '__main__':
    main()
