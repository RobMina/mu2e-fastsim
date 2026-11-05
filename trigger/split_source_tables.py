"""Split already-exported truth by exact original parent keys without rerunning art."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

import event_keys
import integrity


KEYS = ['run', 'subrun', 'event']


def structured(keys):
    keys = np.ascontiguousarray(keys, dtype='<u8')
    dtype = np.dtype([(key, '<u8') for key in KEYS[:keys.shape[1]]])
    return keys.view(dtype).ravel()


def split(sources, tables, output):
    if (output.exists() or not sources or not tables
            or len(set(map(str, sources))) != len(sources)
            or len(set(map(str, tables))) != len(tables)):
        raise ValueError('need unique ordered sources/exports and a new split directory')
    source_keys = []
    for path in sources:
        with uproot.open(path) as root:
            source_keys.append(event_keys.read(root['Events']))
    all_keys = np.concatenate(source_keys)
    if len(np.unique(all_keys, axis=0)) != len(all_keys):
        raise ValueError('overlapping original parent populations')
    schemas, data = {}, {}
    expected_names = None
    for path in tables:
        with uproot.open(path) as root:
            names = set(root['dump'].keys(recursive=False, cycle=False))
            if expected_names is not None and names != expected_names:
                raise ValueError('truth tree sets differ across exports')
            expected_names = names
            for name, tree in root['dump'].items(recursive=False, cycle=False):
                if tree.classname != 'TTree':
                    raise ValueError('truth export contains an unsupported object')
                arrays = tree.arrays(library='np')
                schema = {key: value.dtype for key, value in arrays.items()}
                required = KEYS[:2] if name == 'subruns' else KEYS
                if not set(required).issubset(schema):
                    raise ValueError(f'truth tree {name} lacks required identity columns')
                if name in schemas and schema != schemas[name]:
                    raise ValueError('truth schemas differ across exports')
                schemas[name] = schema
                data.setdefault(name, []).append(arrays)
    if not schemas:
        raise ValueError('truth export contains no trees')
    arrays = {name: {key: np.concatenate([part[key] for part in parts]) for key in schemas[name]}
              for name, parts in data.items()}
    tree_keys = {}
    for name, values in arrays.items():
        required = KEYS[:2] if name == 'subruns' else KEYS
        keys = np.column_stack([values[key] for key in required]).astype(np.uint64)
        if not np.all(np.isin(structured(keys), structured(all_keys[:, :len(required)]))):
            raise ValueError(f'truth tree {name} contains identities outside original parents')
        tree_keys[name] = structured(keys)
    output.mkdir(parents=True)
    rows = []
    for ordinal, (path, keys) in enumerate(zip(sources, source_keys)):
        target = output / f'parent_{ordinal + 1}.root'
        counts = {}
        with uproot.create(target) as root:
            for name, values in arrays.items():
                width = 2 if name == 'subruns' else 3
                mask = np.isin(tree_keys[name], structured(keys[:, :width]))
                counts[name] = int(np.count_nonzero(mask))
                root.mktree('dump/' + name, schemas[name])
                if np.any(mask):
                    root['dump/' + name].extend({key: value[mask] for key, value in values.items()})
        rows.append({'ordinal': ordinal, 'events': len(keys), 'source': str(path),
                     'table': str(target), 'tree_rows': counts})
    totals = {name: len(keys) for name, keys in tree_keys.items()}
    receipt = {'events': len(all_keys), 'parents': rows, 'export_tree_rows': totals,
        'subrun_metadata_policy': 'copy matching run/subrun rows to each intersecting parent; not additive exposure',
        'source_files': integrity.files(sources), 'export_files': integrity.files(tables),
        'output_files': integrity.files([row['table'] for row in rows]),
        'scope': 'exact event-key partition of saved truth, not transport or physics certification'}
    (output / 'split.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (output / 'event-counts.txt').write_text(''.join(f"{row['events']}\n" for row in rows))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, required=True)
    parser.add_argument('--tables', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = split(args.sources, args.tables, args.output)
    print(f"split {result['events']} original events into {len(result['parents'])} parent tables")


if __name__ == '__main__':
    main()
