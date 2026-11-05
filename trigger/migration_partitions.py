"""Saved all52 migrations by contiguous original parent; no art/transport rerun or SEM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

import compare
import event_keys
import hybrid_replay
import integrity
import path_state
import replay
import schedule_registry


MIGRATIONS = ('stock_pass', 'replay_pass', 'stock_only', 'replay_only', 'mismatches')


def _rows(stock, fast, names, stock_sequences, fast_sequences):
    rows = path_state.compare_matrices(stock, fast, names, stock_sequences, fast_sequences)
    a, b = (stock & 3) == 1, (fast & 3) == 1
    for col, name in enumerate(names):
        counts = [np.count_nonzero(mask[:, col]) for mask in (a, b, a & ~b, ~a & b, a != b)]
        rows[name].update(zip(MIGRATIONS, map(int, counts)))
        rows[name]['delta_fraction'] = (counts[1] - counts[0]) / len(stock)
    return rows


def _check_sums(aggregate, partitions):
    if sum(row['events'] for row in partitions) != aggregate['events']:
        raise ValueError('partition event counts differ from aggregate')
    for name, total in aggregate['paths'].items():
        for field, value in total.items():
            if field == 'delta_fraction':
                continue  # Fractions have different denominators; only counts are additive.
            pieces = [row['paths'][name][field] for row in partitions]
            summed = ({state: sum(piece[state] for piece in pieces) for state in value}
                      if isinstance(value, dict) else sum(pieces))
            if summed != value:
                raise ValueError(f'partition counts differ from aggregate: {name}/{field}')


def summarize(stock, fast, names, stock_sequences, fast_sequences, counts):
    """Return {aggregate: {events, paths}, partitions: ordered [{start, stop, events, paths}]}.

    Positive counts delimit contiguous original-parent slices, NOT sorted subruns.
    Status words come from one continuously evolving whole-process filter history;
    slicing never resets counters or supplies independent statistical replicates.
    delta_fraction is (replay_pass - stock_pass) / events, including zero-pass paths.
    """
    names, counts = list(names), list(counts)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('need nonempty unique path names')
    stock, fast = np.asarray(stock), np.asarray(fast)
    a, _, _ = path_state.decode(stock)
    b, _, _ = path_state.decode(fast)
    if stock.shape != fast.shape or stock.shape[1] != len(names):
        raise ValueError('need matching scheduled path matrices')
    if np.any((a == 0) | (a == 3) | (b == 0) | (b == 3)):
        raise ValueError('partition diagnostic requires no Ready/Exception paths')
    if (not counts or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                          or n <= 0 for n in counts) or sum(map(int, counts)) != len(stock)):
        raise ValueError('positive integer parent counts must sum exactly to matrix rows')
    if any(name not in stock_sequences or name not in fast_sequences for name in names):
        raise ValueError('missing scheduled path sequence')
    aggregate = {'events': len(stock), 'paths': _rows(stock, fast, names, stock_sequences, fast_sequences)}
    partitions, start = [], 0
    for number, count in enumerate(counts, 1):
        stop = start + int(count)
        partitions.append({'partition': number, 'start': start, 'stop': stop, 'events': int(count),
            'paths': _rows(stock[start:stop], fast[start:stop], names, stock_sequences, fast_sequences)})
        start = stop
    _check_sums(aggregate, partitions)
    return {'aggregate': aggregate, 'partitions': partitions}


def analyze(art, config_path, prepared, comparison_path):
    art, config_path, prepared, comparison_path = map(Path, (art, config_path, prepared, comparison_path))
    config_text = config_path.read_text()
    manifest, result = hybrid_replay.preflight(prepared, config_text)
    comparison = json.loads(comparison_path.read_text())
    evidence = integrity.files([prepared / 'manifest.json', prepared / 'replay.fcl'])
    if comparison.get('validation_evidence') != evidence or comparison.get('saved_replay_file') != integrity.digest(art):
        raise ValueError('completed comparison evidence or saved art bytes differ')
    for field in ('all_scheduled_paths_measured', 'accepted_bits_crosschecked_with_log',
                  'root_event_ids_independently_decoded', 'trigger_schedule_registry_verified'):
        if comparison.get(field) is not True:
            raise ValueError(f'need completed comparison: {field}')
    if comparison.get('full_production_history_certified') != manifest['full_production_history_certified']:
        raise ValueError('comparison production-history certification differs')
    original, candidate, names, bits = compare.validate_manifest(manifest)
    stock_paths = replay.paths(replay.production_config(config_text, original))
    fast_paths = replay.paths(result)
    scheduled = [name for _, name, _ in stock_paths
                 if name != 'DigitizePath' and not name.startswith('Triggerable')]
    if (len(names) != 52 or scheduled != names
            or [(bit, name) for bit, name, _ in fast_paths] != list(zip(manifest['bits'], names))):
        raise ValueError('need all52 scheduled physics paths in saved replay order')
    stock_names = [name for _, name, _ in stock_paths]
    if any(name not in stock_names or int(stock_paths[stock_names.index(name)][0]) != bit
           for name, bit in zip(names, bits)):
        raise ValueError('original scheduled names/bits differ from manifest')
    if set(comparison.get('decision_migrations', {})) != set(names) or set(comparison.get('paths', {})) != set(names):
        raise ValueError('comparison must contain every scheduled path')
    # art processes each source in supplied order, with numeric IDs sorted WITHIN that file.
    # Never sort the concatenation: parent traversal can be 1,10,2 rather than 1,2,10.
    sources = result['source']['fileNames']
    keys, counts = [], []
    for source in sources:
        with uproot.open(source) as root:
            values = event_keys.read(root['Events'])
        counts.append(len(values))
        keys.append(values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))])
    if not keys:
        raise ValueError('empty candidate source list')
    keys = np.concatenate(keys)
    if len(keys) != manifest['expected_events'] or len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('candidate event count or duplicate identities differ')
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    with uproot.open(art, custom_classes=classes) as root:
        path_state.check_schema(root)
        tree = root['Events']
        if tree.classname != 'TTree' or tree.num_entries != len(keys):
            raise ValueError('wrong saved Events type or count')
        saved_keys = event_keys.read(tree)
        if len(np.unique(saved_keys, axis=0)) != len(saved_keys) or not np.array_equal(keys, saved_keys):
            raise ValueError('saved art event IDs/order differ from candidate parent traversal')
        stock, stock_id = path_state.read_status(tree, original, len(stock_paths))
        fast, fast_id = path_state.read_status(tree, candidate, len(fast_paths))
        registry = {original: schedule_registry.verify(root, stock_id, stock_paths),
                    candidate: schedule_registry.verify(root, fast_id, fast_paths)}
    if len(stock) != len(keys) or len(fast) != len(keys) or comparison.get('events') != len(keys):
        raise ValueError('saved matrices/comparison event count differs')
    if (comparison.get('parameter_set_ids') != {original: stock_id, candidate: fast_id}
            or comparison.get('schedule_registry') != registry):
        raise ValueError('saved schedule registry differs from comparison')
    stock = stock[:, [stock_names.index(name) for name in names]]
    report = summarize(stock, fast, names, {name: seq for _, name, seq in stock_paths},
                       {name: seq for _, name, seq in fast_paths}, counts)
    for name, row in report['aggregate']['paths'].items():
        prior, states = comparison['decision_migrations'][name], comparison['paths'][name]
        if (any(prior.get(field) != row[field] for field in MIGRATIONS)
                or prior.get('paired_efficiency_difference') != row['delta_fraction']
                or any(states.get(field) != value for field, value in row.items()
                       if field not in (*MIGRATIONS, 'delta_fraction'))):
            raise ValueError(f'computed aggregate differs from completed comparison for {name}')
    for row, source in zip(report['partitions'], sources):
        row['source'] = str(Path(source).resolve())
    report.update(events=len(keys), paths=names, schedule_registry=registry,
        inputs=integrity.files([art, config_path, prepared / 'manifest.json', prepared / 'replay.fcl',
                                comparison_path, *sources, *[row['path'] for row in manifest['reference_integrity']]]),
        full_production_history_certified=manifest['full_production_history_certified'],
        scope='contiguous original-parent diagnostics from one continuously evolving filter history; no counter reset',
        limitations=['common-seed resets correlate original parents; partitions are not independent replicas',
                     'no independent error bars or SEM, generated exposure, or causal attribution',
                     'primary-only transport, separate electronics RNG and retained stock calo history; no physics closure'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('art', 'config', 'prepared', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse partition diagnostic overwrite')
    report = analyze(args.art, args.config, args.prepared, args.comparison)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(f"saved partition diagnostic: {report['events']} events, {len(report['partitions'])} parents")


if __name__ == '__main__':
    main()
