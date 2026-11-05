"""Saved all52 common-FILTER reach/return diagnostics; never launch art or transport."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

import compare
import event_keys
import integrity
import path_state
import replay
import schedule_registry


MIGRATIONS = ('stock_pass', 'replay_pass', 'stock_only', 'replay_only', 'mismatches')
COMPLETION_FLAGS = ('all_scheduled_paths_measured', 'accepted_bits_crosschecked_with_log',
                    'input_event_ids_verified_by_log', 'root_event_ids_independently_decoded',
                    'trigger_schedule_registry_verified', 'full_production_history_certified')


def _same_bytes(saved, actual):
    """Admit copied saved evidence by ordered exact bytes, not original path spelling."""
    return (isinstance(saved, list) and len(saved) == len(actual)
            and all(isinstance(a, dict) and all(a.get(key) == b.get(key) for key in ('bytes', 'sha256'))
                    for a, b in zip(saved, actual)))


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _counts(masks, events):
    counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    return {'counts': counts,
            'event_fractions': {name: _fraction(count, events) for name, count in counts.items()}}


def _sequence(sequences, name):
    if name not in sequences:
        raise ValueError(f'missing scheduled path sequence: {name}')
    sequence = sequences[name]
    if (not isinstance(sequence, (list, tuple))
            or any(not isinstance(label, str) or not label for label in sequence)):
        raise ValueError('need literal module-label sequences')
    if len(set(sequence)) != len(sequence):
        raise ValueError('repeated module in path is unsupported for stage flow')
    return list(sequence)


def summarize(stock, fast, names, stock_sequences, fast_sequences, filter_labels):
    """Vectorized counts for each common filter, including a rejecting endpoint.

    filter_labels is the set/iterable of labels classified as FILTER in BOTH
    configurations, NOT a module-type-name heuristic. Producers (including shared
    fits) are never stages. Workers have one-based positions; a Fail at position k
    reached k but did not return pass. A Pass must have index == sequence length.
    Empty populations are supported with null fractions; zero-reach stages remain.
    No event iteration, history reconstruction or independent uncertainty estimate.
    """
    names = list(names)
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('need nonempty unique path names')
    if isinstance(filter_labels, (str, bytes)):
        raise ValueError('need an iterable of common FILTER labels, not a string')
    labels = list(filter_labels)
    if (any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('need unique common FILTER labels')
    labels = set(labels)
    stock, fast = np.asarray(stock), np.asarray(fast)
    a, ia, _ = path_state.decode(stock)
    b, ib, _ = path_state.decode(fast)
    if stock.shape != fast.shape or stock.shape[1] != len(names):
        raise ValueError('need matching scheduled path matrices')
    if np.any((a == 0) | (a == 3) | (b == 0) | (b == 3)):
        raise ValueError('stage flow requires no Ready/Exception paths')
    seq_a = {name: _sequence(stock_sequences, name) for name in names}
    seq_b = {name: _sequence(fast_sequences, name) for name in names}
    # This validates all execution counts, even paths with no common filters.
    final = path_state.compare_matrices(stock, fast, names, seq_a, seq_b)
    events, rows = len(stock), {}
    for col, name in enumerate(names):
        passed_a, passed_b = a[:, col] == 1, b[:, col] == 1
        masks = (passed_a, passed_b, passed_a & ~passed_b, ~passed_a & passed_b,
                 passed_a != passed_b)
        final[name].update(zip(MIGRATIONS, (int(np.count_nonzero(mask)) for mask in masks)))
        final[name]['paired_efficiency_difference'] = _fraction(
            final[name]['replay_pass'] - final[name]['stock_pass'], events)
        stages = {}
        positions_b = {label: pos for pos, label in enumerate(seq_b[name], 1)}
        for pos_a, label in enumerate(seq_a[name], 1):
            if label not in labels or label not in positions_b:
                continue
            pos_b = positions_b[label]
            reached_a = passed_a | (ia[:, col] >= pos_a)
            reached_b = passed_b | (ib[:, col] >= pos_b)
            returned_a = reached_a & (passed_a | (ia[:, col] > pos_a))
            returned_b = reached_b & (passed_b | (ib[:, col] > pos_b))
            both = reached_a & reached_b
            reach = _counts({'neither': ~reached_a & ~reached_b, 'both': both,
                'stock_only': reached_a & ~reached_b, 'fast_only': ~reached_a & reached_b}, events)
            arms = {}
            for arm, reached, returned in (('stock', reached_a, returned_a), ('fast', reached_b, returned_b)):
                row = _counts({'reached': reached, 'returned_pass': returned,
                               'returned_fail': reached & ~returned}, events)
                count = row['counts']
                row['conditional_on_reached'] = {'denominator': count['reached'],
                    'pass_fraction': _fraction(count['returned_pass'], count['reached']),
                    'fail_fraction': _fraction(count['returned_fail'], count['reached'])}
                arms[arm] = row
            paired = _counts({'stock_pass': both & returned_a, 'fast_pass': both & returned_b,
                'both_pass': both & returned_a & returned_b,
                'both_fail': both & ~returned_a & ~returned_b,
                'lost': both & returned_a & ~returned_b,
                'gained': both & ~returned_a & returned_b,
                'mismatch': both & (returned_a != returned_b)}, events)
            denominator = reach['counts']['both']
            paired['conditional_on_both_reached'] = {'denominator': denominator,
                'fractions': {key: _fraction(count, denominator) for key, count in paired['counts'].items()}}
            stages[label] = {'stock_position': pos_a, 'fast_position': pos_b,
                             **arms, 'reach': reach, 'both_reached': paired}
        rows[name] = {'final_path': final[name], 'filters': stages}
    return rows


def _admit(art, config_path, prepared, comparison_path):
    """Bind saved bytes to completed postflight; deliberately do NOT call preflight.

    hybrid_replay.preflight would re-open the historical sources. Their original
    census is evidence in the completed comparison, not work to repeat here.
    Exact art hash binds the newly decoded IDs/order to that prior independent
    input-ID check: the comparison does not itself store an event-key array.
    """
    config_text = config_path.read_text()
    fcl_text = (prepared / 'replay.fcl').read_text()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    comparison = json.loads(comparison_path.read_text())
    evidence = integrity.files([prepared / 'manifest.json', prepared / 'replay.fcl'])
    if (manifest.get('config_sha256') != hashlib.sha256(config_text.encode()).hexdigest()
            or manifest.get('replay_fcl_sha256') != hashlib.sha256(fcl_text.encode()).hexdigest()
            or not _same_bytes(comparison.get('validation_evidence'), evidence)
            or not _same_bytes([comparison.get('saved_replay_file')], [integrity.digest(art)])):
        raise ValueError('saved config/FCL/manifest/art bytes differ from completed comparison')
    if manifest.get('full_production_history_certified') is not True:
        raise ValueError('need a completed full-history hybrid diagnostic')
    for field in COMPLETION_FLAGS:
        if comparison.get(field) is not True:
            raise ValueError(f'need completed comparison: {field}')
    original, candidate, names, bits = compare.validate_manifest(manifest)
    stock_config = replay.production_config(config_text, original)
    fast_config = replay.parse_expanded(fcl_text)
    stock_paths, fast_paths = replay.paths(stock_config), replay.paths(fast_config)
    scheduled = [name for _, name, _ in stock_paths
                 if name != 'DigitizePath' and not name.startswith('Triggerable')]
    stock_names = [name for _, name, _ in stock_paths]
    if (len(names) != 52 or scheduled != names
            or [(bit, name) for bit, name, _ in fast_paths] != list(zip(manifest['bits'], names))
            or any(name not in stock_names or stock_paths[stock_names.index(name)][0] is None
                   or int(stock_paths[stock_names.index(name)][0]) != bit
                   for name, bit in zip(names, bits))):
        raise ValueError('need all52 exact scheduled physics paths/bits in saved replay order')
    if (set(comparison.get('paths', {})) != set(names)
            or set(comparison.get('decision_migrations', {})) != set(names)):
        raise ValueError('completed comparison must contain all52 paths')
    return manifest, comparison, stock_config, fast_config, stock_paths, fast_paths


def analyze(art, config_path, prepared, comparison_path):
    """Read only five saved artifacts; missing original source files are allowed."""
    art, config_path, prepared, comparison_path = map(Path, (art, config_path, prepared, comparison_path))
    manifest, comparison, stock_config, fast_config, stock_paths, fast_paths = _admit(
        art, config_path, prepared, comparison_path)
    names = manifest['paths']
    original, candidate = manifest['original_process'], manifest['replay_process']
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    with uproot.open(art, custom_classes=classes) as root:
        path_state.check_schema(root)
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False) if key.startswith('Events;')]
        if not cycles:
            raise ValueError('missing saved Events tree')
        highest_cycle = max(cycles)
        tree = root[f'Events;{highest_cycle}']
        if (tree.classname != 'TTree' or tree.num_entries <= 0
                or tree.num_entries != comparison.get('events')
                or tree.num_entries != manifest.get('expected_events')
                or highest_cycle != comparison.get('highest_cycle')):
            raise ValueError('saved highest-cycle Events type/count differs from completed comparison')
        keys = event_keys.read(tree)
        if (keys.shape != (tree.num_entries, 3) or keys.dtype.kind not in 'ui'
                or np.any(keys <= 0) or len(np.unique(keys, axis=0)) != len(keys)):
            raise ValueError('invalid or duplicate saved ROOT event identities')
        stock, stock_id = path_state.read_status(tree, original, len(stock_paths))
        fast, fast_id = path_state.read_status(tree, candidate, len(fast_paths))
        registry = {original: schedule_registry.verify(root, stock_id, stock_paths),
                    candidate: schedule_registry.verify(root, fast_id, fast_paths)}
    if (stock.shape != (len(keys), len(stock_paths)) or fast.shape != (len(keys), len(fast_paths))):
        raise ValueError('saved status matrix counts differ from ROOT identities')
    if (comparison.get('parameter_set_ids') != {original: stock_id, candidate: fast_id}
            or comparison.get('schedule_registry') != registry):
        raise ValueError('saved original/replay registry differs from completed comparison')
    stock_names = [name for _, name, _ in stock_paths]
    stock = stock[:, [stock_names.index(name) for name in names]]
    arm_filters = [set(config['physics'].get('filters', {})) for config in (stock_config, fast_config)]
    # Fail endpoints must really be filters, even if that label is not common.
    for words, paths, filters in ((stock, stock_paths, arm_filters[0]), (fast, fast_paths, arm_filters[1])):
        state, index, _ = path_state.decode(words)
        sequences = {name: seq for _, name, seq in paths}
        for col, name in enumerate(names):
            endpoints = path_state.endpoints(state[:, col], index[:, col], sequences[name])
            if np.any((state[:, col] == 2) & ~np.isin(endpoints, list(filters))):
                raise ValueError('failed path endpoint is not a configured FILTER')
    rows = summarize(stock, fast, names, {name: seq for _, name, seq in stock_paths},
                     {name: seq for _, name, seq in fast_paths}, arm_filters[0] & arm_filters[1])
    for name, row in rows.items():
        final = row['final_path']
        states, prior = comparison['paths'][name], comparison['decision_migrations'][name]
        if (any(states.get(field) != value for field, value in final.items()
                if field not in (*MIGRATIONS, 'paired_efficiency_difference'))
                or any(prior.get(field) != final[field] for field in (*MIGRATIONS, 'paired_efficiency_difference'))):
            raise ValueError(f'stage final path state/loss/gain differs from completed comparison: {name}')
    totals = {field: sum(row['final_path'][field] for row in rows.values()) for field in MIGRATIONS}
    mismatch_fields = [field for field in rows[names[0]]['final_path'] if field.endswith('mismatches')]
    for field in mismatch_fields:
        total = sum(row['final_path'][field] for row in rows.values())
        if field in comparison and comparison[field] != total:
            raise ValueError(f'aggregate stage final states differ from completed comparison: {field}')
    return {'events': len(keys), 'highest_cycle': highest_cycle, 'paths': rows,
        'final_path_totals': totals,
        'schedule_registry': registry, 'full_production_history_certified': True,
        'completed_comparison_reconciled': True,
        'root_event_ids_independently_decoded': True,
        'identity_binding': 'unique saved ROOT IDs/order bound by exact art hash to completed independent input-ID comparison; no source recensus',
        'inputs': integrity.files([art, config_path, prepared / 'manifest.json', prepared / 'replay.fcl', comparison_path]),
        'scope': 'saved-only descriptive all52 common-FILTER reach and return counts, including rejecting endpoints',
        'index_convention': 'one-based executed-worker count; Pass requires full sequence length',
        'fraction_denominators': {'event_fractions': 'all saved events, including unreached events',
            'conditional_on_reached': 'events reaching the filter in this arm',
            'conditional_on_both_reached': 'events reaching the filter in both arms (selected paired diagnostic)',
            'zero_denominator': 'null'},
        'limitations': ['paired both-reach is a selected diagnostic, not an isolated causal filter effect or causal origin',
            'common-seed parent electronics resets are correlated; no independent replicas, independent errors or SEM',
            'primary-only transport omits secondaries/returns; separate electronics RNG and retained stock-calorimeter history',
            'no physics closure, generated-event efficiency, threshold tuning, art/reco/transport launch or source history reprocessing']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('art', 'config', 'prepared', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('refuse stage-flow diagnostic overwrite')
    report = analyze(args.art, args.config, args.prepared, args.comparison)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"saved stage-flow diagnostic: {report['events']} events, {len(report['paths'])} paths")


if __name__ == '__main__':
    main()
