"""Count mapped endpoint pairs for saved stock/fast path migrations; no art or physics rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import uproot

import integrity
import path_state
import replay


def summarize(stock, fast, names, stock_sequences, fast_sequences):
    if stock.shape != fast.shape or stock.shape[1] != len(names):
        raise ValueError('endpoint diagnostic needs matching scheduled path matrices')
    a, ia, _ = path_state.decode(stock)
    b, ib, _ = path_state.decode(fast)
    if np.any((a == 0) | (a == 3) | (b == 0) | (b == 3)):
        raise ValueError('endpoint migration diagnostic requires no Ready/Exception paths')
    rows = {}
    for col, name in enumerate(names):
        ea = path_state.endpoints(a[:, col], ia[:, col], stock_sequences[name])
        eb = path_state.endpoints(b[:, col], ib[:, col], fast_sequences[name])
        groups = {'lost': (a[:, col] == 1) & (b[:, col] == 2),
                  'gained': (a[:, col] == 2) & (b[:, col] == 1),
                  'both_fail_different_endpoint': (a[:, col] == 2) & (b[:, col] == 2) & (ea != eb)}
        result = {}
        for label, mask in groups.items():
            pairs, counts = np.unique(np.column_stack([ea[mask], eb[mask]]), axis=0, return_counts=True)
            order = np.argsort(-counts, kind='stable')
            result[label] = {'events': int(np.count_nonzero(mask)), 'endpoint_pairs': [
                {'stock': str(pairs[i, 0]), 'fast': str(pairs[i, 1]), 'events': int(counts[i])} for i in order]}
        rows[name] = result
    return rows


def analyze(art, config_path, prepared, comparison_path):
    config_text = config_path.read_text()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    comparison = json.loads(comparison_path.read_text())
    fcl_text = (prepared / 'replay.fcl').read_text()
    if (manifest['config_sha256'] != hashlib.sha256(config_text.encode()).hexdigest()
            or manifest['replay_fcl_sha256'] != hashlib.sha256(fcl_text.encode()).hexdigest()
            or comparison['saved_replay_file']['sha256'] != integrity.digest(art)['sha256']):
        raise ValueError('saved decision/config/FCL bytes differ from completed diagnostic')
    stock_paths = replay.paths(replay.production_config(config_text, manifest['original_process']))
    fast_paths = replay.paths(replay.parse_expanded(fcl_text))
    names = manifest['paths']
    if [name for _, name, _ in fast_paths] != names or len(names) != 52:
        raise ValueError('need all52 scheduled physics paths in saved replay order')
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    with uproot.open(art, custom_classes=classes) as root:
        path_state.check_schema(root)
        stock, _ = path_state.read_status(root['Events'], manifest['original_process'], len(stock_paths))
        fast, _ = path_state.read_status(root['Events'], manifest['replay_process'], len(names))
    stock_names = [name for _, name, _ in stock_paths]
    stock = stock[:, [stock_names.index(name) for name in names]]
    if len(stock) != comparison['events']:
        raise ValueError('saved event count differs from completed comparison')
    rows = summarize(stock, fast, names, {name: seq for _, name, seq in stock_paths},
                     {name: seq for _, name, seq in fast_paths})
    for name, row in rows.items():
        prior = comparison['decision_migrations'][name]
        if row['lost']['events'] != prior['stock_only'] or row['gained']['events'] != prior['replay_only']:
            raise ValueError('endpoint loss/gain counts differ from independent completed comparison')
    return {'events': len(stock), 'paths': rows,
        'inputs': integrity.files([art, config_path, prepared / 'manifest.json', prepared / 'replay.fcl', comparison_path]),
        'full_production_history_certified': comparison['full_production_history_certified'],
        'scope': 'saved path last-executed-filter migration counts; endpoint is not the causal origin of a discrepancy',
        'limitations': ['primary-only transport, separate electronics RNG and retained stock calo history',
                        'no threshold changes, new events, physics attribution or independent-shard errors']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('art', 'config', 'prepared', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse endpoint diagnostic overwrite')
    report = analyze(args.art, args.config, args.prepared, args.comparison)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"saved endpoint diagnostic: {report['events']} events, {len(report['paths'])} paths")


if __name__ == '__main__':
    main()
