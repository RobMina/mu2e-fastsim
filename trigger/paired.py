"""Paired equal-shard errors for conditional trigger-efficiency differences."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarize(reports):
    if len(reports) < 2:
        raise ValueError('paired per-shard error needs at least two independent shards')
    first = reports[0]
    names = set(first['paths'])
    if not names:
        raise ValueError('empty paths')
    for report in reports:
        if report['comparison'] != 'accepted_boolean_only' or report['full_art_state_compared']:
            raise ValueError('unsupported comparison kind')
        if set(report['paths']) != names or not report['input_event_ids_verified']:
            raise ValueError('path sets must match and input IDs must be independently verified')
        n = report['events']
        if type(n) is not int or n <= 0:
            raise ValueError('invalid shard event count')
        for name, row in report['paths'].items():
            for field in ('stock_pass', 'replay_pass', 'stock_only', 'replay_only', 'mismatches'):
                if type(row[field]) is not int or not 0 <= row[field] <= n:
                    raise ValueError('invalid decision or migration count')
            if (row['bit'] != first['paths'][name]['bit'] or
                row['stock_only'] > row['stock_pass'] or row['replay_only'] > row['replay_pass'] or
                row['mismatches'] != row['stock_only'] + row['replay_only'] or
                row['replay_pass'] - row['stock_pass'] != row['replay_only'] - row['stock_only']):
                raise ValueError('inconsistent path bits or paired migration counts')
    count = len(reports)
    result = {}
    for name in sorted(names):
        values = [(r['paths'][name]['replay_pass'] - r['paths'][name]['stock_pass']) / r['events'] for r in reports]
        mean = math.fsum(values) / count
        sem = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (count * (count - 1)))
        result[name] = {'bit': first['paths'][name]['bit'], 'mean_paired_difference': mean,
                        'paired_shard_sem': sem,
                        'mismatches': sum(r['paths'][name]['mismatches'] for r in reports),
                        'stock_only': sum(r['paths'][name]['stock_only'] for r in reports),
                        'replay_only': sum(r['paths'][name]['replay_only'] for r in reports)}
    return {'shards': count, 'events': sum(r['events'] for r in reports), 'paths': result,
            'weighting': 'equal independent shards; standard error of shard paired differences',
            'denominator': 'conditional on stored input events; not absolute generated-event efficiency',
            'caveats': ['zero observed mismatches is not a finite-sample bound on an unseen mismatch probability',
                        'caller must establish independent, nonoverlapping shards and the development/holdout split']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = [p.resolve() for p in args.reports]
    if len(set(sources)) != len(sources):
        raise ValueError('duplicate report files')
    report = summarize([json.loads(p.read_text()) for p in args.reports])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"shards={report['shards']} events={report['events']}; paired equal-shard errors")


if __name__ == '__main__':
    main()
