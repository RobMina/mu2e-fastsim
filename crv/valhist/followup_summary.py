"""Compact read-only summaries and explicit completed-cell ensemble manifests."""
import argparse
import json
from pathlib import Path

import numpy as np

from .library_variants import _read_sealed


def ensemble_manifest(campaign, out, indices):
    record = _read_sealed(Path(campaign))
    cells = [c for c in record['cells'] if c['index'] in indices]
    if {c['index'] for c in cells} != set(indices):
        raise ValueError('unknown index')
    normalized = []
    for cell in cells:
        complete = json.loads((Path(cell['directory']) / 'completion.json').read_text())
        if not complete['completed'] or complete['changed_inputs']:
            raise ValueError('incomplete or drifted cell')
        normalized.append({**{k: cell[k] for k in ('index', 'draw', 'shard', 'directory', 'library_metadata', 'seed')},
                           'expected_generated_events': cell['expected_events']})
    result = {'schema': 'valhist.library_ensemble.v1', 'run': 1400, 'cells': normalized}
    from .library_ensemble import validate_manifest
    validate_manifest(result, Path(out).parent)
    with Path(out).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f'wrote {out}: {len(cells)} completed cells')


def attribution(report):
    result = json.loads(Path(report).read_text())
    print('shards', result['shards'], 'stock', result['stock_dump_status'])
    for comparison, observables in result['comparisons'].items():
        print('\nCOMPARISON', comparison)
        for name, observable in observables.items():
            print(name, observable['status'])
            for partition, entry in observable.get('regions', {}).get('visible', {}).items():
                global_shape = entry['global_shape']['shape']
                null = global_shape['centered_tv_null']
                print(' ', partition, 'TV', global_shape['total_variation'],
                      'null95', null['quantiles'][2] if null else None)
                for group, values in entry['groups'].items():
                    direct = values['signed_contribution']
                    within, mix = values['within'], values['mix']
                    print('   ', group, 'N', sum(values['counts_ref']), sum(values['counts_cand']),
                          'contributionL1half', np.abs(direct).sum() / 2 if direct is not None else None,
                          'withinL1half', np.abs(within).sum() / 2 if within is not None else None,
                          'mixL1half', np.abs(mix).sum() / 2 if mix is not None else None,
                          'localTV', values['local']['shape']['total_variation'])
    for arm in ('a1', 'e2e', 'a0g_dump'):
        rows = [s['arms'][arm] for s in result['sources'] if arm in s['arms']]
        if rows:
            print('closure', arm, {name: {'nonclosing_shards': sum(not r['closure'][name]['exact'] for r in rows),
                                        'l1_counts': sum(r['closure'][name]['l1_count_difference'] for r in rows)}
                                    for name in rows[0]['closure']})


def ensemble(report):
    result = json.loads(Path(report).read_text())
    print('report keys', sorted(result))
    for arm, metrics in result['metrics'].items():
        print(arm, json.dumps(metrics['cluster_rate'], allow_nan=False))


def reco_detail(report):
    result = json.loads(Path(report).read_text())
    for shard, item in result['per_shard'].items():
        print('SHARD', shard, 'events', item['events'], 'totals', item['totals'])
        for key in ('pulse_identity', 'cluster_alignment', 'identical_pulses'):
            print(key, json.dumps(item.get(key), allow_nan=False))
        for name in ('pe', 'time', 'flags'):
            stats = item['pulse_fields'][name]
            print('pulse', name, json.dumps({k: stats[k] for k in
                  ('rows', 'exact_mismatches', 'beyond_tolerance', 'delta')}, allow_nan=False))
        for sector, stats in item['sectors'].items():
            print('sector', sector, json.dumps({k: v for k, v in stats.items()
                                                if k not in ('pulse_fields', 'cluster_fields', 'fields')}, allow_nan=False))
        print('membership summary', json.dumps({k: v for k, v in item['membership'].items()
                                                if k not in ('removed', 'added')}, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('ensemble-manifest')
    p.add_argument('--campaign', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--indices', nargs='+', type=int, required=True)
    for command in ('attribution', 'ensemble', 'reco-detail'):
        p = sub.add_parser(command)
        p.add_argument('report')
    args = vars(parser.parse_args())
    command = args.pop('command')
    {'ensemble-manifest': ensemble_manifest, 'attribution': attribution,
     'ensemble': ensemble, 'reco-detail': reco_detail}[command](**args)


if __name__ == '__main__':
    main()
