"""Print compact measured summaries from Thread B uncertainty reports."""
import argparse
import json
from pathlib import Path


def summarize(path):
    path = Path(path)
    result = json.loads(path.read_text())
    print(path)
    if 'paired_rates' in result:
        print('generated events/arm:', result['generated_events_per_arm'])
        for name, value in result['paired_rates'].items():
            print(name, json.dumps(value))
    elif 'histograms' in result:
        print('shards:', len(result['shards']))
        for name, hist in result['histograms'].items():
            for region, value in hist['regions'].items():
                shape, rate = value['shape'], value['rate_ratio']
                null = shape.get('centered_tv_null') or {}
                print(name, region, json.dumps(dict(ref=value['total_ref'], cand=value['total_cand'],
                      ratio=rate['value'], ratio_ci=rate['percentile_ci'], tv=shape['total_variation'],
                      null_quantiles=null.get('quantiles'), tail=null.get('tail_probability_plus_one'))))
    elif 'observables' in result:
        print('events:', result['events'], 'rates:', json.dumps(result['total_cluster_rate']))
        for name, observable in result['observables'].items():
            for region in ('inclusive', 'visible'):
                value = observable[region]
                print(name, region, json.dumps({k: value.get(k) for k in
                      ('tv', 'conditional_null_tv_quantiles', 'linearized_tv_uncertainty_quantiles')}))
    elif 'components' in result:
        for arm, metrics in result['components'].items():
            print(arm, 'cluster_rate', json.dumps(metrics['cluster_rate']))
            for name, component in metrics.items():
                if name == 'cluster_rate':
                    continue
                negative = component['negative_component']
                print(arm, name, 'negative_variance_bins', sum(negative), '/', len(negative))
    elif 'certification' in result:
        print(json.dumps(result['certification'], indent=2))
        for name, variant in result['variants'].items():
            print(name, json.dumps(variant))
    else:
        raise ValueError('unrecognized uncertainty report schema')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports', nargs='+')
    a = p.parse_args()
    for path in a.reports:
        summarize(path)


if __name__ == '__main__':
    main()
