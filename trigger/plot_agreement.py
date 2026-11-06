"""Per-path agreement of stored and replayed trigger decisions for one or more compare.py reports.

Use: python3 trigger/plot_agreement.py --sample CE=ce.json --sample Cosmic=cosmic.json \
         --cost CE=8.2 --cost Cosmic=9.9 --output trigger_replay.png --numbers numbers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', 'D', '^']


def load(spec):
    label, _, path = spec.partition('=')
    if not path:
        raise ValueError(f'expected LABEL=PATH, got {spec!r}')
    return label, json.loads(Path(path).read_text())


def summary(report):
    rows = report['paths']
    exact = sum(row['mismatches'] == 0 for row in rows.values())
    return {'events': report['events'], 'paths': len(rows), 'paths_exact': exact,
            'event_path_decisions': report['events'] * len(rows), 'mismatches': report['mismatches'],
            'nonstateful_mismatches': report['nonstateful_mismatches'],
            'stateful_paths': report['stateful_paths'],
            'differing_paths': {name: {k: row[k] for k in ('stock_pass', 'replay_pass', 'stock_only', 'replay_only')}
                                for name, row in rows.items() if row['mismatches']},
            'stock_pass': {name: row['stock_pass'] for name, row in rows.items()}}


def plot(samples, costs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(samples[0][1]['paths'])
    for label, report in samples[1:]:
        if list(report['paths']) != names:
            raise ValueError(f'{label} has a different path list')
    x = np.arange(len(names))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                                      gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.06})
    floor = 0.3 / max(report['events'] for _, report in samples)
    width = 0.8 / len(samples)
    for i, (label, report) in enumerate(samples):
        rows = [report['paths'][name] for name in names]
        stock = np.array([row['stock_pass'] for row in rows]) / report['events']
        replay = np.array([row['replay_pass'] for row in rows]) / report['events']
        same = 100 * (1 - np.array([row['mismatches'] for row in rows]) / report['events'])
        s = summary(report)
        offset = (i - (len(samples) - 1) / 2) * width
        cost = f', {costs[label]}' if label in costs else ''
        top.plot(x + offset, np.maximum(stock, floor), MARKERS[i], color=COLORS[i], ms=6, ls='none',
                 label=f"{label} stored: {s['paths_exact']}/{s['paths']} paths exact, {s['events']:,} events{cost}")
        top.plot(x + offset, np.maximum(replay, floor), MARKERS[i], mfc='none', mec='black', mew=0.9, ms=9,
                 ls='none', label='replay (open marker)' if i == len(samples) - 1 else None)
        bottom.plot(x + offset, same, MARKERS[i], color=COLORS[i], ms=6, ls='none', label=label)
        for xi, row, value in zip(x, rows, same):
            if row['mismatches']:
                decimals = max(2, int(np.ceil(-np.log10(100 - value))) + 1)
                bottom.annotate(f"{value:.{decimals}f}% ({row['mismatches']:,})", (xi + offset, value),
                                fontsize=7, ha='left', va='center', xytext=(6, -9 if i == 0 else 0),
                                textcoords='offset points',
                                bbox={'boxstyle': 'square,pad=0.1', 'fc': 'white', 'ec': 'none', 'alpha': 0.85})
    top.set_yscale('log')
    top.axhline(floor, color='0.6', lw=0.8, ls=':')
    top.text(-0.5, floor * 1.15, 'no event accepted', ha='left', va='bottom', fontsize=8, color='0.4')
    top.set_ylim(floor * 0.6, 1.5)
    top.set_ylabel('accepted fraction of stored events')
    top.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(samples) + 1, frameon=False)
    top.grid(axis='y', color='0.9', lw=0.6)
    bottom.set_ylabel('events with the\nsame decision (%)')
    stateful = set().union(*(report['stateful_paths'] for _, report in samples))
    for xi, name in zip(x, names):
        if name in stateful:
            for axis in (top, bottom):
                axis.axvspan(xi - 0.5, xi + 0.5, color='0.92', zorder=0, lw=0)
    bottom.set_xticks(x, names, rotation=90, fontsize=7)
    for tick, name in zip(bottom.get_xticklabels(), names):
        if name in stateful:
            tick.set_fontweight('bold')
    lowest = min(100 * (1 - row['mismatches'] / r['events']) for _, r in samples for row in r['paths'].values())
    bottom.set_ylim(min(99.5, lowest - 0.4), 100.25)
    bottom.axhline(100, color='0.6', lw=0.8, ls=':')
    bottom.grid(axis='y', color='0.9', lw=0.6)
    bottom.set_xlim(-0.7, len(names) - 0.3)
    top.text(0.42, 0.16, 'shaded, bold: paths with a stateful counter\n(lumiStream subrun/buffer state, '
             'HelixFilter phi-prescale call count)', transform=top.transAxes, fontsize=8, va='bottom',
             ha='center')
    for axis in (top, bottom):
        axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(output, dpi=200, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', action='append', required=True, help='LABEL=compare.py report')
    parser.add_argument('--cost', action='append', default=[], help='LABEL=stock trigger cost text for the legend')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numbers', type=Path)
    args = parser.parse_args()
    samples = [load(spec) for spec in args.sample]
    costs = dict(spec.split('=', 1) for spec in args.cost)
    plot(samples, costs, args.output)
    if args.numbers:
        numbers = {label: {**summary(report), **({'stock_ms_per_event': costs[label]} if label in costs else {})}
                   for label, report in samples}
        args.numbers.write_text(json.dumps(numbers, indent=2) + '\n')


if __name__ == '__main__':
    main()
