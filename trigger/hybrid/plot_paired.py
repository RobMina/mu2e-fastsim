"""Per-path acceptance of two paired replays (pair.py report) and their paired difference with errors.

Use: python3 trigger/hybrid/plot_paired.py paired.json --output paired.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REFERENCE, CANDIDATE = '#2a78d6', '#eb6834'


def plot(report, output, title=None, names=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ref, cand = report['reference'], report['candidate']
    ref_name, cand_name = names or (ref, cand)
    names = list(report['paths'])
    rows = [report['paths'][name] for name in names]
    x = np.arange(len(names))
    tested = np.array([row['status'] == 'tested' for row in rows])
    floor = 0.3 / report['events']
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True,
                                      gridspec_kw={'height_ratios': [1.6, 1], 'hspace': 0.06})
    for xi, ok in zip(x, tested):
        if not ok:
            for axis in (top, bottom):
                axis.axvspan(xi - 0.5, xi + 0.5, color='0.93', zorder=0, lw=0)
    r = np.array([row[f'{ref}_acceptance'] for row in rows])
    c = np.array([row[f'{cand}_acceptance'] for row in rows])
    top.plot(x[tested] - 0.15, np.maximum(r, floor)[tested], 'o', color=REFERENCE, ms=6, ls='none',
             label=f'{ref_name} digis')
    top.plot(x[tested] + 0.15, np.maximum(c, floor)[tested], 's', color=CANDIDATE, ms=6, ls='none',
             label=f'{cand_name} digis')
    top.set_yscale('log')
    top.set_ylim(max(floor, 0.5 * min(v for v in np.concatenate([r[tested], c[tested]]) if v > 0)), 1.5)
    top.set_ylabel(f'acceptance\n({report["events"]:,} events)')
    head = title or f'{cand_name} vs {ref_name} digis, same events, {report.get("event_mode") or ""} trigger replay'
    top.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, title=head,
               title_fontsize=9, alignment='left')
    diff = 100 * np.array([row['paired_difference'] for row in rows])
    error = 100 * np.array([row['paired_error'] for row in rows])
    bottom.axhline(0, color='0.5', lw=0.8)
    bottom.errorbar(x[tested], diff[tested], yerr=error[tested], fmt='s', color=CANDIDATE, ms=5, capsize=2,
                    elinewidth=1)
    for xi, row, d, e in zip(x, rows, diff, error):
        if row['status'] == 'tested' and row['pull'] is not None and abs(row['pull']) > 2:
            bottom.annotate(f"{row['pull']:+.1f}σ", (xi, d - e if d < 0 else d + e), fontsize=6.5, ha='center',
                            va='top' if d < 0 else 'bottom', xytext=(0, -3 if d < 0 else 3),
                            textcoords='offset points', rotation=90)
    bottom.set_ylabel(f'{cand_name} − {ref_name}\n(pp, paired)')
    low = float(np.min(diff[tested] - error[tested])) if tested.any() else -0.5
    high = float(np.max(diff[tested] + error[tested])) if tested.any() else 0.5
    pad = 0.35 * max(high - low, 1.0)
    bottom.set_ylim(min(low, -0.5) - pad, max(high, 0.5) + pad)
    bottom.set_xticks(x, names, rotation=90, fontsize=7)
    for tick, row in zip(bottom.get_xticklabels(), rows):
        if row['status'] != 'tested':
            tick.set_color('0.55')
    top.text(0.66, 0.22, 'grey: not tested (disabled in this event mode,\nor no event accepted in either arm)',
             transform=top.transAxes, fontsize=8, ha='center', va='bottom', color='0.35')
    bottom.set_xlim(-0.7, len(names) - 0.3)
    for axis in (top, bottom):
        axis.grid(axis='y', color='0.9', lw=0.6)
        axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(output, dpi=200, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='pair.py output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title')
    parser.add_argument('--names', help='display names REFERENCE,CANDIDATE (default: the report labels)')
    args = parser.parse_args()
    plot(json.loads(args.report.read_text()), args.output, args.title,
         args.names.split(',') if args.names else None)


if __name__ == '__main__':
    main()
