"""Plots of the saved-state response diagnosis; no response generation or refitting."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SAMPLES = ('data', 'nominal', 'best')
COLORS = dict(data='black', nominal='#277da1', best='#d1495b')
LABELS = dict(data='P4 data', nominal='Nominal', best='Pilot best')


def _finish(fig, path, title):
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.text(0.5, 0.01, 'Saved in-range histograms only; binned estimates; no significance or causal attribution.',
             ha='center', fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _curves(ax, edges, samples, xlabel):
    for k in SAMPLES:
        ax.stairs(samples[k]['probability'], edges, color=COLORS[k], label=LABELS[k])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('In-range probability / bin')
    ax.grid(alpha=0.2)


def scalar_plots(report, dest):
    for block, data in report['blocks'].items():
        fig, axs = plt.subplots(4, 3, figsize=(13, 12))
        for row, (slot, info) in enumerate(data['sectors'].items()):
            for col, (key, label) in enumerate((('pe_layer', 'Positive cluster layer PE'),
                                                 ('tspread', 'Corrected first-pulse RMS [ns]'),
                                                 ('dt_cluster', 'Selected cluster minus EX [ns]'))):
                ax = axs[row, col]
                if key not in info['scalar']:
                    ax.axis('off')
                    continue
                item = info['scalar'][key]
                _curves(ax, item['edges'], item['samples'], label)
                ax.set_title(f'{slot}: {key}')
                if row == 0 and col == 0:
                    ax.legend(fontsize=9)
        _finish(fig, dest / f'shapes_{block}.png', f'Held-out {block.upper()}: data / nominal / pilot best')


def conditional_plots(report, dest):
    for block, data in report['blocks'].items():
        fig, axs = plt.subplots(4, 3, figsize=(14, 12))
        for row, (slot, info) in enumerate(data['sectors'].items()):
            for side in (0, 1):
                ax = axs[row, side]
                cells = [c for c in info['pe_along']['cells'] if c['side'] == side]
                x = [(c['along_low'] + c['along_high']) / 2 for c in cells]
                for k in SAMPLES:
                    y = [c['samples'][k]['q50'] if c['common_support'] else np.nan for c in cells]
                    ax.plot(x, y, 'o-', color=COLORS[k], label=LABELS[k], markersize=4)
                if not any(c['common_support'] for c in cells):
                    ax.text(0.5, 0.5, 'No common populated cells', ha='center', transform=ax.transAxes)
                ax.set(title=f'{slot}, readout side {side}', xlabel='Measured along / half-length',
                       ylabel='Median pulse PE / path factor')
                ax.grid(alpha=0.2)
            cells = info['pe_path']['cells']
            x = [(c['path_low'] + c['path_high']) / 2 for c in cells]
            ax = axs[row, 2]
            for k in SAMPLES:
                y = [c['samples'][k]['q50'] if c['common_support'] else np.nan for c in cells]
                ax.plot(x, y, 'o-', color=COLORS[k], label=LABELS[k], markersize=4)
            ax.set(title=f'{slot}, raw PE vs measured path', xlabel='Measured path factor', ylabel='Median pulse PE')
            ax.grid(alpha=0.2)
        axs[0, 0].legend(fontsize=9)
        _finish(fig, dest / f'conditional_pe_{block}.png',
                f'Held-out {block.upper()}: common populated cells only (threshold in JSON)')


def timing_plots(report, dest):
    for block, data in report['blocks'].items():
        fig, axs = plt.subplots(3, 2, figsize=(15, 11))
        for row, slot in enumerate(('T1', 'T2', 'TAG')):
            item = data['sectors'][slot]['module_timing']
            cells = [c for c in item['cells'] if c['common_support']]
            x = np.arange(len(cells))
            labels = [f"{c['ex_module']}/{c['module']}" for c in cells]
            for col, key in enumerate(('q50', 'rms')):
                ax = axs[row, col]
                for k in SAMPLES:
                    vals = [c['samples'][k][key] for c in cells]
                    ax.plot(x, vals, 'o-', color=COLORS[k], label=LABELS[k], markersize=4)
                ax.set_xticks(x, labels, rotation=65, ha='right', fontsize=8)
                ax.set(title=f'{slot}: module-pair {key}', ylabel=f'In-range {key} [ns]')
                ax.grid(alpha=0.2)
        axs[0, 0].legend(fontsize=9)
        _finish(fig, dest / f'module_timing_{block}.png',
                f'Held-out {block.upper()}: no timing refit; same P4 shifts/background')
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.7))
        for ax, slot in zip(axs, ('T1', 'T2', 'TAG')):
            item = data['sectors'][slot]['module_timing']
            if item['standardized']:
                _curves(ax, item['edges'], item['standardized'], 'Selected cluster minus EX [ns]')
            else:
                ax.text(0.5, 0.5, 'No common populated cells', ha='center', transform=ax.transAxes)
            ax.set_title(f'{slot}: fixed data module-pair fractions')
            ax.set_xlim(-22, 22)
        if axs[0].get_legend_handles_labels()[0]:
            axs[0].legend(fontsize=9)
        _finish(fig, dest / f'timing_standardized_{block}.png',
                f'Held-out {block.upper()}: mixture diagnostic, not a new objective')


def score_plots(report, dest):
    for block, data in report['blocks'].items():
        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        for row, (name, label) in enumerate((('pe_layer', 'Positive cluster layer PE'),
                                            ('tspread', 'Corrected first-pulse RMS [ns]'),
                                            ('dt_cluster', 'Selected cluster minus EX [ns]'))):
            item = data['sectors']['T1']['scalar'][name]
            _curves(axs[row, 0], item['edges'], item['samples'], label)
            axs[row, 0].set_title(f'T1: {name}')
            dec = item.get('score_decomposition')
            if dec:
                edges = np.asarray(item['edges'])
                x = (edges[1:] + edges[:-1]) / 2
                delta = np.asarray(dec['bin_best_minus_nominal'])
                axs[row, 1].bar(x, delta, width=np.diff(edges), color=np.where(delta < 0, '#277da1', '#d1495b'))
                axs[row, 1].axhline(0, color='black', linewidth=0.7)
                axs[row, 1].set(title=f"Term change = {dec['best_minus_nominal']:+.3f}",
                                xlabel=label, ylabel='Contribution to term best minus nominal')
        axs[0, 0].legend(fontsize=9)
        _finish(fig, dest / f't1_score_bins_{block}.png',
                f'Held-out {block.upper()}: saved-score decomposition (negative improves)')


def plot_controls(report, directory):
    dest = Path(directory)
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True, exist_ok=False)
    labels = ('P4 data', 'Nominal', 'Full best', 'T1 attenuation only', 'EX speed only')
    colors = ('black', '#277da1', '#d1495b', '#2a9d8f', '#9467bd')
    for block, row in report['blocks'].items():
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (name, xlabel) in zip(axs, (('pe_layer', 'Positive cluster layer PE'),
                                           ('tspread', 'Corrected first-pulse RMS [ns]'),
                                           ('dt_cluster', 'Selected cluster minus EX [ns]'))):
            full = row['full_best']['sectors']['T1']['scalar'][name]
            curves = [full['samples'][k]['probability'] for k in ('data', 'nominal', 'best')]
            curves += [row['controls'][control]['sectors']['T1']['scalar'][name]['samples']['best']['probability']
                       for control in ('t1atten', 'exspeed')]
            for curve, label, color in zip(curves, labels, colors):
                ax.stairs(curve, full['edges'], label=label, color=color)
            ax.set(title=f'T1: {name}', xlabel=xlabel, ylabel='In-range probability / bin')
            ax.grid(alpha=0.2)
        axs[0].legend(fontsize=8)
        _finish(fig, dest / f'control_shapes_{block}.png', f'Block {block.upper()}: isolated response controls, fixed P4 calibration')
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        candidates = [row['full_best'], row['controls']['t1atten'], row['controls']['exspeed']]
        terms = ('pe_layer_T1', 'tspread_T1', 'dt_cluster_T1')
        x = np.arange(3)
        for i, (candidate, label, color) in enumerate(zip(candidates, labels[2:], colors[2:])):
            shifts = [candidate['saved_score_deltas']['terms'][term]['candidate_minus_nominal'] for term in terms]
            axs[0].bar(x + (i - 1) * 0.25, shifts, width=0.25, label=label, color=color)
            times = [candidate['timing_mean_shifts_ns'][slot]['dt_cluster'] for slot in ('T1', 'T2', 'TAG')]
            axs[1].bar(x + (i - 1) * 0.25, times, width=0.25, label=label, color=color)
        axs[0].set_xticks(x, terms)
        axs[0].set(title='Saved term change (negative improves)', ylabel='Term candidate minus nominal')
        axs[1].set_xticks(x, ('T1', 'T2', 'TAG'))
        axs[1].set(title='Mean cluster time relative to EX', ylabel='Candidate minus nominal [ns]')
        for ax in axs:
            ax.axhline(0, color='black', linewidth=0.7)
            ax.grid(axis='y', alpha=0.2)
        axs[0].legend(fontsize=8)
        _finish(fig, dest / f'control_effects_{block}.png', f'Block {block.upper()}: contrasts, not an additive decomposition')


def plot_report(report, directory):
    dest = Path(directory)
    scalar_plots(report, dest)
    conditional_plots(report, dest)
    timing_plots(report, dest)
    score_plots(report, dest)


def main():
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-report', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    report = json.loads(Path(a.control_report).read_text())
    if report.get('kind') != 'saved-control-response-diagnosis':
        p.error('expected a saved control diagnosis report')
    plot_controls(report, a.out)


if __name__ == '__main__':
    main()
