"""Reduce saved pair reports to checked summary tables and plots; no capture reads or fit."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import joint_compare as J, joint_pairs as P, validation as V


def validate(report):
    J.require(report.get('comparison') == 'joint-pairs' and report.get('passed') is True, 'incomplete pair report')
    for sample in report['samples'].values():
        J.require(sample['legacy_pair_census_and_moments_equal'] is True, 'legacy pair gate missing')
        for name in P.T.G.BODY_NAMES:
            f = sample['pair_flow'][name]
            J.require(f['all_pairs'] == f['excluded_pe_pairs']+f['excluded_coordinates_after_pe_pairs']+f['inrange_pairs'],
                      'exclusive pair flow mismatch')
        for rows in sample['strata'].values():
            for c in rows:
                J.require(np.isclose(c['second_moment'], c['variance']+c['mean']**2, rtol=1e-10, atol=1e-8),
                          'pair variance identity mismatch')
    for mode in report['standardization'].values():
        for body in mode.values():
            for r in [body['all_pe']] + body['diagonal_pe']:
                n = r['common_cells']
                J.require(len(r['data_pair_weights']) == n and (not n or np.isclose(sum(r['data_pair_weights']), 1.)),
                          'support weight mismatch')
                for sample in r['samples'].values():
                    J.require(sample['common_pairs'] <= sample['inrange_pairs'], 'invalid support census')
    for r in report['fixed_support_multiplicity'].values():
        if r['common_exact_cells']:
            J.require(abs(r['samples']['data']['second_moment_reweighting']) < 1e-10,
                      'data target changed in fixed-support contrast')


def multiplicity_leverage(row):
    keys = np.asarray(row['keys'], dtype=np.int64)
    if not len(keys):
        return dict(coarse_cells=0, multi_N_coarse_cells=0, data_weight_in_multi_N_coarse_cells=0.)
    coarse, inv, counts = np.unique(keys[:, :6], axis=0, return_inverse=True, return_counts=True)
    multi = counts > 1
    weights = np.asarray(row['data_pair_weights'])
    return dict(coarse_cells=len(coarse), multi_N_coarse_cells=int(multi.sum()),
                data_weight_in_multi_N_coarse_cells=float(weights[multi[inv]].sum()))


def summarize(reports):
    result = dict(schema=1, comparison='joint-pairs-summary', blocks={})
    data = None
    for block, r in reports.items():
        validate(r)
        if data is not None:
            J.require(data == r['samples']['data'], 'independent A/B measured data reductions differ')
        data = r['samples']['data']
        support = {}
        for mode, bodies in r['standardization'].items():
            support[mode] = {name: {which: [dict(common_cells=v['common_cells'], samples=v['samples']) for v in value]
                                    if which == 'diagonal_pe' else dict(common_cells=value['common_cells'], samples=value['samples'])
                                    for which, value in body.items()} for name, body in bodies.items()}
        leverage = {name: multiplicity_leverage(row['all_pe']) for name, row in r['standardization']['exact'].items()}
        result['blocks'][block] = dict(support=support, fixed_support_multiplicity=r['fixed_support_multiplicity'],
            multiplicity_leverage=leverage,
            flows={name: s['pair_flow'] for name, s in r['samples'].items()})
    result['identical_measured_data_diagnostics'] = True
    return result


def _fmt(value, percent=False):
    return 'unsupported' if value is None else f'{value:.1%}' if percent else f'{value:.3f}'


def markdown(summary):
    lines = ['# Thread C: common-support same-bar/end SiPM timing', '',
      'All values below are mean pair time-difference² [ns²], not single-sensor variance.',
      'PE bins are sensor oriented; group means and line propagation cancel. No calibration/background refit.', '',
      '## T1 exact-N support (body, side, along, path, both sensor PEs, exact multiplicity)', '',
      '| Block/body | Cells | Nominal | Atten | Yield | Joint | Data | Data pair coverage |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for block, r in summary['blocks'].items():
        for body in ('T10', 'T11'):
            row = r['support']['exact'][body]['all_pe']
            s = row['samples']
            vals = [s[a]['data_weights']['second_moment'] for a in (*J.ARMS, 'data')]
            lines.append(f"| {block}/{body} | {row['common_cells']} | "+' | '.join(map(_fmt, vals))+
                         f" | {_fmt(s['data']['coverage'], True)} |")
    lines += ['', '## Same exact-N-supported rows: multiplicity-mixture reweighting', '',
              'Fine-data-weighted minus own-N-mixture/coarse-data-weighted moment; identical retained rows.', '',
              '| Block/body | Fine / coarse cells | Nominal | Atten | Yield | Joint | Multi-N cells / data weight |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for block, r in summary['blocks'].items():
        for body in ('T10', 'T11'):
            row = r['fixed_support_multiplicity'][body]
            vals = [row['samples'][a]['second_moment_reweighting'] for a in J.ARMS] if row['common_exact_cells'] else [None]*4
            lever = r['multiplicity_leverage'][body]
            lines.append(f"| {block}/{body} | {row['common_exact_cells']} / {row['coarse_cells']} | "+
                         ' | '.join(map(_fmt, vals))+f" | {lever['multi_N_coarse_cells']} / " +
                         f"{lever['data_weight_in_multi_N_coarse_cells']:.1%} |")
    lines += ['', '## All-body banded-N comparison and coverage', '',
              '| Block/body | Cells | Nominal | Joint | Data | Data pair coverage |',
              '|---|---:|---:|---:|---:|---:|']
    for block, r in summary['blocks'].items():
        for body, entry in r['support']['band'].items():
            row = entry['all_pe']
            s = row['samples']
            vals = [s[a]['data_weights']['second_moment'] for a in ('nominal', 'joint', 'data')]
            lines.append(f"| {block}/{body} | {row['common_cells']} | "+' | '.join(map(_fmt, vals))+
                         f" | {_fmt(s['data']['coverage'], True)} |")
    lines += ['', '## Interpretation limits', '',
      '- Exact-N support is sparse; excluded cells remain unresolved. Modes have different targets/support.',
      '- Within-bin PE means are recorded and not forced equal; measured cell conditioning is not exact continuous matching.',
      '- Pair counts and group-cell incidences are correlated, not independent statistics. No significance assigned.',
      '- Historical P4 physical mapping is not certified; sensor identity here follows retained offline IDs.',
      '- Arrival fluctuations versus pulse fitting remain unidentifiable without suitable upstream observables.', '']
    return '\n'.join(lines)


def plots(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'nominal': 'tab:blue', 'atten': 'tab:orange', 'yield': 'tab:green', 'joint': 'tab:red', 'data': 'black'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(5)
    labels = ['0–20', '20–40', '40–60', '60–100', '100–200']
    for j, body in enumerate(('T10', 'T11')):
        for block, report in summary['blocks'].items():
            rows = report['support']['band'][body]['diagonal_pe']
            style = '-' if block == 'A' else '--'
            for name in (*J.ARMS, 'data'):
                y = [r['samples'][name]['data_weights']['second_moment'] for r in rows]
                axes[0, j].plot(x, y, marker='o', linestyle=style, color=colors[name], label=f'{name} {block}')
            coverage = [r['samples']['data']['coverage'] for r in rows]
            axes[1, j].plot(x, coverage, marker='o', linestyle=style, color='black', label=f'data support {block}')
            for p, r in enumerate(rows):
                axes[1, j].annotate(str(r['common_cells']), (p, coverage[p] or 0.),
                                   xytext=(0, 7 if block == 'A' else -13), textcoords='offset points', ha='center', fontsize=8)
        axes[0, j].set_title(f'{body}: fixed body/side/along/path/two-PE/N-band fractions')
        axes[0, j].set_ylabel('mean pair difference² [ns²]')
        axes[0, j].set_ylim(bottom=0)
        axes[1, j].set_title('Data common/in-range pair coverage; labels = common cells')
        axes[1, j].set_ylabel('data pair coverage')
        axes[1, j].set_ylim(-.03, 1.)
        for row in range(2):
            axes[row, j].set_xticks(x, labels)
            axes[row, j].set_xlabel('both sensor PEs in this bin')
            axes[row, j].grid(alpha=.25)
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Same bar + end, distinct sensors · common support · descriptive only, no refit', fontsize=11)
    fig.savefig(out/'pair_pe_common_support.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    modes = ('none', 'band', 'exact', 'layer_band')
    for j, body in enumerate(('T10', 'T11')):
        for block, report in summary['blocks'].items():
            for name in ('nominal', 'joint', 'data'):
                y = [report['support'][m][body]['all_pe']['samples'][name]['data_weights']['second_moment'] for m in modes]
                axes[j].plot(range(4), y, marker='o', linestyle='-' if block == 'A' else '--', color=colors[name], label=f'{name} {block}')
        axes[j].set_title(body)
        axes[j].set_xticks(range(4), ['no N', 'N band', 'exact N', 'layer + N band'])
        axes[j].set_ylabel('mean pair difference² [ns²]')
        axes[j].set_ylim(bottom=0)
        axes[j].grid(alpha=.25)
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle('Each mode has different support: do not read mode-to-mode shifts as a multiplicity-only effect', fontsize=10)
    fig.savefig(out/'pair_support_modes.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('a', 'b', 'outdir'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args(argv)
    out = Path(args.outdir)
    J.require(not out.exists(), 'output directory already exists')
    receipts = {}
    for source in (__file__, P.__file__, J.__file__, V.__file__):
        J._receipt(source, receipts)
    reports = {block: J._json(path, receipts) for block, path in (('A', args.a), ('B', args.b))}
    summary = summarize(reports)
    del reports
    summary['files'] = list(receipts.values())
    out.mkdir(parents=True)
    plots(summary, out)
    (out/'summary.md').write_text(markdown(summary))
    for r in receipts.values():
        J.require(V.digest(r['path']) == r['sha256'], 'report source/input changed')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(out/'summary.md')


if __name__ == '__main__':
    main()
