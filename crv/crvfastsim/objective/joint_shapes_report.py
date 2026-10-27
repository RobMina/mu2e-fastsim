"""Compact retained-shape results; no ROOT read, simulation, refit or uncertainty resampling."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import joint_shape_stats as S, validation as V


def validate(report):
    if report.get('comparison') != 'joint-retained-pulse-shapes' or report.get('completed') is not True:
        raise ValueError('incomplete shape report')
    if report.get('full_file') is not True:
        raise ValueError('summary requires the full original data file, not a pilot')
    for mode in ('band', 'exact'):
        if report['original_population_replay'][mode]['full_census_and_moments_equal'] is not True:
            raise ValueError('missing historical data population gate')
    for block in report['fixed_support'].values():
        for mode in block.values():
            for row in mode.values():
                if not row['supported']:
                    continue
                m = row['means']
                if not np.isclose(m['le2']-m['dt2'], S.ALPHA**2*m['dbeta2']-2*S.ALPHA*m['dt_dbeta'],
                                  atol=1e-8, rtol=1e-10):
                    raise ValueError('shape covariance identity mismatch')
                if not 0 <= row['shape_pair_coverage'] <= 1:
                    raise ValueError('invalid shape coverage')
                if row['shape_pair_coverage'] == 1 and not np.isclose(
                        row['historical_data']['data_weights']['second_moment'], m['dt2'], atol=1e-8, rtol=1e-10):
                    raise ValueError('original timing moment changed')


def markdown(report):
    validate(report)
    lines = ['# Retained P4 pulse-shape associations', '',
             f"{report['events']:,} events; {report['selected_flow']['selected_first_pulses']:,} exact joined selected pulses; "
             f"{report['selected_flow']['retained_members']:,} original timing members.",
             'Data only, fixed previous common-support cells/weights; no response/conditions change.',
             'SEs cluster by full event and condition on support/weights; serial dependence and support learning are excluded.',
             'Derived LE is peak minus 0.985 beta algebra, not an independent measured time or a calibrated resolution.', '',
             '## T1 exact-N support', '',
             '| Support report | Body | Pairs kept/original; events | Shape retention | Peak² ± SE | Derived LE² ± SE | Difference ± SE | Beta² term | Covariance term |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: f'{x:.4f}'
    for path, block in report['fixed_support'].items():
        for body in ('T10', 'T11'):
            r = block['exact'][body]
            if not r['supported']:
                lines.append(f"| {Path(path).name} | {body} | unsupported: {r['reason']}; "
                             f"{r.get('empty_cells', 0)} empty of {r['cells']} frozen cells | | | | | | |")
                continue
            m, se, alg = r['means'], r['conditional_event_se'], r['algebra']
            lines.append(f"| {Path(path).name} | {body} | {r['pairs']}/{r['historical_data']['common_pairs']}; "
                         f"{r['contributing_events']} | {r['shape_pair_coverage']:.2%} | "
                         f"{fmt(m['dt2'])} ± {fmt(se['dt2'])} | {fmt(m['le2'])} ± {fmt(se['le2'])} | "
                         f"{fmt(r['le_minus_peak'])} ± {fmt(r['le_minus_peak_conditional_event_se'])} | "
                         f"{fmt(alg['beta_term'])} | {fmt(alg['covariance_term'])} |")
    lines += ['', '## All-body banded-N data associations', '',
              '| Support report | Body | Cells | Historical data coverage | Shape retention | Peak² | Derived LE² | Beta low/high | Negative-chi2 pair fraction |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for path, block in report['fixed_support'].items():
        for body, r in block['band'].items():
            if not r['supported']:
                lines.append(f"| {Path(path).name} | {body} | {r['cells']} | unsupported: {r['reason']}; "
                             f"{r.get('empty_cells', 0)} empty cells | | | | | |")
                continue
            m = r['means']
            lines.append(f"| {Path(path).name} | {body} | {r['cells']} | {r['historical_data']['coverage']:.1%} | "
                         f"{r['shape_pair_coverage']:.2%} | "
                         f"{fmt(m['dt2'])} | {fmt(m['le2'])} | {m['beta_low']:.3f}/{m['beta_high']:.3f} | "
                         f"{m['either_chi2_negative']:.3%} |")
    if 'model_implied_calibration' in report:
        lines += ['', '## Model-implied calibration ratios', '',
                  'Channel median gains inferred algebraically, divided by the frozen MC constants.',
                  'Not a historical DB-payload certificate or measured contribution to the timing excess.', '',
                  '| Body | Channels | Area ratio median (min–max) | Height ratio median (min–max) | Largest within-channel relative range area/height |',
                  '|---|---:|---:|---:|---:|']
        for name, row in report['model_implied_calibration']['bodies'].items():
            a, h = row['area_ratio_channel_medians'], row['height_ratio_channel_medians']
            if a['count']:
                lines.append(f"| {name} | {row['channels']} | {a['median']:.4f} ({a['min']:.4f}–{a['max']:.4f}) | "
                             f"{h['median']:.4f} ({h['min']:.4f}–{h['max']:.4f}) | "
                             f"{row['max_area_channel_full_range_relative']:.2g}/{row['max_height_channel_full_range_relative']:.2g} |")
    lines += ['', '## Limits', '', *['- '+line for line in report['cautions']], '']
    return '\n'.join(lines)


def run(path, out):
    source = Path(path)
    before = V.digest(source)
    report = json.loads(source.read_text())
    text = markdown(report)
    if V.digest(source) != before:
        raise ValueError('source report changed')
    if out is not None:
        dest = Path(out)
        dest.mkdir(parents=True, exist_ok=False)
        (dest / 'summary.md').write_text(text)
        (dest / 'receipt.json').write_text(json.dumps(dict(source=str(source), sha256=before,
            reporter={str(p): V.digest(p) for p in (Path(__file__), Path(S.__file__))}), indent=2))
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    p.add_argument('--out')
    args = p.parse_args(argv)
    run(args.report, args.out)


if __name__ == '__main__':
    main()
