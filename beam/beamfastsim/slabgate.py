"""Report saved slabstudy diagnostics: --study DIR [DIR ...] --out NEW.json.

No transport is run. All gates are pointwise threshold-transform diagnostics,
not physical-stop or ordinary exit-energy closure, and never a global pass.
Historical source identities are retained, not checked against current sources.
"""
import argparse
import json
from pathlib import Path

from . import slabstudy
from .reference_audit import _identity

UNVERIFIED = 'UNVERIFIED'
SCOPE = __doc__


def _gate(value, fields):
    value = value or {}
    return dict({k: value.get(k) for k in fields}, state=value.get('state', UNVERIFIED))


def _threshold(summary, reference, comparison, tolerance):
    fast = summary.get('classes', {}).get('threshold_stop', {})
    ref = reference.get('fraction_intervals', {}).get('threshold_stop', {})
    delta = comparison.get('classes', {}).get('threshold_stop', {})
    interval = delta.get('interval_99pct')
    state = UNVERIFIED
    if interval is not None and tolerance is not None:
        lo, hi = interval
        state = ('FAIL' if lo > tolerance or hi < -tolerance else
                 'PASS' if lo >= -tolerance and hi <= tolerance else 'INDETERMINATE')
    return dict(fast_fraction_bounds=fast.get('fraction_bounds'),
                reference_fraction_bounds=ref.get('fraction_bounds'),
                delta_fraction_bounds=delta.get('delta_fraction_bounds'),
                interval_99pct=interval, tolerance=tolerance, state=state)


def rows(report, study_index):
    args, refs = report['args'], report.get('g4', {})
    audit = report.get('split_mean_audit') or {}
    for arm, item in report['arms'].items():
        family = 'loss' if arm == 'no_scatter' else 'full'
        preferred = [family + suffix for suffix in ('_finest', '_finer', '_fine', '')]
        label = next((name for name in preferred if name in refs), None)
        reference = refs.get(label, {})
        comparison = item.get('vs_g4', {}).get(label) or {}
        distribution = comparison.get('threshold_aligned_distribution') or {}
        summary = item['summary']
        mean = _gate(distribution.get('mean_exit'), ('delta', 'se', 'tolerance'))
        variance = _gate(distribution.get('variance'), ('delta', 'se', 'tolerance'))
        fast_var = distribution.get('candidate', {}).get('variance')
        ref_var = distribution.get('reference', {}).get('variance')
        variance.update(fast=fast_var, reference=ref_var,
                        ratio=fast_var / ref_var if fast_var is not None and ref_var else None,
                        relative=variance['delta'] / ref_var if variance['delta'] is not None and ref_var else None)
        cdf = _gate(distribution.get('cdf'), ('distance', 'interval', 'radius_99pct_dkw'))
        endpoint = summary.get('endpoint_identification') or {}
        yield dict(study_index=study_index, cell=args['cell'], pdg=args['pdg'],
                   material=report.get('material', UNVERIFIED),
                   loss_model=args.get('loss_model', 'hybrid'), arm=arm, reference=label,
                   reference_state='MATCHED' if label and comparison else UNVERIFIED,
                   reference_configuration_qualification=report.get('reference_configuration_qualification', UNVERIFIED),
                   mean=mean, variance=variance, cdf=cdf,
                   threshold=_threshold(summary, reference, comparison, report.get('tolerances', {}).get('stop_abs')),
                   threshold_distribution_identified=comparison.get('threshold_distribution_identified'),
                   endpoint_counts=dict(fast=summary.get('endpoint_counts'), reference=reference.get('endpoint_counts')),
                   n=dict(fast=summary.get('n'), reference=reference.get('n')),
                   unresolved=dict(fast=summary.get('unresolved'), reference=reference.get('unresolved', 0) if label else None),
                   audits=dict(range=audit.get('range_interpolation_audit'),
                               mean={k: v for k, v in audit.items() if not isinstance(v, (list, dict))},
                               raw_mean_bounds_MeV=endpoint.get('raw', {}).get('mean_bounds_MeV'),
                               threshold_mean_bounds_MeV=endpoint.get('thresholded', {}).get('mean_bounds_MeV'),
                               csda_comparison=endpoint.get('csda_comparison')))


def summarize(directories):
    result = dict(schema_version=1, scope=SCOPE, physical_stop_closure=False,
                  global_pass=None, reporting_sources=[_identity(Path(__file__))], studies=[], rows=[])
    checked = []
    for index, directory in enumerate(directories):
        path = Path(directory).resolve() / 'summary.json'
        identity = _identity(path)
        report = json.loads(path.read_text())
        if report.get('complete') is not True:
            raise ValueError(f'completed slabstudy required: {path}')
        records = [identity] + report['inputs'] + report['outputs']
        slabstudy.verify(records)
        checked.extend(records)
        historical = {key: report.get(key) for key in ('sources', 'core_sources', 'response_generation_provenance')}
        result['studies'].append(dict(original_report=identity, inputs=report['inputs'], outputs=report['outputs'],
                                     historical_source_provenance=historical, historical_sources_verified=False,
                                     generation={key: report.get(key) for key in
                                                 ('args', 'environment', 'transport_binding', 'sample_identity')},
                                     threshold_MeV=report.get('threshold_MeV'), tolerances=report.get('tolerances'),
                                     reference_configuration_qualification=report.get('reference_configuration_qualification', UNVERIFIED)))
        result['rows'].extend(rows(report, index))
    slabstudy.verify(checked)
    return result


def _format(value):
    if value is None:
        return UNVERIFIED
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(_format(v) for v in value) + ']'
    return str(value)


def print_table(report):
    print('study\tcell\tPDG\tarm\tref\tfast_threshold\tref_threshold\tvariance_ratio\tCDF\tmean_state\tvariance_state\tCDF_state\tthreshold_state\tunresolved')
    for row in report['rows']:
        values = [row['study_index'], row['cell'], row['pdg'], row['arm'], row['reference'],
                  row['threshold']['fast_fraction_bounds'], row['threshold']['reference_fraction_bounds'],
                  row['variance']['ratio'], row['cdf']['distance'], row['mean']['state'],
                  row['variance']['state'], row['cdf']['state'], row['threshold']['state'], row['unresolved']['fast']]
        print('\t'.join(_format(v) for v in values))
    print('# All states are diagnostics, not physical-stop closure or a global pass.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', nargs='+', required=True)
    parser.add_argument('--out', required=True, help='new JSON file; never overwritten')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    report = summarize(args.study)
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(content)
    print_table(report)
    return report


if __name__ == '__main__':
    main()
