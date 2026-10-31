"""Read-only independent step/bin contrasts with a reused refined/168 reference.

A two-level factorial separates settings in this released setup, not the cause
of an unavailable historical signature or a limit of numerical convergence.
All-primary threshold distributions retain every event. No sampler is run.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import edgefactorial as ef, edgeg4, foilreport, g4tables, physics_tables as pt
from . import lossdiagnostic as ld, lossexplain as le, slabrefs
from .foilcompare import contrast, moments
from .reference_audit import _identity

ORDER = ('stock_step_bins7', 'stock_step_bins168', 'refined_step_bins7', 'full_refined')
WEIGHTS = {'step_at_bins7': (-1, 0, 1, 0), 'step_at_bins168': (0, -1, 0, 1),
           'bins_at_stock_step': (-1, 1, 0, 0), 'bins_at_refined_step': (0, 0, -1, 1),
           'step_main_average': (-.5, -.5, .5, .5), 'bins_main_average': (-.5, .5, -.5, .5),
           'step_bin_interaction': (1, -1, -1, 1)}
PAIRS = {'step_at_bins7': (2, 0), 'step_at_bins168': (3, 1),
         'bins_at_stock_step': (1, 0), 'bins_at_refined_step': (3, 2)}


def factorial(samples):
    """Linear independent-sample contrasts; shared-arm contrast covariance retained."""
    if len(samples) != 4:
        raise ValueError('four independent factorial arms required')
    rows = []
    for sample in samples:
        y = np.asarray(sample, float)
        stats = moments(y)
        below = y < .1
        rows.append(dict(mean=stats['mean'], variance=stats['variance'], threshold_fraction=float(below.mean()),
                         mean_se=stats['mean_se'], variance_se=stats['variance_se'],
                         threshold_fraction_se=float(below.std(ddof=1) / np.sqrt(len(y)))))
    weights = np.asarray(list(WEIGHTS.values()), float)
    estimates, covariances = {}, {}
    for metric in ('mean', 'variance', 'threshold_fraction'):
        values = np.array([r[metric] for r in rows])
        variances = np.array([r[metric + '_se'] ** 2 for r in rows])
        covariance = (weights * variances) @ weights.T
        delta = weights @ values
        estimates[metric] = {name: dict(delta=float(delta[i]), se=float(np.sqrt(covariance[i, i])),
                                       signed_3se_interval=[float(delta[i] - 3 * np.sqrt(covariance[i, i])),
                                                           float(delta[i] + 3 * np.sqrt(covariance[i, i]))])
                             for i, name in enumerate(WEIGHTS)}
        covariances[metric] = covariance.tolist()
    return dict(arm_order=ORDER, contrast_order=list(WEIGHTS), weights=WEIGHTS, arms=rows,
                estimates=estimates, contrast_covariance=covariances,
                scope='Independent-primary pointwise moment/influence SEs, correlated contrasts, not simultaneous or causal; no interaction acceptance gate')


def threshold(energy):
    energy = np.asarray(energy, float)
    return np.where(energy < .1, 0., energy)


def shape_contrast(candidate, reference):
    c, r = threshold(candidate), threshold(reference)
    decomposition = le.explain(c, r)
    caps = {}
    for cap in (.3, .5, .75, 1.):
        cm, rm = moments(np.minimum(c, cap)), moments(np.minimum(r, cap))
        caps[str(cap)] = dict(candidate=cm, reference=rm, delta=cm['variance']-rm['variance'],
                              se=float(np.hypot(cm['variance_se'], rm['variance_se'])))
    return dict(variance_explanation=decomposition, capped_variance_diagnostics=caps,
                scope='Fixed all-primary caps retain every entry; diagnostic estimands only, not replacement gates or causal components')


def run(root, reused, studies, recovered, package, out):
    root, reused, out = Path(root), Path(reused), Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, scope=__doc__, physics_closure=False, default_changed=False,
                  references={}, factorial={}, pairs={}, fast={}, historical={}, inputs=[],
                  sources=[_identity(p) for p in sorted(Path(__file__).parent.glob('*.py'))])
    samples = {}
    for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
        refs, records = slabrefs.load_references([root, reused], cell, 13, thickness, energy, cosine)
        report['inputs'] += records
        if set(refs) != set(ef.MODES) | set(edgeg4.MODES):
            raise ValueError('complete new and reused control campaigns required')
        samples[cell] = {}
        runtime = None
        seeds = set()
        for mode in ORDER:
            directory = (reused if mode == 'full_refined' else root) / (cell + '_' + mode)
            receipt = json.loads((directory / 'receipt.json').read_text())
            case = receipt['case']
            if case['events'] != 5000 or case['seed'] in seeds:
                raise ValueError('independent complete 5000-event factorial arms required')
            seeds.add(case['seed'])
            for item in receipt['sources'] + receipt['artifacts'] + receipt['tables'] + [receipt['output']]:
                edgeg4.checked(item)
            observed_runtime = (directory / 'runtime.sha256').read_bytes()
            if runtime is not None and runtime != observed_runtime:
                raise ValueError('factorial runtime/binary/source identities differ')
            runtime = observed_runtime
            signature = edgeg4.signature((directory / 'expanded.fcl').read_text(), case)
            ef.macro_signature((directory / 'preinit.mac').read_text(), case)
            native = ef.native_signature(directory, case)
            census, arrays, _ = foilreport.audit(directory)
            if census['primaries'] != 5000 or census['other']:
                raise ValueError('complete highest-cycle factorial census required')
            if any(not np.array_equal(arrays[k], refs[mode][k]) for k in arrays):
                raise ValueError('fresh ROOT reduction differs from sealed reference')
            tables = g4tables.compare(directory, package, 'G4_Al', require_audited_export=True)
            if case['bins'] == 168 and not tables['all_arrays_equal']:
                raise ValueError('dense tables differ from completed fast-response package')
            material_tables, couple, paths = pt.load_target(directory / 'tables', material='G4_Al')
            grid = np.geomspace(.1, 3., 1001)
            dense_directory = reused / (cell + '_full_refined')
            dense_tables, dense_couple, _ = pt.load_target(dense_directory / 'tables', material='G4_Al')
            if couple != dense_couple:
                raise ValueError('factorial cuts/material differs from reused dense arm')
            interpolation = {name: dict(max_abs_relative=float(np.max(np.abs(
                pt.interpolate(vector, grid) / pt.interpolate(dense_tables[name], grid) - 1))))
                for name, vector in material_tables.items()}
            report['references'][cell + '_' + mode] = dict(census=census, effective_physics=signature, native=native,
                table_comparison=tables, diagnostic_interpolation_to_dense=interpolation,
                interpolation_scope='SciPy spline diagnostic on 0.1-3 MeV, not G4 interpolation parity',
                mixture=ld.mixture(arrays['exit_MeV']), receipt=_identity(directory / 'receipt.json'))
            samples[cell][mode] = arrays['exit_MeV']
        values = [threshold(samples[cell][mode]) for mode in ORDER]
        report['factorial'][cell] = factorial(values)
        report['pairs'][cell] = {name: dict(contrast(samples[cell][ORDER[a]], samples[cell][ORDER[b]], energy),
                    saved_shape=shape_contrast(samples[cell][ORDER[a]], samples[cell][ORDER[b]]))
                                  for name, (a, b) in PAIRS.items()}
    historical = ld.analyze(recovered, Path(package).parent)
    report['inputs'] += historical['inputs']
    for cell, (energy, _, _) in edgeg4.CELLS.items():
        item = historical['references'][cell + '_full']
        with np.load(item['identity']['path'], allow_pickle=False) as saved:
            old = saved['exit_MeV']
        report['historical'][cell] = dict(qualification='UNVERIFIED original monoenergetic signature; no retrospective certification',
            current_stock_minus_historical=contrast(samples[cell]['stock_step_bins7'], old, energy))
    for directory in studies:
        study, arms, identity = le.read_study(directory)
        report['inputs'].append(identity)
        cell = study['args']['cell']
        energy = edgeg4.CELLS[cell][0]
        for arm, arrays in arms.items():
            y = le.threshold_array(arrays, energy)
            report['fast'][Path(directory).name + '_' + arm] = dict(
                qualification=study['reference_configuration_qualification'],
                references={mode: contrast(y, values, energy) for mode, values in samples[cell].items()})
            report['inputs'].append(study['arms'][arm]['output'])
    for item in report['inputs'] + report['sources']:
        edgeg4.checked(item)
    report['complete'] = True
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, cells=len(report['factorial']), references=len(report['references']))), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--reused-root', required=True)
    parser.add_argument('--study', nargs='+', required=True)
    parser.add_argument('--recovered-root', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    run(args.root, args.reused_root, args.study, args.recovered_root, args.package, args.out)


if __name__ == '__main__':
    main()
