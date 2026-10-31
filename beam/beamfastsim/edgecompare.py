"""Read-only edge-control comparison using new audited G4 and already-finished fast arms.

No sampler execution or tuning. Old stock-G4 comparisons retain their missing
configuration qualification; new refined/lateral references keep original ROOT.
Angular and chord summaries condition on exits and are not causal explanations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from . import edgeg4, edgeangles, foilreport, g4tables, physics_tables
from . import lossdiagnostic as ld, lossexplain as le, slabdiffusion as sd
from .foilcompare import contrast
from .reference_audit import _identity


def run(root, studies, recovered, package, out):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError(out)
    references, records, signatures = {}, [], {}
    out.mkdir(parents=True, exist_ok=False)
    for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
        references[cell] = {}
        for mode in edgeg4.MODES:
            directory = root / (cell + '_' + mode)
            receipt = json.loads((directory / 'receipt.json').read_text())
            if receipt.get('complete') is not True:
                raise ValueError('completed edge reference required')
            for item in receipt['sources'] + receipt['artifacts'] + receipt['tables'] + [receipt['output']]:
                edgeg4.checked(item)
            report, data, _ = foilreport.audit(directory)
            if report['other'] or report['primaries'] != 5000:
                raise ValueError('complete 5000-event bounded reference required')
            signature = edgeg4.signature((directory / 'expanded.fcl').read_text(), receipt['case'])
            tables = g4tables.compare(directory, package, 'G4_Al', require_audited_export=True)
            native, couple, paths = physics_tables.load_target(directory / 'tables', material='G4_Al')
            signatures[cell + '_' + mode] = dict(effective_physics=signature, table_numeric_comparison=tables,
                                                couple=couple, census=report, receipt=_identity(directory / 'receipt.json'))
            if not tables['all_arrays_equal'] or not tables['material_equal']:
                raise ValueError('new edge tables differ numerically from completed fast response package')
            records += [_identity(directory / 'receipt.json')] + [_identity(p) for p in paths]
            with uproot.open(directory / 'foil.root', array_cache=None) as file:
                steps, _ = foilreport.highest(file, 'readback/ntstepper', arrays=False)
                angles = edgeangles.first_exit(steps, data)
            np.savez_compressed(out / (cell + '_' + mode + '_angles.npz'), **angles)
            references[cell][mode] = data
            signatures[cell + '_' + mode]['angles'] = edgeangles.summary(data['exit_MeV'], data['exit'], angles['direction'], cosine)
            signatures[cell + '_' + mode]['mixture'] = ld.mixture(data['exit_MeV'])
    historical = ld.analyze(recovered, Path(package).parent)
    result = dict(complete=False, scope=__doc__, physics_closure=False, g4=signatures, fast={}, g4_controls={},
                  historical_configuration='UNVERIFIED: original stock monoenergetic bundles absent',
                  inputs=records + historical['inputs'], sources=[_identity(p) for p in
                    (Path(__file__), Path(edgeangles.__file__), Path(edgeg4.__file__))])
    for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
        refs = references[cell]
        a, b = refs['full_refined_no_lateral'], refs['full_refined']
        result['g4_controls'][cell + '_no_lateral_minus_refined'] = contrast(a['exit_MeV'], b['exit_MeV'], energy)
        tag = cell + '_full'
        item = historical['references'][tag]
        with np.load(item['identity']['path'], allow_pickle=False) as saved:
            stock = saved['exit_MeV']
        result['g4_controls'][cell + '_refined_minus_historical_stock'] = dict(
            distribution=contrast(b['exit_MeV'], stock, energy), reference_configuration='UNVERIFIED historical stock')
    for directory in studies:
        report, arms, identity = le.read_study(directory)
        result['inputs'].append(identity)
        cell = report['args']['cell']
        energy, thickness, cosine = edgeg4.CELLS[cell]
        for arm, arrays in arms.items():
            y = le.threshold_array(arrays, energy)
            entry = dict(seed=report['args']['seed'], n=len(y),
                         original_reference_qualification=report['reference_configuration_qualification'],
                         endpoint_scope='Threshold transform only; positive floor is not physical stop', references={})
            exits = np.isin(np.asarray(arrays['status']).ravel(), [sd.TRANSMITTED, sd.REFLECTED])
            entry['angles'] = edgeangles.summary(np.asarray(arrays['raw_energy']).ravel(), exits,
                                                 arrays['direction'], cosine)
            for mode, ref in references[cell].items():
                entry['references'][mode] = dict(distribution=contrast(y, ref['exit_MeV'], energy),
                    variance_explanation=le.explain(y, np.where(ref['exit_MeV'] < .1, 0., ref['exit_MeV'])))
            result['fast'][Path(directory).name + '_' + arm] = entry
            result['inputs'].append(report['arms'][arm]['output'])
    for item in result['inputs'] + result['sources']:
        edgeg4.checked(item)
    result['outputs'] = [_identity(p) for p in sorted(out.glob('*.npz'))]
    result['complete'] = True
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, fast_arms=len(result['fast']), g4_arms=len(result['g4']))), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--study', nargs='+', required=True)
    parser.add_argument('--recovered-root', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    run(args.root, args.study, args.recovered_root, args.package, args.out)


if __name__ == '__main__':
    main()
