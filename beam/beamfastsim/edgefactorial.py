"""Bounded step/bin controls at two edges, reusing the existing refined/168 arm.

Stock step means no StepFunctionMuHad override in the same released runtime,
not certification of a missing historical configuration. Native grids and cuts
are checked before reference finalization. No convergence or model promotion.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import edgeg4, physics_tables as pt, slabrefs
from .reference_audit import _identity

MODES = {'stock_step_bins7': (None, .001, 7),
         'stock_step_bins168': (None, .001, 168),
         'refined_step_bins7': (.005, .00005, 7)}
SEEDS = {'moved_edge': (2026103031, 2026103032, 2026103033),
         'oblique_edge': (2026103041, 2026103042, 2026103043)}


def prepare(out, cell, mode, events, seed):
    fraction, final_range, bins = MODES[mode]
    return edgeg4._prepare(out, cell, mode, events, seed, step_fraction=fraction,
                           final_range=final_range, bins=bins, lateral=True)


def macro_signature(text, case):
    lines = text.splitlines()
    bins = case['bins']
    expected = ['/control/verbose 2', '/run/particle/setStoredInAscii 1',
                '/process/eLoss/CSDARange true', f'/process/em/binsPerDecade {bins}',
                f'/process/em/binsPerDecade {bins}']
    if case['step_fraction'] is not None:
        expected.append('/process/eLoss/StepFunctionMuHad '
                        f"{case['step_fraction']:.17g} {case['final_range']:.17g} mm")
    if lines != expected or not case['lateral_displacement']:
        raise ValueError('factorial macro/settings mismatch')
    return dict(bins_per_decade=bins, step_override=case['step_fraction'] is not None,
                step_scope='explicit refined' if case['step_fraction'] is not None else 'inherited released runtime')


def native_signature(directory, case):
    tables, couple, paths = pt.load_target(Path(directory) / 'tables', material='G4_Al')
    if (couple['density_g_cm3'] != 2.699
            or couple['range_cuts_mm'] != {'gamma': .01, 'e-': .01, 'e+': .01, 'proton': 0.}
            or not np.isclose(couple['energy_cuts_MeV']['e-'], .03410575, rtol=0, atol=1e-12)):
        raise ValueError('factorial native material/cuts mismatch')
    grids = {}
    for name, vector in tables.items():
        energy = vector['energy']
        observed = (len(energy) - 1) / np.log10(energy[-1] / energy[0])
        if not np.isclose(observed, case['bins'], rtol=0, atol=.1):
            raise ValueError('factorial native binning mismatch: ' + name)
        grids[name] = dict(nodes=len(energy), minimum_MeV=float(energy[0]),
                           maximum_MeV=float(energy[-1]), observed_bins_per_decade=float(observed))
    return dict(couple=couple, grids=grids, inputs=[_identity(p) for p in paths])


def finalize(out):
    out = Path(out)
    receipt = json.loads((out / 'prepared.json').read_text())
    case = receipt['case']
    if receipt['mode'] not in MODES or (case['step_fraction'], case['final_range'], case['bins']) != MODES[receipt['mode']]:
        raise ValueError('unknown factorial control settings')
    for item in receipt['sources'] + receipt['prepared']:
        edgeg4.checked(item)
    macro = macro_signature((out / 'preinit.mac').read_text(), case)
    native = native_signature(out, case)
    report = edgeg4.finalize(out)
    audit = dict(complete=True, macro=macro, native=native, census=report,
                 receipt=_identity(out / 'receipt.json'), sources=[_identity(__file__), _identity(pt.__file__)])
    (out / 'factorial_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    return audit


def collect(root):
    root = Path(root).resolve()
    path = root / 'manifest.json'
    if path.exists():
        raise FileExistsError(path)
    runs, seeds = {}, set()
    for cell in edgeg4.CELLS:
        for mode in MODES:
            tag = cell + '_' + mode
            out = root / tag
            manifest = json.loads((root / (tag + '.manifest.json')).read_text())
            if manifest.get('complete') is not True or set(manifest['runs']) != {tag}:
                raise ValueError('complete factorial single-control manifest required')
            run = manifest['runs'][tag]
            for item in (run['receipt'], run['scan_npz']):
                edgeg4.checked(item)
            audit = json.loads((out / 'factorial_audit.json').read_text())
            edgeg4.checked(audit['receipt'])
            for item in audit['sources'] + audit['native']['inputs']:
                edgeg4.checked(item)
            receipt = json.loads((out / 'receipt.json').read_text())
            case = receipt['case']
            if case['events'] != 5000 or case['seed'] in seeds or audit.get('complete') is not True:
                raise ValueError('5000 independent audited factorial primaries required')
            seeds.add(case['seed'])
            macro_signature((out / 'preinit.mac').read_text(), case)
            native_signature(out, case)
            runs[tag] = dict(run, factorial_audit=_identity(out / 'factorial_audit.json'))
    path.write_text(json.dumps(dict(complete=True, runs=runs, scope=__doc__), indent=2) + '\n')
    try:
        for cell, (energy, thickness, cosine) in edgeg4.CELLS.items():
            refs, records = slabrefs.load_references(root, cell, 13, thickness, energy, cosine)
            if set(refs) != set(MODES):
                raise ValueError('all factorial control modes required')
            for item in records:
                edgeg4.checked(item)
    except Exception:
        path.unlink()
        raise
    return _identity(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', required=True)
    p.add_argument('--cell', choices=edgeg4.CELLS, required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--events', type=int, default=5000)
    p.add_argument('--seed', type=int, required=True)
    p = sub.add_parser('finalize')
    p.add_argument('--out', required=True)
    p = sub.add_parser('collect')
    p.add_argument('--root', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args.out, args.cell, args.mode, args.events, args.seed)
    elif args.command == 'finalize':
        finalize(args.out)
    else:
        print(json.dumps(collect(args.root)), flush=True)


if __name__ == '__main__':
    main()
