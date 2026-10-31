"""Prepare and audit bounded released-G4 edge controls with original artifacts retained.

Preparation never runs G4. The external private-cache Offline runner executes the
prepared FCL. Finalization refuses altered inputs, incomplete logs or unresolved
outcomes. Matched refined lateral on/off is not a proof of reference convergence.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from . import foilclosure as fc, foilreport as fr
from .lossarchive import block, scalar
from .reference_audit import _identity

CELLS = {'moved_edge': (2.02, .2, 1.), 'oblique_edge': (2.3, .2, .8)}
MODES = {'full_refined': True, 'full_refined_no_lateral': False}


def checked(item):
    actual = _identity(item['path'])
    if any(actual[k] != item[k] for k in ('bytes', 'sha256')):
        raise ValueError('changed edge input/artifact: ' + item['path'])
    return actual


def prepare(out, cell, mode, events, seed):
    return _prepare(out, cell, mode, events, seed, step_fraction=.005, final_range=.00005,
                    bins=168, lateral=MODES[mode])


def _prepare(out, cell, mode, events, seed, *, step_fraction, final_range, bins, lateral):
    energy, thickness, cosine = CELLS[cell]
    fc.validate_case(energy, thickness, cosine, 13, events, seed)
    if events > 5000:
        raise ValueError('bounded edge control permits at most 5000 events')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    geom, gun = fc.render(energy, thickness, cosine, 13)
    macro, config = fc.control_text(events, seed, export_tables=True, step_fraction=step_fraction,
                                   final_range=final_range, bins=bins, export_bins=bins,
                                   lateral_displacement=lateral)
    (out / 'tables').mkdir()
    template = Path(fc.__file__).resolve().parents[1] / 'fcl/foilclosure.fcl'
    for name, text in {'foil_geom.txt': geom, 'foil_gun.txt': gun, 'preinit.mac': macro,
                       'base.fcl': template.read_text(), 'job.fcl': config}.items():
        (out / name).write_text(text)
    case = dict(energy_MeV=energy, thickness_mm=thickness, cosine=cosine, pdg=13,
                events=events, seed=seed, energy_max_MeV=energy, cosine_min=cosine,
                step_fraction=step_fraction, final_range=final_range, bins=bins, no_msc=False,
                export_tables=True, lateral_displacement=lateral)
    source = Path(__file__).resolve().parents[1]
    files = sorted((source / 'beamfastsim').glob('*.py'))
    files += [template, source / 'art/edge_offline_uva.sh']
    mount_runner = source.parent / 'container/offline/offline_uva.sh'
    if mount_runner.is_file():
        files.append(mount_runner)
    receipt = dict(complete=False, release=fc.RELEASE, case=case, cell=cell, mode=mode,
                   sources=[_identity(p) for p in files],
                   prepared=[_identity(out / name) for name in ('base.fcl', 'job.fcl', 'foil_geom.txt',
                                                               'foil_gun.txt', 'preinit.mac')],
                   scope=__doc__, producer='edgeg4 external released Offline private-cache runner')
    (out / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def signature(text, case):
    physics = block(block(block(text, 'physics'), 'producers'), 'g4run')
    settings = block(physics, 'physics')
    expected = dict(physicsListName='ShieldingM', minRangeCut=.01,
                    protonProductionCut=0., setMuHadLateralDisplacement=case['lateral_displacement'],
                    useDensityEffectInIonizationLossCalc=False)
    actual = {k: scalar(settings, k) for k in expected}
    if actual != expected:
        raise ValueError('effective G4 physics signature mismatch: ' + repr(actual))
    if scalar(block(text, 'source'), 'maxEvents') != case['events']:
        raise ValueError('effective source count mismatch')
    return actual


def finalize(out):
    out = Path(out).resolve()
    receipt = json.loads((out / 'prepared.json').read_text())
    for item in receipt['sources'] + receipt['prepared']:
        checked(item)
    if int((out / 'exit_code.txt').read_text()) != 0:
        raise ValueError('external Offline runner failed')
    text = (out / 'job.log').read_text()
    if fc.macro_errors(text) or 'Art has completed and will exit with status 0' not in text:
        raise ValueError('completed error-free G4 log required')
    case = receipt['case']
    if not re.search(r'TrigReport Events total\s*=\s*' + str(case['events']) + r'\b', text):
        raise ValueError('completed runtime event count mismatch')
    commands = (out / 'preinit.mac').read_text().splitlines()
    if not commands or commands[0] != '/control/verbose 2':
        raise ValueError('required leading G4 verbosity control missing')
    for command in commands[1:]:
        if command not in text:
            raise ValueError('unconfirmed G4 macro command: ' + command)
    effective = signature((out / 'expanded.fcl').read_text(), case)
    runtime = (out / 'runtime.txt').read_text()
    if 'al9-prof-e29-p084' not in runtime or fc.RELEASE not in runtime:
        raise ValueError('released runtime identity mismatch')
    elapsed = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)', (out / 'art.time').read_text())
    if not elapsed:
        raise ValueError('missing art wall-time record')
    parts = [float(x) for x in elapsed[1].split(':')]
    wall = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    names = ('base.fcl', 'job.fcl', 'foil_geom.txt', 'foil_gun.txt', 'preinit.mac', 'expanded.fcl',
             'job.log', 'runtime.txt', 'runtime.sha256', 'launcher.txt', 'launcher.log',
             'art.time', 'total.time', 'cache.txt', 'exit_code.txt', 'prepared.json')
    tables = [_identity(p) for p in sorted((out / 'tables').iterdir()) if p.is_file()]
    if not tables or not any(Path(p['path']).name == 'couple.dat' for p in tables):
        raise ValueError('missing run-specific physics table export')
    receipt.update(complete=True, exit_code=0, wall_s=wall, effective_physics=effective, tables=tables,
                   finalizer=_identity(__file__),
                   artifacts=[_identity(out / n) for n in names], output=_identity(out / 'foil.root'))
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    try:
        report, arrays, _ = fr.audit(out)
        if report['other'] or report['primaries'] != case['events']:
            raise ValueError('unresolved G4 outcomes or incomplete census')
        np.savez_compressed(out / 'scan.npz', **arrays)
        (out / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        run = dict(scan=receipt['cell'], mode=receipt['mode'], pdg=13, thickness_mm=case['thickness_mm'],
                   receipt=_identity(out / 'receipt.json'), scan_npz=_identity(out / 'scan.npz'), census=report)
        manifest = dict(complete=True, runs={out.name: run}, scope=__doc__)
        (out.parent / (out.name + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    except Exception:
        receipt['complete'] = False
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        raise
    for item in receipt['sources'] + receipt['artifacts'] + receipt['tables'] + [receipt['output']]:
        checked(item)
    print(json.dumps(dict(complete=True, cell=receipt['cell'], mode=receipt['mode'], census=report)), flush=True)
    return report


def collect(root):
    """Seal a complete four-control campaign and verify it through the strict reference loader."""
    from . import slabrefs
    root = Path(root).resolve()
    path = root / 'manifest.json'
    if path.exists():
        raise FileExistsError(path)
    runs, seeds = {}, set()
    for cell in CELLS:
        for mode in MODES:
            tag = cell + '_' + mode
            manifest = json.loads((root / (tag + '.manifest.json')).read_text())
            if manifest.get('complete') is not True or set(manifest['runs']) != {tag}:
                raise ValueError('complete single-control manifest required')
            item = manifest['runs'][tag]
            checked(item['receipt'])
            checked(item['scan_npz'])
            receipt = json.loads((root / tag / 'receipt.json').read_text())
            seed = receipt['case']['seed']
            if seed in seeds:
                raise ValueError('independent control seeds required')
            seeds.add(seed)
            runs[tag] = item
    path.write_text(json.dumps(dict(complete=True, runs=runs, scope=__doc__), indent=2) + '\n')
    try:
        for cell, (energy, thickness, cosine) in CELLS.items():
            refs, records = slabrefs.load_references(root, cell, 13, thickness, energy, cosine)
            if set(refs) != set(MODES):
                raise ValueError('both qualified edge-control modes required')
            for item in records:
                checked(item)
    except Exception:
        path.unlink()
        raise
    return _identity(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', required=True)
    p.add_argument('--cell', choices=CELLS, required=True)
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
