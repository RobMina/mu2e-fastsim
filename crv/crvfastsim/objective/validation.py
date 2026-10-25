"""Explicit-input, one-pass response validation outside the BO leaderboard."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .. import tables
from . import crvresponse as R


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def keys(path, tree):
    return np.unique(tables.read(path, tree, tables.KEY).to_records(index=False))


def inspect_inputs(paths, excluded=()):
    """Certify counts and disjoint generated-event identities, not just distinct filenames."""
    paths = [str(Path(p).resolve()) for p in paths]
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError('inputs must be a nonempty list of unique files')
    blocked = [keys(p, 'gen') for p in excluded]
    seen, rows = [], []
    for path in paths:
        gen, step = keys(path, 'gen'), keys(path, 'steps')
        if not len(step) or np.setdiff1d(step, gen).size:
            raise ValueError(f'empty steps or missing generated-event identities: {path}')
        if any(np.intersect1d(gen, other).size for other in blocked + seen):
            raise ValueError(f'generated-event overlap: {path}')
        if len(np.unique(gen['run'])) != 1 or np.any(gen['event'] >= 2 ** 20):
            raise ValueError(f'input outside R3 event-key encoding: {path}')
        seen.append(gen)
        rows.append(dict(path=path, sha256=digest(path), events_with_steps=len(step), events=len(gen)))
    if len(np.unique(np.concatenate(seen)['run'])) != 1:
        raise ValueError('R3 validation requires one run across the input set')
    return rows


def inventory(paths, excluded, assets, out):
    rows = inspect_inputs(paths, excluded)
    manifest = dict(schema=1, inputs=rows,
                    excluded=[dict(path=str(Path(p).resolve()), sha256=digest(p)) for p in excluded],
                    assets={k: dict(path=str(Path(p).resolve()), sha256=digest(p)) for k, p in assets.items()},
                    events_with_steps=sum(x['events_with_steps'] for x in rows),
                    events=sum(x['events'] for x in rows))
    with open(out, 'x') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify(manifest):
    for row in manifest['inputs'] + list(manifest['assets'].values()):
        if digest(row['path']) != row['sha256']:
            raise ValueError(f"input bytes changed: {row['path']}")


def revision(package):
    import importlib.util
    root = Path(importlib.util.find_spec(package).origin).resolve().parent.parent
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError(f'validation requires clean tracked code: {root}')
    return dict(path=str(root), sha=git('rev-parse', 'HEAD'))


def runtime_assets():
    from .. import geometry
    from ..response.lookup import lookup_dir
    from ..reco.r3 import DEFAULT_COUNTERS
    from ..reco.waveform import WaveformConfig
    root = Path(lookup_dir())
    return {'counters': geometry.find_counters(DEFAULT_COUNTERS),
            'waveform': geometry.find_counters(WaveformConfig().single_pe_file),
            **{name: str(root / name) for name in ('LookupTable_6000_0', 'LookupTable_2370_1',
                                                   'LookupTable_3200_0', 'photonMap.root')}}


def run(manifest_path, out, seed, knobs, jax=False, calibration=False):
    from ..reco import r3
    from ..kit_task import score_result
    manifest = json.loads(Path(manifest_path).read_text())
    verify(manifest)
    code = {p: revision(p) for p in ('crvfastsim', 'mu2efastsim')}
    assets = {k: v['path'] for k, v in manifest['assets'].items()}
    if calibration and not {'background', 'time_shift', 'reference'} <= assets.keys():
        raise ValueError('calibration requires background, time_shift and reference assets')
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=False)
    runtime = {k: dict(path=str(Path(v).resolve()), sha256=digest(v)) for k, v in runtime_assets().items()}
    import platform
    import importlib.metadata
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'awkward', 'uproot')}
    if jax:
        versions.update({p: importlib.metadata.version(p) for p in ('jax', 'jaxlib')})
    provenance = dict(manifest=manifest, manifest_sha256=digest(manifest_path), code=code, seed=seed,
                      runtime_assets=runtime, python=platform.python_version(), versions=versions,
                      knobs=knobs, jax=jax, calibration=calibration)
    (dest / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    result = r3.run([x['path'] for x in manifest['inputs']], assets['calib'], assets['deviation'],
                    target=manifest['events_with_steps'], max_passes=1, seed=seed, jax=jax, v05=True,
                    objective=True, knobs=knobs, background=assets.get('background') if calibration else None,
                    time_shift=assets.get('time_shift') if calibration else None,
                    emit=str(dest / 'r3.json'))
    if result['events_with_steps'] != manifest['events_with_steps'] or \
            result['response_state']['events'] != manifest['events']:
        raise ValueError('finished R3 counts differ from certified one-pass input counts')
    if len(result['passes']) != 1 or result['timers'].get('response_overflow_batches', 0):
        raise ValueError('validation requires one complete pass without overflow')
    verify(manifest)
    if any(digest(v['path']) != v['sha256'] for v in runtime.values()):
        raise ValueError('runtime assets changed during validation')
    if code != {p: revision(p) for p in code}:
        raise ValueError('code changed during validation')
    ref = json.loads(Path(assets['reference']).read_text())
    if calibration:
        score = score_result(result, ref)
    else:
        if ref['events'] != manifest['events']:
            raise ValueError('stock reference exposure differs from input census')
        score = R.objective(result['response_state'], ref)
    R.write_json(str(dest), 'comparison.json', score)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--inputs', nargs='+', required=True)
    inv.add_argument('--exclude', nargs='*', default=[])
    inv.add_argument('--asset', nargs='+', required=True, help='NAME=PATH; calib, deviation, reference, ...')
    inv.add_argument('--out', required=True)
    ev = sub.add_parser('run')
    ev.add_argument('--manifest', required=True)
    ev.add_argument('--out', required=True)
    ev.add_argument('--seed', type=int, required=True)
    ev.add_argument('--knob', nargs='*', default=[])
    ev.add_argument('--jax', action='store_true')
    ev.add_argument('--calibration', action='store_true')
    a = p.parse_args()
    if a.cmd == 'inventory':
        m = inventory(a.inputs, a.exclude, dict(x.split('=', 1) for x in a.asset), a.out)
        print(json.dumps({k: m[k] for k in ('events', 'events_with_steps')}))
    else:
        run(a.manifest, a.out, a.seed, {k: float(v) for k, v in (x.split('=', 1) for x in a.knob)},
            a.jax, a.calibration)


if __name__ == '__main__':
    main()
