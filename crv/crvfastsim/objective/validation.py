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


def _read_json(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, label, positive=False):
    _require(type(value) is int and value >= int(positive), f'invalid {label}: {value!r}')
    return value


def _finite(value, label):
    _require(type(value) in (int, float) and np.isfinite(value), f'nonfinite or invalid {label}')
    return value


def _result_counts(result):
    step = _count(result['events_with_steps'], 'events_with_steps', positive=True)
    empty = _count(result['events_without_steps'], 'events_without_steps')
    target = _count(result['target'], 'target', positive=True)
    _count(result['seed'], 'seed')
    events = _count(result['response_state']['events'], 'response_state events', positive=True)
    _require(step == target, 'incomplete event target')
    _require(events == step + empty, 'response exposure event count mismatch')
    passes = result['passes']
    _require(isinstance(passes, list) and passes, 'missing passes')
    totals = [_count(x['events'], 'pass events', positive=True) for x in passes]
    _require(sum(totals) == step, 'pass event totals mismatch')
    _require(not result['timers'].get('response_overflow_batches', 0), 'response overflow')
    _require(isinstance(result['knobs'], dict), 'invalid knobs')
    return dict(events_with_steps=step, events_without_steps=empty, events=events,
                target=target, pass_events=totals)


def _report(out, report):
    # Never clobber a receipt, including when the caller accidentally names an input.
    with open(out, 'x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def compare_replay(reference, candidate, out):
    """Exact saved-response equivalence; timing, throughput and device are irrelevant."""
    ref, cand = _read_json(reference), _read_json(candidate)
    counts = _result_counts(ref)
    _require(counts == _result_counts(cand), 'replay actual counts/target/pass totals mismatch')
    for key in ('seed', 'knobs'):
        _require(ref[key] == cand[key], f'replay {key} mismatch')
    _require(ref['response_state'] == cand['response_state'], 'replay response_state mismatch')
    return _report(out, dict(schema=1, comparison='compare-replay', passed=True,
                            reference=dict(path=str(reference), sha256=digest(reference)),
                            candidate=dict(path=str(candidate), sha256=digest(candidate)),
                            counts=counts, seed=ref['seed'], knobs=ref['knobs'],
                            response_state_equal=True))


def _validate_point(result, score, provenance):
    from ..kit_task import TERMS
    counts = _result_counts(result)
    exposure = _finite(result['response_state']['exposure_ns'], 'response exposure_ns')
    _require(exposure > 0, 'response exposure_ns must be positive')
    window = result.get('resolved_config', {}).get('timing', {}).get('window')
    if window is not None:
        _require(len(window) == 2 and window[1] > window[0], 'invalid response window')
        _require(exposure == counts['events'] * (window[1] - window[0]),
                 'response time exposure differs from manifest event census and window')
    manifest = provenance['manifest']
    _require(manifest['schema'] == 1, 'unsupported manifest schema')
    rows = manifest['inputs']
    _require(isinstance(rows, list) and rows, 'missing manifest inputs')
    paths = [x['path'] for x in rows]
    _require(len(paths) == len(set(paths)), 'duplicate manifest inputs')
    for row in rows:
        step = _count(row['events_with_steps'], 'manifest input steps', positive=True)
        _require(_count(row['events'], 'manifest input events', positive=True) >= step,
                 'manifest input exposure below step count')
        _require(bool(row['sha256']), 'missing input hash')
    for key in ('events', 'events_with_steps'):
        _count(manifest[key], f'manifest {key}', positive=True)
        _require(sum(x[key] for x in rows) == manifest[key] == counts[key],
                 f'manifest {key} count mismatch')
    _require(len(result['passes']) == 1, 'paired requires one complete pass')
    if 'cache_files' in result:
        _require(result['cache_files'] == paths, 'result inputs differ from manifest')
    if 'cache_events' in result:
        _require(result['cache_events'] == manifest['events_with_steps'], 'cache event count mismatch')
    for key in ('seed', 'knobs'):
        _require(result[key] == provenance[key], f'result/provenance {key} mismatch')
    _require(provenance['calibration'] is True, 'paired requires calibration scoring')
    _require({'calib', 'deviation', 'reference', 'background', 'time_shift'} <= manifest['assets'].keys(),
             'missing calibration/reference assets')
    _require(bool(provenance['manifest_sha256']) and bool(provenance['runtime_assets']) and
             {'crvfastsim', 'mu2efastsim'} <= provenance['code'].keys(), 'incomplete provenance')
    for group in (manifest['assets'], provenance['runtime_assets']):
        _require(all(row.get('sha256') for row in group.values()), 'missing asset hashes')
    _require(all(row.get('sha') for row in provenance['code'].values()), 'missing code revisions')
    required = set(TERMS)
    _require(len(required) == 32 and set(score['terms']) == required, 'required 32 term names mismatch')
    _require(score['n_terms'] == 32 and not score.get('excluded'), 'incomplete 32-term score')
    if score.get('required_terms') is not None:
        _require(len(score['required_terms']) == 32 and set(score['required_terms']) == required,
                 'recorded required term names mismatch')
    _finite(score['objective'], 'objective')
    for key in ('objective_all', 'objective_raw', 'sigma', 'sigma_ref', 'n0'):
        if key in score:
            _finite(score[key], key)
    for name, term in score['terms'].items():
        _finite(term['chi2ndf'], name)
        for key, value in term.items():
            _finite(value, f'{name}.{key}')
    return counts


def paired(directory, out):
    """Compare two completed nominal/best blocks using recorded census/provenance only.

    Disjoint names rely on the inventory's prior event-identity census; no ROOT
    input or simulation/calibration assets are reopened or rescored here.
    """
    root = Path(directory)
    points, receipts = {}, {}
    for name in ('a_nominal', 'a_best', 'b_nominal', 'b_best'):
        paths = {k: root / name / f'{k}.json' for k in ('r3', 'comparison', 'provenance')}
        data = {k: _read_json(v) for k, v in paths.items()}
        counts = _validate_point(data['r3'], data['comparison'], data['provenance'])
        points[name] = data
        receipts[name] = dict(counts=counts, files={k: dict(path=str(v), sha256=digest(v))
                                                  for k, v in paths.items()})
    baseline = points['a_nominal']['provenance']
    for name, data in points.items():
        prov = data['provenance']
        for key in ('code', 'runtime_assets', 'calibration'):
            _require(prov[key] == baseline[key], f'{name}: {key} mismatch')
        _require(prov['manifest']['assets'] == baseline['manifest']['assets'], f'{name}: assets mismatch')
    blocks = {}
    for block in ('a', 'b'):
        nominal, best = (points[f'{block}_{point}'] for point in ('nominal', 'best'))
        pn, pb = nominal['provenance'], best['provenance']
        for key in ('manifest', 'manifest_sha256', 'seed'):
            _require(pn[key] == pb[key], f'block {block}: {key} mismatch')
        for key in ('jax', 'noise', 'v05', 'batch_events', 'background', 'time_shift'):
            _require(nominal['r3'].get(key) == best['r3'].get(key), f'block {block}: {key} mismatch')
        for key in ('python', 'versions', 'jax'):
            _require(pn.get(key) == pb.get(key), f'block {block}: {key} mismatch')
        _require(nominal['r3']['response_state']['exposure_ns'] ==
                 best['r3']['response_state']['exposure_ns'], f'block {block}: time exposure mismatch')
        n, b = nominal['comparison'], best['comparison']
        _require(n.get('n0') == b.get('n0'), f'block {block}: score normalization mismatch')
        def difference(x, y):
            delta = _finite(y - x, 'best-minus-nominal difference')
            return dict(nominal=x, best=y, best_minus_nominal=delta)
        blocks[block] = dict(seed=pn['seed'], counts=receipts[f'{block}_nominal']['counts'],
                             objective=difference(n['objective'], b['objective']),
                             terms={k: difference(n['terms'][k]['chi2ndf'], b['terms'][k]['chi2ndf'])
                                    for k in sorted(n['terms'])})
    pa, pb = (points[f'{block}_nominal']['provenance'] for block in ('a', 'b'))
    _require(pa['seed'] != pb['seed'], 'blocks require distinct seeds')
    names = [{Path(row['path']).name for row in p['manifest']['inputs']} for p in (pa, pb)]
    _require(names[0].isdisjoint(names[1]), 'blocks require disjoint manifest input names')
    return _report(out, dict(schema=1, comparison='paired', passed=True, blocks=blocks, inputs=receipts,
                            caution='Two pairs are a consistency check only, not a significance, precision '
                            'variance or convergence result. Conditional candidate/reference sigmas are not '
                            'paired-difference errors and are not combined. Disjoint manifest input names '
                            'rely on the previously certified event-identity census; it is not rerun here.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--inputs', nargs='+', required=True)
    inv.add_argument('--exclude', nargs='*', default=[])
    inv.add_argument('--asset', nargs='+', required=True, help='NAME=PATH; calib, deviation, reference, ...')
    inv.add_argument('--out', required=True)
    stock = sub.add_parser('stock-count')
    stock.add_argument('--ntuple', required=True)
    stock.add_argument('--accumulator', required=True)
    stock.add_argument('--out', required=True)
    replay = sub.add_parser('compare-replay')
    replay.add_argument('--reference', required=True)
    replay.add_argument('--candidate', required=True)
    replay.add_argument('--out', required=True)
    pair = sub.add_parser('paired')
    pair.add_argument('--directory', required=True)
    pair.add_argument('--out', required=True)
    ev = sub.add_parser('run')
    ev.add_argument('--manifest', required=True)
    ev.add_argument('--out', required=True)
    ev.add_argument('--seed', type=int, required=True)
    ev.add_argument('--knob', nargs='*', default=[])
    ev.add_argument('--jax', action='store_true')
    ev.add_argument('--calibration', action='store_true')
    a = p.parse_args()
    if a.cmd in ('compare-replay', 'paired'):
        try:
            report = (compare_replay(a.reference, a.candidate, a.out) if a.cmd == 'compare-replay'
                      else paired(a.directory, a.out))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            p.error(str(exc))
        print(f"{a.cmd}: PASS; report {a.out}")
        if 'caution' in report:
            print(report['caution'])
    elif a.cmd == 'stock-count':
        import uproot
        with uproot.open(a.ntuple) as f:
            cycles = [k for k in f.keys(recursive=True, cycle=True) if k.split(';')[0] == R.TREE]
            key = max(cycles, key=lambda k: int(k.rsplit(';', 1)[1]))
            count = f[key].num_entries
        expected = json.loads(Path(a.accumulator).read_text())['events']
        if count != expected:
            raise ValueError(f'stock entry count {count} != accumulator events {expected}')
        with open(a.out, 'x') as f:
            json.dump(dict(ntuple=a.ntuple, tree=key, entries=count, accumulator=a.accumulator,
                           accumulator_sha256=digest(a.accumulator)), f, indent=2)
        print(f'{key}: {count} entries; matches saved accumulator')
    elif a.cmd == 'inventory':
        m = inventory(a.inputs, a.exclude, dict(x.split('=', 1) for x in a.asset), a.out)
        print(json.dumps({k: m[k] for k in ('events', 'events_with_steps')}))
    else:
        run(a.manifest, a.out, a.seed, {k: float(v) for k, v in (x.split('=', 1) for x in a.knob)},
            a.jax, a.calibration)


if __name__ == '__main__':
    main()
