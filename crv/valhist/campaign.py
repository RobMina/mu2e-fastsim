"""Certify explicit Thread B shard outputs and prepare read-only paired analyses."""
import argparse
import json
import posixpath
from pathlib import Path

import numpy as np

from .paired_rates import paired_ratio, read
from .pilot import digest


COUNT_KEYS = tuple('crvRecoDQM/' + name for name in ('NPc', 'PEc', 'X_extracted', 'Y_extracted', 'Z_extracted'))


PAIR_NAMES = ('a1_a0g', 'e2e_a0g', 'e2e_a1', 'floor')


def _absolute(path):
    """Normalize recorded remote paths without resolving them on the analysis host."""
    if not isinstance(path, str) or not path.startswith('/'):
        raise ValueError(f'expected absolute provenance path: {path!r}')
    return posixpath.normpath(path)


def logical_physics(manifest):
    """Return a flat logical-name -> hash/identity mapping (library excluded).

    Source paths are relative to recorded crv/core/area roots, never this host's
    checkout. Helper scripts below crv/valhist and the per-shard fastC input are
    omitted. CRV/core git heads/status are provenance, not simulation identity:
    the complete recorded simulation-source inventory is compared instead.
    Frozen area/backing paths and DQM HEAD remain mandatory identities.
    This helper permits the legacy inventory; campaign comparison alone grants
    shard 11 its explicit, narrowly classified missing-hash exception.
    """
    try:
        roots = {name: _absolute(manifest[name]['path']) for name in ('crv', 'core')}
        roots['area'] = _absolute(manifest['area'])
        result = {'identity:area': roots['area'], 'identity:backing': _absolute(manifest['backing']),
                  'identity:dqm_head': manifest['dqm']['head']}
        if not result['identity:dqm_head']:
            raise ValueError('missing DQM head')
        if _absolute(manifest['dqm']['path']) != roots['area'] + '/DQM':
            raise ValueError('DQM checkout is outside frozen area')
        files = manifest['files']
        excluded = {_absolute(manifest[k]) for k in ('fastc', 'library')}
        if any(manifest[k] not in files for k in ('fastc', 'library')):
            raise ValueError('missing transport/library hash')
        for asset in manifest.get('asset_paths', []):
            if asset not in files:
                raise ValueError(f'missing declared physics asset hash: {asset}')
        if manifest.get('asset_paths_missing'):
            raise ValueError('missing declared physics assets')
        for filename, sha in files.items():
            path = _absolute(filename)
            if not isinstance(sha, str) or not sha:
                raise ValueError(f'missing physics hash: {filename}')
            if path in excluded:
                continue
            if path.startswith(roots['crv'] + '/valhist/'):
                continue
            # Longest prefix wins if the core and detector checkouts are nested.
            key = 'external/' + path.lstrip('/')
            for name, root in sorted(roots.items(), key=lambda kv: len(kv[1]), reverse=True):
                if path.startswith(root + '/'):
                    key = name + '/' + path[len(root) + 1:]
                    break
            if key in result:
                raise ValueError(f'duplicate logical physics input: {key}')
            result[key] = sha
        if not any(k.startswith('crv/crvfastsim/') for k in result):
            raise ValueError('missing detector physics source inventory')
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError(f'incomplete physics manifest: {exc}') from exc


def _legacy_hash(key):
    """Only the original pilot's known omissions, never all external inputs."""
    return (key.startswith('core/mu2efastsim/') or
            '/CRVConditions/' in key or key.rsplit('/', 1)[-1].startswith('LookupTable_') or
            key.rsplit('/', 1)[-1] in ('photonMap.root', 'singlePEWaveform_v3.txt'))


def _physics_campaign(manifests):
    if set(manifests) != set(range(1, 21)):
        raise ValueError('require exactly shards 1..20')
    logical = {s: logical_physics(m) for s, m in manifests.items()}
    baseline = logical[1]
    required = {
        'core source': lambda k: k.startswith('core/mu2efastsim/'),
        'FHiCL': lambda k: k.startswith('area/CrvFastSim/') and k.endswith('.fcl'),
        'art source': lambda k: k.startswith('area/CrvFastSim/src/'),
        'CrvFastSim library': lambda k: k.startswith('area/build/') and '/CrvFastSim/lib/' in k and k.endswith('.so'),
        'DQM library': lambda k: k.startswith('area/build/') and '/DQM/lib/' in k and k.endswith('.so'),
        'DQM binning': lambda k: k == 'area/backing/Offline/CRVDQM/data/CRVDQM_binning_v3_extracted.txt',
        'calibration': lambda k: k.endswith('/crvcalib_sim_best_v1_5_run1400.csv'),
        'photon conditions': lambda k: k.endswith('/crvphoton_sim_best_v1_5_run1400.csv'),
        'LUT': lambda k: k.rsplit('/', 1)[-1].startswith('LookupTable_'),
        'photon map': lambda k: k.endswith('/photonMap.root'),
        'single PE template': lambda k: k.endswith('/singlePEWaveform_v3.txt'),
    }
    for label, predicate in required.items():
        if not any(predicate(k) for k in baseline):
            raise ValueError(f'missing required physics inventory: {label}')
    exceptions = []
    library = manifests[1]['files'][manifests[1]['library']]
    for shard, current in logical.items():
        manifest = manifests[shard]
        if manifest['seed'] != shard:
            raise ValueError(f'wrong nominal seed for shard {shard}')
        if manifest['files'][manifest['library']] != library:
            raise ValueError('nominal campaign uses different secondary libraries')
        missing, extra = baseline.keys() - current.keys(), current.keys() - baseline.keys()
        if extra or (missing and (shard != 11 or not all(_legacy_hash(k) for k in missing))):
            raise ValueError(f'physics inventory differs for shard {shard}: missing={sorted(missing)}, extra={sorted(extra)}')
        changed = [k for k in current.keys() & baseline.keys() if current[k] != baseline[k]]
        if changed:
            raise ValueError(f'shared physics input/identity changed for shard {shard}: {sorted(changed)}')
        if missing:
            exceptions.append(dict(shard=11, missing_hashes=sorted(missing),
                                   reason='Original pilot pre-run core/LUT/template hashes were not recorded; not retroactively certified.'))
    return dict(logical_inputs=baseline, library_sha256=library, exceptions=exceptions)


def _certify_outputs(directory, shard, manifest, completion):
    if (not completion.get('completed') or completion.get('changed_inputs') or
            completion.get('expected_generated_events') != 50000 or
            manifest['generated'].get('unique_events') != 50000):
        raise ValueError(f'invalid completion/exposure shard {shard}')
    keys, values, hashes = [], {}, {}
    for arm in ('a1', 'e2e'):
        path = directory / arm / 'valhist.root'
        saved = completion['arms'][arm]
        sha = digest(path)
        if not saved['completed'] or sha != saved['sha256'] or _absolute(saved['path']) != str(path):
            raise ValueError(f'output hash/path mismatch or incomplete {shard}/{arm}')
        k, v, meta = read(path)
        expected = manifest['generated']['event_keys_sha256']
        evidence = saved['trees']['events']
        if (len(k) != 50000 or not np.all(k[:, :2] == [1400, shard]) or
                meta['rows'] != 50000 or meta['unique_events'] != 50000 or
                evidence['rows'] != 50000 or evidence['unique_events'] != 50000):
            raise ValueError(f'wrong persisted event census {shard}/{arm}')
        if meta['event_keys_sha256'] != expected or evidence['event_keys_sha256'] != expected:
            raise ValueError(f'event identity differs from transport/completion {shard}/{arm}')
        if digest(path) != sha:
            raise ValueError(f'output changed while reading {shard}/{arm}')
        keys.append(k)
        values[arm], hashes[arm] = v, sha
    if not np.array_equal(*keys):
        raise ValueError(f'arm event identities differ: {shard}')
    return keys[0], values, hashes


def _pair_rows(data, directory, shard):
    identity = f'001400_{shard:08d}'
    base = Path(data) / 'valhist/g4ref_v1'
    a1, e2e = (str(directory / arm / 'valhist.root') for arm in ('a1', 'e2e'))
    historical = {arm: str(base / arm / f'valhist.{identity}.root') for arm in ('a0g', 'a0', 'a0p')}
    return {name: dict(shard_id=identity, ref=ref, cand=cand, expected_events=50000)
            for name, ref, cand in [('a1_a0g', historical['a0g'], a1),
                                    ('e2e_a0g', historical['a0g'], e2e), ('e2e_a1', a1, e2e),
                                    ('floor', historical['a0'], historical['a0p'])]}


def shard_paths(data, tag):
    data = Path(data).resolve()
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('tag must contain only letters, digits, underscore and hyphen')
    matched_pilot = data / 'valhist' / f'{tag}_s00000011'
    return [(s, data / 'valhist' / ('thread_b_dqm_20261025_s11'
                                  if s == 11 and not matched_pilot.exists() else f'{tag}_s{s:08d}'))
            for s in range(1, 21)]


def inventory(data, tag):
    rows = []
    for shard, directory in shard_paths(data, tag):
        complete = directory / 'completion.json'
        row = dict(shard=shard, directory=str(directory), completed=False)
        if complete.is_file():
            result = json.loads(complete.read_text())
            row.update(completed=bool(result.get('completed')), completion=result)
        rows.append(row)
    return rows


def prepare(data, tag, out):
    data, out = Path(data).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    rows = inventory(data, tag)
    if not all(r['completed'] for r in rows):
        raise ValueError('incomplete shards: ' + str([r['shard'] for r in rows if not r['completed']]))
    pairs = {key: [] for key in PAIR_NAMES}
    keys, values = [], {'a1': [], 'e2e': []}
    provenance, manifests, input_hashes = [], {}, {}
    for row in rows:
        shard, directory = row['shard'], Path(row['directory'])
        manifest_path, completion_path = directory / 'manifest.json', directory / 'completion.json'
        manifest_sha, completion_sha = digest(manifest_path), digest(completion_path)
        manifest = json.loads(manifest_path.read_text())
        completion = json.loads(completion_path.read_text())
        k, v, hashes = _certify_outputs(directory, shard, manifest, completion)
        keys.append(k)
        for arm in values:
            values[arm].append(v[arm])
            input_hashes[str(directory / arm / 'valhist.root')] = hashes[arm]
        if shard in manifests:
            raise ValueError('duplicate campaign shard')
        manifests[shard] = manifest
        for name, pair in _pair_rows(data, directory, shard).items():
            for role in ('ref', 'cand'):
                path = pair[role]
                if path not in input_hashes:
                    input_hashes[path] = digest(path)
            pairs[name].append(pair)
        if digest(manifest_path) != manifest_sha or digest(completion_path) != completion_sha:
            raise ValueError('campaign provenance changed during preparation')
        provenance.append(dict(shard=shard, manifest=str(manifest_path),
                               manifest_sha256=manifest_sha, completion_sha256=completion_sha,
                               source_crv=manifest['crv'], source_core=manifest['core'],
                               library=manifest['library'], library_sha256=manifest['files'][manifest['library']],
                               seed=manifest['seed'], generated=manifest['generated']))
    physics = _physics_campaign(manifests)
    allkeys = np.concatenate(keys)
    if len(allkeys) != 1000000 or len(np.unique(allkeys, axis=0)) != len(allkeys):
        raise ValueError('wrong campaign exposure or events reused across shards')
    metrics = set().union(*(v.keys() for arms in values.values() for v in arms))
    combined = {arm: {m: np.concatenate([v.get(m, np.zeros(50000)) for v in shards]) for m in metrics}
                for arm, shards in values.items()}
    rates = {m: paired_ratio(combined['a1'][m], combined['e2e'][m]) for m in sorted(metrics)}
    result = dict(schema_version=2, data=str(data), tag=tag, shards=20, generated_events_per_arm=len(allkeys),
                  provenance=provenance, paired_rates=rates, physics=physics, input_sha256=input_hashes,
                  pair_manifests={}, limitations=[
                      'Rates are conditional on the shared secondary library and conditions.',
                      'Historical A0g/A0/A0p histograms do not establish complete persisted event pairing.',
                      'Any reused legacy shard-11 missing hashes are listed explicitly in physics.exceptions.'])
    out.mkdir(parents=True)
    for name, contents in pairs.items():
        path = out / f'pairs_{name}.json'
        path.write_text(json.dumps(contents, indent=2) + '\n')
        result['pair_manifests'][name] = dict(file=path.name, sha256=digest(path))
    (out / 'campaign.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def validate_report_inputs(out):
    """Read-only certification; return (campaign_record, {pair_name: verified_rows}).

    Reject legacy unbound report directories: prepare a NEW report directory to
    bind original historical files, never refresh or rewrite those references.
    """
    try:
        return _validate_report_inputs(Path(out))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'incomplete campaign report binding: {exc}') from exc


def _validate_report_inputs(out):
    record = json.loads((out / 'campaign.json').read_text())
    if (record.get('schema_version') != 2 or record.get('shards') != 20 or
            record.get('generated_events_per_arm') != 1000000):
        raise ValueError('require full certified campaign with bound report inputs')
    provenance = record['provenance']
    if len(provenance) != 20 or {p['shard'] for p in provenance} != set(range(1, 21)):
        raise ValueError('require exactly 20 unique provenance shards')
    directories = dict(shard_paths(record['data'], record['tag']))
    pairs, expected_pairs, manifests = {}, {n: [] for n in PAIR_NAMES}, {}
    if set(record['pair_manifests']) != set(PAIR_NAMES):
        raise ValueError('missing pair manifest inventory')
    # Validate pair bytes/structure before any expensive ROOT reads.
    for name in PAIR_NAMES:
        path = out / f'pairs_{name}.json'
        binding = record['pair_manifests'][name]
        if binding['file'] != path.name or digest(path) != binding['sha256']:
            raise ValueError(f'pair manifest changed: {name}')
        rows = json.loads(path.read_text())
        if (not isinstance(rows, list) or len(rows) != 20 or
                {r['shard_id'] for r in rows} != {f'001400_{s:08d}' for s in range(1, 21)}):
            raise ValueError(f'pair manifest requires exactly 20 unique shards: {name}')
        pairs[name] = rows
    used = set()
    for p in sorted(provenance, key=lambda p: p['shard']):
        shard = p['shard']
        directory = directories[shard]
        path = directory / 'manifest.json'
        complete = directory / 'completion.json'
        if p['manifest'] != str(path) or digest(path) != p['manifest_sha256']:
            raise ValueError('campaign provenance changed')
        if digest(complete) != p['completion_sha256']:
            raise ValueError('campaign completion changed')
        manifest, completion = json.loads(path.read_text()), json.loads(complete.read_text())
        manifests[shard] = manifest
        _, _, hashes = _certify_outputs(directory, shard, manifest, completion)
        for arm, sha in hashes.items():
            if record['input_sha256'].get(str(directory / arm / 'valhist.root')) != sha:
                raise ValueError('candidate hash differs from preparation')
        for name, pair in _pair_rows(record['data'], directory, shard).items():
            for role in ('ref', 'cand'):
                filename = pair[role]
                sha = record['input_sha256'].get(filename)
                if sha is None:
                    raise ValueError('missing report input hash')
                if filename not in used and digest(filename) != sha:
                    raise ValueError(f'report input hash changed: {filename}')
                used.add(filename)
            expected_pairs[name].append(pair)
    if used != set(record['input_sha256']):
        raise ValueError('report input inventory differs')
    for name in PAIR_NAMES:
        if sorted(pairs[name], key=lambda r: r['shard_id']) != expected_pairs[name]:
            raise ValueError(f'pair paths/hashes/exposure differ from certified inputs: {name}')
    if _physics_campaign(manifests) != record['physics']:
        raise ValueError('campaign physics certification changed')
    return record, pairs


def analyze(out, replicas=10000):
    from .paired_shapes import run
    out = Path(out)
    validate_report_inputs(out)
    for name in PAIR_NAMES:
        run(out / f'pairs_{name}.json', out / f'shapes_{name}.json',
            select=r'^crvRecoDQM/(NPc|PEc|[XYZ]_extracted)$', count_keys=COUNT_KEYS,
            replicas=replicas, seed=20261025)
        print(f'completed {name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('status', 'prepare'):
        p = sub.add_parser(name)
        p.add_argument('--data', required=True)
        p.add_argument('--tag', required=True)
        if name == 'prepare':
            p.add_argument('--out', required=True)
    p = sub.add_parser('analyze')
    p.add_argument('out')
    p.add_argument('--replicas', type=int, default=10000)
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'status':
        rows = inventory(**args)
        print(json.dumps(dict(completed=[r['shard'] for r in rows if r['completed']],
                              incomplete=[r['shard'] for r in rows if not r['completed']]), indent=2))
    elif command == 'prepare':
        result = prepare(**args)
        print(json.dumps({k: result[k] for k in ('shards', 'generated_events_per_arm', 'paired_rates')}, indent=2))
    else:
        analyze(**args)


if __name__ == '__main__':
    main()
