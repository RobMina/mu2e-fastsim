"""Paired event diagnostics for two disjoint donor-file halves and the frozen nominal library."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import event_shapes
from .paired_rates import paired_ratio, read
from .pilot import digest
from .report_uncertainty import FEATURES


def certify(directory):
    directory = Path(directory)
    record = json.loads((directory / 'completion.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not record['completed'] or record['changed_inputs'] or record['expected_generated_events'] != 50000:
        raise ValueError(f'incomplete or drifted library replay: {directory}')
    for arm in ('a1', 'e2e'):
        if digest(directory / arm / 'valhist.root') != record['arms'][arm]['sha256']:
            raise ValueError(f'output hash changed: {directory}/{arm}')
    return dict(manifest=manifest, completion=record,
                manifest_sha256=digest(directory / 'manifest.json'),
                completion_sha256=digest(directory / 'completion.json'))


def run(data, out, replicas=999):
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    directories = dict(nominal=data / 'valhist/thread_b_expand_20261025_s00000011',
                       half_even=data / 'valhist/thread_b_library_replay_20261025_half_even_s11',
                       half_odd=data / 'valhist/thread_b_library_replay_20261025_half_odd_s11')
    certification = {k: certify(v) for k, v in directories.items()}
    from .campaign import logical_physics
    identities = [logical_physics(v['manifest']) for v in certification.values()]
    if any(v != identities[0] for v in identities[1:]):
        raise ValueError('library contrast requires matching response/reco source and conditions')
    library_metadata = {}
    donor_sets = []
    for name in ('half_even', 'half_odd'):
        manifest = certification[name]['manifest']
        meta_path = Path(manifest['library']).with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        if meta['generated_scaled_to'] != 5000000 or meta['provenance']['variant'] != name:
            raise ValueError('wrong half-library exposure or variant')
        if meta['provenance']['library_sha256'] != manifest['files'][manifest['library']]:
            raise ValueError('half-library metadata hash disagrees with replay')
        donor_sets.append(set(meta['inputs']))
        library_metadata[name] = dict(path=str(meta_path), sha256=digest(meta_path), metadata=meta)
    if donor_sets[0] & donor_sets[1] or any(len(s) != 10 for s in donor_sets):
        raise ValueError('half libraries must contain ten disjoint donor files each')
    keys = {k: v['manifest']['generated']['event_keys_sha256'] for k, v in certification.items()}
    if len(set(keys.values())) != 1:
        raise ValueError('library arms use different generated-event identities')
    transport = {v['manifest']['files'][v['manifest']['fastc']] for v in certification.values()}
    if len(transport) != 1:
        raise ValueError('library arms use different retained transport')
    seeds = {v['manifest']['seed'] for v in certification.values()}
    if len(seeds) != 1:
        raise ValueError('half-library contrast requires declared common seed')
    out.mkdir(parents=True)
    result = dict(scope='One-shard split-library sensitivity, not a calibrated finite-library standard error',
                  certification=certification, library_metadata=library_metadata, comparisons={}, limitations=[
                      'One donor-file split cannot estimate a full-library one-sigma uncertainty.',
                      'Both halves use 5M generated normalization and refit quantile edges.',
                      'Shared generator denominator is fixed; its uncertainty is omitted.',
                      'Common seeds do not give identical random assignments when multiplicities differ.',
                      'Conditional event errors do not estimate the library uncertainty or remove response noise.'])
    for ref, cand in [('half_even', 'half_odd'), ('nominal', 'half_even'), ('nominal', 'half_odd')]:
        for arm in ('a1', 'e2e'):
            name = f'{cand}_vs_{ref}_{arm}'
            pr, pc = (directories[k] / arm / 'valhist.root' for k in (ref, cand))
            kr, vr, _ = read(pr)
            kc, vc, _ = read(pc)
            if not np.array_equal(kr, kc):
                raise ValueError('event identities differ')
            zero = np.zeros(len(kr))
            rates = {key: paired_ratio(vr.get(key, zero), vc.get(key, zero))
                     for key in sorted(vr.keys() | vc.keys())}
            event_shapes.run([dict(ref=str(pr), cand=str(pc))], FEATURES, out / f'{name}.json',
                             replicas=replicas, seed=20261025)
            result['comparisons'][name] = rates
            print(name, json.dumps(rates['ncluster']), flush=True)
    (out / 'library_contrast.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--replicas', type=int, default=999)
    a = p.parse_args()
    run(a.data, a.out, a.replicas)


if __name__ == '__main__':
    main()
