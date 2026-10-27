"""Small-ensemble donor-file bootstrap variance, separating conditional draw noise."""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from .library_contrast import certify
from .pilot import digest


LIMITATIONS = [
    'Diagnostic variance estimate from a small donor-file bootstrap ensemble, not calibrated confidence coverage.',
    'Conditional on one validation shard, fixed gen denominator, transport, stock baseSeed and conditions.',
    'Independent tier1b/fast-response seeds are used within each library; stock random streams remain fixed.',
    'Between-library variance includes donor composition and refitted quantile-bin edges.',
    'A negative noise-subtracted component is retained; it is not evidence of negative physical variance.',
    'No extrapolation from this shard to the full campaign; no DQM pass or significance claim.',
]


def components(values):
    """Balanced nested random-effects method of moments over library and draw axes."""
    values = np.asarray(values, float)
    if values.ndim < 2 or min(values.shape[:2]) < 2 or not np.isfinite(values).all():
        raise ValueError('need finite (>=2 libraries, >=2 draws, ...) values')
    libraries, draws = values.shape[:2]
    means = values.mean(axis=1)
    within = values.var(axis=1, ddof=1).mean(axis=0)
    between_means = means.var(axis=0, ddof=1)
    library = between_means - within / draws
    return dict(libraries=libraries, draws=draws, mean=values.mean(axis=(0, 1)).tolist(),
                library_means=means.tolist(), conditional_draw_variance=within.tolist(),
                library_mean_variance=between_means.tolist(),
                library_variance_noise_subtracted=library.tolist(),
                library_sd_nonnegative=np.sqrt(np.maximum(library, 0)).tolist(),
                negative_component=(library < 0).tolist())


def build_identity(info):
    provenance = info['provenance']
    return {k: provenance[k] for k in ('plan_sha256', 'build_code_sha256', 'counters')} | {'gen': info['gen']}


def replay_identity(manifest):
    from .campaign import logical_physics
    return logical_physics(manifest)


def run(data, out, indices):
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    if len(indices) < 2 or len(set(indices)) != len(indices) or any(i not in range(1, 9) for i in indices):
        raise ValueError('need at least two unique planned replica indices in 1..8')
    provenance, vectors, edges = [], {a: {} for a in ('a1', 'e2e')}, {}
    transport_hash = None
    common_build = common_replay = None
    for index in sorted(indices):
        variant = f'bootstrap_{index:04d}'
        build = data / 'valhist' / f'thread_b_library_20261025_{variant}_library/library.json'
        if digest(build) != build.with_suffix('.json.sha256').read_text().strip():
            raise ValueError('bootstrap build metadata seal mismatch')
        info = json.loads(build.read_text())
        sealed_plan = build.parent / 'plan.json'
        if digest(sealed_plan) != info['provenance']['plan_sha256']:
            raise ValueError('bootstrap plan seal mismatch')
        if info['provenance']['variant'] != variant or info['generated_scaled_to'] != 10000000:
            raise ValueError('wrong bootstrap library metadata')
        identity = build_identity(info)
        if common_build is not None and identity != common_build:
            raise ValueError('bootstrap builds use different donor plan, denominator, code or geometry')
        common_build = identity
        local = {a: {} for a in vectors}
        for draw in (1, 2):
            seed = 200000 + 100 * index + draw
            directory = data / 'valhist' / f'thread_b_library_bootstrap_20261025_{variant}_seed{seed}_s11'
            evidence = certify(directory)
            manifest = evidence['manifest']
            if manifest['seed'] != seed or manifest['files'][manifest['library']] != info['provenance']['library_sha256']:
                raise ValueError('bootstrap seed or library hash mismatch')
            physics = replay_identity(manifest)
            if common_replay is not None and physics != common_replay:
                raise ValueError('bootstrap replays use different conditions or physics inputs')
            common_replay = physics
            sha = manifest['files'][manifest['fastc']]
            if transport_hash is not None and transport_hash != sha:
                raise ValueError('retained transport differs across bootstrap arms')
            transport_hash = sha
            provenance.append(dict(variant=variant, draw=draw, seed=seed, evidence=evidence,
                                   library_metadata_sha256=digest(build)))
            for arm in vectors:
                with uproot.open(directory / arm / 'valhist.root') as root:
                    events = root['crvdump/events'].arrays(['run', 'subrun', 'event', 'ncluster'], library='np')
                    if len(events['event']) != 50000 or not np.all(events['run'] == 1400) or not np.all(events['subrun'] == 11):
                        raise ValueError('wrong validation shard population')
                    local[arm].setdefault('cluster_rate', []).append(float(events['ncluster'].sum()) / 50000)
                    for name in ('PEc', 'NPc', 'X_extracted', 'Y_extracted', 'Z_extracted'):
                        hist = root['crvRecoDQM/' + name]
                        booking = (hist.classname, hist.axis().edges().tolist())
                        if name in edges and edges[name] != booking:
                            raise ValueError('histogram booking differs across bootstrap draws')
                        edges[name] = booking
                        counts = hist.values(flow=True).astype(float)
                        if not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
                            raise ValueError('invalid bootstrap shape')
                        local[arm].setdefault(name + '_inclusive_fraction', []).append(counts / counts.sum())
                if digest(directory / arm / 'valhist.root') != evidence['completion']['arms'][arm]['sha256']:
                    raise ValueError('bootstrap input changed while reading')
        for arm, metrics in local.items():
            for key, draws in metrics.items():
                vectors[arm].setdefault(key, []).append(draws)
    results = {arm: {name: components(values) for name, values in metrics.items()} for arm, metrics in vectors.items()}
    results['e2e_minus_a1'] = {name: components(np.asarray(vectors['e2e'][name]) - np.asarray(values))
                               for name, values in vectors['a1'].items()}
    report = dict(scope='Conditional donor-file bootstrap variance diagnostic', indices=indices,
                  provenance=provenance, histogram_bookings=edges, components=results, limitations=LIMITATIONS)
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for arm, metrics in results.items():
        print(arm, json.dumps(metrics['cluster_rate']), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--indices', nargs='+', type=int, required=True)
    a = p.parse_args()
    run(a.data, a.out, a.indices)


if __name__ == '__main__':
    main()
