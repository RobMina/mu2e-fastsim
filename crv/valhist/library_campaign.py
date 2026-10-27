"""Plan and execute explicit additional library/shard cells using frozen physics."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .library_variants import _read_sealed, _sealed_json, sha256
from .library_contrast import certify


ANALYSIS = Path('/exp/mu2e/app/users/ram2aq/analysis')


def plan(data, out, indices, shards, tag):
    data, out = Path(data).resolve(), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if len(set(indices)) != len(indices) or not indices or any(i not in range(1, 9) for i in indices):
        raise ValueError('unique indices in 1..8 required')
    if len(set(shards)) != len(shards) or len(shards) < 2 or any(s not in range(11, 21) for s in shards):
        raise ValueError('at least two distinct denominator-disjoint shards 11..20 required')
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        raise ValueError('invalid tag')
    source = data / 'valhist/thread_b_library_plan_20261025/plan.json'
    sealed = _read_sealed(source)
    cells = []
    for index in sorted(indices):
        variant = f'bootstrap_{index:04d}'
        if variant not in sealed['variants']:
            raise ValueError('unplanned library')
        metadata = data / f'valhist/thread_b_library_20261025_{variant}_library/library.json'
        for shard in sorted(shards):
            for draw in (1, 2):
                oldseed = 200000 + 100 * index + draw
                old = data / f'valhist/thread_b_library_bootstrap_20261025_{variant}_seed{oldseed}_s11'
                reuse = shard == 11 and old.exists()
                seed = oldseed if reuse else 3000000 + 10000 * index + 100 * shard + draw
                directory = old if reuse else data / f'valhist/{tag}_{variant}_seed{seed}_s{shard:02d}'
                if reuse:
                    evidence = certify(directory)
                    if evidence['manifest']['seed'] != seed:
                        raise ValueError('old seed mismatch')
                elif directory.exists():
                    raise FileExistsError(directory)
                cells.append(dict(index=index, draw=draw, shard=shard, directory=str(directory),
                                  library_metadata=str(metadata), seed=seed, expected_events=50000,
                                  reused=reuse))
    result = dict(schema='valhist.library_campaign.v1', data=str(data), tag=tag, cells=cells,
                  donor_plan=str(source), donor_plan_sha256=sha256(source),
                  frozen_physics=str(ANALYSIS / 'crv-thread-b-dqm-20261025'),
                  uncertainty_code=str(ANALYSIS / 'crv-thread-b-uncertainty-20261025'),
                  core=str(ANALYSIS / 'core-thread-b-dqm-20261025'),
                  area='/exp/mu2e/app/users/ram2aq/crvval',
                  counters='/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00/Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt',
                  limitations=['Fixed shared denominator; these are donor-only replays.',
                               'Same library shared across all shards; distinct conditional seeds per new cell.',
                               'Stock baseSeed and conditions remain fixed.'])
    _sealed_json(out, result)
    print(f'planned {len(cells)} cells; {sum(c["reused"] for c in cells)} retained')
    return result


def preflight(data):
    """Check the established frozen area and physics against accepted campaign hashes."""
    path = Path(data) / 'valhist/thread_b_expand_20261025_s00000011/manifest.json'
    record = json.loads(path.read_text())
    changed = [name for name, expected in record['files'].items() if sha256(name) != expected]
    if changed:
        raise ValueError('accepted frozen physics inputs changed: ' + str(changed))
    print(f'unchanged accepted campaign inputs: {len(record["files"])}', flush=True)


def run(manifest, indices):
    record = _read_sealed(Path(manifest))
    if record.get('schema') != 'valhist.library_campaign.v1':
        raise ValueError('unsupported campaign')
    if sha256(record['donor_plan']) != record['donor_plan_sha256']:
        raise ValueError('donor plan drift')
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('explicit unique library indices required')
    cells = [c for c in record['cells'] if c['index'] in indices]
    if {c['index'] for c in cells} != set(indices):
        raise ValueError('unplanned index')
    for cell in cells:
        directory = Path(cell['directory'])
        if cell['reused']:
            certify(directory)
        elif directory.exists():
            raise FileExistsError(directory)
    env = dict(os.environ, CRVFS_DATA=record['data'], M2FS_CORE=record['core'],
               CRVVAL_AREA=record['area'], PYTHONPATH=record['uncertainty_code'] + ':' + record['core'],
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for index in indices:
        selected = [c for c in cells if c['index'] == index]
        metadata = Path(selected[0]['library_metadata'])
        variant = f'bootstrap_{index:04d}'
        if not metadata.parent.exists():
            subprocess.run(['python3', '-m', 'valhist.library_variants', 'build', record['donor_plan'],
                            '--variant', variant, '--counters', record['counters'], '--outdir', str(metadata.parent)],
                           cwd=record['uncertainty_code'], env=env, check=True)
        library = _read_sealed(metadata)
        root = metadata.parent / 'library.root'
        if (library['provenance']['plan_sha256'] != record['donor_plan_sha256'] or
                library['provenance']['variant'] != variant or
                sha256(root) != library['provenance']['library_sha256']):
            raise ValueError('library provenance mismatch')
        for cell in selected:
            if cell['reused']:
                continue
            shard = cell['shard']
            identity = f'001400_{shard:08d}'
            fastc = Path(record['data']) / f'thread_a_ledger_20261025_c3895aae/extracted20/tables/fastC.{identity}.root'
            gen = f'/pnfs/mu2e/scratch/users/ram2aq/crv_fastsim/g4ref_v1/gen.ram2aq.CosmicCRYExtractedRef.v0.{identity}.art'
            print(f'BEGIN index={index} draw={cell["draw"]} shard={shard} seed={cell["seed"]}', flush=True)
            subprocess.run(['bash', record['frozen_physics'] + '/valhist/run_pilot.sh', cell['directory'],
                            str(fastc), str(root), gen, str(cell['seed'])],
                           cwd=record['frozen_physics'], env=env, check=True)
            certify(cell['directory'])
            print(f'END index={index} draw={cell["draw"]} shard={shard}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--indices', nargs='+', type=int, required=True)
    p.add_argument('--shards', nargs='+', type=int, required=True)
    p.add_argument('--tag', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--data', required=True)
    p = sub.add_parser('run')
    p.add_argument('manifest')
    p.add_argument('--indices', nargs='+', type=int, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    {'plan': plan, 'run': run, 'preflight': preflight}[command](**args)


if __name__ == '__main__':
    main()
