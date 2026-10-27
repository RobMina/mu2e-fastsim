"""Run explicitly scoped Thread B event/shard diagnostics after full campaign certification."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import campaign, event_shapes, paired_shapes


FEATURES = {
    'pe': [0, 100, 200, 300, 400, 500, 700, 1000, 1500, 2500, 5000],
    'npulses': [0, 4, 8, 12, 16, 24, 32, 48, 64, 100],
    'x': np.linspace(-10000, 2000, 61).tolist(),
    'y': np.linspace(-6000, 6000, 61).tolist(),
    'z': np.linspace(19000, 27000, 41).tolist(),
    'sectortype': np.arange(-0.5, 13, 1).tolist(),
}


def run(out, replicas=999):
    out = Path(out)
    record, verified_pairs = campaign.validate_report_inputs(out)
    specs = verified_pairs['e2e_a1']
    pairs = [{k: row[k] for k in ('ref', 'cand')} for row in specs]
    event_shapes.run(pairs, FEATURES, out / 'event_shapes_e2e_a1.json', replicas=replicas, seed=20261025)
    print('completed event_shapes_e2e_a1', flush=True)
    for name in ('a1_a0g', 'e2e_a0g', 'e2e_a1', 'floor'):
        rows = verified_pairs[name]
        heldout = [r for r in rows if int(r['shard_id'].split('_')[1]) > 10]
        path = out / f'pairs_{name}_heldout.json'
        with path.open('x') as stream:
            json.dump(heldout, stream, indent=2)
        paired_shapes.run(path, out / f'shapes_{name}_heldout.json',
                          select=r'^crvRecoDQM/(NPc|PEc|[XYZ]_extracted)$',
                          count_keys=campaign.COUNT_KEYS, replicas=10000, seed=20261025)
        print(f'completed heldout {name}', flush=True)


def pilot(data, out, replicas=999):
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    pairs = [dict(ref=str(data / 'valhist/thread_b_dqm_20261025_s11/a1/valhist.root'),
                  cand=str(data / 'valhist/thread_b_dqm_20261025_s11/e2e/valhist.root'))]
    event_shapes.run(pairs, FEATURES, out, replicas=replicas, seed=20261025)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    full = sub.add_parser('full')
    full.add_argument('out')
    full.add_argument('--replicas', type=int, default=999)
    one = sub.add_parser('pilot')
    one.add_argument('--data', required=True)
    one.add_argument('--out', required=True)
    one.add_argument('--replicas', type=int, default=999)
    a = vars(p.parse_args())
    command = a.pop('command')
    (run if command == 'full' else pilot)(**a)


if __name__ == '__main__':
    main()
