"""Save canonical synthetic reco outputs for exact candidate/batching comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    from . import chain_jax, cli, synth
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--events', type=int, default=1000)
    ap.add_argument('--against')
    a = ap.parse_args()
    G = cli._coinc_geom()
    W = synth.chain_events(a.events, G, seed=20261028)
    outs, timers = chain_jax.run(W, G, np.zeros(4 * len(G.sector), np.int64),
                               n_chunks=2, batch=a.batch, keep_pulses=True, with_pos=True)
    arrays = {}
    for k in outs[0]:
        if k == 'pulses':
            for col in outs[0][k]:
                arrays['pulse__' + col] = np.concatenate([o[k][col] for o in outs])
        elif k.startswith('cl_'):
            arrays[k] = np.concatenate([o[k] for o in outs])
    order = np.lexsort((arrays['cl_start'], arrays['cl_type'], arrays['cl_event']))
    arrays = {k: v[order] if k.startswith('cl_') else v for k, v in arrays.items()}
    p = Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / 'arrays.npz', **arrays)
    result = dict(timers=timers, batch=a.batch)
    if a.against:
        from .pvc_profile import _differences
        with np.load(Path(a.against) / 'arrays.npz') as ref:
            result['comparison'] = _differences(arrays, ref)
    (p / 'parity.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
