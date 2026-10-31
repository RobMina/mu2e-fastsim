"""Minimal synthetic oneAPI H32/K8 coincidence compiler reproducer.

No experiment geometry, conditions, input data or core package is needed.
Run with ``JAX_PLATFORMS=oneapi python3 -m
crvfastsim.reco.pvc_finddense_repro``; a compiler abort is expected on affected
plugin/driver combinations. Use ``--hits 16`` for a control shape.
"""
import argparse
import json

import numpy as np


def main():
    import jax
    from . import coinc as C, coinc_jax as J
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hits', type=int, default=32)
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--groups', type=int, default=1)
    a = ap.parse_args()
    h = a.hits
    valid = np.ones((1, h), bool)
    index = np.arange(h)[None, :]
    side = index % 2
    layer = (index // 2) % 4
    bar = index // 2
    first = valid.copy()
    x = index.astype(float) * 20
    y = layer * 20.
    t = np.full((1, h), 100.)
    pe = np.full((1, h), 10.)
    pos = np.stack((x, y, np.zeros_like(x)), -1)
    cu = np.array([float(getattr(C.EXTRACTED_V04['EX'], k)) for k in J.CUT_COLS])
    cuts = np.broadcast_to(cu, (1, h, len(cu)))
    tp = np.array([20., -1400., 1400., 1., 140., 20., 34.5, 1.])
    print(json.dumps(dict(jax=jax.__version__, devices=[str(d) for d in jax.devices()], hits=h, slots=a.slots)), flush=True)
    xs = [valid, index, side, bar, side, first, layer, bar, x, y, t, pe, pos, cuts]
    xs = [np.repeat(v, a.groups, axis=0) for v in xs]
    out = J.find_dense(*xs, tp, cu, ld=2, ns=(3,), uniform=True, K=a.slots)
    jax.block_until_ready(out)
    print('Compiled and executed successfully', flush=True)


if __name__ == '__main__':
    main()
