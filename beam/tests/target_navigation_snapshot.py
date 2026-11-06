"""Capture or compare synthetic raw CE/flash states across a geometry-only refactor.

Run as a script before and after an edit, using the same environment. Artifacts
are local regression fixtures, not physics samples or a throughput benchmark.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from beamfastsim import cechain as C
from beamfastsim import cechain_jax as CE
from beamfastsim import flash_navigator_jax as FL
from beamfastsim import geometry as G


def snapshots():
    if not jax.config.x64_enabled:
        raise ValueError('Snapshot comparison requires JAX_ENABLE_X64=1')
    beam = G.BeamGeometry(-3904., np.array([5800., 5822.]), np.full(2, 21.5), np.full(2, 75.),
                          np.full(2, 2.5), 4204.6, 25.4, 235., 525., 300., .5, 6900., 7900.,
                          900., 4280., 8530.)
    geom = replace(C.CEGeometry(beam, 900., 900., 1., z_plane=8000.), ring_t=2., ring_hl=1.)
    x = np.array([[-3864., 0., 5800.], [-3864., 0., 5796.5], [-3864., 0., 5826.],
                  [-3604.5, 0., 7000.], [-3004.5, 0., 7000.], [-3606.5, 0., 6900.5],
                  [-3894., 0., 5700.], [-3864., 0., 5700.]])
    u = np.array([[.2, .1, np.sqrt(.95)], [0., 0., 1.], [0., 0., -1.], [1., 0., 0.],
                  [1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 0., -1.]])
    results = {}
    for dtype in (np.float32, np.float64):
        for name, module, positron in (('ce', CE, False), ('flash_e', FL, False), ('flash_p', FL, True)):
            for field in ((0., 0., 0.), (.05, .02, 1.)):
                label = f'{name}_{np.dtype(dtype).name}_field{int(field[2])}'
                args = (geom, dtype, 1., 30., True, True, True)
                if module is CE:
                    init, run = module._build(*args, None, False, field)
                else:
                    init, run = module._build(*args, False, field, positron=positron, photon_capacity=16)
                n = len(x)
                state = init((), jnp.asarray(x, dtype), jnp.asarray(u, dtype), jnp.full(n, 100., dtype),
                             jnp.zeros(n, dtype), jnp.asarray([0] + [-1] * (n - 1), jnp.int32),
                             jnp.asarray([2.5] + [0.] * (n - 1), dtype))
                state['fate'] = state['fate'].at[-1].set(C.PLANE)
                for stop in (1, 32):
                    used, out = run((), jax.random.PRNGKey(203), state, jnp.int32(stop), n, n)
                    results[f'{label}_{stop}/iterations'] = np.asarray(used)
                    for path, value in jax.tree_util.tree_flatten_with_path(out)[0]:
                        key = '/'.join(str(item.key) for item in path)
                        results[f'{label}_{stop}/{key}'] = np.asarray(value)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if (args.out is None) == (args.reference is None):
        parser.error('choose exactly one of --out or --reference')
    current = snapshots()
    if args.out is not None:
        with args.out.open('xb') as stream:
            np.savez(stream, **current)
        print(f'Captured {len(current)} arrays across 24 raw-state cases')
        return
    with np.load(args.reference, allow_pickle=False) as old:
        assert set(current) == set(old.files)
        for name, actual in current.items():
            expected = old[name]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
            np.testing.assert_array_equal(actual, expected, err_msg=name)
            assert actual.tobytes() == expected.tobytes(), name
    print(f'Exact byte parity: {len(current)} arrays across 24 raw-state cases')


if __name__ == '__main__':
    main()
