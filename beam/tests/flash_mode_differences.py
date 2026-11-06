"""Print all raw cross-mode differences for the small vacuum CI regression fixtures."""
from dataclasses import replace
import json
import os
import platform

import jax
import jax.numpy as jnp
import numpy as np

from beamfastsim import flash_navigator_jax as FN
from test_flash_target_jax import AXIS, KEY, _geom, _starts
from test_flash_vacuum_fast import MODES, _build, _initial


def differences(reference, candidate):
    left, lt = jax.tree_util.tree_flatten_with_path(reference)
    right, rt = jax.tree_util.tree_flatten_with_path(candidate)
    assert lt == rt
    rows = []
    for (path, a), (_, b) in zip(left, right):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        if a.tobytes() == b.tobytes():
            continue
        row = dict(path=jax.tree_util.keystr(path), shape=list(a.shape), dtype=str(a.dtype),
                   unequal=int(np.count_nonzero(a != b)))
        if a.dtype.kind == 'f':
            assert np.isfinite(a).all() and np.isfinite(b).all()
            delta = np.abs(a - b)
            spacing = np.abs(np.spacing(np.maximum(np.abs(a), np.abs(b))))
            ulps = np.divide(delta, spacing, out=np.zeros_like(delta), where=spacing > 0)
            row.update(max_abs=float(np.max(delta)), max_spacing_units=float(np.max(ulps)),
                       zero_changes=int(np.count_nonzero(((a == 0) | (b == 0)) &
                                                        ((a != b) | (np.signbit(a) != np.signbit(b))))))
            indices = np.flatnonzero(a != b)[:8]
            row['examples'] = [dict(index=int(i), reference=float(a.flat[i]).hex(),
                                    candidate=float(b.flat[i]).hex()) for i in indices]
        rows.append(row)
    return rows


def report(label, reference, candidate):
    print(json.dumps(dict(case=label, differences=differences(reference, candidate))), flush=True)


def bucket_cases():
    geom = _geom(foil_z=(5800., 5803.), ht=.25, plane=5810.)
    x, u = _starts(8, z=5790.)
    x[4:, 0] += 100.
    reference = None
    modes = MODES + tuple(dict(skip_empty=True, vacuum_fast=True, vacuum_bucket=b) for b in (1, 4, 16))
    for mode in modes:
        init, run = FN._build(geom, np.float64, 1., 30000., False, True, True, False,
                             (0., 0., 1.), photon_capacity=32, **mode)
        state = _initial(init, x, u, 100.)
        full = run((), KEY, state, jnp.int32(1000), 8, 8)
        first, mid = run((), KEY, state, jnp.int32(7), 8, 8)
        last, split = run((), KEY, mid, jnp.int32(993), 8, 8)
        report(f'bucket chunks {mode}', full, (first + last, split))
        if reference is None:
            reference = full
        else:
            report(f'bucket modes {mode}', reference, full)


def msc_cases():
    geom = _geom(foil_z=(5800., 5803.), ht=.25, plane=8000.)
    geom = replace(geom, beam=replace(geom.beam, ipa_t=.05), opa_t=.05, ring_t=.05, ring_hl=.5)
    x, u = _starts(32, z=5798.75)
    x[24:26] = [AXIS + 299.5, 0., 7000.]
    x[26:28] = [AXIS + 899.5, 0., 7000.]
    x[28:30] = [AXIS + 299.45, 0., 6900.5]
    x[30:] = [AXIS + 10., 0., 7000.]
    u[24:30] = [1., 0., 0.]
    for positron in (False, True):
        reference = None
        for mode in MODES:
            init, run = _build(geom, msc=True, positron=positron, s_max=8., field=(.05, .02, .5), **mode)
            state = _initial(init, x, u)
            full = run((), KEY, state, jnp.int32(1000), 32, 32)
            if reference is None:
                reference = full
            else:
                report(f'msc positron={positron} {mode}', reference, full)


if __name__ == '__main__':
    print(json.dumps(dict(platform=platform.platform(), python=platform.python_version(),
                          jax=jax.__version__, numpy=np.__version__, x64=jax.config.x64_enabled,
                          xla_flags=os.environ.get('XLA_FLAGS', ''))), flush=True)
    bucket_cases()
    msc_cases()
