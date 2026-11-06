"""Compact vacuum bursts retain canonical material lanes and the shared RNG clock."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from beamfastsim import flash_navigator_jax as FN
from beamfastsim import cechain as C
from test_flash_target_jax import _geom, _starts, KEY
from test_flash_vacuum_fast import _exact, _bucket_roundoff


@pytest.mark.parametrize("bucket", [1, 4, 16])
def test_compact_tail_caps_padding_and_material_fallback(bucket):
    geom = _geom(foil_z=(5800., 5803.), ht=.25, plane=5810.)
    x, u = _starts(8, z=5790.)
    x[4:, 0] += 100.
    outputs = []
    for mode in ({}, dict(skip_empty=True, vacuum_fast=True, vacuum_bucket=bucket)):
        init, run = FN._build(geom, np.float64, 1., 30000., False, True, True, False,
                             (0., 0., 1.), photon_capacity=32, **mode)
        st = init((), jnp.asarray(x), jnp.asarray(u), jnp.full(8, 100.), jnp.zeros(8),
                  jnp.full(8, -1, jnp.int32), jnp.zeros(8))
        full = run((), KEY, st, jnp.int32(1000), 8, 8)
        na, split = run((), KEY, st, jnp.int32(7), 8, 8)
        nb, split = run((), KEY, split, jnp.int32(993), 8, 8)
        _exact(full, (na + nb, split))
        assert int(full[0]) == int(full[1]["slab"]["it"])
        assert not np.any(np.asarray(full[1]["fate"]) == C.LIVE)
        outputs.append(full)
    _bucket_roundoff(*outputs)


def test_compact_vacuum_reaches_cap_and_preserves_dead_rows():
    geom = _geom(foil_z=(5000.,), ht=.05, plane=8500.)
    x, u = _starts(8, z=5700.)
    u[:] = [1., 0., 0.]
    outputs = []
    for bucket in (0, 4):
        init, run = FN._build(geom, np.float64, 1., 30000., False, True, True, False,
                             (0., 0., 1.), photon_capacity=32, vacuum_bucket=bucket)
        st = init((), jnp.asarray(x), jnp.asarray(u), jnp.full(8, .1), jnp.zeros(8),
                  jnp.full(8, -1, jnp.int32), jnp.zeros(8))
        st["fate"] = st["fate"].at[2:].set(C.PLANE)
        st['x'] = st['x'].at[2:, 0].add(jnp.arange(1., 7.))
        st['t'] = st['t'].at[2:].set(jnp.arange(1., 7.))
        st['s'] = st['s'].at[2:].set(jnp.arange(11., 17.))
        result = run((), KEY, st, jnp.int32(1200), 8, 8)
        for name in ('x', 'u', 'T', 't', 's', 'fate', 'inl', 'nvac', 'nstall', 'nadapt'):
            _exact({name: st[name][2:]}, {name: result[1][name][2:]})
        assert int(result[0]) == int(result[1]["slab"]["it"]) == 1200
        np.testing.assert_array_equal(result[1]["nvac"], [1200, 1200, 0, 0, 0, 0, 0, 0])
        outputs.append(result)
    for name in outputs[0][1]:
        a, b = outputs[0][1][name], outputs[1][1][name]
        if name in ("x", "u"):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-10, err_msg=name)
        else:
            _exact(a, b)
    _exact(outputs[0][0], outputs[1][0])
