"""Hybrid handoffs preserve the canonical slab clock and include candidate iterations once."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from beamfastsim import flash_navigator_jax as FN
from beamfastsim.flash_hybrid import make_runner
from test_flash_target_jax import _geom, _starts, KEY
from test_flash_vacuum_fast import _exact


@pytest.mark.parametrize("adaptive", [False, True])
def test_handoffs_caps_and_canonical_rng(adaptive):
    geom = _geom(foil_z=(5800., 5803.), ht=.25, plane=5810.)
    x, u = _starts(8, z=5790.)
    init, run = FN._build(geom, np.float64, 1., 30000., False, True, True, False,
                          (0., 0., 1.), photon_capacity=32, skip_empty=True)
    st = init((), jnp.asarray(x), jnp.asarray(u), jnp.full(8, 100.), jnp.zeros(8),
              jnp.full(8, -1, jnp.int32), jnp.zeros(8))
    expected = run((), KEY, st, jnp.int32(1000), 8, 8)
    diag = {}
    runner = make_runner(run, (), KEY, st, 8, 8, 16, diag, adaptive=adaptive)
    a, partial = runner((), KEY, st, jnp.int32(7))
    b, result = runner((), KEY, partial, jnp.int32(993))
    _exact(expected, (a + b, result))
    assert diag["bursts"] > 1 and diag["vacuum_iterations"] > 0 and diag["device_iterations"] > 0


def test_public_source_hybrid_and_validation():
    from beamfastsim.flash_target_jax import target_source
    geom = _geom(foil_z=(5800.,), ht=.25, plane=5805.)
    x, u = _starts(8, z=5790.)
    base = target_source(KEY, geom, [0., 0., 0.], x, u, 100., 1e16, 11, msc=False)
    fast = target_source(KEY, geom, [0., 0., 0.], x, u, 100., 1e16, 11, msc=False,
                         cpu_vacuum=16, cpu_adaptive=True)
    for name in ("state", "records", "count", "t0", "capped", "overflow", "loopers"):
        _exact(base[name], fast[name])
    _exact(base["diagnostics"]["iterations"], fast["diagnostics"]["iterations"])
    assert fast["diagnostics"]["scheduling"]["11"]["vacuum_iterations"] > 0
    for value in (-1, 1.5, True):
        with pytest.raises(ValueError, match="cpu_vacuum"):
            target_source(KEY, geom, [0., 0., 0.], x, u, 100., 0., 11, cpu_vacuum=value)
    with pytest.raises(ValueError, match="cpu_adaptive"):
        target_source(KEY, geom, [0., 0., 0.], x, u, 100., 0., 11, cpu_adaptive=True)
