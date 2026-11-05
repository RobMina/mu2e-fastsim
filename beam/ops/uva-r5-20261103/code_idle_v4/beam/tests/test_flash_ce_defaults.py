"""Optional byte regression against a caller-supplied pristine CE navigator source."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from beamfastsim import cechain as C
from beamfastsim import cechain_jax as CJ
from beamfastsim import geometry as G


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_default_ce_matches_pristine_source_byte_exact(dtype):
    path = os.environ.get("FLASH_CE_BASELINE_SOURCE")
    if not path or not Path(path).is_file():
        pytest.skip("Provide pristine CE navigator source for explicit byte regression")
    spec = importlib.util.spec_from_file_location("beamfastsim._flash_ce_baseline", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    b = G.BeamGeometry(0., np.array([5., 10.]), np.zeros(2), np.full(2, 75.), np.full(2, .0528),
                       -50., 1., 100., 200., 300., .5, 100., 200., 900., 100., 200.)
    g = C.CEGeometry(b, 900., 900., 1., z_plane=20., z_up=-20., z_ds=-30., coll5_z0=-29., coll5_z1=-28.)
    x = np.array([[10., 0., 5.], [10., 0., 0.], [0., 0., 12.], [10., 0., 5.]])
    u = np.array([[0., 0., 1.], [.1, 0., 1.], [0., 0., 1.], [0., 0., -1.]])
    kw = dict(key=jax.random.PRNGKey(13), geom=g, field=[0., 0., 1.], x0=x, u0=u,
              p0=100., t0=123., h=1., msc=True, dtype=dtype, min_bucket=4, chunk=16)
    a, b = old.transport(**kw), CJ.transport(**kw)
    for name in a._fields:
        av, bv = getattr(a, name), getattr(b, name)
        if av is None:
            assert bv is None
        else:
            assert av.dtype == bv.dtype and av.shape == bv.shape, name
            assert av.tobytes() == bv.tobytes(), name
