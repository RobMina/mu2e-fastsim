"""The split photon kernels (oneAPI path) give the same photons as the fused one on the same key."""
import numpy as np
import pytest

import jax


def test_device_photons_split_equals_fused():
    from crvfastsim.response import bench, photons_jax
    from crvfastsim.reco.coinc_jax import is_oneapi
    if is_oneapi():
        pytest.skip("the fused kernel aborts the oneAPI compiler; the split path is checked there by v4sipm --device")
    rng = np.random.default_rng(3)
    L = bench.synthetic_table(rng, prob=0.0015)
    dt = photons_jax.DeviceTable(L)
    rows = bench.synthetic_rows(rng, 300)              # ~4e4 arrivals: plenty for the equality check
    key = jax.random.PRNGKey(11)
    a = photons_jax.device_photons(dt, rows, key, split=False)
    b = photons_jax.device_photons(dt, rows, key, split=True)
    va, vb = np.asarray(a["valid"]), np.asarray(b["valid"])
    assert va.sum() > 1000 and np.array_equal(va, vb)
    for k in ("step", "chan", "time"):
        assert np.array_equal(np.asarray(a[k])[va], np.asarray(b[k])[vb]), k
    assert bool(a["overflow"]) == bool(b["overflow"])
