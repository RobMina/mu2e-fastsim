"""The DriftANN/BkgANN MLP runs in float32 as stock (SOFIE) with full float32 matmuls in every device mode, not
XLA's default (TF32 on Ampere and later)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from trkfastsim.reco import driftann  # noqa: E402


def test_mlp_is_float32_with_highest_matmul_precision():
    rng = np.random.default_rng(0)
    layers = [(rng.normal(size=(4, 8)).astype(np.float32), rng.normal(size=8).astype(np.float32)),
              (rng.normal(size=(8, 1)).astype(np.float32), rng.normal(size=1).astype(np.float32))]
    x = rng.normal(size=(16, 4))
    out = driftann.mlp(layers, x)
    h = np.maximum(x.astype(np.float32) @ layers[0][0] + layers[0][1], 0)
    ref = 1 / (1 + np.exp(-(h @ layers[1][0] + layers[1][1])))[:, 0]
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5)
    assert "HIGHEST" in jax.jit(lambda v: driftann.mlp(layers, v)).lower(x).as_text()
