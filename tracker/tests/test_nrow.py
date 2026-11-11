"""Sampled tracker rows (opt-in tracker_rows=sampled): library build, fallbacks, and the device draw."""
import numpy as np
import pytest

from trkfastsim.recoresp import nrow as NR

AXIS, BZ = np.array([-3904.0, 0.0]), 1.0


def _training(rng, n):
    """Entry states on helices with |p| ~ 100 MeV/c; nrow grows with tan(dip)^-1 (more turns at low pitch)."""
    pm = rng.uniform(95, 106, n)
    td = rng.uniform(0.5, 1.0, n)
    pt = pm / np.sqrt(1 + td ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    p = np.column_stack([pt * np.cos(phi), pt * np.sin(phi), pt * td])
    r = rng.uniform(380, 450, n)
    a = rng.uniform(0, 2 * np.pi, n)
    x = np.column_stack([AXIS[0] + r * np.cos(a), AXIS[1] + r * np.sin(a), np.full(n, -1500.0)])
    nrow = rng.poisson(30 / td)
    return rng.uniform(size=n) < 0.9, nrow, x, p


def test_build_keeps_every_track_and_falls_back():
    rng = np.random.default_rng(1)
    has, nrow, x, p = _training(rng, 20000)
    tab = NR.build(has, nrow, x, p, AXIS, BZ, min_count=20)
    assert tab["n_training"] == has.sum()
    assert np.all(tab["cell_count"] >= 20)
    assert set(np.unique(tab["cell_level"])) <= {0, 1, 2}
    np.testing.assert_allclose(tab["training_level_fraction"].sum(), 1.0)
    frac = tab["training_level_fraction"]
    assert frac[0] > 0.4 and frac[0] + frac[1] > 0.99      # sparse cells borrow their (|p|, tan dip) pool
    with pytest.raises(ValueError):
        NR.build(np.zeros(3, bool), np.zeros(3), x[:3], p[:3], AXIS, BZ)


def test_sampled_nrow_follows_the_training_conditionals(tmp_path):
    rng = np.random.default_rng(2)
    has, nrow, x, p = _training(rng, 40000)
    path = tmp_path / "lib.npz"
    NR.save(path, NR.build(has, nrow, x, p, AXIS, BZ))
    tab = NR.load(path)
    h2, n2, x2, p2 = _training(np.random.default_rng(3), 20000)
    got = NR.sample_numpy(tab, np.random.default_rng(4), h2, x2, p2)
    assert np.all(got[~h2] == 0)
    td = p2[:, 2] / np.hypot(p2[:, 0], p2[:, 1])
    for lo, hi in ((0.5, 0.65), (0.85, 1.0)):
        m = h2 & (td >= lo) & (td < hi)
        assert abs(got[m].mean() - n2[m].mean()) < 4 * n2[m].std() / np.sqrt(m.sum()) + 0.5


def test_device_sample_matches_the_host_distribution(tmp_path):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    rng = np.random.default_rng(5)
    has, nrow, x, p = _training(rng, 30000)
    tab = NR.build(has, nrow, x, p, AXIS, BZ)
    dn = NR.to_device(tab)
    got = np.asarray(jax.jit(NR.sample)(dn, jax.random.PRNGKey(0), jnp.asarray(has), jnp.asarray(x), jnp.asarray(p)))
    host = NR.sample_numpy(tab, np.random.default_rng(6), has, x, p)
    assert got.dtype.kind == "i" and np.all(got[~has] == 0)
    assert abs(got[has].mean() - host[has].mean()) < 0.5
    assert np.isin(got[has], tab["pool"]).all()
