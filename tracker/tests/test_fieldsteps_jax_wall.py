"""sim.fieldsteps_jax opt-in complete_outer_wall: the device wall column equals fieldwall.WallIndex (synthetic geometry)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from trkfastsim.sim import fieldsteps as fs  # noqa: E402
from trkfastsim.sim import fieldsteps_jax as FJ  # noqa: E402
from trkfastsim.sim.fieldwall import WallIndex  # noqa: E402

from test_fieldsteps_jax import _geometry, _grid, _lines, _setup  # noqa: E402


def _chords(rng, n):
    """Straight chords through the two straw layers: random, plus grazing ones whose closest approach to a straw axis
    lies between the gas and the outer wall radius (wall only, no gas)."""
    o, d = _lines(n, rng)
    L = np.full(n, 90.0)
    geom = _geometry()
    k = n // 2
    js = rng.integers(0, len(geom.mid), k)
    b = rng.uniform(geom.radius + 1e-4, geom.wall_rmax - 1e-4, k)
    ang = rng.uniform(0.2, 1.2, k) * rng.choice([-1, 1], k)
    dd = np.c_[np.zeros(k), np.sin(ang), np.cos(ang)]
    perp = np.c_[np.zeros(k), np.cos(ang), -np.sin(ang)]
    mid = geom.mid[js] + np.c_[rng.uniform(-300, 300, k), np.zeros(k), np.zeros(k)]
    o[:k] = mid + perp * b[:, None] - dd * 20.0
    d[:k] = dd
    L[:k] = 40.0
    return o, d, L


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_device_complete_wall_equals_the_wall_index(dtype):
    B = np.array([0.0, 0.0, 1.0])
    geom, env, trk = _setup(B, chord_mm=5.0, complete_outer_wall=True, dtype=dtype)
    o, d, L = _chords(np.random.default_rng(5), 400)
    sp, _, _ = FJ._straws(trk.T, trk.cfg, trk.pwidth, jnp.asarray(o, dtype), jnp.asarray(d, dtype), jnp.asarray(L, dtype))
    ref = WallIndex(geom, geom.wall_rmax, fs.ChordIndex).columns(o, d, L)[0]
    got = np.asarray(sp["wall_all"], float)
    # float32: near-tangent wall chords are ill-conditioned (sqrt of a small discriminant), a few um per chord
    tol = 1e-9 if dtype == np.float64 else 3e-3
    np.testing.assert_allclose(got, ref, atol=tol)
    assert abs(got.sum() / ref.sum() - 1) < (1e-12 if dtype == np.float64 else 1e-3)
    gated = np.asarray(sp["wall"], float).sum(1)
    assert (ref[:200] > 0).all()
    assert (ref[:200] > gated[:200] + 1e-6).mean() > 0.5          # the grazing chords add wall-only pieces


def test_default_tracker_has_no_complete_wall_and_keeps_its_rows():
    B = np.array([0.0, 0.0, 1.0])
    _, _, a = _setup(B, chord_mm=5.0)
    _, _, b = _setup(B, chord_mm=5.0, complete_outer_wall=False)
    assert a.cfg.complete_wall is False and a.cfg == b.cfg
    o, d = _lines(100, np.random.default_rng(7))
    ra = FJ.transport(a, jax.random.PRNGKey(3), o, d * 105.0, np.zeros(len(o)))
    rb = FJ.transport(b, jax.random.PRNGKey(3), o, d * 105.0, np.zeros(len(o)))
    for k in ra.steps:
        assert np.array_equal(np.asarray(ra.steps[k]), np.asarray(rb.steps[k]))
    sp, _, _ = FJ._straws(a.T, a.cfg, a.pwidth, jnp.asarray(o), jnp.asarray(d), jnp.full(len(o), 90.0))
    assert "wall_all" not in sp


def test_wall_only_tracks_lose_energy_only_with_the_complete_wall():
    """Tracks running inside one straw's wall (between the gas and the outer radius, along the straw axis): no gas
    passage either way; they lose energy only with the complete wall."""
    B = np.array([0.0, 0.0, 1e-6])
    geom = _geometry()
    js = np.arange(4, 40, 4)
    b = 0.5 * (geom.radius + geom.wall_rmax)
    o = geom.mid[js] + np.c_[np.full(len(js), -300.0), np.full(len(js), b), np.zeros(len(js))]
    d = np.tile([1.0, 0.0, 0.0], (len(js), 1))
    fin = {}
    for flag in (False, True):
        _, _, trk = _setup(B, chord_mm=5.0, msc=False, complete_outer_wall=flag)
        r = FJ.transport(trk, jax.random.PRNGKey(4), o, d * 105.0, np.zeros(len(o)))
        assert int(np.asarray(r.steps["valid"]).sum()) == 0
        f = r.final
        fin[flag] = np.sqrt(np.asarray(f["px"]) ** 2 + np.asarray(f["py"]) ** 2 + np.asarray(f["pz"]) ** 2)
    np.testing.assert_allclose(fin[False], 105.0, atol=1e-6)
    assert (fin[True] < 104.0).all()
