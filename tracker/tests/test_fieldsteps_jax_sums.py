"""sim.fieldsteps_jax: per-material and per-lane sums without float scatter-adds (synthetic geometry, no data files).

A float scatter-add with repeated indices sums in a run-dependent order on a GPU (atomics), so two identical runs of
the transport differed in the last bits of a few tracks, enough to move a tracker crossing now and then. The support
column and the energy loss per lane are reductions instead; these tests keep it so and check the support column
against world.CoaxialShells.column.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mu2efastsim import world as Wd  # noqa: E402
from trkfastsim.sim import fieldsteps as fs  # noqa: E402
from trkfastsim.sim import fieldsteps_jax as FJ  # noqa: E402
from trkfastsim.sim import steps as S  # noqa: E402

from test_fieldsteps_jax import _geometry, _grid  # noqa: E402


def _interior():
    """Coaxial shells about the z axis with repeated materials inside one chord: two copper rings overlapping in z,
    two aluminium half-sectors on the same z interval (one interval, two members), and a thin ring of a third radius."""
    I3 = np.eye(3)
    tubes = [("G4_Cu", I3, np.array([0.0, 0.0, 0.0]), 100.0, 110.0, 300.0, 0.0, 2 * np.pi),
             ("G4_Cu", I3, np.array([0.0, 0.0, 250.0]), 100.0, 110.0, 300.0, 0.0, 2 * np.pi),
             ("G4_Al", I3, np.array([0.0, 0.0, 100.0]), 100.0, 110.0, 400.0, 0.0, np.pi),
             ("G4_Al", I3, np.array([0.0, 0.0, 100.0]), 100.0, 110.0, 400.0, np.pi, 0.5 * np.pi),
             ("G4_Cu", I3, np.array([0.0, 0.0, 0.0]), 60.0, 61.0, 600.0, 0.0, 2 * np.pi)]
    interior = S.TrackerInterior.__new__(S.TrackerInterior)
    interior.shells = S.InteriorColumn(Wd.CoaxialShells(tubes, axis_point=[0.0, 0.0, 0.0]))
    interior.nsub = 8
    return interior


def _tracker(dtype=np.float64, **kw):
    geom = _geometry()
    env = fs.Envelope(geom, pad_r=10.0, pad_z=10.0, axis_xy=(0.0, 0.0))
    interior = _interior()
    B = np.array([0.0, 0.0, 1.0])
    return FJ.Tracker(geom, _grid(B), interior, env, np.zeros(3), B, dtype=dtype, **kw), interior


def _chords(n, rng):
    """Chords from inside r < 40 mm outwards through the shells, with a z component."""
    r, ph = rng.uniform(0, 40, n), rng.uniform(0, 2 * np.pi, n)
    x = np.c_[r * np.cos(ph), r * np.sin(ph), rng.uniform(-300, 500, n)]
    d = np.c_[rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 0.6, n)]
    return x, d / np.linalg.norm(d, axis=1)[:, None], rng.uniform(20, 200, n)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_support_column_per_material_equals_the_numpy_shells(dtype):
    trk, interior = _tracker(dtype)
    x, c, L = _chords(3000, np.random.default_rng(11))
    sl, over = FJ._support(trk.T, trk.cfg, trk.groups, jnp.asarray(x, dtype), jnp.asarray(c, dtype),
                           jnp.asarray(L, dtype))
    sl = np.asarray(sl, float)
    assert int(over) == 0
    ref = interior.shells.column(x, c, L, nsub=trk.cfg.nsub)
    assert set(ref) == set(trk.sup_names)
    tol = 1e-9 if dtype == np.float64 else 2e-3
    for i, m in enumerate(trk.sup_names):
        np.testing.assert_allclose(sl[:, i], ref[m], rtol=tol, atol=tol, err_msg=m)
    assert (sl > 0).sum(0).min() > 50          # every material is crossed by many chords


def _float_scatter_adds(jaxpr, acc=None):
    acc = [] if acc is None else acc
    for e in jaxpr.eqns:
        if e.primitive.name == "scatter-add" and any(jnp.issubdtype(v.aval.dtype, jnp.floating) for v in e.outvars):
            acc.append(str(e.outvars[0].aval))
        for v in e.params.values():
            for s in (v if isinstance(v, (tuple, list)) else (v,)):
                j = getattr(s, "jaxpr", s)
                if hasattr(j, "eqns"):
                    _float_scatter_adds(j, acc)
    return acc


def test_the_chord_kernel_has_no_float_scatter_add():
    """With support shells, energy loss and scattering on: no float scatter-add anywhere in one transport chord."""
    trk, _ = _tracker()
    n = 64
    x, c, _ = _chords(n, np.random.default_rng(3))
    s = FJ._state(jnp.asarray(x), jnp.asarray(c), jnp.full(n, 104.0), jnp.zeros(n), jnp.ones(n, bool))
    rec = FJ._records(n, jnp.float64)
    tk_ = ((trk.T, trk.groups, trk.env, trk.fn, trk.pwidth, trk.lnt0, trk.dlnt, trk.mc), trk.cfg)
    jx = jax.make_jaxpr(lambda s_, r_, k: FJ._chord(tk_, s_, r_, jnp.arange(n), k, 0))(s, rec, jax.random.PRNGKey(1))
    assert _float_scatter_adds(jx.jaxpr) == []
