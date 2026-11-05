"""Single-step synthetic foil/query/core contract; not full finite-foil navigation."""
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from mu2efastsim import eslab_jax as SJ, materials as M
from beamfastsim import flash_foil_frames as F
from beamfastsim import flash_foil_query as Q


@pytest.fixture(scope="module")
def slab_tables():
    table = SJ.lane_tables([(M.ALUMINIUM, 20., .03411, .00099, 1.)], msc=True)
    return table._replace(tab=table.tab._replace(**{
        name: np.full_like(getattr(table.tab, name), value) for name, value in
        dict(sig_b=0., sig_d=0., sig_a=0., dedx=21., rng=100., lam1=100.,
             hard=0., soft=0., soft_all=0.).items()}))


def _setup(dtype, rim=True, gap=.01):
    table = F.build_table(SimpleNamespace(axis_x=0., foil_z=[0.], foil_ht=[10.], foil_rin=[1.], foil_rout=[3.]))
    n = 4
    x = np.tile(np.array([3.-gap if rim else 1.+gap, 0., .5], dtype), (n, 1))
    u = np.tile(np.array([.6 if rim else -.6, 0., .8], dtype), (n, 1))
    field = np.tile(np.array([.1, .2, .7], dtype), (n, 1))
    frame = F.entry_frame(table, np.zeros(n, np.int32), x, u, field)
    state = SJ.lane_state(np.full(n, 105., dtype), frame["local_u"], frame["depth"], 20.)
    state["x"] = jnp.asarray(frame["local_x"])
    state["pend"] = jnp.full(n, .1, dtype)
    return table, frame, state


def _run(tables, frame, state, query, msc, field):
    return jax.jit(partial(SJ.lane_step_emissions_navigation, jax.random.PRNGKey(17), tables,
                           jnp.zeros(4, jnp.int32), B=jnp.asarray(frame["local_B"]) if field else None,
                           msc=msc, positron=False, boundary_query=query))(state)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("msc,field", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rim", [False, True])
def test_actual_leg_hits_finite_rim_or_hole_before_zface(slab_tables, monkeypatch, dtype, msc, field, rim):
    if not hasattr(SJ, "lane_step_emissions_navigation"):
        pytest.skip("Transactional boundary-query core interface required")
    monkeypatch.setattr(SJ.eloss_jax, "sample_loss_electron", lambda key, mc, T, length, tcut, mean, *a, **kw: mean)
    table, frame, state = _setup(dtype, rim)
    out, photons, status = _run(slab_tables, frame, state, Q.make_query(table, np.zeros(4, np.int32), frame), msc, field)
    assert not np.asarray(status.unresolved).any()
    assert np.asarray(out["exited"]).all() and not np.asarray(out["stopped"]).any()
    assert not np.asarray(photons.valid).any()
    xyz = F.to_global(frame, np.asarray(out["x"]), point=True)
    tolerance = 2.e-6 if dtype == np.float32 else 2.e-12
    np.testing.assert_allclose(np.hypot(xyz[:, 0], xyz[:, 1]), 3. if rim else 1., atol=tolerance, rtol=0)
    assert (np.asarray(out["path"]) < .1).all()
    np.testing.assert_array_equal(out["pend"], 0.)
    np.testing.assert_allclose(out["cont"], 21*(np.asarray(state["pend"])+np.asarray(out["path"])),
                               atol=5*tolerance, rtol=5*tolerance)


def test_subresolution_float32_start_refuses_without_tuning_guards(slab_tables):
    if not hasattr(SJ, "lane_step_emissions_navigation"):
        pytest.skip("Transactional boundary-query core interface required")
    table, frame, state = _setup(np.float32, gap=1.e-5)
    query = Q.make_query(table, np.zeros(4, np.int32), frame)
    out, photons, status = _run(slab_tables, frame, state, query, True, True)
    np.testing.assert_array_equal(status.unresolved, True)
    for name in state.keys() - {"it"}:
        assert np.asarray(out[name]).tobytes() == np.asarray(state[name]).tobytes(), name
    for value in photons:
        assert not np.asarray(value).any()


def test_unknown_foil_query_rolls_back_instead_of_transporting_through_vacuum(slab_tables):
    if not hasattr(SJ, "lane_step_emissions_navigation"):
        pytest.skip("Transactional boundary-query core interface required")
    table, frame, state = _setup(np.float64)
    query = Q.make_query(table, np.array([-1, -1, -1, -1]), frame)
    out, photons, status = _run(slab_tables, frame, state, query, True, True)
    np.testing.assert_array_equal(status.unresolved, True)
    for name in state.keys() - {"it"}:
        assert np.asarray(out[name]).tobytes() == np.asarray(state[name]).tobytes(), name
    assert out["it"] == state["it"] + 1
    for value in photons:
        assert not np.asarray(value).any()
