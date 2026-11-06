"""Core-hook/finite-annulus component integration, not complete source navigation."""
from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp

from beamfastsim import flash_annulus_jax as A
from mu2efastsim import eslab_jax as SJ
from mu2efastsim import materials as M


@pytest.fixture(scope="module")
def tables():
    layers = [(M.BY_NAME['G4_Al'], 4., .03411, .00099, .1)]
    return {charge: SJ.lane_tables(layers, positron=charge, msc=True) for charge in (False, True)}


@pytest.mark.parametrize('positron', [False, True])
@pytest.mark.parametrize('field', [False, True])
@pytest.mark.parametrize('msc', [False, True])
def test_physical_tables_do_not_advance_or_emit_outside_annulus(tables, positron, field, msc):
    n = 32
    x = np.tile([5. - 1e-6, 0., 2.], (n, 1))
    u = np.tile([1., 0., 0.], (n, 1))
    x[n // 2:, 0] = 1. + 1e-6
    u[n // 2:, 0] = -1.
    st = SJ.lane_state(jnp.full(n, 100.), jnp.asarray(u), jnp.full(n, 2.), 4., positron=positron)
    st['x'] = jnp.asarray(x)
    st['pend'] = jnp.full(n, .001)
    B = jnp.broadcast_to(jnp.array([0., 0., 1.]), (n, 3)) if field else None

    def boundary(x, u, p, q, B, smax):
        return A.crossing(x, u, p, q, B, smax, 2., 2., 1., 5.).distance

    step = jax.jit(partial(SJ.lane_step_photons, jax.random.PRNGKey(19), tables[positron],
                            jnp.zeros(n, jnp.int32), B=B, positron=positron, msc=msc,
                            boundary=boundary, safety=lambda x: A.safety(x, 2., 2., 1., 5.)))
    after, photons = step(st)
    assert np.asarray(after['exited']).all()
    assert not np.asarray(after['stopped']).any()
    assert not np.asarray(photons.valid).any()
    np.testing.assert_array_equal(after['pend'], 0.)
    np.testing.assert_allclose(after['path'], 1e-6, rtol=2e-6, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(after['x'])[:, :2], axis=1),
                                np.r_[np.full(n // 2, 5.), np.ones(n // 2)], atol=1e-12)
    np.testing.assert_allclose(after['T'] + after['cont'] + after['eph'] + after['ede'] + after['ean'],
                                st['T'] + SJ.ME * after['nan'], atol=1e-12)


def test_component_hole_crossing_reentry_keeps_depth_and_path(tables, monkeypatch):
    monkeypatch.setattr(SJ.eloss_jax, 'sample_loss_electron', lambda key, mc, T, length, tcut, mean, positron, **kw: mean)
    lt = tables[False]
    tab = lt.tab._replace(sig_b=np.zeros_like(lt.tab.sig_b), sig_d=np.zeros_like(lt.tab.sig_d),
                          dedx=np.full_like(lt.tab.dedx, .1))
    lt = lt._replace(tab=tab)
    x, u = jnp.array([[1.2, 0., 2.7]]), jnp.array([[-1., 0., 0.]])

    def boundary(x, u, p, q, B, smax):
        return A.crossing(x, u, p, q, B, smax, 2., 2., 1., 5.).distance

    step = jax.jit(partial(SJ.lane_step_photons, jax.random.PRNGKey(20), lt,
                            jnp.zeros(1, jnp.int32), msc=False, boundary=boundary,
                            safety=lambda x: A.safety(x, 2., 2., 1., 5.)))
    start = SJ.lane_state(jnp.array([100.]), u, x[:, 2], 4.)
    start['x'] = x
    first, record = step(start)
    assert first['exited'][0] and not record.valid[0]
    np.testing.assert_allclose(first['path'], [.2], atol=1e-12)
    momentum = jnp.sqrt(first['T'] * (first['T'] + 2 * SJ.ME))
    entry = A.crossing(first['x'], first['d'], momentum, -1., None, 10., 2., 2., 1., 5., entering=True)
    assert entry.supported[0] and entry.surface[0] == 4
    np.testing.assert_allclose(entry.distance, [2.], atol=1e-12)
    point = first['x'] + entry.distance[:, None] * first['d']
    again = SJ.lane_state(first['T'], first['d'], point[:, 2], 4.)
    again['x'] = point
    second, record = step(again)
    assert second['exited'][0] and not record.valid[0]
    np.testing.assert_allclose(second['x'], [[-5., 0., 2.7]], atol=1e-12)
    np.testing.assert_allclose(first['path'] + entry.distance + second['path'], [6.2], atol=1e-12)
    np.testing.assert_allclose(second['T'], [100. - .1 * 4.2], atol=1e-12)
