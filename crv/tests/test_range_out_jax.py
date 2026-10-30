"""New-default device transport: exact CSDA knots and uncapped delta/substep contracts."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as j
from crvfastsim import range_out_jax as R, transport as T, transport_jax as J
from mu2efastsim import eloss as E, materials as M
from test_transport_jax import gen, geom


def test_csda_interpolation_matches_reference():
    mat = M.BY_NAME["G4_Al"]
    knots = R.table(mat)
    p = np.geomspace(1, 1e5, 128)
    length = np.geomspace(.01, 10000, 128)
    normal = np.linspace(-3, 3, 128)
    for g in (None, normal):
        ref = T._range_loss(mat, p, length, g)
        got = jax.jit(R.range_loss)(j.asarray(p), j.asarray(length), knots, None if g is None else j.asarray(g))
        np.testing.assert_allclose(got, ref, rtol=2e-12, atol=2e-10)


@pytest.mark.parametrize("p,cut", [(40., .0872), (3000., .0872), (3000., .001)])
def test_uncapped_delta_energy_law(p, cut):
    n = 20000
    pp, ll = np.full(n, p), np.full(n, 10.)
    idx, values = E.sample_deltas(np.random.default_rng(981), M.POLYSTYRENE, pp, ll, cut)
    ref = np.bincount(idx, weights=values, minlength=n)
    got = np.asarray(jax.jit(R.delta_sum)(jax.random.key(751), j.asarray(pp), j.asarray(ll), j.full(n, cut)))
    se = np.sqrt(ref.var() / n + got.var() / n)
    assert abs(ref.mean() - got.mean()) < 6 * se
    assert np.isfinite(got).all() and (got >= 0).all()
    np.testing.assert_allclose(R.delta_rate(j.asarray(pp[:5]), cut), E.delta_rate_per_mm(M.POLYSTYRENE, pp[:5], cut), rtol=2e-14)


def test_substeps_stop_ledger_and_padding():
    p = j.asarray([30., 50., 1000., 50.])
    length = j.asarray([20., 20., 20., 0.])
    keys = tuple(jax.random.split(jax.random.key(17), 3))
    dep, delta, path, stopped = jax.jit(R.bar_loss)(keys, p, length, j.full(4, .0872))
    tin = j.sqrt(p * p + E.MMU**2) - E.MMU
    assert np.isfinite(dep).all() and np.isfinite(delta).all()
    assert bool(stopped[0]) and 0 < float(path[0]) < 20.
    np.testing.assert_allclose((dep + delta)[stopped], tin[stopped], rtol=2e-12, atol=1e-12)
    assert not bool(stopped[2])
    assert float(dep[3]) == float(delta[3]) == 0.


def test_fixed_loss_draws_match_numpy_bar_loop(monkeypatch):
    g = geom(8)
    generated = gen(12)
    p = np.array([10., 20., 30., 40., 50., 60., 70., 80., 120., 500., 3000., 10000.])
    generated["py"] = -p
    monkeypatch.setattr(T, "world_loss", lambda rng, o, d, p, world, edges:
                        (p, np.zeros(len(p)), np.ones(len(p), bool), None, None))
    monkeypatch.setattr(T, "msc_to_plane", lambda rng, o, d, p, *args, **kw: (o, d, np.zeros(len(p))))
    def fixed_loss(rng, mat, p, length, cut):
        return E.dedx_restricted(mat, p, cut) * length
    monkeypatch.setattr(E, "sample_loss", fixed_loss)
    monkeypatch.setattr(E, "sample_deltas", lambda rng, mat, p, length, cut: (np.arange(len(p)), .2 * length))
    monkeypatch.setattr(R.sampling, "sample_loss", lambda key, mat, p, length, cut: R.ej.dedx_restricted(mat, p, cut) * length)
    monkeypatch.setattr(R, "delta_sum", lambda key, p, length, cut: .2 * length)
    ref = T.transport(generated, g, np.random.default_rng(18), stage="C", world=[], range_out=True)[0]
    dev = J.DeviceTransport(g, [], stage="C", range_out=True, batch_size=12)
    o = generated[["x", "y", "z"]].to_numpy()
    d = np.tile([0., -1., 0.], (len(p), 1))
    b, ti, to, pin, path, kinetic, vis, keep, over = jax.device_get(dev.bar_kernel(
        jax.random.key(18), j.asarray(o), j.asarray(d), j.asarray(p), j.zeros(len(p)), j.ones(len(p), bool)))
    assert not over
    it, rank = np.nonzero(keep)
    np.testing.assert_array_equal(ref.event, it)
    np.testing.assert_array_equal(ref.bar, b[keep])
    for name, actual in (("len", path[keep]), ("edep", vis[keep]), ("p1", J.p_after(pin[keep], kinetic[keep]))):
        np.testing.assert_allclose(actual, ref[name], rtol=2e-11, atol=2e-10, err_msg=name)
    np.testing.assert_allclose(pin[keep], -ref.py0, rtol=2e-11, atol=2e-10)


def test_stage_c_range_schema_and_repeat():
    g = geom()
    a = gen(9, p=80.)
    dev = J.DeviceTransport(g, [], stage="C", range_out=True, batch_size=16)
    one = J.transport(a, g, np.random.default_rng(33), stage="C", prepared=dev)
    two = J.transport(a, g, np.random.default_rng(33), stage="C", prepared=dev)
    import pandas as pd
    for x, y in zip(one, two):
        pd.testing.assert_frame_equal(x, y)
    assert (one[0].edep >= 0).all() and (one[0].p1 >= 0).all()
    with pytest.raises(ValueError, match="match"):
        J.transport(a, g, np.random.default_rng(33), stage="C", prepared=dev, range_out=False)
    with pytest.raises(ValueError, match="stage C"):
        J.DeviceTransport(g, stage="A", range_out=True)
