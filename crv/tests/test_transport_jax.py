"""Synthetic device transport contracts; no experiment data required."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
from crvfastsim import crossings as nc, transport as nt
from crvfastsim import crossings_jax as jc, transport_jax as jt
from mu2efastsim import materials, transport as ct, eloss_jax as ej


def geom(n=4):
    return SimpleNamespace(nbars=n, center=np.c_[np.zeros(n), np.arange(n)*-30., np.zeros(n)],
                           halflen=np.tile([100., 10., 100.], (n, 1)))


def gen(n=12, p=1000.):
    return pd.DataFrame(dict(run=np.ones(n, int), subrun=np.ones(n, int), event=np.arange(n), idx=np.zeros(n, int),
                             pdg=np.full(n, 13), x=np.zeros(n), y=np.full(n, 6000.), z=np.zeros(n),
                             px=np.zeros(n), py=np.full(n, -p), pz=np.zeros(n), t=np.zeros(n)))


def test_crossings_exact_and_parallel():
    g = geom()
    o = np.array([[0., 6000., 0.], [200., 6000., 0.], [0., 0., 0.], [0., 20., 0.]])
    d = np.array([[0., -1., 0.], [0., -1., 0.], [1., 0., 0.], [0., 1., 0.]])
    ref = nc.crossings(o, d, g, backend="numpy")
    x = jax.device_get(jc.crossings(o, d, g.center-g.halflen, g.center+g.halflen))
    it, k = np.nonzero(x.valid)
    cand = np.c_[it, x.bar[it, k], x.entry[it, k], x.exit[it, k], x.exit[it, k]-x.entry[it, k]]
    cand = cand[np.lexsort((cand[:, 1], cand[:, 0]))]
    np.testing.assert_array_equal(cand[:, :2], ref[:, :2])
    np.testing.assert_allclose(cand[:, 2:], ref[:, 2:], rtol=0., atol=1e-10)
    assert not x.overflow


def test_overflow_is_not_silent():
    g = geom()
    dev = jt.DeviceTransport(g, stage="A", capacity=1, batch_size=8)
    with pytest.raises(RuntimeError, match="capacity"):
        jt.transport(gen(), g, np.random.default_rng(1), stage="A", prepared=dev)


def test_repeat_schema_multi_muon_and_padding():
    g = geom()
    a = gen(13)
    a.loc[1, "event"], a.loc[1, "idx"] = 0, 1
    dev = jt.DeviceTransport(g, stage="A", batch_size=8)
    s, p = jt.transport(a, g, np.random.default_rng(8), stage="A", prepared=dev)
    s2, p2 = jt.transport(a, g, np.random.default_rng(8), stage="A", prepared=dev)
    sn, pn = nt.transport(a, g, np.random.default_rng(8), stage="A")
    pd.testing.assert_frame_equal(s, s2)
    pd.testing.assert_frame_equal(p, p2)
    pd.testing.assert_frame_equal(p, pn)
    assert list(s) == list(sn)
    assert s.bar.dtype == sn.bar.dtype == np.dtype("int64")
    assert set(s.sp) == {1, 2}
    assert len(s) == 52
    assert np.isfinite(s.to_numpy()).all()
    assert (s.edep > 0).all() and (s.p1 >= 0).all()
    np.testing.assert_allclose(s[["len", "x0", "y0", "z0", "x1", "y1", "z1", "t0"]],
                               sn[["len", "x0", "y0", "z0", "x1", "y1", "z1", "t0"]], atol=1e-10)


def test_range_out_no_extension_and_numpy_endpoint_convention():
    g = geom(30)
    a = gen(8, p=10.)
    s, _ = jt.transport(a, g, np.random.default_rng(1), stage="A", air_loss=False, batch_size=8)
    assert (s.groupby("event").size() == 1).all()
    assert (s.p1 == 0).all()
    assert (s.len < 20).all()
    np.testing.assert_allclose(s.y1, -10.)


def test_convex_column_matches_core():
    planes = np.array([[1,0,0,2], [-1,0,0,2], [0,1,0,2], [0,-1,0,2], [0,0,1,2], [0,0,-1,2]], float)
    pieces = [("G4_AIR", planes, np.full(3, -2.), np.full(3, 2.))]
    o = np.array([[0., 10., 0.], [3., 10., 0.], [2., 10., 0.]])
    d = np.tile([0., -1., 0.], (3, 1))
    data, names = jt.pack(pieces)
    cand = jt.column(o, d, np.full(3, 20.), *data, len(names))
    from mu2efastsim.world import column
    np.testing.assert_allclose(np.asarray(cand)[:, 0], column(pieces, o, d, np.full(3, 20.))["G4_AIR"])
    with pytest.raises(ValueError, match="explicit material"):
        jt.pack([("unrecognised", planes, np.full(3, -2.), np.full(3, 2.))])


def test_convex_column_replay_with_many_same_material_pieces():
    from mu2efastsim import world as wm
    rng = np.random.default_rng(24)
    pieces = []
    for width in np.geomspace(.001, 20., 129):
        lo, hi = np.full(3, -width), np.full(3, width)
        planes = np.c_[np.r_[np.eye(3), -np.eye(3)], np.full(6, width)]
        pieces.append(("G4_AIR", planes, lo, hi))
    data, names = jt.pack(pieces)
    o = np.c_[rng.uniform(-.1, .1, 257), np.full(257, 30.), rng.uniform(-.1, .1, 257)]
    d = np.tile([0., -1., 0.], (len(o), 1))
    length = np.full(len(o), 60.)
    ref = wm.column(pieces, o, d, length)["G4_AIR"]
    first = np.asarray(jt.column(o, d, length, *data, len(names)))
    for _ in range(5):
        np.testing.assert_array_equal(jt.column(o, d, length, *data, len(names)), first)
    np.testing.assert_allclose(first[:, 0], ref, rtol=2e-14, atol=2e-12)


def test_empty_population_and_unsupported():
    g = geom()
    a = gen(0)
    s, p = jt.transport(a, g, np.random.default_rng(1), stage="A")
    assert len(s) == len(p) == 0
    with pytest.raises(ValueError, match="straight-line"):
        nt.transport(gen(), g, np.random.default_rng(1), transport_backend="jax", below_msc=True)
    with pytest.raises(ValueError, match="unknown transport"):
        nt.transport(gen(), g, np.random.default_rng(1), transport_backend="other")


def test_stage_c_bypass_population_and_empty_world():
    g = geom()
    a = gen(8)
    a.loc[0, ["x", "px", "py"]] = [0., 10., 1000.]
    a.loc[1, ["y", "px", "py"]] = [-200., 0., 1000.]
    dev = jt.DeviceTransport(g, [], stage="C", batch_size=8)
    s, p = jt.transport(a, g, np.random.default_rng(4), stage="C", prepared=dev)
    assert 0 not in set(s.event)
    assert 1 in set(s.event)
    assert np.isfinite(s.to_numpy()).all()
    assert len(p) == len(a)


def test_slab_loss_distribution():
    n = 10000
    p, length = np.full(n, 500.), np.full(n, 100.)
    ref = ct.slab_loss(np.random.default_rng(9), materials.CONCRETE_MARS, p, length)
    cand = np.asarray(jt.slab_loss(jax.random.key(9), ej.consts(materials.CONCRETE_MARS), p, length))
    se = np.sqrt(ref.var()/n + cand.var()/n)
    assert abs(ref.mean()-cand.mean()) < 5*se
    assert np.isfinite(cand).all() and (cand >= 0).all()
