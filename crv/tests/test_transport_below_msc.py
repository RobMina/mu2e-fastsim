"""Below-plane scattering (transport below_msc / scattered_crossings) on synthetic bars and slabs."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import transport as T
from mu2efastsim import eloss as E, materials as M, world as W
from mu2efastsim import transport as CT


def _bars(ys, half=(10., .5, 10.)):
    ys = np.asarray(ys, float)
    return SimpleNamespace(nbars=len(ys), center=np.c_[np.zeros(len(ys)), ys, np.zeros(len(ys))],
                           halflen=np.tile(half, (len(ys), 1)))


def _piece(name, bottom, top, half=10.):
    planes = np.array([[1, 0, 0, half], [-1, 0, 0, half], [0, 1, 0, top],
                       [0, -1, 0, -bottom], [0, 0, 1, half], [0, 0, -1, half]], float)
    return name, planes, np.array([-half, bottom, -half]), np.array([half, top, half])


def _gen(p, n=1, x=0.):
    p = np.repeat(np.asarray(p, float), n)
    return pd.DataFrame(dict(run=1, subrun=1, event=np.arange(len(p)), idx=0, pdg=13,
                             x=x, y=6., z=0., px=0., py=-p, pz=0., t=0.))


def _stage_c(monkeypatch, xx0=0.):
    monkeypatch.setattr(T, "world_loss", lambda rng, o, d, p, world, edges:
                        (p, np.full(len(p), xx0), np.ones(len(p), bool), None, None))
    monkeypatch.setattr(T, "msc_to_plane", lambda rng, o, d, p, *args, **kw: (o, d, np.zeros(len(p))))


def _mock_loss(monkeypatch, bar_loss=10., slab=5.):
    monkeypatch.setattr(E, "sample_loss", lambda rng, mat, p, length, cut: np.full_like(p, bar_loss))
    monkeypatch.setattr(E, "dedx_restricted", lambda mat, p, cut: np.full_like(p, 0. if mat == M.AIR else 10.))
    monkeypatch.setattr(E, "birks", lambda mat, loss, length: loss)
    monkeypatch.setattr(T, "slab_loss", lambda rng, mat, p, length: np.full_like(p, slab))


def _run(monkeypatch, geom, gen, below, below_msc):
    return T.transport(gen, geom, np.random.default_rng(7), stage="C", world=[], below=below, below_msc=below_msc)[0]


@pytest.mark.parametrize("slabs", [[], [(M.ALUMINIUM.name, 3., 3.5), (M.CONCRETE_MARS.name, 2.5, 3.)]])
def test_without_scattering_equals_straight_line(monkeypatch, slabs):
    _stage_c(monkeypatch)
    _mock_loss(monkeypatch)
    monkeypatch.setattr(CT, "highland_var_increment", lambda p, x0, x1, mass=E.MMU: np.zeros(np.shape(p)))
    geom, gen = _bars([4., 2., 0.]), _gen([100., 300.])
    below = W.PieceSet([_piece(*s) for s in slabs]).build_grid(1.)
    a = _run(monkeypatch, geom, gen, below, False).reset_index(drop=True)
    b = _run(monkeypatch, geom, gen, below, True).reset_index(drop=True)
    # a muon that ranges out in a bar ends at its residual range here (the straight-line rows keep the bar exit)
    stopped = (b.p1 == 0).to_numpy()
    assert stopped.any() == bool(slabs)
    for c in ("x1", "y1", "z1"):
        a.loc[stopped, c] = b.loc[stopped, c]
    np.testing.assert_allclose(b.y1[stopped], b.y0[stopped] - b.len[stopped])
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12)


def test_straight_option_is_default():
    import inspect
    assert inspect.signature(T.transport).parameters["below_msc"].default is False


def test_range_out_in_gap_blocks_later_bars(monkeypatch):
    _stage_c(monkeypatch)
    _mock_loss(monkeypatch, slab=1e4)
    geom = _bars([4., 0.])
    below = W.PieceSet([_piece(M.ALUMINIUM.name, 2., 3.)]).build_grid(1.)
    st = _run(monkeypatch, geom, _gen([100., 300.]), below, True)
    assert st.bar.tolist() == [0, 0]


def test_slab_scattering_matches_highland(monkeypatch):
    """A 50 mm aluminium slab 1 m above a wide bar: plane-angle RMS = Highland, displacement from the lever arm."""
    _stage_c(monkeypatch)
    monkeypatch.setattr(T, "slab_loss", lambda rng, mat, p, length: np.zeros_like(p))
    monkeypatch.setattr(E, "dedx_restricted", lambda mat, p, cut, mass=E.MMU: np.zeros_like(p))
    monkeypatch.setattr(E, "sample_loss", lambda rng, mat, p, length, cut: np.full_like(p, 1e-3))
    p, n = 1000., 20000
    geom = _bars([-1000.], half=(5000., .5, 5000.))
    below = W.PieceSet([_piece(M.ALUMINIUM.name, -50., 0., half=5000.)]).build_grid(500.)
    gen = _gen([p], n)
    gen["y"] = 1.
    st = T.transport(gen, geom, np.random.default_rng(3), stage="C", world=[], below=below, below_msc=True)[0]
    x = 50. / E.radiation_length_mm(M.ALUMINIUM)
    th0 = E.highland_theta0(np.array([p]), np.array([x]))[0]
    ang = st.px0.to_numpy() / -st.py0.to_numpy()
    assert len(st) == n
    assert abs(ang.std() / th0 - 1) < 0.03
    # slab centroid 25 mm below the origin, bar top 999.5 mm below: lever 974.5 mm plus the in-slab spread
    lever = np.sqrt(974.5 ** 2 + 50. ** 2 / 12)
    assert abs(st.x0.std() / (th0 * lever) - 1) < 0.03
    assert abs(np.corrcoef(ang, st.x0)[0, 1] - 974.5 / lever) < 0.02


def test_bar_scattering_turns_later_bars(monkeypatch):
    """Ten thick bars: the bar material alone turns the track; the first bar's entry direction is the plane one."""
    _stage_c(monkeypatch, xx0=0.)
    monkeypatch.setattr(E, "dedx_restricted", lambda mat, p, cut, mass=E.MMU: np.zeros_like(p))
    monkeypatch.setattr(E, "sample_loss", lambda rng, mat, p, length, cut: np.full_like(p, 1e-3))
    geom = _bars(-100. * np.arange(10), half=(5000., 10., 5000.))
    below = W.PieceSet([]).build_grid(1.)
    st = T.transport(_gen([500.], 5000), geom, np.random.default_rng(5), stage="C", world=[], below=below,
                     below_msc=True)[0]
    first, last = st[st.bar == 0], st[st.bar == 9]
    np.testing.assert_allclose(first.px0, 0., atol=1e-9)
    x9 = 9 * 20. / E.radiation_length_mm(M.POLYSTYRENE)
    th = E.highland_theta0(np.array([500.]), np.array([x9]))[0]
    ang = last.px0.to_numpy() / -last.py0.to_numpy()
    assert abs(ang.std() / th - 1) < 0.05
    assert (st.groupby("event").bar.apply(lambda b: (np.diff(b.to_numpy()) == 1).all())).all()


def test_below_msc_requires_stage_c():
    with pytest.raises(ValueError):
        T.transport(_gen([100.]), _bars([0.]), np.random.default_rng(1), stage="A",
                    below=W.PieceSet([]).build_grid(1.), below_msc=True)


def test_below_margin_defaults():
    from crvfastsim import runconfig as R
    assert R.RUN2_V10.below_margin == 20000.0
    for rc in (R.EXTRACTED_V04, R.RUN1A_V01, R.RUN1B_V01):
        assert rc.below_margin == 2000.0
    assert not any(rc.below_msc for rc in R.REGISTRY.values())


def test_load_below_margin_reaches_far_material(monkeypatch):
    """A dirt block 10 m outside the CRV box is below-plane material only with a margin beyond 10 m."""
    geom = _bars([0.])
    pieces = [_piece("MBOverburden", -5., -1.), _piece("MBOverburden", -5., -1.)]
    far = pieces[1]
    shift = np.array([10000., 0, 0])
    planes = far[1].copy()
    planes[:, 3] += planes[:, :3] @ shift
    pieces[1] = (far[0], planes, far[2] + shift, far[3] + shift)
    monkeypatch.setattr(W, "load", lambda path, y_min=None, names=None: list(pieces))
    assert len(T.load_below("x.gdml", geom, y_plane=1.0)) == 1
    assert len(T.load_below("x.gdml", geom, y_plane=1.0, margin=20000.0)) == 2
