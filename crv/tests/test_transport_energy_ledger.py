"""Below-plane energy continuity and stopping with synthetic bars and fixed loss draws."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim import transport as T
from mu2efastsim import eloss as E, materials as M, world as W


def _geometry():
    return SimpleNamespace(nbars=3, center=np.array([[0., 4., 0.], [0., 2., 0.], [0., 0., 0.]]),
                           halflen=np.tile([10., .5, 10.], (3, 1)))


def _gen(p=(100., 300.)):
    p = np.asarray(p)
    return pd.DataFrame(dict(run=1, subrun=1, event=np.arange(len(p)), idx=0, pdg=13,
                             x=0., y=6., z=0., px=0., py=-p, pz=0., t=0.))


def _p_after(p, loss):
    tk = np.maximum(np.hypot(p, E.MMU) - E.MMU - loss, 0.)
    return np.sqrt(tk * (tk + 2 * E.MMU))


def _mock_loss(monkeypatch, bar_loss=10., sheet_loss=5., air_dedx=0.):
    def sample(rng, mat, p, length, cut):
        return np.full_like(p, bar_loss if mat == M.POLYSTYRENE else sheet_loss)

    def dedx(mat, p, cut):
        return np.full_like(p, air_dedx if mat == M.AIR else 10.)

    monkeypatch.setattr(E, "sample_loss", sample)
    monkeypatch.setattr(E, "dedx_restricted", dedx)
    monkeypatch.setattr(E, "birks", lambda mat, loss, length: loss)


def _run(monkeypatch, stage="A", **kwargs):
    if stage == "C":
        monkeypatch.setattr(T, "world_loss", lambda rng, o, d, p, world, edges:
                            (p, np.zeros(len(p)), np.ones(len(p), bool), None, None))
        monkeypatch.setattr(T, "msc_to_plane", lambda rng, o, d, p, *args, **kw:
                            (o, d, np.zeros(len(p))))
    return T.transport(_gen(), _geometry(), np.random.default_rng(1), stage=stage, world=[], **kwargs)[0]


@pytest.mark.parametrize("stage", ["A", "C"])
def test_bar_exit_equals_next_entry_without_gap_loss(monkeypatch, stage):
    _mock_loss(monkeypatch)
    st = _run(monkeypatch, stage=stage)
    p0 = np.repeat([100., 300.], 3)
    bar = st.bar.to_numpy()
    pin = np.linalg.norm(st[["px0", "py0", "pz0"]], axis=1)
    np.testing.assert_allclose(pin, _p_after(p0, 10. * bar))
    np.testing.assert_allclose(st.p1, _p_after(p0, 10. * (bar + 1)))
    np.testing.assert_allclose(st.p1.to_numpy().reshape(-1, 3)[:, :-1], pin.reshape(-1, 3)[:, 1:])


@pytest.mark.parametrize("stage", ["A", "C"])
def test_air_energy_combines_with_carried_bar_energy(monkeypatch, stage):
    _mock_loss(monkeypatch, air_dedx=.25)
    st = _run(monkeypatch, stage=stage)
    p0 = np.repeat([100., 300.], 3)
    bar = st.bar.to_numpy()
    air_loss = .25 * (1.5 + 2. * bar)
    pin = np.linalg.norm(st[["px0", "py0", "pz0"]], axis=1)
    np.testing.assert_allclose(pin, _p_after(p0, air_loss + 10. * bar))
    np.testing.assert_allclose(st.p1, _p_after(p0, air_loss + 10. * (bar + 1)))


@pytest.mark.parametrize("straggling", [False, True])
def test_absorber_uses_energy_and_current_momentum(monkeypatch, straggling):
    _mock_loss(monkeypatch)
    sheet = (np.array([[-10., 3., -10.]]), np.array([[10., 3.5, 10.]]), M.ALUMINIUM)
    st = _run(monkeypatch, absorbers=sheet, absorber_straggling=straggling)
    p0 = np.repeat([100., 300.], 3)
    bar = st.bar.to_numpy()
    pin = np.linalg.norm(st[["px0", "py0", "pz0"]], axis=1)
    np.testing.assert_allclose(pin, _p_after(p0, 10. * bar + 5. * (bar > 0)))


def test_range_out_in_absorber_blocks_all_later_bars(monkeypatch):
    _mock_loss(monkeypatch, sheet_loss=100.)
    sheet = (np.array([[-10., 3., -10.]]), np.array([[10., 3.5, 10.]]), M.ALUMINIUM)
    st = _run(monkeypatch, absorbers=sheet)
    assert st.loc[st.event == 0, "bar"].tolist() == [0]
    assert st.loc[st.event == 1, "bar"].tolist() == [0, 1, 2]


def _piece(name, bottom, top):
    planes = np.array([[1, 0, 0, 10], [-1, 0, 0, 10], [0, 1, 0, top],
                       [0, -1, 0, -bottom], [0, 0, 1, 10], [0, 0, -1, 10]], float)
    return name, planes, np.array([-10., bottom, -10.]), np.array([10., top, 10.])


def test_gap_material_energy_matches_absorber_route(monkeypatch):
    _mock_loss(monkeypatch)
    seen = []

    def slab(rng, mat, p, length):
        seen.append((mat.name, p.copy()))
        return np.full_like(p, 5.)

    monkeypatch.setattr(T, "slab_loss", slab)
    below = W.PieceSet([_piece(M.ALUMINIUM.name, 3., 3.5),
                        _piece(M.CONCRETE_MARS.name, 2.5, 3.)]).build_grid(1.)
    st = _run(monkeypatch, below=below)
    p0 = np.repeat([100., 300.], 3)
    bar = st.bar.to_numpy()
    np.testing.assert_allclose(np.linalg.norm(st[["px0", "py0", "pz0"]], axis=1),
                               _p_after(p0, 10. * bar + 10. * (bar > 0)))
    np.testing.assert_allclose(seen[0][1], _p_after(np.array([100., 300.]), 10.))
    np.testing.assert_allclose(seen[1][1], _p_after(np.array([100., 300.]), 15.))


@pytest.mark.parametrize("straggling", [False, True])
def test_absorber_sampler_receives_current_momentum(monkeypatch, straggling):
    _mock_loss(monkeypatch)
    seen = []
    sample, dedx = E.sample_loss, E.dedx_restricted

    def record_sample(rng, mat, p, length, cut):
        if mat == M.ALUMINIUM:
            seen.append(p.copy())
        return sample(rng, mat, p, length, cut)

    def record_dedx(mat, p, cut):
        if mat == M.ALUMINIUM:
            seen.append(p.copy())
        return dedx(mat, p, cut)

    monkeypatch.setattr(E, "sample_loss", record_sample)
    monkeypatch.setattr(E, "dedx_restricted", record_dedx)
    sheet = (np.array([[-10., 3., -10.]]), np.array([[10., 3.5, 10.]]), M.ALUMINIUM)
    _run(monkeypatch, absorbers=sheet, absorber_straggling=straggling)
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0], _p_after(np.array([100., 300.]), 10.))


def test_gap_range_out_skips_stopped_lane_in_later_material(monkeypatch):
    _mock_loss(monkeypatch)
    seen = []

    def slab(rng, mat, p, length):
        seen.append(p.copy())
        kinetic = np.sqrt(p * p + E.MMU * E.MMU) - E.MMU
        return np.where(p < 100., kinetic, 5.)

    monkeypatch.setattr(T, "slab_loss", slab)
    below = W.PieceSet([_piece(M.ALUMINIUM.name, 3., 3.5),
                        _piece(M.CONCRETE_MARS.name, 2.5, 3.)]).build_grid(1.)
    st = _run(monkeypatch, below=below)
    assert st.loc[st.event == 0, "bar"].tolist() == [0]
    assert st.loc[st.event == 1, "bar"].tolist() == [0, 1, 2]
    assert len(seen[0]) == 2 and len(seen[1]) == 1
    assert (seen[1] > 0.).all()


def test_air_loss_false_preserves_bar_continuity(monkeypatch):
    _mock_loss(monkeypatch, air_dedx=1000.)
    st = _run(monkeypatch, air_loss=False)
    pin = np.linalg.norm(st[["px0", "py0", "pz0"]], axis=1)
    np.testing.assert_allclose(pin, _p_after(np.repeat([100., 300.], 3), 10. * st.bar.to_numpy()))


def test_stage_c_starts_from_surviving_plane_energy(monkeypatch):
    _mock_loss(monkeypatch, air_dedx=.25)

    def world_loss(rng, o, d, p, world, edges):
        pp = _p_after(p, np.array([100., 20.]))
        return pp, np.zeros(len(p)), np.array([False, True]), None, None

    def msc(rng, o, d, p, *args, **kwargs):
        end = o.copy()
        end[:, 1] = 5.
        return end, d, np.ones(len(p))

    monkeypatch.setattr(T, "world_loss", world_loss)
    monkeypatch.setattr(T, "msc_to_plane", msc)
    st = T.transport(_gen(), _geometry(), np.random.default_rng(1), stage="C", world=[])[0]
    assert st.event.tolist() == [1, 1, 1]
    bar = st.bar.to_numpy()
    np.testing.assert_allclose(np.linalg.norm(st[["px0", "py0", "pz0"]], axis=1),
                               _p_after(300., 20. + .25 * (.5 + 2. * bar) + 10. * bar))


@pytest.mark.parametrize("air_dedx,low_bars", [(30., []), (10., [0])])
def test_air_range_out_has_no_later_steps(monkeypatch, air_dedx, low_bars):
    _mock_loss(monkeypatch, air_dedx=air_dedx)
    st = _run(monkeypatch)
    assert st.loc[st.event == 0, "bar"].tolist() == low_bars
    assert st.loc[st.event == 1, "bar"].tolist() == [0, 1, 2]
    assert np.isfinite(st.select_dtypes(include="number").to_numpy()).all()
    assert (st.p1 >= 0.).all()


def test_range_out_in_bar_caps_energy_and_blocks_later_bars(monkeypatch):
    _mock_loss(monkeypatch, bar_loss=25.)
    st = _run(monkeypatch)
    low = st[st.event == 0]
    assert low.bar.tolist() == [0, 1]
    np.testing.assert_allclose(low.edep.sum(), np.hypot(100., E.MMU) - E.MMU)
    assert low.p1.iloc[-1] == 0.
    assert st.loc[st.event == 1, "bar"].tolist() == [0, 1, 2]
