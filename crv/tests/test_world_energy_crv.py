"""CRV default world-loss energy ledger and banded material-order regressions."""
import numpy as np
import pytest

from crvfastsim import transport as T
from mu2efastsim import eloss, materials


def _piece(name, bottom, top):
    planes = np.array([[1, 0, 0, 10], [-1, 0, 0, 10], [0, 1, 0, top],
                       [0, -1, 0, -bottom], [0, 0, 1, 10], [0, 0, -1, 10]], float)
    return name, planes, np.array([-10., bottom, -10.]), np.array([10., top, 10.])


def _remaining_p(p, loss):
    kinetic = np.maximum(np.hypot(p, eloss.MMU) - eloss.MMU - loss, 0.)
    return np.sqrt(kinetic * (kinetic + 2 * eloss.MMU))


@pytest.mark.parametrize("banded", [False, True])
@pytest.mark.parametrize("edges", [[3., 0.], [3., 2., 1., 0.]])
def test_crv_world_energy_arguments(monkeypatch, banded, edges):
    p = np.array([100., 300., 3000.])
    pieces = [_piece(materials.ALUMINIUM.name, 2., 3.),
              _piece(materials.CONCRETE_MARS.name, 1., 2.)]
    world = T.banded_world(pieces, edges) if banded else pieces
    seen, air = [], []

    def slab(rng, mat, momentum, length):
        seen.append((mat.name, momentum.copy()))
        return np.full_like(momentum, 20. if mat == materials.ALUMINIUM else 5.)

    def dedx(mat, momentum, cut):
        air.append(momentum.copy())
        return np.full_like(momentum, 2.)

    monkeypatch.setattr(T, "slab_loss", slab)
    monkeypatch.setattr(eloss, "dedx_restricted", dedx)
    out = T.world_loss(np.random.default_rng(1), np.tile([0., 3., 0.], (len(p), 1)),
                       np.tile([0., -1., 0.], (len(p), 1)), p, world, edges)
    assert [name for name, _ in seen] == [materials.ALUMINIUM.name, materials.CONCRETE_MARS.name]
    np.testing.assert_allclose(seen[0][1], p)
    np.testing.assert_allclose(seen[1][1], _remaining_p(p, 20.))
    np.testing.assert_allclose(air[-1], _remaining_p(p, 25.))
    np.testing.assert_allclose(out[0], _remaining_p(p, 27.))
    assert out[2].all()


def test_crv_world_air_overloss_returns_zero_momentum(monkeypatch):
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, 1000.))
    out = T.world_loss(np.random.default_rng(1), np.array([[0., 1., 0.]]),
                       np.array([[0., -1., 0.]]), np.array([100.]), [], [1., 0.])
    np.testing.assert_array_equal(out[0], 0.)
    assert not out[2].any()


def test_crv_world_banded_matches_piece_list():
    pieces = [_piece(materials.ALUMINIUM.name, 2., 3.),
              _piece(materials.CONCRETE_MARS.name, 1., 2.)]
    p = np.array([30., 100., 300., 3000.])
    o, d = np.tile([0., 3., 0.], (len(p), 1)), np.tile([0., -1., 0.], (len(p), 1))
    edges = [3., 0.]
    a = T.world_loss(np.random.default_rng(5), o, d, p, pieces, edges)
    b = T.world_loss(np.random.default_rng(5), o, d, p, T.banded_world(pieces, edges), edges)
    for va, vb in zip(a, b):
        np.testing.assert_array_equal(va, vb)


def test_crv_world_stopped_lane_stays_stopped(monkeypatch):
    p = np.array([100., 300.])
    pieces = [_piece(materials.ALUMINIUM.name, 1., 2.),
              _piece(materials.CONCRETE_MARS.name, 0., 1.)]
    seen = []

    def slab(rng, mat, momentum, length):
        seen.append(momentum.copy())
        return np.hypot(momentum, eloss.MMU) - eloss.MMU

    monkeypatch.setattr(T, "slab_loss", slab)
    out = T.world_loss(np.random.default_rng(1), np.tile([0., 2., 0.], (2, 1)),
                       np.tile([0., -1., 0.], (2, 1)), p, pieces, [2., 0.])
    np.testing.assert_allclose(seen[1], 0., atol=1e-6)
    np.testing.assert_array_equal(out[0], 0.)
    assert not out[2].any()


@pytest.mark.parametrize("transport", [T, T.core_tr])
def test_world_air_before_solid_and_mixed_survival(monkeypatch, transport):
    p = np.array([100., 300., 3000.])
    o, d = np.tile([0., 3., 0.], (3, 1)), np.tile([0., -1., 0.], (3, 1))
    o[2, 0] = 20.
    seen = []

    def slab(rng, mat, momentum, length):
        seen.append(momentum.copy())
        kinetic = np.sqrt(momentum ** 2 + eloss.MMU ** 2) - eloss.MMU
        return np.where(momentum < 100., kinetic, 5.)

    monkeypatch.setattr(transport, "slab_loss", slab)
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.full_like(p, 2.))
    world = [_piece(materials.ALUMINIUM.name, 1., 2.)]
    out = transport.world_loss(np.random.default_rng(1), o, d, p, world, [3., 2., 1., 0.])
    np.testing.assert_allclose(seen[0], _remaining_p(p[:2], 2.))
    np.testing.assert_allclose(out[0], [0., _remaining_p(p[1], 9.), _remaining_p(p[2], 6.)])
    np.testing.assert_array_equal(out[2], [False, True, True])


@pytest.mark.parametrize("transport", [T, T.core_tr])
def test_world_survival_threshold_is_not_a_momentum_floor(monkeypatch, transport):
    kinetic = np.array([.49, .5, .51])
    p = np.sqrt(kinetic * (kinetic + 2 * eloss.MMU))
    monkeypatch.setattr(eloss, "dedx_restricted", lambda mat, p, cut: np.zeros_like(p))
    out = transport.world_loss(np.random.default_rng(1), np.tile([0., 1., 0.], (3, 1)),
                               np.tile([0., -1., 0.], (3, 1)), p, [], [1., 0.])
    np.testing.assert_allclose(out[0], p)
    np.testing.assert_array_equal(out[2], [False, False, True])


def test_crv_world_empty_input():
    out = T.world_loss(np.random.default_rng(1), np.empty((0, 3)), np.empty((0, 3)),
                       np.empty(0), [], [1., 0.])
    assert all(value.shape[0] == 0 for value in out)
