"""Unit tests: conversion-electron chain to the tracker entry plane (beamfastsim.cechain) in a uniform field."""
from dataclasses import replace

import numpy as np
import pytest

from beamfastsim import cechain as C
from beamfastsim import geometry as G

AXIS = -3904.0


def _geom(foil_z=(5800.0, 5822.0, 5844.0), ipa=(300.0, 0.5, 6900.0, 7900.0), opa=(454.0, 728.4, 20.0)):
    n = len(foil_z)
    beam = G.BeamGeometry(AXIS, np.array(foil_z), np.full(n, 21.5), np.full(n, 75.0), np.full(n, 0.0528),
                          4204.6, 25.4, 235.0, 525.0, ipa[0], ipa[1], ipa[2], ipa[3], opa[0], 4280.0, 8530.0)
    return C.CEGeometry(beam, opa[0], opa[1], opa[2], z_plane=8539.9)


def _starts(n, rng, z=5700.0, r=40.0, cos=(0.5, 0.8)):
    ct = rng.uniform(*cos, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - ct ** 2)
    u = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    x = np.tile([AXIS + r, 0.0, z], (n, 1))
    return x, u


def test_vacuum_helix_reaches_the_plane_with_unchanged_momentum_and_light_speed_time():
    rng = np.random.default_rng(1)
    g = _geom(foil_z=(5000.0,), ipa=(300.0, 0.5, 9000.0, 9001.0), opa=(900.0, 900.0, 1.0))
    x, u = _starts(500, rng, r=10.0, cos=(0.6, 0.8))
    res = C.transport(rng, g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0)
    assert (res.fate == C.PLANE).all()
    assert np.allclose(res.x[:, 2], g.z_plane, atol=1e-6)
    assert np.allclose(res.u[:, 2], u[:, 2], atol=1e-9)          # pitch is conserved in a uniform axial field
    assert np.allclose(res.path, (g.z_plane - 5700.0) / u[:, 2], rtol=1e-7)
    beta = C._beta(res.T)
    assert np.allclose(res.t, res.path / (C.C_MM_NS * beta), rtol=1e-9)
    rmax, _ = C.helix_rmax(g, x, u, C.P_CE_AL, 1.0)
    rplane = C.radial(g, res.x)
    assert (rplane <= rmax + 1e-6).all()


@pytest.mark.slow   # NumPy msc chain through foils: minutes at any sample size
def test_birth_in_a_foil_loses_energy_and_counts_the_crossing():
    rng = np.random.default_rng(2)
    g = _geom()
    n = 800
    x, u = _starts(n, rng, z=5800.0, r=40.0)
    res = C.transport(rng, g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0)
    T0 = np.hypot(C.P_CE_AL, 0.511) - 0.511
    assert (res.nfoil >= 1).all()
    reached = res.fate == C.PLANE
    assert reached.mean() > 0.95
    loss = T0 - res.T[reached]
    assert 0.0 <= loss.min() and np.median(loss) < 1.0
    assert set(np.unique(res.fate)) <= set(range(len(C.FATES)))


def test_foil_births_without_msc_cross_foils_and_shells_to_the_plane():
    # the per-PR material path of the NumPy chain (the msc version above runs with --runslow): births inside the
    # foils, IPA and OPA crossings; msc off keeps the slab stepping short
    rng = np.random.default_rng(9)
    g = replace(_geom(foil_z=tuple(5800.0 + 22.0 * np.arange(5)), ipa=(300.0, 0.5, 6000.0, 7000.0)),
                ring_t=9.525, ring_hl=6.35)
    n = 6
    k = rng.integers(0, 5, n)
    r = np.sqrt(rng.uniform(21.5 ** 2, 75.0 ** 2, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    x = np.stack([AXIS + r * np.cos(ph), r * np.sin(ph), g.beam.foil_z[k] + rng.uniform(-0.05, 0.05, n)], axis=1)
    res = C.transport(rng, g, [0.0, 0.0, 1.0], x, C.isotropic(rng, n), C.P_CE_AL, 0.0, msc=False)
    T0 = np.hypot(C.P_CE_AL, 0.511) - 0.511
    assert (res.nfoil >= 1).all() and res.nipa.sum() > 0 and res.nopa.sum() > 0
    assert (res.fate == C.PLANE).all()
    assert np.allclose(res.x[:, 2], g.z_plane, atol=1e-6)
    assert (res.T < T0).all() and (res.T > 0).all()
    assert (res.t >= res.path / C.C_MM_NS - 1e-6).all()


def test_helix_rmax_follows_the_charge_sign():
    g = _geom()
    x = np.array([[AXIS + 100.0, 0.0, 6000.0]])
    u = np.array([[1.0, 0.0, 0.0]])
    rmax, R = C.helix_rmax(g, x, u, 100.0, 1.0)
    assert R[0] == pytest.approx(100.0 / (0.299792458 * 1.0))
    assert rmax[0] == pytest.approx(np.hypot(100.0, R[0]) + R[0])


def test_upstream_and_wall_losses_end_the_track():
    rng = np.random.default_rng(3)
    g = _geom(foil_z=(5000.0,), ipa=(300.0, 0.5, 9000.0, 9001.0), opa=(900.0, 900.0, 1.0))
    x = np.tile([AXIS + 10.0, 0.0, 5700.0], (2, 1))
    u = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    res = C.transport(rng, g, [0.0, 0.0, 0.0], x, u, C.P_CE_AL, 0.0)     # no field: straight lines
    assert list(res.fate) == [C.UPSTREAM, C.WALL]
    assert res.path[0] < 5700.0 - g.z_up + 20.0 and res.path[1] < g.r_wall + 20.0


def test_recording_plane_keeps_the_vacuum_state_at_its_first_crossing():
    rng = np.random.default_rng(4)
    g = _geom(foil_z=(5000.0,), ipa=(300.0, 0.5, 9000.0, 9001.0), opa=(900.0, 900.0, 1.0))
    x, u = _starts(200, rng, r=10.0, cos=(0.6, 0.8))
    res = C.transport(rng, g, [0.0, 0.0, 1.0], x, u, C.P_CE_AL, 0.0, z_record=6276.0)
    assert res.rec_ok.all() and np.allclose(res.rec_x[:, 2], 6276.0, atol=1e-6)
    assert np.allclose(res.rec_t, (6276.0 - 5700.0) / u[:, 2] / (C.C_MM_NS * C._beta(res.rec_T)), rtol=1e-7)
    assert np.allclose(res.rec_T, res.T)


def test_shell_curvature_term_gives_the_straight_chord_through_a_thin_cylinder():
    # no field: a straight line leaving a cylinder of radius R at a grazing angle crosses a wall of thickness t
    # along sqrt((R + t)^2 - b^2) - sqrt(R^2 - b^2), b = R cos(alpha); the flat slab alone would give t / sin(alpha)
    from mu2efastsim import ehelix
    g = _geom()
    R, t, n = 300.0, 0.5, 200
    alpha = np.full(n, 0.01)
    x = np.tile([AXIS + R, 0.0, 7000.0], (n, 1))
    u = np.stack([np.sin(alpha), np.cos(alpha), np.zeros(n)], axis=1)
    e1, e2, nn = C._frame_radial(g, x, np.ones(n, bool))
    p = np.full(n, C.P_CE_AL)
    T = np.hypot(p, 0.51099895) - 0.51099895
    dBz = (u * e2).sum(1) * p / (ehelix.K * R)
    B = np.zeros((n, 3))
    B[:, 2] = dBz
    _, _, _, _, path = C._cross_layer(np.random.default_rng(5), G.IPA_PE, T, x, u, e1, e2, nn, t, B, None, msc=False)
    b = R * np.cos(alpha)
    exact = np.sqrt((R + t) ** 2 - b ** 2) - np.sqrt(R ** 2 - b ** 2)
    assert np.median(path) == pytest.approx(exact[0], rel=0.03)
    assert exact[0] < 0.5 * t / np.sin(alpha[0])
