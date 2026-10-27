"""Synthetic algorithm tests, not validation of the below-plane physics approximation."""
from types import SimpleNamespace
import inspect

import numpy as np
import pandas as pd
import pytest

from crvfastsim import transport as T
from mu2efastsim import eloss as E, materials as M, world as W
from mu2efastsim import transport as CT


def bars(ys=(0.,), half=(1., .5, 10.)):
    return SimpleNamespace(nbars=len(ys), center=np.c_[np.zeros(len(ys)), ys, np.zeros(len(ys))],
                           halflen=np.tile(half, (len(ys), 1)))


def gen(x=0., p=500., n=1):
    return pd.DataFrame(dict(run=1, subrun=1, event=np.arange(n), idx=0, pdg=13,
                             x=x, y=6., z=0., px=0., py=-p, pz=0., t=0.))


def slab(bottom=-20., top=6., half=30.):
    planes = np.array([[1, 0, 0, half], [-1, 0, 0, half], [0, 1, 0, top],
                       [0, -1, 0, -bottom], [0, 0, 1, half], [0, 0, -1, half]], float)
    return W.PieceSet([(M.ALUMINIUM.name, planes, np.array([-half, bottom, -half]),
                       np.array([half, top, half]))]).build_grid(10.)


def empty():
    return W.PieceSet([]).build_grid(10.)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(T, 'world_loss', lambda rng, o, d, p, world, edges:
                        (p, np.zeros(len(p)), np.ones(len(p), bool), None, None))
    monkeypatch.setattr(T, 'msc_to_plane', lambda rng, o, d, p, *a, **kw: (o, d, np.zeros(len(p))))
    monkeypatch.setattr(E, 'sample_loss', lambda rng, mat, p, length, cut: .2 * length)
    monkeypatch.setattr(E, 'dedx_restricted', lambda mat, p, cut: np.zeros_like(p))
    monkeypatch.setattr(E, 'birks', lambda mat, loss, length: loss)
    monkeypatch.setattr(T, 'slab_loss', lambda rng, mat, p, length: .01 * length)
    monkeypatch.setattr(CT, 'highland_var_increment', lambda p, x0, x1: np.zeros_like(p))


def run(g, geometry=None, below=None, **kw):
    options = dict(stage='C', world=[], below=empty() if below is None else below, below_msc=True,
                   below_scatter_in=True, scatter_in_step_mm=1., scatter_in_envelope_mm=10.)
    options.update(kw)
    return T.transport(g, bars() if geometry is None else geometry, np.random.default_rng(9), **options)


def test_near_miss_recovers_without_initial_box_or_bar(model, monkeypatch):
    calls = 0
    def kick(pos, d, dth, dy):
        nonlocal calls
        calls += 1
        if calls == 1:
            d = np.tile(np.array([-.3, -1., 0.]) / np.sqrt(1.09), (len(d), 1))
        return pos, d
    monkeypatch.setattr(T, '_kick', kick)
    diagnostic = {}
    g = gen(x=2., n=3)
    # Neither legacy admission nor the legacy bar-conditioned algorithm can recover this track.
    old = run(g, below=slab(), below_scatter_in=False)[0]
    assert old.empty
    st, parts = run(g, below=slab(), scatter_in_diagnostics=diagnostic)
    assert len(st) == len(parts) == 3
    assert st.bar.eq(0).all() and st.sp.eq(1).all()
    assert diagnostic['recovered_initial_misses'] == 3
    assert diagnostic['complete'] and not diagnostic['cap_hit']
    assert diagnostic['admitted'] == 3
    assert diagnostic['escaped'] == 3
    # The only pre-entry material is uniform passive slab: exactly travelled distance is charged,
    # not the abandoned pre-kick candidate gap or a second copy of the traversed piece.
    beta = 500. / np.sqrt(500. ** 2 + E.MMU ** 2)
    distance = st.t0.to_numpy() * beta * T.C_MM_PER_NS
    pin = np.sqrt(st.px0 ** 2 + st.py0 ** 2 + st.pz0 ** 2)
    np.testing.assert_allclose(pin, CT._p_after(np.full(3, 500.), .01 * distance), rtol=1e-12)


def test_miss_after_kick_is_not_permanently_dropped(model, monkeypatch):
    calls = 0
    def kick(pos, d, dth, dy):
        nonlocal calls
        calls += 1
        if calls == 1:
            d = np.tile(np.array([1., -1., 0.]) / np.sqrt(2.), (len(d), 1))
        elif calls == 3:
            d = np.tile(np.array([-.4, -1., 0.]) / np.sqrt(1.16), (len(d), 1))
        return pos, d
    monkeypatch.setattr(T, '_kick', kick)
    st, _ = run(gen(), below=slab())
    assert len(st) == 1
    assert calls > 3


def test_reentry_has_one_bar_ledger_and_actual_last_exit(model, monkeypatch):
    turns = 0
    def kick(pos, d, dth, dy):
        nonlocal turns
        # Reverse in the gap below the bar, then leave the upper domain through the same bar.
        if turns == 0 and (pos[:, 1] < -.6).any():
            d = d.copy()
            d[:, 1] *= -1
            turns += 1
        return pos, d
    monkeypatch.setattr(T, '_kick', kick)
    st, _ = run(gen(), below=slab())
    assert turns == 1 and len(st) == 1
    np.testing.assert_allclose(st.len, 2.)
    np.testing.assert_allclose(st.edep, .4)
    np.testing.assert_allclose(st.y0, .5)
    np.testing.assert_allclose(st.y1, .5)
    assert st.t1.iloc[0] - st.t0.iloc[0] > 2. / T.C_MM_PER_NS


def test_stop_in_bar_once_and_no_later_bar(model, monkeypatch):
    monkeypatch.setattr(E, 'sample_loss', lambda rng, mat, p, length, cut: 1000. * length)
    st, _ = run(gen(p=100.), geometry=bars((0., -3.)), scatter_in_step_mm=.2)
    assert len(st) == 1 and st.bar.iloc[0] == 0
    tk = np.sqrt(100. ** 2 + E.MMU ** 2) - E.MMU
    np.testing.assert_allclose(st.edep, tk)
    np.testing.assert_allclose(st.len, tk / 1000.)
    np.testing.assert_allclose(st.y1, st.y0 - st.len)
    np.testing.assert_allclose(st.p1, 0.)


def test_stop_in_passive_before_bar(model, monkeypatch):
    monkeypatch.setattr(T, 'slab_loss', lambda rng, mat, p, length: 1e5 * length)
    diagnostic = {}
    assert run(gen(), below=slab(), scatter_in_diagnostics=diagnostic)[0].empty
    assert diagnostic['stopped'] == 1
    assert diagnostic['segments'] == 1


def test_only_traversed_segment_material_is_charged(model, monkeypatch):
    lengths = []
    monkeypatch.setattr(T, 'slab_loss', lambda rng, mat, p, length: lengths.append(length.copy()) or .01 * length)
    # A huge old-style pre-kick gap must never be submitted as a full column/loss.
    run(gen(), geometry=bars((-100.,)), below=slab(-200., 6.), scatter_in_step_mm=2.)
    assert max(np.concatenate(lengths)) <= 2. + 1e-12


@pytest.mark.parametrize('step', [.1, 1., 20.])
def test_zero_scatter_matches_straight_deterministic_loss(model, step):
    g, geometry = gen(n=2), bars((4., 2., 0.))
    expected = run(g, geometry, below_msc=False, below_scatter_in=False)[0]
    actual = run(g, geometry, scatter_in_step_mm=step)[0]
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-11, rtol=1e-11)


def test_identity_multiple_muons(model):
    g = gen(n=2)
    g['event'] = 7
    g['idx'] = [0, 3]
    st, parts = run(g)
    assert st.sp.tolist() == [1, 2]
    assert parts.key.tolist() == [1, 2]
    assert not st.duplicated(['run', 'subrun', 'event', 'sp', 'bar']).any()


def test_fail_closed_cap(model):
    diagnostic = {}
    with pytest.raises(RuntimeError, match='no partial result'):
        run(gen(), scatter_in_max_steps=1, scatter_in_diagnostics=diagnostic)
    assert diagnostic['cap_hit'] and not diagnostic['complete']
    assert diagnostic['max_segments_per_track'] == 1


@pytest.mark.parametrize('kw', [dict(stage='A'), dict(below_msc=False), dict(scatter_in_step_mm=0),
    dict(scatter_in_step_mm=np.nan), dict(scatter_in_envelope_mm=-1), dict(scatter_in_envelope_mm=np.inf),
    dict(scatter_in_max_steps=0), dict(scatter_in_max_steps=1.5), dict(scatter_in_diagnostics=[])])
def test_invalid_options(model, kw):
    with pytest.raises(ValueError):
        run(gen(), **kw)


def test_requires_below(model):
    with pytest.raises(ValueError, match='requires'):
        T.transport(gen(), bars(), np.random.default_rng(1), stage='C', world=[], below_msc=True,
                    below_scatter_in=True)


def test_invalid_direct_input(model):
    with pytest.raises(ValueError, match='unit directions'):
        T.scatter_in_crossings(np.random.default_rng(1), np.zeros((1, 3)), np.zeros((1, 3)),
                               np.array([500.]), np.zeros(1), bars(), empty(), np.ones(1))


def test_default_and_explicit_false_are_bit_identical():
    assert inspect.signature(T.transport).parameters['below_scatter_in'].default is False
    g, geometry = gen(n=4), bars((4., 2., 0.))
    for below_msc in (False, True):
        kw = dict(stage='C', world=[], below=slab(), below_msc=below_msc)
        rng0, rng1 = np.random.default_rng(13), np.random.default_rng(13)
        a = T.transport(g, geometry, rng0, **kw)
        b = T.transport(g, geometry, rng1, below_scatter_in=False, **kw)
        for x, y in zip(a, b):
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        assert rng0.bit_generator.state == rng1.bit_generator.state


def test_wider_admission_also_processes_above_world(model, monkeypatch):
    seen = []
    def world(rng, o, d, p, *args):
        seen.extend(o[:, 0])
        return p, np.zeros(len(p)), np.ones(len(p), bool), None, None
    monkeypatch.setattr(T, 'world_loss', world)
    diagnostic = {}
    run(gen(x=2000.), scatter_in_envelope_mm=3000., scatter_in_step_mm=100., scatter_in_diagnostics=diagnostic)
    assert seen == [2000.]
    assert diagnostic['above_processed'] == diagnostic['admitted'] == 1


def test_outside_envelope_is_not_admitted(model):
    diagnostic = {}
    st, _ = run(gen(x=20.), scatter_in_diagnostics=diagnostic)
    assert st.empty and diagnostic['admitted'] == 0
    assert diagnostic['complete']


def test_endpoint_diffusion_refinement_moves_toward_slab_covariance(model, monkeypatch):
    """Step refinement diagnostic in a constant-power toy slab, not a G4 agreement test."""
    monkeypatch.setattr(CT, 'highland_var_increment', lambda p, x0, x1: .001 * (x1 - x0))
    monkeypatch.setattr(T, 'slab_loss', lambda rng, mat, p, length: np.zeros_like(p))
    n = 10000
    o = np.tile([0., 10., 0.], (n, 1))
    d = np.tile([0., -1., 0.], (n, 1))
    geometry = bars((-20.,), half=(100., .5, 100.))
    below = slab(0., 10., 100.)
    theta = np.sqrt(.001 * 10. / E.radiation_length_mm(M.ALUMINIUM))
    continuum = theta * np.sqrt(24.5 ** 2 + 10. ** 2 / 12.)
    observed = []
    for step in (10., 2.):
        diagnostic = {}
        out = T.scatter_in_crossings(np.random.default_rng(3), o, d, np.full(n, 500.), np.zeros(n),
                                     geometry, below, np.ones(1), step_mm=step, envelope_mm=10.,
                                     diagnostics=diagnostic)
        assert len(out[0]) == n and diagnostic['complete']
        sigma = out[2][:, 0].std()
        levers = 29.5 - np.arange(step, 10. + .01, step)
        expected = theta * np.sqrt(np.mean(levers ** 2))
        assert abs(sigma / expected - 1.) < .04
        observed.append(sigma)
    assert abs(observed[1] - continuum) < abs(observed[0] - continuum)


@pytest.mark.parametrize('p,x', [(0., 0.), (np.nan, 0.), (500., np.inf)])
def test_invalid_generator_state(model, p, x):
    with pytest.raises(ValueError, match='finite'):
        run(gen(x=x, p=p))
