"""Clustered local angular-budget reductions; no G4, table export or transport."""
import numpy as np
import pytest

from beamfastsim import edgemoments as em, foilreport as fr
from test_edgeangles import Chunks
from test_foilclosure import fixture


@pytest.mark.parametrize('chunk', [1, 2, 3, 100])
def test_moment_reduction_cross_chunks_and_post_threshold_selection(chunk):
    p, s, case = fixture()
    s['px'][2], s['pz'][2] = s['pz'][2], 0.
    _, ref = fr.census(p, s, case)
    out = em.reduce_steps(Chunks(s), ref, np.array([.1, 3.]), np.ones(2), step_size=chunk)
    np.testing.assert_array_equal(out['pairs'].sum(axis=1), [2., 0.])
    np.testing.assert_allclose(out['sum_one_minus_cos'].sum(axis=1), [2., 0.])
    np.testing.assert_allclose(out['budget_left'].sum(axis=1), [.2, 0.])
    np.testing.assert_allclose(out['budget_trapezoid'].sum(axis=1), [.2, 0.])
    np.testing.assert_allclose(out['sum_theta4'].sum(axis=1), [2*(np.pi/2)**4, 0.])
    np.testing.assert_array_equal(out['large_jumps_0p6'].sum(axis=1), [2., 0.])
    np.testing.assert_allclose(out['moment_large_jumps_0p6'].sum(axis=1), [2., 0.])


def test_clustered_ratio_not_iid_step_error():
    x, y = np.array([0., 2., 3., 1.]), np.array([.1, .2, .1, .3])
    result = em.ratio(x, y)
    value = x.sum()/y.sum()
    influence = (x-value*y)/y.mean()
    assert result['value'] == pytest.approx(value)
    assert result['se'] == pytest.approx(influence.std(ddof=1)/2)
    assert em.ratio(x, y*0) is None
    with pytest.raises(ValueError, match='per-primary'):
        em.ratio(x[:1], y[:1])


def test_no_scattering_has_zero_moments_and_expected_chord_budget():
    p, s, case = fixture()
    _, ref = fr.census(p, s, case)
    out = em.reduce_steps(Chunks(s), ref, np.array([.1, 3.]), np.full(2, 2.), step_size=2)
    assert not out['sum_one_minus_cos'].any()
    np.testing.assert_allclose(out['budget_left'].sum(axis=1), [.1, 0.])


def test_moment_reduction_refuses_bad_census_and_extrapolation():
    p, s, case = fixture()
    _, ref = fr.census(p, s, case)
    with pytest.raises(ValueError, match='domain'):
        em.reduce_steps(Chunks(s), ref, np.array([.1, 1.]), np.ones(2), step_size=2)
    ref['exit'][0] = False
    with pytest.raises(ValueError, match='census'):
        em.reduce_steps(Chunks(s), ref, np.array([.1, 3.]), np.ones(2), step_size=2)


def test_small_q_tail_screen_excludes_large_budget_not_large_angle():
    p, s, case = fixture()
    s['px'][2], s['pz'][2] = s['pz'][2], 0.
    _, ref = fr.census(p, s, case)
    out = em.reduce_steps(Chunks(s), ref, np.array([.1, 3.]), np.full(2, 1000.), step_size=2)
    assert out['small_q_pairs'].sum() == 2
    assert out['small_q_large_jumps_0p6'].sum() == 2
    assert out['small_q_gaussian_large_jumps_0p6'].sum() < 1e-100
    np.testing.assert_allclose(out['small_q_theta4'].sum(), out['sum_theta4'].sum())
    out = em.reduce_steps(Chunks(s), ref, np.array([.1, 3.]), np.full(2, .1), step_size=2)
    assert out['small_q_pairs'].sum() == 0
    assert out['small_q_large_jumps_0p6'].sum() == 0
