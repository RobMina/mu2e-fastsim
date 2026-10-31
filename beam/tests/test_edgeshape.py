"""All-entry tail bookkeeping, influence uncertainties and chunked step diagnostics."""
import json
from pathlib import Path

import numpy as np
import pytest

from beamfastsim import edgeshape as es, foilreport as fr
from test_edgeangles import Chunks
from test_foilclosure import fixture


def test_variance_partition_closure_including_empty_groups():
    y = np.array([0., 0., .2, .3, .9, 1.4])
    masks = np.column_stack([y < .1, (y >= .1) & (y < 1.), y >= 1., y < 0.])
    result = es.variance_parts(y, masks, ['zero', 'bulk', 'tail', 'empty'])
    assert result['closure'] == pytest.approx(0., abs=1e-15)
    assert sum(x['count'] for x in result['components'].values()) == len(y)
    assert result['components']['empty'] == dict(count=0, contribution=0., se=0.)
    influence = (y-y.mean())**2-y.var()
    assert np.sum(result['covariance']) == pytest.approx(influence.var(ddof=1)/len(y))
    json.dumps(result, allow_nan=False)


def test_partition_influence_matches_weight_derivative():
    y = np.array([0., .1, .4, .5, 1.])
    masks = np.column_stack([y < .4, y >= .4])
    result = es.variance_parts(y, masks, ['low', 'high'])
    w = np.full(len(y), 1/len(y))
    derivatives = []
    for row in range(len(y)):
        perturb = np.eye(len(y))[row]-w
        h = 1e-5
        a, b = w+h*perturb, w-h*perturb
        va = (masks*(y-a@y)[:, None]**2*a[:, None]).sum(axis=0)
        vb = (masks*(y-b@y)[:, None]**2*b[:, None]).sum(axis=0)
        derivatives.append((va-vb)/(2*h))
    expected = np.cov(np.asarray(derivatives), rowvar=False, ddof=1)/len(y)
    np.testing.assert_allclose(result['covariance'], expected, rtol=1e-8, atol=1e-12)


def test_partition_must_be_exhaustive_and_exclusive():
    for mask in [np.zeros((3, 2), bool), np.ones((3, 2), bool)]:
        with pytest.raises(ValueError, match='partition'):
            es.variance_parts([0., .1, .2], mask, ['a', 'b'])


def test_caps_are_diagnostic_and_retain_entries():
    y = np.array([0., 0., .2, .3, .4, 1.5])
    result = es.tails(y, np.array([0, 0, 1, 1, 1, 1], bool), np.array([0, 0, 1, 1, 1, 0], bool))
    assert result['caps']['0.5']['capped_population_variance'] == pytest.approx(np.minimum(y, .5).var())
    assert result['caps']['0.5']['removed_population_variance'] == pytest.approx(y.var()-np.minimum(y, .5).var())
    assert result['caps']['0.5']['affected']['n'] == len(y)
    assert result['energy_variance_parts']['components']['1_and_above']['count'] == 1
    assert result['topology']['components']['back_exit']['count'] == 1
    assert result['back_exit_probability']['count'] == 1
    assert es.probability(np.zeros(8000, bool))['interval_99pct'][1] > 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('data', [[1.], [0., np.nan], [[1., 2.]]])
def test_vectors_refuse_unidentified_or_invalid(data):
    with pytest.raises(ValueError, match='finite'):
        es.tails(data)


def test_correlation_uncertainty_and_constant_vector():
    x = np.arange(10.)
    result = es.correlation(x, x)
    assert result['value'] == pytest.approx(1.)
    assert result['se'] < 1e-15
    assert es.correlation(x, np.ones(10)) is None
    with pytest.raises(ValueError, match='paired'):
        es.correlation(x, x[:4])


def test_angle_cohorts_are_explicitly_outcome_conditioned():
    direction = np.array([[0., 0., 1.], [1., 0., 0.], [np.nan, np.nan, np.nan], [0., 0., 1.]])
    result = es.exit_statistics(np.array([.3, .4, 0., .05]), [True, True, False, True],
                               [True, False, False, True], direction, np.array([1., .7, .8, 1.]), 1.)
    assert result['all_exits']['n'] == 3
    assert result['above_exits']['mean_deflection_cosine']['mean'] == .5
    assert result['above_forward_exits']['n'] == 1
    assert sum(map(sum, result['joint_counts'])) == 2
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('chunk', [1, 2, 3, 7, 100])
def test_adjacent_step_features_cross_chunks_and_ignore_stoppers_zero_vector(chunk):
    p, s, case = fixture()
    s['px'][2], s['pz'][2], s['code'][2] = s['pz'][2], 0., 29
    s['code'][1] = 59
    _, ref = fr.census(p, s, case)
    out = es.step_features(Chunks(s), ref, step_size=chunk)
    np.testing.assert_array_equal(out['foil_steps'], [2, 2])
    np.testing.assert_array_equal(out['msc_ended_steps'], [1, 0])
    np.testing.assert_array_equal(out['coulomb_ended_steps'], [1, 0])
    np.testing.assert_array_equal(out['jumps_over_0p6'], [2, 0])
    np.testing.assert_allclose(out['max_jump_rad'], [np.pi/2, 0.])
    np.testing.assert_allclose(out['sum_one_minus_cos'], [2., 0.])


def test_step_feature_first_exit_is_not_reentry(tmp_path):
    p, s, case = fixture()
    _, ref = fr.census(p, s, case)
    s = {k:np.insert(v, 4, v[2]) for k, v in s.items()}
    s['pz'][4], s['px'][4] = 0., 10.
    out = es.step_features(Chunks(s), ref, step_size=2)
    np.testing.assert_array_equal(out['foil_steps'], [2, 2])
    np.testing.assert_array_equal(out['jumps_over_0p6'], [0, 0])


def test_step_feature_refuses_changed_census():
    p, s, case = fixture()
    _, ref = fr.census(p, s, case)
    ref['foil_steps'][0] += 1
    with pytest.raises(ValueError, match='census'):
        es.step_features(Chunks(s), ref, step_size=2)


def test_energy_standardization_exact_symmetric_composition_shape():
    energy = np.tile([.15, .25, .4, .8], 10)
    a = np.tile([.5, .4, .3, .2], 10)
    b = a-.02
    result = es.standardize_angle(energy, a, energy, b)
    assert result['available']
    assert result['composition']['delta'] == pytest.approx(0., abs=1e-14)
    assert result['conditional_shape']['delta'] == pytest.approx(.02)
    assert result['closure'] == pytest.approx(0., abs=1e-14)
    counts = [5, 10, 15, 20]
    e = np.repeat([.15, .25, .4, .8], counts)
    angle = np.repeat([.5, .4, .3, .2], counts)
    result = es.standardize_angle(e, angle, energy, a)
    assert result['conditional_shape']['delta'] == pytest.approx(0., abs=1e-14)
    assert result['composition']['delta'] == pytest.approx(angle.mean()-a.mean())
    assert result['closure'] == pytest.approx(0., abs=1e-14)
    assert not es.standardize_angle(energy[:2], a[:2], energy, b)['available']


def test_standardization_covariance_sums_to_independent_mean_error():
    e = np.tile([.15, .25, .4, .8], 10)
    a = np.linspace(.1, .6, len(e))
    b = np.linspace(.1, .5, len(e))**2
    result = es.standardize_angle(e, a, e, b)
    covariance = np.asarray(result['covariance'])
    assert covariance.sum() == pytest.approx(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))


def test_median_order_interval_and_momentum_chord_metrics():
    result = es.quantile_interval(np.arange(100.))
    assert result['lower'] < 49.5 < result['upper']
    small = es.quantile_interval([1., 2.])
    assert small['lower'] is None and small['upper'] is None
    p, s, case = fixture()
    _, ref = fr.census(p, s, case)
    result = es.step_features(Chunks(s), ref, step_size=1)
    np.testing.assert_array_equal(result['transverse_chord_sum_mm'], [0., 0.])
    np.testing.assert_allclose(result['adjacent_chord_sum_mm'], ref['chord_sum_mm'])
