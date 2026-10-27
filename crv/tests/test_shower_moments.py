"""Exact shower moments, globally coupled file bootstrap, and paired-event sufficient statistics."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from valhist.shower_moments import (
    ProbabilityOverflowError,
    bootstrap_probability_guard,
    conditional_moments,
    file_bootstrap,
    paired_covariance,
)


def enumerate_primaries(outcomes):
    draws = np.array([np.sum(draw, axis=0) for draw in product(*outcomes)])
    centered = draws - draws.mean(axis=0)
    return draws.mean(axis=0), centered.T @ centered / len(draws)


def test_independent_zero_template_outcomes_and_full_covariance():
    features = sparse.csr_matrix([[1, 2, 0], [3, 0, 4], [0, 1, 2]])
    result = conditional_moments(features, [0, 0, 1], [4, 3, 0], [2, 1, 7])
    cell0 = [[1, 2, 0], [3, 0, 4], [0, 0, 0], [0, 0, 0]]
    cell1 = [[0, 1, 2], [0, 0, 0], [0, 0, 0]]
    mean, cov = enumerate_primaries([cell0, cell0, cell1])
    assert_allclose(result['mean_per_cell'], [[1, .5, 1], [0, 1 / 3, 2 / 3], [0, 0, 0]])
    assert_allclose(result['total_mean'], mean)
    assert_allclose(result['sampling_cov'], cov, atol=8 * np.finfo(float).eps)
    assert result['sampling_cov'][0, 2] != 0
    assert result['sampling_cov'][1, 2] < 0


def test_double_primary_variance_is_twice_not_four_times():
    features = sparse.csr_matrix([[2, 3]])
    one = conditional_moments(features, [0], [2], [1])
    two = conditional_moments(features, [0], [2], [2])
    assert_allclose(two['total_mean'], 2 * one['total_mean'])
    assert_allclose(two['sampling_cov'], 2 * one['sampling_cov'])
    expected_mean, expected_cov = enumerate_primaries([[[0, 0], [2, 3]]] * 2)
    assert_allclose(two['total_mean'], expected_mean)
    assert_allclose(two['sampling_cov'], expected_cov)


@pytest.mark.parametrize('features,cell,ngen,neval', [
    (sparse.csr_matrix((0, 3)), [], [0, 5], [100, 100]),
    (sparse.csr_matrix((3, 3)), [0, 0, 1], [2, 1], [100, 100]),
    (sparse.csr_array((0, 3)), [], [], []),
    (sparse.csr_matrix([[0., 0., 0.]]), [0], [1], [0]),
])
def test_sparse_empty_null_and_all_zero(features, cell, ngen, neval):
    result = conditional_moments(features, cell, ngen, neval)
    assert_array_equal(result['mean_per_cell'], np.zeros((len(ngen), 3)))
    assert_array_equal(result['total_mean'], np.zeros(3))
    assert_array_equal(result['sampling_cov'], np.zeros((3, 3)))


def test_integer_persisted_denominator_is_not_reestimated():
    result = conditional_moments(sparse.csr_matrix([[6]]), [0], [3.0], [2])
    assert_allclose(result['total_mean'], [4])
    assert_allclose(result['sampling_cov'], [[16]])
    with pytest.raises(ValueError, match='integer-valued'):
        conditional_moments(sparse.csr_matrix([[6]]), [0], [3.2], [2])


@pytest.mark.parametrize('change', [
    {'features': np.array([[1., 2.]])},
    {'features': sparse.csc_matrix([[1., 2.]])},
    {'features': sparse.csr_matrix([[-1., 2.]])},
    {'features': sparse.csr_matrix([[np.nan, 2.]])},
    {'features': sparse.csr_matrix([[np.inf, 2.]])},
    {'features': sparse.csr_matrix([[1j, 2.]])},
    {'features': sparse.csr_matrix((2, 2))},
    {'cell': [1]},
    {'cell': [-1]},
    {'cell': [0.]},
    {'cell': []},
    {'ngen': [0]},
    {'ngen': [-1]},
    {'ngen': [np.nan]},
    {'ngen': [np.inf]},
    {'ngen': [[2]]},
    {'neval': [-1]},
    {'neval': [1.5]},
    {'neval': [np.inf]},
    {'neval': []},
])
def test_malformed_moment_support(change):
    args = dict(features=sparse.csr_matrix([[1., 2.]]), cell=[0], ngen=[2], neval=[3])
    args.update(change)
    with pytest.raises(ValueError):
        conditional_moments(**args)


def test_more_templates_than_ngen_rejected_even_if_features_are_zero():
    with pytest.raises(ValueError, match='template count'):
        conditional_moments(sparse.csr_matrix((3, 2)), [0, 0, 0], [2], [0])


def test_exact_multinomial_covariance_and_sample_summaries():
    contributions = np.array([[0., 0.], [1., 4.], [5., 2.]])
    result = file_bootstrap(contributions, [1, 2], [0, 0], [10], replicas=12000, seed=17)
    exact_draws = np.array([contributions[list(draw)].sum(axis=0) for draw in product(range(3), repeat=3)])
    assert_allclose(result['nominal'], [6, 6])
    assert_allclose(result['bootstrap_cov'], np.cov(exact_draws, rowvar=False, ddof=0))
    assert_allclose(result['bootstrap_cov'], 2 * np.cov(contributions, rowvar=False, ddof=1))
    weights = np.random.default_rng(17).multinomial(3, np.full(3, 1 / 3), size=12000)
    expected = weights @ contributions
    assert_array_equal(result['replica_totals'], expected)
    assert_array_equal(result['sampled_mean'], expected.mean(axis=0))
    assert_array_equal(result['quantile_levels'], [.025, .16, .5, .84, .975])
    assert_allclose(result['quantiles'], np.quantile(expected, result['quantile_levels'], axis=0))
    assert_allclose(np.cov(expected, rowvar=False, ddof=0), result['bootstrap_cov'], atol=.4)
    assert result['probability_guard']['invalid_replicas'] == 0


def test_file_bootstrap_matches_duplicating_all_templates_with_fixed_denominator():
    features = sparse.csr_matrix([[2., 0.], [0., 4.], [3., 5.], [1., 0.]])
    files, cell = np.array([0, 0, 1, 2]), np.array([0, 1, 0, 1])
    ngen, neval = np.array([12, 12]), np.array([9, 6])
    incidence = sparse.csr_matrix((np.ones(4), (files, np.arange(4))), shape=(3, 4))
    contributions = (incidence @ features.multiply((neval / ngen)[cell, None])).toarray()
    result = file_bootstrap(contributions, files, cell, ngen, replicas=50, seed=31)
    weights = np.random.default_rng(31).multinomial(3, np.full(3, 1 / 3), size=50)
    for weight, total in zip(weights, result['replica_totals']):
        repeated = np.repeat(np.arange(4), weight[files])
        actual = conditional_moments(features[repeated], cell[repeated], ngen, neval)
        assert_allclose(total, actual['total_mean'])
    assert_allclose(result['nominal'], conditional_moments(features, cell, ngen, neval)['total_mean'])


def test_upper_bound_certification_and_strict_overflow_report():
    weights = np.array([[2, 0], [1, 1], [0, 2]])
    legal = bootstrap_probability_guard([0, 0, 1], [0, 0, 1], [4, 2, 0], weights)
    assert_array_equal(legal['upper_bound_per_cell'], [1, 1, 0])
    assert legal['certified_all_replicas']
    report = bootstrap_probability_guard([0, 0], [0, 0], [2], weights, strict=False)
    assert_array_equal(report['upper_bound_per_cell'], [2])
    assert not report['certified_all_replicas']
    assert report['invalid_replicas'] == 1
    assert report['invalid_cell_replica_pairs'] == 1
    assert_array_equal(report['invalid_replica_indices'], [0])
    assert_array_equal(report['invalid_cell_indices'], [0])
    with pytest.raises(ProbabilityOverflowError, match='no replicas discarded') as exc:
        bootstrap_probability_guard([0, 0], [0, 0], [2], weights)
    assert exc.value.report['invalid_replicas'] == 1


def test_bound_is_sufficient_not_a_rejection_of_legal_sampled_replicas():
    report = bootstrap_probability_guard([0, 0], [0, 0], [2], [[1, 1], [0, 2]])
    assert not report['certified_all_replicas']
    assert report['invalid_replicas'] == 0


def test_file_bootstrap_never_filters_invalid_replicas():
    seed, replicas = 23, 100
    expected_invalid = np.count_nonzero(np.random.default_rng(seed).multinomial(2, [.5, .5], size=replicas)[:, 0] > 1)
    with pytest.raises(ProbabilityOverflowError) as exc:
        file_bootstrap([[5.], [0.]], [0, 0], [0, 0], [2], replicas=replicas, seed=seed)
    assert exc.value.report['invalid_replicas'] == expected_invalid > 0
    assert exc.value.report['replicas'] == replicas


def test_bootstrap_rank_warning_and_single_file():
    with pytest.warns(RuntimeWarning, match='F-1=1'):
        result = file_bootstrap([[1., 2., 3.], [2., 3., 4.]], [], [], [], replicas=3)
    assert result['covariance_rank_upper_bound'] == 1
    assert np.linalg.matrix_rank(result['bootstrap_cov']) == 1
    with pytest.warns(RuntimeWarning, match='F-1=0'):
        single = file_bootstrap([[2., 3.]], [0], [0], [1], replicas=3)
    assert_array_equal(single['bootstrap_cov'], np.zeros((2, 2)))
    assert_array_equal(single['replica_totals'], [[2, 3]] * 3)


@pytest.mark.parametrize('replicas', [0, -1, 1.5, True])
def test_bad_bootstrap_replica_counts(replicas):
    with pytest.raises(ValueError, match='positive integer'):
        file_bootstrap([[1.], [2.]], [], [], [0], replicas=replicas)


@pytest.mark.parametrize('weights', [[[1, 0]], [[1.5, .5]], [[-1, 3]], [[np.nan, 2]], [], np.empty((0, 2))])
def test_bad_bootstrap_weights(weights):
    with pytest.raises(ValueError):
        bootstrap_probability_guard([], [], [0], weights)


@pytest.mark.parametrize('files,cells,ngen', [([2], [0], [2]), ([0], [1], [2]), ([0], [0], [0]), ([0], [], [2]), ([.5], [0], [2])])
def test_bad_bootstrap_support(files, cells, ngen):
    with pytest.raises(ValueError):
        file_bootstrap([[1.], [2.]], files, cells, ngen, replicas=3)


@pytest.mark.parametrize('dense', [False, True])
@pytest.mark.parametrize('with_muon', [False, True])
def test_centered_paired_covariance_against_direct_residuals_including_zero_events(dense, with_muon):
    h = np.array([[0, 0], [2, 0], [0, 1], [1, 2], [0, 0]], dtype=float)
    mean = np.array([[1., 2., 0.], [0., 1., 3.]])
    g4 = np.array([[0, 0, 0], [3, 2, 1], [0, 2, 4], [2, 3, 4], [0, 0, 0]], dtype=float)
    muon = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=float)
    r = muon - g4 if with_muon else -g4
    residual = h @ mean + r
    hth, htr, rtr = h.T @ h, h.T @ r, r.T @ r
    if not dense:
        hth, htr, rtr = sparse.csr_matrix(hth), sparse.csr_matrix(htr), sparse.csr_matrix(rtr)
    result = paired_covariance(mean, hth, htr, rtr, h.sum(axis=0), r.sum(axis=0), len(h))
    assert_allclose(result['residual_total'], residual.sum(axis=0))
    assert_allclose(result['covariance'], len(h) * np.cov(residual, rowvar=False, ddof=1), atol=1e-14)
    assert not np.allclose(result['covariance'], residual.T @ residual)


def test_paired_zero_residual_and_no_cells():
    result = paired_covariance(np.empty((0, 2)), sparse.csr_matrix((0, 0)), sparse.csr_matrix((0, 2)), np.zeros((2, 2)), [], [0, 0], 5)
    assert_array_equal(result['residual_total'], [0, 0])
    assert_array_equal(result['covariance'], np.zeros((2, 2)))


@pytest.mark.parametrize('change', [
    {'nEvents': 1}, {'nEvents': 2.5}, {'nEvents': True},
    {'hth': np.zeros((2, 2))}, {'htr': np.zeros((1, 3))},
    {'rtr': [[1, 2], [0, 1]]}, {'rsum': [np.nan, 0]},
    {'counts': [-1]}, {'counts': [1.5]}, {'mean_per_cell': [[-1, 1]]},
])
def test_bad_paired_inputs(change):
    args = dict(mean_per_cell=[[1, 1]], hth=[[1]], htr=[[0, 0]], rtr=np.zeros((2, 2)), counts=[1], rsum=[0, 0], nEvents=3)
    args.update(change)
    with pytest.raises(ValueError):
        paired_covariance(**args)


def test_probability_report_counts_replicas_separately_from_cell_pairs():
    report = bootstrap_probability_guard([0, 0], [0, 1], [1, 1], [[2, 0], [1, 1]], strict=False)
    assert report['invalid_replicas'] == 1
    assert report['invalid_cell_replica_pairs'] == 2
    assert_array_equal(report['invalid_cell_indices'], [0, 1])


@pytest.mark.parametrize('contributions', [np.empty((0, 2)), [[-1.], [2.]], [[np.nan], [2.]], [[np.inf], [2.]], [1., 2.]])
def test_bad_bootstrap_contributions(contributions):
    with pytest.raises(ValueError):
        file_bootstrap(contributions, [], [], [0], replicas=2)


def test_noncanonical_sparse_feature_entries_are_additive():
    features = sparse.csr_matrix((np.array([1., 2., 3.]), np.array([0, 0, 1]), np.array([0, 3])), shape=(1, 2))
    result = conditional_moments(features, [0], [2], [1])
    assert_allclose(result['total_mean'], [1.5, 1.5])
    assert_allclose(result['sampling_cov'], np.full((2, 2), 2.25))
