"""Multinomial aggregate replicas are the exact Bernoulli/uniform-template sampler law."""
import numpy as np
from scipy import sparse
import pytest

from valhist.shower_sampling_tails import multinomial_replicas
from valhist.shower_moments import conditional_moments


def test_multinomial_moments_match_exact_independent_primary_sampling():
    x = sparse.csr_matrix([[2., 1.], [4., 3.], [1., 5.]])
    cell = np.array([0, 0, 1]); ngen = np.array([8, 5]); neval = np.array([7, 9])
    moments = conditional_moments(x, cell, ngen, neval)
    samples = multinomial_replicas(x, cell, ngen, neval, 50000, 71, batch=1000)
    np.testing.assert_allclose(samples.mean(axis=0), moments["total_mean"], rtol=.005)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), moments["sampling_cov"], rtol=.02)


def test_multinomial_zero_and_probability_one():
    x = sparse.csr_matrix([[2., 1.]])
    sample = multinomial_replicas(x, np.array([0]), np.array([1]), np.array([10]), 10, 12)
    np.testing.assert_array_equal(sample, np.tile([20., 10.], (10, 1)))
    sample = multinomial_replicas(x, np.array([0]), np.array([1]), np.array([0]), 10, 12)
    assert not sample.any()
    with pytest.raises(ValueError, match="support"):
        multinomial_replicas(x, np.array([0]), np.array([0]), np.array([1]), 10, 12)
