"""Exact donor-feature moments and context/fallback selection checks."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from crvfastsim import tier1b as T
from valhist import descendant_expectation as E
from valhist.shower_moments import conditional_moments


def test_pooled_and_context_moments_include_empty_crossings():
    lx = pd.DataFrame({"xid": np.arange(4), "bin": [0] * 4, "ctx": [0, 0, 1, 1]})
    features = sparse.csr_matrix([[0., 0.], [2., 6.], [4., 8.], [0., 0.]])
    x, cell = E.pool(lx, features)
    n = np.bincount(cell, minlength=9)
    want = np.zeros(9, dtype=int)
    want[0], want[8] = 2, 3
    result = conditional_moments(x, cell, n, want)
    outcomes0, outcomes8 = features.toarray()[:2], features.toarray()
    expected = 2 * outcomes0.mean(0) + 3 * outcomes8.mean(0)
    covariance = 2 * np.cov(outcomes0, rowvar=False, ddof=0) + 3 * np.cov(outcomes8, rowvar=False, ddof=0)
    np.testing.assert_allclose(result["total_mean"], expected)
    np.testing.assert_allclose(result["sampling_cov"], covariance)


def test_exact_features_keep_otherbar_destinations_and_zero_rows():
    geom = SimpleNamespace(nbars=2, center=np.array([[0., 5., 0.], [0., 0., 0.]]),
                           halflen=np.ones((2, 3)), local_to_world=np.tile([1, 0, 2], (2, 1)),
                           sector=np.zeros(2, dtype=int), module=np.zeros(2, dtype=int), layer=np.arange(2))
    lx = pd.DataFrame({"xid": [0, 1], "bin": [0, 0], "ctx": [0, 0]})
    desc = pd.DataFrame({"xid": [0, 0], "samebar": [0, 1], "edep": [2., 99.],
                         "dcx": [0., 0.], "dcy": [-5., 0.], "dcz": [0., 0.]})
    x = E.feature_matrix(lx, desc, np.array([0, 0]), geom).toarray()
    assert x[0, 1] == 1 and x[0, 17] == 2
    assert x.sum() == 3 and not x[1].any()
    desc.loc[0, "dcy"] = -50
    with pytest.raises(ValueError, match="exact geometry"):
        E.feature_matrix(lx, desc, np.array([0, 0]), geom)


def test_context_selection_agrees_with_sampler_fallback(monkeypatch):
    lx = pd.DataFrame({"xid": np.arange(21), "bin": [0] * 21, "ctx": [0] * 20 + [1]})
    fx = pd.DataFrame({"bar": [0, 1], "px0": [0., 0.], "py0": [-1., -1.], "pz0": [0., 0.], "len": [1., 1.]})
    monkeypatch.setattr(T, "_group_index", lambda *args: np.zeros(2, dtype=int))
    monkeypatch.setattr(T, "_bin", lambda *args: np.zeros(2, dtype=int))
    monkeypatch.setattr(T, "crossing_context", lambda *args: np.array([0, 1]))
    code, fallback = E.evaluation_codes(fx, lx, None, None, True)
    np.testing.assert_array_equal(code, [0, 8])
    assert fallback == 1
    code, fallback = E.evaluation_codes(fx, lx, None, None, False)
    np.testing.assert_array_equal(code, [8, 8])
    assert fallback == 2


def test_replica_mc_error_is_not_sampler_sd():
    from valhist.descendant_replicas import summarize
    before = np.tile(np.arange(1., 5.)[:, None], (1, 32))
    after = before / 2
    result = summarize(before, after, np.ones(32))
    np.testing.assert_allclose(result["post"]["sector_mean"], np.full(8, 5.))
    np.testing.assert_allclose(result["post"]["sector_mean_mc_se"],
                               np.asarray(result["post"]["sector_sampling_sd"]) / 2)
    np.testing.assert_allclose(result["post_minus_pre"]["sector_mean"], np.full(8, -5.))


def test_crossing_identity_contract_fails_closed():
    with pytest.raises(ValueError, match="ordered contiguous"):
        E.cell_codes(pd.DataFrame({"xid": [0, 0], "bin": [0, 0], "ctx": [0, 0]}))
    with pytest.raises(ValueError, match="context"):
        E.cell_codes(pd.DataFrame({"xid": [0], "bin": [0], "ctx": [8]}))
