"""Matched numerator/denominator influence keeps their correlation and the shared bank."""
import numpy as np
import pytest

from valhist.shower_source_study import ratio_influence, bank_influence, certify_source_pairs
from valhist import shower_uncertainty as U
import json
import pandas as pd


def test_source_pairing_checks_payload_not_just_event_labels(tmp_path, monkeypatch):
    old, full = tmp_path / "old.root", tmp_path / "full.root"
    old.write_bytes(b"old"); full.write_bytes(b"full")
    (tmp_path / "plan.json").write_text(json.dumps({"donors": [{"old_table": str(old), "old_sha256": U.sha256(old)}]}))
    donor = pd.DataFrame({"run": [1400], "subrun": [1], "event": [2], "idx": [0], "pdg": [22], "e": [50.]})
    frames = {str(old): donor, str(full): donor.copy()}
    monkeypatch.setattr(U.tables, "read", lambda path, tree: frames[path])
    meta = {"campaign": str(tmp_path), "gen": [str(full)]}
    records, _ = certify_source_pairs(meta)
    assert records[0]["original_product_payload_identical"]
    frames[str(full)].loc[0, "e"] = 51.
    with pytest.raises(AssertionError):
        certify_source_pairs(meta)


def test_matched_proportional_response_has_zero_file_uncertainty():
    den = np.array([[4., 2.], [3., 1.], [5., 3.]])
    response = np.array([[2., 5.], [7., 1.]])
    num = den[:, :, None] * response[None, :, :]
    got = ratio_influence(num, den, np.array([100., 200.]))
    np.testing.assert_allclose(got, 0, atol=1e-12)


def test_influence_matches_finite_difference_global_file_weights():
    den = np.array([[4., 2.], [3., 1.], [5., 3.]])
    num = np.array([[[1., 2.], [3., 1.]], [[5., 1.], [2., 4.]], [[2., 3.], [1., 2.]]])
    neval = np.array([100., 200.]); direction = np.array([1., -2., 1.]); eps = 1e-5
    def expected(weights):
        return np.einsum("c,cd->d", neval / (weights @ den), np.einsum("f,fcd->cd", weights, num))
    derivative = (expected(1 + eps * direction) - expected(1 - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(direction @ ratio_influence(num, den, neval), derivative, rtol=1e-8)


def test_shared_bank_contrast_derivative_preserves_covariance():
    den = np.array([[4., 2.], [3., 1.], [5., 3.]])
    num = np.arange(12, dtype=float).reshape(3, 2, 2) + 1
    prod = np.array([[10., 20.], [30., 10.]])
    neval = np.array([100., 200.]); scale = 20.; direction = np.array([1., -1., 0.]); eps = 1e-5
    influence = ratio_influence(num, den, neval) - bank_influence(prod, den, neval, scale)
    def contrast(w):
        return np.einsum("c,cd->d", neval / (w @ den), np.einsum("f,fcd->cd", w, num) - prod / scale)
    derivative = (contrast(1 + eps * direction) - contrast(1 - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(direction @ influence, derivative, rtol=1e-8)
    np.testing.assert_allclose(influence.sum(axis=0), 0, atol=1e-12)


def test_unsupported_ratio_fails_closed():
    with pytest.raises(ValueError, match="unsupported"):
        ratio_influence(np.ones((2, 1, 1)), np.zeros((2, 1)), np.ones(1))
