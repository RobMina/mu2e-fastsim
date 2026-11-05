"""An identical marginal acceptance does not imply closure conditional on one selected draw."""
import numpy as np
import pytest

from beamfastsim.target_conditioning import selection_null


def test_equal_unconditional_acceptance_has_conditional_deficit():
    a, b = np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
    r = selection_null(a, b, np.array([0, 0, 1, 1]))
    assert r['unconditional']['ratio'] == 1
    assert r['candidate_acceptance_given_base'] == 0.5
    assert r['base_acceptance_given_candidate'] == 0.5
    r = selection_null(a, a, np.array([0, 0, 1, 1]))
    assert r['candidate_acceptance_given_base'] == 1
    with pytest.raises(ValueError):
        selection_null(a, b[:2], np.zeros(4))
