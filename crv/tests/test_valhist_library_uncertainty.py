"""Nested donor-library variance components retain draw noise and negative estimates."""
import numpy as np
import pytest

from valhist.library_uncertainty import build_identity, components


def test_identity_requires_common_plan_denominator_code_and_geometry():
    base = dict(gen=['gen.root'], provenance=dict(plan_sha256='plan', build_code_sha256={'code': 'sha'},
                                                counters={'path': 'geom', 'sha256': 'geomsha'}))
    identity = build_identity(base)
    for key in ('plan_sha256', 'build_code_sha256', 'counters'):
        changed = dict(base, provenance=dict(base['provenance'], **{key: 'other'}))
        assert build_identity(changed) != identity
    assert build_identity(dict(base, gen=['different.root'])) != identity
    assert build_identity(dict(base, inputs=['bootstrap donor repeat'])) == identity


def test_zero_draw_noise_keeps_between_library_variance():
    values = np.array([[1, 1], [2, 2], [3, 3]], float)
    got = components(values)
    assert got['conditional_draw_variance'] == 0
    assert got['library_variance_noise_subtracted'] == 1
    assert got['library_sd_nonnegative'] == 1


def test_draw_noise_removed_but_negative_retained():
    got = components([[0, 2], [0, 2], [0, 2]])
    assert got['library_mean_variance'] == 0
    assert got['conditional_draw_variance'] == 2
    assert got['library_variance_noise_subtracted'] == -1
    assert got['negative_component'] is True
    assert got['library_sd_nonnegative'] == 0


def test_vector_statistics_match_explicit_formula():
    x = np.arange(36.).reshape(3, 2, 6)
    got = components(x)
    np.testing.assert_allclose(got['library_variance_noise_subtracted'],
                               x.mean(1).var(0, ddof=1) - x.var(1, ddof=1).mean(0) / 2)


@pytest.mark.parametrize('values', [[[1, 2]], [[1], [2]], [1, 2], [[np.nan, 2], [3, 4]]])
def test_invalid_ensembles_rejected(values):
    with pytest.raises(ValueError):
        components(values)
