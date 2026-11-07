"""Independent scalar population/delete-block oracles; synthetic inputs only."""
import json
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from calofastsim.showers.conditional_stats import moments


FIELDS = ("mean", "std", "covariance", "correlation")


def scalar_arm(values, x):
    """Decimal arithmetic on exact input floats, not implementation helpers."""
    n = len(values)
    if not n:
        return dict(n=0, **{key: None for key in FIELDS})
    with localcontext() as ctx:
        ctx.prec = 80
        y = [Decimal.from_float(float(v)) for v in values]
        covariate = [Decimal.from_float(float(v)) for v in x]
        mean, mx = sum(y) / n, sum(covariate) / n
        vy = sum((v - mean) ** 2 for v in y) / n
        vx = sum((v - mx) ** 2 for v in covariate) / n
        covariance = sum((a - mean) * (b - mx) for a, b in zip(y, covariate)) / n
        correlation = float(covariance / (vy * vx).sqrt()) if vy and vx else None
        return dict(n=n, mean=float(mean), std=float(vy.sqrt()),
                    covariance=float(covariance), correlation=correlation)


def scalar_oracle(fast, stock, x, group, fast_mask=None, stock_mask=None):
    """Literally delete original blocks and recompute both selected samples."""
    fm = [True] * len(fast) if fast_mask is None else list(fast_mask)
    sm = [True] * len(stock) if stock_mask is None else list(stock_mask)

    def arms(delete=None):
        selected_f = [i for i in range(len(fast)) if fm[i] and group[i] != delete]
        selected_s = [i for i in range(len(stock)) if sm[i] and group[i] != delete]
        return (scalar_arm([fast[i] for i in selected_f], [x[i] for i in selected_f]),
                scalar_arm([stock[i] for i in selected_s], [x[i] for i in selected_s]))

    def difference(f, s):
        return {key: None if f[key] is None or s[key] is None else f[key] - s[key]
                for key in FIELDS}

    f, s = arms()
    delta = difference(f, s)
    groups = sorted(set(group))
    leaves = [difference(*arms(g)) for g in groups]
    errors = {}
    for key in FIELDS:
        values = [leave[key] for leave in leaves]
        if len(groups) < 2 or delta[key] is None or any(v is None for v in values):
            errors[key] = None
        else:
            mean = math.fsum(values) / len(values)
            errors[key] = math.sqrt((len(groups) - 1) / len(groups)
                                    * math.fsum((v - mean) ** 2 for v in values))
    return dict(fast=f, stock=s, difference=delta, error=errors)


def assert_result(actual, expected, *, rtol=2e-12, atol=2e-12):
    assert actual.keys() == expected.keys()
    for section in expected:
        assert actual[section].keys() == expected[section].keys()
        for key, value in expected[section].items():
            got = actual[section][key]
            if value is None:
                assert got is None, (section, key, got)
            else:
                assert got == pytest.approx(value, rel=rtol, abs=atol), (section, key, got, value)
    json.dumps(actual, allow_nan=False)


def test_explicit_population_normalization_and_same_query_covariate():
    f, s, x = [0., 2., 4., 6.], [4., 3., 2., 1.], [1., 3., 5., 7.]
    result = moments(f, s, x, [0, 0, 7, 9])
    assert result["fast"] == dict(n=4, mean=3., std=math.sqrt(5.), covariance=5., correlation=1.)
    assert result["stock"]["covariance"] == -2.5
    assert result["stock"]["correlation"] == pytest.approx(-1.)
    assert result["difference"]["covariance"] == 7.5
    assert_result(result, scalar_oracle(f, s, x, [0, 0, 7, 9]))


@pytest.mark.parametrize("selection", range(32))
def test_unequal_blocks_separate_selection_against_scalar_delete_oracle(selection):
    # Original rows have gaps: selected-row//100 would give different blocks.
    rows = np.array([700, 1, 101, 703, 3, 402, 401, 709, 4, 405, 710, 1900, 6])
    group = rows // 100
    index = np.arange(len(rows))
    fast = ((index * 17) % 23).astype(float) - 7
    stock = ((index * index * 7) % 31).astype(float) - 9
    x = (index % 5).astype(float) * .7 + index / 11
    fm = ((index * 3 + selection) % 7) >= selection % 5
    sm = ((index * 5 + selection // 3) % 9) > selection % 7
    result = moments(fast, stock, x, group, fm, sm)
    assert_result(result, scalar_oracle(fast, stock, x, group, fm, sm))
    assert result["fast"]["n"] == sum(fm)
    assert result["stock"]["n"] == sum(sm)


@pytest.mark.parametrize("fm,sm", [
    ([True, True, False, False, False, False], [False, False, True, True, False, False]),
    ([True, False, True, False, False, False], [False, True, True, True, False, False]),
    ([True, True, True, True, False, False], [True, True, True, True, False, False]),
])
def test_original_union_includes_unselected_and_one_arm_only_blocks(fm, sm):
    fast, stock, x, group = [1., 5., 2., 7., 100., 0.], [8., 2., 3., 9., 4., 0.], [2., 4., 1., 5., 8., 3.], [0, 0, 2, 2, 19, 19]
    result = moments(fast, stock, x, group, fm, sm)
    assert_result(result, scalar_oracle(fast, stock, x, group, fm, sm))
    # The absent block is still a replicate; deleting it returns the full sample.
    if fm == sm:
        restricted = moments(fast[:4], stock[:4], x[:4], group[:4])
        assert result["error"]["mean"] != pytest.approx(restricted["error"]["mean"])


def test_joint_selection_is_caller_intersection_not_implicit_pairwise_filter():
    f, s, x, g = np.arange(8.), np.arange(8.)[::-1] ** 2, np.arange(8.) ** 2, np.repeat([4, 11, 19, 23], 2)
    fm, sm = f > 2, s > 3
    own = moments(f, s, x, g, fm, sm)
    joint = fm & sm
    both = moments(f, s, x, g, joint, joint)
    assert own["fast"]["n"] == 5 and own["stock"]["n"] == 6
    assert both["fast"]["n"] == both["stock"]["n"] == 3
    assert_result(own, scalar_oracle(f, s, x, g, fm, sm))
    assert_result(both, scalar_oracle(f, s, x, g, joint, joint))


@pytest.mark.parametrize("fast,stock,x,group,fm,sm", [
    ([], [], [], [], None, None),
    ([0.], [0.], [9.], [55], None, None),
    ([2., 7.], [8., 1.], [1., 3.], [0, 1], None, None),
    ([2., 7., 9.], [8., 1., 6.], [1., 3., 2.], [0, 1, 2], [True, False, False], None),
    ([2., 7.], [8., 1.], [1., 3.], [0, 1], [False, False], None),
    ([2., 7.], [8., 1.], [1., 3.], [0, 1], [False, False], [False, False]),
    ([2., 7., 3.], [8., 1., 4.], [1., 3., 2.], [1, 1, 1], None, None),
    ([0., 0., 0., 0.], [0., 0., 0., 0.], [1., 2., 3., 4.], [0, 1, 1, 2], None, None),
    ([1., 2., 3., 4.], [3., 5., 1., 2.], [0., 0., 0., 0.], [0, 1, 1, 2], None, None),
    ([0., 0., 0., 1.], [3., 5., 1., 2.], [0., 0., 0., 1.], [0, 1, 1, 2], None, None),
])
def test_sparse_empty_singleton_zero_and_degenerate(fast, stock, x, group, fm, sm):
    assert_result(moments(fast, stock, x, group, fm, sm),
                  scalar_oracle(fast, stock, x, group, fm, sm))


def test_singleton_deleted_correlation_missing_but_other_errors_available():
    result = moments([1., 4.], [3., 2.], [2., 8.], [0, 1])
    assert result["error"]["correlation"] is None
    assert result["error"]["std"] == result["error"]["covariance"] == 0.
    assert result["error"]["mean"] == 2.


def test_zero_variance_is_exact_not_nan_or_small_positive():
    y = np.full(103, .1)
    result = moments(y, y, np.arange(103.), np.arange(103) // 10)
    for arm in ("fast", "stock"):
        assert result[arm]["std"] == result[arm]["covariance"] == 0.
        assert result[arm]["correlation"] is None
    assert result["error"] == dict(mean=0., std=0., covariance=0., correlation=None)


@pytest.mark.parametrize("offset", [0., 1e8, 1e12, -1e12])
def test_large_common_location_preserves_small_width_and_covariance(offset):
    d = np.array([0., .25, -.5, .75, -1., 1.25, -1.5, 1.75, 2., -2.25, 2.5, -2.75])
    f, s, x = offset + d, -offset + d[::-1] * .5, offset + d * 2
    g = np.repeat([0, 3, 8, 19], [2, 3, 3, 4])
    actual = moments(f, s, x, g)
    expected = scalar_oracle(f, s, x, g)
    # Decimal oracle's published float means lose some low bits at large offset;
    # the implementation centers errors before adding back the common location.
    for section in ("fast", "stock", "difference"):
        for field in FIELDS:
            assert actual[section][field] == pytest.approx(expected[section][field], abs=3e-12)
    for field in ("std", "covariance", "correlation"):
        assert actual["error"][field] == pytest.approx(expected["error"][field], abs=3e-12)
    baseline = moments(d, d[::-1] * .5, d * 2, g)
    assert actual["error"]["mean"] == pytest.approx(baseline["error"]["mean"], abs=3e-12)


@pytest.mark.parametrize("magnitude", [1e7, 1e12, 1e16])
def test_delete_extreme_block_does_not_subtract_nearly_equal_squares(magnitude):
    # Deleting the extreme block leaves a strictly constant arm and x: raw
    # sum-of-squares subtraction gives negative variance or spurious width.
    f = [magnitude, 3., 3., 3., 3., 3.]
    s = [-magnitude, -2., -2., -2., -2., -2.]
    x = [magnitude, 1., 1., 1., 1., 1.]
    g = [0, 1, 1, 2, 3, 3]
    actual = moments(f, s, x, g)
    assert_result(actual, scalar_oracle(f, s, x, g), rtol=3e-12, atol=3.)
    assert actual["error"]["correlation"] is None
    assert actual["fast"]["correlation"] == pytest.approx(1.)
    assert actual["stock"]["correlation"] == pytest.approx(-1.)


def test_cancellation_covariance_and_identical_arms():
    x = np.tile([-1., 1., 1., -1.], 8)
    y = np.tile([1., 1., -1., -1.], 8) * 1e7
    group = np.arange(len(x)) // 4
    actual = moments(y, y, x, group)
    assert_result(actual, scalar_oracle(y, y, x, group), atol=2e-9)
    assert actual["fast"]["covariance"] == pytest.approx(0., abs=2e-9)
    assert actual["difference"] == dict.fromkeys(FIELDS, 0.)
    assert actual["error"] == dict.fromkeys(FIELDS, 0.)


def test_permutation_block_relabeling_and_arm_swap():
    f, s, x = np.arange(17.) ** 2, np.arange(17.)[::-1] ** 1.5, np.arange(17.) % 5
    g = np.arange(17) // 3
    fm, sm = np.arange(17) % 3 != 1, np.arange(17) % 4 != 2
    result = moments(f, s, x, g, fm, sm)
    order = np.array([16, 0, 7, 3, 9, 12, 1, 11, 6, 2, 10, 4, 13, 5, 15, 8, 14])
    permuted = moments(f[order], s[order], x[order], (900 - 17 * g)[order], fm[order], sm[order])
    assert_result(permuted, result)
    swap = moments(s, f, x, g, sm, fm)
    assert_result(swap, dict(fast=result["stock"], stock=result["fast"], error=result["error"],
                             difference={key: -value for key, value in result["difference"].items()}))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["fast", "stock", "x"])
def test_nonfinite_selected_rejected_and_explicitly_unselected_ignored(bad, field):
    data = dict(fast=np.array([1., 2., 3., 4.]), stock=np.array([4., 3., 2., 1.]),
                x=np.array([1., 5., 3., 2.]), group=np.array([0, 0, 1, 2]))
    data[field][1] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        moments(**data)
    mask = np.array([True, False, True, True])
    fm = mask if field in ("fast", "x") else None
    sm = mask if field in ("stock", "x") else None
    assert_result(moments(**data, fast_mask=fm, stock_mask=sm),
                  scalar_oracle(**data, fast_mask=fm, stock_mask=sm))
    if field == "x":
        with pytest.raises(ValueError, match="nonfinite"):
            moments(**data, fast_mask=mask)


@pytest.mark.parametrize("change", [
    dict(fast=[[1., 2.]]), dict(stock=[1.]), dict(x=[1., 2., 3.]), dict(group=[0]),
    dict(group=[0., 1.]), dict(group=[0., np.nan]), dict(group=["a", "b"]),
    dict(fast_mask=[1, 0]), dict(stock_mask=[True]), dict(stock_mask=[[True, False]]),
])
def test_reject_malformed_inputs(change):
    data = dict(fast=[1., 2.], stock=[3., 4.], x=[2., 1.], group=[0, 1])
    data.update(change)
    with pytest.raises(ValueError):
        moments(**data)


def test_no_mutation_and_masked_nonfinite_empty_samples_are_strict_json():
    f = np.array([np.nan, np.inf])
    s = np.array([-np.inf, np.nan])
    x = np.array([np.nan, np.nan])
    mask = np.zeros(2, dtype=bool)
    snapshots = [a.copy() for a in (f, s, x, mask)]
    result = moments(f, s, x, [0, 5], mask, mask)
    assert result["fast"]["n"] == result["stock"]["n"] == 0
    assert all(value is None for value in result["error"].values())
    json.dumps(result, allow_nan=False)
    for original, saved in zip((f, s, x, mask), snapshots):
        np.testing.assert_array_equal(original, saved)


@pytest.mark.parametrize("magnitude", [1e12, 1e16, 1e20])
@pytest.mark.parametrize("extreme_block", [0, 2, 9])
def test_deleting_extreme_preserves_nonconstant_small_remaining_values(magnitude, extreme_block):
    f = [magnitude, 1., 1.25, 1.5, 1.75, 2.]
    s = [-magnitude, 1., 1.25, 1.5, 1.75, 2.]
    x = [magnitude, 1., 1.25, 1.5, 1.75, 2.]
    g = [extreme_block, 1, 1, 3, 4, 4]
    actual = moments(f, s, x, g)
    expected = scalar_oracle(f, s, x, g)
    assert actual["error"]["correlation"] == pytest.approx(1.5)
    # Width DIFFERENCES subtract quantities of order magnitude. Their absolute
    # rounding floor scales with those quantities, not with the tiny difference.
    for section in ("difference", "error"):
        assert actual[section]["std"] == pytest.approx(
            expected[section]["std"], abs=8 * np.spacing(magnitude))
        expected[section]["std"] = actual[section]["std"]
    assert_result(actual, expected, rtol=3e-12, atol=1e-9)


def test_mean_error_uses_paired_not_independent_arm_replicates():
    f = np.array([1., 3., 2., 6., 9., 11., 8.])
    s = f + 23.
    x = np.array([0., 1., 3., 2., 7., 4., 6.])
    result = moments(f, s, x, [0, 0, 5, 8, 8, 8, 14])
    assert result["error"]["mean"] == 0.
    assert result["difference"]["mean"] == -23.
    assert_result(result, scalar_oracle(f, s, x, [0, 0, 5, 8, 8, 8, 14]))


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_scale_covariance_and_width_without_degeneracy_threshold(scale):
    f = np.array([1., -2., 3., -4., 5., 0., -7., 8.]) * scale
    s = np.array([3., 2., -1., 7., 9., -5., 0., -4.]) * scale
    x = np.array([1., 2., 4., 3., 7., 6., 9., 8.]) * scale
    g = [0, 0, 3, 3, 3, 6, 6, 8]
    result = moments(f, s, x, g)
    assert_result(result, scalar_oracle(f, s, x, g), rtol=2e-12, atol=1e-30)
    assert all(result[arm]["correlation"] is not None for arm in ("fast", "stock"))


def test_one_deleted_degenerate_covariate_invalidates_only_correlation_error():
    f, s, x, g = [2., 1., 4., 6.], [1., 5., 2., 3.], [1., 0., 0., 0.], [0, 1, 1, 2]
    result = moments(f, s, x, g)
    assert_result(result, scalar_oracle(f, s, x, g))
    assert result["difference"]["correlation"] is not None
    assert result["error"]["correlation"] is None
    assert all(result["error"][key] is not None for key in ("mean", "std", "covariance"))


def test_full_mean_does_not_lose_small_remainder_to_large_center():
    f, s, x, g = [1e12, -1e12, 1.], [0., 0., 0.], [1., 2., 3.], [0, 1, 2]
    result = moments(f, s, x, g)
    assert result["fast"]["mean"] == 1 / 3
    assert_result(result, scalar_oracle(f, s, x, g), rtol=2e-12, atol=1e-12)
