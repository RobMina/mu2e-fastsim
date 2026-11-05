"""Pure analytic diagnostics only; no transport or physics validation."""
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from beamfastsim.flash_photon_chords import (
    NONFINITE_LENGTH, NONFINITE_MOMENTUM, NONFINITE_POSITION,
    NONPOSITIVE_LENGTH, NUMERIC_OVERFLOW, ZERO_CHORD, ZERO_MOMENTUM,
    PhotonChordDiagnostics, diagnose_photon_chords,
)


def diagnose(pre, post, momentum, length):
    return diagnose_photon_chords(pre, post, momentum, np.asarray(length, dtype=np.float32))


def test_analytic_forward_backward_orthogonal_noncollinear_and_duplicates():
    post = [[3, 4, 0], [3, 4, 0], [3, 4, 0], [3, 4, 0], [3, 4, 0]]
    momentum = [[6, 8, 0], [-3, -4, 0], [-4, 3, 0], [0, 0, 2], [6, 8, 0]]
    result = diagnose(np.zeros((5, 3)), post, momentum, [5, 6, 4, 5, 5])
    assert isinstance(result, PhotonChordDiagnostics)
    np.testing.assert_array_equal(result.chord_length, [5] * 5)
    np.testing.assert_array_equal(result.chord_minus_length, [0, -1, 1, 0, 0])
    np.testing.assert_allclose(result.direction_cosine, [1, -1, 0, 0, 1], atol=1e-16)
    np.testing.assert_allclose(result.transverse_sine, [0, 0, 1, 1, 0], atol=1e-16)
    np.testing.assert_array_equal(result.reason, 0)
    oblique = diagnose([[0, 0, 0]], [[3, 4, 0]], [[1, 0, 0]], [5])
    np.testing.assert_allclose(oblique.direction_cosine, [.6])
    np.testing.assert_allclose(oblique.transverse_sine, [.8])


def test_float32_rounding_signed_residual_and_storage_not_a_cut():
    exact = np.float64(1) + 2.0 ** -25
    result = diagnose([[0, 0, 0], [0, 0, 0]], [[exact, 0, 0], [2, 0, 0]],
                      [[1, 0, 0]] * 2, [exact, 1])
    np.testing.assert_array_equal(result.chord_minus_length, [2.0 ** -25, 1])
    np.testing.assert_array_equal(result.length_storage_ulp, [2.0 ** -23] * 2)
    np.testing.assert_array_equal(result.reason, 0)
    assert not hasattr(result, "valid")
    assert not hasattr(result, "resolved")
    assert not hasattr(result, "complete")


def test_translation_no_mutation_and_frozen_fields():
    pre = np.array([[0, 0, 0], [8, 4, -2]], dtype=np.int64)
    post = pre + np.array([3, 4, 0])
    momentum = np.array([[6, 8, 0], [-6, -8, 0]], dtype=np.float32)
    length = np.array([5, 6], dtype=np.float32)
    snapshots = [value.copy() for value in (pre, post, momentum, length)]
    result = diagnose_photon_chords(pre, post, momentum, length)
    offset = np.array([-3900, 50, 6000])
    translated = diagnose_photon_chords(pre + offset, post + offset, momentum, length)
    for field in fields(result):
        np.testing.assert_array_equal(getattr(result, field.name), getattr(translated, field.name))
    for value, snapshot in zip((pre, post, momentum, length), snapshots):
        np.testing.assert_array_equal(value, snapshot)
    with pytest.raises(FrozenInstanceError):
        result.reason = np.zeros(2)
    assert all(not np.shares_memory(result.chord_length, value) for value in (pre, post, momentum, length))


def test_empty_shapes_and_dtypes():
    result = diagnose(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), [])
    for field in fields(result):
        array = getattr(result, field.name)
        assert array.shape == (0,)
        assert array.dtype == (np.uint8 if field.name == "reason" else np.float64)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1., 1e200, 1e300])
def test_stable_norms_across_scales(scale):
    result = diagnose([[0, 0, 0]], np.array([[3., 4., 0]]) * scale,
                      np.array([[0., 0., 2.]]) * scale, [1])
    np.testing.assert_allclose(result.chord_length / scale, [5])
    np.testing.assert_array_equal(result.direction_cosine, [0])
    np.testing.assert_allclose(result.transverse_sine, [1])
    np.testing.assert_array_equal(result.reason, 0)


def test_smallest_subnormal_norms_and_length_spacing():
    tiny = np.nextafter(np.float64(0), np.float64(1))
    tiny32 = np.nextafter(np.float32(0), np.float32(1))
    result = diagnose([[0, 0, 0]], [[tiny, 0, 0]], [[tiny, 0, 0]], [tiny32])
    np.testing.assert_array_equal(result.chord_length, [tiny])
    np.testing.assert_array_equal(result.direction_cosine, [1])
    np.testing.assert_array_equal(result.transverse_sine, [0])
    np.testing.assert_array_equal(result.length_storage_ulp, [np.float64(tiny32)])
    np.testing.assert_array_equal(result.reason, 0)


def test_large_momentum_norm_need_not_be_representable_for_direction():
    huge = np.finfo(np.float64).max
    result = diagnose([[0, 0, 0]], [[1, 1, 0]], [[huge, huge, 0]], [1])
    np.testing.assert_allclose(result.direction_cosine, [1])
    np.testing.assert_allclose(result.transverse_sine, [0])
    np.testing.assert_array_equal(result.reason, 0)


def test_chord_norm_overflow_preserves_available_direction_and_ulp():
    huge = np.finfo(np.float64).max
    result = diagnose([[0, 0, 0]], [[huge, huge, 0]], [[1, 1, 0]], [1])
    assert np.isnan(result.chord_length[0])
    assert np.isnan(result.chord_minus_length[0])
    np.testing.assert_allclose(result.direction_cosine, [1])
    np.testing.assert_allclose(result.transverse_sine, [0])
    assert np.isfinite(result.length_storage_ulp[0])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


def test_subtraction_overflow_is_not_nonfinite_input_or_zero_chord():
    huge = np.finfo(np.float64).max
    result = diagnose([[-huge, 0, 0]], [[huge, 0, 0]], [[1, 0, 0]], [1])
    for value in (result.chord_length, result.chord_minus_length,
                  result.direction_cosine, result.transverse_sine):
        assert np.isnan(value[0])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


def test_float32_max_spacing_overflow_is_not_length_rejection():
    huge = np.finfo(np.float32).max
    result = diagnose([[0, 0, 0]], [[1, 0, 0]], [[1, 0, 0]], [huge])
    assert np.isnan(result.length_storage_ulp[0])
    np.testing.assert_array_equal(result.chord_minus_length, [1 - np.float64(huge)])
    np.testing.assert_array_equal(result.direction_cosine, [1])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


@pytest.mark.parametrize("bad_momentum", [False, True])
def test_normalization_underflow_is_explicit(bad_momentum):
    vector = [np.finfo(np.float64).max, np.nextafter(0., 1.), 0]
    result = diagnose([[0, 0, 0]], [[1, 0, 0]] if bad_momentum else [vector],
                      [vector] if bad_momentum else [[1, 0, 0]], [1])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])
    assert np.isfinite(result.chord_length[0])
    assert np.isfinite(result.chord_minus_length[0])
    assert np.isnan(result.direction_cosine[0])
    assert np.isnan(result.transverse_sine[0])


def test_dot_underflow_does_not_erase_available_cross_norm():
    result = diagnose([[0, 0, 0]], [[1, 1e-200, 0]], [[0, 1e-200, 1]], [1])
    assert np.isnan(result.direction_cosine[0])
    np.testing.assert_array_equal(result.transverse_sine, [1])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


def test_cross_underflow_does_not_erase_available_dot():
    result = diagnose([[0, 0, 0]], [[1, 1e-200, 0]], [[1, 0, 1e-200]], [1])
    np.testing.assert_array_equal(result.direction_cosine, [1])
    assert np.isnan(result.transverse_sine[0])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_positions_preserve_length_ulp(value):
    result = diagnose([[value, 0, 0]], [[1, 0, 0]], [[1, 0, 0]], [1])
    assert np.isnan(result.chord_length[0])
    assert np.isnan(result.chord_minus_length[0])
    assert np.isnan(result.direction_cosine[0])
    assert np.isnan(result.transverse_sine[0])
    assert np.isfinite(result.length_storage_ulp[0])
    np.testing.assert_array_equal(result.reason, [NONFINITE_POSITION])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_momentum_preserves_chord_and_residual(value):
    result = diagnose([[0, 0, 0]], [[3, 4, 0]], [[value, 0, 0]], [6])
    np.testing.assert_array_equal(result.chord_length, [5])
    np.testing.assert_array_equal(result.chord_minus_length, [-1])
    assert np.isnan(result.direction_cosine[0])
    assert np.isnan(result.transverse_sine[0])
    np.testing.assert_array_equal(result.reason, [NONFINITE_MOMENTUM])


@pytest.mark.parametrize("value,reason", [(np.nan, NONFINITE_LENGTH), (np.inf, NONFINITE_LENGTH),
    (-np.inf, NONFINITE_LENGTH), (0, NONPOSITIVE_LENGTH), (-0., NONPOSITIVE_LENGTH),
    (-2, NONPOSITIVE_LENGTH)])
def test_bad_length_preserves_chord_and_direction(value, reason):
    result = diagnose([[0, 0, 0]], [[3, 4, 0]], [[3, 4, 0]], [value])
    np.testing.assert_array_equal(result.chord_length, [5])
    np.testing.assert_allclose(result.direction_cosine, [1])
    np.testing.assert_allclose(result.transverse_sine, [0])
    assert np.isnan(result.length_storage_ulp[0])
    if np.isfinite(value):
        np.testing.assert_array_equal(result.chord_minus_length, [5 - value])
    else:
        assert np.isnan(result.chord_minus_length[0])
    np.testing.assert_array_equal(result.reason, [reason])


def test_mixed_conditions_stay_aligned_and_bits_accumulate():
    result = diagnose([[0, 0, 0], [0, 0, 0], [np.nan, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [3, 4, 0], [1, 0, 0], [3, 4, 0]],
                      [[0, 0, 0], [0, 0, 0], [np.inf, 0, 0], [3, 4, 0]],
                      [-1, 6, np.nan, 5])
    np.testing.assert_array_equal(result.reason,
        [ZERO_CHORD | ZERO_MOMENTUM | NONPOSITIVE_LENGTH, ZERO_MOMENTUM,
         NONFINITE_POSITION | NONFINITE_MOMENTUM | NONFINITE_LENGTH, 0])
    np.testing.assert_allclose(result.chord_length, [0, 5, np.nan, 5], equal_nan=True)
    np.testing.assert_allclose(result.chord_minus_length, [1, -1, np.nan, 0], equal_nan=True)
    np.testing.assert_allclose(result.direction_cosine, [np.nan, np.nan, np.nan, 1], equal_nan=True)
    assert np.isnan(result.transverse_sine[:3]).all()


def test_zero_chord_without_zero_momentum_has_no_direction():
    result = diagnose([[2, 3, 4]], [[2, 3, 4]], [[1, 0, 0]], [1])
    np.testing.assert_array_equal(result.chord_length, [0])
    np.testing.assert_array_equal(result.chord_minus_length, [-1])
    assert np.isnan(result.direction_cosine[0]) and np.isnan(result.transverse_sine[0])
    np.testing.assert_array_equal(result.reason, [ZERO_CHORD])


@pytest.mark.parametrize("slot,bad", [
    (0, [0, 0, 0]), (1, [[1, 0, 0], [2, 0, 0]]), (2, np.zeros((1, 2))),
    (0, [["0", "0", "0"]]), (1, np.zeros((1, 3), dtype=complex)),
    (2, np.zeros((1, 3), dtype=bool)), (0, np.zeros((1, 3), dtype=object)),
    (3, [1.]), (3, np.array([1.], dtype=np.float64)), (3, np.array([1], dtype=int)),
    (3, np.array(1, dtype=np.float32)), (3, np.ones((1, 1), dtype=np.float32)),
    (3, np.ones(2, dtype=np.float32)), (3, np.array([True])),
])
def test_bad_schema(slot, bad):
    args = [np.zeros((1, 3)), np.ones((1, 3)), np.ones((1, 3)), np.ones(1, dtype=np.float32)]
    args[slot] = bad
    with pytest.raises(ValueError):
        diagnose_photon_chords(*args)


def test_mixed_numeric_and_input_conditions_do_not_hide_each_other():
    huge = np.finfo(np.float64).max
    result = diagnose([[-huge, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[huge, 0, 0], [3, 4, 0], [3, 4, 0]],
                      [[0, 0, 0], [np.nan, 0, 0], [3, 4, 0]],
                      [-1, np.finfo(np.float32).max, np.nan])
    np.testing.assert_array_equal(result.reason,
        [NUMERIC_OVERFLOW | ZERO_MOMENTUM | NONPOSITIVE_LENGTH,
         NUMERIC_OVERFLOW | NONFINITE_MOMENTUM, NONFINITE_LENGTH])
    np.testing.assert_allclose(result.chord_length, [np.nan, 5, 5], equal_nan=True)
    assert np.isfinite(result.chord_minus_length[1])
    np.testing.assert_allclose(result.direction_cosine, [np.nan, np.nan, 1], equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, ">f4"])
def test_real_numeric_promotion_readonly_strided_and_endian(dtype):
    pre = np.zeros((2, 6), dtype=dtype)[:, ::2]
    post = np.tile([3, 0, 4, 0, 0, 0], (2, 1)).astype(dtype)[:, ::2]
    momentum = post.copy()
    length = np.array([5, 5], dtype=">f4")
    for array in (pre, post, momentum, length):
        array.flags.writeable = False
    result = diagnose_photon_chords(pre, post, momentum, length)
    np.testing.assert_array_equal(result.chord_length, [5, 5])
    np.testing.assert_allclose(result.direction_cosine, [1, 1])
    np.testing.assert_array_equal(result.reason, 0)
    assert result.chord_length.dtype == np.float64


@pytest.mark.skipif(np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp,
                    reason="platform has no wider real floating dtype")
@pytest.mark.parametrize("slot", [0, 2])
@pytest.mark.parametrize("magnitude", ["overflow", "underflow"])
def test_wider_float_conversion_loss_is_numerical_not_input_nonfiniteness(slot, magnitude):
    args = [np.zeros((1, 3), dtype=np.longdouble), np.ones((1, 3)),
            np.ones((1, 3), dtype=np.longdouble), np.ones(1, dtype=np.float32)]
    value = (np.finfo(np.longdouble).max if magnitude == "overflow"
             else np.finfo(np.longdouble).tiny)
    args[slot][0, 0] = value
    result = diagnose_photon_chords(*args)
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])
    assert np.isnan(result.direction_cosine[0])
    assert np.isnan(result.transverse_sine[0])
    assert np.isfinite(result.length_storage_ulp[0])
    if slot == 2:
        assert np.isfinite(result.chord_length[0])
        assert np.isfinite(result.chord_minus_length[0])


def test_4096_rows_against_vectorized_independent_oracle():
    rng = np.random.default_rng(763)
    pre, post, momentum = rng.normal(size=(3, 4096, 3))
    length = rng.uniform(.1, 10, 4096).astype(np.float32)
    result = diagnose_photon_chords(pre, post, momentum, length)
    delta = post - pre
    chord = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    pnorm = np.sqrt(np.einsum("ij,ij->i", momentum, momentum))
    cosine = np.einsum("ij,ij->i", delta, momentum) / (chord * pnorm)
    cross = np.cross(delta, momentum)
    sine = np.sqrt(np.einsum("ij,ij->i", cross, cross)) / (chord * pnorm)
    np.testing.assert_allclose(result.chord_length, chord, rtol=5e-16)
    np.testing.assert_allclose(result.chord_minus_length, chord - length, atol=2e-15)
    np.testing.assert_allclose(result.direction_cosine, cosine, atol=5e-16)
    np.testing.assert_allclose(result.transverse_sine, sine, atol=5e-16)
    np.testing.assert_array_equal(result.length_storage_ulp, np.spacing(length).astype(np.float64))
    np.testing.assert_array_equal(result.reason, 0)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("offset,step", [(2**53, 1), (2**53 + 1, 2),
                                         (2**53 + 2, 2), (2**63 - 3, 1)])
def test_integer_conversion_uncertainty_cannot_report_zero_or_wrong_chord(dtype, offset, step):
    pre = np.array([[offset, 0, 0]], dtype=dtype)
    post = np.array([[offset + step, 1, 0]], dtype=dtype)
    result = diagnose(pre, post, [[1, 0, 0]], [1])
    for metric in (result.chord_length, result.chord_minus_length,
                   result.direction_cosine, result.transverse_sine):
        assert np.isnan(metric[0])
    assert np.isfinite(result.length_storage_ulp[0])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])
    # Also exercise complete endpoint collapse, not only a wrong nonzero chord.
    post[0, 1] = 0
    result = diagnose(pre, post, [[1, 0, 0]], [1])
    assert np.isnan(result.chord_length[0])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_integer_negative_and_positive_outside_exact_range(sign):
    pre = np.array([[sign * 2**53, 0, 0]], dtype=np.int64)
    post = np.array([[sign * (2**53 + 1), 0, 0]], dtype=np.int64)
    result = diagnose(pre, post, [[1, 0, 0]], [1])
    assert np.isnan(result.chord_length[0])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


@pytest.mark.parametrize("dtype,sign", [(np.int64, -1), (np.int64, 1), (np.uint64, 1)])
def test_exact_integer_threshold_remains_available(dtype, sign):
    pre = np.array([[sign * (2**53 - 1), 0, 0]], dtype=dtype)
    post = np.array([[sign * 2**53, 0, 0]], dtype=dtype)
    momentum = np.array([[2**53, 0, 0]], dtype=dtype)
    result = diagnose(pre, post, momentum, [1])
    np.testing.assert_array_equal(result.chord_length, [1])
    np.testing.assert_array_equal(result.chord_minus_length, [0])
    np.testing.assert_array_equal(result.direction_cosine, [sign])
    np.testing.assert_array_equal(result.transverse_sine, [0])
    np.testing.assert_array_equal(result.reason, [0])


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("value", [2**53 + 1, 2**53 + 2])
def test_integer_momentum_loss_preserves_good_chord(dtype, value):
    result = diagnose([[0, 0, 0]], [[3, 4, 0]],
                      np.array([[value, 1, 0]], dtype=dtype), [6])
    np.testing.assert_array_equal(result.chord_length, [5])
    np.testing.assert_array_equal(result.chord_minus_length, [-1])
    assert np.isfinite(result.length_storage_ulp[0])
    assert np.isnan(result.direction_cosine[0])
    assert np.isnan(result.transverse_sine[0])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
                    reason="platform has no wider real floating precision")
@pytest.mark.parametrize("slot", [0, 1, 2])
@pytest.mark.parametrize("collapse", [False, True])
def test_wider_float_roundtrip_precision_loss(slot, collapse):
    one = np.longdouble(1)
    changed = one + np.finfo(np.longdouble).eps
    args = [np.array([[one, 0, 0]], dtype=np.longdouble),
            np.array([[one if collapse else 2, 0, 0]], dtype=np.longdouble),
            np.array([[one, 0, 0]], dtype=np.longdouble)]
    args[slot][0, 0] = changed
    if slot == 2:
        args[1][0, 0] = 2
    result = diagnose(*args, [1])
    np.testing.assert_array_equal(result.reason, [NUMERIC_OVERFLOW])
    assert np.isnan(result.direction_cosine[0])
    assert np.isnan(result.transverse_sine[0])
    assert np.isfinite(result.length_storage_ulp[0])
    if slot == 2:
        np.testing.assert_array_equal(result.chord_length, [1])
        np.testing.assert_array_equal(result.chord_minus_length, [0])
    else:
        assert np.isnan(result.chord_length[0])
        assert np.isnan(result.chord_minus_length[0])


def test_conversion_loss_and_other_vectors_nonfinite_input_bits_are_independent():
    bad = np.array([[2**53 + 1, 0, 0]], dtype=np.int64)
    position_loss = diagnose(bad, bad, [[np.nan, 0, 0]], [1])
    momentum_loss = diagnose([[np.inf, 0, 0]], [[1, 0, 0]], bad, [1])
    np.testing.assert_array_equal(position_loss.reason,
                                  [NUMERIC_OVERFLOW | NONFINITE_MOMENTUM])
    np.testing.assert_array_equal(momentum_loss.reason,
                                  [NUMERIC_OVERFLOW | NONFINITE_POSITION])
    for result in (position_loss, momentum_loss):
        assert np.isnan(result.chord_length[0])
        assert np.isnan(result.chord_minus_length[0])
        assert np.isnan(result.direction_cosine[0])
        assert np.isnan(result.transverse_sine[0])
        assert np.isfinite(result.length_storage_ulp[0])
