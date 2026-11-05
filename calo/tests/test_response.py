"""Synthetic source-conformance checks ONLY, not stock execution validation.

Arithmetic expectations follow the inspected Offline source and product types;
no detector constants, conditions data, art execution or fit validation is used.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calofastsim import response as numpy_response


@pytest.fixture(params=["numpy", "jax"])
def backend(request):
    if request.param == "numpy":
        yield numpy_response
    else:
        jax = pytest.importorskip("jax")
        from calofastsim import response_jax
        with jax.enable_x64():
            yield response_jax


def evaluate(backend, name, *args, **kwargs):
    function = getattr(backend, name)
    if backend is numpy_response:
        return function(*args, **kwargs)
    import jax
    static = ("n_groups",) if name == "weighted_reco_digis" else ()
    return jax.jit(function, static_argnames=static)(*args, **kwargs)


def test_package_import_does_not_import_jax():
    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); "
         "import calofastsim; assert 'jax' not in sys.modules", root],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("birks", [False, True])
@pytest.mark.parametrize("longitudinal", [False, True])
def test_energy_source_order_and_switches(backend, birks, longitudinal):
    rng = np.random.default_rng(215)
    g4 = rng.uniform(-2, 15, (31, 3)).astype(np.float32)
    quenched = rng.uniform(-3, 11, (31, 3)).astype(np.float32)
    z = rng.uniform(-1, 13, (31, 1)).astype(np.float32)
    length = np.array([3.1, 7.6, 9.2], dtype=np.float32)
    slope = rng.uniform(-1, 2, (31, 1)).astype(np.float32)
    expected = quenched if birks else g4
    if longitudinal:
        normalized = np.divide(z, length, dtype=np.float32)
        centered = np.subtract(normalized, np.float32(0.5), dtype=np.float32)
        scaled = np.multiply(slope, centered, dtype=np.float32)
        factor = np.add(scaled, np.float32(1), dtype=np.float32)
        expected = np.multiply(expected, factor, dtype=np.float32)
    actual = evaluate(backend, "corrected_energy", g4, quenched, z, length, slope,
                      birks=birks, longitudinal=longitudinal)
    assert actual.dtype == np.float32
    if backend is numpy_response:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_energy_dynamic_switch_arrays_and_unclipped_position(backend):
    actual = evaluate(backend, "corrected_energy", [2, 2], [4, 4], [-1, 3], 2, 1,
                      birks=np.array([True, False]),
                      longitudinal=np.array([False, True]))
    np.testing.assert_array_equal(actual, np.array([4, 4], dtype=np.float32))


def test_energy_scalar_and_empty(backend):
    scalar = evaluate(backend, "corrected_energy", 2, 1, 1, 2, 0.2)
    assert scalar.shape == () and scalar.dtype == np.float32
    assert float(scalar) == 1
    empty = evaluate(backend, "corrected_energy", np.empty((0, 2)), 1, 1, 2, 0.2)
    assert empty.shape == (0, 2)


def test_means_shape_clamp_and_float32(backend):
    energy = np.array([[2], [-3]], dtype=np.float64)
    yields = np.array([[1, 2], [3, -4], [5, 6]], dtype=np.float64)
    result = evaluate(backend, "photoelectron_mean", energy, yields)
    assert result.shape == (2, 3, 2) and result.dtype == np.float32
    np.testing.assert_array_equal(result, [[[2, 4], [6, 0], [10, 12]],
                                          [[0, 0], [0, 12], [0, 0]]])
    assert evaluate(backend, "photoelectron_mean", 2, [1, 3]).shape == (2,)
    assert evaluate(backend, "photoelectron_mean", np.empty(0), [1, 3]).shape == (0, 2)
    with pytest.raises(ValueError, match="trailing"):
        evaluate(backend, "photoelectron_mean", [1, 2], [1, 2, 3])


def test_lround_halves_neighbors_and_large_values(backend):
    half = np.float32(0.5)
    means = np.array([0, np.nextafter(half, np.float32(0)), half, 1.5, 2.5,
                      3.5, 8388609, 2147483520], dtype=np.float32)
    result = evaluate(backend, "nonnegative_lround", means)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, [0, 0, 1, 2, 3, 4, 8388609, 2147483520])
    assert evaluate(backend, "nonnegative_lround", 2.5).shape == ()
    assert evaluate(backend, "nonnegative_lround", np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("invalid", [-0.1, np.nan, np.inf, -np.inf, 2147483648.0])
def test_lround_host_guards(invalid):
    with pytest.raises(ValueError):
        numpy_response.nonnegative_lround([invalid])


TIMING = dict(pbtime=-3, mbtime=100, event_length=300, onspill=False,
              digitization_start=-250, digitization_end=25,
              digitization_buffer=5, time_from_protons_to_dr_marker=2)


def test_time_fmod_negative_single_shift_and_no_upper_reject(backend):
    times = np.array([-353, -253, -153, -3, 24, 50, 250], dtype=np.float64)
    folded, accepted = evaluate(backend, "fold_step_time", times, **TIMING)
    assert folded.dtype == np.float64
    np.testing.assert_array_equal(folded, [-53, 47, -153, -3, 24, 50, 250])
    np.testing.assert_array_equal(accepted, np.ones(times.shape, dtype=bool))
    options = dict(TIMING, digitization_start=10)
    folded, accepted = evaluate(backend, "fold_step_time", [-153, 5, 6, 50], **options)
    np.testing.assert_array_equal(folded, [-153, 5, 6, 50])
    np.testing.assert_array_equal(accepted, [False, False, True, True])


def test_time_dynamic_spill_periods_and_empty(backend):
    folded, accepted = evaluate(backend, "fold_step_time", [-153, -153],
                                **dict(TIMING, onspill=np.array([True, False])))
    np.testing.assert_array_equal(folded, [47, -153])
    np.testing.assert_array_equal(accepted, [True, True])
    empty, mask = evaluate(backend, "fold_step_time", np.empty((0, 2)), **TIMING)
    assert empty.shape == mask.shape == (0, 2)


def test_timing_float32_correction_order(backend):
    options = dict(TIMING, pbtime=-0.3, mbtime=100,
                   digitization_start=2**24, digitization_end=2**24,
                   time_from_protons_to_dr_marker=2**24, digitization_buffer=0.25)
    start = ((np.float32(options["digitization_start"]) - np.float32(-0.3))
             - np.float32(2**24)) - np.float32(0.25)
    end = (np.float32(2**24) - np.float32(-0.3)) - np.float32(2**24)
    time = np.array([-0.5, 0, 0.5], dtype=np.float64)
    pb = np.float64(np.float32(-0.3))
    expected = np.fmod(time + pb, 300) - pb
    expected = np.where(expected < max(100.0, float(end)) - 300, expected + 300, expected)
    folded, accepted = evaluate(backend, "fold_step_time", time, **options)
    np.testing.assert_array_equal(folded, expected)
    np.testing.assert_array_equal(accepted, expected >= start)


def test_weighted_given_membership_double_weights_float_storage(backend):
    values = np.array([[1.1, -4], [8.3, 2], [5.2, 7], [9.9, -1]], dtype=np.float64)
    errors = np.array([[0.13, 3], [1.7, 0.31], [0.77, 2.1], [3.3, 0.17]])
    groups = np.array([1, 0, 1, 0], dtype=np.int32)
    result, uncertainty = evaluate(backend, "weighted_reco_digis", values, errors,
                                   groups, n_groups=3)
    e = errors.astype(np.float32).astype(np.float64)
    v = values.astype(np.float32).astype(np.float64)
    w = (np.float64(1) / e) / e
    sums = np.stack([w[1] + w[3], w[0] + w[2]])
    weighted = np.stack([w[1] * v[1] + w[3] * v[3], w[0] * v[0] + w[2] * v[2]])
    assert result.dtype == uncertainty.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(result)[:2], (weighted / sums).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(uncertainty)[:2], (1 / np.sqrt(sums)).astype(np.float32))
    assert np.all(np.isnan(np.asarray(result)[2]))
    assert np.all(np.isposinf(np.asarray(uncertainty)[2]))


def test_weighted_broadcast_singleton_empty_and_no_selection(backend):
    result, error = evaluate(backend, "weighted_reco_digis", [3, 7], 2,
                             np.array([0, 1]), n_groups=2)
    np.testing.assert_array_equal(result, [3, 7])
    np.testing.assert_array_equal(error, [2, 2])
    result, error = evaluate(backend, "weighted_reco_digis", np.empty((0, 2)),
                             np.empty((0, 2)), np.empty(0, dtype=np.int32), n_groups=2)
    assert result.shape == error.shape == (2, 2)
    assert np.all(np.isnan(result)) and np.all(np.isposinf(error))
    result, error = evaluate(backend, "weighted_reco_digis", np.empty(0),
                             np.empty(0), np.empty(0, dtype=np.int32), n_groups=0)
    assert result.shape == error.shape == (0,)


@pytest.mark.parametrize("errors", [0, -1, np.nan, np.inf, -np.inf, 1e-100])
def test_weight_errors_host_guards(errors):
    with pytest.raises(ValueError, match="errors"):
        numpy_response.weighted_reco_digis([1], [errors], [0], 1)


@pytest.mark.parametrize("groups,n_groups", [([1], 1), ([-1], 1), ([0.0], 1), ([0], -1)])
def test_weight_membership_host_guards(groups, n_groups):
    with pytest.raises(ValueError):
        numpy_response.weighted_reco_digis([1], [1], groups, n_groups)


def test_additional_host_guards():
    with pytest.raises(ValueError, match="groups"):
        numpy_response.weighted_reco_digis([1, 2], 1, [0], 1)
    with pytest.raises(ValueError, match="values"):
        numpy_response.weighted_reco_digis([np.inf], 1, [0], 1)
    with pytest.raises(ValueError, match="crystal_length"):
        numpy_response.corrected_energy(1, 1, 1, 0, 1)
    with pytest.raises(ValueError, match="finite"):
        numpy_response.fold_step_time(np.nan, **TIMING)
    with pytest.raises(ValueError, match="positive"):
        numpy_response.fold_step_time(1, **dict(TIMING, event_length=0))


def test_jax_does_not_mutate_config_and_requires_source_double():
    jax = pytest.importorskip("jax")
    from calofastsim import response_jax
    original = jax.config.x64_enabled
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="x64"):
            response_jax.fold_step_time(0, **TIMING)
        with pytest.raises(ValueError, match="x64"):
            response_jax.weighted_reco_digis([1], [1], [0], 1)
        assert not jax.config.x64_enabled
    assert jax.config.x64_enabled == original


def test_very_late_finite_times_keep_double_fmod_not_float_or_clipping(backend):
    times = np.array([0., 2.**24 + .25, 6.359561891378894e14, 5.012094393544822e15], dtype=np.float64)
    options = dict(TIMING, pbtime=-.3, mbtime=1695., event_length=1795., onspill=True,
                   digitization_start=450., digitization_end=1795., digitization_buffer=10.,
                   time_from_protons_to_dr_marker=0.)
    pb = np.float64(np.float32(-.3))
    expected = np.fmod(times + pb, np.float64(np.float32(1695.))) - pb
    corrected_end = np.float64(np.float32(np.float32(1795.) - np.float32(-.3)))
    expected = np.where(expected < corrected_end - np.float64(1695.), expected + np.float64(1695.), expected)
    corrected_start = np.float32(np.float32(np.float32(450.) - np.float32(-.3)) - np.float32(10.))
    folded, accepted = evaluate(backend, "fold_step_time", times, **options)
    np.testing.assert_array_equal(folded, expected)
    np.testing.assert_array_equal(accepted, expected >= corrected_start)
    assert np.all(np.isfinite(folded))


def test_hand_derived_golden_values():
    """Values worked out by hand from the source formulas, independent of the implementation."""
    # Birks 9 MeV at z = 150 of 200 mm, LRU 0.05: 9 * (0.05 * (0.75 - 0.5) + 1) = 9.1125
    e = numpy_response.corrected_energy(10.0, 9.0, 150.0, 200.0, 0.05)
    assert e == np.float32(9.1125)
    # 30 PE/MeV on both SiPMs: 273.375 -> 273 PE; 0.5 rounds up (lround), 2.5 -> 3
    mean = numpy_response.photoelectron_mean(e, [30.0, 30.0])
    np.testing.assert_array_equal(mean, np.float32(273.375))
    np.testing.assert_array_equal(numpy_response.nonnegative_lround([273.375, 0.5, 2.5, 2.4999]), [273, 1, 3, 2])
    # pb -200, period 1695, window 450-1795, buffer 10, marker 200: corrected start 440, max hit time 1795
    # t = 1000: fmod(800, 1695) + 200 = 1000, accepted; t = 2000: fmod(1800, 1695) + 200 = 305 < 440, rejected;
    # t = 50: fmod(-150, 1695) + 200 = 50 < 1795 - 1695, shifted to 1745, accepted
    hit, ok = numpy_response.fold_step_time(np.array([1000.0, 2000.0, 50.0]), pbtime=-200.0, mbtime=1695.0,
                                            event_length=1700.0, onspill=True, digitization_start=450.0,
                                            digitization_end=1795.0, digitization_buffer=10.0,
                                            time_from_protons_to_dr_marker=200.0)
    np.testing.assert_array_equal(hit, [1000.0, 305.0, 1745.0])
    np.testing.assert_array_equal(ok, [True, False, True])
    # inverse-variance pair: values 10, 20 with errors 1, 2 -> (10 + 20/4) / 1.25 = 12, error 1/sqrt(1.25)
    v, err = numpy_response.weighted_reco_digis([10.0, 20.0], [1.0, 2.0], np.array([0, 0]), 1)
    np.testing.assert_allclose([v[0], err[0]], [12.0, 1.0 / np.sqrt(1.25)], rtol=1e-7)
