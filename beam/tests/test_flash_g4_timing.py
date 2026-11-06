"""Conservative brackets from synthetic product-qualified exporter records."""
import numpy as np
import pytest

from beamfastsim.flash_g4_timing import (
    FIELDS, PHOTON_FIELDS, crossing_intervals, float32_interval,
)


def fixture(times=(1., 2., 3.), *, two=False, end=4., birth=0.):
    n = len(times)
    particles = {field: np.ones(1, np.uint64) for field in FIELDS["particles"]}
    particles.update(time=np.array([birth], np.float32), end_time=np.array([end], np.float32))
    steps = {field: np.ones(n, np.uint64) for field in FIELDS["steps"]}
    steps.update(product=np.full(n, 2, np.uint64), index=np.arange(n, dtype=np.uint64),
                 time=np.asarray(times, np.float64))
    photons = {field: np.ones(1, np.uint64) for field in PHOTON_FIELDS}
    photons.update(first_step_product=np.array([2], np.uint64),
                   first_stepindex=np.array([0], np.uint64),
                   continuation_stepindex=np.array([1], np.uint64),
                   two_segment_reconstruction=np.array([two]), ambiguous=np.array([False]))
    return {"particles": particles, "steps": steps}, photons


def assert_invalid(result):
    assert not result["valid"].any()
    assert np.isnan(result["lower_ns"]).all()
    assert np.isnan(result["upper_ns"]).all()
    assert not result["upper_source"].any()


def test_rounding_asymmetric_at_positive_and_negative_powers_of_two():
    values = np.array([1., 2., 1024., -1., -2., -1024.], np.float32)
    lo, hi = float32_interval(values)
    exact = values.astype(np.float64)
    expected_lo = (np.nextafter(values, np.float32(-np.inf)).astype(np.float64) + exact) / 2
    expected_hi = (np.nextafter(values, np.float32(np.inf)).astype(np.float64) + exact) / 2
    np.testing.assert_array_equal(lo, expected_lo)
    np.testing.assert_array_equal(hi, expected_hi)
    np.testing.assert_array_equal(hi[:3] - exact[:3], 2 * (exact[:3] - lo[:3]))
    np.testing.assert_array_equal(exact[3:] - lo[3:], 2 * (hi[3:] - exact[3:]))
    assert lo.dtype == hi.dtype == np.dtype(np.float64)


def test_rounding_zero_subnormal_and_largest_finite():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    largest = np.finfo(np.float32).max
    values = np.array([0., -0., tiny, -tiny, largest, -largest], np.float32)
    lo, hi = float32_interval(values)
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert lo[0] == lo[1] == -float(tiny) / 2
    assert hi[0] == hi[1] == float(tiny) / 2
    assert lo[2] == float(tiny) / 2 and hi[2] == 1.5 * float(tiny)
    assert hi[4] == (float(largest) + 2.**128) / 2
    assert lo[5] == -hi[4]


def test_only_exact_widened_float32_is_accepted():
    exact = float(np.float32(.1))
    values = np.array([exact, .1, np.nextafter(exact, np.inf), np.inf, -np.inf, np.nan, 1e300])
    lo, hi = float32_interval(values)
    assert np.isfinite(lo[0]) and lo[0] < exact < hi[0]
    assert np.isnan(lo[1:]).all() and np.isnan(hi[1:]).all()
    assert float32_interval(np.array(1., np.float32))[0].shape == ()
    assert float32_interval(np.array([], np.float64))[0].shape == (0,)


@pytest.mark.parametrize("values", [[1, 2], [True], ["1"], [1 + 0j], np.array([1.], np.float16)])
def test_rounding_rejects_non_float32_float64_dtype(values):
    with pytest.raises(ValueError, match="floating dtype"):
        float32_interval(values)


def test_ordinary_uses_raw_pre_time_not_npz_or_flight_diagnostic():
    tables, photons = fixture(times=(1.123456789, 2.987654321))
    photons.update(first_pre_time_ns=np.array([-100.]), segmentflight_ns=np.array([900.]))
    result = crossing_intervals(tables, photons)
    assert result["valid"].tolist() == [True]
    assert result["lower_ns"].tolist() == [1.123456789]
    assert result["upper_ns"].tolist() == [2.987654321]
    assert result["upper_source"].tolist() == [1]


def test_two_segment_uses_step_after_continuation():
    tables, photons = fixture(two=True)
    result = crossing_intervals(tables, photons)
    assert result["valid"].all()
    assert result["lower_ns"].tolist() == [1.]
    assert result["upper_ns"].tolist() == [3.]
    assert result["upper_source"].tolist() == [1]


@pytest.mark.parametrize("two,times", [(False, (1.,)), (True, (1., 2.))])
def test_defined_end_fallback_includes_rounding_midpoint(two, times):
    tables, photons = fixture(times=times, two=two, end=2.)
    result = crossing_intervals(tables, photons)
    assert result["valid"].all()
    assert result["lower_ns"].tolist() == [1.]
    assert result["upper_ns"][0] == float32_interval(np.array([2.], np.float32))[1][0]
    assert result["upper_source"].tolist() == [2]


@pytest.mark.parametrize("times,end", [((1., 1.), 1.), ((1.,), 1.)])
def test_equal_stored_times_are_not_reversed(times, end):
    tables, photons = fixture(times=times, end=end)
    result = crossing_intervals(tables, photons)
    assert result["valid"].all()
    assert result["upper_ns"][0] >= result["lower_ns"][0]


@pytest.mark.parametrize("end,defined", [(np.inf, 1), (-np.inf, 1), (np.nan, 1), (0., 1), (4., 0)])
def test_invalid_or_missing_end_cannot_bound_last_step(end, defined):
    tables, photons = fixture(times=(1.,), end=end)
    tables["particles"]["end_defined"][:] = defined
    assert_invalid(crossing_intervals(tables, photons))


def test_non_float32_end_and_birth_are_invalid_not_silently_rounded():
    tables, photons = fixture(times=(1.,))
    tables["particles"]["end_time"] = np.array([1.1], np.float64)
    assert_invalid(crossing_intervals(tables, photons))
    tables, photons = fixture()
    tables["particles"]["time"] = np.array([.1], np.float64)
    assert_invalid(crossing_intervals(tables, photons))


def test_next_step_does_not_require_end_defined_or_finite_end():
    tables, photons = fixture(end=np.nan)
    tables["particles"]["end_defined"][:] = 0
    result = crossing_intervals(tables, photons)
    assert result["valid"].all() and result["upper_source"].tolist() == [1]


@pytest.mark.parametrize("times", [(2., 1., 3.), (1., 3., 2.), (np.nan, 2.), (1., np.inf), (1., 2., np.nan)])
def test_nonmonotonic_or_nonfinite_sequence_invalidates_without_fallback(times):
    tables, photons = fixture(times=times)
    assert_invalid(crossing_intervals(tables, photons))


def test_preceding_nonmonotonic_step_invalidates_selected_later_crossing():
    tables, photons = fixture(times=(2., 1., 3., 4.))
    photons["first_stepindex"][:] = 2
    assert_invalid(crossing_intervals(tables, photons))


@pytest.mark.parametrize("field,value", [("first_step_product", 3), ("first_stepindex", 99),
                                         ("product", 99), ("key", 99), ("event", 99),
                                         ("run", 99), ("subrun", 99)])
def test_mismatched_photon_full_identity_is_invalid(field, value):
    tables, photons = fixture()
    photons[field][:] = value
    assert_invalid(crossing_intervals(tables, photons))


@pytest.mark.parametrize("field,value", [("particle_product", 3), ("particle_key", 99),
                                         ("particle_state", 0), ("particle_state", 2),
                                         ("particle_state", 3)])
def test_wrong_first_step_pointer_is_invalid(field, value):
    tables, photons = fixture()
    tables["steps"][field][0] = value
    assert_invalid(crossing_intervals(tables, photons))


@pytest.mark.parametrize("index", [0, 2, 99])
def test_continuation_must_be_immediate_same_photon_same_product_step(index):
    tables, photons = fixture(two=True)
    photons["continuation_stepindex"][:] = index
    assert_invalid(crossing_intervals(tables, photons))


def test_continuation_with_wrong_pointer_is_invalid():
    tables, photons = fixture(two=True)
    tables["steps"]["particle_key"][1] = 99
    assert_invalid(crossing_intervals(tables, photons))


@pytest.mark.parametrize("tree", ["particles", "steps"])
def test_duplicate_raw_full_keys_raise(tree):
    tables, photons = fixture()
    tables[tree] = {field: np.concatenate([value, value[:1]]) for field, value in tables[tree].items()}
    with pytest.raises(ValueError, match="Duplicate/ambiguous"):
        crossing_intervals(tables, photons)


@pytest.mark.parametrize("flag,value", [("exit", False), ("exit_candidate", False), ("ambiguous", True)])
def test_no_exit_or_ambiguous_spatial_exit_is_invalid(flag, value):
    tables, photons = fixture()
    photons[flag][:] = value
    assert_invalid(crossing_intervals(tables, photons))


def test_no_steps_no_particles_and_no_photons():
    tables, photons = fixture(times=())
    assert_invalid(crossing_intervals(tables, photons))
    tables["particles"] = {field: value[:0] for field, value in tables["particles"].items()}
    assert_invalid(crossing_intervals(tables, photons))
    photons = {field: value[:0] for field, value in photons.items()}
    result = crossing_intervals(tables, photons)
    assert all(value.shape == (0,) for value in result.values())


def test_unsorted_full_uint64_keys_product_and_event_collisions():
    base = np.uint64(2**63 + 100)
    pieces = [fixture(times=(float(i + 1), float(i + 2))) for i in range(5)]
    for i, (tables, photons) in enumerate(pieces):
        for table in (tables["particles"], tables["steps"], photons):
            table["run"][:] = base + np.uint64(i == 3)
            table["subrun"][:] = base + np.uint64(i == 4)
            table["event"][:] = base + np.uint64(i == 2)
        tables["particles"]["product"][:] = photons["product"][:] = base + np.uint64(i == 1)
        tables["particles"]["key"][:] = photons["key"][:] = np.iinfo(np.uint64).max
        tables["steps"]["particle_product"][:] = tables["particles"]["product"][0]
        tables["steps"]["particle_key"][:] = np.iinfo(np.uint64).max
        tables["steps"]["product"][:] = photons["first_step_product"][:] = base + np.uint64(10 + (i == 1))
        tables["steps"]["index"][:] = [np.iinfo(np.uint64).max - 1, np.iinfo(np.uint64).max]
        photons["first_stepindex"][:] = np.iinfo(np.uint64).max - 1
    tables = {tree: {field: np.concatenate([piece[0][tree][field] for piece in pieces])
                     for field in FIELDS[tree]} for tree in FIELDS}
    photons = {field: np.concatenate([piece[1][field] for piece in pieces]) for field in PHOTON_FIELDS}
    tables["particles"] = {field: value[::-1] for field, value in tables["particles"].items()}
    tables["steps"] = {field: value[[7, 0, 9, 4, 1, 8, 5, 3, 6, 2]] for field, value in tables["steps"].items()}
    result = crossing_intervals(tables, photons)
    assert result["valid"].all()
    np.testing.assert_array_equal(result["lower_ns"], np.arange(1., 6.))
    np.testing.assert_array_equal(result["upper_ns"], np.arange(2., 7.))
    np.testing.assert_array_equal(result["upper_source"], np.ones(5))


def test_other_step_product_is_never_used_as_next_step():
    tables, photons = fixture(times=(1., 2.), end=3.)
    tables["steps"]["product"][1] = 3
    result = crossing_intervals(tables, photons)
    assert result["valid"].all()
    assert result["upper_source"].tolist() == [2]
    assert result["upper_ns"][0] > 3.


def test_birth_interval_consistency_and_end_after_continuation():
    tables, photons = fixture(birth=2.)
    assert_invalid(crossing_intervals(tables, photons))
    tables, photons = fixture(times=(1., 3.), two=True, end=2.)
    assert_invalid(crossing_intervals(tables, photons))


def test_malformed_reconstruction_flag_cannot_choose_ordinary_upper():
    tables, photons = fixture()
    photons["two_segment_reconstruction"] = np.array([2], np.uint32)
    assert_invalid(crossing_intervals(tables, photons))


def test_interleaved_other_photon_steps_are_skipped_without_float_key_promotion():
    tables, photons = fixture(times=(1., 10., 2.))
    tables["particles"] = {field: np.repeat(value, 2) for field, value in tables["particles"].items()}
    tables["particles"]["key"] = np.array([2**63, 2**63 + 1], np.uint64)
    photons["key"][:] = np.uint64(2**63)
    tables["steps"]["particle_key"] = np.array([2**63, 2**63 + 1, 2**63], np.uint64)
    result = crossing_intervals(tables, photons)
    assert result["valid"].all()
    assert result["upper_ns"].tolist() == [2.]
    tables["steps"]["particle_key"][0] = np.uint64(2**63 + 1)
    assert_invalid(crossing_intervals(tables, photons))


def test_missing_fields_bad_shapes_and_nonfloating_step_times_raise():
    tables, photons = fixture()
    del tables["particles"]["end_defined"]
    with pytest.raises(ValueError, match="missing required"):
        crossing_intervals(tables, photons)
    tables, photons = fixture()
    tables["steps"]["time"] = np.ones((3, 1))
    with pytest.raises(ValueError, match="one-dimensional"):
        crossing_intervals(tables, photons)
    tables, photons = fixture()
    tables["steps"]["time"] = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="floating dtype"):
        crossing_intervals(tables, photons)
