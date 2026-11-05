"""Synthetic stock-algorithm grouping contracts, not a fitted-pulse oracle."""
import numpy as np
import pytest

from calofastsim.hits import make_hits


def run(times, sipm=None, events=None, **options):
    n = len(times)
    return make_hits(np.zeros(n, dtype=int) if events is None else np.asarray(events),
                     np.zeros(n, dtype=int) if sipm is None else np.asarray(sipm),
                     times, np.ones(n), np.arange(n) + 1, np.ones(n),
                     n_events=options.get("n_events", 1), n_crystals=options.get("n_crystals", 2),
                     time4merge=options.get("time4merge", 6.))


def test_split_is_relative_to_first_not_previous_and_singleton_survives():
    result = run([0., 4., 8.], [0, 1, 0])
    np.testing.assert_array_equal(result["member_offsets"], [0, 2, 3])
    np.testing.assert_array_equal(result["nSiPMs"], [2, 1])
    np.testing.assert_array_equal(result["time"], [2., 8.])
    np.testing.assert_array_equal(result["energy"], [1.5, 3.])


def test_same_sensor_is_not_deduplicated_and_exact_window_equality_merges():
    result = run([6., 0.], [1, 1])
    np.testing.assert_array_equal(result["member_index"], [1, 0])
    np.testing.assert_array_equal(result["nSiPMs"], [2])
    np.testing.assert_array_equal(result["time"], [3.])


def test_crystal_total_multiplicity_and_event_isolation():
    result = run([0., 0., 20., 1., 1.], [0, 2, 0, 0, 2], [0, 0, 0, 1, 1], n_events=2)
    np.testing.assert_array_equal(result["event_index"], [0, 0])
    np.testing.assert_array_equal(result["crystal"], [0, 0])
    np.testing.assert_array_equal(result["nSiPMs"], [1, 1])
    np.testing.assert_array_equal(result["member_index"], [0, 2])


def test_hit_order_is_event_then_crystal_then_first_time():
    result = run([20., 0., 1., 30., 2., 1.], [2, 0, 1, 3, 1, 0], [0, 1, 1, 0, 0, 0], n_events=2)
    np.testing.assert_array_equal(result["event_index"], [0, 0, 0, 1])
    np.testing.assert_array_equal(result["crystal"], [0, 1, 1, 0])
    np.testing.assert_array_equal(result["nSiPMs"], [2, 1, 1, 2])
    np.testing.assert_array_equal(result["member_index"], [5, 4, 0, 3, 1, 2])


def test_ties_are_explicit_and_empty_input_is_valid():
    result = run([1., 1., 10.], [0, 1, 0])
    np.testing.assert_array_equal(result["equal_time_order_unspecified"], [True, False])
    result = run([])
    np.testing.assert_array_equal(result["member_offsets"], [0])
    assert result["time"].size == 0
    np.testing.assert_array_equal(run([1.])["member_offsets"], [0])


def test_float32_difference_precedes_double_window_comparison():
    result = run([1., 1. + 1.e-8], time4merge=0.)
    np.testing.assert_array_equal(result["nSiPMs"], [2])
    np.testing.assert_array_equal(result["equal_time_order_unspecified"], [True])


@pytest.mark.parametrize("options", [dict(time4merge=-1.), dict(time4merge=np.nan),
                                     dict(n_events=0), dict(n_crystals=0)])
def test_bad_geometry_or_window_refused(options):
    with pytest.raises(ValueError):
        run([1., 2.], **options)


def test_invalid_identifiers_and_fit_errors_refused():
    with pytest.raises(ValueError, match="integer"):
        run([1., 2.], sipm=[0., 1.])
    with pytest.raises(ValueError, match="outside"):
        run([1., 2.], sipm=[0, 4])
    with pytest.raises(ValueError, match="positive"):
        make_hits([0, 0], [0, 1], [1., 2.], [0., 1.], [1., 2.], [1., 1.],
                  n_events=1, n_crystals=2, time4merge=6.)


def test_jax_rounds_match_numpy():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(9)
    n = 400
    args = (rng.integers(0, 5, n), rng.integers(0, 40, n), rng.uniform(0, 40, n).astype(np.float32),
            rng.uniform(0.1, 1, n), rng.uniform(1, 50, n), rng.uniform(0.1, 2, n))
    kw = dict(n_events=5, n_crystals=20, time4merge=6.0)
    a = make_hits(*args, **kw)
    b = make_hits(*args, **kw, backend="jax")
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
