"""Diagnostic timing boundaries, plus production gate and synthetic waveform contracts."""
import math

import awkward as ak
import numpy as np
import pytest

from crvfastsim.reco import waveform
from crvfastsim.veto import apply, replay_timing as timing
from crvfastsim.veto.card import VetoCard


PBTIMES = [-224.46, -219.46, -214.46, -209.46, -204.46]


def test_wrap_arrivals_negative_multiperiod_and_broadcast_origins():
    offsets = np.array([-3415.0, -1695.0, -25.0, 0.0, 1695.0, 3415.0])
    origins = np.array([0.0, 224.4600067138672])[:, None]
    expected = origins + np.array([1670.0, 0.0, 1670.0, 0.0, 0.0, 25.0])
    np.testing.assert_array_equal(timing.wrap_arrivals(origins + offsets, origins), expected)
    np.testing.assert_array_equal(timing.wrap_arrivals([-21.0, -1.0, 20.0], period=10.0), [9.0, 9.0, 0.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["time", "origin"])
def test_wrap_arrivals_rejects_nonfinite(field, bad):
    kwargs = {"time": [0.0, 200.0], "origin": 0.0}
    kwargs[field] = [0.0, bad]
    with pytest.raises(ValueError):
        timing.wrap_arrivals(**kwargs)


@pytest.mark.parametrize("period", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_wrap_and_step_reject_invalid_period(period):
    with pytest.raises(ValueError):
        timing.wrap_arrivals([200.0], period=period)
    with pytest.raises(ValueError):
        timing.stock_step_mask([np.nan], [np.nan], period=period)


@pytest.mark.parametrize(
    "t0,t1,expected",
    [(50.0, 50.0, False), (50.0, 500.0, True), (50.0, 1600.0, True),
     (500.0, 50.0, True), (500.0, 500.0, True), (500.0, 1600.0, True),
     (1600.0, 50.0, False), (1600.0, 500.0, True), (1600.0, 1600.0, False)],
)
def test_stock_step_three_clause_truth_table(t0, t1, expected):
    assert bool(timing.stock_step_mask(t0, t1)) is expected


@pytest.mark.parametrize("origin", [0.0, 224.4600067138672])
def test_stock_step_strict_boundaries_including_wrap_clause(origin):
    lo, hi = origin + 100.0, origin + 1500.0
    before, after = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
    t0 = [before, lo, lo, hi, after, after, hi, after]
    t1 = [before, before, lo, after, hi, after, before, lo]
    expected = [False, True, True, True, True, False, True, True]
    np.testing.assert_array_equal(timing.stock_step_mask(t0, t1, origin), expected)
    assert not timing.stock_step_mask(after, before, origin)


@pytest.mark.parametrize("origin", [0.0, 224.4600067138672])
def test_stock_step_weird_wrap_and_long_spans_are_not_interval_overlap(origin):
    t0 = origin + np.array([1600.0, 50.0, 50.0, 1600.0, 500.0])
    t1 = origin + np.array([1745.0, 1600.0, 3440.0, 4990.0, 3890.0])
    np.testing.assert_array_equal(timing.stock_step_mask(t0, t1, origin), [False, True, False, False, True])
    assert t0[2] < origin + 500.0 < t1[2]


def test_stock_step_nan_endpoints_skip_without_poisoning_valid_rows():
    t0 = [np.nan, 500.0, np.nan, 500.0]
    t1 = [500.0, np.nan, np.nan, 600.0]
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(timing.stock_step_mask(t0, t1), [False, False, False, True])


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
@pytest.mark.parametrize("other", [500.0, np.nan])
def test_stock_step_infinite_endpoint_fails_even_with_nan_partner(bad, other):
    for endpoints in ((bad, other), (other, bad)):
        with pytest.raises(ValueError):
            timing.stock_step_mask(*endpoints)


@pytest.mark.parametrize("kwargs", [
    {"origin": np.nan}, {"origin": np.inf}, {"origin": -np.inf},
    {"start": np.nan}, {"end": np.inf}, {"margin": np.nan},
    {"margin": -1.0}, {"start": 1501.0, "end": 1500.0},
])
def test_stock_step_invalid_interval_or_origin(kwargs):
    with pytest.raises(ValueError):
        timing.stock_step_mask(500.0, 600.0, **kwargs)


def test_arrival_arms_preserve_one_draw_and_wrap_before_upper_cut():
    step = np.array([1, 0, 1, 0, 0, 1, 2, 2])
    times = np.array([-25.0, 100.0, 1500.0, np.nextafter(1500.0, np.inf),
                      1695.0, 1795.0, 3590.0, 6685.0])
    accepted = np.array([False, True, True])
    original = times.copy()
    arms = timing.arrival_arms(step, times, accepted)
    assert set(arms) == {"step_only", "arrival_only", "stock_timing"}
    expected_folded = [1670.0, 100.0, 1500.0, times[3], 0.0, 100.0, 200.0, 1600.0]
    expected_masks = {
        "step_only": [True, True, True, False, False, False, False, False],
        "arrival_only": [False, True, True, False, True, True, True, False],
        "stock_timing": [False, False, True, False, False, True, True, False],
    }
    for name, (arm_times, mask) in arms.items():
        np.testing.assert_array_equal(arm_times, original if name == "step_only" else expected_folded)
        np.testing.assert_array_equal(mask, expected_masks[name])
        assert arm_times.shape == mask.shape == step.shape
    np.testing.assert_array_equal(times, original)
    np.testing.assert_array_equal(step, [1, 0, 1, 0, 0, 1, 2, 2])
    np.testing.assert_array_equal(accepted, [False, True, True])
    assert not np.shares_memory(arms["step_only"][0], times)


def test_arrivals_wrap_after_interpolation_and_negative_multiperiod_optical_delays():
    t0, t1, fraction = 1600.0, 1795.0, 0.5
    emission = t0 + fraction * (t1 - t0)
    delays = np.array([-3440.0, -5.0, 0.0, 3415.0, 6685.0])
    raw = emission + delays
    arms = timing.arrival_arms(np.zeros(5, dtype=np.int64), raw, [True])
    expected = [1647.5, 1692.5, 2.5, 27.5, 1602.5]
    np.testing.assert_array_equal(arms["step_only"][0], raw)
    for name in ("arrival_only", "stock_timing"):
        np.testing.assert_array_equal(arms[name][0], expected)
        np.testing.assert_array_equal(arms[name][1], [False, False, True, True, False])
    wrapped_endpoints = timing.wrap_arrivals([t0, t1])
    wrong_emission = wrapped_endpoints[0] + fraction * np.diff(wrapped_endpoints)[0]
    assert wrong_emission == 850.0
    assert arms["arrival_only"][0][2] == 2.5
    assert arms["arrival_only"][0][2] != wrong_emission


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_arrival_arms_reject_nonfinite_photons_even_for_rejected_steps(bad):
    with pytest.raises(ValueError):
        timing.arrival_arms(np.array([0]), [bad], [False])
    with pytest.raises(ValueError):
        timing.arrival_arms(np.array([0]), [100.0], [True], end=bad)


@pytest.mark.parametrize("step,time,accepted", [
    ([0], [100.0, 200.0], [True]),
    ([[0]], [[100.0]], [True]),
    ([0], [100.0], [[True]]),
    (np.array([0.0]), [100.0], [True]),
    (np.array([True]), [100.0], [True]),
    ([-1], [100.0], [True]),
    ([1], [100.0], [True]),
    ([0], [100.0], []),
])
def test_arrival_arms_reject_misaligned_or_invalid_step_indices(step, time, accepted):
    with pytest.raises(ValueError):
        timing.arrival_arms(np.asarray(step), time, accepted)


def test_empty_diagnostic_inputs():
    assert timing.wrap_arrivals([]).shape == (0,)
    assert timing.stock_step_mask([], []).shape == (0,)
    assert timing.digi_start_mask([]).shape == (0,)
    arms = timing.arrival_arms(np.array([], dtype=np.int64), [], [])
    for times, mask in arms.values():
        assert times.shape == mask.shape == (0,)
        assert mask.dtype == np.bool_


@pytest.mark.parametrize("pbtime", PBTIMES)
def test_float32_pbtime_absolute_boundaries(pbtime):
    origin = -float(np.float32(pbtime))
    assert origin != -pbtime
    lo, hi = origin + 200.0, origin + 1500.0
    starts = [np.nextafter(lo, -np.inf), lo, np.nextafter(lo, np.inf),
              np.nextafter(hi, -np.inf), hi, np.nextafter(hi, np.inf)]
    np.testing.assert_array_equal(timing.digi_start_mask(starts, origin), [False, True, True, True, True, False])
    np.testing.assert_array_equal(timing.stock_step_mask(starts, starts, origin), [True, True, True, True, True, False])
    assert not timing.digi_start_mask(-pbtime + 200.0, origin)
    assert timing.digi_start_mask(hi, -pbtime) == np.bool_(False)


def test_boundary_table_float32_phases_and_corrected_fmod_operation_order():
    table = timing.boundary_table()
    assert len(table) == 5 * 7 * 3
    np.testing.assert_array_equal(table.w.unique(), -np.asarray(PBTIMES, np.float32).astype(float))
    assert not table.duplicated(["w", "edge_relative", "epsilon_direction"]).any()
    for row in table.itertuples(index=False):
        edge = row.w + row.edge_relative
        expected_raw = edge if row.epsilon_direction == 0 else np.nextafter(edge, row.epsilon_direction * np.inf)
        assert row.raw_time == expected_raw
        expected_folded = math.fmod(expected_raw - row.w, 1695.0) + row.w
        if expected_folded < row.w:
            expected_folded += 1695.0
        assert row.wrapped_absolute == expected_folded
        assert row.zero_length_step_kept == (row.w + 100.0 <= expected_folded <= row.w + 1500.0)
        assert row.sipm_upper_kept == (expected_folded <= row.w + 1500.0)
        assert row.digi_start_kept == (row.w + 200.0 <= expected_raw <= row.w + 1500.0)
    exact = table[table.epsilon_direction == 0]
    np.testing.assert_array_equal(exact.zero_length_step_kept, np.tile([False, False, True, True, True, False, False], 5))
    np.testing.assert_array_equal(exact.digi_start_kept, np.tile([False, False, False, True, True, False, False], 5))


def test_digi_start_mask_inclusive_and_last_sample_may_exceed_end():
    starts = np.array([np.nextafter(200.0, -np.inf), 200.0, 1500.0, np.nextafter(1500.0, np.inf)])
    np.testing.assert_array_equal(timing.digi_start_mask(starts), [False, True, True, False])
    assert 1500.0 + 11 * 12.5 > 1500.0
    origins = np.array([0.0, 224.4600067138672])[:, None]
    np.testing.assert_array_equal(timing.digi_start_mask(origins + [200.0, 1500.0], origins), np.ones((2, 2), bool))


@pytest.mark.parametrize("kwargs", [
    {"start": np.nan}, {"start": np.inf}, {"start": -np.inf},
    {"origin": np.nan}, {"origin": np.inf}, {"origin": -np.inf},
    {"lo": np.nan}, {"hi": np.inf}, {"lo": 1501.0},
])
def test_digi_start_mask_rejects_nonfinite_or_reversed_window(kwargs):
    with pytest.raises(ValueError):
        timing.digi_start_mask(**({"start": 500.0} | kwargs))


def test_actual_event_gate_end_1700_is_exclusive_and_all_coincidences_must_pass():
    before = np.nextafter(1700.0, -np.inf)
    after = np.nextafter(1700.0, np.inf)
    ends = [[before], [1700.0], [after], [], [before, 1700.0], [before, before]]
    coincs = ak.zip({"time": ak.Array([[800.0] * len(event) for event in ends]),
                     "timeStart": ak.Array([[700.0] * len(event) for event in ends]),
                     "timeEnd": ak.Array(ends)})
    card = VetoCard(gate_start_lo=600.0, gate_end_hi=1700.0)
    np.testing.assert_array_equal(apply.gate(coincs, card), [True, False, False, True, False, True])


@pytest.fixture
def synthetic_waveform(monkeypatch):
    cfg = waveform.WaveformConfig(noise=0.0)
    monkeypatch.setattr(waveform, "single_pe_template", lambda *args: np.full(200, 0.1))

    def synthesize(charge_time, window):
        return waveform.synthesize(np.array([0]), np.array([charge_time]),
                                   np.array([cfg.single_pe_reference_charge]), np.array([0.0]),
                                   cfg=cfg, window=window)

    return synthesize


@pytest.mark.parametrize("endpoint,window", [(200.0, (200.0, 1500.0)), (1500.0, (200.0, 1500.0))])
def test_actual_waveform_start_window_endpoints_inclusive(synthetic_waveform, endpoint, window):
    result = synthetic_waveform(endpoint + 12.5, window)
    np.testing.assert_array_equal(result["start"], [endpoint])
    assert result["volt"].shape == result["adc"].shape == (1, 12)
    assert result["starttdc"][0] == int(endpoint / 12.5)


def test_actual_waveform_window_nextafter_rejects_boundary_start(synthetic_waveform):
    lower = synthetic_waveform(212.5, (np.nextafter(200.0, np.inf), 1500.0))
    np.testing.assert_array_equal(lower["start"], [212.5])
    upper = synthetic_waveform(1512.5, (200.0, np.nextafter(1500.0, -np.inf)))
    assert len(upper["start"]) == 0


def test_actual_waveform_keeps_samples_past_window_end(synthetic_waveform):
    result = synthetic_waveform(1512.5, (200.0, 1500.0))
    np.testing.assert_array_equal(result["start"], [1500.0])
    sample_times = result["start"][0] + np.arange(12) * 12.5
    assert sample_times[-1] == 1637.5
    np.testing.assert_array_equal(result["volt"][0, 1:9], np.full(8, 0.1))
    assert np.all(sample_times[1:9] > 1500.0)
    assert np.all(result["adc"][0, 1:9] > 100)
