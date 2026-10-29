"""Cell admission must happen before either diagnostic or production dense synthesis."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import bounded_waveform as B, replay_waveform_control as W
from test_veto_replay_waveform_control import channels, synthetic, charges


def test_guarded_synthesis_identical_and_bound_covers_real_allocations(channels, synthetic, charges):
    cfg, _ = synthetic
    expected = W.synthesize(charges, channels, 71, cfg)
    actual, receipt = B.synthesize(charges, channels, 71, cfg)
    assert receipt["charge_reach_cells"] == len(charges) * 10
    assert receipt["waveform_cells_upper"] >= len(actual[1])
    assert receipt["peak_memory_certified"] is False
    for key, value in actual[0].items():
        np.testing.assert_array_equal(value, expected[0][key])
    for a, b in zip(actual[1:], expected[1:]):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.parametrize("resource,cap", [("charges", 1), ("reach_cells", 1), ("waveform_cells", 1), ("max_length", 12), ("template_cells", 1)])
def test_reject_before_unbounded_synthesis(monkeypatch, channels, synthetic, charges, resource, cap):
    cfg, _ = synthetic
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("synthesis reached before cap"))
    with pytest.raises(ValueError, match="limit exceeded"):
        B.synthesize(charges, channels, 71, cfg, limits=replace(B.Limits(), **{resource: cap}))


@pytest.mark.parametrize("case", ["negative", "nonfinite", "missing", "huge_time", "bad_limit"])
def test_invalid_inputs(channels, synthetic, charges, case):
    cfg, _ = synthetic
    q, c, limits = charges.copy(), channels.copy(), B.Limits()
    if case == "negative":
        q.loc[0, "charge"] = -1
    elif case == "nonfinite":
        q.loc[0, "time"] = np.inf
    elif case == "missing":
        c = c[c.chan != 0]
    elif case == "huge_time":
        q.loc[0, "time"] = 2**50
    else:
        limits = replace(limits, reach_cells=True)
    with pytest.raises(ValueError):
        B.admit(q, c, 71, cfg, limits=limits)


def test_empty_charges_keeps_quiet_phases(channels, synthetic, charges):
    cfg, _ = synthetic
    products, receipt = B.synthesize(charges.iloc[:0], channels, 71, cfg)
    assert receipt["waveform_cells_upper"] == receipt["charge_reach_cells"] == 0
    assert len(products[3]) == len(channels)


def test_many_clusters_bounds_padded_matrix_not_sum_of_lengths(channels, synthetic):
    cfg, _ = synthetic
    q = pd.DataFrame({"event": 1, "chan": 1, "time": np.r_[np.arange(300., 650., 50.), 900., 1200.]})
    q["charge"] = cfg.single_pe_reference_charge
    products, receipt = B.synthesize(q, channels, 71, cfg)
    assert receipt["clusters"] == 3
    assert receipt["waveform_cells_upper"] == 3 * receipt["waveform_length_upper"]
    assert receipt["waveform_cells_upper"] >= len(products[1])


@pytest.mark.parametrize("max_time,precision,limits", [
    (1e9, 1e9, B.Limits()),
    (100., .5, replace(B.Limits(), reach_cells=9)),
    (100., .5, replace(B.Limits(), max_length=9)),
    (20_000., 20_000., replace(B.Limits(), max_length=100_000)),
])
def test_empty_charges_cannot_bypass_reach_cap(monkeypatch, channels, synthetic, charges, max_time, precision, limits):
    cfg, _ = synthetic
    cfg = replace(cfg, single_pe_max_time=max_time, single_pe_precision=precision, samples_zs=1)
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("unbounded helper reached"))
    monkeypatch.setattr(B.WF, "single_pe_template", lambda *a: pytest.fail("template reached before reach cap"))
    with pytest.raises(ValueError, match="reach cell limit"):
        B.synthesize(charges.iloc[:0], channels, 71, cfg, limits=limits)


@pytest.mark.parametrize("field,value", [
    ("noise", np.nan), ("noise", np.inf), ("noise", -1.), ("noise", True),
    ("min_voltage", np.nan), ("min_voltage", np.inf), ("min_voltage", -1.),
    ("adc_conversion", np.nan), ("adc_conversion", np.inf), ("adc_conversion", 0.),
    ("adc_conversion", -1.), ("adc_conversion", True), ("adc_conversion", "2300"),
    ("min_adc", -1), ("max_adc", 4096), ("max_adc", 99), ("min_adc", 101),
    ("pedestal", -1), ("pedestal", 4096), ("pedestal", 100.), ("pedestal", True),
    ("min_adc", 0.), ("max_adc", np.nan), ("single_pe_precision", True),
    ("single_pe_reference_charge", 0.),
])
def test_bad_configuration_rejected_before_helper(monkeypatch, channels, synthetic, charges, field, value):
    cfg, _ = synthetic
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("helper reached before config validation"))
    with pytest.raises(ValueError):
        B.synthesize(charges, channels, 71, replace(cfg, **{field: value}))


@pytest.mark.parametrize("window", [
    (np.nan, 1500.), (200., np.nan), (-np.inf, 1500.), (200., np.inf),
    (1500., 200.), (200.,), (200., 1500., 2000.), (True, 1500.), ("200", "1500"), None,
])
def test_bad_window_rejected_before_helper(monkeypatch, channels, synthetic, charges, window):
    cfg, _ = synthetic
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("helper reached before window validation"))
    with pytest.raises(ValueError, match="window"):
        B.synthesize(charges, channels, 71, cfg, window=window)


@pytest.mark.parametrize("target,column,dtype", [
    ("charges", "event", bool), ("charges", "chan", bool),
    ("charges", "time", bool), ("charges", "charge", bool),
    ("charges", "time", str), ("charges", "charge", object),
    ("charges", "charge", complex), ("charges", "charge", np.float32),
    ("charges", "time", np.int8), ("charges", "time", np.int32),
    ("channels", "event", bool), ("channels", "chan", bool),
    ("channels", "roc", bool), ("channels", "feb", bool),
    ("channels", "roc", float), ("channels", "feb", object),
])
def test_invalid_numeric_dtypes_rejected_before_helper(monkeypatch, channels, synthetic, charges, target, column, dtype):
    cfg, _ = synthetic
    q, c = charges.copy(), channels.copy()
    frame = q if target == "charges" else c
    frame[column] = frame[column].astype(dtype)
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("helper reached before dtype validation"))
    with pytest.raises(ValueError, match="dtype|float64"):
        B.synthesize(q, c, 71, cfg)


@pytest.mark.parametrize("case", ["ratio_overflow", "ratio_cap", "adc_wrap", "cluster_sum", "noise", "conversion", "negative", "integer_time"])
def test_extreme_ranges_rejected_before_helper(monkeypatch, channels, synthetic, charges, case):
    cfg, _ = synthetic
    q = charges.iloc[:1].copy()
    if case == "ratio_overflow":
        q["charge"] = np.finfo(float).max
        cfg = replace(cfg, single_pe_reference_charge=np.nextafter(0., 1.))
    elif case == "ratio_cap":
        q["charge"] = (B.MAX_CHARGE_RATIO + 1) * cfg.single_pe_reference_charge
    elif case == "adc_wrap":
        q["charge"] = 1000 * cfg.single_pe_reference_charge
    elif case == "cluster_sum":
        q["charge"] = 150 * cfg.single_pe_reference_charge
        q = pd.concat([q, q], ignore_index=True)
    elif case == "noise":
        cfg = replace(cfg, noise=np.finfo(float).max)
    elif case == "conversion":
        cfg = replace(cfg, adc_conversion=np.finfo(float).max)
    elif case == "negative":
        q["charge"] = -np.finfo(float).max
    else:
        q["time"] = np.iinfo(np.int64).min
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("unsafe amplitude reached helper"))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        with pytest.raises(ValueError):
            B.synthesize(q, channels, 71, cfg)


@pytest.mark.parametrize("config", [
    {"noise": 0., "min_voltage": 0.},
    {"min_adc": 10, "pedestal": 200, "max_adc": 3000, "adc_conversion": 1000.},
    {"pedestal": 0, "noise": 0.},
])
def test_supported_configuration_matches_frozen_helper(channels, synthetic, charges, config):
    cfg, _ = synthetic
    cfg = replace(cfg, **config)
    expected = W.synthesize(charges, channels, 71, cfg)
    actual, _ = B.synthesize(charges, channels, 71, cfg)
    for key in expected[0]:
        np.testing.assert_array_equal(actual[0][key], expected[0][key])
    for a, b in zip(actual[1:], expected[1:]):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_inclusive_start_window_preserves_full_tails(channels, synthetic, charges):
    cfg, _ = synthetic
    cfg = replace(cfg, noise=0.)
    q = charges.iloc[:1].copy()
    baseline = W.synthesize(q, channels, 71, cfg)[0]
    assert len(baseline["start"]) == 1
    start = baseline["start"][0]
    for window in ((start, start), (start, start + 1), (start - 1, start)):
        expected = W.synthesize(q, channels, 71, cfg, window)
        actual, _ = B.synthesize(q, channels, 71, cfg, window)
        for key in baseline:
            np.testing.assert_array_equal(actual[0][key], baseline[key])
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        for a, b in zip(actual[1:], expected[1:]):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert actual[0]["volt"].shape == (1, cfg.samples_zs)
        assert start + (cfg.samples_zs - 1) * 12.5 > window[1]
        assert (actual[0]["volt"][0, 1:] > 0).any()
    after = np.nextafter(start, np.inf)
    actual, _ = B.synthesize(q, channels, 71, cfg, (after, after))
    assert len(actual[0]["chan"]) == 0


@pytest.mark.parametrize("seed", [True, np.bool_(False), 71., "71"])
def test_seed_requires_nonboolean_integer(monkeypatch, channels, synthetic, charges, seed):
    cfg, _ = synthetic
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("invalid seed reached helper"))
    with pytest.raises(ValueError, match="seed"):
        B.synthesize(charges, channels, seed, cfg)


@pytest.mark.parametrize("field,value", [
    ("noise", 10**400), ("single_pe_precision", "0.5"),
    ("single_pe_max_time", np.finfo(float).max),
    ("single_pe_precision", np.nextafter(0., 1.)),
])
def test_extreme_configuration_does_not_overflow_before_rejection(monkeypatch, channels, synthetic, charges, field, value):
    cfg, _ = synthetic
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("extreme config reached helper"))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        with pytest.raises(ValueError):
            B.synthesize(charges.iloc[:0], channels, 71, replace(cfg, **{field: value}))


@pytest.mark.parametrize("template", [
    np.array([np.nan]), np.array([np.inf]), np.array([1 + 0j]), np.array([]),
    np.ones((2, 2)), np.array([np.finfo(float).max]),
])
def test_invalid_template_rejected_before_helper(monkeypatch, channels, synthetic, charges, template):
    cfg, _ = synthetic
    monkeypatch.setattr(B.WF, "single_pe_template", lambda *a: template)
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("invalid template reached helper"))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        with pytest.raises(ValueError):
            B.synthesize(charges, channels, 71, cfg)


def test_tiny_template_period_rejected_before_index_narrowing(monkeypatch, channels, synthetic, charges):
    cfg, _ = synthetic
    cfg = replace(cfg, single_pe_max_time=1e-20, single_pe_precision=1e-20)
    monkeypatch.setattr(W, "synthesize", lambda *a: pytest.fail("unsafe index reached helper"))
    with pytest.raises(ValueError, match="template index"):
        B.synthesize(charges, channels, 71, cfg)


def test_large_but_safe_saturating_pulse_is_not_clipped_before_helper(channels, synthetic, charges):
    cfg, _ = synthetic
    cfg = replace(cfg, noise=0.)
    q = charges.iloc[:1].copy()
    q["charge"] = 100 * cfg.single_pe_reference_charge
    expected = W.synthesize(q, channels, 71, cfg)
    actual, _ = B.synthesize(q, channels, 71, cfg)
    for key in expected[0]:
        np.testing.assert_array_equal(actual[0][key], expected[0][key])
    assert (actual[0]["adc"] == cfg.max_adc).any()
    assert actual[0]["volt"].max() > (cfg.max_adc - cfg.pedestal) / cfg.adc_conversion
