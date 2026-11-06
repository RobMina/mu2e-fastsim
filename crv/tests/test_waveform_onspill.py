"""Synthetic stock ZS scan tests; no detector assets or stochastic physics tuning."""
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax import lax

from crvfastsim.reco import waveform_jax as J
from crvfastsim.reco.waveform import WaveformConfig
from mu2efastsim import precision as dprec

PERIOD = J.PERIOD
CFG = replace(WaveformConfig(), min_voltage=0.5, single_pe_precision=PERIOD)


def dense_args(values, t_start, *, n=None, keep=None, noise=0.0):
    """A one-tick template and fixed charges give exactly the requested samples."""
    values = np.atleast_2d(np.asarray(values, float))
    C, Q = values.shape
    t_start = np.broadcast_to(np.asarray(t_start, float), (C,)).copy()
    n = np.full(C, Q) if n is None else np.asarray(n)
    keep = np.ones(C, bool) if keep is None else np.asarray(keep)
    times = t_start[:, None] + np.arange(Q)[None, :] * PERIOD
    return (times, values, n, t_start, keep, np.ones(1), jax.random.PRNGKey(719),
            Q + CFG.samples_zs, CFG.samples_zs, PERIOD, noise, CFG.min_voltage,
            CFG.adc_conversion, CFG.pedestal, CFG.min_adc, CFG.max_adc)


def stock_reference(wave, last, t_start, keep, window, zs=12):
    """Literal CrvWaveformsGenerator nested loops, including both increments of i."""
    starts = np.zeros(wave.shape, bool)
    rows, indices, voltages = [], [], []
    for r in range(len(wave)):
        if not keep[r]:
            continue
        full_waveform = wave[r, :last[r]]
        i = 0
        while i < len(full_waveform):  # outer for (...; ...; ++i)
            trigger = any(full_waveform[j] > CFG.min_voltage
                          for j in range(max(i, 2) - 2, min(i + 3, len(full_waveform))))
            if trigger:
                start_time = t_start[r] + i * PERIOD
                if start_time < window[0] or start_time > window[1]:
                    i += 1  # continue executes only the outer increment
                    continue
                starts[r, i] = True
                rows.append(r)
                indices.append(i)
                voltage = []
                for _ in range(zs):  # inner ++i, ++singleWaveformIndex
                    voltage.append(full_waveform[i] if i < len(full_waveform) else 0.0)
                    i += 1
                voltages.append(voltage)
                i -= 1  # stock --i cancels the outer increment after an accepted digi
            i += 1
    return starts, np.asarray(rows, int), np.asarray(indices, int), np.asarray(voltages).reshape(-1, zs)


def check_stock(args, window):
    outputs = tuple(np.asarray(x) for x in J.synth_dense(
        *args, stock_scan=True, window_start=window[0], window_end=window[1]))
    wave, adc, starts, last = outputs
    expected, rows, indices, voltages = stock_reference(wave, last, args[3], args[4], window)
    np.testing.assert_array_equal(starts, expected)
    pk = dict(t_start=args[3], keep=args[4], chan=np.arange(len(wave)),
              tdc0=np.zeros(len(wave)), est=last)
    digis = J.to_digis(pk, wave, adc, starts, last, CFG, window)
    np.testing.assert_array_equal(digis["chan"], rows)
    np.testing.assert_array_equal(digis["start"], args[3][rows] + indices * PERIOD)
    np.testing.assert_array_equal(digis["volt"], voltages)
    expected_adc = np.clip((voltages * CFG.adc_conversion + CFG.pedestal + 0.5).astype(np.int16),
                           CFG.min_adc, CFG.max_adc)
    np.testing.assert_array_equal(digis["adc"], expected_adc)
    return outputs, digis


def test_pre_gate_trigger_does_not_hide_in_gate_continuation():
    args = dense_args(np.ones(35), 175.0)
    outputs, digis = check_stock(args, (200.0, 1500.0))
    np.testing.assert_array_equal(np.flatnonzero(outputs[2][0]), [2, 14, 26])
    assert digis["start"][0] == 200.0
    # An ungated pre-window start consumes 12 ticks before the host drops it.
    legacy_starts = np.asarray(J.synth_dense(*args)[2])
    np.testing.assert_array_equal(np.flatnonzero(legacy_starts[0]), [0, 12, 24])
    legacy_times = args[3][0] + np.flatnonzero(legacy_starts[0]) * PERIOD
    assert legacy_times[legacy_times >= 200.0][0] == 325.0


def test_upper_start_is_inclusive_and_tail_is_not_cropped():
    args = dense_args(np.ones(5), 1475.0)
    _, digis = check_stock(args, (1500.0, 1500.0))
    np.testing.assert_array_equal(digis["start"], [1500.0])
    # Samples at 1512.5 and 1525 survive; beyond fullWaveform is zero padded.
    np.testing.assert_array_equal(digis["volt"][0], [1.0] * 3 + [0.0] * 9)
    np.testing.assert_array_equal(digis["adc"][0, 3:], np.full(9, CFG.pedestal))
    _, excluded = check_stock(args, (1499.0, 1500.0 - 0.125))
    assert len(excluded["start"]) == 0


def test_stock_nested_loops_resume_at_12_with_decrement():
    args = dense_args(np.ones(40), 200.0)
    outputs, _ = check_stock(args, (200.0, 1500.0))
    np.testing.assert_array_equal(np.flatnonzero(outputs[2][0]), [0, 12, 24, 36])
    # Missing endpoints mean an unbounded stock scan, not a host-only crop.
    unbounded = J.synth_dense(*args, stock_scan=True)
    np.testing.assert_array_equal(unbounded[2], outputs[2])


def test_literal_scan_mixed_lengths_phases_masks_and_threshold():
    values = np.random.default_rng(81).choice([0.0, 0.5, 0.75, 1.0], (6, 48))
    values[4] = 0.5  # strict > minVoltage, not >=
    args = dense_args(values, [162.5, 193.25, 1462.5, 200.0, 200.0, 1500.125],
                      n=[48, 37, 8, 0, 48, 48], keep=[True, True, True, True, True, False])
    outputs, _ = check_stock(args, (200.0, 1500.0))
    assert not outputs[2][3:].any()


@partial(dprec.jit, static_argnames=("L", "zs"))
def legacy_dense(T, W, n, t_start, keep, tmpl, key, L, zs, precision, noise, min_voltage,
                 adc_conv, pedestal, adc_lo, adc_hi):
    """Frozen pre-opt-in kernel for bitwise seeded regression, including noisy samples."""
    C, Q = T.shape
    qmask = jnp.arange(Q)[None, :] < n[:, None]
    ts = t_start[:, None] + jnp.arange(L, dtype=dprec.FLOAT)[None, :] * PERIOD
    dt = ts[:, None, :] - T[:, :, None]
    ti = jnp.rint(dt / precision).astype(jnp.int64)
    ok = (dt >= 0) & (ti < tmpl.shape[0]) & qmask[:, :, None]
    contrib = jnp.where(ok, tmpl[jnp.clip(ti, 0, tmpl.shape[0] - 1)] * W[:, :, None], 0.0)
    wave = contrib.sum(1)
    last = jnp.where(ok, jnp.arange(L)[None, None, :] + 1, 0).max((1, 2))
    wave = wave + noise * jax.random.normal(key, wave.shape) * (jnp.arange(L)[None, :] < last[:, None])
    valid = jnp.arange(L)[None, :] < last[:, None]
    above = (wave > min_voltage) & valid
    pad = jnp.zeros((C, 2), bool)
    ap = jnp.concatenate([pad, above, pad], 1)
    win = ap[:, 0:L] | ap[:, 1:L + 1] | ap[:, 2:L + 2] | ap[:, 3:L + 3] | ap[:, 4:L + 4]
    win = win & valid & keep[:, None]

    def step(next_i, i):
        start = win[:, i] & (next_i <= i)
        return jnp.where(start, i + zs, next_i), start

    _, starts = lax.scan(step, jnp.zeros(C, jnp.int64), jnp.arange(L))
    starts = starts.T
    adc = jnp.clip((wave * adc_conv + pedestal + 0.5).astype(jnp.int16), adc_lo, adc_hi)
    return wave, adc, starts, last


@pytest.mark.parametrize("noise", [0.0, 0.04])
def test_default_is_bitwise_legacy_with_seeded_noise(noise):
    values = np.random.default_rng(53).uniform(0.0, 1.0, (3, 40))
    args = dense_args(values, [175.0, 1462.5, 190.0], n=[40, 31, 17],
                      keep=[True, True, False], noise=noise)
    expected = legacy_dense(*args)
    default = J.synth_dense(*args)
    disabled = J.synth_dense(*args, stock_scan=False, window_start=200.0, window_end=1500.0)
    stock = J.synth_dense(*args, stock_scan=True, window_start=200.0, window_end=1500.0)
    for before, after, explicit in zip(expected, default, disabled):
        np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(explicit, before)
    for index in (0, 1, 3):  # opt-in only changes start selection, never analog/noise/ADC
        np.testing.assert_array_equal(stock[index], default[index])


def test_bucketed_synth_forwards_stock_gate(monkeypatch):
    from crvfastsim.reco import cli, waveform
    monkeypatch.setattr(J, "single_pe_template", lambda *unused: np.ones(7))
    cfg = replace(CFG, single_pe_max_time=7 * PERIOD)
    q = {"row": np.array([0]), "time": np.array([187.5]), "charge": np.array([cfg.single_pe_reference_charge])}
    stock = cli._synth_jax_bucketed(J, q, np.array([0.]), cfg, window=(200., 1500.), stock_scan=True)
    # Supply the same test template to the NumPy reference as to the device path.
    monkeypatch.setattr(waveform, "single_pe_template", lambda *unused: np.ones(7))
    reference = waveform.synthesize(q["row"], q["time"], q["charge"], np.array([0.]), cfg, window=(200., 1500.))
    for key in ("chan", "starttdc", "adc"):
        np.testing.assert_array_equal(stock[key], reference[key])
    empty = cli._synth_jax_bucketed(J, {k: v[:0] for k, v in q.items()}, np.array([]), cfg,
                                   window=(200., 1500.), stock_scan=True)
    assert empty["adc"].shape == (0, cfg.samples_zs)


def test_synthesize_forwards_window_only_when_opted_in(monkeypatch):
    monkeypatch.setattr(J, "single_pe_template", lambda *unused: np.ones(7))
    cfg = replace(CFG, single_pe_max_time=7 * PERIOD)
    args = (np.array([0]), np.array([187.5]), np.array([cfg.single_pe_reference_charge]), np.array([0.0]))
    default = J.synthesize(*args, cfg=cfg, window=(200.0, 1500.0))
    stock = J.synthesize(*args, cfg=cfg, window=(200.0, 1500.0), stock_scan=True)
    assert len(default["start"]) == 0
    np.testing.assert_array_equal(stock["start"], [200.0])
    np.testing.assert_array_equal(stock["volt"][0], [1.0] * 6 + [0.0] * 6)
