"""Opt-in thermal attempts share the existing JAX SiPM recovery and ancestry machinery."""
from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from crvfastsim.response import sipm, sipm_jax as sj


def _map():
    pmap = np.zeros((40, 40))
    pmap[18, 18] = 1
    return pmap


def _simulate(cfg=None, **kwargs):
    return sj.simulate([], [], [], _map(), cfg or sipm.SiPMConfig(thermal_rate=0.01),
                       np.random.default_rng(81), jax.random.PRNGKey(7), **kwargs)


def _echo(args, cfg, *, pixel_override=None):
    n = int(np.asarray(args[2]).sum())
    return {"slot": args[0][:n], "time": args[1][:n], "root": np.arange(n),
            "fired": np.ones(n, bool), "charge": np.ones(n), "pe": np.ones(n), "overflow": False}


@pytest.mark.slow
def test_attempt_moments_uniform_pixels_and_empty_labels(monkeypatch):
    captured = []

    def record(args, cfg, *, pixel_override=None):
        n = int(np.asarray(args[2]).sum())
        captured.append(np.asarray(pixel_override)[:n])
        return _echo(args, cfg, pixel_override=pixel_override)

    monkeypatch.setattr(sj, "run_device_sipm", record)
    cfg = sipm.SiPMConfig(thermal_rate=0.08)
    labels = np.arange(12000) * 200 + 17
    out = _simulate(cfg, t_start=10, t_end=20, thermal_chans=labels)
    counts = np.bincount(np.searchsorted(labels, out["chan"]), minlength=len(labels))
    lam = cfg.thermal_rate / cfg.p_fire(cfg.overvoltage) * 10
    assert abs(counts.mean() - lam) < 5 * np.sqrt(lam / len(labels))
    assert abs(counts.var() - lam) < 5 * np.sqrt((lam + 2 * lam**2) / len(labels))
    assert (counts == 0).any() and (counts > 0).any()
    assert np.all(out["photon"] == -1)
    assert np.all((out["time"] >= 10) & (out["time"] < 20))
    assert abs(out["time"].mean() - 15) < 0.1
    assert abs(out["time"].var() - 100 / 12) < 0.2
    pixels = np.concatenate(captured)
    for coord in (pixels // sj.NPIX, pixels % sj.NPIX):
        assert coord.min() == 0 and coord.max() == 39
        bins = np.bincount(coord, minlength=40)
        assert np.max(np.abs(bins - len(coord) / 40)) < 5 * np.sqrt(len(coord) / 40)
    assert np.any(pixels == 0)


@pytest.mark.slow
def test_optical_sentinel_and_dark_ancestry(monkeypatch):
    sentinel_counts = []

    def echo_descendant(args, cfg, *, pixel_override=None):
        result = _echo(args, cfg, pixel_override=pixel_override)
        pixels = np.asarray(pixel_override)[:len(result["root"])]
        sentinel_counts.append(np.count_nonzero(pixels == -1))
        root = np.flatnonzero(pixels >= 0)[0]
        for name in ("slot", "time", "charge", "pe", "fired"):
            result[name] = np.r_[np.asarray(result[name]), result[name][root]]
        result["root"] = np.r_[result["root"], root]
        return result

    monkeypatch.setattr(sj, "run_device_sipm", echo_descendant)
    out = sj.simulate([17, 17], [1, 2], np.array([101, 102], dtype=np.uint64), _map(),
                      sipm.SiPMConfig(thermal_rate=0.1), np.random.default_rng(81), jax.random.PRNGKey(7),
                      t_end=100, thermal_chans=[17, 23])
    assert sum(sentinel_counts) == 2
    assert set(out["photon"]) == {-1, 101, 102}
    assert np.all(out["photon"][out["chan"] == 23] == -1)
    assert 23 in out["chan"]


def test_optical_and_dark_share_device_recovery():
    class FixedDark:
        def poisson(self, lam, size):
            return np.ones(size, np.int64)

        def random(self, size):
            return np.full(size, 0.5)

        def integers(self, low, high, size):
            return np.zeros(size, np.int64)

    pmap = np.zeros((40, 40))
    pmap[0, 0] = 1
    cfg = sipm.SiPMConfig(thermal_rate=0.1, p1=1, p2=0.001, cross_talk=0, inactive=())
    out = sj.simulate(np.zeros(64, np.int64), np.zeros(64), np.arange(64), pmap, cfg,
                      FixedDark(), jax.random.PRNGKey(7), t_end=2, thermal_chans=[0])
    dark = out["photon"] == -1
    assert dark.sum() == 1 and (~dark).sum() == 4
    np.testing.assert_allclose(out["pe"][dark], [1 - np.exp(-1 / cfg.time_constant)], rtol=1e-6)
    np.testing.assert_array_equal(out["time"][dark], [1])


@pytest.mark.slow   # statistical moments
def test_empty_input_device_moments():
    cfg = sipm.SiPMConfig(thermal_rate=0.01, cross_talk=0, inactive=())
    out = _simulate(cfg, t_end=100, thermal_chans=np.arange(256))
    counts = np.bincount(out["chan"], minlength=256)
    assert abs(counts.mean() - 1) < 5 / np.sqrt(256)
    assert abs(counts.var() - 1) < 5 * np.sqrt(3 / 256)
    assert np.all(out["photon"] == -1)
    assert np.all((out["time"] >= 0) & (out["time"] <= 100))
    assert np.all((out["pe"] > 0) & (out["pe"] <= 1))
    ref = sipm.simulate([], [], [], _map(), cfg, np.random.default_rng(82), t_end=100,
                        thermal_chans=np.arange(256))
    ref_counts = np.bincount(ref["chan"].astype(np.int64), minlength=256)
    assert abs(counts.mean() - ref_counts.mean()) < 5 * np.sqrt(2 / 256)
    assert abs(counts.var() - ref_counts.var()) < 5 * np.sqrt(6 / 256)


def _device_args(cfg, times, slots=None, q_ct=0):
    times = np.asarray(times, float)
    n = len(times)
    return (jnp.asarray(np.zeros(n, np.int64) if slots is None else slots), jnp.asarray(times),
            jnp.ones(n, bool), jax.random.PRNGKey(9), jnp.asarray(np.cumsum(_map().ravel())), 40,
            jnp.zeros(sj.NP2, bool), q_ct, cfg.overvoltage, cfg.time_constant, cfg.p1, cfg.p2,
            cfg.capacitance, n, 16, 16, 16, n + 48)


@pytest.mark.slow
def test_preexisting_device_calls_and_shared_recovery():
    cfg = sipm.SiPMConfig(p1=1, p2=0.001, inactive=())
    args = _device_args(cfg, [0, 0, 20])
    old = sj.device_sipm(*args, 8, 1, 1, 40, 40)
    sentinel = sj.device_sipm(*args, 8, 1, 1, 40, 40, pixel_override=jnp.full(3, -1))
    for key in old:
        np.testing.assert_array_equal(old[key], sentinel[key])
    shared = sj.device_sipm(*args, 8, 1, 1, 40, 40, pixel_override=jnp.full(3, 5 * sj.NPIX + 7))
    np.testing.assert_array_equal(np.asarray(shared["fired"])[:3], [True, False, True])
    np.testing.assert_allclose(np.asarray(shared["pe"])[[0, 2]], [1, 1 - np.exp(-20 / cfg.time_constant)], rtol=1e-6)
    dead = np.zeros(sj.NP2, bool)
    dead[5 * sj.NPIX + 7] = True
    dead_args = (*args[:6], jnp.asarray(dead), *args[7:])
    result = sj.device_sipm(*dead_args, 8, 1, 1, 40, 40, pixel_override=jnp.full(3, 5 * sj.NPIX + 7))
    assert not np.any(result["fired"])


@pytest.mark.slow
def test_device_dark_crosstalk_descendants():
    cfg = sipm.SiPMConfig(p1=1, p2=0.001, inactive=())
    args = _device_args(cfg, [4], q_ct=1)
    args = (*args[:-2], 64, 97)
    result = sj.device_sipm(*args, 8, 1, 8, 40, 40, pixel_override=jnp.array([0]))
    assert not bool(result["overflow"])
    roots = np.asarray(result["root"])[np.asarray(result["fired"])]
    assert len(roots) > 1
    np.testing.assert_array_equal(np.array([-1])[roots], -np.ones(len(roots)))


def test_retry_preserves_pixels(monkeypatch):
    seen = []
    override = jnp.array([17, 32])

    def device(*args, pixel_override=None):
        seen.append(pixel_override)
        return {"stream_overflow": len(seen) == 1, "n_long_streams": 2, "max_long_stream": 65}

    monkeypatch.setattr(sj, "device_sipm", device)
    sj.run_device_sipm((), sipm.SiPMConfig(), pixel_override=override)
    assert len(seen) == 2 and all(item is override for item in seen)


def test_zero_thermal_unchanged_path_and_rng(monkeypatch):
    run = sj.run_device_sipm
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return run(*args, **kwargs)

    monkeypatch.setattr(sj, "run_device_sipm", record)
    cfg = sipm.SiPMConfig()
    rng1, rng2 = np.random.default_rng(29), np.random.default_rng(29)
    chan, times, photons = np.repeat(np.arange(4), 8), np.tile(np.arange(8.), 4), np.arange(32)
    pmap = np.zeros((40, 40))
    pmap[4, 7] = 1
    a = sj.simulate(chan, times, photons, pmap, cfg, rng1, jax.random.PRNGKey(7), 1e5, 1 << 23)
    b = sj.simulate(chan, times, photons, pmap, cfg, rng2, jax.random.PRNGKey(7),
                    t_start=10, thermal_chans=[-99])
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert calls and all(kwargs == {} for kwargs in calls)
    assert rng1.bit_generator.state == rng2.bit_generator.state == np.random.default_rng(29).bit_generator.state
    if sj.dprec.SINGLE:
        np.testing.assert_array_equal(a["photon"], [2, 3, 4, 9, 11, 14, 15, 17, 23, 30])
        np.testing.assert_array_equal(a["pe"], [1, 1, 1, 1, 1, 1, 0.3630906045436859, 1, 1, 1])
        np.testing.assert_array_equal(a["chan"], [0, 0, 0, 1, 1, 1, 1, 2, 2, 3])
        np.testing.assert_array_equal(a["time"], [2, 3, 4, 1, 3, 6, 7, 1, 7, 6])
        np.testing.assert_array_equal(a["charge"], [2.6519999846326414e-13] * 6 +
                                      [9.62916202394519e-14] + [2.6519999846326414e-13] * 3)


def test_overflow_fails_closed_only_for_thermal(monkeypatch):
    monkeypatch.setattr(sj, "run_device_sipm", lambda *a, **k: {"overflow": True})
    with pytest.raises(RuntimeError, match="thermal device overflow"):
        _simulate(t_end=100, thermal_chans=[0, 1])
    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return {k: np.array([1]) for k in ("chan", "time", "charge", "pe", "photon")}

    monkeypatch.setattr(sipm, "simulate", fallback)
    sj.simulate([1], [1], [1], _map(), sipm.SiPMConfig(), np.random.default_rng(1), jax.random.PRNGKey(1))
    assert len(calls) == 1 and calls[0][1] == {"t_end": 1e5}


@pytest.mark.parametrize("rate", [-1, np.nan, np.inf])
def test_invalid_rates(rate):
    with pytest.raises(ValueError, match="thermal_rate"):
        _simulate(sipm.SiPMConfig(thermal_rate=rate), thermal_chans=[0])


@pytest.mark.parametrize("kwargs", [dict(t_start=-1), dict(t_end=-1), dict(t_start=2, t_end=1),
                                    dict(t_start=np.nan), dict(t_end=np.inf)])
def test_invalid_windows(kwargs):
    with pytest.raises(ValueError, match="t_start"):
        _simulate(thermal_chans=[0], **kwargs)


@pytest.mark.parametrize("labels", [None, [], [0, 0], [-1], [np.nan], [np.inf], [1.5], [[1]], [True],
                                    np.array([2**63], dtype=np.uint64)])
def test_invalid_thermal_labels(labels):
    with pytest.raises(ValueError, match="thermal_chans"):
        _simulate(thermal_chans=labels)


@pytest.mark.parametrize("changes", [dict(n_pixels_x=0), dict(n_pixels_y=42), dict(n_pixels_x=2.5),
                                     dict(n_pixels_x=True), dict(inactive=((41, 0),)),
                                     dict(inactive=((-1, 0),)), dict(inactive=((1.5, 0),))])
def test_invalid_pixels(changes):
    with pytest.raises(ValueError, match="pixel"):
        _simulate(replace(sipm.SiPMConfig(thermal_rate=0.1), **changes), thermal_chans=[0])


@pytest.mark.parametrize("changes", [dict(trap0_prob=0.1), dict(trap1_prob=0.1), dict(trap0_prob=-1),
                                     dict(trap1_prob=np.nan)])
def test_traps_still_rejected(changes):
    with pytest.raises(NotImplementedError, match="trap afterpulses"):
        _simulate(replace(sipm.SiPMConfig(), **changes))


@pytest.mark.parametrize("p1", [0, -1, np.nan, np.inf, 10])
def test_invalid_thermal_firing_probability(p1):
    with pytest.raises(ValueError, match="firing probability"):
        _simulate(sipm.SiPMConfig(thermal_rate=0.1, p1=p1), thermal_chans=[0])


def test_zero_window_and_unknown_keyword():
    out = _simulate(t_start=10, t_end=10, thermal_chans=[0, 1])
    assert all(len(value) == 0 for value in out.values())
    with pytest.raises(TypeError, match="unexpected keyword"):
        _simulate(thermal_channels=[0])


@pytest.mark.parametrize("chan,time,photon", [([-1], [0], [0]), ([0.5], [0], [0]),
                                             ([0], [np.nan], [0]), ([0], [], [0]),
                                             ([[0]], [0], [0])])
def test_invalid_optical_inputs(chan, time, photon):
    with pytest.raises(ValueError):
        sj.simulate(chan, time, photon, _map(), sipm.SiPMConfig(), np.random.default_rng(1),
                    jax.random.PRNGKey(1))
