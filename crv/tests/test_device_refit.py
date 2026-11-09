"""reco.chain_jax device refit: the waveforms the short pulse fit flags (long, many pulses) get the same pulses as the
numpy full-length refit, to fit tolerance; rows that still overflow go to numpy; flag off keeps the numpy path."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from crvfastsim.reco import chain_jax as CJX               # noqa: E402
from crvfastsim.reco import pulsefit as PF                 # noqa: E402


def waveforms(W=64, L=40, seed=3):
    """Long waveforms with 1-6 well separated pulses (more than the 3 short-fit rounds, longer than 12 samples)."""
    rng = np.random.default_rng(seed)
    t = (100 + np.arange(L)) * PF.PERIOD
    adc = np.zeros((W, L), np.int64)
    for i in range(W):
        y = np.zeros(L)
        for j in range(1 + i % 6):
            y += PF.gumbel(t, rng.uniform(400, 2500), t[3 + 6 * j] + rng.uniform(-3, 3), rng.uniform(12, 18))
        adc[i] = np.clip(np.round(y + 2045 + rng.normal(0, 2.0, L)), 0, 4095)
    n = np.full(W, L)
    n[::7] = 20
    return dict(adc=adc, n=n, start=np.full(W, 100), ped=np.full(W, 2045.0), area=np.full(W, 400.0),
                height=np.full(W, 30.0))


def test_device_refit_matches_numpy_full_length():
    Wv = waveforms()
    g = np.arange(len(Wv["n"]))
    cfg = PF.RecoConfig()
    ref = PF.reconstruct_full(Wv["adc"], Wv["n"], Wv["start"], Wv["ped"], Wv["area"], Wv["height"], cfg)
    dev, again = CJX._device_refit(Wv, g, cfg)
    assert not again.any()
    o = np.lexsort((dev["seq"], dev["wf"]))
    dev = {k: v[o] for k, v in dev.items()}
    np.testing.assert_array_equal(dev["wf"], ref["wf"])
    np.testing.assert_array_equal(dev["seq"], ref["seq"])
    good = ~ref["failed"]
    np.testing.assert_allclose(dev["pe"][good], ref["pe"][good], rtol=2e-3)
    np.testing.assert_allclose(dev["time"][good], ref["time"][good], atol=0.05)
    assert np.bincount(ref["wf"]).max() == 6                # more pulses than the short fit's 3 rounds


def test_collect_routes_flagged_rows(monkeypatch):
    Wv = waveforms(W=8)
    rows = np.arange(8)
    cfg = PF.RecoConfig()
    W2 = len(rows)
    short = dict(found=np.zeros((W2, 3), bool), pe=np.zeros((W2, 3)), time=np.zeros((W2, 3)),
                 rounds_exhausted=np.zeros(W2, bool), window_overflow=np.zeros(W2, bool),
                 input_truncated=np.ones(W2, bool))
    numpy_calls = []
    real = PF.reconstruct_full
    monkeypatch.setattr(PF, "reconstruct_full", lambda *a, **k: numpy_calls.append(len(a[1])) or real(*a, **k))
    off = CJX._pulse_collect([(short, W2)], Wv, rows, cfg)
    assert numpy_calls == [8] and off[4] == 8
    numpy_calls.clear()
    counts = {}
    on = CJX._pulse_collect([(short, W2)], Wv, rows, cfg, True, counts)
    assert numpy_calls == [] and on[4] == 0 and counts["pulse_device_refit"] == 8
    np.testing.assert_array_equal(on[0], off[0])
    np.testing.assert_array_equal(on[1], off[1])
    np.testing.assert_allclose(on[2], off[2], rtol=2e-3)
