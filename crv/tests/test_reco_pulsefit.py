"""Unit tests of crvfastsim.reco: the batched Gumbel fit recovers injected pulses, the peak finder follows the stock
rules, digi grouping concatenates continuing digis, and the waveform synthesis is linear in charge and zero-suppressed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim.reco import pulsefit as PF                 # noqa: E402
from crvfastsim.reco import waveform as WF                 # noqa: E402

CFG = PF.RecoConfig()


def _wave(A, mu, b, n=24, start_tdc=10, ped=100.0, noise=0.0, seed=0):
    t = (start_tdc + np.arange(n)) * PF.PERIOD
    y = PF.gumbel(t, A, mu, b)
    rng = np.random.default_rng(seed)
    return np.round(y + ped + rng.normal(0, noise, n)).astype(np.int64)


def test_single_pulse_fit_recovers_parameters():
    W = 200
    rng = np.random.default_rng(1)
    A = rng.uniform(300, 2000, W)
    mu = (10 + 6) * PF.PERIOD + rng.uniform(-6, 6, W)
    b = rng.uniform(14, 24, W)
    adc = np.stack([_wave(A[i], mu[i], b[i]) for i in range(W)])
    res = PF.reconstruct(adc, np.full(W, 24), np.full(W, 10), np.full(W, 100.0), np.ones(W), np.ones(W), CFG)
    first = res["seq"] == 0
    assert first.sum() == W
    ok = ~res["failed"][first]
    assert ok.mean() > 0.95
    assert np.median(np.abs(res["fit_mu"][first][ok] - mu[ok])) < 0.3
    assert np.median(np.abs(res["fit_b"][first][ok] / b[ok] - 1)) < 0.03
    # PE = A*beta / calib (calib 1)
    assert np.median(np.abs(res["pe"][first][ok] / (A[ok] * b[ok]) - 1)) < 0.03


def test_below_threshold_gives_no_pulse():
    adc = _wave(30.0, 16 * PF.PERIOD, 19.0)[None, :]
    res = PF.reconstruct(adc, np.array([24]), np.array([10]), np.array([100.0]), np.ones(1), np.ones(1), CFG)
    assert len(res["wf"]) == 0


def test_pedestal_undershoot_uses_first_sample():
    adc = _wave(800.0, 16 * PF.PERIOD, 19.0, ped=90.0)[None, :]
    res = PF.reconstruct(adc, np.array([24]), np.array([10]), np.array([100.0]), np.ones(1), np.ones(1), CFG)
    assert not res["pedfromdb"][0] and res["ped"][0] == adc[0, 0]


def test_concat_waveforms_joins_continuing_digis():
    key = np.array([1, 1, 1, 2])
    bar = np.array([5, 5, 6, 5])
    sipm = np.zeros(4, np.int64)
    tdc = np.array([10, 22, 10, 10])
    lens = np.array([12, 12, 12, 12])
    flat = np.arange(48)
    gid, mat, n, start, first = PF.concat_waveforms(key, bar, sipm, tdc, flat, lens)
    assert list(gid) == [0, 0, 1, 2]
    assert list(n) == [24, 12, 12] and list(start) == [10, 10, 10]
    assert np.array_equal(mat[0, :24], np.arange(24))


def test_waveform_linear_and_zero_suppressed():
    cfg = WF.WaveformConfig()
    t = 300.0 + np.linspace(0.0, 6.0, 30)                  # 30 single-PE avalanches (the stock skip rule counts charges)
    q = np.full(30, cfg.single_pe_reference_charge)
    chan = np.zeros(30, np.int64)
    a = WF.synthesize(chan, t, q, np.array([3.0]), cfg)
    b = WF.synthesize(chan, t, 2 * q, np.array([3.0]), cfg)
    assert len(a["chan"]) >= 1
    assert np.allclose(b["volt"][0], 2 * a["volt"][0])
    assert np.allclose((a["start"] - 3.0) / PF.PERIOD, np.rint((a["start"] - 3.0) / PF.PERIOD))
    tiny = WF.synthesize(chan[:1], t[:1], np.array([0.5 * cfg.single_pe_reference_charge]), np.array([3.0]), cfg)
    assert len(tiny["chan"]) == 0


def test_waveform_then_reco_roundtrip():
    cfg = WF.WaveformConfig()
    rng = np.random.default_rng(3)
    nq = 40
    t = np.sort(300.0 + rng.exponential(8.0, nq))
    q = np.full(nq, cfg.single_pe_reference_charge)
    dg = WF.synthesize(np.zeros(nq, np.int64), t, q, np.array([0.0]), cfg)
    lens = np.full(len(dg["chan"]), cfg.samples_zs)
    gid, mat, n, start, first = PF.concat_waveforms(np.zeros(len(lens), np.int64), dg["chan"],
                                                    np.zeros(len(lens), np.int64), dg["starttdc"],
                                                    dg["adc"].ravel(), lens)
    res = PF.reconstruct(mat, n, start, np.full(len(n), 100.0), np.ones(len(n)), np.ones(len(n)), CFG)
    assert len(res["wf"]) >= 1
    assert 290.0 < res["time"][0] < 360.0


def test_direct_model_threshold_and_resampling():
    from crvfastsim.reco import direct as DR
    rng = np.random.default_rng(5)
    C = 40000
    q = rng.uniform(2.0, 30.0, C)
    feat = {"q": q, "pfrac": rng.uniform(0.3, 0.9, C), "t10": np.zeros(C)}
    found = rng.random(C) < 1 / (1 + np.exp(-(q - 9.0) / 0.7))
    npul = found.astype(np.int64)
    first = {"pe": np.where(found, 0.9 * q, np.nan), "dt": np.where(found, 5.0, np.nan),
             "beta": np.where(found, 18.0, np.nan), "failed": np.zeros(C, bool)}
    m = DR.DirectModel.fit(feat, npul, first)
    s = m.sample(feat, np.random.default_rng(6))
    for lo, hi in ((6, 8), (8, 10), (10, 12)):
        sel = (q >= lo) & (q < hi)
        assert abs((s["n"][sel] > 0).mean() - found[sel].mean()) < 0.03
    assert abs(np.nanmedian(s["pe"] / q) - 0.9) < 1e-6


def test_direct_v2_resamples_whole_response():
    """DirectModelV2 returns a training cluster's full response: count, pulse ordering and spacing, TDC0 shift."""
    import numpy as np
    from crvfastsim.reco.direct import N_MODELLED, DirectModelV2
    rng = np.random.default_rng(3)
    C = 2000
    feat = {"q": np.full(C, 40.0), "pfrac": np.full(C, 0.8), "t10": rng.uniform(100, 200, C)}
    n = rng.integers(1, 3, C)
    tdc0 = rng.uniform(0, 12.5, C)
    pul = {k: np.full((C, N_MODELLED), np.nan) for k in ("pe", "time", "beta")}
    pul["failed"] = np.zeros((C, N_MODELLED), bool)
    pul["pe"][:, 0], pul["time"][:, 0], pul["beta"][:, 0] = 38.0, feat["t10"] - tdc0 + 30.0, 15.0
    two = n == 2
    pul["pe"][two, 1], pul["time"][two, 1], pul["beta"][two, 1] = 5.0, pul["time"][two, 0] + 40.0, 14.0
    m = DirectModelV2.fit(feat, n, pul, tdc0)
    t_new = rng.uniform(0, 12.5, C)
    s = m.sample(feat, t_new, rng)
    assert set(np.unique(s["n"])) == {1, 2}
    assert np.allclose(s["time"][:, 0], feat["t10"] - t_new + 30.0)
    t2 = s["n"] == 2
    assert np.allclose(s["time"][t2, 1] - s["time"][t2, 0], 40.0) and np.all(np.isnan(s["time"][~t2, 1]))


def test_noise_partner_and_model():
    """Noise flag: a pulse pair on two SiPMs of one counter is signal, a lone pulse is noise; the fitted model
    reproduces the per-event count and keeps generated pulses on the channels seen in training."""
    import numpy as np
    from crvfastsim.reco import noise as N
    rng = np.random.default_rng(5)
    rows = []
    for e in range(4000):
        b = rng.integers(0, 100)
        rows += [(e, b * 4, 500.0, 30.0), (e, b * 4 + 2, 503.0, 28.0)]          # signal pair
        for _ in range(rng.poisson(1.5)):
            rows.append((e, 4 * rng.integers(100, 110) + 1, rng.uniform(0, 1e5), 6.0))   # noise channels 401..437
    a = np.array(rows)
    o = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
    a = a[o]
    P = {"ev": a[:, 0].astype(np.int64), "chan": a[:, 1].astype(np.int64), "time": a[:, 2], "pe": a[:, 3],
         "beta": np.full(len(a), 14.0), "failed": np.zeros(len(a), bool), "wf": np.arange(len(a))}
    wi, noise = N.noise_waveforms(P["ev"], P["chan"], P["time"], P["wf"])
    assert not noise[P["pe"] > 20].any() and noise[P["pe"] < 10].all()
    m = N.NoiseModel.fit(P, 4000, 2304)
    s = m.sample(4000, rng)
    assert abs(len(s["ev"]) / 4000 - 1.5) < 0.1
    assert set(np.unique(s["chan"])) <= set(range(401, 440, 4))
    assert np.allclose(s["pe"], 6.0)
