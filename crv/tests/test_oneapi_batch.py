"""Shape-bucketed waveform synthesis (used on oneAPI) gives the same digis as the fixed-batch path without noise, and
the same digi count and noise statistics with it."""
import numpy as np


def _inputs(n_events=400, seed=4):
    from crvfastsim.reco import repro
    row, t, q, tdc0 = repro._charges(n_events, seed)
    return {"row": row, "time": t, "charge": q}, tdc0


def test_bucketed_equals_batched_noise_off():
    from crvfastsim.reco import cli, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    Q, tdc0 = _inputs()
    a = cli._synth_jax_batched(WJ, Q, tdc0, cfg, 500)
    b = cli._synth_jax_bucketed(WJ, Q, tdc0, cfg, budget_gib=0.01)    # small budget: several calls per shape
    assert len(a["adc"]) > 1000
    for k in ("chan", "start", "starttdc", "adc"):
        assert np.array_equal(a[k], b[k]), k
    assert np.max(np.abs(a["volt"] - b["volt"])) < 1e-12


def test_bucketed_noise_statistics():
    from crvfastsim.reco import cli, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    Q, tdc0 = _inputs()
    ref = cli._synth_jax_batched(WJ, Q, tdc0, cfg, 500)                              # noise off
    out = {}
    for name, f in (("batched", lambda: cli._synth_jax_batched(WJ, Q, tdc0, cfg, 500, noise=cfg.noise, seed=1)),
                    ("bucketed", lambda: cli._synth_jax_bucketed(WJ, Q, tdc0, cfg, noise=cfg.noise, seed=1,
                                                               budget_gib=0.01))):
        a = f()
        ka = {(c, s): i for i, (c, s) in enumerate(zip(a["chan"], a["starttdc"]))}
        both = [(ka[(c, s)], j) for j, (c, s) in enumerate(zip(ref["chan"], ref["starttdc"])) if (c, s) in ka]
        i, j = np.array(both).T
        out[name] = (len(a["adc"]), np.std(a["volt"][i] - ref["volt"][j]))
    (na, sa), (nb, sb) = out["batched"], out["bucketed"]
    assert abs(na - nb) <= 0.002 * na
    assert abs(sb / sa - 1) < 0.05 and abs(sa / cfg.noise - 1) < 0.1 and abs(sb / cfg.noise - 1) < 0.1


def test_bucketed_matches_synthesize_with_label_tdc0():
    """chain.digis on oneAPI: bucketed synthesis with SiPM labels and a callable TDC0 == waveform_jax.synthesize."""
    from crvfastsim.reco import cli, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    Q, tdc0 = _inputs(200, 9)
    labels = Q["row"] * 7 + 4096                                   # sparse labels, as event key * 4096 + channel
    lut = lambda lab: tdc0[(lab - 4096) // 7]
    a = WJ.synthesize(labels, Q["time"], Q["charge"], lut, cfg)
    b = cli._synth_jax_bucketed(WJ, {"row": labels, "time": Q["time"], "charge": Q["charge"]}, lut, cfg,
                                budget_gib=0.01)
    for k in ("chan", "start", "starttdc", "adc"):
        assert np.array_equal(a[k], b[k]), k


def test_chain_digis_bucketed_default_equals_unbucketed():
    """chain.digis now uses bucketed synthesis on every JAX backend (one padded call ran out of memory on A100s):
    noise off, the default and ChainConfig(unbucketed_synth=True) give the same digis."""
    from types import SimpleNamespace
    from crvfastsim import chain
    Q, _ = _inputs(300, 11)
    labels = Q["row"] * 7 + 4096
    QC = {"chan": labels, "time": Q["time"], "charge": Q["charge"]}
    cond = SimpleNamespace()
    orig = chain.tdc0_per_feb
    chain.tdc0_per_feb = lambda lab, cond, rng: (np.arange(len(lab)) % 12).astype(np.float64)
    try:
        a = chain.digis(QC, cond, chain.ChainConfig(jax=True, noise=False), np.random.default_rng(1))
        b = chain.digis(QC, cond, chain.ChainConfig(jax=True, noise=False, unbucketed_synth=True), np.random.default_rng(1))
    finally:
        chain.tdc0_per_feb = orig
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
