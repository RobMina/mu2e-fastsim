"""JAX pulse reco (pulsefit_jax) against the numpy reference (pulsefit) on synthetic waveforms: identical pulse finding
and parameters to fit tolerance, including double pulses, below-threshold, undershoot pedestal and saturation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
jax = pytest.importorskip("jax")
from crvfastsim.reco import pulsefit as PF                 # noqa: E402
from crvfastsim.reco import pulsefit_jax as PJ             # noqa: E402


def _batch(W=300, L=36, seed=2):
    rng = np.random.default_rng(seed)
    t = (100 + np.arange(L)) * PF.PERIOD
    adc = np.zeros((W, L), np.int64)
    for i in range(W):
        y = PF.gumbel(t, rng.uniform(150, 3000), t[4] + rng.uniform(-6, 6), rng.uniform(12, 24))
        if i % 3 == 0:   # second pulse
            y += PF.gumbel(t, rng.uniform(100, 1500), t[12] + rng.uniform(-6, 6), rng.uniform(12, 24))
        adc[i] = np.clip(np.round(y + 2045 + rng.normal(0, 2.0, L)), 0, 4095)
    adc[0, :] = 2045                       # flat: no pulse
    adc[1, 0] = 2030                       # pedestal undershoot
    n = np.full(W, L)
    n[5] = 12
    return adc, n, np.full(W, 100), np.full(W, 2045.0), np.full(W, 400.0), np.full(W, 30.0)


def test_jax_matches_numpy():
    args = _batch()
    cfg = PF.RecoConfig()
    ref = PF.reconstruct(*args, cfg)
    d = PJ.reconstruct_dense(*[np.asarray(a) for a in args], cfg=cfg)
    out = PJ.to_flat(d)
    assert not out["window_overflow"].any()
    assert np.array_equal(out["wf"], ref["wf"]) and np.array_equal(out["seq"], ref["seq"])
    assert np.array_equal(out["failed"], ref["failed"])
    good = ~ref["failed"]
    assert np.max(np.abs(out["pe"][good] / ref["pe"][good] - 1)) < 1e-6
    assert np.max(np.abs(out["time"][good] - ref["time"][good])) < 1e-4
    assert np.array_equal(out["wf_near_threshold"], ref["wf_near_threshold"])


def test_waveform_jax_matches_numpy():
    from crvfastsim.reco import waveform as WF
    from crvfastsim.reco import waveform_jax as WJ
    cfg = WF.WaveformConfig()
    rng = np.random.default_rng(4)
    rows = []
    for c in range(60):
        n = rng.integers(1, 60)
        t = np.sort(300.0 + rng.exponential(10.0, n) + (500.0 if c % 5 == 0 else 0.0) * (np.arange(n) > n // 2))
        rows.append((np.full(n, c), t, rng.uniform(0.8, 1.2, n) * cfg.single_pe_reference_charge))
    chan = np.concatenate([r[0] for r in rows]); t = np.concatenate([r[1] for r in rows]); q = np.concatenate([r[2] for r in rows])
    tdc0 = rng.random(60) * 12.5
    a = WF.synthesize(chan, t, q, tdc0, cfg)
    b = WJ.synthesize(chan, t, q, tdc0, cfg)
    assert len(a["chan"]) == len(b["chan"]) > 0
    assert np.array_equal(a["chan"], b["chan"]) and np.array_equal(a["starttdc"], b["starttdc"])
    assert np.array_equal(a["adc"], b["adc"])
    assert np.max(np.abs(a["volt"] - b["volt"])) < 1e-12


def test_coinc_jax_matches_numpy():
    """JAX coincidence finder == coinc.py (scalar and vectorized) on synthetic tracks, noise and showers, with small
    tiers so promotion and the numpy fallback are exercised."""
    from crvfastsim.reco import coinc as C
    from crvfastsim.reco import coinc_jax as CJ
    from crvfastsim.reco import synth
    from crvfastsim.reco.cli import _coinc_geom
    G = _coinc_geom()
    P = synth.coinc_events(300, G, seed=7, shower_frac=0.1, shower_mean=40)
    status = np.zeros(4 * len(G.sector), np.int64)
    status[np.random.default_rng(1).choice(len(status), 40, replace=False)] = 1   # PE compensation path
    cols = [P[k] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")]
    out = CJ.find(*cols, G, status, tiers=((8, 4, 256), (16, 6, 128), (32, 8, 64)))
    assert len(out["fallback_events"]) > 0
    ref, refv = set(), set()
    ev = P["ev"]
    for e in np.unique(ev):
        a, b = np.searchsorted(ev, e), np.searchsorted(ev, e, side="right")
        p = {k: P[k][a:b] for k in ("idx", "bar", "sipm", "time", "pe", "seq")}
        ref |= {(int(e), r["sectortype"], tuple(sorted(r["pulses"]))) for r in C.find_event(p, G, status)}
        refv |= {(int(e), r["sectortype"], tuple(sorted(r["pulses"]))) for r in C.find_event(p, G, status, vectorized=True)}
    assert len(ref) > 100
    assert refv == ref
    assert CJ.cluster_keys(out) == ref
