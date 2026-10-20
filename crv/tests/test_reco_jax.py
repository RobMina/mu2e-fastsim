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


def test_chain_jax_chunking_and_consistency():
    """Pipelined pulses -> coincidences: the clusters do not depend on the chunking and equal coinc_jax.find run on the
    chain's own pulses."""
    from crvfastsim.reco import chain_jax, synth
    from crvfastsim.reco import coinc_jax as CJ
    from crvfastsim.reco.cli import _coinc_geom
    G = _coinc_geom()
    W = synth.chain_events(400, G, seed=11)
    status = np.zeros(4 * len(G.sector), np.int64)
    key = lambda outs: set().union(*[CJ.cluster_keys(o) for o in outs])
    a, _ = chain_jax.run(W, G, status, n_chunks=1, batch=4096)
    b, _ = chain_jax.run(W, G, status, n_chunks=3, batch=4096)
    assert len(key(a)) > 100 and key(a) == key(b)
    import dataclasses
    cfg = dataclasses.replace(PF.RecoConfig(), max_pulses=3)
    rows = np.arange(len(W["ev"]))
    wrow, seq, pe, tm, _ = chain_jax._pulse_collect(chain_jax._pulse_dispatch(W, rows, cfg, 4096, 12), W, rows,
                                                    PF.RecoConfig())
    ev = W["ev"][wrow]
    ref = CJ.find(ev, np.arange(len(ev)) - np.searchsorted(ev, ev), W["bar"][wrow], W["sipm"][wrow], tm, pe, seq, G,
                  status)
    assert CJ.cluster_keys(ref) == key(a)


def test_repro_bit_identical_per_seed():
    """R5: the jitted stages are bit-identical for the same seed and differ for another seed."""
    from crvfastsim.reco import repro
    from crvfastsim.reco.cli import _coinc_geom
    G = _coinc_geom()
    a, b, c = (repro.fingerprint(repro.run(150, G, seed=s, batch=2048)) for s in (5, 5, 6))
    assert a == b
    assert all(a[k] != c[k] for k in repro.STAGES)


def test_knobs_are_runtime_data_no_recompile():
    """Physics settings of the pulse reco, waveform synthesis and coincidence finder are data: sweeping them reuses the
    compiled kernels (one compilation per shape), and the result follows the knob."""
    import dataclasses
    from crvfastsim.reco import coinc as C
    from crvfastsim.reco import coinc_jax as CJ
    from crvfastsim.reco import synth
    from crvfastsim.reco import waveform as WF
    from crvfastsim.reco import waveform_jax as WJ
    from crvfastsim.reco.cli import _coinc_geom
    args = [np.asarray(a) for a in _batch(W=64)]
    base = PF.RecoConfig()
    n0 = PJ._reconstruct_dense._cache_size()
    npul = []
    for thr in (30.0, 40.0, 80.0, 160.0):
        for b in (17.0, 19.0):
            cfg = dataclasses.replace(base, min_adc_difference=thr, default_beta=b, max_beta=45.0)
            npul.append(int(PJ.reconstruct_dense(*args, cfg=cfg)["npulse"].sum()))
    assert PJ._reconstruct_dense._cache_size() - n0 == 1
    assert npul[0] >= npul[2] >= npul[4] >= npul[6] and npul[0] > npul[6]
    # waveform: noise and threshold
    rng = np.random.default_rng(1)
    chan = np.repeat(np.arange(20), 15)
    t = 300.0 + np.sort(rng.exponential(8.0, (20, 15)), 1).ravel()
    q = np.full(len(t), WF.WaveformConfig().single_pe_reference_charge)
    n1 = WJ.synth_dense._cache_size()
    nd = []
    for noise, mv in ((0.0, 0.0275), (4e-4, 0.0275), (8e-4, 0.05)):
        cfg = dataclasses.replace(WF.WaveformConfig(), noise=noise, min_voltage=mv)
        nd.append(len(WJ.synthesize(chan, t, q, np.zeros(20), cfg, rng_key=__import__("jax").random.PRNGKey(0),
                                    Q=16, L=24)["chan"]))
    assert WJ.synth_dense._cache_size() - n1 == 1 and nd[0] > 0
    # coincidence finder: PE threshold and time window of every sector
    G = _coinc_geom()
    P = synth.coinc_events(200, G, seed=3)
    status = np.zeros(4 * len(G.sector), np.int64)
    cols = [P[k] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")]
    tiers = ((32, 8, 256), (64, 12, 64))
    CJ.find(*cols, G, status, tiers=tiers)
    n2 = CJ.find_dense._cache_size()
    ncl = []
    for thr, dt in ((4.0, 20.0), (8.0, 10.0), (30.0, 10.0)):
        cuts = {k: dataclasses.replace(c, pe_threshold=thr, max_dt=dt if k != "EX" else 2 * dt)
                for k, c in C.EXTRACTED_V04.items()}
        ncl.append(len(CJ.find(*cols, G, status, cfg=C.FinderConfig(cuts=cuts), tiers=tiers)["cl_event"]))
    assert CJ.find_dense._cache_size() == n2
    assert ncl[0] >= ncl[1] >= ncl[2] and ncl[0] > ncl[2]


def test_coinc_jax_per_sector_cuts():
    """Cuts that differ between sectors of one type (Run 2 style) match the numpy port event by event."""
    import dataclasses
    from crvfastsim.reco import coinc as C
    from crvfastsim.reco import coinc_jax as CJ
    from crvfastsim.reco import synth
    from crvfastsim.reco.cli import _coinc_geom
    G = _coinc_geom()
    P = synth.coinc_events(600, G, seed=21, shower_frac=0.1, shower_mean=40)
    status = np.zeros(4 * len(G.sector), np.int64)
    cuts = dict(C.EXTRACTED_V04)
    for m in ("M1", "M3", "M5", "M7"):
        cuts[m] = dataclasses.replace(cuts[m], coincidence_layers=3, max_dt_adjacent=6.0)
    cuts["M2"] = dataclasses.replace(cuts["M2"], pe_threshold=20.0, max_slope_diff=1.0)
    cuts["M4"] = dataclasses.replace(cuts["M4"], min_slope=-1.0, max_slope=1.0, final_max_distance=80.0)
    cuts["T1"] = dataclasses.replace(cuts["T1"], coincidence_layers=3, max_dt=14.0)
    cuts["T2"] = dataclasses.replace(cuts["T2"], min_cluster_pes=400.0)
    cfg = C.FinderConfig(cuts=cuts)
    assert CJ.type_params(G, cfg)[4].ns == (3, 4)
    out = CJ.find(*[P[k] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")], G, status, cfg=cfg,
                  tiers=((16, 6, 256), (64, 12, 64)))
    ref, ev = set(), P["ev"]
    for e in np.unique(ev):
        a, b = np.searchsorted(ev, e), np.searchsorted(ev, e, side="right")
        p = {k: P[k][a:b] for k in ("idx", "bar", "sipm", "time", "pe", "seq")}
        ref |= {(int(e), r["sectortype"], tuple(sorted(r["pulses"]))) for r in C.find_event(p, G, status, cfg)}
    assert len(ref) > 100
    assert CJ.cluster_keys(out) == ref


def test_chain_jax_adds_db_time_offsets():
    """chain_jax.run shifts pulse times by the per-waveform DB time offset W['toff'] (CrvRecoPulsesFinder
    useTimeOffsetDB); with toff = 0 the clusters equal the run without it, and a uniform shift moves every cluster
    time by the same amount."""
    from crvfastsim.reco import chain_jax, synth
    from crvfastsim.reco.cli import _coinc_geom
    G = _coinc_geom()
    W = synth.chain_events(150, G, seed=4)
    status = np.zeros(4 * len(G.sector), np.int64)
    cat = lambda outs, k: np.concatenate([o[k] for o in outs])
    a, _ = chain_jax.run(W, G, status, n_chunks=1, batch=4096)
    b, _ = chain_jax.run(dict(W, toff=np.zeros(len(W["ev"]))), G, status, n_chunks=1, batch=4096)
    c, _ = chain_jax.run(dict(W, toff=np.full(len(W["ev"]), 7.5)), G, status, n_chunks=1, batch=4096)
    assert np.array_equal(cat(a, "cl_start"), cat(b, "cl_start")) and len(cat(a, "cl_start")) > 50
    assert np.allclose(cat(c, "cl_start") - cat(a, "cl_start"), 7.5)


def test_backend_tiers_skip_h32_on_oneapi(monkeypatch):
    """On oneAPI the H=32 tier is dropped (SPIR-V compile abort); other backends keep every tier."""
    import jax
    from crvfastsim.reco import coinc_jax as CJ
    tiers = ((16, 6, 1), (32, 8, 1), (64, 12, 1))
    assert CJ.backend_tiers(tiers) == (tiers if jax.devices()[0].platform != "oneapi" else (tiers[0], tiers[2]))

    class Dev:
        platform = "oneapi"
    monkeypatch.setattr(CJ.jax, "devices", lambda *a: [Dev()])
    assert CJ.backend_tiers(tiers) == (tiers[0], tiers[2])
