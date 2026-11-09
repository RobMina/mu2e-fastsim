"""R5 reproducibility: run the jitted fast-reco stages on seeded synthetic inputs and fingerprint their outputs.

Stages: waveform synthesis with noise (waveform_jax, JAX threefry key), pulse reco (pulsefit_jax, 3 rounds + numpy
redo) and the pipelined coincidence finder (chain_jax). `run` returns the output arrays; `fingerprint` hashes them per
stage. Same seed + same backend must give identical hashes (bit-identical); `compare` takes the outputs of two
backends (CPU vs GPU) and reports, per stage, the identical fraction, the largest differences and KS of the physics
columns (the R5 statistical-equality check: the PRNG is backend independent, so differences come only from float
rounding in the fits).
"""
import hashlib

import numpy as np

from mu2efastsim import match

STAGES = ("waveforms", "pulses", "clusters")


def _charges(n_events, seed, nchan=2304):
    """~20 SiPM charge clusters per event, KPP-like, as flat (chan row, time, charge) sorted by row then time."""
    from .waveform import WaveformConfig
    cfg = WaveformConfig()
    rng = np.random.default_rng(seed)
    C = 20 * n_events
    n = np.maximum(rng.poisson(28, C), 1)
    t0 = rng.uniform(100, 90000, C)
    row = np.repeat(np.arange(C), n)
    t = np.repeat(t0, n) + rng.exponential(8.0, n.sum())
    q = rng.uniform(0.8, 1.2, n.sum()) * cfg.single_pe_reference_charge
    o = np.lexsort((t, row))
    return row[o], t[o], q[o], rng.random(C) * 12.5


def run(n_events, G, seed=20261019, batch=16384):
    import jax
    from . import chain_jax, synth, waveform_jax as WJ
    from .waveform import WaveformConfig
    out = {}
    row, t, q, tdc0 = _charges(n_events, seed)
    d = WJ.synthesize(row, t, q, tdc0, WaveformConfig(), rng_key=jax.random.PRNGKey(seed))
    out["waveforms"] = {"chan": d["chan"], "starttdc": d["starttdc"], "adc": d["adc"]}
    W = synth.chain_events(n_events, G, seed=seed)
    status = np.zeros(4 * len(G.sector), np.int64)
    import dataclasses
    from . import pulsefit
    cfg = dataclasses.replace(pulsefit.RecoConfig(), max_pulses=3)
    rows = np.arange(len(W["ev"]))
    wrow, seq, pe, tm, _ = chain_jax._pulse_collect(chain_jax._pulse_dispatch(W, rows, cfg, batch, 12), W, rows,
                                                    pulsefit.RecoConfig())
    out["pulses"] = {"wf": wrow, "seq": seq, "pe": pe, "time": tm}
    outs, _ = chain_jax.run(W, G, status, n_chunks=2, batch=batch)
    cl = {k: np.concatenate([o[k] for o in outs]) for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end",
                                                             "cl_slope", "cl_avgtime")}
    o = np.lexsort((cl["cl_start"], cl["cl_type"], cl["cl_event"]))
    out["clusters"] = {k: v[o] for k, v in cl.items()}
    return out


def fingerprint(out):
    h = {}
    for s in STAGES:
        m = hashlib.sha256()
        for k in sorted(out[s]):
            m.update(k.encode())
            m.update(np.ascontiguousarray(out[s][k]).tobytes())
        h[s] = m.hexdigest()[:16]
    return h


def flatten(out):
    return {f"{s}__{k}": v for s in STAGES for k, v in out[s].items()}


def unflatten(z):
    out = {s: {} for s in STAGES}
    for key in z.files if hasattr(z, "files") else z:
        s, k = key.split("__", 1)
        out[s][k] = z[key]
    return out


def compare(a, b):
    """Backend comparison of two run() outputs."""
    res = {}
    wa, wb = a["waveforms"], b["waveforms"]
    ka = np.stack([wa["chan"], wa["starttdc"]], 1)
    kb = np.stack([wb["chan"], wb["starttdc"]], 1)
    ia, ib = match.join(ka, kb)
    same = (wa["adc"][ia] == wb["adc"][ib]).all(1) if len(ia) else np.zeros(0, bool)
    res["waveforms"] = {"digis": [int(len(ka)), int(len(kb))], "matched": int(len(ia)),
                        "sample_identical_frac": float(same.mean()) if len(ia) else None,
                        "max_abs_adc_diff": int(np.abs(wa["adc"][ia].astype(int) - wb["adc"][ib].astype(int)).max())
                        if len(ia) else None,
                        "adc_ks": dict(zip(("d", "p"), match.ks_2samp(wa["adc"].ravel() + 0.0, wb["adc"].ravel() + 0.0)))}
    pa, pb = a["pulses"], b["pulses"]
    ka, kb = np.stack([pa["wf"], pa["seq"]], 1), np.stack([pb["wf"], pb["seq"]], 1)
    ia, ib = match.join(ka, kb)
    res["pulses"] = {"pulses": [int(len(ka)), int(len(kb))], "matched": int(len(ia)),
                     "identical_frac": float(((pa["pe"][ia] == pb["pe"][ib]) & (pa["time"][ia] == pb["time"][ib])).mean())
                     if len(ia) else None,
                     "max_rel_pe_diff": float(np.max(np.abs(pa["pe"][ia] / pb["pe"][ib] - 1))) if len(ia) else None,
                     "max_abs_time_diff_ns": float(np.max(np.abs(pa["time"][ia] - pb["time"][ib]))) if len(ia) else None,
                     "pe_ks": dict(zip(("d", "p"), match.ks_2samp(pa["pe"], pb["pe"]))),
                     "time_ks": dict(zip(("d", "p"), match.ks_2samp(pa["time"], pb["time"])))}
    ca, cb = a["clusters"], b["clusters"]
    res["clusters"] = {"clusters": [int(len(ca["cl_event"])), int(len(cb["cl_event"]))],
                       "per_event_type_counts_equal": bool(np.array_equal(
                           np.unique(np.stack([ca["cl_event"], ca["cl_type"]], 1), axis=0, return_counts=True)[1],
                           np.unique(np.stack([cb["cl_event"], cb["cl_type"]], 1), axis=0, return_counts=True)[1]))
                       if len(ca["cl_event"]) == len(cb["cl_event"]) else False,
                       "pe_ks": dict(zip(("d", "p"), match.ks_2samp(ca["cl_pe"], cb["cl_pe"]))),
                       "avgtime_ks": dict(zip(("d", "p"), match.ks_2samp(ca["cl_avgtime"], cb["cl_avgtime"])))}
    if len(ca["cl_event"]) == len(cb["cl_event"]):
        res["clusters"]["max_abs_pe_diff"] = float(np.max(np.abs(ca["cl_pe"] - cb["cl_pe"]))) if len(ca["cl_pe"]) else 0.0
    ks_ok = all(v["p"] > 0.01 for st in res.values() for k, v in st.items() if k.endswith("_ks"))
    res["pass_statistical"] = bool(ks_ok)
    return res
