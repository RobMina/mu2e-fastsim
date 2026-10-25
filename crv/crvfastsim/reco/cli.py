"""Fast CRV reco commands: `python3 -m crvfastsim reco-<cmd> REF.root [...]` (alias: `python3 -m crvfastsim.reco <cmd>`)

  vr2       V-R2: stock digis -> fast pulse reco (numpy) vs the stock pulses in the same table
  vr2stock  oracle check: Offline's MakeCrvRecoPulses through PyROOT on the dumped digis vs the dumped pulses
            (proves the table and the pulse grouping reproduce the art job; needs a built muse area)
  vr1       V-R1: dumped SiPM charges -> fast waveforms + digis vs the dumped digis (noise off: compare digi MC
            voltages too)
  direct    direct path: fit the charge -> pulse model on half of the events, sample the other half, compare with
            the stock pulses and with the waveform path (fidelity and cost per event)
  cost      wall time per event of both paths on the same events (CPU, this host)

"""
import argparse
import json
import os
import sys
import time

import awkward as ak
import numpy as np

from mu2efastsim import match, plots, stats
from . import compare, pulsefit, reftables, waveform
from .direct import N_MODELLED, DirectModel, DirectModelV2, assign_pulses, cluster_features

NCHAN_PER_BAR = 4


def _digis(paths, max_events=None):
    d = reftables.read(paths, "digis")
    ev = reftables.event_key(d)
    if max_events:
        keep = np.isin(ev, np.unique(ev)[:max_events])
        d, ev = d[keep], ev[keep]
    adc = d.adc
    return {"ev": ev, "bar": ak.to_numpy(d.bar), "sipm": ak.to_numpy(d.sipm), "starttdc": ak.to_numpy(d.starttdc),
            "nzs": ak.to_numpy(d.nzs), "adc_flat": ak.to_numpy(ak.flatten(adc)), "lens": ak.to_numpy(ak.num(adc)),
            "idx": ak.to_numpy(d.idx)}


def _pulses(paths, events=None):
    p = reftables.read(paths, "pulses")
    ev = reftables.event_key(p)
    if events is not None:
        keep = np.isin(ev, events)
        p, ev = p[keep], ev[keep]
    chan = ak.to_numpy(p.bar) * NCHAN_PER_BAR + ak.to_numpy(p.sipm)
    wf0 = ak.to_numpy(ak.firsts(p.wfidx))
    return {"ev": ev, "chan": chan, "seq": ak.to_numpy(p.seq), "pe": ak.to_numpy(p.pe), "time": ak.to_numpy(p.time),
            "beta": ak.to_numpy(p.beta), "failed": (ak.to_numpy(p.flags) & 1).astype(bool),
            "key": compare.pulse_key(ev, wf0, ak.to_numpy(p.seq)), "wf": wf0}


def _prepare(paths, max_events, cfg):
    """Group digis into waveforms and attach conditions; returns arrays for pulsefit and the grouping."""
    D = _digis(paths, max_events)
    C = reftables.calib(paths)
    zs = D["nzs"] == 0
    # stock grouping runs over the collection in order; NZS digis come only with NZSdata (not used here)
    sel = {k: (v[zs] if len(v) == len(zs) else v) for k, v in D.items() if k not in ("adc_flat", "lens")}
    lens = D["lens"][zs]
    flat = D["adc_flat"][np.repeat(zs, D["lens"])]
    gid, mat, n, start, first = pulsefit.concat_waveforms(sel["ev"], sel["bar"], sel["sipm"], sel["starttdc"], flat,
                                                          lens)
    chan = sel["bar"][first] * NCHAN_PER_BAR + sel["sipm"][first]
    ok = np.ones(len(first), bool)
    if cfg.ignore_channels:
        ok = ~((C["status"][chan] >> 1) & 1).astype(bool)
    return {"ev": sel["ev"][first][ok], "chan": chan[ok], "digi0": sel["idx"][first][ok], "mat": mat[ok], "n": n[ok],
            "start": start[ok],
            "ped": C["pedestal"][chan[ok]], "area": C["pulsearea"][chan[ok]],
            "height": C["pulseheight"][chan[ok]], "toff": C["timeoffset"][chan[ok]], "events": np.unique(D["ev"])}


def _with_offsets(res, W, cfg):
    t_off = W["toff"][res["wf"]] * cfg.time_offset_scale if cfg.use_time_offset_db else 0.0
    return res["time"] + t_off   # offspill (extracted, KPP): TDC0time = 0 in CrvRecoPulsesFinder


def _pad_batches(W, L, batch):
    """Fixed-shape batches for the jitted kernel: rows padded to a multiple of `batch`, samples to L."""
    nw = len(W["n"])
    nb = max(1, -(-nw // batch))
    tot = nb * batch
    mat = np.zeros((tot, L), np.int64)
    m = min(L, W["mat"].shape[1])
    mat[:nw, :m] = W["mat"][:, :m]
    pad = lambda a, v: np.concatenate([a, np.full(tot - nw, v, dtype=np.asarray(a).dtype)])
    return (mat, pad(np.minimum(W["n"], L), 0), pad(W["start"], 0), pad(W["ped"].astype(float), 0.0),
            pad(W["area"].astype(float), 1.0), pad(W["height"].astype(float), 1.0)), nw, nb


def _run_jax(W, cfg, L=36, batch=65536, rounds=None):
    """pulsefit_jax over all waveforms in fixed (batch, L) shapes; longer waveforms and window overflows -> numpy."""
    import dataclasses

    import jax
    from . import pulsefit_jax as PJ
    cfg_np = cfg
    if rounds:
        cfg = dataclasses.replace(cfg, max_pulses=rounds)
    long_ = W["n"] > L
    (mat, n, st, ped, area, height), nw, nb = _pad_batches(W, L, batch)
    parts = []
    t0 = time.perf_counter()
    for b in range(nb):
        s = slice(b * batch, (b + 1) * batch)
        d = PJ.reconstruct_dense(mat[s], n[s], st[s], ped[s], area[s], height[s], cfg=cfg)
        parts.append(jax.device_get(d))
    dt = time.perf_counter() - t0
    dense = {k: np.concatenate([p[k] for p in parts])[:nw] for k in parts[0]}
    flat = PJ.to_flat(dense)
    redo = long_ | flat["window_overflow"] | flat["rounds_exhausted"]
    if redo.any():
        rows = np.flatnonzero(redo)
        sub = pulsefit.reconstruct_full(W["mat"][rows], W["n"][rows], W["start"][rows], W["ped"][rows], W["area"][rows],
                                   W["height"][rows], cfg_np)
        keep = ~np.isin(flat["wf"], rows)
        flat2 = {k: np.concatenate([flat[k][keep], (rows[sub[k]] if k == "wf" else sub[k])])
                 for k in ("wf", "seq", "pe", "time", "failed")}
        o = np.lexsort((flat2["seq"], flat2["wf"]))
        near = flat["wf_near_threshold"].copy()
        near[rows] = sub["wf_near_threshold"]
        flat = {k: v[o] for k, v in flat2.items()}
        flat["wf_near_threshold"] = near
    return flat, dt, int(redo.sum()), nb


def cmd_vr2(args, oracle=False, use_jax=False):
    cfg = pulsefit.RecoConfig()
    W = _prepare(args.inputs, args.max_events, cfg)
    t0 = time.perf_counter()
    extra = {}
    if use_jax:
        res, _, nredo, nb = _run_jax(W, cfg, args.length, args.batch, args.rounds)
        extra = {"numpy_fallback_waveforms": nredo, "batches": nb, "backend": __import__("jax").default_backend()}
    elif oracle:
        from . import stockfit
        ped_used = np.where(W["ped"] - W["mat"][:, 0] > cfg.pedestal_undershoot_threshold, W["mat"][:, 0], W["ped"])
        res = stockfit.reconstruct(W["mat"], W["n"], W["start"], ped_used, W["area"], W["height"], cfg)
    else:
        res = pulsefit.reconstruct(W["mat"], W["n"], W["start"], W["ped"], W["area"], W["height"], cfg)
    dt = time.perf_counter() - t0
    fast = {"key": compare.pulse_key(W["ev"][res["wf"]], W["digi0"][res["wf"]], res["seq"]), "pe": res["pe"],
            "time": _with_offsets(res, W, cfg), "failed": res["failed"]}
    ref = _pulses(args.inputs, W["events"])
    near = res.get("wf_near_threshold")
    excl = np.stack([W["ev"], W["digi0"]], 1)[near] if near is not None else None
    out = compare.vr2(fast, ref, excl)
    out.update({"events": int(len(W["events"])), "waveforms": int(len(W["n"])), "wall_s": dt,
                "us_per_event": 1e6 * dt / max(len(W["events"]), 1),
                "mode": "stock-oracle" if oracle else ("fast-jax" if use_jax else "fast"), **extra})
    _report(args, "vr2stock" if oracle else ("vr2jax" if use_jax else "vr2"), out)


def _bench_waveforms(args):
    """Throughput of the jitted waveform synthesis on synthetic SiPM-charge clusters (compile excluded)."""
    import jax
    from . import waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    rng = np.random.default_rng(args.seed)
    ne = args.max_events or 50000
    C = ne * 20                                    # KPP-like: ~20 digitised SiPM clusters per event
    Q = 64                                         # charges per cluster (median ~25 PE in data; padded)
    n = np.minimum(rng.poisson(28, C), Q)
    t0 = rng.uniform(100, 90000, C)
    T = t0[:, None] + np.sort(rng.exponential(8.0, (C, Q)), 1)
    W = np.where(np.arange(Q)[None, :] < n[:, None], rng.uniform(0.8, 1.2, (C, Q)), 0.0)
    tdc0 = rng.random(C) * 12.5
    t_start = np.ceil((t0 - 12.5 - tdc0) / 12.5) * 12.5 + tdc0
    tmpl = WJ.single_pe_template(cfg.single_pe_file, cfg.single_pe_precision, cfg.single_pe_stretch,
                                 cfg.single_pe_max_time)
    keep = n * tmpl.max() >= cfg.min_voltage
    L = 32
    B = args.batch
    nb = -(-C // B)
    pad = lambda a, v: np.concatenate([a, np.full((nb * B - C,) + a.shape[1:], v, a.dtype)])
    arrs = [jax.device_put(pad(x, v)) for x, v in ((T, 0.0), (W, 0.0), (n, 0), (t_start, 0.0), (keep, False))]
    key = jax.random.PRNGKey(args.seed)
    f = lambda b, k: WJ.synth_dense(*[a[b * B:(b + 1) * B] for a in arrs], tmpl, k, L, cfg.samples_zs,
                                    cfg.single_pe_precision, cfg.noise, cfg.min_voltage, cfg.adc_conversion,
                                    cfg.pedestal, cfg.min_adc, cfg.max_adc)
    t = time.perf_counter()
    jax.block_until_ready(f(0, key))
    comp = time.perf_counter() - t
    times = []
    for _ in range(args.repeat):
        t = time.perf_counter()
        jax.block_until_ready([f(b, jax.random.fold_in(key, b)) for b in range(nb)])
        times.append(time.perf_counter() - t)
    best = min(times)
    return {"stage": "waveforms", "backend": jax.default_backend(), "device": str(jax.devices()[0]), "events": ne,
            "clusters": C, "Q": Q, "L": L, "batch": B, "compile_s": comp, "pass_s": times, "events_per_s": ne / best,
            "us_per_event": 1e6 * best / ne,
            "stock_reference_us_per_event": {"CrvWaveforms_plus_CrvDigi_estimate": "part of the 19.8 ms CRV response"}}


def _coinc_geom():
    from ..geometry import find_counters, from_offline_config, parse_simpleconfig
    from . import coinc as C
    cf = find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt")
    return C.sector_geom(from_offline_config(cf), parse_simpleconfig(open(cf).read()))


def _ref_pulses_sorted(paths):
    p = reftables.read(paths, "pulses")
    ev = reftables.event_key(p)
    o = np.argsort(ev, kind="stable")
    out = {k: ak.to_numpy(p[k])[o] for k in ("idx", "bar", "sipm", "time", "pe", "seq")}
    out["ev"] = ev[o]
    return out


def _bench_coinc(args):
    """Throughput of the JAX coincidence finder (coinc_jax.find incl. host packing and numpy fallback)."""
    import jax
    from . import coinc_jax as CJ
    G = _coinc_geom()
    if args.inputs == ["synthetic"]:
        from . import synth
        P = synth.coinc_events(args.max_events or 50000, G, seed=args.seed)
        status = np.zeros(4 * len(G.sector), np.int64)
    else:
        P = _ref_pulses_sorted(args.inputs)
        status = reftables.calib(args.inputs)["status"]
    cols = [P[k] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")]
    ne = len(np.unique(P["ev"]))
    t0 = time.perf_counter()
    CJ.find(*cols, G, status)
    compile_s = time.perf_counter() - t0
    times, tms = [], []
    for _ in range(args.repeat):
        tm = {}
        t0 = time.perf_counter()
        out = CJ.find(*cols, G, status, timers=tm)
        times.append(time.perf_counter() - t0)
        tms.append(tm)
    best = int(np.argmin(times))
    return {"backend": jax.default_backend(), "device": str(jax.devices()[0]), "events": int(ne),
            "pulses": int(len(P["ev"])), "clusters": int(len(out["cl_event"])),
            "fallback_events": int(len(out["fallback_events"])), "first_call_s": compile_s, "pass_s": times,
            "timers_best": tms[best], "events_per_s": ne / times[best], "us_per_event": 1e6 * times[best] / ne}


def _bench_chain(args):
    """Pulses -> coincidences in one pipelined pass (chain_jax) on synthetic waveforms that make coincidences."""
    import jax
    from . import chain_jax, synth
    G = _coinc_geom()
    ne = args.max_events or 100000
    W = synth.chain_events(ne, G, seed=args.seed, L=args.length)
    status = np.zeros(4 * len(G.sector), np.int64)
    rounds = args.rounds or 3
    t0 = time.perf_counter()
    chain_jax.run(W, G, status, n_chunks=args.chunks, rounds=rounds, batch=args.batch, L=args.length)   # compile
    first = time.perf_counter() - t0
    best, tms = None, []
    for _ in range(args.repeat):
        tm = {}
        outs, tm = chain_jax.run(W, G, status, n_chunks=args.chunks, rounds=rounds, batch=args.batch, L=args.length,
                                 timers=tm)
        tms.append(tm)
        best = tm if best is None or tm["wall_s"] < best["wall_s"] else best
    ncl = int(sum(len(o["cl_event"]) for o in outs))
    nfb = int(sum(len(o["fallback_events"]) for o in outs))
    return {"backend": jax.default_backend(), "device": str(jax.devices()[0]), "events": best["events"],
            "waveforms": best["waveforms"], "pulses": best.get("pulses"), "clusters": ncl, "coinc_fallback_events": nfb,
            "rounds": rounds, "chunks": args.chunks, "first_call_s": first, "pass_wall_s": [t["wall_s"] for t in tms],
            "timers_best": best, "events_per_s": best["events"] / best["wall_s"],
            "us_per_event": 1e6 * best["wall_s"] / best["events"]}


def cmd_bench(args):
    """Throughput of the jitted pulse reco (or, with --stage waveforms, the waveform synthesis); compile excluded."""
    import jax
    if getattr(args, "stage", "pulses") == "waveforms":
        return _report(args, "bench_waveforms", _bench_waveforms(args))
    if getattr(args, "stage", "pulses") == "coinc":
        return _report(args, "bench_coinc", _bench_coinc(args))
    if getattr(args, "stage", "pulses") == "chain":
        return _report(args, "bench_chain", _bench_chain(args))
    from . import pulsefit_jax as PJ
    cfg = pulsefit.RecoConfig()
    if args.inputs == ["synthetic"]:
        from . import synth
        ne = args.max_events or 200000
        adc, n_, st_, ped_, area_, h_ = synth.waveforms(ne, seed=args.seed, L=args.length)
        W = {"mat": adc, "n": n_, "start": st_, "ped": ped_, "area": area_, "height": h_, "events": np.arange(ne)}
    else:
        W = _prepare(args.inputs, args.max_events, cfg)
    (mat, n, st, ped, area, height), nw, nb = _pad_batches(W, args.length, args.batch)
    if args.rounds:
        import dataclasses
        cfg = dataclasses.replace(cfg, max_pulses=args.rounds)
    devs = jax.devices()
    t0 = time.perf_counter()
    jax.block_until_ready(PJ.reconstruct_dense(mat[:args.batch], n[:args.batch], st[:args.batch], ped[:args.batch],
                                               area[:args.batch], height[:args.batch], cfg=cfg))
    compile_s = time.perf_counter() - t0
    dev = [jax.device_put(x, devs[0]) for x in (mat, n, st, ped, area, height)]
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        outs = [PJ.reconstruct_dense(*[x[b * args.batch:(b + 1) * args.batch] for x in dev], cfg=cfg)
                for b in range(nb)]
        jax.block_until_ready(outs)
        times.append(time.perf_counter() - t0)
    best = min(times)
    exhausted = int(np.asarray(outs[0]["rounds_exhausted"]).sum()) if outs else 0
    ne = len(W["events"])
    out = {"backend": jax.default_backend(), "device": str(devs[0]), "n_devices": len(devs), "events": int(ne),
           "waveforms": int(nw), "batch": args.batch, "length": args.length, "batches": nb, "compile_s": compile_s,
           "rounds": cfg.max_pulses, "rounds_exhausted_first_batch": exhausted,
           "pass_s": times, "events_per_s": ne / best, "waveforms_per_s": nw / best,
           "us_per_event": 1e6 * best / max(ne, 1),
           "stock_reference_us_per_event": {"CrvRecoPulsesFinder_A8_kpp": 3730, "CrvRecoPulsesFinder_A6_mc": 1560}}
    _report(args, "bench", out)


def cmd_inspect(args):
    """Waveforms whose pulses differ: ADCs, pedestal, fast pulses and reference pulses, for --keys from a vr2 JSON."""
    keys = json.load(open(args.keys))
    want = np.array(keys.get("unmatched_fast_keys", []) + keys.get("unmatched_ref_keys", []), np.int64).reshape(-1, 3)
    cfg = pulsefit.RecoConfig()
    W = _prepare(args.inputs, None, cfg)
    wkey = np.stack([W["ev"], W["digi0"]], 1)
    ia, _ = compare._join(wkey, np.unique(want[:, :2], axis=0))
    if not len(ia):
        print("no waveform found for the keys")
        return
    sub = {k: (v[ia] if isinstance(v, np.ndarray) and len(v) == len(W["ev"]) else v) for k, v in W.items()}
    res = pulsefit.reconstruct(sub["mat"], sub["n"], sub["start"], sub["ped"], sub["area"], sub["height"], cfg)
    ref = _pulses(args.inputs, np.unique(sub["ev"]))
    out = []
    for i in range(len(ia)):
        rsel = (ref["key"][:, 0] == sub["ev"][i]) & (ref["key"][:, 1] == sub["digi0"][i])
        fsel = res["wf"] == i
        out.append({"event_key": int(sub["ev"][i]), "digi0": int(sub["digi0"][i]), "chan": int(sub["chan"][i]),
                    "start_tdc": int(sub["start"][i]), "n": int(sub["n"][i]), "ped_db": float(sub["ped"][i]),
                    "adc": sub["mat"][i, :sub["n"][i]].tolist(),
                    "fast": {k: res[k][fsel].tolist() for k in ("seq", "pe", "time", "height", "beta", "failed",
                                                                "fit_A", "fit_mu", "fit_b", "ped")},
                    "ref": {k: ref[k][rsel].tolist() for k in ("seq", "pe", "time", "beta", "failed")}})
    _report(args, "inspect", {"waveforms": out})


def _charges(paths, events=None):
    q = reftables.read(paths, "charges")
    ev = reftables.event_key(q)
    if events is not None:
        keep = np.isin(ev, events)
        q, ev = q[keep], ev[keep]
    nq = ak.to_numpy(ak.num(q.time))
    chan_row = ak.to_numpy(q.bar) * NCHAN_PER_BAR + ak.to_numpy(q.sipm)
    rowid = np.arange(len(nq))
    return {"ev_row": ev, "chan_row": chan_row, "row": np.repeat(rowid, nq), "time": ak.to_numpy(ak.flatten(q.time)),
            "charge": ak.to_numpy(ak.flatten(q.charge)), "pe": ak.to_numpy(ak.flatten(q.pe))}


def _vr1_noise(args, Q, out, m, mev, mchan, rv, radc, cfg):
    """V-R1 with noise: the reference is the stock chain with noise on (same charges, same TDC0). Per-channel ADC
    distributions (all samples of all digis) by KS, plus digi counts per channel and sample-level summaries."""
    fch = Q["chan_row"][out["chan"]]
    fadc = out["adc"].astype(np.int64)
    nch = int(max(fch.max(initial=0), mchan.max(initial=0))) + 1
    nd_f, nd_r = np.bincount(fch, minlength=nch), np.bincount(mchan, minlength=nch)
    # per-channel ADC KS over all samples; channels with >= min_digis digis in both arms
    of, orr = np.argsort(fch, kind="stable"), np.argsort(mchan, kind="stable")
    sf, sr = np.r_[0, np.cumsum(nd_f)], np.r_[0, np.cumsum(nd_r)]
    chans = np.flatnonzero((nd_f >= args.min_digis) & (nd_r >= args.min_digis))
    ps, ds = [], []
    for c in chans:
        a = fadc[of[sf[c]:sf[c + 1]]].ravel()
        b = radc[orr[sr[c]:sr[c + 1]]].astype(np.int64).ravel()
        d, p = match.ks_2samp(a + 0.0, b + 0.0)
        ds.append(d)
        ps.append(p)
    ps, ds = np.array(ps), np.array(ds)
    # pedestal-region samples (first two of each digi: before the signal crosses threshold) -> noise width
    integ_f, integ_r = (fadc - cfg.pedestal).sum(1), (radc.astype(np.int64) - cfg.pedestal).sum(1)
    ntest = len(chans)
    res = {"digis_fast": int(len(fch)), "digis_ref": int(len(mchan)),
           "digis_per_event": {"fast": len(fch) / args._nev, "ref": len(mchan) / args._nev},
           "channels_tested": int(ntest), "min_digis_per_channel": args.min_digis,
           "per_channel_ks_p": {"min": float(ps.min()) if ntest else None,
                                "frac_below_0.01": float((ps < 0.01).mean()) if ntest else None,
                                "expected_frac_below_0.01": 0.01,
                                "quantiles_5_50_95": np.quantile(ps, [0.05, 0.5, 0.95]).tolist() if ntest else None},
           "per_channel_ks_d_max": float(ds.max()) if ntest else None,
           "digi_count_ratio_per_channel_q": np.quantile(nd_f[chans] / nd_r[chans], [0.05, 0.5, 0.95]).tolist()
           if ntest else None,
           "ks_note": "per-channel and pooled ADC KS are dominated by the shared signal (noise 0.92 ADC rms); the "
                      "noise itself is tested by noise_residual (each arm minus the noise-free reference)",
           "digi_integral": compare.distribution_summary({"pe": integ_f}, {"pe": integ_r}, columns=("pe",))["pe"],
           "pooled_adc_ks": dict(zip(("d", "p"), match.ks_2samp(fadc.ravel() + 0.0, radc.astype(np.int64).ravel() + 0.0)))}
    if args.nonoise_ref:
        res["noise_residual"] = _noise_residual(args, out, Q, mev, mchan, radc, cfg)
    # with independent noise, a per-channel KS at p>0.01 fails 1 % of channels by chance: pass if the failing
    # fraction is consistent with 1 % (binomial, 3 sigma)
    if ntest:
        k, n = int((ps < 0.01).sum()), ntest
        res["pass_per_channel_ks"] = bool(k <= 0.01 * n + 3 * np.sqrt(0.01 * 0.99 * n))
        res["pass"] = res["pass_per_channel_ks"] and res.get("noise_residual", {}).get("pass", True)
    return res


def _noise_residual(args, out, Q, mev, mchan, radc, cfg):
    """Noise test proper: on digis present in all three tables (fast-with-noise, stock-with-noise, stock-noise-free;
    same event, channel, startTDC) the sample residual of each noisy arm against the noise-free stock digi. Same
    width and shape = same noise model; mean ~0 = no bias; neighbour-sample correlation (stock noise is white)."""
    m0 = reftables.read([args.nonoise_ref], "digimc", columns=["run", "subrun", "event", "bar", "sipm", "start",
                                                                "tdc0", "volt"])
    ev0 = reftables.event_key(m0)
    keep = np.isin(ev0, np.unique(mev))
    m0, ev0 = m0[keep], ev0[keep]
    ch0 = ak.to_numpy(m0.bar) * NCHAN_PER_BAR + ak.to_numpy(m0.sipm)
    v0 = ak.to_numpy(ak.fill_none(ak.pad_none(m0.volt, cfg.samples_zs, clip=True), 0.0))
    adc0 = np.clip((v0 * cfg.adc_conversion + cfg.pedestal + 0.5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    st0 = np.rint((ak.to_numpy(m0.start) - ak.to_numpy(m0.tdc0)) / pulsefit.PERIOD).astype(np.int64)
    k0 = np.stack([ev0, ch0, st0], 1).astype(np.int64)
    m = reftables.read(args.inputs, "digimc", columns=["run", "subrun", "event", "start", "tdc0"])
    m = m[np.isin(reftables.event_key(m), np.unique(mev))]
    kr = np.stack([mev, mchan, np.rint((ak.to_numpy(m.start) - ak.to_numpy(m.tdc0)) / pulsefit.PERIOD)], 1).astype(np.int64)
    kf = np.stack([Q["ev_row"][out["chan"]], Q["chan_row"][out["chan"]], out["starttdc"]], 1).astype(np.int64)
    i0r, ir = compare._join(k0, kr)
    i0f, iff = compare._join(k0, kf)
    to_r, to_f = np.full(len(k0), -1), np.full(len(k0), -1)
    to_r[i0r], to_f[i0f] = ir, iff
    both = np.flatnonzero((to_r >= 0) & (to_f >= 0))
    rr = radc[to_r[both]].astype(np.int64) - adc0[both]
    rf = out["adc"][to_f[both]].astype(np.int64) - adc0[both]
    # samples away from saturation and the ADC floor
    ok = (adc0[both] > cfg.min_adc + 5) & (adc0[both] < cfg.max_adc - 5)
    a, b = rf[ok].astype(float), rr[ok].astype(float)
    corr = lambda r: float(np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1])
    d, p = match.ks_2samp(a, b)
    expect = cfg.noise * cfg.adc_conversion
    return {"digis_in_all_three": int(len(both)), "samples": int(ok.sum()),
            "expected_rms_adc": float(np.sqrt(expect ** 2 + 1 / 12)),
            "rms_adc": {"fast": float(a.std()), "ref": float(b.std())}, "mean_adc": {"fast": float(a.mean()),
                                                                                  "ref": float(b.mean())},
            "hist_-4_to_4": {"fast": np.bincount(np.clip(a, -4, 4).astype(int) + 4, minlength=9).tolist(),
                             "ref": np.bincount(np.clip(b, -4, 4).astype(int) + 4, minlength=9).tolist()},
            "ks": {"d": d, "p": p}, "neighbour_corr": {"fast": corr(rf), "ref": corr(rr)},
            "pass": bool(p > 0.01)}


def cmd_vr1(args):
    """Noise-free V-R1 against the dumped digi MC voltages (rng None), with TDC0 taken from the reference digi MC.
    --noise: the fast arm adds noise (JAX key or numpy rng); compare with a noise-on reference statistically."""
    cfg = waveform.WaveformConfig()
    Q = _charges(args.inputs)
    events = np.unique(Q["ev_row"])
    if args.max_events:
        events = events[:args.max_events]
        keep = np.isin(Q["ev_row"][Q["row"]], events)
        Q = {**Q, **{k: Q[k][keep] for k in ("row", "time", "charge", "pe")}}
    m = reftables.read(args.inputs, "digimc")
    mev = reftables.event_key(m)
    mk = np.isin(mev, events)
    m, mev = m[mk], mev[mk]
    mchan = ak.to_numpy(m.bar) * NCHAN_PER_BAR + ak.to_numpy(m.sipm)
    # TDC0 is drawn per FEB and event in the stock chain: take it from any reference digi on the same FEB
    feb = reftables.calib(args.inputs)["feb"]
    mkey = np.stack([mev, feb[mchan]], 1)
    lab_key = np.stack([Q["ev_row"], feb[Q["chan_row"]]], 1)
    _, fi = np.unique(compare._as_void(mkey), return_index=True)
    tdc0_row = compare.lookup(lab_key, mkey[fi], ak.to_numpy(m.tdc0)[fi])
    n_no_tdc0 = int(np.isnan(tdc0_row).sum())
    tdc0_row = np.nan_to_num(tdc0_row, nan=0.0)
    noise = getattr(args, "noise", False)
    t0 = time.perf_counter()
    if getattr(args, "jax", False):
        from . import waveform_jax
        from .coinc_jax import is_oneapi
        if is_oneapi():
            out = _synth_jax_bucketed(waveform_jax, Q, tdc0_row, cfg, noise=cfg.noise if noise else 0.0, seed=args.seed)
        else:
            out = _synth_jax_batched(waveform_jax, Q, tdc0_row, cfg, args.batch_clusters,
                                     noise=cfg.noise if noise else 0.0, seed=args.seed)
    else:
        out = waveform.synthesize(Q["row"], Q["time"], Q["charge"], tdc0_row, cfg,
                                  rng=np.random.default_rng(args.seed) if noise else None)
    dt = time.perf_counter() - t0
    fkey = np.stack([Q["ev_row"][out["chan"]], Q["chan_row"][out["chan"]], out["starttdc"]], 1).astype(np.int64)
    rv = ak.to_numpy(ak.fill_none(ak.pad_none(m.volt, cfg.samples_zs, clip=True), 0.0))
    radc = np.clip((rv * cfg.adc_conversion + cfg.pedestal + 0.5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    rstart = np.rint((ak.to_numpy(m.start) - ak.to_numpy(m.tdc0)) / pulsefit.PERIOD).astype(np.int64)
    rkey = np.stack([mev, mchan, rstart], 1).astype(np.int64)
    if noise:
        args._nev = len(events)
        res = _vr1_noise(args, Q, out, m, mev, mchan, rv, radc, cfg)
        res.update({"events": int(len(events)), "wall_s": dt, "us_per_event": 1e6 * dt / max(len(events), 1),
                    "fast": "jax" if getattr(args, "jax", False) else "numpy",
                    "note": "reference = stock chain with noise on (crvreco_mc); same charges and TDC0"})
        return _report(args, "vr1noise", res)
    res = compare.vr1({"key": fkey, "adc": out["adc"]}, {"key": rkey, "adc": radc})
    ia, ib = compare._join(fkey, rkey)
    dv = np.abs(out["volt"][ia] - rv[ib]) if len(ia) else np.zeros((0, 1))
    ua = np.setdiff1d(np.arange(len(fkey)), ia)
    unknown = np.isnan(compare.lookup(lab_key, mkey[fi], ak.to_numpy(m.tdc0)[fi]))
    extra = {"n": int(len(ua)), "n_on_feb_without_reference_tdc0": int(unknown[out["chan"][ua]].sum()),
             "start_minus_window_end_ns": np.quantile(out["start"][ua] - args.window_end, [0, 0.5, 1]).tolist()
             if len(ua) else None,
             "start_rel_first_charge_ns": None,
             "examples": fkey[ua][:10].tolist()}
    res.update({"unmatched_fast": extra, "sipm_rows_without_feb_tdc0": n_no_tdc0,
                "note": "fast path noise off; compare with a reference made with CrvWaveforms.noise 0",
                "volt_abs_diff_quantiles_mV": (1e3 * np.quantile(dv, [0.5, 0.9, 0.99])).tolist() if dv.size else None,
                "events": int(len(events)), "wall_s": dt, "us_per_event": 1e6 * dt / max(len(events), 1)})
    _report(args, "vr1", res)


def _synth_jax_batched(WJ, Q, tdc0_row, cfg, batch, noise=0.0, seed=0):
    """JAX synthesis over all SiPM rows in fixed-shape batches of whole rows (Q and L padded per batch to a bucket)."""
    rows = Q["row"]
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    edges = list(starts[::batch]) + [len(rows)]
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        sl = slice(a, b)
        pk = WJ.pack(rows[sl], Q["time"][sl], Q["charge"][sl], tdc0_row, cfg)
        Qb = int(2 ** np.ceil(np.log2(max(pk["T"].shape[1], 1))))
        Lb = int(2 ** np.ceil(np.log2(max(pk["L"], 1))))
        pk = WJ.pack(rows[sl], Q["time"][sl], Q["charge"][sl], tdc0_row, cfg, Q=Qb, L=Lb)
        wave, adc, st, last = WJ.synth_dense(pk["T"], pk["W"], pk["n"], pk["t_start"], pk["keep"], pk["tmpl"],
                                             __import__("jax").random.fold_in(__import__("jax").random.PRNGKey(seed),
                                                                              a), pk["L"], cfg.samples_zs,
                                             cfg.single_pe_precision, noise, cfg.min_voltage, cfg.adc_conversion,
                                             cfg.pedestal, cfg.min_adc, cfg.max_adc)
        parts.append(WJ.to_digis(pk, np.asarray(wave), np.asarray(adc), np.asarray(st), np.asarray(last), cfg,
                                 (0.0, 1e5)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


SYNTH_BUCKET_GIB = 2.0   # [C, Q, L] float64 per call in _synth_jax_bucketed


def _p2(x):
    return (2 ** np.ceil(np.log2(np.maximum(x, 1)))).astype(np.int64)


def _synth_jax_bucketed(WJ, Q, tdc0_row, cfg, noise=0.0, seed=0, budget_gib=SYNTH_BUCKET_GIB, window=(0.0, 1e5)):
    """_synth_jax_batched with charge clusters grouped by padded shape (power-of-two charges Q and samples L) and each
    call filled up to budget_gib: few calls (~30 for a 22 k-event MC file instead of ~300) and no batch padded to the
    largest cluster. Used on oneAPI, where many small noisy calls intermittently abort the Level Zero runtime
    (urEventWait on an internal event), and by the chain (chain.digis). Noise off: same digis as _synth_jax_batched;
    noise on: another noise draw. tdc0_row: array indexed by row, or a callable of the row labels (WJ.pack)."""
    import jax
    rows, t, q = Q["row"], Q["time"], Q["charge"]
    cid = waveform.charge_clusters(rows, t, cfg)
    if not len(cid):
        return _synth_jax_batched(WJ, Q, tdc0_row, cfg, 1, noise, seed)
    C = int(cid[-1]) + 1
    first = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    nq = np.bincount(cid, minlength=C)
    span = np.zeros(C)
    np.maximum.at(span, cid, t - t[first][cid])
    need = (np.ceil(span + cfg.single_pe_max_time) / pulsefit.PERIOD).astype(np.int64) + 2 + cfg.samples_zs  # as WJ.pack
    qb, lb = _p2(nq), _p2(need)
    shape = qb * (1 << 20) + lb
    root = jax.random.PRNGKey(seed)
    parts, call = [], 0
    for k in np.unique(shape):
        Qb, Lb = int(k >> 20), int(k & ((1 << 20) - 1))
        cl = np.flatnonzero(shape == k)                                 # clusters in their original order
        cap = int(2 ** np.floor(np.log2(max(1, budget_gib * 2 ** 30 / (Qb * Lb * 8)))))
        for a in range(0, len(cl), cap):
            c = cl[a:a + cap]
            lens = nq[c]
            idx = np.repeat(first[c] - np.r_[0, np.cumsum(lens)[:-1]], lens) + np.arange(lens.sum())
            pk = WJ.pack(rows[idx], t[idx], q[idx], tdc0_row, cfg, Q=Qb, L=Lb)
            n = len(pk["n"])
            Cp = cap if n == cap else int(_p2(n))                       # few distinct shapes -> few compiles
            pad = lambda x, v: np.concatenate([x, np.full((Cp - n,) + x.shape[1:], v, x.dtype)]) if Cp > n else x
            wave, adc, st, last = WJ.synth_dense(pad(pk["T"], 0.0), pad(pk["W"], 0.0), pad(pk["n"], 0),
                                                 pad(pk["t_start"], 0.0), pad(pk["keep"], False), pk["tmpl"],
                                                 jax.random.fold_in(root, call), Lb, cfg.samples_zs,
                                                 cfg.single_pe_precision, noise, cfg.min_voltage, cfg.adc_conversion,
                                                 cfg.pedestal, cfg.min_adc, cfg.max_adc)
            parts.append(WJ.to_digis(pk, np.asarray(wave)[:n], np.asarray(adc)[:n], np.asarray(st)[:n],
                                     np.asarray(last)[:n], cfg, window))
            call += 1
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    o = np.lexsort((out["start"], out["chan"]))                         # the row order of _synth_jax_batched
    return {k: v[o] for k, v in out.items()}


def _direct_inputs(paths, max_events):
    """Charge clusters, the stock pulses assigned to them (time-ordered, DB time offset removed) and the reference
    TDC0 of each cluster's (event, FEB)."""
    cfg = waveform.WaveformConfig()
    Q = _charges(paths)
    events = np.unique(Q["ev_row"])
    if max_events:
        events = events[:max_events]
    keep = np.isin(Q["ev_row"][Q["row"]], events)
    feat = cluster_features(Q["row"][keep], Q["time"][keep], Q["pe"][keep], cfg)
    P = _pulses(paths, events)
    C_ = reftables.calib(paths)
    rk = (Q["ev_row"].astype(np.int64) << 16) | Q["chan_row"]
    pk = (P["ev"].astype(np.int64) << 16) | P["chan"]
    srt = np.argsort(rk)
    pos = np.clip(np.searchsorted(rk[srt], pk), 0, len(srt) - 1)
    prow = np.where(rk[srt][pos] == pk, srt[pos], -1)
    tn = P["time"] - C_["timeoffset"][P["chan"]]
    cl = np.full(len(pk), -1, np.int64)
    okp = prow >= 0
    cl[okp] = assign_pulses(feat, prow[okp], tn[okp], cfg)
    Cn = len(feat["q"])
    npul = np.bincount(cl[cl >= 0], minlength=Cn)
    a = np.flatnonzero(cl >= 0)
    o = a[np.lexsort((tn[a], cl[a]))]
    first = np.searchsorted(cl[o], cl[o])
    rank = np.arange(len(o)) - first
    m = rank < N_MODELLED
    pul = {k: np.full((Cn, N_MODELLED), np.nan) for k in ("pe", "time", "beta")}
    pul["failed"] = np.zeros((Cn, N_MODELLED), bool)
    for k, src in (("pe", P["pe"]), ("time", tn), ("beta", P["beta"]), ("failed", P["failed"])):
        pul[k][cl[o][m], rank[m]] = src[o][m]
    # reference TDC0 per (event, FEB) from the digi MC
    md = reftables.read(paths, "digimc", columns=["run", "subrun", "event", "bar", "sipm", "tdc0"])
    mev = reftables.event_key(md)
    mch = ak.to_numpy(md.bar) * NCHAN_PER_BAR + ak.to_numpy(md.sipm)
    feb = C_["feb"]
    cev, cch = Q["ev_row"][feat["chan"]], Q["chan_row"][feat["chan"]]
    kref = mev * 4096 + feb[mch]
    u, fi = np.unique(kref, return_index=True)
    kc = cev * 4096 + feb[cch]
    j = np.clip(np.searchsorted(u, kc), 0, len(u) - 1)
    tdc0 = np.where(u[j] == kc, ak.to_numpy(md.tdc0)[fi][j], np.nan)
    return dict(feat=feat, npul=npul, pul=pul, tdc0=tdc0, cev=cev, cch=cch, calib=C_, P=P, cl=cl, events=events,
                feb=feb)


def _coinc_from(ev, chan, time, pe, seq, G, status):
    from . import coinc_jax as CJ
    o = np.lexsort((seq, time, chan, ev))
    ev, chan, time, pe, seq = ev[o], chan[o], time[o], pe[o], seq[o]
    idx = np.arange(len(ev)) - np.searchsorted(ev, ev)
    return CJ.find(ev, idx, chan // NCHAN_PER_BAR, chan % NCHAN_PER_BAR, time, pe, seq, G, status)


def cmd_direct(args):
    """Direct path v2: fit on even events, test on odd. Per cluster pulse count and the first N_MODELLED pulses vs
    stock, then the coincidence finder on direct-path pulses vs on stock pulses of the same events (V-R4 style)."""
    if getattr(args, "v1", False):
        return cmd_direct_v1(args)
    D = _direct_inputs(args.inputs, args.max_events)
    feat, npul, pul, cev = D["feat"], D["npul"], D["pul"], D["cev"]
    Cn = len(feat["q"])
    rng = np.random.default_rng(args.seed)
    train = (cev % 2) == 0
    fsel = lambda m: {k: (v[m] if isinstance(v, np.ndarray) and len(v) == Cn else v) for k, v in feat.items()}
    fit = train & (np.isfinite(D["tdc0"]) | (npul == 0))   # pulsed clusters need their FEB's reference TDC0
    model = DirectModelV2.fit(fsel(fit), npul[fit], {k: v[fit] for k, v in pul.items()}, np.nan_to_num(D["tdc0"])[fit])
    te = ~train
    ft = fsel(te)
    # one TDC0 per (event, FEB), uniform over a clock period, as stock
    kc = cev[te] * 4096 + D["feb"][D["cch"][te]]
    uk, inv = np.unique(kc, return_inverse=True)
    t0 = time.perf_counter()
    tdc0 = (args.feb_phase_offset + rng.random(len(uk)) * args.feb_phase_spread)[inv]
    s = model.sample(ft, tdc0, rng)
    dt_sample = time.perf_counter() - t0
    toff = D["calib"]["timeoffset"][D["cch"][te]]
    n_ref, n_fast = npul[te], s["n"]
    ref_p = {k: v[te] for k, v in pul.items()}
    out = {"model": "v2 joint resampling, TDC0 per (event, FEB)", "clusters_test": int(te.sum()),
           "events_test": int(len(np.unique(cev[te]))),
           "npulse_hist": {"ref": (np.bincount(np.minimum(n_ref, 4), minlength=5) / len(n_ref)).tolist(),
                           "fast": (np.bincount(np.minimum(n_fast, 4), minlength=5) / len(n_fast)).tolist()},
           "found_frac_vs_q": _eff_vs_q(ft["q"], n_ref > 0, n_fast > 0)}
    for k in range(min(2, N_MODELLED)):
        fast = {"pe": s["pe"][:, k], "time": s["time"][:, k], "beta": s["beta"][:, k]}
        ref = {"pe": ref_p["pe"][:, k], "time": ref_p["time"][:, k], "beta": ref_p["beta"][:, k]}
        if k > 0:
            fast["time"] = s["time"][:, k] - s["time"][:, 0]
            ref["time"] = ref_p["time"][:, k] - ref_p["time"][:, 0]
        out[f"pulse{k}"] = {"summary": compare.distribution_summary(fast, ref),
                            "binned": {c: _band(fast[c], ref[c]) for c in ("pe", "beta", "time")}}
    out["failed_frac"] = {"ref": float(ref_p["failed"][:, 0][n_ref > 0].mean()),
                          "fast": float(s["failed"][:, 0][n_fast > 0].mean())}
    # downstream: coincidences from stock pulses vs direct-path pulses, same test events
    G = _coinc_geom()
    status = D["calib"]["status"]
    P = D["P"]
    pm = (P["ev"] % 2) == 1
    ref_cl = _coinc_from(P["ev"][pm], P["chan"][pm], P["time"][pm], P["pe"][pm], P["seq"][pm], G, status)
    h = s["has"]
    ci, ki = np.nonzero(h)
    t_fast = s["time"][ci, ki] + toff[ci]
    fast_cl = _coinc_from(cev[te][ci], D["cch"][te][ci], t_fast, s["pe"][ci, ki], ki, G, status)
    ne = out["events_test"]
    per_type = lambda c: {int(t): int((c["cl_type"] == t).sum()) for t in np.unique(np.r_[ref_cl["cl_type"],
                                                                                          fast_cl["cl_type"]])}
    out["coincidences"] = {"per_event": {"ref": len(ref_cl["cl_type"]) / ne, "fast": len(fast_cl["cl_type"]) / ne},
                           "ratio": len(fast_cl["cl_type"]) / max(len(ref_cl["cl_type"]), 1),
                           "per_sector_type": {"ref": per_type(ref_cl), "fast": per_type(fast_cl)},
                           "cluster_pe": _band(fast_cl["cl_pe"], ref_cl["cl_pe"]),
                           "cluster_pe_median": {"ref": float(np.median(ref_cl["cl_pe"])),
                                                 "fast": float(np.median(fast_cl["cl_pe"]))},
                           "cluster_duration_median_ns": {
                               "ref": float(np.median(ref_cl["cl_end"] - ref_cl["cl_start"])),
                               "fast": float(np.median(fast_cl["cl_end"] - fast_cl["cl_start"]))}}
    out["v_r4_checks"] = {"coincidences_per_event_within_1pct": abs(out["coincidences"]["ratio"] - 1) <= 0.01,
                          "cluster_pe_median_within_2pct": abs(out["coincidences"]["cluster_pe_median"]["fast"] /
                                                               out["coincidences"]["cluster_pe_median"]["ref"] - 1) <= 0.02,
                          "pulse_pe_band": out["pulse0"]["binned"]["pe"]["max_abs_dev_minus_2sigma"] is not None and
                          out["pulse0"]["binned"]["pe"]["max_abs_dev_minus_2sigma"] <= BAND_TOL["pe"]}
    out["unassigned_ref_pulses"] = {"n": int((D["cl"] < 0).sum()), "frac": float((D["cl"] < 0).mean())}
    out["us_per_event_sample"] = 1e6 * dt_sample / max(ne, 1)
    os.makedirs(args.outdir, exist_ok=True)
    model.save(os.path.join(args.outdir, "direct_model_v2.npz"))
    _report(args, "direct", out)


def cmd_noise(args):
    """Noise-pulse model: fit on even events of a data table, generate for the odd events, compare with the held-out
    data (count per event, channels, PE, time), then coincidences from (data signal pulses + generated noise) vs the
    stock coincidences of the same events."""
    from . import noise as N
    G = _coinc_geom()
    sl = dict(sector=G.sector_type[G.sector], layer=G.layer) if not args.counter_only else {}
    P = _pulses(args.inputs)
    evs_all = reftables.event_key(reftables.read(args.inputs, "events", columns=["run", "subrun", "event"]))
    evs_all = np.unique(evs_all)
    nchan = len(reftables.calib(args.inputs)["status"])
    sub = None
    if args.mc:
        Pm = _pulses([args.mc])
        nm = len(np.unique(reftables.event_key(reftables.read([args.mc], "events", columns=["run", "subrun", "event"]))))
        wi, iso = N.noise_waveforms(Pm["ev"], Pm["chan"], Pm["time"], Pm["wf"], **sl)
        fst = np.unique(wi, return_index=True)[1]
        sel = fst[iso[fst]]
        sub = np.bincount(Pm["chan"][sel], minlength=nchan) / nm
    tr = (P["ev"] % 2) == 0
    ntr, nte = int(((evs_all % 2) == 0).sum()), int(((evs_all % 2) == 1).sum())
    model = N.NoiseModel.fit({k: v[tr] for k, v in P.items()}, ntr, nchan, subtract_rate=sub, **sl)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    g = model.sample(nte, rng, N.NoiseConfig(rate_scale=args.rate_scale))
    dt = time.perf_counter() - t0
    te = {k: v[~tr] for k, v in P.items()}
    wi, noise = N.noise_waveforms(te["ev"], te["chan"], te["time"], te["wf"], **sl)
    fst = np.unique(wi, return_index=True)[1]
    nwf = fst[noise[fst]]
    cnt_ref = np.bincount(np.searchsorted(np.unique(te["ev"]), te["ev"][nwf]), minlength=nte)
    cnt_fast = np.bincount(g["ev"][g["seq"] == 0], minlength=nte)
    ch_ref = np.bincount(te["chan"][nwf], minlength=nchan)
    ch_fast = np.bincount(g["chan"][g["seq"] == 0], minlength=nchan)
    ok = ch_ref > 50
    ref_p = {"pe": te["pe"][noise], "time": te["time"][noise], "beta": te["beta"][noise]}
    fast_p = {"pe": g["pe"], "time": g["time"], "beta": g["beta"]}
    out = {"events_train": ntr, "events_test": nte, "mc_subtracted": bool(sub is not None),
           "gamma_k": model.gamma_k, "library_waveforms": int(len(model.lib)),
           "noise_waveforms_per_event": {"ref": float(cnt_ref.mean()), "fast": float(cnt_fast.mean())},
           "count_var": {"ref": float(cnt_ref.var()), "fast": float(cnt_fast.var())},
           "count_hist": {"ref": np.bincount(np.minimum(cnt_ref, 8), minlength=9).tolist(),
                          "fast": np.bincount(np.minimum(cnt_fast, 8), minlength=9).tolist()},
           "pulses_per_noise_waveform": {"ref": float(noise.sum() / max(len(nwf), 1)),
                                         "fast": float(len(g["ev"]) / max((g["seq"] == 0).sum(), 1))},
           "channel_rate_ratio_q": np.quantile(ch_fast[ok] / ch_ref[ok], [0.05, 0.5, 0.95]).tolist(),
           "pulses": compare.distribution_summary(fast_p, ref_p),
           "pulses_binned": {c: _band(fast_p[c], ref_p[c]) for c in ("pe", "beta")},
           "us_per_event_sample": 1e6 * dt / max(nte, 1)}
    # downstream: stock coincidences vs (signal pulses of the test events + generated noise)
    status = reftables.calib(args.inputs)["status"]
    ue = np.unique(te["ev"])
    test_events = evs_all[(evs_all % 2) == 1]
    ref_cl = _coinc_from(te["ev"], te["chan"], te["time"], te["pe"], te["seq"], G, status)
    sig = ~noise
    gev = test_events[g["ev"]]
    cat = lambda a, b: np.concatenate([a, b])
    fast_cl = _coinc_from(cat(te["ev"][sig], gev), cat(te["chan"][sig], g["chan"]), cat(te["time"][sig], g["time"]),
                          cat(te["pe"][sig], g["pe"]), cat(te["seq"][sig], g["seq"]), G, status)
    none_cl = _coinc_from(te["ev"][sig], te["chan"][sig], te["time"][sig], te["pe"][sig], te["seq"][sig], G, status)
    pt = lambda c: {int(t): int((c["cl_type"] == t).sum()) for t in (1, 2, 3, 4)}
    out["coincidences"] = {"stock": pt(ref_cl), "signal_plus_model_noise": pt(fast_cl), "signal_only": pt(none_cl),
                           "per_event": {"stock": len(ref_cl["cl_type"]) / nte,
                                         "signal_plus_model_noise": len(fast_cl["cl_type"]) / nte,
                                         "signal_only": len(none_cl["cl_type"]) / nte}}
    os.makedirs(args.outdir, exist_ok=True)
    model.save(os.path.join(args.outdir, "noise_model.npz"))
    _report(args, "noise", out)


def cmd_vr4(args):
    """V-R4: fast chain (fast response + fast reco) vs the stock chain from the same G4 CrvSteps, per sector group.
    inputs: CrvStep tables (crvsteps.<shard>.root); --refs: the stock CrvRecoTableDumper files of the same shards.
    --merge: inputs are --emit JSONs to sum."""
    from . import vr4
    if args.merge:
        return vr4.report(vr4.merge(args.inputs), args.outdir)
    if not (args.refs and args.calib):
        raise SystemExit("reco-vr4 needs --refs and --calib (or --merge)")
    return vr4.run(args.inputs, args.refs, args.calib, args.outdir, deviation=args.deviation, jax=args.jax,
                   max_events=args.max_events, seed=args.seed, emit=args.emit, noise=not args.no_noise,
                   batch_events=args.batch_events)


def cmd_r3(args):
    """R3: cached CrvStep tables replayed through the whole fast chain (+ objective histograms) to --target events with
    CRV steps, timed per stage. --merge: inputs are per-process --emit JSONs (node totals; --wall = job wall [s])."""
    from . import r3
    if args.merge:
        out = r3.merge(args.inputs, wall=args.wall)
        st = out.pop("response_state", None)
        plots.write_json(args.outdir, "r3_merge.json", out)
        if st is not None:
            json.dump(st, open(os.path.join(args.outdir, "r3_response_state.json"), "w"))
        print(json.dumps(out, indent=1))
        return out
    if not args.calib:
        raise SystemExit("reco-r3 needs --calib")
    return r3.run(args.inputs, args.calib, deviation=args.deviation, target=args.target, jax=args.jax, seed=args.seed,
                  batch_events=args.batch_events, emit=args.emit, objective=args.objective, noise=not args.no_noise,
                  time_limit=args.time_limit, v05=args.v05,
                  knobs=dict(kv.split("=", 1) for kv in (args.knob or [])), background=args.background, time_shift=args.time_shift)


def cmd_background(args):
    """Out-of-time background model (reco.background). Three modes:
      --emit PART.npz          one partial per input file list (data: library kept with --keep-frac; --mc: rate only)
      --merge --out MODEL.npz  inputs are data partials, --mc partials subtracted; model saved mode 600
      --out MODEL.npz          scan + merge in one process (small tests only)
    The model and partials hold data-derived content (Mu2e-internal): keep them with the conditions, mode 700 dirs."""
    from . import background as BGM
    from crvfastsim.objective import bars
    inputs, mc = _expand_inputs(args.inputs), _expand_inputs(args.mc or [])
    if args.emit:
        frac = 0.0 if args.rate_only else args.keep_frac
        parts = [BGM.scan_file(p, frac, args.seed + i, args.max_events) for i, p in enumerate(inputs)]
        part = parts[0]
        for q in parts[1:]:                      # several files in one partial: sum the counts, offset the library
            off = int(part["lib"]["group"].max()) + 1 if len(part["lib"]["group"]) else 0
            for k in ("events", "exposure_ns", "groups", "cnt_sum", "cnt_sumsq"):
                part[k] += q[k]
            part["windows"] = sorted(set(part["windows"]) | set(q["windows"]))
            part["lib"] = {k: np.r_[part["lib"][k], (q["lib"][k] + off) if k == "group" else q["lib"][k]] for k in BGM.LIB_KEYS}
            part["file"] = part["file"] + "," + q["file"]
        BGM.save_partial(args.emit, part)
        print(json.dumps({k: v for k, v in part.items() if k != "lib"}, default=float))
        return part
    if args.merge:
        m = BGM.BackgroundModel.from_partials([BGM.load_partial(p) for p in inputs], [BGM.load_partial(p) for p in mc])
    else:
        m = BGM.BackgroundModel.fit(inputs, mc, entry_stop=args.max_events, keep_frac=args.keep_frac, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    m.save(args.out)
    os.chmod(args.out, 0o600)
    slot, _, _ = bars.bar_slot_body_layer(m.bar)
    one = [len(np.unique(m.bar[a:a + c])) == 1 for a, c in zip(m.start[:20000], m.count[:20000])]
    out = dict(m.info, net_rate_per_event_per_us=1e3 * m.rate_per_ns, gamma_k=m.gamma_k,
               pulses_by_slot={n: int((slot == i).sum()) for i, n in enumerate(bars.SLOT_NAME)},
               pe_median=float(np.median(m.pe)) if len(m.pe) else None, groups_one_bar=float(np.mean(one)) if one else None)
    print(json.dumps({k: v for k, v in out.items() if k not in ("data_files", "mc_files")}, indent=1, default=float))
    if args.outdir:
        plots.write_json(args.outdir, "background_model.json", out)
    return out


def _expand_inputs(paths):
    import glob
    out = []
    for p in paths:
        if p.endswith(".txt") and os.path.isfile(p):
            out += [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
        else:
            out += sorted(glob.glob(p)) or [p]
    return out


def cmd_vr4eff(args):
    """V-R4 EX layer efficiency on crvonly_v05 events. --prep NTUPLE OUT.npz (GPVM, light: tracks + stock pulses);
    inputs = CrvStep table, --prep-npz, --calib, --emit (grid: fast chain); --merge JSONs."""
    from . import vr4eff
    if args.prep:
        return vr4eff.prep(args.inputs[0], args.prep)
    if args.merge:
        out = vr4eff.merge(args.inputs)
        plots.write_json(args.outdir, "vr4eff_stats.json", out)
        for L, r in out["layers"].items():
            print(f"EX layer {L}: den {r['den']}  eff stock {100*r['eff_stock']:.2f} fast {100*r['eff_fast']:.2f}  "
                  f"diff {100*r['diff']:+.3f}+-{100*r['diff_err']:.3f} pts | {'ok' if r['pass'] else 'FAIL'}")
        print(f"V-R4 EX layer efficiency {'PASS' if out['pass'] else 'FAIL'} over {out['files']} files")
        return out
    return vr4eff.run(args.inputs[0], args.prep_npz, args.calib, args.emit, deviation=args.deviation, jax=args.jax,
                      seed=args.seed, batch_events=args.batch_events, max_events=args.max_events)


def cmd_repro(args):
    """R5: seeded synthetic runs of the jitted stages. Same seed twice in one process (and against --against, a saved
    run of the same backend from another process) must be bit-identical; --against from another backend is compared
    statistically. Writes repro_<backend>.npz for later comparisons."""
    import jax
    from . import repro
    G = _coinc_geom()
    ne = args.max_events or 5000
    a = repro.run(ne, G, seed=args.seed)
    b = repro.run(ne, G, seed=args.seed)
    c = repro.run(ne, G, seed=args.seed + 1)
    fa, fb, fc = repro.fingerprint(a), repro.fingerprint(b), repro.fingerprint(c)
    be = jax.default_backend()
    out = {"backend": be, "device": str(jax.devices()[0]), "events": ne, "seed": args.seed, "fingerprint": fa,
           "same_seed_identical": {s: fa[s] == fb[s] for s in repro.STAGES},
           "other_seed_differs": {s: fa[s] != fc[s] for s in repro.STAGES}}
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, f"repro_{be}.npz"), **repro.flatten(a))
    if args.against:
        z = np.load(args.against)
        other = repro.unflatten(z)
        fo = repro.fingerprint(other)
        out["against"] = {"file": args.against, "fingerprint": fo, "identical": {s: fo[s] == fa[s] for s in repro.STAGES},
                          "comparison": repro.compare(a, other)}
    out["pass_bit_identical"] = all(out["same_seed_identical"].values()) and all(out["other_seed_differs"].values())
    _report(args, "repro", out)


def cmd_direct_v1(args):
    """Fit on even events, test on odd: pulse count per cluster, first-pulse PE/time/beta vs stock pulses."""
    cfg = waveform.WaveformConfig()
    Q = _charges(args.inputs)
    events = np.unique(Q["ev_row"])
    if args.max_events:
        events = events[:args.max_events]
    P = _pulses(args.inputs, events)
    rows_ev = Q["ev_row"][Q["row"]]
    keep = np.isin(rows_ev, events)
    lab = Q["row"][keep]
    feat = cluster_features(lab, Q["time"][keep], Q["pe"][keep], cfg)
    # pulse -> charges row (same event and channel) -> cluster
    rk = (Q["ev_row"].astype(np.int64) << 16) | Q["chan_row"]
    pk = (P["ev"].astype(np.int64) << 16) | P["chan"]
    srt = np.argsort(rk)
    pos = np.clip(np.searchsorted(rk[srt], pk), 0, len(srt) - 1)
    prow = np.where(rk[srt][pos] == pk, srt[pos], -1)
    cl = np.full(len(pk), -1, np.int64)
    okp = prow >= 0
    C_ = reftables.calib(args.inputs)
    toff = C_["timeoffset"][P["chan"]]
    t_nooff = P["time"] - toff
    cl[okp] = assign_pulses(feat, prow[okp], t_nooff[okp], cfg)
    Cn = len(feat["q"])
    npul = np.bincount(cl[cl >= 0], minlength=Cn)
    # first pulse per cluster = lowest seq among assigned
    firstp = np.full(Cn, -1, np.int64)
    a = np.flatnonzero(cl >= 0)
    o = a[np.lexsort((P["seq"][a], cl[a]))]
    u, fi = np.unique(cl[o], return_index=True)
    firstp[u] = o[fi]
    hp = firstp >= 0
    first = {"pe": np.where(hp, P["pe"][np.maximum(firstp, 0)], np.nan),
             "dt": np.where(hp, t_nooff[np.maximum(firstp, 0)] - feat["t10"], np.nan),
             "beta": np.where(hp, P["beta"][np.maximum(firstp, 0)], np.nan),
             "failed": np.where(hp, P["failed"][np.maximum(firstp, 0)], False)}
    cev = Q["ev_row"][feat["chan"]]
    train = (cev % 2) == 0
    tr = {k: (v[train] if isinstance(v, np.ndarray) and len(v) == Cn else v) for k, v in feat.items()}
    model = DirectModel.fit(tr, npul[train], {k: v[train] for k, v in first.items()})
    te = {k: (v[~train] if isinstance(v, np.ndarray) and len(v) == Cn else v) for k, v in feat.items()}
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    s = model.sample(te, rng)
    dt = time.perf_counter() - t0
    ntest = int(len(np.unique(cev[~train])))
    ref = {"pe": first["pe"][~train], "time": first["dt"][~train] + feat["t10"][~train], "beta": first["beta"][~train]}
    fast = {"pe": s["pe"], "time": s["time"], "beta": s["beta"]}
    n_ref, n_fast = npul[~train], s["n"]
    out = {"clusters_test": int((~train).sum()), "events_test": ntest,
           "pulses_per_cluster_mean": {"ref": float(n_ref.mean()), "fast": float(n_fast.mean())},
           "found_frac": {"ref": float((n_ref > 0).mean()), "fast": float((n_fast > 0).mean())},
           "found_frac_vs_q": _eff_vs_q(te["q"], n_ref > 0, n_fast > 0),
           "first_pulse": compare.distribution_summary(fast, ref),
           "first_pulse_binned": {c: _band(fast[c], ref[c]) for c in ("pe", "beta")},
           "failed_frac": {"ref": float(first["failed"][~train][n_ref > 0].mean()) if (n_ref > 0).any() else None,
                           "fast": float(s["failed"][n_fast > 0].mean()) if (n_fast > 0).any() else None},
           "unassigned_ref_pulses": int((cl < 0).sum()), "wall_s_sample": dt,
           "us_per_event": 1e6 * dt / max(ntest, 1)}
    out["first_pulse_binned_pass"] = {c: bool(out["first_pulse_binned"][c]["max_abs_dev_minus_2sigma"] is not None and
                                              out["first_pulse_binned"][c]["max_abs_dev_minus_2sigma"] <= BAND_TOL[c])
                                      for c in BAND_TOL}
    os.makedirs(args.outdir, exist_ok=True)
    model.save(os.path.join(args.outdir, "direct_model.npz"))
    _report(args, "direct", out)


BAND_TOL = {"pe": 0.03, "beta": 0.03}   # binned-ratio tolerance (V-R4's +-3 %), 2 sigma allowed


def _band(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.ratio_band(a[np.isfinite(a)], b[np.isfinite(b)], q=(0.02, 0.98), nbins=40, min_ref=400)
    out = {k: r[k] for k in ("max_abs_dev", "max_abs_dev_minus_2sigma", "nbins_used")}
    return out


def _eff_vs_q(q, a, b, edges=(0, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100, 1e9)):
    e = np.asarray(edges, float)
    i = np.clip(np.searchsorted(e, q, side="right") - 1, 0, len(e) - 2)
    n = np.bincount(i, minlength=len(e) - 1)
    return [{"q_lo": float(e[k]), "q_hi": float(e[k + 1]), "n": int(n[k]),
             "ref": float(np.bincount(i, weights=a, minlength=len(e) - 1)[k] / n[k]) if n[k] else None,
             "fast": float(np.bincount(i, weights=b, minlength=len(e) - 1)[k] / n[k]) if n[k] else None}
            for k in range(len(e) - 1)]


def cmd_cost(args):
    """Both fast paths on the same events, CPU wall per event (synthesis + reco vs direct sampling)."""
    cfg_w, cfg_r = waveform.WaveformConfig(), pulsefit.RecoConfig()
    Q = _charges(args.inputs)
    events = np.unique(Q["ev_row"])[:args.max_events] if args.max_events else np.unique(Q["ev_row"])
    keep = np.isin(Q["ev_row"][Q["row"]], events)
    lab, tq, qq, pe = Q["row"][keep], Q["time"][keep], Q["charge"][keep], Q["pe"][keep]
    rng = np.random.default_rng(args.seed)
    tdc0 = rng.random(len(Q["chan_row"])) * pulsefit.PERIOD
    t0 = time.perf_counter()
    dg = waveform.synthesize(lab, tq, qq, tdc0, cfg_w, rng=rng)
    t1 = time.perf_counter()
    lens = np.full(len(dg["chan"]), cfg_w.samples_zs)
    ev_d = Q["ev_row"][dg["chan"]]
    gid, mat, n, start, first = pulsefit.concat_waveforms(ev_d, Q["chan_row"][dg["chan"]], np.zeros_like(ev_d),
                                                          dg["starttdc"], dg["adc"].ravel(), lens)
    W = len(n)
    res = pulsefit.reconstruct(mat, n, start, np.full(W, float(cfg_w.pedestal)), np.full(W, 1.0), np.full(W, 1.0),
                               cfg_r)
    t2 = time.perf_counter()
    feat = cluster_features(lab, tq, pe, cfg_w)
    t3 = time.perf_counter()
    model = DirectModel.load(args.model) if args.model else None
    if model is not None:
        model.sample(feat, rng)
    t4 = time.perf_counter()
    ne = max(len(events), 1)
    out = {"events": int(len(events)), "charges": int(len(lab)), "digis": int(len(dg["chan"])),
           "pulses_waveform_path": int(len(res["wf"])),
           "us_per_event": {"waveform_synthesis": 1e6 * (t1 - t0) / ne, "pulse_reco": 1e6 * (t2 - t1) / ne,
                            "waveform_path_total": 1e6 * (t2 - t0) / ne, "direct_features": 1e6 * (t3 - t2) / ne,
                            "direct_sample": 1e6 * (t4 - t3) / ne if model is not None else None,
                            "direct_path_total": 1e6 * (t4 - t2) / ne if model is not None else None},
           "stock_reference_ms_per_event": {"CrvRecoPulsesFinder_A6": 1.56, "CrvRecoPulsesFinder_A8": 3.73,
                                            "CRV_response_total": 19.8, "note": "stock TimeTracker means (docs/fastreco.md)"}}
    _report(args, "cost", out)


def _report(args, name, out):
    plots.write_json(args.outdir, f"{name}_stats.json", out)
    print(json.dumps(plots._clean(out), indent=1)[:4000])


RECO_CMDS = ("vr2", "vr2stock", "vr2jax", "bench", "vr1", "vr3", "vr4", "vr4eff", "r3", "direct", "noise", "repro", "cost", "inspect",
             "background")


def cmd_vr3(args):
    """V-R3: stock pulses -> ported coincidence finder vs the stock clusters, per event and sector type."""
    from ..geometry import find_counters, from_offline_config, parse_simpleconfig
    from . import coinc as C
    cf = find_counters("Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt")
    G = C.sector_geom(from_offline_config(cf), parse_simpleconfig(open(cf).read()))
    status = reftables.calib(args.inputs)["status"]
    p = reftables.read(args.inputs, "pulses")
    pev = reftables.event_key(p)
    c = reftables.read(args.inputs, "clusters")
    cev = reftables.event_key(c)
    evs = np.unique(pev)[:args.max_events] if args.max_events else np.unique(pev)
    cols = {k: ak.to_numpy(p[k]) for k in ("idx", "bar", "sipm", "time", "pe", "seq")}
    order = np.argsort(pev, kind="stable")
    lo, hi = np.searchsorted(pev[order], evs), np.searchsorted(pev[order], evs, side="right")
    corder = np.argsort(cev, kind="stable")
    clo, chi = np.searchsorted(cev[corder], evs), np.searchsorted(cev[corder], evs, side="right")
    cpul, cst = c.pulses, ak.to_numpy(c.sectortype)
    if getattr(args, "jax", False):
        return _vr3_jax(args, G, status, c, cev, evs)
    cnum = {k: ak.to_numpy(c[k]) for k in ("pe", "start", "end", "slope", "avgtime")}
    ident = nref = nfast = 0
    dprop = {k: [] for k in cnum}
    bad = []
    t0 = time.perf_counter()
    for k, e in enumerate(evs):
        sel = order[lo[k]:hi[k]]
        fast = C.find_event({kk: v[sel] for kk, v in cols.items()}, G, status)
        rows = corder[clo[k]:chi[k]]
        ref = {(int(cst[i]), tuple(sorted(ak.to_list(cpul[i])))): i for i in rows}
        fs = {(r["sectortype"], tuple(sorted(r["pulses"]))): r for r in fast}
        nref += len(ref)
        nfast += len(fs)
        if set(ref) == set(fs):
            ident += 1
            for key, i in ref.items():
                r = fs[key]
                for q, src in (("pe", "pe"), ("start", "start"), ("end", "end"), ("slope", "slope"),
                               ("avgtime", "avgtime")):
                    dprop[q].append(abs(r[src] - cnum[q][i]))
        elif len(bad) < 10:
            bad.append({"event_key": int(e), "ref": [list(x) for x in ref], "fast": [list(x) for x in fs]})
    dt = time.perf_counter() - t0
    out = {"events": int(len(evs)), "events_identical": int(ident), "clusters_ref": int(nref), "clusters_fast": int(nfast),
           "identical_cluster_lists": bool(ident == len(evs)),
           "max_abs_property_diff": {q: (float(np.max(v)) if v else None) for q, v in dprop.items()},
           "examples": bad, "wall_s": dt, "us_per_event": 1e6 * dt / max(len(evs), 1)}
    out["pass"] = out["identical_cluster_lists"]
    _report(args, "vr3", out)


def _vr3_jax(args, G, status, c, cev, evs):
    from . import coinc_jax as CJ
    P = _ref_pulses_sorted(args.inputs)
    m = np.isin(P["ev"], evs)
    cols = [P[k][m] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")]
    CJ.find(*cols, G, status)   # compile
    tm = {}
    t0 = time.perf_counter()
    out = CJ.find(*cols, G, status, timers=tm)
    dt = time.perf_counter() - t0
    cm = np.isin(cev, evs)
    keys = [(int(e), int(t), tuple(sorted(ak.to_list(x)))) for e, t, x in
            zip(cev[cm], ak.to_numpy(c.sectortype)[cm], c.pulses[cm])]
    refi = {k: i for k, i in zip(keys, np.flatnonzero(cm))}
    fk = CJ.cluster_keys(out)
    ref = set(refi)
    bad_ev = {k[0] for k in ref ^ fk}
    # cluster properties of the matched clusters
    o = np.lexsort((out["mem_idx"], out["mem_cl"]))
    cl, mi = out["mem_cl"][o].astype(np.int64), out["mem_idx"][o].astype(np.int64)
    b = np.flatnonzero(np.diff(cl)) + 1
    dprop = {q: 0.0 for q in ("pe", "start", "end", "slope", "avgtime")}
    for i, g in zip(cl[np.r_[0, b]] if len(cl) else [], np.split(mi, b)):
        j = refi.get((int(out["cl_event"][i]), int(out["cl_type"][i]), tuple(g.tolist())))
        if j is not None:
            for q in dprop:
                dprop[q] = max(dprop[q], float(abs(out["cl_" + q][i] - c[q][j])))
    res = {"finder": "coinc_jax", "events": int(len(evs)), "events_identical": int(len(evs) - len(bad_ev)),
           "clusters_ref": len(ref), "clusters_fast": len(fk), "identical_cluster_lists": not bad_ev,
           "max_abs_property_diff": dprop, "fallback_events": int(len(out["fallback_events"])), "timers": tm,
           "wall_s": dt, "us_per_event": 1e6 * dt / max(len(evs), 1), "examples": sorted(bad_ev)[:10]}
    res["pass"] = res["identical_cluster_lists"]
    _report(args, "vr3", res)


def add_reco_args(sp, name):
    """Arguments of one reco subcommand (shared by `crvfastsim reco-<name>` and `crvfastsim.reco <name>`)."""
    sp.add_argument("inputs", nargs="+", help="CrvRecoTableDumper files (globs allowed, quoted); bench also "
                    "accepts the word 'synthetic' (KPP-like waveforms, reco/synth.py)")
    sp.add_argument("--outdir", default=None)
    sp.add_argument("--max-events", type=int, default=None)
    sp.add_argument("--seed", type=int, default=20261016)
    if name in ("vr2jax", "bench"):
        sp.add_argument("--length", type=int, default=36, help="padded samples per waveform (longer -> numpy)")
        sp.add_argument("--batch", type=int, default=65536, help="waveforms per jitted call")
        sp.add_argument("--rounds", type=int, default=None,
                        help="pulse rounds in the jitted kernel (default RecoConfig.max_pulses = 8); waveforms with a "
                        "pulse in the last round are redone by the numpy path")
    if name == "bench":
        sp.add_argument("--repeat", type=int, default=3)
        sp.add_argument("--chunks", type=int, default=8, help="chain stage: event chunks in the pipeline")
        sp.add_argument("--stage", default="pulses", choices=["pulses", "waveforms", "coinc", "chain"],
                        help="pulses: jitted pulse reco; waveforms: jitted waveform synthesis (synthetic charges); "
                        "coinc: JAX coincidence finder (synthetic tracks+noise, or reference pulses)")
    if name == "vr1":
        sp.add_argument("--window-end", type=float, default=1e5, help="digitisation window end [ns] (offspill 1e5)")
        sp.add_argument("--jax", action="store_true", help="use the JAX waveform synthesis (waveform_jax)")
        sp.add_argument("--batch-clusters", type=int, default=2000, help="SiPM rows per jitted call (--jax)")
        sp.add_argument("--noise", action="store_true", help="fast arm with noise; reference must be the noise-on "
                        "table (crvreco_mc); statistical per-channel comparison instead of sample identity")
        sp.add_argument("--nonoise-ref", default=None, help="--noise: the noise-free reference of the same events "
                        "(crvreco_mcnonoise) for the residual test")
        sp.add_argument("--min-digis", type=int, default=200, help="--noise: channels with fewer digis are not tested")
    if name == "vr3":
        sp.add_argument("--jax", action="store_true", help="use the JAX coincidence finder (coinc_jax)")
    if name == "inspect":
        sp.add_argument("--keys", required=True, help="a vr2/vr2stock stats JSON (its unmatched_*_keys)")
    if name == "direct":
        sp.add_argument("--v1", action="store_true", help="the v1 model (first pulse only, independent residuals)")
        sp.add_argument("--feb-phase-spread", type=float, default=12.5,
                        help="FEB clock phase (TDC0) drawn uniform in [offset, offset+spread] ns per (event, FEB); "
                        "stock MC 12.5 (one period); a tuning knob for data")
        sp.add_argument("--feb-phase-offset", type=float, default=0.0)
    if name == "vr4":
        sp.add_argument("--refs", nargs="+", default=None, help="stock CrvRecoTableDumper files, same shards and order")
        sp.add_argument("--calib", default=None, help="CRVCalib CSV of the reference job (crvdump/calib dump)")
        sp.add_argument("--deviation", default=None, help="CRVPhoton yield deviations CSV")
        sp.add_argument("--jax", action="store_true", help="JAX twins for the response, waveforms and reco")
        sp.add_argument("--no-noise", action="store_true", help="no electronics noise in the fast waveforms")
        sp.add_argument("--emit", default=None, help="write the accumulators to this JSON (batchable)")
        sp.add_argument("--merge", action="store_true", help="inputs are --emit JSONs: sum and report")
        sp.add_argument("--batch-events", type=int, default=None, help="events per fast-chain batch (JAX default 250; "
                        "smaller bounds memory)")
    if name == "r3":
        sp.add_argument("--calib", default=None, help="CRVCalib CSV (channel map, status, constants)")
        sp.add_argument("--deviation", default=None, help="CRVPhoton yield deviations CSV")
        sp.add_argument("--target", type=int, default=1_000_000, help="events with CRV steps to process (replays the cache)")
        sp.add_argument("--jax", action="store_true", help="JAX: fused device response, JAX waveforms and reco")
        sp.add_argument("--no-noise", action="store_true")
        sp.add_argument("--objective", action="store_true", help="fill the crvfastsim.objective response histograms")
        sp.add_argument("--batch-events", type=int, default=2000)
        sp.add_argument("--time-limit", type=float, default=None, help="stop after this many seconds (wall)")
        sp.add_argument("--v05", action="store_true", help="crvonly_v05 settings: one FEB phase per event, CRVTime net zero")
        sp.add_argument("--background", default=None, help="out-of-time background model npz (reco-background fit)")
        sp.add_argument("--time-shift", default=None, help="timecal JSON (objective.timecal): its mc_channel_shift [ns] "
                        "is added to the MC pulse times, bringing the MC to the data period's module/FEB timing")
        sp.add_argument("--knob", nargs="*", default=None, help="response knobs NAME=VALUE (chain.KNOBS), e.g. "
                        "photon_yield_scale=1.1 cross_talk=0.06")
        sp.add_argument("--emit", default=None, help="per-process JSON")
        sp.add_argument("--merge", action="store_true", help="inputs are --emit JSONs of one node job")
        sp.add_argument("--wall", type=float, default=None, help="--merge: the job wall [s] (default: slowest process)")
    if name == "vr4eff":
        sp.add_argument("--prep", default=None, help="with an EventNtuple input: write tracks + stock pulses here")
        sp.add_argument("--prep-npz", default=None)
        sp.add_argument("--calib", default=None)
        sp.add_argument("--deviation", default=None)
        sp.add_argument("--jax", action="store_true")
        sp.add_argument("--emit", default=None)
        sp.add_argument("--merge", action="store_true")
        sp.add_argument("--batch-events", type=int, default=None)
    if name == "repro":
        sp.add_argument("--against", default=None, help="repro_<backend>.npz from another run (process or backend)")
    if name == "background":
        sp.add_argument("--mc", nargs="*", default=None, help="MC EventNtuples (or a .txt list) whose own out-of-time rate is subtracted")
        sp.add_argument("--out", default=None, help="model npz (with --merge, or a one-process fit)")
        sp.add_argument("--emit", default=None, help="write one partial (npz) for the inputs")
        sp.add_argument("--merge", action="store_true", help="inputs (and --mc) are partials")
        sp.add_argument("--rate-only", action="store_true", help="--emit: no library (MC partials)")
        sp.add_argument("--keep-frac", type=float, default=0.003, help="fraction of data groups kept in the library")
    if name == "noise":
        sp.add_argument("--mc", default=None, help="MC reference table: its isolated-waveform rate is subtracted")
        sp.add_argument("--rate-scale", type=float, default=1.0)
        sp.add_argument("--counter-only", action="store_true",
                        help="noise = no partner on the same counter only (default also requires none in another "
                        "layer of the sector)")
    if name == "cost":
        sp.add_argument("--model", default=None, help="direct_model.npz from the direct subcommand")


def run_reco(name, a):
    a.outdir = a.outdir or f"plots/reco_{name}"
    {"vr2": cmd_vr2, "vr2stock": lambda x: cmd_vr2(x, oracle=True), "vr1": cmd_vr1, "vr3": cmd_vr3, "direct": cmd_direct, "noise": cmd_noise, "background": cmd_background, "repro": cmd_repro, "vr4": cmd_vr4, "vr4eff": cmd_vr4eff, "r3": cmd_r3,
     "cost": cmd_cost, "inspect": cmd_inspect, "vr2jax": lambda x: cmd_vr2(x, use_jax=True),
     "bench": cmd_bench}[name](a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="crvfastsim.reco", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in RECO_CMDS:
        add_reco_args(sub.add_parser(name), name)
    a = p.parse_args(argv)
    run_reco(a.cmd, a)
