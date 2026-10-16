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
from .direct import DirectModel, assign_pulses, cluster_features

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
            "key": compare.pulse_key(ev, wf0, ak.to_numpy(p.seq))}


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
        sub = pulsefit.reconstruct(W["mat"][rows], W["n"][rows], W["start"][rows], W["ped"][rows], W["area"][rows],
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


def cmd_bench(args):
    """Throughput of the jitted pulse reco on the reference waveforms (compile excluded; repeated passes)."""
    import jax
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


def cmd_vr1(args):
    """Noise-free V-R1 against the dumped digi MC voltages (rng None), with TDC0 taken from the reference digi MC."""
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
    t0 = time.perf_counter()
    out = waveform.synthesize(Q["row"], Q["time"], Q["charge"], tdc0_row, cfg, rng=None)
    dt = time.perf_counter() - t0
    fkey = np.stack([Q["ev_row"][out["chan"]], Q["chan_row"][out["chan"]], out["starttdc"]], 1).astype(np.int64)
    rv = ak.to_numpy(ak.fill_none(ak.pad_none(m.volt, cfg.samples_zs, clip=True), 0.0))
    radc = np.clip((rv * cfg.adc_conversion + cfg.pedestal + 0.5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    rstart = np.rint((ak.to_numpy(m.start) - ak.to_numpy(m.tdc0)) / pulsefit.PERIOD).astype(np.int64)
    rkey = np.stack([mev, mchan, rstart], 1).astype(np.int64)
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


def cmd_direct(args):
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


RECO_CMDS = ("vr2", "vr2stock", "vr2jax", "bench", "vr1", "direct", "cost", "inspect")


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
    if name == "vr1":
        sp.add_argument("--window-end", type=float, default=1e5, help="digitisation window end [ns] (offspill 1e5)")
    if name == "inspect":
        sp.add_argument("--keys", required=True, help="a vr2/vr2stock stats JSON (its unmatched_*_keys)")
    if name == "cost":
        sp.add_argument("--model", default=None, help="direct_model.npz from the direct subcommand")


def run_reco(name, a):
    a.outdir = a.outdir or f"plots/reco_{name}"
    {"vr2": cmd_vr2, "vr2stock": lambda x: cmd_vr2(x, oracle=True), "vr1": cmd_vr1, "direct": cmd_direct,
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
