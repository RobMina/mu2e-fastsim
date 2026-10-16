"""CLI of crvfastsim.reco:  python3 -m crvfastsim.reco <subcommand> REF.root [...] [--outdir DIR]

  vr2       V-R2: stock digis -> fast pulse reco (numpy) vs the stock pulses in the same table
  vr2stock  oracle check: Offline's MakeCrvRecoPulses through PyROOT on the dumped digis vs the dumped pulses
            (proves the table and the pulse grouping reproduce the art job; needs a built muse area)
  vr1       V-R1: dumped SiPM charges -> fast waveforms + digis vs the dumped digis (noise off: compare digi MC
            voltages too)
  direct    direct path: fit the charge -> pulse model on half of the events, sample the other half, compare with
            the stock pulses and with the waveform path (fidelity and cost per event)
  cost      wall time per event of both paths on the same events (CPU, this host)

A separate CLI until the package's shared cli.py is free to edit (other work in progress there).
"""
import argparse
import json
import os
import sys
import time

import awkward as ak
import numpy as np

from .. import plots
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
    return {"ev": ev, "chan": chan, "seq": ak.to_numpy(p.seq), "pe": ak.to_numpy(p.pe), "time": ak.to_numpy(p.time),
            "beta": ak.to_numpy(p.beta), "failed": (ak.to_numpy(p.flags) & 1).astype(bool),
            "key": compare.pulse_key(ev, chan, ak.to_numpy(p.seq))}


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
    return {"ev": sel["ev"][first][ok], "chan": chan[ok], "mat": mat[ok], "n": n[ok], "start": start[ok],
            "ped": C["pedestal"][chan[ok]], "area": C["pulsearea"][chan[ok]],
            "height": C["pulseheight"][chan[ok]], "toff": C["timeoffset"][chan[ok]], "events": np.unique(D["ev"])}


def _with_offsets(res, W, cfg):
    t_off = W["toff"][res["wf"]] * cfg.time_offset_scale if cfg.use_time_offset_db else 0.0
    return res["time"] + t_off   # offspill (extracted, KPP): TDC0time = 0 in CrvRecoPulsesFinder


def cmd_vr2(args, oracle=False):
    cfg = pulsefit.RecoConfig()
    W = _prepare(args.inputs, args.max_events, cfg)
    t0 = time.perf_counter()
    if oracle:
        from . import stockfit
        ped_used = np.where(W["ped"] - W["mat"][:, 0] > cfg.pedestal_undershoot_threshold, W["mat"][:, 0], W["ped"])
        res = stockfit.reconstruct(W["mat"], W["n"], W["start"], ped_used, W["area"], W["height"], cfg)
    else:
        res = pulsefit.reconstruct(W["mat"], W["n"], W["start"], W["ped"], W["area"], W["height"], cfg)
    dt = time.perf_counter() - t0
    fast = {"key": compare.pulse_key(W["ev"][res["wf"]], W["chan"][res["wf"]], res["seq"]), "pe": res["pe"],
            "time": _with_offsets(res, W, cfg), "failed": res["failed"]}
    ref = _pulses(args.inputs, W["events"])
    out = compare.vr2(fast, ref)
    out.update({"events": int(len(W["events"])), "waveforms": int(len(W["n"])), "wall_s": dt,
                "us_per_event": 1e6 * dt / max(len(W["events"]), 1), "mode": "stock-oracle" if oracle else "fast"})
    _report(args, "vr2stock" if oracle else "vr2", out)


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
    # one label per (event, channel): row index of the charges table
    lab_key = (Q["ev_row"].astype(np.int64) << 16) | Q["chan_row"]
    tdc0_by = {}
    mkey = (mev.astype(np.int64) << 16) | mchan
    _, fi = np.unique(mkey, return_index=True)
    tdc0_map_keys, tdc0_map_vals = mkey[fi], ak.to_numpy(m.tdc0)[fi]
    srt = np.argsort(tdc0_map_keys)
    pos = np.searchsorted(tdc0_map_keys[srt], lab_key)
    found = (pos < len(srt)) & (tdc0_map_keys[srt][np.clip(pos, 0, len(srt) - 1)] == lab_key)
    tdc0_row = np.where(found, tdc0_map_vals[srt][np.clip(pos, 0, len(srt) - 1)], 0.0)
    t0 = time.perf_counter()
    out = waveform.synthesize(Q["row"], Q["time"], Q["charge"], tdc0_row, cfg, rng=None)
    dt = time.perf_counter() - t0
    fkey = (Q["ev_row"][out["chan"]].astype(np.int64) << 24) ^ (Q["chan_row"][out["chan"]].astype(np.int64) << 12) ^ \
        out["starttdc"].astype(np.int64)
    rv = ak.to_numpy(ak.fill_none(ak.pad_none(m.volt, cfg.samples_zs, clip=True), 0.0))
    radc = np.clip((rv * cfg.adc_conversion + cfg.pedestal + 0.5).astype(np.int16), cfg.min_adc, cfg.max_adc)
    rstart = np.rint((ak.to_numpy(m.start) - ak.to_numpy(m.tdc0)) / pulsefit.PERIOD).astype(np.int64)
    rkey = (mev << 24) ^ (mchan.astype(np.int64) << 12) ^ rstart
    res = compare.vr1({"key": fkey, "adc": out["adc"]}, {"key": rkey, "adc": radc})
    ia, ib = compare._join(fkey, rkey)
    dv = np.abs(out["volt"][ia] - rv[ib]) if len(ia) else np.zeros((0, 1))
    res.update({"note": "noise off in the fast path; the reference digi MC voltages include stock noise (0.4 mV)",
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
           "failed_frac": {"ref": float(first["failed"][~train][n_ref > 0].mean()) if (n_ref > 0).any() else None,
                           "fast": float(s["failed"][n_fast > 0].mean()) if (n_fast > 0).any() else None},
           "unassigned_ref_pulses": int((cl < 0).sum()), "wall_s_sample": dt,
           "us_per_event": 1e6 * dt / max(ntest, 1)}
    os.makedirs(args.outdir, exist_ok=True)
    model.save(os.path.join(args.outdir, "direct_model.npz"))
    _report(args, "direct", out)


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
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"{name}_stats.json")
    plots.write_json(path, out)
    print(json.dumps(plots._clean(out), indent=1)[:4000])


def main(argv=None):
    p = argparse.ArgumentParser(prog="crvfastsim.reco", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("vr2", "vr2stock", "vr1", "direct", "cost"):
        sp = sub.add_parser(name)
        sp.add_argument("inputs", nargs="+", help="CrvRecoTableDumper files (globs allowed, quoted)")
        sp.add_argument("--outdir", default=None)
        sp.add_argument("--max-events", type=int, default=None)
        sp.add_argument("--seed", type=int, default=20261016)
        if name == "cost":
            sp.add_argument("--model", default=None, help="direct_model.npz from the direct subcommand")
    a = p.parse_args(argv)
    a.outdir = a.outdir or f"plots/reco_{a.cmd}"
    {"vr2": cmd_vr2, "vr2stock": lambda x: cmd_vr2(x, oracle=True), "vr1": cmd_vr1, "direct": cmd_direct,
     "cost": cmd_cost}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
