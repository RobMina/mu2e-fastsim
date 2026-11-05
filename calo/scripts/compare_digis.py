#!/usr/bin/env python3
"""Fast (JAX/NumPy) vs stock CaloDigis on the same saved CaloShowerSteps.

Noise off (arm ``det``): digis matched by (event, SiPM, t0) must agree sample by sample. Noise on (arm ``noi``,
same deterministic readouts): the per-event digi count difference and the length, peak, t0 and summed-ADC shapes
(chi2/ndf), optionally against a second stock seed as the stock-vs-stock reference.
``--steps`` is the stock art file itself or a table export (see ``calofastsim.stockio.load_steps``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import awkward as ak
import numpy as np

from calofastsim import stockio
from calofastsim.digi import DigiConfig, digitize, noise_pedestal
from calofastsim.readout import ReadoutConfig, make_readouts


def fast_digis(steps, timing, sim, pulse, *, noise=None, rng=None, ro_cfg=None, table=None, backend="numpy"):
    ro_cfg = ro_cfg or ReadoutConfig(pe_stat=False, travel=False)
    ev, sipm, t, count = make_readouts(steps["event"], steps["crystal"], steps["energy_g4"], steps["energy_birks"],
                                       steps["time"], steps["z_local"], steps["lru"], steps["yields"], timing, ro_cfg,
                                       table=table, rng=rng)
    key = ev * 10000 + sipm
    ukey, row = np.unique(key, return_inverse=True)
    noise_arg = None
    if noise is not None:
        def offsets(length):
            return np.trunc(rng.random(len(length)) * (noise.size - np.asarray(length))).astype(np.int64)
        noise_arg = (noise, noise_pedestal(noise), offsets)
    d = digitize(row, t, timing["pbtime"][ev], stockio.adc_per_pe(sim, sipm), ukey.size, pulse, DigiConfig(),
                 weight=count, noise=noise_arg, marker=timing["marker"][ev], backend=backend)
    d["event"], d["sipm"] = ukey[d["row"]] // 10000, ukey[d["row"]] % 10000
    return d


def stock_digis(ntuple_path, arm, events):
    pos, a = stockio.load_ntuple(ntuple_path, arm, events)
    c = a["calodigis"]
    n = ak.to_numpy(ak.num(c["calodigis.SiPMID_"]))
    wf = c["calodigis.waveform_"]
    length = ak.to_numpy(ak.flatten(ak.num(wf, axis=2)))
    return {"event": np.repeat(pos, n), "sipm": ak.to_numpy(ak.flatten(c["calodigis.SiPMID_"])),
            "t0": ak.to_numpy(ak.flatten(c["calodigis.t0_"])), "peakpos": ak.to_numpy(ak.flatten(c["calodigis.peakpos_"])),
            "offsets": np.r_[0, np.cumsum(length)], "samples": ak.to_numpy(ak.flatten(wf, axis=None))}


def summaries(d):
    length = np.diff(d["offsets"])
    member = np.repeat(np.arange(length.size), length)
    peak = np.zeros(length.size, np.int64)
    np.maximum.at(peak, member, d["samples"])
    total = np.bincount(member, weights=d["samples"], minlength=length.size)
    return {"length": length, "peak": peak, "sum": total, "t0": d["t0"]}


def keyed(d):
    return (d["event"].astype(np.int64) * 10000 + d["sipm"]) * 100000 + d["t0"]


def exact_compare(fast, stock):
    kf, ks = keyed(fast), keyed(stock)
    common, i_f, i_s = np.intersect1d(kf, ks, assume_unique=False, return_indices=True)
    lf, ls = np.diff(fast["offsets"]), np.diff(stock["offsets"])
    same_len = lf[i_f] == ls[i_s]
    same = np.zeros(common.size, bool)
    maxdiff = np.zeros(common.size, np.int64)
    ok = np.flatnonzero(same_len)
    if ok.size:
        n = lf[i_f][ok]
        mf = np.repeat(fast["offsets"][i_f][ok], n) + _ranges(n)
        ms = np.repeat(stock["offsets"][i_s][ok], n) + _ranges(n)
        diff = np.abs(fast["samples"][mf] - stock["samples"][ms])
        member = np.repeat(np.arange(ok.size), n)
        md = np.zeros(ok.size, np.int64)
        np.maximum.at(md, member, diff)
        maxdiff[ok] = md
        same[ok] = (md == 0) & (fast["peakpos"][i_f][ok] == stock["peakpos"][i_s][ok])
    return {"n_fast": int(kf.size), "n_stock": int(ks.size), "n_matched_key": int(common.size),
            "n_identical": int(same.sum()), "n_same_length": int(same_len.sum()),
            "max_sample_diff": int(maxdiff.max(initial=0)), "n_maxdiff_le1": int((maxdiff <= 1).sum()),
            "unmatched_fast": int(kf.size - common.size), "unmatched_stock": int(ks.size - common.size)}


def _ranges(n):
    return np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)


def shape_compare(fast, stock, name, bins):
    hf, _ = np.histogram(fast, bins)
    hs, _ = np.histogram(stock, bins)
    ok = (hf + hs) > 0
    chi2 = float((((hf - hs) ** 2) / np.maximum(hf + hs, 1))[ok].sum())
    return {"quantity": name, "chi2": chi2, "ndf": int(ok.sum()), "mean_fast": float(np.mean(fast)),
            "mean_stock": float(np.mean(stock)), "n_fast": int(np.size(fast)), "n_stock": int(np.size(stock))}


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", required=True, help="stock art file with CaloShowerSteps, or a table export")
    p.add_argument("--ntuple", required=True, help="stock_chain EventNtuple file")
    p.add_argument("--sim-crystal", required=True)
    p.add_argument("--pulse", required=True)
    p.add_argument("--noise", required=True)
    p.add_argument("--backend", default="numpy", choices=["numpy", "jax"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--baseline-ntuple", help="same chain, independent stock seed: stock-vs-stock reference")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.backend == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
    sim = stockio.load_sim_crystal(args.sim_crystal)
    events, timing, steps = stockio.load_steps(args.steps, sim)
    pulse = stockio.load_pulse(args.pulse)
    noise = stockio.load_noise(args.noise)
    report = {"events": int(events.size), "backend": args.backend, "seed": args.seed,
              "commit": git_sha(Path(__file__).parent), "inputs": vars(args)}
    t0 = time.perf_counter()
    fast_det = fast_digis(steps, timing, sim, pulse, backend=args.backend)
    report["time_det_s"] = time.perf_counter() - t0
    stock_det = stock_digis(args.ntuple, "det", events)
    report["noise_off"] = exact_compare(fast_det, stock_det)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    fast_noi = fast_digis(steps, timing, sim, pulse, noise=noise, rng=rng, backend=args.backend)
    report["time_noi_s"] = time.perf_counter() - t0
    stock_noi = stock_digis(args.ntuple, "noi", events)
    sf, ss = summaries(fast_noi), summaries(stock_noi)
    nf = np.bincount(fast_noi["event"], minlength=events.size)
    ns = np.bincount(stock_noi["event"], minlength=events.size)
    report["noise_on"] = {
        "n_fast": int(nf.sum()), "n_stock": int(ns.sum()),
        "per_event_diff_mean": float(np.mean(nf - ns)), "per_event_diff_err": float(np.std(nf - ns) / np.sqrt(nf.size)),
        "shapes": [shape_compare(sf["length"], ss["length"], "length", np.arange(16, 120, 2)),
                   shape_compare(sf["peak"], ss["peak"], "peak_adc", np.r_[np.arange(0, 200, 5), np.arange(200, 4200, 100)]),
                   shape_compare(sf["t0"], ss["t0"], "t0_ns", np.arange(150, 1650, 25)),
                   shape_compare(sf["sum"], ss["sum"], "sum_adc", np.geomspace(300, 2e5, 50))]}
    if args.baseline_ntuple:
        base = stock_digis(args.baseline_ntuple, "noi", events)
        sb = summaries(base)
        nb = np.bincount(base["event"], minlength=events.size)
        report["noise_on_stock_vs_stock"] = {
            "n_baseline": int(nb.sum()), "per_event_diff_mean": float(np.mean(nb - ns)),
            "per_event_diff_err": float(np.std(nb - ns) / np.sqrt(nb.size)),
            "shapes": [shape_compare(sb["length"], ss["length"], "length", np.arange(16, 120, 2)),
                       shape_compare(sb["peak"], ss["peak"], "peak_adc", np.r_[np.arange(0, 200, 5), np.arange(200, 4200, 100)]),
                       shape_compare(sb["t0"], ss["t0"], "t0_ns", np.arange(150, 1650, 25)),
                       shape_compare(sb["sum"], ss["sum"], "sum_adc", np.geomspace(300, 2e5, 50))]}
        for s_ in report["noise_on_stock_vs_stock"]["shapes"]:
            print("stock-vs-stock", s_)
    np.savez_compressed(Path(args.out).with_suffix(".npz"),
                        **{f"fast_noi_{k}": v for k, v in sf.items()}, **{f"stock_noi_{k}": v for k, v in ss.items()})
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("noise_off", "time_det_s", "time_noi_s")}, indent=1))
    for s in report["noise_on"]["shapes"]:
        print(s)


if __name__ == "__main__":
    main()
