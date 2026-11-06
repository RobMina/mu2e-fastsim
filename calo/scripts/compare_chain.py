#!/usr/bin/env python3
"""Fast calorimeter chain from stock CaloShowerSteps vs stock digi + reco (production settings) on the same steps.

Paired by event: the fast arm and the stock arm see the same steps with independent randomness, so the reference
for every paired difference is a second stock seed (stock' - stock). Writes ``numbers.json``, ``calo_reco.png``
(overlays with pull panels) and ``calo_reco_paired.png`` (paired differences).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from calofastsim import stockio
from calofastsim.chain import run_chain
from calofastsim.readout import ReadoutConfig

STOCK, FAST = "#2a78d6", "#eb6834"


def stock_arm(ntuple, arm, events):
    import compare_reco
    pos, digis, reco, hits, pbt = compare_reco.stock_tables(ntuple, arm, events)
    return {"hits": hits, "clusters": hits.pop("clusters"), "n_digis": int(digis["t0"].size), "n_reco": int(reco["e"].size)}


def leading(cl, n_events):
    """Highest-energy cluster per event (clusters are sorted by event, then decreasing energy); NaN if none."""
    out = {k: np.full(n_events, np.nan) for k in ("energy", "time", "x", "y", "size")}
    first = np.r_[True, cl["event"][1:] != cl["event"][:-1]] if cl["event"].size else np.zeros(0, bool)
    ev = cl["event"][first]
    for k in out:
        out[k][ev] = np.asarray(cl[k], np.float64)[first]
    return out


def gauss_core(e, width=1.0):
    """Mean and sigma of a Gaussian fitted to the CE peak core (1 MeV bins, iterated window [mu-1.5s, mu+2s])."""
    from scipy.optimize import curve_fit
    e = e[np.isfinite(e)]
    mu, sig = float(np.median(e[e > 75.0])), 3.0
    bins = np.arange(40.0, 115.0 + width, width)
    h, edges = np.histogram(e, bins)
    c = 0.5 * (edges[1:] + edges[:-1])
    f = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)
    for _ in range(6):
        w = (c > mu - 1.5 * sig) & (c < mu + 2.0 * sig) & (h > 0)
        p, _ = curve_fit(f, c[w], h[w], p0=[h[w].max(), mu, sig], sigma=np.sqrt(h[w]), absolute_sigma=True)
        mu, sig = float(p[1]), float(abs(p[2]))
    return mu, sig


def cluster_metrics(e, tail=10.0):
    """CE leading-cluster peak and core width, and the low-tail fraction (E below peak - ``tail`` MeV)."""
    e = e[np.isfinite(e)]
    mu, sig = gauss_core(e)
    core = e[e > 75.0]
    q16, q50, q84 = np.percentile(core, [16, 50, 84])
    return {"peak": mu, "width": sig, "low_tail": float(np.mean(e < mu - tail)), "n": int(e.size),
            "mean": float(np.mean(e)), "core_median": float(q50), "core_halfwidth_16_84": float(0.5 * (q84 - q16))}


def bootstrap_diff(fe, se, n_boot=200, seed=7):
    """Paired event bootstrap of fast - stock for peak, width and low-tail fraction."""
    rng = np.random.default_rng(seed)
    keys = ("peak", "width", "low_tail", "core_median", "core_halfwidth_16_84")
    base = {k: cluster_metrics(fe)[k] - cluster_metrics(se)[k] for k in keys}
    samples = {k: [] for k in base}
    for _ in range(n_boot):
        i = rng.integers(0, fe.size, fe.size)
        try:
            mf, ms = cluster_metrics(fe[i]), cluster_metrics(se[i])
        except (RuntimeError, ValueError, TypeError):
            continue
        for k in base:
            samples[k].append(mf[k] - ms[k])
    return {k: {"diff": float(base[k]), "err": float(np.std(samples[k])), "n_boot": len(samples[k])} for k in base}


def paired(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    return {"n": int(d.size), "mean": float(np.mean(d)), "err": float(np.std(d) / np.sqrt(max(d.size, 1))),
            "rms": float(np.std(d)), "q16_50_84": [float(v) for v in np.percentile(d, [16, 50, 84])]}


def shape(a, b, bins):
    ha, _ = np.histogram(a[np.isfinite(a)], bins)
    hb, _ = np.histogram(b[np.isfinite(b)], bins)
    ok = (ha + hb) > 0
    return {"chi2": float((((ha - hb) ** 2) / np.maximum(ha + hb, 1))[ok].sum()), "ndf": int(ok.sum())}


def single_hits(h, n_events):
    """(event, crystal) -> (E, t) for crystals with exactly one hit."""
    key = h["event"].astype(np.int64) * 2000 + h["crystal"]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    one = cnt[inv] == 1
    return key[one], np.asarray(h["e"], np.float64)[one], np.asarray(h["t"], np.float64)[one]


def hit_pairs(a, b, n_events):
    ka, ea, ta = single_hits(a, n_events)
    kb, eb, tb = single_hits(b, n_events)
    _, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    return ea[ia], eb[ib], ta[ia], tb[ib]


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def plot(out_dir, fast_lead, s1_lead, fast_hits, s1_hits, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    panels = [("leading cluster energy [MeV]", fast_lead["energy"], s1_lead["energy"], np.arange(40, 111, 2.0), False),
              ("hit energy [MeV]", np.asarray(fast_hits["e"], float), np.asarray(s1_hits["e"], float),
               np.geomspace(0.3, 110, 41), True),
              ("leading cluster radius [mm]", np.hypot(fast_lead["x"], fast_lead["y"]),
               np.hypot(s1_lead["x"], s1_lead["y"]), np.arange(350, 680, 10), False)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.8), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for k, (label, f, s, bins, logx) in enumerate(panels):
        hf, _ = np.histogram(f[np.isfinite(f)], bins)
        hs, _ = np.histogram(s[np.isfinite(s)], bins)
        ax, pax = axes[0, k], axes[1, k]
        ax.stairs(hs, bins, fill=True, color=STOCK, alpha=0.35, label="stock")
        ax.stairs(hs, bins, color=STOCK, lw=1.2)
        ax.stairs(hf, bins, color=FAST, lw=2, label="fast")
        ax.set_ylabel("entries")
        ax.grid(alpha=0.25, lw=0.5)
        pull = (hf - hs) / np.sqrt(np.maximum(hf + hs, 1))
        c = 0.5 * (bins[1:] + bins[:-1]) if not logx else np.sqrt(bins[1:] * bins[:-1])
        pax.axhline(0, color="#888", lw=0.8)
        pax.plot(c, pull, "o", color=FAST, ms=3.5)
        pax.set_ylim(-4, 4)
        pax.set_ylabel("pull")
        pax.set_xlabel(label)
        if logx:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ok = (hf + hs) > 0
        ax.set_title(f"$\\chi^2$/ndf = {float((((hf - hs) ** 2) / np.maximum(hf + hs, 1))[ok].sum()):.0f}/{int(ok.sum())}",
                     fontsize=9, loc="right")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Calorimeter reco, 2,000 CE: fast chain vs stock Offline on the same CaloShowerSteps", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "calo_reco.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    spec = [("leading cluster $\\Delta E$ [MeV]", "dE", np.arange(-6, 6.01, 0.25)),
            ("leading cluster $\\Delta t$ [ns]", "dt", np.arange(-2.5, 2.501, 0.1)),
            ("leading cluster $\\Delta$position [mm]", "dr", np.arange(0, 12.01, 0.25))]
    for ax, (label, key, bins) in zip(axes, spec):
        f, s = pairs[key]
        ax.hist(s, bins, histtype="stepfilled", color=STOCK, alpha=0.35, label="stock' $-$ stock")
        ax.hist(s, bins, histtype="step", color=STOCK, lw=1.2)
        ax.hist(f, bins, histtype="step", color=FAST, lw=2, label="fast $-$ stock")
        ax.set_xlabel(label)
        ax.set_ylabel("events")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Same events, independent randomness: fast vs stock compared with stock vs stock (second seed)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "calo_reco_paired.png", dpi=150)
    plt.close(fig)


def headline(r, args):
    """The deck numbers: one line per comparison, value and error."""
    lc, pl, ph = r["leading_cluster"], r["paired_leading_cluster"], r["paired_single_hits"]
    h = {"events": 2000, "events_with_cluster_fast_stock": [r["events_with_cluster"]["fast"],
                                                            r["events_with_cluster"]["stock"]],
         "cluster_peak_MeV_fast_stock": [lc["fast"]["peak"], lc["stock"]["peak"]],
         "cluster_peak_diff_MeV": [lc["fast_minus_stock"]["peak"]["diff"], lc["fast_minus_stock"]["peak"]["err"]],
         "cluster_width_MeV_fast_stock": [lc["fast"]["width"], lc["stock"]["width"]],
         "cluster_width_diff_MeV": [lc["fast_minus_stock"]["width"]["diff"], lc["fast_minus_stock"]["width"]["err"]],
         "cluster_low_tail_fast_stock": [lc["fast"]["low_tail"], lc["stock"]["low_tail"]],
         "cluster_low_tail_diff": [lc["fast_minus_stock"]["low_tail"]["diff"], lc["fast_minus_stock"]["low_tail"]["err"]],
         "paired_cluster_dE_MeV_mean_err_rms": [pl["dE"]["fast_minus_stock"][k] for k in ("mean", "err", "rms")],
         "paired_cluster_dE_rms_stock_vs_stock_MeV": pl["dE"]["stock2_minus_stock"]["rms"],
         "paired_cluster_dt_ns_mean_err_rms": [pl["dt"]["fast_minus_stock"][k] for k in ("mean", "err", "rms")],
         "paired_cluster_dt_rms_stock_vs_stock_ns": pl["dt"]["stock2_minus_stock"]["rms"],
         "paired_cluster_dpos_mm_mean_rms": [pl["dr"]["fast_minus_stock"][k] for k in ("mean", "rms")],
         "paired_cluster_dpos_mm_stock_vs_stock_mean_rms": [pl["dr"]["stock2_minus_stock"][k] for k in ("mean", "rms")],
         "paired_hit_dE_MeV_mean_err_rms": [ph["energy"]["fast_minus_stock"][k] for k in ("mean", "err", "rms")],
         "paired_hit_dE_rms_stock_vs_stock_MeV": ph["energy"]["stock2_minus_stock"]["rms"],
         "paired_hit_dt_ns_mean_err_rms": [ph["time"]["fast_minus_stock"][k] for k in ("mean", "err", "rms")],
         "paired_hit_dt_rms_stock_vs_stock_ns": ph["time"]["stock2_minus_stock"]["rms"],
         "chi2_ndf_cluster_energy": [r["shapes"]["leading_cluster_energy"]["fast"][k] for k in ("chi2", "ndf")],
         "chi2_ndf_hit_energy": [r["shapes"]["hit_energy"]["fast"][k] for k in ("chi2", "ndf")]}
    if args.digi_report:
        d = json.loads(Path(args.digi_report).read_text())
        h["digis_noise_off_identical_of_stock"] = [d["noise_off"]["n_identical"], d["noise_off"]["n_stock"]]
        h["digis_noise_on_count_diff_per_event"] = [d["noise_on"]["per_event_diff_mean"], d["noise_on"]["per_event_diff_err"]]
        h["digis_noise_on_chi2_ndf"] = {s["quantity"]: [s["chi2"], s["ndf"]] for s in d["noise_on"]["shapes"]}
    if args.reco_report:
        d = json.loads(Path(args.reco_report).read_text())[r["arm"]]
        rd, hi, cl = d["reco_digis"], d["hits_from_stock_recodigis"], d["clusters_from_stock_hits"]
        h["templatefit_on_stock_digis"] = {"paired": [rd["n_paired"], rd["n_stock"]],
                                           "energy_within_1pct": rd["energy_rel"]["frac_abs_lt_1pct"],
                                           "time_within_0p1ns": rd["time_ns"]["frac_abs_lt_0p1"]}
        h["hits_on_stock_recodigis_exact"] = [hi["energy_exact_frac"], hi["time_exact_frac"], hi["n_stock"]]
        h["clusters_on_stock_hits"] = {"n": [cl["n_fast"], cl["n_stock"]], "energy_equal": cl["energy_exact_frac"],
                                       "size_equal": cl["size_equal_frac"], "time_rms_ns": cl["time_ns"]["rms"],
                                       "position_max_mm": cl["position_mm"]["q01_16_50_84_99"][-1]}
    for item in args.timing:
        name, path = item.split("=", 1)
        t = json.loads(Path(path).read_text())
        h.setdefault("throughput_events_per_s", {})[name] = t["events_per_s_warm_median"]
    if args.stock_ms_per_event:
        h.setdefault("throughput_events_per_s", {})["stock_cpu_slot"] = 1000.0 / args.stock_ms_per_event
    return h


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", required=True)
    p.add_argument("--ntuple", required=True, help="stock_chain ntuple, seed 1")
    p.add_argument("--ntuple2", required=True, help="stock_chain ntuple, independent seed")
    p.add_argument("--geometry", required=True)
    p.add_argument("--sim-crystal", required=True)
    p.add_argument("--pulse", required=True)
    p.add_argument("--noise", required=True)
    p.add_argument("--propagation", required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--arm", default="prd")
    p.add_argument("--digi-report", help="compare_digis.py JSON to fold into the headline numbers")
    p.add_argument("--reco-report", help="compare_reco.py JSON to fold into the headline numbers")
    p.add_argument("--timing", action="append", default=[], metavar="NAME=JSON",
                   help="time_chain.py output to fold in (repeatable)")
    p.add_argument("--stock-ms-per-event", type=float, help="stock calo digi+reco CPU time per event (TimeTracker)")
    p.add_argument("--device-digitize", action="store_true", help="on-device digitisation (digi_device)")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import compare_reco
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim = stockio.load_sim_crystal(args.sim_crystal)
    events, timing, steps = stockio.load_steps(args.steps, sim)
    pulse, noise = stockio.load_pulse(args.pulse), stockio.load_noise(args.noise)
    table = stockio.load_propagation(args.propagation)
    geom = compare_reco.load_geometry(args.geometry)
    n = events.size
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    fast = run_chain(steps, timing, lambda s: stockio.adc_per_pe(sim, s), pulse, noise, geom, rng,
                     ro_cfg=ReadoutConfig(), table=table, adc2mev=stockio.adc_to_mev,
                     device_digitize=args.device_digitize)
    t_fast = time.perf_counter() - t0
    s1, s2 = stock_arm(args.ntuple, args.arm, events), stock_arm(args.ntuple2, args.arm, events)
    lf, l1, l2 = leading(fast["clusters"], n), leading(s1["clusters"], n), leading(s2["clusters"], n)

    def pos_d(a, b):
        return np.hypot(a["x"] - b["x"], a["y"] - b["y"])

    pairs = {"dE": (lf["energy"] - l1["energy"], l2["energy"] - l1["energy"]),
             "dt": (lf["time"] - l1["time"], l2["time"] - l1["time"]),
             "dr": (pos_d(lf, l1), pos_d(l2, l1))}
    both = np.isfinite(lf["energy"]) & np.isfinite(l1["energy"])
    both2 = np.isfinite(l2["energy"]) & np.isfinite(l1["energy"])
    hfe, hse, hft, hst = hit_pairs(fast["hits"], s1["hits"], n)
    h2e, h1e, h2t, h1t = hit_pairs(s2["hits"], s1["hits"], n)
    report = {
        "sample": "2,000 CE (R6 pilot, MDC2025ay transport), CaloShowerStepMaker steps; stock Offline v13_39_00",
        "commit": git_sha(Path(__file__).parent), "seed_fast": args.seed, "arm": args.arm,
        "counts": {k: {"fast": int(fast[k]["event"].size), "stock": int(s1[k]["event"].size),
                       "stock2": int(s2[k]["event"].size)} for k in ("hits", "clusters")},
        "events_with_cluster": {"fast": int(np.isfinite(lf["energy"]).sum()), "stock": int(np.isfinite(l1["energy"]).sum()),
                                "stock2": int(np.isfinite(l2["energy"]).sum())},
        "leading_cluster": {"fast": cluster_metrics(lf["energy"]), "stock": cluster_metrics(l1["energy"]),
                            "stock2": cluster_metrics(l2["energy"]),
                            "fast_minus_stock": bootstrap_diff(lf["energy"][both], l1["energy"][both]),
                            "stock2_minus_stock": bootstrap_diff(l2["energy"][both2], l1["energy"][both2])},
        "paired_leading_cluster": {k: {"fast_minus_stock": paired(*([pairs[k][0], np.zeros_like(pairs[k][0])])),
                                       "stock2_minus_stock": paired(*([pairs[k][1], np.zeros_like(pairs[k][1])]))}
                                   for k in pairs},
        "paired_single_hits": {"energy": {"fast_minus_stock": paired(hfe, hse), "stock2_minus_stock": paired(h2e, h1e)},
                               "time": {"fast_minus_stock": paired(hft, hst), "stock2_minus_stock": paired(h2t, h1t)}},
        "shapes": {"leading_cluster_energy": {"fast": shape(lf["energy"], l1["energy"], np.arange(40, 111, 2.0)),
                                              "stock2": shape(l2["energy"], l1["energy"], np.arange(40, 111, 2.0))},
                   "hit_energy": {"fast": shape(np.asarray(fast["hits"]["e"], float), np.asarray(s1["hits"]["e"], float),
                                                np.geomspace(0.3, 110, 41)),
                                  "stock2": shape(np.asarray(s2["hits"]["e"], float), np.asarray(s1["hits"]["e"], float),
                                                  np.geomspace(0.3, 110, 41))},
                   "leading_cluster_radius": {"fast": shape(np.hypot(lf["x"], lf["y"]), np.hypot(l1["x"], l1["y"]),
                                                            np.arange(350, 680, 10)),
                                              "stock2": shape(np.hypot(l2["x"], l2["y"]), np.hypot(l1["x"], l1["y"]),
                                                              np.arange(350, 680, 10))}},
        "fast_chain_seconds": t_fast, "fast_refit_skipped_digis": fast["n_refit_skipped"],
        "fast_nonfinite_reco_digis": fast["n_nonfinite_reco"], "fast_nonfinite_examples": fast["nonfinite_examples"],
        "inputs": vars(args)}
    report["headline"] = headline(report, args)
    (out / "numbers.json").write_text(json.dumps(report, indent=1) + "\n")
    np.savez_compressed(out / "leading_clusters.npz", **{f"{a}_{k}": v for a, d in (("fast", lf), ("stock", l1),
                                                                                  ("stock2", l2)) for k, v in d.items()})
    plot(out, lf, l1, fast["hits"], s1["hits"], pairs)
    print(json.dumps({k: report[k] for k in ("counts", "events_with_cluster", "leading_cluster", "fast_chain_seconds",
                                                 "fast_nonfinite_reco_digis", "fast_nonfinite_examples")},
                     indent=1))


if __name__ == "__main__":
    main()
