"""Saved-input comparison for a shower library, with simple response cluster sums."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import uproot

from .export import EVENT, join, keys, read
from .library import load, sample
from .response import cluster_sum, place


def response_inputs(path):
    with uproot.open(path) as source:
        events = source["response/events"].arrays(list(EVENT) + ["pbtime", "mbtime", "event_length", "marker", "spill", "crystal_length"], library="np")
        steps = source["response/steps"].arrays(["crystal", "lru"], library="np")
    ids = np.unique(steps["crystal"])
    lo = np.full(int(ids.max()) + 1, np.inf)
    hi = np.full(len(lo), -np.inf)
    np.minimum.at(lo, steps["crystal"], steps["lru"])
    np.maximum.at(hi, steps["crystal"], steps["lru"])
    if np.any(lo[ids] != hi[ids]):
        raise ValueError("this simple comparison needs constant per-crystal LRU conditions")
    lo[~np.isfinite(lo)] = np.nan
    return events, lo


def apply_response(backend, steps, query, event_conditions, lru, *, start, end, buffer):
    mapped = steps["mapped"]
    row = steps["entry"]
    crystal = steps["crystal"][mapped]
    if np.any(crystal >= len(lru)) or not np.all(np.isfinite(lru[crystal])):
        raise ValueError("sampled crystal has no saved response conditions")
    timing = {name: event_conditions[field][row] for name, field in (
        ("pbtime", "pbtime"), ("mbtime", "mbtime"), ("event_length", "event_length"),
        ("time_from_protons_to_dr_marker", "marker"))}
    folded, accepted = backend.fold_step_time(steps["time"], onspill=event_conditions["spill"][row] == 1,
                                             digitization_start=start, digitization_end=end, digitization_buffer=buffer, **timing)
    energy = np.zeros(len(row))
    energy[mapped] = backend.corrected_energy(steps["energy_g4"][mapped], steps["energy_birks"][mapped],
                                             steps["local_position"][mapped, 2],
                                             event_conditions["crystal_length"][row[mapped]], lru[crystal])
    corrected = dict(steps, time=folded)
    return cluster_sum(corrected, len(query["energy"]), energies=energy, accepted=accepted & mapped)


def paired_mean(fast, stock, group):
    """Paired block-jackknife error; blocks are not production shards."""
    fast, stock = np.asarray(fast, float), np.asarray(stock, float)
    unique, inverse = np.unique(group, return_inverse=True)
    total = np.bincount(inverse)
    delta = fast - stock
    sums = np.bincount(inverse, weights=delta)
    if len(unique) < 2 or np.any(len(delta) - total == 0):
        error = None
    else:
        leave = (delta.sum() - sums) / (len(delta) - total)
        error = float(np.sqrt((len(unique) - 1) / len(unique) * np.sum((leave - leave.mean()) ** 2)))
    return dict(fast=float(fast.mean()), stock=float(stock.mean()), difference=float(delta.mean()), error=error)


def paired_ratio(fast, stock, group):
    fast, stock = np.asarray(fast, float), np.asarray(stock, float)
    unique, inverse = np.unique(group, return_inverse=True)
    fs = np.bincount(inverse, weights=fast)
    ss = np.bincount(inverse, weights=stock)
    if stock.sum() == 0:
        return dict(ratio=None, error=None)
    denominator = stock.sum() - ss
    if len(unique) < 2 or np.any(denominator == 0):
        error = None
    else:
        leave = (fast.sum() - fs) / denominator
        error = float(np.sqrt((len(unique) - 1) / len(unique) * np.sum((leave - leave.mean()) ** 2)))
    return dict(ratio=float(fast.sum() / stock.sum()), error=error)


def paired_hist(fast, stock, group, edges):
    unique, inverse = np.unique(group, return_inverse=True)
    xedges = np.arange(len(unique) + 1) - .5
    f = np.histogram2d(inverse, fast, bins=(xedges, edges))[0]
    s = np.histogram2d(inverse, stock, bins=(xedges, edges))[0]
    delta = f - s
    variance = len(unique) * np.var(delta, axis=0, ddof=1) if len(unique) > 1 else np.full(len(edges) - 1, np.nan)
    difference = delta.sum(axis=0)
    usable = variance > 0
    pull = np.divide(difference, np.sqrt(variance), out=np.full(len(difference), np.nan), where=usable)
    return dict(edges=np.asarray(edges).tolist(), fast=f.sum(axis=0).tolist(), stock=s.sum(axis=0).tolist(),
                pull=np.where(np.isfinite(pull), pull, None).tolist(),
                variance=np.where(np.isfinite(variance), variance, None).tolist(), chi2=float(np.sum(pull[usable] ** 2)),
                ndf=int(usable.sum()), chi2_method="diagonal paired event-block covariance; no fitted normalization",
                zero_variance_nonzero_difference_bins=int(np.sum((variance == 0) & (difference != 0))))


def plot(histograms, path, title, *, labels=("Stock CSS + response sum", "Sampled CSS + response sum")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(histograms), figsize=(6 * len(histograms), 6),
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for i, (label, h) in enumerate(histograms.items()):
        edges, f, s = np.asarray(h["edges"]), np.asarray(h["fast"]), np.asarray(h["stock"])
        centers = (edges[1:] + edges[:-1]) / 2
        axes[0, i].stairs(s, edges, color="#0072B2", label=labels[0])
        axes[0, i].stairs(f, edges, color="#D55E00", label=labels[1])
        axes[0, i].set_ylabel("Entries / bin")
        axes[0, i].legend(fontsize=8)
        axes[0, i].set_title(f"Paired chi²/ndf = {h['chi2']:.1f}/{h['ndf']}")
        axes[1, i].axhline(0, color="gray", lw=1)
        axes[1, i].plot(centers, h["pull"], ".", color="#D55E00")
        axes[1, i].set_ylabel("Pull")
        axes[1, i].set_xlabel(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--response-input", required=True, type=Path)
    parser.add_argument("--response-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", choices=("development", "heldout"), required=True)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--digitization-start", type=float, required=True)
    parser.add_argument("--digitization-end", type=float, required=True)
    parser.add_argument("--digitization-buffer", type=float, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("calo_response_backend", args.response_source)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    library, metadata = load(args.library)
    entries, css, _, census, total = read(args.input)
    selected = entries["event_row"] < metadata["split_row"]
    if args.arm == "heldout":
        selected = ~selected
    query = {k: v[selected] for k, v in entries.items()}
    index = np.full(len(selected), -1, dtype=np.int64)
    index[selected] = np.arange(selected.sum())
    take = index[css["entry"]] >= 0
    stock = {k: v[take] for k, v in css.items()}
    stock["entry"] = index[stock["entry"]]
    stock = place(library, stock)
    if not np.all(stock["mapped"]):
        bad = stock["local_position"][~stock["mapped"]]
        raise ValueError(f"{len(bad)} stock CSS outside geometry; local extrema {bad.min(axis=0)}, {bad.max(axis=0)}")
    begin = time.perf_counter()
    draw = sample(library, query, np.random.default_rng(args.seed), orientation="disk")
    placed = place(library, draw)
    draw_seconds = time.perf_counter() - begin
    conditions, lru = response_inputs(args.response_input)
    if len(conditions["event"]) != census["events"]:
        raise ValueError("response conditions event count differs from export")
    with uproot.open(args.input) as source:
        ids = source["dump/events"].arrays(list(EVENT), library="np")
    order = join(keys(conditions, EVENT), keys(ids, EVENT)[query["event_row"]])
    conditions = {k: v[order] for k, v in conditions.items()}
    options = dict(start=args.digitization_start, end=args.digitization_end, buffer=args.digitization_buffer)
    f = apply_response(backend, placed, query, conditions, lru, **options)
    s = apply_response(backend, stock, query, conditions, lru, **options)
    group = query["event_row"] // 100
    histograms = {"Cluster-sum energy [MeV]": paired_hist(f["energy"], s["energy"], group, np.arange(0, 111, 5))}
    positive = (f["energy"] > 0) & (s["energy"] > 0)
    histograms["Cluster-sum time [ns], both positive"] = paired_hist(
        f["time"][positive], s["time"][positive], group[positive], np.arange(250, 1901, 50))
    momentum = np.sqrt(query["energy"] ** 2 - .51099895 ** 2)
    histograms["E/p at disk front (truth diagnostic)"] = paired_hist(
        f["energy"] / momentum, s["energy"] / momentum, group, np.linspace(0, 1.2, 25))
    missing = ~placed["mapped"]
    result = dict(arm=args.arm, seed=args.seed, entries=len(query["energy"]), source_events=census["events"],
                  split=metadata["split"], statistic="paired 100-source-event block jackknife; not independent production shards",
                  model="Deterministic response + one energy-weighted cluster sum; not stock cluster reconstruction",
                  energy_mean=paired_mean(f["energy"], s["energy"], group),
                  acceptance_above_10_mev=paired_mean(f["energy"] > 10, s["energy"] > 10, group),
                  acceptance_ratio=paired_ratio(f["energy"] > 10, s["energy"] > 10, group),
                  truth_ep_mean=paired_mean(f["energy"] / momentum, s["energy"] / momentum, group),
                  fallback_entries=int(draw["fallback"].sum()), unsupported_entries=int((~draw["supported"]).sum()),
                  unmapped_css=int(missing.sum()),
                  unmapped_visible_energy_fraction=float(draw["energy_birks"][missing].sum() / draw["energy_birks"].sum()),
                  mean_time_difference_ns=float(np.mean(f["time"][positive] - s["time"][positive])),
                  mean_position_distance_mm=float(np.mean(np.linalg.norm(f["position"][positive] - s["position"][positive], axis=1))),
                  sample_and_place_seconds=draw_seconds, sample_and_place_entries_per_second=len(query["energy"]) / draw_seconds,
                  stock_rate_comparison=None, histograms=histograms,
                  non_front_positive_events=census["non_front_entry_positive_events"],
                  total_source_visible_positive_events=int(np.sum(total > 0)))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "numbers.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.out / "clusters.npz", event_row=query["event_row"], fast_energy=f["energy"], stock_energy=s["energy"],
                        fast_time=f["time"], stock_time=s["time"], fast_position=f["position"], stock_position=s["position"],
                        donor=draw["donor"], fallback=draw["fallback"])
    plot({"Cluster-sum energy [MeV]": histograms["Cluster-sum energy [MeV]"]}, args.out / "calo_ce.png",
         f"CE {args.arm}: direct-electron disk-front entries; simple cluster sum")
    print(json.dumps({k: v for k, v in result.items() if k != "histograms"}, indent=2))


if __name__ == "__main__":
    main()
