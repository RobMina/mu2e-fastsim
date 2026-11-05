"""Render saved CE-chain measurements without rerunning transport or held-out draws."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def public_numbers(report):
    """Summary statistics only, excluding input arrays, paths, job IDs and allocation costs."""
    pairs = {k: v for k, v in report.items() if "_vs_" in k}
    t = report["timing"]
    return dict(events=report["events"], counts=report["counts"], comparisons=pairs,
                energy_mean=report["energy_mean"], energy_chi2=report["energy_shape"]["chi2"],
                energy_ndf=report["energy_shape"]["ndf"], error_method=report["error_method"], model=report["model"],
                versions=report["versions"], backend=report["backend"],
                first_chain_seconds=t["measurements"][0]["chain_seconds"],
                warm_chain_seconds=t["warm_chain_seconds"], warm_generated_per_second=t["warm_generated_per_second"],
                stock_grid_slot_equivalent=t["stock_grid_slot_equivalent"], stock_generated_per_second=t["stock_generated_per_second"],
                boundary_note=t["boundary_note"])


def reco_numbers(report):
    keys = ("events", "fast_selected", "stock_selected", "acceptance_difference", "acceptance_ratio", "energy_mean",
            "both_selected", "fast_only", "stock_only", "paired_time_ns", "paired_position_distance_mm", "timing", "reco_commit", "device")
    out = {k: report[k] for k in keys if k in report}
    out.update(energy_chi2=report["energy_shape"]["chi2"], energy_ndf=report["energy_shape"]["ndf"])
    out["reco_commit"] = report.get("reco_commit", report.get("cal1_commit"))
    return out


def figure(reports, out, reco=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, values, errors = [], [], []
    for device, report in reports.items():
        for key, name in (("fast_visible_vs_stock_visible", "shower / G4 visible"),
                          ("fast_response_vs_stock_response", "shower / G4 response sum"),
                          ("proxy_wall_vs_stock_visible", "cesob wall proxy / G4 visible")):
            pair = report[key]
            labels.append(device + ": " + name)
            values.append(pair["ratio"])
            errors.append(pair["ratio_error"])
    for device, report in (reco or {}).items():
        labels.append(device + ": reconstructed cluster / stock")
        values.append(report["acceptance_ratio"]["ratio"])
        errors.append(report["acceptance_ratio"]["error"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.8, 1]})
    y = np.arange(len(labels))
    axes[0].axvline(1, color="gray", lw=1)
    axes[0].errorbar(values, y, xerr=errors, fmt="o", color="#0072B2", capsize=3)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Acceptance ratio (>10 MeV); paired event-block error")
    speed = reco if reco else reports
    names = list(speed)
    key = "generated_per_second" if reco else "warm_generated_per_second"
    rates = [speed[k]["timing"][key] for k in names]
    bars = axes[1].bar(names, rates, color=["#0072B2", "#D55E00"][:len(names)])
    for bar, value in zip(bars, rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")
    axes[1].set_ylabel("Generated CE / s (warm 1,000-event batch)")
    axes[1].set_ylim(0, max(rates) * 1.25)
    fig.suptitle("CE birth → calorimeter; held-out generated-event denominator", fontsize=11)
    fig.text(.5, .015, "Timing through reconstructed clusters. Primary-only fast transport; the shower model remains diagnostic." if reco else
             "Primary-only fast route to response sum; not stock cluster reconstruction. Shower model remains diagnostic.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--reco-cpu", type=Path)
    parser.add_argument("--reco-gpu", type=Path)
    parser.add_argument("--front-entry-report", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    reports = {"CPU": json.loads(args.cpu.read_text())}
    if args.gpu:
        reports["MIG + host"] = json.loads(args.gpu.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    reco = {k: json.loads(path.read_text()) for k, path in (("CPU", args.reco_cpu), ("MIG + host", args.reco_gpu)) if path}
    summary = {k: public_numbers(v) for k, v in reports.items()}
    summary["reconstruction"] = {k: reco_numbers(v) for k, v in reco.items()}
    (args.out / "chain_numbers.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    combined = {"generated_event_chain": summary}
    if args.front_entry_report:
        combined["stock_front_entry_diagnostic"] = json.loads(args.front_entry_report.read_text())
    (args.out / "numbers.json").write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n")
    figure(reports, args.out / "calo_chain.png", reco)
    if reco:
        from .compare import plot
        shape = {k + " leading-cluster energy [MeV]": v["energy_shape"] for k, v in reco.items()}
        plot(shape, args.out / "calo_cluster.png", "CE chain through calorimeter reconstruction; paired generated-event blocks",
             labels=("G4 + stock reconstruction", "Fast transport + fast reconstruction"))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
