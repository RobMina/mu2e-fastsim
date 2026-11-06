"""Make a slide-ready throughput plot from validated single-pass measurement receipts."""
import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(report, output):
    """Show actual process throughput, without claiming grid-slot equivalence."""
    runs = report["runs"]
    if len(runs) != 3 or any(run["validation"]["outputs"]["equal"] is not True for run in runs):
        raise ValueError("three validated single-pass measurements required")
    hardware = (report.get("hardware") or {}).get("text", "")
    expected = ["fresh_private_cache", "populated_cache_fresh_process", "populated_cache_fresh_process"]
    if ("A100" not in hardware or re.search(r"\bcpus=2\b", hardware) is None
            or [run.get("cache_state") for run in runs] != expected
            or report["stock"].get("endpoint") != "UVA historical 1 CPU; not grid or GPVM"):
        raise ValueError("this figure requires the recorded UVA A100/2-CPU and historical 1-CPU configuration")
    stock = report["stock"]["external_process"]["events_per_s"]
    rates = np.array([stock] + [run["python_process"]["events_per_s"] for run in runs], dtype=float)
    if not np.all(np.isfinite(rates) & (rates > 0)):
        raise ValueError("finite positive process rates required")
    labels = ["Stock Offline\n1 UVA CPU", "Fast, fresh compilation cache\n1 A100 + 2 host CPUs",
              "Fast, populated cache: run 1\n1 A100 + 2 host CPUs", "Fast, populated cache: run 2\n1 A100 + 2 host CPUs"]
    with plt.rc_context({"font.size": 13, "font.family": "DejaVu Sans"}):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        fig.subplots_adjust(left=.32, right=.94, top=.78, bottom=.27)
        y = np.arange(len(rates))[::-1]
        ax.barh(y, rates, color=["#85837e", "#7397c7", "#2a78d6", "#2a78d6"], height=.58)
        ax.set_xscale("log")
        ax.set_xlim(min(.5, rates.min() / 2), rates.max() * 1.65)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Primary events / s — process launch through output and exit (log scale)")
        ax.grid(axis="x", which="major", alpha=.2)
        ax.set_axisbelow(True)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
        ax.tick_params(axis="y", length=0)
        for position, rate in zip(y, rates):
            ax.text(rate * 1.06, position, f"{rate:.2f}", va="center", fontweight="bold")
        fig.suptitle("Run 1A pileup: tracker digitisation + CRV response/reco", fontsize=19, y=.96)
        fig.text(.5, .88, f"{report['events']:,} primary events per run, including empty detector events; single-pass fresh processes",
                 ha="center", fontsize=13)
        fig.text(.03, .12, "Existing detector-step inputs → tracker digis/ADCs + CRV digis, pulses and clusters.\n"
                 "Includes loading, validation, sampling/mixing, response and output; excludes prior transport, staging and queue wait.\n"
                 "Historical stock CPU vs GPU-backed fast path: different hardware, timing inputs and diagnostic schemas.\n"
                 "No calorimeter, trigger, track fitting or full art output; development result, not grid-slot equivalence.",
                 fontsize=10, va="center")
        fig.savefig(output, dpi=200)
        plt.close(fig)


def public_numbers(report):
    """Export only collaborator-readable quantities, not internal paths or job IDs."""
    return {"events": report["events"], "denominator": report["denominator"],
            "boundary": report["boundary"], "excluded": report["excluded"],
            "cache_scope": report["cache_scope"], "comparison_scope": report["comparison_scope"],
            "status": "development only; not a certified grid/GPVM comparison",
            "stock": {"endpoint": "historical UVA CPU, 1 core, 8 GB RAM",
                      "process_wall_s": report["stock"]["external_process"]["wall_s"],
                      "events_per_s": report["stock"]["external_process"]["events_per_s"]},
            "fast": [{"cache_state": run["cache_state"], "endpoint": "UVA A100-80GB + 2 host CPUs, 64 GB RAM",
                      "process_wall_s": run["python_process"]["wall_s"],
                      "events_per_s": run["python_process"]["events_per_s"],
                      "container_wall_s": run["outer_container_process"]["wall_s"],
                      "container_events_per_s": run["outer_container_process"]["events_per_s"],
                      "setup_s": run["stages"]["setup_s"], "write_s": run["stages"]["write_s"],
                      "response_s": run["stages"]["passes"][0]["combined_response_s"],
                      "saved_output_and_sampling_exact": True} for run in report["runs"]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--numbers", type=Path, help="optional collaborator-facing measurement JSON")
    args = parser.parse_args(argv)
    if args.out.exists() or (args.numbers is not None and args.numbers.exists()):
        parser.error("refusing to overwrite an existing figure or numbers")
    report = json.loads(args.report.read_text())
    public = public_numbers(report) if args.numbers is not None else None
    make_figure(report, args.out)
    if args.numbers is not None:
        with args.numbers.open("x") as stream:
            stream.write(json.dumps(public, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
