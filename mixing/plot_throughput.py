"""Plot separately scoped endpoint measurements from validated single-pass receipts."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENDPOINT_SCOPE = ("Separate endpoint measurements, not a speedup comparison: devices, "
                  "input boundaries and output schemas differ.")


def hardware_label(metadata):
    """Never infer hardware from a site, legacy schema, or private raw provenance."""
    label = (metadata or {}).get("label")
    if label is None:
        return "hardware not recorded"
    if not isinstance(label, str) or not label.strip() or any(c in label for c in "\r\n"):
        raise ValueError("hardware metadata needs a nonempty single-line public label")
    return label.strip()


def validated_rates(report):
    runs = report["runs"]
    if len(runs) != 3 or any(run["validation"]["outputs"]["equal"] is not True for run in runs):
        raise ValueError("three validated single-pass measurements required")
    expected = ["fresh_private_cache", "populated_cache_fresh_process", "populated_cache_fresh_process"]
    if [run.get("cache_state") for run in runs] != expected:
        raise ValueError("expected a fresh cache followed by two populated-cache fresh processes")
    rates = np.array([report["stock"]["external_process"]["events_per_s"]]
                     + [run["python_process"]["events_per_s"] for run in runs], dtype=float)
    if not np.all(np.isfinite(rates) & (rates > 0)):
        raise ValueError("finite positive process rates required")
    return rates


def make_figure(report, output):
    """Use separate axes and boundaries; neither compute nor display a rate ratio."""
    rates = validated_rates(report)
    fast_label = hardware_label(report.get("hardware"))
    stock_label = hardware_label(report["stock"].get("hardware"))
    with plt.rc_context({"font.size": 11, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})
        fig.subplots_adjust(left=.28, right=.94, top=.81, bottom=.25, hspace=1.2)
        panels = [(axes[0], rates[:1], ["Historical stock"], ["#85837e"],
                   f"Stock art process — {stock_label}\nNative mixing + framework/source I/O + native detector diagnostics"),
                  (axes[1], rates[1:], ["Fresh compilation cache", "Populated cache: run 1", "Populated cache: run 2"],
                   ["#7397c7", "#2a78d6", "#2a78d6"],
                   f"Fast Python processes — {fast_label}\nCached source steps → sampling/mixing + detector response + reduced ROOT output")]
        for ax, values, labels, colors, title in panels:
            y = np.arange(len(values))[::-1]
            ax.barh(y, values, color=colors, height=.58)
            ax.set_xlim(0, values.max() * 1.25)
            ax.set_yticks(y, labels)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Primary events / s — process launch through output and exit")
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
            for edge in ("top", "right", "left"):
                ax.spines[edge].set_visible(False)
            ax.tick_params(axis="y", length=0)
            for position, rate in zip(y, values):
                ax.text(rate + values.max() * .025, position, f"{rate:.2f}", va="center", fontweight="bold")
        fig.suptitle("Run 1A pileup: tracker digitisation + CRV response/reco", fontsize=18, y=.97)
        fig.text(.5, .91, f"{report['events']:,} primary events per run, including empty detector events; independent axes",
                 ha="center", fontsize=12)
        fig.text(.03, .11, ENDPOINT_SCOPE + "\n"
                 "Fast includes loading, validation, sampling/mixing, response and output; excludes prior extraction/transport, staging and queue wait.\n"
                 "Fresh means private compilation caches, not cold storage; populated-cache runs are fresh processes, not in-process warm passes.\n"
                 "No calorimeter, trigger, track fitting or full art output; development result, not grid-slot equivalence.",
                 fontsize=10, va="center")
        fig.savefig(output, dpi=200)
        plt.close(fig)


def public_numbers(report):
    """Whitelist public quantities; ignore obsolete fields in frozen legacy receipts."""
    validated_rates(report)
    return {"events": report["events"], "denominator": report["denominator"],
            "boundary": report["boundary"], "excluded": report["excluded"],
            "cache_scope": report["cache_scope"], "comparison_scope": report["comparison_scope"],
            "endpoint_scope": ENDPOINT_SCOPE,
            "status": "development only; not a certified grid/GPVM comparison",
            "stock": {"endpoint": hardware_label(report["stock"].get("hardware")),
                      "scope": report["stock"]["external_process"].get("scope", "external art process, native inputs and diagnostics"),
                      "process_wall_s": report["stock"]["external_process"]["wall_s"],
                      "events_per_s": report["stock"]["external_process"]["events_per_s"]},
            "fast": [{"cache_state": run["cache_state"], "endpoint": hardware_label(report.get("hardware")),
                      "scope": run["python_process"].get("scope", "external Python process, cached steps and reduced diagnostics"),
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
