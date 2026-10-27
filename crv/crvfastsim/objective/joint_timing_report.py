"""Summarize and plot saved joint-timing reports without reopening captures or recalibrating."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import joint_compare as J, validation as V


def summarize(reports):
    out = dict(standardization={}, all_bodies={}, same_end_pe={}, hardware={}, checks={})
    for block, report in reports.items():
        J.require(report.get("comparison") == "joint-timing" and report.get("passed") is True,
                  "requires completed joint-timing report")
        out["standardization"][block] = {}
        for mode, bodies in report["standardization"].items():
            out["standardization"][block][mode] = {body: dict(common_cells=v["all_pe"]["common_cells"],
                 samples=v["all_pe"]["samples"]) for body, v in bodies.items()}
        out["all_bodies"][block] = {name: dict(flow=s["flow"], event_multiplicity=s["event_multiplicity"])
                                           for name, s in report["samples"].items()}
        out["same_end_pe"][block] = {name: s["pairs"]["same_bar_same_end_distinct_sipm_equal_pe_bin"]
                                    for name, s in report["samples"].items()}
        out["hardware"][block] = {name: s["hardware"] for name, s in report["samples"].items()}
        cancellation, decomposition = [], []
        for sample in report["samples"].values():
            for row in sample["pairs"]["same_bar_same_end_distinct_sipm"]:
                if row["pairs"]:
                    cancellation.append(abs(row["propagation_difference_second"]))
                    cancellation.append(abs(row["difference_second"]-row["raw_difference_second"]))
            for mode in sample["strata"].values():
                for cell in mode["cells"]:
                    decomposition.append(abs(cell["second_moment"]-cell["raw_second_moment"]-
                                             cell["propagation_second_moment"]+2*cell["raw_propagation"]))
        out["checks"][block] = dict(max_same_end_cancellation_ns2=max(cancellation, default=0.),
                                    max_cell_decomposition_error_ns2=max(decomposition, default=0.))
        J.require(max(cancellation + decomposition, default=0.) < 1e-6, "timing decomposition failed")
    values = list(reports.values())
    J.require(all(r["samples"]["data"] == values[0]["samples"]["data"] for r in values),
              "A/B measured data diagnostics differ")
    out["checks"]["identical_measured_data_diagnostics"] = True
    return out


def text(summary):
    lines = ["# Thread C retained timing diagnostics", "",
             "Descriptive moments only; calibration/background unchanged. Exact-N support is narrower than the",
             "four-coordinate support. Differences between conditioning modes are not a multiplicity-only effect.", "",
             "## Exact multiplicity common support: selected-hit residual second moment [ns²]", "",
             "| Block/body | Cells | Nominal | Atten | Yield | Joint | Data | Data coverage |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for block, modes in summary["standardization"].items():
        for body in ("T10", "T11"):
            row = modes["exact"][body]
            s = row["samples"]
            vals = [s[a]["data_weighted"]["second_moment"] for a in (*J.ARMS, "data")]
            lines.append(f"| {block}/{body} | {row['common_cells']} | " +
                         " | ".join(f"{v:.3f}" if v is not None else "unsupported" for v in vals) +
                         f" | {s['data']['coverage']:.1%} |")
    lines += ["", "## All-body multiplicity-band support census", "",
              "| Block/body | Cells | Nominal / joint / data second moment [ns²] | Data coverage |",
              "|---|---:|---|---:|"]
    for block, modes in summary["standardization"].items():
        for body, row in modes["band"].items():
            s = row["samples"]
            vals = [s[a]["data_weighted"]["second_moment"] for a in ("nominal", "joint", "data")]
            coverage = s['data']['coverage']
            lines.append(f"| {block}/{body} | {row['common_cells']} | " +
                         " / ".join(f"{v:.3f}" if v is not None else "unsupported" for v in vals) +
                         f" | {coverage:.1%} |" if coverage is not None else f"| {block}/{body} | 0 | unsupported | n/a |")
    lines += ["", "## Same-bar/same-end distinct-SiPM pairs, both pulses 100–200 PE", "",
              "The line correction cancels. Pair populations are not position/path/multiplicity standardized.", "",
              "| Block/body/arm | Pairs | Event/slot groups | Mean difference² [ns²] | Group-weighted [ns²] |",
              "|---|---:|---:|---:|---:|"]
    for block, samples in summary["same_end_pe"].items():
        for name in ("nominal", "joint", "data"):
            for row in samples[name]:
                if row["body"] in (4, 5) and row["bin"] == 4:
                    lines.append(f"| {block}/T1{row['body']-4}/{name} | {row['pairs']} | {row['groups']} | "
                                 f"{row['difference_second']:.3f} | {row['group_weighted_difference_second']:.3f} |")
    lines += ["", "## Evidence limitations", "",
              "- Pair correlations include shared-mean anticorrelation; negative values alone do not identify electronics.",
              "- FEB labels project the frozen MC channel map onto data; physical P4 mapping is not certified here.",
              "- Photon arrivals, fit shape/quality and waveform/clock samples were not retained.",
              "- Data belong to the scoring reference; neither independent calibration holdout nor significance is claimed.", ""]
    return "\n".join(lines)


def plots(reports, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"nominal": "tab:blue", "atten": "tab:orange", "yield": "tab:green", "joint": "tab:red", "data": "black"}
    x = np.arange(5)
    labels = ["0–20", "20–40", "40–60", "60–100", "100–200"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, body in enumerate(("T10", "T11")):
        for block, report in reports.items():
            for arm in (*J.ARMS, "data"):
                style = "-" if block == "A" else "--"
                rows = report["standardization"]["band"][body]["by_pe"]
                y = [r["samples"][arm]["data_weighted"]["second_moment"] for r in rows]
                axes[0, col].plot(x, y, marker="o", linestyle=style, color=colors[arm],
                                  label=f"data ({block} support)" if arm == "data" else f"{arm} {block}")
                pairs = report["samples"][arm]["pairs"]["same_bar_same_end_distinct_sipm_equal_pe_bin"]
                y = [r["difference_second"] for r in pairs if r["body"] == col+4]
                axes[1, col].plot(x, y, marker="o", linestyle=style, color=colors[arm])
        axes[0, col].set_title(f"{body}: body/along/path/PE/N-band common support")
        axes[1, col].set_title(f"{body}: same bar + end, distinct SiPM; correction cancels")
        for row in range(2):
            axes[row, col].set_xticks(x, labels)
            axes[row, col].set_xlabel("measured PE bin (both pulses for pair panel)")
            axes[row, col].grid(alpha=.25)
            axes[row, col].set_ylim(bottom=0)
        axes[0, col].set_ylabel("selected-pulse residual second moment [ns²]")
        axes[1, col].set_ylabel("mean pair time-difference² [ns²]")
    axes[0, 1].legend(ncol=2, fontsize=8)
    fig.suptitle("Frozen calibration/background · descriptive only · pair panels not fully standardized", fontsize=11)
    fig.savefig(out/"t1_pe_timing.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(14, 11), constrained_layout=True)
    first = reports["A"]
    for body, ax in enumerate(axes.flat):
        for name in ("nominal", "joint", "data"):
            rows = first["samples"][name]["hardware"]["layer"]
            rows = [r for r in rows if r["key"][0] == body]
            ax.plot([r["key"][1] for r in rows], [r["mean"] for r in rows], "o-", color=colors[name], label=name)
        ax.set_title(first["axes"]["bodies"][body])
        ax.set_xticks(range(4))
        ax.axhline(0, color="grey", linewidth=.5)
        ax.grid(alpha=.2)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean residual [ns]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("All sectors/bodies, both sides retained · A · unstandardized hardware means, not calibration offsets")
    fig.savefig(out/"all_body_layer_means.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args(argv)
    out = Path(args.outdir)
    J.require(not out.exists(), "output directory already exists")
    receipts = {}
    reports = {b: J._json(path, receipts) for b, path in (("A", args.a), ("B", args.b))}
    J._receipt(__file__, receipts)
    summary = summarize(reports)
    summary["files"] = list(receipts.values())
    out.mkdir(parents=True)
    plots(reports, out)
    (out/"summary.md").write_text(text(summary))
    for record in receipts.values():
        J.require(V.digest(record["path"]) == record["sha256"], "report input/source changed")
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(out/"summary.md")


if __name__ == "__main__":
    main()
