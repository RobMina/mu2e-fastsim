"""Plot a measured paired occupancy report without concealing selection or coverage."""
import argparse
import json
from pathlib import Path

import numpy as np


def plot(report, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = report["paired"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    for column, (key, xlabel) in enumerate((("planes", "Tracker plane"), ("sectors", "CRV sector"))):
        data = paired[key]
        x = np.arange(len(data["stock"]))
        axes[0, column].step(x, data["stock"], where="mid", color="#0072B2", label="Stock saved digis + reco")
        axes[0, column].step(x, data["fast"], where="mid", color="#D55E00", label="Four-stream overlay + response")
        ratio = np.asarray(data["ratio"], dtype=float)
        error = np.asarray(data["ratio_error"], dtype=float)
        axes[1, column].errorbar(x, ratio, yerr=error, fmt="o", ms=3, color="#D55E00", capsize=2)
        axes[1, column].axhline(1, color="0.4", lw=1)
        axes[2, column].axhline(0, color="0.4", lw=1)
        axes[2, column].axhspan(-2, 2, color="0.92")
        axes[2, column].plot(x, np.asarray(data["pull"], dtype=float), "o", ms=3, color="#009E73")
        axes[0, column].set_ylabel("Straw digis / event" if key == "planes" else "CRV pulses / event")
        axes[1, column].set_ylabel("Overlay / stock")
        axes[2, column].set_ylabel("Paired pull")
        axes[2, column].set_xlabel(xlabel)
        chi2 = data["chi2_diagonal"]
        text = "undefined" if chi2 is None else f"{chi2:.1f}"
        axes[0, column].set_title(f"Diagonal chi-square / ndf = {text} / {data['ndf_diagonal']}", fontsize=11)
        for row in range(3):
            axes[row, column].grid(alpha=0.2)
        if key == "sectors":
            axes[2, column].set_xticks(x, report["sector_names"])
            axes[2, column].set_xlim(-0.5, len(x) - 0.5)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"{title}\n{paired['matched_events']:,} paired primary events", fontsize=15)
    fig.text(0.5, 0.015, "Conditioned on stock-selected event IDs; independent pileup draws. "
             "CRV T1/T2 only: no side-sector closure.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Stock-frame pileup: selected-CE diagnostic")
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(json.loads(args.report.read_text()), args.out, args.title)


if __name__ == "__main__":
    main()
