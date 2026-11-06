"""Plot saved reference diagnostics without transport, fitted scales or board claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MISSING_SE_NOTE = "Unavailable event SE omitted (e.g. n=1); missing bars do not imply zero uncertainty."


def components(origin, comparison, arm):
    """Return eV/input-event components and raw mean SE, or None if unavailable."""
    n = comparison["events"]
    if n <= 0 or origin["input_events"] != n:
        raise ValueError("origins and comparison denominators disagree")
    table = origin["foil_ancestry"]
    status = np.asarray(table["status"])
    foil = np.asarray(table["foil_born_observed"], bool)
    complete = np.asarray(table["ancestry_complete"], bool)
    energy = np.asarray(table["ionizing_mev"], float)
    if not np.all(complete) or np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("component plot requires complete finite ancestry")
    masks = (status == 1, (status == 0) & foil, (status == 0) & ~foil, status == 2)
    sums = np.asarray([energy[use].sum() for use in masks])
    totals = comparison["totals"]
    expected = [totals[k][arm]["sum_mev"] for k in ("owned", "residual", "unknown")]
    if not np.allclose([sums[0], sums[1:3].sum(), sums[3]], expected, rtol=1e-12, atol=1e-14):
        raise ValueError("ancestry components disagree with comparison")
    if not np.isclose(sums.sum(), totals["raw"][arm]["sum_mev"], rtol=1e-12, atol=1e-14):
        raise ValueError("component sum disagrees with raw energy")
    error = totals["raw"][arm]["event_mean_se"]
    return sums * 1e6 / n, None if error is None else error * 1e6


def plot_spectra(low, high, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    missing_se = False
    for col, (name, report) in enumerate((("Low anchor", low), ("High anchor", high))):
        edges = np.asarray(report["energy_bin_lower_edges_mev"])
        n = report["input_events"]
        if n < 1 or not np.array_equal(edges, [0., .001, .01, .1, 1., 3., 10., 30., 100.]):
            raise ValueError("unsupported spectrum denominator/bin convention")
        labels = ["<.001", ".001–.01", ".01–.1", ".1–1", "1–3", "3–10", "10–30", "30–100", "≥100"]
        x = np.arange(len(edges))
        for row, metric in enumerate(("photons", "energy_mev")):
            ax, data = axes[row, col], report["spectra"][metric]
            for model, color, offset in (("fast", "#0072B2", -.09), ("g4", "#D55E00", .09)):
                error = np.asarray(data[model + "_total_event_se"], dtype=float) / n
                missing_se |= np.isnan(error).any()
                ax.errorbar(x + offset, np.asarray(data[model + "_total"]) / n,
                    yerr=None if np.isnan(error).all() else error, fmt="o", ms=4, capsize=3,
                    color=color, label="Fast weighted" if model == "fast" else "G4 realized")
            ax.set_xticks(x, labels, rotation=40, ha="right")
            ax.grid(alpha=.2)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Photons / input event" if row == 0 else "Photon energy sum [MeV / input event]")
            if row == 0:
                ax.set_title(f"{name} — {n:,} matched DS events")
                ax.legend(fontsize=9)
            else:
                ax.set_xlabel("Photon energy bin [MeV]")
    fig.suptitle("Primary brems: uncollided first-foil exit\nAbsolute spectra; event sampling errors; no fitted normalization", fontsize=13)
    caveat = "Development samples; independent RNG; planar charged foils, no shower closure. Loopers/unknowns retained as caveats."
    fig.supxlabel(caveat + ("\n" + MISSING_SE_NOTE if missing_se else ""), fontsize=9)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_components(low, high, comparison, out):
    import matplotlib.pyplot as plt

    values, error = zip(*(components(origin, comparison, arm) for origin, arm in ((low, "low"), (high, "high"))))
    values = np.stack(values)
    error = np.asarray(error, dtype=float)
    available = np.isfinite(error)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    labels = ("Foil-photon attributed", "Residual with foil-born ancestry", "Residual without foil-born ancestry", "Unresolved attribution")
    colors = ("#0072B2", "#D55E00", "#999999", "#CC79A7")
    bottom = np.zeros(2)
    for index, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(np.arange(2), values[:, index], bottom=bottom, width=.55, label=label, color=color)
        bottom += values[:, index]
    if available.any():
        ax.errorbar(np.arange(2)[available], bottom[available], yerr=error[available], fmt="none",
                    color="black", capsize=6, label="Raw total: event sampling SE")
    ax.set_xticks([0, 1], ["Low anchor", "High anchor"])
    ax.set_ylabel("Raw tracker ionization [eV / TS3 input event]")
    upper = np.r_[bottom, bottom[available] + error[available]].max()
    ax.set_ylim(0, upper * 1.25 if upper > 0 else 1.)
    ax.grid(axis="y", alpha=.2)
    ax.set_title(f"Inclusive reference: {comparison['events']:,} events per design\nAncestry components, not a fitted fast prediction")
    ax.legend(loc="upper left", fontsize=9)
    caveat = ("Residual is not necessarily upstream background. Rare showers dominate residual uncertainty.\n"
              "No resampling, detector aggregation/selection or POT normalization; not board-budget closure.")
    fig.supxlabel(caveat + ("\n" + MISSING_SE_NOTE if not available.all() else ""), fontsize=9)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("spectrum-low", "spectrum-high", "origins-low", "origins-high", "comparison", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args(argv)
    reports = {name: json.loads(getattr(a, name).read_text()) for name in
               ("spectrum_low", "spectrum_high", "origins_low", "origins_high", "comparison")}
    a.out.mkdir(parents=True, exist_ok=False)
    plot_spectra(reports["spectrum_low"], reports["spectrum_high"], a.out / "flash_reference_spectra.png")
    plot_components(reports["origins_low"], reports["origins_high"], reports["comparison"], a.out / "flash_reference_components.png")
    provenance = {name: str(getattr(a, name)) for name in reports}
    (a.out / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
