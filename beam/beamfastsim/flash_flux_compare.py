"""Paired S1 primary-brems comparison, not a full-shower or detector closure test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flash_flux import _identity_rows

FIELDS = ("transparent_photons", "uncollided_photons", "transparent_energy_mev", "uncollided_energy_mev")
DESIGNS = ("bpzup21", "bpzup12")
COLORS = ("#0072B2", "#E69F00")
SCOPE = "Primary brems, uncollided first emitting-foil exit; no cascades/annihilation."


def paired(low, high, n):
    """Ratio of sums with paired residual SE; unrepresented events are known zeros."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    if (low.ndim != 1 or low.shape != high.shape or isinstance(n, bool) or
            not np.isfinite(n) or int(n) != n or n < max(1, len(low))):
        raise ValueError("Invalid paired arrays or source-event denominator")
    if not (np.isfinite(low).all() and np.isfinite(high).all()) or np.any(low < 0) or np.any(high < 0):
        raise ValueError("Totals must be finite and nonnegative")
    a, b = float(low.sum()), float(high.sum())
    ratio = b / a if a else None
    se = float(np.sqrt(n / (n - 1) * np.sum((high - ratio * low) ** 2)) / a) if a and n > 1 else None
    mean_se = lambda x: float(np.sqrt(max(0., np.sum(x*x) - x.sum()**2/n) / (n*(n-1)))) if n > 1 else None
    return dict(low_sum=a, high_sum=b, low_per_event=a/n, high_per_event=b/n,
                low_se=mean_se(low), high_se=mean_se(high), ratio=ratio, ratio_se=se,
                events=int(n), status="unmeasurable: zero low arm" if not a else "ok" if n > 1 else "SE unavailable: one event")


def read_arm(path, n):
    report = json.loads((path / "flux.json").read_text())
    if report.get("source_events") != n:
        raise ValueError("Both reports must declare the requested source_events denominator")
    if report.get("dropped_births", 0) or report.get("capped_parents", 0):
        raise ValueError("Truncated sources (dropped births or capped parents) cannot be compared")
    with np.load(path / "event_totals.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in ("event", *FIELDS)}
    event = data["event"]
    if event.ndim != 2 or event.shape[1] != 3 or event.dtype.kind not in "iu" or len(np.unique(event, axis=0)) != len(event):
        raise ValueError("Event identities must be unique integer (run, subrun, event) rows")
    data["event"] = event = _identity_rows(event.T)
    for key in FIELDS:
        if data[key].shape != (len(event),):
            raise ValueError("Event totals must match event identities")
        paired(data[key], data[key], n)
        if not np.isclose(data[key].sum(), report["totals"][key]):
            raise ValueError("Report totals disagree with event totals")
    return report, data


def event_histograms(path, event, edges, totals=None):
    """Recover event-level bin correlations from the photon-to-source parent map."""
    if not (path / "births.npz").exists() or not (path / "foil_exit.npz").exists():
        return None
    with np.load(path / "births.npz", allow_pickle=False) as z:
        source = _identity_rows([z["source_" + k] for k in ("run", "subrun", "event")])
    dtype = np.dtype([(name, "u8") for name in ("run", "subrun", "event")])
    keys = np.ascontiguousarray(_identity_rows(event.T)).view(dtype).reshape(-1)
    query = np.ascontiguousarray(source).view(dtype).reshape(-1)
    order = np.argsort(keys)
    index = np.searchsorted(keys[order], query)
    if np.any(index >= len(keys)) or (len(query) and np.any(keys[order[index]] != query)):
        raise ValueError("Birth source event absent from event universe")
    parent_event = order[index]
    with np.load(path / "foil_exit.npz", allow_pickle=False) as z:
        parent, energy, weight, foil = (z[k] for k in ("parent", "energy", "weight", "foil_exit"))
    if (parent.ndim != 1 or any(v.shape != parent.shape for v in (energy, weight, foil)) or
            parent.dtype.kind not in "iu" or np.any(parent < 0) or np.any(parent >= len(source)) or
            not np.isfinite(energy).all() or not np.isfinite(weight).all() or np.any(weight < 0)):
        raise ValueError("Invalid photon parent, energy or weight")
    if totals is not None:
        weights = (foil.astype(float), weight, foil*energy, weight*energy)
        for key, w in zip(FIELDS, weights):
            reconstructed = np.bincount(parent_event[parent], weights=w, minlength=len(event))
            if not np.allclose(reconstructed, totals[key]):
                raise ValueError("Photon parent mapping disagrees with event totals")
    bins = np.searchsorted(edges, energy, side="right") - 1
    bins[energy == edges[-1]] = len(edges) - 2
    use = (bins >= 0) & (bins < len(edges) - 1)
    hist = {}
    for kind, w in (("transparent", foil.astype(float)), ("uncollided", weight)):
        h = np.zeros((len(event), len(edges) - 1))
        np.add.at(h, (parent_event[parent[use]], bins[use]), w[use])
        hist[kind] = h
    return hist


def spectrum(low, high, low_total, high_total, n):
    """Bin ratio SE and shape pulls accounting for the fitted overall count ratio."""
    fit = paired(low_total, high_total, n)["ratio"]
    result = []
    for l, h in zip(low.T, high.T):
        item = paired(l, h, n)
        delta = pull = shape_se = None
        if fit is not None:
            residual = h - fit*l
            # Linearized fitted-normalization residual, including cross-bin covariance.
            influence = residual - l.sum()/low_total.sum() * (high_total - fit*low_total)
            delta = float(residual.sum()/n)
            if n > 1:
                shape_se = float(np.sqrt(max(0., np.sum(influence**2) - influence.sum()**2/n)/(n*(n-1))))
                if shape_se > 0:
                    pull = delta/shape_se
        result.append(dict(**item, shape_difference_per_event=delta, shape_se=shape_se, shape_pull=pull))
    return result


def shape_chi2(low, high, low_total, high_total, n):
    """Paired fitted-normalization shape statistic with event covariance."""
    fit = paired(low_total, high_total, n)["ratio"]
    if fit is None or n < 2:
        return dict(chi2=None, ndf=0, chi2_ndf=None)
    residual = high - fit * low
    influence = residual - (high_total - fit * low_total)[:, None] * (low.sum(axis=0) / low_total.sum())
    delta = residual.sum(axis=0) / n
    covariance = (influence.T @ influence - np.outer(influence.sum(axis=0), influence.sum(axis=0)) / n) / (n * (n - 1))
    eigen, vectors = np.linalg.eigh(covariance)
    use = eigen > max(0., eigen.max(initial=0.)) * 1e-10
    ndf = int(use.sum())
    chi2 = float(np.sum((delta @ vectors[:, use]) ** 2 / eigen[use])) if ndf else None
    return dict(chi2=chi2, ndf=ndf, chi2_ndf=chi2 / ndf if ndf else None,
                scope="Retained covariance eigenspace; fitted count ratio, no independent-bin assumption")


def compare(low, high, out, n):
    reports, data = zip(*(read_arm(p, n) for p in (low, high)))
    if not np.array_equal(data[0]["event"], data[1]["event"]):
        raise ValueError("Exact same event arrays required; missing events cannot be assumed zero")
    event = data[0]["event"]
    shards, group = np.unique(event[:, :2], axis=0, return_inverse=True)
    metrics, by_shard = {}, []
    for key in FIELDS:
        l, h = (d[key] for d in data)
        sl, sh = (np.bincount(group, weights=v, minlength=len(shards)) for v in (l, h))
        cluster = paired(sl, sh, max(1, len(shards)))
        metrics[key] = dict(**paired(l, h, n), shard_ratio_se=cluster["ratio_se"])
    for i, shard in enumerate(shards):
        use = group == i
        by_shard.append(dict(run_subrun=shard.tolist(), represented_events=int(use.sum()),
                             metrics={k: paired(data[0][k][use], data[1][k][use], int(use.sum())) for k in FIELDS}))
    edges = np.asarray(reports[0]["energy_edges_mev"], float)
    if (not np.array_equal(edges, reports[1]["energy_edges_mev"]) or len(edges) < 2 or
            not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or edges[0] <= 0):
        raise ValueError("Matching, positive, increasing energy bin edges required")
    hist = [event_histograms(p, event, edges, d) for p, d in zip((low, high), data)]
    spectra = {}
    for kind in ("transparent", "uncollided"):
        raw = [np.asarray(r["histograms"][kind], float) for r in reports]
        if any(v.shape != (len(edges)-1,) or not np.isfinite(v).all() or np.any(v < 0) for v in raw):
            raise ValueError("Invalid energy histograms")
        for matrix, counts in zip(hist, raw):
            if matrix is not None and not np.allclose(matrix[kind].sum(axis=0), counts):
                raise ValueError("Photon histograms disagree with report")
        if all(h is not None for h in hist):
            args = (hist[0][kind], hist[1][kind], data[0][kind+"_photons"], data[1][kind+"_photons"], n)
            bins, chi2 = spectrum(*args), shape_chi2(*args)
        else:
            bins = [dict(ratio=float(h/l) if l else None, ratio_se=None, shape_pull=None) for l, h in zip(*raw)]
            chi2 = dict(chi2=None, ndf=0, chi2_ndf=None)
        spectra[kind] = dict(low_per_event=(raw[0]/n).tolist(), high_per_event=(raw[1]/n).tolist(), bins=bins, shape_test=chi2,
                             uncertainty="paired event delta method" if all(h is not None for h in hist) else "shape uncertainty not evaluated: birth-parent mapping unavailable")
    result = dict(scope=SCOPE, designs=dict(low=DESIGNS[0], high=DESIGNS[1]), source_events=n,
                  represented_events=len(event), known_zero_input_events=n-len(event), shard_count=len(shards),
                  metrics=metrics, per_shard=by_shard, energy_edges_mev=edges.tolist(), spectra=spectra,
                  s0_deposit_ratio_reference=3.68145,
                  caveats=["Any loopers make this a path-truncated source diagnostic, not completed flux.",
                           "S0 ratio is detector deposit, NOT flux; not a paired board comparison or closure target.",
                           "Not an all-66 result or physics-validated prediction; all source times retained.",
                           "Same event identities required; excess declared source events are shared known zeros.",
                           "Shard SE uses ratio of sums and cluster residuals, not mean of shard ratios; few (e.g. four) shards are fragile.",
                           "Shard per-event means use represented events only; extra zero-input events have no shard allocation.",
                           "Shape pulls are diagnostics against the fitted observed count ratio, not independent-bin tests.",
                           "Spectra cover the declared energy bins only; count ratios use all energies."],
                  diagnostics=[{k: r.get(k) for k in ("dropped_births", "capped_parents", "loopers", "unsupported_attenuation_photons", "nonfoil_brems")} for r in reports])
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, out)
    return result


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fmt = lambda v: "n/a" if v is None else f"{v:.4g}"
    loopers = [d.get("loopers") or 0 for d in result["diagnostics"]]
    caveat = f"Path-truncated source: {loopers[0]}/{loopers[1]} loopers (low/high)." if any(loopers) else "No path-loopers."
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key in zip(axes.flat, FIELDS):
        m = result["metrics"][key]
        for i, (arm, color) in enumerate(zip(("low", "high"), COLORS)):
            ax.errorbar(i, m[arm+"_per_event"], yerr=m[arm+"_se"], fmt="o", color=color, capsize=4)
        ax.set(xticks=[0, 1], xticklabels=DESIGNS, xlim=(-.5, 1.5),
               ylabel="MeV / input event" if "energy" in key else "Photons / input event",
               title=key.replace("_", " ").replace(" mev", ""))
        ax.text(.03, .96, f"High/low = {fmt(m['ratio'])} ± {fmt(m['ratio_se'])}\nShard SE = {fmt(m['shard_ratio_se'])}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.margins(y=.3)
    fig.suptitle("Paired two-point primary-brems flux (event SE)")
    fig.text(.5, .02, SCOPE + "\nS0 deposit ratio = 3.68145: detector deposit, NOT flux or closure."
             "\n" + caveat + " Few-shard SE is fragile.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(out / "ratio_summary.png", dpi=160)
    plt.close(fig)
    edges = np.asarray(result["energy_edges_mev"])
    centers = np.sqrt(edges[:-1]*edges[1:])
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex="col", gridspec_kw={"height_ratios": [3, 2, 1]})
    for col, kind in enumerate(("transparent", "uncollided")):
        s = result["spectra"][kind]
        for arm, name, color in zip(("low", "high"), DESIGNS, COLORS):
            axes[0, col].stairs(s[arm+"_per_event"], edges, color=color, label=name)
        axes[0, col].set(title=kind.capitalize(), ylabel="Photons / input event / bin")
        axes[0, col].legend()
        value = lambda key: np.array([np.nan if b.get(key) is None else b[key] for b in s["bins"]])
        ratio, error = value("ratio"), value("ratio_se")
        axes[1, col].plot(centers, ratio, "o", color=COLORS[0], ms=3)
        valid = np.isfinite(error) & np.isfinite(ratio)
        axes[1, col].errorbar(centers[valid], ratio[valid], yerr=error[valid], fmt="none", color=COLORS[0])
        fit = result["metrics"][kind+"_photons"]["ratio"]
        if fit is not None:
            axes[1, col].axhline(fit, color=COLORS[1], ls="--", label="Observed count ratio (fit)")
            axes[1, col].legend(fontsize=8)
        axes[1, col].set(ylabel="High / low (paired SE)")
        axes[2, col].plot(centers, value("shape_pull"), "o", color=COLORS[0], ms=3)
        axes[2, col].axhline(0, color="0.5", lw=1)
        axes[2, col].set(xscale="log", xlabel="Photon energy [MeV]", ylabel="Shape pull")
        stat = s["shape_test"]
        label = s["uncertainty"] + f"; shape chi2/ndf = {fmt(stat['chi2'])}/{stat['ndf']}"
        axes[0, col].text(.02, .98, label, va="top", transform=axes[0, col].transAxes, fontsize=7, wrap=True)
    fig.suptitle("Primary-brems energy spectrum; fitted-normalization shape diagnostics")
    fig.text(.5, .02, SCOPE + "\n" + caveat + " No detector-response closure.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(out / "energy_spectrum.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("low", "high", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-events", type=int, required=True)
    args = parser.parse_args(argv)
    compare(args.low, args.high, args.out, args.source_events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
