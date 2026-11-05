"""Paired C4 comparisons on a fixed cached stock track selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from . import VetoCard, vetoed
from .apply import gate
from .metrics import fraction

KEY = ("run", "subrun", "event")


def align(reference, candidate):
    """Reorder complete candidate event rows by identity, including empty responses."""
    left = pd.DataFrame({k: ak.to_numpy(reference[k]) for k in KEY})
    right = pd.DataFrame({k: ak.to_numpy(candidate[k]) for k in KEY})
    right["row"] = np.arange(len(right))
    joined = left.merge(right, on=list(KEY), how="left", validate="one_to_one", sort=False)
    if len(left) != len(right) or joined.row.isna().any():
        raise ValueError("fast response must contain every cached event, including empty responses")
    return candidate[joined.row.to_numpy(np.int64)]


def paired_rate(stock, fast, groups):
    """Count ratios and differences with paired cluster-robust errors by input shard."""
    stock, fast = np.asarray(stock, float), np.asarray(fast, float)
    groups = np.asarray(groups)
    if stock.shape != fast.shape or stock.ndim != 1 or groups.shape != stock.shape:
        raise ValueError("stock, fast and shard arrays must be aligned one-dimensional arrays")
    _, inverse = np.unique(groups, return_inverse=True)
    n = np.bincount(inverse).astype(float)
    a = np.bincount(inverse, weights=fast)
    b = np.bincount(inverse, weights=stock)
    size, shards = len(stock), len(n)
    ratio = float(a.sum() / b.sum()) if b.sum() else None
    difference = float((a.sum() - b.sum()) / size) if size else None
    ratio_se = difference_se = None
    if shards > 1:
        factor = shards / (shards - 1)
        if ratio is not None:
            ratio_se = float(np.sqrt(factor * np.sum((a - ratio * b) ** 2)) / b.sum())
        difference_se = float(np.sqrt(factor * np.sum((a - b - difference * n) ** 2)) / size)
    return dict(stock=int(b.sum()), fast=int(a.sum()), n=size, shards=shards,
                ratio=ratio, ratio_se=ratio_se, difference=difference, difference_se=difference_se)


def compare(selection, card, fast=None, pmin=85., pmax=200., sectors=(), edges=None):
    """Keep the stock-selected denominator fixed; report fast event-gate migration separately."""
    p, t = selection.p_mid, selection.t_mid
    if not ak.all(ak.num(p) == ak.num(t)):
        raise ValueError("cached track momentum and time are not aligned")
    stock_v, stock_s = vetoed(t, selection.coincs, card, return_sector=True)
    stock_gate = gate(selection.coincs, card)
    use = (p > pmin) & (p < pmax) & stock_gate
    flat = lambda x: ak.to_numpy(ak.flatten(x[use]))
    v, sector, momentum = flat(stock_v), flat(stock_s), flat(p)
    event_groups = (ak.to_numpy(selection.shard) if "shard" in selection.fields else
                    pd.MultiIndex.from_arrays([ak.to_numpy(selection[k]) for k in ("run", "subrun")]).factorize()[0])
    groups = flat(ak.broadcast_arrays(ak.Array(event_groups)[:, None], p)[0])
    result = dict(events=len(selection), tracks=int(ak.sum(ak.num(p))), selected=len(v),
                  vetoed=fraction(v), inefficiency=fraction(~v),
                  gate_policy="fixed stock-selected denominator; fast gate failures reported separately",
                  uncertainty="paired cluster-robust delta method",
                  shard_definition="shard column" if "shard" in selection.fields else "run/subrun",
                  card=asdict(card))
    candidate = align(selection, fast) if fast is not None else None
    if candidate is not None:
        fv, fs = vetoed(t, candidate.coincs, card, return_sector=True)
        fv, fs = flat(fv), flat(fs)
        fg = flat(ak.broadcast_arrays(ak.Array(gate(candidate.coincs, card))[:, None], p)[0])
        result["fast"] = dict(vetoed=fraction(fv), inefficiency=fraction(~fv), gate_failed_tracks=int((~fg).sum()))
        result["paired_vetoed"] = paired_rate(v, fv, groups)
        result["paired_inefficiency"] = paired_rate(~v, ~fv, groups)
        result["discordant"] = dict(stock_only=int((v & ~fv).sum()), fast_only=int((fv & ~v).sum()))
    sector_values = set(map(int, sectors)) | set(map(int, ak.to_numpy(ak.flatten(selection.coincs.sectorType))))
    if candidate is not None:
        sector_values |= set(map(int, ak.to_numpy(ak.flatten(candidate.coincs.sectorType))))
    result["by_sector"] = {}
    for s in sorted(sector_values):
        row = dict(stock=int((sector == s).sum()), n=len(v))
        if candidate is not None:
            row = paired_rate(sector == s, fs == s, groups)
        result["by_sector"][str(s)] = row
    result["sector_definition"] = "exclusive highest-PE vetoing coincidence attribution, not a sector-conditioned inefficiency"
    edges = np.asarray(edges if edges is not None else np.linspace(pmin, pmax, 24), float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        raise ValueError("momentum edges must be finite and increasing")
    bins = np.searchsorted(edges, momentum, side="right") - 1
    rows = []
    for i in range(len(edges) - 1):
        mask = bins == i
        row = dict(stock=int((mask & ~v).sum()), n=int(mask.sum()))
        if candidate is not None:
            row = paired_rate((~v)[mask], (~fv)[mask], groups[mask])
            se = row["difference_se"]
            row["pull"] = row["difference"] / se if se is not None and se > 0 else None
        rows.append(row)
    result["momentum"] = dict(edges=edges.tolist(), bins=rows, quantity="unvetoed tracks per stock-selected track")
    if candidate is not None:
        pulls = [r["pull"] for r in rows if r["pull"] is not None]
        degenerate = sum(r["n"] > 0 and r["difference_se"] == 0 and r["difference"] != 0 for r in rows)
        result["momentum"]["chi2"] = float(np.sum(np.square(pulls))) if pulls and not degenerate else None
        result["momentum"]["ndf"] = len(pulls)
        result["momentum"]["zero_variance_discrepant_bins"] = degenerate
        result["momentum"]["chi2_definition"] = "sum of squared paired per-bin pulls; null if unestimable; bin-to-bin covariance not included"
    return result


def plot(result, path, label):
    """Save a stock/fast inefficiency overlay and paired pull panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.asarray(result["momentum"]["edges"])
    rows = result["momentum"]["bins"]
    n = np.asarray([r["n"] for r in rows])
    centers = (edges[:-1] + edges[1:]) / 2
    fig, (ax, lower) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for key, color, name in (("stock", "#0072B2", "Stock"), ("fast", "#D55E00", label)):
        if key == "fast" and "fast" not in result:
            continue
        counts = np.asarray([r[key] for r in rows])
        values = np.divide(counts, n, out=np.full(len(n), np.nan), where=n > 0)
        from .metrics import jeffreys
        intervals = np.asarray([jeffreys(int(k), int(total)) for k, total in zip(counts, n)])
        errors = np.maximum(np.vstack((values - intervals[:, 0], intervals[:, 1] - values)), 0.)
        ax.errorbar(centers, values, yerr=errors, fmt="o-", markersize=3, color=color, label=name)
    ax.set(ylabel="CRV veto inefficiency", ylim=(-0.02, 1.02),
           title=f"Run 1A cached selection: {result['selected']:,} tracks")
    if "paired_inefficiency" in result:
        rate = result["paired_inefficiency"]
        text = f"Fast/stock = {rate['ratio']:.4f}" if rate["ratio"] is not None else "Fast/stock undefined"
        if rate["ratio_se"] is not None:
            text += f" ± {rate['ratio_se']:.4f} (paired)"
        ax.text(.02, .05, text, transform=ax.transAxes)
    ax.legend()
    if "fast" in result:
        lower.plot(centers, [np.nan if r["pull"] is None else r["pull"] for r in rows], "o", color="#D55E00")
        same = [i for i, r in enumerate(rows) if r["n"] > 0 and r["difference"] == 0 and r["difference_se"] == 0]
        if same:
            lower.plot(centers[same], np.zeros(len(same)), "x", color="0.4", label="identical decisions; pull undefined")
            lower.legend(fontsize=7, loc="upper right")
    else:
        lower.text(.5, .5, "Fast response pending — stock replay only", transform=lower.transAxes, ha="center")
    lower.axhline(0, color="0.4", lw=.8)
    lower.set(xlabel="Selected track momentum [MeV/c]", ylabel="Paired pull")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", nargs="+")
    parser.add_argument("--card", required=True, help="VetoCard JSON or saved veto_stats.json")
    parser.add_argument("--fast", nargs="+", help="parquet with run/subrun/event and coincs, including empty events")
    parser.add_argument("--label", default="Fast")
    parser.add_argument("--sectors", nargs="*", type=int, default=[])
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    saved = json.loads(Path(args.card).read_text())
    values = dict(saved.get("card", saved))
    values.pop("sha", None)
    selection = ak.concatenate([ak.from_parquet(path) for path in args.selection])
    fast = ak.concatenate([ak.from_parquet(path) for path in args.fast]) if args.fast else None
    result = compare(selection, VetoCard.from_dict(values), fast, sectors=args.sectors)
    result["fast_label"] = args.label if args.fast else None
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, out / "crv_veto.png", args.label)
    print(json.dumps({k: result[k] for k in ("selected", "vetoed", "inefficiency")}))


if __name__ == "__main__":
    main()
