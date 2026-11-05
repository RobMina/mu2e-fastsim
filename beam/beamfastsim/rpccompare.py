"""Compare generated internal-RPC pairs with saved stock RPCGun histograms.

This is a source-kinematics comparison, not a transport or event-yield test.
The two samples have independent kinematic draws. Flow bins enter the shape
statistic; plots show ordinary bins and their independent-draw pulls.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .rpcspec import ME

HISTOGRAMS = {
    "photon_energy": ("hmomentum", r"Photon energy [MeV]"),
    "p": ("hElecMom", r"Electron momentum [MeV/c]"),
    "positron_p": ("hPosiMom", r"Positron momentum [MeV/c]"),
    "pair_mass": ("hMee", r"Pair invariant mass [MeV/c$^2$]"),
}


def shape_comparison(stock, fast):
    """Two-sample Pearson shape test; adjacent sparse bins pooled for chi2."""
    stock, fast = np.asarray(stock, float), np.asarray(fast, float)
    if (stock.ndim != 1 or stock.shape != fast.shape or not np.isfinite(stock).all()
            or not np.isfinite(fast).all() or np.any(stock < 0) or np.any(fast < 0)
            or stock.sum() <= 0 or fast.sum() <= 0):
        raise ValueError("need nonnegative, nonempty same-sized histogram counts")
    scale = stock.sum() / fast.sum()
    variance = scale * (stock + fast)
    pulls = np.divide(scale * fast - stock, np.sqrt(variance),
                      out=np.zeros_like(stock), where=variance > 0)
    expected = stock + scale * fast
    cumulative = np.cumsum(expected)
    cuts = np.searchsorted(cumulative, np.arange(50., cumulative[-1], 50.)) + 1
    bounds = np.unique(np.r_[0, cuts, stock.size]).astype(int)
    s, f = np.add.reduceat(stock, bounds[:-1]), np.add.reduceat(fast, bounds[:-1])
    if s.size > 1 and s[-1] + scale * f[-1] < 50:
        s[-2] += s[-1]
        f[-2] += f[-1]
        s, f = s[:-1], f[:-1]
    v = scale * (s + f)
    good = v > 0
    chi2 = float(np.sum((scale * f[good] - s[good]) ** 2 / v[good]))
    ndf = max(int(good.sum()) - 1, 0)
    return dict(stock_entries=int(stock.sum()), fast_entries=int(fast.sum()), scale=float(scale),
                chi2=chi2, ndf=ndf, chi2_ndf=chi2 / ndf if ndf else None,
                pulls=pulls.tolist(), flow_stock=stock[[0, -1]].tolist(), flow_fast=fast[[0, -1]].tolist())


def compare(stock_path, source_path, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.4), sharex="col", height_ratios=[3, 1])
    reports = {}
    with np.load(source_path, allow_pickle=False) as source, uproot.open(stock_path) as root:
        meta = json.loads(str(source["metadata"])) if "metadata" in source else {}
        n = len(source["p"])
        ee, ep = np.hypot(source["p"], ME), np.hypot(source["positron_p"], ME)
        pair_p = source["p"][:, None] * source["u"] + source["positron_p"][:, None] * source["positron_u"]
        values = dict(photon_energy=ee + ep, p=source["p"], positron_p=source["positron_p"],
                      pair_mass=np.sqrt(np.maximum(0, (ee + ep) ** 2 - np.sum(pair_p ** 2, axis=1))))
        for col, (key, (histname, label)) in enumerate(HISTOGRAMS.items()):
            hist = root["generate/RPCGun/" + histname]
            edges, stock = hist.axis().edges(), hist.values(flow=True)
            fast = np.bincount(np.searchsorted(edges, values[key], side="right"), minlength=edges.size + 1)
            r = shape_comparison(stock, fast)
            reports[key] = r
            np.savez_compressed(outdir / (key + ".npz"), edges=edges, stock=stock, fast=fast)
            axes[0, col].stairs(stock[1:-1], edges, color="#0072B2", label="Stock RPCGun")
            axes[0, col].stairs(r["scale"] * fast[1:-1], edges, color="#D55E00", label="Fast (equal area)")
            axes[0, col].set_title(r"$\chi^2$/ndf = " + f"{r['chi2']:.1f}/{r['ndf']}")
            axes[0, col].set_ylabel("Captures / bin")
            centres = (edges[:-1] + edges[1:]) / 2
            axes[1, col].plot(centres, r["pulls"][1:-1], ".", color="#D55E00", markersize=3)
            axes[1, col].axhline(0, color="black", linewidth=.7)
            axes[1, col].set_ylabel("Pull")
            axes[1, col].set_xlabel(label)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Internal RPC source only: {reports['p']['stock_entries']:,} stock / {n:,} fast captures")
        fig.tight_layout()
        fig.savefig(outdir / "rpc_source.png", dpi=170)
        plt.close(fig)
    report = dict(scope="internal RPC source kinematics only; no VD13 transport or yield result",
                  events=n, source=meta, shapes=reports,
                  error_method="independent kinematic draws; two-sample Pearson statistic using pooled null variance, one normalization dof removed",
                  shape_bins="adjacent bins pooled to combined expected count >=50, including flows; pulls unpooled")
    (outdir / "numbers.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stock", required=True, help="saved stock RPCGun TFileService ROOT file")
    ap.add_argument("--source", required=True, help="rpcspec output NPZ")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    report = compare(a.stock, a.source, a.outdir)
    for key, r in report["shapes"].items():
        print(f"{key}: chi2/ndf={r['chi2']:.3f}/{r['ndf']}")


if __name__ == "__main__":
    main()
