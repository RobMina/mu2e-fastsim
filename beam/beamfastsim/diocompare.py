"""Fast DIO electrons (diospec starts through cechain) against the stock DIO dts at the tracker entry plane.

Arms are paired per shard: shard s is one stock file and the fast electrons drawn from the same stops events
(`diospec starts --ref`), so both arms have the same generated electrons n_s. The stock events passed the production
PrimaryFilter, so yields are compared inside the filter-implying selection of cecompare (momentum at the plane above
P_MIN, helix maximum radius above an RMAX cut). Per shard: k_s (stock) and k'_s (fast) selected electrons.

- yield per generated electron in each arm and R = sum k' / sum k, with the error from the per-shard residuals
  k'_s - R k_s (shards are the independent units);
- shapes at the plane (momentum, cos theta, radius, time), normalised per generated electron: per-bin pulls and the
  two-sample chi2 / ndf on fixed bins (bins with at least MIN_BIN entries in the sum).

Development shards (`--dev-shards`, the lowest shard ids) and held-out shards are reported separately and combined.
Every generated start must have one final row across the supplied chunks. Empty selected arms retain their exposure;
unavailable statistics are null, including the ratio error for fewer than two shards. Histogram flows are counted
separately from the windowed chi2. The fast selection uses a z-plane, not the finite stock VD13 disk acceptance.
"""
import argparse
import glob
import json

import numpy as np

from mu2efastsim.eslab_g4 import chi2, summary

from . import cechain as C
from . import cecompare as CC

DIO_CODE = 166
BINS = {"p": np.arange(80.0, 105.01, 0.25), "cos": np.linspace(0.4, 0.9, 51), "r": np.arange(0.0, 650.1, 25.0),
        "t": np.arange(0.0, 2500.1, 50.0)}
MIN_BIN = 10


def stock_plane(ref):
    r = np.load(ref)
    pv = np.stack([r["pxvd_front"], r["pyvd_front"], r["pzvd_front"]], axis=1).astype(float)
    p = np.linalg.norm(pv, axis=1)
    return dict(x=np.stack([r["xvd_front"], r["yvd_front"], r["zvd_front"]], axis=1).astype(float),
                u=pv / np.maximum(p, 1e-12)[:, None], p=p, t=r["tvd_front"].astype(float), t0=r["time"].astype(float),
                ok=r["isvd_front"].astype(bool), shard=r["ifile"].astype(int))


def fast_plane(paths, starts):
    if not paths:
        raise ValueError("no fast output chunks: expected every generated start exactly once")
    f = CC.fast_plane(paths)
    with np.load(starts) as s:
        shard, ids, counts = s["shard"], s["shard_id"], s["shard_generated"]
        actual_ids, actual_counts = np.unique(shard, return_counts=True)
        if (shard.ndim != 1 or shard.size != s["x"].size or ids.ndim != 1 or counts.shape != ids.shape
                or np.unique(ids).size != ids.size or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 0) or counts.sum() != shard.size):
            raise ValueError("starts shard_generated counts must match the start rows for each unique shard_id")
        order = np.argsort(ids)
        positive = order[counts[order] > 0]
        if not (np.array_equal(ids[positive], actual_ids) and np.array_equal(counts[positive], actual_counts)):
            raise ValueError("starts shard_generated counts must match the start rows for each unique shard_id")
        idx = f["idx"]
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("fast output start indices must be a one-dimensional integer array")
        if np.any((idx < 0) | (idx >= shard.size)):
            raise ValueError(f"fast output start index out of range [0, {shard.size})")
        if np.unique(idx).size != idx.size:
            raise ValueError("duplicate fast output start indices across chunks")
        if idx.size != shard.size:
            raise ValueError(f"missing fast output start indices: expected {shard.size}, found {idx.size}")
        if any(f[k].shape[0] != idx.size for k in ("fate", "x", "u", "T", "t", "t0")):
            raise ValueError("fast output row counts must match final start indices")
        f["shard"] = shard[idx]
        return f, dict(zip(ids.tolist(), counts.tolist()))


def ratio_by_shard(k_stock, k_fast, n):
    """Per-generated yields of both arms and R = sum k_fast / sum k_stock with the shard-residual error."""
    k_stock, k_fast, n = (np.asarray(v, float) for v in (k_stock, k_fast, n))
    S, F, N = k_stock.sum(), k_fast.sum(), n.sum()
    R = F / S if S > 0 and N > 0 else None
    m = k_stock.size
    err = np.sqrt(m / (m - 1) * ((k_fast - R * k_stock) ** 2).sum()) / S if R is not None and m > 1 else None
    return dict(shards=int(m), generated=int(N), stock=int(S), fast=int(F),
                y_stock=S / N if N > 0 else None, y_fast=F / N if N > 0 else None,
                y_stock_err=np.sqrt(S) / N if N > 0 else None, y_fast_err=np.sqrt(F) / N if N > 0 else None,
                ratio=R, ratio_err=err)


def shapes(so, fo, n_gen):
    """Per-generated histograms with Poisson errors, windowed chi2 and separate flow counts."""
    out = {}
    for k, e in BINS.items():
        a, b = so[k], fo[k]
        hs, hf = np.histogram(a, e)[0].astype(float), np.histogram(b, e)[0].astype(float)
        es, ef = np.sqrt(hs), np.sqrt(hf)
        den = np.sqrt(es ** 2 + ef ** 2)
        pull = np.where(den > 0, (hf - hs) / np.where(den > 0, den, 1.0), 0.0)   # same n_gen in both arms
        m = (hs + hf) >= MIN_BIN
        c2 = float((pull[m] ** 2).sum()) if n_gen > 0 and a.size and b.size and m.any() else None
        sc2, snd = chi2(a, b, e) if hs.sum() > 0 and hf.sum() > 0 else (None, 0)
        norm = lambda h: (h / n_gen).tolist() if n_gen > 0 else [None] * h.size
        flows = lambda v: dict(under=int((v < e[0]).sum()), over=int((v > e[-1]).sum()))
        out[k] = dict(edges=e.tolist(), stock=norm(hs), fast=norm(hf), stock_err=norm(es), fast_err=norm(ef),
                      pull=pull.tolist(), chi2=c2, ndf=int(m.sum()), chi2_ndf=c2 / int(m.sum()) if c2 is not None else None,
                      shape_chi2=sc2, shape_ndf=max(snd - 1, 0),
                      shape_chi2_ndf=sc2 / (snd - 1) if snd > 1 else None,
                      stock_summary=summary(a), fast_summary=summary(b), stock_flow=flows(a), fast_flow=flows(b))
    return out


def compare(ref, fast_glob, starts, geom, field, dev_shards):
    s = stock_plane(ref)
    f, gen = fast_plane(sorted(glob.glob(fast_glob)), starts)
    so_all, fo_all = CC.observables(geom, field, s), CC.observables(geom, field, f)
    ss, fs = s["shard"][s["ok"]], f["shard"][f["ok"]]
    ids = np.array(sorted(i for i in gen if gen[i] > 0), dtype=int)
    groups = dict(dev=ids[:dev_shards], heldout=ids[dev_shards:], all=ids)
    out = dict(stock_rows=int(s["ok"].size), stock_at_plane=int(s["ok"].sum()), fast_rows=int(f["fate"].size),
               fast_fates={n: int((f["fate"] == k).sum()) for k, n in enumerate(C.FATES)},
               fast_wall_s=f["wall"], fast_us_per_e=1e6 * f["wall"] / f["fate"].size if f["fate"].size else None,
               generated=int(sum(gen[i] for i in ids)), dev_shards=int(min(dev_shards, ids.size)), cuts={})
    for rc in CC.RMAX_CUTS:
        sel_s = (so_all["p"] > CC.P_MIN) & (so_all["rmax"] > rc)
        sel_f = (fo_all["p"] > CC.P_MIN) & (fo_all["rmax"] > rc)
        c = {}
        for g, members in groups.items():
            ks = np.array([(sel_s & (ss == i)).sum() for i in members])
            kf = np.array([(sel_f & (fs == i)).sum() for i in members])
            c[g] = dict(yield_=ratio_by_shard(ks, kf, [gen[i] for i in members]))
            if rc == CC.RMAX_CUTS[0]:
                ms, mf = sel_s & np.isin(ss, members), sel_f & np.isin(fs, members)
                c[g]["shapes"] = shapes({k: v[ms] for k, v in so_all.items()}, {k: v[mf] for k, v in fo_all.items()},
                                        sum(gen[i] for i in members))
        out["cuts"][str(rc)] = c
    return out


def plot(rep, path, group="heldout", cut="450.0"):
    """Fast vs stock per generated electron at the plane (p, cos theta, r) with pull panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sh = rep["cuts"][cut][group]["shapes"]
    y = rep["cuts"][cut][group]["yield_"]
    keys = (("p", "momentum at TT_FrontHollow [MeV/c]"), ("cos", r"cos $\theta$ at TT_FrontHollow"),
            ("r", "radius at TT_FrontHollow [mm]"))
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.2), sharex="col", gridspec_kw=dict(height_ratios=(3, 1), hspace=0.05, wspace=0.18))
    for j, (k, lab) in enumerate(keys):
        d = sh[k]
        e = np.array(d["edges"])
        c = 0.5 * (e[1:] + e[:-1])
        ax[0, j].errorbar(c, np.asarray(d["stock"], float), yerr=np.asarray(d["stock_err"], float),
                          fmt="o", ms=3, color="black", label="Geant4 (stock dts)")
        ax[0, j].stairs(np.asarray(d["fast"], float), e, color="#E69F00", lw=1.6, label="fast (cechain)")
        if j == 0:
            ax[0, j].set_ylabel("per generated electron / bin")
            ax[1, j].set_ylabel("pull")
            positive = any(np.any(np.asarray(d[q], float) > 0) for q in ("stock", "fast"))
            ax[0, j].set_yscale("log" if k == "p" and positive else "linear")
        stat = f"{d['chi2']:.1f}/{d['ndf']}" if d["chi2"] is not None else "unavailable"
        ax[0, j].text(0.03, 0.95, rf"window $\chi^2$/ndf = {stat}", transform=ax[0, j].transAxes,
                      va="top", fontsize=9)
        ax[1, j].bar(c, d["pull"], width=e[1:] - e[:-1], color="#0072B2")
        ax[1, j].axhline(0, color="black", lw=0.8)
        ax[1, j].set_ylim(-4, 4)
        ax[1, j].set_xlabel(lab)
    ax[0, 0].legend(fontsize=9, loc="lower left")
    fmt = lambda v: f"{v:.4f}" if v is not None else "unavailable"
    fig.suptitle(f"DIO electrons at the tracker entry: yield fast / Geant4 = {fmt(y['ratio'])} +- {fmt(y['ratio_err'])}"
                 f" ({y['generated']:,} generated per arm)", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def numbers(rep, group="heldout"):
    """Headline numbers: per-group yields and ratio for every r_max cut, shape chi2 and mean shifts at the first cut."""
    out = dict(generated_per_arm=rep["generated"], dev_shards=rep["dev_shards"],
               fast_us_per_e=rep["fast_us_per_e"], selection=f"p > {CC.P_MIN} MeV/c at the plane, helix r_max > cut")
    for rc, c in rep["cuts"].items():
        out[f"rmax_{float(rc):.0f}"] = {g: {k: c[g]["yield_"][k] for k in ("generated", "stock", "fast", "y_stock",
                                                                            "y_fast", "ratio", "ratio_err")}
                                       for g in ("dev", "heldout", "all")}
    sh = rep["cuts"][str(CC.RMAX_CUTS[0])][group]["shapes"]
    out["shapes_" + group] = {}
    for k, v in sh.items():
        s, f = v["stock_summary"], v["fast_summary"]
        available = s["n"] > 0 and f["n"] > 0
        out["shapes_" + group][k] = dict(chi2=v["chi2"] if available else None, ndf=v["ndf"],
                                        mean_stock=s.get("mean"), mean_fast=f.get("mean"),
                                        mean_diff_err=float(np.hypot(s["err"], f["err"])) if available else None)
        out["shapes_" + group][k].update({q: v[q] for q in ("stock_flow", "fast_flow") if q in v})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.diocompare", description=__doc__.split("\n")[0])
    ap.add_argument("--ref", help="stock reference table (cereference with the DIO code), one file per shard")
    ap.add_argument("--starts", help="diospec starts npz the fast run used (--ref of cechain)")
    ap.add_argument("--fast", help="glob of cechain output npz files")
    ap.add_argument("--dev-shards", type=int, default=50)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--offline-root")
    ap.add_argument("--maps", nargs="+")
    ap.add_argument("--out", help="report json")
    ap.add_argument("--plot", nargs=2, metavar=("REPORT", "PNG"), help="only plot an existing report")
    ap.add_argument("--numbers", nargs=2, metavar=("REPORT", "JSON"), help="only write the headline numbers")
    a = ap.parse_args(argv)
    if a.plot or a.numbers:
        if a.plot:
            plot(json.load(open(a.plot[0])), a.plot[1])
        if a.numbers:
            with open(a.numbers[1], "w") as fh:
                json.dump(numbers(json.load(open(a.numbers[0]))), fh, indent=1, default=float)
        return
    geom, field = C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps)
    rep = compare(a.ref, a.fast, a.starts, geom, field, a.dev_shards)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    y = {g: rep["cuts"]["450.0"][g]["yield_"] for g in ("dev", "heldout", "all")}
    print(json.dumps(dict(generated=rep["generated"], fates=rep["fast_fates"],
                          ratio={g: tuple(round(v[k], 4) if v[k] is not None else None for k in ("ratio", "ratio_err"))
                                 for g, v in y.items()}),
                     default=float))


if __name__ == "__main__":
    main()
