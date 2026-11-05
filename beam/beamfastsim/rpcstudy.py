"""RPC electron-marginal transport study on saved stock events and pion windows.

Use rpcspec for pairs and the existing cechain --ref interface for transport.
Saved dts events are detector-filtered: their paired comparison is conditional.
A source-window comparison quotes the saved-stock yield per generated capture,
not inclusive Geant4 acceptance. No detector-filter implication is assumed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import rpcspec as S

BINS = {"p": np.arange(0., 140.01, 2.), "cos": np.linspace(-1., 1., 41), "r": np.arange(0., 1000.01, 25.)}


def starts(window, out, repeats=1, seed=1):
    """One pair per saved production-window occurrence, preserving its shard."""
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    with np.load(window, allow_pickle=False) as s:
        required = ("xstop", "ystop", "zstop", "tstop", "pdg", "at_rest", "foil", "codestop",
                    "production_ifile", "rpc_event_occurrence")
        a = {k: np.asarray(s[k]) for k in required}
    n = len(a["pdg"])
    if not n or any(v.shape != (n,) or not np.isfinite(v).all() for v in a.values()):
        raise ValueError("production-window columns must be finite, equally sized vectors")
    if not np.all((a["pdg"] == -211) & a["at_rest"] & (a["foil"] >= 0)
                  & (a["codestop"] == 78) & (a["tstop"] >= 350)):
        raise ValueError("window must contain selected physical target pion stops")
    identity = np.column_stack((a["production_ifile"], a["rpc_event_occurrence"]))
    if np.unique(identity, axis=0).shape[0] != n:
        raise ValueError("window must have one selected pion per capture occurrence")
    rows = np.tile(np.arange(n), repeats)
    tic = time.perf_counter()
    pair = S.internal_pair(np.random.default_rng(seed),
                           S.PhotonSpectrum().sample(np.random.default_rng(seed + 1), rows.size))
    wall = time.perf_counter() - tic
    outdata = dict(x=a["xstop"][rows], y=a["ystop"][rows], z=a["zstop"][rows], time=a["tstop"][rows],
                   pdg=np.full(rows.size, 11, np.int32), shard=a["production_ifile"][rows].astype(int),
                   occurrence=a["rpc_event_occurrence"][rows].astype(int), window_row=rows, **pair)
    for j, k in enumerate(("px", "py", "pz")):
        outdata[k] = pair["p"] * pair["u"][:, j]
    np.savez_compressed(out, **outdata, seed=seed, repeats=repeats, generator_wall_s=wall)
    report = dict(captures=int(rows.size), input_occurrences=n, repeats=repeats,
                  generator_wall_s=wall, seed=seed,
                  per_shard={str(int(k)): int(v) for k, v in zip(*np.unique(outdata["shard"], return_counts=True))})
    print(json.dumps(report))
    return report


def clustered_ratio(stock, fast, exposure):
    """Ratio and yield difference with paired per-shard residual errors."""
    stock, fast, exposure = (np.asarray(v, float) for v in (stock, fast, exposure))
    if (stock.ndim != 1 or stock.size == 0 or stock.shape != fast.shape or stock.shape != exposure.shape
            or np.any(exposure <= 0) or np.any(stock < 0) or np.any(fast < 0)):
        raise ValueError("need nonnegative per-shard counts and positive common exposure")
    m, ns, nf, n = len(stock), stock.sum(), fast.sum(), exposure.sum()
    ratio = nf / ns if ns > 0 else None
    delta = (nf - ns) / n
    def err(residual, norm):
        return float(np.sqrt(m / (m - 1) * np.sum(residual ** 2)) / norm) if m > 1 else None
    return dict(shards=m, exposure=int(n), stock_count=float(ns), fast_count=float(nf),
                stock_yield=float(ns / n), fast_yield=float(nf / n),
                ratio=float(ratio) if ratio is not None else None,
                ratio_error=err(fast - ratio * stock, ns) if ratio is not None else None,
                difference=float(delta), difference_error=err(fast - stock - delta * exposure, n),
                per_shard_stock=stock.tolist(), per_shard_fast=fast.tolist(),
                per_shard_exposure=exposure.tolist())


def hist_by_shard(values, selected, shard, members, edges):
    """Counts including flows; every listed shard gets a row even if empty."""
    si = np.searchsorted(members, shard)
    match = np.isin(shard, members) & selected
    b = np.searchsorted(edges, values[match], side="right")
    return np.bincount(si[match] * (len(edges) + 1) + b,
                       minlength=len(members) * (len(edges) + 1)).reshape(len(members), len(edges) + 1)


def shape_report(stock, fast, edges):
    """Normalised shape differences with paired-shard errors and marginal chi2."""
    stock, fast = np.asarray(stock, float), np.asarray(fast, float)
    m = len(stock)
    ns, nf = stock.sum(), fast.sum()
    if ns == 0 or nf == 0:
        return dict(edges=edges.tolist(), stock=stock.sum(axis=0).tolist(), fast=fast.sum(axis=0).tolist(),
                    chi2=None, ndf=0, pulls=[None] * stock.shape[1], ratio=[None] * stock.shape[1])
    scale = ns / nf
    s, f = stock.sum(axis=0), fast.sum(axis=0)
    pooled = (s + scale * f) / (2 * ns)
    residual = scale * fast - stock - pooled * (scale * fast.sum(axis=1) - stock.sum(axis=1))[:, None]
    variance = m * residual.var(axis=0, ddof=1) if m > 1 else np.full_like(s, np.nan)
    delta = scale * f - s
    valid = (s + scale * f >= 20) & (variance > 0) & np.isfinite(variance)
    pulls = np.divide(delta, np.sqrt(variance), out=np.full_like(s, np.nan), where=variance > 0)
    ratio = np.divide(scale * f, s, out=np.full_like(s, np.nan), where=s > 0)
    finite = lambda x: [float(v) if np.isfinite(v) else None for v in x]
    return dict(edges=edges.tolist(), stock=s.tolist(), fast=(scale * f).tolist(), scale=float(scale),
                pulls=finite(pulls), ratio=finite(ratio), difference_error=finite(np.sqrt(variance)),
                chi2=float(np.sum(pulls[valid] ** 2)), ndf=max(int(valid.sum()) - 1, 0),
                zero_variance_nonzero_difference=int(np.sum((variance == 0) & (delta != 0))),
                chi2_scope="sum of marginal paired-shard pulls squared; bin covariance ignored, descriptive with few shards")


def plane_quantities(table, stock=False, axis_x=-3904.):
    """Full-row arrays; missing crossings remain false instead of dropping rows."""
    if stock:
        pvec = np.column_stack([table[k + "vd_front"] for k in ("px", "py", "pz")]).astype(float)
        p = np.linalg.norm(pvec, axis=1)
        x = np.column_stack([table[k + "vd_front"] for k in ("x", "y", "z")]).astype(float)
        ok = np.asarray(table["isvd_front"], bool)
        cos = np.divide(pvec[:, 2], p, out=np.zeros_like(p), where=p > 0)
    else:
        p = np.sqrt(table["T"] * (table["T"] + 2 * 0.51099895))
        x, cos = table["x"], table["u"][:, 2]
        ok = table["fate"] == 1
    return dict(ok=ok, p=p, cos=cos, r=np.hypot(x[:, 0] - axis_x, x[:, 1]))


def compare(ref_path, fast_path, members, *, starts_path=None, axis_x=-3904., pmin=60., vd_radius=850.1):
    """Compare either exact saved starts or matched-stop-window source draws."""
    if not np.isfinite([axis_x, pmin, vd_radius]).all() or pmin < 0 or vd_radius <= 0:
        raise ValueError("finite plane geometry and nonnegative momentum cut required")
    members = np.sort(np.asarray(members, int))
    if members.size < 1 or np.unique(members).size != members.size:
        raise ValueError("need distinct shard IDs")
    with np.load(ref_path, allow_pickle=False) as r, np.load(fast_path, allow_pickle=False) as f:
        idx = f["idx"].astype(int)
        g = plane_quantities(r, stock=True, axis_x=axis_x)
        b = plane_quantities(f, axis_x=axis_x)
        gs = r["ifile"].astype(int)
        if starts_path is None:
            if not np.array_equal(np.sort(idx), np.arange(len(gs))):
                raise ValueError("paired fast output must cover each saved reference start once")
            expected_x = np.column_stack([r[k][idx] for k in ("x", "y", "z")])
            expected_p = np.column_stack([r[k][idx] for k in ("px", "py", "pz")])
            if not (np.allclose(f["x0"], expected_x, atol=1e-8, rtol=0)
                    and np.allclose(f["p0"][:, None] * f["u0"], expected_p, atol=1e-8, rtol=1e-10)
                    and np.allclose(f["t0"], r["time"][idx], atol=1e-8, rtol=0)):
                raise ValueError("paired fast output does not share stock starts")
            fs = gs[idx]
            exposure = np.array([np.sum(gs == k) for k in members])
            scale = np.ones(members.size)
            scope = "same saved detector-filtered starts; conditional electron-marginal comparison"
        else:
            with np.load(starts_path, allow_pickle=False) as st:
                if not np.array_equal(np.sort(idx), np.arange(len(st["shard"]))):
                    raise ValueError("fast output must contain every generated start once")
                fs = st["shard"][idx].astype(int)
                expected_x = np.column_stack([st[k][idx] for k in ("x", "y", "z")])
                expected_p = np.column_stack([st[k][idx] for k in ("px", "py", "pz")])
                if not (np.allclose(f["x0"], expected_x, atol=1e-8, rtol=0)
                        and np.allclose(f["p0"][:, None] * f["u0"], expected_p, atol=1e-8, rtol=1e-10)
                        and np.allclose(f["t0"], st["time"][idx], atol=1e-8, rtol=0)):
                    raise ValueError("fast output does not match generated starts")
            exposure = np.asarray(r["gencount"], int)[members]
            fast_n = np.array([np.sum(fs == k) for k in members])
            if np.any(fast_n == 0):
                raise ValueError("generated source shard is missing")
            scale = exposure / fast_n
            scope = "same pion-stop windows; independent pair draws; saved-stock yield, not inclusive Geant4 acceptance"
        out = dict(scope=scope, members=members.tolist(), fast_total_rows=int(len(idx)),
                   stock_total_rows=int(len(gs)), stock_total_generated=int(np.sum(r["gencount"])),
                   fast_wall_s=float(f["wall"]), vd_radius_mm=vd_radius, selections={})
        for name, threshold in (("vd13", 0.), ("vd13_pmin", pmin)):
            sa = g["ok"] & (g["p"] > threshold) & (g["r"] < vd_radius)
            sb = b["ok"] & (b["p"] > threshold) & (b["r"] < vd_radius)
            ks = np.array([np.sum(sa & (gs == k)) for k in members])
            kf = np.array([np.sum(sb & (fs == k)) for k in members]) * scale
            report = dict(yield_=clustered_ratio(ks, kf, exposure), pmin=threshold, shapes={})
            for key, edges in BINS.items():
                hs = hist_by_shard(g[key], sa, gs, members, edges)
                hf = hist_by_shard(b[key], sb, fs, members, edges) * scale[:, None]
                report["shapes"][key] = shape_report(hs, hf, edges)
            out["selections"][name] = report
        return out


def plot(report, output):
    """VD13 momentum, angle and radius overlays with unclipped paired-shard pulls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panel = report["selections"]["vd13_pmin"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 5.5), sharex="col", height_ratios=(3, 1))
    labels = {"p": "Momentum at VD13 [MeV/c]", "cos": r"cos $\theta$ at VD13", "r": "Radius at VD13 [mm]"}
    for j, (key, label) in enumerate(labels.items()):
        h = panel["shapes"][key]
        edges = np.asarray(h["edges"])
        ax[0, j].stairs(h["stock"][1:-1], edges, label="Saved stock", color="#0072B2")
        ax[0, j].stairs(h["fast"][1:-1], edges, label="Fast (equal area)", color="#D55E00")
        ax[0, j].set_title(f"Marginal $\\chi^2$/ndf = {h['chi2']:.1f}/{h['ndf']}")
        ax[0, j].set_ylabel("Electrons / bin")
        ax[1, j].plot((edges[:-1] + edges[1:]) / 2, h["pulls"][1:-1], ".", color="#D55E00")
        ax[1, j].axhline(0, color="black", linewidth=.7)
        ax[1, j].set_ylabel("Shard pull")
        ax[1, j].set_xlabel(label)
    y = panel["yield_"]
    ax[0, 0].legend(fontsize=9)
    fig.suptitle(f"RPC electron marginal, p > {panel['pmin']:g} MeV/c: fast/saved-stock "
                 f"{y['ratio']:.4f} ± {y['ratio_error']:.4f}\n{report['scope']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("starts")
    g.add_argument("--window", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--repeats", type=int, default=1)
    g.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--fast", required=True)
    c.add_argument("--starts")
    c.add_argument("--shards", type=int, nargs="+", required=True)
    c.add_argument("--axis-x", type=float, default=-3904.)
    c.add_argument("--pmin", type=float, default=60.)
    c.add_argument("--vd-radius", type=float, default=850.1, help="Run 1A TT_FrontHollow disk radius [mm]")
    c.add_argument("--out", required=True)
    c.add_argument("--png")
    a = ap.parse_args(argv)
    if a.command == "starts":
        starts(a.window, a.out, a.repeats, a.seed)
    else:
        report = compare(a.ref, a.fast, a.shards, starts_path=a.starts, axis_x=a.axis_x, pmin=a.pmin,
                         vd_radius=a.vd_radius)
        Path(a.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if a.png:
            plot(report, a.png)
        print(json.dumps({k: v["yield_"] for k, v in report["selections"].items()}))


if __name__ == "__main__":
    main()
