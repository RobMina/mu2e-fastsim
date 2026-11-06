"""Opt-in Run1B paired-shard TS comparison; no spatial/time matching.

Independent shards, not particles, set uncertainties. Spectral chi2 is only a
diagonal approximation (includes yield, ignores bin covariance); few shards do
not support precision significance. No Poisson floor or chi2 p-value is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

BINS = {"momentum": np.arange(0., 201., 10.), "elapsed_time": np.arange(0., 501., 20.)}
MUONS = {"mu-": 13, "mu+": -13}


def _load(a):
    if isinstance(a, (str, Path)):
        with np.load(a, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    return {k: np.asarray(v) for k, v in a.items()}


def _keys(a):
    x = np.ascontiguousarray(a, dtype=np.int64)
    return x.view(np.dtype([(f"k{i}", np.int64) for i in range(x.shape[1])])).ravel()


def _validate(a, reference):
    n = len(a["id"])
    shapes = {"id": (n, 4), "shard": (n,), "pdg": (n,), "x": (n, 3),
              "p": (n, 3), "t": (n,)}
    shapes.update({"stock_" + k: shapes[k] for k in ("x", "p", "t", "pdg")} if reference
                  else {"fate": (n,)})
    if reference:
        shapes["stock_reached"] = (n,)
    for k, shape in shapes.items():
        if a[k].shape != shape:
            raise ValueError(f"invalid {k} shape: expected {shape}")
    for k in ("id", "shard", "pdg", "stock_pdg" if reference else "fate"):
        if a[k].dtype.kind not in "iu" or (a[k].dtype.kind == "u" and np.any(a[k] > np.iinfo(np.int64).max)):
            raise ValueError(f"{k} must be integer within int64 range")
    keys = _keys(a["id"])
    if len(np.unique(keys)) != n:
        raise ValueError("duplicate IDs")
    reached = a["stock_reached"] if reference else a["fate"] == 1
    if not np.all(np.isin(reached, [0, 1])):
        raise ValueError("stock_reached must be boolean")
    for prefix, mask in (("", np.ones(n, bool)), ("stock_", reached.astype(bool))) if reference else (("", reached),):
        for k in ("x", "p", "t"):
            if not np.all(np.isfinite(a[prefix + k][mask])):
                raise ValueError(f"nonfinite {prefix + k} on selected state")
    return keys


def _charge(pdg):
    pdg = np.asarray(pdg, dtype=np.int64)
    if np.any(~np.isin(np.abs(pdg), [13, 211])):
        raise ValueError("input species must be charged pion or muon")
    return np.where(np.abs(pdg) == 211, np.sign(pdg), -np.sign(pdg))


def _json(x):
    """Strict JSON, with null for undefined estimates (never NaN/Infinity)."""
    a = np.asarray(x)
    return np.where(np.isfinite(a), a, None).tolist()


def _paired(stock, fast, n):
    """Estimating-equation errors for counts of shape (shards, ...)."""
    s, f = np.asarray(stock, float), np.asarray(fast, float)
    S, F, N, K = s.sum(0), f.sum(0), float(np.sum(n)), len(n)
    cohort = np.asarray(n).reshape((K,) + (1,) * (s.ndim - 1))
    delta = (F - S) / N if N else np.full_like(S, np.nan)
    ratio = np.divide(F, S, out=np.full_like(S, np.nan), where=S > 0)
    def se(residual, denominator):
        if K < 2:
            return np.full_like(S, np.nan)
        variance = K / (K - 1) * np.sum(residual ** 2, axis=0)
        return np.divide(np.sqrt(variance), denominator, out=np.full_like(S, np.nan),
                         where=np.asarray(denominator) > 0)
    # Cross-multiplication preserves zero variance for exactly proportional counts.
    err = se((f - s) * N - (F - S) * cohort, N * N)
    pull = np.divide(delta, err, out=np.full_like(S, np.nan), where=err > 0)
    pull = np.where((err == 0) & (delta == 0), 0., pull)
    singular = (err == 0) & (delta != 0) & np.isfinite(delta)
    return {"stock": S.astype(int).tolist(), "fast": F.astype(int).tolist(),
            "stock_fraction": _json(S / N if N else np.full_like(S, np.nan)),
            "fast_fraction": _json(F / N if N else np.full_like(F, np.nan)),
            "difference": _json(delta), "difference_se": _json(err),
            "ratio": _json(ratio), "ratio_se": _json(se(f * S - F * s, S * S)),
            "pull": _json(pull), "zero_variance_nonzero_difference": singular.tolist()}


def _hist(values, mask, shard, k, edges):
    # Include the last edge, as numpy.histogram does; retain both flow bins.
    b = np.searchsorted(edges, values[mask], side="right")
    b[values[mask] == edges[-1]] = len(edges) - 1
    h = np.bincount(shard[mask] * (len(edges) + 1) + b,
                    minlength=k * (len(edges) + 1)).reshape(k, len(edges) + 1)
    return h[:, 1:-1], h[:, [0, -1]].sum(0).tolist()


def _shards(r):
    """Keep independent (run, shard) units, including parent-only empty units."""
    rows = np.column_stack((r["id"][:, 0], r["shard"]))
    parent = r.get("parent_shard_ids", r.get("parent_shard", np.empty((0, 2), dtype=int)))
    if parent.dtype.kind not in "iu":
        raise ValueError("parent shard IDs must be integer")
    if parent.ndim == 1:
        runs = np.unique(rows[:, 0])
        if len(runs) > 1 and len(parent):
            raise ValueError("multi-run parent inventory requires (run, shard) IDs")
        parent = np.column_stack((np.full(len(parent), runs[0] if len(runs) else 0), parent))
    if parent.ndim != 2 or parent.shape[1] != 2:
        raise ValueError("parent_shard_ids must have shape (K, 2)")
    if "parent_shard_event_counts" in r:
        counts = r["parent_shard_event_counts"]
        if counts.shape != (len(parent),) or counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("invalid parent_shard_event_counts")
    return rows, np.unique(np.concatenate((rows, parent)), axis=0)


def compare(reference, fast, shards=None, *, geometry_label="Run1B"):
    """Return strict-JSON-ready paired metrics from NPZ paths or array mappings.

    Without shards, every reference ID is required. Explicit shards select both
    tables, but missing rows *within* that cohort still raise. parent_shard_ids
    optionally lists (run, shard) pairs including zero-input shards; denominators always
    count source rows, never parent event counts. Runtime describes the full fast
    file, not a prorated shard subset.
    """
    if geometry_label not in ("Run1A", "Run1B"):
        raise ValueError("geometry_label must be Run1A or Run1B")
    r, f = _load(reference), _load(fast)
    rk, fk = _validate(r, True), _validate(f, False)
    order = np.argsort(rk)
    pos = np.searchsorted(rk[order], fk)
    if len(fk) and (not len(rk) or np.any(pos == len(rk))):
        raise ValueError("fast IDs absent from reference")
    index = order[pos]
    if np.any(rk[index] != fk):
        raise ValueError("fast IDs absent from reference")
    if np.any(r["shard"][index] != f["shard"]):
        raise ValueError("shard mismatch for matched IDs")
    row_units, units = _shards(r)
    inventory = np.unique(units[:, 1])
    chosen = inventory if shards is None else np.asarray(shards)
    if chosen.ndim != 1 or chosen.dtype.kind not in "iu" or np.any(~np.isin(chosen, inventory)):
        raise ValueError("invalid/unknown selected shards")
    chosen = np.unique(chosen)
    units = units[np.isin(units[:, 1], chosen)]
    ri, fi = np.flatnonzero(np.isin(r["shard"], chosen)), np.flatnonzero(np.isin(f["shard"], chosen))
    if len(ri) != len(fi):
        raise ValueError("missing fast rows in chosen reference cohort; use explicit --shards for a subset")
    metadata = {}
    if "nested_id" in r:
        metadata = {"cohort_definition": "known root entrance lineages",
                    "nested_tracks_represented_by_ancestor": int(np.isin(r["nested_id"][:, 1], chosen).sum()),
                    "upstream_ancestry_incomplete_sources": int(r["source_parent_ancestry_incomplete"][ri].sum()),
                    "parent_events": int(r["parent_shard_event_counts"][np.isin(r["parent_shard_ids"][:, 1], chosen)].sum())}
    fate_values, fate_counts = np.unique(f["fate"][fi], return_counts=True)
    # Reorder the reference to fast's exact-ID order; no coordinates enter this join.
    r = {k: r[k][index[fi]] for k in ("id", "shard", "pdg", "t", "stock_reached", "stock_pdg", "stock_p", "stock_t")}
    seconds = np.asarray(f["seconds"])
    if seconds.shape != () or not np.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be a finite nonnegative scalar")
    f = {k: f[k][fi] for k in ("fate", "pdg", "p", "t")}
    group = np.searchsorted(_keys(units), _keys(row_units[index[fi]]))
    q, K = _charge(r["pdg"]), len(units)
    values = {"momentum": (np.linalg.norm(r["stock_p"], axis=1), np.linalg.norm(f["p"], axis=1)),
              "elapsed_time": (r["stock_t"] - r["t"], f["t"] - r["t"])}
    out = {"shards": chosen.tolist(), "independent_shards": units.tolist(), "nshards": K, "ninput": len(ri),
           "fast_seconds_full_file": float(seconds), "charges": {}, "warnings": [],
           "reference_metadata": metadata, "fast_fate_counts": dict(zip(map(str, fate_values), map(int, fate_counts))),
           "fast_inputs_per_second": float(len(fk) / seconds) if seconds > 0 else None,
           "stock_grid_slots_replaced": None,
           "geometry_label": geometry_label,
           "scope": ("Run1B mu/pi transport development, not Run1A validation; no matched TS-only stock timing"
                     if geometry_label == "Run1B" else
                     "Run1A mu/pi transport development comparison; no matched TS-only stock timing"),
           "statistics": "Independent-shard estimating equations; no Poisson floor. Spectral chi2 is diagonal, includes yield, ignores bin covariance; few-shard approximation, no p-value."}
    if K < 10:
        out["warnings"].append("Few independent shards: uncertainties/diagonal chi2 are not precision significance.")
    for charge in (-1, 1):
        cohort = q == charge
        n = np.bincount(group[cohort], minlength=K)
        sm, fm = cohort & r["stock_reached"].astype(bool), cohort & (f["fate"] == 1)
        count = lambda m: np.bincount(group[m], minlength=K)
        c = {"ninput": int(n.sum()), "ninput_shard": n.tolist(),
             "ds": _paired(count(sm), count(fm), n), "muons": {}}
        for name, pdg in MUONS.items():
            s, a = sm & (r["stock_pdg"] == pdg), fm & (f["pdg"] == pdg)
            sc, fc = count(s), count(a)
            m = {**_paired(sc, fc, n), "stock_shard": sc.tolist(), "fast_shard": fc.tolist(), "histograms": {}}
            for variable, edges in BINS.items():
                sv, fv = values[variable]
                hs, sf = _hist(sv, s, group, K, edges)
                hf, ff = _hist(fv, a, group, K, edges)
                h = _paired(hs, hf, n)
                pull = np.asarray(h["pull"], float)
                used = (hs.sum(0) + hf.sum(0)) > 0
                chi2 = np.sum(pull[used] ** 2) if np.all(np.isfinite(pull[used])) else np.nan
                h.update(edges=edges.tolist(), stock_shard=hs.tolist(), fast_shard=hf.tolist(),
                         stock_flow=sf, fast_flow=ff, chi2_diagonal=_json(chi2), chi2_bins=int(used.sum()))
                m["histograms"][variable] = h
            c["muons"][name] = m
        primary = c["muons"]["mu-" if charge < 0 else "mu+"]
        c["headline"] = {k: primary[k] for k in ("stock", "fast", "ratio", "ratio_se", "difference", "difference_se")}
        c["headline"]["shapes"] = {v: {"chi2": h["chi2_diagonal"], "ndf": h["chi2_bins"]}
                                    for v, h in primary["histograms"].items()}
        out["charges"][str(charge)] = c
    positive = out["charges"]["1"]["muons"]["mu+"]
    if min(positive["stock"], positive["fast"]) < 10:
        out["warnings"].append("Scarce positive muons in at least one arm: not a precision comparison.")
    if metadata.get("upstream_ancestry_incomplete_sources", 0):
        out["warnings"].append(f"Known-root cohort: upstream ancestry incomplete for {metadata['upstream_ancestry_incomplete_sources']:,} inputs.")
    return out


def plot(result, path, label="development"):
    """Write 2x2 per-input-charge spectra, each with a paired pull panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if label not in ("development", "held-out"):
        raise ValueError("label must be development or held-out")
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 2, hspace=.45, wspace=.3)
    for row, charge in enumerate((-1, 1)):
        c = result["charges"][str(charge)]
        for col, variable in enumerate(BINS):
            sub = grid[row, col].subgridspec(2, 1, height_ratios=(3, 1), hspace=.06)
            ax = fig.add_subplot(sub[0]); pa = fig.add_subplot(sub[1], sharex=ax)
            for name, style in (("mu-", "-"), ("mu+", "--")):
                m = c["muons"][name]
                if not (m["stock"] or m["fast"]) and MUONS[name] != -13 * charge:
                    continue
                h = m["histograms"][variable]
                edges = np.asarray(h["edges"]); centres = (edges[1:] + edges[:-1]) / 2
                for arm, color in (("stock", "#0072B2"), ("fast", "#D55E00")):
                    ax.stairs(np.asarray(h[arm]) / max(c["ninput"], 1), edges,
                              color=color, linestyle=style, label=f"{arm} {name}")
                pa.plot(centres, np.asarray(h["pull"], float), style, color="#333333", linewidth=1)
                bad = np.asarray(h["zero_variance_nonzero_difference"])
                if bad.any():
                    pa.scatter(centres[bad], np.sign(np.asarray(h["difference"], float)[bad]) * 5,
                               marker="x", color="#D55E00", label="undefined (zero SE)")
                    pa.legend(fontsize=6)
            headline = c["headline"]
            ratio = (f"{headline['ratio']:.2f} ± {headline['ratio_se']:.2f}" if headline["ratio"] is not None
                     and headline["ratio_se"] is not None else "undefined")
            detail = (f"mu yield fast/stock: {ratio}" if min(headline["stock"], headline["fast"]) >= 10
                      else f"mu counts fast/stock: {headline['fast']}/{headline['stock']} (sparse)")
            ax.set_title(f"Input charge {charge:+d}; N={c['ninput']}\n{detail}", fontsize=11)
            ax.set_ylabel("DS muons / input / bin"); ax.legend(fontsize=8)
            ax.tick_params(labelbottom=False)
            pa.axhline(0, color="gray", linewidth=.7); pa.set_ylabel("paired\npull")
            pa.set_xlabel("Momentum [MeV/c]" if variable == "momentum" else "Elapsed time [ns]")
            if result["nshards"] < 2:
                pa.text(.02, .75, "SE undefined: <2 shards", transform=pa.transAxes, fontsize=7)
    fig.suptitle(f"{result.get('geometry_label', 'Run1B')} {label.upper()} — paired TS comparison", fontsize=15)
    fig.text(.5, .015, "\n".join(result["warnings"] + ["Diagonal spectral chi2: approximation; no p-value. Undefined pulls marked x, not finite significance."]),
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.91, bottom=.14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--fast")
    parser.add_argument("--plot-json", help="render an existing numbers.json without recomputing statistics")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, nargs="+")
    parser.add_argument("--label", choices=("development", "held-out"), default="development")
    parser.add_argument("--geometry-label", choices=("Run1A", "Run1B"), default="Run1B")
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.plot_json:
        result = json.loads(Path(args.plot_json).read_text())
    else:
        if not args.reference or not args.fast:
            parser.error("--reference and --fast are required unless --plot-json is used")
        result = compare(args.reference, args.fast, args.shards, geometry_label=args.geometry_label)
        result["label"] = args.label
        (out / "numbers.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, out / "ts.png", result.get("label", args.label))


if __name__ == "__main__":
    main()
