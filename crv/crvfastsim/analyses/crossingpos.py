"""crossingpos -- where primary muons cross the CRV bars, fast sim vs G4 (diagnostic for V2 per-sector pulse spectra).

A crossing is the primary muon's steps in one bar of one event (first entry, last exit). For each, the crossing
midpoint in the bar frame: along the length (signed, + towards side 1 of the bar, i.e. towards larger coordinate on
the length axis), across the width and the thickness, each over the half length. Per sector group:
  * the distribution of the position along the bar (20 bins over [-1, 1]) and its binned ratio fast/G4;
  * crossings present in both arms (same event and bar: the same CRY muon, since both arms share the primaries)
    and the fast - G4 differences of position along the bar (mm), across the width (mm) and of path length (mm).
Single-ended sectors (T1: SiPMs at side 1 only) turn a shift along the bar directly into a pulse-size shift.

    crossingpos FAST.root [...] --compare G4.root [...] --outdir DIR
"""
import numpy as np
import pandas as pd

from mu2efastsim import plots

from .. import geometry, tables
from .census import SECTOR_GROUP

GROUPS = ("EX", "T1", "T2", "M")
EDGES = np.linspace(-1.0, 1.0, 21)
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def crossings(paths, geom):
    """One row per (event, bar) primary-muon crossing: local midpoint over half lengths, path length."""
    steps = tables.read_many(paths, "steps", tables.KEY + ["bar", "sp", "t0", "len", "x0", "y0", "z0", "x1", "y1", "z1"])
    parts = tables.read_many(paths, "particles", tables.KEY + ["key", "pdg", "isprim"])
    prim = parts[(parts.isprim == 1) & (parts.pdg.abs() == tables.PRIMARY_MUON_PDG)]
    s = steps.merge(prim[tables.KEY + ["key"]].rename(columns={"key": "sp"}), on=tables.KEY + ["sp"])
    s = s.sort_values(tables.KEY + ["bar", "t0"], kind="stable")
    g = s.groupby(tables.KEY + ["bar"], sort=False)
    a, b = g[["x0", "y0", "z0"]].first(), g[["x1", "y1", "z1"]].last()
    x = a.reset_index()
    x["len"] = g.len.sum().to_numpy()
    bar = x.bar.to_numpy()
    mid = 0.5 * (a.to_numpy() + b.to_numpy())
    loc = mid - geom.center[bar]
    r = np.arange(len(bar))
    ax = geom.local_to_world[bar]
    h = geom.halflen[bar]
    x["along_mm"] = loc[r, ax[:, 2]]
    x["along"] = x.along_mm / h[r, ax[:, 2]]
    x["width_mm"] = loc[r, ax[:, 1]]
    x["group"] = SECTOR_GROUP[geom.sector[bar]]
    return x[tables.KEY + ["bar", "group", "along", "along_mm", "width_mm", "len"]]


def _quant(v):
    return [float(q) for q in np.quantile(v, (0.05, 0.16, 0.5, 0.84, 0.95))] if len(v) else None


def run(inputs, outdir, compare=None, counters=None, **kw):
    if not compare:
        raise SystemExit("crossingpos needs --compare <G4 CrvStep tables of the same shards>")
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    xf, xr = crossings(tables.expand(inputs), geom), crossings(tables.expand(compare), geom)
    m = xf.merge(xr, on=tables.KEY + ["bar"], suffixes=("_f", "_r"))
    out = {}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(4 * len(GROUPS), 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for j, g in enumerate(GROUPS):
        f, r = xf[xf.group == g], xr[xr.group == g]
        hf = np.histogram(f.along, bins=EDGES)[0].astype(float)
        hr = np.histogram(r.along, bins=EDGES)[0].astype(float)
        ratio = (hf / max(hf.sum(), 1)) / np.maximum(hr / max(hr.sum(), 1), 1e-12)
        err = ratio * np.sqrt(1 / np.maximum(hf, 1) + 1 / np.maximum(hr, 1))
        mg = m[m.group_f == g]
        d_al, d_w, d_l = (mg.along_mm_f - mg.along_mm_r), (mg.width_mm_f - mg.width_mm_r), (mg.len_f - mg.len_r)
        out[g] = {"n_fast": len(f), "n_ref": len(r), "n_matched": len(mg),
                  "along_mean_fast": float(f.along.mean()), "along_mean_ref": float(r.along.mean()),
                  "along_ratio_chi2_ndf": float(np.sum(((ratio - 1) / err) ** 2) / len(ratio)),
                  "along_ratio": ratio, "d_along_mm_q": _quant(d_al), "d_along_mm_mean": float(d_al.mean()),
                  "d_width_mm_q": _quant(d_w), "d_len_mm_q": _quant(d_l)}
        plots.ratio_panel(axes[0, j], axes[1, j], EDGES, hr, hf, "position along bar / half length (+ = side 1)")
        axes[0, j].set_title(f"{g} (N_ref={len(r):,})")
    plots.save(fig, outdir, "crossingpos_along.png")
    plots.write_json(outdir, "crossingpos_stats.json", out)
    for g, o in out.items():
        print(f"{g:3s} crossings {o['n_ref']:,} -> {o['n_fast']:,} (matched {o['n_matched']:,}); mean along "
              f"{o['along_mean_ref']:+.4f} -> {o['along_mean_fast']:+.4f}; along-ratio chi2/ndf "
              f"{o['along_ratio_chi2_ndf']:.2f}")
        print(f"    fast-G4 matched: along [mm] q05/16/50/84/95 {np.round(o['d_along_mm_q'], 2)} mean "
              f"{o['d_along_mm_mean']:+.2f}; width {np.round(o['d_width_mm_q'], 2)}; path {np.round(o['d_len_mm_q'], 2)}")
        print("    along ratio by bin (side 0 -> side 1): " + " ".join(f"{v:.2f}" for v in o["along_ratio"]))
    return out
