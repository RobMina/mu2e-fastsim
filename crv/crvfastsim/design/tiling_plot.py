"""Top-down tiling diagram of the CRV top sectors (T1..T5 and their fillers) of a design geometry.

The view looks down the Mu2e y axis onto the z-x plane. The colour of each point is the number of scintillator layers
a vertical track there crosses: full depth, one layer short, or two or more short. Cracks and holes show up in red.
On top it draws each module's outline (all its layers and segments) in its sector's colour, with fillers dashed.
Zoom panels follow each boundary between neighbouring sectors, where cracks and overlaps occur.

    python -m crvfastsim.design.tiling_plot --out top.png                       # nominal Run 2
    python -m crvfastsim.design.tiling_plot --design '{"n_layers": 6, ...}' --placed --out a.png
    python -m crvfastsim.design.tiling_plot --design a.json --placed --legacy --out a_v3.png   # the old rules

--placed lays the design out and places it (layout.py, placement.py, as fom.evaluate with a placement). Without it
the design is drawn as module.build tiles it on the stock positions (the stock-position model). The summary JSON
next to the PNG lists every boundary's gap or overlap per layer and the area of the top plane by layers crossed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np

TOP = ("T",)                     # sector-name prefixes drawn
GRID_MM = 5.0


def top_boxes(dg, prefixes=TOP):
    """Per top-sector bar: sector name, module, layer, segment and the world box [lo, hi] (x, z) of the bar."""
    names = list(dg.sector_names)
    rows = []
    for s, nm in enumerate(names):
        if not nm.startswith(prefixes):
            continue
        m = np.flatnonzero(dg.sector == s)
        hw = np.abs(dg.halflen[m][:, :, None] * dg.axes[s][None, :, :]).sum(1)     # world half extents [n, 3]
        c = dg.center[m]
        for j, b in enumerate(m):
            rows.append((nm, int(dg.module[b]), int(dg.layer[b]), int(dg.segment[b]),
                         c[j, 0] - hw[j, 0], c[j, 0] + hw[j, 0], c[j, 2] - hw[j, 2], c[j, 2] + hw[j, 2]))
    return rows, int(max(dg.n_layers[i] for i, n in enumerate(names) if n.startswith(prefixes)))


def layer_map(rows, n_layers, grid=GRID_MM):
    """Grid (z, x) of the number of distinct layers a vertical track crosses; returns map, z edges, x edges."""
    x0 = min(r[4] for r in rows) - 100; x1 = max(r[5] for r in rows) + 100
    z0 = min(r[6] for r in rows) - 100; z1 = max(r[7] for r in rows) + 100
    ze, xe = np.arange(z0, z1 + grid, grid), np.arange(x0, x1 + grid, grid)
    zc, xc = 0.5 * (ze[1:] + ze[:-1]), 0.5 * (xe[1:] + xe[:-1])
    count = np.zeros((len(zc), len(xc)), np.int16)
    for lay in range(n_layers):
        hit = np.zeros_like(count, bool)
        for r in rows:
            if r[2] != lay:
                continue
            iz = slice(np.searchsorted(zc, r[6]), np.searchsorted(zc, r[7]))
            ix = slice(np.searchsorted(xc, r[4]), np.searchsorted(xc, r[5]))
            hit[iz, ix] = True
        count += hit
    return count, ze, xe


def module_outlines(rows):
    """{(sector, module, segment): (xlo, xhi, zlo, zhi)} over all layers."""
    out = {}
    for nm, mod, _, seg, xl, xh, zl, zh in rows:
        k = (nm, mod, seg)
        if k in out:
            a = out[k]
            out[k] = (min(a[0], xl), max(a[1], xh), min(a[2], zl), max(a[3], zh))
        else:
            out[k] = (xl, xh, zl, zh)
    return out


def boundaries(rows, n_layers):
    """Gap (> 0) or overlap (< 0) [mm] per layer between sectors that follow each other along z: the next sector's
    first bar edge minus the previous sector's last bar edge, layer by layer."""
    ext = {}
    for nm, _, lay, _, _, _, zl, zh in rows:
        a = ext.setdefault(nm, {}).setdefault(lay, [np.inf, -np.inf])
        a[0], a[1] = min(a[0], zl), max(a[1], zh)
    order = sorted(ext, key=lambda n: min(v[0] for v in ext[n].values()))
    out = []
    for a, b in zip(order, order[1:]):
        gaps = [round(ext[b][l][0] - ext[a][l][1], 1) for l in range(n_layers) if l in ext[a] and l in ext[b]]
        out.append(dict(prev=a, next=b, gap_mm_per_layer=gaps, z=round(ext[a][max(ext[a])][1], 1)))
    return out


def draw(dg, out, title, prefixes=TOP, zooms=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Rectangle

    rows, nl = top_boxes(dg, prefixes)
    count, ze, xe = layer_map(rows, nl)
    bnd = boundaries(rows, nl)
    cmap = ListedColormap(["#ffffff", "#d7301f", "#fdae61", "#e0f3db"])        # none, >= 2 short, 1 short, full
    norm = BoundaryNorm([-0.5, 0.5, nl - 1.5, nl - 0.5, nl + 0.5], cmap.N)
    sectors = sorted({r[0] for r in rows}, key=lambda n: min(r[6] for r in rows if r[0] == n))
    # outlines in dark, neutral colours only: green / orange / red belong to the coverage map
    pal = ("#1b2a49", "#4b3b8f", "#00525e", "#3d3d3d", "#6b4226", "#2f4f2f")
    col = {n: pal[i % len(pal)] for i, n in enumerate(sorted({n.rstrip("F") for n in sectors}))}
    outl = module_outlines(rows)
    # zooms: the boundaries between stock sectors (T2|T3, T3|T4, T4|T5 on Run 2; a filler counts with its parent),
    # the worst first when some are open
    stock_b = [b for b in bnd if b["prev"].rstrip("F") != b["next"].rstrip("F")]
    worst = sorted(stock_b, key=lambda b: (-max(abs(g) for g in b["gap_mm_per_layer"]) if b["gap_mm_per_layer"] else 0,
                                          -b["z"]))
    zb = sorted(worst[:zooms], key=lambda b: b["z"])
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, max(len(zb), 1), height_ratios=[1.25, 1])
    axm = fig.add_subplot(gs[0, :])
    panels = [(axm, None)] + [(fig.add_subplot(gs[1, i]), b) for i, b in enumerate(zb)]
    for ax, b in panels:
        ax.pcolormesh(ze / 1000, xe / 1000, count.T, cmap=cmap, norm=norm, shading="flat", rasterized=True)
        for (nm, mod, seg), (xl, xh, zl, zh) in outl.items():
            fill = nm.endswith("F") and nm.rstrip("F") in col
            ax.add_patch(Rectangle((zl / 1000, xl / 1000), (zh - zl) / 1000, (xh - xl) / 1000, fill=False,
                                   lw=0.5 if b is None else 1.0, ls="--" if fill else "-",
                                   ec=col[nm.rstrip("F")] if nm.rstrip("F") in col else "k"))
        ax.set_xlabel("z [m]")
        ax.set_ylabel("x [m]")
        if b is not None:
            ax.set_xlim((b["z"] - 300) / 1000, (b["z"] + 300) / 1000)
            g = b["gap_mm_per_layer"]
            ax.set_title(f"{b['prev']} | {b['next']}: " + ("gap" if max(g) > 0 else "overlap")
                         + f" {min(g):.0f}..{max(g):.0f} mm / layer", fontsize=9)
    for nm in sectors:
        zs = [r for r in rows if r[0] == nm]
        zc = 0.5 * (min(r[6] for r in zs) + max(r[7] for r in zs)) / 1000
        axm.text(zc, max(r[5] for r in zs) / 1000 + 0.12, nm, ha="center", va="bottom", fontsize=8,
                 color=col.get(nm.rstrip("F"), "k"))
    axm.set_title(title, fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, color=cmap(i)) for i in (3, 2, 1)] + [
        Rectangle((0, 0), 1, 1, fill=False, ec="#3d3d3d"), Rectangle((0, 0), 1, 1, fill=False, ec="#3d3d3d", ls="--")]
    axm.legend(handles, [f"{nl} layers", f"{nl - 1} layers", f"<= {nl - 2} layers", "module outline",
                         "filler module"], loc="lower right", fontsize=8, ncol=5, framealpha=0.9)
    axm.set_ylim(xe[0] / 1000, xe[-1] / 1000 + 0.45)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    cell = GRID_MM * GRID_MM / 1e6
    inside = count > 0
    summary = dict(title=title, n_layers=nl, boundaries=bnd,
                   area_m2={"full": float((count == nl).sum() * cell), "one_short": float((count == nl - 1).sum() * cell),
                            "two_or_more_short_inside": float(((count <= nl - 2) & inside).sum() * cell)})
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def geometry(design, placed, close_up=True, remedies="", counters=None, legacy=False):
    """The design geometry: module.build on the stock positions, or the placed layout (first feasible variant)."""
    from . import fom as F
    from . import layout as L
    from . import module as M
    from . import placement as P
    ctx = F.Context() if counters is None else F.Context(counters=counters)
    if not placed:
        return M.build(design, ctx.counters), None
    lc = L.LayoutConfig(split=True, close_up=False) if legacy else dataclasses.replace(L.LayoutConfig(),
                                                                                         close_up=close_up)
    cfg = P.PlacementConfig(layout=lc, remedies=P.parse_remedies(remedies), policy="first", tie_chains=not legacy)
    vs = P.place(design, ctx.counters, cfg)
    v = next((x for x in vs if x.feasible), vs[0])
    return L.build(design, v.text, None, v.layout), v


def main(argv=None):
    from . import module as M
    p = argparse.ArgumentParser(prog="python -m crvfastsim.design.tiling_plot")
    p.add_argument("--design", help="JSON object or file (module.Design keys); absent = nominal Run 2")
    p.add_argument("--placed", action="store_true", help="lay out and place the design (layout.py, placement.py)")
    p.add_argument("--no-close-up", action="store_true", help="placed layout without closing up adjacent sectors")
    p.add_argument("--legacy", action="store_true", help="the G1 option-B v3 rules the first placed scores and the "
                   "Geant4 references used: length splits, no close-up, every sector shifted alone")
    p.add_argument("--remedies", default="", help="placement remedies (placement.parse_remedies syntax)")
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = {}
    if a.design:
        d = json.load(open(a.design)) if os.path.exists(a.design) else json.loads(a.design)
    design = M.Design(**d)
    dg, v = geometry(design, a.placed, not a.no_close_up, a.remedies, legacy=a.legacy)
    kind = ("placed" + (" (v3 rules)" if a.legacy else " (no close-up)" if a.no_close_up else "")) if a.placed \
        else "stock positions"
    title = a.title or ("nominal Run 2" if not d else "design") + f", {kind}: top sectors seen from above"
    if v is not None:
        segmented = int(dg.segment.max()) > 0 if len(dg.segment) else False
        title += (f"; joint stagger {v.stagger_mm:g} mm" if segmented else "") + ("" if v.feasible else ", NOT FEASIBLE")
    s = draw(dg, a.out, title)
    print(json.dumps(dict(out=a.out, area_m2=s["area_m2"],
                          boundaries=[(b["prev"], b["next"], min(b["gap_mm_per_layer"]), max(b["gap_mm_per_layer"]))
                                      for b in s["boundaries"]])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
