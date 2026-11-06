"""Figures of the Run 1A V-series (run1a_vseries.json): cluster PE and layer multiplicity, fast vs stock, with pulls."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STOCK, FAST, RESP, LIB = "#000000", "#0072B2", "#E69F00", "#D55E00"   # Okabe-Ito


def _steps(ax, edges, y, **kw):
    ax.stairs(y, edges, **kw)


def _pulls(ax, x, res, color, marker, label):
    ax.plot(x, res["pull"], marker, color=color, ms=4, label=label, ls="none")


def _merged(edges, res, display_top):
    """Edges of the tail-merged bins of res (paired_shape 'bins'), the open top edge drawn at display_top."""
    e = np.asarray(edges, float)
    e = np.where(np.isfinite(e) & (e < 1e8), e, display_top)
    g = res["bins"]
    return np.array([e[x[0]] for x in g] + [e[g[-1][-1] + 1]])


def make(result, out, labels=None):
    """labels: optional {"stock": legend of the stock arm} (the Offline release of the reference)."""
    stock_label = (labels or {}).get("stock", "Geant4 + stock response/reco")
    out = Path(out)
    e = result["e2e"]
    rc = result["response_control"]
    ec = result["e2e_vs_response_control"]
    shards = len(result["shards"])

    k = "e2e/cluster_pe"
    edges = _merged(result["edges"]["cluster_pe"], e[k], 3200.0)
    width = np.diff(edges) / 100.0
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    _steps(a, edges, np.asarray(e[k]["ref"]) / width, color=STOCK, lw=1.6, label=stock_label)
    _steps(a, edges, np.asarray(e[k]["fast"]) / width, color=FAST, lw=1.4, label="fast end to end")
    if rc[k]["bins"] == e[k]["bins"]:
        _steps(a, edges, np.asarray(rc[k]["fast"]) / width, color=RESP, lw=1.2, ls="--",
               label="Geant4 + fast response/reco")
    a.set_yscale("log")
    a.set_ylabel(f"coincidence clusters / 100 PE ({shards} x 50k events)")
    a.legend(fontsize=8, frameon=False)
    r, rl = e["e2e/clusters"], ec["e2e/clusters"]
    a.set_title(f"Run 1A CRV clusters, end to end / stock {r['ratio']:.3f} +- {r['err']:.3f} (chi2/ndf {e[k]['chi2_ndf']:.2f})\n"
                f"transport + library: end to end / Geant4 steps, same response {rl['ratio']:.3f} +- {rl['err']:.3f} "
                f"(chi2/ndf {ec[k]['chi2_ndf']:.2f})\nnull expectation of chi2/ndf with {shards} shards: {e[k]['chi2_ndf_null']:.2f}", fontsize=8)
    x = 0.5 * (edges[1:] + edges[:-1])
    _pulls(b, x, e[k], FAST, "o", "end to end vs stock")
    xl = 0.5 * (_merged(result["edges"]["cluster_pe"], ec[k], 3200.0)[1:] + _merged(result["edges"]["cluster_pe"], ec[k], 3200.0)[:-1])
    _pulls(b, xl, ec[k], LIB, "^", "end to end vs Geant4 + fast response")
    b.legend(fontsize=7, frameon=False, loc="upper right")
    for y in (-2, 0, 2):
        b.axhline(y, color="0.6", lw=0.8, ls=":" if y else "-")
    b.set_ylabel("paired pull")
    b.set_xlabel("cluster PE (last bin: above 3000)")
    fig.tight_layout()
    fig.savefig(out / "crv_pe.png", dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    k = "e2e/layers"
    groups = e[k]["bins"]                                # tail-merged layer bins (edges 1, 2, 3, 4, 5)
    names = [str(g[0] + 1) if len(g) == 1 else (f"<= {g[-1] + 1}" if i == 0 else f">= {g[0] + 1}")
             for i, g in enumerate(groups)]
    xs0 = np.arange(len(groups))
    w = 0.27
    arms = ((e[k]["ref"], STOCK, stock_label), (e[k]["fast"], FAST, "fast end to end"))
    if rc[k]["bins"] == groups:
        arms += ((rc[k]["fast"], RESP, "Geant4 + fast response"),)
    for off, (y, c, lab) in zip((-w, 0, w), arms):
        y = np.asarray(y, float)
        a1.bar(xs0 + off, y / y.sum(), width=w, color=c, label=lab)
    a1.set_yscale("log")
    a1.set_ylim(top=5)
    a1.set_xticks(xs0, names)
    a1.set_xlabel("layers in the coincidence cluster")
    a1.set_ylabel("fraction of clusters")
    a1.set_title(f"layer multiplicity, chi2/ndf: end to end vs stock {e[k]['chi2_ndf']:.2f}, "
                 f"vs Geant4 + fast response {ec[k]['chi2_ndf']:.2f}", fontsize=8)
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    qty = [(f"e2e/{s}/{q}", f"{s} {lab}") for s in ("T1", "T2") for q, lab in (("pulses", "pulses >= 5 PE"),
                                                                              ("clusters", "clusters"))]
    xs = np.arange(len(qty))
    for off, (res, c, m, lab) in zip((-0.18, 0, 0.18), ((e, FAST, "o", "end to end / stock"),
                                                        (rc, RESP, "s", "Geant4 + fast response / stock"),
                                                        (ec, LIB, "^", "end to end / Geant4 + fast response"))):
        a2.errorbar(xs + off, [res[q]["ratio"] for q, _ in qty], yerr=[res[q]["err"] for q, _ in qty], fmt=m,
                    color=c, label=lab, capsize=3)
    a2.axhline(1, color="0.6", lw=0.8)
    a2.set_xticks(xs, [lab for _, lab in qty], fontsize=8)
    a2.set_ylabel("ratio (paired per-shard error)")
    a2.set_title(f"pulses and clusters per sector ({shards} x 50k events)", fontsize=9)
    a2.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "crv_layers.png", dpi=150)
    plt.close(fig)
