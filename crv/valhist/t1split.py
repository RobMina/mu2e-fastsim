"""Where does a per-sector crossings excess (V1) sit? Primary-muon bar crossings of fast tables (several seeds, summed)
against G4 on the same generated events, for one sector group, split by: whether the muon also crossed EX (and above
or below it), crossing path length, |p| at entry, position along the bar, module, layer and whether the bar is an edge
counter of its module. Ratios fast/G4 per bin with errors (fast summed over seeds, scaled to one seed).

    python -m valhist.t1split --group T1 --g4 'crvsteps.{s}.root' --fast 'dirA/fastC.{s}.root' 'dirB/fastC.{s}.root' ...
"""
import argparse
import json

import numpy as np

from crvfastsim import geometry, tables
from crvfastsim.analyses import v2compare as V2


def _key(df):
    return (df.subrun.to_numpy().astype(np.int64) << 20) | df.event.to_numpy().astype(np.int64)


def crossings(paths, g4):
    s = tables.read_many(paths, "steps", tables.KEY + ["sp", "bar", "len", "edep", "y0", "px0", "py0", "pz0",
                                                        "x0", "z0"])
    p = tables.read_many(paths, "particles", tables.KEY + ["key", "pdg", "isprim"])
    s = s.merge(p.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"])
    s = s[(s.isprim == 1) & (s.pdg.abs() == tables.PRIMARY_MUON_PDG)].copy()
    s["k"] = _key(s)
    s["pmag"] = np.linalg.norm(s[["px0", "py0", "pz0"]].to_numpy(), axis=1)
    agg = (s.sort_values("y0", ascending=False).groupby(["k", "sp", "bar"], sort=False)
           .agg(len=("len", "sum"), p=("pmag", "first"), x=("x0", "first"), z=("z0", "first"), y=("y0", "first"))
           .reset_index())
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="T1")
    ap.add_argument("--g4", required=True)
    ap.add_argument("--fast", nargs="+", required=True, help="patterns with {s}; one per seed")
    ap.add_argument("--shards", default="1-20")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    geom = geometry.from_offline_config(geometry.find_counters(V2.DEFAULT_COUNTERS))
    grp = np.full(len(geom.sector), "", object)
    for n, secs in V2.GROUPS.items():
        grp[np.isin(geom.sector, secs)] = n
    lo, hi = (int(x) for x in a.shards.split("-"))
    sub = [f"001400_{k:08d}" for k in range(lo, hi + 1)]
    G = crossings([a.g4.format(s=s) for s in sub], True)
    F = [crossings([pat.format(s=s) for s in sub], False) for pat in a.fast]
    nseed = len(F)
    # counters per module: edge = first or last counter of its module layer
    cmax = {}
    for b in range(len(geom.sector)):
        key = (geom.sector[b], geom.module[b], geom.layer[b])
        cmax[key] = max(cmax.get(key, 0), geom.counter[b])
    edge = np.array([geom.counter[b] in (0, cmax[(geom.sector[b], geom.module[b], geom.layer[b])])
                     for b in range(len(geom.sector))])
    lax = geom.local_to_world[:, 2]                       # Mu2e axis of the bar length
    def annotate(X):
        X = X.copy()
        X["grp"] = grp[X.bar.to_numpy()]
        ex = X[X.grp == "EX"].groupby(["k", "sp"]).y.max()
        X["has_ex"] = [(k, sp) in ex.index for k, sp in zip(X.k, X.sp)]
        X["edge"] = edge[X.bar.to_numpy()]
        X["module"] = geom.module[X.bar.to_numpy()]
        X["layer"] = geom.layer[X.bar.to_numpy()]
        c = geom.center[X.bar.to_numpy()]
        h = geom.halflen[X.bar.to_numpy()]
        ax = lax[X.bar.to_numpy()]
        pos = np.where(ax == 0, X.x.to_numpy() - c[:, 0], X.z.to_numpy() - c[:, 2])
        X["along"] = pos / np.take_along_axis(h, ax[:, None], 1)[:, 0]   # -1 .. 1 along the bar
        return X[X.grp == a.group]
    Gs = annotate(G)
    Fs = [annotate(f) for f in F]
    out = {"group": a.group, "g4": len(Gs), "fast_per_seed": [len(f) for f in Fs],
           "ratio_total": float(np.mean([len(f) for f in Fs]) / len(Gs) - 1)}
    splits = {
        "has_ex": lambda X: X.has_ex.map({True: "with EX", False: "no EX"}),
        "edge_counter": lambda X: X.edge.map({True: "edge", False: "inner"}),
        "len_mm": lambda X: np.digitize(X.len, [15, 20, 22, 25, 30, 40, 80]),
        "p_GeV": lambda X: np.digitize(X.p / 1e3, [0.3, 0.6, 1, 2, 4, 10, 30]),
        "along": lambda X: np.digitize(np.abs(X.along), [0.25, 0.5, 0.75, 0.9, 0.97, 1.01]),
        "module": lambda X: X.module,
        "layer": lambda X: X.layer,
    }
    import pandas as pd
    vc = lambda v: pd.Series(np.asarray(v)).value_counts()
    for name, f in splits.items():
        g = vc(f(Gs))
        fc = sum(vc(f(x)).reindex(g.index).fillna(0) for x in Fs) / nseed
        rows = {}
        for b in sorted(g.index, key=str):
            n_g, n_f = float(g[b]), float(fc[b])
            err = np.sqrt(1 / max(n_g, 1) + 1 / max(n_f * nseed, 1))
            rows[str(b)] = {"g4": int(n_g), "fast": round(n_f, 1), "rel": n_f / n_g - 1, "err": err}
        out[name] = rows
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
    for name in splits:
        print(name, "  ".join(f"{b}: {r['rel']:+.3f}+-{r['err']:.3f} (n={r['g4']})" for b, r in out[name].items()))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
