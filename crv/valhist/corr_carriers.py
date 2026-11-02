"""What carries the adjacent-layer correlation of muon-descendant energy, from CrvStep tables of one or more arms.

Two definitions are filled side by side:
  module  per (event, EX module) with a primary-muon deposit in all four layers: descendant energy per layer (all muons of
          the event together), mean of the three adjacent-layer Pearson correlations
  muon    per (event, primary muon, layer rank) in a sector group, over all pairs of ranks L, L+1 the muon crosses:
          descendant energy of that muon in the rank (modules pooled), one Pearson correlation over the pairs

Reference arms (with the particle ccode column) also split the covariance of the muon definition, per step-tree root (the
ancestor that is a direct child of the primary muon): same-root term by root process, kinetic energy, deposit and birth
place; between-cell term of the (momentum, path) cells; joint (x, y) cells; muon kinetic-energy loss between the crossings.
Errors are shard jackknives (shards are independent).

    python -m valhist.corr_carriers fill --arm g4='DIR/g4all.{s}.root' --arm lt4='DIR/fast1b.{s}.root' --shards 22-120:2 --out J
    python -m valhist.corr_carriers report J [J2 ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, tier1b as T
from crvfastsim.analyses.census import classify
from . import steplayers as S

K = tables.KEY
MU_MASS = 105.658
COLS = K + ["bar", "sp", "t0", "edep", "len", "px0", "py0", "pz0", "x0", "y0", "z0", "x1", "y1", "z1"]
PROCESS = {31: "muIoni", 34: "muPairProd", 30: "muBrems", 116: "muonNuclear", 14: "Decay", 111: "Decay", 166: "Decay",
           114: "Decay", 32: "capture", 165: "capture"}
KE_EDGES = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0])       # root kinetic energy classes [MeV]
DEP_EDGES = np.array([1.0, 3.0, 10.0, 30.0, 100.0])             # root visible deposit classes over all its steps [MeV]
XY_EDGES = np.array([1e-9, 1.0, 3.0, 10.0, 30.0])               # joint cells of the pair energies [MeV]
DT_EDGES = np.array([5.0, 10.0, 20.0, 50.0, 100.0])             # muon kinetic-energy loss between the crossings [MeV]
P_EDGES = np.array([400.0, 800.0, 2000.0, 8000.0])              # crossing |p| classes [MeV/c]
L_EDGES = np.array([40.0, 60.0])                                # crossing path classes [mm]


def _sums(a, b):
    return np.array([len(a), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()], float)


def _pearson(s):
    n, x, y, xx, yy, xy = s
    if n == 0:
        return float("nan")
    return (xy / n - x * y / n ** 2) / np.sqrt((xx / n - (x / n) ** 2) * (yy / n - (y / n) ** 2))


def _cov(s):
    n, x, y, _, _, xy = s
    return xy / n - x * y / n ** 2


def module_stats(s, geom):
    """Sums of the module definition: n, sum w[4], sum w w^T over EX modules with a muon deposit in all four layers."""
    x = s[(s.bar < 256) & (s.cls.isin(("mu_primary", "mu_descendant")))].copy()
    x["mod"], x["lay"] = x.bar // 64, (x.bar // 16) % 4
    x["mu"] = np.where(x.cls == "mu_primary", x.edep, 0.0)
    x["de"] = np.where(x.cls == "mu_descendant", x.edep, 0.0)
    g = x.groupby(K + ["mod", "lay"], sort=False)[["mu", "de"]].sum().reset_index()
    g = g[g.mu > 0]
    w = g.pivot_table(index=K + ["mod"], columns="lay", values="de", aggfunc="sum").dropna().to_numpy()
    return {"n": float(len(w)), "w": w.sum(0).tolist(), "ww": (w.T @ w).ravel().tolist()}


def module_corr(m):
    n, w, ww = m["n"], np.asarray(m["w"]), np.asarray(m["ww"]).reshape(4, 4)
    c = ww / n - np.outer(w, w) / n ** 2
    d = np.sqrt(np.diag(c))
    return float(np.mean([c[i, i + 1] / (d[i] * d[i + 1]) for i in range(3)]))


def muon_pairs(s, geom):
    """Pairs (rows: event, muon, group, rank L, x, y, and per crossing T, path) of adjacent layer ranks of a muon."""
    lay = T.layer_rank(geom)
    gi = T._group_index(geom, s.bar.to_numpy())
    s = s.assign(gi=gi, lay=lay[s.bar.to_numpy()])
    key = K + ["prim_key", "gi", "lay"]
    mu = s[s.cls == "mu_primary"].sort_values(K + ["t0"], kind="stable")
    mu = mu.assign(p=np.sqrt(mu.px0 ** 2 + mu.py0 ** 2 + mu.pz0 ** 2))
    a = mu.groupby(key, sort=False).agg(p=("p", "first"), path=("len", "sum"), t=("t0", "first")).reset_index()
    d = s[s.cls == "mu_descendant"].groupby(key, sort=False).edep.sum().rename("de").reset_index()
    a = a.merge(d, on=key, how="left").fillna({"de": 0.0})
    b = a.assign(lay=a.lay - 1)
    pr = a.merge(b, on=key, suffixes=("", "_n"))
    return pr.rename(columns={"de": "x", "de_n": "y"})


def decomposition(s, parts, pr):
    """Same-root covariance sums of the EX pairs pr: per (process, KE class, deposit class, born in a bar, split)."""
    root = pd.DataFrame({**{k: parts[k].to_numpy() for k in K}, "sp": parts.key.to_numpy(), "root": T.step_roots(parts)})
    d = s[(s.cls == "mu_descendant") & (s.gi == 0)].merge(root, on=K + ["sp"])
    p = parts.rename(columns={"key": "root"})
    p = p.assign(ke=p.se - p.pdg.abs().map(T.MASS).fillna(0.0), born=np.where(p.inbar, "bar", "gap"))
    d = d.merge(p[K + ["root", "ke", "ccode", "born"]], on=K + ["root"], how="left")
    d["rde"] = d.groupby(K + ["prim_key", "root"]).edep.transform("sum")
    e = d.groupby(K + ["prim_key", "root", "lay"], sort=False).agg(
        xr=("edep", "sum"), ke=("ke", "first"), ccode=("ccode", "first"), born=("born", "first"),
        rde=("rde", "first")).reset_index()
    n = e.assign(lay=e.lay - 1).rename(columns={"xr": "yr"})[K + ["prim_key", "root", "lay", "yr"]]
    r = e.merge(n, on=K + ["prim_key", "root", "lay"]).merge(pr[K + ["prim_key", "lay"]], on=K + ["prim_key", "lay"])
    r["proc"] = r.ccode.map(PROCESS).fillna("other")
    r["kec"] = np.searchsorted(KE_EDGES, r.ke.to_numpy())
    r["depc"] = np.searchsorted(DEP_EDGES, r.rde.to_numpy())
    r["xy"] = r.xr * r.yr
    g = r.groupby(["proc", "kec", "depc", "born"], sort=False).xy.agg(["sum", "size"]).reset_index()
    out = {f"{a}|{b}|{c}|{d_}": [float(u), float(v)] for a, b, c, d_, u, v in g.itertuples(index=False)}
    return out, attach_split(s, d, r)


def attach_split(s, d, r):
    """Of the same-root product sums r: the part the library's step attachment keeps on one crossing (own bar first, else
    the nearest crossing entry of the muon, as tier1b.build), the part from one track, and the part with both ends in the
    muon's own bars. Returned per root KE class."""
    x = s[s.cls == "mu_primary"].sort_values(K + ["t0"], kind="stable").groupby(K + ["prim_key", "bar"], sort=False).agg(
        ex=("x0", "first"), ey=("y0", "first"), ez=("z0", "first")).reset_index().rename(columns={"bar": "xbar"})
    x["xc"] = np.arange(len(x))
    m = d[K + ["prim_key", "root", "lay", "sp", "bar", "edep", "x0", "y0", "z0", "x1", "y1", "z1"]].reset_index(drop=True)
    m["did"] = np.arange(len(m))
    j = m.merge(x, on=K + ["prim_key"])
    dist = np.sqrt((0.5 * (j.x0 + j.x1) - j.ex) ** 2 + (0.5 * (j.y0 + j.y1) - j.ey) ** 2 + (0.5 * (j.z0 + j.z1) - j.ez) ** 2)
    j["score"] = np.where(j.bar.to_numpy() == j.xbar.to_numpy(), -1.0, dist)
    best = j.loc[j.groupby("did").score.idxmin(), ["did", "xc", "score"]]
    m = m.merge(best, on="did")
    m["own"] = m.score < 0
    ids = K + ["prim_key", "root", "lay"]

    def part(e, on):
        a = e.groupby(ids + on, sort=False).edep.sum().rename("xp").reset_index()
        b = a.assign(lay=a.lay - 1).rename(columns={"xp": "yp"})
        return a.merge(b, on=ids + on).merge(r[ids + ["kec"]], on=ids)

    res = {}
    for name, e, on in (("attach", m, ["xc"]), ("track", m, ["sp"]), ("own", m[m.own], [])):
        q = part(e, on)
        v = (q.xp * q.yp).groupby(q.kec).sum()
        res[name] = {str(i): float(t) for i, t in v.items()}
    res["all"] = {str(i): float(t) for i, t in r.xy.groupby(r.kec).sum().items()}
    return res


def cell_sums(pr, code):
    """Per integer cell of the pairs: sums (n, x, y, xx, yy, xy)."""
    f = pd.DataFrame({"c": code, "n": 1.0, "x": pr.x.to_numpy(), "y": pr.y.to_numpy()})
    f["xx"], f["yy"], f["xy"] = f.x ** 2, f.y ** 2, f.x * f.y
    g = f.groupby("c").sum()
    return {str(c): v.tolist() for c, v in zip(g.index, g.to_numpy())}


def fill_shard(path, geom, with_roots):
    st = tables.read(path, "steps", COLS)
    pc = K + ["key", "parent", "pdg"] + (["ccode", "se", "sx", "sy", "sz"] if with_roots else [])
    pa = tables.read(path, "particles", pc)
    s = classify(st, pa, geom.sector)
    out = {"module": module_stats(s, geom), "muon": {}}
    pr = muon_pairs(s, geom)
    for gi, g in enumerate(T.GROUPS):
        q = pr[pr.gi == gi]
        out["muon"][g] = _sums(q.x.to_numpy(), q.y.to_numpy()).tolist()
    q = pr[pr.gi == 0].reset_index(drop=True)
    xc = np.searchsorted(XY_EDGES, q.x.to_numpy())
    yc = np.searchsorted(XY_EDGES, q.y.to_numpy())
    out["joint"] = cell_sums(q, xc * 10 + yc)
    pc_ = np.searchsorted(P_EDGES, np.minimum(q.p, q.p_n).to_numpy())
    lc = np.searchsorted(L_EDGES, q.path.to_numpy()) * 3 + np.searchsorted(L_EDGES, q.path_n.to_numpy())
    out["pathp"] = cell_sums(q, pc_ * 10 + lc)
    te = lambda p: np.sqrt(p ** 2 + MU_MASS ** 2) - MU_MASS
    dt = np.abs(te(q.p.to_numpy()) - te(q.p_n.to_numpy()))
    out["dtmu"] = cell_sums(q, np.searchsorted(DT_EDGES, dt))
    out["edep_mean"] = float(s[s.cls == "mu_descendant"].edep.sum())
    nst = s.groupby(K, sort=False).size().to_numpy()
    out["nstep"] = np.histogram(nst, np.arange(-0.5 + 10, 1000.5 + 10, 10))[0].tolist()
    if with_roots:
        lay = T.layer_rank(geom)
        pos = pa[["sx", "sy", "sz"]].to_numpy()
        pa = pa.assign(inbar=T.bar_containing(pos, geom) >= 0)
        s = s.assign(gi=T._group_index(geom, s.bar.to_numpy()), lay=lay[s.bar.to_numpy()])
        out["same"], out["split"] = decomposition(s, pa, q)
    return out


def _add(a, b):
    """Sum nested dicts of numbers / lists."""
    if isinstance(b, dict):
        return {k: (_add(a[k], b[k]) if k in a and k in b else (a[k] if k in a else b[k])) for k in {*a, *b}}
    if isinstance(b, list):
        return (np.asarray(a) + np.asarray(b)).tolist()
    return a + b


def fill(arms, shard_list, out, counters=None):
    if Path(out).exists():
        raise FileExistsError(out)
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    res = {"shards": shard_list, "arms": {a: [] for a in arms}}
    for a, pat in arms.items():
        for k in shard_list:
            res["arms"][a].append(fill_shard(pat.format(s=S.shard(k), k=k), geom, a.startswith("g4")))
            print(a, k, flush=True)
    Path(out).write_text(json.dumps(res))


def _jack(rows, fn):
    """fn over the sum of all shards, and its shard jackknife error."""
    tot = rows[0]
    for r in rows[1:]:
        tot = _add(tot, r)
    val = fn(tot)
    n = len(rows)
    jk = []
    for r in rows:
        jk.append(fn(_sub(tot, r)))
    jk = np.asarray(jk, float)
    return val, np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))


def _sub(a, b):
    if isinstance(a, dict):
        return {k: (_sub(a[k], b[k]) if k in b else a[k]) for k in a}
    if isinstance(a, list):
        return (np.asarray(a) - np.asarray(b)).tolist()
    return a - b


def _cells(tot, name, key):
    """Sums of the cells of tot[name] selected by key(code)."""
    v = [np.asarray(x) for c, x in tot[name].items() if key(int(c))]
    return np.sum(v, axis=0) if v else np.zeros(6)


def report(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    arms = {}
    for r in runs:
        for a, rows in r["arms"].items():
            arms.setdefault(a, []).extend(rows)
    print(f"{'arm':>10} {'module def':>18} " + " ".join(f"{'muon ' + g:>18}" for g in T.GROUPS))
    for a, rows in arms.items():
        m, me = _jack(rows, lambda t: module_corr(t["module"]))
        c = [f"{m:7.4f} +- {me:.4f}".rjust(18)]
        for g in T.GROUPS:
            v, e = _jack(rows, lambda t: _pearson(t["muon"][g]))
            c.append(f"{v:7.4f} +- {e:.4f}".rjust(18))
        print(f"{a:>10} " + " ".join(c))
    print("\nNStep per event, paired per-shard chi2/ndf against the first arm (10-step bins), mean steps per event ratio")
    first = np.asarray([r["nstep"] for r in next(iter(arms.values()))], float)
    for a, rows in list(arms.items())[1:]:
        h = np.asarray([r["nstep"] for r in rows], float)
        n = len(h)
        d = h - first
        v = n / (n - 1) * ((d - d.mean(0)) ** 2).sum(0)
        q = v > 0
        print(f"{a:>10} {(d.sum(0)[q] ** 2 / v[q]).sum() / q.sum() * (n - 3) / (n - 1):6.2f}")
    for a, rows in arms.items():
        tot = rows[0]
        for r in rows[1:]:
            tot = _add(tot, r)
        s = np.asarray(tot["muon"]["EX"])
        cov = _cov(s)
        print(f"\n== {a}: EX muon pairs {s[0]:.0f}, mean x {s[1] / s[0]:.3f} y {s[2] / s[0]:.3f} MeV, cov {cov:.3f} MeV2")
        print("joint cells (x class, y class; edges 0, 1, 3, 10, 30 MeV): share of pairs, share of cov")
        mx, my = s[1] / s[0], s[2] / s[0]
        for c, v in sorted(tot["joint"].items(), key=lambda kv: int(kv[0])):
            v = np.asarray(v)
            sh = (v[5] - my * v[1] - mx * v[2] + v[0] * mx * my) / s[0] / cov
            print(f"  {int(c):3d} {v[0] / s[0]:8.5f} {sh:7.3f}")
        print("muon kinetic-energy loss between the crossings (edges 5, 10, 20, 50, 100 MeV): pairs, cov share, within corr")
        for c, v in sorted(tot["dtmu"].items(), key=lambda kv: int(kv[0])):
            v = np.asarray(v)
            sh = (v[5] - my * v[1] - mx * v[2] + v[0] * mx * my) / s[0] / cov
            print(f"  {c} {v[0] / s[0]:8.5f} {sh:7.3f} {_pearson(v):7.3f}")
        cells = [np.asarray(v) for v in tot["pathp"].values()]
        between = sum(v[0] * (v[1] / v[0] - mx) * (v[2] / v[0] - my) for v in cells) / s[0]
        print(f"between (p, path) cells: cov share {between / cov:.3f}")
        if "same" not in tot:
            continue
        f = pd.DataFrame([k.split("|") + v for k, v in tot["same"].items()],
                         columns=["proc", "kec", "depc", "born", "xy", "n"])
        same = f.xy.sum() / s[0]
        print(f"same-root product / total product: {same / (s[5] / s[0]):.3f}; same-root / cov: {same / cov:.3f}; "
              f"mean-product baseline {mx * my:.3f}, total product {s[5] / s[0]:.3f}, same-root {same:.3f}")
        for col in ("proc", "kec", "depc", "born"):
            g = f.groupby(col)[["xy", "n"]].sum()
            print(f"  by {col}: " + ", ".join(f"{i}: {v.xy / f.xy.sum():.3f} ({v.n:.0f})" for i, v in g.iterrows()))
        al = sum(tot["split"]["all"].values())
        for n_ in ("attach", "track", "own"):
            print(f"  same-root product {n_}: {sum(tot['split'][n_].values()) / al:.3f}; by KE class " + ", ".join(
                f"{i}: {tot['split'][n_].get(i, 0) / v:.3f}" for i, v in sorted(tot["split"]["all"].items())))
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill")
    f.add_argument("--arm", action="append", required=True, metavar="NAME=PATTERN", help="arms named g4* get the root split")
    f.add_argument("--shards", required=True, help="first-last:step")
    f.add_argument("--out", required=True)
    f.add_argument("--counters")
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    a = p.parse_args()
    if a.cmd == "fill":
        lo, _, st = a.shards.partition(":")
        x, _, y = lo.partition("-")
        fill(dict(s.split("=", 1) for s in a.arm), list(range(int(x), int(y or x) + 1, int(st or 1))), a.out, a.counters)
    else:
        report(a.runs)


if __name__ == "__main__":
    main()
