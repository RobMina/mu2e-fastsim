"""Which events populate a cluster-PE range: clusters of a chain file (valhist.fasttables) joined to the steps of their event
and sector. Class per cluster: multi (two or more primary muons cross the sector), shower (non-muon energy above FRAC of the
sector's), delta (muon-descendant energy above FRAC), muon (the rest).

    python -m valhist.pe_tail --arm g4='CHAIN.{s}.root':'STEPS.{s}.root' --arm ... --shards 22-120:2 --out J
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from crvfastsim import geometry, tables
from crvfastsim.analyses.census import classify
from . import steplayers as S

K = tables.KEY
PE_EDGES = np.array([480.0, 640.0, 800.0, 1200.0, 2000.0])
FRAC = 0.25
CLASSES = ("muon", "delta", "shower", "multi")


def sector_table(path, geom):
    st = tables.read(path, "steps", K + ["bar", "sp", "edep", "len"])
    pa = tables.read(path, "particles", K + ["key", "parent", "pdg"])
    s = classify(st, pa, geom.sector)
    s["sec"] = np.minimum(geom.sector[s.bar.to_numpy()], 3) + 1   # the clusters' sector type: EX, T1, T2, taggers
    s["mu"] = np.where(s.cls == "mu_primary", s.edep, 0.0)
    s["de"] = np.where(s.cls == "mu_descendant", s.edep, 0.0)
    s["nm"] = np.where(s.cls.str.startswith("nonmu"), s.edep, 0.0)
    s["path"] = np.where(s.cls == "mu_primary", s.len, 0.0)
    t = s.groupby(K + ["sec"], sort=False)[["mu", "de", "nm", "path"]].sum().reset_index()
    m = s[s.cls == "mu_primary"].groupby(K + ["sec"], sort=False).agg(nmu=("prim_key", "nunique"), ncr=("bar", "nunique")).reset_index()
    return t.merge(m, on=K + ["sec"], how="left").fillna({"nmu": 0, "ncr": 0})


def one(chain, steps, geom):
    with uproot.open(chain) as f:
        c = f["crvdump/clusters"].arrays(["run", "subrun", "event", "sectortype", "pe"], library="pd")
    c = c.rename(columns={"sectortype": "sec"})
    for k in K:
        c[k] = c[k].astype(np.int64)
    t = sector_table(steps, geom)
    for k in K:
        t[k] = t[k].astype(np.int64)
    j = c.merge(t, on=K + ["sec"], how="left").fillna(0.0)
    tot = np.maximum(j.mu + j.de + j.nm, 1e-9)
    j["cl"] = np.select([j.nmu >= 2, j.nm / tot > FRAC, j.de / tot > FRAC], [3, 2, 1], 0)
    j["pc"] = np.searchsorted(PE_EDGES, j.pe.to_numpy())
    g = j.groupby(["pc", "cl"]).agg(n=("pe", "size"), mu=("mu", "sum"), de=("de", "sum"), nm=("nm", "sum"), path=("path", "sum"),
                                    ncr=("ncr", "sum"), unm=("nmu", lambda v: int((v == 0).sum()))).reset_index()
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=CHAIN:STEPS")
    p.add_argument("--shards", required=True, help="first-last:step")
    p.add_argument("--out")
    a = p.parse_args()
    lo, _, st = a.shards.partition(":")
    x, _, y = lo.partition("-")
    shards = list(range(int(x), int(y or x) + 1, int(st or 1)))
    geom = geometry.from_offline_config(geometry.find_counters(S.COUNTERS))
    res = {}
    for s in a.arm:
        name, _, rest = s.partition("=")
        ch, _, stp = rest.partition(":")
        rows = [one(ch.format(s=S.shard(k), k=k), stp.format(s=S.shard(k), k=k), geom).assign(shard=k) for k in shards]
        res[name] = pd.concat(rows, ignore_index=True)
    arms = list(res)
    ref = res[arms[0]]
    print("PE class edges", PE_EDGES.tolist(), "; class", CLASSES, "; counts per arm, ratio to the first with shard-jackknife error")
    n = len(shards)
    for pc in sorted(ref.pc.unique()):
        for cl in sorted(ref.cl.unique()):
            r0 = ref[(ref.pc == pc) & (ref.cl == cl)]
            if not len(r0):
                continue
            c = [f"pe{pc} {CLASSES[cl]:>6} n {int(r0.n.sum()):7d} mu {r0.mu.sum() / r0.n.sum():6.1f} de {r0.de.sum() / r0.n.sum():6.1f} "
                 f"nm {r0.nm.sum() / r0.n.sum():6.1f} path {r0.path.sum() / r0.n.sum():6.0f} ncr {r0.ncr.sum() / r0.n.sum():5.1f} nomatch {int(r0.unm.sum())}"]
            for a2 in arms[1:]:
                q = res[a2]
                q = q[(q.pc == pc) & (q.cl == cl)]
                d = q.groupby("shard").n.sum().reindex(shards).fillna(0).to_numpy() - r0.groupby("shard").n.sum().reindex(shards).fillna(0).to_numpy()
                c.append(f"{a2} {q.n.sum() / r0.n.sum():6.3f}+-{np.sqrt(n / (n - 1) * ((d - d.mean()) ** 2).sum()) / r0.n.sum():.3f}")
            print(" ".join(c))
    if a.out:
        Path(a.out).write_text(json.dumps({k: v.to_dict(orient="list") for k, v in res.items()}))


if __name__ == "__main__":
    main()
