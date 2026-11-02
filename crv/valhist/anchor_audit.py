"""Where the descendant energy of root-coherent (anchored) tier-1b draws goes, per sector group of the fast crossing that
draws it, against the independent draw on the same crossings and the G4 descendant energy of the same events.

Status of each anchored row: own (its tree starts on the drawn crossing), forward (placed on a later crossing of the fast
muon), kept (an unfed crossing's row entered from behind), free (no crossing with its symbol: placed at its offset from
the anchor), and whether it found a bar (dropped otherwise). Sums over shards, energies in MeV.

    python -m valhist.anchor_audit --library L --sidecar S --templates T --transport 'DIR/fastC.{s}.root' \
        --reference 'DIR/g4all.{s}.root' --shards 22-40:2 [--hard] --out J
"""
import argparse
import json
from pathlib import Path

import numpy as np

from crvfastsim import geometry, tables, tier1b as T
from crvfastsim.analyses.census import SECTOR_GROUP, classify
from . import descendant_templates as DT
from . import descendant_topology as D
from . import steplayers as S

BASE = dict(context=True, contain=True, point=True, topology=True, chord=True, drop_unvisited=True)


def _by_group(geom, bars, w):
    g = SECTOR_GROUP[geom.sector[np.asarray(bars, np.int64)]]
    return {k: float(w[g == k].sum()) for k in T.GROUPS}


def _ranks(nx):
    prv = np.full(len(nx), -1, np.int64)
    prv[nx[nx >= 0]] = np.flatnonzero(nx >= 0)
    rank = np.zeros(len(nx), np.int64)
    cur, r = np.flatnonzero(prv < 0), 0
    while len(cur):
        rank[cur] = r
        cur = nx[cur]
        cur, r = cur[cur >= 0], r + 1
    return rank


def _rank_counts(geom, bars, rank):
    return {g: np.bincount(np.minimum(rank[SECTOR_GROUP[geom.sector[bars]] == g], 5), minlength=6).tolist()
            for g in T.GROUPS}


def run(a):
    geom = geometry.from_offline_config(geometry.find_counters(S.COUNTERS))
    lib = T.read_library(a.library)
    lib = T.attach_topology(lib, *D.read_sidecar(a.sidecar, a.library), geom)
    lib = T.attach_templates(lib, DT.read_sidecar(a.templates, a.library), geom)
    lib = T.attach_hard(lib, geom)
    lib = T.attach_anchors(lib, geom, a.scope)
    res = {}
    for k in a.shards:
        fx = tables.read(a.transport.format(s=S.shard(k), k=k), "steps")
        rc = _rank_counts(geom, fx.bar.to_numpy(), _ranks(T._next_crossing(fx, geom)))
        res["fast/ranks"] = {g: (np.add(res["fast/ranks"][g], rc[g]).tolist() if "fast/ranks" in res else rc[g])
                             for g in T.GROUPS}
        for name, extra in (("plain", {}), ("anchor", {"anchor": True})):
            tr = {}
            T._desc_for_crossings(fx, lib, geom, np.random.default_rng([k, 1000]), trace=tr, hard=a.hard, **BASE, **extra)
            xb = fx.bar.to_numpy()[tr["xi"]]
            e, kept = tr["source_edep"], tr["kept"]
            if name == "plain":
                cats = {"all": np.ones(len(e), bool)}
            else:
                dk, free = tr["anchor_dk"], tr["anchor_free"]
                on_pick = tr["xid"] == tr["pick"][tr["xi"]]
                how = tr["anchor_how"]
                cats = {"own": dk == 0, "forward": (dk > 0) & ~free & ~on_pick, "kept": (dk > 0) & on_pick & ~free,
                        "free": free, "dk1": (dk == 1) & ~free & ~on_pick, "dk2+": (dk > 1) & ~free & ~on_pick,
                        "exact": how == 1, "walked": how == 2, "anchor_layer": how == 3}
            for c, m in cats.items():
                for st, mm in (("placed", m & kept), ("dropped", m & ~kept)):
                    key = f"{name}/{c}/{st}"
                    v = _by_group(geom, xb[mm], e[mm])
                    res[key] = {g: res.get(key, {}).get(g, 0.0) + v[g] for g in T.GROUPS}
        if k == a.shards[0]:                          # library side: energy entered from behind by distance, per group
            ld, A = lib["libdesc"], lib["anchor"]
            xb = lib["xgeo"]["bar"][ld.xid.to_numpy()]
            e, dk = ld.edep.to_numpy(), A["dk"]
            for c, m in (("all", np.ones(len(e), bool)), ("dk1", dk == 1), ("dk2+", dk > 1)):
                res[f"library/{c}"] = _by_group(geom, xb[m], e[m])
            res["library/ranks"] = _rank_counts(geom, lib["xgeo"]["bar"], _ranks(A["next"]))
        ref = a.reference.format(s=S.shard(k), k=k)
        st = tables.read(ref, "steps", tables.KEY + ["bar", "sp", "edep"])
        pa = tables.read(ref, "particles", tables.KEY + ["key", "parent", "pdg"])
        c = classify(st, pa, geom.sector)
        m = (c.cls == "mu_descendant").to_numpy()
        v = _by_group(geom, c.bar.to_numpy()[m], c.edep.to_numpy()[m])
        res["g4/desc"] = {g: res.get("g4/desc", {}).get(g, 0.0) + v[g] for g in T.GROUPS}
    print(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for n in ("library", "sidecar", "templates", "transport", "reference", "out"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--shards", required=True, help="first-last:step")
    p.add_argument("--hard", action="store_true")
    p.add_argument("--scope", default="module", choices=("module", "muon"))
    a = p.parse_args()
    lo, _, st = a.shards.partition(":")
    x, _, y = lo.partition("-")
    a.shards = list(range(int(x), int(y or x) + 1, int(st or 1)))
    run(a)


if __name__ == "__main__":
    main()
