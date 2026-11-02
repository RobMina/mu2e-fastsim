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


def run(a):
    geom = geometry.from_offline_config(geometry.find_counters(S.COUNTERS))
    lib = T.read_library(a.library)
    lib = T.attach_topology(lib, *D.read_sidecar(a.sidecar, a.library), geom)
    lib = T.attach_templates(lib, DT.read_sidecar(a.templates, a.library), geom)
    lib = T.attach_hard(lib, geom)
    lib = T.attach_anchors(lib, geom)
    res = {}
    for k in a.shards:
        fx = tables.read(a.transport.format(s=S.shard(k), k=k), "steps")
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
                cats = {"own": dk == 0, "forward": (dk > 0) & ~free & ~on_pick, "kept": (dk > 0) & on_pick & ~free,
                        "free": free}
            for c, m in cats.items():
                for st, mm in (("placed", m & kept), ("dropped", m & ~kept)):
                    key = f"{name}/{c}/{st}"
                    v = _by_group(geom, xb[mm], e[mm])
                    res[key] = {g: res.get(key, {}).get(g, 0.0) + v[g] for g in T.GROUPS}
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
    a = p.parse_args()
    lo, _, st = a.shards.partition(":")
    x, _, y = lo.partition("-")
    a.shards = list(range(int(x), int(y or x) + 1, int(st or 1)))
    run(a)


if __name__ == "__main__":
    main()
