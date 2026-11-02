"""Independent donors behind the hard draws: library counts of hard muons and of step trees that join crossings, and what
two seeded draws over a set of shards actually use (repeat rate, effective donors, overlap between the seeds).

Effective donors: N_eff = (sum c)^2 / sum c^2 over donors drawn c times, and with energy weight (sum c e)^2 / sum (c e)^2.
A donor is a library muon for the hard arm, a library crossing with a joining tree above --e-min for the anchored arm.

    python -m valhist.corr_donors --library L --sidecar S --templates T --transport 'DIR/fastC.{s}.root' --shards 22-120:2 \
        --mode hard|anchor --out J
"""
import argparse
import json
from pathlib import Path

import numpy as np

from crvfastsim import geometry, tables, tier1b as T
from . import descendant_templates as DT
from . import descendant_topology as D
from . import steplayers as S

BASE = dict(context=True, contain=True, point=True, topology=True, chord=True)
CUTS = (3.0, 10.0, 30.0, 100.0, 300.0)


def _neff(c, e=None):
    w = c.astype(float) if e is None else c * e
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))


def library_counts(lib, geom, mode):
    g = lib["xgeo"]
    _, mu = np.unique(g["mu"], return_inverse=True)
    ld = lib["libdesc"]
    out = {"crossings": int(len(g["bar"])), "muons": int(mu.max() + 1), "desc_rows": int(len(ld)),
           "desc_edep": float(ld.edep.sum())}
    de = np.bincount(mu[ld.xid.to_numpy()], weights=ld.edep.to_numpy(), minlength=out["muons"])
    out["muons_above"] = {str(c): int((de > c).sum()) for c in CUTS}
    out["muon_energy_above"] = {str(c): float(de[de > c].sum() / de.sum()) for c in CUTS}
    if mode == "anchor":
        A = lib["anchor"]
        fwd = A["dk"] > 0
        tree = np.unique(ld.root.to_numpy() + mu[ld.xid.to_numpy()] * (1 << 31), return_inverse=True)[1]
        multi = np.zeros(tree.max() + 1, bool)
        multi[tree[fwd]] = True
        e = np.bincount(tree, weights=ld.edep.to_numpy())
        out.update(trees=int(len(e)), trees_multi=int(multi.sum()), forward_edep_share=A["forward_edep_share"],
                   multi_edep_share=float(e[multi].sum() / e.sum()),
                   trees_multi_above={str(c): int((e[multi] > c).sum()) for c in CUTS},
                   multi_energy_above={str(c): float(e[multi][e[multi] > c].sum() / e.sum()) for c in CUTS})
    return out, de, mu


def run(a):
    geom = geometry.from_offline_config(geometry.find_counters(S.COUNTERS))
    lib = T.read_library(a.library)
    lib = T.attach_topology(lib, *D.read_sidecar(a.sidecar, a.library), geom)
    lib = T.attach_templates(lib, DT.read_sidecar(a.templates, a.library), geom)
    lib = T.attach_hard(lib, geom)
    opts = dict(BASE)
    if a.mode == "hard":
        opts["hard"] = True
    else:
        lib = T.attach_anchors(lib, geom)
        opts["anchor"] = True
    res, de, mu = library_counts(lib, geom, a.mode)
    res["hard_muons"], res["hard_energy_share"] = lib["hard"]["n_hard"], lib["hard"]["hard_energy_share"]
    res["hard_keys"] = int(len(lib["hard"]["kcat"]))
    cnt = lib["hard"]["kcat"].cnt.to_numpy()
    res["hard_per_key_q"] = np.quantile(cnt, [0.5, 0.9, 0.99]).tolist()
    n = len(de)
    use = {}
    ld = lib["libdesc"]
    for seed in (0, 7):
        c = np.zeros(n if a.mode == "hard" else len(lib["libxing"]), np.int64)
        for k in a.shards:
            f = a.transport.format(s=S.shard(k), k=k)
            fx = tables.read(f, "steps")
            tr = {}
            rng = np.random.default_rng([k, 1000, seed] if seed else [k, 1000])
            T._desc_for_crossings(fx, lib, geom, rng, trace=tr, drop_unvisited=True, **opts)
            p = tr["pick"]
            if a.mode == "hard":
                ishard = lib["hard"]["is_hard_xing"][p]
                mu_fast = fx.groupby(T.KEY + ["sp"], sort=False).ngroup().to_numpy()
                # one count per fast muon that took a hard library muon
                u = np.unique(np.c_[mu_fast[ishard], mu[p[ishard]]], axis=0)
                np.add.at(c, u[:, 1], 1)
            else:
                np.add.at(c, p, 1)
        use[seed] = c
    if a.mode == "hard":
        donors = lib["hard"]["is_hard_xing"]
        hm = np.zeros(n, bool)
        hm[mu[donors]] = True
        en = de
        sel = {"hard": hm}
    else:
        A = lib["anchor"]
        fwd = A["dk"] > 0
        en = np.bincount(A["axid"][fwd], weights=ld.edep.to_numpy()[fwd], minlength=len(lib["libxing"]))
        sel = {f"forward>{c:g}": en > c for c in CUTS}
    res["draws"] = {}
    for name, m in sel.items():
        c0, c1 = use[0][m], use[7][m]
        e = en[m]
        res["draws"][name] = {"library_donors": int(m.sum()), "draws": int(c0.sum()), "used": int((c0 > 0).sum()),
                              "repeat_rate": float(1 - (c0 > 0).sum() / max(c0.sum(), 1)), "neff": _neff(c0),
                              "neff_energy": _neff(c0, e), "library_neff_energy": _neff(np.ones(len(e)), e),
                              "overlap_seeds": float(((c0 > 0) & (c1 > 0)).sum() / max((c0 > 0).sum(), 1)),
                              "max_draws": int(c0.max()) if len(c0) else 0}
    print(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for n in ("library", "sidecar", "templates", "transport", "out"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--shards", required=True, help="first-last:step")
    p.add_argument("--mode", choices=("hard", "anchor"), required=True)
    a = p.parse_args()
    lo, _, st = a.shards.partition(":")
    x, _, y = lo.partition("-")
    a.shards = list(range(int(x), int(y or x) + 1, int(st or 1)))
    run(a)


if __name__ == "__main__":
    main()
