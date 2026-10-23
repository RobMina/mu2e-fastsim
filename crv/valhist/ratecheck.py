"""Muon-led events with CRV steps: G4 vs the stored fast tables vs fresh stage-C reruns of the fast transport (several
seeds), and the overlaps. Tells whether the fast transport delivers fewer muons to the bars (a rate deficit) and
whether the events it delivers are a subset of G4's (narrow scattering) or a reshuffle (random swaps).

    python -m valhist.ratecheck --g4 crvsteps...root ... --fast fast1b...root ... --gen gen...root ... [--seeds 3]
"""
import argparse
import json

import numpy as np

from crvfastsim import geometry, tables, transport as TR
from crvfastsim.analyses import v2compare as V2
from crvfastsim.analyses.transport import default_gdml
from mu2efastsim import world as world_mod


def _key(df):
    return (df.subrun.to_numpy().astype(np.int64) << 20) | df.event.to_numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", nargs="+", required=True)
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--margin", type=float, default=None, help="diagnostic: near_crv footprint margin [mm] (default 1500)")
    ap.add_argument("--msc-tail", action="store_true")
    ap.add_argument("--tail-k", type=float, default=None)
    ap.add_argument("--sliced-msc", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.margin is not None:   # diagnostic only: transport() calls near_crv(o, d, geom) with the module default
        _orig = TR.near_crv
        TR.near_crv = lambda o, d, geom, margin=a.margin: _orig(o, d, geom, margin)
    geom = geometry.from_offline_config(geometry.find_counters(V2.DEFAULT_COUNTERS))
    world = [p for p in world_mod.load(default_gdml(), y_min=TR.Y_MSC_PLANE) if p[0] != "G4_POLYSTYRENE"]
    out = {"per_file": []}
    tot = {}
    for gf, ff, nf in zip(a.g4, a.fast, a.gen):
        gs = tables.read(gf, "steps", tables.KEY + ["sp"])
        gp = tables.read(gf, "particles", tables.KEY + ["key", "pdg", "isprim"])
        gs = gs.merge(gp.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], how="left")
        G = set(_key(gs[(gs.isprim == 1) & (gs.pdg.abs() == 13)]))
        fs = tables.read(ff, "steps", tables.KEY + ["sp"])
        fp = tables.read(ff, "particles", tables.KEY + ["key", "pdg", "isprim"])
        fs = fs.merge(fp.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], how="left")
        F = set(_key(fs[(fs.isprim == 1) & (fs.pdg.abs() == 13)]))
        gen = tables.read(nf, "gen")
        row = {"file": nf.rsplit("/", 1)[-1], "g4": len(G), "stored": len(F), "stored_and_g4": len(F & G)}
        R = []
        for s in range(a.seeds):
            opts = {k: v for k, v in (("msc_tail", a.msc_tail), ("tail_k", a.tail_k), ("sliced_msc", a.sliced_msc)) if v}
            st, _ = TR.transport(gen, geom, np.random.default_rng([97, s, len(out["per_file"])]), stage="C", world=world,
                                 **opts)
            R.append(set(_key(st)))
        row["rerun"] = [len(r) for r in R]
        row["rerun_and_g4"] = [len(r & G) for r in R]
        row["rerun_and_stored"] = [len(r & F) for r in R]
        row["rerun_pair_overlap"] = len(R[0] & R[1]) if len(R) > 1 else None
        out["per_file"].append(row)
        for k, v in row.items():
            if k == "file":
                continue
            tot[k] = (np.add(tot[k], v) if k in tot else np.array(v, float)) if v is not None else None
        print(row, flush=True)
    out["total"] = {k: (v.tolist() if v is not None else None) for k, v in tot.items()}
    print(json.dumps(out["total"], indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
