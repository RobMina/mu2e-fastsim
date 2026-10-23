"""CRV visible energy and muon delivery of fast step tables against the G4 tables on the same generated events.

Per arm: total visible energy (steps.edep), events with CRV steps, events with steps of a primary muon, and against G4:
events with G4 primary-muon steps that the arm misses, and the G4 energy in them.

    python -m valhist.stepsum --g4 'crvsteps.*' --arm NAME='dir/fast1b.*' [--arm ...] [--shards 11-20]
"""
import argparse
import json

import numpy as np

from crvfastsim import tables


def _key(df):
    return (df.subrun.to_numpy().astype(np.int64) << 20) | df.event.to_numpy().astype(np.int64)


def load(paths):
    s = tables.read_many(paths, "steps", tables.KEY + ["sp", "edep"])
    p = tables.read_many(paths, "particles", tables.KEY + ["key", "pdg", "isprim"])
    s["k"], p["k"] = _key(s), _key(p)
    s = s.merge(p[["k", "key", "pdg", "isprim"]].rename(columns={"key": "sp"}), on=["k", "sp"], how="left")
    s["prim_mu"] = (s.isprim == 1) & (s.pdg.abs() == tables.PRIMARY_MUON_PDG)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", required=True, help="pattern with {s} for the subrun, e.g. dir/crvsteps.{s}.root")
    ap.add_argument("--arm", action="append", required=True, help="NAME=pattern with {s}")
    ap.add_argument("--shards", default="11-20")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.shards.split("-"))
    sub = [f"001400_{k:08d}" for k in range(lo, hi + 1)]
    G = load([a.g4.format(s=s) for s in sub])
    gE = G.groupby("k").edep.sum()
    gmu = set(G[G.prim_mu].k)
    out = {"g4": {"edep_GeV": G.edep.sum() / 1e3, "events": int(G.k.nunique()), "mu_events": len(gmu)}}
    for spec in a.arm:
        name, pat = spec.split("=", 1)
        F = load([pat.format(s=s) for s in sub])
        fmu = set(F[F.prim_mu].k)
        miss = gmu - set(F.k)
        out[name] = {"edep_GeV": F.edep.sum() / 1e3, "edep_rel_g4": F.edep.sum() / G.edep.sum() - 1,
                     "events": int(F.k.nunique()), "mu_events": len(fmu),
                     "mu_events_rel_g4": len(fmu) / len(gmu) - 1, "mu_events_err": float(np.sqrt(1 / len(fmu) + 1 / len(gmu))),
                     "g4_mu_events_missed": len(miss), "g4_GeV_in_missed": float(gE.reindex(list(miss)).sum() / 1e3)}
    out = json.loads(json.dumps(out, default=float))
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
