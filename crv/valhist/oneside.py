"""Events with CRV digis in only one of two arms on the same generated events: classify them by the CRY primary
(gen tables: primary muon present or not, pdg of the leading primary) and count their digis, per class.

    python -m valhist.oneside --ref REF.root ... --cand CAND.root ... --gen GEN.root ...
"""
import argparse
import json

import numpy as np

from crvfastsim import tables
from .deficit import _evkey, _read


def digis_per_event(paths):
    d = _read(paths, "digis", ["subrun", "event"])
    k, n = np.unique(_evkey(d), return_counts=True)
    return dict(zip(k.tolist(), n.tolist()))


def bars_per_event(paths):
    """event key -> set of bars with digis"""
    d = _read(paths, "digis", ["subrun", "event", "bar"])
    k, b = _evkey(d), np.asarray(d.bar)
    out = {}
    for kk, bb in zip(k.tolist(), b.tolist()):
        out.setdefault(kk, set()).add(bb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R, C = digis_per_event(a.ref), digis_per_event(a.cand)
    g = tables.read_many(a.gen, "gen", tables.KEY + ["pdg", "e"])
    g["key"] = (g.subrun.to_numpy().astype(np.int64) << 20) | g.event.to_numpy().astype(np.int64)
    has_mu = g.assign(mu=g.pdg.abs() == tables.PRIMARY_MUON_PDG).groupby("key").mu.any()
    lead = g.sort_values("e", ascending=False).drop_duplicates("key").set_index("key").pdg

    def cls(k):
        if bool(has_mu.get(k, False)):
            return "muon"
        p = int(abs(lead.get(k, 0)))
        return {22: "gamma", 2112: "neutron", 11: "electron", 2212: "proton"}.get(p, "other")

    out = {}
    for name, keys, dig in (("ref_only", set(R) - set(C), R), ("cand_only", set(C) - set(R), C),
                            ("both_ref", set(R) & set(C), R), ("both_cand", set(R) & set(C), C)):
        row = {}
        for k in keys:
            c = cls(k)
            e = row.setdefault(c, [0, 0])
            e[0] += 1
            e[1] += dig[k]
        out[name] = {c: {"events": v[0], "digis": v[1]} for c, v in sorted(row.items())}
    # muon one-side events: sector group of their digis and the primary muon's energy and zenith
    from crvfastsim import geometry
    from crvfastsim.analyses import v2compare as V2
    geom = geometry.from_offline_config(geometry.find_counters(V2.DEFAULT_COUNTERS))
    grp = np.full(len(geom.sector), "", object)
    for name, secs in V2.GROUPS.items():
        grp[np.isin(geom.sector, secs)] = name
    BR, BC = bars_per_event(a.ref), bars_per_event(a.cand)
    mu = tables.primary_muons(g.assign(px=0.0) if "px" not in g else g)
    mu_e = mu.set_index((mu.subrun.to_numpy().astype(np.int64) << 20) | mu.event.to_numpy().astype(np.int64)).e
    for name, keys, B in (("ref_only", set(R) - set(C), BR), ("cand_only", set(C) - set(R), BC)):
        mk = [k for k in keys if cls(k) == "muon"]
        sec = {}
        for k in mk:
            gs = sorted({grp[b] for b in B[k]})
            sec["+".join(gs)] = sec.get("+".join(gs), 0) + 1
        e = np.array([mu_e.get(k, np.nan) for k in mk])
        out[name]["muon_detail"] = {"groups": dict(sorted(sec.items(), key=lambda x: -x[1])),
                                    "nbars_median": float(np.median([len(B[k]) for k in mk])) if mk else 0,
                                    "mu_e_quartiles_MeV": np.nanpercentile(e, [25, 50, 75]).round(0).tolist() if mk else []}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
