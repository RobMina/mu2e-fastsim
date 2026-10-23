"""Where does a digi/pulse deficit between two arms sit? Both arms are CrvRecoTableDumper files on the same events
(e.g. A0g = G4 steps and A1 = fast steps, both injected by CrvStepsFromTable on the gen stream, or the V2 ref/fast
dumps restricted to common events). Counts per sector group and per bin of SiPM charge (PE), pulse PE and pulse time;
digis per SiPM charge; and per-event matching: of the events with CRV activity in either arm, how many have it in one
only, and how the per-event digi count differs.

    python -m valhist.deficit --ref REF.root ... --cand CAND.root ... [--out OUT.json]
"""
import argparse
import json

import awkward as ak
import numpy as np
import uproot

from crvfastsim import geometry, tables
from crvfastsim.analyses import v2compare as V2

GROUPS = V2.GROUPS


def _read(paths, tree, cols):
    out = []
    for p in tables.expand(paths):
        with uproot.open(p) as f:
            if f"crvdump/{tree}" in f:
                out.append(f[f"crvdump/{tree}"].arrays(cols))
    return ak.concatenate(out) if out else None


def _evkey(a):
    return (ak.to_numpy(a.subrun).astype(np.int64) << 20) | ak.to_numpy(a.event).astype(np.int64)


def _group_of_bar(geom):
    g = np.full(len(geom.sector), "", object)
    for name, secs in GROUPS.items():
        g[np.isin(geom.sector, secs)] = name
    return g


def summarize(paths, grp, events=None):
    d = _read(paths, "digis", ["subrun", "event", "bar", "sipm", "adc"])
    p = _read(paths, "pulses", ["subrun", "event", "bar", "sipm", "pe", "time", "seq"])
    q = _read(paths, "charges", ["subrun", "event", "bar", "sipm", "pe"])
    res = {}
    dk, pk = _evkey(d), _evkey(p)
    if events is not None:
        dsel, psel = np.isin(dk, events), np.isin(pk, events)
        d, p, dk, pk = d[dsel], p[psel], dk[dsel], pk[psel]
        if q is not None:
            qk = _evkey(q)
            q = q[np.isin(qk, events)]
    dg, pg = grp[ak.to_numpy(d.bar)], grp[ak.to_numpy(p.bar)]
    pe = ak.to_numpy(p.pe)
    tm = ak.to_numpy(p.time)
    for g in GROUPS:
        m = pg == g
        res[g] = {"digis": int((dg == g).sum()), "pulses": int(m.sum()), "pulses_seq0": int((m & (ak.to_numpy(p.seq) == 0)).sum()),
                  "pe_hist": np.histogram(pe[m], bins=PE_BINS)[0].tolist(),
                  "time_hist": np.histogram(tm[m], bins=T_BINS)[0].tolist()}
    if q is not None:
        qpe = ak.to_numpy(ak.sum(q.pe, axis=1)) if q.pe.ndim > 1 else ak.to_numpy(q.pe)
        qg = grp[ak.to_numpy(q.bar)]
        for g in GROUPS:
            m = qg == g
            res[g]["sipms_with_charge"] = int(m.sum())
            res[g]["sipm_pe_hist"] = np.histogram(qpe[m], bins=PE_BINS)[0].tolist()
    res["_events_with_digis"] = np.unique(dk)
    res["_digis_per_event"] = dict(zip(*np.unique(dk, return_counts=True)))
    return res


PE_BINS = np.array([0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 120, 200, 400, 1e9])
T_BINS = np.array([-1e9, 0, 400, 450, 500, 600, 800, 1000, 2000, 5000, 1e4, 5e4, 1e5, 1e9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--counters", default=None)
    a = ap.parse_args()
    geom = geometry.from_offline_config(geometry.find_counters(a.counters or V2.DEFAULT_COUNTERS))
    grp = _group_of_bar(geom)
    R, C = summarize(a.ref, grp), summarize(a.cand, grp)
    er, ec = R.pop("_events_with_digis"), C.pop("_events_with_digis")
    nr, nc = R.pop("_digis_per_event"), C.pop("_digis_per_event")
    both = np.intersect1d(er, ec)
    out = {"events_with_digis": {"ref": len(er), "cand": len(ec), "both": len(both),
                                 "ref_only": int(len(er) - len(both)), "cand_only": int(len(ec) - len(both))},
           "digis_in_ref_only_events": int(sum(nr[e] for e in np.setdiff1d(er, both))),
           "digis_in_cand_only_events": int(sum(nc[e] for e in np.setdiff1d(ec, both))),
           "digis_in_common_events": {"ref": int(sum(nr[e] for e in both)), "cand": int(sum(nc[e] for e in both))},
           "pe_bins": PE_BINS.tolist(), "t_bins": T_BINS.tolist(), "groups": {}}
    for g in GROUPS:
        r, c = R[g], C[g]
        rel = lambda x, y: (y - x) / x if x else float("nan")
        err = lambda x, y: np.sqrt(1 / x + 1 / y) if x and y else float("nan")
        row = {k: {"ref": r[k], "cand": c[k], "rel": rel(r[k], c[k]), "err": err(r[k], c[k])}
               for k in ("digis", "pulses", "pulses_seq0") + (("sipms_with_charge",) if "sipms_with_charge" in r else ())}
        for h in ("pe_hist", "time_hist") + (("sipm_pe_hist",) if "sipm_pe_hist" in r else ()):
            row[h] = {"ref": r[h], "cand": c[h], "rel": [rel(x, y) for x, y in zip(r[h], c[h])]}
        out["groups"][g] = row
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}, indent=1, default=int))
    for g, row in out["groups"].items():
        print(g, {k: f"{v['ref']} -> {v['cand']} ({v['rel']:+.4f} +- {v['err']:.4f})" for k, v in row.items()
                  if "hist" not in k})
        for h in ("sipm_pe_hist", "pe_hist", "time_hist"):
            if h in row:
                print("   ", h, " ".join(f"{x:+.3f}" if np.isfinite(x) else "nan" for x in row[h]["rel"]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=int)


if __name__ == "__main__":
    main()
