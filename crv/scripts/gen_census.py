"""Census of gen tables (CRY primaries of every generated event): per-file integrity and physics summary, and a
comparison between two gen sets (e.g. crygen_v1 vs g4ref_v1).

usage: python gen_census.py OUT.json SET=GLOB [SET=GLOB ...]
  e.g. python gen_census.py census.json crygen_v1='crygen_v1_tables/gen.*.root' g4ref_v1='g4ref_v1_tables/gen.*.root'

Per file: tree cycles of dump/gen (the highest is read), rows, events (unique run/subrun/event), subrun vs file name,
duplicate keys, livetime and primaries from dump/subruns, events per live second, primary pdg census, and for the
primary muon of each event: mean energy, cos(zenith) mean, charge ratio. Set level: totals and the spread of the
per-file rates (pull of each file against the set mean, Poisson).
"""
import json
import re
import sys

import numpy as np
import uproot

from crvfastsim import tables

MU = 13


def one(path):
    out = {"file": path.rsplit("/", 1)[-1]}
    with uproot.open(path) as f:
        out["gen_cycles"] = sorted(int(k.split(";")[1]) for k in f["dump"].keys() if k.startswith("gen;"))
        g = f["dump/gen"].arrays(["run", "subrun", "event", "pdg", "py", "e"], library="pd")
        s = f["dump/subruns"].arrays(library="pd")
    ev = g[tables.KEY].drop_duplicates()
    m = re.search(r"_(\d{8})\.root$", path)
    out.update(rows=len(g), events=len(ev), subruns=sorted(int(x) for x in g.subrun.unique()),
               subrun_from_name=int(m.group(1)) if m else None,
               runs=sorted(int(x) for x in g.run.unique()),
               livetime=float(s.livetime.sum()), primaries_subruns=int(s.primaries.sum()), n_subrun_rows=len(s))
    out["dup_rows"] = int(g.duplicated(tables.KEY + ["pdg", "e"]).sum())
    out["events_per_s"] = out["events"] / out["livetime"] if out["livetime"] else None
    pdg = g.pdg.value_counts()
    out["pdg"] = {int(k): int(v) for k, v in pdg.items()}
    mu = tables.primary_muons(g)
    p = np.sqrt(np.maximum(mu.e.to_numpy() ** 2 - 105.658 ** 2, 0))
    cz = -mu.py.to_numpy() / np.where(p > 0, p, 1)          # Mu2e y is up: downward muon has py < 0
    out.update(mu_events=len(mu), mu_e_mean=float(mu.e.mean()), mu_e_median=float(mu.e.median()),
               mu_cosz_mean=float(cz.mean()), mu_plus_over_minus=float((mu.pdg == -MU).sum() / max((mu.pdg == MU).sum(), 1)))
    return out


def summarise(rows):
    ev = np.array([r["events"] for r in rows], float)
    lt = np.array([r["livetime"] for r in rows], float)
    rate = ev.sum() / lt.sum()
    pull = (ev - rate * lt) / np.sqrt(rate * lt)
    pdg = {}
    for r in rows:
        for k, v in r["pdg"].items():
            pdg[k] = pdg.get(k, 0) + v
    nrow = sum(r["rows"] for r in rows)
    mu = sum(r["mu_events"] for r in rows)
    w = np.array([r["mu_events"] for r in rows], float)
    return {"files": len(rows), "events": int(ev.sum()), "rows": nrow, "livetime_s": float(lt.sum()),
            "events_per_s": float(rate), "rate_pull_rms": float(pull.std()), "rate_pull_maxabs": float(np.abs(pull).max()),
            "pdg_fraction": {k: v / nrow for k, v in sorted(pdg.items(), key=lambda kv: -kv[1])},
            "mu_events": mu, "mu_fraction_of_events": mu / ev.sum(),
            "mu_e_mean": float(np.average([r["mu_e_mean"] for r in rows], weights=w)),
            "mu_cosz_mean": float(np.average([r["mu_cosz_mean"] for r in rows], weights=w)),
            "mu_plus_over_minus": float(np.average([r["mu_plus_over_minus"] for r in rows], weights=w))}


def problems(rows):
    bad = []
    for r in rows:
        why = []
        if r["subruns"] != [r["subrun_from_name"]]:
            why.append(f"subruns {r['subruns']} != name {r['subrun_from_name']}")
        if r["n_subrun_rows"] != 1:
            why.append(f"{r['n_subrun_rows']} subrun rows")
        if r["dup_rows"]:
            why.append(f"{r['dup_rows']} duplicate rows")
        if why:
            bad.append({"file": r["file"], "why": why})
    return bad


def main():
    out_path, sets = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    res = {}
    for name, pat in sets.items():
        rows = []
        for p in tables.expand([pat]):
            rows.append(one(p))
            print(f"{name} {rows[-1]['file']} events {rows[-1]['events']} live {rows[-1]['livetime']:.3f}", flush=True)
        res[name] = {"summary": summarise(rows), "problems": problems(rows), "files": rows}
        print(name, json.dumps(res[name]["summary"], indent=1), "problems:", res[name]["problems"], flush=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
