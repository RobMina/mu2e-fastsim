"""tier1b-build / tier1b-apply -- the G4 library of muon descendants and non-muon primaries (crvfastsim/tier1b.py).

    tier1b-build crvsteps.<train>.root [...] --gen gen.<train>.root [...] --out lib.root
    tier1b-apply fastC.*.root --library lib.root --outdir DIR [--seed N]   (writes fast1b.<shard>.root)

Build on training shards and apply to others (the library never sees the events it is compared with). The fast
tables carry their gen tree, which supplies the non-muon CRY primaries.
"""
import json
import os

import numpy as np

from .. import geometry, tables, tier1b

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def _geom(counters):
    return geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))


def build(inputs, outdir, gen=None, out=None, counters=None, **kw):
    if not gen or not out:
        raise SystemExit("tier1b-build needs --gen and --out")
    lib, info = tier1b.build(tables.expand(inputs), tables.expand(gen), _geom(counters))
    tier1b.write_library(out, lib)
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump({"inputs": tables.expand(inputs), "gen": tables.expand(gen), **info}, fh, indent=2)
    print(info)


def apply(inputs, outdir, library=None, seed=1, counters=None, **kw):
    if not library:
        raise SystemExit("tier1b-apply needs --library")
    geom = _geom(counters)
    lib = tier1b.read_library(library)
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), 1_000 + i])
        st, pa, gen = tables.read(f, "steps"), tables.read(f, "particles"), tables.read(f, "gen")
        sub = tables.read(f, "subruns")
        steps, parts, info = tier1b.sample(st, pa, gen, lib, geom, rng)
        name = os.path.basename(f).replace("fastC.", "fast1b.").replace("fastB.", "fast1b.").replace("fastA.", "fast1b.")
        tables.write(os.path.join(outdir, name), steps, parts, gen, sub, as_ttree=True)
        print(f"# {f}: {info}", flush=True)


GROUPS = ("EX", "T1", "T2", "M")
EXTRA_Q = (0.5, 0.9, 0.99)


def _per_crossing(paths, geom):
    """Per primary-muon crossing: muon visible energy and the descendants' visible energy in the same bar; plus the
    descendant energy outside the muon's bars and the non-muon energy, per sector group (totals)."""
    from .census import SECTOR_GROUP, classify
    steps = tables.read_many(paths, "steps", tables.KEY + ["bar", "sp", "edep"])
    parts = tables.read_many(paths, "particles", tables.KEY + ["key", "parent", "pdg"])
    parts["isprim"] = (parts.parent < 0).astype(int)
    s = classify(steps, parts, geom.sector)
    s["group"] = SECTOR_GROUP[geom.sector[s.bar.to_numpy()]]
    mu = s[s.cls == "mu_primary"].groupby(tables.KEY + ["bar"]).edep.sum().rename("mu").reset_index()
    d = s[s.cls == "mu_descendant"].groupby(tables.KEY + ["bar"]).edep.sum().rename("extra").reset_index()
    x = mu.merge(d, on=tables.KEY + ["bar"], how="left").fillna({"extra": 0.0})
    x["group"] = SECTOR_GROUP[geom.sector[x.bar.to_numpy()]]
    out = d.merge(mu[tables.KEY + ["bar"]].assign(m=1), on=tables.KEY + ["bar"], how="left")
    out = out[out.m.isna()]
    out["group"] = SECTOR_GROUP[geom.sector[out.bar.to_numpy()]]
    nm = s[s.cls.str.startswith("nonmu")]
    return x, out.groupby("group").extra.sum(), nm.groupby("group").edep.sum(), \
        nm.groupby(tables.KEY + ["bar"]).edep.sum().reset_index().assign(group=lambda f: SECTOR_GROUP[geom.sector[f.bar.to_numpy()]])


def check(inputs, outdir, compare=None, counters=None, **kw):
    """tier1b-check FAST1B.root [...] --compare crvsteps.root [...]: step-level tier-1b closure per sector group."""
    from mu2efastsim import plots
    if not compare:
        raise SystemExit("tier1b-check needs --compare <G4 CrvStep tables of the same (held-out) shards>")
    geom = _geom(counters)
    xf, of, nf, bf = _per_crossing(tables.expand(inputs), geom)
    xr, orr, nr, br = _per_crossing(tables.expand(compare), geom)
    res = {}
    for g in GROUPS:
        a, b = xf[xf.group == g], xr[xr.group == g]
        if not len(b):
            continue
        res[g] = {"crossings_fast": len(a), "crossings_ref": len(b),
                  "extra_over_muon_fast": float(a.extra.sum() / a.mu.sum()), "extra_over_muon_ref": float(b.extra.sum() / b.mu.sum()),
                  "frac_with_extra_fast": float((a.extra > 0).mean()), "frac_with_extra_ref": float((b.extra > 0).mean()),
                  "extra_q_fast": [float(v) for v in np.quantile(a.extra, EXTRA_Q)],
                  "extra_q_ref": [float(v) for v in np.quantile(b.extra, EXTRA_Q)],
                  "bar_total_q_fast": [float(v) for v in np.quantile(a.mu + a.extra, EXTRA_Q)],
                  "bar_total_q_ref": [float(v) for v in np.quantile(b.mu + b.extra, EXTRA_Q)],
                  "desc_outside_muon_bars_per_crossing_fast": float(of.get(g, 0.0) / len(a)),
                  "desc_outside_muon_bars_per_crossing_ref": float(orr.get(g, 0.0) / len(b)),
                  "nonmu_edep_fast_over_ref": float(nf.get(g, 0.0) / max(nr.get(g, 0.0), 1e-9)),
                  "nonmu_bar_hits_fast_over_ref": float((bf.group == g).sum() / max((br.group == g).sum(), 1))}
    plots.write_json(outdir, "tier1b_check.json", res)
    for g, r in res.items():
        print(f"{g:3s} extra/mu {r['extra_over_muon_ref']:.3f}->{r['extra_over_muon_fast']:.3f}  with extra "
              f"{r['frac_with_extra_ref']:.3f}->{r['frac_with_extra_fast']:.3f}  extra q50/90/99 "
              f"{np.round(r['extra_q_ref'], 2)}->{np.round(r['extra_q_fast'], 2)}  bar total q50/90/99 "
              f"{np.round(r['bar_total_q_ref'], 2)}->{np.round(r['bar_total_q_fast'], 2)}  desc outside/xing "
              f"{r['desc_outside_muon_bars_per_crossing_ref']:.3f}->{r['desc_outside_muon_bars_per_crossing_fast']:.3f}  "
              f"nonmu edep f/r {r['nonmu_edep_fast_over_ref']:.3f}, bar hits f/r {r['nonmu_bar_hits_fast_over_ref']:.3f}")
    return res
