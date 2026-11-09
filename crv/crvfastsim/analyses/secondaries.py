"""secondaries -- what tier 1b must add: G4 CrvSteps that are not the primary muon's own, from CrvStep tables.

Design input for the tier-1b library (docs/plan.md section 6). Every step is classed by its primary (census.classify).
Muon descendants are split by the creation process of their "root secondary" (the ancestor that is a direct child of
the primary muon), by where that root secondary was born (inside a CRV bar or outside), and by whether the step's bar
is one the primary muon itself crossed in that event ("muon bar"). Per primary-muon crossing: the extra visible energy
its descendants leave in the same bar. Non-muon primaries: per pdg, the gen-level count, the fraction that leave CRV
steps, their visible energy and kinetic energy. Event classes: muon crossing yes/no x other deposits yes/no.

    secondaries crvsteps.*.root [--gen gen.*.root] --outdir DIR
"""
import numpy as np
import pandas as pd

from mu2efastsim import plots

from .. import geometry, tables
from .census import SECTOR_GROUP, classify

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
# Offline ProcessCode ids (MCDataProducts/inc/ProcessCode.hh) of the processes that matter here
PROCESS = {12: "compt", 13: "conv", 14: "Decay", 16: "eBrem", 17: "eIoni", 30: "muBrems", 31: "muIoni", 34: "muPairProd",
           40: "phot", 56: "mu2ePrimary", 111: "DecayWithSpin", 114: "DIO", 115: "NuclearCapture", 116: "muonNuclear",
           32: "muMinusCaptureAtRest", 63: "muMinusAtomicCapture", 21: "hIoni", 58: "hadElastic", 97: "neutronInelastic",
           101: "protonInelastic", 109: "photonNuclear", 108: "electronNuclear"}
MASS = {11: 0.511, 13: 105.658, 22: 0.0, 211: 139.570, 321: 493.677, 2212: 938.272, 2112: 939.565}
EXTRA_EDGES = np.r_[0.0, np.geomspace(0.01, 100.0, 41)]
KE_EDGES = np.geomspace(0.1, 1e5, 61)


def root_child(parts):
    """Per particle row: key of its ancestor whose parent is the primary (itself for a primary's child), -1 for primaries."""
    p = parts[tables.KEY + ["key", "parent"]]
    lut = pd.Series(np.arange(len(p)), index=pd.MultiIndex.from_frame(p[tables.KEY + ["key"]]))
    par = lut.reindex(pd.MultiIndex.from_frame(p[tables.KEY + ["parent"]])).to_numpy()
    par = np.where(np.isnan(par), -1, par).astype(np.int64)
    cur = np.arange(len(p))
    for _ in range(1000):
        pc = par[cur]
        go = (pc >= 0) & (np.where(pc >= 0, par[np.maximum(pc, 0)], -1) >= 0)
        if not go.any():
            break
        cur = np.where(go, pc, cur)
    return np.where(par >= 0, p.key.to_numpy()[cur], -1), cur


def bar_containing(pos, geom, chunk=20000):
    """Bar index whose box contains each point (-1 if none)."""
    out = np.full(len(pos), -1, np.int64)
    for i in range(0, len(pos), chunk):
        d = np.abs(pos[i:i + chunk, None, :] - geom.center[None, :, :]) <= geom.halflen[None, :, :]
        inside = d.all(axis=2)
        hit = inside.any(axis=1)
        out[i:i + chunk][hit] = inside[hit].argmax(axis=1)
    return out


def _pname(code):
    return PROCESS.get(int(code), f"code{int(code)}")


def analyse(paths, geom):
    steps = tables.read_many(paths, "steps", tables.KEY + ["bar", "sp", "edep", "px0", "py0", "pz0"])
    parts = tables.read_many(paths, "particles", tables.KEY + ["key", "parent", "pdg", "ccode", "isprim",
                                                               "sx", "sy", "sz", "se"])
    s = classify(steps, parts, geom.sector)
    rk, rrow = root_child(parts)
    parts = parts.assign(root=rk, root_ccode=parts.ccode.to_numpy()[rrow],
                         root_bar=bar_containing(parts[["sx", "sy", "sz"]].to_numpy()[rrow], geom))
    s = s.merge(parts[tables.KEY + ["key", "root_ccode", "root_bar"]].rename(columns={"key": "sp"}),
                on=tables.KEY + ["sp"], how="left")
    s["group"] = SECTOR_GROUP[geom.sector[s.bar.to_numpy()]]
    tot = float(s.edep.sum())
    out = {"n_events": int(s[tables.KEY].drop_duplicates().shape[0]), "edep_total_MeV": tot}
    out["edep_frac_by_class"] = (s.groupby("cls").edep.sum() / tot).to_dict()
    # muon crossings and the muon's bars per event
    mu = s[s.cls == "mu_primary"]
    xing = mu.groupby(tables.KEY + ["bar"], sort=False).agg(edep=("edep", "sum"), px=("px0", "first"),
                                                            py=("py0", "first"), pz=("pz0", "first")).reset_index()
    xing["p"] = np.sqrt(xing.px ** 2 + xing.py ** 2 + xing.pz ** 2)
    mubar = xing[tables.KEY + ["bar"]].assign(mubar=True)
    s = s.merge(mubar, on=tables.KEY + ["bar"], how="left")
    s["mubar"] = s.mubar.fillna(False).astype(bool)
    # muon descendants
    d = s[s.cls == "mu_descendant"].copy()
    d["proc"] = d.root_ccode.fillna(0).astype(int).map(_pname).to_numpy()
    d["born"] = np.where(d.root_bar.to_numpy() >= 0, np.where(d.root_bar.to_numpy() == d.bar.to_numpy(),
                                                               "same_bar", "other_bar"), "outside_crv_bars")
    dt = float(d.edep.sum()) or 1.0
    out["mu_desc"] = {
        "edep_frac_of_total": dt / tot,
        "by_process": (d.groupby("proc").edep.sum() / dt).sort_values(ascending=False).head(12).to_dict(),
        "by_birth": (d.groupby("born").edep.sum() / dt).to_dict(),
        "in_muon_bar": float(d.loc[d.mubar, "edep"].sum() / dt),
        "by_group_in_muon_bar": {g: float(x.loc[x.mubar, "edep"].sum() / max(x.edep.sum(), 1e-9))
                                 for g, x in d.groupby("group")},
    }
    # extra energy per muon crossing from its descendants in the same bar
    extra = d[d.mubar].groupby(tables.KEY + ["bar"]).edep.sum().rename("extra").reset_index()
    xe = xing.merge(extra, on=tables.KEY + ["bar"], how="left").fillna({"extra": 0.0})
    xe["group"] = SECTOR_GROUP[geom.sector[xe.bar.to_numpy()]]
    out["per_crossing"] = {}
    for g, x in xe.groupby("group"):
        r = x.extra / x.edep
        out["per_crossing"][g] = {"n": int(len(x)), "frac_with_extra": float((x.extra > 0).mean()),
                                  "extra_mean_MeV": float(x.extra.mean()), "muon_edep_mean_MeV": float(x.edep.mean()),
                                  "extra_over_muon_mean": float(x.extra.sum() / x.edep.sum()),
                                  "extra_q50_q90_q99": [float(v) for v in np.quantile(x.extra, [0.5, 0.9, 0.99])],
                                  "ratio_q90": float(np.quantile(r, 0.9))}
    pb = np.quantile(xe.p, [0, 0.25, 0.5, 0.75, 1.0])
    xe["pbin"] = np.clip(np.searchsorted(pb, xe.p, "right") - 1, 0, 3)
    out["extra_vs_p"] = {f"{pb[i]:.0f}-{pb[i + 1]:.0f} MeV/c": float(x.extra.mean())
                         for i, x in xe.groupby("pbin")}
    out["h_extra"] = {g: np.histogram(x.extra, bins=EXTRA_EDGES)[0] for g, x in xe.groupby("group")}
    out["desc_steps_outside_muon_bars"] = {"edep_frac_of_desc": float(d.loc[~d.mubar, "edep"].sum() / dt),
                                           "bars_per_event": float(d[~d.mubar].groupby(tables.KEY).bar.nunique().sum()
                                                                   / out["n_events"])}
    # non-muon primaries
    prim = parts[parts.isprim == 1].copy()
    prim["apdg"] = prim.pdg.abs()
    prim["ke"] = prim.se - prim.apdg.map(MASS).fillna(0.0)
    nm = s[s.cls.str.startswith("nonmu")]
    # primary of each non-muon step from census.classify (prim_key)
    ph = nm.groupby(tables.KEY + ["prim_key"]).edep.sum().rename("edep").reset_index()
    ph = ph.merge(prim.rename(columns={"key": "prim_key"})[tables.KEY + ["prim_key", "apdg", "ke"]],
                  on=tables.KEY + ["prim_key"], how="left")
    out["nonmu"] = {"edep_frac_of_total": float(nm.edep.sum() / tot), "by_pdg": {}}
    for a, x in ph.groupby("apdg"):
        out["nonmu"]["by_pdg"][int(a)] = {"primaries_with_steps": int(len(x)), "edep_frac_of_total": float(x.edep.sum() / tot),
                                          "ke_q10_q50_q90_MeV": [float(v) for v in np.quantile(x.ke, [0.1, 0.5, 0.9])],
                                          "edep_per_primary_mean_MeV": float(x.edep.mean())}
    out["nonmu"]["by_group"] = (nm.groupby("group").edep.sum() / s.groupby("group").edep.sum()).to_dict()
    # event classes
    ev = s.groupby(tables.KEY).agg(has_mu=("cls", lambda c: (c == "mu_primary").any()),
                                   has_nonmu=("cls", lambda c: c.str.startswith("nonmu").any()))
    out["events"] = {"mu_only_or_desc": int((ev.has_mu & ~ev.has_nonmu).sum()),
                     "mu_and_nonmu": int((ev.has_mu & ev.has_nonmu).sum()),
                     "nonmu_without_mu_crossing": int((~ev.has_mu & ev.has_nonmu).sum()),
                     "neither": int((~ev.has_mu & ~ev.has_nonmu).sum())}
    return out, xe


def gen_census(paths):
    """Per pdg: CRY primaries per generated event (gen stream: every generated event)."""
    g = tables.read_many(paths, "gen", tables.KEY + ["pdg", "e"])
    nev = g[tables.KEY].drop_duplicates().shape[0]
    g["apdg"] = g.pdg.abs()
    c = g.groupby("apdg").size()
    return {"generated_events": int(nev), "per_event_by_pdg": {int(k): float(v / nev) for k, v in c.items()},
            "events_with_muon_frac": float(g[g.apdg == 13][tables.KEY].drop_duplicates().shape[0] / nev),
            "particles_per_event_mean": float(len(g) / nev)}


def run(inputs, outdir, gen=None, counters=None, **kw):
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    out, xe = analyse(inputs, geom)
    if gen:
        out["gen"] = gen_census(gen)
    plots.write_json(outdir, "secondaries_stats.json", out)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, h in out["h_extra"].items():
        h = np.asarray(h, float)
        ax.stairs(h / max(h.sum(), 1), EXTRA_EDGES, label=g)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("descendant visible energy in the muon's bar [MeV]")
    ax.set_ylabel("fraction of crossings")
    ax.legend()
    plots.save(fig, outdir, "secondaries_extra.png")
    f = out["edep_frac_by_class"]
    print("visible energy by class: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in f.items()))
    md = out["mu_desc"]
    print(f"muon descendants: in muon bar {100 * md['in_muon_bar']:.1f}%; birth " +
          ", ".join(f"{k} {100 * v:.1f}%" for k, v in md["by_birth"].items()))
    print("  by root process: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in md["by_process"].items()))
    for g, r in out["per_crossing"].items():
        print(f"  {g}: crossings {r['n']:,}, with extra {100 * r['frac_with_extra']:.1f}%, extra/muon "
              f"{100 * r['extra_over_muon_mean']:.1f}%, extra q50/q90/q99 {r['extra_q50_q90_q99']}")
    print("  extra vs p: " + ", ".join(f"{k}: {v:.3f}" for k, v in out["extra_vs_p"].items()))
    print(f"non-muon: {100 * out['nonmu']['edep_frac_of_total']:.1f}% of visible energy; by group " +
          ", ".join(f"{k} {100 * v:.1f}%" for k, v in out["nonmu"]["by_group"].items()))
    for a, r in sorted(out["nonmu"]["by_pdg"].items(), key=lambda kv: -kv[1]["edep_frac_of_total"]):
        print(f"  pdg {a}: {r['primaries_with_steps']:,} primaries with steps, {100 * r['edep_frac_of_total']:.1f}% of "
              f"edep, KE q10/50/90 {[round(v, 1) for v in r['ke_q10_q50_q90_MeV']]} MeV, edep/primary "
              f"{r['edep_per_primary_mean_MeV']:.1f}")
    print("events: " + ", ".join(f"{k} {v:,}" for k, v in out["events"].items()))
    if gen:
        print("gen: " + str(out["gen"]))
    return out
