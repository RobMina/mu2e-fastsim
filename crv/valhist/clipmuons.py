"""Diagnose muon events where G4 and the fast steps disagree on CRV activity (the fast-steps digi deficit).

For events whose leading CRY primary is a muon, compares the G4 CrvStep tables (crvsteps.*) with the fast step tables
(fast1b.*) per event: which particles deposit in G4 (primary muon vs secondaries), how many bars each arm touches, the
muon's distance of closest approach to the bars it misses, and whether G4's primary muon was deflected (its entry
direction at the first CRV step vs the generated direction).

    python -m valhist.clipmuons --g4 crvsteps...root ... --fast fast1b...root ... [--emin-kev 0] [--out OUT.json]
"""
import argparse
import json

import numpy as np

from crvfastsim import geometry, tables
from crvfastsim.analyses import v2compare as V2


def _key(df):
    return (df.subrun.to_numpy().astype(np.int64) << 20) | df.event.to_numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4", nargs="+", required=True)
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--emin-kev", type=float, default=0.0, help="visible energy per event below which a bar is ignored")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    geom = geometry.from_offline_config(geometry.find_counters(V2.DEFAULT_COUNTERS))
    grp = np.full(len(geom.sector), "", object)
    for name, secs in V2.GROUPS.items():
        grp[np.isin(geom.sector, secs)] = name

    gs = tables.read_many(a.g4, "steps", tables.KEY + ["bar", "sp", "edep", "x0", "y0", "z0", "px0", "py0", "pz0", "len"])
    gp = tables.read_many(a.g4, "particles", tables.KEY + ["key", "parent", "pdg", "isprim", "ccode"])
    gen = tables.read_many(a.g4, "gen", tables.KEY + ["pdg", "px", "py", "pz", "e", "x", "y", "z"])
    fs = tables.read_many(a.fast, "steps", tables.KEY + ["bar", "sp", "edep", "len"])
    fp = tables.read_many(a.fast, "particles", tables.KEY + ["key", "pdg", "isprim"])
    gs["k"], gp["k"], gen["k"], fs["k"], fp["k"] = _key(gs), _key(gp), _key(gen), _key(fs), _key(fp)

    mu = tables.primary_muons(gen)
    mu["k"] = _key(mu)
    muk = set(mu.k.tolist())
    # G4 particle class: primary muon (isprim, |pdg| 13) or secondary
    gp["prim_mu"] = (gp.isprim == 1) & (gp.pdg.abs() == 13)
    gs = gs.merge(gp[["k", "key", "prim_mu", "pdg", "ccode"]].rename(columns={"key": "sp"}), on=["k", "sp"], how="left")
    gs["prim_mu"] = gs.prim_mu.fillna(False).astype(bool)
    # fast: primary muon = tier 1a (isprim and |pdg| 13), else library
    fp["prim_mu"] = (fp.isprim == 1) & (fp.pdg.abs() == 13)
    fs = fs.merge(fp[["k", "key", "prim_mu"]].rename(columns={"key": "sp"}), on=["k", "sp"], how="left")
    fs["prim_mu"] = fs.prim_mu.fillna(False).astype(bool)

    eg = gs.groupby("k").edep.sum()
    ef = fs.groupby("k").edep.sum()
    thr = a.emin_kev * 1e-3
    g_ev = set(eg[eg > thr].index) & muk
    f_ev = set(ef[ef > thr].index) & muk
    g_only, f_only, both = sorted(g_ev - f_ev), sorted(f_ev - g_ev), sorted(g_ev & f_ev)

    def describe(keys, S, label):
        S = S[S.k.isin(keys)]
        per = S.groupby("k").agg(nbar=("bar", "nunique"), edep=("edep", "sum"), mu_edep=("edep", lambda x: 0.0))
        pm = S[S.prim_mu].groupby("k").edep.sum()
        per["mu_edep"] = pm.reindex(per.index).fillna(0.0)
        per["mu_frac"] = per.mu_edep / per.edep.where(per.edep > 0, 1.0)
        return {"events": len(per), "nbar_median": float(per.nbar.median()) if len(per) else 0,
                "edep_MeV_quartiles": per.edep.quantile([.25, .5, .75]).round(3).tolist() if len(per) else [],
                "events_with_primary_muon_steps": int((per.mu_edep > 0).sum()),
                "events_secondaries_only": int((per.mu_edep == 0).sum()),
                "edep_from_primary_muon_frac": float(per.mu_edep.sum() / max(per.edep.sum(), 1e-12))}

    out = {"muon_events": len(muk), "g4_with_crv": len(g_ev), "fast_with_crv": len(f_ev), "both": len(both),
           "g4_only": describe(g_only, gs, "g4"), "fast_only": describe(f_only, fs, "fast"),
           "both_g4": describe(both, gs, "g4"), "both_fast": describe(both, fs, "fast")}

    # G4-only events with primary-muon steps: was the muon deflected? compare its direction at the first CRV step with
    # the generated direction; and the straight line's distance of closest approach to the bars G4 hit
    m = mu.set_index("k")
    gsel = gs[gs.k.isin(g_only) & gs.prim_mu].sort_values(["k", "y0"], ascending=[True, False])
    first = gsel.drop_duplicates("k").set_index("k")
    if len(first):
        d0 = m.loc[first.index, ["px", "py", "pz"]].to_numpy()
        d0 /= np.linalg.norm(d0, axis=1)[:, None]
        d1 = first[["px0", "py0", "pz0"]].to_numpy()
        p1 = np.linalg.norm(d1, axis=1)
        d1 /= np.where(p1 > 0, p1, 1)[:, None]
        ang = np.degrees(np.arccos(np.clip((d0 * d1).sum(1), -1, 1)))
        # straight-line miss distance from the generated line to the first G4 step point
        o = m.loc[first.index, ["x", "y", "z"]].to_numpy()
        q = first[["x0", "y0", "z0"]].to_numpy()
        w = q - o
        miss = np.linalg.norm(w - (w * d0).sum(1)[:, None] * d0, axis=1)
        p0 = np.linalg.norm(m.loc[first.index, ["px", "py", "pz"]].to_numpy(), axis=1)
        # where the transport stands for these muons: footprint cut (no MSC, no world material) and expected MSC width
        from crvfastsim import transport as TR
        near = TR.near_crv(o, d0, geom)
        th_expect = None
        out["g4_only_near_crv_frac"] = float(near.mean())
        out["g4_only_by_cut"] = {
            "outside_footprint": {"events": int((~near).sum()), "deflection_deg_median": float(np.median(ang[~near])) if (~near).any() else None},
            "inside_footprint": {"events": int(near.sum()),
                                 "deflection_deg_quartiles": np.percentile(ang[near], [25, 50, 75, 90]).round(2).tolist() if near.any() else [],
                                 "miss_mm_quartiles": np.percentile(miss[near], [25, 50, 75, 90]).round(1).tolist() if near.any() else []}}
        # world column above the plane for the inside-footprint ones (stage C material, g/cm2): air-only vs thick
        try:
            import os
            from mu2efastsim import world as W
            pieces = [q for q in W.load(os.path.join(os.environ["CRVFS_DATA"], "geometry/mu2e_extracted_v04.gdml"),
                                        y_min=TR.Y_MSC_PLANE) if q[0] != "G4_POLYSTYRENE"]
            L = (o[:, 1] - TR.Y_MSC_PLANE) / -np.where(d0[:, 1] < 0, d0[:, 1], -1)
            col = W.column(pieces, o, d0, np.maximum(L, 0))
            solid = sum(col.values()) if col else np.zeros(len(o))
            for lab, m_ in (("inside", near), ("outside", ~near)):
                out["g4_only_by_cut"][f"{lab}_footprint"]["solid_mm_quartiles"] = (
                    np.percentile(solid[m_], [25, 50, 75]).round(0).tolist() if m_.any() else [])
            thin = near & (solid < 100)
            out["g4_only_by_cut"]["inside_thin_column"] = {
                "events": int(thin.sum()), "deflection_deg_quartiles":
                np.percentile(ang[thin], [25, 50, 75, 90]).round(2).tolist() if thin.any() else []}
        except Exception as e:    # diagnostics only
            out["world_column_error"] = repr(e)
        out["g4_only_primary_muon"] = {
            "events": len(first), "deflection_deg_quartiles": np.percentile(ang, [25, 50, 75, 90]).round(2).tolist(),
            "miss_mm_quartiles": np.percentile(miss, [25, 50, 75, 90]).round(1).tolist(),
            "p_gen_MeV_quartiles": np.percentile(p0, [25, 50, 75]).round(0).tolist(),
            "p_at_crv_MeV_quartiles": np.percentile(p1, [25, 50, 75]).round(0).tolist(),
            "first_bar_group": {g: int((grp[first.bar.to_numpy()] == g).sum()) for g in V2.GROUPS},
            "first_step_len_mm_quartiles": first.len.quantile([.25, .5, .75]).round(2).tolist(),
            "frac_miss_gt_20mm": float((miss > 20).mean()), "frac_miss_gt_100mm": float((miss > 100).mean())}
        # the same for G4 muons in events BOTH arms see, for scale
        bsel = gs[gs.k.isin(both) & gs.prim_mu].sort_values(["k", "y0"], ascending=[True, False]).drop_duplicates("k")
        bsel = bsel.set_index("k")
        o2, q2 = m.loc[bsel.index, ["x", "y", "z"]].to_numpy(), bsel[["x0", "y0", "z0"]].to_numpy()
        d2 = m.loc[bsel.index, ["px", "py", "pz"]].to_numpy()
        d2 /= np.linalg.norm(d2, axis=1)[:, None]
        w2 = q2 - o2
        miss2 = np.linalg.norm(w2 - (w2 * d2).sum(1)[:, None] * d2, axis=1)
        out["both_primary_muon_miss_mm_quartiles"] = np.percentile(miss2, [25, 50, 75, 90]).round(1).tolist()
    # secondaries-only G4 events: creation codes of the depositing particles
    sec = gs[gs.k.isin(g_only) & ~gs.prim_mu]
    out["g4_only_secondary_pdg"] = {int(k): int(v) for k, v in sec.drop_duplicates(["k", "sp"]).pdg.value_counts().head(8).items()}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
