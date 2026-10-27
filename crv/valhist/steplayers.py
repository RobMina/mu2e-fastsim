"""Where two CrvStep-table arms on the same generated events differ: per sector group, module and layer, by step origin,
particle, length, energy deposit, steps per particle-bar and position along the bar; and what that does to the light.

Every quantity is a sum over cells (e.g. group x layer x origin) of a per-step weight: the step count n and the visible
energy e. Event-clustered raw second moments are accumulated exactly: sum_e n_e^2 per arm and sum_e(c_e-r_e)^2
for the paired difference. These estimate Poisson-bootstrap variances, not centered fixed-exposure variances.
The diagonal position score is the deck diagnostic, not a calibrated chi-square test; library uncertainty and
cross-bin covariance are not included. Ratio errors use paired delta-method residuals c_e-(C/R)r_e.
Layer rank 0 is the top layer of its module (largest thickness coordinate, y in every extracted sector).
This campaign tool assumes single-run, nonoverlapping shards with equal generated exposure; audit those inputs first.

    python -m valhist.steplayers breakdown --ref 'DIR/g4all.{s}.root' --cand 'DIR/fast1b.{s}.root' --shards 1-20 --out J
    python -m valhist.steplayers response  --ref ... --cand ... --shards 11-14 --calib CSV --deviation CSV --out J
    python -m valhist.steplayers report J [--views grp_lay_cls,...]
    python -m valhist.steplayers posyz --arm a0g='DIR/g4all.{s}.root' --arm a1='...' [--arm ...] --shards 1-20 --out J
    python -m valhist.steplayers clusters RESPONSE_JSON [...]

posyz is the deck's step y/z statistic (fsdeck.crv_yz: CRVDQM v3 extracted binning, float32 start positions, exact
event-clustered variance sum_e n_eb^2 per arm, flag chi2/ndf > 2) per sector group, every arm against the first.
clusters reads the cluster y/z views of a response JSON (exact clustered variance per arm, and paired).

response runs the same numpy fast chain (photons, SiPM charges, waveforms, pulses, coincidences: validated against the
stock modules, A2/A3/A4) on both step tables, so any difference is the steps': charge PE per layer and origin, bars per
layer above PE thresholds, reco pulses per layer, clusters per sector and their y.
"""
import argparse
import json

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables
from crvfastsim import tier1b as T1B
from crvfastsim.analyses.census import SECTOR_GROUP, classify

COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
LEN_EDGES = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 60, 1e9])
E_EDGES = np.array([0, 0.002, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 5, 10, 1e9])
U_EDGES = np.linspace(-1, 1, 11)
NPER_EDGES = np.array([1, 2, 3, 5, 10, 1e9])
PE_EDGES = np.array([0, 1, 5, 10, 20, 50, 100, 1e9])
P_EDGES = np.array([0, 0.3, 1, 3, 10, 30, 100, 1e9])  # step start |p| [MeV/c]
PDGC = {13: "mu", 11: "e", 22: "gamma", 2212: "p", 2112: "n"}
OUTSIDE_MM = 1.0
STEP_COLS = tables.KEY + ["bar", "sp", "t0", "edep", "len", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0"]
SUMS = ["n_ref", "n2_ref", "e_ref", "e2_ref", "n_cand", "n2_cand", "e_cand", "e2_cand", "dn2", "de2"]

# views: name -> cell columns (n = steps, e = visible energy [MeV]; for the response views n = entries, e = PE)
VIEWS = {
    "grp": ["grp"],
    "grp_lay": ["grp", "lay"],
    "grp_lay_cls": ["grp", "lay", "cls"],
    "grp_mod_lay": ["grp", "mod", "lay"],
    "grp_lay_cls_pdg": ["grp", "lay", "cls", "pdgc"],
    "grp_lay_dwhere": ["grp", "lay", "dwhere"],
    "grp_ctx_dwhere": ["grp", "ctx", "dwhere"],
    "grp_mod_ctx_dwhere": ["grp", "mod", "ctx", "dwhere"],
    "grp_cls_lenbin": ["grp", "cls", "lenbin"],
    "grp_cls_ebin": ["grp", "cls", "ebin"],
    "grp_cls_pbin": ["grp", "cls", "pbin"],
    "grp_lay_cls_ubin": ["grp", "lay", "cls", "ubin"],
    "grp_cls_outside": ["grp", "cls", "outside"],
    "grp_cls_pdg_outside": ["grp", "cls", "pdgc", "outside"],
}
# per (particle, bar): number of steps, cells by origin and multiplicity bin
NPER_VIEWS = {"pb_grp_lay_cls": ["grp", "lay", "cls"], "pb_grp_cls_nper": ["grp", "cls", "nper"]}


def shard(k):
    return f"001400_{k:08d}"


def shards(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def bar_table(geom):
    """Per bar: sector group, module label, layer rank (0 = top of its module)."""
    ta = geom.local_to_world[:, 0]
    df = pd.DataFrame({"sector": geom.sector, "module": geom.module, "layer": geom.layer,
                       "tpos": geom.center[np.arange(geom.nbars), ta]})
    lp = df.groupby(["sector", "module", "layer"]).tpos.transform("mean")
    df["lay"] = (lp.groupby([df.sector, df.module]).rank(method="dense", ascending=False).astype(int) - 1).to_numpy()
    names = np.asarray(geom.sector_names, dtype=object)
    df["grp"] = SECTOR_GROUP[geom.sector]
    df["mod"] = [f"{names[s]}.{m}" for s, m in zip(df.sector, df.module)]
    return df


def evkey(df):
    if len(df) and (not df.run.eq(1400).all() or not df.event.between(0, (1 << 20) - 1).all()
                    or not df.subrun.between(0, (1 << 39) - 1).all()):
        raise ValueError("step-layer campaign requires run 1400 and chain-compatible event/subrun IDs")
    return (df.subrun.to_numpy().astype(np.int64) << 24) | df.event.to_numpy().astype(np.int64)


def _cut(v, edges):
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)


def annotate(steps, parts, geom, bt):
    """Steps with the cell columns of every view."""
    s = classify(steps.assign(_row=np.arange(len(steps))), parts, geom.sector)
    s = s.merge(parts[tables.KEY + ["key", "pdg"]].rename(columns={"key": "sp", "pdg": "spdg"}),
                on=tables.KEY + ["sp"], how="left")
    s["ev"] = evkey(s)
    b = s.bar.to_numpy()
    for c in ("grp", "mod", "lay"):
        s[c] = bt[c].to_numpy()[b]
    ap = s.spdg.abs().fillna(0).astype(np.int64).to_numpy()
    s["pdgc"] = pd.Series(ap).map(PDGC).fillna("other").to_numpy()
    # descendants: in a bar crossed by their own primary muon, or elsewhere
    # and the crossing context (tier1b.crossing_context) of the muon crossing in their bar, -1 elsewhere
    x = (s.loc[s.cls == "mu_primary"].sort_values("t0", kind="stable")
         .groupby(["ev", "prim_key", "bar"], sort=False).first().reset_index())
    x["ctx"] = T1B.crossing_context(x.assign(sp=x.prim_key), geom) if len(x) else np.zeros(0, np.int64)
    s = s.merge(x[["ev", "prim_key", "bar", "ctx"]].assign(xb=1), on=["ev", "prim_key", "bar"], how="left")
    s["dwhere"] = np.where(s.cls != "mu_descendant", "-", np.where(s.xb.fillna(0) == 1, "xbar", "otherbar"))
    s["ctx"] = s.ctx.fillna(-1).astype(np.int64)
    s["lenbin"] = _cut(s.len.to_numpy(), LEN_EDGES)
    s["ebin"] = _cut(s.edep.to_numpy(), E_EDGES)
    p = np.sqrt(s.px0.to_numpy() ** 2 + s.py0.to_numpy() ** 2 + s.pz0.to_numpy() ** 2)
    s["pbin"] = _cut(p, P_EDGES)
    r0 = s[["x0", "y0", "z0"]].to_numpy()
    c, h = geom.center[b], geom.halflen[b]
    la = geom.local_to_world[b, 2]
    rows = np.arange(len(s))
    u = (r0[rows, la] - c[rows, la]) / h[rows, la]
    s["ubin"] = _cut(u, U_EDGES)
    s["outside"] = (np.abs(r0 - c) > h + OUTSIDE_MM).any(axis=1).astype(np.int64)
    s["w"] = s.edep.to_numpy()
    return s.sort_values("_row", kind="stable").reset_index(drop=True)   # row i = input step i


def accumulate(ref, cand, cols, wcol="w"):
    """Per cell: per-arm sums of n and weight, their event-clustered second moments, and the paired variance of the
    difference cand - ref (one DataFrame row per cell, columns SUMS)."""
    parts = []
    for arm, d in (("ref", ref), ("cand", cand)):
        g = d.groupby(cols + ["ev"], sort=False, observed=True)[wcol].agg(["size", "sum"]).reset_index()
        g.columns = cols + ["ev", "n", "e"]
        g["arm"] = arm
        parts.append(g)
    a = pd.concat(parts, ignore_index=True)
    a["n"] = a.n.astype(float)
    a["n2"], a["e2"] = a.n ** 2, a.e ** 2
    pa = a.groupby(cols + ["arm"], observed=True)[["n", "n2", "e", "e2"]].sum().unstack("arm", fill_value=0.0)
    pa.columns = [f"{q}_{arm}" for q, arm in pa.columns]
    for c in SUMS[:8]:
        if c not in pa.columns:
            pa[c] = 0.0
    sg = np.where(a.arm == "cand", 1.0, -1.0)
    a["dn"], a["de"] = sg * a.n, sg * a.e
    d = a.groupby(cols + ["ev"], observed=True)[["dn", "de"]].sum()
    d["dn2"], d["de2"] = d.dn ** 2, d.de ** 2
    pd_ = d.groupby(level=list(range(len(cols))))[["dn2", "de2"]].sum()
    out = pa.join(pd_, how="outer").fillna(0.0)[SUMS].reset_index()
    return out


def per_particle_bar(s):
    """One row per (event, particle, bar) with its step count and summed visible energy."""
    g = s.groupby(["ev", "sp", "bar"], sort=False).agg(nstep=("edep", "size"), w=("edep", "sum"),
                                                       grp=("grp", "first"), lay=("lay", "first"),
                                                       cls=("cls", "first")).reset_index()
    g["nper"] = _cut(g.nstep.to_numpy(), NPER_EDGES)
    return g


def _merge_views(acc, new):
    for k, v in new.items():
        acc[k] = v if k not in acc else pd.concat([acc[k], v]).groupby(list(v.columns[:-len(SUMS)]),
                                                                       as_index=False)[SUMS].sum()
    return acc


def _geom():
    geom = geometry.from_offline_config(geometry.find_counters(COUNTERS))
    return geom, bar_table(geom)


def _read(path, cols=STEP_COLS):
    return tables.read(path, "steps", cols), tables.read(path, "particles",
                                                              tables.KEY + ["key", "parent", "pdg", "isprim"])


def _dump(acc, out, meta):
    res = {"meta": meta, "views": {}}
    for k, v in acc.items():
        cols = list(v.columns[:-len(SUMS)])
        res["views"][k] = {"cols": cols, "sums": SUMS,
                           "rows": [[(x.item() if hasattr(x, "item") else x) for x in r] for r in v.itertuples(index=False)]}
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"wrote {out}")


def breakdown(ref_pat, cand_pat, shard_list, out):
    geom, bt = _geom()
    acc, info = {}, {"inside_tol_mm": OUTSIDE_MM, "shards": shard_list}
    for k in shard_list:
        R = annotate(*_read(ref_pat.format(s=shard(k), k=k)), geom, bt)
        C = annotate(*_read(cand_pat.format(s=shard(k), k=k)), geom, bt)
        new = {name: accumulate(R, C, cols) for name, cols in VIEWS.items()}
        pr, pc = per_particle_bar(R), per_particle_bar(C)
        new |= {name: accumulate(pr, pc, cols) for name, cols in NPER_VIEWS.items()}
        acc = _merge_views(acc, new)
        print(f"{shard(k)}: ref {len(R):,} steps, cand {len(C):,}", flush=True)
    info["layers"] = bt.groupby("grp").lay.max().astype(int).to_dict()
    _dump(acc, out, {"ref": ref_pat, "cand": cand_pat, **info})


# ----------------------------------------------------------------------------------------------------- response
def _chain_arm(steps, parts, geom, bt, cond, cfg, ccfg, tabs, pmap, rng):
    """Fast numpy chain on one step table: per-charge, per-(event, bar), pulse and cluster frames."""
    from crvfastsim import chain
    s = annotate(steps, parts, geom, bt)
    Q = chain.charges(steps, parts, geom, cond, cfg, rng, tabs=tabs, pmap=pmap)
    row = np.asarray(Q["photon"], np.int64)
    q = pd.DataFrame({"ev": s.ev.to_numpy()[row], "bar": s.bar.to_numpy()[row], "w": np.asarray(Q["pe"], float)})
    for c in ("grp", "mod", "lay", "cls"):
        q[c] = s[c].to_numpy()[row]
    bars = q.groupby(["ev", "bar"], sort=False).w.sum().reset_index()
    for c in ("grp", "mod", "lay"):
        bars[c] = bt[c].to_numpy()[bars.bar.to_numpy()]
    bars["pebin"] = _cut(bars.w.to_numpy(), PE_EDGES)
    D = chain.digis(Q, cond, cfg, rng)
    W = chain.reco_waveforms(D, cond)
    outs, _ = chain.clusters(W, cond, geom, ccfg, cfg, detail=True, rng=rng)
    o = outs[0]
    P = o["pulses"]
    # chain event keys are (subrun << 20) | event; ours (subrun << 24) | event
    re_ev = lambda e: ((np.asarray(e, np.int64) >> 20) << 24) | (np.asarray(e, np.int64) & ((1 << 20) - 1))
    pul = pd.DataFrame({"ev": re_ev(P["ev"]), "bar": np.asarray(P["bar"], np.int64), "w": np.asarray(P["pe"], float)})
    for c in ("grp", "mod", "lay"):
        pul[c] = bt[c].to_numpy()[pul.bar.to_numpy()]
    pul["pebin"] = _cut(pul.w.to_numpy(), PE_EDGES)
    cl = pd.DataFrame({"ev": re_ev(o["cl_event"]), "ctype": np.asarray(o["cl_type"], np.int64),
                       "w": np.asarray(o["cl_pe"], float)})
    pos = np.asarray(o["cl_pos"], float).reshape(-1, 3)
    cl["ybin"] = np.searchsorted(Y_EDGES, pos[:, 1], side="right") if len(cl) else np.zeros(0, np.int64)
    cl["zbin"] = np.searchsorted(Z_EDGES, pos[:, 2], side="right") if len(cl) else np.zeros(0, np.int64)
    return q, bars, pul, cl


Y_EDGES = np.linspace(4202.4, 4743.6, 101)   # crvRecoDQM Y_extracted booking (CRVDQM binning v3, extracted)
Z_EDGES = np.linspace(20787.95, 24903.05, 101)
RESP_VIEWS = {
    "pe_grp_lay_cls": ("q", ["grp", "lay", "cls"]), "pe_grp_mod_lay": ("q", ["grp", "mod", "lay"]),
    "bars_grp_lay_pebin": ("bars", ["grp", "lay", "pebin"]), "bars_grp_lay": ("bars", ["grp", "lay"]),
    "pul_grp_lay": ("pul", ["grp", "lay"]), "pul_grp_mod_lay": ("pul", ["grp", "mod", "lay"]),
    "pul_grp_lay_pebin": ("pul", ["grp", "lay", "pebin"]),
    "cl_ctype": ("cl", ["ctype"]), "cl_ctype_ybin": ("cl", ["ctype", "ybin"]), "cl_ctype_zbin": ("cl", ["ctype", "zbin"]),
}


def response(ref_pat, cand_pat, shard_list, out, calib, deviation=None, seed=1):
    from crvfastsim import chain, runconfig
    from crvfastsim.response import photons as PH
    from crvfastsim.response import sipm as SI
    rc = runconfig.get(None, None)
    geom, ccfg = rc.geometry(), rc.counters_cfg()
    bt = bar_table(geom)
    cond = chain.Conditions.from_csv(calib, deviation)
    cfg = chain.ChainConfig(jax=False)
    tabs, pmap = PH.load_tables(cfg.response), SI.photon_map()
    acc = {}
    for k in shard_list:
        fr = {}
        for arm, pat in (("ref", ref_pat), ("cand", cand_pat)):
            rng = np.random.default_rng([int(seed), k])   # same draws per shard in both arms
            fr[arm] = dict(zip(("q", "bars", "pul", "cl"),
                               _chain_arm(*_read(pat.format(s=shard(k), k=k), None), geom, bt, cond, cfg, ccfg, tabs, pmap, rng)))
        new = {name: accumulate(fr["ref"][src], fr["cand"][src], cols) for name, (src, cols) in RESP_VIEWS.items()}
        acc = _merge_views(acc, new)
        print(f"{shard(k)}: PE ref {fr['ref']['q'].w.sum():.0f} cand {fr['cand']['q'].w.sum():.0f}; clusters "
              f"{len(fr['ref']['cl'])} / {len(fr['cand']['cl'])}", flush=True)
    _dump(acc, out, {"ref": ref_pat, "cand": cand_pat, "shards": shard_list, "calib": calib, "deviation": deviation,
                     "seed": seed, "position_bins": "root_flow", "y_edges": [float(Y_EDGES[0]), float(Y_EDGES[-1]), len(Y_EDGES) - 1],
                     "z_edges": [float(Z_EDGES[0]), float(Z_EDGES[-1]), len(Z_EDGES) - 1]})


def layout():
    """Per sector, module and layer rank: bars, length and thickness axes, and the bar-box extent in x, y, z [mm]."""
    geom, bt = _geom()
    lo, hi = geom.center - geom.halflen, geom.center + geom.halflen
    df = bt.assign(la=geom.local_to_world[:, 2], ta=geom.local_to_world[:, 0], **{f"{c}lo": lo[:, i] for i, c in
                   enumerate("xyz")}, **{f"{c}hi": hi[:, i] for i, c in enumerate("xyz")})
    g = df.groupby(["grp", "mod", "lay"]).agg(bars=("la", "size"), la=("la", "first"), ta=("ta", "first"),
                                              xlo=("xlo", "min"), xhi=("xhi", "max"), ylo=("ylo", "min"),
                                              yhi=("yhi", "max"), zlo=("zlo", "min"), zhi=("zhi", "max"))
    pd.set_option("display.width", 250, "display.max_rows", 2000)
    print(g.round(1).to_string())


def same(a, b):
    """Whether two step tables hold identical steps and particles (row order included); prints the first difference."""
    for tree in ("steps", "particles"):
        x, y = tables.read(a, tree), tables.read(b, tree)
        try:
            pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True), check_dtype=False)
        except AssertionError as e:
            print(f"{tree}: differ ({len(x)} vs {len(y)} rows): {str(e).splitlines()[:4]}")
            return False
        print(f"{tree}: identical ({len(x)} rows)")
    return True


# --------------------------------------------------------------------------------------------- step positions
BOOKING = "/exp/mu2e/data/users/ram2aq/crv_fastsim/valhist/g4ref_v1/a0/valhist.001400_00000001.root"
POS_GROUPS = ("all", "EX", "T1", "T2", "M")
SECTOR_TYPE = {1: "EX", 2: "T1", 3: "T2", 4: "M"}   # CrvCoincidenceCluster sector type of the extracted sectors
FLAG = 2.0                                            # chi2/ndf flag without a floor (deck rule)


def booking_edges(path=BOOKING):
    """crvRecoDQM Y/Z_extracted bin edges of a frozen A0 shard (falls back to the v3 constants)."""
    try:
        import uproot
        with uproot.open(path) as f:
            return {v: f[f"crvRecoDQM/{v.upper()}_extracted"].axis().edges() for v in ("y", "z")}
    except (OSError, KeyError, ImportError):
        return {"y": Y_EDGES, "z": Z_EDGES}


def posfill(ev, v, edges):
    """Counts with flow (ROOT bin numbering) and w2 = sum over events of the squared per-event bin count."""
    nb = len(edges) + 1
    idx = np.searchsorted(edges, v, side="right")
    n = np.bincount(idx, minlength=nb).astype(float)
    if not len(v):
        return n, np.zeros(nb)
    u, c = np.unique(np.asarray(ev, np.int64) * nb + idx, return_counts=True)
    return n, np.bincount(u % nb, weights=c.astype(float) ** 2, minlength=nb)


def posstat(n1, w1, n2, w2):
    """Deck statistic with the exact event-clustered variance on the booked bins: sum (n2-n1)^2 / (w1+w2) over bins
    with entries, divided by their number; flag above FLAG. Totals include flow."""
    a, b, va = np.asarray(n1)[1:-1], np.asarray(n2)[1:-1], np.asarray(w1)[1:-1] + np.asarray(w2)[1:-1]
    m = (a + b) > 0
    chi = float(((b[m] - a[m]) ** 2 / va[m]).sum() / max(int(m.sum()), 1))
    t1, t2 = float(np.sum(n1)), float(np.sum(n2))
    return {"chi2ndf_w2": chi, "ndf": int(m.sum()), "flag": chi > FLAG, "n_ref": t1, "n_cand": t2,
            "rel": t2 / t1 - 1 if t1 else None, "flow_ref": float(n1[0] + n1[-1]), "flow_cand": float(n2[0] + n2[-1])}


def posyz(arms, shard_list, out, booking=BOOKING):
    """Step start y/z per sector group with the CRVDQM extracted binning, for arms [(name, pattern), ...]; every arm
    is compared with the first (fsdeck.crv_yz statistic: float32 positions, exact clustered variance, flag > 2)."""
    edges = booking_edges(booking)
    _, bt = _geom()
    grp = bt.grp.to_numpy()
    acc = {a: {g: {v: [0.0, 0.0] for v in edges} for g in POS_GROUPS} for a, _ in arms}
    for k in shard_list:
        for name, pat in arms:
            s = tables.read(pat.format(s=shard(k), k=k), "steps", tables.KEY + ["bar", "y0", "z0"])
            ev, g_of = evkey(s), grp[s.bar.to_numpy()]
            pos = {v: s[f"{v}0"].to_numpy().astype(np.float32).astype(np.float64) for v in edges}
            for g in POS_GROUPS:
                m = np.ones(len(s), bool) if g == "all" else g_of == g
                for v, e in edges.items():
                    n, w2 = posfill(ev[m], pos[v][m], e)
                    acc[name][g][v][0] = acc[name][g][v][0] + n
                    acc[name][g][v][1] = acc[name][g][v][1] + w2
        print(f"{shard(k)} done", flush=True)
    ref = arms[0][0]
    res = {"meta": {"arms": dict(arms), "shards": shard_list, "booking": booking},
           "edges": {v: [float(x) for x in e] for v, e in edges.items()},
           "hists": {a: {g: {v: {"n": list(map(float, nw[0])), "w2": list(map(float, nw[1]))} for v, nw in d.items()}
                         for g, d in per.items()} for a, per in acc.items()}, "cmp": {}}
    for name, _ in arms[1:]:
        res["cmp"][f"{name}_vs_{ref}"] = {f"{g}/{v}": posstat(*acc[ref][g][v], *acc[name][g][v])
                                          for g in POS_GROUPS for v in edges}
    with open(out, "w") as f:
        json.dump(res, f)
    print_posyz(res)
    print(f"wrote {out}")


def print_posyz(res):
    for c, per in res["cmp"].items():
        print(f"\n== {c}: step start position, diagonal deck score (clustered second moments), total cand/ref - 1")
        for g in POS_GROUPS:
            y, z = per[f"{g}/y"], per[f"{g}/z"]
            print(f"{g:>4}  y {y['chi2ndf_w2']:7.2f}{'*' if y['flag'] else ' '}  z {z['chi2ndf_w2']:7.2f}"
                  f"{'*' if z['flag'] else ' '}  N {y['n_ref']:>10,.0f} -> {y['n_cand']:>10,.0f} ({100 * y['rel']:+.2f} %)")
        print(f"flags: {sum(r['flag'] for k, r in per.items() if not k.startswith('all'))} / 8 sector histograms")


def clusters(path, view_y="cl_ctype_ybin", view_z="cl_ctype_zbin"):
    """Cluster y/z per sector type from a `response` JSON: chi2/ndf with the exact clustered variance of each arm
    (independent arms, the deck rule) and with the paired per-event variance of the difference (same events)."""
    js = json.load(open(path))
    out = {}
    tot = pd.DataFrame(js["views"]["cl_ctype"]["rows"], columns=js["views"]["cl_ctype"]["cols"] + SUMS)
    for var, view in (("y", view_y), ("z", view_z)):
        v = js["views"][view]
        df = pd.DataFrame(v["rows"], columns=v["cols"] + SUMS)
        for ct, ib in df.groupby("ctype"):
            visible = np.ones(len(ib), bool)
            if js["meta"].get("position_bins") == "root_flow":
                visible = (ib[f"{var}bin"] >= 1) & (ib[f"{var}bin"] <= js["meta"][f"{var}_edges"][2])
            flow = ib.loc[~visible, ["n_ref", "n_cand"]].sum()
            m = ((ib.n_ref + ib.n_cand) > 0) & visible
            diff2 = (ib.n_cand - ib.n_ref)[m] ** 2
            t = tot[tot.ctype == ct].iloc[0]
            out[f"{SECTOR_TYPE.get(int(ct), ct)}/{var}"] = {
                "chi2ndf_w2": float((diff2 / (ib.n2_ref + ib.n2_cand)[m]).sum() / max(int(m.sum()), 1)),
                "chi2ndf_paired": float((diff2 / ib.dn2[m].where(ib.dn2[m] > 0, 1)).sum() / max(int(m.sum()), 1)),
                "ndf": int(m.sum()), "n_ref": float(t.n_ref), "n_cand": float(t.n_cand),
                "position_bins": js["meta"].get("position_bins", "legacy_clipped"),
                "flow_ref": float(flow.n_ref), "flow_cand": float(flow.n_cand),
                "rel": float(t.n_cand / t.n_ref - 1),
                "rel_err_paired": float(ratio_error(t.n_ref, t.n_cand, t.n2_ref, t.n2_cand, t.dn2)),
                "scaled_difference_err": float(np.sqrt(t.dn2) / t.n_ref)}
    return out


def print_clusters(path):
    c = clusters(path)
    print(f"\n== clusters ({path}): diagonal deck score (paired score), count cand/ref - 1")
    for k, r in c.items():
        print(f"{k:>5}  {r['chi2ndf_w2']:6.2f}{'*' if r['chi2ndf_w2'] > FLAG else ' '} ({r['chi2ndf_paired']:6.2f})  "
              f"ndf {r['ndf']:3d}  N {r['n_ref']:>7,.0f} -> {r['n_cand']:>7,.0f} "
              f"({100 * r['rel']:+.2f} +- {100 * r['rel_err_paired']:.2f} %)"
              f"  flow {r['flow_ref']:.0f}/{r['flow_cand']:.0f} [{r['position_bins']}]")
    return c


# ------------------------------------------------------------------------------------------------------- report
def ratio_error(ref, cand, ref2, cand2, diff2):
    """Paired ratio delta-method SE from event-clustered raw moments, conditional on the fixed library."""
    ref, cand = np.asarray(ref, float), np.asarray(cand, float)
    ratio = np.divide(cand, ref, out=np.zeros_like(cand), where=ref > 0)
    cross = 0.5 * (np.asarray(ref2) + np.asarray(cand2) - np.asarray(diff2))
    variance = np.maximum(np.asarray(cand2) + ratio ** 2 * np.asarray(ref2) - 2 * ratio * cross, 0)
    return np.divide(np.sqrt(variance), ref, out=np.full_like(cand, np.nan), where=ref > 0)


def table(js, view, q="n"):
    """DataFrame of one view: ref, cand, cand/ref - 1 with its paired error, and the clustered chi2 of each cell."""
    v = js["views"][view]
    df = pd.DataFrame(v["rows"], columns=v["cols"] + v["sums"])
    r, c = df[f"{q}_ref"], df[f"{q}_cand"]
    df["diff"] = c - r
    df["err"] = np.sqrt(df[f"d{q}2"])
    df["rel"] = np.where(r > 0, c / r.where(r > 0, 1) - 1, np.nan)
    df["rel_err"] = ratio_error(r, c, df[f"{q}2_ref"], df[f"{q}2_cand"], df[f"d{q}2"])
    df["scaled_difference_err"] = np.where(r > 0, df.err / r.where(r > 0, 1), np.nan)
    df["pull"] = np.where(df.err > 0, df["diff"] / df.err.where(df.err > 0, 1), 0.0)
    df["pull_indep"] = np.where(df[f"{q}2_ref"] + df[f"{q}2_cand"] > 0,
                                df["diff"] / np.sqrt((df[f"{q}2_ref"] + df[f"{q}2_cand"]).where(lambda x: x > 0, 1)), 0)
    return df[v["cols"] + [f"{q}_ref", f"{q}_cand", "diff", "err", "rel", "rel_err", "pull", "pull_indep"]]


def report(path, views=None, q=("n", "e")):
    js = json.load(open(path))
    pd.set_option("display.width", 250, "display.max_rows", 2000, "display.max_columns", 30)
    for view in (views or js["views"]):
        for qq in q:
            print(f"\n== {view} [{qq}]")
            print(table(js, view, qq).to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def main():
    ap = argparse.ArgumentParser(prog="valhist.steplayers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("breakdown", "response"):
        p = sub.add_parser(name)
        p.add_argument("--ref", required=True, help="reference step tables, {s} = subrun, {k} = shard index, e.g. DIR/g4all.{s}.root")
        p.add_argument("--cand", required=True, help="candidate step tables, {s} = subrun")
        p.add_argument("--shards", default="1-20")
        p.add_argument("--out", required=True)
        if name == "response":
            p.add_argument("--calib", required=True, help="CRVCalib + channel map CSV")
            p.add_argument("--deviation", default=None, help="CRVPhoton deviations CSV")
            p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("posyz")
    p.add_argument("--arm", action="append", required=True, metavar="NAME=PATTERN",
                   help="step tables of one arm, {s} = subrun, {k} = shard; the first arm is the reference")
    p.add_argument("--shards", default="1-20")
    p.add_argument("--booking", default=BOOKING, help="ROOT file with crvRecoDQM/{Y,Z}_extracted (bin edges)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("clusters")
    p.add_argument("json", nargs="+", help="response JSON(s)")
    sub.add_parser("layout")
    p = sub.add_parser("same")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("report")
    p.add_argument("json")
    p.add_argument("--views", default=None, help="comma-separated view names (default all)")
    p.add_argument("--q", default="n,e", help="n (entries) and/or e (energy or PE)")
    a = ap.parse_args()
    if a.cmd == "breakdown":
        breakdown(a.ref, a.cand, shards(a.shards), a.out)
    elif a.cmd == "response":
        response(a.ref, a.cand, shards(a.shards), a.out, a.calib, a.deviation, a.seed)
    elif a.cmd == "posyz":
        posyz([tuple(x.split("=", 1)) for x in a.arm], shards(a.shards), a.out, a.booking)
    elif a.cmd == "same":
        raise SystemExit(0 if same(a.a, a.b) else 1)
    elif a.cmd == "layout":
        layout()
    elif a.cmd == "clusters":
        for j in a.json:
            print_clusters(j)
    else:
        report(a.json, a.views.split(",") if a.views else None, a.q.split(","))


if __name__ == "__main__":
    main()
