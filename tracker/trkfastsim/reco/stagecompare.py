"""Per-stage, event-by-event comparison of two reco-stage dumps (plan (c) reference harness).

Input: the `events` tree of `StrawHitTableDumper` with DumpTracks (fcl `dump_recostages_ce.fcl`): straw and panel
ComboHits (`s_`, `p_`), time clusters (`tc_`, one or more collections), helix seeds (`hs_`), KalSeed fits (`kl_`)
with their hits (`kh_`) and LoopHelix parameters (`kl_lh_*`) at the segment nearest t0. The arms are two runs over
the same Geant4 events: the same digis (a replay, or a device stage fed by stock), or another digitisation seed (the
stock-vs-stock floor).

Objects are matched inside common events by shared straws (StrawId), mutual best match with at least MIN_SHARED
shared straws and at least half of the smaller straw list: an identity match that holds across digitisations,
never by distance or time. Straw hits are compared one to one only when both arms have the same digis.
Every number is computed over flat arrays (pandas joins, numpy), with no per-event loops.

CLI: python3 -m trkfastsim.reco.stagecompare --a A.root [...] --b B.root [...] [--out stagecompare.json]
"""
import argparse
import json

import awkward as ak
import numpy as np
import pandas as pd
import uproot

TREE = "events"
MIN_SHARED = 5
QS = (0.05, 0.16, 0.5, 0.84, 0.95)
DIGI = ("d_sid", "d_tdc0", "d_tdc1", "d_tot0", "d_tot1", "d_pmp")
CH_INT = ("sid", "nsh", "ncombo", "eend", "flag")
CH_FLOAT = ("time", "edep", "dtime", "ptime", "wdist", "uvar", "vvar", "timevar", "qual", "x", "y", "z")
TC_PANEL = {0: True, 1: True, 2: False}   # dump_recostages_ce.fcl: TimeClusterFinderDe, CalTimePeakFinderDe -> makePH
HS_PAR = ("t0", "rad", "lam", "cx", "cy", "fz0")
LH_PAR = ("rad", "lam", "cx", "cy", "phi0", "t0")


def load(paths, tree=TREE):
    """The dump's branches as one awkward record array over the files, plus the event keys."""
    arrs = []
    for p in paths:
        t = uproot.open(p)[tree]
        arrs.append(t.arrays([k for k in t.keys() if not k.startswith("d_adc") and k != "d_nadc"]))
    ev = ak.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    keys = np.stack([ak.to_numpy(ev[k]).astype(np.int64) for k in ("run", "subrun", "event")], 1)
    return {"ev": ev, "keys": keys, "fields": set(ev.fields)}


def match_events(ka, kb):
    """Entry indices of the events present in both, in A's order."""
    _, inv = np.unique(np.concatenate([ka, kb]), axis=0, return_inverse=True)
    inv = inv.ravel()
    _, ia, ib = np.intersect1d(inv[:len(ka)], inv[len(ka):], return_indices=True)
    o = np.argsort(ia)
    return ia[o], ib[o]


def _q(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return {"n": int(len(a)), "mean": float(a.mean()), "q": [float(x) for x in np.quantile(a, QS)]}


def _frac(m):
    m = np.asarray(m, bool)
    return {"n": int(len(m)), "frac": float(m.mean()) if len(m) else None}


def _objects(ev, n, ind, panel=None):
    """Flat (event, object, straw) table of jagged objects: n[event][object] hit counts, ind the flat hit indices per
    event into makeSH (panel=False) or makePH (panel=True, expanded to the panel's straw hits)."""
    cnt = ak.to_numpy(ak.flatten(ev[n]))
    nev = ak.to_numpy(ak.num(ev[n], axis=1))
    evo = np.repeat(np.arange(len(nev)), nev)                      # event of each object
    obj = np.arange(len(cnt)) - np.repeat(np.cumsum(nev) - nev, nev)  # object index within its event
    idx = ak.to_numpy(ak.flatten(ev[ind])).astype(np.int64)
    e_h = np.repeat(evo, cnt)
    o_h = np.repeat(obj, cnt)
    off_s = np.cumsum(ak.to_numpy(ak.num(ev["s_sid"], axis=1))) - ak.to_numpy(ak.num(ev["s_sid"], axis=1))
    sid_s = ak.to_numpy(ak.flatten(ev["s_sid"])).astype(np.int64)
    if panel:
        npc = ak.to_numpy(ak.flatten(ev["p_ncombo"])).astype(np.int64)
        nph = ak.to_numpy(ak.num(ev["p_ncombo"], axis=1))
        off_p = np.cumsum(nph) - nph
        start = np.cumsum(npc) - npc                                # start of each panel hit in the flat p_ind
        pind = ak.to_numpy(ak.flatten(ev["p_ind"])).astype(np.int64)
        g = off_p[e_h] + idx                                        # flat panel-hit index
        k = npc[g]
        e_h, o_h = np.repeat(e_h, k), np.repeat(o_h, k)
        pos = np.repeat(start[g], k) + (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k))
        idx = pind[pos]
    sid = sid_s[off_s[e_h] + idx]
    return pd.DataFrame({"ev": e_h, "obj": o_h, "sid": sid}).drop_duplicates()


def match(da, db, ia, ib, nev_a, nev_b):
    """Mutual best matches of objects between arms (common events ia/ib of nev_a/nev_b events), by shared straws."""
    ma = np.full(nev_a, -1, np.int64)
    ma[ia] = np.arange(len(ia))
    mb = np.full(nev_b, -1, np.int64)
    mb[ib] = np.arange(len(ib))
    A = da.assign(ce=ma[da["ev"].to_numpy()])
    B = db.assign(ce=mb[db["ev"].to_numpy()])
    A, B = A[A.ce >= 0], B[B.ce >= 0]
    na = A.groupby(["ce", "obj"]).size().rename("na")
    nb = B.groupby(["ce", "obj"]).size().rename("nb")
    j = A.merge(B, on=["ce", "sid"], suffixes=("_a", "_b"))
    s = j.groupby(["ce", "obj_a", "obj_b"]).size().rename("shared").reset_index()
    s = s.join(na, on=["ce", "obj_a"]).join(nb, on=["ce", "obj_b"])
    s = s[(s.shared >= MIN_SHARED) & (2 * s.shared >= np.minimum(s.na, s.nb))]
    best_a = s.sort_values(["shared"], ascending=False, kind="stable").drop_duplicates(["ce", "obj_a"])
    best_b = s.sort_values(["shared"], ascending=False, kind="stable").drop_duplicates(["ce", "obj_b"])
    m = best_a.merge(best_b, on=["ce", "obj_a", "obj_b", "shared", "na", "nb"])
    m["jaccard"] = m.shared / (m.na + m.nb - m.shared)
    m["ev_a"], m["ev_b"] = ia[m.ce.to_numpy()], ib[m.ce.to_numpy()]
    return m


def _per_object(ev, prefix_n, cols, sel=None):
    """Flat per-object columns with (event, object-in-event) keys; sel restricts to a collection."""
    nev = ak.to_numpy(ak.num(ev[prefix_n], axis=1))
    evo = np.repeat(np.arange(len(nev)), nev)
    obj = np.arange(len(evo)) - np.repeat(np.cumsum(nev) - nev, nev)
    d = {"ev": evo, "obj": obj, **{c: ak.to_numpy(ak.flatten(ev[c])) for c in cols}}
    df = pd.DataFrame(d)
    return df if sel is None else df[sel(df)].assign(obj=lambda x: x.groupby("ev").cumcount())


def _subset(ev, n, ind, coll, c):
    """ev with fields _n and _ind: n and ind restricted to the objects of collection c."""
    keep = ev[coll] == c
    cnt = ev[n]
    rep = np.repeat(ak.to_numpy(ak.flatten(keep)), ak.to_numpy(ak.flatten(cnt)))
    ind_k = ak.unflatten(ak.to_numpy(ak.flatten(ev[ind]))[rep], ak.to_numpy(ak.sum(ak.where(keep, cnt, 0), axis=1)))
    return ak.with_field(ak.with_field(ev, cnt[keep], "_n"), ind_k, "_ind")


def _rows_equal(x, y):
    """Per event: the jagged rows x and y have the same length and the same values."""
    ok = ak.to_numpy(ak.num(x, axis=1) == ak.num(y, axis=1))
    eq = np.zeros(len(ok), bool)
    if ok.any():
        cnt = ak.to_numpy(ak.num(x[ok], axis=1))
        bad = np.bincount(np.repeat(np.arange(len(cnt)), cnt), minlength=len(cnt),
                          weights=ak.to_numpy(ak.flatten(x[ok])) != ak.to_numpy(ak.flatten(y[ok])))
        eq[np.flatnonzero(ok)] = bad == 0
    return eq


def compare_hits(A, B, ia, ib):
    """Straw and panel hits one to one, when the digis are the same (else only the counts)."""
    a, b = A["ev"][ia], B["ev"][ib]
    same = np.ones(len(ia), bool)
    for k in DIGI:
        if k in A["fields"] and k in B["fields"]:
            same &= _rows_equal(a[k], b[k])
    out = {"events": int(len(ia)), "same_digis": _frac(same)}
    for pre in ("s_", "p_"):
        na, nb = ak.to_numpy(ak.num(a[pre + "sid"], axis=1)), ak.to_numpy(ak.num(b[pre + "sid"], axis=1))
        r = {"count_equal": _frac(na == nb), "count_equal_same_digis": _frac((na == nb)[same])}
        m = same & (na == nb)
        if m.any():
            aa, bb = a[m], b[m]
            for k in CH_INT:
                if pre + k in A["fields"]:
                    r[k + "_equal"] = _frac(ak.to_numpy(ak.flatten(aa[pre + k] == bb[pre + k])))
            for k in CH_FLOAT:
                if pre + k in A["fields"]:
                    r[k + "_absdiff"] = _q(np.abs(ak.to_numpy(ak.flatten(aa[pre + k] - bb[pre + k]))))
        out[pre.rstrip("_")] = r
    return out


def compare_tc(A, B, ia, ib):
    out = {}
    colls = sorted(set(np.unique(ak.to_numpy(ak.flatten(A["ev"]["tc_coll"])))) |
                   set(np.unique(ak.to_numpy(ak.flatten(B["ev"]["tc_coll"])))))
    for c in colls:
        ea = _subset(A["ev"], "tc_n", "tc_ind", "tc_coll", c)
        eb = _subset(B["ev"], "tc_n", "tc_ind", "tc_coll", c)
        da = _objects(ea, "_n", "_ind", panel=TC_PANEL.get(int(c), False))
        db = _objects(eb, "_n", "_ind", panel=TC_PANEL.get(int(c), False))
        m = match(da, db, ia, ib, len(A["keys"]), len(B["keys"]))
        na = ak.to_numpy(ak.num(ea["_n"], axis=1))[ia]
        nb = ak.to_numpy(ak.num(eb["_n"], axis=1))[ib]
        ta = _per_object(A["ev"], "tc_coll", ["tc_coll", "tc_t0"], lambda d: d.tc_coll == c)
        tb = _per_object(B["ev"], "tc_coll", ["tc_coll", "tc_t0"], lambda d: d.tc_coll == c)
        mm = m.merge(ta.rename(columns={"ev": "ev_a", "obj": "obj_a"}), on=["ev_a", "obj_a"]) \
              .merge(tb.rename(columns={"ev": "ev_b", "obj": "obj_b"}), on=["ev_b", "obj_b"], suffixes=("_a", "_b"))
        dt0 = (mm.tc_t0_a - mm.tc_t0_b).to_numpy()
        out[str(int(c))] = {"objects": [int(na.sum()), int(nb.sum())], "count_equal": _frac(na == nb),
                            "matched": int(len(m)), "jaccard": _q(m.jaccard), "jaccard_gt_0.98": _frac(m.jaccard > 0.98),
                            "dt0": _q(dt0), "abs_dt0_lt_0.1": _frac(np.abs(dt0) < 0.1)}
    return out


def compare_hs(A, B, ia, ib):
    out = {}
    if "hs_n" not in A["fields"] or "hs_n" not in B["fields"]:
        return out
    colls = sorted(set(np.unique(ak.to_numpy(ak.flatten(A["ev"]["hs_coll"])))) |
                   set(np.unique(ak.to_numpy(ak.flatten(B["ev"]["hs_coll"])))))
    cols = ["hs_coll", "hs_status"] + [f"hs_{p}" for p in HS_PAR]
    for c in colls:
        ea = _subset(A["ev"], "hs_n", "hs_ind", "hs_coll", c)
        eb = _subset(B["ev"], "hs_n", "hs_ind", "hs_coll", c)
        m = match(_objects(ea, "_n", "_ind"), _objects(eb, "_n", "_ind"), ia, ib, len(A["keys"]), len(B["keys"]))
        na = ak.to_numpy(ak.num(ea["_n"], axis=1))[ia]
        nb = ak.to_numpy(ak.num(eb["_n"], axis=1))[ib]
        ta = _per_object(A["ev"], "hs_coll", cols, lambda d: d.hs_coll == c)
        tb = _per_object(B["ev"], "hs_coll", cols, lambda d: d.hs_coll == c)
        mm = m.merge(ta.rename(columns={"ev": "ev_a", "obj": "obj_a"}), on=["ev_a", "obj_a"]) \
              .merge(tb.rename(columns={"ev": "ev_b", "obj": "obj_b"}), on=["ev_b", "obj_b"], suffixes=("_a", "_b"))
        r = {"objects": [int(na.sum()), int(nb.sum())], "count_equal": _frac(na == nb), "matched": int(len(m)),
             "jaccard": _q(m.jaccard), "status_equal": _frac(mm.hs_status_a == mm.hs_status_b)}
        for p in HS_PAR:
            r["d_" + p] = _q(mm[f"hs_{p}_a"] - mm[f"hs_{p}_b"])
        out[str(int(c))] = r
    return out


def compare_fit(A, B, ia, ib):
    cols = ["kl_status", "kl_chi2", "kl_ndof", "kl_fitcon", "kl_nactive", "kl_segmom", "kl_segmomerr"]
    pars = [f"lh_{p}" for p in LH_PAR if f"kl_lh_{p}" in A["fields"] and f"kl_lh_{p}" in B["fields"]]
    cols += [f"kl_{p}" for p in pars] + [f"kl_{p}err" for p in pars]
    m = match(_objects(A["ev"], "kl_nhit", "kh_ind"), _objects(B["ev"], "kl_nhit", "kh_ind"), ia, ib,
              len(A["keys"]), len(B["keys"]))
    na = ak.to_numpy(ak.num(A["ev"]["kl_nhit"], axis=1))[ia]
    nb = ak.to_numpy(ak.num(B["ev"]["kl_nhit"], axis=1))[ib]
    ta = _per_object(A["ev"], "kl_nhit", cols)
    tb = _per_object(B["ev"], "kl_nhit", cols)
    mm = m.merge(ta.rename(columns={"ev": "ev_a", "obj": "obj_a"}), on=["ev_a", "obj_a"]) \
          .merge(tb.rename(columns={"ev": "ev_b", "obj": "obj_b"}), on=["ev_b", "obj_b"], suffixes=("_a", "_b"))
    r = {"objects": [int(na.sum()), int(nb.sum())], "count_equal": _frac(na == nb), "matched": int(len(m)),
         "jaccard": _q(m.jaccard), "status_equal": _frac(mm.kl_status_a == mm.kl_status_b),
         "chi2ndf_rel": _q((mm.kl_chi2_a / mm.kl_ndof_a) / (mm.kl_chi2_b / mm.kl_ndof_b) - 1),
         "fitcon_diff": _q(mm.kl_fitcon_a - mm.kl_fitcon_b), "nactive_diff": _q(mm.kl_nactive_a - mm.kl_nactive_b)}
    for p in pars + ["segmom"]:
        e = mm[f"kl_{p}err_b"] if p != "segmom" else mm["kl_segmomerr_b"]
        pull = ((mm[f"kl_{p}_a"] - mm[f"kl_{p}_b"]) / e.where(e > 0)).to_numpy()
        r["pull_" + p] = _q(pull)
        r["abs_pull_lt_0.1_" + p] = _frac(np.abs(pull[np.isfinite(pull)]) < 0.1)
    # hit states of the matched fits, joined by straw
    ha = _hits(A["ev"]).rename(columns={"ev": "ev_a", "trk": "obj_a"})
    hb = _hits(B["ev"]).rename(columns={"ev": "ev_b", "trk": "obj_b"})
    hj = m[["ev_a", "obj_a", "ev_b", "obj_b"]].merge(ha, on=["ev_a", "obj_a"]).merge(
        hb, on=["ev_b", "obj_b", "sid"], suffixes=("_a", "_b"))
    r["hits_joined"] = int(len(hj))
    r["hit_active_equal"] = _frac(hj.act_a == hj.act_b)
    r["hit_ambig_equal"] = _frac(hj.amb_a == hj.amb_b)
    return r


def _hits(ev):
    nev = ak.to_numpy(ak.num(ev["kh_trk"], axis=1))
    return pd.DataFrame({"ev": np.repeat(np.arange(len(nev)), nev),
                         "trk": ak.to_numpy(ak.flatten(ev["kh_trk"])), "sid": ak.to_numpy(ak.flatten(ev["kh_sid"])),
                         "act": ak.to_numpy(ak.flatten(ev["kh_active"])), "amb": ak.to_numpy(ak.flatten(ev["kh_ambig"]))
                         }).drop_duplicates(["ev", "trk", "sid"])


def compare(A, B):
    ia, ib = match_events(A["keys"], B["keys"])
    res = {"events": [int(len(A["keys"])), int(len(B["keys"])), int(len(ia))], "hits": compare_hits(A, B, ia, ib)}
    if "tc_n" in A["fields"] and "tc_n" in B["fields"]:
        res["timeclusters"] = compare_tc(A, B, ia, ib)
        res["helices"] = compare_hs(A, B, ia, ib)
        res["fits"] = compare_fit(A, B, ia, ib)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m trkfastsim.reco.stagecompare", description=__doc__.split("\n")[0])
    ap.add_argument("--a", nargs="+", required=True, help="dump files of arm A")
    ap.add_argument("--b", nargs="+", required=True, help="dump files of arm B (the reference)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = compare(load(a.a), load(a.b))
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
