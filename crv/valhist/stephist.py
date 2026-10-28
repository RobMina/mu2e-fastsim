"""ValCrvStep histograms refilled from CrvStep tables, with exact event-clustered variances, and the deck comparison.

Offline Validation/src/ValCrvStep.cc booking and ROOT fixed-bin lookup are reproduced (float32 CrvStep members promoted
to double), so a refill equals the art histogram bin for bin (`closure` checks it). Step start y/z per sector group use
the CRVDQM v3 extracted booking of `steplayers.posyz`. Every histogram keeps, per bin with flow, the entries n and
w2 = sum over events of n_eb^2; components split the steps by origin (primary muon, muon descendant, non-muon).

Statistics per (reference, candidate) pair, bins with entries, flow excluded, first bin cut for E/SMom/SMom2:
  deck   chi2/ndf with Poisson variance n1+n2 (position y/z: exact clustered variance, as the deck), flag > FLAG
  exact  chi2/ndf with the exact clustered variance w2_1+w2_2 of independent arms
  paired chi2/ndf with sum_e (c_eb-r_eb)^2, valid only when both arms are on the same generated events
Yields include flow; their SE are the clustered independent sqrt(W1+W2) and the paired sqrt(sum_e (c_e-r_e)^2).
All are conditional on the fixed library; none is a calibrated p-value.

    python -m valhist.stephist closure --tables 'DIR/fast1b.{s}.root' --root 'DIR/s{k:08d}/a1/valhist.root' \
        --dir Validation/CrvSteps_fastCrvSteps_noName --shards 11 --out J
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import tables
from crvfastsim.analyses.census import classify
from . import steplayers as S

# ValCrvStep booking: name -> (nbins, lo, hi)
BOOK = {"NStep": (101, -9.5, 1000.5), "bar": (100, -0.5, 5503.5), "t": (100, 0.0, 2000.0),
        "t2": (100, -100.0, 100.0e3), "lt": (100, 0.0, 10.0), "E": (100, 0.0, 0.2), "lE": (100, -9.0, 2.0),
        "posx": (100, -7500.0, 1000.0), "SMom": (100, 0.0, 100.0), "SMom2": (100, 0.0, 100000.0),
        "pathLen": (100, 0.0, 100.0)}
FIRST_BIN_CUT = {"E", "SMom", "SMom2"}
TARGET = ("E", "lE", "pathLen", "bar", "NStep", "SMom")
COMPONENTS = ("all", "mu", "desc", "nonmu")
COMP_OF = {"mu_primary": "mu", "mu_descendant": "desc", "nonmu_primary": "nonmu", "nonmu_descendant": "nonmu"}
FLAG = 2.0
STEP_COLS = tables.KEY + ["bar", "sp", "t0", "edep", "len", "x0", "y0", "z0", "px0", "py0", "pz0"]


def root_bin(x, nbins, lo, hi):
    """TAxis::FindBin for a fixed-width axis: 0 underflow, nbins+1 overflow."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(len(x), np.int64)
    under, over = x < lo, ~(x < hi)
    mid = ~(under | over)
    out[under], out[over] = 0, nbins + 1
    out[mid] = 1 + (nbins * (x[mid] - lo) / (hi - lo)).astype(np.int64)
    return out


def values(s):
    """Per step, the value ValCrvStep fills for each histogram (CrvStep float members promoted to double)."""
    f32 = lambda c: s[c].to_numpy().astype(np.float32).astype(np.float64)
    e = s.edep.to_numpy().astype(np.float32)
    t = s.t0.to_numpy().astype(np.float64)
    p = np.sqrt(f32("px0") ** 2 + f32("py0") ** 2 + f32("pz0") ** 2)
    with np.errstate(divide="ignore"):
        le = np.where(e > 0, np.log10(e, dtype=np.float32).astype(np.float64), -10.0)
    return {"bar": s.bar.to_numpy().astype(np.float64), "t": t, "t2": t, "lt": np.log10(np.maximum(t, 1e-5)),
            "E": e.astype(np.float64), "lE": le, "posx": f32("x0"), "SMom": p, "SMom2": p, "pathLen": f32("len")}


def clustered(ev, idx, nb):
    """Entries per bin and the exact clustered second moment sum_e n_eb^2."""
    n = np.bincount(idx, minlength=nb).astype(float)
    if not len(idx):
        return n, np.zeros(nb)
    u, c = np.unique(np.asarray(ev, np.int64) * nb + idx, return_counts=True)
    return n, np.bincount(u % nb, weights=c.astype(float) ** 2, minlength=nb)


_GEOM = {}


def _bars():
    if not _GEOM:
        geom, bt = S._geom()
        _GEOM.update(geom=geom, grp=bt.grp.to_numpy())
    return _GEOM


def bin_indices(steps, comp, events, edges):
    """Per histogram key, (event id, bin index, component) arrays of one arm and shard; NStep per event; _total keys
    count every step (one bin) per sector group, for the yields and their clustered second moments."""
    ev = S.evkey(steps)
    v = values(steps)
    out = {k: (ev, root_bin(v[k], *BOOK[k]), comp) for k in v}
    eid = S.evkey(events)
    per = pd.Series(np.ones(len(ev)), index=ev).groupby(level=0).size()
    if not pd.Index(per.index).isin(eid).all():
        raise ValueError("steps outside the generated event population")
    m = pd.Series(0, index=eid).add(per, fill_value=0).to_numpy()
    out["NStep"] = (np.asarray(eid, np.int64), root_bin(m, *BOOK["NStep"]), np.full(len(eid), "all", object))
    grp = _bars()["grp"][steps.bar.to_numpy()]
    zero = np.zeros(len(steps), np.int64)
    for g in S.POS_GROUPS:
        m_ = np.ones(len(steps), bool) if g == "all" else grp == g
        out[f"_total/{g}"] = (ev[m_], zero[m_], comp[m_])
    for v_, e in edges.items():
        pos = steps[f"{v_}0"].to_numpy().astype(np.float32).astype(np.float64)
        idx = np.searchsorted(e, pos, side="right")
        for g in S.POS_GROUPS:
            m_ = np.ones(len(steps), bool) if g == "all" else grp == g
            out[f"{g}/{v_}"] = (ev[m_], idx[m_], comp[m_])
    return out


def nbins_of(key, edges):
    if key.startswith("_total"):
        return 1
    return BOOK[key][0] + 2 if key in BOOK else len(edges[key.split("/")[1]]) + 1


def fill(steps, parts, events, edges):
    """{key: {component: (n, w2)}} for one arm and shard, plus the indices kept for pairing."""
    s = classify(steps, parts, _bars()["geom"].sector)
    if len(s) != len(steps):
        raise ValueError("classification changed the step count")
    comp = s.cls.map(COMP_OF).fillna("unknown").to_numpy(dtype=object)
    idx = bin_indices(steps, comp, events, edges)
    h = {}
    for k, (ev, b, c) in idx.items():
        nb = nbins_of(k, edges)
        h[k] = {"all": clustered(ev, b, nb)}
        if k != "NStep":
            for name in COMPONENTS[1:] + ("unknown",):
                m = c == name
                if m.any() or name != "unknown":
                    h[k][name] = clustered(ev[m], b[m], nb)
    return h, idx


def paired(ref_idx, cand_idx, edges):
    """Per key, sum over events of (c_eb - r_eb)^2 per bin and of (c_e - r_e)^2 for the total, same events."""
    out = {}
    for k in ref_idx:
        nb = nbins_of(k, edges)
        (er, br, _), (ec, bc, _) = ref_idx[k], cand_idx[k]
        d = pd.DataFrame({"ev": np.r_[er, ec], "b": np.r_[br, bc], "w": np.r_[-np.ones(len(br)), np.ones(len(bc))]})
        cell = d.groupby(["ev", "b"], sort=False).w.sum()
        bins = cell.index.get_level_values("b").to_numpy()
        out[k] = np.bincount(bins, weights=cell.to_numpy() ** 2, minlength=nb)
    return out


def add(acc, new):
    for k, comps in new.items():
        a = acc.setdefault(k, {})
        for c, (n, w2) in comps.items():
            if c in a:
                a[c] = (a[c][0] + n, a[c][1] + w2)
            else:
                a[c] = (n.copy(), w2.copy())
    return acc


def add_paired(acc, new):
    for k, d2 in new.items():
        acc[k] = acc[k] + d2 if k in acc else d2.copy()
    return acc


def chi2(n1, n2, var, cut):
    a, b, v = np.asarray(n1)[1:-1], np.asarray(n2)[1:-1], np.asarray(var)[1:-1]
    if cut:
        a, b, v = a[1:], b[1:], v[1:]
    m = (a + b) > 0
    if not m.any():
        return None, 0
    return float(((b[m] - a[m]) ** 2 / np.maximum(v[m], 1e-300)).sum() / int(m.sum())), int(m.sum())


def yield_stat(ref, cand, comp="all", d2=None):
    """Relative yield difference with the clustered independent and (same events) paired standard errors."""
    (n1, w1), (n2, w2) = ref[comp], cand[comp]
    t1, t2 = float(n1.sum()), float(n2.sum())
    r = {"n_ref": t1, "n_cand": t2, "rel": t2 / t1 - 1 if t1 else None,
         "rel_se_independent": float(np.sqrt(w1.sum() + w2.sum()) / t1) if t1 else None}
    if d2 is not None and t1:
        r["rel_se_paired"] = float(np.sqrt(d2.sum()) / t1)
    if t1 and r["rel_se_independent"]:
        r["pull_independent"] = r["rel"] / r["rel_se_independent"]
    if "rel_se_paired" in r and r["rel_se_paired"]:
        r["pull_paired"] = r["rel"] / r["rel_se_paired"]
    return r


def compare(ref, cand, pair=None):
    """Per key: deck/exact/paired chi2/ndf and the deck flag; per _total key the yield (component 'all') and the
    component yields with their independent clustered errors."""
    out = {}
    for k in ref:
        d2 = pair.get(k) if pair is not None else None
        if k.startswith("_total"):
            out[k] = yield_stat(ref[k], cand[k], "all", d2)
            out[k]["components"] = {c: yield_stat(ref[k], cand[k], c) for c in COMPONENTS[1:]
                                    if c in ref[k] and c in cand[k]}
            continue
        (n1, w1), (n2, w2) = ref[k]["all"], cand[k]["all"]
        cut = k in FIRST_BIN_CUT
        pos = "/" in k
        deck, ndf = chi2(n1, n2, (w1 + w2) if pos else (n1 + n2), cut)
        exact, _ = chi2(n1, n2, w1 + w2, cut)
        r = {"deck": deck, "ndf": ndf, "exact": exact, "flag": deck is not None and deck > FLAG,
             "variance": "exact clustered" if pos else "Poisson"}
        if d2 is not None:
            r["paired"], _ = chi2(n1, n2, d2, cut)
        out[k] = r
    return out


def to_json(acc):
    return {k: {c: {"n": n.tolist(), "w2": w.tolist()} for c, (n, w) in comps.items()} for k, comps in acc.items()}


def from_json(js):
    return {k: {c: (np.asarray(v["n"]), np.asarray(v["w2"])) for c, v in comps.items()} for k, comps in js.items()}


# ------------------------------------------------------------------------------------------------- closure vs art
def closure(tables_pat, root_pat, directory, shard_list, gen_pat, out):
    """Refill each shard's table and compare with the art ValCrvStep histograms bin by bin (flow included)."""
    import uproot
    res = []
    for k in shard_list:
        f = tables_pat.format(s=S.shard(k), k=k)
        steps, parts = S._read(f, STEP_COLS)
        events = tables.read(gen_pat.format(s=S.shard(k), k=k) if gen_pat else f, "gen", tables.KEY).drop_duplicates()
        h, _ = fill(steps, parts, events, S.booking_edges())
        row = {"shard": k, "table": f, "root": root_pat.format(s=S.shard(k), k=k), "mismatch": {}}
        with uproot.open(row["root"]) as r:
            for key in BOOK:
                art = r[f"{directory}/{key}"].values(flow=True)
                diff = np.abs(art - h[key]["all"][0])
                if diff.sum():
                    row["mismatch"][key] = {"entries": float(diff.sum() / 2), "bins": int((diff > 0).sum())}
        row["entries"] = float(h["bar"]["all"][0].sum())
        res.append(row)
        print(json.dumps(row), flush=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("closure", help="refill tables and compare with art ValCrvStep histograms")
    c.add_argument("--tables", required=True)
    c.add_argument("--root", required=True)
    c.add_argument("--dir", required=True, help="ValCrvStep directory, e.g. Validation/CrvSteps_steps_noName")
    c.add_argument("--gen", help="table pattern holding the full generated population (default: --tables)")
    c.add_argument("--shards", default="11")
    c.add_argument("--out", required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    closure(a.tables, a.root, a.dir, S.shards(a.shards), a.gen, a.out)


if __name__ == "__main__":
    main()
