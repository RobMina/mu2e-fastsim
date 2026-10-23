"""Fast-chain deadtime frames vs the stock coincidences of the same events (F-fast validation, plan V-E4).

Stock: the EventNtuple crvcoincs of the events (nts of the same subruns). Fast: crvfastsim.veto.fastframes output.
Reports frames, coincidences per frame, fraction of frames with a coincidence, PE and time distributions (quantiles,
two-sample KS), per-event agreement (same events with/without a coincidence), and, with a selection parquet, F
deadtime on both frame sets for a card.

    python -m crvfastsim.veto.compareframes FAST.parquet [...] --nts nts.root [...] [--selection SEL.parquet ...]
        [--cosmic-ana CUTS.yaml CUTSET | --card JSON] [--pmin 85 --pmax 200] --outdir DIR
"""
from __future__ import annotations

import argparse
import json
import os

import awkward as ak
import numpy as np

COLS = ("time", "timeStart", "timeEnd", "PEs", "nLayers", "sectorType")


def stock_frames(paths, keys=None):
    """EventNtuple crvcoincs per event as frames (run, subrun, event, coincs); keys: restrict to these event keys."""
    import uproot
    out = []
    for p in paths:
        with uproot.open(p) as f:
            t = f["EventNtuple/ntuple"]
            ev = {k: t[f"evtinfo/{k}"].array(library="np") for k in ("run", "subrun", "event")}
            co = t.arrays([f"crvcoincs.{c}" for c in COLS], library="ak")
        r = ak.zip({"run": ev["run"].astype(np.int64), "subrun": ev["subrun"].astype(np.int64),
                    "event": ev["event"].astype(np.int64),
                    "coincs": ak.zip({c: co[f"crvcoincs.{c}"] for c in COLS})}, depth_limit=1)
        out.append(r)
    a = ak.concatenate(out)
    if keys is not None:
        a = a[np.isin(evkey(a), keys)]
    return a


def evkey(a):
    return (ak.to_numpy(a.subrun).astype(np.int64) << 20) | ak.to_numpy(a.event).astype(np.int64)


def summary(fr):
    n = len(fr)
    nc = ak.to_numpy(ak.num(fr.coincs.time))
    pe = ak.to_numpy(ak.flatten(fr.coincs.PEs))
    t = ak.to_numpy(ak.flatten(fr.coincs.time))
    q = lambda x: [float(v) for v in np.percentile(x, [10, 50, 90])] if len(x) else []
    return {"frames": int(n), "coincs": int(nc.sum()), "frac_frames_with_coinc": float((nc > 0).mean()) if n else float("nan"),
            "coincs_per_frame": float(nc.mean()) if n else float("nan"), "pe_q10_50_90": q(pe), "time_q10_50_90": q(t),
            "sectors": {int(s): int(c) for s, c in zip(*np.unique(ak.to_numpy(ak.flatten(fr.coincs.sectorType)), return_counts=True))}}


def compare(fast, stock):
    from scipy.stats import ks_2samp
    kf, ks = evkey(fast), evkey(stock)
    common = np.intersect1d(kf, ks)
    f = fast[np.isin(kf, common)][np.argsort(kf[np.isin(kf, common)])]
    s = stock[np.isin(ks, common)][np.argsort(ks[np.isin(ks, common)])]
    hf, hs = ak.to_numpy(ak.num(f.coincs.time)) > 0, ak.to_numpy(ak.num(s.coincs.time)) > 0
    out = {"common_frames": int(len(common)), "fast": summary(f), "stock": summary(s),
           "both": int((hf & hs).sum()), "fast_only": int((hf & ~hs).sum()), "stock_only": int((~hf & hs).sum())}
    for c in ("PEs", "time"):
        a, b = ak.to_numpy(ak.flatten(f.coincs[c])), ak.to_numpy(ak.flatten(s.coincs[c]))
        out[f"ks_{c}"] = float(ks_2samp(a, b).pvalue) if len(a) and len(b) else float("nan")
    # per-event time shift on events where both have exactly one coincidence
    one = (ak.to_numpy(ak.num(f.coincs.time)) == 1) & (ak.to_numpy(ak.num(s.coincs.time)) == 1)
    if one.any():
        d = ak.to_numpy(ak.firsts(f.coincs.time[one])) - ak.to_numpy(ak.firsts(s.coincs.time[one]))
        r = ak.to_numpy(ak.firsts(f.coincs.PEs[one])) / ak.to_numpy(ak.firsts(s.coincs.PEs[one]))
        out["matched_single"] = {"n": int(one.sum()), "dt_median": float(np.median(d)), "dt_q10_90": [float(x) for x in np.percentile(d, [10, 90])],
                                 "pe_ratio_median": float(np.median(r))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fast", nargs="+")
    ap.add_argument("--nts", nargs="+", required=True)
    ap.add_argument("--selection", nargs="*")
    ap.add_argument("--cosmic-ana", nargs=2, metavar=("CUTS_YAML", "CUTSET"))
    ap.add_argument("--card")
    ap.add_argument("--pmin", type=float, default=85.0)
    ap.add_argument("--pmax", type=float, default=200.0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    fast = ak.concatenate([ak.from_parquet(p) for p in a.fast])
    stock = stock_frames(a.nts, evkey(fast))
    out = compare(fast, stock)
    if a.selection and (a.cosmic_ana or a.card):
        from .card import VetoCard
        from .deadtime import factorised, read_coincs
        card = VetoCard.from_dict(json.load(open(a.card))) if a.card else VetoCard.from_mu2e_cosmic_ana(*a.cosmic_ana)
        ev, _ = read_coincs(a.selection)
        t, p = ak.to_numpy(ak.flatten(ev["t_mid"])), ak.to_numpy(ak.flatten(ev["p_mid"]))
        t = t[(p > a.pmin) & (p < a.pmax)]
        kf = evkey(fast)
        common = np.isin(kf, evkey(stock))
        out["F_fast"] = factorised(t, fast[common].coincs, card)
        out["F_stock_same_frames"] = factorised(t, stock.coincs, card)
        out["card"] = card.__dict__ | {"sha": card.sha()}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(a.outdir, "compare_frames.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
