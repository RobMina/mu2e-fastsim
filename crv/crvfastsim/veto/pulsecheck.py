"""F-fast diagnostic: fast pulses vs the stock pulses of the same events, per (event, channel), on one mixed-step
subrun (crvfastsim.veto.fastframes inputs). Reports: channels with a pulse in stock only / fast only / both, the PE
ratio and time difference on matched channels, both split by sector and PE bin, and the step-level input census
(steps, steps with an unknown pdg, steps per event) so that a missing input shows up before the response does.

    python -m crvfastsim.veto.pulsecheck mixsteps.root --conditions crvreco_mcs.root --photon-dev CRVPhoton.txt
        --pbtime nts.root [--run-config run1a_v01] [--max-events N] --outdir DIR
"""
from __future__ import annotations

import argparse
import json
import os

import awkward as ak
import numpy as np

from .. import chain, geometry, tables
from ..reco import pulsefit, reftables
from ..response import photons as PH
from ..response import sipm as SI
from . import fastframes as FF


def fast_pulses(st, pa, cond, run, seed=1, chunk=500):
    resp, fcfg, counters, window = FF.run_config(run)
    geom = geometry.from_offline_config(geometry.find_counters(counters))
    cfg = chain.ChainConfig(response=resp, window=window)
    tabs, pmap = PH.load_tables(resp), SI.photon_map()
    rng = np.random.default_rng(seed)
    sk = chain.event_key(st)
    ue = np.unique(sk)
    out = {k: [] for k in ("ev", "chan", "pe", "time")}
    qs = {k: [] for k in ("key", "n", "t0")}
    nq = 0
    for a in range(0, len(ue), chunk):
        sub = st[np.isin(sk, ue[a:a + chunk])].reset_index(drop=True)
        Q = chain.charges(sub, pa, geom, cond, cfg, rng, tabs=tabs, pmap=pmap)
        nq += len(Q["time"])
        if not len(Q["time"]):
            continue
        lab = np.asarray(Q["chan"])                               # event key * CHAN_MULT + channel
        kq = (lab // chain.CHAN_MULT) * 8192 + lab % chain.CHAN_MULT
        u, inv = np.unique(kq, return_inverse=True)
        qs["key"].append(u); qs["n"].append(np.bincount(inv, weights=np.asarray(Q["charge"]) / np.mean(Q["charge"])))
        t0q = np.full(len(u), np.inf); np.minimum.at(t0q, inv, np.asarray(Q["time"])); qs["t0"].append(t0q)
        D = chain.digis(Q, cond, cfg, rng)
        if not len(D["chan"]):
            continue
        W = chain.reco_waveforms(D, cond)
        P = pulsefit.reconstruct(W["adc"], W["n"], W["start"], W["ped"], W["area"], W["height"], pulsefit.RecoConfig())
        P["time"] = P["time"] + cond.timeoffset[W["chan"]][P["wf"]]
        out["ev"].append(W["ev"][P["wf"]]); out["chan"].append(W["chan"][P["wf"]])
        out["pe"].append(P["pe"]); out["time"].append(P["time"])
    Qs = {k: np.concatenate(v) if v else np.zeros(0) for k, v in qs.items()}
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}, geom, nq, Qs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--photon-dev")
    ap.add_argument("--pbtime", nargs="+", required=True)
    ap.add_argument("--run-config", default="run1a_v01")
    ap.add_argument("--max-events", type=int, default=2000)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    cond = FF.conditions_from_dump(a.conditions, a.photon_dev)
    kp, wp = FF.read_pbtime(a.pbtime)
    st = tables.read(a.steps, "steps")
    pa = tables.read(a.steps, "particles")
    keep = np.sort(np.unique(chain.event_key(st)))[:a.max_events]
    st = st[np.isin(chain.event_key(st), keep)].reset_index(drop=True)
    pdg = st.merge(pa[tables.KEY + ["key", "pdg"]].rename(columns={"key": "sp"}), on=tables.KEY + ["sp"],
                   how="left").pdg.fillna(0).to_numpy()
    mass, charge, known = PH.particle_props(pdg)
    census = {"events_with_steps": int(len(keep)), "steps": int(len(st)), "steps_unknown_pdg": int((~known).sum()),
              "edep_unknown_pdg_frac": float(st.edep.to_numpy()[~known].sum() / max(st.edep.sum(), 1e-12)),
              "pdg_top": {int(k): int(c) for k, c in zip(*np.unique(pdg, return_counts=True))
                          if c >= 0.01 * len(pdg) or k == 0}}
    st, _ = FF.fold_steps(st, kp, wp)
    F, geom, nq, Qs = fast_pulses(st, pa, cond, a.run_config)
    F["time"] = F["time"] + wp[np.argsort(kp)][np.clip(np.searchsorted(np.sort(kp), F["ev"]), 0, len(kp) - 1)]
    S = reftables.read([a.conditions], "pulses", ["subrun", "event", "bar", "sipm", "pe", "time"])
    sev = (ak.to_numpy(S.subrun).astype(np.int64) << 20) | ak.to_numpy(S.event).astype(np.int64)
    m = np.isin(sev, keep)
    S = {"ev": sev[m], "chan": (ak.to_numpy(S.bar) * 4 + ak.to_numpy(S.sipm))[m], "pe": ak.to_numpy(S.pe)[m],
         "time": ak.to_numpy(S.time)[m]}
    # strongest pulse per (event, channel) on each side
    def best(X):
        k = X["ev"] * 8192 + X["chan"]
        o = np.lexsort((-X["pe"], k))
        f = np.r_[True, k[o][1:] != k[o][:-1]]
        return {kk: v[o][f] for kk, v in X.items()} | {"key": k[o][f]}
    Fb, Sb = best(F), best(S)
    both = np.intersect1d(Fb["key"], Sb["key"])
    fi, si = np.searchsorted(Fb["key"], both), np.searchsorted(Sb["key"], both)
    sec = geom.sector[(Sb["chan"] // 4).astype(int)]
    names = list(geom.sector_names)
    res = {"census": census, "fast_charges": int(nq), "fast_pulses": int(len(F["ev"])), "stock_pulses": int(len(S["ev"])),
           "chan_both": int(len(both)), "chan_stock_only": int(len(Sb["key"]) - len(both)),
           "chan_fast_only": int(len(Fb["key"]) - len(both))}
    r = Fb["pe"][fi] / Sb["pe"][si]
    dt = Fb["time"][fi] - Sb["time"][si]
    res["matched_pe_ratio_q25_50_75"] = [float(x) for x in np.percentile(r, [25, 50, 75])]
    res["matched_dt_q10_50_90"] = [float(x) for x in np.percentile(dt, [10, 50, 90])]
    so = ~np.isin(Sb["key"], both)
    res["stock_only_pe_q10_50_90"] = [float(x) for x in np.percentile(Sb["pe"][so], [10, 50, 90])] if so.any() else []
    # stock-only channels: did the fast chain make SiPM charges there, and when (folded frame + w = reco frame)?
    ks = Sb["key"][so]
    j = np.clip(np.searchsorted(Qs["key"], ks), 0, max(len(Qs["key"]) - 1, 0))
    hasq = (Qs["key"][j] == ks) if len(Qs["key"]) else np.zeros(len(ks), bool)
    w_of = lambda ev: wp[np.argsort(kp)][np.clip(np.searchsorted(np.sort(kp), ev), 0, len(kp) - 1)]
    res["stock_only_detail"] = {
        "n": int(so.sum()), "with_fast_charges": int(hasq.sum()),
        "fast_charge_pe_q10_50_90": [float(x) for x in np.percentile(Qs["n"][j][hasq], [10, 50, 90])] if hasq.any() else [],
        "stock_time_q10_50_90": [float(x) for x in np.percentile(Sb["time"][so], [10, 50, 90])] if so.any() else [],
        "fast_charge_t0_minus_stock_q10_50_90": [float(x) for x in np.percentile(
            Qs["t0"][j][hasq] + w_of(Sb["ev"][so][hasq]) - Sb["time"][so][hasq], [10, 50, 90])] if hasq.any() else []}
    res["fast_charges_by_sector"] = {n: int(np.isin(geom.sector[((Qs["key"] % 8192) // 4).astype(int)], [s]).sum())
                                     for s, n in enumerate(geom.sector_names)}
    res["by_sector"] = {}
    for s, n in enumerate(names):
        ms = sec == s
        msb = sec[si] == s
        res["by_sector"][n] = {"stock_channels": int(ms.sum()), "stock_only": int((ms & so).sum()),
                               "pe_ratio_median": float(np.median(r[msb])) if msb.any() else float("nan"),
                               "stock_pe_sum": float(Sb["pe"][ms].sum()),
                               "fast_pe_sum": float(Fb["pe"][geom.sector[(Fb["chan"] // 4).astype(int)] == s].sum())}
    edges = [0, 5, 10, 20, 40, 80, 1e9]
    res["stock_only_frac_by_pe"] = {f"{edges[i]}-{edges[i + 1]}": float(so[(Sb["pe"] >= edges[i]) & (Sb["pe"] < edges[i + 1])].mean())
                                    for i in range(len(edges) - 1) if ((Sb["pe"] >= edges[i]) & (Sb["pe"] < edges[i + 1])).any()}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(res, open(os.path.join(a.outdir, "pulsecheck.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
