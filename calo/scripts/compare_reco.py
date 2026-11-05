#!/usr/bin/env python3
"""Fast calorimeter reco vs stock, stage by stage on the same stock inputs (paired), from a stock_chain ntuple.

- reco digis: template fit of the stock CaloDigis vs the stock CaloRecoDigis linked to them;
- hits: CaloHitMaker grouping of the stock reco digis vs the stock CaloHits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np

from calofastsim import stockio
from calofastsim.cluster import Geometry, make_clusters
from calofastsim.hits import make_hits
from calofastsim.templatefit import FitConfig, fit_digis


def flat(a, name):
    return ak.to_numpy(ak.flatten(a[name]))


def stock_tables(ntuple, arm, events):
    pos, a = stockio.load_ntuple(ntuple, arm, events)
    d, r, h = a["calodigis"], a["calorecodigis"], a["calohits"]
    nd = ak.to_numpy(ak.num(d["calodigis.SiPMID_"]))
    nr = ak.to_numpy(ak.num(r["calorecodigis.eDep_"]))
    nh = ak.to_numpy(ak.num(h["calohits.eDep_"]))
    wf = d["calodigis.waveform_"]
    digis = {"event": np.repeat(pos, nd), "first": np.repeat(np.cumsum(nd) - nd, nd),
             "sipm": flat(d, "calodigis.SiPMID_"), "t0": flat(d, "calodigis.t0_"),
             "length": ak.to_numpy(ak.flatten(ak.num(wf, axis=2))),
             "samples": ak.to_numpy(ak.fill_none(ak.pad_none(ak.flatten(wf, axis=1), 128, clip=True), 0))}
    reco = {"event": np.repeat(pos, nr), "digi": flat(r, "calorecodigis.caloDigiIdx_") + np.repeat(np.cumsum(nd) - nd, nr),
            "sipm": flat(r, "calorecodigis.SiPMID_"), "e": flat(r, "calorecodigis.eDep_"),
            "e_err": flat(r, "calorecodigis.eDepErr_"), "t": flat(r, "calorecodigis.time_"),
            "t_err": flat(r, "calorecodigis.timeErr_"), "chi2": flat(r, "calorecodigis.chi2_"),
            "ndf": flat(r, "calorecodigis.ndf_")}
    hits = {"event": np.repeat(pos, nh), "crystal": flat(h, "calohits.crystalId_"), "nsipm": flat(h, "calohits.nSiPMs_"),
            "e": flat(h, "calohits.eDep_"), "e_err": flat(h, "calohits.eDepErr_"), "t": flat(h, "calohits.time_"),
            "t_err": flat(h, "calohits.timeErr_")}
    c = a["caloclusters"]
    nc = ak.to_numpy(ak.num(c["caloclusters.energyDep_"]))
    clusters = {"event": np.repeat(pos, nc), "energy": flat(c, "caloclusters.energyDep_"),
                "time": flat(c, "caloclusters.time_"), "x": flat(c, "caloclusters.cog_.fCoordinates.fX"),
                "y": flat(c, "caloclusters.cog_.fCoordinates.fY"), "z": flat(c, "caloclusters.cog_.fCoordinates.fZ"),
                "size": flat(c, "caloclusters.size_"), "disk": flat(c, "caloclusters.diskID_"),
                "split": flat(c, "caloclusters.isSplit_")}
    hits["clusters"] = clusters
    pbt = ak.to_numpy(a["evtinfo"]["pbtime"])
    return pos, digis, reco, hits, pbt


def fast_reco_digis(digis, pbt_event, pulse, max_chi2=2.0, cfg=FitConfig()):
    """Template fit of padded digis -> flat reco digis (digi index, sipm, E, E err, t, t err, chi2, ndf)."""
    fit = fit_digis(digis["t0"], digis["samples"], digis["length"], pulse, cfg)
    ok = fit["valid"] & ((fit["chi2"] / np.maximum(fit["ndf"], 1).astype(np.float32))[:, None] <= max_chi2)
    di, slot = np.nonzero(ok)
    order = np.lexsort((fit["peak_time"][di, slot], di))
    di, slot = di[order], slot[order]
    f32 = np.float32
    adc2mev = stockio.adc_to_mev(digis["sipm"][di])
    return {"digi": di, "event": digis["event"][di], "sipm": digis["sipm"][di],
            "e": (fit["amp"][di, slot] * adc2mev).astype(f32), "e_err": (fit["amp_err"][di, slot] * adc2mev).astype(f32),
            "t": (fit["time"][di, slot] - pbt_event[di]).astype(f32), "t_err": fit["time_err"][di, slot].astype(f32),
            "chi2": fit["chi2"][di].astype(f32), "ndf": fit["ndf"][di],
            "refit_skipped": fit["refit_skipped"][di]}, fit


def pair_by_digi(fast, stock, tmax=5.0):
    """Pair fast and stock reco digis from the same digi, nearest time within ``tmax`` ns (one-to-one in order)."""
    kf = fast["digi"] * 0 + fast["digi"]
    ks = stock["digi"]
    common = np.intersect1d(np.unique(kf), np.unique(ks))
    first_f = np.flatnonzero(np.r_[True, kf[1:] != kf[:-1]])
    first_s = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    i_f = first_f[np.searchsorted(kf[first_f], common)]
    i_s = first_s[np.searchsorted(ks[first_s], common)]
    close = np.abs(fast["t"][i_f] - stock["t"][i_s]) < tmax
    return i_f[close], i_s[close], common.size


def resid_stats(a, b, rel=False):
    d = (a - b) / np.where(b != 0, b, 1) if rel else a - b
    q = np.percentile(d, [1, 16, 50, 84, 99]) if d.size else [np.nan] * 5
    return {"n": int(d.size), "mean": float(np.mean(d)) if d.size else None, "rms": float(np.std(d)) if d.size else None,
            "q01_16_50_84_99": [float(v) for v in q], "frac_abs_lt_1pct" if rel else "frac_abs_lt_0p1":
                float(np.mean(np.abs(d) < (0.01 if rel else 0.1))) if d.size else None}


def compare_reco_digis(digis, reco, pbt, pos, pulse, cfg=FitConfig()):
    pbt_digi = pbt[np.searchsorted(pos, digis["event"])] if pos.size else np.zeros(0)
    fast, fit = fast_reco_digis(digis, pbt_digi.astype(np.float64), pulse, cfg=cfg)
    i_f, i_s, n_common = pair_by_digi(fast, reco)
    nf = np.bincount(fast["digi"], minlength=digis["t0"].size)
    ns = np.bincount(reco["digi"], minlength=digis["t0"].size)
    return {"n_digis": int(digis["t0"].size), "n_fast": int(fast["e"].size), "n_stock": int(reco["e"].size),
            "digis_same_npeaks": float(np.mean(nf == ns)), "n_paired": int(i_f.size),
            "refit_skipped_digis": int(fit["refit_skipped"].sum()),
            "energy_rel": resid_stats(fast["e"][i_f], reco["e"][i_s], rel=True),
            "time_ns": resid_stats(fast["t"][i_f], reco["t"][i_s]),
            "energy_err_rel": resid_stats(fast["e_err"][i_f], reco["e_err"][i_s], rel=True),
            "time_err_ns": resid_stats(fast["t_err"][i_f], reco["t_err"][i_s])}, fast


def compare_hits(reco, hits, n_events):
    """CaloHitMaker on the stock reco digis vs the stock hits, matched by (event, crystal, time order)."""
    got = make_hits(reco["event"].astype(np.int64), reco["sipm"].astype(np.int64), reco["t"], reco["t_err"],
                    reco["e"], reco["e_err"], n_events=n_events, n_crystals=1348, time4merge=6.0)
    kf = got["event_index"].astype(np.int64) * 2000 + got["crystal"]
    ks = hits["event"].astype(np.int64) * 2000 + hits["crystal"]
    of = np.lexsort((got["time"], kf))
    os_ = np.lexsort((hits["t"], ks))
    same_count = kf.size == ks.size and np.array_equal(kf[of], ks[os_])
    out = {"n_fast": int(kf.size), "n_stock": int(ks.size), "same_crystal_lists": bool(same_count)}
    if same_count:
        out["energy_exact_frac"] = float(np.mean(got["energy"][of] == hits["e"][os_]))
        out["time_exact_frac"] = float(np.mean(got["time"][of] == hits["t"][os_]))
        out["energy_rel"] = resid_stats(got["energy"][of], hits["e"][os_], rel=True)
        out["time_ns"] = resid_stats(got["time"][of], hits["t"][os_])
        out["nsipm_equal"] = float(np.mean(got["nSiPMs"][of] == hits["nsipm"][os_]))
    return out


def load_geometry(path):
    return Geometry.load(path)


def rank_in_event(event):
    first = np.r_[0, np.flatnonzero(np.diff(event)) + 1]
    counts = np.diff(np.r_[first, event.size])
    return np.arange(event.size) - np.repeat(first, counts)


def compare_clusters(fast, stock, n_events):
    """Clusters matched per event by energy rank (both sorted by decreasing energy)."""
    kf = fast["event"] * 1000 + rank_in_event(fast["event"])
    ks = stock["event"] * 1000 + rank_in_event(stock["event"])
    common, i_f, i_s = np.intersect1d(kf, ks, return_indices=True)
    nf = np.bincount(fast["event"], minlength=n_events)
    ns = np.bincount(stock["event"], minlength=n_events)
    pos = np.hypot(fast["x"][i_f] - stock["x"][i_s], fast["y"][i_f] - stock["y"][i_s])
    return {"n_fast": int(kf.size), "n_stock": int(ks.size), "events_same_count": float(np.mean(nf == ns)),
            "n_paired": int(common.size), "energy_exact_frac": float(np.mean(np.isclose(fast["energy"][i_f],
                                                                                        stock["energy"][i_s], rtol=1e-6))),
            "size_equal_frac": float(np.mean(fast["size"][i_f] == stock["size"][i_s])),
            "energy_rel": resid_stats(fast["energy"][i_f], stock["energy"][i_s], rel=True),
            "time_ns": resid_stats(fast["time"][i_f], stock["time"][i_s]),
            "position_mm": resid_stats(pos, np.zeros_like(pos)),
            "z_mm": resid_stats(fast["z"][i_f], stock["z"][i_s]),
            "disk_equal_frac": float(np.mean(fast["disk"][i_f] == stock["disk"][i_s])),
            "split_equal_frac": float(np.mean(fast["split"][i_f] == stock["split"][i_s]))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", required=True)
    p.add_argument("--ntuple", required=True)
    p.add_argument("--pulse", required=True)
    p.add_argument("--geometry", required=True, help="R6 export with dump/crystals and dump/disks")
    p.add_argument("--arms", default="noi,prd")
    p.add_argument("--iterations", type=int, default=FitConfig().iterations)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    events, _, _ = stockio.load_steps(args.steps)
    pulse = stockio.load_pulse(args.pulse)
    report = {"inputs": vars(args)}
    for arm in args.arms.split(","):
        pos, digis, reco, hits, pbt = stock_tables(args.ntuple, arm, events)
        geom = load_geometry(args.geometry)
        fast_cl = make_clusters({"event": hits["event"], "crystal": hits["crystal"], "time": hits["t"],
                                 "energy": hits["e"]}, events.size, geom)
        report[arm] = {"reco_digis": compare_reco_digis(digis, reco, pbt, pos, pulse, FitConfig(iterations=args.iterations))[0],
                       "hits_from_stock_recodigis": compare_hits(reco, hits, events.size),
                       "clusters_from_stock_hits": compare_clusters(fast_cl, hits["clusters"], events.size)}
        print(arm, json.dumps(report[arm], indent=1))
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
