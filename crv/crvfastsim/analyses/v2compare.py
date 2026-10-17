"""v2compare -- validation level V2 (docs/plan.md section 5): stock CRV response + reco on two CrvStep sources.

Inputs are CrvRecoTableDumper files (directory crvdump/) of two arms run through the same stock digitisation and CRV
reco on the same art events (scripts/run_v2_arms.sh): the candidate (fast sim, or any table via CrvStepsFromTable) and
--compare the reference. Per sector group (EX, T1, T2, M = taggers M1-M8; clusters by CrvSectorType 1-4):

  * PE per pulse (sequence index 0): median                     |rel| <= 2 %   (2 sigma of its statistical error allowed)
  * PE spectrum over 5-150 PE, 20 bins of equal reference content  binned ratio +-5 % after 2 sigma; chi2/ndf reported
  * events with a pulse of PE >= RATE_PE in the bar, all 576 bars  chi2/ndf <= 1.5 and no bar beyond 5 sigma
    (counted once per event and bar: counting each SiPM's pulse made ~4 correlated entries per crossing; user 2026-10-16)
  * coincidence clusters per event                               |rel| <= 1 %   (2 sigma allowed)
  * per-layer pulse efficiency of primary-muon bar crossings     |diff| <= 0.3 points (2 sigma allowed). DIAGNOSTIC:
    the denominator is each arm's own truth crossings (--steps / --compare-steps); the agreed observable uses
    extrapolated tracks and the module-gap width D, which this does not compute.

Rates are per event of each arm's art job. With --generated N (and --compare-generated M) they are per generated event
instead: needed when the arms ran on different event sets, e.g. the candidate on the gen stream (every generated event)
and the reference on the G4-accepted dts (G4-rejected events have no CRV step above 0.1 MeV, and the digitisation makes
no digis without charge, so they would add nothing).
Fixed-edge histograms make it batchable: --emit writes one JSON of counts; --merge sums them.

    v2compare FAST.root [...] --compare REF.root [...] [--steps FASTSTEPS.root ... --compare-steps REFSTEPS.root ...]
    v2compare ... --emit batch_NN.json ;  v2compare "batches/*.json" --merge --outdir DIR
"""
import json
import os

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from mu2efastsim import match, plots

from .. import geometry, tables
from ..reco import reftables

GROUPS = {"EX": [0], "T1": [1], "T2": [2], "M": list(range(3, 11))}
SECTOR_TYPE = {"EX": 1, "T1": 2, "T2": 3, "M": 4}
NBARS = 576
NLAYERS = 4
PE_EDGES = np.linspace(0.0, 300.0, 3001)
RATE_PE = 5.0
EFF_PE = 8.0  # PEthreshold of every sector at the data operating point (epilog_extracted_v04)
SPEC_RANGE = (5.0, 150.0)
THRESH = {"pe_median_rel": 0.02, "pe_ratio_band": 0.05, "bar_rate_chi2_ndf": 1.5, "bar_rate_max_pull": 5.0,
          "clusters_per_event_rel": 0.01, "layer_eff_diff": 0.003}
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def _new_arm():
    return {"events": 0, "files": 0, "bar_rate": [0] * NBARS,
            "groups": {g: {"npulse": 0, "h_pe": [0] * (len(PE_EDGES) - 1), "nclus": 0,
                           "eff_den": [0] * NLAYERS, "eff_num": [0] * NLAYERS} for g in GROUPS}}


def _new_state():
    return {"fast": _new_arm(), "ref": _new_arm()}


def _add(a, b):
    return (np.asarray(a) + np.asarray(b)).tolist()


def _fill_arm(A, reco_paths, step_paths, geom):
    ev = reftables.read(reco_paths, "events", ["run", "subrun", "event"])
    p = reftables.read(reco_paths, "pulses", ["run", "subrun", "event", "bar", "seq", "pe"])
    c = reftables.read(reco_paths, "clusters", ["sectortype"])
    A["events"] += len(ev)
    A["files"] += len(tables.expand(reco_paths))
    bar = ak.to_numpy(p.bar).astype(np.int64)
    pe = ak.to_numpy(p.pe).astype(float)
    s0 = ak.to_numpy(p.seq) == 0
    # bars hit per event (one count per event and bar, not per SiPM pulse), so the counts are Poisson
    evk = reftables.event_key(p)
    hb = s0 & (pe >= RATE_PE)
    eb = np.unique(np.stack([evk[hb], bar[hb]], 1), axis=0)
    A["bar_rate"] = _add(A["bar_rate"], np.bincount(eb[:, 1], minlength=NBARS))
    sec = geom.sector[bar]
    st = ak.to_numpy(c.sectortype) if c is not None else np.zeros(0, int)
    for g, secs in GROUPS.items():
        G = A["groups"][g]
        m = s0 & np.isin(sec, secs)
        G["npulse"] += int(m.sum())
        G["h_pe"] = _add(G["h_pe"], np.histogram(pe[m], bins=PE_EDGES)[0])
        G["nclus"] += int((st == SECTOR_TYPE[g]).sum())
    if step_paths:
        _fill_eff(A, p, reftables.event_key(ev), step_paths, geom)


def _fill_eff(A, p, ev_keys, step_paths, geom):
    """Per-layer efficiency of the arm's own primary-muon crossings: max seq-0 pulse PE over the bar's SiPMs >= EFF_PE."""
    steps = tables.read_many(step_paths, "steps", tables.KEY + ["bar", "sp", "edep", "len", "t0"])
    parts = tables.read_many(step_paths, "particles", tables.KEY + ["key", "pdg", "isprim"])
    x = tables.primary_muon_crossings(steps, parts, geom.sector)
    evk = reftables.event_key(p)
    s0 = ak.to_numpy(p.seq) == 0
    pk = np.stack([evk[s0], ak.to_numpy(p.bar)[s0].astype(np.int64)], 1)
    pe = ak.to_numpy(p.pe)[s0].astype(float)
    # max PE per (event, bar): sort by key then PE, keep the last of each key
    order = np.lexsort((pe, pk[:, 1], pk[:, 0]))
    pk, pe = pk[order], pe[order]
    last = np.r_[np.any(pk[1:] != pk[:-1], axis=1), True]
    xev = (x.run.to_numpy().astype(np.int64) << 40) | (x.subrun.to_numpy().astype(np.int64) << 20) | \
        x.event.to_numpy().astype(np.int64)
    xk = np.stack([xev, x.bar.to_numpy().astype(np.int64)], 1)
    # crossings only in events the reco saw (the art job's universe)
    seen = np.isin(xev, ev_keys)
    maxpe = match.lookup(xk[seen], pk[last], pe[last], missing=0.0)
    xb = x.bar.to_numpy()[seen]
    lay, sec = geom.layer[xb].astype(np.int64), geom.sector[xb]
    hit = maxpe >= EFF_PE
    for g, secs in GROUPS.items():
        G = A["groups"][g]
        m = np.isin(sec, secs)
        G["eff_den"] = _add(G["eff_den"], np.bincount(lay[m], minlength=NLAYERS)[:NLAYERS])
        G["eff_num"] = _add(G["eff_num"], np.bincount(lay[m & hit], minlength=NLAYERS)[:NLAYERS])


def _merge_into(tot, d):
    for arm in ("fast", "ref"):
        T, D = tot[arm], d[arm]
        T["events"] += D["events"]
        T["files"] += D["files"]
        T["bar_rate"] = _add(T["bar_rate"], D["bar_rate"])
        for g in GROUPS:
            for k in ("npulse", "nclus"):
                T["groups"][g][k] += D["groups"][g][k]
            for k in ("h_pe", "eff_den", "eff_num"):
                T["groups"][g][k] = _add(T["groups"][g][k], D["groups"][g][k])


def hist_median(h, edges):
    """Median by linear interpolation in the cumulative histogram, and its large-sample error 1/(2 f(m) sqrt(N))."""
    h = np.asarray(h, float)
    n = h.sum()
    if n < 50:
        return float("nan"), float("nan")
    cdf = np.r_[0.0, np.cumsum(h)] / n
    m = float(np.interp(0.5, cdf, edges))
    k = min(int(np.searchsorted(edges, m, "right")) - 1, len(h) - 1)
    lo, hi = max(k - 5, 0), min(k + 6, len(h))
    dens = h[lo:hi].sum() / n / (edges[hi] - edges[lo])
    return m, float(1.0 / (2.0 * dens * np.sqrt(n))) if dens > 0 else float("nan")


def ratio_band(hf, hr, edges, lo, hi, nbins=20, min_ref=100):
    """fast/ref of unit-normalised spectra in nbins of equal reference content over [lo, hi)."""
    hf, hr = np.asarray(hf, float), np.asarray(hr, float)
    i0, i1 = int(np.searchsorted(edges, lo)), int(np.searchsorted(edges, hi))
    hf, hr = hf[i0:i1], hr[i0:i1]
    if hr.sum() < nbins * min_ref:
        return {"max_abs_dev": None, "max_abs_dev_minus_2sigma": None, "chi2_ndf": None, "nbins_used": 0}
    cdf = np.cumsum(hr) / hr.sum()
    cut = np.searchsorted(cdf, np.arange(1, nbins) / nbins)
    rf = np.array([a.sum() for a in np.split(hf, cut)])
    rr = np.array([a.sum() for a in np.split(hr, cut)])
    ok = rr >= min_ref
    r = (rf[ok] / max(hf.sum(), 1)) / (rr[ok] / hr.sum())
    err = r * np.sqrt(1 / np.maximum(rf[ok], 1) + 1 / rr[ok])
    return {"max_abs_dev": float(np.abs(r - 1).max()),
            "max_abs_dev_minus_2sigma": float(np.maximum(np.abs(r - 1) - 2 * err, 0).max()),
            "chi2_ndf": float(np.sum(((r - 1) / err) ** 2) / ok.sum()), "nbins_used": int(ok.sum())}


def bar_rates(nf, nr, ef, er):
    """chi2/ndf and max |pull| of per-bar counts, the fast counts scaled to the reference's event count."""
    nf, nr = np.asarray(nf, float), np.asarray(nr, float)
    s = er / ef
    ok = (nf + nr) > 0
    pull = (s * nf[ok] - nr[ok]) / np.sqrt(s * s * nf[ok] + nr[ok])
    worst = int(np.flatnonzero(ok)[np.argmax(np.abs(pull))]) if ok.any() else -1
    return {"chi2_ndf": float(np.sum(pull ** 2) / max(ok.sum(), 1)), "max_abs_pull": float(np.abs(pull).max()),
            "worst_bar": worst, "nbars_used": int(ok.sum())}


def summarise(S):
    F, R = S["fast"], S["ref"]
    out = {"events_fast": F["events"], "events_ref": R["events"], "thresholds": THRESH, "groups": {}}
    br = bar_rates(F["bar_rate"], R["bar_rate"], F["events"], R["events"])
    out["bar_rate"] = br
    ok_all = br["chi2_ndf"] <= THRESH["bar_rate_chi2_ndf"] and br["max_abs_pull"] <= THRESH["bar_rate_max_pull"]
    out["bar_rate"]["pass"] = bool(ok_all)
    for g in GROUPS:
        GF, GR = F["groups"][g], R["groups"][g]
        if GR["npulse"] == 0:
            continue
        (mf, emf), (mr, emr) = hist_median(GF["h_pe"], PE_EDGES), hist_median(GR["h_pe"], PE_EDGES)
        res = {"npulse_ref": GR["npulse"], "npulse_fast": GF["npulse"],
               "pulses_per_event_rel": (GF["npulse"] / F["events"]) / (GR["npulse"] / R["events"]) - 1,
               "pe_median_ref": mr, "pe_median_fast": mf, "pe_median_rel": mf / mr - 1,
               "pe_median_rel_err": abs(mf / mr) * np.hypot(emf / mf, emr / mr),
               "pe_ratio": ratio_band(GF["h_pe"], GR["h_pe"], PE_EDGES, *SPEC_RANGE)}
        cf, cr = GF["nclus"] / F["events"], GR["nclus"] / R["events"]
        res.update({"nclus_ref": GR["nclus"], "nclus_fast": GF["nclus"], "clusters_per_event_rel": cf / cr - 1,
                    "clusters_per_event_rel_err": (cf / cr) * np.sqrt(1 / max(GF["nclus"], 1) + 1 / GR["nclus"])})
        checks = {
            "pe_median": max(abs(res["pe_median_rel"]) - 2 * res["pe_median_rel_err"], 0) <= THRESH["pe_median_rel"],
            "pe_ratio": (res["pe_ratio"]["max_abs_dev_minus_2sigma"] or 0) <= THRESH["pe_ratio_band"],
            "clusters": max(abs(res["clusters_per_event_rel"]) - 2 * res["clusters_per_event_rel_err"], 0)
            <= THRESH["clusters_per_event_rel"],
        }
        if sum(GR["eff_den"]):
            ef = np.asarray(GF["eff_num"], float) / np.maximum(GF["eff_den"], 1)
            er = np.asarray(GR["eff_num"], float) / np.maximum(GR["eff_den"], 1)
            err = np.sqrt(ef * (1 - ef) / np.maximum(GF["eff_den"], 1) + er * (1 - er) / np.maximum(GR["eff_den"], 1))
            res.update({"layer_eff_fast": ef, "layer_eff_ref": er, "layer_eff_diff": ef - er,
                        "layer_eff_diff_err": err, "layer_eff_den_ref": GR["eff_den"]})
            res["diagnostic_layer_eff_ok"] = bool(np.all(np.maximum(np.abs(ef - er) - 2 * err, 0)
                                                         <= THRESH["layer_eff_diff"]))
        res["checks"] = checks
        res["pass"] = all(checks.values())
        ok_all &= res["pass"]
        out["groups"][g] = res
    out["pass"] = bool(ok_all)
    return out


def report(S, outdir, labels=("reference", "fast sim")):
    summ = summarise(S)
    plots.write_json(outdir, "v2compare_stats.json", summ)
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(4 * len(GROUPS), 5), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    edges = PE_EDGES[::10]  # 1 PE bins for display
    for j, g in enumerate(GROUPS):
        hf = np.add.reduceat(np.asarray(S["fast"]["groups"][g]["h_pe"], float), np.arange(0, len(PE_EDGES) - 1, 10))
        hr = np.add.reduceat(np.asarray(S["ref"]["groups"][g]["h_pe"], float), np.arange(0, len(PE_EDGES) - 1, 10))
        plots.ratio_panel(axes[0, j], axes[1, j], edges, hr, hf, "PE per pulse (seq 0)", labels=labels)
        axes[0, j].set_title(f"{g}  (N_ref={int(hr.sum()):,})")
        axes[0, j].set_xlim(0, 150)
    plots.save(fig, outdir, "v2compare_pe.png")
    br = summ["bar_rate"]
    print(f"bars: chi2/ndf {br['chi2_ndf']:.2f}  max|pull| {br['max_abs_pull']:.1f} (bar {br['worst_bar']})  "
          f"| {'ok' if br['pass'] else 'FAIL'}")
    for g, r in summ["groups"].items():
        flags = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items())
        eff = ""
        if "layer_eff_diff" in r:
            eff = "  dEff[pts] " + " ".join(f"{100 * d:+.2f}" for d in r["layer_eff_diff"])
        print(f"{g:3s} pulses/ev {r['pulses_per_event_rel']:+.4f}  PEmed {r['pe_median_ref']:.2f}->"
              f"{r['pe_median_fast']:.2f} ({r['pe_median_rel']:+.4f}+-{r['pe_median_rel_err']:.4f})  "
              f"band {r['pe_ratio']['max_abs_dev_minus_2sigma']}  chi2 {r['pe_ratio']['chi2_ndf']}  "
              f"clus/ev {r['clusters_per_event_rel']:+.4f}+-{r['clusters_per_event_rel_err']:.4f}{eff} | {flags}")
    print(f"V2 {'PASS' if summ['pass'] else 'FAIL'}  over {summ['events_ref']:,} events")
    return summ


def run(inputs, outdir, compare=None, steps=None, compare_steps=None, emit=None, merge=False, counters=None,
        generated=None, compare_generated=None, **kw):
    S = _new_state()
    if merge:
        for f in tables.expand(inputs):
            _merge_into(S, json.load(open(f)))
        return report(S, outdir)
    if not compare:
        raise SystemExit("v2compare needs --compare <reference CrvRecoTableDumper files>")
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    _fill_arm(S["fast"], inputs, steps, geom)
    _fill_arm(S["ref"], compare, compare_steps, geom)
    if generated:
        S["fast"]["events"] = int(generated)
        S["ref"]["events"] = int(compare_generated or generated)
    if emit:
        os.makedirs(os.path.dirname(emit) or ".", exist_ok=True)
        json.dump(S, open(emit, "w"))
        print(f"wrote {emit}")
        return S
    return report(S, outdir)
