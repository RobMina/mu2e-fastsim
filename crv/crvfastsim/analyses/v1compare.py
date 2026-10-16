"""v1compare -- validation level V1 (docs/plan.md section 5): primary-muon CRV bar crossings, fast sim vs G4.

Inputs are CrvStep tables (crvsteps.*.root from CrvStepTableDumper, or the fast sim's equivalent); --compare names the
G4 reference tables. A crossing is the sum of the primary muon's CrvSteps in one bar (tables.primary_muon_crossings).
Per sector group (EX, T1, T2, M = taggers M1-M8):

  * crossings per accepted event                         threshold |rel| <= 1 %
  * path length: mean, and binned ratio over the ref 5-95 % range    0.5 %, +-2 % (2 sigma allowed)
  * visible energy: mean, MPV, and binned ratio                      1 %, 1 %, +-3 % (2 sigma allowed)
  * start-time bias of the crossings present in both arms (same event and bar)    <= 0.1 ns (reported)

Fixed-edge histograms make it batchable: --emit writes one JSON of counts per arm and file group; --merge sums them.
Per-crossing statistics come from the histograms (mean from sums, MPV from the edep histogram).

    v1compare FAST.root [...] --compare G4.root [...] --outdir DIR
    v1compare FAST.root [...] --compare G4.root [...] --emit batch_NN.json
    v1compare "batches/*.json" --merge --outdir DIR
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from .. import geometry, plots, tables

GROUPS = {"EX": [0], "T1": [1], "T2": [2], "M": list(range(3, 11))}
EDGES = {"len": np.linspace(0.0, 120.0, 601), "edep": np.linspace(0.0, 30.0, 601), "dt": np.linspace(-5.0125, 5.0125, 402)}  # 0.025 ns bins centred on 0
THRESH = {"crossings_per_event_rel": 0.01, "len_mean_rel": 0.005, "len_ratio_band": 0.02,
          "edep_mean_rel": 0.01, "edep_mpv_rel": 0.01, "edep_ratio_band": 0.03, "t0_bias_ns": 0.1}
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def _new_arm():
    return {"events": 0, "files": 0,
            "groups": {g: {"n": 0, "len_sum": 0.0, "edep_sum": 0.0,
                           "h_len": [0] * (len(EDGES["len"]) - 1),
                           "h_edep": [0] * (len(EDGES["edep"]) - 1)} for g in GROUPS}}


def _new_state():
    return {"fast": _new_arm(), "ref": _new_arm(),
            "dt": {g: [0] * (len(EDGES["dt"]) - 1) for g in GROUPS}, "matched": {g: 0 for g in GROUPS}}


def _fill_arm(A, xing, n_events):
    A["events"] += int(n_events)
    A["files"] += 1
    for g, secs in GROUPS.items():
        x = xing[xing.sector.isin(secs)]
        G = A["groups"][g]
        G["n"] += int(len(x))
        G["len_sum"] += float(x.len.sum())
        G["edep_sum"] += float(x.edep.sum())
        for col in ("len", "edep"):
            h, _ = np.histogram(x[col].to_numpy(), bins=EDGES[col])
            G[f"h_{col}"] = (np.asarray(G[f"h_{col}"]) + h).tolist()


def _arm_crossings(paths, sector_of_bar):
    steps = tables.read_many(paths, "steps")
    parts = tables.read_many(paths, "particles", ["run", "subrun", "event", "key", "pdg", "isprim"])
    events = parts[tables.KEY].drop_duplicates()
    return tables.primary_muon_crossings(steps, parts, sector_of_bar), events


def _accumulate(S, fast_paths, ref_paths, sector_of_bar):
    xf, ef = _arm_crossings(fast_paths, sector_of_bar)
    xr, er = _arm_crossings(ref_paths, sector_of_bar)
    # event universe = union of both arms (README section 5: why not the common events)
    universe = ef.merge(er, on=tables.KEY, how="outer")
    _fill_arm(S["fast"], xf, len(universe))
    _fill_arm(S["ref"], xr, len(universe))
    m = xf.merge(xr, on=tables.KEY + ["bar"], suffixes=("_f", "_r"))
    for g, secs in GROUPS.items():
        mg = m[m.sector_f.isin(secs)]
        S["matched"][g] += int(len(mg))
        h, _ = np.histogram((mg.t0_f - mg.t0_r).to_numpy(), bins=EDGES["dt"])
        S["dt"][g] = (np.asarray(S["dt"][g]) + h).tolist()


def _merge_into(tot, d):
    for arm in ("fast", "ref"):
        tot[arm]["events"] += d[arm]["events"]
        tot[arm]["files"] += d[arm]["files"]
        for g in GROUPS:
            T, D = tot[arm]["groups"][g], d[arm]["groups"][g]
            for k in ("n", "len_sum", "edep_sum"):
                T[k] += D[k]
            for k in ("h_len", "h_edep"):
                T[k] = (np.asarray(T[k]) + np.asarray(D[k])).tolist()
    for g in GROUPS:
        tot["matched"][g] += d["matched"][g]
        tot["dt"][g] = (np.asarray(tot["dt"][g]) + np.asarray(d["dt"][g])).tolist()


def _hist_mpv(h, edges, lo_q=0.15, hi_q=0.70):
    """MPV and its statistical error from a Moyal (Landau approximation) fit to the histogram core.

    Fit range: from the lo_q to the hi_q quantile of the histogram (the peak and its rising and falling edges,
    excluding the long high tail). Weighted least squares on counts, Poisson errors. Returns (mpv, err).
    A peak-finder (argmax, or argmax after smoothing) moved by +-4 % on the 16 k tagger crossings with the smoothing
    window (2026-10-16), so it cannot test a 1 % threshold there; the fit uses the whole core.
    """
    from scipy.optimize import curve_fit
    h = np.asarray(h, float)
    c = 0.5 * (edges[:-1] + edges[1:])
    if h.sum() < 100:
        return float("nan"), float("nan")
    cdf = np.cumsum(h) / h.sum()
    sel = (cdf >= lo_q) & (cdf <= hi_q) & (h > 0)

    def moyal(x, a, mu, sig):
        z = (x - mu) / sig
        return a * np.exp(-0.5 * (z + np.exp(-z)))

    k = int(np.argmax(np.where(sel, h, 0)))
    p0 = (h[k] / np.exp(-0.5), c[k], 0.15 * c[k])
    try:
        par, cov = curve_fit(moyal, c[sel], h[sel], p0=p0, sigma=np.sqrt(h[sel]), absolute_sigma=True, maxfev=5000)
        return float(par[1]), float(np.sqrt(cov[1, 1]))
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")


def _hist_ratio_band(hf, hr, edges, q=(0.05, 0.95), min_ref=400):
    """Binned-ratio check on fixed-edge histograms, rebinned to 20 bins over the reference's q-range."""
    hf, hr = np.asarray(hf, float), np.asarray(hr, float)
    cdf = np.cumsum(hr) / max(hr.sum(), 1)
    lo_i, hi_i = int(np.searchsorted(cdf, q[0])), int(np.searchsorted(cdf, q[1])) + 1
    groups = np.array_split(np.arange(lo_i, min(hi_i, len(hr))), 20)
    rf = np.array([hf[g].sum() for g in groups if len(g)])
    rr = np.array([hr[g].sum() for g in groups if len(g)])
    ok = rr >= min_ref
    if not ok.any():
        return {"max_abs_dev": None, "max_abs_dev_minus_2sigma": None, "nbins_used": 0}
    r = (rf[ok] / hf.sum()) / (rr[ok] / hr.sum())
    err = r * np.sqrt(1 / np.maximum(rf[ok], 1) + 1 / rr[ok])
    return {"max_abs_dev": float(np.abs(r - 1).max()),
            "max_abs_dev_minus_2sigma": float(np.maximum(np.abs(r - 1) - 2 * err, 0).max()),
            "nbins_used": int(ok.sum()), "range": [float(edges[lo_i]), float(edges[min(hi_i, len(hr))])]}


def summarise(S):
    out = {"events_union": S["ref"]["events"], "thresholds": THRESH, "groups": {}}
    ok_all = True
    for g in GROUPS:
        F, R = S["fast"]["groups"][g], S["ref"]["groups"][g]
        if R["n"] == 0:
            continue
        res = {"n_ref": R["n"], "n_fast": F["n"], "crossings_per_event_rel": F["n"] / R["n"] - 1}
        for col in ("len", "edep"):
            res[f"{col}_mean_rel"] = (F[f"{col}_sum"] / max(F["n"], 1)) / (R[f"{col}_sum"] / R["n"]) - 1
            res[f"{col}_ratio"] = _hist_ratio_band(F[f"h_{col}"], R[f"h_{col}"], EDGES[col])
        (mr, er), (mf, ef) = _hist_mpv(R["h_edep"], EDGES["edep"]), _hist_mpv(F["h_edep"], EDGES["edep"])
        res["edep_mpv_ref"], res["edep_mpv_ref_err"] = mr, er
        res["edep_mpv_fast"], res["edep_mpv_fast_err"] = mf, ef
        res["edep_mpv_rel"] = mf / mr - 1
        res["edep_mpv_rel_err"] = abs(mf / mr) * np.hypot(er / mr, ef / mf)
        hdt = np.asarray(S["dt"][g], float)
        c = 0.5 * (EDGES["dt"][:-1] + EDGES["dt"][1:])
        res["matched_frac_of_ref"] = S["matched"][g] / R["n"]
        res["t0_bias_ns"] = float(c[np.searchsorted(np.cumsum(hdt), hdt.sum() / 2)]) if hdt.sum() else None
        checks = {
            "crossings": abs(res["crossings_per_event_rel"]) <= THRESH["crossings_per_event_rel"],
            "len_mean": abs(res["len_mean_rel"]) <= THRESH["len_mean_rel"],
            "len_ratio": (res["len_ratio"]["max_abs_dev_minus_2sigma"] or 0) <= THRESH["len_ratio_band"],
            "edep_mean": abs(res["edep_mean_rel"]) <= THRESH["edep_mean_rel"],
            "edep_mpv": max(abs(res["edep_mpv_rel"]) - 2 * res["edep_mpv_rel_err"], 0) <= THRESH["edep_mpv_rel"],
            "edep_ratio": (res["edep_ratio"]["max_abs_dev_minus_2sigma"] or 0) <= THRESH["edep_ratio_band"],
            "t0_bias": res["t0_bias_ns"] is None or abs(res["t0_bias_ns"]) <= THRESH["t0_bias_ns"],
        }
        res["checks"] = checks
        res["pass"] = all(checks.values())
        ok_all &= res["pass"]
        out["groups"][g] = res
    out["pass"] = bool(ok_all)
    return out


def report(S, outdir):
    summ = summarise(S)
    plots.write_json(outdir, "v1compare_stats.json", summ)
    for col, xl in (("len", "path length per crossing [mm]"), ("edep", "visible energy per crossing [MeV]")):
        fig, axes = plt.subplots(2, len(GROUPS), figsize=(4 * len(GROUPS), 5), sharex="col",
                                 gridspec_kw={"height_ratios": [3, 1]})
        for j, g in enumerate(GROUPS):
            F, R = S["fast"]["groups"][g], S["ref"]["groups"][g]
            plots.ratio_panel(axes[0, j], axes[1, j], EDGES[col], np.asarray(R[f"h_{col}"], float),
                              np.asarray(F[f"h_{col}"], float), xl)
            axes[0, j].set_title(f"{g}  (N_ref={R['n']:,})")
            if col == "len":
                axes[0, j].set_xlim(15, 60)
            else:
                axes[0, j].set_xlim(0, 12)
        plots.save(fig, outdir, f"v1compare_{col}.png")
    for g, r in summ["groups"].items():
        flags = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["checks"].items())
        print(f"{g:3s} n_ref={r['n_ref']:>8,} xing {r['crossings_per_event_rel']:+.4f}  len {r['len_mean_rel']:+.4f}  "
              f"edep {r['edep_mean_rel']:+.4f} mpv {r['edep_mpv_rel']:+.4f}  t0 {r['t0_bias_ns']}  | {flags}")
    print(f"V1 {'PASS' if summ['pass'] else 'FAIL'}  over {summ['events_union']:,} events (union of both arms)")
    return summ


def run(inputs, outdir, compare=None, emit=None, merge=False, counters=None, **kw):
    S = _new_state()
    if merge:
        for f in tables.expand(inputs):
            _merge_into(S, json.load(open(f)))
        return report(S, outdir)
    if not compare:
        raise SystemExit("v1compare needs --compare <G4 reference tables>")
    sector_of_bar = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS)).sector
    _accumulate(S, inputs, compare, sector_of_bar)
    if emit:
        os.makedirs(os.path.dirname(emit) or ".", exist_ok=True)
        json.dump(S, open(emit, "w"))
        print(f"wrote {emit}")
        return S
    return report(S, outdir)
