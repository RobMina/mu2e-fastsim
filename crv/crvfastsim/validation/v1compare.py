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
import warnings

import matplotlib.pyplot as plt
import numpy as np

from mu2efastsim import plots

from .. import geometry, runconfig, tables

EXTRACTED_GROUPS = {"EX": [0], "T1": [1], "T2": [2], "M": list(range(3, 11))}
GROUPS = dict(EXTRACTED_GROUPS)
EDGES = {"len": np.linspace(0.0, 120.0, 601), "edep": np.linspace(0.0, 30.0, 601), "dt": np.linspace(-5.0125, 5.0125, 402)}  # 0.025 ns bins centred on 0
THRESH = {"crossings_per_event_rel": 0.01, "len_mean_rel": 0.005, "len_ratio_band": 0.02,
          "edep_mean_rel": 0.01, "edep_mpv_rel": 0.01, "edep_ratio_band": 0.03, "t0_bias_ns": 0.1,
          "ratio_chi2_ndf": 1.5}  # binned-ratio pass rule (user, 2026-10-16); the bands are reported only
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


def _valid_hist(h):
    return bool(np.all(np.isfinite(h)) and np.all(h >= 0) and np.isfinite(h.sum()) and h.sum() > 0)


def _hist_mpv(h, edges, lo_q=0.15, hi_q=0.70):
    """Fit the unchanged Moyal core; reject unsupported fits before applying the 2-sigma allowance."""
    from scipy.optimize import OptimizeWarning, curve_fit
    h = np.asarray(h, float)
    c = 0.5 * (edges[:-1] + edges[1:])
    result = {"valid": False, "reason": None, "mpv": None, "error": None, "nbins_fit": 0}
    if not _valid_hist(h) or h.sum() < 100:
        return dict(result, reason="invalid_or_insufficient_histogram")
    cdf = np.cumsum(h) / h.sum()
    sel = (cdf >= lo_q) & (cdf <= hi_q) & (h > 0)
    result["nbins_fit"] = int(sel.sum())
    if sel.sum() < 4:
        return dict(result, reason="insufficient_fit_bins")
    lo, hi = float(c[sel][0]), float(c[sel][-1])
    width = hi - lo
    result["range"] = [lo, hi]

    def moyal(x, a, mu, sig):
        z = (x - mu) / sig
        return a * np.exp(-0.5 * (z + np.exp(-z)))

    k = int(np.argmax(np.where(sel, h, 0)))
    p0 = (h[k] / np.exp(-0.5), c[k], 0.15 * c[k])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                par, cov = curve_fit(moyal, c[sel], h[sel], p0=p0, sigma=np.sqrt(h[sel]),
                                     absolute_sigma=True, maxfev=5000)
    except (RuntimeError, ValueError, TypeError, OptimizeWarning):
        return dict(result, reason="fit_failed")
    result["parameters"] = np.asarray(par).tolist()
    if not np.all(np.isfinite(par)) or par[0] <= 0 or par[2] <= 0 or par[1] <= 0 or not lo <= par[1] <= hi:
        return dict(result, reason="unphysical_or_out_of_core_fit")
    if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T) or np.any(np.linalg.eigvalsh(cov) <= 0):
        return dict(result, reason="invalid_covariance")
    error = float(np.sqrt(cov[1, 1]))
    scale_error = float(np.sqrt(cov[2, 2]))
    result.update(mpv=float(par[1]), error=error, scale_error=scale_error)
    if par[2] > width or error > width or scale_error > width:
        return dict(result, reason="unidentified_core_fit")
    return dict(result, valid=True)


def _hist_ratio_band(hf, hr, edges, q=(0.05, 0.95), min_ref=400):
    """Binned-ratio check on fixed-edge histograms, rebinned to 20 bins over the reference's q-range."""
    hf, hr = np.asarray(hf, float), np.asarray(hr, float)
    result = {"max_abs_dev": None, "max_abs_dev_minus_2sigma": None, "chi2_ndf": None,
              "nbins_used": 0, "valid": False, "reason": None}
    if not _valid_hist(hf) or not _valid_hist(hr):
        return dict(result, reason="invalid_or_empty_histogram")
    cdf = np.cumsum(hr) / hr.sum()
    lo_i, hi_i = int(np.searchsorted(cdf, q[0])), int(np.searchsorted(cdf, q[1])) + 1
    groups = np.array_split(np.arange(lo_i, min(hi_i, len(hr))), 20)
    rf = np.array([hf[g].sum() for g in groups if len(g)])
    rr = np.array([hr[g].sum() for g in groups if len(g)])
    ok = rr >= min_ref
    result.update(nbins_used=int(ok.sum()), range=[float(edges[lo_i]), float(edges[min(hi_i, len(hr))])])
    if not ok.any():
        return dict(result, reason="no_populated_comparison_bins")
    if np.any(rf[ok] == 0):
        return dict(result, reason="empty_candidate_comparison_bin")
    r = (rf[ok] / hf.sum()) / (rr[ok] / hr.sum())
    err = r * np.sqrt(1 / rf[ok] + 1 / rr[ok])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        chi2 = float(np.sum(((r - 1) / err) ** 2) / ok.sum())
    if not np.isfinite(chi2):
        return dict(result, reason="nonfinite_ratio_statistic")
    return dict(result, valid=True, chi2_ndf=chi2, max_abs_dev=float(np.abs(r - 1).max()),
                max_abs_dev_minus_2sigma=float(np.maximum(np.abs(r - 1) - 2 * err, 0).max()))


def _within(value, limit):
    return bool(value is not None and np.isfinite(value) and abs(value) <= limit)


def summarise(S):
    events = S["ref"]["events"]
    events_valid = bool(np.isfinite(events) and events > 0 and S["fast"]["events"] == events)
    out = {"events_union": events, "events_valid": events_valid, "thresholds": THRESH, "groups": {}}
    for g in GROUPS:
        F, R = S["fast"]["groups"][g], S["ref"]["groups"][g]
        populated = F["n"] > 0 and R["n"] > 0
        res = {"n_ref": R["n"], "n_fast": F["n"],
               "crossings_per_event_rel": F["n"] / R["n"] - 1 if R["n"] > 0 else float("nan")}
        reasons = {}
        if not events_valid:
            reasons["events"] = "invalid_event_universe"
        if not populated:
            reasons["population"] = "empty_required_group_arm"
        for col in ("len", "edep"):
            res[f"{col}_mean_rel"] = ((F[f"{col}_sum"] / F["n"]) / (R[f"{col}_sum"] / R["n"]) - 1
                                       if populated and R[f"{col}_sum"] > 0 else float("nan"))
            shape = _hist_ratio_band(F[f"h_{col}"], R[f"h_{col}"], EDGES[col])
            res[f"{col}_ratio"] = shape
            if not shape["valid"]:
                reasons[f"{col}_ratio"] = shape["reason"]
        fits = {arm: _hist_mpv(A["h_edep"], EDGES["edep"]) for arm, A in (("ref", R), ("fast", F))}
        res["edep_mpv_fits"] = fits
        for arm, fit in fits.items():
            res[f"edep_mpv_{arm}"] = fit["mpv"]
            res[f"edep_mpv_{arm}_err"] = fit["error"]
            if not fit["valid"]:
                reasons[f"edep_mpv_{arm}"] = fit["reason"]
        mpv_valid = all(fit["valid"] for fit in fits.values())
        res["edep_mpv_rel"], res["edep_mpv_rel_err"] = float("nan"), float("nan")
        if mpv_valid:
            mr, er = fits["ref"]["mpv"], fits["ref"]["error"]
            mf, ef = fits["fast"]["mpv"], fits["fast"]["error"]
            res["edep_mpv_rel"] = mf / mr - 1
            res["edep_mpv_rel_err"] = abs(mf / mr) * np.hypot(er / mr, ef / mf)
        hdt = np.asarray(S["dt"][g], float)
        c = 0.5 * (EDGES["dt"][:-1] + EDGES["dt"][1:])
        matched = S["matched"][g]
        time_valid = bool(_valid_hist(hdt) and np.isfinite(matched) and matched == int(matched)
                          and 0 < hdt.sum() <= matched <= min(F["n"], R["n"]))
        res["matched_frac_of_ref"] = matched / R["n"] if R["n"] > 0 else None
        res["matched"], res["matched_in_time_range"] = matched, float(hdt.sum())
        res["t0_bias_ns"] = float(c[np.searchsorted(np.cumsum(hdt), hdt.sum() / 2)]) if time_valid else None
        if not time_valid:
            reasons["t0_bias"] = "missing_or_invalid_matched_time_histogram"
        checks = {
            "events": events_valid,
            "population": bool(populated),
            "crossings": _within(res["crossings_per_event_rel"], THRESH["crossings_per_event_rel"]),
            "len_mean": _within(res["len_mean_rel"], THRESH["len_mean_rel"]),
            "len_ratio": res["len_ratio"]["valid"] and _within(res["len_ratio"]["chi2_ndf"], THRESH["ratio_chi2_ndf"]),
            "edep_mean": _within(res["edep_mean_rel"], THRESH["edep_mean_rel"]),
            "edep_mpv": mpv_valid and _within(max(abs(res["edep_mpv_rel"]) - 2 * res["edep_mpv_rel_err"], 0), THRESH["edep_mpv_rel"]),
            "edep_ratio": res["edep_ratio"]["valid"] and _within(res["edep_ratio"]["chi2_ndf"], THRESH["ratio_chi2_ndf"]),
            "t0_bias": time_valid and _within(res["t0_bias_ns"], THRESH["t0_bias_ns"]),
        }
        res["checks"], res["invalid_reasons"] = checks, reasons
        res["pass"] = all(checks.values())
        out["groups"][g] = res
    out["pass"] = bool(events_valid and out["groups"] and all(r["pass"] for r in out["groups"].values()))
    return out


def report(S, outdir):
    summ = summarise(S)
    plots.write_json(outdir, "v1compare_stats.json", summ)
    for col, xl in (("len", "path length per crossing [mm]"), ("edep", "visible energy per crossing [MeV]")):
        fig, axes = plt.subplots(2, len(GROUPS), figsize=(4 * len(GROUPS), 5), sharex="col",
                                 gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
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


def set_groups(run_config=None):
    """Sector groups of the comparison: the extracted groups (EX, T1, T2, M) or, for another run configuration, one
    group per sector type of its counters file (Run 2: R1-R6, L1-L3, T1-T5, E1-E2, U, D1-D4, C1-C2; every sector)."""
    global GROUPS
    if not run_config or run_config == runconfig.DEFAULT:
        GROUPS = dict(EXTRACTED_GROUPS)
        return
    rc = runconfig.get(run_config)
    g, c = rc.geometry(), rc.counters_cfg()
    names = runconfig.sector_type_names(g, c)
    st = [int(c[f"crs.sectorType{n}"]) for n in g.sector_names]
    GROUPS = {names[t]: [i for i, x in enumerate(st) if x == t] for t in sorted(names)}


def run(inputs, outdir, compare=None, emit=None, merge=False, counters=None, run_config=None, **kw):
    set_groups(run_config)
    if run_config and not counters:
        counters = runconfig.get(run_config).counters
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
