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
With run_config (name or RunConfig), geometry and all sector-type groups come from that configuration. Named states
carry an immutable configuration snapshot and fail closed on unsupported metrics; see docs/named-response.md.
Without run_config, the historical extracted state schema and supported-sample decisions are retained.

    v2compare FAST.root [...] --compare REF.root [...] [--steps FASTSTEPS.root ... --compare-steps REFSTEPS.root ...]
    v2compare ... --emit batch_NN.json ;  v2compare "batches/*.json" --merge --outdir DIR
"""
import dataclasses
import hashlib
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


@dataclasses.dataclass(frozen=True)
class ResponseLayout:
    """Immutable snapshot: never retain the registry's mutable sector/cut dictionaries."""
    identity: str
    nbars: int
    nlayers: int
    groups: tuple  # (display name, sector type, tuple of geometry sector indices)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def response_layout(run_config, geom=None):
    """Resolve a name or RunConfig into a geometry/config-bound V2 layout."""
    from .. import runconfig
    rc = runconfig.get(run_config) if isinstance(run_config, str) else run_config
    geom = rc.geometry() if geom is None else geom
    ccfg = rc.counters_cfg()
    names = runconfig.sector_type_names(geom, ccfg)
    groups = tuple((label, int(t), tuple(i for i, n in enumerate(geom.sector_names)
                                        if int(ccfg[f"crs.sectorType{n}"]) == t))
                   for t, label in names.items())
    if not groups or len({g[0] for g in groups}) != len(groups) or geom.nbars <= 0:
        raise ValueError("named V2 requires nonempty, uniquely named sector-type groups")
    nlayers = int(ccfg["crs.nLayers"])
    if nlayers <= 0 or np.any((geom.sector < 0) | (geom.sector >= len(geom.sector_names))):
        raise ValueError("invalid named geometry")
    arrays = {f.name: (getattr(geom, f.name).tolist() if isinstance(getattr(geom, f.name), np.ndarray)
                       else getattr(geom, f.name)) for f in dataclasses.fields(geom)}
    payload = {"schema": "crv-v2-named-v1", "run_config": dataclasses.asdict(rc),
               "geometry_sha256": hashlib.sha256(_canonical(arrays).encode()).hexdigest(),
               "counters_sha256": hashlib.sha256(_canonical(ccfg).encode()).hexdigest(),
               "sector_names": list(geom.sector_names), "nbars": geom.nbars, "nlayers": nlayers,
               "groups": groups}
    return ResponseLayout(_canonical(payload), geom.nbars, nlayers, groups)


def _state_layout(S):
    if "configuration" not in S:
        return None
    identity = S["configuration"]
    data = json.loads(identity)
    if data["schema"] != "crv-v2-named-v1" or not data["groups"]:
        raise ValueError("unsupported named V2 configuration")
    return ResponseLayout(identity, data["nbars"], data["nlayers"],
                          tuple((g, t, tuple(secs)) for g, t, secs in data["groups"]))


def _new_arm(layout=None):
    groups = [g for g, _, _ in layout.groups] if layout else GROUPS
    A = {"events": 0, "files": 0, "bar_rate": [0] * (layout.nbars if layout else NBARS),
         "groups": {g: {"npulse": 0, "h_pe": [0] * (len(PE_EDGES) - 1), "nclus": 0,
                         "eff_den": [0] * (layout.nlayers if layout else NLAYERS),
                         "eff_num": [0] * (layout.nlayers if layout else NLAYERS)} for g in groups}}
    if layout:
        A["missing_trees"] = []
    return A


def _new_state(run_config=None, geom=None, *, layout=None):
    if run_config is not None:
        layout = response_layout(run_config, geom)
    S = {"fast": _new_arm(layout), "ref": _new_arm(layout)}
    if layout:
        S["configuration"] = layout.identity
    return S


def _add(a, b):
    return (np.asarray(a) + np.asarray(b)).tolist()


def _fill_arm(A, reco_paths, step_paths, geom, layout=None):
    groups = {g: secs for g, _, secs in layout.groups} if layout else GROUPS
    types = {g: t for g, t, _ in layout.groups} if layout else SECTOR_TYPE
    nbars = layout.nbars if layout else NBARS
    if layout:
        import uproot
        missing = set(A["missing_trees"])
        for path in tables.expand(reco_paths):
            with uproot.open(path) as f:
                missing.update(k for k in ("events", "pulses", "clusters") if f"crvdump/{k}" not in f)
        A["missing_trees"] = sorted(missing)
    ev = reftables.read(reco_paths, "events", ["run", "subrun", "event"])
    p = reftables.read(reco_paths, "pulses", ["run", "subrun", "event", "bar", "seq", "pe"])
    c = reftables.read(reco_paths, "clusters", ["sectortype"])
    if layout:
        if ev is None:
            ev = ak.Array({k: np.zeros(0, np.int64) for k in ("run", "subrun", "event")})
        if p is None:
            p = ak.Array({k: np.zeros(0, np.int64) for k in ("run", "subrun", "event", "bar", "seq", "pe")})
    A["events"] += len(ev)
    A["files"] += len(tables.expand(reco_paths))
    bar = ak.to_numpy(p.bar).astype(np.int64)
    if layout and np.any((bar < 0) | (bar >= nbars) | (ak.to_numpy(p.bar) != bar)):
        raise ValueError("pulse bar outside named geometry")
    pe = ak.to_numpy(p.pe).astype(float)
    if layout and not np.isfinite(pe).all():
        raise ValueError("nonfinite pulse PE")
    s0 = ak.to_numpy(p.seq) == 0
    # bars hit per event (one count per event and bar, not per SiPM pulse), so the counts are Poisson
    evk = reftables.event_key(p)
    hb = s0 & (pe >= RATE_PE)
    eb = np.unique(np.stack([evk[hb], bar[hb]], 1), axis=0)
    A["bar_rate"] = _add(A["bar_rate"], np.bincount(eb[:, 1], minlength=nbars))
    sec = geom.sector[bar]
    st = ak.to_numpy(c.sectortype) if c is not None else np.zeros(0, int)
    if layout and np.any(~np.isin(st, list(types.values()))):
        raise ValueError("cluster sector type outside named geometry")
    for g, secs in groups.items():
        G = A["groups"][g]
        m = s0 & np.isin(sec, secs)
        G["npulse"] += int(m.sum())
        G["h_pe"] = _add(G["h_pe"], np.histogram(pe[m], bins=PE_EDGES)[0])
        G["nclus"] += int((st == types[g]).sum())
    if step_paths:
        _fill_eff(A, p, reftables.event_key(ev), step_paths, geom, layout)


def _fill_eff(A, p, ev_keys, step_paths, geom, layout=None):
    """Per-layer efficiency of the arm's own primary-muon crossings: max seq-0 pulse PE over the bar's SiPMs >= EFF_PE."""
    groups = {g: secs for g, _, secs in layout.groups} if layout else GROUPS
    nlayers = layout.nlayers if layout else NLAYERS
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
    last = np.r_[np.any(pk[1:] != pk[:-1], axis=1), True] if len(pk) else np.zeros(0, bool)
    xev = (x.run.to_numpy().astype(np.int64) << 40) | (x.subrun.to_numpy().astype(np.int64) << 20) | \
        x.event.to_numpy().astype(np.int64)
    xk = np.stack([xev, x.bar.to_numpy().astype(np.int64)], 1)
    # crossings only in events the reco saw (the art job's universe)
    seen = np.isin(xev, ev_keys)
    maxpe = (match.lookup(xk[seen], pk[last], pe[last], missing=0.0)
             if len(pk) and seen.any() else np.zeros(int(seen.sum())))
    xb = x.bar.to_numpy()[seen]
    lay, sec = geom.layer[xb].astype(np.int64), geom.sector[xb]
    hit = maxpe >= EFF_PE
    for g, secs in groups.items():
        G = A["groups"][g]
        m = np.isin(sec, secs)
        G["eff_den"] = _add(G["eff_den"], np.bincount(lay[m], minlength=nlayers)[:nlayers])
        G["eff_num"] = _add(G["eff_num"], np.bincount(lay[m & hit], minlength=nlayers)[:nlayers])


def _merge_into(tot, d):
    if "producer_receipt_audit" in tot or "producer_receipt_audit" in d:
        raise ValueError("receipt-audited V2 states cannot be merged; revalidate original inputs")
    if ("configuration" in tot) != ("configuration" in d) or tot.get("configuration") != d.get("configuration"):
        raise ValueError("cannot merge different V2 configurations/geometries or named and legacy states")
    layout = _state_layout(tot)
    groups = [g for g, _, _ in layout.groups] if layout else GROUPS
    if layout:
        # Validate both complete schemas before mutating any accumulated count.
        for S in (tot, d):
            for arm in ("fast", "ref"):
                A = S[arm]
                if len(A["bar_rate"]) != layout.nbars or set(A["groups"]) != set(groups):
                    raise ValueError("named V2 state geometry/group mismatch")
                for G in A["groups"].values():
                    for k, n in (("h_pe", len(PE_EDGES) - 1), ("eff_den", layout.nlayers),
                                 ("eff_num", layout.nlayers)):
                        if len(G[k]) != n:
                            raise ValueError("named V2 state histogram shape mismatch")
    for arm in ("fast", "ref"):
        T, D = tot[arm], d[arm]
        if layout:
            T["missing_trees"] = sorted(set(T["missing_trees"]) | set(D["missing_trees"]))
        T["events"] += D["events"]
        T["files"] += D["files"]
        T["bar_rate"] = _add(T["bar_rate"], D["bar_rate"])
        for g in groups:
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
    if not ok.any():
        return {"max_abs_dev": None, "max_abs_dev_minus_2sigma": None, "chi2_ndf": None, "nbins_used": 0}
    r = (rf[ok] / max(hf.sum(), 1)) / (rr[ok] / hr.sum())
    err = r * np.sqrt(1 / np.maximum(rf[ok], 1) + 1 / rr[ok])
    return {"max_abs_dev": float(np.abs(r - 1).max()),
            "max_abs_dev_minus_2sigma": float(np.maximum(np.abs(r - 1) - 2 * err, 0).max()),
            "chi2_ndf": float(np.sum(np.divide((r - 1) ** 2, err ** 2, out=np.full_like(r, np.inf), where=err > 0)) / ok.sum()), "nbins_used": int(ok.sum())}


def bar_rates(nf, nr, ef, er):
    """chi2/ndf and max |pull| of per-bar counts, the fast counts scaled to the reference's event count."""
    nf, nr = np.asarray(nf, float), np.asarray(nr, float)
    if ef <= 0 or er <= 0 or not np.any(nf + nr):
        return {"chi2_ndf": float("nan"), "max_abs_pull": float("nan"), "worst_bar": -1, "nbars_used": 0}
    s = er / ef
    ok = (nf + nr) > 0
    pull = (s * nf[ok] - nr[ok]) / np.sqrt(s * s * nf[ok] + nr[ok])
    worst = int(np.flatnonzero(ok)[np.argmax(np.abs(pull))]) if ok.any() else -1
    return {"chi2_ndf": float(np.sum(pull ** 2) / max(ok.sum(), 1)), "max_abs_pull": float(np.abs(pull).max()),
            "worst_bar": worst, "nbars_used": int(ok.sum())}


def _summarise_named(S):
    """Same tolerances as legacy V2, but unavailable evidence is never a pass."""
    layout = _state_layout(S)
    F, R = S["fast"], S["ref"]
    exposure = F["events"] > 0 and R["events"] > 0
    missing = {a: S[a].get("missing_trees", []) for a in ("fast", "ref")}
    br = bar_rates(F["bar_rate"], R["bar_rate"], F["events"], R["events"])
    supported = exposure and br["nbars_used"] > 0 and not any(missing.values())
    br["pass"] = bool(supported and br["chi2_ndf"] <= THRESH["bar_rate_chi2_ndf"]
                      and br["max_abs_pull"] <= THRESH["bar_rate_max_pull"])
    br["status"] = "unsupported" if not supported else ("pass" if br["pass"] else "fail")
    if not supported:
        br["chi2_ndf"] = br["max_abs_pull"] = None
    out = {"configuration": json.loads(layout.identity), "events_fast": F["events"], "events_ref": R["events"],
           "thresholds": dict(THRESH), "bar_rate": br, "groups": {}, "missing_trees": missing,
           "layer_efficiency_status": "diagnostic_only_truth_crossings_not_track_efficiency_or_gap_width"}
    for g, t, secs in layout.groups:
        GF, GR = F["groups"].get(g), R["groups"].get(g)
        res = {"sector_type": t, "sector_indices": list(secs), "checks": {}, "unsupported": [],
               "layer_efficiency_status": "diagnostic_only_not_computed"}
        out["groups"][g] = res
        if GF is None or GR is None:
            res.update(checks={"pe_median": False, "pe_ratio": False, "clusters": False},
                       unsupported=["missing_group"], status="unsupported", **{"pass": False})
            continue
        context = exposure and not any(missing.values()) and bool(secs)
        if not context:
            res["unsupported"].append("missing_exposure_or_tree_or_sectors")
        res.update(npulse_ref=GR["npulse"], npulse_fast=GF["npulse"],
                   nclus_ref=GR["nclus"], nclus_fast=GF["nclus"])
        res["pulses_per_event_rel"] = ((GF["npulse"] / F["events"]) / (GR["npulse"] / R["events"]) - 1
                                       if exposure and GR["npulse"] else None)
        (mf, emf), (mr, emr) = hist_median(GF["h_pe"], PE_EDGES), hist_median(GR["h_pe"], PE_EDGES)
        median_ok = context and np.isfinite([mf, emf, mr, emr]).all() and mf > 0 and mr > 0
        rel, err = (mf / mr - 1, abs(mf / mr) * np.hypot(emf / mf, emr / mr)) if median_ok else (None, None)
        res.update(pe_median_ref=mr if np.isfinite(mr) else None, pe_median_fast=mf if np.isfinite(mf) else None,
                   pe_median_rel=rel, pe_median_rel_err=err)
        res["checks"]["pe_median"] = bool(median_ok and max(abs(rel) - 2 * err, 0) <= THRESH["pe_median_rel"])
        if not median_ok:
            res["unsupported"].append("pe_median_requires_50_histogram_entries_each_arm_and_finite_estimate")
        band = ratio_band(GF["h_pe"], GR["h_pe"], PE_EDGES, *SPEC_RANGE)
        res["pe_ratio"] = band
        i0, i1 = np.searchsorted(PE_EDGES, SPEC_RANGE)
        band_ok = (context and sum(GF["h_pe"][i0:i1]) > 0 and band["nbins_used"] > 0
                   and band["max_abs_dev_minus_2sigma"] is not None
                   and np.isfinite(band["max_abs_dev_minus_2sigma"]))
        if band["chi2_ndf"] is not None and not np.isfinite(band["chi2_ndf"]):
            band["chi2_ndf"] = None
        res["checks"]["pe_ratio"] = bool(band_ok and band["max_abs_dev_minus_2sigma"] <= THRESH["pe_ratio_band"])
        if not band_ok:
            res["unsupported"].append("pe_spectrum_requires_2000_reference_entries_in_5_150_and_candidate_support")
        cluster_ok = context and GF["nclus"] > 0 and GR["nclus"] > 0
        rel = ((GF["nclus"] / F["events"]) / (GR["nclus"] / R["events"]) - 1) if cluster_ok else None
        err = ((rel + 1) * np.sqrt(1 / GF["nclus"] + 1 / GR["nclus"])) if cluster_ok else None
        res.update(clusters_per_event_rel=rel, clusters_per_event_rel_err=err)
        res["checks"]["clusters"] = bool(cluster_ok and max(abs(rel) - 2 * err, 0) <= THRESH["clusters_per_event_rel"])
        if not cluster_ok:
            res["unsupported"].append("clusters_require_nonzero_counts_each_arm")
        if sum(GR["eff_den"]):
            df, dr = np.asarray(GF["eff_den"]), np.asarray(GR["eff_den"])
            ef = np.asarray(GF["eff_num"], float) / np.maximum(df, 1)
            er = np.asarray(GR["eff_num"], float) / np.maximum(dr, 1)
            err = np.sqrt(ef * (1 - ef) / np.maximum(df, 1) + er * (1 - er) / np.maximum(dr, 1))
            res.update(layer_eff_fast=ef.tolist(), layer_eff_ref=er.tolist(), layer_eff_diff=(ef - er).tolist(),
                       layer_eff_diff_err=err.tolist(), layer_eff_den_ref=GR["eff_den"],
                       layer_efficiency_status="diagnostic_only_fixed_8_PE_truth_crossings",
                       diagnostic_layer_eff_ok=bool(np.all((df > 0) & (dr > 0)) and
                                                    np.all(np.maximum(np.abs(ef - er) - 2 * err, 0) <= THRESH["layer_eff_diff"])))
        res["pass"] = all(res["checks"].values())
        res["status"] = "unsupported" if res["unsupported"] else ("pass" if res["pass"] else "fail")
    out["pass"] = br["pass"] and all(r["pass"] for r in out["groups"].values())
    unsupported = br["status"] == "unsupported" or any(r["status"] == "unsupported" for r in out["groups"].values())
    out["status"] = "unsupported" if unsupported else ("pass" if out["pass"] else "fail")
    out["metrics_pass"], out["metrics_status"] = out["pass"], out["status"]
    out["input_provenance_status"] = "unverified_requested_configuration_not_bound_to_both_inputs"
    if "producer_receipt_audit" in S:
        out["producer_receipt_audit"] = S["producer_receipt_audit"]
        out["input_provenance_status"] = "partial_fast_receipt_bound_stock_configuration_not_certified"
    out["pass"], out["status"] = False, "unsupported"
    return out


def summarise(S):
    if "configuration" in S:
        return _summarise_named(S)
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
    groups = [g for g, _, _ in _state_layout(S).groups] if "configuration" in S else GROUPS
    fig, axes = plt.subplots(2, len(groups), figsize=(4 * len(groups), 5), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]})
    edges = PE_EDGES[::10]  # 1 PE bins for display
    for j, g in enumerate(groups):
        empty = [0] * (len(PE_EDGES) - 1)
        hf = np.add.reduceat(np.asarray(S["fast"]["groups"].get(g, {}).get("h_pe", empty), float),
                             np.arange(0, len(PE_EDGES) - 1, 10))
        hr = np.add.reduceat(np.asarray(S["ref"]["groups"].get(g, {}).get("h_pe", empty), float),
                             np.arange(0, len(PE_EDGES) - 1, 10))
        plots.ratio_panel(axes[0, j], axes[1, j], edges, hr, hf, "PE per pulse (seq 0)", labels=labels)
        axes[0, j].set_title(f"{g}  (N_ref={int(hr.sum()):,})")
        axes[0, j].set_xlim(0, 150)
    plots.save(fig, outdir, "v2compare_pe.png")
    br = summ["bar_rate"]
    if "configuration" in S:
        print(f"bars: {br}")
        for g, r in summ["groups"].items():
            print(f"{g}: {r['status']} checks={r['checks']} unsupported={r['unsupported']}")
        print(f"Named metrics {summ['metrics_status'].upper()}; V2 UNSUPPORTED over {summ['events_ref']:,} events "
              f"({summ['input_provenance_status']}; layer efficiency diagnostic only)")
        return summ
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
        generated=None, compare_generated=None, run_config=None, candidate_receipt=None, conditions_manifest=None, **kw):
    receipt_mode = candidate_receipt is not None or conditions_manifest is not None
    if receipt_mode and (not candidate_receipt or not conditions_manifest or run_config is None or merge or emit
                         or generated is not None or compare_generated is not None or counters):
        raise ValueError("receipt audit needs named config, both receipt/conditions manifest, no emit/merge/overrides")
    geom = None
    if run_config is not None:
        from .. import runconfig
        rc = runconfig.get(run_config) if isinstance(run_config, str) else run_config
        if counters:
            rc = dataclasses.replace(rc, counters=counters)
        geom = rc.geometry()
        S = _new_state(rc, geom)
    else:
        S = _new_state()
    if merge:
        for f in tables.expand(inputs):
            _merge_into(S, json.load(open(f)))
        return report(S, outdir)
    if not compare:
        raise SystemExit("v2compare needs --compare <reference CrvRecoTableDumper files>")
    if geom is None:
        geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    layout = _state_layout(S)
    if receipt_mode:
        from .. import response_receipt as RR
        fast_paths = [os.path.realpath(p) for p in tables.expand(inputs)]
        ref_paths = [os.path.realpath(p) for p in tables.expand(compare)]
        if len(fast_paths) != 1 or len(ref_paths) != 1:
            raise ValueError("receipt audit currently requires exactly one file per arm")
        evidence = RR.validate_receipt(fast_paths[0], candidate_receipt, run_config=rc, geom=geom,
                                       conditions_manifest=conditions_manifest)
        receipt = RR._json(candidate_receipt)
        source_key = "input" if receipt["metadata"]["stage"] == "reco" else "event_reference"
        source = receipt["inputs"].get(source_key)
        reference = RR._record(ref_paths[0])
        if source is None or any(source[k] != reference[k] for k in ("sha256", "size")):
            raise ValueError("comparison reference is not the producer's recorded stock population source")
        if RR.event_identity(fast_paths[0]) != RR.event_identity(ref_paths[0]):
            raise ValueError("receipt comparison event populations differ")
        S["producer_receipt_audit"] = {"fast": evidence, "reference": reference,
                                      "scope": "conditional_saved_stock_population_not_full_v2"}
    _fill_arm(S["fast"], fast_paths if receipt_mode else inputs, steps, geom, layout)
    _fill_arm(S["ref"], ref_paths if receipt_mode else compare, compare_steps, geom, layout)
    if receipt_mode:
        RR.validate_receipt(fast_paths[0], candidate_receipt, run_config=rc, geom=geom,
                            conditions_manifest=conditions_manifest)
        if RR._record(ref_paths[0]) != reference:
            raise ValueError("reference changed during V2 filling")
    if layout:
        if compare_generated is not None and generated is None:
            raise ValueError("compare_generated requires generated")
        for value in (generated, compare_generated):
            if value is not None and (isinstance(value, bool) or not np.isfinite(value)
                                      or value <= 0 or int(value) != value):
                raise ValueError("named V2 generated denominators must be positive integers")
    if generated:
        S["fast"]["events"] = int(generated)
        S["ref"]["events"] = int(compare_generated or generated)
    if emit:
        os.makedirs(os.path.dirname(emit) or ".", exist_ok=True)
        json.dump(S, open(emit, "w"))
        print(f"wrote {emit}")
        return S
    return report(S, outdir)
