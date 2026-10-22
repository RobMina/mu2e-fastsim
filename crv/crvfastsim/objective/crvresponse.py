"""crvresponse -- CRV response observables on CRV-only lines and the MC-vs-data objective (plan
docs/crv_only_alignment_plan.md §2.3 and §5.2; the R3/V-R5 objective of the CRV fastreco).

One accumulator for every source: stock EventNtuples (KPP data, crvonly MC) and the fast chain (crvfastsim.reco.ntview
gives the same branches), so data, stock MC and fast MC are histogrammed by the same code. Per event, the line comes
from crvline.fit_arrays (largest EX cluster plus in-time T1/T2/tagger clusters, seq-0 pulses only).

Observables (per sector slot EX/T1/T2/TAG unless noted; "along" = line coordinate along the bar minus the bar centre,
in units of the half length; side 0 is the low end):
  pe_along   PE per pulse at normal incidence (PE / sqrt(1 + sx^2 + sz^2)) vs along, per side   [light yield, attenuation]
  pe_path    raw PE per pulse vs the path factor sqrt(1 + sx^2 + sz^2)                          [yield vs path]
  asym       ln(PE0/PE1) per bar with both ends vs along (EX, T2)                               [attenuation, reflector]
  tdiff      t0 - t1 per bar with both ends vs along (EX, T2)                                   [fibre speed, side offsets]
  tspread    rms of the propagation-corrected first-pulse times in the chosen cluster          [decay times, jitter]
  dt_cluster chosen-cluster time minus the EX cluster time (T1, T2, TAG)                       [module/sector timing]
  nlayers    layers in the chosen cluster                                                      [thresholds, dead SiPMs]
  pe_layer   PEsPerLayer of the chosen cluster                                                 [thresholds]
  oot_pe     PE of unclustered seq-0 pulses more than OOT_DT from every cluster, and their rate per event per us
                                                                                                [thermal rate, crosstalk]
Objective (objective()): per histogram a size-free shape distance, the triangular discrimination
D = sum_i (p_i - q_i)^2 / (p_i + q_i) of the two parent shapes p (candidate) and q (reference), estimated from the
counts a, b (entries A, B) with the Poisson bias removed from the numerator: (a/A - b/B)^2 - a/A^2 - b/B^2. D does not
depend on A or B (the ROOT Chi2Test UU chi2/ndf it replaces grew with the sample size, and its bins where one sample
is nearly empty scaled with A/B). It is quoted on the chi2/ndf scale of two N0-entry histograms, chi2ndf = 1 + N0 D/ndf
(1 = identical shapes; noise can take it below 1). sigma is its statistical noise from Poisson resampling of the
candidate (sigma: the obs_noise a BO needs) and of the reference (sigma_ref: common to every BO point, a
bias rather than point-to-point noise; large only when the reference histogram is small). Per slot,
(ln(rate / rate_ref) / RATE_TOL)^2 for the out-of-time rate (size-free; sigma from the candidate count). Total = mean
over all terms (each observable weighs the same). raw_chi2ndf / objective_raw keep the UU chi2/ndf for reference.

    crvresponse NTUPLE.root [...] --emit X.json [--window-ns W]
    crvresponse "dir/*.json" --merge --outdir DIR [--compare "ref/*.json"]   (objective of inputs vs compare)
"""
import glob
import json
import os

import awkward as ak
import numpy as np

from . import bars as G
from . import crvline as CL

SLOTS = CL.SLOT_NAME
HALF = {"EX": 3000.0, "T1": 1185.0, "T2": 1600.0, "TAG": 1100.0}   # half bar lengths, extracted v04 [mm]
ZSLOT = (True, False, True, False)    # EX, T2 stack in z (bars along x); T1, taggers stack in x (bars along z)
OOT_DT = 500.0                        # ns from every cluster for an out-of-time pulse
RATE_TOL = 0.1                        # relative tolerance of the out-of-time rate term
N0 = 10_000                           # nominal entries per histogram for the size-free chi2/ndf scale
MIN_SPREAD_HITS = 3
E = {"along": np.linspace(-1, 1, 11), "pe": np.linspace(0, 200, 41), "path": np.linspace(1, 3, 9),
     "asym": np.linspace(-3, 3, 31), "tdiff": np.linspace(-50, 50, 41), "tspread": np.linspace(0, 10, 41),
     "dt": np.linspace(-25, 25, 51), "nl": np.arange(-0.5, 5.0, 1.0), "pel": np.linspace(0, 400, 41),
     "oot": np.linspace(0, 40, 41)}
SHAPES = {"pe_along": ("side", "along", "pe"), "pe_path": ("path", "pe"), "asym": ("along", "asym"),
          "tdiff": ("along", "tdiff"), "tspread": ("tspread",), "dt_cluster": ("dt",), "nlayers": ("nl",),
          "pe_layer": ("pel",), "oot_pe": ("oot",)}
SLOTS_OF = {"asym": ("EX", "T2"), "tdiff": ("EX", "T2"), "dt_cluster": ("T1", "T2", "TAG")}
V_FIBER = CL.V_FIBER
TREE = "EventNtuple/ntuple"


def _shape(name):
    return tuple(2 if a == "side" else len(E[a]) - 1 for a in SHAPES[name])


def new_state():
    S = {"events": 0, "files": 0, "exposure_ns": 0.0, "lines_z": 0, "lines_x": 0, "lines_xz": 0, "h": {}, "oot_n": {}}
    for name in SHAPES:
        for s in SLOTS_OF.get(name, SLOTS):
            S["h"][f"{name}_{s}"] = np.zeros(_shape(name)).tolist()
    for s in SLOTS:
        S["oot_n"][s] = 0
    return S


def _fill(S, key, *cols):
    name = key.rsplit("_", 1)[0]
    axes = SHAPES[name]
    if len(cols[0]) == 0:
        return
    bins = [np.arange(-0.5, 2.0, 1.0) if a == "side" else E[a] for a in axes]
    h = np.histogramdd(np.stack([np.asarray(c, float) for c in cols], 1), bins=bins)[0] if len(axes) > 1 else \
        np.histogram(np.asarray(cols[0], float), bins=bins[0])[0]
    S["h"][key] = (np.asarray(S["h"][key]) + h).tolist()


def accumulate_arrays(c, p, S, window_ns):
    """Add one block of events: c, p jagged arrays with crvline's COINC_BR / PULSE_BR fields (one entry per event)."""
    o = CL.fit_arrays(c, p)
    nev = o["nev"]
    S["events"] += int(nev)
    S["exposure_ns"] += float(nev) * float(window_ns)
    okz, okx = o["okz"], o["okx"]
    S["lines_z"] += int(okz.sum())
    S["lines_x"] += int(okx.sum())
    S["lines_xz"] += int((okz & okx).sum())
    e, s, side, pe, t, y = o["hit_evt"], o["hit_slot"], o["hit_side"], o["hit_pe"], o["hit_time"], o["hit_y"]
    zs = np.isin(s, [0, 2])
    yy = y - CL.Y_REF
    # along-bar line coordinate minus the bar centre: EX/T2 from the x line, T1/taggers from the z line
    along_abs = np.where(zs, o["x0"][e] + o["sx"][e] * yy, o["z0"][e] + o["sz"][e] * yy)
    centre = np.where(zs, o["hit_px"], o["hit_pz"])
    half = np.array([HALF[n] for n in SLOTS])[s]
    al = (along_abs - centre) / half
    has_al = np.where(zs, okx[e], okz[e]) & (np.abs(al) <= 1.0)
    both = okz[e] & okx[e]
    pathf = np.sqrt(1.0 + o["sx"][e] ** 2 + o["sz"][e] ** 2)
    for k, name in enumerate(SLOTS):
        m = s == k
        mb = m & both & has_al
        _fill(S, f"pe_along_{name}", side[mb], al[mb], pe[mb] / pathf[mb])
        mp = m & both
        _fill(S, f"pe_path_{name}", pathf[mp], pe[mp])
    # per bar (event, body, layer, stacking position) and side: PE sums and PE-weighted first-pulse times
    bk = ((e.astype(np.int64) * CL.N_BODY + o["hit_body"]) * 4 + o["hit_layer"]) * 4096 + \
        np.round(o["hit_u"] / 10.0).astype(np.int64) % 4096
    ub, inv = np.unique(bk, return_inverse=True)
    pe_s = [np.bincount(inv[side == q], pe[side == q], len(ub)) for q in (0, 1)]
    tw_s = [np.bincount(inv[side == q], (pe * t)[side == q], len(ub)) for q in (0, 1)]
    first = np.zeros(len(ub), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    sb, alb, hb = s[first], al[first], has_al[first]
    ok2 = (pe_s[0] > 0) & (pe_s[1] > 0) & hb
    for k, name in ((0, "EX"), (2, "T2")):
        m = ok2 & (sb == k)
        _fill(S, f"asym_{name}", alb[m], np.log(pe_s[0][m] / pe_s[1][m]))
        _fill(S, f"tdiff_{name}", alb[m], tw_s[0][m] / pe_s[0][m] - tw_s[1][m] / pe_s[1][m])
    # intra-cluster spread of propagation-corrected times (distance to the hit's readout end along the line)
    dist = np.where(side == 0, (al + 1.0), (1.0 - al)) * half
    tc = t - dist / V_FIBER
    gk = e.astype(np.int64) * 4 + s
    mt = has_al
    ug, gi = np.unique(gk[mt], return_inverse=True)
    n = np.bincount(gi, minlength=len(ug))
    m1 = np.bincount(gi, tc[mt], len(ug)) / np.maximum(n, 1)
    m2 = np.bincount(gi, tc[mt] ** 2, len(ug)) / np.maximum(n, 1)
    rms = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))
    gs = ug % 4
    for k, name in enumerate(SLOTS):
        m = (gs == k) & (n >= MIN_SPREAD_HITS)
        _fill(S, f"tspread_{name}", rms[m])
    f = o["feat"]
    for name in SLOTS:
        hs = o["has"][name]
        _fill(S, f"nlayers_{name}", f[f"nl_{name}"][hs])
        if name != "EX":
            _fill(S, f"dt_cluster_{name}", (f[f"t_{name}"] - f["t_EX"])[hs])
    chosen, _ = CL._choose_clusters(c)
    pl = c["crvcoincs.PEsPerLayer[4]"]
    idx = ak.local_index(c["crvcoincs.sectorType"])
    for k, name in enumerate(SLOTS):
        pick = idx == chosen[k][:, None]
        v = ak.to_numpy(ak.flatten(ak.flatten(pl[pick], axis=1), axis=None))
        _fill(S, f"pe_layer_{name}", v[v > 0])
    # out-of-time unclustered first pulses
    pt, ct = p["crvpulses.time"], c["crvcoincs.time"]
    pr = ak.cartesian([pt, ct], axis=1, nested=True)
    a, b = ak.unzip(pr)
    dmin = ak.fill_none(ak.min(np.abs(a - b), axis=2), np.inf)
    oot = (p["crvpulses.crvHitIndex"] < 0) & (p["crvpulses.sequenceIndex"] == 0) & (dmin > OOT_DT)
    bid = ak.to_numpy(ak.flatten(p["crvpulses.barId"][oot]))
    ope = ak.to_numpy(ak.flatten(p["crvpulses.PEs"][oot]))
    slot, _, _ = G.bar_slot_body_layer(bid)
    for k, name in enumerate(SLOTS):
        m = slot == k
        S["oot_n"][name] += int(m.sum())
        _fill(S, f"oot_pe_{name}", ope[m])
    return o


def accumulate_file(path, S, window_ns=None, chunk=CL.CHUNK, entry_stop=None):
    """Stock EventNtuple: read COINC_BR and PULSE_BR in chunks and accumulate. window_ns None: the digitisation window
    is the largest pulse time rounded to the nearest 100 us (KPP data 400 us, MC 100 us)."""
    import uproot
    t = uproot.open(path)[TREE]
    n = t.num_entries if entry_stop is None else min(entry_stop, t.num_entries)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        c = t.arrays(CL.COINC_BR, entry_start=a, entry_stop=b)
        p = t.arrays(CL.PULSE_BR, entry_start=a, entry_stop=b)
        w = window_ns
        if w is None:
            tmax = float(ak.max(p["crvpulses.time"], axis=None) or 0.0)
            w = max(1e5, np.round(tmax / 1e5) * 1e5)
        S.setdefault("window_ns", []).append(float(w))
        accumulate_arrays(c, p, S, w)
    S["files"] += 1


def add_empty_events(S, n, window_ns):
    """Count n events with no CRV activity (no pulses, no lines): they add exposure, so the out-of-time rates stay per
    generated event, as in a stock EventNtuple, which holds every event. The fast chain replays only events with CRV
    steps and adds the rest here."""
    S["events"] += int(n)
    S["exposure_ns"] += float(n) * float(window_ns)
    return S


def merge_states(states):
    T = new_state()
    for S in states:
        for k in ("events", "files", "exposure_ns", "lines_z", "lines_x", "lines_xz"):
            T[k] += S[k]
        for k, v in S["h"].items():
            T["h"][k] = (np.asarray(T["h"][k]) + np.asarray(v)).tolist()
        for k, v in S["oot_n"].items():
            T["oot_n"][k] += v
        T["window_ns"] = sorted(set(T.get("window_ns", [])) | set(S.get("window_ns", [])))
    return T


def chi2_shape(a, b):
    """Two-sample homogeneity chi2 of unweighted histograms (ROOT Chi2Test "UU"): sum over bins filled in either of
    (sqrt(B/A) a - sqrt(A/B) b)^2 / (a + b); a bin empty in one sample is bounded by the other's statistics, so a
    large reference does not inflate it. Returns (chi2, ndf)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    A, B = a.sum(), b.sum()
    if A <= 0 or B <= 0:
        return 0.0, 0
    m = (a + b) > 0
    num = (np.sqrt(B / A) * a[m] - np.sqrt(A / B) * b[m]) ** 2
    return float((num / (a[m] + b[m])).sum()), int(m.sum() - 1)


N_BOOT = 64                           # Poisson resamplings of the candidate for sigma


def tri_disc(a, b):
    """Bias-corrected triangular discrimination of count arrays a [..., n], b [n]; returns D^ per leading index."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    A, B = a.sum(-1, keepdims=True), b.sum()
    pa, qb = a / A, b / B
    den = pa + qb
    num = (pa - qb) ** 2 - a / A ** 2 - b / B ** 2
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).sum(-1)


def shape_term(a, b, n0=N0, rng=None):
    """Size-free shape term of histograms a (candidate) and b (reference): chi2/ndf at n0 entries per histogram, its
    sigma (candidate resampled), D^, the raw UU chi2 and ndf. None if either histogram is empty."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    c2, ndf = chi2_shape(a, b)
    if ndf <= 0:
        return None
    d = float(tri_disc(a, b))
    rng = rng if rng is not None else np.random.default_rng(20261022)
    ab = rng.poisson(a, (N_BOOT, a.size)).astype(float)
    ok = ab.sum(1) > 0
    sd = float(np.std(tri_disc(ab[ok], b))) if ok.sum() > 1 else float("nan")
    bb = rng.poisson(b, (N_BOOT, b.size)).astype(float)
    okb = bb.sum(1) > 0
    sdr = float(np.std([tri_disc(a, x) for x in bb[okb]])) if okb.sum() > 1 else float("nan")
    return {"chi2ndf": 1.0 + n0 * d / ndf, "sigma": n0 * sd / ndf, "sigma_ref": n0 * sdr / ndf, "d": d, "ndf": ndf,
            "chi2": c2,
            "raw_chi2ndf": c2 / ndf, "entries": float(a.sum()), "entries_ref": float(b.sum())}


def objective(S, R, n0=N0):
    """Objective of S (candidate, e.g. MC) against R (reference, e.g. data). Returns the size-free total (objective,
    with its statistical sigma), the old raw-chi2 total (objective_raw) and the per-term values."""
    terms = {}
    rng = np.random.default_rng(20261022)
    for k in S["h"]:
        t = shape_term(S["h"][k], R["h"][k], n0, rng)
        if t is not None:
            terms[k] = t
    for s in SLOTS:
        ns, nr = S["oot_n"][s], R["oot_n"][s]
        es, er = S["exposure_ns"] * 1e-3, R["exposure_ns"] * 1e-3    # per event per us
        if es > 0 and er > 0 and ns + nr > 0:
            rs, rr = max(ns, 0.5) / es, max(nr, 0.5) / er
            x = float(np.log(rs / rr))
            v = (x / RATE_TOL) ** 2
            sig = 2.0 * abs(x) / RATE_TOL ** 2 / np.sqrt(max(ns, 0.5))
            sigr = 2.0 * abs(x) / RATE_TOL ** 2 / np.sqrt(max(nr, 0.5))
            terms[f"oot_rate_{s}"] = {"rate": rs, "rate_ref": rr, "chi2ndf": v, "raw_chi2ndf": v, "sigma": float(sig),
                                      "sigma_ref": float(sigr)}
    if not terms:
        return {"objective": float("nan"), "n_terms": 0, "terms": terms}
    n = len(terms)
    tot = float(np.mean([v["chi2ndf"] for v in terms.values()]))
    sig = float(np.sqrt(sum(v["sigma"] ** 2 for v in terms.values())) / n)
    sigr = float(np.sqrt(sum(v["sigma_ref"] ** 2 for v in terms.values())) / n)
    raw = float(np.mean([v["raw_chi2ndf"] for v in terms.values()]))
    return {"objective": tot, "sigma": sig, "sigma_ref": sigr, "objective_raw": raw, "n0": n0, "n_terms": n,
            "terms": terms}


def _expand(inputs):
    out = []
    for p in inputs:
        out += sorted(glob.glob(p)) or [p]
    return out


def write_json(outdir, name, obj):
    """Stats dict to outdir/name; NaN/inf become null so the JSON stays standard."""
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    with open(path, "w") as fh:
        json.dump(clean(obj), fh, indent=2)
    print(f"wrote {path}")
    return path


def report(S, outdir, R=None, n0=N0):
    res = {k: S[k] for k in ("events", "files", "exposure_ns", "lines_z", "lines_x", "lines_xz")}
    res["window_ns"] = S.get("window_ns")
    res["oot_rate_per_event_per_us"] = {s: S["oot_n"][s] / max(S["exposure_ns"] * 1e-3, 1e-300) for s in SLOTS}
    if R is not None:
        o = res["objective"] = objective(S, R, n0)
        print(f"objective {o['objective']:.3f} +- {o['sigma']:.3f} (candidate) +- {o['sigma_ref']:.3f} (reference) "
              f"(size-free, N0 {n0:g}) over {o['n_terms']} terms; raw chi2/ndf mean {o['objective_raw']:.3f}")
        worst = sorted(o["terms"].items(), key=lambda kv: -kv[1]["chi2ndf"])[:8]
        for k, v in worst:
            print(f"  {k:24s} chi2/ndf@N0 {v['chi2ndf']:9.2f} +- {v['sigma']:7.2f} +- {v['sigma_ref']:7.2f}   "
                  f"raw {v['raw_chi2ndf']:9.2f}")
    write_json(outdir, "crvresponse_stats.json", res)
    return res


def run(inputs, outdir, config=None, emit=None, merge=False, compare=None, entry_stop=None, window_ns=None, n0=N0,
        **kw):
    if merge:
        S = merge_states([json.load(open(f)) for f in _expand(inputs)])
        R = merge_states([json.load(open(f)) for f in _expand(compare)]) if compare else None
        return report(S, outdir, R, n0)
    S = new_state()
    for f in _expand(inputs):
        accumulate_file(f, S, window_ns=window_ns, entry_stop=entry_stop)
        print(f"# {f}: cumulative {S['events']:,} events", flush=True)
    if emit:
        os.makedirs(os.path.dirname(emit) or ".", exist_ok=True)
        json.dump(S, open(emit, "w"))
        print(f"wrote {emit}")
        return S
    return report(S, outdir)
