"""timecal -- module time-offset calibration from CRV-only lines (class D of the tracker-CRV census: measured from data,
the same workflow run on MC, applied to both arms; one set per calibration period).

Input: the dtp histograms of a crvresponse accumulator: per event, the chosen T1 / T2 / tagger cluster time minus the
chosen EX cluster time, binned per (EX module, other module) pair. Model, per pair (e, m):

    median dt_data(e, m) - median dt_MC(e, m) = d_m - d_e     (d: data - MC time offset of a body; EX0 = 0, the gauge)

fitted by weighted least squares over every pair populated in both arms (medians from the histograms, weight from
both arms' entries and core widths). The pair medians themselves are not module offsets: time of flight and the
light's path along the bars depend on which modules a muon crosses (v05 MC, perfect timing: T1 - EX is 9.8 ns from EX0
and 2.0 ns from EX3), and that geometry cancels only in the data - MC difference. Taggers (one per pair, ~100 MC
entries each at 125 k events) are grouped in the MC by default; the data keeps one offset per tagger.

    o = fit(state)            -> {"offset": [16], "error": [16], "pairs": n, "rms_resid": ...}
    correction = data - MC    (per body, per period) -> apply_to_state / chain time offsets
"""
import numpy as np

from . import crvline as CL
from .crvresponse import DTP_EDGES, EX_BODIES

MIN_ENTRIES = 200      # per (EX module, module) pair; the MC (5 files, 125 k events) has ~100 per tagger pair
C = 0.5 * (DTP_EDGES[1:] + DTP_EDGES[:-1])


def _quantile(h, q):
    c = np.cumsum(h)
    if c[-1] <= 0:
        return np.nan
    x = q * c[-1]
    i = int(np.searchsorted(c, x))
    lo = c[i - 1] if i > 0 else 0.0
    f = (x - lo) / max(h[i], 1e-30)
    return DTP_EDGES[i] + f * (DTP_EDGES[i + 1] - DTP_EDGES[i])


def pair_medians(dtp, min_entries=MIN_ENTRIES):
    """(e, m, median, sigma_core, n) for every populated pair; medians from a +-10 ns window around the raw median."""
    dtp = np.asarray(dtp, float)
    out = []
    for e in range(dtp.shape[0]):
        for m in range(dtp.shape[1]):
            h = dtp[e, m]
            n = h.sum()
            if n < min_entries:
                continue
            md = _quantile(h, 0.5)
            hw = np.where(np.abs(C - md) < 10.0, h, 0.0)
            md = _quantile(hw, 0.5)
            s = 0.5 * (_quantile(hw, 0.84) - _quantile(hw, 0.16))
            out.append((e, m, md, s, hw.sum()))
    return out


def fit(state, min_entries=MIN_ENTRIES, group_taggers=False):
    """Body time offsets [ns] (EX0 = 0) from a crvresponse accumulator's dtp. Bodies with no pair stay NaN.
    group_taggers: the eight taggers share one offset (their dtp summed): MC samples too small per tagger, and the MC
    taggers have no per-tagger timing by construction."""
    dtp = np.asarray(state["dtp"], float)
    if group_taggers:
        dtp = dtp.copy()
        dtp[:, 8] = dtp[:, 8:].sum(axis=1)
        dtp[:, 9:] = 0.0
    P = pair_medians(dtp, min_entries)
    nb = CL.N_BODY
    seen = sorted({e for e, *_ in P} | {m for _, m, *_ in P})
    free = [b for b in seen if b != 0]
    col = {b: i for i, b in enumerate(free)}
    A = np.zeros((len(P), len(free)))
    y, w = np.zeros(len(P)), np.zeros(len(P))
    for r, (e, m, md, s, n) in enumerate(P):
        if m in col:
            A[r, col[m]] += 1.0
        if e in col:
            A[r, col[e]] -= 1.0
        y[r], w[r] = md, n / max(1.25 * s, 0.3) ** 2
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    cov = np.linalg.pinv((A * w[:, None]).T @ A)
    off, err = np.full(nb, np.nan), np.full(nb, np.nan)
    off[0], err[0] = 0.0, 0.0
    for b, i in col.items():
        off[b], err[b] = sol[i], np.sqrt(max(cov[i, i], 0.0))
    if group_taggers and np.isfinite(off[8]):
        off[9:], err[9:] = off[8], err[8]
    res = y - A @ sol
    return {"offset": off.tolist(), "error": err.tolist(), "pairs": len(P),
            "rms_resid": float(np.sqrt(np.average(res ** 2, weights=w))) if len(P) else None,
            "pair_table": [[CL.BODY_NAMES[e], CL.BODY_NAMES[m], md, s, n, float(md - (off[m] - off[e]))]
                           for e, m, md, s, n in P],
            "bodies": list(CL.BODY_NAMES)}


def fit_difference(data_state, mc_state, min_entries=MIN_ENTRIES, group_mc_taggers=True):
    """Per-body data - MC time offsets [ns] (EX0 = 0) from the pair-median differences. Returns offsets, errors, the
    pair table and the weighted rms of the residuals (the part a module offset cannot absorb)."""
    dd = np.asarray(data_state["dtp"], float)
    dm = np.asarray(mc_state["dtp"], float)
    if group_mc_taggers:
        dm = dm.copy()
        dm[:, 8:] = dm[:, 8:].sum(axis=1, keepdims=True)
    PD = {(e, m): (md, sg, n) for e, m, md, sg, n in pair_medians(dd, min_entries)}
    PM = {(e, m): (md, sg, n) for e, m, md, sg, n in pair_medians(dm, min_entries)}
    keys = sorted(set(PD) & set(PM))
    nb = CL.N_BODY
    seen = sorted({e for e, _ in keys} | {m for _, m in keys})
    free = [b for b in seen if b != 0]
    col = {b: i for i, b in enumerate(free)}
    A = np.zeros((len(keys), len(free)))
    y, w = np.zeros(len(keys)), np.zeros(len(keys))
    for r, (e, m) in enumerate(keys):
        (a, sa, na), (b, sb, nb_) = PD[(e, m)], PM[(e, m)]
        if m in col:
            A[r, col[m]] += 1.0
        if e in col:
            A[r, col[e]] -= 1.0
        y[r] = a - b
        w[r] = 1.0 / ((1.25 * max(sa, 0.3)) ** 2 / na + (1.25 * max(sb, 0.3)) ** 2 / nb_)
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    cov = np.linalg.pinv((A * w[:, None]).T @ A)
    off, err = np.full(nb, np.nan), np.full(nb, np.nan)
    off[0], err[0] = 0.0, 0.0
    for bb, i in col.items():
        off[bb], err[bb] = sol[i], np.sqrt(max(cov[i, i], 0.0))
    res = y - A @ sol
    return {"offset": off.tolist(), "error": err.tolist(), "pairs": len(keys),
            "rms_resid": float(np.sqrt(np.average(res ** 2, weights=w))) if len(keys) else None,
            "chi2ndf": float((res ** 2 * w).sum() / max(len(keys) - len(free), 1)) if len(keys) else None,
            "pair_table": [[CL.BODY_NAMES[e], CL.BODY_NAMES[m], PD[(e, m)][0], PM[(e, m)][0], float(res[r])]
                           for r, (e, m) in enumerate(keys)],
            "bodies": list(CL.BODY_NAMES)}


def apply_to_state(state, corr):
    """Shift an accumulator's dt_cluster and dtp histograms as if every body's times moved by corr[body] (bin-level,
    0.5 ns dtp bins, 1 ns dt_cluster bins: the shift is rounded to the bin). Returns a new state; for checks only, the
    real application is in the chain (time offsets per channel)."""
    import copy
    from .crvresponse import E, SLOTS
    S = copy.deepcopy(state)
    dtp = np.asarray(S["dtp"], float)
    new = np.zeros_like(dtp)
    bw = DTP_EDGES[1] - DTP_EDGES[0]
    for e in range(dtp.shape[0]):
        for m in range(dtp.shape[1]):
            k = int(np.rint((corr[m] - corr[e]) / bw))
            new[e, m] = np.roll(dtp[e, m], k)
            if k > 0:
                new[e, m][:k] = 0
            elif k < 0:
                new[e, m][k:] = 0
    S["dtp"] = new.tolist()
    # rebuild dt_cluster_<slot> from the shifted dtp (only pairs with a known EX module contribute)
    slot_of = np.array([0] * 4 + [1] * 2 + [2] * 2 + [3] * 8)
    for k, name in enumerate(SLOTS):
        if name == "EX":
            continue
        h = new[:, slot_of == k].sum(axis=(0, 1))
        S["h"][f"dt_cluster_{name}"] = np.histogram(np.repeat(0.5 * (DTP_EDGES[1:] + DTP_EDGES[:-1]), 1), bins=E["dt"],
                                                    weights=h)[0].tolist()
    return S


def main(argv=None):
    """python -m crvfastsim.objective.timecal --data P.json [...] --mc MC.json [...] --out CORR.json
    Inputs are crvresponse accumulators (with dtp); data states are merged into one period. Writes the data - MC body
    offsets and correction = -(data - MC): add it to the DATA times of that period to bring them to the MC time frame
    (crvresponse --time-corr), or subtract it from MC times to bring MC to the data frame."""
    import argparse
    import json
    from .crvresponse import merge_states
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--mc", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--per-tagger-mc", action="store_true", help="do not group the MC taggers")
    a = ap.parse_args(argv)
    D = merge_states([json.load(open(f)) for f in a.data])
    M = merge_states([json.load(open(f)) for f in a.mc])
    f = fit_difference(D, M, group_mc_taggers=not a.per_tagger_mc)
    off = np.asarray(f["offset"], float)
    corr = -np.where(np.isfinite(off), off, 0.0)
    out = {"label": a.label, "fit": f, "correction": corr.tolist(), "bodies": list(CL.BODY_NAMES),
           "data_files": a.data, "mc_files": a.mc}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.label}: pairs {f['pairs']}, residual rms {f['rms_resid']:.2f} ns, chi2/ndf {f['chi2ndf']:.1f}")
    print("data - MC offsets " + " ".join(f"{n}:{o:+.2f}+-{e:.2f}" for n, o, e in zip(CL.BODY_NAMES, f["offset"], f["error"])
                                         if o == o))


if __name__ == "__main__":
    main()
