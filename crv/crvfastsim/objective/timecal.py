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
from .crvresponse import DTP_EDGES, EX_BODIES, FEB_GROUPS, FTP_EDGES

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


def feb_offsets(data_state, mc_state, min_entries=200):
    """Per (body, FEB group) data - MC offset of the first-pulse time relative to its cluster mean [ns], EX and T2
    (groups: layer pair x side). Medians of the ftp histograms; a module's four groups average to ~0 by construction
    (the module offset is fitted separately), so these refine the module calibration within a module."""
    fd, fm = np.asarray(data_state["ftp"], float), np.asarray(mc_state["ftp"], float)
    cc = 0.5 * (FTP_EDGES[1:] + FTP_EDGES[:-1])

    def med(h):
        c = np.cumsum(h)
        return float(np.interp(0.5 * c[-1], c, cc)) if c[-1] >= min_entries else np.nan
    out = np.full((CL.N_BODY, FEB_GROUPS), np.nan)
    for b in list(range(4)) + [6, 7]:
        for g in range(FEB_GROUPS):
            out[b, g] = med(fd[b, g]) - med(fm[b, g])
    return out


def channel_table(module_corr, feb_corr=None, nchan=2304):
    """Per-channel time shift [ns] (channel = 4 barId + SiPMId) from per-body offsets and optional (body, FEB group)
    offsets: the table the chains add to pulse times (fast chain: timeoffset; stock: a CRVTime text table)."""
    from .bars import bar_slot_body_layer
    ch = np.arange(nchan)
    bar, sipm = ch // 4, ch % 4
    slot, body, layer = bar_slot_body_layer(bar)
    t = np.asarray(module_corr, float)[body].copy()
    if feb_corr is not None:
        fc = np.nan_to_num(np.asarray(feb_corr, float))
        grp = (layer >= 2).astype(np.int64) * 2 + (sipm % 2)
        t += fc[body, grp]
    return t


def write_crvtime_digi(path, base_offsets, shift, note=""):
    """CRVTime text table for the stock MC DIGITISATION job only, so that MC pulse times move by +shift [ns] per channel.
    crvonly_v05 digitisation applies CRVTime with timeOffsetScale -1 and the reco adds the DB table (Sim_best, the base
    here): net = base - T_digi, so T_digi = base - shift. Use with services.DbService.textFile in the digi job, never in
    the reco job (which keeps the DB table, as for data)."""
    return write_crvtime(path, base_offsets, -np.asarray(shift, float), note + " (digi-only: base - shift)")


def write_crvtime(path, base_offsets, shift, cid_note=""):
    """DbService text table CRVTime: base per-channel timeOffset (e.g. the Sim_best table the MC used) plus shift."""
    off = np.asarray(base_offsets, float) + np.asarray(shift, float)
    with open(path, "w") as fh:
        fh.write(f"TABLE CRVTime\n# timeOffset + module/FEB time calibration from CRV-only lines {cid_note}\n")
        for c, o in enumerate(off):
            fh.write(f"{c},{o:.6f}\n")
    return path


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
    ap.add_argument("--no-feb", action="store_true", help="module offsets only (no FEB-group refinement)")
    ap.add_argument("--crvtime-base", default=None, help="CRVCalib CSV of the MC (timeoffset column): also write the "
                    "digi-only CRVTime text table <out>.crvtime_digi.txt for the stock MC")
    a = ap.parse_args(argv)
    D = merge_states([json.load(open(f)) for f in a.data])
    M = merge_states([json.load(open(f)) for f in a.mc])
    f = fit_difference(D, M, group_mc_taggers=not a.per_tagger_mc)
    off = np.asarray(f["offset"], float)
    corr = -np.where(np.isfinite(off), off, 0.0)
    feb = feb_offsets(D, M) if ("ftp" in D and "ftp" in M and not a.no_feb) else None
    out = {"label": a.label, "fit": f, "correction": corr.tolist(), "bodies": list(CL.BODY_NAMES),
           "feb_offset": None if feb is None else np.where(np.isfinite(feb), feb, None).tolist(),
           "mc_channel_shift": channel_table(np.where(np.isfinite(off), off, 0.0), feb).tolist(),
           "data_files": a.data, "mc_files": a.mc}
    json.dump(out, open(a.out, "w"), indent=1)
    if a.crvtime_base:
        base = np.loadtxt(a.crvtime_base, delimiter=",", comments="#")
        b = np.zeros(len(out["mc_channel_shift"]))
        b[base[:, 0].astype(int)] = base[:, 4]
        p = a.out.rsplit(".json", 1)[0] + ".crvtime_digi.txt"
        write_crvtime_digi(p, b, out["mc_channel_shift"], f"{a.label}")
        print("wrote", p)
    print(f"{a.label}: pairs {f['pairs']}, residual rms {f['rms_resid']:.2f} ns, chi2/ndf {f['chi2ndf']:.1f}")
    print("data - MC offsets " + " ".join(f"{n}:{o:+.2f}+-{e:.2f}" for n, o, e in zip(CL.BODY_NAMES, f["offset"], f["error"])
                                         if o == o))
    if feb is not None:
        print("FEB groups (L01S0 L01S1 L23S0 L23S1) data - MC: " + "; ".join(
            f"{CL.BODY_NAMES[b]} " + " ".join(f"{x:+.2f}" for x in feb[b]) for b in (0, 1, 2, 3, 6, 7)))


if __name__ == "__main__":
    main()
