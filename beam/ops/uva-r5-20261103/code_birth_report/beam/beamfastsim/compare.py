"""Fast vs G4 comparison on the same DS-entrance muons: stop fraction, per-foil profile, stop radius, stopped-muon
entry momentum, per-muon agreement (both stop / one stops), and plots.
"""
import json
import os

import numpy as np

from . import transport


def _binom(k, n):
    f = k / n
    return f, np.sqrt(f * (1 - f) / n)


def run(g, f, geom, outdir):
    os.makedirs(outdir, exist_ok=True)
    n = len(f["fate"])
    g = {k: v[:n] for k, v in g.items()}
    g_stop = g["foil"] >= 0
    f_stop = f["fate"] == 1
    neg = g["q"] < 0
    m = {"n_muons": n, "n_mu_minus": int(neg.sum())}
    for lab, sel in (("all", np.ones(n, bool)), ("mu_minus", neg)):
        fg, eg = _binom(g_stop[sel].sum(), sel.sum())
        ff, ef = _binom(f_stop[sel].sum(), sel.sum())
        m[f"stop_frac_g4_{lab}"], m[f"stop_frac_fast_{lab}"] = fg, ff
        m[f"stop_frac_ratio_{lab}"] = ff / fg
        m[f"stop_frac_ratio_err_{lab}"] = (ff / fg) * np.hypot(eg / fg, ef / ff)
    m["fast_fates"] = {transport.FATES[k]: int(v) for k, v in zip(*np.unique(f["fate"], return_counts=True))}
    m["g4_end_codes"] = {int(k): int(v) for k, v in zip(*np.unique(g["codestop"], return_counts=True))}
    m["g4_at_rest_not_in_foil"] = int((g["at_rest"] & ~g_stop).sum())
    both = g_stop & f_stop
    m["per_muon"] = {"both_stop": int(both.sum()), "g4_only": int((g_stop & ~f_stop).sum()),
                     "fast_only": int((~g_stop & f_stop).sum()), "neither": int((~g_stop & ~f_stop).sum())}
    m["same_foil_given_both"] = float(np.mean(g["foil"][both] == f["foil"][both])) if both.any() else None
    m["abs_foil_diff_mean_given_both"] = float(np.mean(np.abs(g["foil"][both] - f["foil"][both]))) if both.any() else None
    nf = len(geom.foil_z)
    hg = np.bincount(g["foil"][g_stop], minlength=nf)
    hf = np.bincount(f["foil"][f_stop], minlength=nf)
    m["foil_profile_g4"], m["foil_profile_fast"] = hg.tolist(), hf.tolist()
    sel = hg > 0
    m["foil_profile_chi2_ndf"] = float(np.sum((hf[sel] - hg[sel]) ** 2 / (hf[sel] + hg[sel])) / max(sel.sum() - 1, 1))
    rg = np.hypot(g["xstop"][g_stop] - geom.axis_x, g["ystop"][g_stop])
    rf = np.hypot(f["x"][f_stop, 0] - geom.axis_x, f["x"][f_stop, 1])
    p0 = np.sqrt(g["px"] ** 2 + g["py"] ** 2 + g["pz"] ** 2)
    m["stop_r_mean_g4"], m["stop_r_mean_fast"] = float(rg.mean()), float(rf.mean())
    m["stop_r_ks_p"] = float(stats_ks(rg, rf))
    m["entry_p_of_stopped_mean_g4"] = float(p0[g_stop].mean())
    m["entry_p_of_stopped_mean_fast"] = float(p0[f_stop].mean())
    m["meta"] = json.loads(str(f["meta"])) if "meta" in f else None
    _plots(outdir, geom, hg, hf, rg, rf, p0, g_stop, f_stop)
    with open(os.path.join(outdir, "metrics.json"), "w") as fo:
        json.dump(m, fo, indent=1)
    return m


def stats_ks(a, b):
    from mu2efastsim import match
    r = match.ks_2samp(a, b)
    return r[1] if isinstance(r, tuple) else r


def _plots(outdir, geom, hg, hf, rg, rf, p0, gs, fs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    i = np.arange(len(hg))
    ax[0, 0].step(i, hg, where="mid", label="Geant4")
    ax[0, 0].step(i, hf, where="mid", label="fast")
    ax[0, 0].set(xlabel="foil index", ylabel="stops", title="stops per foil")
    ax[0, 0].legend()
    b = np.linspace(0, 80, 41)
    ax[0, 1].hist(rg, b, histtype="step", label="Geant4")
    ax[0, 1].hist(rf, b, histtype="step", label="fast")
    ax[0, 1].set(xlabel="stop radius [mm]", title="stop radius")
    b = np.linspace(0, 120, 61)
    ax[1, 0].hist(p0, b, histtype="step", color="gray", label="all DS-entrance muons")
    ax[1, 0].hist(p0[gs], b, histtype="step", label="stopped, Geant4")
    ax[1, 0].hist(p0[fs], b, histtype="step", label="stopped, fast")
    ax[1, 0].set(xlabel="|p| at DS entrance [MeV/c]", title="entry momentum")
    ax[1, 0].legend()
    eff_g = np.histogram(p0[gs], b)[0] / np.maximum(np.histogram(p0, b)[0], 1)
    eff_f = np.histogram(p0[fs], b)[0] / np.maximum(np.histogram(p0, b)[0], 1)
    c = 0.5 * (b[1:] + b[:-1])
    ax[1, 1].plot(c, eff_g, drawstyle="steps-mid", label="Geant4")
    ax[1, 1].plot(c, eff_f, drawstyle="steps-mid", label="fast")
    ax[1, 1].set(xlabel="|p| at DS entrance [MeV/c]", ylabel="stopping probability", title="stopping vs momentum")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "compare.png"), dpi=110)
