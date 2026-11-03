"""Fast conversion-electron chain against the Geant4 CeEndpoint reference at the tracker entry plane.

The Geant4 events passed the production PrimaryFilter (a particle above 50 MeV/c with at least 12 straw steps, or
45 MeV in the calorimeter), so a yield per generated electron is only comparable inside a selection that implies the
filter: momentum at the plane above P_MIN and helix maximum radius above an RMAX cut (the helix then crosses the
straw annulus over the whole tracker). Yields use fast runs from a stop table (unbiased starts); paired runs (the
Geant4 starts) give per-electron differences for the same start. Distributions: two-sample chi2 on 40 equal-
probability bins of the Geant4 sample, Poisson errors (the arms are treated as independent).
"""
import argparse
import glob
import json

import numpy as np

from mu2efastsim.eslab_g4 import chi2, quantile_edges, summary

from . import cechain as C

P_MIN = 60.0
RMAX_CUTS = (450.0, 500.0, 550.0)
NBINS = 40


E_CE = float(np.hypot(C.P_CE_AL, 0.51099895))        # conversion-electron total energy [MeV]
DE_EDGES = np.arange(-2.0, 0.2501, 0.25)              # tail bins of E - E_CE [MeV]
HIST_W = 0.02                                         # histogram bin for the MPV and FWHM [MeV]
N_BOOT = 200


def _peak(h, lo):
    """(MPV, FWHM) of histograms h (..., nb) with bins HIST_W from lo: parabola through the maximum bin and its
    neighbours; half-maximum crossings interpolated linearly."""
    nb = h.shape[-1]
    i = np.clip(np.argmax(h, axis=-1), 1, nb - 2)
    take = lambda k: np.take_along_axis(h, (i + k)[..., None], -1)[..., 0]
    a, b, c = take(-1), take(0), take(1)
    den = a - 2 * b + c
    off = np.where(den != 0, 0.5 * (a - c) / np.where(den != 0, den, 1.0), 0.0)
    centre = lo + (i + 0.5 + np.clip(off, -1, 1)) * HIST_W
    half = 0.5 * b
    j = np.arange(nb)
    below = h < half[..., None]
    left = np.where(below & (j < i[..., None]), j, -1).max(-1)
    right = np.where(below & (j > i[..., None]), j, nb).min(-1)
    hl, hl1 = take(left - i), take(left - i + 1)
    hr, hr1 = take(np.minimum(right, nb - 1) - i), take(right - i - 1)
    xl = lo + (left + 0.5 + (half - hl) / np.where(hl1 != hl, hl1 - hl, 1.0)) * HIST_W
    xr = lo + (right - 0.5 + (hr1 - half) / np.where(hr1 != hr, hr1 - hr, 1.0)) * HIST_W
    return centre, xr - xl


def energy_metrics(dE, n_norm, rng):
    """E - E_CE at the plane: MPV and FWHM with bootstrap errors (multinomial on the 0.02 MeV histogram), mean,
    and the fraction per normalisation count in 0.25 MeV bins from -2 to +0.25 MeV and below -2 MeV."""
    lo = -3.0
    edges = np.arange(lo, 0.5 + 1e-9, HIST_W)
    h = np.histogram(dE, edges)[0]
    mpv, fwhm = _peak(h[None, :].astype(float), lo)
    boot = rng.multinomial(h.sum(), h / max(h.sum(), 1), N_BOOT).astype(float)
    bm, bf = _peak(boot, lo)
    k = np.histogram(dE, DE_EDGES)[0]
    tail = (dE < DE_EDGES[0]).sum()
    frac = lambda c: dict(f=float(c / n_norm), err=float(np.sqrt(c * (1 - c / n_norm)) / n_norm))
    return dict(n=int(dE.size), mpv=float(mpv[0]), mpv_err=float(bm.std()), fwhm=float(fwhm[0]),
                fwhm_err=float(bf.std()), mean=float(dE.mean()), mean_err=float(dE.std() / np.sqrt(dE.size)),
                bins=[float(v) for v in DE_EDGES], frac=[frac(c) for c in k], below=frac(tail))


def g4_plane(ref):
    r = np.load(ref)
    x = np.stack([r["xvd_front"], r["yvd_front"], r["zvd_front"]], axis=1).astype(float)
    pv = np.stack([r["pxvd_front"], r["pyvd_front"], r["pzvd_front"]], axis=1).astype(float)
    p = np.linalg.norm(pv, axis=1)
    ok = r["isvd_front"].astype(bool)
    return dict(x=x, u=pv / np.maximum(p, 1e-12)[:, None], p=p, t=r["tvd_front"].astype(float),
                t0=r["time"].astype(float), ok=ok, gen=int(r["gencount"].sum()))


def fast_plane(paths):
    parts = [np.load(f) for f in sorted(paths)]
    cat = {k: np.concatenate([q[k] for q in parts]) for k in ("idx", "fate", "x", "u", "T", "t", "t0", "nfoil",
                                                               "nipa", "nopa", "path")}
    cat["p"] = np.sqrt(cat["T"] * (cat["T"] + 2 * 0.51099895))
    cat["ok"] = cat["fate"] == C.PLANE
    cat["wall"] = float(sum(float(q["wall"]) for q in parts))
    return cat


def observables(geom, field, s):
    """Plane observables for the rows that reached it."""
    ok = s["ok"]
    x, u, p = s["x"][ok], s["u"][ok], s["p"][ok]
    Bz = field(x)[:, 2]
    rmax, _ = C.helix_rmax(geom, x, u, p, Bz)
    return dict(p=p, cos=u[:, 2], r=C.radial(geom, x), rmax=rmax, tof=(s["t"] - s["t0"])[ok], t=s["t"][ok],
                dE=np.hypot(p, 0.51099895) - E_CE)


def _ratio(k1, n1, k2, n2):
    a, b = k1 / n1, k2 / n2
    ea, eb = np.sqrt(a * (1 - a) / n1), np.sqrt(b * (1 - b) / n2)
    return dict(g4=a, g4_err=ea, fast=b, fast_err=eb, ratio=b / a, ratio_err=b / a * np.hypot(ea / a, eb / b))


def compare(ref, fast_glob, geom, field, mode):
    g = g4_plane(ref)
    f = fast_plane(glob.glob(fast_glob))
    go, fo = observables(geom, field, g), observables(geom, field, f)
    out = dict(mode=mode, g4_rows=int(g["ok"].size), g4_gen=g["gen"], fast_rows=int(f["fate"].size),
               fast_fates={n: int((f["fate"] == k).sum()) for k, n in enumerate(C.FATES)},
               fast_wall_s=f["wall"], fast_us_per_e=1e6 * f["wall"] / max(f["fate"].size, 1),
               fast_crossings=dict(foil=float(f["nfoil"].mean()), ipa=float(f["nipa"].mean()),
                                   opa=float(f["nopa"].mean())), cuts={})
    for rc in RMAX_CUTS:
        gs = (go["p"] > P_MIN) & (go["rmax"] > rc)
        fs = (fo["p"] > P_MIN) & (fo["rmax"] > rc)
        c = dict(n_g4=int(gs.sum()), n_fast=int(fs.sum()))
        if mode == "stops":
            c["yield"] = _ratio(gs.sum(), g["gen"], fs.sum(), f["fate"].size)
        else:
            c["kept_fraction"] = _ratio(gs.sum(), g["ok"].size, fs.sum(), f["fate"].size)
        rng = np.random.default_rng(1)
        ng, nf = (g["gen"], f["fate"].size) if mode == "stops" else (int(gs.sum()), int(fs.sum()))
        c["energy"] = dict(norm="per generated" if mode == "stops" else "per selected",
                           g4=energy_metrics(go["dE"][gs], ng, rng), fast=energy_metrics(fo["dE"][fs], nf, rng))
        for k in ("p", "cos", "r", "rmax", "tof", "t"):
            a, b = go[k][gs], fo[k][fs]
            e = quantile_edges(a, NBINS, lo=min(a.min(), b.min()), hi=max(a.max(), b.max()))
            c2, nd = chi2(a, b, e)
            c[k] = dict(g4=summary(a), fast=summary(b), chi2=c2, ndf=nd - 1, chi2_ndf=c2 / max(nd - 1, 1))
        out["cuts"][str(rc)] = c
    if mode == "ref":
        # same start: momentum and time-of-flight differences for rows that reached the plane in both arms
        # same start, both arms at the plane inside the selection (p > P_MIN, r_max > the first cut)
        gsel = np.zeros(g["ok"].size, bool)
        gsel[np.flatnonzero(g["ok"])] = (go["p"] > P_MIN) & (go["rmax"] > RMAX_CUTS[0])
        fsel = np.zeros(f["ok"].size, bool)
        fsel[np.flatnonzero(f["ok"])] = (fo["p"] > P_MIN) & (fo["rmax"] > RMAX_CUTS[0])
        both = fsel & gsel[f["idx"]]
        j = f["idx"][both]
        dp = f["p"][both] - g["p"][j]
        dt = (f["t"] - f["t0"])[both] - (g["t"] - g["t0"])[j]
        out["paired"] = dict(n=int(both.sum()), reach_fast=float(f["ok"].mean()), sel_fast_given_g4=float(
            both.sum() / max(gsel[f["idx"]].sum(), 1)), dp=summary(dp), dtof=summary(dt))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.cecompare", description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--fast", required=True, help="glob of cechain output npz files")
    ap.add_argument("--mode", choices=("stops", "ref"), required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = compare(a.ref, a.fast, C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps), a.mode)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("fast_rows", "fast_fates", "fast_us_per_e")}, default=float))


if __name__ == "__main__":
    main()
