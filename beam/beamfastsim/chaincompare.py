"""Full-chain fast vs Geant4 stopping-target metrics on the same DS-entrance muons, with errors, for several fast arms.

Metrics (per charge selection): stop fraction ratio fast/G4 with the independent-binomial error (as the deck) and the
paired error (bootstrap over muons, both arms resampled together); per-foil profile chi2/ndf as the deck
(sum (f-g)^2/(f+g) over filled bins, ndf = filled bins - 1); stops in each foil with the ratio fast/G4 and its
Poisson error; mean stop foil index and radius.
  python3 -m beamfastsim.chaincompare --g4 g4ref.npz --fast A.npz [B.npz ...] --labels a [b ...] --nfoil 49 --out x.json
"""
import argparse
import json

import numpy as np

AXIS_X = -3904.0


def metrics(g, f, nfoil, sel=None, boot=400, seed=3):
    n = len(f["fate"])
    g = {k: v[:n] for k, v in g.items()}
    sel = np.ones(n, bool) if sel is None else sel[:n]
    gs, fs = (g["foil"] >= 0) & sel, (f["fate"] == 1) & sel
    m = int(sel.sum())
    hg = np.bincount(g["foil"][gs], minlength=nfoil)[:nfoil]
    hf = np.bincount(f["foil"][fs], minlength=nfoil)[:nfoil]
    fg, ff = gs.sum() / m, fs.sum() / m
    eg, ef = np.sqrt(fg * (1 - fg) / m), np.sqrt(ff * (1 - ff) / m)
    ratio = ff / fg
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(sel)
    rb = []
    for _ in range(boot):
        i = rng.choice(idx, len(idx))
        rb.append(fs[i].mean() / gs[i].mean())
    filled = (hg + hf) > 0
    chi2 = float(((hf - hg)[filled] ** 2 / (hf + hg)[filled]).sum() / max(int(filled.sum()) - 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        foil_ratio = hf / hg
        foil_err = foil_ratio * np.sqrt(1 / np.maximum(hf, 1) + 1 / np.maximum(hg, 1))
    rg = np.hypot(g["xstop"][gs] - AXIS_X, g["ystop"][gs])
    rf = np.hypot(f["x"][fs, 0] - AXIS_X, f["x"][fs, 1])
    return dict(muons=m, stops_g4=int(gs.sum()), stops_fast=int(fs.sum()), stop_frac_g4=float(fg),
                stop_frac_fast=float(ff), ratio=float(ratio), ratio_err_indep=float(ratio * np.hypot(eg / fg, ef / ff)),
                ratio_err_paired=float(np.std(rb)), foil_chi2_ndf=chi2, ndf=int(filled.sum()) - 1,
                first_foil_g4=int(hg[0]), first_foil_fast=int(hf[0]), first_foil_ratio=float(foil_ratio[0]),
                first_foil_ratio_err=float(foil_err[0]), foil_ratio=np.nan_to_num(foil_ratio).tolist(),
                foil_ratio_err=np.nan_to_num(foil_err).tolist(), profile_g4=hg.tolist(), profile_fast=hf.tolist(),
                mean_foil_g4=float(g["foil"][gs].mean()), mean_foil_fast=float(f["foil"][fs].mean()),
                mean_r_g4=float(rg.mean()), mean_r_fast=float(rf.mean()),
                both=int((gs & fs).sum()), g4_only=int((gs & ~fs).sum()), fast_only=int((~gs & fs).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--g4", required=True)
    ap.add_argument("--fast", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--nfoil", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    labels = a.labels or a.fast
    with np.load(a.g4) as fg:
        g = {k: fg[k] for k in fg.files}
    out = {}
    for label, path in zip(labels, a.fast):
        with np.load(path) as ff:
            f = {k: ff[k] for k in ff.files if k != "meta"}
            meta = json.loads(str(ff["meta"])) if "meta" in ff.files else {}
        q = g["q"][:len(f["fate"])]
        out[label] = dict(fast=path, params=meta.get("params"), all=metrics(g, f, a.nfoil),
                          mu_minus=metrics(g, f, a.nfoil, q < 0), mu_plus=metrics(g, f, a.nfoil, q > 0))
        x = out[label]["all"]
        print(f"{label:24s} ratio {x['ratio']:.4f} +- {x['ratio_err_indep']:.4f} (paired {x['ratio_err_paired']:.4f})"
              f"  chi2/ndf {x['foil_chi2_ndf']:.2f}  foil0 {x['first_foil_fast']}/{x['first_foil_g4']} ="
              f" {x['first_foil_ratio']:.3f} +- {x['first_foil_ratio_err']:.3f}  mu+ ratio"
              f" {out[label]['mu_plus']['ratio']:.3f} +- {out[label]['mu_plus']['ratio_err_indep']:.3f}", flush=True)
    with open(a.out, "w") as fo:
        json.dump(dict(g4=a.g4, nfoil=a.nfoil, arms=out), fo, indent=1)


if __name__ == "__main__":
    main()
