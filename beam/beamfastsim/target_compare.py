"""Paired electron-marginal VD 13 comparison for DIO/internal RPC (not external conversion).

This compares transport on identical *kept* G4 starts, not an unbiased source
yield. Production filtering and RPC pair selection remain separate contracts.
Paired per-row differences use cecompare's p/rmax acceptance proxy (implication unproven);
histogram differences use paired source-file clusters to retain correlations.
"""
import argparse
import glob
import json

import numpy as np

from . import cechain as C
from . import cecompare as CC


def paired_hist(a, b, sa, sb, groups, edges):
    """Full row-paired count differences, with per-file cluster variance, including flow bins."""
    nbin = len(edges) + 1
    _, gi = np.unique(groups, return_inverse=True)
    n = int(gi.max()) + 1
    ha = np.bincount(gi[sa] * nbin + np.searchsorted(edges, a[sa], side="right"),
                     minlength=n * nbin).reshape(n, nbin)
    hb = np.bincount(gi[sb] * nbin + np.searchsorted(edges, b[sb], side="right"),
                     minlength=n * nbin).reshape(n, nbin)
    d = (hb - ha).astype(float)
    delta = d.sum(axis=0)
    variance = n * d.var(axis=0, ddof=1) if n > 1 else None
    return dict(edges=edges.tolist(), base=ha.sum(axis=0).tolist(), cand=hb.sum(axis=0).tolist(),
                diff=delta.tolist(), diff_err=np.sqrt(variance).tolist() if variance is not None else None, clusters=n,
                variance_method="N times sample variance of per-file differences; no chi2 claimed for sparse bins")


def clustered_mean(values, groups):
    """Mean and file-cluster sandwich error, allowing unequal rows per file."""
    values = np.asarray(values, float)
    _, gi = np.unique(groups, return_inverse=True)
    if not values.size:
        return dict(n=0, mean=None, err=None, clusters=0)
    n = int(gi.max()) + 1
    mean = values.mean()
    influence = np.bincount(gi, weights=values - mean, minlength=n)
    err = np.sqrt(n / (n - 1) * np.sum(influence ** 2)) / values.size if n > 1 else None
    return dict(n=int(values.size), mean=float(mean), err=float(err) if err is not None else None, clusters=n)


def migration_report(sa, sb, g4_ok, fast_fate, groups):
    """Separate reach losses from boundary migration without assigning a physics cause."""
    both = sa & sb
    lost = sa & ~sb
    gained = ~sa & sb
    return dict(both=int(both.sum()), base_only=int(lost.sum()), cand_only=int(gained.sum()),
        neither=int((~sa & ~sb).sum()), base_only_at_fast_plane=int((lost & (fast_fate == C.PLANE)).sum()),
        cand_only_without_g4_plane=int((gained & ~g4_ok).sum()),
        base_only_fates={name: int((lost & (fast_fate == i)).sum()) for i, name in enumerate(C.FATES)},
        clustered_diff=clustered_mean(sb.astype(float) - sa, groups))


def compare(ref, fast_glob, geom, field):
    g = CC.g4_plane(ref)
    f = CC.fast_plane(glob.glob(fast_glob))
    with np.load(ref) as r:
        idx = f["idx"].astype(int)
        if np.unique(idx).size != idx.size or idx.size < 1 or np.any(idx < 0) or np.any(idx >= len(r["x"])):
            raise ValueError("fast rows must identify unique valid reference rows")
        parts = [np.load(p) for p in sorted(glob.glob(fast_glob))]
        x0 = np.concatenate([p["x0"] for p in parts])
        u0 = np.concatenate([p["u0"] for p in parts])
        p0 = np.concatenate([p["p0"] for p in parts])
        expected_x = np.column_stack([r[k][idx] for k in ("x", "y", "z")])
        expected_p = np.column_stack([r[k][idx] for k in ("px", "py", "pz")])
        if not (np.allclose(x0, expected_x, rtol=0, atol=1e-8)
                and np.allclose(p0[:, None] * u0, expected_p, rtol=1e-10, atol=1e-8)
                and np.allclose(f["t0"], r["time"][idx], rtol=0, atol=1e-8)):
            raise ValueError("fast transport did not use the reference starts")
        groups = r["ifile"][idx]
        for p in parts:
            p.close()
    n = idx.size
    obs = []
    for state in (g, f):
        o = CC.observables(geom, field, state)
        full = {}
        for k, v in o.items():
            full[k] = np.full(state["ok"].size, np.nan)
            full[k][state["ok"]] = v
        obs.append(full)
    a = {k: v[idx] for k, v in obs[0].items()}
    b = obs[1]
    out = dict(mode="kept-reference-electron-marginal", rows=n, g4_generated=g["gen"],
               generated_yield_validated=False, start_identity_verified=True, fast_wall_s=f["wall"],
               fast_electrons_per_s=n / f["wall"] if f["wall"] > 0 else None, cuts={})
    for rc in CC.RMAX_CUTS:
        sa = g["ok"][idx] & (a["p"] > CC.P_MIN) & (a["rmax"] > rc)
        sb = f["ok"] & (b["p"] > CC.P_MIN) & (b["rmax"] > rc)
        both = sa & sb
        c = dict(kept_yield=CC._paired_mean(sa, sb), same_start={}, histograms={},
                 migration=migration_report(sa, sb, g["ok"][idx], f["fate"], groups))
        lost_plane = sa & ~sb & f["ok"]
        c["migration"]["base_only_plane_cut_failures"] = dict(
            momentum_only=int((lost_plane & (b["p"] <= CC.P_MIN) & (b["rmax"] > rc)).sum()),
            radius_only=int((lost_plane & (b["p"] > CC.P_MIN) & (b["rmax"] <= rc)).sum()),
            both=int((lost_plane & (b["p"] <= CC.P_MIN) & (b["rmax"] <= rc)).sum()))
        c["symmetric_interior"] = []
        for margin in (0., 25., 50., 100.):
            interior = (g["ok"][idx] & f["ok"] & (a["p"] > 70.) & (b["p"] > 70.)
                        & (a["rmax"] > rc + margin) & (b["rmax"] > rc + margin))
            c["symmetric_interior"].append(dict(radius_margin_mm=margin, n=int(interior.sum()),
                interpretation="both-arm selected shape diagnostic, never an acceptance or yield test",
                dp=clustered_mean((b["p"] - a["p"])[interior], groups[interior]),
                dtof=clustered_mean((b["tof"] - a["tof"])[interior], groups[interior])))
        c["kept_yield"]["clustered_diff"] = clustered_mean(sb.astype(float) - sa, groups)
        ratio = c["kept_yield"]["ratio"]
        influence = clustered_mean(sb.astype(float) - ratio * sa, groups) if np.isfinite(ratio) else None
        c["kept_yield"]["clustered_ratio_err"] = (influence["err"] / sa.mean()
            if influence is not None and influence["err"] is not None else None)
        c["birth_momentum_bands"] = []
        for lo, hi in zip((0., 60., 80., 95., 105.), (60., 80., 95., 105., np.inf)):
            band = (p0 >= lo) & (p0 < hi)
            if not band.any():
                continue
            c["birth_momentum_bands"].append(dict(lo=lo, hi=hi if np.isfinite(hi) else None,
                starts=int(band.sum()), base_selected=int((sa & band).sum()), cand_selected=int((sb & band).sum()),
                yield_diff=clustered_mean((sb.astype(float) - sa)[band], groups[band]),
                paired_dp=clustered_mean((b["p"] - a["p"])[both & band], groups[both & band])))
        if not np.isfinite(c["kept_yield"]["ratio"]):
            c["kept_yield"]["ratio"] = c["kept_yield"]["ratio_err"] = None
        for k in ("p", "cos", "r", "rmax", "tof", "t"):
            diff = b[k][both] - a[k][both]
            c["same_start"][k] = clustered_mean(diff, groups[both])
            selected = np.r_[a[k][sa], b[k][sb]]
            if not selected.size:
                continue
            lo, hi = float(selected.min()), float(selected.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, 41)
            c["histograms"][k] = paired_hist(a[k], b[k], sa, sb, groups, edges)
        out["cuts"][str(rc)] = c
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--fast", required=True)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_common.txt")
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = compare(a.ref, a.fast, C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps))
    with open(a.out, "w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "cuts"}))
    for rc, c in report["cuts"].items():
        print(json.dumps(dict(rmax_cut=rc, kept_yield=c["kept_yield"], same_start=c["same_start"])))


if __name__ == "__main__":
    main()
