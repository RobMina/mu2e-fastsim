"""Single-foil stopping-edge scans: G4 slabs with a gun flat in momentum and cos(theta), and the fast layer response.

campaign (GPVM, SimJob MDC2025ac shell): for each scan (foil thickness, kinetic-energy range, cos range), charge and
G4 mode, one beamfastsim.foilclosure job; each finished job is audited at once (foilreport.audit) into scan.npz.
compare (anywhere): the production layer (beamfastsim.foilfast) on the same G4 entries, for each fast physics arm;
stop probability (exit below 0.1 MeV) in bins of entry energy and cos(theta), binomial errors, chi2/ndf, and the
edge energy T50 (50 % stop) per cos bin by linear interpolation, with bootstrap errors.
  python3 -m beamfastsim.foilscan campaign --out DIR [--scans ...] [--modes ...] [--charges 13 -13] [--events N]
  python3 -m beamfastsim.foilscan compare --campaign DIR [DIR ...] --out DIR [--replicas R]
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

# scan: (thickness mm, kinetic energy min, max MeV, cos min, max); flat in momentum and cos(theta). Cells (emin = emax,
# cmin = cmax) are the monoenergetic single-foil pilot cases of beamfastsim.foilcampaign.
SCANS = {"t1056": (0.1056, 1.0, 2.6, 0.5, 1.0), "t2": (0.2, 1.5, 3.6, 0.5, 1.0), "t4": (0.4, 2.2, 5.0, 0.5, 1.0),
         "nom_edge": (0.1056, 1.4, 1.4, 1.0, 1.0), "moved_edge": (0.2, 2.02, 2.02, 1.0, 1.0),
         "nom_5": (0.1056, 5.0, 5.0, 1.0, 1.0), "moved_5": (0.2, 5.0, 5.0, 1.0, 1.0),
         "moved_20": (0.2, 20.0, 20.0, 1.0, 1.0), "oblique_edge": (0.2, 2.3, 2.3, 0.8, 0.8)}
COS = (0.5, 1.0)
# mode: (no_msc, step_fraction, final_range, bins, default events). Stock = the chain reference settings (7 bins per
# decade, stock step function); finest = the converged pilot refinement (168 bins, 0.005 / 0.00005 mm).
MODES = {"full": (False, None, .001, None, 100000), "full_fine": (False, .05, .001, None, 50000),
         "loss_fine": (True, .05, .001, None, 50000), "full_finest": (False, .005, .00005, 168, 20000),
         "loss": (True, None, .001, None, 50000), "loss_finest": (True, .005, .00005, 168, 50000)}
COS_EDGES = np.array([0.5, 0.625, 0.75, 0.875, 1.0])
FAST_ARMS = {"legacy": dict(physics="legacy"),
             "legacy_s32": dict(physics="legacy", n_slices=32),
             "range_g4_nopath": dict(physics="g4", msc_path=False, table="slab"),
             "g4": dict(physics="g4", table="slab")}                 # slab tables: the G4 reference is a G4_Al slab


def campaign(a):
    from .foilclosure import run_case
    from .foilreport import audit
    from .reference_audit import _identity
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(complete=False, scans={k: SCANS[k] for k in a.scans}, cos=COS, modes={m: MODES[m] for m in a.modes},
                    charges=a.charges, runs={}, sources=[_identity(Path(__file__))])
    started = time.monotonic()
    for scan in a.scans:
        i = list(SCANS).index(scan)                         # seed offset independent of the requested subset
        thickness, emin, emax, cmin, cmax = SCANS[scan]
        for mode in a.modes:
            no_msc, frac, final, bins, events = MODES[mode]
            j = list(MODES).index(mode)
            for pdg in a.charges:
                k = 0 if pdg == 13 else 1
                tag = f"{scan}_{mode}_{'mum' if pdg == 13 else 'mup'}" + (f"_r{a.seed_offset}" if a.seed_offset else "")
                if time.monotonic() - started > a.budget_seconds:
                    manifest["unstarted_from"] = tag
                    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                    raise SystemExit(2)
                n = a.events or int(events * a.events_scale)
                run_case(out / tag, energy=emin, energy_max=emax, thickness=thickness, cosine=cmax,
                         cosine_min=cmin, pdg=pdg, events=n, seed=2026102700 + 100 * i + 10 * j + k + 100000 * a.seed_offset,
                         no_msc=no_msc, step_fraction=frac, final_range=final, bins=bins)
                report, store, _ = audit(out / tag)
                np.savez_compressed(out / tag / "scan.npz", **store)
                if a.drop_root:
                    (out / tag / "foil.root").unlink()       # hash stays in receipt.json; scan.npz is the audited product
                manifest["runs"][tag] = dict(scan=scan, mode=mode, pdg=pdg, thickness_mm=thickness, seed_offset=a.seed_offset,
                                             receipt=_identity(out / tag / "receipt.json"),
                                             scan_npz=_identity(out / tag / "scan.npz"),
                                             census={x: report[x] for x in ("primaries", "exits", "physical_stops",
                                                                           "threshold_stops", "back_exits",
                                                                           "foil_steps_quantiles")},
                                             wall_s=report["wall_s"])
                manifest["wall_s"] = time.monotonic() - started
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
                print(json.dumps(dict(tag=tag, **manifest["runs"][tag]["census"], wall_s=report["wall_s"])), flush=True)
    manifest["complete"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))


def _bins(scan):
    _, emin, emax, _, _ = SCANS[scan]
    return np.linspace(emin, emax, int(round((emax - emin) / 0.05)) + 1)


def stop_table(entry, cosine, stop, e_edges, weight=1.0):
    """Stops / entries per (energy, cos) bin; weight scales counts (1/replicas for fast arms)."""
    h = lambda w: np.histogram2d(entry, cosine, bins=(e_edges, COS_EDGES), weights=w)[0]
    return h(stop.astype(float)) * weight, h(np.ones(len(entry))) * weight


def t50(e_edges, k, n):
    """Edge energy per cos bin where the stop probability crosses 0.5 (linear interpolation), nan if absent."""
    centres = 0.5 * (e_edges[1:] + e_edges[:-1])
    out = np.full(n.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = k / n
    for c in range(n.shape[1]):
        y = f[:, c]
        ok = np.flatnonzero((y[:-1] >= 0.5) & (y[1:] < 0.5))
        if len(ok):
            i = ok[-1]
            out[c] = centres[i] + (y[i] - 0.5) / (y[i] - y[i + 1]) * (centres[i + 1] - centres[i])
    return out


def compare_run(g, scan, thickness, pdg, replicas, boot=200, seed=5):
    from . import foilfast, transport
    e_edges = _bins(scan)
    entry, cosine = g["entry_MeV"], g["entry_cos"]
    gstop = g["exit_MeV"] < 0.1
    kg, ng = stop_table(entry, cosine, gstop, e_edges)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(entry), len(entry)) for _ in range(boot)]
    t50_g = t50(e_edges, kg, ng)
    t50_gb = np.array([t50(e_edges, *stop_table(entry[i], cosine[i], gstop[i], e_edges)) for i in idx])
    res = dict(g4=dict(n=int(len(entry)), stop_fraction=float(gstop.mean()),
                       physical_stop_fraction=float(g["physical_stop"].mean()), t50_MeV=t50_g.tolist(),
                       subthreshold_exit_fraction=float(np.mean(g["exit"] & (g["exit_MeV"] < .1))),
                       t50_err_MeV=np.nanstd(t50_gb, axis=0).tolist()), fast={})
    for arm, kw in FAST_ARMS.items():
        prm = transport.params(**kw)
        ex, st, tp, raw = foilfast.respond(entry, cosine, thickness, prm, charge=-1 if pdg == 13 else 1,
                                           replicas=replicas, seed=seed, raw=True)
        kf, nf = stop_table(np.tile(entry, replicas), np.tile(cosine, replicas), st.ravel(), e_edges, 1.0 / replicas)
        ok = (ng > 0)
        # binomial variance of each arm's bin fraction (fast: replicas of the same entries, conservative)
        fg = np.where(ok, kg / np.maximum(ng, 1), 0)
        ff = np.where(ok, kf / np.maximum(nf, 1), 0)
        var = fg * (1 - fg) / np.maximum(ng, 1) + ff * (1 - ff) / np.maximum(ng * replicas, 1)
        use = ok & (var > 0)
        chi2 = float(np.sum((ff - fg)[use] ** 2 / var[use]))
        t50_f = t50(e_edges, kf, nf)
        t50_fb = np.array([t50(e_edges, *stop_table(np.tile(entry[i], replicas), np.tile(cosine[i], replicas),
                                                    st[:, i].ravel(), e_edges, 1.0 / replicas)) for i in idx[:50]])
        dstop = float(st.mean() - gstop.mean())
        se = float(np.sqrt(gstop.mean() * (1 - gstop.mean()) / len(entry)
                           + st.mean() * (1 - st.mean()) / (len(entry) * replicas)))
        surv_g, surv_f = g["exit_MeV"][~gstop], ex[~st]
        res["fast"][arm] = dict(params=prm._asdict(), stop_fraction=float(st.mean()), stop_delta=dstop,
                                stop_delta_err=se, chi2=chi2, ndf=int(use.sum()),
                                t50_MeV=t50_f.tolist(), t50_err_MeV=np.nanstd(t50_fb, axis=0).tolist(),
                                t50_delta_MeV=(t50_f - t50_g).tolist(),
                                survivor_mean_exit_delta_MeV=float(surv_f.mean() - surv_g.mean()),
                                true_over_geometric_path_mean=float(np.mean(tp * cosine[None, :] / thickness)),
                                physical_stop_fraction=float(np.mean(raw <= 0)),
                                subthreshold_exit_fraction=float(np.mean((raw > 0) & (raw < .1))),
                                table=dict(k=kf.tolist(), n=nf.tolist()))
    res["g4"]["table"] = dict(k=kg.tolist(), n=ng.tolist(), e_edges=e_edges.tolist(), cos_edges=COS_EDGES.tolist())
    return res


def compare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(fast_arms=FAST_ARMS, replicas=a.replicas, runs={})
    for root in a.campaign:
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        for tag, run in manifest["runs"].items():
            emin, emax = SCANS[run["scan"]][1:3]
            if emin == emax:
                continue                                    # monoenergetic cells: beamfastsim.foilcells
            with np.load(Path(root) / tag / "scan.npz") as f:
                g = {k: f[k] for k in f.files}
            if np.any(g["other"]):
                raise ValueError(f"unresolved G4 outcomes in {tag}")
            r = compare_run(g, run["scan"], run["thickness_mm"], run["pdg"], a.replicas)
            r.update(scan=run["scan"], mode=run["mode"], pdg=run["pdg"], thickness_mm=run["thickness_mm"])
            report["runs"][tag] = r
            line = {arm: (round(v["stop_delta"], 4), round(v["stop_delta_err"], 4), round(v["chi2"] / max(v["ndf"], 1), 2))
                    for arm, v in r["fast"].items()}
            print(tag, f"G4 stop {r['g4']['stop_fraction']:.4f}", json.dumps(line), flush=True)
            (out / "summary.json").write_text(json.dumps(report, indent=1, allow_nan=True))
    (out / "summary.json").write_text(json.dumps(report, indent=1, allow_nan=True))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("campaign")
    s.add_argument("--out", required=True)
    s.add_argument("--scans", nargs="+", choices=list(SCANS), default=list(SCANS))
    s.add_argument("--modes", nargs="+", choices=list(MODES), default=["full", "full_fine", "loss_fine"])
    s.add_argument("--charges", nargs="+", type=int, choices=[13, -13], default=[13, -13])
    s.add_argument("--events-scale", type=float, default=1.0)
    s.add_argument("--events", type=int, help="events per run (overrides the mode default and --events-scale)")
    s.add_argument("--seed-offset", type=int, default=0, help="independent replica of the same runs (tag suffix _rN)")
    s.add_argument("--budget-seconds", type=float, default=5400.)
    s.add_argument("--drop-root", action="store_true", help="delete each foil.root after its audit (grid workers)")
    s = sp.add_parser("compare")
    s.add_argument("--campaign", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--replicas", type=int, default=4)
    a = ap.parse_args(argv)
    {"campaign": campaign, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
