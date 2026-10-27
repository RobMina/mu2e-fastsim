"""Precision single-foil cells: G4 slab references (both charges) against fast layer arms, with an endpoint contract.

Inputs are audited `beamfastsim.foilscan` campaigns (one or more directories, e.g. one per grid job); runs of the same
(scan, mode, charge) with different seed offsets are pooled as independent samples. For every monoenergetic scan
(SCANS entries with emin = emax and cmin = cmax) and charge:

* G4 internal contrasts: full_finest - loss_finest (msc), full - full_finest (stock vs converged settings);
* fast arms on the same entry (energy, path = thickness / cos):
    threshold-relative range-mean arms of `beamfastsim.rangefluct` (endpoint at 0.1 MeV, the October 25 diagnostic):
      fast_local_s4 (the legacy mean law), fast_range_s32, g4_unrestricted_s32, g4_split_s32;
    production layer arms of `beamfastsim.foilfast` (transport._layer, G4_Al slab tables for the g4 law):
      legacy_s4 (transport default), legacy_s32, range_g4_s4 / _s32 (CSDA range from zero, no path correction),
      g4path_s4 (the "g4" preset: range from zero plus the msc true-path correction);
  each scored with `foilcompare.contrast` (threshold-aligned stop, mean, variance, 99 % DKW CDF) against loss_finest,
  and the path arm also against full_finest and stock full;
* endpoint contract (production arms, which keep sub-0.1 MeV exits): physical stop = range-out inside the foil
  (G4: zero momentum in the foil), subthreshold exit = exit with 0 < T < 0.1 MeV; exact-binomial 99 % difference
  intervals against the 0.02 band, separately for each class.

  python3 -m beamfastsim.foilcells --campaign DIR [DIR ...] --slab-tables EXPORT/tables --out DIR [--samples N]
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import beta

from . import fluctstudy as fs, physics_tables as pt, rangefluct as rf
from .foilcompare import contrast, moments
from .foilscan import SCANS
from .reference_audit import _identity
from .slicestudy import verify_hashes

BAND = .02
RF_ARMS = {"fast_local_s4": ("fast_default", "local", False, 4), "fast_range_s32": ("fast_default", "range", False, 32),
           "g4_unrestricted_s32": ("g4_full", "range", False, 32), "g4_split_s32": ("g4_cut", "range", True, 32)}
LAYER_ARMS = {"legacy_s4": dict(physics="legacy", n_slices=4), "legacy_s32": dict(physics="legacy", n_slices=32),
              "range_g4_s4": dict(physics="g4", msc_path=False, n_slices=4),
              "range_g4_s32": dict(physics="g4", msc_path=False, n_slices=32),
              "g4path_s4": dict(physics="g4", n_slices=4)}
REFERENCES = {"loss_finest": "converged loss-only (no msc)", "full_finest": "converged full physics",
              "full": "stock settings = chain reference"}


def cells():
    return {k: v for k, v in SCANS.items() if v[1] == v[2] and v[3] == v[4]}


def binomial_difference(k1, n1, k2, n2, band=BAND):
    """Conservative 99 % interval of p1 - p2 from two 99.5 % exact (Clopper-Pearson) intervals, and its verdict."""
    def ci(k, n):
        return (0. if k == 0 else float(beta.ppf(.0025, k, n - k + 1)), 1. if k == n else float(beta.ppf(.9975, k + 1, n - k)))
    (a0, a1), (b0, b1) = ci(int(k1), int(n1)), ci(int(k2), int(n2))
    lo, hi = a0 - b1, a1 - b0
    return dict(candidate=k1 / n1, reference=k2 / n2, delta=k1 / n1 - k2 / n2, interval_99pct=[lo, hi], band=band,
                state="FAIL" if lo > band or hi < -band else "PASS" if lo >= -band and hi <= band else "INDETERMINATE")


def endpoint(raw, g4):
    """Physical-stop and subthreshold-exit classes of a production arm (raw outgoing T) against a G4 sample."""
    raw = np.asarray(raw, float).ravel()
    n, m = len(raw), len(g4["exit_MeV"])
    sub_g4 = int(np.sum(g4["exit"] & (g4["exit_MeV"] < .1)))
    return dict(physical_stop=binomial_difference(np.sum(raw <= 0), n, np.sum(g4["physical_stop"]), m),
                subthreshold_exit=binomial_difference(np.sum((raw > 0) & (raw < .1)), n, sub_g4, m),
                threshold_stop=binomial_difference(np.sum(raw < .1), n, np.sum(g4["exit_MeV"] < .1), m),
                subthreshold_exit_MeV=dict(fast=moments(raw[(raw > 0) & (raw < .1)]) if np.sum((raw > 0) & (raw < .1)) > 1 else None,
                                           g4=moments(g4["exit_MeV"][g4["exit"] & (g4["exit_MeV"] < .1)]) if sub_g4 > 1 else None))


def load_runs(campaigns):
    """{(scan, mode, pdg): pooled G4 arrays} from audited campaigns, hash-checked, no unresolved outcomes."""
    pooled, inputs, seen = {}, [], set()
    for root in campaigns:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if not manifest.get("complete"):
            raise ValueError(f"incomplete campaign {root}")
        for tag, run in manifest["runs"].items():
            path = root / tag / "scan.npz"
            item = dict(run["scan_npz"], path=str(path))
            verify_hashes([item], [_identity(path)], f"scan {tag}")
            with np.load(path) as f:
                g = {k: f[k] for k in f.files}
            if np.any(g["other"]):
                raise ValueError(f"unresolved G4 outcomes in {tag}")
            key = (run["scan"], run["mode"], int(run["pdg"]))
            sample = (*key, run.get("seed_offset", 0))
            if sample in seen:
                raise ValueError(f"duplicate response sample {sample}")
            seen.add(sample)
            pooled.setdefault(key, []).append(g)
            inputs.append(dict(tag=tag, campaign=str(root), seed_offset=run.get("seed_offset", 0), **item))
    out = {k: {name: np.concatenate([g[name] for g in v]) for name in v[0]} for k, v in pooled.items()}
    return out, inputs


def g4_summary(g):
    n = len(g["exit_MeV"])
    sub = g["exit"] & (g["exit_MeV"] < .1)
    return dict(n=n, threshold_stop=float(np.mean(g["exit_MeV"] < .1)), physical_stop=float(g["physical_stop"].mean()),
                subthreshold_exit=float(sub.mean()), back_exit=float(np.mean(g["exit"] & (g["exit_z_mm"] < 0))),
                exit=moments(np.where(g["exit_MeV"] < .1, 0., g["exit_MeV"])))


def main(argv=None):
    from . import foilfast, transport
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", nargs="+", required=True)
    ap.add_argument("--slab-tables", required=True, help="G4_Al slab export tables/ (both charges)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=1000000)
    ap.add_argument("--charges", type=int, nargs="+", choices=[13, -13], default=[13, -13])
    ap.add_argument("--cells", nargs="+", choices=list(cells()), default=list(cells()))
    a = ap.parse_args(argv)
    if a.samples <= 1 or len(set(a.charges)) != len(a.charges):
        ap.error("samples > 1 and unique charges required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runs, inputs = load_runs(a.campaign)
    missing = [(cell, mode, pdg) for cell in a.cells for pdg in a.charges for mode in REFERENCES
               if (cell, mode, pdg) not in runs]
    if missing:
        raise ValueError(f"missing required references: {missing}")
    from . import g4tables
    table_checks = {"slab": g4tables.compare(a.slab_tables, g4tables.SLAB, "G4_Al", require_audited_export=True)}
    if not table_checks["slab"]["all_arrays_equal"]:
        raise ValueError("shipped slab tables differ from the audited export")
    laws = {}
    for pdg, name in ((13, "mu-"), (-13, "mu+")):
        tables, info, paths = pt.load_target(a.slab_tables, material="G4_Al", particle=name)
        laws[pdg] = fs.laws(tables, info["energy_cuts_MeV"]["e-"], 40.)
    report = dict(complete=False, band=BAND, references=REFERENCES, rf_arms=RF_ARMS, layer_arms=LAYER_ARMS,
                  samples=a.samples, charges=a.charges, inputs=inputs, table_checks=table_checks, cells={})
    for ci, cell in enumerate(a.cells):
        thickness, energy, _, cosine, _ = SCANS[cell]
        for pdg in a.charges:
            ref = {mode: runs[(cell, mode, pdg)] for mode in REFERENCES if (cell, mode, pdg) in runs}
            q = "mum" if pdg == 13 else "mup"
            r = dict(g4={m: g4_summary(g) for m, g in ref.items()}, g4_contrasts={}, arms={})
            for cand, base in (("full_finest", "loss_finest"), ("full", "full_finest")):
                if cand in ref and base in ref:
                    r["g4_contrasts"][f"{cand}_minus_{base}"] = contrast(ref[cand]["exit_MeV"], ref[base]["exit_MeV"], energy)
            seed = 20261027 + 10 * ci + (0 if pdg == 13 else 5)
            for arm, (law, mode, deltas, slices) in RF_ARMS.items():
                res = rf.run_layer(np.array([energy]), np.array([thickness / cosine]), laws[pdg][law], slices=slices,
                                   replicas=a.samples, seed=seed, mode=mode, deltas=deltas)
                x = res["exit_MeV"].ravel()
                saved = out / f"{cell}_{q}_{arm}.npz"
                np.savez_compressed(saved, **res)
                r["arms"][arm] = dict(kind="rangefluct", output=_identity(saved), vs={"loss_finest": contrast(x, ref["loss_finest"]["exit_MeV"], energy)},
                                      forced_fraction=float(res["forced_range_stop"].mean()))
            for arm, kw in LAYER_ARMS.items():
                prm = transport.params(**kw, table="slab")
                _, _, tpath, raw = foilfast.respond(np.array([energy]), np.array([cosine]), thickness, prm,
                                                    charge=-1 if pdg == 13 else 1, replicas=a.samples, seed=seed, raw=True)
                x = np.where(raw < .1, 0., raw).ravel()
                vs = {m: contrast(x, ref[m]["exit_MeV"], energy) for m in ref
                      if m == "loss_finest" or prm.msc_path or arm == "legacy_s4"}
                ends = {m: endpoint(raw, ref[m]) for m in vs}
                saved = out / f"{cell}_{q}_{arm}.npz"
                np.savez_compressed(saved, raw_exit_MeV=raw, threshold_exit_MeV=x, effective_path_mm=tpath)
                r["arms"][arm] = dict(kind="layer", output=_identity(saved), params=prm._asdict(), vs=vs, endpoint=ends,
                                      true_over_geometric_path=float(np.mean(tpath) * cosine / thickness))
            report["cells"][f"{cell}_{q}"] = r
            line = {arm: {m: (v["stop"]["state"][0], v["mean_exit"]["state"][0], v["variance"]["state"][0], v["cdf"]["state"][0])
                          for m, v in x["vs"].items()} for arm, x in r["arms"].items()}
            print(f"{cell}_{q}", json.dumps(dict(g4={m: round(v["threshold_stop"], 4) for m, v in r["g4"].items()}, arms=line)),
                  flush=True)
            (out / "summary.json").write_text(json.dumps(report, indent=1, allow_nan=False))
    from mu2efastsim import eloss, eloss_jax
    report["core_sources"] = [_identity(eloss.__file__), _identity(eloss_jax.__file__)]
    report["table_npz"] = _identity(g4tables.SLAB)
    report["scope"] = "Selected-cell pilot only; endpoint, distribution and full-chain acceptance remain separate gates"
    report.update(complete=True, wall_s=time.monotonic() - started,
                  sources=[_identity(Path(__file__).with_name(n)) for n in ("foilcells.py", "foilfast.py", "layerphys.py",
                           "transport.py", "rangefluct.py", "fluctstudy.py", "foilcompare.py", "foilscan.py", "g4tables.py")])
    (out / "summary.json").write_text(json.dumps(report, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
