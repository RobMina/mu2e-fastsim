"""Compare completed G4 single-foil cases with opt-in fast mean-law arms, never rerunning a G4 reference.

Pointwise diagnostic intervals are not simultaneous confidence or accepted full-chain tolerances. G4 physical
stops and subthreshold exits remain separate; threshold-aligned comparisons collapse both to zero explicitly.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import ks_2samp, beta

from . import physics_tables as pt, rangefluct as rf, fluctstudy as fs
from .foilreport import audit, table_audit
from .foilcampaign import TOLERANCES
from .reference_audit import _identity
from .slicestudy import verify_hashes

ARMS = ("fast_local_1MeV", "fast_range_1MeV", "g4_range_unrestricted", "g4_range_cut_only", "g4_range_cut_split")


def moments(x):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) < 2 or np.any(~np.isfinite(x)):
        raise ValueError("finite vector with at least two independent samples required")
    variance = float(x.var(ddof=1))
    m4 = float(np.mean((x - x.mean()) ** 4))
    variance_se = float(np.sqrt(max(0., (m4 - (len(x) - 3) / (len(x) - 1) * variance**2) / len(x))))
    return dict(n=len(x), mean=float(x.mean()), mean_se=float(np.sqrt(variance / len(x))),
                variance=variance, variance_se=variance_se,
                quantiles=np.quantile(x, [0, .01, .1, .5, .9, .99, 1]).tolist())


def verdict(delta, se, tolerance):
    if not all(np.isfinite(v) for v in (delta, se, tolerance)) or se < 0 or tolerance <= 0:
        raise ValueError("finite delta, nonnegative SE and positive tolerance required")
    low, high = max(0., abs(delta) - 3 * se), abs(delta) + 3 * se
    return dict(delta=delta, se=se, tolerance=tolerance, abs_3se_interval=[low, high],
                state="FAIL" if low > tolerance else "PASS" if high <= tolerance else "INDETERMINATE")


def stop_interval(candidate, reference):
    """Conservative 99% difference interval from two 99.5% exact binomial intervals."""
    bounds = []
    for array in (candidate, reference):
        n, k = len(array), int(np.sum(np.asarray(array) < .1))
        bounds.append((0. if k == 0 else float(beta.ppf(.0025, k, n - k + 1)),
                       1. if k == n else float(beta.ppf(.9975, k + 1, n - k))))
    lo, hi = bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0]
    tolerance = TOLERANCES["stop_abs"]
    return dict(delta=float(np.mean(np.asarray(candidate) < .1) - np.mean(np.asarray(reference) < .1)),
                interval_99pct=[lo, hi], tolerance=tolerance,
                state="FAIL" if lo > tolerance or hi < -tolerance else "PASS" if lo >= -tolerance and hi <= tolerance else "INDETERMINATE")


def contrast(candidate, reference, entry, *, threshold_aligned=True):
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if threshold_aligned:
        c, r = np.where(c < .1, 0., c), np.where(r < .1, 0., r)
    cm, rm = moments(c), moments(r)
    delta = cm["mean"] - rm["mean"]
    se = np.hypot(cm["mean_se"], rm["mean_se"])
    mean_tolerance = max(TOLERANCES["mean_loss_floor_MeV"], TOLERANCES["mean_loss_relative"] * (entry - rm["mean"]))
    var_delta = cm["variance"] - rm["variance"]
    var_se = np.hypot(cm["variance_se"], rm["variance_se"])
    distance = float(ks_2samp(c, r).statistic)
    radius = float(np.sqrt(np.log(400.) / (2 * len(c))) + np.sqrt(np.log(400.) / (2 * len(r))))
    cdf_low, cdf_high = max(0., distance - radius), min(1., distance + radius)
    return dict(candidate=cm, reference=rm, mean_exit=verdict(delta, se, mean_tolerance),
        stop=stop_interval(c, r),
        variance=verdict(var_delta, var_se, max(TOLERANCES["variance_relative"] * rm["variance"], 1e-15)),
        cdf=dict(distance=distance, radius_99pct_dkw=radius, interval=[cdf_low, cdf_high],
                 state="FAIL" if cdf_low > TOLERANCES["cdf_abs"] else "PASS" if cdf_high <= TOLERANCES["cdf_abs"] else "INDETERMINATE"),
        threshold_aligned=threshold_aligned,
        uncertainty="Independent-event pointwise diagnostic intervals; not paired G4 histories or simultaneous acceptance")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--table-export", required=True)
    ap.add_argument("--table-reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=50000)
    ap.add_argument("--slices", nargs="+", type=int, default=[4, 16, 32])
    a = ap.parse_args(argv)
    if a.samples < 2 or any(s <= 0 for s in a.slices) or len(set(a.slices)) != len(a.slices):
        raise ValueError("at least two samples and unique positive slice counts required")
    root, out = Path(a.campaign), Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("completed campaign required")
    material = table_audit(a.table_export, a.table_reference)
    if not material["mu-"]["compatible"]:
        raise ValueError("negative-muon material equivalence gate failed")
    tables, info, paths = pt.load_target(Path(a.table_export) / "tables", material="G4_Al")
    laws = fs.laws(tables, info["energy_cuts_MeV"]["e-"], 40.)
    source_paths = [Path(__file__), Path(rf.__file__), Path(fs.__file__), Path(pt.__file__),
                    Path(__file__).with_name("foilreport.py"), Path(__file__).with_name("foilclosure.py"),
                    Path(__file__).with_name("foilcampaign.py"), Path(__file__).with_name("layerstudy.py"),
                    Path(__file__).with_name("geometry.py"), Path(__file__).with_name("reference_audit.py"),
                    Path(__file__).with_name("slicestudy.py"), Path(__file__).with_name("g4ref.py"),
                    Path(rf.eloss.__file__), Path(rf.eloss_jax.__file__), Path(rf.eloss.__file__).with_name("materials.py")]
    source_identities = [_identity(p) for p in source_paths]
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, tolerances=TOLERANCES, material_check=material, g4={}, fast={}, comparisons={}, g4_checks={})
    inputs = [_identity(root / "manifest.json")]
    starts = time.monotonic()
    per_case = {}
    for tag, item in manifest["runs"].items():
        verify_hashes([item["receipt"]], [_identity(item["receipt"]["path"])], "campaign receipt")
        inputs.append(item["receipt"])
        summary, data, _ = audit(Path(item["receipt"]["path"]).parent)
        if np.any(data["other"]):
            raise ValueError("unresolved G4 outcomes; no silent exclusion")
        np.savez_compressed(out / f"g4_{tag}.npz", **data)
        report["g4"][tag] = summary
        case_name = next(c[0] for c in manifest["cases"] if tag.startswith(c[0] + "_"))
        mode = tag[len(case_name) + 1:]
        per_case.setdefault(case_name, {})[mode] = data
    for index, (label, energy, thickness, cosine) in enumerate(manifest["cases"]):
        if label not in per_case:
            continue
        saved = per_case[label]
        for cand, base in (("loss_fine", "loss"), ("full", "loss"), ("full", "loss_fine"),
                           ("loss_finer", "loss_gridfine"), ("loss_finest", "loss_finer")):
            if cand in saved and base in saved:
                report["g4_checks"][f"{label}_{cand}_minus_{base}"] = contrast(saved[cand]["exit_MeV"], saved[base]["exit_MeV"], energy)
        for slices in a.slices:
            for arm in ARMS:
                law, mode, deltas = fs.ARMS[arm]
                result = rf.run_layer(np.array([energy]), np.array([thickness / cosine]), laws[law],
                                     slices=slices, replicas=a.samples, seed=20261026 + index,
                                     mode=mode, deltas=deltas)
                tag = f"{label}_{arm}_s{slices}"
                np.savez_compressed(out / f"fast_{tag}.npz", **result)
                outgoing = result["exit_MeV"].ravel()
                report["fast"][tag] = dict(exit=moments(outgoing), stop_fraction=float(result["stop"].mean()),
                    hard_mean_MeV=float(result["hard_applied"].mean()), hard_mean_count=float(result["hard_count"].mean()),
                    forced_fraction=float(result["forced_range_stop"].mean()),
                    continuous_cap_MeV=float(result["continuous_cap_excess"].sum()), hard_cap_MeV=float(result["hard_cap_excess"].sum()))
                for gmode, data in saved.items():
                    r = contrast(outgoing, data["exit_MeV"], energy)
                    r["g4_physical_stop_fraction"] = float(data["physical_stop"].mean())
                    r["g4_subthreshold_exit_fraction"] = float(np.mean(data["exit"] & (data["exit_MeV"] < .1)))
                    report["comparisons"][f"{tag}_vs_{gmode}"] = r
                if "loss_fine" in saved:
                    comparison = report["comparisons"][f"{tag}_vs_loss_fine"]
                    print(json.dumps(dict(tag=tag, stop=report["fast"][tag]["stop_fraction"],
                        mean=outgoing.mean(), variance=outgoing.var(ddof=1),
                        verdicts={k: comparison[k]["state"] for k in ("stop", "mean_exit", "variance", "cdf")})), flush=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    report.update(complete=True, wall_s=time.monotonic() - starts)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for item in inputs + source_identities:
        verify_hashes([item], [_identity(item["path"])], "source/input during comparison")
    (out / "manifest.json").write_text(json.dumps(dict(inputs=inputs, tables=[_identity(p) for p in paths],
        sources=source_identities, arguments=vars(a), report=_identity(out / "summary.json"),
        outputs=[_identity(p) for p in sorted(out.glob("*.npz"))]), indent=2))
    print(json.dumps(dict(complete=True, wall_s=report["wall_s"], comparisons=len(report["comparisons"]))), flush=True)


if __name__ == "__main__":
    main()
