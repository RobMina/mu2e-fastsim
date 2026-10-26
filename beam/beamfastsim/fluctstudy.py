"""Saved-negative-entry fluctuation study with separated numerical, mean-law and hard-electron arms.

Not transport or G4 stochastic closure. Each replica is conditional on the same baseline-fast entrance rows.
See beam/README.md for the continuous/discrete contract and threshold exhaustion convention.
"""
import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np
from scipy.integrate import cumulative_trapezoid

from mu2efastsim import eloss, eloss_jax
from . import g4ref, geometry, layerstudy as ls, physics_tables as pt, rangefluct as rf
from .convergence import cluster_mean
from .rangestudy import negative_entries
from .reference_audit import _identity
from .slicestudy import CORE_FILES, verify_hashes


ARMS = {
    "fast_local_1MeV": ("fast_default", "local", False),
    "fast_range_1MeV": ("fast_default", "range", False),
    "fast_range_unrestricted": ("fast_full", "range", False),
    "fast_range_cut_only": ("fast_cut", "range", False),
    "fast_range_cut_split": ("fast_cut", "range", True),
    "g4_range_unrestricted": ("g4_full", "range", False),
    "g4_range_cut_only": ("g4_cut", "range", False),
    "g4_range_cut_split": ("g4_cut", "range", True),
}
PAIRS = (("fast_range_1MeV", "fast_local_1MeV"),
         ("g4_range_unrestricted", "fast_range_unrestricted"),
         ("fast_range_cut_split", "fast_range_cut_only"),
         ("g4_range_cut_split", "g4_range_cut_only"),
         ("fast_range_cut_split", "fast_range_unrestricted"),
         ("g4_range_cut_split", "g4_range_unrestricted"))


def laws(tables, cut, maximum, points=65537):
    energy = np.geomspace(rf.THRESHOLD, maximum, points)
    power = {name: eloss.dedx_restricted(geometry.STOPPING_TARGET_AL, ls.momentum(energy), tc)
             for name, tc in (("fast_default", 1.), ("fast_full", 1e9), ("fast_cut", cut))}
    power.update(g4_full=pt.interpolate(tables["DEDXnr"], energy), g4_cut=pt.interpolate(tables["DEDX"], energy))
    cuts = dict(fast_default=1., fast_full=1e9, fast_cut=cut, g4_full=1e9, g4_cut=cut)
    return {name: rf.MeanLaw(energy, s, cumulative_trapezoid(1 / s, energy, initial=0), cuts[name])
            for name, s in power.items()}


def mc_summary(values):
    """Conditional response-replica mean/SE, not input or reference uncertainty."""
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()),
                conditional_mc_se=float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)


def summarize(result, event_ids):
    stop, outgoing = result["stop"], result["exit_MeV"]
    report = dict(expected_stops=mc_summary(stop.sum(axis=1)),
                  expected_stop_fraction_event_cluster=cluster_mean(stop.mean(axis=0), event_ids),
                  exit_MeV_quantiles_all=ls.quantiles(outgoing.ravel()),
                  exit_MeV_quantiles_survivors=ls.quantiles(outgoing[~stop]),
                  mean_exit_MeV=mc_summary(outgoing.mean(axis=1)),
                  expected_forced_stops=mc_summary(result["forced_range_stop"].sum(axis=1)),
                  expected_hard_count=mc_summary(result["hard_count"].sum(axis=1)), energy_budget={})
    for name in ("continuous_applied", "hard_requested", "hard_applied", "continuous_cap_excess",
                 "hard_cap_excess", "threshold_residual"):
        report["energy_budget"][name] = dict(mean_MeV_per_entry=float(result[name].mean()),
            max_MeV=float(result[name].max()), nonzero_entry_replica_pairs=int(np.count_nonzero(result[name])))
    return report


def paired(candidate, baseline):
    if candidate["stop"].shape != baseline["stop"].shape:
        raise ValueError("paired arrays must have identical replica/entry layout")
    difference = candidate["stop"].astype(float) - baseline["stop"]
    return dict(stop_count_delta=mc_summary(difference.sum(axis=1)),
                mean_exit_MeV_delta=mc_summary((candidate["exit_MeV"] - baseline["exit_MeV"]).mean(axis=1)))


def load_inputs(layer_root, table_root):
    lm = json.loads((layer_root / "manifest.json").read_text())
    tm = json.loads((table_root / "manifest.json").read_text())
    identities = [_identity(layer_root / "manifest.json"), _identity(table_root / "manifest.json"),
                  _identity(table_root / "summary.json")]
    identities.extend(lm[k] for k in ("study", "summary", "history", "frozen"))
    study = json.loads(Path(lm["study"]["path"]).read_text())
    summary = json.loads(Path(lm["summary"]["path"]).read_text())
    if not summary["complete"] or not all(summary["replay_exact"].values()):
        raise ValueError("completed exact saved-layer replay required")
    identities.append(study["input"])
    identities.extend(tm["artifacts"])
    for item in identities:
        verify_hashes([item], [_identity(item["path"])], "saved input")
    core_dir = Path(eloss.__file__).parent
    verify_hashes(study["core"], [_identity(core_dir / p) for p in CORE_FILES], "unchanged core physics")
    table_path = next(Path(p["path"]).parent for p in tm["artifacts"] if Path(p["path"]).name == "couple.dat")
    tables, info, consumed = pt.load_target(table_path)
    recorded = {str(Path(p["path"]).resolve()) for p in tm["artifacts"]}
    if not {str(p.resolve()) for p in consumed} <= recorded:
        raise ValueError("table manifest omits consumed files")
    if info != json.loads((table_root / "summary.json").read_text())["couple"] or info["particle"] != "mu-":
        raise ValueError("negative-muon table/couple mismatch")
    g = g4ref.load(study["input"]["path"])
    frozen, charges = negative_entries(g4ref.load(lm["frozen"]["path"]), g)
    return frozen, g, charges, tables, info, identities


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer-report", required=True)
    ap.add_argument("--table-report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicas", type=int, default=64)
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--slices", nargs="+", type=int, default=[4, 16, 32])
    ap.add_argument("--budget-seconds", type=float, default=1200.)
    a = ap.parse_args(argv)
    if a.replicas < 2 or a.seed < 0 or any(s <= 0 for s in a.slices) or len(set(a.slices)) != len(a.slices):
        raise ValueError("at least two replicas, nonnegative seed and unique positive slices required")
    if not np.isfinite(a.budget_seconds) or a.budget_seconds <= 0:
        raise ValueError("positive finite budget required")
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    frozen, g, charges, tables, info, inputs = load_inputs(Path(a.layer_report), Path(a.table_report))
    maximum = max(float(frozen["entry_MeV"].max()) * 1.01, 10.)
    model = laws(tables, info["energy_cuts_MeV"]["e-"], maximum)
    fine = laws(tables, info["energy_cuts_MeV"]["e-"], maximum, points=131073)
    rows, energy, length = frozen["row"], frozen["entry_MeV"], frozen["length_mm"]
    ids = np.column_stack([g[k][rows] for k in ("run", "subrun", "evt")])
    source_paths = [Path(__file__), Path(rf.__file__), Path(ls.__file__), Path(pt.__file__),
                    Path(__file__).with_name("rangestudy.py"), Path(__file__).with_name("convergence.py"),
                    Path(__file__).with_name("reference_audit.py"), Path(__file__).with_name("slicestudy.py"),
                    Path(__file__).with_name("g4ref.py"), Path(geometry.__file__)]
    sources = [_identity(p) for p in source_paths] + [_identity(Path(eloss.__file__).parent / p) for p in CORE_FILES]
    report = dict(complete=False, charge_selection=charges, replicas=a.replicas, seed=a.seed, slices=a.slices,
                  g4_first_among_entries=int(np.sum(g["foil"][rows] == 0)),
                  g4_negative_first_full_block=int(np.sum((g["foil"] == 0) & (g["pdg"] == 13))),
                  entry_MeV_quantiles=ls.quantiles(energy), length_mm_quantiles=ls.quantiles(length),
                  already_subthreshold_rows=rows[energy < rf.THRESHOLD].tolist(),
                  models={}, runs={}, comparisons={}, slice_comparisons={},
                  scope="Frozen fast-entry diagnostic; response replicas are not independent input samples or G4 validation")
    for name, law in model.items():
        reference, stop = ls.range_out(energy, length, (law.energy, law.ranges))
        finer, fine_stop = ls.range_out(energy, length, (fine[name].energy, fine[name].ranges))
        report["models"][name] = dict(deterministic_stops=int(stop.sum()),
            grid_stop_disagreements=int(np.sum(stop != fine_stop)),
            grid_max_abs_exit_MeV=float(np.max(np.abs(reference - finer))),
            entrants_with_Tmax_above_cut=int(np.sum(eloss.kinematics(ls.momentum(energy))[-1] > law.cut)))
    check_energy = np.array([.1, .2, .5, 1., 2., 3., 5., 10., 20.])
    hard = rf.delta_mean_per_mm(check_energy, info["energy_cuts_MeV"]["e-"])
    gap = pt.interpolate(tables["DEDXnr"], check_energy) - pt.interpolate(tables["DEDX"], check_energy)
    report["split_mean_check"] = dict(energy_MeV=check_energy.tolist(), analytic_hard_MeV_mm=hard.tolist(),
        g4_unrestricted_minus_restricted_MeV_mm=gap.tolist(),
        split_minus_unrestricted_MeV_mm=(hard - gap).tolist(),
        note="Analytic hard spectrum is not release-matched G4; aggregate DEDX split may include other processes")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "entries.npz", **frozen, event_ids=ids, g4_first=g["foil"][rows] == 0)
    manifest = dict(inputs=inputs, sources=sources, arguments=vars(a), couple=info, jax=jax.__version__,
        numpy=np.__version__, devices=[str(d) for d in jax.devices()], arms=ARMS,
        contract="Range relative to 0.1 MeV; bypass fluctuations if path > residual range. Analytic hard spectrum at slice entrance; continuous then capped hard subtraction. No electron transport/deposition or G4 scheduler parity.",
        uncertainty="Conditional replica MC SE separately from event-cluster sampling diagnostic; no G4 error budget")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    started = time.monotonic()
    previous = {}
    completed = []
    for slices in a.slices:
        outputs = {}
        for name, (law_name, mode, deltas) in ARMS.items():
            if time.monotonic() - started > a.budget_seconds:
                report["budget_exhausted_before"] = f"{name}_s{slices}"
                break
            then = time.monotonic()
            result = rf.run_layer(energy, length, model[law_name], mode=mode, deltas=deltas,
                                  slices=slices, replicas=a.replicas, seed=a.seed)
            tag = f"{name}_s{slices}"
            np.savez_compressed(out / f"{tag}.npz", **result)
            report["runs"][tag] = summarize(result, ids)
            report["runs"][tag]["wall_s"] = time.monotonic() - then
            outputs[name] = {k: result[k] for k in ("stop", "exit_MeV")}
            print(json.dumps(dict(arm=tag, **{k: report["runs"][tag][k] for k in
                ("expected_stops", "expected_forced_stops", "expected_hard_count", "wall_s")})), flush=True)
        for cand, base in PAIRS:
            if cand in outputs and base in outputs:
                report["comparisons"][f"{cand}_minus_{base}_s{slices}"] = paired(outputs[cand], outputs[base])
        if previous:
            for name in outputs.keys() & previous.keys():
                report["slice_comparisons"][f"{name}_s{slices}_minus_s{completed[-1]}"] = paired(outputs[name], previous[name])
        previous = outputs
        if len(outputs) == len(ARMS):
            completed.append(slices)
        report["wall_s"] = time.monotonic() - started
        (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        if "budget_exhausted_before" in report:
            break
    for item in inputs + sources:
        verify_hashes([item], [_identity(item["path"])], "input/source during study")
    report["complete"] = len(completed) == len(a.slices)
    manifest["outputs"] = [_identity(p) for p in sorted(out.glob("*.npz"))]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(complete=report["complete"], wall_s=report["wall_s"], out=str(out))), flush=True)
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
