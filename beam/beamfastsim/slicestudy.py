"""Extend saved convergence pilots to 16/32 slices and compare a fixed common-stopper population.

Requires a whole-event pilot with input_block.npz; reuses its nominal and slices8 outputs. Selection on all arms'
stop/foil outcomes is a diagnostic, not a measured-variable response model or a G4 timing comparison.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from . import g4ref, geometry
from .convergence import cluster_mean, summarize
from .reference_audit import _identity


PHYSICS_FILES = ("transport.py", "geometry.py", "simpleconfig.py", "fieldmap.py")
CORE_FILES = ("eloss.py", "eloss_jax.py", "materials.py")


def fixed_population(g, arms):
    """Report adjacent slice changes on one intersection, with fixed same/different-foil subsets."""
    if set(arms) != {4, 8, 16, 32}:
        raise ValueError("Exactly 4, 8, 16, 32 slice arms required")
    n = len(g["evt"])
    for f in arms.values():
        if any(len(f[k]) != n for k in ("fate", "foil", "t", "x")):
            raise ValueError("All arm arrays must match the selected input length")
        if not np.all(np.isfinite(f["t"])) or not np.all(np.isfinite(f["x"])):
            raise ValueError("Nonfinite transport output")
    ids = np.column_stack([g[k] for k in ("run", "subrun", "evt")])
    common = np.logical_and.reduce([arms[s]["fate"] == 1 for s in (4, 8, 16, 32)])
    same = common & np.logical_and.reduce([arms[s]["foil"] == arms[4]["foil"] for s in (8, 16, 32)])
    populations = {"common_stoppers": common, "same_foil_all_arms": same,
                   "changed_foil_any_arm": common & ~same}
    result = {}
    for name, mask in populations.items():
        record = dict(n=int(mask.sum()), rows=np.flatnonzero(mask).tolist(),
                      flight_time_ns={str(s): cluster_mean((arms[s]["t"] - g["time"])[mask], ids[mask])
                                      for s in (4, 8, 16, 32)}, differences={})
        for lo, hi in ((4, 8), (8, 16), (16, 32), (4, 32)):
            record["differences"][f"{hi}_minus_{lo}"] = dict(
                time_ns=cluster_mean((arms[hi]["t"] - arms[lo]["t"])[mask], ids[mask]),
                foil_index=cluster_mean((arms[hi]["foil"] - arms[lo]["foil"])[mask], ids[mask]),
                z_mm=cluster_mean((arms[hi]["x"][:, 2] - arms[lo]["x"][:, 2])[mask], ids[mask]))
        result[name] = record
    return result


def verify_hashes(saved, current, label):
    """Require identical ordered content, not site-specific paths."""
    if [p["sha256"] for p in saved] != [p["sha256"] for p in current]:
        raise ValueError(f"{label} changed since the saved pilot; do not mix physics baselines")


def main(argv=None):
    from . import __main__ as cli
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geom")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    ap.add_argument("--budget-seconds", type=float, default=1200)
    a = ap.parse_args(argv)
    if not a.seeds or len(set(a.seeds)) != len(a.seeds) or a.budget_seconds <= 0:
        ap.error("unique nonempty seeds and positive budget required")
    baseline = Path(a.baseline)
    manifest = json.loads((baseline / "manifest.json").read_text())
    saved_summary = json.loads((baseline / "summary.json").read_text())
    input_path = baseline / "input_block.npz"
    verify_hashes([manifest["selected_input"]], [_identity(input_path)], "selected input")
    code_dir = Path(__file__).resolve().parent
    core_dir = Path(__import__("mu2efastsim").__file__).resolve().parent
    saved_code = {Path(p["path"]).name: p for p in manifest["code"]}
    physics = [_identity(code_dir / p) for p in PHYSICS_FILES]
    core = [_identity(core_dir / p) for p in CORE_FILES]
    verify_hashes([saved_code[p] for p in PHYSICS_FILES], physics, "beam physics")
    verify_hashes(manifest["core_code"], core, "core physics")
    cli._geom(a.geom, target_contract=True)
    sources = []
    geometry.from_offline(a.geom or cli.GEOM_FILE, cli.OFFLINE_ROOT, source_files=sources)
    geo = [_identity(p) for p in sources]
    field = [_identity(Path(cli.MAPS) / f"{cli.MAP}.{ext}") for ext in ("header", "bin")]
    verify_hashes(manifest["geometry"], geo, "geometry")
    verify_hashes(manifest["field"], field, "field")
    g = g4ref.load(input_path)
    n = len(g["evt"])
    saved = {}
    provenance = []
    for seed in a.seeds:
        if seed not in manifest["seeds"]:
            raise ValueError("Seed absent from saved pilot")
        saved[seed] = {}
        for slices, name in ((4, "nominal"), (8, "slices8")):
            tag = f"{name}_seed{seed}"
            fpath = baseline / f"{tag}.npz"
            f = g4ref.load(fpath)
            meta = json.loads(str(f["meta"]))
            expected = dict(ds=5.0, n_slices=slices, tcut=1.0, max_steps=3000,
                            z_up=4100.0, r_wall=950.0, msc=True, decay=True)
            if meta != saved_summary["runs"][tag]["meta"] or meta["params"] != expected or len(f["fate"]) != n:
                raise ValueError(f"Saved arm metadata/length mismatch: {tag}")
            saved[seed][slices] = f
            provenance.append(_identity(fpath))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(complete=False, runs={}, fixed_population={}, unstarted=[])
    (out / "manifest.json").write_text(json.dumps(dict(
        baseline_manifest=_identity(baseline / "manifest.json"), input=_identity(input_path),
        reused_arms=provenance, physics=physics, core=core, geometry=geo, field=field,
        code=[_identity(p) for p in sorted(code_dir.glob("*.py"))], seeds=a.seeds,
        note="Fixed common-stopper/foil subsets are outcome-conditioned diagnostics; no G4 timing closure"), indent=2))
    for seed in a.seeds:
        arms = dict(saved[seed])
        for slices in (16, 32):
            tag = f"slices{slices}_seed{seed}"
            if time.monotonic() - started >= a.budget_seconds:
                report["unstarted"].append(tag)
                continue
            args = SimpleNamespace(g4=str(input_path), out=str(out / f"{tag}.npz"), geom=a.geom, n=n,
                                   ds=5.0, max_steps=3000, slices=slices, tcut=1.0, no_msc=False, no_decay=False,
                                   seed=seed, batch=n, chunk=200, repeat=0)
            cli.cmd_run(args)
            f = g4ref.load(args.out)
            arms[slices] = f
            report["runs"][tag] = dict(summarize(g, f, arms[4]), meta=json.loads(str(f["meta"])))
            (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        if set(arms) == {4, 8, 16, 32}:
            report["fixed_population"][str(seed)] = fixed_population(g, arms)
        (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    report["complete"] = not report["unstarted"]
    report["wall_s"] = time.monotonic() - started
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for seed, populations in report["fixed_population"].items():
        for name, p in populations.items():
            print(json.dumps(dict(seed=seed, population=name, n=p["n"], differences=p["differences"])), flush=True)
    print(f"complete={report['complete']} wall_s={report['wall_s']:.3f}", flush=True)


if __name__ == "__main__":
    main()
