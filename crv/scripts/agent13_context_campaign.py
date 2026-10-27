"""Run local prepared-context benchmarks with exact comparison and no resume."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# Import the sibling comparator without importing the physics packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent13_compare_context as compare_context


RUNS = (
    ("n50000_compact0", (11,), 0, 20261027, 4, True, False),
    ("n50000_compact1", (11,), 1, 20261027, 4, True, False),
    ("context_heldout_reference", (12, 13), 1, 20261028, 2, False, True),
    ("context_heldout_candidate", (12, 13), 1, 20261028, 4, True, True),
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_snapshot():
    sources = {}
    for package in ("crvfastsim", "mu2efastsim"):
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f"package unavailable in current interpreter: {package}")
        for root in spec.submodule_search_locations:
            for path in Path(root).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".json"):
                    sources[str(path.resolve())] = digest(path)
    return sources


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)


def require_file(path):
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty local file: {path}")
    return path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path, help="local immutable asset root")
    p.add_argument("--inputs-dir", required=True, type=Path)
    p.add_argument("--lookup-dir", type=Path, help="optical table root, default --data")
    p.add_argument("--outdir", required=True, type=Path, help="new private directory for detailed NPZ products")
    p.add_argument("--summary-dir", required=True, type=Path, help="existing directory for JSON and logs only")
    p.add_argument("--reference-dir", type=Path, help="saved n50000_compact{0,1}.json; absent references are deferred")
    for name in ("library", "calib", "deviation", "gdml"):
        p.add_argument("--" + name, type=Path, help="explicit local asset path overriding --data defaults")
    return p


def validate_report(report, expected, compact, prepared, sources):
    if report.get("status") != "complete" or report.get("repeat_fingerprints_identical") is not True:
        raise ValueError("completed repeat-identical report required")
    arguments = report["arguments"]
    if any(arguments.get(key, False if isinstance(value, bool) else None) != value
           for key, value in expected.items()):
        raise ValueError("completed report does not match requested campaign settings/input ordering")
    policy = "prepared_once" if prepared else "reload_each_pass"
    if report.get("context_policy") != policy or (prepared and report.get("context_audits_passed") is not True):
        raise ValueError("completed report has invalid context policy/audits")
    if report["hardware_software"]["environment"].get("CRVFASTSIM_COMPACT_DISPATCH") != str(compact):
        raise ValueError("completed report has incorrect compact dispatch")
    if report["provenance"]["source_sha256"] != sources:
        raise ValueError("subprocess source snapshot differs from campaign source snapshot")
    passes = report["passes"]
    if len(passes) != expected["repeat"]:
        raise ValueError("completed report has incorrect repeat count")
    dataset = []
    fingerprints = None
    for index, item in enumerate(passes):
        files = item["files"]
        if [f["input"] for f in files] != expected["inputs"]:
            raise ValueError("completed pass input ordering differs")
        current = [f["fingerprint"] for f in files]
        if fingerprints is not None and current != fingerprints:
            raise ValueError("repeat output differs: no tolerance allowed")
        fingerprints = current
        counts = []
        for f in files:
            selected = f["counts"]["generated_events"]
            full = f["counts"]["input_generated_events"]
            if type(selected) is not int or type(full) is not int or not 0 < selected <= full:
                raise ValueError("invalid measured dataset counts")
            counts.append({"input": f["input"], "input_generated_events": full, "generated_events": selected})
        if sum(f["generated_events"] for f in counts) != item["generated_events"]:
            raise ValueError("completed report denominator mismatch")
        if index and counts != dataset:
            raise ValueError("dataset counts changed across repeats")
        dataset = counts
    return {"files": dataset, "input_generated_events": sum(f["input_generated_events"] for f in dataset),
            "generated_events_per_pass": sum(f["generated_events"] for f in dataset),
            "passes": len(passes), "generated_event_executions": sum(p["generated_events"] for p in passes)}


def run(args):
    if os.path.lexists(args.outdir):
        raise FileExistsError(f"refusing existing output: {args.outdir}")
    outdir, summary = args.outdir.resolve(), args.summary_dir.resolve()
    if not summary.is_dir():
        raise ValueError("--summary-dir must be an existing directory")
    data = args.data.resolve()
    if not data.is_dir():
        raise ValueError("--data must be an existing local directory")
    if args.reference_dir is not None and not args.reference_dir.is_dir():
        raise ValueError("--reference-dir must be an existing directory")
    library = args.library
    if library is None:
        library = data / "lib_prod_as_20files.root"
        if not library.is_file():
            library = data / "tier1b_v7" / "lib_prod_as_20files.root"
    assets = {"library": require_file(library),
              "calib": require_file(args.calib or data / "crvcalib_sim_best_v1_5_run1400.csv"),
              "deviation": require_file(args.deviation or data / "crvphoton_sim_best_v1_5_run1400.csv"),
              "gdml": require_file(args.gdml or data / "mu2e_extracted_v04.gdml")}
    inputs = {n: require_file(args.inputs_dir / f"gen.001400_{n:08d}.root") for n in (11, 12, 13)}
    comparison_names = [f"context_n50000_compact{c}_comparison.json" for c in (0, 1)]
    comparison_names.append("context_heldout_comparison.json")
    destinations = [summary / (r[0] + suffix) for r in RUNS for suffix in (".json", ".log")]
    destinations += [summary / name for name in ["campaign.json", *comparison_names]]
    for path in destinations:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing existing summary artifact: {path}")
    sources = source_snapshot()
    scripts = {str(Path(p).resolve()): digest(p) for p in (__file__, compare_context.__file__)}

    def check_sources():
        if source_snapshot() != sources or any(digest(p) != h for p, h in scripts.items()):
            raise ValueError("source snapshot changed during campaign")

    env = os.environ.copy()
    lookup = args.lookup_dir.resolve() if args.lookup_dir is not None else data
    if not lookup.is_dir():
        raise ValueError("--lookup-dir must be an existing local directory")
    env.update(PYTHONDONTWRITEBYTECODE="1", CRVFS_DATA=str(data), CRVFS_LOOKUP_DIR=str(lookup))
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    outdir.mkdir(parents=True, exist_ok=False)
    campaign = {"schema": 1, "outdir": str(outdir), "summary_dir": str(summary),
                "python_executable": sys.executable, "source_sha256": sources, "script_sha256": scripts,
                "max_events_policy": "50000 event numbers per input file; actual full-key counts are read from reports",
                "stock_physics_validation": "not evaluated", "runs": {}, "comparisons": {}}
    reports = {}
    for name, shards, compact, seed, repeat, prepared, prefetch in RUNS:
        check_sources()
        expected = {"inputs": [str(inputs[n]) for n in shards], "seed": seed, "repeat": repeat,
                    "max_events": 50000, "run_config": "extracted_v04", "window": "offspill",
                    "jax": True, "prefetch": prefetch, **{k: str(v) for k, v in assets.items()}}
        command = [sys.executable, "-m", "crvfastsim.analyses.e2e_bench", *expected["inputs"]]
        for key in ("library", "calib", "deviation", "gdml", "run_config", "window", "seed", "max_events", "repeat"):
            command += ["--" + key.replace("_", "-"), str(expected[key])]
        command += ["--jax", "--outdir", str(outdir / name)]
        if prepared:
            command.append("--reuse-context")
        if prefetch:
            command.append("--prefetch")
        run_env = {**env, "CRVFASTSIM_COMPACT_DISPATCH": str(compact)}
        log = summary / (name + ".log")
        with log.open("x") as stream:
            subprocess.run(command, env=run_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        check_sources()
        report_path = outdir / name / "benchmark.json"
        report = json.loads(report_path.read_text())
        dataset = validate_report(report, expected, compact, prepared, sources)
        copied = summary / (name + ".json")
        with report_path.open("rb") as src, copied.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        reports[name] = report_path
        campaign["runs"][name] = {"command": command, "compact_dispatch": compact, "report": str(copied),
                                  "detailed_report": str(report_path), "log": str(log), "dataset": dataset}
        if shards == (11,):
            reference = args.reference_dir / (name + ".json") if args.reference_dir is not None else None
            if reference is None or not os.path.lexists(reference):
                campaign["comparisons"][name] = {"status": "deferred", "reason": "saved shard11 reference unavailable",
                                                  "reference": str(reference) if reference is not None else None}
            else:
                result = compare_context.compare(reference, report_path)
                path = summary / ("context_" + name + "_comparison.json")
                write_json(path, result)
                campaign["comparisons"][name] = {"status": "complete", "record": str(path)}
    result = compare_context.compare(reports["context_heldout_reference"], reports["context_heldout_candidate"])
    result["comparison_limit"] = "current sequential reload/prepared held-out runs, not interleaved timing; candidate audits included"
    comparison = summary / "context_heldout_comparison.json"
    write_json(comparison, result)
    campaign["comparisons"]["heldout"] = {"status": "complete", "record": str(comparison)}
    check_sources()
    campaign["status"] = "complete"
    write_json(summary / "campaign.json", campaign)
    return campaign


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        p.exit(1, f"context-campaign: {exc}\n")
    print(json.dumps({"status": result["status"], "campaign": str(args.summary_dir / "campaign.json")}))


if __name__ == "__main__":
    main()
