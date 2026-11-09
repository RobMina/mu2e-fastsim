"""Summarize recorded profiles and measured benchmark reports without rerunning jobs."""
import argparse
import json
from pathlib import Path
import statistics


def summary(directory):
    root = Path(directory)
    report = json.loads((root / "benchmark/benchmark.json").read_text())
    warm = report["passes"][1:]
    timers = sorted({key for p in warm for key in p["runchain"]["timers"] if key.endswith("_s")})
    output = {"path": str(root), "device": report["device"],
              "environment": report["hardware_software"]["environment"],
              "repeat_identical": report["repeat_fingerprints_identical"],
              "context_audits_passed": report["context_audits_passed"],
              "generated_events": warm[0]["generated_events"], "counts": warm[0]["counts"],
              "preparation_elapsed_s": report["preparation_elapsed_s"],
              "preparation_timers": report["preparation_timers"],
              "cold_s": report["passes"][0]["elapsed_s"],
              "warm_seconds": [p["elapsed_s"] for p in warm],
              "warm_median_s": statistics.median(p["elapsed_s"] for p in warm),
              "warm_stage_medians_s": {key: statistics.median(p["runchain"]["timers"].get(key, 0) for p in warm)
                                       for key in timers},
              "warm_step_rate_median": statistics.median(p["rates"]["events_with_steps_per_s"] for p in warm)}
    profiles = {}
    for path in sorted(root.glob("*.json")):
        profile = json.loads(path.read_text())
        if profile.get("profiled"):
            profiles[path.stem] = {"elapsed_s": profile["elapsed_s"], "top": profile["functions"][:25],
                                   "tier1b": [row for row in profile["functions"] if row["file"].endswith("tier1b.py")][:20]}
    output["profiles"] = profiles
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [summary(path) for path in args.directories]
    with open(args.out, "x") as stream:
        json.dump(rows, stream, indent=2)
    for row in rows:
        print(json.dumps({key: value for key, value in row.items() if key not in ("profiles", "environment")}, indent=2))
        if row["profiles"]:
            label = sorted(k for k in row["profiles"] if k.startswith("pass"))[-1]
            print("tier1b warm profile", json.dumps(row["profiles"][label]["tier1b"], indent=2))


if __name__ == "__main__":
    main()
