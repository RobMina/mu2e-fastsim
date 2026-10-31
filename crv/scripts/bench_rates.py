"""Summarize measured full-pass rates without adding overlapping stage times."""
import argparse
import json
from pathlib import Path
import statistics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args(argv)
    for name in args.reports:
        p = Path(name)
        report = json.loads(p.read_text())
        if report.get("status") != "complete":
            raise ValueError(f"not complete: {p}")
        print(f"## {p}\n{report['boundary']}")
        print(f"device={report['device']} repeats_identical={report['repeat_fingerprints_identical']}")
        print(json.dumps({k: report[k] for k in ("context_policy", "preparation_elapsed_s", "preparation_timers",
                          "preparation_asset_hash_s", "passes_elapsed_s", "final_provenance_verify_s",
                          "invocation_elapsed_before_report_s", "invocation_generated_events_per_s") if k in report}))
        warm = [item["elapsed_s"] for item in report["passes"][1:]]
        if warm:
            counts = {key: sum(f["counts"][key] for f in report["passes"][0]["files"])
                      for key in ("generated_events", "events_with_steps", "pulses", "clusters")}
            median = statistics.median(warm)
            audits = [item["runchain"]["timers"].get("context_audit_s", 0.) for item in report["passes"]]
            print(json.dumps({"warm_summary": {"elapsed_s": warm, "median_elapsed_s": median,
                  "first_execution_s": report["passes"][0]["elapsed_s"], "counts": counts,
                  "median_rates_per_s": {key: val / median for key, val in counts.items()},
                  "context_audit_range_s": [min(audits), max(audits)]}}))
        for item in report["passes"]:
            counts = {key: sum(f["counts"][key] for f in item["files"])
                      for key in ("generated_events", "events_with_steps", "pulses", "clusters")}
            print(json.dumps({"stage": item["stage"], "elapsed_s": item["elapsed_s"], "counts": counts,
                              "rates_per_s": {key: val / item["elapsed_s"] for key, val in counts.items()},
                              "setup_s": item["runchain"]["timers"]["setup_s"],
                              "stage_timers": item["runchain"]["timers"]}, sort_keys=True))


if __name__ == "__main__":
    main()
