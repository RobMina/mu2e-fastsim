"""Summarize completed audited tier1b chain benchmarks and test logs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--numpy")
    p.add_argument("--jax")
    p.add_argument("--tests")
    p.add_argument("--out")
    a = p.parse_args()
    result = {}
    for name, root in (("numpy", a.numpy), ("jax", a.jax)):
        if not root:
            continue
        report = json.loads((Path(root)/"benchmark.json").read_text())
        warm = report["passes"][1:]
        median = {k: float(np.median([rec["runchain"]["timers"].get(k, 0.) for rec in warm]))
                  for k in warm[0]["runchain"]["timers"] if k.endswith("_s")}
        result[name] = dict(warm_wall_s=[rec["elapsed_s"] for rec in warm],
                            warm_wall_median_s=float(np.median([rec["elapsed_s"] for rec in warm])),
                            warm_stage_medians_s=median,
                            repeat_exact=report.get("repeat_fingerprints_identical"),
                            context_audits=report.get("context_audits_passed"),
                            preparation_s=report.get("preparation_elapsed_s"),
                            passes=[{k: v for k, v in rec.items() if k != "files"} for rec in report["passes"]])
    if a.tests:
        text = Path(a.tests).read_text()
        summary_lines = [line for line in text.splitlines() if re.search(r"\bin [0-9.]+s\b", line)]
        totals = {name: sum(int(value) for line in summary_lines for value in re.findall(rf"\b(\d+) {name}\b", line))
                  for name in ("passed", "failed", "skipped", "error", "errors")}
        files = re.split(r"^== ", text, flags=re.MULTILINE)[1:]
        failures = {block.splitlines()[0]: "\n".join(block.splitlines()[-14:]) for block in files
                    if "FAILED " in block or "ERROR " in block or "ERRORS " in block}
        result["tests"] = dict(totals=totals, files=len(files), failures=failures)
    text = json.dumps(result, indent=2, allow_nan=False)
    if a.out:
        with open(a.out, "x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
