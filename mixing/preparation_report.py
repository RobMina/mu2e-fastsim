"""Validate native preparation outputs and report cross-run descriptive timing, not speedup."""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from container.offline.compare_root import compare
from trigger.timing import finite_sum, parse_log
from .counts import event_ids, event_keys

STEP_DIRS = ("primarySteps", "stopSteps", "eleSteps", "muSteps", "neutralSteps")
CATEGORIES = ("mix", "filters", "CrvPhotons", "other CRV", "step export", "misc")


def category(label, kind):
    if kind == "MixBackgroundFrames" or label.endswith("Mixer"):
        return "mix"
    if kind == "MixingFilter" or label.endswith("Filter"):
        return "filters"
    if label == "CrvPhotons" or kind == "CrvPhotons":
        return "CrvPhotons"
    if "crv" in (label + kind).lower():
        return "other CRV"
    if label in STEP_DIRS or kind == "StrawGasStepTableDumper":
        return "step export"
    return "misc"


def timing(text, expected_events):
    parsed = parse_log(text)
    if parsed["events"] != expected_events:
        raise ValueError("log event count differs from finished ROOT census")
    if min(parsed["cpu_s"], parsed["real_s"], parsed["full_event"]["avg_s"]) <= 0:
        raise ValueError("job and full-event times must be positive finite numbers")
    modules, excluded = {}, []
    for row in parsed["rows"]:
        total = finite_sum([row["stats"]["avg_s"] * row["invocations"]])
        if row["path"] == "source" or row["type"].endswith("(write)"):
            excluded.append({"label": row["label"], "total_s": total})
            continue
        label = row["label"]
        item = modules.setdefault(label, {"type": row["type"], "category": category(label, row["type"]),
                                          "invocations": 0, "total_s": 0.0})
        item["invocations"] += row["invocations"]
        item["total_s"] = finite_sum([item["total_s"], total])
    totals = {name: finite_sum(item["total_s"] for item in modules.values() if item["category"] == name)
              for name in CATEGORIES}
    total = finite_sum(totals.values())
    if total <= 0:
        raise ValueError("module time sum must be positive")
    return {"cpu_s": parsed["cpu_s"], "real_s": parsed["real_s"],
            "real_events_per_s": finite_sum([expected_events / parsed["real_s"]]),
            "full_event_total_s": finite_sum([parsed["full_event"]["avg_s"] * expected_events]),
            "module_total_s": total, "by_module": modules, "by_category_s": totals,
            "excluded_subtimers": excluded,
            "sum_boundary": "Sum of module Avg*nEvts only; excludes Full event, source and write subtimers."}


def census(directory, expected_events):
    path = directory / "response.root"
    with uproot.open(path) as root:
        ids = event_ids(root["crvdump/events"].arrays(["run", "subrun", "event"], library="ak"))
        keys = event_keys(ids)
        if len(ids) != expected_events or np.unique(keys).size != len(ids):
            raise ValueError("finished ROOT census has wrong count or duplicate event identities")
        steps = {name: int(root[f"{name}/sgsteps"].num_entries) for name in STEP_DIRS}
    result = {"root": str(path), "finished_events": len(ids), "steps": steps,
              "total_steps": sum(steps.values()), "timing": timing((directory / "art.log").read_text(), len(ids))}
    metadata = directory / "metadata.json"
    if metadata.is_file():
        saved = json.loads(metadata.read_text())
        node = saved.get("node", saved.get("hostname"))
        if node is not None:
            result["node"] = node
            result["node_source"] = str(metadata)
    return ids, result


def report(reference, candidate, comparison, expected_events=2741):
    if expected_events <= 0:
        raise ValueError("expected-events must be positive")
    reference, candidate, comparison = map(lambda p: Path(p).resolve(), (reference, candidate, comparison))
    supplied = json.loads(comparison.read_text())
    json.dumps(supplied, allow_nan=False)
    if not isinstance(supplied, dict) or supplied.get("equal") is not True:
        raise ValueError("comparison does not certify equal ROOT outputs")
    for role, directory in (("reference", reference), ("candidate", candidate)):
        if role not in supplied or Path(supplied[role]).resolve() != directory / "response.root":
            raise ValueError(f"comparison {role} path does not match response.root")
    left, baseline = census(reference, expected_events)
    right, proposed = census(candidate, expected_events)
    if not np.array_equal(left, right):
        raise ValueError("full ordered run/subrun/event identities differ")
    if baseline["steps"] != proposed["steps"] or baseline["total_steps"] != proposed["total_steps"]:
        raise ValueError("five step-tree counts differ")
    # Reuse the exact comparator and require its entire current report, not an unverified equal flag.
    actual = compare(reference / "response.root", candidate / "response.root")
    normalized = dict(supplied, reference=actual["reference"], candidate=actual["candidate"])
    if (actual["equal"] is not True or normalized != actual
            or any(item.get("equal") is not True for item in supplied.get("objects", []))):
        raise ValueError("comparison objects, entries or content differ from the complete exact ROOT comparison")
    result = {"interpretation": "Cross-run descriptive timing; not a controlled speedup. Startup/queue excluded.",
              "expected_events": expected_events, "comparison": str(comparison), "equal": True,
              "reference": baseline, "candidate": proposed,
              "candidate_minus_reference_s": {key: proposed["timing"][key] - baseline["timing"][key]
                                               for key in ("cpu_s", "real_s", "module_total_s")}}
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "comparison", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-events", type=int, default=2741)
    args = parser.parse_args(argv)
    try:
        result = report(args.reference, args.candidate, args.comparison, args.expected_events)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"preparation report: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
