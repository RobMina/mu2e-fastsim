"""Derive explicit device capacities from measured demand.

    python3 -m mixing.device_capacity RUN [RUN ...] --template CAPS.json --margin 1.25 --out NEW_CAPS.json

RUN is a finished `stream_benchmark` output (per-batch `capacity_usage` in `batches.jsonl`) or a `device_bench`
output (`pass_*/pass_metrics.json`). Every buffer with a measured demand gets `ceil(margin x largest demand)`, rounded
up where a layout needs it (straws to the electronics batch); widths that are padding choices (`clusters_per_step`,
`clusters_per_end` and the end-tier widths) and buffers without a demand counter stay as in the template. The
result still fails loudly: a later batch whose demand exceeds a capacity is rejected and the message names the
buffer with its demand and capacity. It never truncates.
"""
import argparse
import copy
import json
import math
from pathlib import Path

TRACKER_SIZED = ("drift_clusters", "straws", "long_steps", "clusters_per_long_step", "flat_clusters", "minion_steps")
CRV_SIZED = ("substeps", "photons", "thermal", "charge_clusters", "digis", "reco_waveforms", "coinc_hits")
CRV_FLOOR = ("charges_per_cluster", "waveform_samples", "reco_samples")   # never below the template


def demands(paths):
    """Largest demand per buffer name over every batch of every run."""
    out, batches = {}, 0
    for path in map(Path, paths):
        rows = []
        if (path / "batches.jsonl").is_file():
            rows = [json.loads(line)["backend"].get("capacity_usage", {})
                    for line in (path / "batches.jsonl").read_text().splitlines()]
        rows += [json.loads(p.read_text()).get("capacity_usage_max", {}) for p in sorted(path.glob("pass_*/pass_metrics.json"))]
        if not rows:
            raise ValueError(f"no capacity usage recorded under {path}")
        batches += len(rows)
        for row in rows:
            for name, value in row.items():
                out[name] = max(out.get(name, 0), int(value))
    return out, batches


def derive(template, usage, margin):
    if margin < 1:
        raise ValueError("margin must be at least 1")
    caps = copy.deepcopy(template)
    grow = lambda value: int(math.ceil(margin * value))
    for name in caps["rows"]:
        if f"rows/{name}" in usage:
            caps["rows"][name] = max(1, grow(usage[f"rows/{name}"]))
    if "max_donors" in usage:
        caps["max_donors"] = max(1, grow(usage["max_donors"]))
    tracker = caps.get("response", {}).get("tracker")
    if tracker is not None:
        for name in TRACKER_SIZED:
            key = f"response/tracker/{name}"
            if key in usage and name in tracker and tracker[name] is not None:
                tracker[name] = max(1, grow(usage[key]))
        key = "response/tracker/digit_entries_per_straw_milli"
        if key in usage and tracker.get("digit_entries_per_straw") is not None:
            tracker["digit_entries_per_straw"] = min(tracker["pairs_per_straw"],
                                                     math.ceil(margin * usage[key]) / 1000)
        if tracker.get("clusters_per_long_step") is not None:
            tracker["clusters_per_long_step"] = max(tracker["clusters_per_long_step"], tracker["clusters_per_step"] + 1)
        batch = tracker.get("electronics_batch", 1)
        tracker["straws"] = -(-tracker["straws"] // batch) * batch
        for k, tier in enumerate(tracker.get("end_tiers") or (), start=1):
            key = f"response/tracker/end_tier_{k}_straws"
            if key in usage:
                step = min(batch, max(1, grow(usage[key])))
                tier[0] = -(-max(1, grow(usage[key])) // step) * step
    output = caps.get("response", {}).get("output")
    if output is not None and output.get("tracker_rows") is not None and "response/output/tracker_rows" in usage:
        output["tracker_rows"] = max(1, grow(usage["response/output/tracker_rows"]))
    crv = caps.get("response", {}).get("crv")
    if crv is not None:
        for name in CRV_SIZED + CRV_FLOOR:
            key = f"response/crv/{name}"
            if key in usage and name in crv:
                crv[name] = max(crv[name], grow(usage[key])) if name in CRV_FLOOR else max(1, grow(usage[key]))
    return caps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--template", type=Path, required=True, help="capacity JSON whose layout and widths are kept")
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    usage, batches = demands(args.runs)
    caps = derive(json.loads(args.template.read_text()), usage, args.margin)
    with args.out.open("x") as stream:
        stream.write(json.dumps(caps, indent=1) + "\n")
    selected = usage.get("selected_rows")
    print(json.dumps({"batches": batches, "margin": args.margin,
                      "max_selected_rows": None if selected is None else int(math.ceil(args.margin * selected)),
                      "demand": usage}, indent=1))


if __name__ == "__main__":
    main()
