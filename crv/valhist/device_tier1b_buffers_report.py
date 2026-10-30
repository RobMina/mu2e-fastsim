"""Summarize exact device-metadata/replay and separately warmed fixed-bank timings."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(data):
    records = data["records"]
    timed = [r for r in records if "warm" in r]
    medians = {name: [statistics.median(r["warm"][name]) for r in timed] for name in ("numpy", "jax")}
    sums = {name: sum(v) for name, v in medians.items()}
    return dict(shards=len(records), events=sum(r["events"] for r in records),
                metadata_exact=all(r["metadata"]["exact"] for r in records),
                replay_max_abs=max(r["replay"]["max_abs"] for r in records),
                timed_shards=[r["shard"] for r in timed], warm_summed_shard_medians=sums,
                warm_observed_ratio=sums["numpy"] / sums["jax"] if sums["jax"] else None,
                warm_shard_median_ranges={name: [min(v), max(v)] if v else None for name, v in medians.items()},
                warmup_s={str(r["shard"]): r.get("warmup") for r in timed},
                component_summed_shard_medians={c: sum(statistics.median(b[c] for b in r["warm_boundaries"]) for r in timed)
                                               for c in timed[0]["warm_boundaries"][0]} if timed and "warm_boundaries" in timed[0] else {},
                library_load_s=data["library_load_s"], device_prepare_s=data["device_prepare_s"],
                device_array_bytes=data["device_array_bytes"], boundary=data["boundary"], certified=False,
                note="Replay and residency follow-up only; previous six failed independent deck gates are unchanged")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    result = summarize(json.loads(Path(a.summary).read_text()))
    out.write_text(json.dumps(result, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
