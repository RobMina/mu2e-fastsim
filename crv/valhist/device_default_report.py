"""Read-only compact report for new-default device transport and tier1b audits."""
import argparse
import json
from pathlib import Path


def summarize(data, kind):
    if kind == "transport":
        rows = data["shards"]
        return dict(kind=kind, shards=len(rows), histograms=data.get("aggregate_histograms"),
                    stock=data.get("aggregate_v1_vs_stock"), numpy_stock=data.get("aggregate_numpy_v1_vs_stock"),
                    port=data.get("aggregate_port_v1"),
                    repeat_exact=all(r["repeat_exact"] for r in rows.values()),
                    deterministic=all(r["deterministic"]["exact_track_bar"] for r in rows.values()),
                    warm_s={k: [t["total_s"] for t in r["trials"][1:]] for k, r in rows.items()})
    cmp = data["comparison"]
    return dict(kind=kind, shards=len(data["records"]), events=sum(r["events"] for r in data["records"]),
                flagged=data["flagged"], yields={k: v for k, v in cmp.items() if k.startswith("_total")},
                flagged_details={k: cmp[k] for k in data["flagged"]}, component_yields=data["component_yields"],
                moments=data["moments"], max_replay_abs=max(r["replay"]["max_abs"] for r in data["records"]),
                device_prepare_s=data["device_prepare_s"], device_array_bytes=data["device_array_bytes"],
                warm={str(r["shard"]): r.get("warm") for r in data["records"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary")
    p.add_argument("--kind", choices=("transport", "tier1b"), required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = summarize(json.loads(Path(a.summary).read_text()), a.kind)
    with Path(a.out).open("x") as f:
        json.dump(result, f, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
