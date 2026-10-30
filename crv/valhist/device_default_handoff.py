"""Small read-only handoff view of literal full-bank gates and measured chain times."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.root)
    result = {}
    if (root / "hard_full_v21.json").exists():
        hard_path = root / "hard_full_details_v26.json"
        d = json.loads((hard_path if hard_path.exists() else root / "hard_full_v21.json").read_text())
        sums = {name: sum(sorted(r["warm"][name])[len(r["warm"][name]) // 2] for r in d["records"] if "warm" in r)
                for name in ("numpy", "jax")}
        result["sampler"] = dict(events=d["events"], shards=len(d["shards"]), replay_max_abs=d["max_replay_abs"],
            yields={k: {x: v.get(x) for x in ("n_ref", "n_cand", "rel", "rel_se_paired")}
                    for k, v in d["comparison"].items() if k.startswith("_total")},
            flags={k: d["comparison"][k] for k in d["flagged"]}, moments=d["moments"],
            component_yields=d.get("component_yields"), layer_census=d.get("layer_census"),
            warm_sums=sums, observed_ratio=sums["numpy"] / sums["jax"])
    if (root / "range_full_v21.json").exists():
        d = json.loads((root / "range_full_v21.json").read_text())
        result["transport"] = {k: d[k] for k in ("generated_events", "replay_exact", "deterministic_exact", "physics_port_pass", "candidate_stock_pass", "numpy_stock_pass")}
        result["transport"]["flags"] = {k: v for k, v in d["aggregate_histograms"].items() if v.get("flag")}
        result["transport"]["yields"] = {k: {x: v.get(x) for x in ("n_ref", "n_cand", "rel", "rel_se_paired")}
                                            for k, v in d["aggregate_histograms"].items() if k.startswith("_total")}
        for arm in ("port", "candidate_stock", "numpy_stock"):
            result["transport"][arm] = {g: {k: r[k] for k in ("pass", "checks", "len_mean_rel", "edep_mean_rel", "crossings_per_event_rel")}
                                       for g, r in d[arm + "_V1"]["groups"].items()}
    if (root / "native_default_v23.json").exists():
        d = json.loads((root / "native_default_v23.json").read_text())
        result["native"] = dict(preparation_s=d["preparation_s"], configuration=d["configuration"], boundary=d["boundary"],
                                records=[{k: r[k] for k in ("repeat", "events", "steps", "clusters", "types", "timers", "elapsed_s")}
                                         for r in d["records"]])
    for key, filename in (("chain", "chain_default_medians_v21.json"), ("downstream", "chain_default_compare_v21.json"),
                          ("tests", "default_final_tests_v25.json")):
        if (root / filename).exists():
            d = json.loads((root / filename).read_text())
            result[key] = ({n: {k: v for k, v in arm.items() if k != "passes"} for n, arm in d.items()}
                           if key == "chain" else d)
    with open(a.out, "x") as f:
        json.dump(result, f, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
