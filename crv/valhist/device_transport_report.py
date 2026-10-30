"""Summarise retained device-transport validation or end-to-end benchmark JSON."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--paired-tables", nargs=2)
    p.add_argument("--baseline")
    p.add_argument("--stock")
    p.add_argument("--counters")
    args = p.parse_args()
    if args.paired_tables:
        import numpy as np
        from crvfastsim import tables
        a, b = [tables.read(path, "steps") for path in args.paired_tables]
        identity = tables.KEY + ["bar", "sp", "nstep"]
        print("paired_tables", json.dumps(dict(rows=[len(a), len(b)],
            dtypes=[[str(x) for x in f.dtypes] for f in (a,b)],
            identity_equal=bool(np.array_equal(a[identity].to_numpy(), b[identity].to_numpy())),
            stops_equal=bool(np.array_equal(a.p1.to_numpy()==0., b.p1.to_numpy()==0.)),
            column_max_abs={c:float(np.max(np.abs(a[c].to_numpy().astype(float)-b[c].to_numpy().astype(float)), initial=0.))
                            for c in a if len(a)==len(b)})))
    if args.baseline:
        from crvfastsim import geometry
        from crvfastsim.analyses import v1compare as v1
        state = v1._new_state()
        g = geometry.from_offline_config(args.counters)
        v1._accumulate(state, [args.baseline], [args.stock], g.sector)
        v = v1.summarise(state)
        print("numpy_V1", json.dumps({g: {k:z.get(k) for k in
            ("pass", "crossings_per_event_rel", "len_mean_rel", "edep_mean_rel", "checks")}
            for g,z in v["groups"].items()}))
    for path in args.paths:
        d = json.loads(Path(path).read_text())
        print(path)
        if "candidate_stock_V1" in d:
            print("bank", json.dumps({k:d[k] for k in ("generated_events", "physics_port_pass", "candidate_stock_pass", "numpy_stock_pass", "ready_for_handoff")}))
            print("bank_yield", json.dumps(d["aggregate_histograms"]["_total/all"]))
            print("bank_shapes", json.dumps({k:r for k,r in d["aggregate_histograms"].items() if not k.startswith("_total/" )}))
            print("port_EX_edep", json.dumps(d["port_V1"]["groups"]["EX"]["edep_ratio"]))
            continue
        if "physics_port_pass" in d:
            print("certificate_gates", json.dumps({k:d[k] for k in
                ("complete", "physics_port_pass", "stock_V1_pass", "throughput_pass", "ready_for_handoff")}))
            print("aggregate_yield", json.dumps(d.get("aggregate_histograms", {}).get("_total/all")))
            print("speedups", json.dumps(d.get("per_shard_median_speedup_observed_range")))
            for k in ("port_V1", "numpy_V1_vs_stock", "candidate_V1_vs_stock"):
                print(k, json.dumps({g:{key:r.get(key) for key in
                    ("pass", "crossings_per_event_rel", "len_mean_rel", "edep_mean_rel", "checks", "invalid_reasons")}
                    for g,r in d.get(k, {}).get("groups", {}).items()}))
            print("stop_counts", json.dumps(d.get("stop_counts")))
            continue
        if "shards" in d:
            h = d.get("aggregate_histograms", {})
            print("aggregate yields", json.dumps({k: v for k,v in h.items() if k.startswith("_total/")}))
            print("shapes", json.dumps({k:v for k,v in h.items() if not k.startswith("_total/")}))
            for shard, r in d["shards"].items():
                receipt = r["baseline_receipt"]
                print("shard", shard, "rows", r["numpy_steps"], r["jax_steps"], "crossings", r["deterministic"],
                      "baseline_max", max(receipt.get("max_abs", {}).values(), default=None),
                      "seconds", r["numpy_s"], [x["total_s"] for x in r["trials"]])
                for label in ("v1_vs_stock", "numpy_v1_vs_stock"):
                    v = r.get(label)
                    if v:
                        print(label, json.dumps({g: {k: z.get(k) for k in
                            ("pass", "crossings_per_event_rel", "len_mean_rel", "edep_mean_rel", "checks")}
                            for g,z in v["groups"].items()}))
                if "numpy_times_s" in r:
                    nt = r["numpy_times_s"]
                    jt = [x["total_s"] for x in r["trials"][1:]]
                    print("repeat_timing", json.dumps(dict(numpy_s=nt, jax_warm_s=jt,
                        median_speedup=statistics.median(nt)/statistics.median(jt))))
        if "warm" in d and "cold" in d:
            print("profile", json.dumps({k:v for k,v in d.items() if k != "first_batch_crossing_counts"}))
        if "passes" in d:
            print("repeat_exact", d.get("repeat_fingerprints_identical"))
            warm = d["passes"][1:]
            if warm:
                print("chain_medians", json.dumps(dict(wall_s=statistics.median(r["elapsed_s"] for r in warm),
                    generated_events=warm[0]["generated_events"],
                    timers={k:statistics.median(r["runchain"]["timers"].get(k, 0.) for r in warm) for k in
                        ("transport_s", "tier1b_s", "photons_s", "sipm_s", "waveforms_s", "reco_s", "context_audit_s")})))
            for r in d["passes"]:
                print("pass", r["index"], "wall", r["elapsed_s"], "counts", r["counts"],
                      "timers", json.dumps(r["runchain"]["timers"]))


if __name__ == "__main__":
    main()
