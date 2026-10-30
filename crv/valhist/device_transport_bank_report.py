"""Combine fixed-bank transport summaries and retained certificate without selecting shards.

No physics or statistical thresholds are modified. Timing from a reused baseline
is not inferred: transport throughput evidence remains the separate repeat audit.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from crvfastsim.analyses import v1compare as v1
from . import stephist as sh


def combine(args):
    paths = [Path(p) for p in args.summaries]
    seen = set()
    ah, bh, ph = {}, {}, {}
    states = {k:v1._new_state() for k in ("numpy_stock", "candidate_stock", "port")}
    result = dict(shards=[], generated_events=0, replay_exact=True, deterministic_exact=True,
                  transport_throughput_certified=False, ready_for_handoff=False,
                  note="All supplied nonoverlapping shards included; no throughput inference for reused baseline.")
    for path in paths:
        d = json.loads(path.read_text())
        shards = set(d["shards"])
        if seen & shards:
            raise ValueError("overlapping summary shards")
        seen |= shards
        result["generated_events"] += sum(r["generated_events"] for r in d["shards"].values())
        for r in d["shards"].values():
            result["replay_exact"] &= bool(r.get("repeat_exact", r.get("receipts", {}).get("repeat_status") == "certified"))
            det = r.get("deterministic", r.get("receipts", {}).get("deterministic", {}))
            result["deterministic_exact"] &= bool(det.get("exact_track_bar") and det.get("max_distance_delta_mm") == 0.)
        h = d["histogram_accumulators"]
        sh.add(ah, sh.from_json(h["numpy"]))
        sh.add(bh, sh.from_json(h["candidate"]))
        pair = {k:np.asarray(v) for k,v in h["paired_d2"].items()}
        sh.add_paired(ph, pair)
        vs = d.get("v1_states", d.get("V1_states"))
        for k in states:
            v1._merge_into(states[k], vs[k])
        timed_rows = [r for r in d["shards"].values() if len(r.get("numpy_times_s", [])) >= 4 and len(r.get("trials", [])) >= 4]
        timing_pass = len(timed_rows) >= 10 and all(
            np.median(r["numpy_times_s"][1:]) > np.median([t["total_s"] for t in r["trials"][1:]]) for r in timed_rows)
        result["transport_throughput_certified"] |= bool(d.get("throughput_pass") or timing_pass)
    result["shards"] = sorted(int(x) for x in seen)
    result["aggregate_histograms"] = sh.compare(ah, bh, ph)
    for k,state in states.items():
        result[k+"_V1"] = v1.summarise(state)
    result["physics_port_pass"] = bool(result["replay_exact"] and result["deterministic_exact"]
        and result["port_V1"]["pass"] and all(not r["flag"] for k,r in result["aggregate_histograms"].items() if not k.startswith("_total/")))
    result["candidate_stock_pass"] = result["candidate_stock_V1"]["pass"]
    result["numpy_stock_pass"] = result["numpy_stock_V1"]["pass"]
    result["ready_for_handoff"] = bool(result["physics_port_pass"] and result["candidate_stock_pass"]
        and result["numpy_stock_pass"] and result["transport_throughput_certified"])
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("aggregate_histograms", "numpy_stock_V1", "candidate_stock_V1", "port_V1")}))
    for k in states:
        print(k, json.dumps({g:{c:r[c] for c in ("pass", "crossings_per_event_rel", "len_mean_rel", "edep_mean_rel", "checks")}
            for g,r in result[k+"_V1"]["groups"].items()}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    return 0 if combine(args)["ready_for_handoff"] else 1


if __name__ == "__main__":
    sys.exit(main())
