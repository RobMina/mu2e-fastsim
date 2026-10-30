"""Independent native/table end-to-end CRV smoke comparison with clustered count errors."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .. import chain, tables
from . import runchain


def _counts(outputs, live, events):
    stype = np.concatenate([o["cl_type"] for o in outputs]) if outputs else np.empty(0, int)
    event = np.concatenate([o["cl_event"] for o in outputs]) if outputs else np.empty(0, int)
    result = {}
    for k in (1, 2, 3, 4):
        chosen = event[stype == k]
        _, sizes = np.unique(chosen, return_counts=True)
        total = int(np.sum(sizes))
        var = (float(np.dot(sizes, sizes)) - total**2 / events) * events / max(events-1, 1)
        result[str(k)] = {"clusters": total, "per_s": total / live,
                           "se_per_s_conditional_exposure": np.sqrt(max(var, 0)) / live}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--gdml", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--deviation", required=True)
    p.add_argument("--tier1b-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--events", type=int, default=50000)
    p.add_argument("--backend", choices=("native", "device"), default="native")
    p.add_argument("--reuse-reference", help="saved independent table arm from an earlier comparison.json")
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads(Path(a.reuse_reference).read_text()) if a.reuse_reference else None
    records = {"table": saved["arms"]["table"]} if saved else {}
    if saved and records["table"]["stats"]["generated_events"] != a.events:
        raise ValueError("saved reference count differs from requested native events")
    original = chain.clusters
    for name, inputs in (("native", "native"), ("table", a.reference)):
        if name == "table" and saved:
            continue
        seen = []
        def collect(*args, **kw):
            outputs, more = original(*args, **kw)
            seen.extend(outputs)
            return outputs, more
        chain.clusters = collect
        try:
            stats = runchain.run([inputs], str(out / name), gdml=a.gdml, calib=a.calib,
                                  deviation=a.deviation, tier1b_set="strict_v3", tier1b_dir=a.tier1b_dir,
                                  seed=20261030 if name == "native" else 20261031,
                                  jax=a.backend == "device",
                                  **({"gen": a.backend, "gen_events": a.events, "gen_subrun": 900} if name == "native" else {}))
        finally:
            chain.clusters = original
        live = stats["live_seconds"] if name == "native" else tables.livetime([inputs])
        records[name] = {"stats": stats, "live_seconds": live,
                          "clusters": _counts(seen, live, stats["generated_events"])}
    ratios = {}
    for k in ("1", "2", "3", "4"):
        x, y = records["native"]["clusters"][k], records["table"]["clusters"][k]
        ratio = x["per_s"] / y["per_s"] if y["per_s"] else None
        err = (float(np.hypot(x["se_per_s_conditional_exposure"]/y["per_s"],
                             x["per_s"]*y["se_per_s_conditional_exposure"]/y["per_s"]**2)) if y["per_s"] else None)
        ratios[k] = {"native_over_table": ratio, "se_conditional_exposure": err}
    report = {"arms": records, "ratios": ratios,
              "error_method": "independent event-clustered cluster-count errors conditional on measured exposure; no exposure uncertainty included",
              "status": "smoke comparison only, not certification", "same_digis": False, "backend": a.backend, "reference_receipt": a.reuse_reference}
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"ratios": ratios}, indent=2))


if __name__ == "__main__":
    main()
