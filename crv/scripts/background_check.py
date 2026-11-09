"""Out-of-time background check: fast chain without / with the background model against the data reference.

    python3 background_check.py DATAREF.json NOBG_r3.json BG_r3.json [MORE_r3.json ...] [--out OUT.json]

Per sector slot: out-of-time rate per event per us and the oot_pe shape term; then the total objective of each arm.
Inputs are reco-r3 --emit JSONs (response_state) or crvresponse accumulators.
"""
import argparse
import json

import numpy as np

from crvfastsim.objective import crvresponse as R


def state(path):
    j = json.load(open(path))
    return j.get("response_state", j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataref")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ref = state(a.dataref)
    rr = {s: ref["oot_n"][s] / (ref["exposure_ns"] * 1e-3) for s in R.SLOTS}
    out = {"data_oot_rate": rr, "arms": {}}
    print(f"{'arm':28s} " + " ".join(f"{s:>16s}" for s in R.SLOTS) + "   objective (shape oot_pe terms)")
    print(f"{'data':28s} " + " ".join(f"{rr[s]:16.4f}" for s in R.SLOTS))
    for p in a.arms:
        S = state(p)
        o = R.objective(S, ref)
        rs = {s: S["oot_n"][s] / (S["exposure_ns"] * 1e-3) for s in R.SLOTS}
        pe_terms = {s: o["terms"].get(f"oot_pe_{s}", {}).get("chi2ndf") for s in R.SLOTS}
        top = sorted(o["terms"].items(), key=lambda kv: -kv[1]["chi2ndf"])[:5]
        out["arms"][p] = {"oot_rate": rs, "oot_rate_ratio": {s: rs[s] / rr[s] for s in R.SLOTS}, "objective": o["objective"],
                          "sigma": o["sigma"], "oot_pe_terms": pe_terms, "top_terms": {k: v["chi2ndf"] for k, v in top}}
        print(f"{p.split('/')[-1][:28]:28s} " + " ".join(f"{rs[s]:8.4f} ({rs[s] / rr[s]:4.2f})" for s in R.SLOTS)
              + f"   {o['objective']:7.2f} +- {o['sigma']:.2f}  ("
              + ", ".join(f"{s} {pe_terms[s]:.1f}" for s in R.SLOTS if pe_terms[s] is not None) + ")")
        print(f"{'':28s} top terms: " + ", ".join(f"{k} {v['chi2ndf']:.1f}" for k, v in top))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
