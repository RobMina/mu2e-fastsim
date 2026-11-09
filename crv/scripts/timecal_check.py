"""Objective per calibration period before and after the module time calibration (objective.timecal).

    python3 timecal_check.py MC_STATE.json --period P3 "raw/*124336*.json raw/*124571*.json" "corr/*.json" \
        --period P4 "raw/*125026*.json" "corr4/*.json" [--out OUT.json]

MC_STATE: an r3 --emit JSON (response_state) or an accumulator. For each period: the total objective and the timing
terms (dt_cluster_*, tspread_*) against the raw data and against the time-corrected data.
"""
import argparse
import glob
import json

from crvfastsim.objective import crvresponse as R

TERMS = ("dt_cluster_T1", "dt_cluster_T2", "dt_cluster_TAG", "tspread_EX", "tspread_T1", "tspread_T2", "tspread_TAG")


def state(path):
    j = json.load(open(path))
    return j.get("response_state", j)


def merged(spec):
    files = sorted({f for g in spec.split() for f in glob.glob(g)})
    return R.merge_states([state(f) for f in files]), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mc")
    ap.add_argument("--period", nargs=3, action="append", metavar=("NAME", "RAW", "CORR"), required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = state(a.mc)
    out = {}
    for name, raw, corr in a.period:
        res = {}
        for tag, spec in (("raw", raw), ("timecal", corr)):
            ref, n = merged(spec)
            o = R.objective(S, ref)
            res[tag] = {"files": n, "objective": o["objective"], "sigma": o["sigma"],
                        "timing": {k: o["terms"][k]["chi2ndf"] for k in TERMS if k in o["terms"]},
                        "top": {k: v["chi2ndf"] for k, v in sorted(o["terms"].items(), key=lambda kv: -kv[1]["chi2ndf"])[:6]}}
        out[name] = res
        print(f"== {name}: objective raw {res['raw']['objective']:.2f} -> timecal {res['timecal']['objective']:.2f} "
              f"({res['raw']['files']} files)")
        for k in TERMS:
            print(f"   {k:16s} {res['raw']['timing'].get(k, float('nan')):8.1f} -> {res['timecal']['timing'].get(k, float('nan')):8.1f}")
        print("   top after: " + ", ".join(f"{k} {v:.1f}" for k, v in res["timecal"]["top"].items()))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
