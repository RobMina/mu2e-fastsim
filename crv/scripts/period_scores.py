"""Objective per calibration period with the MC-side time calibration: fast MC (r3 JSONs, each with its period's
--time-shift) against that period's uncorrected data, with and without excluded terms side by side.

    python3 period_scores.py --arm P3 MC_P3.json "data/*124336*.json data/*124571*.json" \
        --arm P4 MC_P4.json "data/*125026*.json" [--exclude dt_cluster_T2@P3 ...] [--out OUT.json]

--exclude TERM@PERIOD leaves TERM out of that period's total (both totals are printed). Arms with the same period
label are all scored (e.g. module-only and module+FEB calibrations).
"""
import argparse
import glob
import json

from crvfastsim.objective import crvresponse as R

TIMING = ("dt_cluster_T1", "dt_cluster_T2", "dt_cluster_TAG", "tspread_EX", "tspread_T1", "tspread_T2", "tspread_TAG")


def state(path):
    j = json.load(open(path))
    return j.get("response_state", j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=3, action="append", metavar=("PERIOD", "MC", "DATA"), required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    excl = {}
    for x in a.exclude:
        t, p = x.split("@")
        excl.setdefault(p, []).append(t)
    out, refs = [], {}
    for period, mc, data in a.arm:
        if data not in refs:
            refs[data] = R.merge_states([state(f) for f in sorted({f for g in data.split() for f in glob.glob(g)})])
        o = R.objective(state(mc), refs[data], exclude=excl.get(period, ()))
        row = {"period": period, "mc": mc, "objective": o["objective"], "sigma": o["sigma"],
               "objective_all": o["objective_all"], "excluded": o["excluded"],
               "timing": {k: o["terms"][k]["chi2ndf"] for k in TIMING if k in o["terms"]},
               "top": {k: v["chi2ndf"] for k, v in sorted(o["terms"].items(), key=lambda kv: -kv[1]["chi2ndf"])[:6]}}
        out.append(row)
        ex = f"  (all terms {row['objective_all']:.2f}; excluded {', '.join(row['excluded'])})" if row["excluded"] else ""
        print(f"{period:4s} {mc.split('/')[-1]:28s} objective {row['objective']:6.2f} +- {row['sigma']:.2f}{ex}")
        print("     timing: " + ", ".join(f"{k.replace('dt_cluster_', 'dt').replace('tspread_', 'ts')} {v:.1f}"
                                         for k, v in row["timing"].items()))
        print("     top: " + ", ".join(f"{k} {v:.1f}" for k, v in row["top"].items()))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
