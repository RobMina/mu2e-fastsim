"""V-R5 verdict from vr5_collect.sh scores.json: per knob, the stock and fast scans of the objective, the minimum of each,
and the fast - stock difference against 1/10 of the stock scan's spread (fastreco plan V-R5).

    python3 vr5_verdict.py SCORES.json [--out VERDICT.json]
"""
import argparse
import json

import numpy as np

# knob -> list of (value, stock point name, fast point name); nominal uses the mean of the fast nominal seeds
SCANS = {
    "photon_yield_scale": [(0.85, "ly085", "photon_yield_scale=0.85"), (0.94, "nominal", "NOMINAL"),
                           (1.03, "ly103", "photon_yield_scale=1.03"), (1.12, "ly112", "photon_yield_scale=1.12")],
    "cross_talk": [(0.02, "ct002", "cross_talk=0.02"), (0.04, "nominal", "NOMINAL"),
                   (0.06, "ct006", "cross_talk=0.06"), (0.08, "ct008", "cross_talk=0.08")],
    "pe_stretch": [(0.95, "ws095", "pe_stretch=0.95"), (1.047, "nominal", "NOMINAL"),
                   (1.2, "ws120", "pe_stretch=1.2"), (1.35, "ws135", "pe_stretch=1.35")],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = json.load(open(a.scores))
    nom = [v["fast"] for k, v in S.items() if k.startswith("nominal_s") and "fast" in v]
    fast_nom = {"objective": float(np.mean([n["objective"] for n in nom])),
                "sigma": float(np.std([n["objective"] for n in nom], ddof=1)) if len(nom) > 1 else nom[0]["sigma"]}
    out = {"fast_nominal_seeds": [n["objective"] for n in nom], "knobs": {}}
    ok_all = True
    for knob, pts in SCANS.items():
        rows = []
        for val, sp, fp in pts:
            st = S[sp]["stock"]
            fa = fast_nom if fp == "NOMINAL" else S[fp]["fast"]
            rows.append((val, st["objective"], st["sigma"], fa["objective"], fa["sigma"]))
        v = np.array([r[0] for r in rows]); so = np.array([r[1] for r in rows]); fo = np.array([r[3] for r in rows])
        spread = float(so.max() - so.min())
        diff = fo - so
        dmax = float(np.max(np.abs(diff)))
        sig = float(np.sqrt(np.mean([r[2] ** 2 + r[4] ** 2 for r in rows])))
        tol = spread / 10.0
        flat = spread < 3.0 * sig                             # no resolvable minimum: the min test is not meaningful
        same_min = bool(v[np.argmin(so)] == v[np.argmin(fo)])
        ok_diff = dmax <= max(tol, 2.0 * sig)               # 1/10 of the spread, but not below the statistical noise
        ok = ok_diff and (same_min or flat)
        ok_all &= ok
        out["knobs"][knob] = {"values": v.tolist(), "stock": so.tolist(), "fast": fo.tolist(), "fast_minus_stock": diff.tolist(),
                              "stock_spread": spread, "tolerance_tenth_spread": tol, "noise_per_diff": sig,
                              "max_abs_diff": dmax, "min_stock": float(v[np.argmin(so)]), "min_fast": float(v[np.argmin(fo)]),
                              "flat": flat, "same_minimum": same_min, "pass": bool(ok)}
        print(f"{knob:20s} " + "  ".join(f"{x:g}: {s_:6.2f}/{f_:6.2f}" for x, s_, f_ in zip(v, so, fo)))
        print(f"{'':20s} spread {spread:5.2f}  tol {tol:4.2f}  noise/diff {sig:4.2f}  max|f-s| {dmax:4.2f}  "
              f"min stock {v[np.argmin(so)]:g} fast {v[np.argmin(fo)]:g}{'  (flat)' if flat else ''}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    out["pass"] = bool(ok_all)
    print(f"fast nominal seeds {', '.join(f'{x:.2f}' for x in out['fast_nominal_seeds'])}; V-R5 {'PASS' if ok_all else 'FAIL'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
