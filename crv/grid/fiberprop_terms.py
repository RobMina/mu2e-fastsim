"""Light-propagation knob check: per objective term, the shift of each point from nominal in the stock and fast arms
(scores.json of vr5_collect.sh with VR5_RUNS=fiberprop_v1). A knob is reproduced if the fast shift follows the stock
shift in the terms it moves.

    python3 fiberprop_terms.py SCORES.json [--min-shift 1.0] [--out TERMS.json]
"""
import argparse
import json

PAIRS = [("sp108", "speed_EX=1.08"), ("at058", "atten_T1=0.58"), ("at10_058", "atten_T10=0.58")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--min-shift", type=float, default=1.0, help="list terms whose stock or fast shift exceeds this")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = json.load(open(a.scores))
    nom_s = S["nominal"]["stock"]["terms"]
    nom_f = next(v["fast"]["terms"] for k, v in S.items() if k.startswith("nominal_s") and "fast" in v)
    out = {}
    for sp, fp in PAIRS:
        ts, tf = S[sp]["stock"]["terms"], S[fp]["fast"]["terms"]
        rows = []
        for k in sorted(ts):
            ds, df = ts[k] - nom_s[k], tf[k] - nom_f[k]
            if max(abs(ds), abs(df)) >= a.min_shift:
                rows.append((k, nom_s[k], ds, nom_f[k], df))
        out[sp] = [dict(term=r[0], stock_nominal=r[1], stock_shift=r[2], fast_nominal=r[3], fast_shift=r[4])
                   for r in rows]
        print(f"== {sp} (stock) vs {fp} (fast): terms moved by >= {a.min_shift}")
        print(f"   {'term':22s} {'stock nom':>10s} {'stock shift':>12s} {'fast nom':>10s} {'fast shift':>11s}")
        for k, n1, d1, n2, d2 in sorted(rows, key=lambda r: -abs(r[2])):
            print(f"   {k:22s} {n1:10.1f} {d1:+12.1f} {n2:10.1f} {d2:+11.1f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
