"""Event-clustered downstream comparison of retained audited chain NPZ outputs.

Same generated events permit paired errors despite independent transport and
shower streams. This checks counts and measured pulse/cluster distributions;
it is not a replacement for the collaboration's full DQM/deck certification.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import stephist as sh


BOOK = {"cluster_pe": (100, 0., 1000.), "pulse_pe": (100, 0., 200.),
        "cluster_time": (100, -100., 100000.), "pulse_time": (100, -100., 100000.),
        "cluster_y": (100, -1000., 5000.), "cluster_z": (100, -20000., 20000.)}


def collect(path):
    with np.load(path, allow_pickle=False) as d:
        universe = d["generated_packed"]
        result = {"universe": universe}
        for k in ("clusters", "pulses"):
            prefix = "cl_event" if k == "clusters" else "pulses__ev"
            arrays = [d[n] for n in d.files if n.endswith("__"+prefix)]
            ev = np.concatenate(arrays) if arrays else np.zeros(0, np.int64)
            result[k] = np.bincount(np.searchsorted(universe, ev), minlength=len(universe))
        mapping = {"cluster_pe":"cl_pe", "pulse_pe":"pulses__pe", "cluster_time":"cl_start", "pulse_time":"pulses__time",
                   "cluster_y":"cl_pos", "cluster_z":"cl_pos"}
        for key, leaf in mapping.items():
            chunks = [n[:-len(leaf)] for n in d.files if n.endswith("__"+leaf)]
            values = np.concatenate([d[c+leaf] for c in chunks]) if chunks else np.zeros(0)
            if key in ("cluster_y", "cluster_z"):
                values = values[:, 1 if key.endswith("y") else 2]
            evleaf = "pulses__ev" if key.startswith("pulse_") else "cl_event"
            ev = np.concatenate([d[c+evleaf] for c in chunks]) if chunks else np.zeros(0, np.int64)
            nb, lo, hi = BOOK[key]
            bins = sh.root_bin(values, nb, lo, hi)
            idx = np.searchsorted(universe, ev)*(nb+2) + bins
            matrix = np.bincount(idx, minlength=len(universe)*(nb+2)).reshape(len(universe), nb+2)
            result[key] = matrix
        return result


def exact_outputs(a, b):
    ra, rb = [json.loads((Path(p)/"benchmark.json").read_text()) for p in (a,b)]
    fa, fb = [r["passes"][1]["files"] for r in (ra,rb)]
    return bool(len(fa) == len(fb) and all(x["input"] == y["input"] and x["fingerprint"] == y["fingerprint"]
        for x,y in zip(fa,fb)))


def run(args):
    roots = [Path(p) for p in (args.numpy, args.jax)]
    reports = [json.loads((p/"benchmark.json").read_text()) for p in roots]
    if not all(r.get("repeat_fingerprints_identical") and r.get("context_audits_passed") for r in reports):
        raise ValueError("audited exact-repeat report required")
    files = [r["passes"][1]["files"] for r in reports]
    if len(files[0]) != len(files[1]):
        raise ValueError("unequal input population")
    accum = {}
    for a,b in zip(files[0], files[1]):
        if a["input"] != b["input"]:
            raise ValueError("input order mismatch")
        sides = [collect(root/"pass_001"/rec["npz"]) for root,rec in zip(roots,(a,b))]
        if not np.array_equal(sides[0]["universe"], sides[1]["universe"]):
            raise ValueError("generated population mismatch")
        for key in sides[0]:
            if key == "universe":
                continue
            av,bv = [s[key].astype(float) for s in sides]
            vals = (av.sum(0), bv.sum(0), (av*av).sum(0), (bv*bv).sum(0), ((bv-av)**2).sum(0))
            accum[key] = tuple(x+y for x,y in zip(accum[key],vals)) if key in accum else vals
    result = {}
    for key,(a,b,aa,bb,dd) in accum.items():
        if np.ndim(a) == 0:
            result[key] = dict(numpy=float(a), jax=float(b), relative=float(b/a-1), paired_se=float(np.sqrt(dd)/a),
                               independent_se=float(np.sqrt(aa+bb)/a))
        else:
            score,ndf = sh.chi2(a,b,aa+bb,0.)
            pair,_ = sh.chi2(a,b,dd,0.)
            result[key] = dict(clustered_chi2_ndf=score, paired_chi2_ndf=pair, ndf=ndf,
                               diagnostic_flag=None if score is None else bool(score > sh.FLAG))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--numpy", required=True)
    p.add_argument("--jax", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exact", nargs=2, help="also compare audited per-input fingerprints of two execution schedules")
    args = p.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if args.exact:
        print("schedule_exact", exact_outputs(*args.exact))
    run(args)


if __name__ == "__main__":
    main()
