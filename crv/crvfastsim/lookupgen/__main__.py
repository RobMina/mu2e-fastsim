"""CLI of the lookup-table generator.

  run      counts for a bin set of one sim type -> .npz (bins, gen, ht, he, stats)
  compare  .npz against a reference table (bin by bin) -> JSON summary
  assemble .npz pieces covering every bin of the three sim types -> LookupTable file
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import geometry as G
from . import materials as M
from .writer import KEYS


def counter_from(a):
    if a.section == "stock":
        return G.stock(a.length, a.reflector)
    t, w = (float(x) for x in a.section.split("x"))
    return G.design(t, w, a.length, a.reflector, fiber_diameter=a.fiber_diameter)


def parse_bins(spec, n):
    if spec in (None, "all"):
        return np.arange(n)
    if ":" in spec:
        parts = [int(x) if x else None for x in spec.split(":")]
        return np.arange(n)[slice(*parts)]
    return np.array([int(x) for x in spec.split(",")])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m crvfastsim.lookupgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "assemble"):
        p = sub.add_parser(name)
        p.add_argument("--section", default="stock", help="'stock' or THICKNESSxWIDTH in mm (design counter)")
        p.add_argument("--fiber-diameter", type=float, default=2.0)
        p.add_argument("--length", type=float, required=True)
        p.add_argument("--reflector", type=int, default=0)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="current", choices=("current", "v6"),
                       help="optical model: current CRVStandalone (default) or v6 (the 2019 code that made the "
                            "released tables; validates the transport against them)")
    r = sub.choices["run"]
    r.add_argument("--simtype", type=int, required=True)
    r.add_argument("--bins", default="all", help="a:b[:step], comma list, or all")
    r.add_argument("--photons", type=int, default=None, help="per bin (stock: 1e5 for 0/1, 1e4 for 2)")
    r.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--lanes", type=int, default=1 << 16)
    r.add_argument("--chunk-iters", type=int, default=2000)
    r.add_argument("--checkpoint", action="store_true", help="rewrite --out during the run (bins marked complete)")
    r.add_argument("--checkpoint-interval", type=float, default=0.0,
                   help="seconds between checkpoint writes (0: after every chunk)")
    r.add_argument("--max-steps", type=int, default=None,
                   help="stop chains at this many steps (default 2**19, the exact limit; see docs/lookupgen.md)")
    r.add_argument("--resume", action="store_true",
                   help="keep the complete bins of an existing --out and run only the others (same random streams)")
    sub.choices["assemble"].add_argument("pieces", nargs="+")
    sub.choices["assemble"].add_argument("--note", action="append", default=[],
                                         help="free-text line for the provenance file (repeatable)")
    c = sub.add_parser("compare")
    c.add_argument("npz")
    c.add_argument("reference", help="reference LookupTable file")
    c.add_argument("--nphot-ref", type=int, default=None)
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .run import generate
        ctr = counter_from(a)
        bins = parse_bins(a.bins, G.n_bins(ctr, a.simtype))
        nphot = a.photons or (10000 if a.simtype == 2 else 100000)
        prev, prev_stats = None, None
        if a.resume and os.path.exists(a.out):
            z = np.load(a.out)
            same = (int(z["simtype"]) == a.simtype and int(z["nphot"]) == nphot and str(z["model"]) == a.model
                    and json.loads(str(z["counter"])) == json.loads(json.dumps(ctr.__dict__)))
            if not same:
                raise SystemExit(f"{a.out}: different sim type, photons, model or counter; not resuming")
            keep = z["complete"] & np.isin(z["bins"], bins)
            prev = {k: z[k][keep] for k in KEYS}
            prev_stats = json.loads(str(z["stats"]))
            bins = np.setdiff1d(bins, prev["bins"])
            print(f"resume: {int(keep.sum())} complete bins kept, {len(bins)} to run", flush=True)

        def merged(res):
            if prev is None:
                return res
            out = dict(res)
            for k in KEYS:
                out[k] = np.concatenate([prev[k], np.asarray(res[k]).astype(prev[k].dtype)])
            out["complete"] = np.concatenate([np.ones(len(prev["bins"]), bool), res["complete"]])
            out["stats"] = dict(res["stats"], resumed_from=prev_stats)
            return out

        last = [0.0]

        def save(res, final=False):
            if not final and time.time() - last[0] < a.checkpoint_interval:
                return
            last[0] = time.time()
            res = merged(res)
            tmp = a.out + ".tmp.npz"
            np.savez_compressed(tmp, **{k: res[k] for k in KEYS}, complete=res["complete"], nphot=nphot,
                                simtype=a.simtype, model=a.model, stats=json.dumps(res["stats"]),
                                counter=json.dumps(ctr.__dict__))
            os.replace(tmp, a.out)

        if len(bins) == 0:
            print("resume: nothing left to run")
            return 0
        res = generate(ctr, a.simtype, bins, nphot, a.dtype, a.seed, a.lanes, a.chunk_iters,
                       log=lambda m: print(m, flush=True), checkpoint=save if a.checkpoint else None,
                       model=a.model, max_steps=a.max_steps)
        save(res, final=True)
        print(json.dumps(res["stats"]))
    elif a.cmd == "compare":
        from crvfastsim.response import lookup
        from .compare import compare
        z = np.load(a.npz)
        keep = z["complete"] if "complete" in z.files else np.ones(len(z["bins"]), bool)
        out = {k: z[k][keep] for k in KEYS}
        out["nphot"] = int(z["nphot"])
        st = int(z["simtype"])
        nref = a.nphot_ref or (10000 if st == 2 else 100000)
        L = lookup.load(os.path.basename(a.reference), directory=os.path.dirname(os.path.abspath(a.reference)))
        res = compare(L, st, out, nref, convention=M.MODELS[str(z["model"])].writer)
        res["stats"] = json.loads(str(z["stats"]))
        res["complete_bins"] = int(keep.sum())
        s = json.dumps(res, indent=1)
        if a.out:
            open(a.out, "w").write(s)
        print(s)
    elif a.cmd == "assemble":
        from . import writer as W
        ctr = counter_from(a)
        by = {0: [], 1: [], 2: []}
        prov = {0: [], 1: [], 2: []}
        for f in a.pieces:
            z = np.load(f)
            if str(z["model"]) != a.model:
                raise SystemExit(f"{f}: made with model {z['model']}, not {a.model}")
            if json.loads(str(z["counter"])) != json.loads(json.dumps(ctr.__dict__)):
                raise SystemExit(f"{f}: made for a different counter")
            if "complete" in z.files and not z["complete"].all():
                raise SystemExit(f"{f}: has incomplete bins")
            st = int(z["simtype"])
            by[st].append({k: z[k] for k in KEYS})
            prov[st].append(dict(W.piece_provenance(json.loads(str(z["stats"]))), file=os.path.basename(f),
                                 photons_per_bin=int(z["nphot"]), bins=int(len(z["bins"]))))
        parts = {st: W.merge_parts(v) for st, v in by.items()}
        open(a.out, "wb").write(W.table_bytes(ctr, parts, M.MODELS[a.model].writer))
        with open(a.out + ".provenance.json", "w") as fh:
            json.dump(W.table_provenance(ctr, a.model, prov, a.note), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
