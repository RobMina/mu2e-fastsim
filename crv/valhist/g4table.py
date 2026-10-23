"""Retype G4 CrvStep tables (CrvStepTableDumper) to the fast-table column types CrvStepsFromTable reads, all steps and
particles kept: arm A0g (G4 steps injected on the gen stream, the same event set as the fast-step arms).

    python -m valhist.g4table crvsteps.*.root --outdir DIR      (writes g4all.<subrun>.root per input)
"""
import argparse
import os

import numpy as np

from crvfastsim import tables

STEP_INT = ("nstep", "bar", "sp")
PART_INT = ("key", "parent", "isprim")


def retype(steps, parts):
    s, p = steps.copy(), parts.copy()
    for c in s.columns:
        s[c] = s[c].astype(np.uint32 if c in tables.KEY else np.int64 if c in STEP_INT else np.float64)
    if "nstep" not in s.columns:
        s["nstep"] = s.groupby(tables.KEY, sort=False).bar.transform("size").astype(np.int64)
    for c in p.columns:
        p[c] = p[c].astype(np.uint32 if c in tables.KEY else np.int64 if c in PART_INT else
                           np.int32 if c == "pdg" else np.float64)
    return s, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    for f in tables.expand(a.inputs):
        s, p = retype(tables.read(f, "steps"), tables.read(f, "particles"))
        out = os.path.join(a.outdir, os.path.basename(f).replace("crvsteps.", "g4all."))
        tables.write(out, s, p, as_ttree=True)
        print(f"{f}: {len(s):,} steps, {len(p):,} particles -> {out}", flush=True)


if __name__ == "__main__":
    main()
