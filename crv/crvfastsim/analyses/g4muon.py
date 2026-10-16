"""g4muon -- G4 reference CrvStep tables restricted to the primary muon's own steps, in the fast-table layout.

The fast sim's tier 1a writes only the primary muon (restricted loss; deltas above the production cut and the non-muon
component are tier 1b). Feeding these tables through CrvStepsFromTable gives the V2 arm that is like-for-like with
tier 1a, so V2 separates the response-path closure from the tier-1b deficit. Primary muon = isprim == 1 and
|pdg| == 13 (truth identity). Column types follow the fast tables (int64 ids, float64 values), which the art module reads.

    g4muon crvsteps.*.root --outdir DIR      (writes g4mu.<shard>.root per input)
"""
import os

import numpy as np

from .. import tables

STEP_INT = ("nstep", "bar", "sp")
PART_COLS = tables.KEY + ["key", "parent", "pdg", "isprim"]


def restrict(steps, parts):
    """(steps, particles) of the primary muon only, typed like crvfastsim.transport output."""
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == tables.PRIMARY_MUON_PDG), PART_COLS]
    prim = prim.drop_duplicates(tables.KEY)
    s = steps.merge(prim[tables.KEY + ["key"]].rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], how="inner")
    s = s.copy()
    s["nstep"] = s.groupby(tables.KEY, sort=False).bar.transform("size")
    for c in s.columns:
        if c in tables.KEY:
            s[c] = s[c].astype(np.uint32)
        elif c in STEP_INT:
            s[c] = s[c].astype(np.int64)
        else:
            s[c] = s[c].astype(np.float64)
    p = prim.copy()
    for c in ("key", "parent", "isprim"):
        p[c] = p[c].astype(np.int64)
    p["pdg"] = p.pdg.astype(np.int32)
    return s.reset_index(drop=True), p.reset_index(drop=True)


def run(inputs, outdir, **kw):
    for f in tables.expand(inputs):
        steps = tables.read(f, "steps")
        parts = tables.read(f, "particles", PART_COLS)
        s, p = restrict(steps, parts)
        name = os.path.basename(f).replace("crvsteps.", "g4mu.")
        tables.write(os.path.join(outdir, name), s, p, as_ttree=True)
        print(f"# {f}: {len(s):,} of {len(steps):,} steps are primary-muon, {s[tables.KEY].drop_duplicates().shape[0]:,}"
              f" events", flush=True)
