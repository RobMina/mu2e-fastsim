"""The CrvStep-table interchange format: readers for the CrvStepTableDumper trees (G4 reference) and the fast sim.

Trees live in the TFileService directory ``dump/``: ``steps`` (one row per CrvStep), ``particles`` (SimParticle),
``gen`` (CRY primary, every generated event in the gen.* stream) and ``subruns`` (CosmicLivetime). Column meanings are
in README.md section 2. Events are keyed by (run, subrun, event) in both arms.
"""
import glob

import numpy as np
import pandas as pd
import uproot

KEY = ["run", "subrun", "event"]
DIR = "dump"
PRIMARY_MUON_PDG = 13


def expand(inputs):
    out = []
    for p in inputs:
        out += sorted(glob.glob(p)) or [p]
    return out


def read(path, tree, columns=None):
    with uproot.open(path) as f:
        return f[f"{DIR}/{tree}"].arrays(columns, library="pd")


def read_many(paths, tree, columns=None):
    return pd.concat([read(p, tree, columns) for p in expand(paths)], ignore_index=True)


def livetime(paths):
    """Total CosmicLivetime [s] over the subruns of the given files."""
    s = read_many(paths, "subruns")
    return float(s.drop_duplicates(["run", "subrun"]).livetime.sum())


def primary_muon_crossings(steps, parts, sector_of_bar):
    """Sum the CrvSteps of each event's primary muon per bar: one row per (event, bar) crossing.

    The primary muon is the SimParticle with isprim == 1 and |pdg| == 13 (truth-particle identity, never time or
    distance). G4 can split one bar crossing into several CrvSteps (a new step after each boundary exit), so path length
    and visible energy are summed and the earliest start time kept.
    """
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == PRIMARY_MUON_PDG), KEY + ["key"]]
    s = steps.merge(prim.rename(columns={"key": "sp"}), on=KEY + ["sp"], how="inner")
    g = (s.groupby(KEY + ["bar"], sort=False)
          .agg(edep=("edep", "sum"), len=("len", "sum"), t0=("t0", "min"), nsteps=("edep", "size"))
          .reset_index())
    g["sector"] = np.asarray(sector_of_bar)[g.bar.to_numpy()]
    return g


def write(path, steps, particles, gen=None, subruns=None, as_ttree=False):
    """Write fast-sim tables in the CrvStepTableDumper layout (directory dump/), readable by every analysis here."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with uproot.recreate(path) as f:
        for name, df in (("steps", steps), ("particles", particles), ("gen", gen), ("subruns", subruns)):
            if df is not None and len(df):
                cols = {c: df[c].to_numpy() for c in df.columns}
                if as_ttree:   # uproot >= 5.7 writes RNTuple by default; art/ROOT readers here need a TTree
                    f.mktree(f"{DIR}/{name}", {c: v.dtype for c, v in cols.items()})
                    f[f"{DIR}/{name}"].extend(cols)
                else:
                    f[f"{DIR}/{name}"] = cols
    print(f"wrote {path}")
