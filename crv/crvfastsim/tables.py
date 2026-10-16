"""The CrvStep-table interchange format: the shared flat-table layout of mu2efastsim.tables plus the CRV ``steps`` tree.

Trees live in the TFileService directory ``dump/``: ``steps`` (one row per CrvStep), ``particles`` (SimParticle),
``gen`` (CRY primary, every generated event in the gen.* stream) and ``subruns`` (CosmicLivetime). Column meanings are
in README.md section 2. Events are keyed by (run, subrun, event) in both arms.
"""
import numpy as np

from mu2efastsim import tables as _core
from mu2efastsim.tables import DIR, KEY, PRIMARY_MUON_PDG, expand, livetime, primary_muons, read, read_many  # noqa: F401


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
    _core.write(path, {"steps": steps, "particles": particles, "gen": gen, "subruns": subruns}, as_ttree=as_ttree)
