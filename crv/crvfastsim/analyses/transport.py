"""transport -- run the fast muon transport on CRY primaries and write CrvStep tables (crvfastsim.transport).

Input: gen.*.root tables (CrvStepTableDumper on the G4 gen stream: every generated event's CRY primaries and the
subrun live time). Output: one fast-sim table file per input, same layout, same (run, subrun, event) keys, so
`v1compare FAST --compare G4` matches events one to one. --stage A: air; B: air + MSC; C: + GDML world material.

    transport gen.001400_00000001.root [...] --outdir DIR [--seed N] [--stage A]
"""
import os

import numpy as np

from .. import geometry, tables
from .. import transport as tr
from .. import world as world_mod

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
DEFAULT_GDML = "/exp/mu2e/data/users/ram2aq/crv_fastsim/geometry/mu2e_extracted_v04.gdml"


def run(inputs, outdir, seed=1, stage="A", counters=None, gdml=None, **kw):
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    world = None
    if stage == "C":
        world = [p for p in world_mod.load(gdml or DEFAULT_GDML, y_min=tr.Y_MSC_PLANE) if p[0] != "G4_POLYSTYRENE"]
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), i])
        gen = tables.read(f, "gen")
        sub = tables.read(f, "subruns")
        steps, parts = tr.transport(gen, geom, rng, stage=stage, world=world)
        name = os.path.basename(f).replace("gen.", f"fast{stage}.")
        tables.write(os.path.join(outdir, name), steps, parts, gen, sub)
        print(f"# {f}: {parts.shape[0]:,} primary muons, {steps[tables.KEY].drop_duplicates().shape[0]:,} with CRV "
              f"crossings, {len(steps):,} crossings", flush=True)
