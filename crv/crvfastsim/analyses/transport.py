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
from mu2efastsim import world as world_mod

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
GDML_NAME = "geometry/mu2e_extracted_v04.gdml"  # under $CRVFS_DATA (geometry/README.md)


def default_gdml():
    data = os.environ.get("CRVFS_DATA")
    if not data:
        raise SystemExit("transport --stage C: pass --gdml or set CRVFS_DATA (source site.env)")
    return os.path.join(data, GDML_NAME)


def run(inputs, outdir, seed=1, stage="A", counters=None, gdml=None, world_mods=None, **kw):
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    world = None
    if stage == "C":
        names = []
        world = world_mod.load(gdml or default_gdml(), y_min=tr.Y_MSC_PLANE, names=names)
        if world_mods:                 # geometry sensitivity: named edits of the world (trkfastsim.worldmods)
            from trkfastsim import worldmods
            world, names = worldmods.apply(world_mods, world, names)
            print(f"# world modification: {world_mods}", flush=True)
        world = [p for p in world if p[0] != "G4_POLYSTYRENE"]
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), i])
        gen = tables.read(f, "gen")
        sub = tables.read(f, "subruns")
        steps, parts = tr.transport(gen, geom, rng, stage=stage, world=world)
        name = os.path.basename(f).replace("gen.", f"fast{stage}.")
        tables.write(os.path.join(outdir, name), steps, parts, gen, sub, as_ttree=True)
        print(f"# {f}: {parts.shape[0]:,} primary muons, {steps[tables.KEY].drop_duplicates().shape[0]:,} with CRV "
              f"crossings, {len(steps):,} crossings", flush=True)
