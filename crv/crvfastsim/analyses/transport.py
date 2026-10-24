"""transport -- run the fast muon transport on CRY primaries and write CrvStep tables (crvfastsim.transport).

Input: gen.*.root tables (CrvStepTableDumper on the G4 gen stream: every generated event's CRY primaries and the
subrun live time). Output: one fast-sim table file per input, same layout, same (run, subrun, event) keys, so
`v1compare FAST --compare G4` matches events one to one. --stage A: air; B: air + MSC; C: + GDML world material.

    transport gen.001400_00000001.root [...] --outdir DIR [--seed N] [--stage A]
"""
import dataclasses
import os

import numpy as np

from .. import geometry, runconfig, tables
from .. import transport as tr
from mu2efastsim import world as world_mod

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"
GDML_NAME = "geometry/mu2e_extracted_v04.gdml"  # under $CRVFS_DATA (geometry/README.md)


def default_gdml():
    data = os.environ.get("CRVFS_DATA")
    if not data:
        raise SystemExit("transport --stage C: pass --gdml or set CRVFS_DATA (source site.env)")
    return os.path.join(data, GDML_NAME)


def run(inputs, outdir, seed=1, stage="A", counters=None, gdml=None, world_mods=None, msc_tail=False, tail_k=None,
        sliced_msc=False, first_muon_only=False, no_absorbers=False, run_config=None, **kw):
    rc = runconfig.get(run_config)
    if counters:
        rc = dataclasses.replace(rc, counters=counters)
    geom = rc.geometry()
    tagger = rc.tagger_bar(geom) if rc.name != runconfig.DEFAULT else None
    if rc.name != runconfig.DEFAULT and gdml is None:
        gdml = rc.gdml_path()
    world = None
    ab = None
    if stage == "C" and world_mods:    # geometry sensitivity: named edits of the world (trkfastsim.worldmods)
        names = []
        world = world_mod.load(gdml or default_gdml(), y_min=tr.msc_plane(geom), names=names)
        from trkfastsim import worldmods
        world, names = worldmods.apply(world_mods, world, names)
        print(f"# world modification: {world_mods}", flush=True)
        world = [p for p in world if p[0] != "G4_POLYSTYRENE"]
        if not kw.get("unbanded_world"):
            world = tr.banded_world(world, tr.band_edges(tr.msc_plane(geom)))
        ab = None if no_absorbers else tr.absorber_slabs(gdml or default_gdml())
    elif stage == "C":
        world, ab = tr.load_world(gdml or default_gdml(), banded=not kw.get("unbanded_world", False),
                                  y_plane=tr.msc_plane(geom))
        ab = None if no_absorbers else ab
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), i])
        gen = tables.read(f, "gen")
        sub = tables.read(f, "subruns")
        steps, parts = tr.transport(gen, geom, rng, stage=stage, world=world, msc_tail=msc_tail, tail_k=tail_k,
                                    sliced_msc=sliced_msc, all_muons=not first_muon_only, absorbers=ab, tagger_bar=tagger)
        name = os.path.basename(f).replace("gen.", f"fast{stage}.")
        tables.write(os.path.join(outdir, name), steps, parts, gen, sub, as_ttree=True)
        print(f"# {f}: {parts.shape[0]:,} primary muons, {steps[tables.KEY].drop_duplicates().shape[0]:,} with CRV "
              f"crossings, {len(steps):,} crossings", flush=True)
