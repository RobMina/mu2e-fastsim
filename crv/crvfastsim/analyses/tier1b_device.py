"""Opt-in device tier1b-apply caller; the NumPy analysis caller remains untouched."""
import os

import numpy as np

from .. import tables, tier1b_assets
from ..tier1b_jax import DeviceRandom, DeviceTier1b
from .tier1b import _geom


def apply(inputs, outdir, seed=1, counters=None, tier1b_set=None, tier1b_dir=None, library=None, **kw):
    if library or (tier1b_set is not None and tier1b_set not in ("strict_v3", "strict_v4")):
        raise ValueError("device tier1b-apply supports strict_v3/strict_v4 asset sets only")
    flags = ("crossing_context", "contain_steps", "point_placement", "mirror_offsets", "translate_offsets",
             "topology_sidecar", "template_sidecar", "unvisited_layers", "near_miss", "chord_placement", "follow_muon")
    if any(kw.get(k) for k in flags):
        raise ValueError("device tier1b-apply takes the asset set's own options, not custom placement flags")
    os.makedirs(outdir, exist_ok=True)
    geom = _geom(counters)
    spec = tier1b_assets.resolve(tier1b_set, directory=tier1b_dir)
    dev = DeviceTier1b(tier1b_assets.load(spec, geom), geom)
    for i, path in enumerate(tables.expand(inputs)):
        st, pa, gen = (tables.read(path, tree) for tree in ("steps", "particles", "gen"))
        stream = np.random.default_rng([int(seed), 1000 + i])
        rng = DeviceRandom(int(stream.integers(0, 2**32, dtype=np.uint32)))
        steps, parts, info = dev.sample(st, pa, gen, rng, **spec.options)
        name = os.path.basename(path)
        for stage in ("fastC.", "fastB.", "fastA."):
            name = name.replace(stage, "fast1b.")
        tables.write(os.path.join(outdir, name), steps, parts, gen, tables.read(path, "subruns"), as_ttree=True)
        print(f"# {path}: {info}; boundary={dev.last}", flush=True)
