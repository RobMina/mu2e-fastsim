"""Hybrid device-stage chain from experimental device CRY; host table boundaries are explicit."""
import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np

from mu2efastsim.cosmic_cry import CosmicConfig
from mu2efastsim.cry_device import DeviceCosmicSource
from crvfastsim import chain, tier1b_assets
from crvfastsim.analyses.runchain import prepare
from crvfastsim.tier1b_jax import DeviceRandom, DeviceTier1b
from crvfastsim.transport_jax import DeviceTransport, transport


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    tick = time.perf_counter()
    context = prepare(calib=args.calib, deviation=args.deviation, gdml=args.gdml, jax=True,
                      tier1b_set="strict_v4", tier1b_dir=args.library_dir)
    p = context.payload
    dt = DeviceTransport(p["geom"], p["world"], p["ab"], p["tagger"], p["bar_cut"], range_out=True)
    ds = DeviceTier1b(p["lib"], p["geom"])
    options = tier1b_assets.resolve("strict_v4", directory=args.library_dir).options
    preparation = time.perf_counter() - tick
    records = []
    with DeviceCosmicSource(CosmicConfig.extracted(args.gdml), seed=args.seed) as source:
        for repeat in range(args.repeat):
            phases = {}
            start = time.perf_counter()
            gen, _, receipt = source.generate(args.events, subrun=repeat + 1)
            phases["generation_s"] = time.perf_counter() - start
            rng = np.random.default_rng([args.seed, repeat])
            t = time.perf_counter()
            st, pa = transport(gen, p["geom"], rng, prepared=dt, all_muons=True)
            phases["transport_s"] = time.perf_counter() - t
            t = time.perf_counter()
            st, pa, info = ds.sample(st, pa, gen, DeviceRandom(args.seed + repeat), **options)
            phases["tier1b_s"] = time.perf_counter() - t
            q = chain.charges(st, pa, p["geom"], p["cond"], p["cfg"], rng, tabs=p["tabs"], pmap=p["pmap"], timers=phases)
            d = chain.digis(q, p["cond"], p["cfg"], rng, timers=phases)
            t = time.perf_counter()
            w = chain.reco_waveforms(d, p["cond"])
            phases["waveform_pack_s"] = time.perf_counter() - t
            t = time.perf_counter()
            clusters, _ = chain.clusters(w, p["cond"], p["geom"], p["ccfg"], p["cfg"], timers=phases)
            phases["reco_s"] = time.perf_counter() - t
            types = np.concatenate([c["cl_type"] for c in clusters]) if clusters else np.zeros(0, int)
            record = dict(repeat=repeat, events=args.events, steps=len(st), clusters=len(types),
                          types={str(k): int((types == k).sum()) for k in np.unique(types)},
                          receipt=receipt, timers=phases, tier1b_boundary=ds.last,
                          elapsed_s=time.perf_counter()-start)
            records.append(record)
            print(json.dumps(record), flush=True)
    out.write_text(json.dumps(dict(preparation_s=preparation, records=records,
                                   configuration="strict_v4/range_out", boundary="GPU stage numerics; host generator/transport tables, shower/ancestry/response interfaces; not zero-copy"), allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("calib", "deviation", "gdml", "library-dir", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--events", type=int, default=10000)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=7712)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
