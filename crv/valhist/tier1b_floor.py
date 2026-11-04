"""Seed floor of the tier-1b step histograms: the NumPy sampler twice on the same transport tables, two seeds.

Arm "numpy" draws with seed [shard, 1000] (the seed of the device audit's NumPy arm), arm "numpy_reseed" with
[shard, SEED]. Writes the stephist accumulators of both arms, the paired second moments and the stephist comparison
in the layout of `valhist.device_tier1b` summaries, so `valhist.deck_rule` reads either.

    python -m valhist.tier1b_floor --counters C --library-dir D --transport T --out DIR --shards 22 24 ...
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from crvfastsim import geometry, tables, tier1b as T, tier1b_assets
from . import descendant_audit as A, stephist as H, steplayers as S


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(out / "summary.json")
    geom = geometry.from_offline_config(args.counters)
    spec = tier1b_assets.resolve(args.tier1b_set, directory=args.library_dir)
    lib = tier1b_assets.load(spec, geom)
    hashes = {str(p): A.digest(p) for p in [*spec.paths(), T.__file__, __file__]}
    edges = S.booking_edges()
    names = ("numpy", "numpy_reseed")
    hist, pair, records = {n: {} for n in names}, {}, []
    for shard in args.shards:
        path = str(Path(args.transport) / f"fastC.{S.shard(shard)}.root")
        st, pa, gen = (tables.read(path, tree) for tree in ("steps", "particles", "gen"))
        hashes[path] = A.digest(path)
        events = gen[tables.KEY].drop_duplicates()
        record, idx = dict(shard=shard, events=len(events), input_steps=len(st)), {}
        for name, seed in zip(names, (1000, args.seed)):
            tick = time.perf_counter()
            steps, parts, _ = T.sample(st, pa, gen, lib, geom, np.random.default_rng([shard, seed]), **spec.options)
            hh, idx[name] = H.fill(steps[H.STEP_COLS], parts, events, edges)
            H.add(hist[name], hh)
            record[name] = dict(seconds=time.perf_counter() - tick, rows=len(steps))
        H.add_paired(pair, H.paired(idx["numpy"], idx["numpy_reseed"], edges))
        records.append(record)
        print(json.dumps(record), flush=True)
    comparison = H.compare(hist["numpy"], hist["numpy_reseed"], pair)
    for path, old in hashes.items():
        if A.digest(path) != old:
            raise ValueError(f"input/code changed during run: {path}")
    result = dict(records=records, tier1b=spec.describe(), hashes=hashes, seeds=[1000, args.seed],
                  comparison=comparison, hists={n: H.to_json(h) for n, h in hist.items()},
                  paired={k: v.tolist() for k, v in pair.items()})
    (out / "summary.json").write_text(json.dumps(result, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("counters", "library-dir", "transport", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--shards", type=int, nargs="+", default=[22])
    p.add_argument("--tier1b-set", choices=("strict_v5",), default="strict_v5")
    p.add_argument("--seed", type=int, default=2000, help="second seed component of the reseeded arm")
    run(p.parse_args())


if __name__ == "__main__":
    main()
