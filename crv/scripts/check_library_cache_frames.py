"""Prove attached-library cache identity and exact fixed-seed held-out step/particle frames."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import runconfig, tables, tier1b, tier1b_assets
from crvfastsim.analyses.context_audit import content_fingerprint
from crvfastsim.analyses.library_cache import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", nargs="+", type=int, default=list(range(22, 42, 2)))
    args = parser.parse_args()
    root = Path(args.inputs_root)
    geom = runconfig.get().geometry()
    spec = tier1b_assets.resolve("strict_v4", directory=root / "tier1b_strict_v2")
    start = time.perf_counter()
    reference = tier1b_assets._load_uncached(spec, geom)
    ref_s = time.perf_counter() - start
    start = time.perf_counter()
    candidate = load(spec, geom, args.cache, tier1b_assets._load_uncached)
    candidate_s = time.perf_counter() - start
    if content_fingerprint(reference) != content_fingerprint(candidate):
        raise ValueError("attached library contents differ")
    rows = []
    for shard in args.shards:
        suffix = f"001400_{shard:08d}.root"
        primary = root / "transport_tdsr4" / ("fastC." + suffix)
        st, pa = tables.read(str(primary), "steps"), tables.read(str(primary), "particles")
        gen = tables.read(str(root / "gen_even" / ("gen." + suffix)), "gen")
        result = []
        for lib in (reference, candidate):
            result.append(tier1b.sample(st, pa, gen, lib, geom, np.random.default_rng([20261027, shard]),
                                       showers=True, **spec.options))
        for a, b in zip(result[0][:2], result[1][:2]):
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        if result[0][2] != result[1][2]:
            raise ValueError("sampling diagnostics differ")
        rows.append({"shard": shard, "steps": len(result[0][0]), "particles": len(result[0][1]),
                     "exact_frames": True})
    with open(args.out, "x") as stream:
        json.dump({"all_exact": True, "library_fingerprint": content_fingerprint(reference),
                   "uncached_load_s": ref_s, "candidate_load_s": candidate_s, "shards": rows}, stream, indent=2)


if __name__ == "__main__":
    main()
