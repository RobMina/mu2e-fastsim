"""One-shard factorial controls: strict/default creation and retained payload identities.

No baseline or library is rewritten. Donor identity repair and placement are separate
factors; common seeds do not guarantee aligned nonmuon draws after library changes.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, tier1b as T
from crvfastsim.analyses.census import classify
from .descendant_audit import digest
from . import stepclosure as C
from . import steplayers as S


def showers(steps, parts, geom):
    tagged = classify(steps, parts, geom.sector)
    cols = [c for c in steps if c != "nstep"]
    return tagged.loc[tagged.cls.str.startswith("nonmu"), cols].sort_values(
        tables.KEY + ["sp", "bar", "t0"], kind="stable").reset_index(drop=True)


def build(transport, library, old_default, old_ccp, strict_ccp, out, seed):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    inputs = [transport, library, old_default, old_ccp, strict_ccp]
    hashes = {str(p): digest(p) for p in inputs + [__file__, T.__file__]}
    geom = geometry.from_offline_config(geometry.find_counters(S.COUNTERS))
    st, pa = tables.read(transport, "steps"), tables.read(transport, "particles")
    gen, sub = tables.read(transport, "gen"), tables.read(transport, "subruns")
    lib = T.read_library(library)
    steps, parts, info = T.sample(st, pa, gen, lib, geom, np.random.default_rng([seed, 1000]))
    del lib
    oldst, oldpa = S._read(old_default, None)
    pd.testing.assert_frame_equal(C.primary_rows(steps, parts, descendants=True),
                                  C.primary_rows(oldst, oldpa, descendants=True), check_exact=True, check_dtype=False)
    newst, newpa = S._read(strict_ccp, None)
    pd.testing.assert_frame_equal(showers(steps, parts, geom), showers(newst, newpa, geom),
                                  check_exact=True, check_dtype=False)
    oldcst, oldcpa = S._read(old_ccp, None)
    pd.testing.assert_frame_equal(C.primary_rows(oldcst, oldcpa, descendants=True),
                                  C.primary_rows(newst, newpa, descendants=True), check_exact=True, check_dtype=False)
    pd.testing.assert_frame_equal(showers(oldst, oldpa, geom), showers(oldcst, oldcpa, geom),
                                  check_exact=True, check_dtype=False)
    for p, sha in hashes.items():
        if digest(p) != sha:
            raise ValueError(f"input drift: {p}")
    out.mkdir(parents=True)
    dest = out / Path(transport).name.replace("fastC.", "fast1b.")
    tables.write(str(dest), steps, parts, gen, sub, as_ttree=True)
    result = {"input_sha256": hashes, "output": str(dest), "output_sha256": digest(dest), "seed": seed,
              "muon_family_identity_within_placement": True, "shower_identity_within_library": True,
              "sampler": info, "seconds": time.monotonic() - start,
              "arms": {"old_default": old_default, "old_ccp": old_ccp, "strict_default": str(dest), "strict_ccp": strict_ccp}}
    (out / "controls.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("transport", "library", "old-default", "old-ccp", "strict-ccp", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, required=True)
    build(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
