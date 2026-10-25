"""Select reproducible, disjoint source-file blocks from a completed Cat inventory, without storage queries."""
import argparse
import json
from pathlib import Path

import numpy as np

from .mixframes import MIXERS
from .replay import sha256


def select(names, per_species, block, seed):
    """One common seeded permutation per species; block slices have no overlapping filenames."""
    if per_species <= 0 or block < 0:
        raise ValueError("positive file count and nonnegative block required")
    rng = np.random.default_rng(seed)
    result = {}
    for species in MIXERS:
        available = sorted({n.strip() for n in names if n.strip().startswith(f"dts.mu2e.{species}.")})
        if len(available) < (block + 1) * per_species:
            raise ValueError(f"not enough inventory entries for {species}")
        order = rng.permutation(len(available))[block * per_species:(block + 1) * per_species]
        result[species] = [available[i] for i in order]
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inventory")
    ap.add_argument("--pnfs", required=True)
    ap.add_argument("--per-species", type=int, default=4)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="prior subset JSON manifests to exclude by basename")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() or out.with_suffix(".json").exists():
        raise ValueError("subset output exists")
    excluded = set()
    for path in a.exclude:
        prior = json.loads(Path(path).read_text())
        excluded.update(n for values in prior["selected"].values() for n in values)
    names = [n for n in Path(a.inventory).read_text().splitlines() if n.strip() not in excluded]
    selected = select(names, a.per_species, a.block, a.seed)
    rows = [f"{sp} {a.pnfs}/{sp}/steps.{name}.parquet\n" for sp, names in selected.items() for name in names]
    with out.open("x") as f:
        f.writelines(rows)
    out.with_suffix(".json").write_text(json.dumps({"seed": a.seed, "block": a.block, "selected": selected,
        "inventory_sha256": sha256(a.inventory), "per_species": a.per_species,
        "excluded_manifests": {p: sha256(p) for p in a.exclude}}, indent=2))


if __name__ == "__main__":
    main()
