"""Held-out support-hole bounds without kinematic matching or altering the empirical sampler."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables
from crvfastsim.analyses.census import classify
from valhist import shower_uncertainty as U
from valhist.genassoc_campaign import _new_json
from valhist.library_variants import sha256


def run(cache, evaluation, out):
    meta, index, cells, _, _ = U.load_cache(cache)
    em = json.loads((Path(evaluation) / "evaluation.json").read_text())
    if em["cache_sha256"] != sha256(Path(cache) / "cache.json"):
        raise ValueError("unbound evaluation")
    hashes = U.fingerprint([Path(cache) / "cache.json", Path(evaluation) / "evaluation.json", __file__])
    geom = geometry.from_offline_config(meta["geometry"])
    edges = {k: np.asarray(v) for k, v in meta["edges"].items()}
    bank = pd.MultiIndex.from_frame(cells)
    ntemplates = np.bincount(index["cell"], minlength=len(cells))
    bounds = {"missing_bank": np.zeros(len(meta["names"])), "no_template": np.zeros(len(meta["names"]))}
    rows, inventories = [], []
    for record in em["shards"]:
        cand, ref, genpath = record["paths"]
        hashes.update(U.fingerprint([ref, genpath]))
        gen = U.bin_frame(tables.read(genpath, "gen"))
        ci = bank.get_indexer(pd.MultiIndex.from_frame(gen[U.BIN]))
        missing = ci < 0
        zero = missing.copy()
        zero[~missing] = ntemplates[ci[~missing]] == 0
        gen["missing_bank"] = missing; gen["no_template"] = zero
        # These are event-level G4 bounds, not assignments of a step to the unsupported primary.
        steps = tables.read(ref, "steps", U.KEY + ["bar", "sp", "edep", "y0", "z0"])
        parts = tables.read(ref, "particles", U.KEY + ["key", "parent", "pdg"])
        s = classify(steps, parts, geom.sector)
        if s.cls.eq("unknown").any():
            raise ValueError("unknown ancestry")
        s = s.loc[s.cls.str.startswith("nonmu")]
        eventkeys = U.event_index(s)
        item = {"shard": record["shard"]}
        for label, mask in (("missing_bank", missing), ("no_template", zero)):
            selected = U.event_index(gen.loc[mask, U.KEY].drop_duplicates())
            inside = eventkeys.isin(selected)
            values = U.features(s.loc[inside], np.zeros(int(inside.sum()), dtype=np.int64), 1, geom.sector, edges)
            vector = np.asarray(values.sum(axis=0)).ravel()
            bounds[label] += vector
            item[label] = {"primaries": int(mask.sum()), "events": len(selected), "event_level_nonmu_steps": int(inside.sum()),
                           "event_level_feature_bound": vector[:8].tolist()}
        rows.append(gen.loc[missing, U.KEY + ["idx", "pdg", "e", "apdg", "kebin"]])
        inventories.append(item)
    U.verify(hashes)
    result = {"missing_bank_primaries": pd.concat(rows, ignore_index=True).to_dict("records"),
              "event_bounds": {k: {n: float(v[j]) for j, n in enumerate(meta["names"][:8])} for k, v in bounds.items()},
              "shards": inventories, "input_sha256": hashes,
              "limitation": "All nonmuon G4 activity in events containing an unsupported primary is an upper bound, not primary attribution; no identity matching by time/position or energy repair."}
    _new_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("shards", "input_sha256")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache"); p.add_argument("evaluation"); p.add_argument("--out", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
