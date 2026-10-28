"""Independently validate the exact input population of a completed additive held-out report.

Legacy tables do not prove an art full-stream role. This certificate checks complete
expected event-number coverage, generator payload equality, ancestry, independence
and hashes, and leaves producer/configuration admission explicitly unverified.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables
from . import descendant_audit as A
from . import descendant_inputs as I
from . import shower_uncertainty as U


def contiguous_census(gen, subrun, expected):
    if gen.duplicated(U.KEY + ["idx"]).any():
        raise ValueError("duplicate generator identity")
    ids = gen[U.KEY].drop_duplicates().sort_values(U.KEY)
    if (not ids.run.eq(1400).all() or not ids.subrun.eq(subrun).all() or
            not np.array_equal(ids.event.to_numpy(), np.arange(1, expected + 1))):
        raise ValueError("incomplete expected generated identities")
    return ids


def run(report, out, expected_events_per_shard=50000):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    result = json.loads(Path(report).read_text())
    meta, index, _, _, _ = U.load_cache(result["cache"])
    if A.digest(Path(result["cache"]) / "cache.json") != result["cache_sha256"]:
        raise ValueError("cache changed")
    U.verify(result["input_sha256"])
    U.verify(meta["input_sha256"])
    geom = geometry.from_offline_config(meta["geometry"])
    train = U.event_index(pd.DataFrame(index["donor_events"], columns=U.KEY))
    for path in meta["gen"]:
        train = train.union(U.event_index(tables.read(path, "gen", U.KEY).drop_duplicates()))
    files = list(result["input_sha256"])
    inventory, seen = [], set()
    for row in result["shards"]:
        k = row["shard"]
        gp = [p for p in files if Path(p).name == f"gen.001400_{k:08d}.root"]
        rp = [p for p in files if Path(p).name == f"crvsteps.001400_{k:08d}.root"]
        if len(gp) != 1 or len(rp) != 1:
            raise ValueError("ambiguous input pair")
        gen = tables.read(gp[0], "gen")
        census = contiguous_census(gen, k, expected_events_per_shard)
        st, pa = tables.read(rp[0], "steps"), tables.read(rp[0], "particles")
        I.population(census, st, pa, train, seen)
        _, ancestry = A.annotate(st, pa, geom)
        refgen = tables.read(rp[0], "gen")
        ix = pd.MultiIndex.from_frame(gen[U.KEY + ["idx"]]).get_indexer(pd.MultiIndex.from_frame(refgen[U.KEY + ["idx"]]))
        if (ix < 0).any():
            raise ValueError("reference generator outside population")
        pd.testing.assert_frame_equal(refgen.reset_index(drop=True), gen.iloc[ix][refgen.columns].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        inventory.append({"shard": k, "events": len(census), "ancestry": ancestry, "reference_generator_exact": True})
    U.verify(result["input_sha256"])
    total = sum(r["events"] for r in inventory)
    if total != result["events_inferred"]:
        raise ValueError("report exposure differs")
    certificate = {"report": report, "report_sha256": A.digest(report), "source_sha256": A.digest(__file__),
                   "events": total, "shards": inventory, "complete_expected_id_range": True,
                   "train_eval_overlap": 0, "cross_shard_overlap": 0, "strict_art_role_certified": False,
                   "seconds": time.monotonic() - start,
                   "limitation": "Expected 50k/shard comes from retained campaign record; no schema-v2 full-generator art redump or new producer compatibility proof."}
    with out.open("x") as stream:
        json.dump(certificate, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in certificate.items() if k != "shards"}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report")
    p.add_argument("--out", required=True)
    p.add_argument("--expected-events-per-shard", type=int, default=50000)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
