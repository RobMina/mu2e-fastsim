"""Trace the retained descendant sampler's other-bar migration; no physics correction.

Run with the original shard seeds and CCP options. Source labels are library crossing
labels, while target membership uses the receiving muon's true identity and all bars.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, tier1b as T
from crvfastsim.analyses.census import SECTOR_GROUP
from . import descendant_audit as A
from . import steplayers as S


VIEWS = {"migration": ["source_group", "target_group", "source_kind", "target_kind"],
         "layer": ["target_group", "layer", "source_kind", "target_kind"]}


def features(fx, trace, geom):
    """Label every sampled row, including lost rows; own-bar membership is ancestry keyed."""
    i = trace["xi"]
    xb = fx.bar.to_numpy(dtype=np.int64)[i]
    target = trace["target_bar"]
    good = trace["kept"]
    d = fx.iloc[i][tables.KEY + ["sp"]].reset_index(drop=True)
    d["bar"] = target
    own = pd.MultiIndex.from_frame(d[tables.KEY + ["sp", "bar"]]).isin(
        pd.MultiIndex.from_frame(fx[tables.KEY + ["sp", "bar"]]))
    sector = SECTOR_GROUP[geom.sector]
    d["source_group"] = sector[xb]
    d["target_group"] = np.where(good, sector[np.maximum(target, 0)], "lost")
    d["source_kind"] = np.where(trace["source_samebar"], "samebar", "otherbar")
    d["target_kind"] = np.select([~good, target == xb, own], ["lost", "anchor", "another_own_bar"],
                                 default="otherbar")
    d["layer"] = np.where(good, T.layer_rank(geom)[np.maximum(target, 0)], -1)
    d["w"] = np.asarray(trace["source_edep"], dtype=np.float64)
    d["ev"] = d.groupby(tables.KEY, sort=False).ngroup()
    return d


def audit(library, transport, retained, shard_list, out, counters=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if len(shard_list) != len(set(shard_list)):
        raise ValueError("duplicate shard")
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    started = time.monotonic()
    lib = T.read_library(library)
    manifest = {str(library): A.digest(library), T.__file__: A.digest(T.__file__), __file__: A.digest(__file__)}
    acc, inventory = {}, []
    for k in shard_list:
        tick = time.monotonic()
        path = transport.format(s=S.shard(k), k=k)
        fx = tables.read(path, "steps")
        rng = np.random.default_rng([k, 1000])
        trace = {}
        desc, dropped, clamped, why = T._desc_for_crossings(fx, lib, geom, rng, context=True, contain=True,
                                                           point=True, trace=trace)
        d = features(fx, trace, geom)
        acc = S._merge_views(acc, {name: S.accumulate(d.iloc[:0], d, cols) for name, cols in VIEWS.items()})
        evidence = {"shard": k, "transport": path, "dropped": dropped, "clamped": clamped, "placement": why}
        manifest[path] = A.digest(path)
        if retained:
            ref = retained.format(s=S.shard(k), k=k)
            steps, parts = S._read(ref)
            r, _ = A.annotate(steps, parts, geom)
            cols = tables.KEY + ["bar", "t0", "edep", "len", "x0", "y0", "z0", "x1", "y1", "z1",
                                 "px0", "py0", "pz0"]
            pd.testing.assert_frame_equal(desc[cols].sort_values(cols).reset_index(drop=True),
                                          r[cols].sort_values(cols).reset_index(drop=True), check_dtype=False,
                                          check_exact=True)
            evidence["retained_descendant_payload_exact"] = True
            manifest[ref] = A.digest(ref)
        evidence["seconds"] = time.monotonic() - tick
        inventory.append(evidence)
        print(json.dumps(evidence), flush=True)
    result = {"schema": 1, "diagnostic": "CCP donor-label to receiving-muon bar membership", "seed_rule": "[shard,1000]",
              "uncertainty": "fixed-library sampled trace; moments are event-clustered, not library variance",
              "hashes": manifest, "shards": inventory, "views": A.summarize(acc),
              "seconds": time.monotonic() - started}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"wrote {out}; {result['seconds']:.2f} s", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "transport", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--retained")
    p.add_argument("--shards", default="11-20")
    p.add_argument("--counters")
    a = vars(p.parse_args())
    a["shard_list"] = S.shards(a.pop("shards"))
    audit(**a)


if __name__ == "__main__":
    main()
