"""Fixed-library replica diagnosis of actual descendant placement (not donor uncertainty).

Each replica uses an independent explicit seed and the existing CCP sampler. No
steps/library/response model is altered. Stored pre/post samples retain covariance.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

from crvfastsim import geometry, tables, tier1b as T
from . import descendant_audit as A
from . import descendant_expectation as E
from . import descendant_placement as P
from . import steplayers as S


def totals(frame, column, categories, weight):
    """Sum sector/layer features in one array operation, excluding lost target rows."""
    m = frame[column].isin(categories).to_numpy() & frame.target_group.ne("lost").to_numpy()
    groups = np.select([frame.target_group.to_numpy() == g for g in T.GROUPS], range(4), -1)
    bins = groups[m] * 4 + frame.layer.to_numpy()[m]
    return np.bincount(bins, weights=frame.w.to_numpy()[m] if weight else None, minlength=16)


def post_features(frame):
    return np.r_[totals(frame, "target_kind", ["otherbar"], False),
                 totals(frame, "target_kind", ["otherbar"], True)]


def projection():
    p = np.zeros((8, 32))
    for i in range(8):
        p[i, i * 4:(i + 1) * 4] = 1
    return p


def summarize(before, after, ref):
    """Mean estimator error and sampler SD are different; expose cross-sector covariance."""
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 2 or len(before) < 2:
        raise ValueError("paired replica matrices with at least two draws required")
    project = projection()
    result = {}
    for name, values in (("pre", before), ("post", after), ("post_minus_pre", after - before)):
        sector = values @ project.T
        sd = sector.std(axis=0, ddof=1)
        result[name] = {"mean": values.mean(0).tolist(), "sector_mean": sector.mean(0).tolist(),
                        "sector_sampling_sd": sd.tolist(), "sector_mean_mc_se": (sd / np.sqrt(len(values))).tolist(),
                        "sector_covariance": np.cov(sector, rowvar=False).tolist(),
                        "sector_ratio_to_g4": (sector.mean(0) / (project @ ref)).tolist()}
    return result


def run(cache, transport, reference, shards, replicas, seed, out):
    if replicas < 2:
        raise ValueError("at least two replicas required")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tick = time.monotonic()
    root = Path(cache)
    meta = json.loads((root / "cache.json").read_text())
    for p, sha in meta["hashes"].items():
        if A.digest(p) != sha:
            raise ValueError(f"cache/input drift: {p}")
    geom = geometry.from_offline_config(meta["geometry"])
    lib = T.read_library(meta["library"])
    features = sparse.load_npz(root / "features.npz")
    pre, post, reference_total = np.zeros((replicas, 32)), np.zeros((replicas, 32)), np.zeros(32)
    old_pre, old_post = np.zeros(32), np.zeros(32)
    from .descendant_inputs import population, unique_shards
    donor_ids = pd.MultiIndex.from_frame(pd.read_csv(root / "donor_events.csv"))
    records, hashes, seen = [], {str(root / "cache.json"): A.digest(root / "cache.json"), __file__: A.digest(__file__)}, set()
    for k in unique_shards(shards):
        f, r = (p.format(s=S.shard(k), k=k) for p in (transport, reference))
        fx = tables.read(f, "steps")
        st, pa = S._read(r)
        population(tables.read(f, "gen", tables.KEY), fx, pa, donor_ids, seen)
        rv, _ = E.reference_features(st, pa, geom)
        reference_total += rv
        start = time.monotonic()
        for replica in range(-1, replicas):
            rng = np.random.default_rng([k, 1000] if replica < 0 else [seed, k, replica])
            trace = {}
            _, dropped, _, _ = T._desc_for_crossings(fx, lib, geom, rng, context=True, contain=True, point=True,
                                                     trace=trace)
            if dropped:
                raise ValueError("unexpected loss of sampled descendants")
            _, ix = np.unique(trace["xi"], return_index=True)
            bp = np.asarray(features[trace["xid"][ix]].sum(axis=0)).ravel()
            other = ~trace["source_samebar"]
            np.testing.assert_equal(bp[:16].sum(), int(other.sum()))
            np.testing.assert_allclose(bp[16:].sum(), trace["source_edep"][other].astype(float).sum(), rtol=1e-12)
            ap = post_features(P.features(fx, trace, geom))
            if replica < 0:
                old_pre += bp
                old_post += ap
            else:
                pre[replica] += bp
                post[replica] += ap
        records.append({"shard": k, "crossings": len(fx), "seconds": time.monotonic() - start})
        hashes.update({p: A.digest(p) for p in (f, r)})
        print(json.dumps(records[-1]), flush=True)
    result = {"replicas": replicas, "seed": seed, "shards": records, "hashes": hashes,
              "cache": str(root), "cache_sha256": A.digest(root / "cache.json"), "retained_seed_rule": "[shard,1000]",
              "retained_pre": old_pre.tolist(), "retained_post": old_post.tolist(), "reference": reference_total.tolist(),
              "pre_samples": pre.tolist(), "post_samples": post.tolist(), "summary": summarize(pre, post, reference_total),
              "seconds": time.monotonic() - tick,
              "limitations": ["Fixed-library independent sampler replicas, not finite-library uncertainty.",
                               "MC SE estimates numerical mean uncertainty, not physics/model uncertainty.",
                               "Pre-placement truth destination vs post-placement own-primary bar membership."]}
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"sector_ratios": {a: r["sector_ratio_to_g4"] for a, r in result["summary"].items()},
                      "seconds": result["seconds"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "transport", "reference", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--shards", default="11")
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--seed", type=int, default=202610271)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
