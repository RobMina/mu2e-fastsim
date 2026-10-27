"""Exact multinomial replicas of additive shower totals and retained-seed replay certification."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
import uproot

from crvfastsim import tables
from valhist import shower_uncertainty as U
from valhist.shower_uncertainty_report import jsonable, aligned, stderr
from valhist.genassoc_campaign import _new_json
from valhist.library_variants import sha256


def multinomial_replicas(x, cell, ngen, neval, replicas, seed, batch=128):
    """Sample exact independent-primary aggregate distribution; loops only over cells and batches."""
    if replicas < 2 or batch < 1:
        raise ValueError("need at least two replicas and positive batch")
    count = np.bincount(cell, minlength=len(ngen))
    if np.any(count > ngen) or np.any(neval < 0) or np.any(neval != np.floor(neval)):
        raise ValueError("invalid counts/support")
    rng = np.random.default_rng(seed)
    result = np.zeros((replicas, x.shape[1]))
    for c in np.flatnonzero((count > 0) & (neval > 0)):
        rows = np.flatnonzero(cell == c)
        probability = np.r_[np.full(len(rows), 1 / ngen[c]), 1 - len(rows) / ngen[c]]
        for lo in range(0, replicas, batch):
            hi = min(lo + batch, replicas)
            draw = rng.multinomial(int(neval[c]), probability, size=hi - lo)[:, :-1]
            result[lo:hi] += np.asarray(x[rows].T @ draw.T).T
    return result


def run(cache, evaluation, production_result, outdir, transport, replicas=10000, seed=20261028):
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError(out)
    ep = Path(evaluation)
    em = json.loads((ep / "evaluation.json").read_text())
    result_path = Path(production_result) / "result.json"
    retained = json.loads(result_path.read_text())
    if (retained["cache"] != str(Path(cache).resolve()) or retained["evaluation"] != str(ep.resolve())
            or retained["input_sha256"].get(str(ep.resolve() / "evaluation.json")) != sha256(ep / "evaluation.json")):
        raise ValueError("production result is bound to another evaluation/cache")
    if sha256(ep / "sufficient.npz") != em["output_sha256"] or sha256(Path(cache) / "cache.json") != em["cache_sha256"]:
        raise ValueError("unbound evaluation/cache")
    q = dict(np.load(ep / "sufficient.npz", allow_pickle=False))
    meta, index, cells, x, _ = U.load_cache(cache)
    hashes = U.fingerprint([Path(cache) / "cache.json", ep / "evaluation.json", ep / "sufficient.npz", result_path,
                            meta["library"], __file__, U.__file__])
    sh = tables.read(meta["library"], "libshower"); lg = tables.read(meta["library"], "libgen")
    sampler_source = Path(U.tier1b.__file__).read_text().split("def _desc_for_crossings", 1)[1].split("def _showers_for_gen", 1)[0]
    if re.findall(r"rng\.\w+\([^\n]*", sampler_source) != ["rng.random(len(b)) * (hi - lo)).astype(np.int64)]"]:
        raise ValueError("descendant RNG consumption differs; derive state from actual sampler before replay")
    replay = np.zeros(x.shape[1]); per_shard = []
    for row in em["shards"]:
        k = row["shard"]
        path = transport.format(s=U.shard(k), k=k)
        hashes.update(U.fingerprint([path, row["paths"][2]]))
        with uproot.open(path) as root:
            crossings = root["dump/steps"].num_entries
        gen = tables.read(row["paths"][2], "gen")
        rng = np.random.default_rng([k, 1000]); rng.random(crossings)
        ids = U.compact_draw(gen, sh, lg, rng)
        replay += np.asarray(x[ids].sum(axis=0)).ravel()
        per_shard.append({"shard": k, "transport_crossings": int(crossings), "selected_showers": len(ids)})
    np.testing.assert_allclose(replay, q["sampled"], rtol=1e-12, atol=1e-7)
    evalcells = pd.DataFrame(em["cells"])
    target = U.cell_rows(cells, evalcells)
    neval = q["counts"][target]
    moments = U.M.conditional_moments(x[:, :8].tocsr(), index["cell"], index["ngen"], neval)
    samples = multinomial_replicas(x[:, :8].tocsr(), index["cell"], index["ngen"], neval, replicas, seed)
    expected, sd = moments["total_mean"], stderr(moments["sampling_cov"])
    scalar = []
    for j, name in enumerate(meta["names"][:8]):
        obs = q["sampled"][j]
        low = int((samples[:, j] <= obs).sum()); high = int((samples[:, j] >= obs).sum())
        scalar.append({"feature": name, "observed": float(obs), "mean": float(expected[j]), "sampling_sd": float(sd[j]),
                       "empirical_mean": float(samples[:, j].mean()), "empirical_sd": float(samples[:, j].std(ddof=1)),
                       "lower_tail_count": low, "upper_tail_count": high, "replicas": replicas,
                       "lower_tail_fraction": low / replicas, "upper_tail_fraction": high / replicas,
                       "quantile_levels": [.001, .01, .025, .16, .5, .84, .975, .99, .999],
                       "quantiles": np.quantile(samples[:, j], [.001, .01, .025, .16, .5, .84, .975, .99, .999]).tolist()})
    U.verify(hashes)
    out.mkdir()
    np.savez_compressed(out / "replicas.npz", samples=samples, replay=replay)
    result = {"retained_replay_feature_match": True, "per_shard": per_shard, "scalar": scalar,
              "seed": seed, "replicas": replicas, "input_sha256": hashes,
              "output_sha256": sha256(out / "replicas.npz"),
              "exact_sampling_covariance": moments["sampling_cov"], "empirical_sampling_covariance": np.cov(samples, rowvar=False),
              "limitation": "Conditional sampler distribution only; unadjusted marginal tails across correlated observables, no model p-value."}
    _new_json(out / "tails.json", jsonable(result))
    print(json.dumps({"retained_replay_feature_match": True, "scalar": scalar}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("cache", "evaluation", "production_result", "outdir"):
        p.add_argument(arg)
    p.add_argument("--transport", required=True); p.add_argument("--replicas", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261028)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
