"""Independent held-out additive shower expectation on existing full-generator/G4 pairs.

Uses the strict library's fixed nominal denominator; optional candidate bank is a
sensitivity diagnostic only. No transport, placement, reconstruction or library write.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

from crvfastsim import genassoc, geometry, tables
from crvfastsim.analyses.census import classify, SECTOR_GROUP
from . import shower_uncertainty as U
from . import descendant_inputs as I
from .descendant_audit import digest
from .steplayers import shard


def scalar_features(steps, parts, census, geom):
    tagged = classify(steps, parts, geom.sector)
    if tagged.cls.eq("unknown").any():
        raise ValueError("unknown ancestry")
    d = tagged.loc[tagged.cls.str.startswith("nonmu")]
    row = U.event_rows(d, census)
    group = pd.Index(U.GROUPS).get_indexer(SECTOR_GROUP[geom.sector[d.bar.to_numpy()]])
    x = sparse.coo_matrix((np.r_[np.ones(len(d)), d.edep.to_numpy(dtype=float)],
                           (np.tile(row, 2), np.r_[2 * group, 2 * group + 1])), shape=(len(census), 8)).tocsr()
    return x


def means(features, template_cells, denominator, ncell):
    incidence = sparse.coo_matrix((np.ones(len(template_cells)), (template_cells, np.arange(len(template_cells)))),
                                 shape=(ncell, len(template_cells))).tocsr()
    sums = (incidence @ features).toarray()
    return np.divide(sums, denominator[:, None], out=np.zeros_like(sums), where=denominator[:, None] > 0)


def run(cache, directory, shards, out, candidate_bank=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    meta, index, cells, x, _ = U.load_cache(cache)
    U.verify(meta["input_sha256"])
    geom = geometry.from_offline_config(meta["geometry"])
    x = x[:, :8].tocsr()
    ncell = len(cells)
    denominators = {"nominal": index["ngen"].astype(float)}
    bank_hash = None
    if candidate_bank:
        b = json.loads(Path(candidate_bank).read_text())
        if not b["support"]["ensemble_support_pass"]:
            raise ValueError("candidate bank failed support")
        pd.testing.assert_frame_equal(cells, pd.DataFrame(b["bins"]), check_dtype=False)
        denominators["candidate_bank_diagnostic"] = np.asarray(b["support"]["nominal_scaled_denominator"])
        bank_hash = digest(candidate_bank)
    mu = {a: means(x, index["cell"], den, ncell) for a, den in denominators.items()}
    train = U.event_index(pd.DataFrame(index["donor_events"], columns=U.KEY))
    for f in meta["gen"]:
        train = train.union(U.event_index(genassoc.generator_event_keys(f)))
    totals = {a: {"mean": np.zeros(8), "residual_outer": np.zeros((8, 8)), "reference_outer": np.zeros((8, 8)),
                  "mean_outer": np.zeros((8, 8)), "cross": np.zeros((8, 8))} for a in mu}
    refsum, counts, n = np.zeros(8), np.zeros(ncell, dtype=np.int64), 0
    inventory, hashes, seen = [], {}, set()
    for k in I.unique_shards(shards):
        tick = time.monotonic()
        gp, rp = [Path(directory) / f"{prefix}.{shard(k)}.root" for prefix in ("gen", "crvsteps")]
        gen = tables.read(gp, "gen")
        census = genassoc.generator_event_keys(gp)
        refgen = tables.read(rp, "gen")
        if gen.duplicated(U.KEY + ["idx"]).any() or refgen.duplicated(U.KEY + ["idx"]).any():
            raise ValueError("duplicate generator rows")
        pos = pd.MultiIndex.from_frame(gen[U.KEY + ["idx"]]).get_indexer(pd.MultiIndex.from_frame(refgen[U.KEY + ["idx"]]))
        if (pos < 0).any():
            raise ValueError("reference generators outside full stream")
        pd.testing.assert_frame_equal(refgen.reset_index(drop=True), gen.iloc[pos][refgen.columns].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        st, pa = tables.read(rp, "steps"), tables.read(rp, "particles")
        I.population(census, st, pa, train, seen)
        pop = U.bin_frame(gen)
        cell = pd.MultiIndex.from_frame(cells).get_indexer(pd.MultiIndex.from_frame(pop[U.BIN]))
        rows = U.event_rows(pop, census)
        valid = cell >= 0
        h = sparse.coo_matrix((np.ones(valid.sum()), (rows[valid], cell[valid])), shape=(len(census), ncell)).tocsr()
        reference = scalar_features(st, pa, census, geom)
        refdense = reference.toarray()
        unsupported = np.zeros(len(census), dtype=bool)
        unsupported[rows[~valid]] = True
        counts += np.bincount(cell[valid], minlength=ncell)
        refsum += np.asarray(reference.sum(axis=0)).ravel()
        n += len(census)
        for arm in mu:
            predicted = h @ mu[arm]
            residual = predicted - refdense
            t = totals[arm]
            t["mean"] += predicted.sum(0)
            t["residual_outer"] += residual.T @ residual
            t["reference_outer"] += refdense.T @ refdense
            t["mean_outer"] += predicted.T @ predicted
            t["cross"] += predicted.T @ refdense
        row = {"shard": k, "events_inferred": len(census), "generator_rows": len(gen), "reference_gen_exact": True,
               "unrepresented_primaries": int((~valid).sum()), "unrepresented_event_nonmu_energy": float(refdense[unsupported, 1::2].sum()),
               "seconds": time.monotonic() - tick}
        hashes.update({str(f): digest(f) for f in (gp, rp)})
        inventory.append(row)
        print(json.dumps(row), flush=True)
    result = {"cache": str(cache), "cache_sha256": digest(Path(cache) / "cache.json"), "source_sha256": digest(__file__),
              "candidate_bank_sha256": bank_hash, "input_sha256": hashes, "events_inferred": n,
              "shards": inventory, "reference": refsum.tolist(), "arms": {}}
    for arm, t in totals.items():
        ratio = t["mean"] / refsum
        rcov = t["residual_outer"] - np.outer(t["mean"] - refsum, t["mean"] - refsum) / n
        ratio_outer = t["mean_outer"] + ratio[:, None] * t["reference_outer"] * ratio[None, :]
        ratio_outer -= t["cross"] * ratio[None, :] + ratio[:, None] * t["cross"].T
        sampling = U.M.conditional_moments(x, index["cell"], denominators[arm], counts)["sampling_cov"]
        result["arms"][arm] = {"mean": t["mean"].tolist(), "ratio": ratio.tolist(),
                               "paired_event_ratio_se": (np.sqrt(np.maximum(np.diag(ratio_outer) * n / (n - 1), 0)) / refsum).tolist(),
                               "paired_event_residual_cov": (rcov * n / (n - 1)).tolist(),
                               "sampling_cov": sampling.tolist()}
        print(arm, "energy ratios", ratio[1::2].tolist(), flush=True)
    result.update(seconds=time.monotonic() - start, limitations=[
        "Additive nonmuon step/energy means only; no descendant placement, nonlinear DQM or efficiency validation.",
        "Paired event errors condition on fixed library and denominator; no donor/denominator error.",
        "Legacy event census inferred from gen rows; strict full-stream/empty-event proof pending.",
        "Candidate-bank arm is a sensitivity diagnostic, not an admitted replacement library."])
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"wrote {out}; {result['seconds']:.2f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "directory", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--shards", default="21-220")
    p.add_argument("--candidate-bank")
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
