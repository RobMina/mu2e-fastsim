"""Separate exact sampler fluctuations, whole-file donor uncertainty and held-out residuals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from valhist import shower_moments as M
from valhist import shower_uncertainty as U
from valhist.genassoc_campaign import _new_json
from valhist.library_variants import sha256


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def stderr(covariance):
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-8 * max(1, float(np.max(np.abs(diagonal))))):
        raise ValueError("negative covariance diagonal")
    return np.sqrt(np.maximum(diagonal, 0))


def aligned(cache, evalcells, counts):
    meta, index, cells, x, filecell = U.load_cache(cache)
    target = U.cell_rows(cells, evalcells)
    ncell = len(evalcells)
    ngen = np.zeros(ncell); ngen[target] = index["ngen"]
    tc = target[index["cell"]]
    moments = M.conditional_moments(x, tc, ngen, counts)
    weight = np.divide(counts[target], index["ngen"], out=np.zeros(len(cells)), where=index["ngen"] > 0)
    f = len(meta["donors"])
    mapping = sparse.csr_matrix((np.tile(weight, f), (np.repeat(np.arange(f), len(cells)),
                                                       np.arange(f * len(cells)))), shape=(f, f * len(cells)))
    contrib = (mapping @ filecell).toarray()
    np.testing.assert_allclose(contrib.sum(axis=0), moments["total_mean"], rtol=1e-12, atol=1e-8)
    return meta, index, moments, contrib, ngen, tc


def positions(mean, paired, donor, fixed, g4, g4_second, blocks):
    """Descriptive diagonal scores; full covariance is retained separately, never inverted."""
    result = {}
    for name, (lo, hi) in blocks.items():
        ix = np.arange(lo + 1, hi - 1)
        expect = mean + fixed
        delta = expect[ix] - g4[ix]
        occupied = (expect[ix] + g4[ix]) > 0
        pvar = np.diag(paired)[ix]; dvar = np.diag(donor)[ix]
        def score(var):
            usable = occupied & (var > 0)
            return {"value": float(np.mean(delta[usable] ** 2 / var[usable])) if usable.any() else None,
                    "used_bins": int(usable.sum()), "zero_variance_nonzero_residual_bins":
                    int(np.sum(occupied & (var <= 0) & (np.abs(delta) > 1e-10)))}
        result[name] = {"expected_steps": float(expect[lo:hi].sum()), "g4_steps": float(g4[lo:hi].sum()),
                        "flow_expected": float(expect[lo] + expect[hi - 1]), "flow_g4": float(g4[lo] + g4[hi - 1]),
                        "occupied_bins": int(occupied.sum()), "paired_diagonal_score": score(pvar),
                        "paired_plus_donor_diagonal_score": score(pvar + dvar),
                        "limitation": "Descriptive diagonal score, no calibrated chi-square interpretation."}
    return result


def nominal(cache, evaluation, outdir, replicas=2000, seed=20261027):
    """Conditional-bank production result; do not add sampling variance to expected-versus-G4 residuals."""
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError(out)
    code = {**U.code_identity(), str(Path(__file__).resolve()): sha256(__file__)}
    ep = Path(evaluation)
    em = json.loads((ep / "evaluation.json").read_text())
    if sha256(Path(cache) / "cache.json") != em["cache_sha256"]:
        raise ValueError("evaluation is bound to a different template cache")
    if sha256(ep / "sufficient.npz") != em["output_sha256"]:
        raise ValueError("evaluation sufficient-statistic hash mismatch")
    q = dict(np.load(ep / "sufficient.npz", allow_pickle=False))
    cells = pd.DataFrame(em["cells"])
    meta, index, moments, contrib, ngen, tc = aligned(cache, cells, q["counts"])
    if meta["names"] != em["names"]:
        raise ValueError("incompatible evaluation features")
    boot = M.file_bootstrap(contrib, index["file_index"], tc, ngen, replicas=replicas, seed=seed)
    paired = [M.paired_covariance(moments["mean_per_cell"], q["hth"], q["htr"][i], q["rtr"][i],
                                 q["counts"], q["rsum"][i], em["events"]) for i in (0, 1)]
    mean = moments["total_mean"]
    samp_sd = stderr(moments["sampling_cov"]); donor_sd = stderr(boot["bootstrap_cov"])
    paired_sd = stderr(paired[0]["covariance"])
    total_paired_sd = stderr(paired[1]["covariance"])
    scalar = []
    for j, name in enumerate(meta["names"][:8]):
        ref = q["g4_nonmu"][j]
        scalar.append({"feature": name, "g4_nonmu": float(ref), "sampled_nonmu": float(q["sampled"][j]),
                       "expected_nonmu": float(mean[j]), "expected_over_g4": float(mean[j] / ref) if ref else None,
                       "sampled_over_g4": float(q["sampled"][j] / ref) if ref else None,
                       "sampling_sd": float(samp_sd[j]), "donor_sd": float(donor_sd[j]),
                       "paired_g4_sd": float(paired_sd[j]),
                       "sampling_pull": float((q["sampled"][j] - mean[j]) / samp_sd[j]) if samp_sd[j] else None,
                       "mean_residual": float(mean[j] - ref),
                       "mean_residual_sd_paired_plus_donor": float(np.hypot(paired_sd[j], donor_sd[j])),
                       "conditional_mean_donor_quantiles": boot["quantiles"][:, j].tolist(),
                       "all_g4": float(q["g4"][j]), "fixed_muon": float(q["fixed"][j]),
                       "all_expected": float(q["fixed"][j] + mean[j]),
                       "all_expected_over_g4": float((q["fixed"][j] + mean[j]) / q["g4"][j]) if q["g4"][j] else None,
                       "all_residual_sd_paired_plus_donor": float(np.hypot(total_paired_sd[j], donor_sd[j]))})
    hashes = U.fingerprint([ep / "evaluation.json", ep / "sufficient.npz", Path(cache) / "cache.json", __file__])
    U.verify(meta["input_sha256"])
    U.verify(meta["output_sha256"])
    U.verify(em["input_sha256"])
    out.mkdir()
    np.savez_compressed(out / "moments.npz", mean_per_cell=moments["mean_per_cell"], mean=mean,
                        sampling_cov=moments["sampling_cov"], donor_cov=boot["bootstrap_cov"],
                        paired_nonmu_cov=paired[0]["covariance"], paired_all_cov=paired[1]["covariance"],
                        file_contribution=contrib, bootstrap_totals=boot["replica_totals"])
    result = {"cache": str(Path(cache).resolve()), "evaluation": str(ep.resolve()), "events": em["events"],
              "scalar": scalar, "names": meta["names"], "quantile_levels": boot["quantile_levels"],
              "bootstrap_replicas": replicas, "bootstrap_seed": seed,
              "bootstrap_probability_guard": boot["probability_guard"],
              "donor_covariance_rank_upper_bound": boot["covariance_rank_upper_bound"],
              "cells_missing_bank": cells.loc[(ngen == 0) & (q["counts"] > 0)].assign(
                  evaluation_primaries=q["counts"][(ngen == 0) & (q["counts"] > 0)]).to_dict("records"),
              "cells_without_templates": cells.loc[(np.bincount(tc, minlength=len(cells)) == 0) & (q["counts"] > 0)].assign(
                  evaluation_primaries=q["counts"][(np.bincount(tc, minlength=len(cells)) == 0) & (q["counts"] > 0)]).to_dict("records"),
              "code_sha256": code,
              "positions": positions(mean, paired[1]["covariance"], boot["bootstrap_cov"], q["fixed"], q["g4"],
                                     q["g4_second"], meta["blocks"]),
              "input_sha256": hashes, "moments_sha256": sha256(out / "moments.npz"),
              "limitations": ["Production donor files assumed independent and exchangeable with equal generated exposure.",
                              "Denominator bank held fixed; its uncertainty is excluded.",
                              "Muon-family response held fixed; no muon-model uncertainty estimated.",
                              "Paired event variance and file bootstrap are approximations, not calibrated goodness-of-fit tests.",
                              "Analytic results apply to additive steps/energy/positions, not unique bar occupancy or clusters."]}
    U.verify({**hashes, **code})
    _new_json(out / "result.json", jsonable(result))
    print(json.dumps(jsonable({"scalar": scalar, "positions": result["positions"],
                               "missing_bank": result["cells_missing_bank"]}), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache"); p.add_argument("evaluation"); p.add_argument("outdir")
    p.add_argument("--replicas", type=int, default=2000); p.add_argument("--seed", type=int, default=20261027)
    nominal(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
