"""Corrected production/reference source contrast with matched source/gen file covariance."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from valhist import shower_moments as M
from valhist import shower_uncertainty as U
from valhist.shower_uncertainty_report import aligned, jsonable, stderr, positions
from valhist.genassoc_campaign import _new_json
from valhist.library_variants import sha256


def ratio_influence(numerator, denominator, neval, scale=1.0):
    """Per matched-file first-order contributions to a cell-wise ratio's evaluated sum."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    neval = np.asarray(neval, dtype=float)
    if numerator.ndim != 3 or denominator.shape != numerator.shape[:2] or neval.shape != denominator.shape[1:]:
        raise ValueError("incompatible matched numerator/denominator shapes")
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or scale <= 0:
        raise ValueError("nonfinite matched ratio input")
    if (numerator < 0).any() or (denominator < 0).any() or (neval < 0).any():
        raise ValueError("negative ratio input")
    total = denominator.sum(axis=0)
    sums = numerator.sum(axis=0)
    if (sums[total == 0] > 0).any():
        raise ValueError("unsupported ratio numerator")
    weight = np.divide(neval, scale * total, out=np.zeros_like(neval), where=total > 0)
    slope = np.divide(sums * weight[:, None], total[:, None], out=np.zeros_like(sums), where=total[:, None] > 0)
    raw = np.einsum("fcd,c->fd", numerator, weight) - denominator @ slope
    return raw - raw.mean(axis=0)


def bank_influence(total_numerator, denominator_files, neval, scale):
    """Shared-generator-bank contribution for a statistically independent fixed numerator."""
    denominator = denominator_files.sum(axis=0).astype(float)
    weight = np.divide(neval, scale * denominator ** 2, out=np.zeros_like(denominator), where=denominator > 0)
    raw = -denominator_files @ (total_numerator * weight[:, None])
    return raw - raw.mean(axis=0)


def sum_per_cell(x, cell, ncell):
    return U.grouped_features(x, cell, ncell).toarray()


def certify_source_pairs(meta):
    """Use original collection indices to bind legacy donor generator products to matched fullgen."""
    plan = json.loads((Path(meta["campaign"]) / "plan.json").read_text())
    records, hashes = [], {}
    for row, path in zip(plan["donors"], meta["gen"]):
        hashes.update(U.fingerprint([row["old_table"], path]))
        if hashes[row["old_table"]] != row["old_sha256"]:
            raise ValueError("legacy source donor drift")
        donor = U.tables.read(row["old_table"], "gen")
        full = U.tables.read(path, "gen")
        keys = U.KEY + ["idx"]
        if donor.duplicated(keys).any() or full.duplicated(keys).any():
            raise ValueError("duplicate source generator identity")
        where = pd.MultiIndex.from_frame(full[keys]).get_indexer(pd.MultiIndex.from_frame(donor[keys]))
        if (where < 0).any():
            raise ValueError("source donor generator absent from matched fullgen")
        pd.testing.assert_frame_equal(donor.reset_index(drop=True), full.iloc[where][donor.columns].reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
        records.append({"legacy_donor": row["old_table"], "fullgen": path, "rows": len(donor),
                        "original_product_payload_identical": True})
    U.verify(hashes)
    return records, hashes


def compare(production, source, evaluation, outdir):
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError(out)
    code = {**U.code_identity(), str(Path(__file__).resolve()): sha256(__file__)}
    ep = Path(evaluation)
    em = json.loads((ep / "evaluation.json").read_text())
    if sha256(ep / "sufficient.npz") != em["output_sha256"]:
        raise ValueError("evaluation hash drift")
    if sha256(Path(production) / "cache.json") != em["cache_sha256"]:
        raise ValueError("evaluation not bound to production cache")
    q = dict(np.load(ep / "sufficient.npz", allow_pickle=False))
    cells = pd.DataFrame(em["cells"])
    pm, pi, pmo, pb, png, pc = aligned(production, cells, q["counts"])
    sm, si, smo, sb, sng, sc = aligned(source, cells, q["counts"])
    if pm["names"] != sm["names"] or pm["names"] != em["names"] or pm["edges"] != sm["edges"]:
        raise ValueError("source/evaluation feature mismatch")
    if pm["geometry_sha256"] != sm["geometry_sha256"] or pm["gen"] != sm["gen"]:
        raise ValueError("source/evaluation geometry or fullgen bank differs")
    if not np.array_equal(pi["gen_counts"], si["gen_counts"]) or not np.array_equal(pi["gen_events"], si["gen_events"]):
        raise ValueError("source/fullgen cell inventories differ")
    source_ids = U.event_index(pd.DataFrame(si["donor_events"], columns=U.KEY))
    prod_ids = U.event_index(pd.DataFrame(pi["donor_events"], columns=U.KEY))
    if len(source_ids.intersection(prod_ids)):
        raise ValueError("source and production donor event overlap")
    for row in em["shards"]:
        generated_ids = U.event_index(U.genassoc.generator_event_keys(row["paths"][2]))
        if len(source_ids.intersection(generated_ids)):
            raise ValueError("source donor/evaluation overlap")
    _, _, pcell, px, pf = U.load_cache(production)
    _, _, scall, sx, sf = U.load_cache(source)
    pt = U.cell_rows(pcell, cells); st = U.cell_rows(scall, cells)
    F, C, D = len(sm["donors"]), len(cells), len(pm["names"])
    if len(si["gen_events"]) != F or not np.all(si["gen_events"] == sm["generated_per_file"]):
        raise ValueError("source file pairs must have equal full-generated exposure")
    genfiles = np.zeros((F, C)); genfiles[:, st] = si["gen_counts"]
    source_features = np.zeros((F, C, D))
    source_features[:, st, :] = sf.toarray().reshape(F, len(scall), D)
    source_counts = np.zeros((F, C))
    np.add.at(source_counts, (si["file_index"], sc), 1)
    if (source_counts > genfiles).any():
        raise ValueError("matched source file has more showers than generated cell primaries")
    prod_sum = sum_per_cell(px, pc, C)
    scale = pm["generated"] / sum(si["gen_events"])
    if not np.array_equal(png, genfiles.sum(axis=0) * scale) or not np.array_equal(sng, genfiles.sum(axis=0)):
        raise ValueError("linearized bank covariance requires exact integer scaling without rounding discrepancy")
    pair_records, pair_hashes = certify_source_pairs(sm)
    src_if = ratio_influence(source_features, genfiles, q["counts"])
    bank_if = bank_influence(prod_sum, genfiles, q["counts"], scale)
    pb_center = pb - pb.mean(axis=0)
    cov_prod_donor = pb_center.T @ pb_center
    cov_src = src_if.T @ src_if
    cov_bank = bank_if.T @ bank_if
    contrast_if = src_if - bank_if
    cov_contrast = contrast_if.T @ contrast_if + cov_prod_donor
    src_paired = [M.paired_covariance(smo["mean_per_cell"], q["hth"], q["htr"][i], q["rtr"][i],
                                     q["counts"], q["rsum"][i], em["events"]) for i in (0, 1)]
    prod_paired = M.paired_covariance(pmo["mean_per_cell"], q["hth"], q["htr"][0], q["rtr"][0],
                                    q["counts"], q["rsum"][0], em["events"])
    pmean, smean = pmo["total_mean"], smo["total_mean"]
    scalar = []
    sp_sd = stderr(src_paired[0]["covariance"] + cov_src)
    pp_sd = stderr(prod_paired["covariance"] + cov_prod_donor + cov_bank)
    ds = stderr(cov_contrast)
    for j, name in enumerate(pm["names"][:8]):
        ref = q["g4_nonmu"][j]
        scalar.append({"feature": name, "g4_nonmu": float(ref), "production_mean": float(pmean[j]),
                       "source_mean": float(smean[j]), "production_over_g4": float(pmean[j] / ref) if ref else None,
                       "source_over_g4": float(smean[j] / ref) if ref else None,
                       "source_minus_production": float(smean[j] - pmean[j]), "source_minus_production_sd": float(ds[j]),
                       "source_g4_residual_sd": float(sp_sd[j]), "production_g4_residual_sd": float(pp_sd[j]),
                       "source_paired_file_sd": float(stderr(cov_src)[j]), "production_bank_sd": float(stderr(cov_bank)[j]),
                       "fixed_muon": float(q["fixed"][j]), "all_g4": float(q["g4"][j]),
                       "all_source_mean": float(q["fixed"][j] + smean[j])})
    # Conditional template summaries expose within-cell source effects without changing the model's variables.
    prod_count = np.bincount(pc, minlength=C); src_count = np.bincount(sc, minlength=C)
    src_sum = source_features.sum(axis=0)
    cellinfo = cells.copy()
    cellinfo["neval"] = q["counts"]; cellinfo["production_showers"] = prod_count; cellinfo["source_showers"] = src_count
    cellinfo["production_probability"] = np.divide(prod_count, png, out=np.zeros(C), where=png > 0)
    cellinfo["source_probability"] = np.divide(src_count, sng, out=np.zeros(C), where=sng > 0)
    for j, name in enumerate(pm["names"][:8]):
        cellinfo[name + "/production_expected"] = q["counts"] * pmo["mean_per_cell"][:, j]
        cellinfo[name + "/source_expected"] = q["counts"] * smo["mean_per_cell"][:, j]
    profiles = {}
    for name, meta, idx, x in (("production", pm, pi, px), ("source", sm, si, sx)):
        prof = pd.DataFrame(idx["profile"], columns=meta["profile_columns"])
        momentum = np.linalg.norm(prof[["gen_px", "gen_py", "gen_pz"]].to_numpy(), axis=1)
        prof["cos_y"] = np.divide(prof.gen_py, momentum, out=np.zeros(len(prof)), where=momentum > 0)
        profiles[name] = []
        for (pdg, ke), frame in prof.groupby(U.BIN):
            profiles[name].append({"apdg": int(pdg), "kebin": int(ke), "showers": len(frame),
                                   "quantile_levels": [.1, .5, .9],
                                   "quantiles": {c: np.quantile(frame[c], [.1, .5, .9]).tolist()
                                                 for c in ("ke", "cos_y", "gen_x", "gen_y", "gen_z")}})
    weights = np.random.default_rng(20261029).multinomial(F, np.full(F, 1 / F), size=10000)
    loss = (weights @ genfiles == 0) & (genfiles.sum(axis=0) > 0)
    active = q["counts"] > 0
    loss_info = cells.assign(generator_files=(genfiles > 0).sum(axis=0),
                             denominator_loss_replicas=loss.sum(axis=0), neval=q["counts"])
    loss_info = loss_info.loc[loss.any(axis=0)].copy()
    for j, name in enumerate(pm["names"][:8]):
        loss_info[name + "/nominal_source_contribution"] = (
            q["counts"] * smo["mean_per_cell"][:, j])[loss.any(axis=0)]
        loss_info[name + "/nominal_production_contribution"] = (
            q["counts"] * pmo["mean_per_cell"][:, j])[loss.any(axis=0)]
    support = {"source_cells_with_single_generator_file": int(((genfiles > 0).sum(axis=0) == 1).sum()),
               "source_cells_with_single_shower_file": int(((source_counts > 0).sum(axis=0) == 1).sum()),
               "source_donor_files": F, "production_donor_files": len(pm["donors"]),
               "source_denom_matched": True, "shared_bank_covariance_included": True,
               "source_pair_payload_checks": pair_records, "denominator_loss_trials": 10000,
               "denominator_loss_seed": 20261029,
               "replicas_with_loss_in_evaluated_cell": int(loss[:, active].any(axis=1).sum()),
               "denominator_loss_cells": loss_info.to_dict("records"),
               "denominator_loss_policy": "No replica with missing support is dropped or used as a valid ratio sampler."}
    hashes = {**pair_hashes, **U.fingerprint([Path(production) / "cache.json", Path(source) / "cache.json",
                                             ep / "evaluation.json", ep / "sufficient.npz"])}
    U.verify({**pm["input_sha256"], **sm["input_sha256"], **em["input_sha256"], **code, **hashes})
    out.mkdir()
    np.savez_compressed(out / "source_covariance.npz", production_mean=pmean, source_mean=smean,
                        source_paired_file_cov=cov_src, production_bank_cov=cov_bank,
                        production_donor_cov=cov_prod_donor, source_contrast_cov=cov_contrast,
                        source_g4_paired_cov=src_paired[0]["covariance"], source_all_paired_cov=src_paired[1]["covariance"],
                        source_mean_per_cell=smo["mean_per_cell"], source_matched_file_influence=src_if,
                        production_bank_influence=bank_if)
    result = {"scalar": scalar, "cells": cellinfo.to_dict("records"), "profiles": profiles,
              "support": support, "names": pm["names"],
              "source_positions": positions(smean, src_paired[1]["covariance"], cov_src, q["fixed"], q["g4"],
                                            q["g4_second"], sm["blocks"]),
              "input_sha256": hashes, "code_sha256": code, "covariance_sha256": sha256(out / "source_covariance.npz"),
              "limitations": ["Matched source/fullgen and shared-bank uncertainty uses first-order file-bootstrap ratio influence.",
                              "File units assumed exchangeable and independent; sparse denominator cells need caution.",
                              "Source contrast conditions on held-out generator population and fixed muon family.",
                              "Source change is a diagnostic, not a calibrated model systematic or recommendation to replace donors.",
                              "Generator profiles are retained shower-producing truth diagnostics, not a full generated-population or response-conditioning model.",
                              "Paired-G4 covariance resamples the event population at fixed event count, not exact conditioning on every observed generator."]}
    _new_json(out / "source_comparison.json", jsonable(result))
    print(json.dumps({"scalar": scalar, "support": support}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("production"); p.add_argument("source"); p.add_argument("evaluation"); p.add_argument("outdir")
    compare(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
