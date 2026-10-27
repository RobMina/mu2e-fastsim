"""Read-only compact summary and final binding of corrected shower uncertainty evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, tables
from valhist import shower_uncertainty as U
from valhist.shower_uncertainty_report import jsonable, stderr
from valhist.genassoc_campaign import _new_json
from valhist.library_variants import sha256


def run(study, fullgen_certificate, out):
    study = Path(study).resolve(strict=True)
    files = {"production": study / "production_result/result.json", "source": study / "source_result_v2/source_comparison.json",
             "tails": study / "sampling_tails/tails.json", "evaluation": study / "evaluation/evaluation.json",
             "certificate": Path(fullgen_certificate).resolve(strict=True)}
    hashes = U.fingerprint(list(files.values()) + [__file__])
    reports = {k: json.loads(p.read_text()) for k, p in files.items()}
    certificate = reports["certificate"]
    U.verify(certificate["input_sha256_post"])
    U.verify(certificate["evidence_sha256"])
    if len(certificate["files"]) != 20 or certificate["event_count"] != 1000000 or certificate["cross_file_event_overlap"]:
        raise ValueError("full-generator certificate is incomplete")
    for old, record in certificate["files"].items():
        if sha256(record["v2path"]) != record["digest"] or not record["legacy_values_identical"]:
            raise ValueError("fullgen audit changed or legacy mismatch")
        strict = genassoc.generator_event_keys(record["v2path"])
        legacy = genassoc.generator_event_keys(old)
        if not U.event_index(strict).sort_values().equals(U.event_index(legacy).sort_values()):
            raise ValueError("legacy-inferred census differs from fullgen census; recompute centered statistics")
        if record["zero_generator_events"] != 0:
            raise ValueError("unexpected omitted legacy census events")
    for cache in (study / "production", study / "reference_source"):
        meta = json.loads((cache / "cache.json").read_text())
        for path in meta["gen"]:
            if path not in certificate["files"]:
                raise ValueError("uncertified denominator")
            if meta["input_sha256"][path] != certificate["input_sha256_post"][path]:
                raise ValueError("denominator/certificate hash mismatch")
        U.verify(meta["input_sha256"]); U.verify(meta["output_sha256"])
    for row in reports["evaluation"]["shards"]:
        path = row["paths"][2]
        if path not in certificate["files"] or row["events"] != certificate["files"][path]["event_count"]:
            raise ValueError("uncertified evaluation census")
    prod = reports["production"]; source = reports["source"]
    mompath = study / "production_result/moments.npz"
    if sha256(mompath) != prod["moments_sha256"]:
        raise ValueError("moment hash drift")
    moments = dict(np.load(mompath, allow_pickle=False))
    scalar = []
    for p, s, t in zip(prod["scalar"], source["scalar"], reports["tails"]["scalar"]):
        if not p["feature"] == s["feature"] == t["feature"]:
            raise ValueError("feature mismatch")
        scalar.append({"feature": p["feature"], "sampled_over_g4": p["sampled_over_g4"],
                       "expected_over_g4": p["expected_over_g4"], "source_over_g4": s["source_over_g4"],
                       "sampling_relative_sd": p["sampling_sd"] / p["expected_nonmu"],
                       "donor_relative_sd": p["donor_sd"] / p["expected_nonmu"],
                       "source_contrast_pull": s["source_minus_production"] / s["source_minus_production_sd"],
                       "expected_g4_pull_paired_donor_bank": p["mean_residual"] / s["production_g4_residual_sd"],
                       "observed_sampling_lower_tail": t["lower_tail_fraction"],
                       "observed_sampling_upper_tail": t["upper_tail_fraction"]})
    ix = np.array([0, 2, 4, 6])
    total = {"g4": sum(p["all_g4"] for p in prod["scalar"][::2]),
             "expected": sum(p["all_expected"] for p in prod["scalar"][::2]),
             "sampled": sum(p["sampled_nonmu"] + p["fixed_muon"] for p in prod["scalar"][::2]),
             "source_expected": sum(s["all_source_mean"] for s in source["scalar"][::2])}
    for name, cov in (("sampling_sd", moments["sampling_cov"]), ("donor_sd", moments["donor_cov"]),
                      ("paired_g4_sd", moments["paired_all_cov"])):
        total[name] = float(np.sqrt(cov[np.ix_(ix, ix)].sum()))
    total["expected_residual_sd_paired_donor"] = float(np.hypot(total["paired_g4_sd"], total["donor_sd"]))
    cell = pd.DataFrame(source["cells"])
    species = []
    for pdg, part in cell.groupby("apdg"):
        item = {"apdg": int(pdg), "production_showers": int(part.production_showers.sum()),
                "source_showers": int(part.source_showers.sum()), "evaluation_primaries": int(part.neval.sum())}
        for name in prod["names"][:8]:
            item[name] = {"production": float(part[name + "/production_expected"].sum()),
                          "source": float(part[name + "/source_expected"].sum())}
        species.append(item)
    result = {"scalar": scalar, "total_steps": total, "species_contributions": species,
              "production_positions": prod["positions"], "source_positions": source["source_positions"],
              "zero_generator_events": 0, "fullgen_census_certified": True,
              "legacy_denominator_unchanged": True, "input_sha256": hashes,
              "uncovered_eval_cells": prod["cells_missing_bank"], "zero_template_eval_cells": prod["cells_without_templates"],
              "limitations": ["Marginal tails are correlated and not corrected for selection/multiple comparisons.",
                              "Cross-bin covariance retained, but no calibrated global significance is assigned.",
                              "Whole-file exchangeability and first-order ratio covariance remain assumptions.",
                              "Neither this study nor source similarity certifies nonlinear cluster response or newer main."]}
    U.verify(hashes)
    _new_json(out, jsonable(result))
    print(json.dumps(jsonable({"scalar": scalar, "total_steps": total, "fullgen_census_certified": True,
                               "uncovered_eval_cells": prod["cells_missing_bank"]}), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study"); p.add_argument("fullgen_certificate"); p.add_argument("--out", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
