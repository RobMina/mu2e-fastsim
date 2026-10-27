"""Compact interpretation of retained corrected-source uncertainty and finite Monte Carlo tails."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from valhist.genassoc_campaign import _new_json
from valhist import shower_uncertainty as U


def tail_interval(count, trials, level=.95):
    """Exact binomial interval for the Monte Carlo fraction, not a physics-model confidence interval."""
    if trials < 1 or not 0 <= count <= trials:
        raise ValueError("invalid Monte Carlo count")
    alpha = (1 - level) / 2
    return [float(beta.ppf(alpha, count, trials - count + 1)) if count else 0.,
            float(beta.ppf(1 - alpha, count + 1, trials - count)) if count < trials else 1.]


def run(study, out):
    root = Path(study).resolve(strict=True)
    paths = [root / "summary.json", root / "source_result_v2/source_comparison.json", root / "sampling_tails/tails.json",
             root / "support_audit.json"]
    hashes = U.fingerprint(paths + [__file__])
    summary, source, tails, holes = [json.loads(p.read_text()) for p in paths]
    total = summary["total_steps"]
    ratio = {k + "_rel_g4": total[k] / total["g4"] - 1 for k in ("sampled", "expected", "source_expected")}
    ratio["expected_residual_over_paired_donor_sd"] = (total["expected"] - total["g4"]) / total["expected_residual_sd_paired_donor"]
    positions = {source: {axis: info["paired_plus_donor_diagonal_score"] for axis, info in summary[source].items()}
                 for source in ("production_positions", "source_positions")}
    tail = [{"feature": r["feature"], "count": r["lower_tail_count"], "trials": r["replicas"],
             "fraction": r["lower_tail_fraction"], "MC_binomial_95_interval": tail_interval(r["lower_tail_count"], r["replicas"])}
            for r in tails["scalar"]]
    support = source["support"]
    risk = [r for r in support["denominator_loss_cells"] if r["neval"] > 0]
    contribution = {}
    for row in source["scalar"]:
        feature = row["feature"]
        contribution[feature] = {arm: sum(r[f"{feature}/nominal_{arm}_contribution"] for r in risk)
                                 for arm in ("production", "source")}
        contribution[feature]["source_fraction"] = contribution[feature]["source"] / row["source_mean"]
        contribution[feature]["production_fraction"] = contribution[feature]["production"] / row["production_mean"]
    result = {"total_steps": {**total, **ratio}, "position_scores": positions,
              "sampling_tails_MC": tail, "bank_bootstrap": {
                  "replicas": support["denominator_loss_trials"], "with_evaluated_cell_loss": support["replicas_with_loss_in_evaluated_cell"],
                  "risk_cells": risk, "nominal_contribution_in_risk_cells": contribution},
              "missing_bank_event_bounds": holes["event_bounds"], "input_sha256": hashes,
              "limitations": ["Do not infer calibrated closure from diagonal scores or marginal pulls.",
                              "Sparse matched-ratio bootstrap is not well-defined for replicas losing support; none are silently discarded.",
                              "The first-order source uncertainty remains a conditional diagnostic, especially for tail cells."]}
    U.verify(hashes)
    _new_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sha256",)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study"); p.add_argument("--out", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
