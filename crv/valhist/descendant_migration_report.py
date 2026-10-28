"""Read retained expectation/replica/placement reports into an additive deficit budget.

No fitted correction. Numerical replica-mean errors are conditional on fixed inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .descendant_audit import digest
from .descendant_replicas import projection
from crvfastsim.tier1b import GROUPS


def calculate(expected, replicas, placement):
    p = projection()
    ref = p @ np.asarray(replicas["reference"])
    if not np.allclose(ref, expected["arms"]["context"]["sector_reference"], rtol=1e-12, atol=1e-8):
        raise ValueError("reference population differs")
    pre = np.asarray(replicas["pre_samples"]) @ p.T
    post = np.asarray(replicas["post_samples"]) @ p.T
    exact = np.asarray(expected["arms"]["context"]["sector_mean"])
    sd = np.asarray(expected["arms"]["context"]["sector_sampling_sd"])
    if len(pre) != replicas["replicas"] or len(pre) < 2:
        raise ValueError("replica count mismatch")
    pre_se = sd / np.sqrt(len(pre))
    mean = post.mean(0)
    mc_se = post.std(0, ddof=1) / np.sqrt(len(post))
    budget = []
    for j, group in enumerate(GROUPS):
        i = j + 4
        migration = post[:, i] - pre[:, i]
        budget.append({"sector": group, "g4_otherbar_energy": ref[i], "exact_pre_energy": exact[i],
                       "pre_replica_mean_minus_exact_over_mc_se": (pre[:, i].mean() - exact[i]) / pre_se[i],
                       "post_mean_ratio": mean[i] / ref[i], "post_mean_ratio_mc_se": mc_se[i] / ref[i],
                       "migration_energy": migration.mean(), "migration_mc_se": migration.std(ddof=1) / np.sqrt(len(pre)),
                       "migration_ratio": migration.mean() / ref[i],
                       "retained_post_ratio": (p @ np.asarray(replicas["retained_post"]))[i] / ref[i]})
    rows = placement["views"]["migration"]
    flow = []
    for group in GROUPS:
        selected = [r for r in rows if r["source_kind"] == "otherbar" and r["target_group"] == group]
        own = sum(r["e_cand"] for r in selected if r["target_kind"] in ("anchor", "another_own_bar"))
        outside = sum(r["e_cand"] for r in selected if r["target_kind"] == "otherbar")
        flow.append({"sector": group, "retained_pasted_otherbar_energy": outside,
                     "retained_reclassified_ownbar_energy": own,
                     "reclassified_fraction_of_energy_landing_in_sector": own / (own + outside)})
    return {"budget": budget, "retained_placement": flow,
            "limitation": "Pre/post sector difference combines destination migration with own-bar reclassification; no tuned response, donor error or calibrated significance."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("expectation", "replicas", "placement", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    inputs = [a.expectation, a.replicas, a.placement]
    from .descendant_inputs import bind_reports
    reports = [json.loads(Path(path).read_text()) for path in inputs]
    bind_reports(*reports)
    result = calculate(*reports)
    result["hashes"] = {path: digest(path) for path in inputs + [__file__]}
    with Path(a.out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
