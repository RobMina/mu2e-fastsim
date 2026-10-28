"""Conditional input-sizing scenarios from a certified full-generator support census.

Poisson independent-event rates and equal-sized independent file units are planning
assumptions, not a certified tail model. This does not authorize propagation or jobs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2

from .descendant_audit import digest


def required_support(cells, replicas, failure_budget):
    """Sufficient distinct supporting files using (1-m/n)^n <= exp(-m), union bound."""
    if cells < 1 or replicas < 1 or not 0 < failure_budget < 1:
        raise ValueError("invalid support target")
    return int(np.ceil(np.log(cells * replicas / failure_budget)))


def minimum_files(rates, events_per_file, support, acquisition_failure=.05):
    """Minimum equal-sized file count for the modeled support-acquisition union bound."""
    rates = np.asarray(rates, dtype=float)
    if np.any(rates <= 0) or not np.isfinite(rates).all() or events_per_file < 1 or support < 1:
        raise ValueError("positive finite rates/file size/support required")
    if not 0 < acquisition_failure < 1:
        raise ValueError("invalid acquisition failure target")
    probability = -np.expm1(-rates * events_per_file)
    failure = lambda n: float(binom.cdf(support - 1, n, probability).sum())
    lo, hi = support - 1, support
    while failure(hi) > acquisition_failure:
        hi *= 2
        if hi > 10 ** 12:
            raise ValueError("sizing exceeds supported file count")
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if failure(mid) <= acquisition_failure:
            hi = mid
        else:
            lo = mid
    return {"files": hi, "generated_events": hi * events_per_file,
            "modeled_acquisition_failure_union_bound": failure(hi)}


def calculate(audit, replicas=1000, failure_budget=.01, acquisition_failure=.05, confidence=.95):
    counts = np.asarray(audit["per_file_counts"], dtype=np.int64).sum(axis=0)
    needed = np.asarray(audit["corrected"]["nominal_showers"]) > 0
    events = int(audit["generator_events"])
    file_events = np.asarray(audit["per_file_events"], dtype=np.int64)
    if not np.all(file_events == file_events[0]) or file_events.sum() != events:
        raise ValueError("requires equal-sized full-generator file census")
    if not 0 < confidence < 1 or np.any(counts[needed] <= 0):
        raise ValueError("invalid confidence or unsupported nominal shower cell")
    k = int(needed.sum())
    support = required_support(k, replicas, failure_budget)
    observed = counts[needed]
    mle = observed / events
    lower = .5 * chi2.ppf((1 - confidence) / k, 2 * observed) / events
    result = {"cells": k, "replicas": replicas, "ensemble_failure_budget": failure_budget,
              "required_distinct_supporting_files_per_cell": support,
              "events_per_file": int(file_events[0]), "observed_generated_events": events,
              "point_rate_scenario": minimum_files(mle, int(file_events[0]), support, acquisition_failure),
              "simultaneous_poisson_lower_rate_scenario": minimum_files(lower, int(file_events[0]), support,
                                                                         acquisition_failure),
              "rate_confidence": confidence, "bottleneck_cells": [],
              "limitations": ["New independent bank; counts/rates assumed Poisson with unchanged physical configuration.",
                              "Acquisition probability and rate confidence are distinct; no unconditional 95% claim.",
                              "Raw zero support only; excludes integer-normalization rounding, saturation and unseen/evaluated tails.",
                              "All support and reach-probability gates must be rerun on actual inputs before propagation.",
                              "No runtime, resource request, submission or library publication is implied."]}
    for i in np.flatnonzero(needed & (counts <= 2)):
        result["bottleneck_cells"].append({**audit["bins"][i], "observed_primaries": int(counts[i]),
                                           "rate_per_event": float(counts[i] / events)})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    result = calculate(json.loads(Path(a.audit).read_text()))
    result.update(audit=a.audit, audit_sha256=digest(a.audit), source_sha256=digest(__file__))
    with Path(a.out).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
