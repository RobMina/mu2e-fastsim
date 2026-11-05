"""Combined saved-CSS stock arm: complete count ledger, arrival support and scoped ideal diagnostics."""
from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

import awkward as ak
import numpy as np
import scipy
import uproot

import check_response_control as deterministic
from check_travel_control import exact_table, prefix, readout_keys
from check_propagation import read_table
from calofastsim.compression import lookup
from calofastsim.identities import keys
from calofastsim.poisson import ledger, zero_fill
from calofastsim.poisson_stats import diagnostic as count_diagnostic
from calofastsim.travel import check_support
from calofastsim.travel_stats import diagnostic as travel_diagnostic
from calofastsim.combined_stats import diagnostic as joint_diagnostic


POLICY = {
    "version": 1,
    "arm": "PEStat_and_travel_on_same_saved_CSS",
    "hard_gate": "exact_baseline_invariants_except_reconciled_n_readouts_complete_zero_fill_and_arrival_support",
    "cross_arm_realizations_compared": False,
    "association": "exact_event_CSS_ProductID_index_SiPM_including_absent_zero_count_readouts",
    "retain": "all_events_accepted_zero_means_zero_counts_photons_and_correlated_entries",
    "ideal_model_suite_alpha_union_bound": 0.03,
    "ideal_suite_families": "count_CDF_.01_conditional_travel_CDF_.01_joint_marked_Poisson_.01",
    "production_rng_validated": False,
    "stock_sampler_law_validated": False,
    "stochastic_response_certified": False,
}


def configuration(path, baseline_path):
    """Restore exactly the two stochastic switches and compare canonical token streams."""
    baseline = Path(baseline_path).read_text()
    options = deterministic.configuration(baseline_path)
    text = Path(path).read_text()
    unquoted = re.sub(r'"(?:\\.|[^"\\])*"|#[^\n]*', '""', text)
    if re.search(r"\b\w+(?:\.\w+)+\s*:", unquoted) or "@" in unquoted:
        raise ValueError("canonical expanded FHiCL required")
    blocks = re.findall(r"\bdeterministicRO\s*:\s*\{([^{}]*)\}", text)
    if len(blocks) != 1:
        raise ValueError("one explicit deterministicRO block required")
    restored = text
    for name in ("PEStatCorrection", "addTravelTime"):
        pattern = r"(\b" + name + r"\s*:\s*)true\b"
        if len(re.findall(pattern, blocks[0])) != 1 or len(re.findall(pattern, text)) != 1:
            raise ValueError(f"one explicit {name}:true required")
        restored = re.sub(pattern, r"\g<1>false", restored)
    def canonical(value):
        return tuple(token for token in re.findall(r'"(?:\\.|[^"\\])*"|#[^\n]*|[^\s"#]+', value)
                     if not token.startswith("#"))
    if canonical(restored) != canonical(baseline):
        raise ValueError("combined configuration must differ only by the two stochastic switches")
    return options


def compare(tables, baseline, cdf, times, dz, *, options, expected_events):
    """Use predicted inputs only for complete ledger/support, never separate-arm sampled counts."""
    baseline = prefix(baseline, expected_events)
    for name in ("steps", "summaries"):
        exact_table(tables[name], baseline[name], name)
    if set(tables["events"]) != set(baseline["events"]):
        raise ValueError("events field registry mismatch")
    exact_table({k: v for k, v in tables["events"].items() if k != "n_readouts"},
                {k: v for k, v in baseline["events"].items() if k != "n_readouts"}, "events")
    events, steps, readouts = (tables[name] for name in ("events", "steps", "readouts"))
    if set(readouts) != set(baseline["readouts"]):
        raise ValueError("readouts field registry mismatch")
    deterministic.validate_rows(events, readouts, "n_readouts", "ro_product_id")
    inputs = ledger(events, steps, options)
    counts = ak.to_numpy(ak.num(readouts["pe_times"]))
    filled, order = zero_fill(inputs["identity"], readout_keys(readouts), counts)
    observed = ak.to_numpy(ak.flatten(readouts["pe_times"]))
    if observed.dtype != np.float32 or not np.all(np.isfinite(observed)):
        raise ValueError("finite float32 observed PE time storage required")
    ei = lookup(keys(events, deterministic.EVENT_FIELDS), keys(steps, deterministic.EVENT_FIELDS))
    step_depth = (events["crystal_length"][ei].astype(np.float32) - steps["z"].astype(np.float32)).astype(np.float32)
    depth = step_depth[inputs["step_index"]]
    support = check_support(inputs["time"][order], depth[order], counts, observed, cdf, times, dz)
    zero = inputs["mean"] == 0
    bad_zero = int(np.count_nonzero(filled[zero]))
    hard = support["all_arrivals_in_conditional_float32_support"] and bad_zero == 0
    report = dict(events=expected_events, saved_steps=len(steps["event"]),
                  accepted_steps=int(inputs["accepted"].sum()), time_rejected_steps=int((~inputs["accepted"]).sum()),
                  accepted_CSS_SiPM_ledger_rows=len(filled), persisted_readouts=len(counts),
                  omitted_readouts_zero_filled=int(np.count_nonzero(filled == 0)),
                  observed_photoelectrons=int(filled.sum()), zero_mean_rows=int(zero.sum()),
                  positive_mean_zero_count_rows=int(np.count_nonzero((~zero) & (filled == 0))),
                  zero_mean_nonzero_count_rows=bad_zero, conditional_arrival_support=support,
                  mean_min=float(np.min(inputs["mean"], initial=np.inf)) if len(filled) else None,
                  mean_max=float(np.max(inputs["mean"], initial=0)), summaries=len(tables["summaries"]["event"]),
                  source_branch_ledger_rows={"zero_mean": int(zero.sum()),
                      "positive_lt100": int(np.count_nonzero((inputs["mean"] > 0) & (inputs["mean"] < 100))),
                      "ge100_Gaussian_approximation": int(np.count_nonzero(inputs["mean"] >= 100))},
                  empty_step_events=int(np.count_nonzero(events["n_steps"] == 0)),
                  empty_readout_events=int(np.count_nonzero(events["n_readouts"] == 0)),
                  baseline_event_step_condition_summary_bits_exact_except_reconciled_n_readouts=True,
                  all_combined_hard_gates_passed=bool(hard), cross_arm_realizations_compared=False,
                  production_rng_validated=False, stock_sampler_law_validated=False, stochastic_response_certified=False,
                  digitization_reconstruction_certified=False, unbiased_library_certified=False)
    if hard:
        count = count_diagnostic(inputs["mean"], filled, inputs["sipm"])
        travel = travel_diagnostic(inputs["time"][order], depth[order], counts, observed,
                                   inputs["sipm"][order], cdf, times, dz)
        joint = joint_diagnostic(inputs["mean"], filled, inputs["sipm"], inputs["time"], depth,
                                 order, observed, cdf, times, dz)
        statuses = [item["model_diagnostic_status"] for item in (count, travel, joint)]
        report.update(ideal_Poisson_diagnostic=count, conditional_ideal_uniform_diagnostic=travel,
                      ideal_joint_marked_Poisson_diagnostic=joint,
                      ideal_model_suite_status=("rejected" if "rejected" in statuses else "not_rejected"
                                               if all(s == "not_rejected" for s in statuses)
                                               else "insufficient_coverage_or_power"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    for name in ("baseline", "reference", "baseline-expanded-fcl", "expanded-fcl", "table", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-sha256", "reference-sha256", "table-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for path, digest in ((args.baseline, args.baseline_sha256), (args.reference, args.reference_sha256),
                         (args.table, args.table_sha256)):
        if deterministic.sha256(path) != digest:
            raise ValueError(f"pinned input SHA256 mismatch: {path}")
    options = configuration(args.expanded_fcl, args.baseline_expanded_fcl)
    baseline = deterministic.read_tables(args.baseline)
    full_count = len(baseline["events"]["event"])
    deterministic.reconcile(baseline, args.reference, full_count)
    baseline_report = deterministic.compare(**baseline, options=options, expected_events=full_count)
    if not baseline_report["all_observed_deterministic_fields_exact"]:
        raise ValueError("saved baseline did not independently validate")
    _, cdf, times, dz, axes = read_table(args.table, "h_zposTime")
    report = compare(deterministic.read_tables(args.input), baseline, cdf, times, dz,
                     options=options, expected_events=args.expected_events)
    source_root = Path(__file__).parents[1]
    sources = [Path(__file__), Path(deterministic.__file__), Path(__file__).with_name("check_travel_control.py"),
               Path(__file__).with_name("check_propagation.py")]
    sources += [source_root / "calofastsim" / name for name in
                ("response.py", "poisson.py", "poisson_stats.py", "propagation.py", "travel.py", "travel_stats.py", "combined_stats.py",
                 "identities.py", "compression.py")]
    sources += [source_root / "tests" / name for name in ("test_combined_control.py", "test_combined_stats.py")]
    report.update(policy=POLICY, baseline_validation=baseline_report, table_axes=axes,
                  input_sha256=deterministic.sha256(args.input), baseline_sha256=args.baseline_sha256,
                  reference_sha256=args.reference_sha256, table_sha256=args.table_sha256,
                  expanded_fcl_sha256=deterministic.sha256(args.expanded_fcl),
                  baseline_expanded_fcl_sha256=deterministic.sha256(args.baseline_expanded_fcl),
                  sources_sha256={str(path.relative_to(source_root)): deterministic.sha256(path) for path in sources},
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   awkward=ak.__version__, uproot=uproot.__version__))
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["all_combined_hard_gates_passed"]:
        raise SystemExit("combined hard gates failed; report retained")


if __name__ == "__main__":
    main()
