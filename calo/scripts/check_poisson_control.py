"""Poisson-only saved-step hard gates plus a separately scoped prospective ideal-model diagnostic."""
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
from calofastsim.poisson import ledger, zero_fill
from calofastsim.poisson_stats import diagnostic


POLICY = {
    "version": 1,
    "arm": "PEStat_only_travel_off",
    "hard_gate": "baseline_invariants_exact_complete_accepted_CSS_SiPM_zero_fill_zero_mean_and_no_travel_time_bits",
    "event_delta_allowed": "n_readouts_only_reconciled_against_actual_rows",
    "counts_support": "nonnegative_integer_int32_with_zero_mean_exact_zero_no_finite_tail_cut",
    "association": "exact_event_CSS_ProductID_index_SiPM_never_observed_count_time_or_energy",
    "retain": "all_events_accepted_zero_means_zero_counts_photons_and_correlated_entries",
    "production_rng_validated": False,
    "stock_sampler_law_validated": False,
    "stochastic_response_certified": False,
}


def configuration(path, baseline_path):
    """Only PEStatCorrection may change; retain process/module identity and no-travel settings."""
    baseline = Path(baseline_path).read_text()
    options = deterministic.configuration(baseline_path)
    text = Path(path).read_text()
    unquoted = re.sub(r'"(?:\\.|[^"\\])*"|#[^\n]*', '""', text)
    if re.search(r"\b\w+(?:\.\w+)+\s*:", unquoted) or "@" in unquoted:
        raise ValueError("canonical expanded FHiCL required")
    pattern = r"(\bPEStatCorrection\s*:\s*)true\b"
    blocks = re.findall(r"\bdeterministicRO\s*:\s*\{([^{}]*)\}", text)
    if len(blocks) != 1 or len(re.findall(pattern, blocks[0])) != 1 or len(re.findall(pattern, text)) != 1:
        raise ValueError("one explicit PEStat-enabled deterministicRO block required")
    restored = re.sub(pattern, r"\g<1>false", text)
    def canonical(value):
        return tuple(token for token in re.findall(r'"(?:\\.|[^"\\])*"|#[^\n]*|[^\s"#]+', value)
                     if not token.startswith("#"))
    if canonical(restored) != canonical(baseline):
        raise ValueError("Poisson configuration must differ from baseline only by PEStatCorrection")
    return options


def compare(tables, baseline, *, options, expected_events):
    """Validate immutable inputs/conditions/summaries and zero-fill every accepted input sensor."""
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
    observed_time = ak.to_numpy(ak.flatten(readouts["pe_times"]))
    if observed_time.dtype != np.float32 or not np.all(np.isfinite(observed_time)):
        raise ValueError("finite float32 observed PE time storage required")
    predicted_time = np.repeat(inputs["time"][order], counts)
    time_mismatches = int(np.count_nonzero(observed_time.view(np.uint32) != predicted_time.view(np.uint32)))
    zero = inputs["mean"] == 0
    bad_zero = int(np.count_nonzero(filled[zero]))
    hard = time_mismatches == 0 and bad_zero == 0
    report = dict(events=expected_events, saved_steps=len(steps["event"]),
                  accepted_steps=int(inputs["accepted"].sum()), time_rejected_steps=int((~inputs["accepted"]).sum()),
                  accepted_CSS_SiPM_ledger_rows=len(filled), persisted_readouts=len(counts),
                  omitted_readouts_zero_filled=int(np.count_nonzero(filled == 0)),
                  observed_photoelectrons=int(filled.sum()), zero_mean_rows=int(zero.sum()),
                  positive_mean_zero_count_rows=int(np.count_nonzero((~zero) & (filled == 0))),
                  zero_mean_nonzero_count_rows=bad_zero, no_travel_time_bit_mismatches=time_mismatches,
                  mean_min=float(np.min(inputs["mean"], initial=np.inf)) if len(filled) else None,
                  mean_max=float(np.max(inputs["mean"], initial=0)),
                  source_branch_ledger_rows={"zero_mean": int(zero.sum()),
                      "positive_lt100": int(np.count_nonzero((inputs["mean"] > 0) & (inputs["mean"] < 100))),
                      "ge100_Gaussian_approximation": int(np.count_nonzero(inputs["mean"] >= 100))},
                  summaries=len(tables["summaries"]["event"]),
                  empty_step_events=int(np.count_nonzero(events["n_steps"] == 0)),
                  empty_readout_events=int(np.count_nonzero(events["n_readouts"] == 0)),
                  baseline_event_step_condition_summary_bits_exact_except_reconciled_n_readouts=True,
                  all_poisson_only_hard_gates_passed=hard, production_rng_validated=False,
                  stock_sampler_law_validated=False, stochastic_response_certified=False,
                  digitization_reconstruction_certified=False, unbiased_library_certified=False)
    if hard:
        report["ideal_Poisson_diagnostic"] = diagnostic(inputs["mean"], filled, inputs["sipm"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    for name in ("baseline", "reference", "baseline-expanded-fcl", "expanded-fcl", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-sha256", "reference-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for path, digest in ((args.baseline, args.baseline_sha256), (args.reference, args.reference_sha256)):
        if deterministic.sha256(path) != digest:
            raise ValueError(f"pinned input SHA256 mismatch: {path}")
    options = configuration(args.expanded_fcl, args.baseline_expanded_fcl)
    baseline = deterministic.read_tables(args.baseline)
    full_count = len(baseline["events"]["event"])
    deterministic.reconcile(baseline, args.reference, full_count)
    baseline_report = deterministic.compare(**baseline, options=options, expected_events=full_count)
    if not baseline_report["all_observed_deterministic_fields_exact"]:
        raise ValueError("saved baseline did not independently validate")
    report = compare(deterministic.read_tables(args.input), baseline, options=options, expected_events=args.expected_events)
    report.update(policy=POLICY, baseline_validation=baseline_report, input_sha256=deterministic.sha256(args.input),
                  baseline_sha256=args.baseline_sha256, reference_sha256=args.reference_sha256,
                  expanded_fcl_sha256=deterministic.sha256(args.expanded_fcl),
                  baseline_expanded_fcl_sha256=deterministic.sha256(args.baseline_expanded_fcl),
                  sources_sha256={str(path.relative_to(Path(__file__).parents[1])): deterministic.sha256(path) for path in
                    (Path(__file__), Path(deterministic.__file__), Path(__file__).with_name("check_travel_control.py"),
                     Path(__file__).parents[1] / "calofastsim/response.py", Path(__file__).parents[1] / "calofastsim/poisson.py",
                     Path(__file__).parents[1] / "calofastsim/poisson_stats.py")},
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   awkward=ak.__version__, uproot=uproot.__version__))
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["all_poisson_only_hard_gates_passed"]:
        raise SystemExit("Poisson-only hard gates failed; report retained")


if __name__ == "__main__":
    main()
