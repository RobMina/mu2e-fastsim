"""Travel-only saved-step control: exact baseline invariants and conditional arrival support."""
from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

import check_response_control as deterministic
from check_propagation import read_table, sha256
from calofastsim.compression import lookup
from calofastsim.identities import keys
from calofastsim.travel import check_support
from calofastsim.travel_stats import diagnostic as distribution_diagnostic


POLICY = {
    "version": 1,
    "arm": "travel_only_PEStat_off",
    "hard_gate": "all_baseline_invariants_exact_and_all_PE_bits_in_conditional_support",
    "support": "float32(baseline_stored_float_time + actual_table_float_center), reachable lower_bound bins only",
    "retain": "all_events_zero_PE_steps_empty_responses_and_finite_late_times",
    "association": "exact_event_CSS_product_index_SiPM_never_time_or_energy",
    "production_rng_validated": False,
    "stochastic_distribution_validated": False,
    "same_seed_equivalence_claimed": False,
}


def configuration(path, baseline_path):
    """Require the canonical expansion to differ only by the travel boolean."""
    baseline = Path(baseline_path).read_text()
    options = deterministic.configuration(baseline_path)
    text = Path(path).read_text()
    if re.search(r"\b\w+(?:\.\w+)+\s*:", re.sub(r'"[^"\n]*"|#[^\n]*', '""', text)) or "@" in text:
        raise ValueError("canonical expanded FHiCL required")
    pattern = r"(\baddTravelTime\s*:\s*)true\b"
    blocks = re.findall(r"\bdeterministicRO\s*:\s*\{([^{}]*)\}", text)
    if len(blocks) != 1 or len(re.findall(pattern, blocks[0])) != 1 or len(re.findall(pattern, text)) != 1:
        raise ValueError("one explicit travel-enabled deterministicRO block required")
    restored = re.sub(pattern, r"\g<1>false", text)
    def canonical(value):
        tokens = re.findall(r'"(?:\\.|[^"\\])*"|#[^\n]*|[^\s"#]+', value)
        return tuple(token for token in tokens if not token.startswith("#"))
    if canonical(restored) != canonical(baseline):
        raise ValueError("travel configuration must differ from baseline only by addTravelTime")
    return options


def exact_table(left, right, label):
    if set(left) != set(right):
        raise ValueError(f"{label} field registry mismatch")
    for name in left:
        a, b = left[name], right[name]
        if isinstance(a, ak.Array) or isinstance(b, ak.Array):
            equal = np.array_equal(ak.to_numpy(ak.num(a)), ak.to_numpy(ak.num(b)))
            a, b = ak.to_numpy(ak.flatten(a)), ak.to_numpy(ak.flatten(b))
        else:
            equal = True
        a, b = np.asarray(a), np.asarray(b)
        equal = equal and a.dtype == b.dtype and a.shape == b.shape
        if equal:
            equal = np.array_equal(a.view(np.uint8), b.view(np.uint8))
        if not equal:
            raise ValueError(f"{label} differs from saved baseline in {name}")


def readout_keys(readouts):
    fields = {name: readouts[name] for name in deterministic.EVENT_FIELDS}
    fields.update(product_id=readouts["step_product_id"], index=readouts["step_index"], sipm=readouts["sipm"])
    return keys(fields, deterministic.RO_FIELDS)


def prefix(tables, count):
    if count <= 0 or count > len(tables["events"]["event"]):
        raise ValueError("independent expected event count exceeds saved baseline")
    result = {}
    for name, table in tables.items():
        event_index = lookup(keys(tables["events"], deterministic.EVENT_FIELDS), keys(table, deterministic.EVENT_FIELDS))
        result[name] = {field: value[event_index < count] for field, value in table.items()}
    return result


def compare(tables, baseline, cdf, times, dz, *, expected_events, ideal_uniform_diagnostic=False):
    """No baseline mutation, time substitution, association repair or stochastic test."""
    baseline = prefix(baseline, expected_events)
    for name in ("events", "steps", "summaries"):
        exact_table(tables[name], baseline[name], name)
    events, steps, readouts = (tables[name] for name in ("events", "steps", "readouts"))
    deterministic.validate_rows(events, readouts, "n_readouts", "ro_product_id")
    expected_keys, observed_keys = readout_keys(baseline["readouts"]), readout_keys(readouts)
    if (len(np.unique(observed_keys)) != len(observed_keys) or
            not np.array_equal(np.sort(expected_keys), np.sort(observed_keys))):
        raise ValueError("readout CSS/SiPM identity set differs from baseline")
    order = lookup(expected_keys, observed_keys)
    counts = ak.to_numpy(ak.num(readouts["pe_times"]))
    baseline_counts = ak.to_numpy(ak.num(baseline["readouts"]["pe_times"]))
    if not np.array_equal(counts, baseline_counts[order]):
        raise ValueError("individual readout NPE differs from deterministic baseline")
    baseline_flat = ak.to_numpy(ak.flatten(baseline["readouts"]["pe_times"])).astype(np.float32)
    offsets = np.cumsum(baseline_counts) - baseline_counts
    if np.any(baseline_counts <= 0):
        raise ValueError("empty baseline readout")
    base = baseline_flat[offsets]
    if not np.array_equal(baseline_flat, np.repeat(base, baseline_counts)):
        raise ValueError("baseline is not a no-travel constant-time readout")
    references = {name: readouts[name] for name in deterministic.EVENT_FIELDS}
    references.update(product_id=readouts["step_product_id"], index=readouts["step_index"])
    si = lookup(keys(steps, deterministic.STEP_FIELDS), keys(references, deterministic.STEP_FIELDS))
    ei = lookup(keys(events, deterministic.EVENT_FIELDS), keys(references, deterministic.EVENT_FIELDS))
    depth = (events["crystal_length"][ei].astype(np.float32) - steps["z"][si].astype(np.float32)).astype(np.float32)
    observed = ak.to_numpy(ak.flatten(readouts["pe_times"]))
    if observed.dtype != np.float32:
        raise ValueError("observed PE times must retain float32 stock storage")
    result = check_support(base[order], depth, counts, observed, cdf, times, dz)
    if ideal_uniform_diagnostic:
        result["ideal_uniform_diagnostic"] = distribution_diagnostic(base[order], depth, counts, observed,
                                                                   readouts["sipm"], cdf, times, dz)
    result.update(events=expected_events, saved_steps=len(steps["event"]), readouts=len(counts),
                  summaries=len(tables["summaries"]["event"]), baseline_event_step_condition_summary_bits_exact=True,
                  baseline_readout_pointer_sipm_NPE_exact=True,
                  empty_step_events=int(np.count_nonzero(events["n_steps"] == 0)),
                  empty_readout_events=int(np.count_nonzero(events["n_readouts"] == 0)),
                  all_travel_only_hard_gates_passed=result["all_arrivals_in_conditional_float32_support"],
                  stochastic_response_certified=False, digitization_reconstruction_certified=False,
                  unbiased_library_certified=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--baseline-expanded-fcl", type=Path, required=True)
    parser.add_argument("--expanded-fcl", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--table-sha256", required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--ideal-uniform-diagnostic", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.expected_events <= 0:
        raise ValueError("positive independent event count required")
    for path, digest in ((args.baseline, args.baseline_sha256), (args.reference, args.reference_sha256),
                         (args.table, args.table_sha256)):
        if sha256(path) != digest:
            raise ValueError(f"pinned input SHA256 mismatch: {path}")
    options = configuration(args.expanded_fcl, args.baseline_expanded_fcl)
    baseline = deterministic.read_tables(args.baseline)
    full_count = len(baseline["events"]["event"])
    deterministic.reconcile(baseline, args.reference, full_count)
    baseline_report = deterministic.compare(**baseline, options=options, expected_events=full_count)
    if not baseline_report["all_observed_deterministic_fields_exact"]:
        raise ValueError("saved deterministic baseline did not validate")
    tables = deterministic.read_tables(args.input)
    _, cdf, times, dz, axes = read_table(args.table, "h_zposTime")
    report = compare(tables, baseline, cdf, times, dz, expected_events=args.expected_events,
                     ideal_uniform_diagnostic=args.ideal_uniform_diagnostic)
    report.update(policy=POLICY, baseline_validation=baseline_report, table_axes=axes,
                  input_sha256=sha256(args.input), baseline_sha256=args.baseline_sha256,
                  reference_sha256=args.reference_sha256, table_sha256=args.table_sha256,
                  expanded_fcl_sha256=sha256(args.expanded_fcl),
                  baseline_expanded_fcl_sha256=sha256(args.baseline_expanded_fcl),
                  checker_sha256=sha256(__file__),
                  support_backend_sha256=sha256(Path(__file__).parents[1] / "calofastsim/travel.py"),
                  distribution_backend_sha256=sha256(Path(__file__).parents[1] / "calofastsim/travel_stats.py"),
                  deterministic_checker_sha256=sha256(deterministic.__file__),
                  environment={"python": platform.python_version(), "numpy": np.__version__,
                               "awkward": ak.__version__, "uproot": uproot.__version__})
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["all_travel_only_hard_gates_passed"]:
        raise SystemExit("travel-only hard gates failed; report retained")


if __name__ == "__main__":
    main()
