"""Synthetic combined-arm contract checks, independent of stock execution and RNG law."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_combined_control as control
from check_travel_control import prefix
spec = importlib.util.spec_from_file_location("combined_fixture", ROOT / "tests/test_response_control.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def fixture():
    baseline = fixtures.fixture()
    baseline["readouts"]["pe_times"] = ak.values_astype(baseline["readouts"]["pe_times"], np.float32)
    candidate = copy.deepcopy(baseline)
    candidate["readouts"]["pe_times"] = candidate["readouts"]["pe_times"] + np.float32(1)
    return baseline, candidate


def check(data, baseline, *, count=3):
    return control.compare(data, baseline, [[.5, 1]], [1, 2], 1,
                           options=fixtures.OPTIONS, expected_events=count)


def test_exact_invariants_complete_zero_fill_and_supported_arrivals():
    baseline, data = fixture()
    result = check(data, baseline)
    assert result["all_combined_hard_gates_passed"]
    assert result["accepted_CSS_SiPM_ledger_rows"] == 6
    assert result["omitted_readouts_zero_filled"] == 2
    assert result["conditional_arrival_support"]["unsupported_arrival_bits"] == 0
    assert result["summaries"] == 2
    assert not result["cross_arm_realizations_compared"]
    assert not result["production_rng_validated"]
    assert result["ideal_model_suite_status"] == "insufficient_coverage_or_power"
    assert control.POLICY["ideal_model_suite_alpha_union_bound"] == .03


def test_changed_counts_are_not_compared_to_separate_arm_realizations():
    baseline, data = fixture()
    data["readouts"]["pe_times"] = ak.values_astype(ak.Array([[21.], [22.]*5, [21.]*3, [22.]*2]), np.float32)
    result = check(data, baseline)
    assert result["all_combined_hard_gates_passed"] and result["observed_photoelectrons"] == 11
    assert result["conditional_arrival_support"]["all_arrivals_in_conditional_float32_support"]


def test_missing_positive_readout_zero_filled_before_statistics():
    baseline, data = fixture()
    data["readouts"] = {k: v[1:] for k, v in data["readouts"].items()}
    data["readouts"]["index"] = np.arange(3)
    data["events"]["n_readouts"][0] = 3
    result = check(data, baseline)
    assert result["all_combined_hard_gates_passed"]
    assert result["positive_mean_zero_count_rows"] == 1
    assert result["omitted_readouts_zero_filled"] == 3


def test_all_empty_readout_and_all_empty_inputs_remain_in_manifest():
    baseline, data = fixture()
    data["readouts"] = {k: v[:0] for k, v in data["readouts"].items()}
    data["events"]["n_readouts"][:] = 0
    result = check(data, baseline)
    assert result["all_combined_hard_gates_passed"] and result["positive_mean_zero_count_rows"] == 4
    for name in ("steps", "readouts", "summaries"):
        baseline[name] = {k: v[:0] for k, v in baseline[name].items()}
    for name in ("n_steps", "n_readouts", "n_summaries"):
        baseline["events"][name][:] = 0
    result = check(copy.deepcopy(baseline), baseline)
    assert result["all_combined_hard_gates_passed"] and result["saved_steps"] == 0


@pytest.mark.parametrize("mutation,error", [
    (lambda d: d["events"]["pbtime"].__setitem__(0, 3), "events"),
    (lambda d: d["events"]["n_readouts"].__setitem__(0, 3), "reconcile"),
    (lambda d: d["steps"]["z"].__setitem__(0, 5), "steps"),
    (lambda d: d["summaries"]["energy"].__setitem__(0, 5), "summaries"),
    (lambda d: d["summaries"].__setitem__("member_indices", ak.Array([[2, 0], [0]])), "summaries"),
    (lambda d: d["readouts"]["sipm"].__setitem__(0, 9), "identity"),
    (lambda d: d["readouts"]["step_index"].__setitem__(0, 1), "identity"),
    (lambda d: d["readouts"]["step_index"].__setitem__(0, 0), "duplicate"),
    (lambda d: d["readouts"].__setitem__("pe_times", ak.Array([[21.], [21.], [21.], [21.]])), "float32"),
])
def test_no_invariant_or_association_repair(mutation, error):
    baseline, data = fixture()
    mutation(data)
    with pytest.raises(ValueError, match=error):
        check(data, baseline)


def test_unsupported_arrival_retains_failure_report_and_no_ideal_pass():
    baseline, data = fixture()
    data["readouts"]["pe_times"] = ak.values_astype(ak.Array([[21.5], [21.], [21.], [21.]]), np.float32)
    result = check(data, baseline)
    assert not result["all_combined_hard_gates_passed"]
    assert result["conditional_arrival_support"]["unsupported_arrival_bits"] == 1
    assert "ideal_model_suite_status" not in result


def test_prefix_and_tied_readouts_do_not_change_identity():
    baseline, data = fixture()
    order = np.array([1, 0, 3, 2])
    data["readouts"] = {k: v[order] for k, v in data["readouts"].items()}
    data["readouts"]["index"] = np.arange(4)
    assert check(data, baseline)["all_combined_hard_gates_passed"]
    assert check(prefix(data, 1), baseline, count=1)["events"] == 1


def test_only_two_switches_may_change_and_quoted_tokens_are_preserved(tmp_path):
    baseline, candidate = tmp_path / "baseline.fcl", tmp_path / "candidate.fcl"
    text = fixtures.config_text()
    enabled = text.replace("PEStatCorrection: false", "PEStatCorrection: true").replace("addTravelTime: false", "addTravelTime: true")
    baseline.write_text(text)
    candidate.write_text(enabled)
    assert control.configuration(candidate, baseline) == fixtures.OPTIONS
    for altered in (enabled.replace("addTravelTime: true", "addTravelTime: false"),
                    enabled.replace("PEStatCorrection: true", "PEStatCorrection: false"),
                    enabled + "\nfoo: @table::bar", enabled + "\nphysics.producers.foo: 1"):
        candidate.write_text(altered)
        with pytest.raises(ValueError):
            control.configuration(candidate, baseline)
    baseline.write_text(text + '\nfile: "a # b.root"\n')
    candidate.write_text(enabled + '\nfile: "a# b.root"\n')
    with pytest.raises(ValueError, match="only"):
        control.configuration(candidate, baseline)


def test_driver_provenance_and_no_build_transport():
    driver = ROOT / "scripts/combined_control.sh"
    assert subprocess.run(["bash", "-n", str(driver)]).returncode == 0
    text = driver.read_text()
    for value in ("CALO_INPUT_SHA256", "CALO_BASELINE_SHA256", "CALO_BASELINE_FCL_SHA256", "CALO_PLUGIN_SHA256",
                  "CALO_PRODUCER_SHA256", "CALO_CLHEP_SHA256", "CALO_CLHEP_RANDOM_SHA256", "CALO_TABLE_SHA256",
                  "CET_PLUGIN_PATH", "producer.linkage.txt", "sampler_provenance.sha256", "combined_stats.py",
                  "travel_stats.py", "poisson_stats.py", "COMBINED_CONTROL.md", "source_build_policy.sha256",
                  "cmp baseline_config_prediction", "cmp resolved_inputs", 'test "$CALO_EVENTS" -le 2000'):
        assert value in text
    assert "muse build" not in text and "muse tarball" not in text and "unfiltered_ce_pilot.sh" not in text
    fcl = (ROOT / "art/CaloFastSim/fcl/combined_control.fcl").read_text()
    assert fcl.isascii() and fcl.count("PEStatCorrection : true") == 1 and fcl.count("addTravelTime : true") == 1
    assert "response_control.fcl" in fcl
