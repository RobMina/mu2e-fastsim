"""Synthetic travel-only tests, not a production RNG or stock execution certificate."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest

from calofastsim.propagation import build_cdf
from calofastsim.travel import arrival_support, check_support

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check_travel_control as travel
spec = importlib.util.spec_from_file_location("response_test_fixture", ROOT / "tests/test_response_control.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def test_conditional_support_never_subtracts_base():
    cdf = build_cdf([[1, 2, 3], [0, 0, 0]])
    times = np.array([.001, .01, 1], np.float32)
    base = np.array([20, 2**24, 0], np.float32)
    counts = np.array([3, 3, 1])
    candidates, _ = arrival_support(base, [0, 0, 1], cdf, times, 1)
    observed = np.r_[candidates[0], candidates[1], times[:1]]
    result = check_support(base, [0, 0, 1], counts, observed, cdf, times, 1)
    assert result["all_arrivals_in_conditional_float32_support"]
    assert result["readouts_with_collapsed_arrival_bins"] == 1
    assert not result["stochastic_distribution_validated"]
    observed[0] = np.float32(20.5)
    assert check_support(base, [0, 0, 1], counts, observed, cdf, times, 1)["unsupported_arrival_bits"] == 1


def test_plateau_bins_unreachable_but_collapsed_reachable_group_allowed():
    cdf = np.array([[.5, .5, 1]], np.float32)
    times = np.array([0, .25, 1], np.float32)
    assert check_support([0], [0], np.array([1]), [.25], cdf, times, 1)["unsupported_arrival_bits"] == 1
    assert check_support([2**24], [0], np.array([1]), [2**24], cdf, times, 1)["unsupported_arrival_bits"] == 0


def test_unreachable_candidate_overflow_does_not_drop_valid_readout():
    maximum = np.finfo(np.float32).max
    result = check_support([maximum], [0], np.array([1]), [maximum], [[1, 1]], [0, maximum], 1)
    assert result["all_arrivals_in_conditional_float32_support"]
    with pytest.raises(ValueError, match="reachable"):
        check_support([maximum], [0], np.array([1]), [maximum], [[.5, 1]], [0, maximum], 1)


def test_support_empty_readout_manifest():
    result = check_support([], [], np.array([], np.int64), [], [[1]], [0], 1)
    assert result["photoelectrons"] == 0 and result["all_arrivals_in_conditional_float32_support"]


@pytest.mark.parametrize("base,depth,counts,observed,error", [
    ([0], [0], [0], [], "positive counts"),
    ([0], [0], [1.], [0], "positive counts"),
    ([0], [0], [2], [0], "positive counts"),
    ([np.inf], [0], [1], [0], "finite"),
    ([0], [np.float32(2**31)], [1], [0], "int32"),
    ([0], [0], [1], [np.nan], "nonfinite"),
])
def test_invalid_support_inputs(base, depth, counts, observed, error):
    with pytest.raises(ValueError, match=error):
        check_support(base, depth, np.array(counts), observed, [[1]], [0], 1)


def baseline_candidate():
    baseline = fixtures.fixture()
    baseline["readouts"]["pe_times"] = ak.values_astype(baseline["readouts"]["pe_times"], np.float32)
    candidate = copy.deepcopy(baseline)
    candidate["readouts"]["pe_times"] = candidate["readouts"]["pe_times"] + np.float32(1)
    return baseline, candidate


def test_exact_baseline_invariants_with_empty_zero_pe_and_tied_readout_sort():
    baseline, candidate = baseline_candidate()
    result = travel.compare(candidate, baseline, np.array([[.5, 1]], np.float32), [1, 2], 1, expected_events=3)
    assert result["all_travel_only_hard_gates_passed"]
    assert result["photoelectrons"] == 11
    assert result["empty_readout_events"] == 2
    assert result["baseline_readout_pointer_sipm_NPE_exact"]
    assert not result["unbiased_library_certified"]
    assert not result["stochastic_response_certified"]
    order = np.array([1, 0, 3, 2])
    candidate["readouts"] = {name: values[order] for name, values in candidate["readouts"].items()}
    candidate["readouts"]["index"] = np.arange(4)
    assert travel.compare(candidate, baseline, [[.5, 1]], [1, 2], 1, expected_events=3)["all_travel_only_hard_gates_passed"]


@pytest.mark.parametrize("mutation,error", [
    (lambda d: d["events"]["pbtime"].__setitem__(0, 1), "events"),
    (lambda d: d["steps"]["lru"].__setitem__(0, .2), "steps"),
    (lambda d: d["summaries"]["energy"].__setitem__(0, 4), "summaries"),
    (lambda d: d["readouts"]["step_index"].__setitem__(0, 9), "identity"),
    (lambda d: d["readouts"].__setitem__("pe_times", ak.values_astype(ak.Array([[21.]*2, [21.], [21.]*6, [21.]]), np.float32)), "NPE"),
])
def test_invariant_failures_not_repaired(mutation, error):
    baseline, candidate = baseline_candidate()
    mutation(candidate)
    with pytest.raises(ValueError, match=error):
        travel.compare(candidate, baseline, [[.5, 1]], [1, 2], 1, expected_events=3)


def test_time_failure_retained_and_prefix_without_baseline_rerun():
    baseline, candidate = baseline_candidate()
    candidate["readouts"]["pe_times"] = ak.values_astype(ak.Array([[21.]*3, [21.], [21.]*6, [21.5]]), np.float32)
    result = travel.compare(candidate, baseline, [[.5, 1]], [1, 2], 1, expected_events=3)
    assert not result["all_travel_only_hard_gates_passed"] and result["unsupported_arrival_bits"] == 1
    baseline, candidate = baseline_candidate()
    candidate = travel.prefix(candidate, 1)
    assert travel.compare(candidate, baseline, [[1]], [1], 1, expected_events=1)["all_travel_only_hard_gates_passed"]
    with pytest.raises(ValueError, match="count"):
        travel.prefix(baseline, 4)


def config(tmp_path):
    baseline, candidate = tmp_path / "baseline.fcl", tmp_path / "candidate.fcl"
    text = fixtures.config_text()
    baseline.write_text(text)
    candidate.write_text(text.replace("addTravelTime: false", "addTravelTime: true"))
    return baseline, candidate


def test_only_travel_boolean_may_change(tmp_path):
    baseline, candidate = config(tmp_path)
    assert travel.configuration(candidate, baseline) == fixtures.OPTIONS
    for extra in ("\nphysics.producers.deterministicRO.PEStatCorrection: true", "\nfoo: @table::bar", "\nextra: 1"):
        candidate.write_text(baseline.read_text().replace("addTravelTime: false", "addTravelTime: true") + extra)
        with pytest.raises(ValueError):
            travel.configuration(candidate, baseline)
    candidate.write_text(baseline.read_text().replace("PEStatCorrection: false", "PEStatCorrection: true")
                         .replace("addTravelTime: false", "addTravelTime: true"))
    with pytest.raises(ValueError, match="only"):
        travel.configuration(candidate, baseline)


def test_quoted_whitespace_and_hash_are_not_config_normalized(tmp_path):
    baseline, candidate = config(tmp_path)
    for before, after in (("a b.root", "ab.root"), ("a# one.root", "a# two.root")):
        baseline.write_text(fixtures.config_text() + f'\nfile: "{before}"\n')
        candidate.write_text(fixtures.config_text().replace("addTravelTime: false", "addTravelTime: true")
                             + f'\nfile: "{after}"\n')
        with pytest.raises(ValueError, match="only"):
            travel.configuration(candidate, baseline)


def test_jagged_large_integer_and_float_bits_are_exact():
    for a, b in ((np.array([2**60], np.int64), np.array([2**60 + 1], np.int64)),
                 (np.array([0.], np.float32), np.array([-0.], np.float32))):
        with pytest.raises(ValueError, match="differs"):
            travel.exact_table({"j": ak.unflatten(a, [1])}, {"j": ak.unflatten(b, [1])}, "jagged")
    with pytest.raises(ValueError, match="differs"):
        travel.exact_table({"j": ak.Array([[1], [2]])}, {"j": ak.Array([[1, 2], []])}, "jagged")


def test_no_build_driver_and_minimal_fcl():
    driver = ROOT / "scripts/travel_control.sh"
    text = driver.read_text()
    assert subprocess.run(["bash", "-n", str(driver)]).returncode == 0
    for required in ("CALO_BASELINE_SHA256", "CALO_BASELINE_FCL_SHA256", "CALO_PLUGIN_SHA256", "CALO_INPUT_SHA256",
                     'test "$CALO_EVENTS" -le 2000', 'mkdir "$CALO_OUT"', "cmp baseline_config_prediction",
                     "cmp resolved_inputs", "ldd -r", "source_build_policy.sha256"):
        assert required in text
    assert "muse build" not in text and "muse tarball" not in text and "unfiltered_ce_pilot" not in text
    fcl = (ROOT / "art/CaloFastSim/fcl/travel_control.fcl").read_text()
    assert fcl.isascii() and fcl.count("addTravelTime : true") == 1
    assert "response_control.fcl" in fcl and "CaloShowerStepMaker :" not in fcl
