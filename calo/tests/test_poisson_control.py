"""Synthetic PE-statistics-only invariants/zero-fill tests, not stock execution."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest

from calofastsim.poisson import ledger, zero_fill

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_poisson_control as control
spec = importlib.util.spec_from_file_location("poisson_fixture", ROOT / "tests/test_response_control.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def fixture():
    baseline = fixtures.fixture()
    baseline["readouts"]["pe_times"] = ak.values_astype(baseline["readouts"]["pe_times"], np.float32)
    return baseline, copy.deepcopy(baseline)


def check(data, baseline):
    return control.compare(data, baseline, options=fixtures.OPTIONS, expected_events=3)


def test_all_accepted_step_sipms_include_zero_means_and_missing_readouts():
    baseline, data = fixture()
    result = check(data, baseline)
    assert result["all_poisson_only_hard_gates_passed"]
    assert result["accepted_CSS_SiPM_ledger_rows"] == 6
    assert result["omitted_readouts_zero_filled"] == 2
    assert result["zero_mean_rows"] == 2
    assert result["observed_photoelectrons"] == 11
    assert result["empty_readout_events"] == 2
    assert not result["stock_sampler_law_validated"]
    assert not result["unbiased_library_certified"]
    assert result["ideal_Poisson_diagnostic"]["model_diagnostic_status"] == "insufficient_coverage_or_power"


def test_positive_mean_deterministic_zero_is_still_a_sample_and_can_emit_ro():
    baseline, _ = fixture()
    baseline["steps"]["energy_birks"][3] = .1
    baseline["summaries"]["energy"][1] = np.float32(.1)
    assert fixtures.check(baseline)["all_observed_deterministic_fields_exact"]
    data = copy.deepcopy(baseline)
    result = check(data, baseline)
    assert result["positive_mean_zero_count_rows"] == 2
    assert result["zero_mean_rows"] == 0
    ro = data["readouts"]
    for name in ro:
        if name != "pe_times":
            ro[name] = np.r_[ro[name], ro[name][:1]]
    ro["event"][-1] = 2
    ro["index"][-1] = 0
    ro["step_index"][-1] = 0
    ro["pe_times"] = ak.values_astype(ak.Array([[20.]*3, [20.], [20.]*6, [20.], [50.]]), np.float32)
    data["events"]["n_readouts"][1] = 1
    result = check(data, baseline)
    assert result["all_poisson_only_hard_gates_passed"]
    assert result["positive_mean_zero_count_rows"] == 1
    assert result["persisted_readouts"] == 5


def test_no_readouts_retains_all_positive_mean_and_empty_events():
    baseline, data = fixture()
    data["readouts"] = {k: v[:0] for k, v in data["readouts"].items()}
    data["events"]["n_readouts"][:] = 0
    result = check(data, baseline)
    assert result["all_poisson_only_hard_gates_passed"]
    assert result["omitted_readouts_zero_filled"] == 6
    assert result["positive_mean_zero_count_rows"] == 4
    assert result["empty_readout_events"] == 3


def test_tied_ro_order_is_not_identity():
    baseline, data = fixture()
    order = np.array([1, 0, 3, 2])
    data["readouts"] = {k: v[order] for k, v in data["readouts"].items()}
    data["readouts"]["index"] = np.arange(4)
    assert check(data, baseline)["all_poisson_only_hard_gates_passed"]


@pytest.mark.parametrize("mutation,error", [
    (lambda d: d["events"]["pbtime"].__setitem__(0, 1), "events"),
    (lambda d: d["events"]["n_readouts"].__setitem__(0, 3), "reconcile"),
    (lambda d: d["steps"]["energy_birks"].__setitem__(0, 5), "steps"),
    (lambda d: d["summaries"]["energy"].__setitem__(0, 5), "summaries"),
    (lambda d: d["summaries"].__setitem__("member_indices", ak.Array([[2, 0], [0]])), "summaries"),
    (lambda d: d["readouts"]["sipm"].__setitem__(0, 9), "identity"),
    (lambda d: d["readouts"]["step_index"].__setitem__(0, 1), "identity"),
    (lambda d: d["readouts"]["step_index"].__setitem__(0, 0), "duplicate"),
    (lambda d: d["readouts"].__setitem__("pe_times", ak.Array([[20.], [20.], [20.], [20.]])), "float32"),
])
def test_invariants_cannot_be_repaired(mutation, error):
    baseline, data = fixture()
    mutation(data)
    with pytest.raises(ValueError, match=error):
        check(data, baseline)


def test_time_failure_report_preserved():
    baseline, data = fixture()
    data["readouts"]["pe_times"] = ak.values_astype(ak.Array([[21.]*3, [20.], [20.]*6, [20.]]), np.float32)
    result = check(data, baseline)
    assert not result["all_poisson_only_hard_gates_passed"]
    assert result["no_travel_time_bit_mismatches"] == 3
    assert "ideal_Poisson_diagnostic" not in result


def test_nonpositive_counts_and_large_counts_rejected():
    baseline, _ = fixture()
    inputs = ledger(baseline["events"], baseline["steps"], fixtures.OPTIONS)
    for value in (0, -1, 2**31):
        with pytest.raises(ValueError, match="positive int32"):
            zero_fill(inputs["identity"], inputs["identity"][:1], [value])


def test_zero_mean_nonzero_fails_with_retained_report():
    baseline, data = fixture()
    baseline["steps"]["energy_birks"][0] = 0
    data["steps"]["energy_birks"][0] = 0
    result = check(data, baseline)
    assert not result["all_poisson_only_hard_gates_passed"]
    assert result["zero_mean_nonzero_count_rows"] == 2


def test_only_pestat_boolean_and_quoted_config_bytes(tmp_path):
    baseline, candidate = tmp_path / "baseline.fcl", tmp_path / "candidate.fcl"
    text = fixtures.config_text()
    baseline.write_text(text)
    enabled = text.replace("PEStatCorrection: false", "PEStatCorrection: true")
    candidate.write_text(enabled)
    assert control.configuration(candidate, baseline) == fixtures.OPTIONS
    for altered in (enabled.replace("addTravelTime: false", "addTravelTime: true"),
                    enabled + "\nphysics.producers.foo: true", enabled + "\nfoo: @table::bar"):
        candidate.write_text(altered)
        with pytest.raises(ValueError):
            control.configuration(candidate, baseline)
    baseline.write_text(text + '\nfile: "a # b.root"\n')
    candidate.write_text(enabled + '\nfile: "a# b.root"\n')
    with pytest.raises(ValueError, match="only"):
        control.configuration(candidate, baseline)


def test_no_build_driver_and_minimal_fcl():
    driver = ROOT / "scripts/poisson_control.sh"
    assert subprocess.run(["bash", "-n", str(driver)]).returncode == 0
    text = driver.read_text()
    for value in ("CALO_BASELINE_SHA256", "CALO_BASELINE_FCL_SHA256", "CALO_PLUGIN_SHA256", "CALO_INPUT_SHA256",
                  "cmp baseline_config_prediction", "cmp resolved_inputs", "ldd -r", 'test "$CALO_EVENTS" -le 2000'):
        assert value in text
    assert "muse build" not in text and "muse tarball" not in text
    fcl = (ROOT / "art/CaloFastSim/fcl/poisson_control.fcl").read_text()
    assert fcl.isascii() and fcl.count("PEStatCorrection : true") == 1
    assert "response_control.fcl" in fcl and "addTravelTime" not in fcl
