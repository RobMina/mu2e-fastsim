"""Synthetic checker/export contract tests, not a stock execution certificate."""
import importlib.util
from pathlib import Path

import awkward as ak
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_response_control", ROOT / "scripts/check_response_control.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)
OPTIONS = dict(birks=True, longitudinal=True, digitization_start=10., digitization_end=100., digitization_buffer=0.)


def table(event, **fields):
    ids = dict(run=np.ones(len(event), np.int64), subrun=np.ones(len(event), np.int64), event=np.array(event, np.int64))
    return {**ids, **{name: np.asarray(value) for name, value in fields.items()}}


def fixture():
    events = table([1, 2, 3], step_product_id=[10]*3, ro_product_id=[20]*3, summary_product_id=[30]*3,
                   ewm_product_id=[40]*3, pb_product_id=[50]*3, n_steps=[3, 1, 0], n_readouts=[4, 0, 0],
                   n_summaries=[1, 1, 0], spill=[1]*3, event_length=[300.]*3, pbtime=[0.]*3,
                   mbtime=[100.]*3, marker=[0.]*3, crystal_length=[2.]*3,
                   sim_iov=[[0, 0, 9, 9]]*3, timing_iov=[[0, 0, 9, 9]]*3)
    steps = table([1, 1, 1, 2], product_id=[10]*4, index=[0, 1, 2, 0],
                  sim_product_id=[11]*4, sim_key=[4, 4, 4, 5], crystal=[0]*4, sipm0=[0]*4,
                  time=[20., 5., 120., 50.], energy_g4=[2., 6., 3., 0.], energy_birks=[1., 5., 2., 0.],
                  z=[0., 1., 2., 1.], lru=[1.]*4, yield0=[1.]*4, yield1=[2.]*4)
    readouts = table([1]*4, product_id=[20]*4, index=[0, 1, 2, 3], step_product_id=[10]*4,
                     step_index=[2, 0, 2, 0], sipm=[0, 0, 1, 1])
    readouts["pe_times"] = ak.Array([[20.]*3, [20.], [20.]*6, [20.]])
    summaries = table([1, 2], product_id=[30]*2, index=[0, 0], energy=[3.5, 0.], time=[20., 50.])
    summaries["member_products"] = ak.Array([[10, 10], [10]])
    summaries["member_indices"] = ak.Array([[0, 2], [0]])
    return dict(events=events, steps=steps, readouts=readouts, summaries=summaries)


def check(data, **kwargs):
    return control.compare(**data, options=OPTIONS, expected_events=3, **kwargs)


def test_exact_links_zero_pe_empty_event_and_nonunique_sort_keys():
    report = check(fixture())
    assert report["all_observed_deterministic_fields_exact"]
    assert report["accepted_steps"] == 3
    assert report["saved_steps"] == 4
    assert report["observed_readouts"] == 4
    assert report["observed_photoelectrons"] == 11
    assert report["time_rejected_steps"] == 1
    assert report["accepted_zero_pe_steps"] == 1
    assert report["empty_saved_step_events"] == 1
    assert report["empty_readout_events"] == 2
    assert report["observed_summaries"] == 2
    assert not report["stock_per_step_intermediates_observed"]
    assert not report["unbiased_library_certified"]


@pytest.mark.parametrize("compiled", [False, True])
def test_jax_same_input_discrete_and_observed_fields_for_exact_fixture(compiled):
    from types import SimpleNamespace
    jax = pytest.importorskip("jax")
    from calofastsim import response_jax
    with jax.enable_x64():
        backend = response_jax if not compiled else SimpleNamespace(**{name: jax.jit(getattr(response_jax, name))
            for name in ("fold_step_time", "corrected_energy", "photoelectron_mean", "nonnegative_lround")})
        assert check(fixture(), backend=backend)["all_observed_deterministic_fields_exact"]


@pytest.mark.parametrize("mutation,error", [
    (lambda d: d["events"]["event"].__setitem__(2, 2), "event"),
    (lambda d: d["events"]["step_product_id"].__setitem__(0, 0), "ProductID"),
    (lambda d: d["events"]["n_steps"].__setitem__(0, 2), "reconcile"),
    (lambda d: d["events"]["sim_iov"].__setitem__((0, 0), 8), "interval"),
    (lambda d: d["steps"]["product_id"].__setitem__(0, 100), "registry"),
    (lambda d: d["steps"]["index"].__setitem__(1, 0), "duplicate"),
    (lambda d: d["steps"]["index"].__setitem__(1, 4), "contiguous"),
    (lambda d: d["steps"]["lru"].__setitem__(0, np.nan), "nonfinite"),
    (lambda d: d["events"]["spill"].__setitem__(0, 3), "spill"),
    (lambda d: d["events"]["crystal_length"].__setitem__(0, -2), "length"),
    (lambda d: d["readouts"]["step_index"].__setitem__(1, 2), "duplicate"),
    (lambda d: d["summaries"].__setitem__("member_products", ak.Array([[10], [10]])), "offsets"),
    (lambda d: d["summaries"].__setitem__("member_indices", ak.Array([[0, 1], [0]])), "accepted"),
    (lambda d: d["summaries"].__setitem__("member_indices", ak.Array([[2, 0], [0]])), "order"),
    (lambda d: d["summaries"].__setitem__("member_products", ak.Array([[99, 10], [10]])), "identity"),
    (lambda d: d["steps"]["sim_key"].__setitem__(2, 8), "mixes"),
])
def test_invalid_evidence_refused(mutation, error):
    data = fixture()
    mutation(data)
    with pytest.raises(ValueError, match=error):
        check(data)


@pytest.mark.parametrize("field", ["npe", "pe_times", "summary_energy", "summary_time"])
def test_numeric_failures_remain_failures(field):
    data = fixture()
    if field == "npe":
        data["readouts"]["pe_times"] = ak.Array([[20.]*2, [20.], [20.]*6, [20.]])
    elif field == "pe_times":
        data["readouts"]["pe_times"] = ak.Array([[21.]*3, [20.], [20.]*6, [20.]])
    else:
        data["summaries"]["energy" if field == "summary_energy" else "time"][0] += 1.
    report = check(data)
    assert not report[field]["exact"]
    assert not report["all_observed_deterministic_fields_exact"]


def test_wrong_sipm_is_not_repaired_by_matching_values():
    data = fixture()
    data["readouts"]["sipm"][0] = 9
    report = check(data)
    assert not report["exact_step_sipm_set"]
    assert not report["all_observed_deterministic_fields_exact"]


def test_all_empty_response_keeps_event_manifest():
    data = fixture()
    for name in ("steps", "readouts", "summaries"):
        data[name] = {field: value[:0] for field, value in data[name].items()}
    for field in ("n_steps", "n_readouts", "n_summaries"):
        data["events"][field][:] = 0
    assert check(data)["all_observed_deterministic_fields_exact"]


def config_text():
    return '''controlEWM: {
      module_type: "EventWindowMarkerProducer"
      SpillType: 1
      InitialPhaseShift: 0
      RecoFromMCTruth: true
      RecoFromMCTruthErr: 0
    }
    deterministicRO: {
      module_type: "CaloShowerROMaker"
      PEStatCorrection: false
      addTravelTime: false
      BirksCorrection: true
      LRUCorrection: true
      caloShowerStepCollection: [ "CaloShowerStepMaker::CaloLibraryCE" ]
      eventWindowMarker: "controlEWM::CaloResponseControl"
      protonBunchTimeMC: "controlEWM::CaloResponseControl"
      digitizationStart: 10
      digitizationEnd: 100
      digitizationBuffer: 0
    }'''


def test_expanded_config_requires_original_inputs_and_deterministic_switches(tmp_path):
    path = tmp_path / "control.fcl"
    path.write_text(config_text())
    assert control.configuration(path) == OPTIONS
    for old, new in (("PEStatCorrection: false", "PEStatCorrection: true"),
                     ("addTravelTime: false", "addTravelTime: true"),
                     ("controlEWM::CaloResponseControl", "controlEWM"),
                     ("CaloShowerStepMaker::CaloLibraryCE", "CaloShowerStepMaker"),
                     ("digitizationStart: 10", "digitizationStart: nan")):
        path.write_text(config_text().replace(old, new))
        with pytest.raises(ValueError):
            control.configuration(path)


def test_rejected_steps_do_not_execute_energy_or_rounding():
    data = fixture()
    data["steps"]["energy_birks"][1] = np.finfo(np.float32).max
    data["steps"]["yield0"][1] = np.finfo(np.float32).max
    assert check(data)["all_observed_deterministic_fields_exact"]


@pytest.mark.parametrize("override", ["physics.producers.deterministicRO.PEStatCorrection: true",
                                      "physics.producers.controlEWM.InitialPhaseShift: -0.5",
                                      "physics: @table::Other"])
def test_residual_overrides_or_references_refused(tmp_path, override):
    path = tmp_path / "control.fcl"
    path.write_text(config_text() + "\n" + override)
    with pytest.raises(ValueError, match="canonical expanded"):
        control.configuration(path)


def test_summary_group_collection_order_is_checked():
    data = fixture()
    data["events"]["n_summaries"][0] = 2
    data["steps"]["sim_key"][2] = 5
    data["summaries"] = table([1, 1, 2], product_id=[30]*3, index=[0, 1, 0],
                               energy=[3., .5, 0.], time=[20., 20., 50.])
    data["summaries"]["member_products"] = ak.Array([[10], [10], [10]])
    data["summaries"]["member_indices"] = ak.Array([[2], [0], [0]])
    report = check(data)
    assert report["summary_energy"]["exact"]
    assert not report["summary_group_order_exact"]
    assert not report["all_observed_deterministic_fields_exact"]
    order = np.array([1, 0, 2])
    data["summaries"] = {name: value[order] for name, value in data["summaries"].items()}
    data["summaries"]["index"] = np.array([0, 1, 0])
    assert check(data)["all_observed_deterministic_fields_exact"]


def test_root_tables_and_independent_step_reconciliation(tmp_path):
    import uproot
    data = fixture()
    root_path, reference = tmp_path / "control.root", tmp_path / "original.root"
    with uproot.recreate(root_path) as root:
        for name, values in data.items():
            root.mktree(f"response/{name}", values)
    saved = {**data["steps"], "position_z": data["steps"]["z"].astype(np.float64)}
    with uproot.recreate(reference) as root:
        root.mktree("dump/events", {**data["events"], "g4_status": np.zeros(3, np.int32)})
        root.mktree("dump/showers", saved)
    tables = control.read_tables(root_path)
    control.reconcile(tables, reference, 3)
    assert check(tables)["all_observed_deterministic_fields_exact"]
    tables["steps"]["sim_product_id"][0] = 99
    with pytest.raises(ValueError, match="sim_product_id"):
        control.reconcile(tables, reference, 3)


def test_response_driver_requires_verified_build_and_input_and_never_transports():
    import subprocess
    driver = ROOT / "scripts/response_control.sh"
    text = driver.read_text()
    assert subprocess.run(["bash", "-n", str(driver)], capture_output=True).returncode == 0
    for guard in ('test "$MUSE_WORK_DIR" = "$CALO_AREA"', 'test "$MUSE_ENVSET" = "$CALO_ENVSET"',
                  'test "$plugin" -nt', 'CALO_PLUGIN_SHA256', 'CALO_INPUT_SHA256', 'test "$CALO_EVENTS" -le 2000'):
        assert guard in text
    assert 'mkdir "$CALO_OUT"' in text
    assert 'CaloResponseDumper_module.so' in text
    assert 'exit_code=%s' in text and 'completed_utc=%s' in text
    assert "muse build" not in text and "muse tarball" not in text and "unfiltered_ce_pilot.sh" not in text


def test_new_plugin_has_mirrored_dependencies_and_no_response_reimplementation():
    package = ROOT / "art/CaloFastSim"
    cmake = (package / "CMakeLists.txt").read_text()
    scons = (package / "src/SConscript").read_text()
    assert "cet_build_plugin(CaloResponseDumper" in cmake
    for name in ("CaloConditions", "DAQConditions", "GlobalConstantsService", "ProditionsService", "DbTables"):
        assert f"Offline::{name}" in cmake
        assert f"'mu2e_{name}'" in scons
    source = (package / "src/CaloResponseDumper_module.cc").read_text()
    assert "std::fmod" not in source and "std::lround" not in source
    assert "energyDep()" in source and "caloShowerStep()" in source
    fcl = (package / "fcl/response_control.fcl").read_text()
    assert "g4run" not in fcl and "CaloShowerStepMaker :" not in fcl
    assert "PEStatCorrection : false" in fcl and "addTravelTime : false" in fcl
    assert fcl.isascii()
