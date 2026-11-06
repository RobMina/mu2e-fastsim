"""Diagnostic wiring keeps transport out and both step inputs before response."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "mixing"


def test_one_frame_is_not_claimed_as_stock_multiplicity():
    text = (ROOT / "fcl/one_frame_diagnostic.fcl").read_text()
    assert "one neutral donor" in text
    assert "module_type: ResamplingMixer" in text
    assert "p1: [donor," in text
    assert 'StrawGasStepModules: ["compressDetStepMCs", "donor"]' in text
    assert 'crvStepModuleLabels: ["compressDetStepMCs", "donor"]' in text
    assert "simTimeOffset" not in text
    assert "Mu2eG4" not in text.replace("Offline/Mu2eG4/geom/", "")


def test_no_build_in_drivers():
    for filename in ("run_native_diagnostic.sh", "run_response.sh"):
        text = (ROOT / filename).read_text()
        assert "muse build" not in text and "scons" not in text
        assert 'bash "$OFFLINE_CHAIN_RUNNER"' in text
    native = (ROOT / "run_native_diagnostic.sh").read_text()
    assert 'export OFFLINE_BINDS="${OFFLINE_BINDS:+$OFFLINE_BINDS,}$HERE:$HERE"' in native
    assert "export FHICL_FILE_PATH=%q" in native


def test_saved_digis_are_not_digitized_again():
    text = (ROOT / "fcl/selected_digi_dump.fcl").read_text()
    assert "CrvRecoPackage.CrvRecoSequence" in text
    assert "mc: false" in text
    assert "CRVResponse" not in text
    assert "MixBackgroundFrames" not in text
    assert "CommonMC.DigiSim" not in text
    assert 'StrawDigiTag: "makeSD"' in text
    assert "module_type: EventLister" in text


def test_occupancy_config_changes_only_diagnostics():
    text = (ROOT / "fcl/four_stream_occupancy.fcl").read_text()
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert '#include "four_stream_response.fcl"' in text
    assert lines == ["physics.analyzers.crvdump.mc: false", "services.TimeTracker.printSummary: true"]


def test_four_stream_response_uses_database_rates_and_no_extra_shift():
    text = (ROOT / "fcl/four_stream_response.fcl").read_text()
    assert "module_type: MixBackgroundFrames" in text
    assert "simbookkeeper.useDb: true" in text
    assert "module_type: ProtonBunchIntensityLogNormal" in text
    assert "meanEventsPerProton:" not in text
    assert "SimTimeOffset" not in text
    assert "SelectEvents" not in text
    for stream in ("MuStopPileup", "EleBeamFlash", "MuBeamFlash", "NeutralsFlash"):
        assert f"{stream}Mixer: @local::PU1Mixer" in text
        assert f"{stream}Mixer.mu2e.simStageEfficiencyTags:" in text
    assert text.index("p1: [PBISim") < text.index("@sequence::CommonMC.DigiSim")
