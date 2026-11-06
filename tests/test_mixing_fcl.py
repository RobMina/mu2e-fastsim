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


def test_tracker_crv_preparation_bypasses_only_universal_filters():
    text = (ROOT / "fcl/four_stream_tracker_crv_inputs.fcl").read_text()
    parent = (ROOT / "fcl/four_stream_response.fcl").read_text()
    assert '#include "four_stream_fast_inputs.fcl"' in text
    streams = ("MuStopPileup", "EleBeamFlash", "MuBeamFlash", "NeutralsFlash")
    for stream, directory in zip(streams, ("stopSteps", "eleSteps", "muSteps", "neutralSteps")):
        assert f"{stream}Mixer.mu2e.products.caloShowerStepMixer.mixingMap: []" in text
        assert f"physics.analyzers.{directory}.strawGasSteps: {stream}Mixer" in text
        assert f"{stream}Filter: {{ module_type: MixingFilter MixingModule: {stream}Mixer }}" in parent
    scheduled = text.split("physics.p1:", 1)[1]
    assert "Filter" not in scheduled and "makeSD" not in scheduled
    assert "[PBISim, " + ", ".join(s + "Mixer" for s in streams) in scheduled
    assert "EWMProducer, timingTruth" in scheduled
    assert "CrvDAQPackage.CrvDAQSequence" in scheduled and "CrvRecoPackage.CrvRecoSequence" in scheduled
    assert "SeedService" not in text and "MaxEventsToSkip" not in text
    assert "genParticles:" not in text and "simParticles:" not in text
    assert "crvStepModuleLabels: [compressDetStepMCs, " + ", ".join(s + "Mixer" for s in streams) + "]" in text


def test_preparation_runner_uses_saved_reference_and_new_output():
    text = (ROOT / "run_preparation_uva.sh").read_text()
    assert 'mkdir "$OUT"' in text and 'mkdir -p "$OUT"' not in text
    assert "MIXING_EVENTS=10" in text and "MIXING_EVENTS=-1" in text
    assert '"$REF/response.root" "$OUT/full/response.root"' in text
    assert "compare_root.py" in text and "--expected-events" in text
    assert '$REF:$REF,$OUT:$OUT' in text and text.count('apptainer exec --cleanenv -B "$BINDS"') == 2
    assert "muse build" not in text and "scons" not in text


def test_four_stream_response_uses_database_rates_and_no_extra_shift():
    text = (ROOT / "fcl/four_stream_response.fcl").read_text()
    assert "module_type: MixBackgroundFrames" in text
    assert "simbookkeeper.useDb: true" in text
    assert "module_type: ProtonBunchIntensityLogNormal" in text
    assert "meanEventsPerProton:" not in text
    assert "SimTimeOffset" not in text
    assert "SelectEvents" not in text
    for stream in ("MuStopPileup", "EleBeamFlash", "MuBeamFlash", "NeutralsFlash"):
        assert f"{stream}Mixer: @local::FourStreamMixer" in text
        assert f"{stream}Mixer.mu2e.simStageEfficiencyTags:" in text
    assert text.index("p1: [PBISim") < text.index("@sequence::CommonMC.DigiSim")
