"""Source-only exporter packaging checks, not C++ compilation or art validation."""
from pathlib import Path
import re
import subprocess

import pytest


ART = Path(__file__).parents[1] / "art"
PACKAGE = ART / "FlashReference"


def test_isolated_satellite_no_transport_or_build_script():
    setup = (ART / "prepare_area.sh").read_text()
    assert "muse " not in setup and "scons" not in setup
    assert "ln -s" in setup and "Existing backing differs" in setup
    source = (PACKAGE / "src/FlashReferenceExporter_module.cc").read_text()
    assert "public art::EDAnalyzer" in source
    for forbidden in ("event.put(", "RandFlat", "isPrimary()", "originParticle()"):
        assert forbidden not in source
    assert not re.search(r'\bG4(?:Step|Track)\b', source)
    assert "expectedEvents_ == 0" in source and "count_ != expectedEvents_" in source
    assert 'checkedFill(events_); ++count_' in source
    assert 'catch (...) { state = 3; }' in source
    assert 'ptr.isNull()' in source and 'ptr.get()' in source
    assert 'product = static_cast<U64>(ptr.id().value())' in source
    assert 'key = static_cast<U64>(ptr.key())' in source
    assert 'nKilledByFieldPropagator()' in source
    assert 'p->producedClassName()' in source
    assert 'getMany<SimParticleCollection>()' in source
    assert 'getMany<GenParticleCollection>()' in source
    assert 'sr.getValidHandle<PhysicalVolumeInfoMultiCollection>' in source
    for table in ('"requirements"', '"storage"', '"schema"', '"completion"'):
        assert table in source


def test_source_fields_cover_persistent_accessors():
    source = (PACKAGE / "src/FlashReferenceExporter_module.cc").read_text()
    for accessor in ("postPosition", "postMomentum", "totalEDep", "nonIonizingEDep", "visibleEDep",
                     "properTime", "endProcessCode", "simStage", "startPosXYZ", "endPosXYZ", "startMomXYZT",
                     "endMomXYZT", "startGlobalTime", "endGlobalTime", "startProperTime", "endProperTime",
                     "startVolumeIndex", "endVolumeIndex", "startG4Status", "endG4Status", "daughters",
                     "preLastStepKineticEnergy", "endKineticEnergy", "nSteps", "trackLength",
                     "creationCode", "stoppingCode", "endDefined", "generatorId", "copyNo", "materialName"):
        assert re.search(r'\b' + accessor + r'\(', source), accessor
    assert 'p.ion().excitationEnergy' in source and 'p.ion().floatLevelBaseIndex' in source
    assert 'Float_t' in source and 'Double_t' in source and 'ULong64_t' in source


def test_configs_are_ascii_fail_closed_and_reader_only():
    for path in (PACKAGE / "fcl").glob("*.fcl"):
        text = path.read_text(encoding="ascii")
        assert 'Mu2eG4' not in text and 'SelectEvents' not in text
    raw = (PACKAGE / "fcl/read_raw.fcl").read_text()
    assert 'RootInput maxEvents: 0' in raw
    assert 'producers' not in raw and 'RootOutput' not in raw
    fixture = (PACKAGE / "fcl/fixture.fcl").read_text()
    assert 'numberEventsInSubRun: 2' in fixture
    assert 'physics.analyzers.export.expectedEvents: 4' in fixture
    assert 'outputs.saved' in fixture and 'physics.out: [saved]' in fixture
    assert 'fileProperties: { granularity: "Job" }' in fixture
    assert 'flash_reference_fixture_%#.art' in fixture
    assert 'Services.Core' not in fixture and 'standardServices.fcl' not in fixture
    contract = (PACKAGE / "fcl/export.fcl").read_text()
    for tag in ("BeamFilter:Beam", "g4run:r5PhotonSteps", "g4run:tracker", "g4run:calorimeter", "g4run:CRV", "g4run:virtualdetector"):
        assert tag in contract
    assert 'expectedEvents: 0' in contract


def test_dependency_list_is_direct_and_setup_has_no_environment_side_effect(tmp_path):
    deps = (PACKAGE / "src/SConscript").read_text()
    for lib in ("mu2e_MCDataProducts", "art_Framework_Core", "art_Framework_Principal", "canvas",
                "fhiclcpp_types", "art_root_io_TFileService", "cetlib_except", "CLHEP"):
        assert repr(lib) in deps
    subprocess.run(["bash", "-n", str(ART / "prepare_area.sh")], check=True)


@pytest.mark.parametrize("existing", [False, True])
def test_prepare_area_never_replaces_existing_backing(tmp_path, existing):
    area = tmp_path / "art"
    area.mkdir()
    script = area / "prepare_area.sh"
    script.write_bytes((ART / "prepare_area.sh").read_bytes())
    backing = tmp_path / "runtime"
    (backing / "Offline").mkdir(parents=True)
    (backing / "build").mkdir()
    if existing:
        (area / "backing").symlink_to(backing)
    result = subprocess.run(["bash", str(script), str(backing)], capture_output=True, text=True)
    assert result.returncode == 0 and (area / "backing").resolve() == backing
    other = tmp_path / "other"
    (other / "Offline").mkdir(parents=True)
    (other / "build").mkdir()
    result = subprocess.run(["bash", str(script), str(other)], capture_output=True, text=True)
    assert result.returncode != 0 and (area / "backing").resolve() == backing


def test_confined_fixture_is_fixed_reader_only_no_transport_or_build():
    script = ART / "run_confined_fixture.sh"
    text = script.read_text()
    assert "muse build" not in text and "tarball" not in text and "scons" not in text
    assert "Mu2eG4" not in text and "flash_reference_raw" not in text
    assert "muse setup" in text and '[[ $(readlink -f "$MUSE_WORK_DIR") == "$A" ]]' in text
    assert "flash_reference_apptainer_probe.py" in text
    assert text.index("flash_reference_apptainer_probe.py") < text.index("setupmu2e-art.sh")
    assert text.count("mu2e -c ") == 2
    assert "FlashReference/fcl/fixture.fcl" in text
    assert "FlashReference/fcl/read_fixture.fcl" in text
    assert "/mnt/work/fixture/flash_reference_fixture_1.art" in text
    assert "export.root" in text
    subprocess.run(["bash", "-n", str(script)], check=True)
