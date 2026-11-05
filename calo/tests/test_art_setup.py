"""Area preparation cannot start a build or overwrite an existing backing choice."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "art" / "setup_area.sh"


def run(tmp_path, **overrides):
    area = tmp_path / "art"
    (area / "CaloFastSim" / "src").mkdir(parents=True, exist_ok=True)
    (area / "CaloFastSim" / "src" / "SConscript").touch()
    shutil.copyfile(SCRIPT, area / "setup_area.sh")
    backing = tmp_path / "release"
    (backing / "build" / "al9-prof-e29-p001").mkdir(parents=True, exist_ok=True)
    (backing / "Offline").mkdir(exist_ok=True)
    (backing / "Offline" / ".muse").write_text("ENVSET p001\n")
    env = {**os.environ, "CALO_BACKING": str(backing), "CALO_ENVSET": "p001", **overrides}
    return subprocess.run(["bash", str(area / "setup_area.sh")], env=env, text=True, capture_output=True)


def test_prepare_idempotent_without_setup_or_build(tmp_path):
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    area = tmp_path / "art"
    assert (area / "backing").readlink() == tmp_path / "release"
    assert (area / ".muse").read_text() == "ENVSET p001\n"
    assert "muse build -j 32" in result.stdout
    assert not (area / "build").exists()
    assert run(tmp_path).returncode == 0


@pytest.mark.parametrize("overrides", [{"CALO_ENVSET": ""}, {"CALO_ENVSET": "bad"},
                                      {"CALO_BACKING": "relative"}])
def test_invalid_setup_refused(tmp_path, overrides):
    assert run(tmp_path, **overrides).returncode != 0


@pytest.mark.parametrize("build_envset,metadata", [("p002", "ENVSET p001\n"),
                                                  ("p001", "ENVSET p002\n"),
                                                  ("p001", "# no preferred environment\n")])
def test_backing_envset_must_match_built_stub_and_metadata(tmp_path, build_envset, metadata):
    backing = tmp_path / "different_release"
    (backing / "build" / f"al9-prof-e29-{build_envset}").mkdir(parents=True)
    (backing / "Offline").mkdir()
    (backing / "Offline" / ".muse").write_text(metadata)
    assert run(tmp_path, CALO_BACKING=str(backing)).returncode != 0
    assert not (tmp_path / "art" / "backing").exists()


def test_art_shared_resource_dependency_is_explicit_in_both_build_systems():
    package = SCRIPT.parent / "CaloFastSim"
    assert "'art_Utilities'" in (package / "src" / "SConscript").read_text()
    assert "art::Utilities" in (package / "CMakeLists.txt").read_text()


def test_existing_choices_not_overwritten(tmp_path):
    assert run(tmp_path).returncode == 0
    assert run(tmp_path, CALO_ENVSET="p002").returncode != 0
    other = tmp_path / "other"
    (other / "build" / "al9-prof-e29-p001").mkdir(parents=True)
    (other / "Offline").mkdir()
    (other / "Offline" / ".muse").write_text("ENVSET p001\n")
    assert run(tmp_path, CALO_BACKING=str(other)).returncode != 0
    assert (tmp_path / "art" / "backing").readlink() == tmp_path / "release"
