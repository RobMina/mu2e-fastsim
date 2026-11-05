"""Backing-overlay guards and source-link discovery without a build invocation."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import pytest

HERE = Path(__file__).resolve().parents[1] / "recording/current_head"
spec = importlib.util.spec_from_file_location("recording_overlay", HERE / "prepare_overlay.py")
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / "sources/Offline"
    backing = tmp_path / "ci"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source.mkdir(parents=True)
    (source / ".muse").write_text("ENVSET p110\n")
    (backing / "Offline/.git").mkdir(parents=True)
    (backing / "Offline/.git/HEAD").write_text("a" * 40 + "\n")
    (backing / "Offline/.muse").write_bytes((source / ".muse").read_bytes())
    for package in overlay.PACKAGES:
        path = source / package / "src/SConscript"
        path.parent.mkdir(parents=True)
        path.write_text("ordinary fixture\n")
        target = backing / "Offline" / path.relative_to(source)
        target.parent.mkdir(parents=True)
        target.write_bytes(path.read_bytes())
    override = source / "Mu2eG4/src/instrumented.cc"
    override.write_text("instrumented fixture\n")
    pins = {
        "base_revision": "a" * 40,
        "base_sources": {".muse": overlay.sha(source / ".muse")},
        "patched_sources": {str(override.relative_to(source)): overlay.sha(override)},
    }
    (artifacts / "source-manifest.json").write_text(json.dumps(pins))
    build = backing / "build" / overlay.STUB
    (build / "Offline/lib").mkdir(parents=True)
    (build / ".musebuild").write_text("11/04/26 10:55:26 to 10:56:14\n")
    for name in overlay.REQUIRED_LIBRARIES:
        (build / "Offline/lib" / name).write_text("fixture only\n")
    (build / "Offline/bin").mkdir()
    (build / "Offline/bin/mu2e").write_text("fixture only\n")
    monkeypatch.setattr(overlay, "HERE", artifacts)
    monkeypatch.setattr(overlay, "load_verifier", lambda: SimpleNamespace(verify=lambda root: None))
    return source, backing, tmp_path / "overlay"


def test_exact_backing_and_three_package_real_directory_discovery(fixture):
    source, backing, destination = fixture
    receipt = overlay.prepare(source, backing, destination)
    assert receipt["build_performed"] is False
    assert receipt["source_modified"] is False
    assert overlay.discover(destination / "Offline") == overlay.expected_scripts()
    assert (destination / "Offline/Mu2eG4/src/instrumented.cc").is_symlink()
    assert not (destination / "Offline/Mu2eG4").is_symlink()
    assert overlay.verify(destination) == receipt


def test_reject_existing_destination_without_touching_it(tmp_path):
    with pytest.raises(ValueError, match="must be new"):
        overlay.prepare(tmp_path / "absent", tmp_path / "absent", tmp_path)


def test_reject_wrong_backing_revision_before_creation(fixture):
    source, backing, destination = fixture
    (backing / "Offline/.git/HEAD").write_text("b" * 40 + "\n")
    with pytest.raises(ValueError, match="exact prepared"):
        overlay.prepare(source, backing, destination)
    assert not destination.exists()


def test_reject_non_override_source_drift(fixture):
    source, backing, destination = fixture
    (source / "CaloMC/src/SConscript").write_text("unexpected drift\n")
    with pytest.raises(ValueError, match="differs from backing"):
        overlay.prepare(source, backing, destination)
    assert not destination.exists()


def test_reject_source_link_retargeting(fixture):
    source, backing, destination = fixture
    overlay.prepare(source, backing, destination)
    link = destination / "Offline/Mu2eG4/src/instrumented.cc"
    link.unlink()
    link.symlink_to(source / "CaloMC/src/SConscript")
    with pytest.raises(ValueError, match="source link changed"):
        overlay.verify(destination)


def test_overlay_helper_never_invokes_build_or_setup():
    text = (HERE / "prepare_overlay.py").read_text()
    for forbidden in ("muse build", "muse setup", "muse tarball", "cmake --build", "subprocess.run"):
        assert forbidden not in text
