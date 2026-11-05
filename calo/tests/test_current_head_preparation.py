"""Current-head preparation guards; never prepare a real Offline checkout in tests."""
from pathlib import Path
import importlib.util
import json

import pytest

HERE = Path(__file__).resolve().parents[1] / "recording"
spec = importlib.util.spec_from_file_location("head_preparer", HERE / "prepare_current_head.py")
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


def test_no_build_setup_or_tarball_in_preparer():
    text = (HERE / "prepare_current_head.py").read_text()
    for forbidden in ("muse build", "muse setup", "muse tarball", "scons", "cmake --build"):
        assert forbidden not in text
    assert '"--dry-run"' in text
    assert '"--fuzz=0"' in text


def test_reject_existing_artifact_destination_before_git(tmp_path):
    with pytest.raises(ValueError, match="must be new"):
        preparer.prepare(tmp_path / "absent", tmp_path)


def test_source_hash_drift_rejected(tmp_path):
    (tmp_path / "a").write_text("a")
    preparer.require_sources(tmp_path, {"a": preparer.sha(tmp_path / "a")})
    with pytest.raises(ValueError, match="unreviewed"):
        preparer.require_sources(tmp_path, {"a": "b" * 64})


def test_current_head_receipt_identifies_p110_and_no_build():
    manifest = json.loads((HERE / "current_head/source-manifest.json").read_text())
    assert manifest["base_revision"] == preparer.HEAD
    assert manifest["environment_recommendation"] == "p110"
    assert manifest["build_performed"] is False
    for name, expected in manifest["patches"].items():
        assert preparer.sha(HERE / "current_head" / name) == expected
    assert preparer.sha(HERE / "current_head/lifecycle-refinement.patch") == manifest["lifecycle_refinement"]["patch_sha256"]
    inventory = json.loads((HERE / "inventory/source-manifest.json").read_text())
    for name, sha in inventory["patched"].items():
        assert manifest["patched_sources"][name] == sha
    assert manifest["manifest_review_pins_sha256"] == preparer.sha(HERE / "manifest/source_manifest.json")
    assert manifest["base_sources"]["Mu2eG4/inc/Mu2eG4Config.hh"] == preparer.REVIEWED_DRIFT["Mu2eG4/inc/Mu2eG4Config.hh"]
