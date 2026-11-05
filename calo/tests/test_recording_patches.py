"""Review-artifact integrity tests; not C++ compilation or native recording."""
from pathlib import Path
import hashlib
import importlib.util
import json


RECORDING = Path(__file__).resolve().parents[1] / "recording"
COMP = RECORDING / "compression"
BOUNDARY = RECORDING / "boundary"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MembershipPatchTests = _module("compression_patch_tests", COMP / "test_membership_patch.py").MembershipPatchTests


def test_boundary_patch_exact_artifact_hash_and_payload():
    manifest = json.loads((BOUNDARY / "source-manifest.json").read_text())
    patch = (BOUNDARY / "boundary-recorder.patch").read_bytes()
    assert hashlib.sha256(patch).hexdigest() == manifest["patch_sha256"]
    name = "Mu2eG4/inc/Mu2eG4BoundaryRecorder.hh"
    payload = (BOUNDARY / "payload/Mu2eG4BoundaryRecorder.hh").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["patched"][name]
    assert all("+" + line in patch.decode().splitlines() for line in payload.decode().splitlines())


def test_boundary_patch_default_off_no_physics_mutation_in_added_lines():
    patch = (BOUNDARY / "boundary-recorder.patch").read_text().splitlines()
    removed = [line for line in patch if line.startswith("-") and not line.startswith("---")]
    assert removed == ["-    _spHelper()"]
    added = "\n".join(line[1:] for line in patch if line.startswith("+") and not line.startswith("+++"))
    assert 'boundaryRecorderPrefix {Name("boundaryRecorderPrefix"), ""}' in added
    assert 'if (prefix.empty()) return;' in added
    for forbidden in ("SetTrackStatus", "SetMomentum", "SetPosition", "findAndCount", "G4UniformRand",
                      "GetNavigator", "LocateGlobalPoint", "TFileService", "produces<"):
        assert forbidden not in added
    assert "/*" not in added


def test_boundary_patch_has_event_lifecycle_and_checked_output():
    helper = (BOUNDARY / "payload/Mu2eG4BoundaryRecorder.hh").read_text()
    for token in ("event_begin", "event_end", "born_inside", "terminal_inside", "tracking_resume",
                  "birth_membership_unresolved", "world_null", "particlePtr(&t)", "identity(saved->parent())",
                  "pre.GetTouchable()", "post.GetTouchable()", "max_digits10", "O_EXCL", "::mkdir",
                  "out_.flush()", "rows_ >= maxRows_", "maxBytes_ - bytes_"):
        assert token in helper
