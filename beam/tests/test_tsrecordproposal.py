"""Source-only proposal controls; no native compilation or geometry execution."""
import hashlib
import json
from pathlib import Path

import pytest

from beamfastsim.tsrecordproposal import ProposalError, assemble, verify


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def trees(tmp_path):
    source, proposal = tmp_path / "source", tmp_path / "proposal"
    source.mkdir()
    proposal.mkdir()
    (source / "Mu2eG4/inc").mkdir(parents=True)
    (source / "Mu2eG4/src").mkdir()
    base, changed = b"base\n", b"base\nextra\n"
    (source / "Mu2eG4/inc/Foo.hh").write_bytes(base)
    (source / "Mu2eG4/src/SConscript").write_bytes(b"source discovery\n")
    assets = {"target_recording.patch": b"--- a/Mu2eG4/inc/Foo.hh\n+++ b/Mu2eG4/inc/Foo.hh\n@@ -1 +1,2 @@\n base\n+extra\n",
              "TargetRecording.hh": b"header\n", "TargetRecording.cc": b"implementation\n", "README.md": b"proposal\n"}
    for name, data in assets.items():
        (proposal / name).write_bytes(data)
    pins = {"schema": "target-recording-source-proposal-v1", "offline_revision": "0" * 40,
            "library_eligible": False,
            "source_sha256": {"Mu2eG4/inc/Foo.hh": digest(base),
                              "Mu2eG4/src/SConscript": digest(b"source discovery\n")},
            "patched_sha256": {"Mu2eG4/inc/Foo.hh": digest(changed)},
            "proposal_sha256": {name: digest(data) for name, data in assets.items()}}
    (proposal / "source_pins.json").write_text(json.dumps(pins))
    return source, proposal, tmp_path / "assembled", pins


def test_assemble_new_tree_only(trees):
    source, proposal, dest, _ = trees
    result = assemble(source, proposal, dest)
    assert (dest / "Mu2eG4/inc/Foo.hh").read_bytes() == b"base\nextra\n"
    assert (source / "Mu2eG4/inc/Foo.hh").read_bytes() == b"base\n"
    assert (dest / "Mu2eG4/src/TargetRecording.cc").read_bytes() == b"implementation\n"
    assert result["patch_files_verified"] == 1
    assert result["library_eligible"] is False
    assert not any(result[key] for key in ("build_performed", "native_execution_performed", "simulation_performed"))


@pytest.mark.parametrize("kind", ["source", "asset", "missing", "symlink", "badpatch", "wrongpost"])
def test_refuse_corrupt_or_linked_input(trees, kind):
    source, proposal, dest, pins = trees
    if kind == "source":
        (source / "Mu2eG4/inc/Foo.hh").write_text("other\n")
    elif kind == "asset":
        (proposal / "TargetRecording.hh").write_text("other\n")
    elif kind == "missing":
        (source / "Mu2eG4/inc/Foo.hh").unlink()
    elif kind == "symlink":
        target = source / "Mu2eG4/inc/Foo.hh"
        target.unlink()
        target.symlink_to(proposal / "TargetRecording.hh")
    elif kind == "badpatch":
        patch = b"not a patch\n"
        (proposal / "target_recording.patch").write_bytes(patch)
        pins["proposal_sha256"]["target_recording.patch"] = digest(patch)
    else:
        pins["patched_sha256"]["Mu2eG4/inc/Foo.hh"] = "f" * 64
    (proposal / "source_pins.json").write_text(json.dumps(pins))
    with pytest.raises(ProposalError):
        assemble(source, proposal, dest)
    if dest.exists():
        assert (dest / "ASSEMBLY_FAILED").is_file()


@pytest.mark.parametrize("name", ["../outside", "/outside", "Mu2eG4/../outside", "./outside", "bad\\path"])
def test_refuse_path_escape(trees, name):
    source, proposal, dest, pins = trees
    pins["source_sha256"][name] = "f" * 64
    (proposal / "source_pins.json").write_text(json.dumps(pins))
    with pytest.raises(ProposalError):
        assemble(source, proposal, dest)
    assert not dest.exists()


def test_never_overwrite_destination(trees):
    source, proposal, dest, _ = trees
    dest.mkdir()
    marker = dest / "work"
    marker.write_text("preserve")
    with pytest.raises(ProposalError, match="must not exist"):
        assemble(source, proposal, dest)
    assert marker.read_text() == "preserve"


def test_refuse_nested_destination(trees):
    source, proposal, _, _ = trees
    with pytest.raises(ProposalError, match="overlaps"):
        assemble(source, proposal, source / "new")


def test_real_hook_patch_is_default_off_and_preserves_cut_order():
    root = Path(__file__).resolve().parents[1] / "native"
    patch = (root / "target_recording.patch").read_text()
    added = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = "\n".join(line[1:] for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    assert "OptionalTable<TargetRecording_> TargetRecording" in added
    assert "if (conf_.TargetRecording.hasValue())" in added
    assert 'conf_.inputs().primaryType() != "GenParticles" || conf_.G4InteralFiltering()' in added
    assert "sequential Mu2eG4 only" in added
    assert "targetRecording_->step(*step)" in added
    assert "targetRecording_->cuts(*track, true, false)" in added
    assert "targetRecording_->cuts(*track, false, true)" in added
    assert "killTrack(" not in removed
    assert "return fKill" not in removed
    assert "return fUrgent" not in removed
    assert "src/TargetRecording.cc" in added


def test_real_assets_are_manifest_bound():
    root = Path(__file__).resolve().parents[1] / "native"
    pins = json.loads((root / "source_pins.json").read_text())
    assert pins["library_eligible"] is False
    assert pins["offline_revision"] == "b24d4163e5e8f5ed2b5f72fccb25e4980fec64e2"
    for name, sha in pins["proposal_sha256"].items():
        assert digest((root / name).read_bytes()) == sha
    assert "Mu2eG4/src/SConscript" in pins["source_sha256"]


@pytest.mark.parametrize("name", ["Mu2eG4/inc/TargetRecording.hh", "Mu2eG4/src/TargetRecording.cc",
                                  "PROPOSAL_STATUS.json", "ASSEMBLY_FAILED"])
def test_generated_outputs_cannot_overlay_bound_sources(trees, name):
    source, proposal, dest, pins = trees
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("preserve\n")
    pins["source_sha256"][name] = digest(b"preserve\n")
    (proposal / "source_pins.json").write_text(json.dumps(pins))
    with pytest.raises(ProposalError, match="collides"):
        assemble(source, proposal, dest)
    assert not dest.exists()
    assert path.read_text() == "preserve\n"


@pytest.mark.parametrize("location", ["source", "proposal", "assembled", "existing", "symlink"])
def test_cli_report_cannot_overwrite_protected_trees(trees, tmp_path, location):
    from beamfastsim.tsrecordproposal import main
    source, proposal, dest, _ = trees
    marker = tmp_path / "preserved"
    marker.write_text("preserve\n")
    if location == "source":
        report = source / "newreport"
    elif location == "proposal":
        report = proposal / "newreport"
    elif location == "assembled":
        report = dest / "newreport"
    elif location == "existing":
        report = marker
    else:
        report = tmp_path / "linked"
        report.symlink_to(marker)
    with pytest.raises(ProposalError, match="report"):
        main(["--source-root", str(source), "--proposal-root", str(proposal),
              "--destination", str(dest), "--out", str(report)])
    assert marker.read_text() == "preserve\n"
    assert not dest.exists()


def test_deletion_only_hunk(trees):
    source, proposal, dest, pins = trees
    patch = b"--- a/Mu2eG4/inc/Foo.hh\n+++ b/Mu2eG4/inc/Foo.hh\n@@ -1 +0,0 @@\n-base\n"
    (proposal / "target_recording.patch").write_bytes(patch)
    pins["proposal_sha256"]["target_recording.patch"] = digest(patch)
    pins["patched_sha256"]["Mu2eG4/inc/Foo.hh"] = digest(b"")
    (proposal / "source_pins.json").write_text(json.dumps(pins))
    assemble(source, proposal, dest)
    assert (dest / "Mu2eG4/inc/Foo.hh").read_bytes() == b""
