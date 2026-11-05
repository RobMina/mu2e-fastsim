"""Exact committed checkpoint verification using disposable Git repositories only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

HERE = Path(__file__).resolve().parents[1] / "recording/current_head"
spec = importlib.util.spec_from_file_location("checkpoint_verifier", HERE / "verify_area.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    (root / "existing.txt").write_text("base\n")
    (root / "untouched.txt").write_text("leave alone\n")
    git(root, "add", "existing.txt", "untouched.txt")
    git(root, "commit", "-qm", "base")
    base = git(root, "rev-parse", "HEAD")
    (root / "existing.txt").write_text("prepared\n")
    (root / "added.txt").write_text("new prepared file\n")
    pins = dict(base_revision=base, patched_sources={
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("existing.txt", "added.txt")})
    artifacts = tmp_path / "pins"
    artifacts.mkdir()
    (artifacts / "source-manifest.json").write_text(json.dumps(pins))
    monkeypatch.setattr(verifier, "HERE", artifacts)
    return root, base


def commit(root):
    git(root, "add", "existing.txt", "added.txt")
    git(root, "commit", "-qm", "checkpoint")


def test_original_unstaged_preparation_and_exact_commit_both_pass(source):
    root, _ = source
    verifier.verify(root)
    commit(root)
    verifier.verify(root)


def test_staged_preparation_is_not_a_verified_state(source):
    root, _ = source
    git(root, "add", "existing.txt")
    with pytest.raises(ValueError, match="staged changes"):
        verifier.verify(root)


def test_unrelated_committed_path_is_not_a_checkpoint(source):
    root, _ = source
    (root / "untouched.txt").write_text("unexpected\n")
    git(root, "add", "untouched.txt")
    commit(root)
    with pytest.raises(ValueError, match="exact prepared changeset"):
        verifier.verify(root)


def test_checkpoint_content_not_just_paths_must_match(source):
    root, _ = source
    (root / "existing.txt").write_text("wrong checkpoint\n")
    commit(root)
    (root / "existing.txt").write_text("prepared\n")
    with pytest.raises(ValueError, match="checkpoint source changed"):
        verifier.verify(root)


def test_checkpoint_worktree_must_stay_clean(source):
    root, _ = source
    commit(root)
    (root / "untouched.txt").write_text("unexpected\n")
    with pytest.raises(ValueError, match="working-tree changes"):
        verifier.verify(root)


def test_checkpoint_rejects_untracked_files(source):
    root, _ = source
    commit(root)
    (root / "unexpected.txt").write_text("unexpected\n")
    with pytest.raises(ValueError, match="unexpected untracked"):
        verifier.verify(root)


def test_checkpoint_base_must_be_ancestor(source):
    root, _ = source
    commit(root)
    git(root, "switch", "--orphan", "unrelated")
    (root / "existing.txt").write_text("prepared\n")
    (root / "added.txt").write_text("new prepared file\n")
    commit(root)
    with pytest.raises(subprocess.CalledProcessError):
        verifier.verify(root)


def test_checkpoint_preserves_source_manifest_bytes(source):
    root, _ = source
    pins = verifier.HERE / "source-manifest.json"
    before = pins.read_bytes()
    commit(root)
    verifier.verify(root)
    assert pins.read_bytes() == before


def test_checkpoint_rejects_committed_mode_change(source):
    root, _ = source
    git(root, "add", "existing.txt", "added.txt")
    git(root, "update-index", "--chmod=+x", "added.txt")
    git(root, "commit", "-qm", "mode change")
    with pytest.raises(ValueError, match="type/mode changed"):
        verifier.verify(root)
