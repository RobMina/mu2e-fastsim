"""Frozen-area checks for retained-step replay preparation."""
import pytest

from valhist import tier1b_replay as R


def test_frozen_area_requires_binaries_and_exact_hashes(tmp_path):
    binary = tmp_path / "module.so"
    binary.write_bytes(b"built")
    manifest = {"files": {str(binary): R.digest(binary), "/unrelated/file": "untouched"}}
    assert R.frozen_files(manifest, tmp_path) == {str(binary): R.digest(binary)}
    binary.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        R.frozen_files(manifest, tmp_path)
    with pytest.raises(ValueError, match="no recorded binaries"):
        R.frozen_files({"files": {}}, tmp_path)


def test_dqm_paths_keep_original_shard_and_opposite_side_baseline(tmp_path):
    from valhist.tier1b_dqm_compare import paths
    hist, tables, g4, old = paths(tmp_path / "data", tmp_path / "new", 11)
    assert hist["old_e2e"].parts[-3:] == ("valhist", "e2e_on", "valhist.001400_00000011.root")
    assert hist["e2e"] == tmp_path / "new/e2e/valhist.root"
    assert tables["e2e"] == hist["e2e"]
    assert g4.name == "g4all.001400_00000011.root"
    assert old.name == "fast1b.001400_00000011.root"


def test_replay_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        R.prepare("missing", tmp_path, "missing", "missing", "missing", "missing", 11)
