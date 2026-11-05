"""Guard compact-header preparation without touching a real source tree or building."""
from pathlib import Path
import importlib.util

import pytest

HERE = Path(__file__).resolve().parents[1] / "recording"
spec = importlib.util.spec_from_file_location("compact_preparer", HERE / "current_head/fix_inventory_compact.py")
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


def test_current_compact_payload_retains_caps_and_observational_contract():
    header = (HERE / "inventory/payload/Mu2eG4PlacementInventory.hh").read_bytes()
    preparer.check_header(header)
    for old in (b"static constexpr Count byteLimit = 1073741824;",
                b"inventory.serialize(g4Run, -1)", b"std::pmr::vector<Placement>"):
        with pytest.raises(ValueError, match="contract absent"):
            preparer.check_header(header.replace(old, b"removed", 1))
    with pytest.raises(ValueError, match="nonobservational"):
        preparer.check_header(header + b"\nG4UniformRand\n")


def test_compact_preparation_rejects_drift_before_writes(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    source, overlay = tmp_path / "source", tmp_path / "overlay"
    for path in (artifacts, source, overlay):
        path.mkdir()
    (artifacts / "source-manifest.json").write_text("unreviewed")
    (overlay / "overlay-receipt.json").write_text("unreviewed")
    monkeypatch.setattr(preparer, "HERE", artifacts)
    with pytest.raises(ValueError, match="baseline drift"):
        preparer.apply(source, overlay)
    assert list(source.iterdir()) == []
    assert sorted(p.name for p in overlay.iterdir()) == ["overlay-receipt.json"]


def test_compact_preparer_has_no_setup_build_transport_or_git_writes():
    text = (HERE / "current_head/fix_inventory_compact.py").read_text()
    for forbidden in ("muse build", "muse setup", "muse tarball", "subprocess", "cmake --build"):
        assert forbidden not in text
    assert '"pre-compact-inventory-header.hh"' in text
    assert '"user_rebuild_pending": True' in text
    assert '"xb"' in text
