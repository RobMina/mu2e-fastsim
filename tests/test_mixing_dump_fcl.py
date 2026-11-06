"""Existing frame dumpers preserve the all-event census without builds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "mixing"


def test_legacy_dump_has_all_event_lister():
    text = (ROOT / "fcl/dump_frames_legacy.fcl").read_text()
    assert "module_type: EventLister" in text
    assert "physics.e1: [tracker, crv, events]" in text


def test_dump_driver_uses_existing_runtime():
    text = (ROOT / "run_dump.sh").read_text()
    assert "muse build" not in text and "scons" not in text
    assert 'mu2e -c mixing/fcl/dump_frames.fcl -s "$input" -n -1 -T "$name.root"' in text
    for extra in ("module load", "config_dumper", "product_sizes_dumper", "OFFLINE_TIME", "legacy", "MIXING_EVENTS"):
        assert extra not in text
