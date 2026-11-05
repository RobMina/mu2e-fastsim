"""Bounded fixture outcome gates, not native execution or crash-safety evidence."""
from pathlib import Path
import importlib.util
import sys

import pytest

HERE = Path(__file__).resolve().parents[1] / "recording"
spec = importlib.util.spec_from_file_location("recording_fixture_auditor", HERE / "audit_fixture.py")
auditor = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(HERE))
try:
    spec.loader.exec_module(auditor)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("mode", ("single", "strict", "empty"))
def test_success_requires_zero_exit(mode):
    assert auditor.outcome(mode, 0, "")
    with pytest.raises(ValueError):
        auditor.outcome(mode, 1, "")


@pytest.mark.parametrize("mode,marker", (
    ("strict_empty", "MANIFEST_MULTIPLICITY"),
    ("missing", "MANIFEST_MISSING"),
    ("generator_error", "BADINPUT"),
))
def test_failure_requires_exact_callback_exception(mode, marker):
    assert not auditor.outcome(mode, 1, "---- " + marker + " BEGIN")
    for code, log in ((0, "---- " + marker + " BEGIN"), (9, "Configuration"), (1, "other failure")):
        with pytest.raises(ValueError):
            auditor.outcome(mode, code, log)


def test_audit_separates_failure_prefix_from_completion_and_physics():
    text = (HERE / "audit_fixture.py").read_text()
    assert '"complete_structural_manifest": complete' in text
    assert '"failure_prefix_only": not complete' in text
    assert '"generator_state_independently_replayed": False' in text
    assert '"physics_validated": False' in text
    assert '"library_eligible": False' in text
    assert 'source_log.read_text()' in text
    assert 'validator.audit(' in text
    assert 'subprocess' not in text
