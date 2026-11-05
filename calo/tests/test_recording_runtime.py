"""No-transport harness source contracts; no native framework execution in these tests."""
from pathlib import Path
import importlib.util
import json

import pytest

HERE = Path(__file__).resolve().parents[1] / "recording"
spec = importlib.util.spec_from_file_location("recording_runtime_checks", HERE / "runtime_checks.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_geometry_has_exact_rejecting_path_and_no_macros():
    text = (HERE / "zero_transport.fcl").read_text()
    assert 'init : [ rejectTransport, g4run ]' in text
    assert 'ReturnValue : false' in text
    assert 'trigger_paths : [ init ]' in text
    assert 'end_paths : []' in text
    assert 'module_type : EmptyEvent maxEvents : 1' in text
    assert 'g4Macro : ""' in text
    assert 'visualization : { initMacro : "" GUIMacro : "" }' in text
    assert 'CalIFBCablePanelPOut1' in text
    assert 'producers.g4run :' not in text


def test_fixture_uses_filters_on_single_ordered_trigger_path():
    text = (HERE / "manifest_fixture.fcl").read_text()
    assert 'filters : {' in text
    assert 'analyzers : {' not in text
    assert 'fixture : [ inputEventManifest, generate, incidentManifest ]' in text
    assert 'trigger_paths : [ fixture ]' in text
    assert 'end_paths : []' in text
    assert text.count('REVIEW_NORMALIZED_SHA256') == 2
    assert '@local::physics.' not in text
    assert 'Mu2eG4' not in text


def test_runtime_guard_rejects_other_area(tmp_path, monkeypatch):
    receipt = tmp_path / 'build.json'
    receipt.write_text(json.dumps({'work_dir': str(tmp_path)}))
    monkeypatch.setenv('MUSE_WORK_DIR', str(tmp_path / 'wrong'))
    monkeypatch.setenv('MUSE_ENVSET', 'p110')
    with pytest.raises(ValueError, match='wrong built'):
        runtime.check_environment(tmp_path, receipt)


def test_runtime_rejects_receipt_from_before_source_correction(tmp_path, monkeypatch):
    receipt = tmp_path / 'build.json'
    receipt.write_text(json.dumps({'work_dir': str(tmp_path), 'source_manifest_sha256': '0' * 64}))
    monkeypatch.setenv('MUSE_WORK_DIR', str(tmp_path))
    monkeypatch.setenv('MUSE_ENVSET', 'p110')
    with pytest.raises(ValueError, match='source manifest changed'):
        runtime.check_environment(tmp_path, receipt)


def test_missing_fixture_requests_prior_process_not_absent_current_producer():
    text = (HERE / "runtime_checks.py").read_text()
    assert '"missing::MuBeamResampler"' in text
    assert '"missing::CaloManifestFixture"' not in text


def test_runtime_checks_never_promote_to_physics_or_build():
    text = (HERE / "runtime_checks.py").read_text()
    assert '"transport_requested": False' in text
    assert 'record["native_geometry_certified"] = False' in text
    assert 'record["physics_validated"] = False' in text
    for forbidden in ('muse build', 'muse setup', 'muse tarball', 'cmake --build', 'LD_PRELOAD'):
        assert forbidden not in text
