"""Launch environments are derived solely from the verified separate full-stock plan."""
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import response_launch as launch


def fixture(tmp_path, monkeypatch):
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text('{}')
    plan = {'runtime': {'OFFLINE_IMAGE': '/cvmfs/pinned', 'CVMFSEXEC_DIST': '/private/cvmfsexec',
                       'TRIGGER_AREA': '/cvmfs/released'},
            'runtime_binding': {'runner': {'path': '/private/completed F/runtime/offline_plugins.sh'}},
            'config': {'path': '/private/original.config'}}
    row = {'events': 2811, 'prepared': '/private/control parent'}
    monkeypatch.setattr(launch.control, 'preflight', lambda p, o: (plan, row))
    return plan_file, plan, row


def test_bound_launch_includes_exact_runner_and_reset(tmp_path, monkeypatch):
    file, plan, row = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'new-output'
    output.mkdir()
    binding = launch.prepare(file, 1, output)
    assert binding['parent'] == row and binding['runtime'] == plan['runtime']
    assert json.loads((output / 'execution-binding.json').read_text()) == binding
    selection = json.loads((output / 'runtime-selection.json').read_text())
    assert selection == json.loads((output / 'roundtrip/runtime-selection.json').read_text())
    assert selection['runner'].endswith('runtime/offline_plugins.sh')
    text = (output / 'launch.env').read_text()
    assert "CONTROL_RUNNER='/private/completed F/runtime/offline_plugins.sh'" in text
    assert 'OFFLINE_EVENTS=2811' in text
    assert 'OFFLINE_IMAGE=/cvmfs/pinned' in text
    assert "CONTROL_PREPARED='/private/control parent'" in text
    assert 'muse' not in text
    with pytest.raises(ValueError, match='fresh'):
        launch.prepare(file, 1, output)


def test_failed_gate_produces_no_launch(tmp_path, monkeypatch):
    file, _, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'out'
    output.mkdir()
    def bad(*args):
        raise ValueError('changed source')
    monkeypatch.setattr(launch.control, 'preflight', bad)
    with pytest.raises(ValueError, match='changed source'):
        launch.prepare(file, 1, output)
    assert not list(output.iterdir())
