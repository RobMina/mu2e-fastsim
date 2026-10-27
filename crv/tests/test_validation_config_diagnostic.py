"""Mocked validation wiring, frozen inputs, and diagnostic worktree receipts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from crvfastsim import response_config as RC
from crvfastsim.objective import validation as V


@pytest.fixture
def run_case(tmp_path, monkeypatch):
    from crvfastsim.reco import r3
    from crvfastsim import kit_task
    files = {}
    for name in ('input', 'excluded', 'calib', 'deviation', 'reference', 'background', 'time_shift', 'runtime'):
        path = tmp_path / (name + '.dat')
        path.write_text('{"events": 3}' if name == 'reference' else name)
        files[name] = path
    def row(name):
        return dict(path=str(files[name]), sha256=V.digest(files[name]))
    manifest = dict(schema=1, inputs=[dict(row('input'), events=3, events_with_steps=2)],
                    excluded=[row('excluded')], events=3, events_with_steps=2,
                    assets={k: row(k) for k in ('calib', 'deviation', 'reference', 'background', 'time_shift')})
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    seen = {}
    result = dict(events_with_steps=2, response_state={'events': 3}, passes=[{'events': 2}], timers={})
    def fake_run(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        Path(kwargs['emit']).write_text(json.dumps(result))
        return result
    monkeypatch.setattr(r3, 'run', fake_run)
    monkeypatch.setattr(V, 'revision', lambda p: dict(path='/clean/' + p, sha='clean-head'))
    monkeypatch.setattr(V, 'diagnostic_revision', lambda p: dict(path='/dirty/' + p, sha='dirty-head',
                        status=' M source.py\n?? extra.py', sources={'source.py': 'bytes'},
                        source_sha256='snapshot-hash', mode='diagnostic-worktree', published=False))
    monkeypatch.setattr(V, 'runtime_assets', lambda: {'runtime': str(files['runtime'])})
    monkeypatch.setattr(V.R, 'objective', lambda result, reference: {'objective': 1})
    monkeypatch.setattr(kit_task, 'score_result', lambda result, reference: {'objective': 2})
    out = tmp_path / 'out'
    return SimpleNamespace(files=files, manifest=manifest, manifest_path=manifest_path, seen=seen,
                           result=result, out=out, fake_run=fake_run, r3=r3)


def execute(case, **kwargs):
    return V.run(case.manifest_path, case.out, 101, kwargs.pop('knobs', {}), **kwargs)


def provenance(case):
    return json.loads((case.out / 'provenance.json').read_text())


@pytest.mark.parametrize('kind', ['object', 'inline', 'file'])
def test_config_forwarding_and_canonical_receipt(run_case, kind):
    c = run_case
    source = {'schema': 1, 'sipm': {'overvoltage': 2.9, 'time_constant': 20}, 'noise': False}
    if kind != 'object':
        source = json.dumps(source)
    if kind == 'file':
        path = c.out.parent / 'requested.json'
        path.write_text(source)
        source = path
    assert execute(c, response_config=source, knobs={'time_constant': 17}) is c.result
    kw = c.seen['kwargs']
    cfg = kw['response_config']
    assert cfg['sipm']['overvoltage'] == 2.9 and cfg['sipm']['time_constant'] == 17
    assert cfg['noise'] is False and kw['knobs'] == {'time_constant': 17}
    assert cfg['timing']['phase_scope'] == 'per_event'
    assert cfg['timing']['timeoffset_mode'] == 'zero'
    assert kw['v05'] and kw['objective'] and kw['max_passes'] == 1
    assert kw['joint_capture'] is None
    rec = provenance(c)
    resolved = rec['response_config']
    assert resolved['values'] == cfg
    assert resolved['canonical'] == RC.canonical(cfg)
    assert resolved['sha256'] == hashlib.sha256(resolved['canonical'].encode()).hexdigest()
    assert rec['code'] == rec['code_after'] and rec['code_policy'] == 'clean-revision'
    assert rec['status'] == 'complete' and not rec['diagnostic_code']
    assert resolved['requested_file'] == (dict(path=str(source), sha256=V.digest(source)) if kind == 'file' else None)


@pytest.mark.parametrize('calibration', [False, True])
def test_manifest_background_timecal_and_reference_remain_frozen(run_case, calibration):
    c = run_case
    before = {k: V.digest(p) for k, p in c.files.items()}
    execute(c, calibration=calibration, response_config={'schema': 1, 'background_scale': .8})
    kw = c.seen['kwargs']
    assert c.seen['args'][1:] == (str(c.files['calib']), str(c.files['deviation']))
    for name in ('background', 'time_shift'):
        value = str(c.files[name]) if calibration else None
        assert kw[name] == kw['response_config']['assets'][name] == value
    assert before == {k: V.digest(p) for k, p in c.files.items()}
    assert json.loads((c.out / 'comparison.json').read_text())['objective'] == (2 if calibration else 1)


@pytest.mark.parametrize('patch', [
    {'assets': {key: '/override'} } for key in RC.ASSETS
] + [{'assets': {'reference': '/override'}}, {'waveform': {'single_pe_file': '/override'}},
     {'sipm': {'thermal_rate': 1}}])
def test_config_rejected_before_physics_io(run_case, monkeypatch, patch):
    monkeypatch.setattr(V, 'verify', lambda *args: pytest.fail('physics input read before config validation'))
    monkeypatch.setattr(V, 'runtime_assets', lambda: pytest.fail('runtime assets read'))
    with pytest.raises(ValueError):
        execute(run_case, response_config={'schema': 1, **patch})
    assert not run_case.out.exists()


def test_invalid_knobs_rejected_before_physics_io(run_case, monkeypatch):
    monkeypatch.setattr(V, 'verify', lambda *args: pytest.fail('physics input read'))
    with pytest.raises(ValueError, match='knob'):
        execute(run_case, knobs={'not_a_knob': 1})
    assert not run_case.out.exists()


def test_joint_requires_diagnostic_code_before_any_io(tmp_path):
    with pytest.raises(ValueError, match='requires diagnostic_code'):
        V.run(tmp_path / 'missing.json', tmp_path / 'out', 1, {}, joint_capture=True)
    assert not (tmp_path / 'out').exists()


def test_diagnostic_and_joint_capture_receipts(run_case, monkeypatch):
    c = run_case
    monkeypatch.setattr(V, 'revision', lambda p: pytest.fail('diagnostic used clean revision gate'))
    module = ModuleType('crvfastsim.objective.joint')
    class JointCapture:
        def write(self, path):
            assert path == c.out / 'joint.npz'
            path.write_bytes(b'npz surrogate')
            return {'events': 3, 'diagnostic_only': True}
    module.JointCapture = JointCapture
    monkeypatch.setitem(sys.modules, module.__name__, module)
    execute(c, diagnostic_code=True, joint_capture=True)
    assert isinstance(c.seen['kwargs']['joint_capture'], JointCapture)
    p = provenance(c)
    assert p['diagnostic_code'] and p['joint_capture']
    assert p['code_policy'] == 'unpublished-diagnostic-snapshot'
    assert p['code'] == p['code_after'] and p['status'] == 'complete'
    assert all(not row['published'] and row['status'] for row in p['code'].values())
    for name in ('joint.npz', 'joint.json'):
        assert p['joint_files'][name]['sha256'] == V.digest(c.out / name)
    assert json.loads((c.out / 'joint.json').read_text()) == {'events': 3, 'diagnostic_only': True}


def test_diagnostic_code_alone_is_allowed(run_case):
    execute(run_case, diagnostic_code=True)
    assert provenance(run_case)['code_policy'] == 'unpublished-diagnostic-snapshot'
    assert run_case.seen['kwargs']['joint_capture'] is None
    assert not (run_case.out / 'joint.npz').exists()


@pytest.mark.parametrize('field', ['sha', 'status', 'sources', 'source_sha256'])
def test_diagnostic_source_or_git_mutation_fails_and_records_after(run_case, monkeypatch, field):
    before = V.diagnostic_revision('crvfastsim')
    after = deepcopy(before)
    after[field] = {'source.py': 'new'} if field == 'sources' else 'changed'
    monkeypatch.setattr(V, 'diagnostic_revision', lambda package: deepcopy(after if run_case.seen else before))
    with pytest.raises(ValueError, match='code changed'):
        execute(run_case, diagnostic_code=True)
    p = provenance(run_case)
    assert p['code']['crvfastsim'] == before and p['code_after']['crvfastsim'] == after
    assert p['status'] == 'failed'


@pytest.mark.parametrize('name', ['input', 'excluded', 'calib', 'deviation', 'reference',
                                 'background', 'time_shift', 'runtime', 'manifest', 'config'])
def test_all_input_mutations_fail_with_receipt(run_case, monkeypatch, name):
    c = run_case
    config = c.out.parent / 'requested.json'
    config.write_text('{"schema": 1}')
    target = {'manifest': c.manifest_path, 'config': config}.get(name, c.files.get(name))
    def mutate(*args, **kwargs):
        result = c.fake_run(*args, **kwargs)
        # Retain valid reference JSON so the post-scoring hash check is exercised.
        target.write_text('{"events": 3, "changed": true}')
        return result
    monkeypatch.setattr(c.r3, 'run', mutate)
    with pytest.raises(ValueError, match='changed'):
        execute(c, response_config=config)
    assert provenance(c)['status'] == 'failed'
    assert provenance(c)['integrity_errors']


def test_scoring_mutation_is_also_detected(run_case, monkeypatch):
    def score(*args):
        run_case.files['reference'].write_text('{"events": 3, "changed": true}')
        return {'objective': 1}
    monkeypatch.setattr(V.R, 'objective', score)
    with pytest.raises(ValueError, match='changed'):
        execute(run_case)


def test_physics_failure_still_records_after_snapshot(run_case, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic R3 failure')
    monkeypatch.setattr(run_case.r3, 'run', fail)
    with pytest.raises(RuntimeError, match='synthetic R3 failure'):
        execute(run_case, diagnostic_code=True)
    p = provenance(run_case)
    assert p['code_after'] == p['code'] and p['status'] == 'failed'
    assert p['execution_error'] == 'RuntimeError: synthetic R3 failure'


def test_missing_after_snapshot_is_failed_not_published(run_case, monkeypatch):
    original = V.diagnostic_revision
    def snapshot(package):
        if run_case.seen:
            raise OSError('source disappeared')
        return original(package)
    monkeypatch.setattr(V, 'diagnostic_revision', snapshot)
    with pytest.raises(ValueError, match='source disappeared'):
        execute(run_case, diagnostic_code=True)
    assert provenance(run_case)['code_after'] == {}
    assert provenance(run_case)['status'] == 'failed'


def test_default_run_keeps_clean_revision_gate(run_case, monkeypatch):
    monkeypatch.setattr(V, 'diagnostic_revision', lambda p: pytest.fail('default bypassed clean gate'))
    def dirty(package):
        raise ValueError('validation requires clean tracked code: dirty')
    monkeypatch.setattr(V, 'revision', dirty)
    with pytest.raises(ValueError, match='requires clean tracked code'):
        execute(run_case)
    assert not run_case.out.exists() and not run_case.seen


def test_default_run_rechecks_clean_revision_after(run_case, monkeypatch):
    def clean_then_dirty(package):
        if run_case.seen:
            raise ValueError('validation requires clean tracked code: dirty')
        return dict(path=package, sha='head')
    monkeypatch.setattr(V, 'revision', clean_then_dirty)
    with pytest.raises(ValueError, match='clean tracked code'):
        execute(run_case)
    assert provenance(run_case)['status'] == 'failed'


def test_nonoverwrite_existing_output(run_case):
    c = run_case
    c.out.mkdir()
    sentinel = c.out / 'provenance.json'
    sentinel.write_text('leave existing results alone')
    with pytest.raises(FileExistsError):
        execute(c, diagnostic_code=True)
    assert sentinel.read_text() == 'leave existing results alone'
    assert not c.seen


def test_cli_forwards_new_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(V, 'run', lambda *a, **kw: seen.update(args=a, kwargs=kw))
    monkeypatch.setattr(sys, 'argv', ['validation', 'run', '--manifest', 'manifest.json', '--out', 'new',
                                    '--seed', '17', '--knob', 'time_constant=15', '--response-config',
                                    '{"schema":1}', '--joint-capture', '--diagnostic-code'])
    V.main()
    assert seen['args'] == ('manifest.json', 'new', 17, {'time_constant': 15.}, False, False)
    assert seen['kwargs'] == dict(response_config='{"schema":1}', joint_capture=True, diagnostic_code=True)


def test_revision_original_api_and_gate(tmp_path, monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, 'find_spec', lambda p: SimpleNamespace(origin=str(tmp_path / p / '__init__.py')))
    dirty = []
    def git(command, **kwargs):
        assert command[:3] == ['git', '-C', str(tmp_path)]
        return '?? new.py\n' if dirty and command[3] == 'status' else ('head\n' if command[3] == 'rev-parse' else '')
    monkeypatch.setattr(V.subprocess, 'check_output', git)
    assert V.revision('synthetic') == {'path': str(tmp_path), 'sha': 'head'}
    dirty.append(True)
    with pytest.raises(ValueError, match='clean tracked code'):
        V.revision('synthetic')


def test_snapshot_includes_package_and_support_source_and_detects_changes(tmp_path, monkeypatch):
    import importlib.util
    package = tmp_path / 'synthetic'
    package.mkdir()
    names = ['synthetic/__init__.py', 'synthetic/untracked.py', 'synthetic/ignored/config.json',
             'scripts/check.py', 'tests/test_new.py', 'docs/diagnostic.md', 'setup.cfg', 'entry.py']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(name)
    (package / 'ignore.pyc').write_bytes(b'not source')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda p: SimpleNamespace(origin=str(package / '__init__.py')))
    def git(command, **kwargs):
        if command[3] == 'status':
            return ' M synthetic/__init__.py\n?? synthetic/untracked.py\n'
        if command[3] == 'rev-parse':
            return 'unpublished-head\n'
        if command[3] == 'ls-files':
            return 'synthetic/__init__.py\0entry.py\0'
        pytest.fail(str(command))
    monkeypatch.setattr(V.subprocess, 'check_output', git)
    before = V.diagnostic_revision('synthetic')
    assert set(before['sources']) == set(names)
    assert before['sources'] == {name: V.digest(tmp_path / name) for name in names}
    assert before['sha'] == 'unpublished-head' and not before['published']
    assert before == V.diagnostic_revision('synthetic')
    (package / 'untracked.py').write_text('changed while status stays identical')
    changed = V.diagnostic_revision('synthetic')
    assert before['source_sha256'] != changed['source_sha256']
    (package / 'new.py').write_text('new source')
    assert 'synthetic/new.py' in V.diagnostic_revision('synthetic')['sources']
    (package / 'new.py').unlink()
    assert V.diagnostic_revision('synthetic') == changed
    (package / 'untracked.py').unlink()
    assert 'synthetic/untracked.py' not in V.diagnostic_revision('synthetic')['sources']


def test_real_joint_collector_serialization(run_case, monkeypatch):
    import numpy as np
    c = run_case
    def capture(*args, **kwargs):
        result = c.fake_run(*args, **kwargs)
        c.r3._capture_empty(kwargs['joint_capture'], np.array([[1400, 1155, i] for i in range(3)]), 0)
        return result
    monkeypatch.setattr(c.r3, 'run', capture)
    execute(c, joint_capture=True, diagnostic_code=True)
    summary = json.loads((c.out / 'joint.json').read_text())
    with np.load(c.out / 'joint.npz', allow_pickle=False) as data:
        assert json.loads(str(data['summary_json'])) == summary
    assert provenance(c)['joint_files']['joint.npz']['sha256'] == V.digest(c.out / 'joint.npz')


@pytest.mark.parametrize('changes', [
    {'diagnostic_code': True}, {'code_policy': 'unpublished-diagnostic-snapshot'}, {'status': 'failed'}
])
def test_paired_refuses_diagnostic_or_failed_provenance(changes):
    # Admission must precede reading any result fields; old receipts without these
    # new metadata fields retain their existing comparison behavior.
    with pytest.raises(ValueError, match='diagnostic code|not complete'):
        V._validate_point({}, {}, changes)


def test_calibration_requires_manifest_background_and_timecal(run_case):
    c = run_case
    del c.manifest['assets']['background']
    c.manifest_path.write_text(json.dumps(c.manifest))
    with pytest.raises(ValueError, match='requires background, time_shift'):
        execute(c, calibration=True)
    assert not c.out.exists() and not c.seen


def test_config_override_refused_even_when_matching_manifest(run_case):
    with pytest.raises(ValueError, match='cannot override'):
        execute(run_case, response_config={'schema': 1, 'assets': {'calib': str(run_case.files['calib'])}})
    assert not run_case.out.exists()


def test_snapshot_checks_git_stability_while_hashing(tmp_path, monkeypatch):
    import importlib.util
    package = tmp_path / 'synthetic'
    package.mkdir()
    (package / '__init__.py').write_text('')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda p: SimpleNamespace(origin=str(package / '__init__.py')))
    status_calls = []
    def git(command, **kwargs):
        if command[3] == 'rev-parse':
            return 'head\n'
        if command[3] == 'status':
            status_calls.append(True)
            return '' if len(status_calls) == 1 else ' M synthetic/__init__.py\n'
        return 'synthetic/__init__.py\0'
    monkeypatch.setattr(V.subprocess, 'check_output', git)
    with pytest.raises(ValueError, match='changed while taking diagnostic snapshot'):
        V.diagnostic_revision('synthetic')
