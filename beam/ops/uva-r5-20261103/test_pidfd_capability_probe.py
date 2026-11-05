"""Portable mocks only; no Linux capability operation or remote execution."""
import builtins
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('pidfd_probe_test',
    Path(__file__).with_name('pidfd_capability_probe.py'))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture
def calls(monkeypatch):
    result = []
    monkeypatch.setattr(probe.platform, 'uname', lambda: SimpleNamespace(
        release='4.18.test', version='test', machine='x86_64', node='testnode'))
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: io.BytesIO(
        b'Name:\ttest\nSeccomp:\t0\nNoNewPrivs:\t0\nCapEff:\t0000\n'))
    monkeypatch.setattr(probe.os, 'readlink', lambda p: 'pid:[123]')
    monkeypatch.setattr(probe, 'self_pidfd', lambda: result.append('open_self') or 17)
    monkeypatch.setattr(probe.signal, 'pidfd_send_signal',
                        lambda *a: result.append(('signal', a)), raising=False)
    monkeypatch.setattr(probe.os, 'close', lambda fd: result.append(('close', fd)))
    return result


def test_self_only_success_remains_unverified(calls):
    r = probe.observe()
    assert calls == ['open_self', ('signal', (17, 0, None, 0)), ('close', 17)]
    assert r['capability_available'] and r['diagnostic_completed']
    assert r['self_status']['Seccomp'] == '0'
    assert r['error'] is None
    assert not r['verified'] and not r['fully_enforced'] and r['transport_events'] == 0


def test_enosys_is_observed_not_lifecycle_success(calls, monkeypatch):
    def unsupported():
        calls.append('open_self')
        raise OSError(38, 'Function not implemented')
    monkeypatch.setattr(probe, 'self_pidfd', unsupported)
    r = probe.observe()
    assert r['diagnostic_completed'] and not r['capability_available']
    assert not r['pidfd_open_self'] and not r['self_fd_closed']
    assert r['error']['errno'] == 38 and calls == ['open_self']


@pytest.mark.parametrize('error', [PermissionError(1, 'refused'), RuntimeError('missing')])
def test_signal_zero_failure_closes_once(calls, monkeypatch, error):
    def refuse(*args):
        raise error
    monkeypatch.setattr(probe.signal, 'pidfd_send_signal', refuse)
    r = probe.observe()
    assert r['pidfd_open_self'] and r['self_fd_closed']
    assert not r['capability_available'] and r['error'] is not None
    assert calls == ['open_self', ('close', 17)]


def test_close_failure_never_retried(calls, monkeypatch):
    def refuse(fd):
        calls.append(('failed_close', fd))
        raise OSError(5, 'close fault')
    monkeypatch.setattr(probe.os, 'close', refuse)
    r = probe.observe()
    assert r['pidfd_signal_zero_self'] and not r['capability_available']
    assert r['error']['errno'] == 5 and not r['self_fd_closed']
    assert calls.count(('failed_close', 17)) == 1


def test_status_size_refusal_precedes_capability(calls, monkeypatch):
    monkeypatch.setattr(builtins, 'open', lambda *a, **k: io.BytesIO(b'x' * 16385))
    r = probe.observe()
    assert not calls and not r['capability_available']
    assert r['error']['type'] == 'RuntimeError'


def test_receipt_is_exclusive_and_unsupported_capability_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'OUT', tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '123456')
    monkeypatch.setattr(probe.signal, 'alarm', lambda n: None, raising=False)
    monkeypatch.setattr(probe, 'observe', lambda: {
        'diagnostic_completed': True, 'capability_available': False, 'verified': False})
    assert probe.main() == 0
    receipt = tmp_path / 'pidfd_capability_123456/receipt.json'
    assert receipt.stat().st_size < probe.MAX_RECEIPT
    assert '"capability_available": false' in receipt.read_text()
    with pytest.raises(FileExistsError):
        probe.main()


@pytest.mark.parametrize('job', ['', '../a', '9' * 13, '\u0661'])
def test_job_id_refused_before_any_directory(tmp_path, monkeypatch, job):
    monkeypatch.setattr(probe, 'OUT', tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', job)
    monkeypatch.setattr(probe.signal, 'alarm', lambda n: None, raising=False)
    with pytest.raises(RuntimeError):
        probe.main()
    assert not list(tmp_path.iterdir())
