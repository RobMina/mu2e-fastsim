"""Local diagnostic-only tests: no subprocesses, signals, mounts, or containers."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def model(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "boundary_diagnostic_under_test", Path(__file__).with_name("apptainer_boundary_probe.py"))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    real_signal = probe.signal
    receipt = tmp_path / "apptainer_boundary_probe_42" / "status.json"
    events = []
    knobs = {"restore_fail": False, "mask_failures": 0, "setup_fail": False,
             "reap_fail": False, "code": 0, "deadline": False, "chunks": [b""]}

    def event(name):
        assert not receipt.exists(), "cleanup ran after status publication"
        events.append(name)

    class Pipe:
        def fileno(self):
            return 123

        def close(self):
            event("pipe_close")

    class Child:
        pid = 424242
        stdout = Pipe()
        returncode = 0  # Models a child that has already exited before collection.

        def wait(self, timeout):
            event("wait")
            self.returncode = knobs["code"]
            return self.returncode

    child = Child()

    def popen(*args, **kwargs):
        event("launch")
        assert kwargs["start_new_session"] and kwargs["close_fds"]
        assert "--setup" in args[0]
        return child

    def handler(sig, value):
        event(("handler", sig.name, value))
        assert value is not None
        if value == real_signal.SIG_DFL and knobs["restore_fail"]:
            raise OSError("restore failed " + "\x00\U0001f600\"" * 5000)

    def mask(how, values):
        event(("mask", how, set(values)))
        if how == real_signal.SIG_SETMASK and knobs["mask_failures"]:
            knobs["mask_failures"] -= 1
            raise OSError("mask restore failed")
        return {real_signal.SIGUSR1}

    def killpg(pid, sig):
        event(("killpg", sig))
        assert pid == child.pid
        if knobs["reap_fail"]:
            raise PermissionError("mock reap denied")
        raise ProcessLookupError("quick-exit child already gone")

    def mkdtemp(**kwargs):
        event("setup")
        if knobs["setup_fail"]:
            raise OSError("mock setup failure " + "\x00\U0001f600\"" * 5000)
        local = tmp_path / "retained-work"
        local.mkdir()
        return str(local)

    class Selector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            event("selector_close")

        def register(self, *args):
            pass

        def select(self, timeout):
            return [True]

    clock = [0]

    def monotonic():
        clock[0] += 1
        return clock[0] * (601 if knobs["deadline"] else 0.01)

    monkeypatch.setattr(probe, "DATA", tmp_path)
    monkeypatch.setattr(probe, "signal", SimpleNamespace(
        SIGINT=real_signal.SIGINT, SIGTERM=real_signal.SIGTERM,
        SIG_DFL=real_signal.SIG_DFL, SIG_IGN=real_signal.SIG_IGN,
        SIGKILL=real_signal.SIGKILL, SIG_BLOCK=real_signal.SIG_BLOCK,
        SIG_SETMASK=real_signal.SIG_SETMASK, getsignal=lambda sig: None,
        signal=handler, pthread_sigmask=mask))
    monkeypatch.setattr(probe, "os", SimpleNamespace(
        environ={"SLURM_JOB_ID": "42"}, path=probe.os.path,
        chmod=lambda *args: None, set_blocking=lambda *args: None,
        read=lambda *args: knobs["chunks"].pop(0), killpg=killpg))
    monkeypatch.setattr(probe, "subprocess", SimpleNamespace(
        Popen=popen, DEVNULL=-3, PIPE=-1, STDOUT=-2))
    monkeypatch.setattr(probe, "selectors", SimpleNamespace(
        DefaultSelector=Selector, EVENT_READ=1))
    monkeypatch.setattr(probe, "tempfile", SimpleNamespace(mkdtemp=mkdtemp))
    monkeypatch.setattr(probe, "time", SimpleNamespace(monotonic=monotonic))

    def run():
        result = probe.diagnostic("/mock/apptainer")
        raw = receipt.read_bytes()
        assert len(raw) <= 4096
        report = json.loads(raw)
        log = receipt.with_name("merged.log")
        if log.exists():
            assert log.stat().st_size <= 1024**2
        assert report["verified"] is False
        assert report["fully_enforced"] is False
        assert report["transport_events"] == 0
        assert report["initial_signal_dispositions"] == {"SIGINT": "None", "SIGTERM": "None"}
        assert report["none_handlers_normalized_to_SIG_DFL"] == ["SIGINT", "SIGTERM"]
        restores = [e for e in events if isinstance(e, tuple) and e[0] == "handler"
                    and e[2] == real_signal.SIG_DFL]
        assert len(restores) == 2  # Both attempted, even when the first raises.
        if report["transient_work_retained"]:
            assert Path(report["transient_work_retained"]).is_dir()
        return result, report

    return SimpleNamespace(knobs=knobs, run=run, events=events, signal=real_signal)


def test_quick_exit_none_handlers_normalized_before_success(model):
    result, report = model.run()
    assert result == 0
    assert report["state"] == "child_checks_passed"
    assert report["wrapper_reaped"] is True
    assert report["teardown_errors"] == []


def test_handler_restoration_failure_never_false_pass(model):
    model.knobs["restore_fail"] = True
    result, report = model.run()
    assert result == 1
    assert report["state"] == "failed"
    assert report["returncode"] == 0
    assert len(report["teardown_errors"]) == 2
    assert all("handler restore" in e for e in report["teardown_errors"])


@pytest.mark.parametrize("failure", ["setup_fail", "deadline", "reap_fail"])
def test_failures_have_bounded_failed_status(model, failure):
    model.knobs[failure] = True
    result, report = model.run()
    assert result == 1
    assert report["state"] == "failed"
    if failure == "setup_fail":
        assert "mock setup failure" in report["error"]
        assert report["launch_state"] == "not_started"
    elif failure == "deadline":
        assert "diagnostic deadline" in report["error"]
        assert report["wrapper_reaped"] is True
    else:
        assert any("mock reap denied" in e for e in report["teardown_errors"])


@pytest.mark.parametrize("failures", [1, 2])
def test_mask_restore_retried_and_failure_recorded(model, failures):
    model.knobs["mask_failures"] = failures
    result, report = model.run()
    assert result == 1
    assert report["state"] == "failed"
    assert "mask restore failed" in report["error"]
    attempts = [e for e in model.events if isinstance(e, tuple) and e[0] == "mask"
                and e[1] == model.signal.SIG_SETMASK]
    assert len(attempts) == 2
    assert all(e[2] == {model.signal.SIGUSR1} for e in attempts)
    assert bool(report["teardown_errors"]) == (failures == 2)


def test_setup_child_failure_is_not_pass(model):
    model.knobs["code"] = 1
    result, report = model.run()
    assert result == 1
    assert report["state"] == "failed"
    assert report["returncode"] == 1


def test_oversized_child_log_is_discarded(model):
    model.knobs["chunks"] = [b"x" * 65536] * 17 + [b""]
    result, report = model.run()
    assert result == 0
    assert report["log_bytes"] == 1024**2
    assert report["discarded_log_bytes"] == 65536


@pytest.fixture
def runtime_model(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "runtime_config_under_test", Path(__file__).with_name("apptainer_boundary_probe.py"))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    ce = tmp_path / "runtime"
    ce.mkdir()
    for name in ("cvmfsexec", "mountrepo"):
        (ce / name).write_bytes(("unit-only mock " + name).encode())
    return probe, ce, tmp_path / "cache"


def test_nonesting_and_log_disposition_configuration(runtime_model, monkeypatch, capsys):
    import hashlib
    import inspect
    probe, ce, cache = runtime_model
    for name, attribute in (("cvmfsexec", "CVMFSEXEC_SHA256"),
                            ("mountrepo", "MOUNTREPO_SHA256")):
        monkeypatch.setattr(probe, attribute, hashlib.sha256((ce / name).read_bytes()).hexdigest())
    probe.configure_cvmfsexec(ce, cache)
    config = ce / "dist/etc/cvmfs"
    defaults = (config / "default.local").read_text()
    assert 'CVMFS_CACHE_BASE=' + str(cache) + '\n' in defaults
    assert 'CVMFS_USYSLOG=/dev/null\nCVMFS_DEBUGLOG=\n' in defaults
    assert {path.name for path in (config / "config.d").iterdir()} == {
        repository + ".local" for repository in probe.REPOSITORIES}
    for repository in probe.REPOSITORIES:
        assert (config / "config.d" / (repository + ".local")).read_text() == (
            'CVMFS_USYSLOG=/dev/null\nCVMFS_DEBUGLOG=\n')
    source = inspect.getsource(probe.setup)
    assert '"-N", *REPOSITORIES, "--"' in source
    assert source.index('configure_cvmfsexec(') < source.index('os.execve(')
    report = json.loads(capsys.readouterr().out)
    assert report['repository_logs_disposition'] == 'discard:/dev/null'
    assert report['runtime_mode_observed'] is False and report['verified'] is False


@pytest.mark.parametrize('changed', ['cvmfsexec', 'mountrepo'])
def test_unreviewed_runtime_helper_refused_before_configuration(runtime_model, monkeypatch, changed):
    import hashlib
    probe, ce, cache = runtime_model
    for name, attribute in (("cvmfsexec", "CVMFSEXEC_SHA256"),
                            ("mountrepo", "MOUNTREPO_SHA256")):
        expected = hashlib.sha256((ce / name).read_bytes()).hexdigest()
        monkeypatch.setattr(probe, attribute, '0' * 64 if name == changed else expected)
    with pytest.raises(ValueError, match='unreviewed CVMFSEXEC helper: ' + changed):
        probe.configure_cvmfsexec(ce, cache)
    assert not (ce / 'dist').exists()


@pytest.fixture
def lifecycle_model(runtime_model, monkeypatch):
    from dataclasses import dataclass
    probe, local, _ = runtime_model
    (local / 'work').mkdir()
    report = SimpleNamespace(success=True, ancestry_drained=True)
    calls = []

    def run_tree(command, environment, cwd, **kwargs):
        calls.append((command, environment, cwd, kwargs))
        return report

    real_spec = probe.importlib.util.spec_from_file_location
    real_loader = probe.importlib.util.module_from_spec
    spec = SimpleNamespace(name='r5_subreaper', loader=SimpleNamespace(exec_module=lambda _: None))
    monkeypatch.setattr(probe.importlib.util, 'spec_from_file_location',
                        lambda name, path: spec if name == 'r5_subreaper' else real_spec(name, path))
    monkeypatch.setattr(probe.importlib.util, 'module_from_spec',
                        lambda value: SimpleNamespace(run_tree=run_tree) if value is spec else real_loader(value))
    monkeypatch.setattr('dataclasses.asdict', lambda value: vars(value).copy())
    monkeypatch.setattr(probe.time, 'monotonic_ns', lambda: 2_000_000_000)
    ready = {'nonce': 'normal_' + local.name, 'mode': 'normal', 'intermediate_reaped': True,
             'emergency_expiry_seconds': 30, 'verified': False}
    heartbeats = [{'sequence': 0, 'monotonic_ns': 1_000_000_000}]

    def records():
        (local / 'work/lifecycle-ready.json').write_text(json.dumps(ready) + '\n')
        (local / 'work/lifecycle-heartbeat.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in heartbeats))

    def run():
        records()
        return probe._supervise_lifecycle(local, ['/mock/cvmfsexec', '-N'])

    return SimpleNamespace(probe=probe, local=local, ready=ready, heartbeats=heartbeats,
                           report=report, calls=calls, run=run)


def test_normal_lifecycle_gate_is_explicitly_unverified(lifecycle_model, capsys):
    model = lifecycle_model
    assert model.run() == 0
    assert model.calls[0][0] == ['/mock/cvmfsexec', '-N']
    assert model.calls[0][2:] == ('/', {'timeout': 540, 'external_watchdog_present': True})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[-1]['fully_enforced'] is False
    assert lines[-1]['drained_before_emergency_seconds'] == 1
    assert lines[-1]['transport_events'] == 0


@pytest.mark.parametrize('field,value', [('mode', 'linger'), ('nonce', 'stale'),
    ('intermediate_reaped', False), ('verified', True), ('emergency_expiry_seconds', 31)])
def test_mismatched_lifecycle_record_refused(lifecycle_model, field, value):
    lifecycle_model.ready[field] = value
    with pytest.raises(ValueError, match='mismatched'):
        lifecycle_model.run()


@pytest.mark.parametrize('drained,success', [(False, True), (True, False), (False, False)])
def test_unknown_or_failed_ancestry_cannot_admit(lifecycle_model, drained, success):
    model = lifecycle_model
    model.report.ancestry_drained, model.report.success = drained, success
    assert model.run() == 1


def test_emergency_expiry_is_not_normal_lifecycle_success(lifecycle_model):
    model = lifecycle_model
    model.heartbeats[0]['monotonic_ns'] = -30_000_000_000
    with pytest.raises(ValueError, match='emergency expiry'):
        model.run()


def test_failed_ancestry_never_opens_payload_records(lifecycle_model, monkeypatch):
    model = lifecycle_model
    model.report.success = False
    model.report.ancestry_drained = False

    def forbidden(*args, **kwargs):
        pytest.fail('unknown ancestry must not open fixture records')

    monkeypatch.setattr(model.probe.os, 'open', forbidden)
    assert model.probe._supervise_lifecycle(model.local, ['/mock/cvmfsexec', '-N']) == 1


def test_normal_lifecycle_setup_copy_and_subreaper_command(runtime_model, monkeypatch):
    probe, local, cache = runtime_model
    copies, launches = [], []
    monkeypatch.setattr(probe.signal, 'pthread_sigmask', lambda *args: set())
    monkeypatch.setattr(probe.os, 'readlink', lambda path: 'pid:[100]')
    monkeypatch.setattr(probe.shutil, 'copyfile', lambda a, b: copies.append((Path(a), Path(b))))
    monkeypatch.setattr(probe.shutil, 'copytree', lambda *args, **kwargs: None)
    monkeypatch.setattr(probe, 'configure_cvmfsexec', lambda *args: None)
    monkeypatch.setattr(probe, '_supervise_lifecycle',
                        lambda work, command: launches.append((work, command)) or 0)
    assert probe.setup(local, '/mock/apptainer', lifecycle=True) == 0
    destinations = {str(path.relative_to(local)) for _, path in copies}
    assert {'checks/flash_reference_lifecycle_payload.py', 'subreaper.py',
            'lifecycle.py', 'diagnostic.py'} <= destinations
    assert 'checks/run_confined_fixture.sh' not in destinations
    work, command = launches[0]
    assert work == local and command[0] == str(local / 'cvmfsexec/cvmfsexec')
    assert command[1:6] == ['-N', *probe.REPOSITORIES]
    assert str(local / 'lifecycle.py') in command
    assert command[-2:] == ['--host-pid-namespace', 'pid:[100]']
    assert '--fixture' not in command
