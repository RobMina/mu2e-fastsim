"""Local trusted scripts/mocks only: these tests certify no runtime confinement."""
from collections import UserDict
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

import pytest

from beamfastsim import flash_reference_budget as budget
from beamfastsim import flash_reference_guarded_run as guarded
from beamfastsim import flash_reference_inventory as inventory

SLOT = budget.SLOT_NAMES[0]
RECEIPT = "test-only independent fixture evidence; NOT a runtime certificate"


def setup(tmp_path):
    # Inventory rejects symlink ancestors, including macOS /var -> /private/var.
    tmp_path = tmp_path.resolve()
    root = tmp_path / "saved"
    work = tmp_path / "work"
    root.mkdir()
    work.mkdir(mode=0o700)
    for name in ("raw_1.art", "export.root", "job.fcl"):
        (work / name).write_bytes(name.encode())
    return root, work


def execute(root, work, script="print('hello')", **kwargs):
    kwargs.setdefault("parent_timeout", 3)
    kwargs.setdefault("byte_limit", 4096)
    kwargs.setdefault("status_reserve", 128)
    kwargs.setdefault("log_limit", 32)
    return guarded.execute_and_publish(root, SLOT, [sys.executable, "-c", script],
                                       work=work, **kwargs)


def verify(receipt, argv, work, returncode):
    assert receipt == RECEIPT
    assert argv[0] == sys.executable
    assert work.is_dir()
    assert returncode == 0
    return True  # Trusted local fixture assertion, not real confinement evidence.


def verified(root, work, script="print('hello')", **kwargs):
    kwargs.setdefault("independently_verified_confinement_receipt", RECEIPT)
    kwargs.setdefault("verify_callback", verify)
    return execute(root, work, script, **kwargs)


def charged(root, report):
    slot = root / SLOT
    assert sum(p.stat().st_size for p in slot.iterdir()) == report.saved_bytes <= 4096
    status = json.loads((slot / "status.json").read_text())
    assert status["saved_bytes"] == report.saved_bytes
    return status


def test_fixed_publication_and_false_gate(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    real_popen = subprocess.Popen
    children = []
    def launch(*args, **kwargs):
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["close_fds"] is True
        assert kwargs["start_new_session"] is True
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(guarded.subprocess, "Popen", launch)
    original = budget.SerializedBudget.publish
    def publish(self, *args, **kwargs):
        assert children[0].returncode == 0
        return original(self, *args, **kwargs)
    monkeypatch.setattr(budget.SerializedBudget, "publish", publish)
    report = verified(root, work, "import os; os.write(1,b'out'); os.write(2,b'err')")
    assert report.state == "complete"
    assert report.fully_enforced is False
    assert report.publication_verified_by_caller
    assert report.published == ("output.art", "output.root", "config.fcl")
    assert (root / SLOT / "merged.log").read_bytes() == b"outerr"
    assert (root / SLOT / "output.art").read_bytes() == b"raw_1.art"
    assert charged(root, report)["state"] == "complete"
    with pytest.raises(FileExistsError):
        verified(root, work)


@pytest.mark.parametrize("extra", [{}, {"independently_verified_confinement_receipt": RECEIPT},
                                    {"verify_callback": verify}])
def test_no_self_certification(tmp_path, monkeypatch, extra):
    root, work = setup(tmp_path)
    def unexpected_inventory(*args, **kwargs):
        pytest.fail("inventory requires external verification first")
    monkeypatch.setattr(guarded, "validate_inventory", unexpected_inventory)
    report = execute(root, work, **extra)
    assert report.state == "unverified"
    assert report.fully_enforced is False
    assert not report.published
    assert charged(root, report)["state"] == "failed"


def test_log_flood_drains_past_cap(tmp_path):
    root, work = setup(tmp_path)
    report = verified(root, work, "import os; [os.write(1,b'x'*65536) for _ in range(32)]")
    assert (root / SLOT / "merged.log").stat().st_size == 32
    assert report.discarded_log_bytes == 32 * 65536 - 32
    charged(root, report)


@pytest.mark.parametrize("script", ["import time; time.sleep(30)",
                                    "import os,time; os.close(1); os.close(2); time.sleep(30)",
                                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"])
def test_timeout_quiet_and_no_eof_reaped(tmp_path, script):
    root, work = setup(tmp_path)
    start = time.monotonic()
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, script, parent_timeout=0.15)
    report = error.value.report
    assert time.monotonic() - start < 3
    assert report.wrapper_reaped
    assert not report.published
    assert charged(root, report)["state"] == "failed"


@pytest.mark.parametrize("case", ["nonzero", "denied", "symlink", "overflow", "missing"])
def test_failures_retain_receipts_and_partial_files(tmp_path, case):
    root, work = setup(tmp_path)
    kwargs = {}
    script = "print('hello')"
    if case == "nonzero":
        script = "raise SystemExit(7)"
    elif case == "denied":
        kwargs["verify_callback"] = lambda *args: False
    elif case == "symlink":
        (work / "raw_1.art").unlink()
        (work / "raw_1.art").symlink_to(work / "export.root")
    elif case == "overflow":
        (work / "export.root").write_bytes(b"x" * 10000)
    else:
        (work / "job.fcl").unlink()
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, script, **kwargs)
    report = error.value.report
    assert report.wrapper_reaped
    assert not report.fully_enforced
    assert charged(root, report)["state"] == "failed"
    if case == "overflow":
        assert (root / SLOT / "output.root").stat().st_size > 0
        assert (root / SLOT / "output.art").exists()


def test_pipe_failure_terminates_and_retains_slot(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    def fail(*args):
        raise OSError("injected drain failure")
    monkeypatch.setattr(guarded, "_drain", fail)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, "import time; time.sleep(30)")
    assert error.value.report.wrapper_reaped
    assert charged(root, error.value.report)["state"] == "failed"


def test_fixture_enum_only(tmp_path):
    root, work = setup(tmp_path)
    (work / "raw_1.art").rename(work / "flash_reference_fixture_1.art")
    report = verified(root, work, output_layout=guarded.OutputLayout.NOG4_FIXTURE)
    assert report.state == "complete"
    charged(root, report)


@pytest.mark.parametrize("kwargs", [{"parent_timeout": 0}, {"parent_timeout": float("inf")},
                                   {"output_layout": "../escape"},
                                   {"independently_verified_confinement_receipt": ""}])
def test_invalid_arguments_do_not_reserve(tmp_path, kwargs):
    root, work = setup(tmp_path)
    with pytest.raises(ValueError):
        execute(root, work, **kwargs)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("stage", ["start", "spawn"])
def test_start_and_spawn_failure_keep_failed_receipt(tmp_path, monkeypatch, stage):
    root, work = setup(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("injected " + stage + " failure")
    if stage == "start":
        monkeypatch.setattr(budget.SerializedBudget, "start", fail)
    else:
        monkeypatch.setattr(guarded.subprocess, "Popen", fail)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work)
    report = error.value.report
    assert report.returncode is None
    assert not report.wrapper_reaped
    assert not report.published
    assert charged(root, report)["state"] == "failed"
    assert (root / SLOT / "reservation.json").exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX local fixture")
def test_descendant_holding_pipe_cannot_hide_timeout(tmp_path):
    root, work = setup(tmp_path)
    script = "import os,time; pid=os.fork(); time.sleep(30) if pid==0 else None"
    start = time.monotonic()
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, script, parent_timeout=0.15)
    assert time.monotonic() - start < 3
    assert error.value.report.wrapper_reaped
    assert not error.value.report.published
    charged(root, error.value.report)


def test_unconfirmed_cleanup_is_explicit_and_never_publishes(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    def fail_cleanup(child):
        assert child.poll() is not None  # This fixture leaves no live child.
        raise subprocess.TimeoutExpired(child.args, 0.01)
    monkeypatch.setattr(guarded, "_terminate_and_reap", fail_cleanup)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, "raise SystemExit(2)")
    report = error.value.report
    assert not report.wrapper_reaped
    assert "cleanup unconfirmed" in report.failure
    assert not report.published
    charged(root, report)


@pytest.mark.skipif(not hasattr(signal, "pthread_sigmask"), reason="POSIX signal mask")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_postspawn_pending_interrupt_reaps_child(tmp_path, monkeypatch, sig):
    root, work = setup(tmp_path)
    real_popen = subprocess.Popen
    children = []
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    old_handler = signal.getsignal(sig)
    def interrupt(signum, frame):
        raise KeyboardInterrupt("postspawn pending signal")
    def launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        os.kill(os.getpid(), sig)
        return child
    signal.signal(sig, interrupt)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
    expected_mask = original_mask | {signal.SIGUSR1}
    monkeypatch.setattr(guarded.subprocess, "Popen", launch)
    try:
        with pytest.raises(KeyboardInterrupt) as error:
            verified(root, work, "import time; time.sleep(30)")
        report = error.value.report
        assert report.launch_state == "acquired"
        assert report.wrapper_reaped
        assert children[0].poll() is not None
        assert not report.published
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == expected_mask
        charged(root, report)
    finally:
        signal.signal(sig, old_handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
            child.stdout.close()


def test_postspawn_exception_unknown_launch_retains_work(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    real_popen = subprocess.Popen
    children = []
    def launch(*args, **kwargs):
        children.append(real_popen(*args, **kwargs))
        raise OSError("postspawn handle lost")
    monkeypatch.setattr(guarded.subprocess, "Popen", launch)
    try:
        with pytest.raises(guarded.GuardedRunError) as error:
            verified(root, work, "import time; time.sleep(30)")
        report = error.value.report
        assert report.launch_state == "unknown"
        assert not report.wrapper_reaped
        assert not report.published
        assert "retain work" in report.failure
        assert work.is_dir()
        assert (work / "raw_1.art").exists()
        charged(root, report)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
            child.stdout.close()


@pytest.mark.parametrize("failure", ["drain", "stream"])
def test_discarded_observed_bytes_survive_stream_failure(tmp_path, monkeypatch, failure):
    root, work = setup(tmp_path)
    def chunks(child, deadline):
        yield b"x" * 100
        yield b"y" * 60
        if failure == "drain":
            raise OSError("after observed discard")
    monkeypatch.setattr(guarded, "_drain", chunks)
    if failure == "stream":
        original = budget.SerializedBudget.stream_log
        def fail(self, chunks):
            original(self, chunks)
            raise OSError("after stream discard")
        monkeypatch.setattr(budget.SerializedBudget, "stream_log", fail)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, "import time; time.sleep(30)")
    report = error.value.report
    assert report.discarded_log_bytes == 160 - 32
    assert (root / SLOT / "merged.log").stat().st_size == 32
    assert report.wrapper_reaped
    assert not report.published
    charged(root, report)


def test_flood_then_timeout_preserves_observed_discard(tmp_path):
    root, work = setup(tmp_path)
    script = "import os,time; [os.write(1,b'x'*65536) for _ in range(8)]; time.sleep(30)"
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, script, parent_timeout=0.5)
    report = error.value.report
    assert report.discarded_log_bytes == 8 * 65536 - 32
    assert report.wrapper_reaped
    assert not report.published
    charged(root, report)


def test_stream_failure_counts_observed_not_unread_chunks(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    def chunks(child, deadline):
        yield b"x" * 100
        pytest.fail("unread chunk must not be counted")
    def fail(self, chunks):
        assert next(chunks) == b"x" * 100
        raise OSError("failed before saving observed chunk")
    monkeypatch.setattr(guarded, "_drain", chunks)
    monkeypatch.setattr(budget.SerializedBudget, "stream_log", fail)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, "import time; time.sleep(30)")
    assert error.value.report.discarded_log_bytes == 100
    assert error.value.report.wrapper_reaped
    assert not error.value.report.published
    charged(root, error.value.report)


@pytest.mark.skipif(not hasattr(signal, "pthread_sigmask"), reason="POSIX signal mask")
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, OSError])
def test_restore_failure_before_mask_change_retried_after_cleanup(tmp_path, monkeypatch, exception):
    root, work = setup(tmp_path)
    real_mask = signal.pthread_sigmask
    original_mask = real_mask(signal.SIG_BLOCK, set())
    old_term_handler = signal.getsignal(signal.SIGTERM)
    expected_mask = original_mask | {signal.SIGUSR1}
    real_mask(signal.SIG_SETMASK, expected_mask)
    children = []
    real_popen = subprocess.Popen
    pending = exception("synthetic pending handler")
    restores = []
    def launch(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child
    def mask(how, signals):
        if how == signal.SIG_SETMASK:
            restores.append(set(signals))
            if len(restores) == 1:
                assert real_mask(signal.SIG_BLOCK, set()) == expected_mask | {signal.SIGINT, signal.SIGTERM}
                raise pending
            assert children[0].poll() is not None
            assert children[0].stdout.closed
        return real_mask(how, signals)
    monkeypatch.setattr(guarded.subprocess, "Popen", launch)
    monkeypatch.setattr(guarded.signal, "pthread_sigmask", mask)
    try:
        expected_exception = guarded.GuardedRunError if issubclass(exception, Exception) else exception
        with pytest.raises(expected_exception) as error:
            verified(root, work, "import time; time.sleep(30)")
        assert (error.value.__cause__ if issubclass(exception, Exception) else error.value) is pending
        report = error.value.report
        assert restores == [expected_mask, expected_mask]
        assert real_mask(signal.SIG_BLOCK, set()) == expected_mask
        assert signal.getsignal(signal.SIGTERM) == old_term_handler
        assert report.wrapper_reaped
        assert report.launch_state == "acquired"
        assert not report.published
        assert charged(root, report)["state"] == "failed"
    finally:
        real_mask(signal.SIG_SETMASK, original_mask)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
            child.stdout.close()


@pytest.mark.skipif(not hasattr(signal, "pthread_sigmask"), reason="POSIX signal mask")
def test_wrapper_inherits_mask_and_can_unblock_without_preexec(tmp_path):
    root, work = setup(tmp_path)
    script = (
        "import signal; "
        "mask=signal.pthread_sigmask(signal.SIG_BLOCK,set()); "
        "assert {signal.SIGINT,signal.SIGTERM} <= mask; "
        "signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGINT,signal.SIGTERM}); "
        "assert not ({signal.SIGINT,signal.SIGTERM} & signal.pthread_sigmask(signal.SIG_BLOCK,set())); "
        "print('wrapper unblocked')"
    )
    report = verified(root, work, script)
    assert report.state == "complete"
    assert (root / SLOT / "merged.log").read_bytes() == b"wrapper unblocked\n"


@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_launch_inherits_environment_and_work(tmp_path, monkeypatch, explicit_none):
    root, work = setup(tmp_path)
    monkeypatch.setenv("GUARDED_PARENT_MARKER", "inherited")
    kwargs = {"launch_environment": None, "launch_cwd": None} if explicit_none else {}
    report = execute(root, work,
                     "import os; assert os.environ['GUARDED_PARENT_MARKER']=='inherited'; "
                     "assert os.getcwd()==" + repr(str(work)), **kwargs)
    assert report.state == "unverified"
    assert report.fully_enforced is False


def test_exact_hostile_parent_environment_not_passed(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    hostile = {
        "APPTAINER_BIND": "/:/host:rw", "APPTAINER_BINDPATH": "/secret",
        "APPTAINERENV_HOME": "/bad", "SINGULARITY_BIND": "/:/host",
        "SINGULARITYENV_LD_PRELOAD": "/evil.so", "LD_PRELOAD": "/evil.so",
        "LD_LIBRARY_PATH": "/evil", "DYLD_INSERT_LIBRARIES": "/evil.dylib",
        "PYTHONPATH": "/evil", "BASH_ENV": "/evil", "SECRET_TOKEN": "secret",
        "PATH": "/evil", "HOME": "/bad", "TMPDIR": "/bad",
        "APPTAINER_TMPDIR": "/bad", "APPTAINER_CACHEDIR": "/bad",
        "APPTAINER_CONFIGDIR": "/bad",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    expected = {
        "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "HOME": str(work / "home"), "TMPDIR": str(work / "tmp"),
        "APPTAINER_TMPDIR": str(work / "tmp"),
        "APPTAINER_CACHEDIR": str(work / "cache"),
        "APPTAINER_CONFIGDIR": str(work / "config"),
    }
    real_popen = subprocess.Popen
    launches = []
    def launch(*args, **kwargs):
        assert kwargs["env"] == expected
        assert kwargs["env"] is not expected
        assert kwargs["cwd"] == "/"
        launches.append(kwargs)
        return real_popen(*args, **kwargs)
    monkeypatch.setattr(guarded.subprocess, "Popen", launch)
    report = verified(root, work,
                      "import json,os; print(json.dumps([dict(os.environ),os.getcwd()]))",
                      launch_environment=expected, launch_cwd="/", log_limit=2048)
    assert len(launches) == 1
    observed_environment, observed_cwd = json.loads((root / SLOT / "merged.log").read_text())
    if sys.platform == "darwin":
        # CoreFoundation may add this during interpreter startup, after Popen.
        observed_environment.pop("__CF_USER_TEXT_ENCODING", None)
    assert observed_environment == expected
    assert observed_cwd == "/"
    assert all(os.environ[key] == value for key, value in hostile.items())
    assert report.state == "complete"
    assert report.fully_enforced is False
    assert (root / SLOT / "output.art").read_bytes() == b"raw_1.art"
    charged(root, report)


def test_empty_launch_environment_is_honored(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    monkeypatch.setenv("GUARDED_PARENT_MARKER", "must not inherit")
    environment = {}
    real_popen = subprocess.Popen
    launches = []
    def launch(*args, **kwargs):
        assert kwargs["env"] == {}
        assert kwargs["env"] is not environment
        launches.append(kwargs)
        return real_popen(*args, **kwargs)
    monkeypatch.setattr(guarded.subprocess, "Popen", launch)
    report = execute(root, work,
                     "import os; assert 'GUARDED_PARENT_MARKER' not in os.environ",
                     launch_environment=environment)
    assert len(launches) == 1
    assert report.state == "unverified"
    assert report.fully_enforced is False


def test_mutable_launch_environment_copied_before_reservation(tmp_path, monkeypatch):
    root, work = setup(tmp_path)
    environment = UserDict({"GUARDED_VALUE": "original", "EMPTY_VALUE": ""})
    expected = dict(environment)
    original_start = budget.SerializedBudget.start
    def start(self):
        environment["GUARDED_VALUE"] = "changed"
        environment["SECRET_TOKEN"] = "added after validation"
        return original_start(self)
    monkeypatch.setattr(budget.SerializedBudget, "start", start)
    real_popen = subprocess.Popen
    launches = []
    def launch(*args, **kwargs):
        assert type(kwargs["env"]) is dict
        assert kwargs["env"] == expected
        assert kwargs["env"] is not environment
        launches.append(kwargs)
        return real_popen(*args, **kwargs)
    monkeypatch.setattr(guarded.subprocess, "Popen", launch)
    report = execute(root, work,
                     "import os; assert os.environ['GUARDED_VALUE']=='original'; "
                     "assert os.environ['EMPTY_VALUE']==''; "
                     "assert 'SECRET_TOKEN' not in os.environ",
                     launch_environment=environment)
    assert len(launches) == 1
    assert report.fully_enforced is False


@pytest.mark.parametrize("environment", [
    [], [("KEY", "value")], "KEY=value", 42,
    {"": "value"}, {"BAD=KEY": "value"}, {"BAD\0KEY": "value"},
    {1: "value"}, {b"KEY": "value"}, {"KEY": 1}, {"KEY": None},
    {"KEY": b"value"}, {"KEY": "bad\0value"},
])
def test_invalid_launch_environment_rejected_before_reservation(tmp_path, monkeypatch, environment):
    root, work = setup(tmp_path)
    def reserve(*args, **kwargs):
        pytest.fail("invalid environment must not reserve a slot")
    monkeypatch.setattr(guarded, "SerializedBudget", reserve)
    with pytest.raises(ValueError, match="launch_environment"):
        execute(root, work, launch_environment=environment)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("cwd", [
    "", "relative", ".", "..", b"/", 42, "/tmp/../", "/tmp/./", "//",
    "/tmp//bad", "/tmp/", "/tmp/bad\0cwd", "missing", "file",
])
def test_invalid_launch_cwd_rejected_before_reservation(tmp_path, monkeypatch, cwd):
    root, work = setup(tmp_path)
    if cwd == "missing":
        cwd = str(work / "missing")
    elif cwd == "file":
        cwd = work / "job.fcl"
    def reserve(*args, **kwargs):
        pytest.fail("invalid cwd must not reserve a slot")
    monkeypatch.setattr(guarded, "SerializedBudget", reserve)
    with pytest.raises(ValueError, match="launch_cwd"):
        execute(root, work, launch_cwd=cwd)
    assert list(root.iterdir()) == []


def test_launch_cwd_accepts_absolute_path_object(tmp_path):
    root, work = setup(tmp_path)
    report = execute(root, work, "import os; assert os.getcwd()=='/'", launch_cwd=Path("/"))
    assert report.state == "unverified"
    assert report.fully_enforced is False


@pytest.mark.parametrize("environment,cwd", [(None, None), ({}, "/"), ({"KEY": "value"}, "/")])
def test_run_guarded_forwards_launch_options(tmp_path, monkeypatch, environment, cwd):
    root, work = setup(tmp_path)
    command = [sys.executable, "-c", "pass"]
    result = object()
    def execute_and_publish(*args, **kwargs):
        assert args == (root, SLOT, command)
        assert kwargs == {
            "work": work, "parent_timeout": 3,
            "independently_verified_confinement_receipt": RECEIPT,
            "verify_callback": verify, "output_layout": guarded.OutputLayout.REFERENCE,
            "launch_environment": environment, "launch_cwd": cwd,
        }
        assert kwargs["launch_environment"] is environment
        return result
    monkeypatch.setattr(guarded, "execute_and_publish", execute_and_publish)
    assert guarded.run_guarded(
        root, SLOT, command, work=work, parent_timeout=3,
        confinement_receipt=RECEIPT, verify_callback=verify,
        launch_environment=environment, launch_cwd=cwd) is result


@pytest.mark.parametrize("layout", list(guarded.OutputLayout))
@pytest.mark.parametrize("scratch", [False, True])
def test_external_verification_inventory_publication_order(tmp_path, monkeypatch, layout, scratch):
    root, work = setup(tmp_path)
    if layout is guarded.OutputLayout.NOG4_FIXTURE:
        (work / "raw_1.art").rename(work / "flash_reference_fixture_1.art")
    if scratch:
        for name in ("tmp", "home", "shm"):
            (work / name).mkdir()
    events = []
    real_inventory = guarded.validate_inventory
    real_publish = budget.SerializedBudget.publish
    def external_verify(*args):
        assert verify(*args) is True
        events.append("externalverify")
        return True
    def census(path, selected_layout):
        assert events == ["externalverify"]
        assert path == work
        assert selected_layout is inventory.OutputLayout[layout.name]
        measured = real_inventory(path, selected_layout)
        assert measured.verified is False
        assert measured.empty_directories == (("tmp", "home", "shm") if scratch else ())
        events.append("inventory")
        return measured
    def publish(self, source, destination, **kwargs):
        assert events[:2] == ["externalverify", "inventory"]
        events.append("publish:" + destination)
        return real_publish(self, source, destination, **kwargs)
    monkeypatch.setattr(guarded, "validate_inventory", census)
    monkeypatch.setattr(budget.SerializedBudget, "publish", publish)
    report = verified(root, work, output_layout=layout, verify_callback=external_verify)
    assert events == ["externalverify", "inventory", "publish:output.art",
                      "publish:output.root", "publish:config.fcl"]
    assert report.state == "complete"
    assert report.publication_verified_by_caller is True
    assert report.fully_enforced is False
    assert charged(root, report)["state"] == "complete"


@pytest.mark.parametrize("case", [
    "raw_2.art", "export_1.root", ".hidden.tmp", "RootOutput-unique.tmp",
    "manyfiles", "subdir", "tmp", "home", "shm",
])
def test_strict_inventory_refuses_before_any_publication(tmp_path, monkeypatch, case):
    root, work = setup(tmp_path)
    if case == "manyfiles":
        for index in range(200):
            (work / f"raw_{index + 2}.art").touch()
    elif case == "subdir":
        (work / case).mkdir()
    elif case in ("tmp", "home", "shm"):
        (work / case).mkdir()
        (work / case / ".residual").touch()
    else:
        (work / case).touch()
    before = sorted(str(path.relative_to(work)) for path in work.rglob("*"))
    publications = []
    real_publish = budget.SerializedBudget.publish
    def publish(self, *args, **kwargs):
        publications.append(args)
        return real_publish(self, *args, **kwargs)
    monkeypatch.setattr(budget.SerializedBudget, "publish", publish)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work)
    report = error.value.report
    assert isinstance(error.value.__cause__, inventory.InventoryError)
    assert publications == []
    assert report.published == ()
    assert report.state == "failed"
    assert report.wrapper_reaped
    assert report.publication_verified_by_caller is True
    assert report.fully_enforced is False
    assert sorted(str(path.relative_to(work)) for path in work.rglob("*")) == before
    assert not {"output.art", "output.root", "config.fcl", "manifest.json"} & {
        path.name for path in (root / SLOT).iterdir()}
    assert (root / SLOT / "reservation.json").exists()
    assert charged(root, report)["state"] == "failed"
    with pytest.raises(FileExistsError):
        verified(root, work)


@pytest.mark.parametrize("exception", [inventory.InventoryError, OSError])
def test_inventory_exception_retains_charged_failed_slot(tmp_path, monkeypatch, exception):
    root, work = setup(tmp_path)
    events = []
    failure = exception("injected inventory failure")
    def external_verify(*args):
        assert verify(*args) is True
        events.append("externalverify")
        return True
    def census(path, layout):
        assert path == work
        assert layout is inventory.OutputLayout.REFERENCE
        events.append("inventory")
        raise failure
    def publish(*args, **kwargs):
        events.append("publish")
    monkeypatch.setattr(guarded, "validate_inventory", census)
    monkeypatch.setattr(budget.SerializedBudget, "publish", publish)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, verify_callback=external_verify)
    report = error.value.report
    assert error.value.__cause__ is failure
    assert events == ["externalverify", "inventory"]
    assert "injected inventory failure" in report.failure
    assert report.published == ()
    assert report.wrapper_reaped
    assert report.fully_enforced is False
    assert report.state == "failed"
    assert charged(root, report)["state"] == "failed"
    assert (root / SLOT / "reservation.json").exists()
    assert {path.name for path in work.iterdir()} == {"raw_1.art", "export.root", "job.fcl"}
    assert not {"output.art", "output.root", "config.fcl", "manifest.json"} & {
        path.name for path in (root / SLOT).iterdir()}
    with pytest.raises(FileExistsError):
        verified(root, work)


@pytest.mark.parametrize("decision", [False, None, 1])
def test_refused_external_verification_never_inventories_or_publishes(tmp_path, monkeypatch, decision):
    root, work = setup(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail("inventory/publication requires external callback to return True")
    monkeypatch.setattr(guarded, "validate_inventory", unexpected)
    monkeypatch.setattr(budget.SerializedBudget, "publish", unexpected)
    with pytest.raises(guarded.GuardedRunError) as error:
        verified(root, work, verify_callback=lambda *args: decision)
    report = error.value.report
    assert report.publication_verified_by_caller is False
    assert report.fully_enforced is False
    assert report.published == ()
    assert "caller verification refused" in report.failure
    assert charged(root, report)["state"] == "failed"
