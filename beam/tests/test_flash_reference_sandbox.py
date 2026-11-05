"""Portable construction/guard tests; no bwrap, mount, CVMFS or payload is run."""

import ctypes
from dataclasses import FrozenInstanceError
import errno
import io
import shutil
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from beamfastsim import flash_reference_sandbox as sandbox


@pytest.fixture(autouse=True)
def runtime_dirs(monkeypatch):
    existing = set(sandbox._RUNTIME_PREFIXES)
    isdir = os.path.isdir
    monkeypatch.setattr(sandbox.os.path, "isdir",
                        lambda path: path in existing if path in sandbox._RUNTIME_PREFIXES
                        else isdir(path))
    return existing


@pytest.fixture
def node(tmp_path, monkeypatch):
    # macOS /tmp is a symlink, so model the node-local root using a real directory.
    root = str(tmp_path.resolve())
    monkeypatch.setattr(sandbox, "_LOCAL_ROOTS", (root,))
    return root


@pytest.fixture
def work(node):
    with sandbox.private_work(node) as path:
        yield path


def _values(argv, option, count):
    return [tuple(argv[index + 1:index + 1 + count])
            for index, value in enumerate(argv) if value == option]


def test_fixed_mounts_and_namespace_options(work):
    plan = sandbox.build_command(work, ["/usr/bin/true", "--ro-bind", "/", "/bad"])
    argv = plan.argv
    assert argv[0] == "bwrap"
    assert sandbox._RUNTIME_PREFIXES == (
        "/usr", "/etc", "/opt", "/var", "/cvmfs", "/home", "/scratch", "/sfs"
    )
    assert sandbox._REQUIRED_RUNTIME_PREFIXES == ("/usr", "/etc", "/cvmfs")
    for flag in ("--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                 "--die-with-parent", "--new-session"):
        assert flag in argv
    assert "--unshare-net" not in argv and "--share-net" not in argv
    assert _values(argv, "--cap-drop", 1) == [("ALL",)]
    outer = argv[:argv.index("--")]
    assert _values(outer, "--ro-bind", 2) == [
        *((prefix, prefix) for prefix in sandbox._RUNTIME_PREFIXES),
        (work + "/.empty", "/run"),
        (work + "/.empty", "/var/run"), (work + "/.empty", "/sys"),
    ]
    assert _values(outer, "--bind", 2) == [
        (work, "/mnt/work"), (work + "/tmp", "/tmp"),
        (work + "/tmp", "/var/tmp"), (work + "/shm", "/dev/shm"),
    ]
    assert _values(outer, "--proc", 1) == [("/proc",)]
    assert _values(outer, "--dev", 1) == [("/dev",)]
    assert _values(outer, "--chdir", 1) == [("/mnt/work",)]
    assert "/dev/fuse" not in argv
    assert "--dev-bind" not in argv
    assert _values(outer, "--tmpfs", 1) == [("/",), ("/mnt",)]
    assert _values(outer, "--symlink", 2) == [
        ("usr/" + name, "/" + name) for name in ("bin", "lib", "lib64", "sbin")
    ]
    assert ("/", "/") not in _values(outer, "--ro-bind", 2)
    assert "--ro-bind-try" not in outer
    assert all(source != "/" for source, target in _values(outer, "--bind", 2))
    assert all(target not in ("/usr", "/etc", "/cvmfs", "/bin", "/lib", "/lib64", "/sbin")
               for source, target in _values(outer, "--bind", 2))
    assert _values(outer, "--dir", 1) == [("/mnt/work",)]
    assert argv[-4:] == ("/usr/bin/true", "--ro-bind", "/", "/bad")
    assert plan.verified is False
    with pytest.raises(FrozenInstanceError):
        plan.verified = True


def test_internal_entry_and_private_environment(work):
    argv = sandbox.build_command(work, ["mu2e", "-c", "job.fcl"], file_size_limit=99).argv
    env = dict(_values(argv, "--setenv", 2))
    assert env["HOME"] == "/mnt/work/home"
    for name in ("TMPDIR", "TMP", "TEMP", "SQLITE_TMPDIR", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR"):
        assert env[name] == "/mnt/work/tmp"
    tail = argv[argv.index("--") + 1:]
    assert tail[1] == "-I"
    assert tail[2] == str(Path(sandbox.__file__).resolve())
    assert tail[3:] == ("--inside", "--file-size-limit", "99", "--", "mu2e", "-c", "job.fcl")


def test_private_work_is_fresh_and_removed(node):
    with sandbox.private_work(node) as work:
        assert stat.S_IMODE(os.stat(work).st_mode) == 0o700
        assert set(os.listdir(work)) == {"tmp", "home", "shm", ".empty"}
        for name in sandbox._LAYOUT:
            assert stat.S_IMODE(os.stat(work + "/" + name).st_mode) == 0o700
        first = work
    assert not os.path.exists(first)
    with sandbox.private_work(node) as second:
        assert second != first


@pytest.mark.parametrize("path", ["/etc", "/tmp-other/test", "/locality/test", "/localscratchy/x",
                                  "relative", "", b"/tmp/bytes", "/tmp/../etc", "/tmp/./a",
                                  "/tmp//a", "/tmp/a/", "/tmp/a\0b", "/"])
def test_nonlocal_or_noncanonical_paths_refused(path):
    with pytest.raises((ValueError, TypeError)):
        sandbox.build_command(path, ["true"])


def test_scratch_roots_not_prefix_matches():
    for root in ("/tmp", "/local", "/localscratch"):
        assert root in sandbox._LOCAL_ROOTS
    with pytest.raises(ValueError, match="under"):
        sandbox._local_directory("/tmp-malicious/mnt/work")


def test_symlink_and_symlink_parent_refused(node, work):
    link = Path(node) / "alias"
    link.symlink_to(work, target_is_directory=True)
    for path in (str(link), str(link / "tmp")):
        with pytest.raises(ValueError, match="symlinks"):
            sandbox.build_command(path, ["true"])


def test_not_private_or_not_fresh_refused(work):
    os.chmod(work, 0o755)
    with pytest.raises(ValueError, match="0700"):
        sandbox.build_command(work, ["true"])
    os.chmod(work, 0o700)
    (Path(work) / "unexpected").write_text("not fresh")
    with pytest.raises(ValueError, match="layout"):
        sandbox.build_command(work, ["true"])
    (Path(work) / "unexpected").unlink()
    (Path(work) / "tmp" / "stale").write_text("old")
    with pytest.raises(ValueError, match="empty"):
        sandbox.build_command(work, ["true"])


def test_scratch_child_cannot_be_symlink(work):
    target = Path(work) / "home"
    target.rmdir()
    target.symlink_to(Path(work) / "tmp", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        sandbox.build_command(work, ["true"])


@pytest.mark.parametrize("command", [[], "true", ["--bind"], [""], ["../escape"],
                                     ["./relative"], ["/bin/../bin/true"], ["."],
                                     [".."], ["true", b"bytes"], ["true", "bad\0arg"]])
def test_payload_input_refused(work, command):
    with pytest.raises(ValueError):
        sandbox.build_command(work, command)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "123", None, 2**63])
def test_invalid_file_limit(work, value):
    with pytest.raises(ValueError, match="positive"):
        sandbox.build_command(work, ["true"], file_size_limit=value)


def test_no_arbitrary_bind_or_bwrap_options_api(work):
    with pytest.raises(TypeError):
        sandbox.build_command(work, ["true"], options=["--bind", "/", "/"])
    with pytest.raises(TypeError):
        sandbox.build_command(work, ["true"], binds=[("/persistent", "/mnt/work")])


class FakeFunction:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


class FakeSeccomp:
    def __init__(self, *, unknown=None, load_result=0, rule_result=0):
        self.rules = []
        self.released = []
        self.names = {}
        self.seccomp_init = FakeFunction(lambda action: 42)
        self.seccomp_release = FakeFunction(self.released.append)
        self.seccomp_load = FakeFunction(lambda context: load_result)

        def resolve(name):
            name = name.decode("ascii")
            if name == unknown:
                return -1
            number = len(self.names)
            self.names[number] = name
            return number

        def rule(context, action, number, count, pointer):
            comparison = None if not count else (
                pointer._obj.arg, pointer._obj.op, pointer._obj.datum_a, pointer._obj.datum_b)
            self.rules.append((self.names[number], action, comparison))
            return rule_result

        self.seccomp_syscall_resolve_name = FakeFunction(resolve)
        self.seccomp_rule_add_array = FakeFunction(rule)


def test_seccomp_uses_library_resolution_and_allows_normal_clone(monkeypatch):
    lib = FakeSeccomp()
    monkeypatch.setattr(sandbox, "_seccomp_library", lambda: lib)
    sandbox._install_seccomp()
    assert lib.released == [42]
    for name in sandbox._DENIED_SYSCALLS:
        assert (name, 0x00050000 | errno.EPERM, None) in lib.rules
    assert ("capset", 0x00050000 | errno.EPERM, None) in lib.rules
    assert ("prctl", 0x00050000 | errno.EPERM, (0, 7, 0xFFFFFFFF, 47)) in lib.rules
    assert ("clone3", 0x00050000 | errno.ENOSYS, None) in lib.rules
    clone_rules = [rule for rule in lib.rules if rule[0] == "clone"]
    assert len(clone_rules) == len(sandbox._NAMESPACE_FLAGS)
    for rule, flag in zip(clone_rules, sandbox._NAMESPACE_FLAGS):
        assert rule == ("clone", 0x00050000 | errno.EPERM, (0, 7, flag, flag))
        assert 0x10000 & flag == 0


@pytest.mark.parametrize("kwargs", [{"unknown": "clone3"}, {"unknown": "capset"},
                                    {"unknown": "prctl"}, {"load_result": -1},
                                    {"rule_result": -1}])
def test_seccomp_failures_refuse_and_release(monkeypatch, kwargs):
    lib = FakeSeccomp(**kwargs)
    monkeypatch.setattr(sandbox, "_seccomp_library", lambda: lib)
    with pytest.raises(RuntimeError):
        sandbox._install_seccomp()
    assert lib.released == [42]


def test_library_feature_detection_never_runs_ldd(monkeypatch):
    calls = []

    def missing(name, **kwargs):
        calls.append(name)
        raise OSError("missing")

    monkeypatch.setattr(sandbox.ctypes, "CDLL", missing)
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *a, **k: pytest.fail("no subprocess probe"))
    with pytest.raises(RuntimeError, match="existing libseccomp"):
        sandbox._seccomp_library()
    assert calls == ["libseccomp.so.2", "libseccomp.so"]


def test_inside_sets_limits_and_filters_before_exec(monkeypatch):
    import resource
    events = []
    monkeypatch.setattr(sandbox, "_linux_only", lambda: None)
    monkeypatch.setattr(sandbox.os, "getcwd", lambda: "/mnt/work")
    monkeypatch.setenv("HOME", "/mnt/work/home")
    monkeypatch.setenv("TMPDIR", "/mnt/work/tmp")
    monkeypatch.setattr(resource, "setrlimit", lambda kind, value: events.append((kind, value)))
    monkeypatch.setattr(sandbox, "_no_privileges", lambda: events.append("privileges"))
    monkeypatch.setattr(sandbox, "_install_seccomp", lambda: events.append("seccomp"))
    monkeypatch.setattr(sandbox.os, "execvpe", lambda *a: events.append("exec"))
    sandbox._inside(["true"], 42)
    assert events == [(resource.RLIMIT_CORE, (0, 0)), (resource.RLIMIT_FSIZE, (42, 42)),
                      "privileges", "seccomp", "exec"]


def test_inside_refuses_failed_filter_before_exec(monkeypatch):
    monkeypatch.setattr(sandbox, "_linux_only", lambda: None)
    monkeypatch.setattr(sandbox.os, "getcwd", lambda: "/mnt/work")
    monkeypatch.setenv("HOME", "/mnt/work/home")
    monkeypatch.setenv("TMPDIR", "/mnt/work/tmp")
    import resource
    monkeypatch.setattr(resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(sandbox, "_no_privileges", lambda: None)
    monkeypatch.setattr(sandbox, "_install_seccomp", lambda: (_ for _ in ()).throw(RuntimeError("filter")))
    monkeypatch.setattr(sandbox.os, "execvpe", lambda *a: pytest.fail("must not exec"))
    with pytest.raises(RuntimeError, match="filter"):
        sandbox._inside(["true"], 42)


def test_linux_guard_and_cli_refusal(monkeypatch, capsys):
    monkeypatch.setattr(sandbox.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux"):
        sandbox.run_payload(["true"])
    with pytest.raises(SystemExit) as result:
        sandbox.main(["--", "true"])
    assert result.value.code == 125
    assert "sandbox refused" in capsys.readouterr().err


def test_result_never_claims_verification():
    assert sandbox.SandboxResult(0).verified is False


@pytest.mark.parametrize("failure", [None, "effective", "permitted", "inheritable",
                                    "capget", "ambient", "ambient-unsupported",
                                    "ambient-error", "set-nnp", "get-nnp"])
def test_capabilities_and_no_new_privileges_are_checked(monkeypatch, failure):
    calls = []

    def capget(header, caps):
        calls.append("capget")
        if failure in ("effective", "permitted", "inheritable"):
            setattr(caps[1], failure, 1)
        return -1 if failure == "capget" else 0

    def prctl(option, arg, *rest):
        calls.append((option, arg))
        if option == 47:
            assert arg == 1
            if failure == "ambient":
                return 1
            if failure == "ambient-unsupported" or rest[0] > 40:
                ctypes.set_errno(errno.EINVAL)
                return -1
            if failure == "ambient-error":
                ctypes.set_errno(errno.EPERM)
                return -1
            return 0
        if option == 38:
            return -1 if failure == "set-nnp" else 0
        if option == 39:
            return 0 if failure == "get-nnp" else 1
        pytest.fail("unexpected prctl (bounding set need not be empty)")

    libc = SimpleNamespace(capget=FakeFunction(capget), prctl=FakeFunction(prctl))
    monkeypatch.setattr(sandbox.ctypes, "CDLL", lambda *a, **kw: libc)
    if failure:
        with pytest.raises(RuntimeError):
            sandbox._no_privileges()
    else:
        sandbox._no_privileges()
        assert calls[:3] == [(38, 1), (39, 0), "capget"]
        assert calls.count((47, 1)) == 42


def test_trusted_outer_closes_fds_and_uses_pipe(node, monkeypatch):
    import io
    calls = []
    child = SimpleNamespace(stdout=io.BytesIO(b"payload log\n"), wait=lambda: 7)

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return child

    output = io.BytesIO()
    monkeypatch.setattr(sandbox, "_linux_only", lambda: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    monkeypatch.setattr(sandbox.sys, "stdout", SimpleNamespace(buffer=output))
    result = sandbox.run_payload(["true"], node_tmp=node)
    assert result.returncode == 7 and result.verified is False
    assert output.getvalue() == b"payload log\n"
    argv, options = calls[0]
    assert options == dict(stdin=sandbox.subprocess.DEVNULL, stdout=sandbox.subprocess.PIPE,
                           stderr=sandbox.subprocess.STDOUT, close_fds=True, cwd="/")
    work = _values(argv, "--bind", 2)[0][0]
    assert not Path(work).exists()


@pytest.mark.parametrize("source,error", [
    ("write", BrokenPipeError("closed output")),
    ("flush", OSError("log error")),
    ("read", OSError("quiet child read error")),
    ("read", KeyboardInterrupt()),
    ("wait", KeyboardInterrupt()),
    ("close", OSError("stream close error")),
])
def test_output_errors_and_interruptions_stop_child_before_removing_work(
        node, monkeypatch, source, error):
    events = []
    state = {}

    def event(name):
        assert Path(state["work"]).exists()
        events.append(name)

    class Stream:
        def __init__(self, data):
            self.stream = io.BytesIO(data)

        def read(self, size):
            if source == "read":
                raise error
            return self.stream.read(size)

        def close(self):
            event("close")
            if source == "close":
                raise error
            self.stream.close()

    def wait(timeout=None):
        event(("wait", timeout))
        if timeout is None and source == "wait":
            raise error
        return 0

    child = SimpleNamespace(stdout=Stream(b"log"), wait=wait,
                            terminate=lambda: event("terminate"),
                            kill=lambda: pytest.fail("no escalation needed"))

    def popen(argv, **kwargs):
        state["work"] = _values(argv, "--bind", 2)[0][0]
        return child

    class Output:
        def write(self, chunk):
            if source == "write":
                raise error

        def flush(self):
            if source == "flush":
                raise error

    monkeypatch.setattr(sandbox, "_linux_only", lambda: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    monkeypatch.setattr(sandbox.sys, "stdout", SimpleNamespace(buffer=Output()))
    with pytest.raises(type(error)) as caught:
        sandbox.run_payload(["true"], node_tmp=node)
    assert caught.value is error
    assert "terminate" in events
    assert ("wait", sandbox._CHILD_STOP_TIMEOUT) in events
    assert not Path(state["work"]).exists()


@pytest.mark.parametrize("signal_error", [False, True])
@pytest.mark.parametrize("reap", ["success", "timeout", "interrupt"])
def test_failed_termination_escalates_and_retains_scratch_if_unreaped(
        node, monkeypatch, signal_error, reap):
    state = {}
    events = []

    def stop(name):
        assert Path(state["work"]).exists()
        events.append(name)
        if signal_error:
            raise OSError("cannot signal")

    def wait(timeout):
        assert Path(state["work"]).exists()
        assert timeout == sandbox._CHILD_STOP_TIMEOUT
        events.append("wait")
        if events.count("wait") == 1 or reap == "timeout":
            raise sandbox.subprocess.TimeoutExpired("bwrap", timeout)
        if reap == "interrupt":
            raise KeyboardInterrupt()
        return -9

    class QuietStream:
        def read(self, size):
            raise BrokenPipeError("reader failed with quiet child")

        def close(self):
            # Must not mask the scratch-retention marker.
            raise OSError("close also failed")

    child = SimpleNamespace(stdout=QuietStream(), wait=wait,
                            terminate=lambda: stop("terminate"), kill=lambda: stop("kill"))

    def popen(argv, **kwargs):
        state["work"] = _values(argv, "--bind", 2)[0][0]
        return child

    monkeypatch.setattr(sandbox, "_linux_only", lambda: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    error = BrokenPipeError if reap == "success" else sandbox._UnreapedChild
    try:
        with pytest.raises(error):
            sandbox.run_payload(["true"], node_tmp=node)
        assert events == ["terminate", "wait", "kill", "wait"]
        assert Path(state["work"]).exists() == (reap != "success")
    finally:
        if Path(state["work"]).exists():
            shutil.rmtree(state["work"])


@pytest.mark.parametrize("error", [KeyboardInterrupt(), OSError("spawn failed")])
def test_spawn_exception_with_unknown_child_state_retains_work(node, monkeypatch, error):
    state = {}

    def popen(argv, **kwargs):
        state["work"] = _values(argv, "--bind", 2)[0][0]
        raise error

    monkeypatch.setattr(sandbox, "_linux_only", lambda: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    try:
        with pytest.raises(sandbox._UnreapedChild, match="scratch retained at"):
            sandbox.run_payload(["true"], node_tmp=node)
        assert Path(state["work"]).is_dir()
    finally:
        shutil.rmtree(state["work"])


def test_portable_builder_with_no_runtime_prefixes(work, runtime_dirs):
    runtime_dirs.clear()
    argv = sandbox.build_command(work, ["true"]).argv
    assert _values(argv, "--ro-bind", 2) == [
        (work + "/.empty", target) for target in ("/run", "/var/run", "/sys")
    ]
    assert "--ro-bind-try" not in argv
    assert sandbox.build_command(work, ["true"]).verified is False


@pytest.mark.parametrize("missing", ["/opt", "/var", "/home", "/scratch", "/sfs"])
def test_missing_optional_prefix_is_not_bound(work, runtime_dirs, missing):
    runtime_dirs.remove(missing)
    argv = sandbox.build_command(work, ["true"]).argv
    assert (missing, missing) not in _values(argv, "--ro-bind", 2)
    for prefix in runtime_dirs:
        assert (prefix, prefix) in _values(argv, "--ro-bind", 2)


@pytest.mark.parametrize("missing", ["/usr", "/etc", "/cvmfs"])
def test_execution_requires_runtime_without_fallback(node, monkeypatch, runtime_dirs, missing):
    runtime_dirs.remove(missing)
    monkeypatch.setattr(sandbox, "_linux_only", lambda: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    monkeypatch.setattr(sandbox, "private_work", lambda *a: pytest.fail("must not allocate"))
    with pytest.raises(RuntimeError, match="required runtime directory is missing: " + missing):
        sandbox.run_payload(["true"], node_tmp=node)
