"""Mock-only state tests; never Linux runtime/race/watchdog evidence."""

import builtins
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from io import BytesIO
import signal
from types import SimpleNamespace as NS

import pytest

from beamfastsim import flash_reference_owned_children as tree

Linux = tree._LinuxBackend
ROOT, OTHER, SELF = 100, 101, 99


@pytest.fixture(autouse=True)
def safe(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real fork/signal/Linux syscall forbidden")

    for name in ("fork", "kill", "killpg", "waitpid", "waitid", "pipe", "pipe2"):
        monkeypatch.setattr(tree.os, name, forbidden, raising=False)
    monkeypatch.setattr(tree.signal, "signal", forbidden)
    monkeypatch.setattr(tree, "_LinuxBackend", forbidden)
    for name, value in dict(P_ALL=0, P_PID=1, WEXITED=4, WNOHANG=1,
                            WNOWAIT=0x1000000, CLD_EXITED=1, CLD_KILLED=2,
                            CLD_DUMPED=3, CLD_STOPPED=5).items():
        monkeypatch.setattr(tree.os, name, value, raising=False)
    state = NS(blocked=set(), restores=0, interrupt=None, entry_interrupt=None)

    def mask(how, values):
        old = state.blocked.copy()
        if how == signal.SIG_BLOCK:
            state.blocked |= set(values)
            if values and state.entry_interrupt:
                callback, state.entry_interrupt = state.entry_interrupt, None
                callback()
        elif how == signal.SIG_UNBLOCK:
            state.blocked -= set(values)
        else:
            state.blocked = set(values)
            state.restores += 1
            if state.interrupt:
                callback, state.interrupt = state.interrupt, None
                callback()
        return old

    monkeypatch.setattr(tree.signal, "pthread_sigmask", mask)
    return state


class Fake:
    def __init__(self, mode="normal"):
        self.now, self.mode = 0.0, mode
        self.events, self.sent, self.closed = [], [], []
        self.children_set, self.pending = set(), []
        self.live, self.released, self.generation = set(), False, 0
        self.frozen, self.unknown = False, False

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert 0 <= seconds <= 0.01
        if not self.frozen:
            self.now += seconds

    def prepare(self):
        self.events.append("prepare")

    def pipe(self):
        self.events.append("pipe")
        self.live.update((20, 21))
        return 20, 21

    def fork(self):
        self.events.append("fork")
        self.children_set.add(ROOT)
        return ROOT

    def close(self, fd):
        assert fd in self.live
        self.live.remove(fd)
        self.closed.append(fd)
        if fd == 21 and not self.released:
            self.pending.append((ROOT, 125 << 8))

    def member(self, pid, sig=None):
        assert pid in self.children_set
        self.events.append(("member", pid, sig, self.generation))
        if any(p == pid for p, _ in self.pending):
            return False
        if sig is not None:
            self.sent.append((pid, sig, self.generation))
            if self.mode != "unkillable" and (self.mode != "stubborn" or sig == signal.SIGKILL):
                self.pending.append((pid, int(sig)))
        return True

    def release(self, fd):
        assert fd == 21
        assert self.events[-1] == ("member", ROOT, None, 0)
        self.events.append("release")
        self.released = True
        if self.mode in ("normal", "orphan", "nonzero", "reuse", "flood"):
            self.pending.append((ROOT, 0 if self.mode != "nonzero" else 7 << 8))
        if self.mode in ("orphan", "reuse"):
            self.children_set.add(OTHER)
        if self.mode == "flood":
            self.children_set.update(range(200, 300))
            self.pending.extend((pid, 0) for pid in range(200, 300))

    def wait(self):
        self.events.append("wait")
        if self.pending:
            pid, status = self.pending.pop(0)
            self.children_set.remove(pid)
            if self.mode == "reuse" and pid == OTHER and not self.generation:
                self.children_set.add(pid)
                self.generation = 1
            return pid, status
        if self.children_set or self.unknown:
            return 0, 0
        raise ChildProcessError()

    def children(self):
        self.events.append("children")
        return sorted(self.children_set)


def run(b, **kwargs):
    options = dict(timeout=0.5, external_watchdog_present=True, _backend=b)
    options.update(kwargs)
    report = tree.run_owned_tree(["/fixed/program", "arg"], {"KEY": "value"}, "/work", **options)
    assert not b.live
    assert len(b.closed) == len(set(b.closed))
    return report


def test_normal_gate_immutable_and_never_certificate():
    b = Fake()
    r = run(b)
    assert r.success and r.ancestry_drained and r.attempted and r.released
    assert (r.root_pid, r.root_status, r.reaps, r.signals) == (ROOT, 0, 1, 0)
    assert r.error is None and r.retain_work and not r.verified and not r.certificate
    assert b.events[:5] == ["prepare", "pipe", "fork", ("member", ROOT, None, 0), "release"]
    assert b.events.count("fork") == 1
    with pytest.raises(FrozenInstanceError):
        r.success = False


@pytest.mark.parametrize("mode", ["orphan", "nonzero", "linger", "stubborn", "reuse"])
def test_failure_still_drains_and_preserves_original_status(mode):
    b = Fake(mode)
    r = run(b)
    assert not r.success and r.ancestry_drained and r.error
    if mode == "stubborn":
        assert [s for _, s, _ in b.sent] == [signal.SIGTERM, signal.SIGKILL]
        assert b.now >= 0.6
    if mode == "reuse":
        assert b.sent == [(OTHER, signal.SIGTERM, 0), (OTHER, signal.SIGTERM, 1)]
        assert r.root_status == 0 and r.reaps == 3


def test_root_pid_reuse_cannot_overwrite_first_terminal_status():
    b = Fake("nonzero")
    wait = b.wait
    reused = False

    def adopt():
        nonlocal reused
        result = wait()
        if result[0] == ROOT and not reused:
            reused = True
            b.children_set.add(ROOT)
            b.pending.append((ROOT, 0))
        return result

    b.wait = adopt
    r = run(b)
    assert r.root_status == 7 << 8 and r.reaps == 2 and not r.success


@pytest.mark.parametrize("mode", ["linger", "unkillable"])
def test_empty_listing_never_proves_drained_root_fallback_fresh(mode):
    b = Fake(mode)
    b.children = lambda: []
    r = run(b)
    assert r.signals and not r.success
    assert r.ancestry_drained is (mode == "linger")
    assert b.now <= 1.01


def test_zero_wait_and_empty_proc_after_root_not_echild():
    b = Fake()
    b.unknown = True
    r = run(b)
    assert r.root_status == 0 and not r.ancestry_drained and not r.success


@pytest.mark.parametrize("which", ["prepare", "pipe", "fork", "member", "release", "wait", "children"])
def test_errors_fail_closed(which):
    b = Fake("orphan")

    def bad(*args):
        raise OSError("injected " + which)

    setattr(b, which, bad)
    r = run(b)
    assert not r.success and r.error and r.retain_work
    if which in ("prepare", "pipe"):
        assert not r.attempted
    if which in ("prepare", "pipe", "fork", "member", "release"):
        assert not r.released


def test_failed_close_spent_once_but_not_drain_evidence():
    b = Fake()
    close = b.close

    def bad(fd):
        close(fd)
        if fd == 21:
            raise InterruptedError("close outcome uncertain")

    b.close = bad
    r = run(b)
    assert not r.success and not r.ancestry_drained and len(b.closed) == 2


def test_mask_restore_interrupt_keeps_fork_ownership(safe):
    b = Fake("linger")
    fork = b.fork

    def interrupt():
        raise KeyboardInterrupt("pending")

    def owned_fork():
        result = fork()
        safe.interrupt = interrupt
        return result

    b.fork = owned_fork
    r = run(b)
    assert r.root_pid == ROOT and r.attempted and not r.released
    assert r.ancestry_drained and not r.success and "KeyboardInterrupt" in r.error
    assert safe.blocked == set()


def test_reap_mask_restore_interrupt_preserves_status(safe):
    b = Fake()
    wait = b.wait

    def interrupt():
        raise KeyboardInterrupt()

    def owned_wait():
        result = wait()
        if result[0]:
            safe.interrupt = interrupt
        return result

    b.wait = owned_wait
    r = run(b)
    assert r.root_status == 0 and r.reaps == 1 and r.ancestry_drained
    assert not r.success and "KeyboardInterrupt" in r.error


def test_interrupt_retries_bounded_sticky_failure(monkeypatch):
    monkeypatch.setattr(tree, "MAX_OPS", 40)
    b = Fake("orphan")
    b.frozen = True

    def interrupted():
        raise InterruptedError("cleanup")

    b.children = interrupted
    r = run(b)
    assert not r.success and not r.ancestry_drained and "InterruptedError" in r.error
    assert b.events.count("wait") < 50


@pytest.mark.parametrize("bound", ["MAX_REAPS", "MAX_SIGNALS", "MAX_CHILDREN", "MAX_OPS"])
def test_total_bounds_and_frozen_clock(monkeypatch, bound):
    monkeypatch.setattr(tree, bound, 1 if bound != "MAX_OPS" else 30)
    b = Fake("flood" if bound == "MAX_REAPS" else "orphan")
    if bound == "MAX_SIGNALS":
        b.mode = "reuse"
    if bound == "MAX_CHILDREN":
        b.children = lambda: [OTHER, OTHER + 1]
    if bound == "MAX_OPS":
        b = Fake("unkillable")
        b.frozen = True
    r = run(b)
    assert not r.success and not r.ancestry_drained and "bound" in r.error
    assert r.signals <= tree.MAX_SIGNALS and r.reaps <= tree.MAX_REAPS


@pytest.mark.parametrize("kwargs", [dict(timeout=x) for x in (True, 0, -1, 601, float("nan"), float("inf"))]
                         + [dict(external_watchdog_present=x) for x in (False, 1, None)])
def test_invalid_configuration_no_setup(kwargs):
    b = Fake()
    with pytest.raises(ValueError):
        run(b, **kwargs)
    assert not b.events


@pytest.mark.parametrize("argv,env,cwd", [([], {}, "/"), (["relative"], {}, "/"),
    (["/x", "\0"], {}, "/"), (["/x"], {"bad=key": "v"}, "/"),
    (["/x"], {"K": 1}, "/"), (["/x"], {}, "relative")])
def test_invalid_launch(argv, env, cwd):
    b = Fake()
    with pytest.raises(ValueError):
        tree.run_owned_tree(argv, env, cwd, timeout=1, external_watchdog_present=True, _backend=b)
    assert not b.events


def test_waitpid_ptrace_stop_refused_never_root_status():
    b = Fake()
    b.wait = lambda: (ROOT, (signal.SIGSTOP << 8) | 0x7f)
    r = run(b)
    assert r.root_status is None and not r.ancestry_drained and not r.success
    assert r.reaps == 0 and "ptrace" in r.error and not b.sent


@pytest.fixture
def native(monkeypatch):
    b = Linux.__new__(Linux)
    monkeypatch.setattr(tree.os, "getpid", lambda: SELF)
    return b


@pytest.mark.parametrize("event", [None, NS(si_pid=0, si_code=0)])
def test_exit_between_lease_and_signal_pins_zombie(native, monkeypatch, safe, event):
    calls = []
    zombie = False

    def status(pid):
        assert safe.blocked == {signal.SIGINT, signal.SIGTERM}
        calls.append("parent")
        return SELF

    def waitid(kind, pid, flags):
        nonlocal zombie
        assert (kind, pid, flags) == (tree.os.P_PID, ROOT,
            tree.os.WEXITED | tree.os.WNOHANG | tree.os.WNOWAIT | tree.WALL)
        calls.append("waitid")
        zombie = True
        return event

    def kill(pid, sig):
        assert zombie and calls == ["parent", "waitid"]
        assert safe.blocked == {signal.SIGINT, signal.SIGTERM}
        assert (pid, sig) == (ROOT, signal.SIGTERM)
        calls.append("kill-zombie-pinned")

    native.status = status
    monkeypatch.setattr(tree.os, "waitid", waitid)
    monkeypatch.setattr(tree.os, "kill", kill)
    assert native.member(ROOT, signal.SIGTERM) is True
    assert calls[-1] == "kill-zombie-pinned" and not safe.blocked


@pytest.mark.parametrize("code", [1, 2, 3])
def test_terminal_skip_retains_zombie(native, monkeypatch, code):
    native.status = lambda pid: SELF
    monkeypatch.setattr(tree.os, "waitid", lambda *args: NS(si_pid=ROOT, si_code=code))
    assert native.member(ROOT, signal.SIGKILL) is False


def test_echild_stale_pid_never_signals(native, monkeypatch):
    native.status = lambda pid: SELF

    def echild(*args):
        raise ChildProcessError()

    monkeypatch.setattr(tree.os, "waitid", echild)
    assert native.member(ROOT, signal.SIGTERM) is None


@pytest.mark.parametrize("event", [NS(si_pid=OTHER, si_code=1), NS(si_pid=ROOT, si_code=5)])
def test_wrong_pid_or_ptrace_waitid_refused(native, monkeypatch, event):
    native.status = lambda pid: SELF
    monkeypatch.setattr(tree.os, "waitid", lambda *args: event)
    with pytest.raises(RuntimeError, match="terminal"):
        native.member(ROOT, signal.SIGTERM)


def test_parent_mismatch_and_waitid_error_refuse(native, monkeypatch):
    native.status = lambda pid: OTHER
    with pytest.raises(RuntimeError, match="real direct"):
        native.member(ROOT, signal.SIGTERM)
    native.status = lambda pid: SELF

    def bad(*args):
        raise PermissionError()

    monkeypatch.setattr(tree.os, "waitid", bad)
    with pytest.raises(PermissionError):
        native.member(ROOT, signal.SIGTERM)


@pytest.mark.parametrize("pid", [0, -1, True, 2**31])
def test_nonpositive_or_invalid_pid_never_signals(native, pid):
    with pytest.raises(RuntimeError):
        native.member(pid, signal.SIGKILL)


@pytest.mark.parametrize("replacement", [b"TracerPid:\t4", b"Tgid:\t102"])
def test_status_rejects_tracers_threads(native, monkeypatch, replacement):
    data = b"Pid:\t100\nPPid:\t99\nTgid:\t100\nTracerPid:\t0\nNSpid:\t100\n"
    key = replacement.split(b":")[0] + b":"
    data = b"\n".join(replacement if line.startswith(key) else line for line in data.splitlines())
    monkeypatch.setattr(builtins, "open", lambda *a, **k: BytesIO(data))
    with pytest.raises(RuntimeError, match="identity"):
        native.status(ROOT)


def test_prepare_order_clone_wall_and_no_children(native, monkeypatch):
    calls = []
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter([NS(name=str(SELF))])))
    def status(pid, *, require_host=False):
        assert pid == SELF and require_host is True
        calls.append("status")

    native.status = status
    monkeypatch.setattr(tree.signal, "signal", lambda sig, val: calls.append((sig, val)))

    def waitid(kind, pid, flags):
        assert (kind, pid, flags) == (tree.os.P_ALL, 0,
            tree.os.WEXITED | tree.os.WNOHANG | tree.os.WNOWAIT | tree.WALL)
        calls.append("wait")
        raise ChildProcessError()

    def prctl(command, value, *args):
        calls.append(command)
        if command == 37:
            value._obj.value = int(36 in calls)
        return 0

    monkeypatch.setattr(tree.os, "waitid", waitid)
    native.libc = NS(prctl=prctl)
    native.prepare()
    assert calls == ["status", (signal.SIGCHLD, signal.SIG_DFL), "wait", 37, 36, 37]


@pytest.mark.parametrize("event", [None, NS(si_pid=0), NS(si_pid=ROOT)])
def test_prepare_existing_children_even_zero_result_refused(native, monkeypatch, event):
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter([NS(name=str(SELF))])))
    native.status = lambda pid, *, require_host=False: None
    monkeypatch.setattr(tree.signal, "signal", lambda *args: None)
    monkeypatch.setattr(tree.os, "waitid", lambda *args: event)
    with pytest.raises(RuntimeError, match="preexisting"):
        native.prepare()


def test_clone_wall_reap_flags(native, monkeypatch):
    monkeypatch.setattr(tree.os, "waitpid", lambda pid, flags: (pid, flags))
    assert native.wait() == (-1, tree.os.WNOHANG | tree.WALL)


@pytest.mark.parametrize("data", [b"1 " * 257, b"x", b"0", b"1 1", b"9" * 4097])
def test_listing_bounded_and_invalid(native, monkeypatch, data):
    monkeypatch.setattr(builtins, "open", lambda *a, **k: BytesIO(data))
    with pytest.raises(RuntimeError):
        native.children()


def test_masked_signal_restore_interrupt_accounts_once(safe):
    b = Fake("orphan")
    member = b.member

    def interrupt():
        raise KeyboardInterrupt("after owned signal")

    def send(pid, sig=None):
        result = member(pid, sig)
        if sig is not None:
            assert safe.blocked == {signal.SIGINT, signal.SIGTERM}
            safe.interrupt = interrupt
        return result

    b.member = send
    r = run(b)
    assert r.signals == 1 and r.reaps == 2 and r.ancestry_drained
    assert not r.success and "KeyboardInterrupt" in r.error and len(b.sent) == 1


def test_cleanup_stale_lease_never_signal_or_empty_proof(monkeypatch):
    monkeypatch.setattr(tree, "MAX_OPS", 100)
    b = Fake("orphan")
    member = b.member
    b.frozen = True
    b.member = lambda pid, sig=None: member(pid) if sig is None else None
    r = run(b)
    assert r.root_status == 0 and r.reaps == 1 and r.signals == 0
    assert not r.ancestry_drained and not r.success and not b.sent


def test_terminal_lease_skipped_then_reaped_in_separate_phase():
    b = Fake("orphan")
    member = b.member

    def terminal(pid, sig=None):
        if sig is not None and not b.pending:
            b.pending.append((pid, 0))
        return member(pid, sig)

    b.member = terminal
    r = run(b)
    assert r.success and r.signals == 0 and r.reaps == 2 and not b.sent


@pytest.mark.parametrize("code", [PermissionError, ProcessLookupError, InterruptedError])
def test_signal_syscall_error_never_success(native, monkeypatch, safe, code):
    native.status = lambda pid: SELF
    monkeypatch.setattr(tree.os, "waitid", lambda *args: None)

    def failed(pid, sig):
        assert safe.blocked == {signal.SIGINT, signal.SIGTERM}
        raise code("signal refused")

    monkeypatch.setattr(tree.os, "kill", failed)
    with pytest.raises(code):
        native.member(ROOT, signal.SIGTERM)
    assert not safe.blocked


def test_each_term_and_kill_requires_fresh_status_then_waitid(native, monkeypatch):
    events = []
    native.status = lambda pid: events.append(("parent", pid)) or SELF
    monkeypatch.setattr(tree.os, "waitid", lambda kind, pid, flags: events.append(("waitid", pid)))
    monkeypatch.setattr(tree.os, "kill", lambda pid, sig: events.append(("kill", pid, sig)))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        assert native.member(ROOT, sig) is True
    assert events == [("parent", ROOT), ("waitid", ROOT), ("kill", ROOT, signal.SIGTERM),
                      ("parent", ROOT), ("waitid", ROOT), ("kill", ROOT, signal.SIGKILL)]


def test_actual_bounded_status_parent_precedes_membership(native, monkeypatch):
    events = []

    def opened(path, mode, **kwargs):
        assert path == f"/proc/{ROOT}/status" and kwargs == {"buffering": 0}
        events.append("status")
        return BytesIO(b"Pid: 100\nTgid: 100\nPPid: 99\nTracerPid: 0\nNSpid: 100\n")

    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(tree.os, "waitid", lambda *args: events.append("waitid"))
    assert native.member(ROOT) is True
    assert events == ["status", "waitid"]


@pytest.mark.parametrize("threads", [[], [NS(name="98")], [NS(name="99"), NS(name="100")]])
def test_prepare_rejects_not_exact_own_single_thread(native, monkeypatch, threads):
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter(threads)))
    with pytest.raises(RuntimeError, match="single thread"):
        native.prepare()


def test_child_closes_all_above_stderr_before_exec(native, monkeypatch, safe):
    events = []
    safe.blocked = {signal.SIGINT, signal.SIGTERM}

    class Exit(BaseException):
        pass

    def close(fd):
        assert not safe.blocked
        events.append(("close", fd))

    def execute(path, argv, env):
        events.append(("exec", path, argv, env))
        raise Exit()

    monkeypatch.setattr(tree.os, "close", close)
    monkeypatch.setattr(tree.os, "read", lambda fd, n: b"G")
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter(
        NS(name=str(fd)) for fd in (0, 1, 2, 20, 34, 999))))
    monkeypatch.setattr(tree.os, "chdir", lambda path: events.append(("cwd", path)))
    monkeypatch.setattr(tree.os, "execve", execute)
    monkeypatch.setattr(tree.os, "_exit", lambda code: (_ for _ in ()).throw(Exit()))
    with pytest.raises(Exit):
        native.child(20, 21, ("/fixed",), {"K": "V"}, "/work")
    assert events == [("close", 21), ("close", 20), ("close", 34), ("close", 999),
                      ("cwd", "/work"), ("exec", "/fixed", ("/fixed",), {"K": "V"})]


def test_original_root_absent_listing_after_reap_no_root_fallback():
    b = Fake("orphan")
    b.children = lambda: []
    r = run(b)
    assert r.root_status == 0 and not r.ancestry_drained and not r.signals
    assert not b.sent


def test_full_600_second_normal_budget_not_exhausted():
    b = Fake("linger")
    wait = b.wait

    def later():
        if b.now >= 599 and ROOT in b.children_set and not b.pending:
            b.pending.append((ROOT, 0))
        return wait()

    b.wait = later
    r = run(b, timeout=600)
    assert r.success and 599 <= b.now < 600


def test_normal_op_exhaustion_cleanup_fresh_budget_same_absolute_limit(monkeypatch):
    monkeypatch.setattr(tree, "MAX_OPS", 256)
    b = Fake("stubborn")
    r = run(b, timeout=600)
    assert not r.success and r.ancestry_drained and "operation bound" in r.error
    assert [sig for _, sig, _ in b.sent] == [signal.SIGTERM, signal.SIGKILL]
    assert b.now < 3


def test_repeated_interrupts_do_not_extend_cleanup_deadline():
    b = Fake("orphan")
    attempts = 0

    def interrupted():
        nonlocal attempts
        attempts += 1
        b.now += 0.02
        raise InterruptedError("sticky cleanup interruption")

    b.children = interrupted
    r = run(b)
    assert not r.success and not r.ancestry_drained and "InterruptedError" in r.error
    assert 20 <= attempts <= 26 and 0.5 <= b.now < 0.53


@pytest.mark.parametrize("prior,failure", [(1, None), (0, 1), (0, 2), (0, 3)])
def test_prepare_subreaper_prior_and_all_prctl_errors(native, monkeypatch, prior, failure):
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter([NS(name=str(SELF))])))
    native.status = lambda pid, *, require_host=False: None
    monkeypatch.setattr(tree.signal, "signal", lambda *args: None)

    def no_children(*args):
        raise ChildProcessError()

    calls = []

    def prctl(command, argument, *unused):
        calls.append(command)
        if command == 37:
            argument._obj.value = prior if len(calls) == 1 else 1
        return -1 if len(calls) == failure else 0

    native.libc = NS(prctl=prctl)
    monkeypatch.setattr(tree.os, "waitid", no_children)
    with pytest.raises(RuntimeError, match="subreaper|SUBREAPER"):
        native.prepare()


def test_final_echild_after_deadline_cannot_succeed():
    b = Fake()
    wait = b.wait

    def late_empty():
        try:
            return wait()
        except ChildProcessError:
            b.now = 0.6
            raise

    b.wait = late_empty
    r = run(b)
    assert r.root_status == 0 and r.ancestry_drained
    assert not r.success and "deadline" in r.error


@pytest.mark.parametrize("boundary", ["pipe", "fork", "signal", "reap"])
def test_mask_entry_apply_then_interrupt_no_unrecorded_effect(safe, boundary):
    b = Fake("orphan")
    prior = {signal.SIGHUP}
    safe.blocked = prior.copy()

    def interrupt():
        assert safe.blocked == prior | {signal.SIGINT, signal.SIGTERM}
        if boundary in ("pipe", "fork"):
            assert "fork" not in b.events
        if boundary == "pipe":
            assert "pipe" not in b.events
        if boundary == "signal":
            assert not b.sent
        if boundary == "reap":
            assert "wait" not in b.events
        raise KeyboardInterrupt("block applied before entry")

    if boundary == "pipe":
        prepare = b.prepare

        def arm():
            prepare()
            safe.entry_interrupt = interrupt

        b.prepare = arm
    elif boundary == "fork":
        pipe = b.pipe

        def arm():
            result = pipe()
            safe.entry_interrupt = interrupt
            return result

        b.pipe = arm
    elif boundary == "signal":
        children = b.children

        # Only interrupt once so cleanup can finish on its next iteration.
        def once():
            b.children = children
            interrupt()

        def arm_once():
            result = children()
            safe.entry_interrupt = once
            return result

        b.children = arm_once
    else:
        close = b.close

        def arm(fd):
            close(fd)
            if fd == 20:  # Last gate close, immediately before first drain.
                safe.entry_interrupt = interrupt

        b.close = arm

    r = run(b)
    assert safe.blocked == prior and not r.success and "KeyboardInterrupt" in r.error
    assert r.signals == len(b.sent)
    if boundary in ("pipe", "fork"):
        assert r.root_pid is None and not r.attempted and not r.released
        assert "fork" not in b.events and "wait" not in b.events and not b.sent
    else:
        assert b.events.count("fork") == 1 and r.reaps == 2 and r.ancestry_drained
        assert r.signals == 1


@pytest.mark.parametrize("apply_first", [False, True])
def test_mask_restore_retry_bounded_preserves_failure(monkeypatch, safe, apply_first):
    original = tree.signal.pthread_sigmask
    calls = 0
    safe.blocked = {signal.SIGHUP}

    def mask(how, values):
        nonlocal calls
        if how == signal.SIG_SETMASK:
            calls += 1
            if calls == 1:
                if apply_first:
                    original(how, values)
                raise KeyboardInterrupt("restore interrupted")
        return original(how, values)

    monkeypatch.setattr(tree.signal, "pthread_sigmask", mask)
    with pytest.raises(KeyboardInterrupt, match="restore interrupted"):
        with tree._masked():
            assert safe.blocked == {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
    assert safe.blocked == {signal.SIGHUP} and calls == 2


def test_mask_restore_persistent_error_stops_after_two(monkeypatch):
    original = tree.signal.pthread_sigmask
    calls = 0

    def mask(how, values):
        nonlocal calls
        if how == signal.SIG_SETMASK:
            calls += 1
            raise InterruptedError("persistent restoration failure")
        return original(how, values)

    monkeypatch.setattr(tree.signal, "pthread_sigmask", mask)
    with pytest.raises(InterruptedError):
        with tree._masked():
            pass
    assert calls == 2


@pytest.mark.parametrize("nspids", [b"100", b"100 1", b"100 100", b"100 200 1"])
def test_nested_child_host_pid_only_signalled(native, monkeypatch, nspids):
    data = b"Pid: 100\nTgid: 100\nPPid: 99\nTracerPid: 0\nNSpid: " + nspids + b"\n"
    monkeypatch.setattr(builtins, "open", lambda *a, **k: BytesIO(data))
    calls = []
    monkeypatch.setattr(tree.os, "waitid", lambda kind, pid, flags: calls.append(("wait", pid)))
    monkeypatch.setattr(tree.os, "kill", lambda pid, sig: calls.append(("signal", pid, sig)))
    assert native.member(ROOT, signal.SIGTERM) is True
    assert calls == [("wait", ROOT), ("signal", ROOT, signal.SIGTERM)]


@pytest.mark.parametrize("nspids", [None, b"", b"0", b"100 0", b"100 -1", b"100 +1",
                                   b"100 x", b"101 100", b"100 2147483648",
                                   b"100 00000000001", b"100 " + b"1 " * 32])
def test_status_invalid_nspid_never_membership_or_signal(native, monkeypatch, nspids):
    data = b"Pid: 100\nTgid: 100\nPPid: 99\nTracerPid: 0\n"
    if nspids is not None:
        data += b"NSpid: " + nspids + b"\n"
    monkeypatch.setattr(builtins, "open", lambda *a, **k: BytesIO(data))
    with pytest.raises(RuntimeError):
        native.member(ROOT, signal.SIGTERM)


def test_prepare_nested_self_refused_before_state_changes(native, monkeypatch):
    data = b"Pid: 99\nTgid: 99\nPPid: 98\nTracerPid: 0\nNSpid: 99 1\n"
    monkeypatch.setattr(builtins, "open", lambda *a, **k: BytesIO(data))
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter([NS(name=str(SELF))])))
    with pytest.raises(RuntimeError, match="identity"):
        native.prepare()


@pytest.mark.parametrize("event", ["echild", None, NS(si_pid=ROOT, si_code=1), PermissionError])
def test_missing_status_only_echild_skips_without_signal(native, monkeypatch, event):
    calls = []

    def missing(pid):
        calls.append(("status", pid))
        raise FileNotFoundError("stale proc node")

    def waitid(kind, pid, flags):
        assert (kind, pid, flags) == (tree.os.P_PID, ROOT,
            tree.os.WEXITED | tree.os.WNOHANG | tree.os.WNOWAIT | tree.WALL)
        calls.append(("waitid", pid))
        if event == "echild":
            raise ChildProcessError()
        if event is PermissionError:
            raise PermissionError()
        return event

    native.status = missing
    monkeypatch.setattr(tree.os, "waitid", waitid)
    if event == "echild":
        assert native.member(ROOT, signal.SIGTERM) is None
    else:
        with pytest.raises(PermissionError if event is PermissionError else RuntimeError):
            native.member(ROOT, signal.SIGTERM)
    assert calls == [("status", ROOT), ("waitid", ROOT)]
