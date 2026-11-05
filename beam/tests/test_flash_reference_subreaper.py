"""Portable mocked-call tests ONLY: no real Linux syscalls, fork, proc or signals.

These are state-machine checks, not Linux race/runtime/watchdog evidence.
"""

from dataclasses import FrozenInstanceError
import signal

import pytest

from beamfastsim import flash_reference_subreaper as tree

_BackendClass = tree._LinuxBackend
ROOT, ORPHAN, LATE = 100, 101, 102
SELF, SELF_FD = 99, 19


class FakeBackend:
    def __init__(self, *, mode="normal", fail=None):
        self.now = 0.0
        self.mode, self.fail = mode, fail
        self.events, self.live, self.closed = [], set(), []
        self.forked = self.released = False
        self.root_done = self.root_reaped = False
        self.pending = []
        self.children_left = set()
        self.next_fd = 20
        self.prepare_calls = 0
        self.wait_calls = 0
        self.signals = []
        self.member_fail = False
        self.unknown = False
        self.launch = None
        self.signal_exit = True
        self.adopt_late = False
        self.frontier_empty = False
        self.all_reaps = 0

    def step(self, name):
        self.events.append(name)
        if name == self.fail:
            raise OSError("injected " + name)

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert 0 <= seconds <= 0.01
        self.events.append("sleep")
        self.now += seconds

    def prepare(self):
        self.step("prepare")
        self.prepare_calls += 1
        if self.prepare_calls != 1:
            raise RuntimeError("prior subreaper one")
        with tree._defer_interrupts():
            fd = self.pidfd_open(SELF)
            try:
                self.send(fd, 0)
            finally:
                self.close(fd)

    def alloc(self):
        fd = self.next_fd
        self.next_fd += 1
        self.live.add(fd)
        return fd

    def pipe(self):
        self.step("pipe")
        return self.alloc(), self.alloc()

    def fork(self):
        self.step("fork")
        assert self.prepare_calls == 1
        self.forked = True
        self.children_left.add(ROOT)
        return ROOT

    def child(self, *args):
        raise AssertionError("no actual child path in parent mock")

    def close(self, fd):
        assert fd in self.live
        self.live.remove(fd)
        self.closed.append(fd)
        if fd == 21 and not self.released:
            # EOF gate cancels all child work; child exits without exec.
            self.root_done = True
            self.pending.append((ROOT, 125 << 8))
        self.step("self_close" if fd == SELF_FD else "close")

    def pidfd_open(self, pid):
        if pid == SELF:
            self.step("self_pidfd")
            self.live.add(SELF_FD)
            return SELF_FD
        self.step("pidfd")
        assert pid in self.children_left, "no capture outside current ancestry"
        return self.alloc()

    def member(self, pid):
        self.step("member")
        assert pid in self.children_left, "not a waitable direct child"
        if self.member_fail:
            raise ChildProcessError("not a member")

    def release(self, writer):
        self.step("release")
        assert writer == 21 and writer in self.live
        assert "pidfd" in self.events and "member" in self.events
        self.released = True
        if self.mode in ("normal", "nonzero", "orphan", "flood"):
            self.root_done = True
            self.pending.append((ROOT, 0 if self.mode != "nonzero" else 7 << 8))
        if self.mode == "orphan":
            self.children_left.add(ORPHAN)
        if self.mode == "flood":
            self.children_left.update(range(200, 300))
            self.pending.extend((pid, 0) for pid in range(200, 300))

    def wait(self):
        self.step("wait")
        self.wait_calls += 1
        if self.pending:
            pid, status = self.pending.pop(0)
            self.children_left.remove(pid)
            self.root_reaped |= pid == ROOT
            self.all_reaps += 1
            if pid == ORPHAN and self.adopt_late:
                self.children_left.add(LATE)
            return pid, status
        if self.children_left or self.unknown:
            return 0, 0
        raise ChildProcessError("ECHILD")

    def children(self):
        self.step("children")
        if self.frontier_empty:
            return []
        return sorted(self.children_left)

    def send(self, fd, sig):
        if fd == SELF_FD:
            assert sig == 0 and fd in self.live
            self.step("self_send")
            return
        self.step("send")
        assert self.events[-2] == "member"  # No drain between proof and signal.
        assert fd in self.live
        self.signals.append((fd, sig))
        if self.signal_exit and (self.mode != "stubborn" or sig == signal.SIGKILL):
            # FD allocation: root first, then adopted frontier in sorted order.
            pid = {22: ROOT, 23: ORPHAN, 24: LATE}.get(fd)
            assert pid in self.children_left
            if not any(item[0] == pid for item in self.pending):
                self.pending.append((pid, sig))


def run(backend, **kwargs):
    options = dict(timeout=0.5, external_watchdog_present=True, _backend=backend)
    options.update(kwargs)
    result = tree.run_tree(("/fixed/program", "arg"), {"KEY": "fixed"}, "/work", **options)
    assert not backend.live
    assert len(backend.closed) == len(set(backend.closed))
    return result


@pytest.fixture(autouse=True)
def masks(monkeypatch):
    # Never alter actual process signal masks, even when testing backend methods.
    class Masks:
        blocked = set()
        events = []
        after_restore = None

        def apply(self, how, values):
            previous = self.blocked.copy()
            self.events.append((how, set(values)))
            if how == signal.SIG_BLOCK:
                self.blocked |= set(values)
            elif how == signal.SIG_UNBLOCK:
                self.blocked -= set(values)
            else:
                assert how == signal.SIG_SETMASK
                self.blocked = set(values)
                if self.after_restore is not None:
                    self.after_restore()
            return previous

    state = Masks()
    monkeypatch.setattr(tree.signal, "pthread_sigmask", state.apply)
    return state


@pytest.fixture(autouse=True)
def never_real_backend(monkeypatch):
    def forbidden():
        pytest.fail("tests must never instantiate the real Linux backend")
    monkeypatch.setattr(tree, "_LinuxBackend", forbidden)


def test_normal_single_root_gated_before_exec_and_echild():
    backend = FakeBackend()
    report = run(backend)
    assert report.success and report.ancestry_drained
    assert (report.root_pid, report.root_status, report.reaps, report.captures) == (ROOT, 0, 1, 1)
    assert report.launch_attempted and report.launch_released
    assert report.error is None and not backend.signals
    assert backend.events.index("prepare") < backend.events.index("self_pidfd")
    assert backend.events.index("self_pidfd") < backend.events.index("self_send")
    assert backend.events.index("self_send") < backend.events.index("self_close")
    assert backend.events.index("self_close") < backend.events.index("pipe") < backend.events.index("fork")
    assert backend.events.index("pidfd") < backend.events.index("release") < backend.events.index("wait")
    assert backend.events.count("fork") == 1
    assert report.verified is False and report.certificate is False and report.retain_work is True
    with pytest.raises(FrozenInstanceError):
        report.verified = True


def test_nonzero_is_failure_even_when_drained():
    report = run(FakeBackend(mode="nonzero"))
    assert not report.success and report.ancestry_drained
    assert report.root_status == 7 << 8
    assert "root exited" in report.error


def test_wait_zero_is_not_drained_and_timeout_remains_failure_after_cleanup():
    backend = FakeBackend(mode="linger")
    report = run(backend)
    assert backend.now >= 0.5
    assert report.ancestry_drained and not report.success
    assert report.root_status == signal.SIGTERM
    assert isinstance(report.error, str) and "deadline" in report.error
    assert backend.signals


def test_empty_proc_children_never_establishes_empty_and_root_is_targeted():
    backend = FakeBackend(mode="linger")
    backend.frontier_empty = True
    report = run(backend)
    assert report.ancestry_drained and backend.signals
    assert report.error and not report.success


def test_root_exit_stops_orphan_and_iterates_later_adoption():
    backend = FakeBackend(mode="orphan")
    backend.adopt_late = True
    report = run(backend)
    assert report.ancestry_drained and report.root_status == 0
    assert not report.success and "termination signals" in report.error
    assert report.captures == 3 and report.reaps == 3
    assert len(backend.signals) == 2
    assert all(sig == signal.SIGTERM for _, sig in backend.signals)


def test_term_then_kill_bounded_grace():
    backend = FakeBackend(mode="stubborn")
    report = run(backend)
    assert report.ancestry_drained and not report.success
    assert [sig for _, sig in backend.signals] == [signal.SIGTERM, signal.SIGKILL]
    assert 0.6 <= backend.now < 1.0


def test_unkillable_unknown_keeps_work_and_cannot_claim_finite_cleanup():
    backend = FakeBackend(mode="linger")
    backend.signal_exit = False
    report = run(backend)
    assert not report.success and not report.ancestry_drained
    assert report.root_status is None and report.retain_work
    assert backend.now <= 1.01
    assert [sig for _, sig in backend.signals] == [signal.SIGTERM, signal.SIGKILL]


def test_echild_is_required_not_just_root_status_and_empty_listing():
    backend = FakeBackend()
    backend.unknown = True
    report = run(backend)
    assert report.root_status == 0 and not report.ancestry_drained
    assert not report.success and report.retain_work
    assert "children" in backend.events


def test_cancel_before_root_exit_stops_and_stays_failure():
    backend = FakeBackend(mode="linger")
    report = run(backend, cancel=lambda: True)
    assert report.ancestry_drained and not report.success
    assert "cancelled" in report.error
    assert backend.now < 0.5


def test_interrupt_gets_bounded_cleanup_report():
    def interrupt():
        raise KeyboardInterrupt()
    report = run(FakeBackend(mode="linger"), cancel=interrupt)
    assert report.ancestry_drained and not report.success
    assert "KeyboardInterrupt" in report.error


@pytest.mark.parametrize("failure", ["prepare", "self_pidfd", "self_send", "self_close", "pipe", "fork", "pidfd", "member", "release", "wait", "children", "send", "close"])
def test_injected_failures_are_explicit_never_success(failure):
    mode = "orphan" if failure in ("children", "send") else "normal"
    backend = FakeBackend(mode=mode, fail=failure)
    report = run(backend)
    assert not report.success and report.error and report.retain_work
    if failure in ("self_pidfd", "self_send", "self_close"):
        assert report.root_pid is None and report.root_status is None
        assert not report.launch_attempted and not report.launch_released
        assert not backend.signals and "pipe" not in backend.events and "fork" not in backend.events
    if failure in ("prepare", "pipe", "fork"):
        assert not report.launch_released
    if failure in ("pidfd", "member", "release", "close"):
        assert not report.launch_released  # gate closed without executing payload
    assert backend.events.count("fork") <= 1


def test_nonchild_membership_prevents_any_signal():
    backend = FakeBackend(mode="linger")
    backend.member_fail = True
    report = run(backend)
    assert not report.launch_released and not report.success
    assert not backend.signals


def test_reap_bound_including_clone_children_is_absolute():
    backend = FakeBackend(mode="flood")
    report = run(backend)
    assert report.reaps == tree.MAX_REAPS
    assert backend.all_reaps == tree.MAX_REAPS
    assert not report.ancestry_drained and not report.success
    assert "reap bound" in report.error


def test_capture_bound_is_total_not_simultaneous(monkeypatch):
    monkeypatch.setattr(tree, "MAX_CAPTURES", 2)
    backend = FakeBackend(mode="orphan")
    backend.adopt_late = True
    report = run(backend)
    assert report.captures == 2
    assert not report.success and not report.ancestry_drained
    assert "capture bound" in report.error


def test_loop_bound_even_with_nonadvancing_clock(monkeypatch):
    monkeypatch.setattr(tree, "MAX_LOOPS", 80)
    backend = FakeBackend(mode="linger")
    backend.sleep = lambda _: None
    report = run(backend)
    assert not report.success and report.ancestry_drained
    assert "operation bound" in report.error
    assert [sig for _, sig in backend.signals] == [signal.SIGTERM]


def test_single_shot_second_call_cannot_fork():
    backend = FakeBackend()
    assert run(backend).success
    report = run(backend)
    assert not report.launch_attempted and not report.success
    assert backend.events.count("fork") == 1


@pytest.mark.parametrize("kwargs", [
    {"external_watchdog_present": False}, {"external_watchdog_present": 1},
    {"external_watchdog_present": "yes"}, {"timeout": True}, {"timeout": 0},
    {"timeout": -1}, {"timeout": 600.01}, {"timeout": float("nan")},
    {"timeout": float("inf")}, {"cancel": 1},
])
def test_invalid_config_does_not_touch_backend(kwargs):
    backend = FakeBackend()
    with pytest.raises(ValueError):
        run(backend, **kwargs)
    assert not backend.events


@pytest.mark.parametrize("argv,env,cwd", [
    ([], {}, "/"), (["relative"], {}, "/"), (["/x", "nul\0"], {}, "/"),
    (["/x"], {"bad=key": "x"}, "/"), (["/x"], {"X": 1}, "/"),
    (["/x"], {"": "x"}, "/"), (["/x"], {}, "relative"),
    (["/x"], {}, "/nul\0"), ([True], {}, "/"), (["/x"], None, "/"),
])
def test_launch_configuration_rejected_before_setup(argv, env, cwd):
    backend = FakeBackend()
    with pytest.raises(ValueError):
        tree.run_tree(argv, env, cwd, timeout=1, external_watchdog_present=True, _backend=backend)
    assert not backend.events


def test_timeout_600_accepted_without_real_wait():
    assert run(FakeBackend(), timeout=600).success


def test_pid_reuse_after_reap_gets_new_capture_and_new_signal():
    backend = FakeBackend(mode="orphan")
    original_wait = backend.wait
    original_send = backend.send
    reused = False

    def wait_with_adoption():
        nonlocal reused
        result = original_wait()
        if result[0] == ORPHAN and not reused:
            reused = True
            # A previously nested, now adopted child can reuse a reaped PID.
            backend.children_left.add(ORPHAN)
        return result

    def send(fd, sig):
        if fd == 24:  # New pidfd, despite identical numeric PID.
            backend.step("send")
            assert backend.events[-2] == "member"
            backend.signals.append((fd, sig))
            backend.pending.append((ORPHAN, sig))
        else:
            original_send(fd, sig)

    backend.wait, backend.send = wait_with_adoption, send
    report = run(backend)
    assert report.ancestry_drained and not report.success
    assert report.captures == 3 and report.reaps == 3
    assert len(backend.signals) == 2
    assert [fd for fd, _ in backend.signals] == [23, 24]


@pytest.mark.parametrize("mode,expected", [("normal", 0), ("nonzero", 7 << 8)])
def test_root_pid_reused_after_reap_never_overwrites_original_root_status(mode, expected):
    backend = FakeBackend(mode=mode)
    original_wait = backend.wait
    reused = False

    def wait_with_adoption():
        nonlocal reused
        result = original_wait()
        if result[0] == ROOT and not reused:
            reused = True
            backend.children_left.add(ROOT)
            backend.pending.append((ROOT, 0))
        return result

    backend.wait = wait_with_adoption
    report = run(backend)
    assert report.root_status == expected and report.reaps == 2
    assert report.ancestry_drained and report.success is (expected == 0)
    if expected:
        assert "root exited" in report.error


def test_final_drain_after_execution_deadline_is_not_success():
    backend = FakeBackend()
    original_wait = backend.wait

    def slow_echild():
        try:
            return original_wait()
        except ChildProcessError:
            backend.now = 0.6
            raise

    backend.wait = slow_echild
    report = run(backend)
    assert report.root_status == 0 and report.ancestry_drained
    assert not report.success and "deadline" in report.error


def test_launch_configuration_copied_before_setup():
    backend = FakeBackend()
    argv, env = ["/fixed/program"], {"KEY": "original"}
    original_prepare = backend.prepare

    def prepare():
        argv[:] = ["/different"]
        env["KEY"] = "changed"
        original_prepare()

    # The parent never execs; exercise the copied launch arguments with an
    # injected fork==0/child stub which raises, without any actual fork.
    seen = []
    backend.prepare = prepare
    backend.fork = lambda: 0

    def child(reader, writer, child_argv, child_env, cwd):
        seen.append((child_argv, child_env, cwd))
        raise RuntimeError("mock child, not actual exec")

    backend.child = child
    report = tree.run_tree(argv, env, "/work", timeout=1,
                           external_watchdog_present=True, _backend=backend)
    assert seen == [(("/fixed/program",), {"KEY": "original"}, "/work")]
    assert not report.success and not backend.live


def test_full_allowed_timeout_does_not_hit_early_loop_limit():
    backend = FakeBackend(mode="linger")
    original_wait = backend.wait

    def delayed_exit():
        if backend.now >= 599 and not backend.root_done:
            backend.root_done = True
            backend.pending.append((ROOT, 0))
        return original_wait()

    backend.wait = delayed_exit
    report = run(backend, timeout=600)
    assert report.success and report.ancestry_drained
    assert 599 <= backend.now < 600


@pytest.mark.parametrize("prior,failed,threads,child,accepted", [
    (0, None, 1, False, True), (1, None, 1, False, False),
    (0, "get", 1, False, False), (0, "set", 1, False, False),
    (0, "verify", 1, False, False), (0, None, 2, False, False),
    (0, None, 1, True, False),
])
def test_linux_setup_contract_with_only_mocked_calls(monkeypatch, prior, failed,
                                                     threads, child, accepted):
    from contextlib import nullcontext
    backend = object.__new__(_BackendClass)  # Never initialize real Linux backend.
    calls = []
    state = prior
    get_calls = 0

    class Libc:
        def prctl(self, op, argument, *unused):
            nonlocal state, get_calls
            calls.append(op)
            if op == 37:
                get_calls += 1
                argument._obj.value = state
                return -1 if failed == "get" or (failed == "verify" and get_calls == 2) else 0
            assert op == 36 and argument == 1
            state = 1
            return -1 if failed == "set" else 0

    def waitpid(pid, flags):
        assert pid == -1 and flags == 1 | tree.WALL
        calls.append("wait")
        if child:
            return 0, 0
        raise ChildProcessError()

    backend.libc = Libc()
    monkeypatch.setattr(tree.os, "getpid", lambda: SELF)
    backend.pidfd_open = lambda pid: calls.append(("self_pidfd", pid)) or SELF_FD
    backend.send = lambda fd, sig: calls.append(("self_send", fd, sig))
    backend.close = lambda fd: calls.append(("self_close", fd))
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter(range(threads))))
    monkeypatch.setattr(tree.os, "WNOHANG", 1)
    monkeypatch.setattr(tree.os, "waitpid", waitpid)
    monkeypatch.setattr(tree.signal, "signal", lambda sig, value: calls.append((sig, value)))
    if accepted:
        backend.prepare()
        assert calls == [("self_pidfd", SELF), ("self_send", SELF_FD, 0),
                         ("self_close", SELF_FD), (signal.SIGCHLD, signal.SIG_DFL),
                         "wait", 37, 36, 37]
        assert state == 1
    else:
        with pytest.raises(RuntimeError):
            backend.prepare()


def test_linux_wait_and_membership_options_with_mocked_calls(monkeypatch):
    backend = object.__new__(_BackendClass)
    recorded = []
    for name, value in (("WNOHANG", 1), ("WEXITED", 4), ("WNOWAIT", 0x1000000), ("P_PID", 1)):
        monkeypatch.setattr(tree.os, name, value, raising=False)
    monkeypatch.setattr(tree.os, "waitpid", lambda pid, flags: recorded.append((pid, flags)) or (0, 0))
    monkeypatch.setattr(tree.os, "waitid", lambda *args: recorded.append(args))
    monkeypatch.setattr(tree.os, "getpid", lambda: 77)
    monkeypatch.setattr(tree.os, "open", lambda *args: 8)
    monkeypatch.setattr(tree.os, "read", lambda *args: b"100 (name ) tricky) S 77 0 0")
    monkeypatch.setattr(tree.os, "close", lambda fd: recorded.append(("close", fd)))
    assert backend.wait() == (0, 0)
    backend.member(ROOT)
    assert recorded == [(-1, 1 | tree.WALL),
                        (1, ROOT, 4 | 1 | 0x1000000 | tree.WALL), ("close", 8)]


def test_normal_operation_exhaustion_still_gets_term_and_kill(monkeypatch):
    monkeypatch.setattr(tree, "MAX_LOOPS", 256)
    backend = FakeBackend(mode="stubborn")
    report = run(backend, timeout=600)
    assert not report.success and report.ancestry_drained
    assert "operation bound" in report.error
    assert [sig for _, sig in backend.signals] == [signal.SIGTERM, signal.SIGKILL]
    assert backend.now < 3


def test_nonadvancing_clock_also_bounds_independent_cleanup(monkeypatch):
    monkeypatch.setattr(tree, "MAX_LOOPS", 80)
    backend = FakeBackend(mode="linger")
    backend.signal_exit = False
    backend.sleep = lambda _: None
    report = run(backend, timeout=600)
    assert not report.success and not report.ancestry_drained
    assert "operation bound" in report.error and backend.now == 0
    assert [sig for _, sig in backend.signals] == [signal.SIGTERM]
    assert backend.wait_calls < 2 * tree.MAX_LOOPS


def test_killed_pid_reuse_receives_fresh_term_and_kill():
    backend = FakeBackend(mode="orphan")
    original_wait = backend.wait
    reused = False

    def wait():
        nonlocal reused
        result = original_wait()
        if result[0] == ORPHAN and not reused:
            reused = True
            backend.children_left.add(ORPHAN)
        return result

    def send(fd, sig):
        if fd == SELF_FD:
            assert sig == 0
            backend.step("self_send")
            return
        backend.step("send")
        assert backend.events[-2] == "member"
        backend.signals.append((fd, sig))
        if sig == signal.SIGKILL:
            backend.pending.append((ORPHAN, sig))

    backend.wait, backend.send = wait, send
    report = run(backend)
    assert report.ancestry_drained and not report.success
    assert report.captures == 3 and report.reaps == 3
    # Once grace has expired, the new identity immediately gets a fresh KILL.
    assert backend.signals == [(23, signal.SIGTERM), (23, signal.SIGKILL),
                               (24, signal.SIGKILL)]


def test_resource_acquisitions_and_close_transfers_are_masked(masks):
    backend = FakeBackend()
    interrupts = {signal.SIGINT, signal.SIGTERM}
    prior = {signal.SIGUSR1}
    masks.blocked = prior.copy()
    for name in ("pipe", "fork", "pidfd_open", "close"):
        original = getattr(backend, name)

        def checked(*args, original=original):
            assert interrupts <= masks.blocked
            return original(*args)

        setattr(backend, name, checked)
    assert run(backend).success
    assert masks.blocked == prior
    assert all(how in (signal.SIG_BLOCK, signal.SIG_SETMASK)
               for how, _ in masks.events)


@pytest.mark.parametrize("operation", ["pipe", "fork", "pidfd_open", "close"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_pending_interrupt_after_acquisition_retains_tracked_ownership(masks, operation, interrupt):
    backend = FakeBackend()
    original = getattr(backend, operation)
    triggered = False
    prior = {signal.SIGUSR1}
    masks.blocked = prior.copy()

    def pending(*args):
        nonlocal triggered
        assert {signal.SIGINT, signal.SIGTERM} <= masks.blocked
        result = original(*args)
        if not triggered and args != (SELF,) and args != (SELF_FD,):
            triggered = True

            def deliver():
                masks.after_restore = None
                raise interrupt("mock pending interrupt")

            masks.after_restore = deliver
        return result

    setattr(backend, operation, pending)
    report = run(backend)
    assert triggered and not report.success
    assert interrupt.__name__ in report.error
    assert masks.blocked == prior
    if operation != "pipe":
        assert report.ancestry_drained
    else:
        assert not report.launch_attempted


@pytest.mark.parametrize("failure", ["pipe", "fork", "pidfd", "close"])
def test_failed_resource_operation_restores_prior_mask(masks, failure):
    masks.blocked = {signal.SIGUSR1}
    report = run(FakeBackend(fail=failure))
    assert not report.success
    assert masks.blocked == {signal.SIGUSR1}


@pytest.mark.parametrize("failure", [None, "open", "read", "close", "restore"])
def test_member_proc_fd_masked_until_finally_close(monkeypatch, masks, failure):
    backend = object.__new__(_BackendClass)
    events = []
    prior = {signal.SIGUSR1}
    masks.blocked = prior.copy()
    interrupts = {signal.SIGINT, signal.SIGTERM}
    for name, value in (("WNOHANG", 1), ("WEXITED", 4), ("WNOWAIT", 0x1000000), ("P_PID", 1)):
        monkeypatch.setattr(tree.os, name, value, raising=False)
    monkeypatch.setattr(tree.os, "waitid", lambda *args: None)
    monkeypatch.setattr(tree.os, "getpid", lambda: 77)

    def operation(name, result):
        def call(*args):
            assert interrupts <= masks.blocked
            events.append(name)
            if failure == name:
                raise OSError("mock " + name)
            if name == "open" and failure == "restore":
                def deliver():
                    masks.after_restore = None
                    raise KeyboardInterrupt("mock pending interrupt")
                masks.after_restore = deliver
            return result
        return call

    monkeypatch.setattr(tree.os, "open", operation("open", 8))
    monkeypatch.setattr(tree.os, "read", operation("read", b"100 (name) S 77 0 0"))
    monkeypatch.setattr(tree.os, "close", operation("close", None))
    if failure:
        with pytest.raises(KeyboardInterrupt if failure == "restore" else OSError):
            backend.member(ROOT)
    else:
        backend.member(ROOT)
    assert events == (["open"] if failure == "open" else ["open", "read", "close"])
    assert masks.blocked == prior


def test_child_unblocks_inherited_interrupt_mask_before_gate_read(monkeypatch, masks):
    backend = object.__new__(_BackendClass)
    masks.blocked = {signal.SIGINT, signal.SIGTERM, signal.SIGUSR1}
    events = []

    class ChildExit(BaseException):
        pass

    def close(fd):
        assert masks.blocked == {signal.SIGUSR1}
        events.append(("close", fd))

    def read(fd, count):
        assert masks.blocked == {signal.SIGUSR1}
        events.append(("read", fd, count))
        return b""  # EOF gate: never exec.

    def exit(code):
        raise ChildExit(code)

    monkeypatch.setattr(tree.os, "close", close)
    monkeypatch.setattr(tree.os, "read", read)
    monkeypatch.setattr(tree.os, "_exit", exit)
    with pytest.raises(ChildExit):
        backend.child(20, 21, ("/fixed/program",), {}, "/work")
    assert events == [("close", 21), ("read", 20, 1)]
    assert masks.events == [(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})]


@pytest.mark.parametrize("operation", ["pidfd_open", "member"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, InterruptedError])
def test_cleanup_orphan_pending_restore_retries_and_reaches_echild(masks, operation, interrupt):
    from contextlib import nullcontext

    backend = FakeBackend(mode="orphan")
    backend.signal_exit = False
    prior = {signal.SIGUSR1}
    masks.blocked = prior.copy()
    original = getattr(backend, operation)
    original_send = backend.send
    triggered = False
    acquisitions = []
    original_pidfd = backend.pidfd_open

    def pidfd(pid):
        acquisitions.append(pid)
        return original_pidfd(pid)

    backend.pidfd_open = pidfd
    if operation == "pidfd_open":
        original = pidfd

    def pending(pid):
        nonlocal triggered
        # pidfd_open is already masked by capture; membership masks its own
        # proc-FD region. Deliver only on the corresponding outer restoration.
        with tree._defer_interrupts() if operation == "member" else nullcontext():
            result = original(pid)
            if pid == ORPHAN and not triggered:
                assert backend.root_reaped and backend.released
                assert acquisitions == [SELF, ROOT, ORPHAN]
                triggered = True

                def deliver():
                    masks.after_restore = None
                    raise interrupt("cleanup orphan pending interrupt")

                masks.after_restore = deliver
        return result

    def send(fd, sig):
        original_send(fd, sig)
        if sig == signal.SIGKILL:
            backend.pending.append((ORPHAN, sig))

    setattr(backend, operation, pending)
    backend.send = send
    report = run(backend)
    assert triggered and not report.success and report.ancestry_drained
    assert interrupt.__name__ in report.error
    assert report.root_status == 0 and report.reaps == report.captures == 2
    assert acquisitions == [SELF, ROOT, ORPHAN]  # Owned root/orphan pidfds never reacquired.
    assert backend.signals == [(23, signal.SIGTERM), (23, signal.SIGKILL)]
    assert backend.wait_calls >= 4  # Root, live orphan, orphan reap, then ECHILD.
    assert masks.blocked == prior
    assert 0.1 <= backend.now < 0.5


@pytest.mark.parametrize("advance", [0, 0.02], ids=["operation-budget", "deadline"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit, InterruptedError])
def test_repeated_cleanup_restore_interrupts_do_not_reset_bounds(monkeypatch, masks, advance, interrupt):
    monkeypatch.setattr(tree, "MAX_LOOPS", 1000 if advance else 80)
    backend = FakeBackend(mode="orphan")
    original = backend.member
    attempts = 0

    def member(pid):
        nonlocal attempts
        with tree._defer_interrupts():
            original(pid)
            if pid == ORPHAN:
                assert backend.root_reaped
                attempts += 1
                backend.now += advance

                def deliver():
                    masks.after_restore = None
                    raise interrupt("persistent cleanup interrupt")

                masks.after_restore = deliver

    backend.member = member
    report = run(backend)
    assert not report.success and not report.ancestry_drained
    assert interrupt.__name__ in report.error
    assert 1 < attempts < tree.MAX_LOOPS
    assert report.reaps == 1 and report.captures == 2
    assert report.reaps <= tree.MAX_REAPS and report.captures <= tree.MAX_CAPTURES
    assert not backend.signals
    if advance:
        assert 0.5 <= backend.now < 0.53
    else:
        assert backend.now == 0


@pytest.mark.parametrize('native', [True, False])
def test_reviewed_pidfd_abi_path_never_uses_numeric_signal(monkeypatch, native):
    from types import SimpleNamespace
    calls = []

    class Syscall:
        restype = None

        def __call__(self, number, pid, flags):
            calls.append(('syscall', number.value, pid.value, flags.value))
            return 41

    syscall = Syscall()
    monkeypatch.setattr(tree.sys, 'platform', 'linux')
    monkeypatch.setattr(tree.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(tree.ctypes, 'CDLL', lambda *args, **kwargs: SimpleNamespace(syscall=syscall))
    monkeypatch.setattr(tree.ctypes, 'sizeof', lambda _: 8)
    monkeypatch.setattr(tree.os, 'waitid', lambda *args: None, raising=False)
    monkeypatch.setattr(tree.signal, 'pidfd_send_signal', lambda *args: calls.append(('send', args)), raising=False)
    if native:
        monkeypatch.setattr(tree.os, 'pidfd_open', lambda pid, flags: calls.append(('native', pid, flags)) or 41, raising=False)
    else:
        monkeypatch.delattr(tree.os, 'pidfd_open', raising=False)
    backend = _BackendClass()
    assert backend.pidfd_open(ROOT) == 41
    assert calls == [('native', ROOT, 0)] if native else [('syscall', 434, ROOT, 0)]
    if not native:
        assert syscall.restype is tree.ctypes.c_long
    backend.send(41, signal.SIGTERM)
    assert calls[-1] == ('send', (41, signal.SIGTERM, None, 0))


@pytest.mark.parametrize('machine,size', [('aarch64', 8), ('x86_64', 4), ('unknown', 8)])
def test_raw_pidfd_unreviewed_abi_refused(monkeypatch, machine, size):
    from types import SimpleNamespace
    monkeypatch.setattr(tree.sys, 'platform', 'linux')
    monkeypatch.setattr(tree.platform, 'machine', lambda: machine)
    monkeypatch.setattr(tree.ctypes, 'sizeof', lambda _: size)
    monkeypatch.setattr(tree.ctypes, 'CDLL', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.delattr(tree.os, 'pidfd_open', raising=False)
    monkeypatch.setattr(tree.os, 'waitid', lambda *args: None, raising=False)
    monkeypatch.setattr(tree.signal, 'pidfd_send_signal', lambda *args: None, raising=False)
    with pytest.raises(RuntimeError, match='reviewed x86_64 ABI'):
        _BackendClass()


@pytest.mark.parametrize('error_code', [38, 1, 3, 24])
def test_raw_pidfd_syscall_error_is_not_a_fallback(monkeypatch, error_code):
    from types import SimpleNamespace
    backend = object.__new__(_BackendClass)
    backend._native_pidfd_open = None
    backend.libc = SimpleNamespace(syscall=lambda *args: -1)
    monkeypatch.setattr(tree.ctypes, 'get_errno', lambda: error_code)
    with pytest.raises(OSError) as error:
        backend.pidfd_open(ROOT)
    assert error.value.errno == error_code


@pytest.mark.parametrize("native", [True, False], ids=["native", "raw-x86_64"])
@pytest.mark.parametrize("outcome", [
    "success", "enosys", "send-refused", "send-interrupt", "close-error",
    "close-interrupt", "pending-open", "pending-send", "pending-close",
])
def test_linux_prepare_self_pidfd_capability_before_any_launch(monkeypatch, masks, native, outcome):
    from contextlib import nullcontext
    from types import SimpleNamespace

    backend = FakeBackend()
    probe = object.__new__(_BackendClass)
    calls = []
    interrupts = {signal.SIGINT, signal.SIGTERM}
    prior = {signal.SIGUSR1}
    masks.blocked = prior.copy()

    def pending(operation):
        if outcome == "pending-" + operation:
            def deliver():
                masks.after_restore = None
                raise KeyboardInterrupt("mock pending " + operation)
            masks.after_restore = deliver

    def open_self(pid, flags):
        assert interrupts <= masks.blocked
        assert (pid, flags) == (SELF, 0)
        calls.append("open")
        if outcome == "enosys":
            raise OSError(38, "mock ENOSYS")
        backend.live.add(SELF_FD)
        pending("open")
        return SELF_FD

    def syscall(number, pid, flags):
        assert (number.value, flags.value) == (434, 0)
        try:
            return open_self(pid.value, flags.value)
        except OSError:
            return -1

    def send(fd, sig, info, flags):
        assert interrupts <= masks.blocked
        assert (fd, sig, info, flags) == (SELF_FD, 0, None, 0)
        assert fd in backend.live
        calls.append("send")
        if outcome == "send-refused":
            raise PermissionError(1, "mock signal 0 refused")
        if outcome == "send-interrupt":
            raise KeyboardInterrupt("mock send interrupt")
        pending("send")

    def close(fd):
        assert interrupts <= masks.blocked
        calls.append("close")
        backend.close(fd)
        if outcome == "close-error":
            raise OSError(5, "mock close error")
        if outcome == "close-interrupt":
            raise KeyboardInterrupt("mock close interrupt")
        pending("close")

    def prctl(op, argument, *unused):
        calls.append(op)
        if op == 37:
            argument._obj.value = int(36 in calls)
        else:
            assert op == 36 and argument == 1
        return 0

    def waitpid(pid, flags):
        assert pid == -1 and flags == tree.os.WNOHANG | tree.WALL
        calls.append("wait")
        raise ChildProcessError()

    def signal_disposition(sig, value):
        assert (sig, value) == (signal.SIGCHLD, signal.SIG_DFL)
        calls.append("disposition")

    def prepare():
        backend.prepare_calls += 1
        _BackendClass.prepare(probe)

    def pipe():
        assert calls == ["open", "send", "close", "disposition", "wait", 37, 36, 37]
        calls.append("pipe")
        return FakeBackend.pipe(backend)

    def forbidden(*args):
        pytest.fail("no real Linux syscall or numeric-PID signal allowed")

    probe._native_pidfd_open = open_self if native else None
    probe.libc = SimpleNamespace(syscall=syscall, prctl=prctl)
    probe.close = close
    backend.prepare, backend.pipe = prepare, pipe
    monkeypatch.setattr(tree.ctypes, "get_errno", lambda: 38)
    monkeypatch.setattr(tree.os, "getpid", lambda: SELF)
    monkeypatch.setattr(tree.os, "scandir", lambda path: nullcontext(iter([1])))
    monkeypatch.setattr(tree.os, "waitpid", waitpid)
    monkeypatch.setattr(tree.os, "fork", forbidden)
    monkeypatch.setattr(tree.os, "pipe", forbidden)
    monkeypatch.setattr(tree.os, "pipe2", forbidden, raising=False)
    monkeypatch.setattr(tree.os, "kill", forbidden)
    monkeypatch.setattr(tree.os, "killpg", forbidden)
    monkeypatch.setattr(tree.signal, "signal", signal_disposition)
    monkeypatch.setattr(tree.signal, "pidfd_send_signal", send, raising=False)
    report = run(backend)
    assert masks.blocked == prior
    if outcome == "success":
        assert report.success and report.launch_attempted and report.launch_released
        assert report.captures == 1 and report.reaps == 1
        assert calls == ["open", "send", "close", "disposition", "wait", 37, 36, 37, "pipe"]
        assert backend.events.index("pidfd") < backend.events.index("member")
        assert backend.events.index("member") < backend.events.index("release")
    else:
        assert not report.success and not report.ancestry_drained
        assert not report.launch_attempted and not report.launch_released
        assert report.root_pid is None and report.root_status is None
        assert report.captures == report.reaps == 0
        assert report.error and not backend.signals
        assert "pipe" not in backend.events and "fork" not in backend.events
        assert calls == (["open"] if outcome == "enosys" else ["open", "send", "close"])
        if outcome == "enosys":
            assert "38" in report.error
        elif outcome == "send-refused":
            assert "PermissionError" in report.error
        elif "interrupt" in outcome or outcome.startswith("pending-"):
            assert "KeyboardInterrupt" in report.error
        else:
            assert "OSError" in report.error
    assert calls.count("close") == (0 if outcome == "enosys" else 1)
    assert backend.closed.count(SELF_FD) == (0 if outcome == "enosys" else 1)
