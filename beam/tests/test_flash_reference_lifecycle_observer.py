"""Portable mock-only identity tests; no real proc, pidfds, forks or signals."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from beamfastsim import flash_reference_lifecycle_observer as observer

NS = observer.NamespaceIdentity(7, 42)
PID = 123


def stat_bytes(pid=PID, start=55, comm=b"a ) tricky (comm)\n ) end"):
    # Tokens start at field 3; field 22 is index 19 after comm.
    return str(pid).encode() + b" (" + comm + b") S " + b"0 " * 18 + str(start).encode() + b" 0\n"


class FakeBackend:
    def __init__(self):
        self.live, self.closed, self.events = set(), [], []
        self.next_fd, self.calls, self.now = 10, 0, 0.0
        self.fail_at = self.expire_at = None
        self.stats = [stat_bytes(), stat_bytes()]
        self.statuses = [b"Name:\ttest\nNSpid:\t123 9\n"] * 2
        self.namespaces = [NS, NS]
        self.readiness = [False, False, False]
        self.reuse_between_opens = False
        self.stale = False

    def clock(self):
        return self.now

    def step(self, event):
        self.events.append(event)
        self.calls += 1
        if self.calls == self.expire_at:
            self.now = 10.0
        if self.calls == self.fail_at:
            raise OSError("injected unavailable/race")

    def allocate(self):
        fd = self.next_fd
        self.next_fd += 1
        self.live.add(fd)
        return fd

    def open_proc(self, pid):
        assert pid == PID
        self.step("proc")
        return self.allocate()

    def pidfd_open(self, pid):
        assert pid == PID and 10 in self.live
        self.step("pidfd")
        self.stale = self.reuse_between_opens
        return self.allocate()

    def read(self, fd, name, limit, check):
        assert fd == 10 and fd in self.live
        assert limit == {"stat": observer.STAT_LIMIT, "status": observer.STATUS_LIMIT}[name]
        check()
        self.step(name)
        if self.stale:
            # Old pinned proc dir is dead: MUST NOT re-open /proc/numeric-PID.
            raise ProcessLookupError("old pinned proc, even if replacement has same starttime")
        result = (self.stats if name == "stat" else self.statuses).pop(0)
        check()
        return result

    def open_namespace(self, fd):
        assert fd == 10 and fd in self.live
        self.step("namespace")
        return self.allocate()

    def identity(self, fd):
        assert fd in self.live
        self.step("identity")
        return self.namespaces.pop(0)

    def ready(self, fd):
        assert fd == 11 and fd in self.live
        self.step("ready")
        return self.readiness.pop(0)

    def close(self, fd):
        assert fd in self.live
        self.live.remove(fd)
        self.closed.append(fd)


def capture(backend, **kwargs):
    args = dict(deadline=10.0, trusted_host_proc=True, _backend=backend)
    args.update(kwargs)
    return observer.capture(PID, NS, **args)


def reject(backend, **kwargs):
    with pytest.raises(observer.ObservationError):
        capture(backend, **kwargs)
    assert not backend.live
    assert len(backend.closed) == len(set(backend.closed))


def test_parsers_complex_comm_and_host_to_local_order():
    assert observer.parse_stat(stat_bytes()) == (PID, 55)
    assert observer.parse_nspid(b"Name: x\nNSpid:\t123 28 9\n") == (123, 28, 9)
    assert observer.parse_nspid(b"NSpid: 123\n") == (123,)


@pytest.mark.parametrize("data", [
    b"", "not bytes", b"123 no comm", b"123 (x) S 0", stat_bytes(pid=0),
    stat_bytes(start=-1), stat_bytes(start=2**64), b"x" * (observer.STAT_LIMIT + 1),
    stat_bytes().replace(b") S ", b") QQ "),
    stat_bytes().replace(b") S 0", b") S --1"),
])
def test_stat_rejects_malformed_and_bounded(data):
    with pytest.raises(observer.ObservationError):
        observer.parse_stat(data)


@pytest.mark.parametrize("data", [
    b"", "NSpid: 123", b"Name: x\n", b"NSpid", b"NSpid:\n", b"NSpid: 0\n",
    b"NSpid: -1\n", b"NSpid: 2147483648\n", b"NSpid: 12x\n",
    b"NSpid: 123\nNSpid: 123\n", b"NSpid: 123\n NSpid : 9\n",
    b"NSpid: " + b"1 " * 33, b"NSpid: 123\n" + b"x" * observer.STATUS_LIMIT,
])
def test_nspid_rejects_missing_duplicate_and_bounds(data):
    with pytest.raises(observer.ObservationError):
        observer.parse_nspid(data)


def test_retained_descriptors_report_and_explicit_idempotent_close():
    backend = FakeBackend()
    process = capture(backend)
    assert backend.events == ["proc", "pidfd", "stat", "status", "namespace",
                              "identity", "ready", "stat", "status", "namespace",
                              "identity", "ready"]
    assert backend.live == {10, 11, 12}  # Second namespace read's FD was closed.
    report = process.observe(deadline=10)
    assert (report.host_pid, report.local_pid, report.starttime, report.namespace) == (PID, 9, 55, NS)
    assert not report.captured_process_exited
    assert report.verified is False and report.certificate is False
    with pytest.raises(FrozenInstanceError):
        report.verified = True
    process.close()
    process.close()
    assert not backend.live
    with pytest.raises(observer.ObservationError):
        process.observe(deadline=10)


def test_readiness_is_only_captured_process_exit_not_descendant_proof():
    backend = FakeBackend()
    backend.readiness = [False, False, True]
    with capture(backend) as process:
        report = process.observe(deadline=10)
        assert report.captured_process_exited is True
        assert report.verified is False and report.certificate is False
        assert backend.live == {10, 11, 12}
    assert not backend.live


@pytest.mark.parametrize("which", [0, 1])
def test_equal_starttime_but_pidfd_ready_rejected(which):
    backend = FakeBackend()
    backend.readiness[which] = True
    reject(backend)


def test_same_tick_reuse_between_proc_and_pidfd_cannot_bind_replacement():
    backend = FakeBackend()
    backend.reuse_between_opens = True
    # Numeric replacement would return the very same stat/starttime if reopened.
    reject(backend)
    assert backend.events == ["proc", "pidfd", "stat"]


@pytest.mark.parametrize("change", ["namespace_first", "namespace_second", "host",
                                    "nspid_host", "starttime", "nspid_local"])
def test_identity_mismatches_close_every_descriptor(change):
    backend = FakeBackend()
    if change.startswith("namespace"):
        backend.namespaces[change == "namespace_second"] = observer.NamespaceIdentity(7, 43)
    elif change == "host":
        backend.stats[0] = stat_bytes(pid=124)
    elif change == "nspid_host":
        backend.statuses[0] = b"NSpid: 124 9\n"
    elif change == "starttime":
        backend.stats[1] = stat_bytes(start=56)
    else:
        backend.statuses[1] = b"NSpid: 123 10\n"
    reject(backend)


@pytest.mark.parametrize("step", range(1, 13))
def test_every_capture_operation_error_fails_closed(step):
    backend = FakeBackend()
    backend.fail_at = step
    reject(backend)


@pytest.mark.parametrize("step", range(1, 13))
def test_deadline_checked_throughout_capture(step):
    backend = FakeBackend()
    backend.expire_at = step
    reject(backend)


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf")])
def test_invalid_or_expired_deadline_does_not_open(deadline):
    backend = FakeBackend()
    reject(backend, deadline=deadline)
    assert not backend.events


def test_observation_error_and_deadline_close_all():
    for expire in (False, True):
        backend = FakeBackend()
        process = capture(backend)
        if expire:
            backend.now = 10
        else:
            backend.fail_at = backend.calls + 1
        with pytest.raises(observer.ObservationError):
            process.observe(deadline=10)
        assert not backend.live


def test_explicit_trust_required_and_pid_not_self_certifying():
    backend = FakeBackend()
    reject(backend, trusted_host_proc=False)
    for pid in (True, 0, -1, 2**31, "123"):
        with pytest.raises(observer.ObservationError):
            observer.capture(pid, NS, deadline=10, trusted_host_proc=True, _backend=backend)
    assert not backend.events


def test_unavailable_linux_support_never_falls_back(monkeypatch):
    monkeypatch.setattr(observer.sys, "platform", "not-linux")
    with pytest.raises(observer.ObservationError):
        observer.capture(PID, NS, deadline=10, trusted_host_proc=True)


def test_linux_adapter_descriptor_relative_bounded_reads_and_namespace_pin(monkeypatch):
    # Call adapter methods with ALL OS operations mocked; portable even off Linux.
    backend = object.__new__(observer._LinuxBackend)
    backend.flags = 1
    for name, value in (("O_NOFOLLOW", 2), ("O_DIRECTORY", 4)):
        monkeypatch.setattr(observer.os, name, value, raising=False)
    opened, closed, reads = [], [], []
    chunks = [b"abc", b""]

    def open_fd(path, flags, **kwargs):
        opened.append((path, flags, kwargs))
        return 100 + len(opened)

    def read_fd(fd, size):
        reads.append((fd, size))
        return chunks.pop(0)

    monkeypatch.setattr(observer.os, "open", open_fd)
    monkeypatch.setattr(observer.os, "read", read_fd)
    monkeypatch.setattr(observer.os, "close", closed.append)
    monkeypatch.setattr(observer.os, "fstat", lambda fd: SimpleNamespace(st_dev=7, st_ino=42))
    assert backend.read(10, "stat", 3, lambda: None) == b"abc"
    assert opened == [("stat", 3, {"dir_fd": 10})]
    assert reads == [(101, 4), (101, 1)] and closed == [101]
    chunks[:] = [b"abcd"]
    with pytest.raises(observer.ObservationError):
        backend.read(10, "status", 3, lambda: None)
    assert closed[-1] == 102
    nsfd = backend.open_namespace(10)
    assert opened[-2:] == [("ns", 7, {"dir_fd": 10}), ("pid", 1, {"dir_fd": 103})]
    assert nsfd == 104 and closed[-1] == 103
    assert backend.identity(nsfd) == NS
    backend.close(nsfd)
    assert closed[-1] == 104


def test_linux_adapter_numeric_opens_are_ordered_by_capture_and_nofollow(monkeypatch):
    backend = object.__new__(observer._LinuxBackend)
    backend.flags = 1
    monkeypatch.setattr(observer.os, "O_DIRECTORY", 4, raising=False)
    monkeypatch.setattr(observer.os, "O_NOFOLLOW", 2, raising=False)
    events = []
    monkeypatch.setattr(observer.os, "open", lambda *args: events.append(args) or 10)
    monkeypatch.setattr(observer.os, "pidfd_open", lambda *args: events.append(args) or 11,
                        raising=False)
    assert backend.open_proc(PID) == 10
    assert backend.pidfd_open(PID) == 11
    assert events == [("/proc/123", 7), (123, 0)]


@pytest.mark.parametrize("events, expected", [([], False), ([(11, 1)], True),
                                               ([(11, 16)], True), ([(11, 32)], None)])
def test_linux_adapter_poll_is_nonblocking_and_rejects_errors(monkeypatch, events, expected):
    calls = []
    poller = SimpleNamespace(register=lambda *args: calls.append(args),
                             poll=lambda timeout: calls.append(timeout) or events)
    monkeypatch.setattr(observer.select, "poll", lambda: poller, raising=False)
    for name, value in (("POLLIN", 1), ("POLLHUP", 16)):
        monkeypatch.setattr(observer.select, name, value, raising=False)
    backend = object.__new__(observer._LinuxBackend)
    if expected is None:
        with pytest.raises(observer.ObservationError):
            backend.ready(11)
    else:
        assert backend.ready(11) is expected
    assert calls == [(11, 1), 0]


def test_linux_adapter_closes_temporary_fds_on_read_or_namespace_error(monkeypatch):
    backend = object.__new__(observer._LinuxBackend)
    backend.flags = 1
    monkeypatch.setattr(observer.os, "O_DIRECTORY", 4, raising=False)
    monkeypatch.setattr(observer.os, "O_NOFOLLOW", 2, raising=False)
    closed = []
    monkeypatch.setattr(observer.os, "close", closed.append)
    monkeypatch.setattr(observer.os, "open", lambda *args, **kwargs: 99)

    def fail(*args, **kwargs):
        raise OSError("injected")

    monkeypatch.setattr(observer.os, "read", fail)
    with pytest.raises(OSError):
        backend.read(10, "stat", 5, lambda: None)
    assert closed == [99]

    def open_namespace(path, *args, **kwargs):
        return 98 if path == "ns" else fail()

    monkeypatch.setattr(observer.os, "open", open_namespace)
    with pytest.raises(OSError):
        backend.open_namespace(10)
    assert closed == [99, 98]


def test_namespace_directory_close_error_still_closes_new_pin(monkeypatch):
    backend = object.__new__(observer._LinuxBackend)
    backend.flags = 1
    monkeypatch.setattr(observer.os, "O_DIRECTORY", 4, raising=False)
    monkeypatch.setattr(observer.os, "O_NOFOLLOW", 2, raising=False)
    monkeypatch.setattr(observer.os, "open", lambda path, *a, **kw: 98 if path == "ns" else 99)
    closed = []

    def close(fd):
        closed.append(fd)
        if fd == 98:
            raise OSError("close failed")

    monkeypatch.setattr(observer.os, "close", close)
    with pytest.raises(OSError):
        backend.open_namespace(10)
    assert closed == [98, 99]


@pytest.mark.parametrize("namespace", [None, (7, 42), observer.NamespaceIdentity(-1, 42),
                                       observer.NamespaceIdentity(7, 0),
                                       observer.NamespaceIdentity(True, 42)])
def test_invalid_expected_namespace_rejected_without_os_operations(namespace):
    backend = FakeBackend()
    with pytest.raises(observer.ObservationError):
        observer.capture(PID, namespace, deadline=10, trusted_host_proc=True, _backend=backend)
    assert not backend.events


def test_close_error_still_attempts_all_descriptors_without_retry():
    backend = FakeBackend()
    process = capture(backend)
    close = backend.close

    def close_with_error(fd):
        close(fd)
        if fd == 12:
            raise OSError("injected close error; never retry possibly recycled fd")

    backend.close = close_with_error
    with pytest.raises(observer.ObservationError):
        process.close()
    assert not backend.live
    assert backend.closed == [13, 12, 11, 10]
    process.close()


def test_interrupted_capture_and_observation_release_descriptors():
    for during_capture in (True, False):
        backend = FakeBackend()
        process = None if during_capture else capture(backend)

        def interrupt(fd):
            raise KeyboardInterrupt

        backend.ready = interrupt
        with pytest.raises(KeyboardInterrupt):
            capture(backend) if during_capture else process.observe(deadline=10)
        assert not backend.live


def test_parser_exact_limits_and_maximum_namespace_depth():
    data = stat_bytes(start=2**64 - 1)
    data += b" " * (observer.STAT_LIMIT - len(data))
    assert observer.parse_stat(data) == (PID, 2**64 - 1)
    status = b"NSpid: " + b"1 " * 32 + b"\n"
    status += b"x" * (observer.STATUS_LIMIT - len(status))
    assert observer.parse_nspid(status) == (1,) * 32


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('operation', ['close', 'capture-failure', 'observe-failure', 'context-exit'])
def test_cleanup_interruption_drains_remaining_descriptors(operation, interruption):
    backend = FakeBackend()
    process = None if operation == 'capture-failure' else capture(backend)
    if operation == 'capture-failure':
        backend.fail_at = 6
    elif operation == 'observe-failure':
        backend.fail_at = backend.calls + 1
    close = backend.close

    def interrupt_close(fd):
        close(fd)
        if fd == 12:
            raise interruption('injected cleanup interruption')

    backend.close = interrupt_close
    with pytest.raises(interruption, match='injected cleanup interruption'):
        if operation == 'capture-failure':
            capture(backend)
        elif operation == 'observe-failure':
            process.observe(deadline=10)
        elif operation == 'context-exit':
            with process:
                pass
        else:
            process.close()
    assert not backend.live
    assert len(backend.closed) == len(set(backend.closed))
    assert backend.closed[-3:] == [12, 11, 10]
