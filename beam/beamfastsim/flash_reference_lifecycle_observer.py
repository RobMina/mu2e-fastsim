"""Read-only Linux identity observations; NOT admission or termination evidence.

Caller preconditions: /proc is a trusted, responsive HOST procfs; the supplied
host PID is independently invocation-bound (a nonce/ready file cannot establish
this). expected_namespace comes from an independently trusted LIVE invocation
member, with its namespace FD kept pinned by the caller throughout capture.
A device/inode pair alone does not keep a namespace alive or authenticate a PID.
No namespace discovery, scanning, signals, controller, or publication is provided.

Open the proc directory BEFORE pidfd_open: if that numeric PID is reused between
opens, reads through the old pinned directory fail, even for same-tick starttime
reuse. Starttime comparison alone is insufficient. Hold proc, namespace and pidfd
until explicit close. Namespace magic-link following is intentional; procdir and
ordinary reads are no-follow. Unsupported operations/errors/races fail closed.
Deadlines are cooperative checks, not cancellation of a blocked kernel syscall.
Readiness describes ONLY the captured process, never all descendants, startup,
helpers, cgroups, or an invocation. Identity fields describe capture time, not
renewed invocation membership. All reports are unverified/non-certificates.
"""

from dataclasses import dataclass, field, replace
import math
import os
import select
import sys
import time

STAT_LIMIT = 4096
STATUS_LIMIT = 65536


class ObservationError(ValueError):
    """Identity could not be established/observed; no fallback is permitted."""


@dataclass(frozen=True)
class NamespaceIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class Observation:
    host_pid: int
    local_pid: int
    starttime: int
    namespace: NamespaceIdentity
    captured_process_exited: bool
    verified: bool = field(default=False, init=False)
    certificate: bool = field(default=False, init=False)


def _unsigned(token, maximum, minimum=0):
    if not token or not token.isdigit() or len(token) > 20:
        raise ObservationError("invalid bounded unsigned integer")
    value = int(token)
    if not minimum <= value <= maximum:
        raise ObservationError("integer out of range")
    return value


def parse_stat(data):
    """Return (host PID, field-22 starttime); comm may contain ')' and newlines."""
    if not isinstance(data, bytes) or not 0 < len(data) <= STAT_LIMIT:
        raise ObservationError("stat size/type")
    opening, closing = data.find(b" ("), data.rfind(b")")
    if opening < 1 or closing < opening + 2 or data[closing + 1:closing + 2] != b" ":
        raise ObservationError("stat comm framing")
    pid = _unsigned(data[:opening], 2**31 - 1, 1)
    fields = data[closing + 2:].split()
    if len(fields) < 20 or len(fields[0]) != 1 or fields[0] not in b"R S D Z T W t X x K P I N".split():
        raise ObservationError("stat fields/state")
    for token in fields[1:19]:
        _unsigned(token.removeprefix(b"-"), 2**64 - 1)
    return pid, _unsigned(fields[19], 2**64 - 1)


def parse_nspid(data):
    """Return bounded host-to-local NSpid values; missing/duplicate fields fail."""
    if not isinstance(data, bytes) or not 0 < len(data) <= STATUS_LIMIT:
        raise ObservationError("status size/type")
    rows = []
    for line in data.splitlines():
        key, separator, value = line.partition(b":")
        if key.strip() == b"NSpid":
            if not separator:
                raise ObservationError("malformed NSpid")
            rows.append(value.split())
    if len(rows) != 1 or not 1 <= len(rows[0]) <= 32:
        raise ObservationError("missing, duplicate, or oversized NSpid")
    return tuple(_unsigned(token, 2**31 - 1, 1) for token in rows[0])


def _check(backend, deadline):
    if not math.isfinite(deadline) or backend.clock() >= deadline:
        raise ObservationError("deadline expired/invalid")


def _close_all(backend, fds):
    failed, interrupted = False, None
    while fds:
        try:
            backend.close(fds.pop())
        except BaseException as exc:
            failed = True  # Do not retry close: the descriptor may have been reused.
            if not isinstance(exc, Exception) and interrupted is None:
                interrupted = exc
    if interrupted is not None:
        raise interrupted
    if failed:
        raise ObservationError("descriptor close failed")


class _LinuxBackend:
    clock = staticmethod(time.monotonic)

    def close(self, fd):
        os.close(fd)

    def __init__(self):
        if sys.platform != "linux" or not hasattr(os, "pidfd_open"):
            raise ObservationError("Linux pidfd support required")
        self.flags = os.O_RDONLY | os.O_CLOEXEC

    def open_proc(self, pid):
        return os.open(f"/proc/{pid}", self.flags | os.O_DIRECTORY | os.O_NOFOLLOW)

    def pidfd_open(self, pid):
        return os.pidfd_open(pid, 0)

    def read(self, parent, name, limit, check):
        check()
        fd = os.open(name, self.flags | os.O_NOFOLLOW, dir_fd=parent)
        try:
            data = bytearray()
            while len(data) <= limit:
                check()
                chunk = os.read(fd, limit + 1 - len(data))
                check()
                if not chunk:
                    return bytes(data)
                data.extend(chunk)
            raise ObservationError("proc read exceeds bound")
        finally:
            os.close(fd)

    def open_namespace(self, parent):
        directory = os.open("ns", self.flags | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent)
        pin = None
        try:
            pin = os.open("pid", self.flags, dir_fd=directory)
        finally:
            try:
                os.close(directory)
            except BaseException:
                if pin is not None:
                    os.close(pin)
                raise
        return pin

    def identity(self, fd):
        info = os.fstat(fd)
        return NamespaceIdentity(info.st_dev, info.st_ino)

    def ready(self, fd):
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        events = poller.poll(0)
        if any(mask & ~(select.POLLIN | select.POLLHUP) for _, mask in events):
            raise ObservationError("pidfd poll error")
        return bool(events)


class CapturedProcess:
    """Owns three descriptors. Explicit close is required; no numeric PID actions."""

    def __init__(self, backend, fds, report):
        self._backend, self._fds, self._report = backend, fds, report

    def observe(self, *, deadline):
        """Nonblocking point observation; errors close all retained descriptors."""
        try:
            if not self._fds:
                raise ObservationError("observer closed")
            _check(self._backend, deadline)
            exited = self._backend.ready(self._fds[1])
            _check(self._backend, deadline)
            return replace(self._report, captured_process_exited=exited)
        except BaseException as exc:
            try:
                self.close()
            except Exception:
                pass
            if not isinstance(exc, Exception):
                raise
            raise ObservationError("observation failed") from exc

    def close(self):
        _close_all(self._backend, self._fds)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def capture(host_pid, expected_namespace, *, deadline, trusted_host_proc=False,
            _backend=None):
    """Capture a caller-bound PID, NOT authenticate its invocation membership.

    trusted_host_proc=True explicitly asserts the module's caller preconditions.
    _backend is solely a portable test seam; production uses the Linux backend.
    """
    if (type(host_pid) is not int or not 1 <= host_pid <= 2**31 - 1
            or trusted_host_proc is not True
            or not isinstance(expected_namespace, NamespaceIdentity)
            or type(expected_namespace.device) is not int
            or type(expected_namespace.inode) is not int
            or expected_namespace.device < 0 or expected_namespace.inode <= 0):
        raise ObservationError("trusted host PID and independently pinned namespace required")
    backend, fds = _backend if _backend is not None else _LinuxBackend(), []
    check = lambda: _check(backend, deadline)
    try:
        check()
        fds.append(backend.open_proc(host_pid))  # MUST precede numeric pidfd_open.
        check()
        fds.append(backend.pidfd_open(host_pid))
        check()
        first = parse_stat(backend.read(fds[0], "stat", STAT_LIMIT, check))
        nspid = parse_nspid(backend.read(fds[0], "status", STATUS_LIMIT, check))
        check()
        fds.append(backend.open_namespace(fds[0]))
        check()
        if backend.identity(fds[2]) != expected_namespace or first[0] != host_pid or nspid[0] != host_pid:
            raise ObservationError("host/namespace mismatch")
        check()
        if backend.ready(fds[1]):
            raise ObservationError("captured process already exited")
        check()
        second = parse_stat(backend.read(fds[0], "stat", STAT_LIMIT, check))
        again = parse_nspid(backend.read(fds[0], "status", STATUS_LIMIT, check))
        check()
        fds.append(backend.open_namespace(fds[0]))
        check()
        if backend.identity(fds[3]) != expected_namespace or second != first or again != nspid:
            raise ObservationError("identity changed during capture")
        backend.close(fds.pop())
        check()
        if backend.ready(fds[1]):
            raise ObservationError("captured process exited during capture")
        check()
        return CapturedProcess(backend, fds, Observation(
            host_pid, nspid[-1], first[1], expected_namespace, False))
    except BaseException as exc:
        try:
            _close_all(backend, fds)
        except Exception:
            pass
        if not isinstance(exc, Exception):
            raise
        raise ObservationError("capture failed") from exc
