"""Opt-in, single-shot Linux ancestry-domain experiment; NO admission/certificate.

Call only in a dedicated single-threaded process, with trusted responsive host
procfs, no other reapers, signal-fork handlers, helper threads/processes or external
brokered launches. argv/environment/cwd are copied fixed caller configuration;
stdin MUST already be DEVNULL, stdout/stderr an already merged inherited stream.
Only the child closes all FDs >2 (including saved-output FDs) before exec.

An independent controller/watchdog MUST survive this supervisor's death.
external_watchdog_present=True is caller trust, NOT checked evidence. There is no
parent-death guarantee, finite D-state cleanup guarantee, publication or deletion.
ECHILD describes only this ancestry domain, not external brokers/runtime safety.
setsid, nested namespaces and intermediate subreapers do not escape ancestry;
the exact runtime/helper behavior still requires independent observation.

The prior subreaper flag must be zero; successful setup irreversibly leaves it one
for this dedicated process. Never reset/reuse the process. timeout (<=600s) bounds
normal execution; cleanup gets at most min(timeout, 2s) extra, with 0.1s TERM grace.
Checks are cooperative: a blocked syscall requires the external watchdog.
Catchable INT/TERM are deferred during resource ownership transfers, without
changing caller dispositions. Restoring the mask may deliver a pending signal;
default SIGTERM still kills this supervisor, requiring the external watchdog.
Import performs no syscalls/state changes. _backend is a portable mock-test seam.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import ctypes
import math
import os
import platform
import signal
import sys
import time

WALL = 0x40000000
# Normal and cleanup each have this finite operation budget; reap/capture caps
# remain total. At 10ms sleeps this allows the entire 600s execution deadline.
MAX_REAPS, MAX_CAPTURES, MAX_LOOPS = 64, 256, 1_000_000


@contextmanager
def _defer_interrupts():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


@dataclass(frozen=True)
class TreeReport:
    root_pid: int | None
    root_status: int | None
    launch_attempted: bool
    launch_released: bool
    ancestry_drained: bool
    success: bool
    reaps: int
    captures: int
    error: str | None
    retain_work: bool = field(default=True, init=False)
    verified: bool = field(default=False, init=False)
    certificate: bool = field(default=False, init=False)


class _LinuxBackend:
    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    close = staticmethod(os.close)
    def __init__(self):
        if (sys.platform != "linux" or not hasattr(os, "waitid")
                or not hasattr(signal, "pidfd_send_signal")):
            raise RuntimeError("Linux pidfd/waitid support required")
        self.libc = ctypes.CDLL(None, use_errno=True)
        self._native_pidfd_open = getattr(os, "pidfd_open", None)
        if self._native_pidfd_open is None:
            if platform.machine() != "x86_64" or ctypes.sizeof(ctypes.c_void_p) != 8:
                raise RuntimeError("raw pidfd_open requires reviewed x86_64 ABI")
            self.libc.syscall.restype = ctypes.c_long

    def pidfd_open(self, pid):
        if self._native_pidfd_open is not None:
            return self._native_pidfd_open(pid, 0)
        # Linux x86_64 syscall table: pidfd_open=434; flags remain exactly zero.
        result = self.libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
        if result < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return result

    def prepare(self):
        # Probe capability only; the gated root still requires its own pidfd.
        with _defer_interrupts():
            fd = self.pidfd_open(os.getpid())
            try:
                self.send(fd, 0)
            finally:
                self.close(fd)
        with os.scandir("/proc/self/task") as tasks:
            if next(tasks, None) is None or next(tasks, None) is not None:
                raise RuntimeError("dedicated single-threaded process required")
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # Also clears NOCLDWAIT.
        try:
            os.waitpid(-1, os.WNOHANG | WALL)
        except ChildProcessError:
            pass
        else:
            raise RuntimeError("preexisting children forbidden (including zombies)")
        value = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 0:
            raise RuntimeError("prior subreaper must be zero")
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise RuntimeError("SET_CHILD_SUBREAPER failed")
        if self.libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
            raise RuntimeError("GET_CHILD_SUBREAPER failed")

    def pipe(self):
        return os.pipe2(os.O_CLOEXEC)

    def fork(self):
        return os.fork()

    def child(self, reader, writer, argv, environment, cwd):
        # No work/exec or release of caller FDs before the parent's pidfd gate.
        try:
            # The fork gate inherits the parent's temporary blocked mask.
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
            os.close(writer)
            if os.read(reader, 1) != b"G":
                os._exit(125)
            # Enumerate before closing: includes CLOEXEC descriptors as well.
            with os.scandir("/proc/self/fd") as entries:
                fds = [int(item.name) for item in entries if int(item.name) > 2]
            for fd in fds:
                try:
                    os.close(fd)
                except OSError as exc:
                    if exc.errno != 9:  # scandir's already closed directory FD.
                        raise
            os.chdir(cwd)
            os.execve(argv[0], argv, environment)
        except BaseException:
            os._exit(126)
        os._exit(126)

    def release(self, writer):
        if os.write(writer, b"G") != 1:
            raise RuntimeError("short gate write")

    def wait(self):
        return os.waitpid(-1, os.WNOHANG | WALL)

    def children(self):
        pid = os.getpid()
        with open(f"/proc/self/task/{pid}/children", "rb", buffering=0) as source:
            data = source.read(4097)
        tokens = data.split()
        if len(data) > 4096 or len(tokens) > MAX_CAPTURES:
            raise RuntimeError("bounded children listing exceeded")
        if any(not token.isdigit() or len(token) > 10 for token in tokens):
            raise RuntimeError("invalid children listing")
        pids = [int(token) for token in tokens]
        if any(not 1 <= pid < 2**31 for pid in pids) or len(set(pids)) != len(pids):
            raise RuntimeError("invalid child PID")
        return pids

    def member(self, pid):
        # None means a live, waitable child; a result means an unreaped zombie.
        # WNOWAIT is essential: retain identity until the serialized drain.
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT | WALL)
        # Keep acquisition, finally registration and close ownership atomic with
        # respect to catchable INT/TERM (the read is bounded on trusted procfs).
        with _defer_interrupts():
            fd = os.open(f"/proc/{pid}/stat", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                data = os.read(fd, 4097)
            finally:
                os.close(fd)
        fields = data[data.rfind(b")") + 2:].split()
        if (len(data) > 4096 or data.split(b" (", 1)[0] != str(pid).encode()
                or len(fields) < 2 or fields[1] != str(os.getpid()).encode()):
            raise RuntimeError("not a direct/adopted child")

    def send(self, fd, sig):
        signal.pidfd_send_signal(fd, sig, None, 0)


def run_tree(argv, environment, cwd, *, timeout, external_watchdog_present=False,
             cancel=None, _backend=None):
    """Run exactly one gated root; return unverified metadata, never publish.

    Bad configuration raises ValueError before setup. Setup/launch/cleanup failures
    return a failure report (launch may be unknown); retain work on every outcome.
    cancel is an optional trusted, nonblocking predicate. A cancellation/catchable
    interrupt requests bounded cleanup and remains failure even if ECHILD is later seen.
    Do not call from a library host: signal/subreaper state is intentionally changed.
    """
    if (external_watchdog_present is not True or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or not 0 < timeout <= 600
            or not isinstance(argv, (tuple, list)) or not argv
            or any(not isinstance(x, str) or "\0" in x for x in argv)
            or not os.path.isabs(argv[0]) or not isinstance(environment, dict)
            or any(not isinstance(k, str) or not k or "=" in k or "\0" in k
                   or not isinstance(v, str) or "\0" in v for k, v in environment.items())
            or not isinstance(cwd, str) or not os.path.isabs(cwd) or "\0" in cwd
            or (cancel is not None and not callable(cancel))):
        raise ValueError("fixed absolute launch, finite timeout and explicit watchdog trust required")
    argv, environment = tuple(argv), dict(environment)
    b, root, status, attempted, released = None, None, None, False, False
    reaps = captures = loops = 0
    drained = signalled = False
    error = None
    held, pipes, termed, killed = {}, [], set(), set()

    def fail(exc):
        nonlocal error
        if error is None:
            error = f"{type(exc).__name__}: {exc}"

    def close(owner, key):
        try:
            with _defer_interrupts():
                # Transfer ownership and close together; never retry a possibly
                # recycled descriptor, even when close or mask restoration fails.
                b.close(owner.pop(key))
        except BaseException as exc:
            fail(exc)

    def check(deadline):
        nonlocal loops
        loops += 1
        if loops > MAX_LOOPS or b.clock() >= deadline:
            raise TimeoutError("deadline/operation bound reached")

    def capture(pid, deadline):
        nonlocal captures
        check(deadline)
        if pid not in held:
            if captures >= MAX_CAPTURES:
                raise RuntimeError("capture bound reached")
            with _defer_interrupts():
                held[pid] = b.pidfd_open(pid)
                captures += 1
        check(deadline)
        b.member(pid)
        check(deadline)

    def drain(deadline):
        nonlocal status, reaps, drained
        while True:
            check(deadline)
            if reaps >= MAX_REAPS:
                raise RuntimeError("reap bound reached")
            try:
                pid, result = b.wait()
            except ChildProcessError:
                # Root was observed, and gate can never launch another root.
                drained = status is not None and not pipes
                check(deadline)
                return
            if pid == 0:
                return  # Children remain; NOT an empty proof.
            reaps += 1
            termed.discard(pid)
            killed.discard(pid)  # A later adopted child may reuse this number.
            if pid == root and status is None:
                status = result
            if pid in held:
                close(held, pid)

    try:
        b = _backend if _backend is not None else _LinuxBackend()
        start = b.clock()
        deadline = start + timeout
        cleanup_end = deadline + min(2.0, timeout)
        check(deadline)
        b.prepare()  # GET/SET/GET and ECHILD all precede the sole fork.
        check(deadline)
        with _defer_interrupts():
            pipes.extend(b.pipe())
        check(deadline)
        with _defer_interrupts():
            attempted = True  # A failed fork is conservatively launch-unknown.
            root = b.fork()
            if root == 0:
                b.child(*pipes, argv, environment, cwd)
                raise RuntimeError("child backend returned")
        if root < 1:
            raise RuntimeError("invalid fork result")
        close(pipes, 0)
        if error is not None:
            raise RuntimeError("gate descriptor close failed")
        capture(root, deadline)  # Still direct; no intervening reaper/PID reuse.
        check(deadline)
        b.release(pipes[0])
        released = True
        close(pipes, -1)  # Irreversibly spend the gate before any drain.
        while status is None:
            check(deadline)
            if cancel is not None and cancel() is True:
                raise InterruptedError("cancelled")
            drain(deadline)
            if status is None:
                b.sleep(min(0.01, max(0, deadline - b.clock())))
    except BaseException as exc:
        fail(exc)
    finally:
        for _ in range(len(pipes)):
            close(pipes, -1)  # EOF prevents exec if release never happened.

    if b is not None and root is not None and root > 0:
        try:
            # STOPPING gets a fresh operation budget even after normal exhaustion.
            # Do not reset the total reap/capture caps or the absolute wall limit.
            loops = 0
            stop = b.clock()
            end = min(cleanup_end, stop + min(2.0, timeout))
            escalate = stop + min(0.1, timeout / 2)
            while not drained:
                # Retry recoverable interrupts within this SAME wall/operation
                # budget. Keep this check outside the retry handler so repeated
                # interrupts cannot hide exhaustion; close is never retried.
                check(end)
                try:
                    drain(end)
                    if drained:
                        break
                    check(end)
                    frontier = b.children()  # Cleanup aid ONLY, never empty proof.
                    if status is None and root not in frontier:
                        frontier = [root, *frontier]  # Root need not await proc listing.
                    check(end)
                    for pid in frontier:
                        capture(pid, end)
                        sig = signal.SIGKILL if b.clock() >= escalate else signal.SIGTERM
                        sent = killed if sig == signal.SIGKILL else termed
                        if pid not in sent:
                            check(end)
                            b.member(pid)  # Revalidate immediately before pidfd signal.
                            check(end)
                            try:
                                b.send(held[pid], sig)
                            except ProcessLookupError:
                                pass  # Exit raced; only a later ECHILD can drain.
                            signalled = True
                            sent.add(pid)
                    b.sleep(min(0.01, max(0, end - b.clock())))
                except (KeyboardInterrupt, SystemExit, InterruptedError) as exc:
                    fail(exc)  # Even a later ECHILD cannot turn this into success.
        except BaseException as exc:
            fail(exc)
    for pid in list(held):
        close(held, pid)
    if status is not None and status != 0:
        fail(RuntimeError("root exited nonzero or by signal"))
    if signalled:
        fail(RuntimeError("termination signals required"))
    if drained and error is None and b.clock() >= deadline:
        fail(TimeoutError("ancestry drain not completed before execution deadline"))
    success = released and status == 0 and drained and error is None
    return TreeReport(root, status, attempted, released, drained, success,
                      reaps, captures, error)
