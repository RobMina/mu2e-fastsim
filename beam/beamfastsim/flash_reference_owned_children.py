"""Unlaunched Linux 4.18 owned-child diagnostic; never a certificate.

Use once in a fresh dedicated single-threaded host-namespace process, with trusted
host procfs, no preexisting children, tracees, competing reapers, brokered launches
or callbacks/handlers that fork, wait or change SIGCHLD. Only the gated child may
exec; inherited stdin must be DEVNULL and stdout/stderr an already merged stream.
The child closes every FD above stderr before exec. Subreaper state is permanent.

Numeric signals rely on unreaped child ownership, NOT proc start times: real
parent, thread-group and tracer checks precede WNOWAIT|__WALL membership, and a
live result is immediately signalled without a reap or user callback. An exit in
that interval pins the PID as a zombie under the stated SIGCHLD/reaper contract.
No lease survives a reap. Enumeration is only a cleanup aid, never empty proof.

An independent surviving external watchdog is mandatory caller trust, not checked
evidence. All deadlines/syscalls are cooperative, not D-state bounds. Cleanup has
at most min(timeout, 2s) extra and 0.1s TERM grace. No runtime, confinement,
publication, deletion, transport or supervisor-death guarantee is supplied.
Import changes no state; _backend exists only for portable mocked tests.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import ctypes
import math
import os
import signal
import sys
import time

WALL = 0x40000000
MAX_REAPS, MAX_SIGNALS, MAX_CHILDREN, MAX_OPS = 64, 256, 256, 1_000_000


@contextmanager
def _masked():
    # Snapshot without changing the mask: blocking itself can apply then raise.
    prior = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        yield
    finally:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, prior)
        except BaseException:
            # One bounded, idempotent retry; preserve the first failure if it works.
            signal.pthread_sigmask(signal.SIG_SETMASK, prior)
            raise


@dataclass(frozen=True)
class OwnedTreeReport:
    root_pid: int | None
    root_status: int | None
    attempted: bool
    released: bool
    ancestry_drained: bool
    success: bool
    reaps: int
    signals: int
    error: str | None
    retain_work: bool = field(default=True, init=False)
    verified: bool = field(default=False, init=False)
    certificate: bool = field(default=False, init=False)


def _pid(pid):
    if type(pid) is not int or not 0 < pid < 2**31:
        raise RuntimeError("invalid positive PID")
    return pid


class _LinuxBackend:
    clock, sleep, close = staticmethod(time.monotonic), staticmethod(time.sleep), staticmethod(os.close)

    def __init__(self):
        if sys.platform != "linux" or not hasattr(os, "waitid"):
            raise RuntimeError("Linux waitid required")
        self.libc = ctypes.CDLL(None, use_errno=True)

    def status(self, pid, *, require_host=False):
        with open(f"/proc/{_pid(pid)}/status", "rb", buffering=0) as source:
            data = source.read(65537)
        if len(data) > 65536:
            raise RuntimeError("status read bound exceeded")
        fields = {}
        for line in data.splitlines():
            key, sep, value = line.partition(b":")
            if sep and key in (b"Pid", b"PPid", b"Tgid", b"TracerPid", b"NSpid"):
                if key in fields:
                    raise RuntimeError("duplicate status field")
                fields[key] = value.split()
        for key in (b"Pid", b"PPid", b"Tgid", b"TracerPid"):
            if len(fields.get(key, ())) != 1 or not fields[key][0].isdigit():
                raise RuntimeError("invalid status field")
        nspids = fields.get(b"NSpid", ())
        if (not 1 <= len(nspids) <= 32
                or any(not token.isdigit() or len(token) > 10
                       or not 0 < int(token) < 2**31 for token in nspids)):
            raise RuntimeError("missing/invalid NSpid sequence")
        # Only self must be singleton; nested child NSpids never supply targets.
        if (int(fields[b"Pid"][0]) != pid or int(fields[b"Tgid"][0]) != pid
                or int(fields[b"TracerPid"][0]) != 0
                or int(nspids[0]) != pid or (require_host and len(nspids) != 1)):
            raise RuntimeError("thread/trace/namespace identity refused")
        return int(fields[b"PPid"][0])

    def prepare(self):
        with os.scandir("/proc/self/task") as tasks:
            first = next(tasks, None)
            if first is None or first.name != str(os.getpid()) or next(tasks, None) is not None:
                raise RuntimeError("dedicated single thread required")
        self.status(os.getpid(), require_host=True)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        try:
            os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT | WALL)
        except ChildProcessError:
            pass
        else:
            raise RuntimeError("preexisting children forbidden")
        value = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 0:
            raise RuntimeError("prior subreaper must be zero")
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise RuntimeError("SET_CHILD_SUBREAPER failed")
        if self.libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
            raise RuntimeError("GET_CHILD_SUBREAPER failed")

    def member(self, pid, sig=None):
        _pid(pid)
        if sig not in (None, signal.SIGTERM, signal.SIGKILL):
            raise RuntimeError("invalid signal")
        with _masked():
            try:
                parent = self.status(pid)
            except FileNotFoundError as exc:
                # Missing status skips only on ECHILD, never authorizes a signal.
                try:
                    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT | WALL)
                except ChildProcessError:
                    return None
                raise RuntimeError("missing child status without ECHILD") from exc
            if parent != os.getpid():
                raise RuntimeError("not a real direct/adopted child")
            try:
                event = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT | WALL)
            except ChildProcessError:
                return None
            if event is not None and event.si_pid != 0:
                if (event.si_pid != pid or event.si_code not in
                        (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED)):
                    raise RuntimeError("unexpected waitid PID/terminal code")
                return False
            if sig is not None:
                os.kill(pid, sig)
            return True

    def pipe(self):
        return os.pipe2(os.O_CLOEXEC)

    def fork(self):
        return os.fork()

    def release(self, writer):
        if os.write(writer, b"G") != 1:
            raise RuntimeError("short gate write")

    def child(self, reader, writer, argv, environment, cwd):
        try:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
            os.close(writer)
            if os.read(reader, 1) != b"G":
                os._exit(125)
            with os.scandir("/proc/self/fd") as entries:
                fds = [int(entry.name) for entry in entries if int(entry.name) > 2]
            for fd in fds:
                try:
                    os.close(fd)
                except OSError as exc:
                    if exc.errno != 9:
                        raise
            os.chdir(cwd)
            os.execve(argv[0], argv, environment)
        except BaseException:
            os._exit(126)
        os._exit(126)

    def wait(self):
        return os.waitpid(-1, os.WNOHANG | WALL)

    def children(self):
        with open(f"/proc/self/task/{os.getpid()}/children", "rb", buffering=0) as source:
            data = source.read(4097)
        tokens = data.split()
        if (len(data) > 4096 or len(tokens) > MAX_CHILDREN
                or any(not token.isdigit() or len(token) > 10 for token in tokens)):
            raise RuntimeError("children listing bound/format")
        pids = [_pid(int(token)) for token in tokens]
        if len(set(pids)) != len(pids):
            raise RuntimeError("duplicate children")
        return pids


def run_owned_tree(argv, environment, cwd, *, timeout, external_watchdog_present=False,
                   _backend=None):
    """Run one fixed gated root and report unverified ancestry metadata only."""
    if (external_watchdog_present is not True or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or not 0 < timeout <= 600
            or not isinstance(argv, (tuple, list)) or not argv
            or any(not isinstance(x, str) or "\0" in x for x in argv)
            or not os.path.isabs(argv[0]) or not isinstance(environment, dict)
            or any(not isinstance(k, str) or not k or "=" in k or "\0" in k
                   or not isinstance(v, str) or "\0" in v for k, v in environment.items())
            or not isinstance(cwd, str) or not os.path.isabs(cwd) or "\0" in cwd):
        raise ValueError("fixed absolute launch, finite timeout and explicit watchdog required")
    argv, environment = tuple(argv), dict(environment)
    b = root = status = error = None
    attempted = released = drained = False
    reaps = signals = signal_attempts = ops = 0
    pipes, termed, killed = [], set(), set()
    gate_ok = True

    def fail(exc):
        nonlocal error
        if error is None:
            error = f"{type(exc).__name__}: {exc}"

    def check(end):
        nonlocal ops
        ops += 1
        if ops > MAX_OPS or b.clock() >= end:
            raise TimeoutError("deadline/operation bound reached")

    def close_gate():
        nonlocal gate_ok
        for _ in range(len(pipes)):
            try:
                with _masked():
                    fd = pipes.pop()
                    try:
                        b.close(fd)
                    except BaseException:
                        gate_ok = False
                        raise
            except BaseException as exc:
                fail(exc)

    def drain(end):
        nonlocal status, reaps, drained
        while True:
            check(end)
            if reaps >= MAX_REAPS:
                raise RuntimeError("reap bound reached")
            with _masked():
                try:
                    pid, result = b.wait()
                except ChildProcessError:
                    drained = status is not None and not pipes and gate_ok
                    return
                if pid == 0:
                    return
                _pid(pid)
                if not (os.WIFEXITED(result) or os.WIFSIGNALED(result)):
                    raise RuntimeError("nonterminal waitpid/ptrace stop refused")
                reaps += 1
                termed.discard(pid)
                killed.discard(pid)
                if pid == root and status is None:
                    status = result

    try:
        b = _backend if _backend is not None else _LinuxBackend()
        deadline = b.clock() + timeout
        cleanup_limit = deadline + min(2.0, timeout)
        check(deadline)
        b.prepare()
        check(deadline)
        with _masked():
            pipes.extend(b.pipe())
        check(deadline)
        with _masked():
            attempted = True
            root = b.fork()
            if root == 0:
                b.child(*pipes, argv, environment, cwd)
                raise RuntimeError("child backend returned")
            _pid(root)
        check(deadline)
        with _masked():
            if b.member(root) is not True:
                raise RuntimeError("fresh gated child membership required")
            check(deadline)
            b.release(pipes[1])
            released = True
        close_gate()
        if error is not None:
            raise RuntimeError("gate close failed")
        while status is None:
            drain(deadline)
            if status is None:
                check(deadline)
                b.sleep(min(0.01, max(0, deadline - b.clock())))
    except BaseException as exc:
        fail(exc)
    finally:
        close_gate()

    if b is not None and type(root) is int and root > 0:
        try:
            ops = 0
            stop = b.clock()
            end = min(cleanup_limit, stop + min(2.0, timeout))
            escalate = stop + 0.1
            while not drained:
                check(end)
                try:
                    drain(end)
                    if drained:
                        break
                    check(end)
                    frontier = b.children()
                    if len(frontier) > MAX_CHILDREN:
                        raise RuntimeError("children listing bound")
                    if status is None and root not in frontier:
                        frontier = [root, *frontier]
                    for pid in frontier:
                        check(end)
                        sig = signal.SIGKILL if b.clock() >= escalate else signal.SIGTERM
                        sent = killed if sig == signal.SIGKILL else termed
                        if pid not in sent:
                            if signal_attempts >= MAX_SIGNALS:
                                raise RuntimeError("signal bound reached")
                            with _masked():
                                signal_attempts += 1
                                if b.member(pid, sig) is True:
                                    signals += 1
                                    sent.add(pid)
                    b.sleep(min(0.01, max(0, end - b.clock())))
                except (KeyboardInterrupt, SystemExit, InterruptedError) as exc:
                    fail(exc)
        except BaseException as exc:
            fail(exc)
    if status is not None and status != 0:
        fail(RuntimeError("root exited nonzero or by signal"))
    if signals:
        fail(RuntimeError("termination signals required"))
    if drained and error is None:
        try:
            if b.clock() >= deadline:
                raise TimeoutError("drain completed after execution deadline")
        except BaseException as exc:
            fail(exc)
    success = released and status == 0 and drained and error is None
    return OwnedTreeReport(root, status, attempted, released, drained, success,
                           reaps, signals, error)
