"""Opt-in, no-Geant4 Linux lifecycle fixture for the fixed private sandbox.

Never a general launcher, controller, transport, fault-injection automation or
production gate. Invoke explicitly through the existing trusted sandbox handoff;
this module only CHECKS its inherited security state and never weakens it.
The external observer owns kill/descendant verification. A ready manifest is not
proof of containment or termination. All processes have a fixed 30-second
emergency fallback expiry, NOT containment. An independent outer watchdog is
mandatory; linger/closed-stdio normally require external termination. Readiness
records checked primary mode properties and successful intermediate reap, not
continued liveness, containment, or successful external termination.
Only two exclusive, bounded files in /mnt/work are written. No retries, output
publication, arbitrary command/path arguments, or host-side setup are provided.
"""

import argparse
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import sys
import time


MODES = ("normal", "nonzero", "crash", "linger", "closed-stdio")
EXPIRY_SECONDS = 30
READY_TIMEOUT_SECONDS = 3
PARENT_EXIT_DELAY_SECONDS = 0.25
MAX_FILE_BYTES = 4096
MAX_HEARTBEATS = 30
WORK = "/mnt/work"
READY_NAME = "lifecycle-ready.json"
HEARTBEAT_NAME = "lifecycle-heartbeat.jsonl"
_SECURITY_EXPECTED = {"NoNewPrivs": 1, "CapEff": 0, "CapPrm": 0,
                      "CapInh": 0, "CapAmb": 0, "Seccomp": 2}


def validate_inputs(mode, nonce, outer_pid_namespace):
    """Pure portable validation; no Linux probes or filesystem effects."""
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("unknown lifecycle mode")
    if not isinstance(nonce, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", nonce) is None:
        raise ValueError("nonce must be 1..64 ASCII letters, digits, underscores or hyphens")
    if (not isinstance(outer_pid_namespace, str) or
            re.fullmatch(r"pid:\[[1-9][0-9]{0,19}\]", outer_pid_namespace) is None):
        raise ValueError("external PID namespace must be a bounded pid:[number] token")


def check_security_status(text):
    """Validate a proc status snapshot; bounding capabilities are not grants."""
    if not isinstance(text, str) or len(text) > 16384:
        raise ValueError("invalid proc status size")
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key in _SECURITY_EXPECTED:
            if key in fields:
                raise ValueError("duplicate security status field")
            fields[key] = int(value.strip(), 16 if key.startswith("Cap") else 10)
    if fields != _SECURITY_EXPECTED:
        raise ValueError("requires NNP, empty effective/permitted/inheritable/ambient caps and seccomp filter")
    return fields


def encode_record(record, limit=MAX_FILE_BYTES):
    """One deterministic bounded JSON line, also used for the ready manifest."""
    data = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("ascii")
    if len(data) > limit:
        raise ValueError("record exceeds fixed byte bound")
    return data


def make_ready_manifest(*, mode, nonce, outer_pid_namespace, security, processes,
                        primary_mode_properties):
    """Pure schema builder. Process snapshots are supplied, never probed here."""
    validate_inputs(mode, nonce, outer_pid_namespace)
    if security != _SECURITY_EXPECTED or set(processes) != {"parent", "session_child", "orphan"}:
        raise ValueError("incomplete ready snapshot")
    for process in processes.values():
        if set(process) != {"pid", "ppid", "sid", "pgid", "pid_namespace"}:
            raise ValueError("invalid process schema")
        if any(type(process[key]) is not int or not 0 <= process[key] < 2**31
               for key in ("pid", "ppid", "sid", "pgid")) or process["pid"] == 0:
            raise ValueError("invalid process identity")
        validate_inputs(mode, nonce, process["pid_namespace"])
    if (primary_mode_properties != _expected_primary_mode_properties(mode) or
            any(type(value) is not bool for value in primary_mode_properties.values())):
        raise ValueError("primary mode properties were not checked")
    orphan = processes["orphan"]
    namespace = orphan["pid_namespace"]
    if namespace == outer_pid_namespace or any(
            item["pid_namespace"] != namespace for item in processes.values()):
        raise ValueError("requires one private PID namespace distinct from external namespace")
    if (orphan["ppid"] == processes["session_child"]["pid"] or
            processes["session_child"]["sid"] != processes["session_child"]["pid"] or
            orphan["sid"] != processes["session_child"]["sid"]):
        raise ValueError("requires session-detached, double-forked orphan")
    record = {"schema_version": 1, "mode": mode, "nonce": nonce,
              "pid": orphan["pid"], "ppid": orphan["ppid"],
              "pid_namespace": namespace, "outer_pid_namespace": outer_pid_namespace,
              "processes": processes, "security": security,
              "emergency_expiry_seconds": EXPIRY_SECONDS,
              "heartbeat_max_bytes": MAX_FILE_BYTES,
              "orphan_ignores_sigterm": True,
              "primary_mode_properties": primary_mode_properties,
              "intermediate_reaped": True,
              "verified": False, "transport_events": 0}
    encode_record(record)
    return record


def _snapshot():
    return {"pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(0),
            "pgid": os.getpgrp(), "pid_namespace": os.readlink("/proc/self/ns/pid")}


def _validate_linux():
    if sys.platform != "linux":
        raise RuntimeError("lifecycle payload requires Linux inside the fixed sandbox")


def _check_environment(outer_pid_namespace):
    if (os.getcwd() != WORK or os.environ.get("HOME") != WORK + "/home" or
            os.environ.get("TMPDIR") != WORK + "/tmp"):
        raise RuntimeError("requires fixed private cwd, HOME and TMPDIR")
    for path in (WORK, WORK + "/home", WORK + "/tmp"):
        info = os.lstat(path)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or
                stat.S_IMODE(info.st_mode) != 0o700 or str(Path(path).resolve()) != path):
            raise RuntimeError("requires owned mode-0700 non-symlink private layout")
    namespace = os.readlink("/proc/self/ns/pid")
    if namespace == outer_pid_namespace or os.readlink("/proc/1/ns/pid") != namespace:
        raise RuntimeError("requires private PID/proc namespace distinct from external namespace")
    with open("/proc/self/status", encoding="ascii") as stream:
        security = check_security_status(stream.read(16385))
    if signal.pthread_sigmask(signal.SIG_BLOCK, set()).intersection(
            {signal.SIGTERM, signal.SIGINT, signal.SIGALRM, signal.SIGABRT}):
        raise RuntimeError("lifecycle signals must not be inherited blocked")
    return security


def _arm_expiry(deadline):
    # Every fork child rearms its non-inherited expiry timer before doing work.
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        os._exit(124)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def _sleep_until(target):
    while True:
        remaining = target - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


def _open_output(directory_fd, name):
    if name not in (READY_NAME, HEARTBEAT_NAME):
        raise ValueError("only fixed lifecycle output names are allowed")
    return os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                   0o600, dir_fd=directory_fd)


def _write_bounded(fd, data, used):
    if used + len(data) > MAX_FILE_BYTES:
        raise ValueError("fixed lifecycle file bound exceeded")
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short lifecycle write")
        used += count
        view = view[count:]
    return used


def _close_stdio():
    fd = os.open("/dev/null", os.O_RDWR | os.O_CLOEXEC)
    try:
        for destination in (0, 1, 2):
            os.dup2(fd, destination, inheritable=True)
    finally:
        if fd > 2:
            os.close(fd)


def _expected_primary_mode_properties(mode):
    return {"sigterm_ignored": mode in ("linger", "closed-stdio"),
            "stdio_devnull_checked": mode == "closed-stdio"}


def _check_primary_mode_properties(mode):
    expected = _expected_primary_mode_properties(mode)
    disposition = signal.SIG_IGN if expected["sigterm_ignored"] else signal.SIG_DFL
    if signal.getsignal(signal.SIGTERM) != disposition:
        raise RuntimeError("primary SIGTERM disposition mismatch")
    if expected["stdio_devnull_checked"]:
        null = os.stat("/dev/null")
        if not stat.S_ISCHR(null.st_mode):
            raise RuntimeError("/dev/null must be a character device")
        for fd in (0, 1, 2):
            info = os.fstat(fd)
            if not stat.S_ISCHR(info.st_mode) or info.st_rdev != null.st_rdev:
                raise RuntimeError("primary stdio must refer to /dev/null")
    return expected


def _read_ready(read_fd, ready_deadline):
    """Read one bounded private pipe record; never publish a partial snapshot."""
    data = bytearray()
    while True:
        remaining = ready_deadline - time.monotonic()
        if remaining <= 0 or not select.select([read_fd], [], [], remaining)[0]:
            raise RuntimeError("bounded lifecycle ready wait failed")
        chunk = os.read(read_fd, MAX_FILE_BYTES + 1 - len(data))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_FILE_BYTES:
            raise RuntimeError("oversized lifecycle ready snapshot")
    if not data or not data.endswith(b"\n"):
        raise RuntimeError("incomplete lifecycle ready snapshot")
    record = json.loads(data)
    encode_record(record)
    return record


def _orphan(*, directory_fd, ready_fd, deadline, parent, session_child):
    # Send inherited mode properties only after intermediate-child reparenting.
    ready_deadline = min(deadline, time.monotonic() + READY_TIMEOUT_SECONDS)
    while os.getppid() == session_child["pid"]:
        if time.monotonic() >= ready_deadline:
            raise RuntimeError("double-fork reparenting timeout")
        time.sleep(0.01)
    processes = {"parent": parent, "session_child": session_child, "orphan": _snapshot()}
    heartbeat_fd = _open_output(directory_fd, HEARTBEAT_NAME)
    used = 0
    try:
        used = _write_bounded(heartbeat_fd, encode_record(
            {"sequence": 0, "pid": os.getpid(), "monotonic_ns": time.monotonic_ns()}), used)
        os.close(directory_fd)
        # Only the primary publishes READY_NAME, after validating/reaping its child.
        _write_bounded(ready_fd, encode_record(processes), 0)
        os.close(ready_fd)
        for sequence in range(1, MAX_HEARTBEATS):
            _sleep_until(min(deadline, time.monotonic() + 1))
            if time.monotonic() >= deadline:
                break
            used = _write_bounded(heartbeat_fd, encode_record(
                {"sequence": sequence, "pid": os.getpid(),
                 "monotonic_ns": time.monotonic_ns()}), used)
        _sleep_until(deadline)
    finally:
        os.close(heartbeat_fd)


def run(*, mode, nonce, outer_pid_namespace):
    """Explicit Linux-only entry; never called by portable schema tests."""
    validate_inputs(mode, nonce, outer_pid_namespace)
    _validate_linux()
    start = time.monotonic()
    deadline = start + EXPIRY_SECONDS
    _arm_expiry(deadline)
    security = _check_environment(outer_pid_namespace)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.umask(0o077)
    if mode in ("linger", "closed-stdio"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "closed-stdio":
        _close_stdio()
    primary_mode_properties = _check_primary_mode_properties(mode)
    parent = _snapshot()
    directory_fd = os.open(WORK, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        _arm_expiry(deadline)
        os.close(read_fd)
        try:
            os.setsid()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            session_child = _snapshot()
            orphan = os.fork()
            if orphan != 0:
                os._exit(0)
            _arm_expiry(deadline)
            _orphan(directory_fd=directory_fd, ready_fd=write_fd, deadline=deadline,
                    parent=parent, session_child=session_child)
        except (OSError, ValueError, RuntimeError):
            os._exit(125)
        os._exit(124)
    os.close(write_fd)
    ready_deadline = min(deadline, time.monotonic() + READY_TIMEOUT_SECONDS)
    try:
        processes = _read_ready(read_fd, ready_deadline)
    finally:
        os.close(read_fd)
    while True:
        reaped, status = os.waitpid(child, os.WNOHANG)
        if reaped:
            if status != 0:
                raise RuntimeError("intermediate lifecycle child failed")
            break
        if time.monotonic() >= ready_deadline:
            raise RuntimeError("bounded intermediate reap failed")
        time.sleep(0.01)
    manifest = make_ready_manifest(mode=mode, nonce=nonce,
                                   outer_pid_namespace=outer_pid_namespace,
                                   security=security, processes=processes,
                                   primary_mode_properties=primary_mode_properties)
    manifest_fd = _open_output(directory_fd, READY_NAME)
    try:
        _write_bounded(manifest_fd, encode_record(manifest), 0)
    finally:
        os.close(manifest_fd)
        os.close(directory_fd)
    if mode in ("linger", "closed-stdio"):
        _sleep_until(deadline)
        return 124
    _sleep_until(min(deadline, time.monotonic() + PARENT_EXIT_DELAY_SECONDS))
    if mode == "crash":
        signal.signal(signal.SIGABRT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGABRT)
        os._exit(125)  # Never turn a refused crash into a successful normal exit.
    return 7 if mode == "nonzero" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-lifecycle-payload", action="store_true", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--outer-pid-namespace", required=True)
    args = parser.parse_args(argv)
    try:
        validate_inputs(args.mode, args.nonce, args.outer_pid_namespace)
        return run(mode=args.mode, nonce=args.nonce, outer_pid_namespace=args.outer_pid_namespace)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(125, "Lifecycle payload refused: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
