"""Prescribed ONE-Apptainer child probes; never an output-bound certificate."""

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import stat
import threading


def probe(*, outer_pid_namespace, persistent_targets):
    failures, checks = [], {}
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
                  if ":" in line)
    for key, expected in (("NoNewPrivs", 1), ("CapEff", 0), ("CapPrm", 0),
                          ("CapInh", 0), ("CapAmb", 0), ("Seccomp", 2)):
        value = int(status[key].strip(), 16 if key.startswith("Cap") else 10)
        checks[key] = value
        if value != expected:
            failures.append(key)
    checks["pid_namespace"] = os.readlink("/proc/self/ns/pid")
    checks["pid1_namespace"] = os.readlink("/proc/1/ns/pid")
    if checks["pid_namespace"] == outer_pid_namespace or checks["pid1_namespace"] != checks["pid_namespace"]:
        failures.append("pid_namespace")
    checks["pid1_root"] = os.readlink("/proc/1/root")
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        record = {}
        for name in ("root", "cwd", "ns/pid"):
            try:
                record[name] = os.readlink(entry / name)
            except OSError as exc:
                record[name] = {"errno": exc.errno}
        try:
            record["fds"] = {item.name: os.readlink(item) for item in (entry / "fd").iterdir()}
        except OSError as exc:
            record["fds"] = {"errno": exc.errno}
        try:
            fd = os.open(entry / "mem", os.O_RDWR | os.O_CLOEXEC)
        except OSError as exc:
            record["memory_open_errno"] = exc.errno
        else:
            os.close(fd)
            record["memory_open_allowed"] = True
            failures.append("other_process_memory_accessible:" + entry.name)
        processes[entry.name] = record
    checks["other_visible_processes"] = processes
    descriptors = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            descriptors[name] = os.readlink("/proc/self/fd/" + name)
        except FileNotFoundError:
            continue
    checks["descriptors"] = descriptors
    if set(descriptors) != {"0", "1", "2"} or descriptors.get("0") != "/dev/null" or not all(
            descriptors.get(str(fd), "").startswith("pipe:[") for fd in (1, 2)):
        failures.append("unexpected_descriptors")
    checks["mountinfo"] = Path("/proc/self/mountinfo").read_text()
    mounts = []
    for line in checks["mountinfo"].splitlines():
        pre, post = line.split(" - ", 1)
        fields = pre.split()
        mounts.append({"target": fields[4], "options": fields[5].split(","),
                       "optional": fields[6:], "backing": post})
    checks["mounts"] = mounts
    checks["resolved_readonly_targets"] = {}
    for target in ("/opt/r5/checks", "/opt/r5/area", "/cvmfs", "/run", "/var/run", "/sys"):
        resolved = str(Path(target).resolve(strict=True))
        checks["resolved_readonly_targets"][target] = resolved
        matching = [m for m in mounts if m["target"] == resolved or m["target"].startswith(resolved + "/")]
        if not any(m["target"] == resolved for m in matching) or any("ro" not in m["options"] for m in matching):
            failures.append("readonly_mount:" + target)
    checks["propagated_mounts"] = [m["target"] for m in mounts if any(
        value.startswith(("shared:", "master:", "propagate_from:")) for value in m["optional"])]
    if checks["propagated_mounts"]:
        failures.append("mount_propagation_unresolved")
    for directory in ("/mnt/work", "/tmp", "/var/tmp", "/dev/shm", os.environ["HOME"]):
        path = Path(directory) / "r5-private-write-probe"
        with path.open("xb") as stream:
            stream.write(b"private transient content")
        path.rename(path.with_name("r5-private-renamed-probe"))
    for index, target in enumerate(persistent_targets):
        path = Path(target)
        if path.exists() or path.is_symlink():
            raise ValueError("persistent probe target already exists")
        try:
            with path.open("xb") as stream:
                stream.write(b"UNEXPECTED persistent write")
        except OSError as exc:
            checks["persistent_write_" + str(index)] = exc.errno
            if exc.errno not in (errno.EROFS, errno.EACCES, errno.EPERM, errno.ENOENT):
                failures.append("unexpected_write_error:" + target)
        else:
            failures.append("persistent_write_allowed:" + target)
    checks["devices"] = {path.name: stat.S_IFMT(path.lstat().st_mode) for path in Path("/dev").iterdir()}
    if Path("/dev/fuse").exists() or Path("/dev/pts/ptmx").exists():
        failures.append("unexpected_devices")
    for directory in ("/run", "/var/run"):
        if list(Path(directory).iterdir()):
            failures.append("runtime_paths_not_empty:" + directory)
    result = []
    thread = threading.Thread(target=lambda: result.append(True))
    thread.start()
    thread.join(timeout=5)
    if result != [True] or thread.is_alive():
        failures.append("ordinary_threads")
    libc = ctypes.CDLL(None, use_errno=True)
    for name, arguments in (("unshare", (0x00020000,)),
                            ("mount", (None, None, None, 0, None)),
                            ("setns", (-1, 0)), ("capset", (None, None)),
                            ("ptrace", (0, 0, None, None))):
        ctypes.set_errno(0)
        result = getattr(libc, name)(*arguments)
        checks[name] = {"result": result, "errno": ctypes.get_errno()}
        if result != -1 or ctypes.get_errno() != errno.EPERM:
            failures.append("syscall_not_denied:" + name)
    mask = signal_mask()
    checks["blocked_signals"] = sorted(int(value) for value in mask)
    if {2, 15}.intersection(mask):
        failures.append("inherited_blocked_signals")
    report = {"passed": not failures, "failures": failures, "checks": checks,
              "verified": False, "transport_events": 0,
              "scope": "Prescribed child checks, not exhaustive confinement, lifecycle or output-bound certification."}
    print(json.dumps(report, sort_keys=True))
    return report


def signal_mask():
    import signal
    return signal.pthread_sigmask(signal.SIG_BLOCK, set())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-pid-namespace", required=True)
    parser.add_argument("--persistent-target", action="append", required=True)
    args = parser.parse_args(argv)
    return 0 if probe(outer_pid_namespace=args.outer_pid_namespace,
                      persistent_targets=args.persistent_target)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
