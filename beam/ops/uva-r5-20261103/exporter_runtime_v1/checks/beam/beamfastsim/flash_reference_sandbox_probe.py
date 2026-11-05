"""Small negative/positive payload checks; not a compute-node certificate alone."""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import threading


def probe(*, persistent_targets):
    """Run only as a confined child with explicitly chosen non-existent targets."""
    failures = []
    checks = {}
    for name in ("NoNewPrivs", "CapEff", "CapPrm", "CapInh", "Seccomp"):
        lines = Path("/proc/self/status").read_text().splitlines()
        value = next(line.split(":", 1)[1].strip() for line in lines if line.startswith(name + ":"))
        checks[name] = value
        expected = 1 if name == "NoNewPrivs" else 2 if name == "Seccomp" else 0
        if int(value, 16 if name.startswith("Cap") else 10) != expected:
            failures.append(name)
    if Path("/dev/fuse").exists():
        failures.append("fuse_visible")
    for i, target in enumerate(persistent_targets):
        path = Path(target)
        if path.exists() or path.is_symlink():
            raise ValueError("probe target must not already exist")
        try:
            with path.open("xb") as stream:
                stream.write(b"unexpected-persistent-write")
        except OSError as exc:
            checks["persistent_write_" + str(i)] = exc.errno
            if exc.errno not in (errno.EROFS, errno.EPERM, errno.EACCES, errno.ENOENT):
                failures.append("unexpected_write_error_" + str(i))
        else:
            failures.append("persistent_write_allowed_" + str(i))
    for directory in ("/mnt/work", "/tmp", "/var/tmp", "/dev/shm", os.environ["HOME"]):
        path = Path(directory) / ("probe-temporary-file-" + str(len(directory)))
        with path.open("xb") as stream:
            stream.write(b"temporary")
    result = []
    thread = threading.Thread(target=lambda: result.append("thread_ok"))
    thread.start()
    thread.join()
    if result != ["thread_ok"]:
        failures.append("thread")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.unshare(0x00020000) != -1 or ctypes.get_errno() != errno.EPERM:
        failures.append("unshare_not_denied")
    with open("/proc/self/mountinfo") as stream:
        checks["mountinfo"] = stream.read()
    report = {"passed": not failures, "failures": failures, "checks": checks,
              "scope": "Prescribed child probes only; not exhaustive confinement or output-budget certification."}
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-target", action="append", required=True)
    args = parser.parse_args(argv)
    return 0 if probe(persistent_targets=args.persistent_target)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
