"""Unverified ONE-Apptainer candidate; no transport or publication CLI.

Trusted CVMFSEXEC must mount the pinned image and CVMFS before build_command.
The caller owns fresh node-local runtime and payload directories and launches
with stdin DEVNULL, merged output PIPE, close_fds=True and cwd='/'. It must
sanitize the LAUNCHER environment, not merely request --cleanenv. The runtime
session/cache path is a sibling, never an ancestor bound into the payload.

This is a fixed bind policy and fail-closed pre-exec, not a confinement or
termination certificate. verified remains False even if a child exits zero.
Do not use the old broad-bind Offline runner or nest bwrap/another container.
The approved bound is logical saved contents, not transient storage allocation.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import signal
import stat
import sys


PINNED_IMAGE = ("/cvmfs/singularity.opensciencegrid.org/.images/sha256:a0/"
                "bdaf92d1cfa960cf7f053f9b929fba86e9dbbb97fc8fea51a900489a32bd40")
NO_MOUNT = "bind-paths,hostfs,cwd,sys,devpts"
_LOCAL_ROOTS = ("/tmp", "/local", "/localscratch")
_LAYOUT = ("tmp", "home", "shm")
_CODE_DEST = "/opt/r5/checks"
_AREA_DEST = "/opt/r5/area"
_INPUT_DEST = "/opt/r5/input.art"
_INNER_SCRIPT = _CODE_DEST + "/flash_reference_apptainer.py"
_SITE_CONFIG = "/apps/software/standard/core/apptainer/1.4.5/etc/apptainer/apptainer.conf"
_SITE_CONFIG_SHA256 = "dda6d4177300d8fab88ed4674af0ada106ef2bf894dc2ee8e2b39b0563ce4cce"


@dataclass(frozen=True)
class ApptainerCommand:
    argv: tuple
    environment: dict
    configuration: bytes = b""
    verified: bool = field(default=False, init=False)


def _path(value):
    value = os.fspath(value)
    if not isinstance(value, str) or not value.startswith("/") or value == "/" or any(
            part in ("", ".", "..") for part in value.split("/")[1:]) or any(
            char in value for char in ("\0", ":", ",", "\n", "\r")):
        raise ValueError("canonical absolute bind path without separators is required")
    return value


def _real(value, *, directory=True, private=False):
    value = _path(value)
    current = ""
    for part in value.split("/")[1:]:
        current += "/" + part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("bind sources and ancestors must not be symlinks")
        if current != value and not stat.S_ISDIR(info.st_mode):
            raise ValueError("bind ancestors must be directories")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError("bind source has wrong file type")
    if private and (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("private directories require current uid and mode 0700")
    return value


def _private(value):
    value = _real(value, private=True)
    if not any(value.startswith(root + "/") for root in _LOCAL_ROOTS):
        raise ValueError("private directory must be below a node-local root")
    return value


def _overlap(left, right):
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def private_mount_configuration(site_bytes):
    """Pin the installed config and change only its incoming-propagation setting."""
    if not isinstance(site_bytes, bytes) or hashlib.sha256(site_bytes).hexdigest() != _SITE_CONFIG_SHA256:
        raise ValueError("installed Apptainer config identity changed")
    lines = site_bytes.splitlines(keepends=True)
    active = [index for index, line in enumerate(lines)
              if line.strip() and not line.lstrip().startswith(b"#") and
              line.split(b"=", 1)[0].strip() == b"mount slave"]
    if len(active) != 1 or lines[active[0]] != b"mount slave = yes\n":
        raise ValueError("expected exactly one pinned mount slave directive")
    lines[active[0]] = b"mount slave = no\n"
    return b"".join(lines)


def launcher_environment(runtime, apptainer):
    """Return a fresh allowlist, never merge APPTAINER*/SINGULARITY*/LD_* inputs."""
    return {"PATH": os.path.dirname(apptainer) + ":/usr/bin:/bin",
            "HOME": runtime + "/home", "LANG": "C", "LC_ALL": "C",
            "TMPDIR": runtime + "/tmp", "APPTAINER_TMPDIR": runtime + "/tmp",
            "APPTAINER_CACHEDIR": runtime + "/cache",
            "APPTAINER_CONFIGDIR": runtime + "/config"}


def build_command(*, apptainer, runtime, work, code, area, command,
                  input_file=None, file_size_limit=1024**3):
    """Build explicit read-only binds and private aliases, not arbitrary flags.

    code contains this module, flash_reference_sandbox.py and trusted payloads.
    The caller must exclusively write plan.configuration at the fixed --config
    path with mode 0400, after planning and before launching, and never alter it.
    This requires the verified installed non-setuid binary; rejection is fatal.
    area is the already-built satellite. Sources must be stable, independently
    checked snapshots under caller ownership; lstat validation is not race proof.
    runtime/work are distinct private siblings with their exact fresh layouts.
    CVMFS is the sole fixed shared bind; nested mounts/propagation need probes.
    """
    apptainer = _real(apptainer, directory=False)
    runtime, work = _private(runtime), _private(work)
    code, area = _real(code), _real(area)
    paths = (runtime, work, code, area)
    if any(_overlap(left, right) for i, left in enumerate(paths) for right in paths[i + 1:]):
        raise ValueError("runtime, work, code and area must not overlap")
    for root, layout in ((runtime, ("home", "tmp", "cache", "config", "session", "empty")),
                         (work, _LAYOUT)):
        if set(os.listdir(root)) != set(layout):
            raise ValueError("unexpected or missing fresh private layout")
        for name in layout:
            child = _real(root + "/" + name, private=True)
            if os.listdir(child):
                raise ValueError("private layout must be empty before launch")
    for name in ("flash_reference_apptainer.py", "flash_reference_sandbox.py"):
        _real(code + "/" + name, directory=False)
    if type(file_size_limit) is not int or not 0 < file_size_limit < 2**63:
        raise ValueError("positive signed 64-bit individual file limit is required")
    if not isinstance(command, (list, tuple)) or not command or any(
            not isinstance(arg, str) or "\0" in arg for arg in command):
        raise ValueError("nonempty text payload argv is required")
    _path(command[0])
    config_bytes = private_mount_configuration(Path(_SITE_CONFIG).read_bytes())
    config_path = runtime + "/config/apptainer.conf"
    argv = [apptainer, "--config", config_path, "exec", "--userns", "--cleanenv", "--containall", "--no-home",
            "--no-mount", NO_MOUNT, "--drop-caps", "ALL", "--no-privs", "--no-eval",
            "--disable-cache", "--workdir", runtime + "/session", "--cwd", "/mnt/work"]
    binds = [("/cvmfs", "/cvmfs", "ro"), (code, _CODE_DEST, "ro"),
             (area, _AREA_DEST, "ro"), (work, "/mnt/work", "rw"),
             (work + "/tmp", "/tmp", "rw"), (work + "/tmp", "/var/tmp", "rw"),
             (work + "/shm", "/dev/shm", "rw"),
             (runtime + "/empty", "/run", "ro"),
             (runtime + "/empty", "/var/run", "ro"),
             (runtime + "/empty", "/sys", "ro")]
    if input_file is not None:
        input_file = _real(input_file, directory=False)
        if any(_overlap(input_file, root) for root in paths):
            raise ValueError("input must be separate from all directory binds")
        binds.append((input_file, _INPUT_DEST, "ro"))
    for source, destination, mode in binds:
        argv.extend(("--bind", source + ":" + destination + ":" + mode))
    for key, value in (("TMPDIR", "/mnt/work/tmp"),
                       ("TMP", "/mnt/work/tmp"), ("TEMP", "/mnt/work/tmp"),
                       ("SQLITE_TMPDIR", "/mnt/work/tmp"), ("XDG_CACHE_HOME", "/mnt/work/tmp"),
                       ("XDG_RUNTIME_DIR", "/mnt/work/tmp"), ("PYTHONDONTWRITEBYTECODE", "1")):
        argv.extend(("--env", key + "=" + value))
    argv.extend((PINNED_IMAGE, "/usr/bin/python3", "-I", "-B", _INNER_SCRIPT,
                 "--inside", "--file-size-limit", str(file_size_limit), "--", *command))
    return ApptainerCommand(tuple(argv), launcher_environment(runtime, apptainer), config_bytes)


def _inside(command, limit):
    """Apply the existing security policy directly after the sole container setup."""
    if (sys.platform != "linux" or os.getcwd() != "/mnt/work" or
            os.environ.get("TMPDIR") != "/mnt/work/tmp" or
            str(Path(__file__).resolve()) != _INNER_SCRIPT):
        raise RuntimeError("internal handoff requires the fixed container layout")
    # Apptainer rejects HOME overrides; set the fixed private value before workload.
    home = os.lstat("/mnt/work/home")
    if not stat.S_ISDIR(home.st_mode) or home.st_uid != os.geteuid() or stat.S_IMODE(home.st_mode) != 0o700:
        raise RuntimeError("internal handoff requires the private home directory")
    os.environ["HOME"] = "/mnt/work/home"
    if type(limit) is not int or not 0 < limit < 2**63 or not command:
        raise ValueError("invalid payload or file limit")
    _path(command[0])
    import resource
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "r5_trusted_security", _CODE_DEST + "/flash_reference_sandbox.py")
    security = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = security
    spec.loader.exec_module(security)
    security._linux_only()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    security._no_privileges()
    security._install_seccomp()
    # No runtime-created file/socket/directory descriptor may reach the workload.
    for entry in os.listdir("/proc/self/fd"):
        fd = int(entry)
        if fd > 2:
            try:
                os.close(fd)
            except OSError as exc:
                import errno
                if exc.errno != errno.EBADF:
                    raise
    # The outer guarded supervisor blocks these while acquiring the child handle.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    os.execvpe(command[0], command, os.environ.copy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true", required=True)
    parser.add_argument("--file-size-limit", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        _inside(command, args.file_size_limit)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(125, "Apptainer handoff refused: " + str(exc) + "\n")
    raise RuntimeError("exec unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
