"""Opt-in Linux payload launcher; the existing Offline runner is NOT sandboxed.

CVMFS and the EL9 runtime must already be mounted by the trusted outer runner.
Never run cvmfsexec/Apptainer inside this payload: no FUSE device, writable CVMFS
cache, installation, build, or extra bind API is provided. The existing runtime
and its environment must be trusted. Only fixed runtime prefixes are exposed
from the host, read-only. Fresh private node-local work, its aliases and the
transient tmpfs root are writable; none of that transient root is saved. Missing runtime or payload inputs fail, without a
host-root fallback.
Network access is retained for site QE. Services, readable secrets, network
access, kernel bugs and mount/path races are NOT a general hostile-code boundary.
The allowed prefixes are a path policy, not proof of node-local filesystem type.
The outer runner must exclusively own scratch and prevent concurrent replacement.

Plans and results deliberately have verified=False: a compute-node probe of
mounts, capabilities, seccomp, threads and the real runtime is still required.
RLIMIT_FSIZE limits individual files, NOT aggregate saved output, allocation,
logs or cache; this is not a hard output-budget gate or publication certificate.
No existing launcher is modified or automatically wrapped.

The bounding capability set may remain nonempty: it is a ceiling, not a grant.
Before exec, no_new_privs is enforced and E/P/I/ambient capabilities are checked
empty; seccomp forbids capset and ambient-capability operations. This policy
relies on Linux no_new_privs/capability semantics, not an empty bounding set.

CLI: python -m beamfastsim.flash_reference_sandbox --node-tmp /tmp \\
    --file-size-limit 1073741824 -- /absolute/payload argument
The separate --inside entry is an internal, fail-closed pre-exec step only.
"""

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import errno
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile


_LOCAL_ROOTS = ("/tmp", "/local", "/localscratch")
_LAYOUT = ("tmp", "home", "shm", ".empty")
_RUNTIME_PREFIXES = ("/usr", "/etc", "/opt", "/var", "/cvmfs", "/home", "/scratch", "/sfs")
_REQUIRED_RUNTIME_PREFIXES = ("/usr", "/etc", "/cvmfs")
DEFAULT_FILE_SIZE_LIMIT = 1024**3
_NAMESPACE_FLAGS = (0x00020000, 0x02000000, 0x04000000, 0x08000000,
                    0x10000000, 0x20000000, 0x40000000, 0x00000080)
_DENIED_SYSCALLS = (
    "capset", "mount", "umount2", "pivot_root", "unshare", "setns", "ptrace",
    "process_vm_readv", "process_vm_writev", "fsopen", "fsconfig", "fsmount",
    "fspick", "open_tree", "move_mount", "mount_setattr",
)


@dataclass(frozen=True)
class SandboxCommand:
    argv: tuple[str, ...]
    verified: bool = field(default=False, init=False)


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    verified: bool = field(default=False, init=False)


def _limit(value):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError("file-size limit must be a positive signed 64-bit integer")
    return value


def _text_path(value):
    value = os.fspath(value)
    if not isinstance(value, str) or not value.startswith("/") or "\0" in value:
        raise ValueError("an absolute text path is required")
    if value == "/" or any(p in ("", ".", "..") for p in value.split("/")[1:]):
        raise ValueError("path must be canonical without dot or empty components")
    return value


def _local_directory(value, *, private=False):
    value = _text_path(value)
    if not any(value == root or value.startswith(root + "/") for root in _LOCAL_ROOTS):
        raise ValueError("scratch must be under /tmp, /local or /localscratch")
    current = ""
    for part in value.split("/")[1:]:
        current += "/" + part
        info = os.lstat(current)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("scratch components must be real directories, never symlinks")
    if private:
        if value in _LOCAL_ROOTS:
            raise ValueError("scratch must be a private child, not a scratch root")
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("private scratch must be owned by this uid with mode 0700")
    return value


def _validate_work(value):
    value = _local_directory(value, private=True)
    if set(os.listdir(value)) != set(_LAYOUT):
        raise ValueError("private work must contain only the fresh scratch layout")
    for name in _LAYOUT:
        child = _local_directory(value + "/" + name, private=True)
        if os.listdir(child):
            raise ValueError("scratch subdirectories must be empty before launch")
    return value


def _payload(command):
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("payload must be a nonempty argument list, not shell text")
    if any(not isinstance(arg, str) or "\0" in arg for arg in command):
        raise ValueError("payload arguments must be text without NUL")
    executable = command[0]
    if not executable or executable.startswith("-"):
        raise ValueError("payload must begin with an executable, not an option")
    if "/" in executable:
        _text_path(executable)
    elif executable in (".", ".."):
        raise ValueError("invalid executable")
    return tuple(command)


class _UnreapedChild(RuntimeError):
    """Do not remove scratch while the sandbox may still be running."""


@contextmanager
def private_work(node_tmp="/tmp"):
    """Allocate fresh mode-0700 work; caller must keep its parent trusted/stable."""
    parent = _local_directory(node_tmp)
    work = tempfile.mkdtemp(prefix="flash-payload-", dir=parent)
    remove = True
    try:
        os.chmod(work, 0o700)
        for name in _LAYOUT:
            os.mkdir(os.path.join(work, name), 0o700)
        yield work
    except _UnreapedChild as exc:
        remove = False
        raise _UnreapedChild("sandbox reaping unconfirmed; scratch retained at " + work) from exc
    finally:
        if remove:
            shutil.rmtree(work)


def build_command(privatework, command, *, file_size_limit=DEFAULT_FILE_SIZE_LIMIT):
    """Build a fixed bwrap invocation without invoking Linux tools or probing them.

    Use private_work() for the fresh directory. No arbitrary sandbox flags,
    writable binds, environment overrides, cache mounts or shell parsing exist.
    The builder includes only existing directories from the fixed runtime list.
    It neither requires a Linux layout for planning nor verifies/executes it.
    run_payload requires /usr, /etc and /cvmfs before execution; absent optional
    prefixes are not exposed and payloads needing their inputs fail normally.
    There is no alternate layout, arbitrary bind API or host-root fallback.
    """
    work = _validate_work(privatework)
    payload = _payload(command)
    limit = _limit(file_size_limit)
    interpreter = _text_path(sys.executable)
    script = str(Path(__file__).resolve())
    argv = [
        "bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--die-with-parent", "--new-session", "--cap-drop", "ALL",
        "--tmpfs", "/",
    ]
    for prefix in _RUNTIME_PREFIXES:
        if os.path.isdir(prefix):
            argv.extend(("--ro-bind", prefix, prefix))
    for name in ("bin", "lib", "lib64", "sbin"):
        argv.extend(("--symlink", "usr/" + name, "/" + name))
    argv.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/mnt",
                 "--dir", "/mnt/work", "--bind", work, "/mnt/work"))
    for target in ("/run", "/var/run", "/sys"):
        argv.extend(("--ro-bind", work + "/.empty", target))
    for source, target in (("tmp", "/tmp"), ("tmp", "/var/tmp"), ("shm", "/dev/shm")):
        argv.extend(("--bind", work + "/" + source, target))
    for key, value in (
        ("TMPDIR", "/mnt/work/tmp"), ("TMP", "/mnt/work/tmp"), ("TEMP", "/mnt/work/tmp"),
        ("SQLITE_TMPDIR", "/mnt/work/tmp"), ("HOME", "/mnt/work/home"),
        ("XDG_CACHE_HOME", "/mnt/work/tmp"), ("XDG_RUNTIME_DIR", "/mnt/work/tmp"),
    ):
        argv.extend(("--setenv", key, value))
    argv.extend(("--chdir", "/mnt/work", "--", interpreter, "-I", script, "--inside",
                 "--file-size-limit", str(limit), "--", *payload))
    return SandboxCommand(tuple(argv))


def _linux_only():
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("payload sandbox requires Linux x86_64 or aarch64")


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32)]


class _ArgCompare(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def _no_privileges():
    """Require NNP and empty P/E/I/ambient, not an empty bounding ceiling.

    See capabilities(7), "Transformation of capabilities during execve()", and
    PR_SET_NO_NEW_PRIVS(2): inherited NNP prevents set-ID/file-capability gains.
    Seccomp subsequently denies capset and every ambient-capability operation.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    libc.capget.argtypes = [ctypes.POINTER(_CapHeader), ctypes.POINTER(_CapData)]
    libc.capget.restype = ctypes.c_int
    # NNP prevents exec capability gains even with a nonempty bounding ceiling.
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(39, 0, 0, 0, 0) != 1:
        raise RuntimeError("cannot enforce no_new_privs")
    header = _CapHeader(0x20080522, 0)
    caps = (_CapData * 2)()
    if libc.capget(ctypes.byref(header), caps) != 0:
        raise RuntimeError("cannot inspect payload capabilities")
    if any(c.effective or c.permitted or c.inheritable for c in caps):
        raise RuntimeError("bubblewrap did not drop all payload capabilities")
    for cap in range(64):
        # PR_CAP_AMBIENT_IS_SET: EINVAL at cap 0 cannot prove empty ambient caps.
        result = libc.prctl(47, 1, cap, 0, 0)
        if result == -1 and ctypes.get_errno() == errno.EINVAL and cap > 0:
            break
        if result != 0:
            raise RuntimeError("cannot establish empty ambient capabilities")


def _seccomp_library():
    # Load only an existing library; never invoke ldd, a compiler or an installer.
    for name in ("libseccomp.so.2", "libseccomp.so"):
        try:
            lib = ctypes.CDLL(name, use_errno=True)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("existing libseccomp is required; refusing unsandboxed execution")
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                           ctypes.c_uint, ctypes.POINTER(_ArgCompare)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    return lib


def _install_seccomp():
    lib = _seccomp_library()
    context = lib.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("cannot allocate seccomp filter")
    try:
        def add(name, error=errno.EPERM, comparison=None):
            number = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                raise RuntimeError("libseccomp cannot resolve required syscall " + name)
            pointer = None if comparison is None else ctypes.byref(comparison)
            result = lib.seccomp_rule_add_array(context, 0x00050000 | error, number,
                                                 int(comparison is not None), pointer)
            if result != 0:
                raise RuntimeError("cannot filter syscall " + name)
        for name in _DENIED_SYSCALLS:
            add(name)
        # Mask ignored upper option bits to deny every PR_CAP_AMBIENT operation.
        add("prctl", comparison=_ArgCompare(0, 7, 0xFFFFFFFF, 47))
        add("clone3", errno.ENOSYS)
        for flag in _NAMESPACE_FLAGS:
            add("clone", comparison=_ArgCompare(0, 7, flag, flag))
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("cannot load seccomp filter")
    finally:
        lib.seccomp_release(context)


def _inside(command, file_size_limit):
    _linux_only()
    payload = _payload(command)
    limit = _limit(file_size_limit)
    if os.getcwd() != "/mnt/work" or os.environ.get("HOME") != "/mnt/work/home" or \
            os.environ.get("TMPDIR") != "/mnt/work/tmp":
        raise RuntimeError("--inside requires the prepared bubblewrap environment")
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    _no_privileges()
    _install_seccomp()
    os.execvpe(payload[0], payload, os.environ.copy())


_CHILD_STOP_TIMEOUT = 2.0


def _terminate_and_reap(child):
    """Never use Popen.__exit__: its wait is unbounded, even on pipe errors.

    Reaping bwrap must precede scratch deletion. Its PID namespace and
    --die-with-parent tie the payload lifetime to the launcher; that behavior
    still needs a compute-node probe. If either bounded wait is interrupted,
    try the next stage; if reaping remains unconfirmed retain scratch.
    """
    failure = None
    for stop in (child.terminate, child.kill):
        try:
            stop()
        except BaseException:
            # A failed signal is not evidence of reaping; still wait/escalate.
            pass
        try:
            child.wait(timeout=_CHILD_STOP_TIMEOUT)
            return
        except BaseException as exc:
            failure = exc
    raise _UnreapedChild("cannot confirm sandbox termination after kill") from failure


def run_payload(command, *, node_tmp="/tmp", file_size_limit=DEFAULT_FILE_SIZE_LIMIT):
    """Stream child output through a pipe; never pass host output file descriptors.

    The trusted outer caller owns stdout/log policy. A nonzero result is failure,
    not a complete physics run. Neither zero nor cleanup certifies confinement,
    payload termination on an unprobed node, or aggregate saved-output bounds.
    /usr, /etc and /cvmfs must exist; no reduced-layout execution is provided.
    On output errors or interruptions, terminate/reap with bounded waits, then
    escalate to kill. If reaping cannot be confirmed, refuse and retain scratch.
    """
    _linux_only()
    _payload(command)
    _limit(file_size_limit)
    for prefix in _REQUIRED_RUNTIME_PREFIXES:
        if not os.path.isdir(prefix):
            raise RuntimeError("required runtime directory is missing: " + prefix)
    with private_work(node_tmp) as work:
        plan = build_command(work, command, file_size_limit=file_size_limit)
        try:
            child = subprocess.Popen(plan.argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, close_fds=True, cwd="/")
        except BaseException as exc:
            # A spawn interruption can leave an untracked child: retain scratch.
            raise _UnreapedChild("launcher creation failed; child state unknown") from exc
        try:
            try:
                while True:
                    chunk = child.stdout.read(64 * 1024)
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                result = SandboxResult(child.wait())
                child.stdout.close()
            except BaseException:
                _terminate_and_reap(child)
                raise
            return result
        finally:
            # Preserve any reaping failure and its scratch-retention marker.
            try:
                child.stdout.close()
            except BaseException:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--node-tmp", default="/tmp")
    parser.add_argument("--file-size-limit", type=int, default=DEFAULT_FILE_SIZE_LIMIT)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if args.inside:
            _inside(command, args.file_size_limit)
            raise RuntimeError("exec unexpectedly returned")
        return run_payload(command, node_tmp=args.node_tmp,
                           file_size_limit=args.file_size_limit).returncode
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(125, "sandbox refused: " + str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
