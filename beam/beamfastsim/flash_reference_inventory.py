"""Strict post-termination census of a trusted private reference work directory.

The caller MUST independently establish termination of every payload descendant
and exclude all other writers before calling, and keep the tree stable afterward.
This helper does not terminate, execute, confine, publish, delete, inspect physics
content, or certify anything. A successful inventory always has verified=False;
it is a point-in-time metadata measurement, not an output-bound enforcement gate
or a substitute for safe publication-time opens and identity checks.

Exactly three flat files are required. Only optional EMPTY tmp/home/shm directories
are additionally allowed. RootOutput numbered siblings, unique temporary files,
hidden objects, SQLite residuals and every other name fail closed, even on failed
runs. Zero-byte fixed files are allowed: structure is not proof of valid art/ROOT
content. No caller-provided filenames, globbing, recursion or relaxed mode exists.
Filesystem operations require POSIX descriptor-relative, no-follow support and a
trusted responsive filesystem; concurrent hostile same-UID mutation is not claimed.
"""

from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import Enum
import os
import stat


class OutputLayout(Enum):
    REFERENCE = "reference"
    NOG4_FIXTURE = "nog4_fixture"


# Source names mirror guarded_run's _OUTPUTS, not its publication destinations.
_OUTPUTS = {
    OutputLayout.REFERENCE: ("raw_1.art", "export.root", "job.fcl"),
    OutputLayout.NOG4_FIXTURE: ("flash_reference_fixture_1.art", "export.root", "job.fcl"),
}
_SCRATCH = ("tmp", "home", "shm")


class InventoryError(ValueError):
    """The complete work tree could not be accepted; retain it for diagnosis."""


@dataclass(frozen=True)
class StatIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def _from_stat(cls, info):
        return cls(info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                   info.st_uid, info.st_gid, info.st_size, info.st_mtime_ns,
                   info.st_ctime_ns)


@dataclass(frozen=True)
class MeasuredFile:
    name: str
    identity: StatIdentity

    @property
    def byte_count(self):
        return self.identity.size


@dataclass(frozen=True)
class MeasuredInventory:
    work: str
    layout: OutputLayout
    work_identity: StatIdentity
    files: tuple[MeasuredFile, ...]
    empty_directories: tuple[str, ...]
    verified: bool = field(default=False, init=False)

    @property
    def total_bytes(self):
        return sum(item.byte_count for item in self.files)


def _snapshot(fd):
    return StatIdentity._from_stat(os.fstat(fd))


def _entry(parent, name):
    return StatIdentity._from_stat(os.stat(name, dir_fd=parent, follow_symlinks=False))


def _unchanged(actual, expected, name):
    if actual != expected:
        raise InventoryError("object changed during inventory: " + name)


def _names(fd, allowed):
    # Stream entries: even arbitrarily many extras are not collected or omitted.
    found = set()
    with os.scandir(fd) as entries:
        for entry in entries:
            if entry.name not in allowed:
                raise InventoryError("unexpected entry: " + entry.name)
            found.add(entry.name)
    return found


def validate_inventory(work, layout=OutputLayout.REFERENCE):
    """Measure only the exact fixed layout after independent caller termination.

    ``work`` must be an existing absolute text path without dot components. Every
    ancestor and work itself must be a real directory, never a symlink. Fixed files
    must be singly linked regular files (empty is permitted). The optional scratch
    directories must be real and completely empty, including hidden entries.
    Missing files, any extra object, changed identities and filesystem errors raise
    InventoryError. Nothing is removed or modified, and no content is read.
    ``layout`` must be this module's enum, not a string or caller-defined mapping.
    """
    if not isinstance(layout, OutputLayout):
        raise InventoryError("layout must be a fixed OutputLayout enum")
    try:
        path = os.fspath(work)
    except TypeError as exc:
        raise InventoryError("work must be an absolute text path") from exc
    if not isinstance(path, str) or not path.startswith("/") or "\0" in path or any(
            part in (".", "..") for part in path.split("/")):
        raise InventoryError("work must be an absolute text path without dot components")
    parts = tuple(part for part in path.split("/") if part)
    path = "/" + "/".join(parts)
    expected = _OUTPUTS[layout]
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        with ExitStack() as stack:
            def opened(name, flags, parent=None):
                fd = os.open(name, flags, dir_fd=parent)
                stack.callback(os.close, fd)
                return fd

            root = opened("/", directory_flags)
            parent = root
            ancestry = []
            for part in parts:
                before = _entry(parent, part)
                if not stat.S_ISDIR(before.mode):
                    raise InventoryError("work and ancestors must be real directories: " + part)
                child = opened(part, directory_flags, parent)
                _unchanged(_snapshot(child), before, part)
                ancestry.append((parent, part, child, before))
                parent = child
            work_fd = parent
            work_before = _snapshot(work_fd)
            names = _names(work_fd, frozenset(expected + _SCRATCH))
            missing = set(expected) - names
            if missing:
                raise InventoryError("missing required files: " + ", ".join(sorted(missing)))
            files = []
            checked = []
            for name in expected:
                before = _entry(work_fd, name)
                if not stat.S_ISREG(before.mode) or before.links != 1:
                    raise InventoryError("expected singly linked regular file: " + name)
                fd = opened(name, file_flags, work_fd)
                _unchanged(_snapshot(fd), before, name)
                files.append(MeasuredFile(name, before))
                checked.append((work_fd, name, fd, before))
            scratch = tuple(name for name in _SCRATCH if name in names)
            for name in scratch:
                before = _entry(work_fd, name)
                if not stat.S_ISDIR(before.mode):
                    raise InventoryError("expected real empty scratch directory: " + name)
                fd = opened(name, directory_flags, work_fd)
                _unchanged(_snapshot(fd), before, name)
                _names(fd, frozenset())
                checked.append((work_fd, name, fd, before))
            # Check observed mutation under the caller's stopped-writers precondition.
            if _names(work_fd, frozenset(expected + _SCRATCH)) != names:
                raise InventoryError("entries changed during inventory")
            for directory, name, fd, before in checked:
                _unchanged(_snapshot(fd), before, name)
                _unchanged(_entry(directory, name), before, name)
            _unchanged(_snapshot(work_fd), work_before, path)
            for directory, name, fd, before in ancestry:
                # Ancestor sibling activity is allowed; check ancestor identity/type.
                after = _entry(directory, name)
                if (after.device, after.inode, after.mode) != (
                        before.device, before.inode, before.mode):
                    raise InventoryError("work ancestor replaced: " + name)
            return MeasuredInventory(path, layout, work_before, tuple(files), scratch)
    except OSError as exc:
        raise InventoryError("cannot inventory work: " + str(exc)) from exc
