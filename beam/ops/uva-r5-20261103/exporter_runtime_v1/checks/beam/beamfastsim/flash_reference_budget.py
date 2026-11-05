"""Bound logical saved contents, not filesystem metadata, caches, or temporary data.

This is not a sandbox, executor, transport, or confinement check. Publishing
requires independently verified confinement AND payload termination. The caller
must exclusively own the destination tree and prevent external writers (including
writes through other hard links). Only this trusted, serialized in-process writer
may modify saved files. There is deliberately no reopening, retry, or cleanup API.
"""

from dataclasses import dataclass
import json
import os
import stat
import threading


SLOT_NAMES = (
    "bpzup21_first10", "bpzup12_first10",
    "bpzup21_remaining990", "bpzup12_remaining990",
)
SLOT_BUDGET_BYTES = 5 * 1024**3
TOTAL_BUDGET_BYTES = len(SLOT_NAMES) * SLOT_BUDGET_BYTES
MAX_LOG_BYTES = 1024**2
DEFAULT_STATUS_RESERVE = 4096
MIN_STATUS_RESERVE = 128
CHUNK_BYTES = 64 * 1024
PAYLOAD_FILENAMES = frozenset(("output.art", "output.root", "config.fcl"))
FIXED_FILENAMES = PAYLOAD_FILENAMES | frozenset((
    "reservation.json", "start.json", "status.json", "merged.log", "manifest.json",
))


class SavedOutputError(RuntimeError):
    """Failure snapshot; even incomplete files remain saved and fully charged."""

    def __init__(self, reason, message, saved_bytes, file_bytes=0):
        super().__init__(message)
        self.reason = reason
        self.saved_bytes = saved_bytes
        self.file_bytes = file_bytes


@dataclass(frozen=True)
class SavedFile:
    name: str
    saved_bytes: int
    total_saved_bytes: int


@dataclass(frozen=True)
class LogReceipt:
    saved_bytes: int
    discarded_bytes: int
    total_saved_bytes: int


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _parts(path):
    path = os.fspath(path)
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("a nonempty text filesystem path is required")
    parts = path.split("/")
    if any(part in (".", "..") for part in parts):
        raise ValueError("dot path components are forbidden")
    return path.startswith("/"), [part for part in parts if part]


def _open_directory(path):
    absolute, parts = _parts(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/" if absolute else ".", flags)
    try:
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_source(path):
    absolute, parts = _parts(path)
    if not parts:
        raise ValueError("source must name a regular file")
    parent = ("/" if absolute else "") + "/".join(parts[:-1])
    parent_fd = _open_directory(parent) if parent else os.open(
        ".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        info = _regular_single_link(fd)
        return fd, info
    except BaseException:
        os.close(fd)
        raise


def _regular_single_link(fd):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("only singly linked regular files are allowed")
    return info


class SerializedBudget:
    """Consume one fixed slot under an existing, trusted root directory.

    Existing slots are always refused, including abandoned/partially constructed
    ones. Smaller limits support tests; limits above production caps are invalid.
    All operations are serialized on one object. Successful writes are fsynced;
    failures retain every written byte and all exclusive lifecycle markers.
    """

    def __init__(self, root, slot, *, byte_limit=SLOT_BUDGET_BYTES,
                 log_limit=MAX_LOG_BYTES, status_reserve=DEFAULT_STATUS_RESERVE):
        if slot not in SLOT_NAMES:
            raise ValueError("unknown fixed slot")
        if any(type(value) is not int for value in (byte_limit, log_limit, status_reserve)):
            raise ValueError("limits must be integers")
        if not MIN_STATUS_RESERVE <= status_reserve < byte_limit <= SLOT_BUDGET_BYTES:
            raise ValueError("invalid slot limit or final-status reserve")
        if not 0 <= log_limit <= MAX_LOG_BYTES:
            raise ValueError("invalid log cap")
        self.slot = slot
        self.byte_limit = byte_limit
        self.log_limit = log_limit
        self.status_reserve = status_reserve
        self._saved = 0
        self._files = {}
        self._lock = threading.RLock()
        self._dir_fd = None
        self._log_fd = None
        self._closed = False
        self._start_attempted = False
        self._started = False
        self._finish_attempted = False
        root_fd = _open_directory(root)
        try:
            os.mkdir(slot, mode=0o700, dir_fd=root_fd)
            os.fsync(root_fd)
            self._dir_fd = os.open(slot, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW |
                                   os.O_CLOEXEC, dir_fd=root_fd)
        finally:
            os.close(root_fd)
        try:
            self._write_blob("reservation.json", _json_bytes({
                "slot": slot, "byte_limit": byte_limit, "status_reserve": status_reserve,
            }))
        except BaseException:
            self.close()
            raise

    @property
    def saved_bytes(self):
        with self._lock:
            return self._saved

    @property
    def remaining_bytes(self):
        with self._lock:
            return max(0, self.byte_limit - self.status_reserve - self._saved)

    def _error(self, reason, message, name=None):
        return SavedOutputError(reason, message, self._saved, self._files.get(name, 0))

    def _live(self, require_started=True):
        if self._closed or self._finish_attempted:
            raise self._error("lifecycle", "writer is closed or finalization was attempted")
        if require_started and not self._started:
            raise self._error("lifecycle", "a successful exclusive start is required")

    def _create(self, name):
        if name not in FIXED_FILENAMES:
            raise self._error("name", "filename is not allowlisted")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW |
                     os.O_CLOEXEC, 0o600, dir_fd=self._dir_fd)
        try:
            _regular_single_link(fd)
            self._files[name] = 0
            os.fsync(self._dir_fd)
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _write(self, fd, name, data, *, final=False):
        view = memoryview(data).cast("B")
        offset = 0
        while offset < len(view):
            _regular_single_link(fd)
            ceiling = self.byte_limit if final else self.byte_limit - self.status_reserve
            available = ceiling - self._saved
            if available <= 0:
                raise self._error("budget", "saved-content budget exhausted", name)
            amount = min(len(view) - offset, available, CHUNK_BYTES)
            try:
                written = os.write(fd, view[offset:offset + amount])
            except InterruptedError:
                continue
            if not 0 < written <= amount:
                raise OSError("invalid or zero-length write")
            self._saved += written
            self._files[name] += written
            offset += written

    def _write_blob(self, name, data, *, final=False):
        fd = None
        try:
            fd = self._create(name)
            self._write(fd, name, data, final=final)
            os.fsync(fd)
            return SavedFile(name, self._files[name], self._saved)
        except SavedOutputError:
            raise
        except (OSError, ValueError) as exc:
            raise self._error("io", str(exc), name) from exc
        finally:
            if fd is not None:
                os.close(fd)

    def start(self):
        """Permanently consume the one start attempt, even when marker writing fails."""
        with self._lock:
            self._live(require_started=False)
            if self._start_attempted:
                raise self._error("lifecycle", "start already attempted")
            self._start_attempted = True
            result = self._write_blob("start.json", _json_bytes({"started": True}))
            self._started = True
            return result

    def stream_log(self, chunks):
        """Drain merged byte chunks; discard excess without allocating a full log."""
        with self._lock:
            self._live()
            before = self._files.get("merged.log", 0)
            discarded = 0
            try:
                if self._log_fd is None:
                    self._log_fd = self._create("merged.log")
                for chunk in chunks:
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise ValueError("log chunks must be bytes-like")
                    view = memoryview(chunk).cast("B")
                    allowance = min(self.remaining_bytes,
                                    self.log_limit - self._files["merged.log"])
                    take = min(len(view), allowance)
                    self._write(self._log_fd, "merged.log", view[:take])
                    discarded += len(view) - take
                os.fsync(self._log_fd)
            except SavedOutputError:
                raise
            except (OSError, ValueError) as exc:
                raise self._error("io", str(exc), "merged.log") from exc
            return LogReceipt(self._files["merged.log"] - before, discarded, self._saved)

    def publish(self, source, name, *, confinement_verified=False, payload_terminated=False):
        """Caller MUST independently verify confinement and termination before publish.

        The keyword assertions record caller responsibility, not security checks.
        Only fixed payload names are accepted. A budget failure retains a bounded
        prefix and reports its charge; it never removes or retries the destination.
        """
        with self._lock:
            self._live()
            if confinement_verified is not True or payload_terminated is not True:
                raise self._error("prerequisite", "independently verified confinement and payload termination required")
            if name not in PAYLOAD_FILENAMES:
                raise self._error("name", "not a fixed publishable payload name")
            source_fd = destination_fd = None
            try:
                source_fd, original = _open_source(source)
                destination_fd = self._create(name)
                copied = 0
                while copied < original.st_size:
                    allowance = self.remaining_bytes
                    if not allowance:
                        raise self._error("budget", "source exceeds saved-content allowance", name)
                    chunk = os.read(source_fd, min(CHUNK_BYTES, allowance, original.st_size - copied))
                    if not chunk:
                        raise self._error("source_changed", "source shortened during publication", name)
                    self._write(destination_fd, name, chunk)
                    copied += len(chunk)
                current = _regular_single_link(source_fd)
                if (current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                        original.st_size, original.st_mtime_ns, original.st_ctime_ns):
                    raise self._error("source_changed", "source changed during publication", name)
                os.fsync(destination_fd)
                return SavedFile(name, self._files[name], self._saved)
            except SavedOutputError:
                raise
            except (OSError, ValueError) as exc:
                raise self._error("io", str(exc), name) from exc
            finally:
                if destination_fd is not None:
                    os.close(destination_fd)
                if source_fd is not None:
                    os.close(source_fd)

    def write_manifest(self):
        """Write a single fixed-name, budget-charged inventory receipt."""
        with self._lock:
            self._live()
            return self._write_blob("manifest.json", _json_bytes({
                "slot": self.slot, "files": dict(self._files),
                "saved_bytes_before_manifest": self._saved,
            }))

    def finish(self, state="complete"):
        """Use reserved bytes for one final status including its own saved bytes."""
        with self._lock:
            self._live(require_started=False)
            if state not in ("complete", "failed"):
                raise ValueError("state must be complete or failed")
            if state == "complete" and not self._started:
                raise self._error("lifecycle", "an unstarted slot cannot complete")
            self._finish_attempted = True
            if self._log_fd is not None:
                try:
                    os.fsync(self._log_fd)
                except OSError as exc:
                    raise self._error("io", str(exc), "merged.log") from exc
            total = self._saved
            while True:
                data = _json_bytes({"state": state, "saved_bytes": total})
                next_total = self._saved + len(data)
                if next_total == total:
                    break
                total = next_total
            if len(data) > self.status_reserve:
                raise self._error("budget", "final status does not fit reserved allowance")
            return self._write_blob("status.json", data, final=True)

    def close(self):
        """Close descriptors without deleting, resuming, retrying, or adding receipts."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._log_fd is not None:
                    os.close(self._log_fd)
                    self._log_fd = None
            finally:
                if self._dir_fd is not None:
                    os.close(self._dir_fd)
                    self._dir_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
