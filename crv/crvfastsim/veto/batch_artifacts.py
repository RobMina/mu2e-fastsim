"""Bounded, append-only batch artifacts; not a physics or campaign admission certificate."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

SCHEMA = "veto-batch-artifacts-v1"
INCOMPLETE = "incomplete.json"
FAILED = "failed.json"
COMPLETE = "complete.json"
PENDING = ".complete.pending.json"
_RESERVED = {INCOMPLETE, FAILED, COMPLETE, PENDING}
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}\.[a-z0-9]{1,16}\Z")
_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])\Z")


class ByteLimitExceeded(RuntimeError):
    """A complete proposed write was refused before any of its bytes were written."""


class TransactionFailed(RuntimeError):
    """A failed, incomplete, reused, or altered transaction cannot commit."""


def _canonical(value):
    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise TypeError("provenance must contain only JSON values")
    check(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                       ensure_ascii=True) + "\n").encode("ascii")


def _name(name):
    if (type(name) is not str or not _NAME.fullmatch(name) or name in _RESERVED
            or _DEVICE.fullmatch(name.split(".")[0])):
        raise ValueError("artifact name must be a safe lowercase basename with one extension")
    return name


def _provenance(context):
    payload = _canonical(context)
    if len(payload) > 16384:
        raise ValueError("provenance exceeds 16384 encoded bytes")
    value = json.loads(payload)
    if type(value) is not dict or set(value) != {"registry_sha256", "batch_identity", "arms", "seeds"}:
        raise ValueError("provide registry_sha256, batch_identity, arms and seeds")
    if not isinstance(value["registry_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["registry_sha256"]):
        raise ValueError("registry_sha256 must be a lowercase SHA256 digest")
    if type(value["batch_identity"]) is not dict or not value["batch_identity"]:
        raise ValueError("batch_identity must be a nonempty JSON object")
    arms, seeds = value["arms"], value["seeds"]
    if (type(arms) is not list or not arms or any(type(a) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", a) for a in arms)
            or len(set(arms)) != len(arms)):
        raise ValueError("arms must be distinct explicit names")
    if (type(seeds) is not list or not seeds or any(type(s) is not int or not 0 <= s < 2**64 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct uint64 JSON integers")
    return value, hashlib.sha256(payload).hexdigest()


class _Budget:
    def __init__(self, limit, reserve):
        self.limit, self.reserve, self.used = limit, reserve, 0

    def check(self, count):
        if count > self.limit - self.reserve - self.used:
            raise ByteLimitExceeded("persisted file-byte cap would be exceeded; batch remains incomplete")


class _Writer(io.RawIOBase):
    def __init__(self, fd, budget, poison):
        super().__init__()
        self._fd, self._budget, self._poison = fd, budget, poison
        self.size, self.digest = 0, hashlib.sha256()

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.size

    def write(self, data):
        try:
            if self.closed:
                raise ValueError("write on closed artifact")
            view = memoryview(data).cast("B")
            self._budget.check(len(view))
            total = 0
            while total < len(view):
                count = os.write(self._fd, view[total:])
                if count <= 0:
                    raise OSError("artifact write made no progress")
                self._budget.used += count
                self.size += count
                self.digest.update(view[total:total + count])
                total += count
            return total
        except BaseException:
            self._poison()
            raise

    def finish(self):
        if self.closed:
            raise TransactionFailed("encoder closed its artifact before verification")
        os.fsync(self._fd)
        info = os.fstat(self._fd)
        return {"bytes": self.size, "sha256": self.digest.hexdigest(),
                "identity": (info.st_dev, info.st_ino)}

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as close_error:
            self._poison()
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note("Artifact stream close failed: " + repr(close_error))
        return False

    def close(self):
        if not self.closed:
            try:
                os.close(self._fd)
            finally:
                super().close()


class BatchArtifacts:
    """One new private directory, one frozen output contract, one context-manager use."""

    def __init__(self, directory, *, max_bytes, context, expected_artifacts):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self.directory = Path(directory)
        self._context, self._context_sha256 = _provenance(context)
        if isinstance(expected_artifacts, (str, bytes)):
            raise TypeError("expected_artifacts must be a sequence of names")
        self._expected = tuple(_name(name) for name in expected_artifacts)
        if not self._expected or len(self._expected) > 128 or len(set(self._expected)) != len(self._expected):
            raise ValueError("declare 1..128 distinct expected artifacts")
        self._expected = tuple(sorted(self._expected))
        common = {"schema": SCHEMA, "context_sha256": self._context_sha256,
                  "execution_admitted": False, "physics_truncated": False,
                  "denominator_renormalized": False}
        self._failure_payload = _canonical(dict(common, state="failed"))
        self._initial_payload = _canonical(dict(common, state="incomplete", context=self._context,
                                               max_bytes=max_bytes, expected_artifacts=list(self._expected),
                                               failure_reserve_bytes=len(self._failure_payload)))
        self._budget = _Budget(max_bytes, len(self._failure_payload))
        self._budget.check(len(self._initial_payload))
        self._records, self._claimed = {}, set()
        self._state, self._poisoned, self._active = "new", False, False
        self._dir_fd = None
        self._common = common

    @property
    def persisted_path_bytes(self):
        """Count bytes actually written, counting retained receipt hardlinks twice."""
        return self._budget.used

    def _poison(self):
        self._poisoned = True

    def _open(self, name):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        return _Writer(os.open(name, flags, 0o600, dir_fd=self._dir_fd), self._budget, self._poison)

    def _control(self, name, payload):
        with self._open(name) as stream:
            stream.write(payload)
            return stream.finish()

    def __enter__(self):
        if self._state != "new":
            raise TransactionFailed("a batch transaction cannot be reused")
        self._state = "opening"
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        try:
            self._dir_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._initial_record = self._control(INCOMPLETE, self._initial_payload)
            os.fsync(self._dir_fd)
            parent = os.open(self.directory.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            self._state = "open"
            return self
        except BaseException as exc:
            self._fail(exc)
            self._close_directory()
            raise

    @contextmanager
    def open_artifact(self, name):
        """Yield an unbuffered, nonseekable writer; any encoder failure poisons the batch."""
        try:
            if self._state != "open" or self._poisoned or self._active:
                raise TransactionFailed("transaction is not writable, failed, or already has an open writer")
            _name(name)
            if name not in self._expected or name in self._claimed:
                raise TransactionFailed("artifact is undeclared or already claimed")
            self._claimed.add(name)
            self._active = True
            with self._open(name) as stream:
                yield stream
                if self._poisoned:
                    raise TransactionFailed("an artifact write failed, even if its exception was caught")
                self._records[name] = stream.finish()
        except BaseException:
            self._poison()
            raise
        finally:
            self._active = False

    def write_json(self, name, value):
        """Encode JSON incrementally; nonfinite numbers fail instead of becoming JSON extensions."""
        with self.open_artifact(name) as stream:
            if not name.endswith(".json"):
                raise ValueError("JSON artifact must have .json extension")
            encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)
            for part in encoder.iterencode(value):
                stream.write(part.encode("ascii"))
            stream.write(b"\n")

    def write_parquet(self, name, table, **options):
        """Write an Arrow table or pandas frame directly through the bounded encoder sink."""
        with self.open_artifact(name) as stream:
            import pyarrow as pa
            import pyarrow.parquet as pq
            if not name.endswith(".parquet"):
                raise ValueError("Parquet artifact must have .parquet extension")
            if not isinstance(table, pa.Table):
                table = pa.Table.from_pandas(table, preserve_index=False)
            pq.write_table(table, stream, **options)

    def _verify(self, name, record):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._dir_fd)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or (before.st_dev, before.st_ino) != record["identity"] or before.st_size != record["bytes"]):
                raise TransactionFailed("artifact identity or size changed before commit: " + name)
            os.fsync(fd)
            digest = hashlib.sha256()
            while block := os.read(fd, 1024 * 1024):
                digest.update(block)
            after = os.fstat(fd)
            if (digest.hexdigest() != record["sha256"] or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns or before.st_ctime_ns != after.st_ctime_ns):
                raise TransactionFailed("artifact content changed before commit: " + name)
        finally:
            os.close(fd)

    def _commit(self):
        if self._poisoned or self._active or set(self._records) != set(self._expected):
            raise TransactionFailed("failed, open, or missing expected artifacts")
        visible, opened = self.directory.stat(follow_symlinks=False), os.fstat(self._dir_fd)
        if not stat.S_ISDIR(visible.st_mode) or (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino):
            raise TransactionFailed("transaction directory changed")
        if set(os.listdir(self._dir_fd)) != {INCOMPLETE, *self._expected}:
            raise TransactionFailed("untracked entry in transaction directory")
        self._verify(INCOMPLETE, self._initial_record)
        for name, record in self._records.items():
            self._verify(name, record)
        artifacts = {name: {key: rec[key] for key in ("bytes", "sha256")} for name, rec in sorted(self._records.items())}
        payload = _canonical(dict(self._common, state="complete", context=self._context,
                                  max_bytes=self._budget.limit, artifacts=artifacts,
                                  artifact_bytes=sum(rec["bytes"] for rec in artifacts.values()),
                                  incomplete_sha256=self._initial_record["sha256"],
                                  receipt_staging_retained=True))
        self._budget.check(2 * len(payload))
        record = self._control(PENDING, payload)
        self._verify(PENDING, record)
        os.fsync(self._dir_fd)
        os.link(PENDING, COMPLETE, src_dir_fd=self._dir_fd, dst_dir_fd=self._dir_fd, follow_symlinks=False)
        self._budget.used += len(payload)
        self._state = "complete"

    def _fail(self, original):
        self._state, self._poisoned = "failed", True
        if self._dir_fd is None:
            return
        try:
            self._budget.reserve = 0
            self._control(FAILED, self._failure_payload)
            os.fsync(self._dir_fd)
        except BaseException as marker_error:
            if hasattr(original, "add_note"):
                original.add_note("Failure marker could not be persisted; absent completion still means incomplete: " + repr(marker_error))
                for note in tuple(getattr(marker_error, "__notes__", ())):
                    original.add_note("Failure marker cleanup: " + note)

    def _close_directory(self):
        if self._dir_fd is not None:
            try:
                os.close(self._dir_fd)
            except OSError:
                pass
            self._dir_fd = None

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None:
                self._fail(exc)
                return False
            try:
                self._commit()
            except BaseException as commit_error:
                self._fail(commit_error)
                raise
            return False
        finally:
            self._close_directory()
