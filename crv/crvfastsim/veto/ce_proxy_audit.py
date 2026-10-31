"""Bounded retained CE scalar availability audit, not a selection or physics audit.

Only the externally pinned old report supplies exact ntuple identities. ROOT
UUIDs, header digests and consumed-array digests are NOT full-file hashes or
proof of complete historical execution. No measured calibration is inferred.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import uuid

import numpy as np
import uproot

from .. import dcache

SCHEMA = "ce-proxy-availability-v1"
XROOTD_TIMEOUT = 30
SELECTION_SHA256 = "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
JSON_CAP, SOURCE_CAP, OUTPUT_CAP = 8 << 20, 64 << 10, 2 << 20
FILE_CAP, TOTAL_CAP, READ_CAP = 128 << 20, 1 << 30, 32 << 20
METADATA_CAP, BRANCH_BYTES, PAYLOAD_CAP = 8 << 20, 1 << 20, 16 << 20
MAX_KEYS, MAX_BRANCHES, MAX_BASKETS, TOTAL_BASKETS = 512, 4096, 128, 4096
MAX_ENTRIES, MAX_TOTAL_ENTRIES, EXPECTED_ENTRIES = 10000, 100000, 91226
VERSION_CAP, VERSION_BINS = 64 << 10, 16
INDICES = frozenset((*range(7), *range(8, 12)))
CAMPAIGN = "CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001"
PATH_TAIL = (r"/tape/phy-nts/nts/mu2e/" + CAMPAIGN +
    r"/root/[0-9a-f]{2}/[0-9a-f]{2}/nts\.mu2e\.CeMLeadingLogMix1BB\."
    r"MDC2025au_best_v1_1-001\.001430_([0-9]{8})\.root")


def path_re():
    """Exact campaign path in the site's dCache area (CRVFS_DCACHE)."""
    return re.compile(re.escape(dcache.area()) + PATH_TAIL)
IDS = ("evtinfo/run", "evtinfo/subrun", "evtinfo/event")
KINDS = dict.fromkeys(IDS, "id") | {
    "evtinfo/nprotons": "count", "evtinfo/pbtime": "time", "evtinfo/pbterr": "error",
    **{"lumistream./lumistream." + n: "count" for n in
       ("nProtonTCs", "nTrackerHits", "nCaloHitsD0", "nCaloHitsD1")},
    "lumistream./lumistream.caloEnergy": "energy",
    "lumistream./lumistream.caphriIsValid": "bool"}
COUNTS = ("entries", "finite", "nonfinite", "sentinel_minus_one", "zero", "positive",
          "other_negative", "invalid", "valid")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def local(value):
    require(isinstance(value, (str, Path)) and str(value) and "://" not in str(value), "local path required")
    path = Path(os.path.abspath(value))
    require("pnfs" not in path.parts, "local small files cannot be on PNFS")
    return path


def no_symlinks(path):
    for part in [*reversed(path.parents), path]:
        require(not stat.S_ISLNK(part.lstat().st_mode), "symlinks forbidden")


def small_bytes(path, cap):
    no_symlinks(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= cap, "small-file size/type cap")
    with path.open("rb") as stream:
        payload = stream.read(cap + 1)
    require(len(payload) <= cap, "small-file read cap")
    return payload


def outside(out, excluded):
    require(not os.path.lexists(out), "new output required; overwrite forbidden")
    no_symlinks(out.parent)
    require(out.parent.is_dir(), "existing output directory required")
    require(not any(out.is_relative_to(p) for p in excluded), "output must be outside code/evidence/selected-file parent")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def bindings(payload):
    report = json.loads(payload, object_pairs_hook=unique_object,
                        parse_constant=lambda x: require(False, "nonfinite JSON"))
    require(type(report) is dict, "selection report object required")
    selected = report.get("selection")
    require(type(selected) is dict and type(selected.get("path")) is str and
            selected["path"].startswith("/") and "://" not in selected["path"], "absolute selected path required")
    require(type(selected.get("sha256")) is str and re.fullmatch(r"[0-9a-f]{64}", selected["sha256"]), "selected SHA binding required")
    records = report.get("ntuples")
    require(type(records) is list and len(records) == 11, "exact eleven ntuple records required")
    seen, rows, total = set(), [], 0
    for row in records:
        require(type(row) is dict and type(row.get("path")) is str, "ntuple path required")
        match = path_re().fullmatch(row["path"])
        require(match is not None, "exact PNFS campaign path required")
        index = int(match[1])
        require(index in INDICES and index not in seen, "wrong/duplicate basename index")
        seen.add(index)
        identity = row.get("uuid")
        require(type(identity) is str, "UUID required")
        require(str(uuid.UUID(identity)) == identity, "canonical UUID required")
        trees = row.get("trees")
        require(type(trees) is dict and type(trees.get("EventNtuple/ntuple")) is dict, "saved exact tree required")
        entries = trees["EventNtuple/ntuple"].get("entries")
        require(type(entries) is int and 0 < entries <= MAX_ENTRIES, "saved tree entry cap/type")
        total += entries
        rows.append(dict(path=row["path"], uuid=identity, entries=entries, index=index))
    require(seen == INDICES and total == EXPECTED_ENTRIES and total <= MAX_TOTAL_ENTRIES, "saved total entries differ")
    ledger_parent = None
    if "exposure_ledger" in report:
        ledger = report["exposure_ledger"]
        require(type(ledger) is dict and type(ledger.get("path")) is str and
                bool(ledger["path"]) and "://" not in ledger["path"], "saved ledger pathname required")
        ledger_parent = Path(ledger["path"]).parent
    return sorted(rows, key=lambda r: r["index"]), selected, ledger_parent


class Budget:
    def __init__(self):
        self.total = 0
        self.files = {}


class BoundedFile:
    """No stat/fstat/namespace operations, even on close or header reopen.

    Repeated reads count again, including metadata/header verification. Caps are
    cumulative by exact pathname across both opens, not an RSS guarantee.
    """
    def __init__(self, path, budget):
        self.path, self.budget = path, budget
        self.fd = os.open(path, os.O_RDONLY)
        self.closed = False
        budget.files.setdefault(path, 0)

    def read(self, size=-1):
        require(isinstance(size, (int, np.integer)) and not isinstance(size, (bool, np.bool_)) and
                0 <= size <= READ_CAP, "bounded single read required")
        size = int(size)
        require(self.budget.files[self.path] + size <= FILE_CAP, "per-file read cap")
        require(self.budget.total + size <= TOTAL_CAP, "total read cap")
        data = os.read(self.fd, size)
        self.budget.files[self.path] += len(data)
        self.budget.total += len(data)
        return data

    def tell(self):
        return os.lseek(self.fd, 0, os.SEEK_CUR)

    def seek(self, offset, whence=os.SEEK_SET):
        require(isinstance(offset, (int, np.integer)) and not isinstance(offset, (bool, np.bool_)) and
                whence in (os.SEEK_SET, os.SEEK_CUR), "SEEK_END/unbounded seek forbidden")
        return os.lseek(self.fd, int(offset), whence)

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def xrootd_url(path):
    # Lexical transform only, after the pinned report has validated identities.
    # The door is the site's configured CRVFS_XROOTD; no user URL, alternate campaign or namespace lookup is admitted.
    match = path_re().fullmatch(path) if type(path) is str else None
    require(match is not None and int(match[1]) in INDICES, "exact PNFS campaign path required for XRootD")
    return dcache.xrootd(path)


class XRootDBoundedFile(BoundedFile):
    """Synchronous direct client calls only; offsets are maintained lexically.

    Every open/read/close has a 30-second client timeout. This is not a
    whole-run deadline or a guarantee against a client violating its timeout.
    Short reads fail explicitly, rather than requesting any replacement bytes.
    """
    def __init__(self, path, budget):
        self.path, self.budget = path, budget
        self.url = xrootd_url(path)
        from XRootD import client  # optional dependency; never imported for NFS
        self._file = client.File()
        self.closed, self.position = False, 0
        budget.files.setdefault(path, 0)  # exact PNFS key, including header reopen
        try:
            status, _ = self._file.open(self.url, flags=0, mode=0, timeout=XROOTD_TIMEOUT)
            self._check(status, "open")
        except BaseException:
            self.close()  # attempted even if open failed; never retry
            raise

    def _check(self, status, operation):
        if not status.ok:
            raise OSError(f"XRootD {operation} failed for {self.url}: status={status}")

    def read(self, size=-1):
        require(not self.closed, "read of closed XRootD file")
        require(isinstance(size, (int, np.integer)) and not isinstance(size, (bool, np.bool_)) and
                0 <= size <= READ_CAP, "bounded single read required")
        size = int(size)
        if size == 0:
            return b""  # XRootD read(size=0) is UNBOUNDED: never issue it
        require(self.budget.files[self.path] + size <= FILE_CAP, "per-file read cap")
        require(self.budget.total + size <= TOTAL_CAP, "total read cap")
        status, data = self._file.read(offset=self.position, size=size, timeout=XROOTD_TIMEOUT)
        # Count all bytes returned, even on a failed status or malformed/short
        # response. There is no retry or automatic transport fallback.
        if isinstance(data, bytes):
            self.budget.files[self.path] += len(data)
            self.budget.total += len(data)
            self.position += len(data)
        self._check(status, "read")
        require(isinstance(data, bytes), f"XRootD read payload is not bytes: status={status}")
        require(len(data) <= size, f"XRootD read payload exceeds requested {size}: status={status}")
        require(len(data) == size, f"XRootD short read: requested {size}, received {len(data)}; status={status}")
        return data

    def tell(self):
        require(not self.closed, "tell of closed XRootD file")
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        require(not self.closed, "seek of closed XRootD file")
        require(isinstance(offset, (int, np.integer)) and not isinstance(offset, (bool, np.bool_)) and
                isinstance(whence, (int, np.integer)) and not isinstance(whence, (bool, np.bool_)) and
                whence in (os.SEEK_SET, os.SEEK_CUR), "SEEK_END/unbounded seek forbidden")
        position = int(offset) + (self.position if whence == os.SEEK_CUR else 0)
        require(position >= 0, "negative XRootD position forbidden")
        self.position = position
        return position

    def close(self):
        if not self.closed:
            self.closed = True  # even a failed close is not automatically retried
            status, _ = self._file.close(timeout=XROOTD_TIMEOUT)
            self._check(status, "close")


def xrootd_version():
    import XRootD
    from XRootD import client
    value = getattr(XRootD, "__version__", None) or getattr(client, "__version__", None)
    if value is None:
        from importlib.metadata import PackageNotFoundError, version
        try:
            value = version("XRootD")
        except PackageNotFoundError:
            return None
    return str(value)


def empty_basket_array(chunk, cursor, maximum):
    """Prove a supported resident TObjArray contains only literal null slots."""
    require(0 < maximum <= TOTAL_BASKETS, "tree branch basket allocation cap")
    require(chunk is not None and cursor is not None, "resident basket metadata required")
    raw = getattr(chunk, "_raw_data", None)
    require(raw is not None and not chunk.is_memmap, "resident basket metadata required")
    cursor = cursor.copy()
    start, limit = cursor.index, chunk.stop
    require(chunk.start <= start <= limit and len(raw) == limit - chunk.start,
            "resident basket metadata bounds")

    def take(size, *, signed=False):
        require(0 <= size <= limit - cursor.index, "truncated basket metadata envelope")
        begin = cursor.index - chunk.start
        cursor.skip(size)
        return int.from_bytes(raw[begin:begin + size], "big", signed=signed)

    count = take(4)
    require(count & 0xc0000000 == 0x40000000, "basket metadata bytecount required")
    end = start + (count & 0x3fffffff) + 4
    # The fixed envelope is 25 bytes; a long name adds at most 256 bytes plus its four-byte length.
    require(25 <= end - start <= min(METADATA_CAP, 285 + 4 * maximum) and end <= limit,
            "basket metadata envelope cap/bounds")
    limit = end
    require(take(2) == 3, "unsupported basket TObjArray version/memberwise")
    require(take(2) == 1, "unsupported basket TObject envelope")
    take(4)
    require(not take(4) & uproot.const.kIsReferenced, "referenced basket TObject forbidden")
    name_size = take(1)
    if name_size == 255:
        name_size = take(4)
    require(name_size <= 256, "basket metadata name cap")
    take(name_size)
    size, lower = take(4, signed=True), take(4, signed=True)
    require(0 <= size <= maximum and lower == 0, "basket metadata slot count/lower bound")
    require(end - cursor.index == 4 * size, "basket metadata slot envelope mismatch")
    for _ in range(size):
        require(take(4) == 0, "embedded baskets forbidden during metadata: nonnull basket slot")
    require(cursor.index == end, "basket metadata envelope mismatch")
    return end


@contextmanager
def metadata_limits():
    """Scoped uproot hooks bound key counts before key-list allocation.

    This CLI is deliberately single-threaded; do not run concurrent uproot work
    in this context. Hooks are restored even on errors. Bounds also cover ROOT
    streamer/key decompression, not just selected branch arrays.
    """
    directory, key = uproot.reading.ReadOnlyDirectory, uproot.reading.ReadOnlyKey
    old_read, old_keys, old_object = directory.hook_before_read_keys, directory.hook_before_keys, key.get_uncompressed_chunk_cursor
    model = uproot.model.Model
    old_postprocess, old_before = model.hook_before_postprocess, model.hook_before_read
    cursor_type = uproot.source.cursor.Cursor
    old_skip = cursor_type.skip_over

    def read_keys(self, **kwargs):
        require(0 <= self._fNbytesKeys <= METADATA_CAP, "directory metadata byte cap")
        return old_read(self, **kwargs)

    def keys(self, **kwargs):
        require(0 <= kwargs["num_keys"] <= MAX_KEYS, "directory key count cap")
        return old_keys(self, **kwargs)

    def obj(self):
        require(0 <= self.data_uncompressed_bytes <= METADATA_CAP and
                0 <= self.data_compressed_bytes <= METADATA_CAP, "ROOT metadata object byte cap")
        return old_object(self)

    def skip(self, chunk, context):
        breadcrumbs = context.get("breadcrumbs", ())
        branch = breadcrumbs[-1] if breadcrumbs else None
        saved = getattr(branch, "_cursor_baskets", None)
        if getattr(branch, "classname", None) == "TBranch" and saved is not None and saved.index == self.index:
            end = empty_basket_array(chunk, saved, branch.member("fMaxBaskets"))
            require(old_skip(self, chunk, context) and self.index == end,
                    "basket metadata skip fallback forbidden")
            return True
        return old_skip(self, chunk, context)

    def before(self, **kwargs):
        breadcrumbs = kwargs["context"].get("breadcrumbs", ())
        if self.classname == "TBranch":
            require(type(self) in uproot.models.TBranch.Model_TBranch.known_versions.values(),
                    "unsupported TBranch metadata reader")
        require(self.classname != "TObjArrayOfTBaskets", "lazy embedded basket deserialization forbidden")
        if self.classname == "TBasket":
            require("basket_num" in kwargs["context"] and
                    not any(x.classname in ("TTree", "TBranch", "TBranchElement") for x in breadcrumbs),
                    "TBasket deserialization forbidden during metadata")
        if self.classname == "TObjArray":
            for branch in breadcrumbs[:-1]:
                saved = getattr(branch, "_cursor_baskets", None)
                require(not (getattr(branch, "classname", None) == "TBranch" and saved is not None and
                             saved.index == kwargs["cursor"].index),
                        "fBaskets deserialization fallback forbidden during metadata")
        return old_before(self, **kwargs)

    def postprocess(self, **kwargs):
        # Validate every base branch, including matching coverage, before uproot's fallback or lazy decode.
        if self.classname == "TBranch":
            maximum, written = self.member("fMaxBaskets"), self.member("fWriteBasket")
            require(0 < maximum <= TOTAL_BASKETS and 0 <= written < maximum,
                    "tree branch basket allocation cap")
            seeks, offsets, sizes = (np.asarray(self.member(name)) for name in
                                    ("fBasketSeek", "fBasketEntry", "fBasketBytes"))
            require(all(x.ndim == 1 and x.dtype.kind in "iu" and len(x) <= maximum
                        for x in (seeks, offsets, sizes)), "tree branch basket metadata bounds")
            normal = min(int(written), int(np.flatnonzero(seeks == 0)[0]) if np.any(seeks == 0) else len(seeks))
            require(normal < len(offsets), "basket-header fallback forbidden during metadata")
            require(all(len(x) == maximum for x in (seeks, offsets, sizes)),
                    "tree branch basket metadata lengths")
            entries = self.member("fEntries")
            children, leaves = len(self.member("fBranches")), len(self.member("fLeaves"))
            require(0 <= entries <= MAX_ENTRIES and children <= MAX_BRANCHES and leaves <= MAX_BRANCHES,
                    "tree branch metadata count cap")
            structural = (children > 0 and entries > 0 and normal == written == offsets[normal] == 0 and
                          not np.any(seeks) and not np.any(sizes) and not np.any(offsets) and
                          self.member("fTotBytes") == self.member("fZipBytes") == 0)
            require(entries == offsets[normal] or structural,
                    "embedded baskets forbidden during metadata: "
                    f"name={self.member('fName')[:256]!r}, entries={entries}, "
                    f"normal={normal}, boundary={offsets[normal]}, written={written}, "
                    f"children={children}, leaves={leaves}, "
                    f"total_bytes={self.member('fTotBytes')}, zip_bytes={self.member('fZipBytes')}")
            empty_basket_array(kwargs.get("chunk"), getattr(self, "_cursor_baskets", None), maximum)
            self._members["fBaskets"] = []
        return old_postprocess(self, **kwargs)

    model.hook_before_read = before
    cursor_type.skip_over = skip
    model.hook_before_postprocess = postprocess
    directory.hook_before_read_keys, directory.hook_before_keys = read_keys, keys
    key.get_uncompressed_chunk_cursor = obj
    try:
        yield
    finally:
        directory.hook_before_read_keys, directory.hook_before_keys = old_read, old_keys
        key.get_uncompressed_chunk_cursor = old_object
        model.hook_before_postprocess = old_postprocess
        model.hook_before_read = old_before
        cursor_type.skip_over = old_skip


def exact_key(directory, name, *, optional=False):
    # Only immediate key metadata; never recursive namespace discovery.
    matches = [k for k in directory._keys if k.fName == name]
    if not matches:
        require(optional, "missing exact ROOT key: " + name)
        return None
    cycles = [k.fCycle for k in matches]
    require(len(set(cycles)) == len(cycles), "duplicate ROOT key cycle")
    return max(matches, key=lambda k: k.fCycle)


def tree_header(root, expected):
    require(str(root.file.uuid) == expected["uuid"], "ROOT UUID differs")
    dkey = exact_key(root, "EventNtuple")
    require(dkey.fClassName in ("TDirectory", "TDirectoryFile"), "EventNtuple directory class")
    directory = dkey.get()
    tkey = exact_key(directory, "ntuple")
    require(tkey.fClassName == "TTree", "exact ntuple must be TTree")
    tree = tkey.get()
    require(tree.num_entries == expected["entries"] and 0 < tree.num_entries <= MAX_ENTRIES, "ROOT entries differ/cap")
    return directory, tree, dict(directory=dkey.fCycle, ntuple=tkey.fCycle)


def version(directory):
    key = exact_key(directory, "version", optional=True)
    if key is None:
        return dict(available=False, reason="exact EventNtuple/version absent")
    meta = dict(cycle=key.fCycle, classname=key.fClassName,
                uncompressed_bytes=key.data_uncompressed_bytes, compressed_bytes=key.data_compressed_bytes)
    if key.fClassName != "TH1I":
        return dict(available=False, reason="exact version is not TH1I; not deserialized", **meta)
    require(0 <= key.data_uncompressed_bytes <= VERSION_CAP and
            0 <= key.data_compressed_bytes <= VERSION_CAP, "version object byte cap")
    hist = key.get()
    require(0 <= hist.member("fNcells") <= VERSION_BINS + 2 and
            0 <= hist.axis().member("fNbins") <= VERSION_BINS, "version bin cap")
    values, flow = hist.values(flow=False), hist.values(flow=True)
    require(values.ndim == 1 and len(values) <= VERSION_BINS and len(flow) == len(values) + 2, "version shape cap")
    labels = hist.axis().labels()
    require(labels is None or (len(labels) <= VERSION_BINS and all(len(s) <= 256 for s in labels)), "version label cap")
    parts = [int(x) for x in values] if len(values) == 3 and np.all(values >= 0) else None
    return dict(available=True, **meta, values=values.tolist(), values_with_flow=flow.tolist(),
                axis_labels=labels, axis_edges=hist.axis().edges().tolist(),
                major_minor_patch=parts, equals_reference_6_13_2=None if parts is None else parts == [6, 13, 2],
                interpretation="Persisted version observation only; differing version is a finding, not a failure or writer/configuration proof.")


def branch_metadata(branch, name, entries):
    require(name in KINDS, "branch is not allowlisted")
    require(branch.classname in ("TBranch", "TBranchElement"), "branch class not scalar-admissible")
    require(len(branch.branches) == 0, "branch must be an exact scalar leaf")
    interpretation = branch.interpretation
    require(isinstance(interpretation, uproot.interpretation.numerical.AsDtype), "non-scalar/jagged interpretation")
    dtype = np.dtype(interpretation.numpy_dtype)
    require(dtype.fields is None and dtype.subdtype is None and dtype.kind in "biuf" and dtype.itemsize <= 8, "non-scalar numeric dtype")
    kind = KINDS[name]
    require(dtype.kind in ("iu" if kind in ("id", "energy") else "biu" if kind in ("count", "bool") else "f"), "scalar dtype domain differs")
    require(branch.num_entries == entries, "branch entries differ")
    require(0 <= branch.member("fMaxBaskets") <= TOTAL_BASKETS, "branch basket allocation cap")
    require(branch._embedded_baskets is not None and len(branch._embedded_baskets) == 0,
            "embedded baskets not admitted before payload")
    require(0 < branch.num_baskets <= MAX_BASKETS and branch._num_normal_baskets == branch.num_baskets, "basket count/embedded basket cap")
    require(0 <= branch.uncompressed_bytes <= BRANCH_BYTES and
            0 <= branch.compressed_bytes <= BRANCH_BYTES, "branch declared byte cap")
    uncompressed = compressed = stop = 0
    for i in range(branch.num_baskets):  # basket metadata, never events
        start, end = branch.basket_entry_start_stop(i)
        require(start == stop and start <= end <= entries, "basket entry bounds")
        stop = end
        key = branch.basket_key(i)  # header only, not basket() which decompresses
        require(0 <= key.data_uncompressed_bytes <= BRANCH_BYTES and
                0 <= key.data_compressed_bytes <= BRANCH_BYTES, "basket byte cap")
        uncompressed += key.data_uncompressed_bytes
        compressed += key.data_compressed_bytes
    require(stop == entries and max(uncompressed, compressed) <= BRANCH_BYTES, "basket total payload cap")
    return dict(classname=branch.classname, interpretation=str(interpretation), numpy_dtype=dtype.str,
                entries=entries, baskets=branch.num_baskets, uncompressed_bytes=uncompressed,
                compressed_bytes=compressed, declared_uncompressed_bytes=int(branch.uncompressed_bytes))


def summary(array, kind):
    finite = np.isfinite(array)
    sentinel = finite & (array == -1) & (kind in ("count", "energy"))
    invalid = ~finite | (finite & (array < 0) & ~sentinel)
    if kind == "bool":
        invalid |= finite & (array != 0) & (array != 1)
    # EventInfo time fields have no documented -1 sentinel; signed pbtime is valid.
    if kind == "time":
        invalid = ~finite
    valid = finite & ~sentinel & ~invalid
    values = array[valid]
    counts = dict(entries=len(array), finite=int(finite.sum()), nonfinite=int((~finite).sum()),
        sentinel_minus_one=int(sentinel.sum()), zero=int((finite & (array == 0)).sum()),
        positive=int((finite & (array > 0)).sum()), other_negative=int((finite & (array < 0) & ~sentinel).sum()),
        invalid=int(invalid.sum()), valid=int(valid.sum()))
    stats = None
    if len(values):
        floats = values.astype(np.float64)
        scale = float(np.max(np.abs(floats))) or 1.
        scaled = floats / scale  # avoid overflow even where longdouble == double
        stats = dict(min=values.min().item(), max=values.max().item(), mean=float(scaled.mean() * scale),
                     quantile_probabilities=[0, .25, .5, .75, 1],
                     quantiles=(np.quantile(scaled, [0, .25, .5, .75, 1]) * scale).tolist(),
                     quantile_method="linear; all per-file valid values, scaled float64 arithmetic (not sampled)")
    return dict(counts=counts, valid_statistics=stats,
                valid_rule="finite; -1 missing only for count/energy; nonnegative counts/energy/pbterr, boolean 0/1; all signed pbtime allowed",
                zero_interpretation="Persisted zero, distinct from count/energy -1; EventInfo zero can be unfilled; filling/calibration semantics not established.")


def array_digest(array):
    return dict(sha256=sha(array.tobytes(order="C")), dtype=array.dtype.str, shape=list(array.shape),
                order="C", bytes=int(array.nbytes), scope="consumed array C bytes only; dtype/shape recorded separately")


def inspect_file(expected, budget, *, transport="nfs"):
    require(transport in ("nfs", "xrootd"), "unknown transport")
    opener = BoundedFile if transport == "nfs" else XRootDBoundedFile
    path = expected["path"]
    executor = uproot.source.futures.TrivialExecutor()
    options = dict(object_cache=None, array_cache=None, decompression_executor=executor,
                   interpretation_executor=executor, use_threads=False, minimal_ttree_metadata=True)
    with opener(path, budget) as stream:
        header_sha = sha(stream.read(100))
        stream.seek(0)
        with uproot.open(stream, **options) as root:
            directory, tree, cycles = tree_header(root, expected)
            ver = version(directory)
            names = tree.keys(recursive=True, full_paths=True)
            require(len(names) <= MAX_BRANCHES and len(names) == len(set(names)), "tree branch count/duplicate cap")
            require(all(n in names for n in IDS), "required exact ID leaves missing")
            metadata, branches = {}, {}
            # Admit every requested scalar before reading any array payload.
            for name in KINDS:
                if name in names:
                    branches[name] = tree[name]
                    metadata[name] = branch_metadata(branches[name], name, expected["entries"])
            require(sum(m["baskets"] for m in metadata.values()) <= TOTAL_BASKETS, "total basket cap")
            require(sum(m["uncompressed_bytes"] for m in metadata.values()) <= PAYLOAD_CAP, "total scalar payload cap")
            results, ids = {}, []
            for name, kind in KINDS.items():
                if name not in metadata:
                    results[name] = dict(available=False, reason="exact optional leaf absent; no aliases substituted")
                    continue
                array = branches[name].array(library="np", array_cache=None,
                    decompression_executor=executor, interpretation_executor=executor)
                require(isinstance(array, np.ndarray) and array.shape == (expected["entries"],) and
                        array.dtype.str == metadata[name]["numpy_dtype"], "consumed scalar shape/dtype differs")
                results[name] = dict(available=True, metadata=metadata[name], consumed_array=array_digest(array), **summary(array, kind))
                if name in IDS:
                    require(np.all(array >= 0) and np.all(array <= np.iinfo(np.int64).max), "invalid full ID domain")
                    ids.append(array.astype(np.int64))
            identities = np.column_stack(ids)
            require(len(np.unique(identities, axis=0)) == len(identities), "duplicate full IDs within file")
    payload_read_bytes = budget.files[path]
    # A new descriptor and ROOT object verify UUID/count/cycles, without arrays.
    with opener(path, budget) as stream:
        require(sha(stream.read(100)) == header_sha, "ROOT header changed")
        stream.seek(0)
        with uproot.open(stream, **options) as root:
            _, _, after_cycles = tree_header(root, expected)
            require(after_cycles == cycles, "ROOT chosen cycles changed")
    return dict(**expected, chosen_cycles=cycles, header_sha256=header_sha, header_bytes=100,
                header_reopened=True, transport=transport,
                resolved_url=xrootd_url(path) if transport == "xrootd" else None, version=ver, scalars=results,
                read_bytes_before_reopen=payload_read_bytes, read_bytes=budget.files[path]), identities


def aggregate(files):
    result = {}
    for name in KINDS:
        present = [f["scalars"][name] for f in files if f["scalars"][name]["available"]]
        result[name] = dict(available_files=len(present), unavailable_files=len(files) - len(present),
            counts={key: sum(p["counts"][key] for p in present) for key in COUNTS},
            aggregate_quantiles=None, interpretation="Exact sums over available files only; no fabricated aggregate quantiles.")
    return result


def run(selection_report, out, *, expected_selection_sha256, transport="nfs"):
    require(transport in ("nfs", "xrootd"), "unknown transport")
    require(expected_selection_sha256 == SELECTION_SHA256, "mandatory hardbound selection SHA256 differs")
    source, selection_report, out = local(__file__), local(selection_report), local(out)
    excluded = [source.parents[2], selection_report.parent]
    outside(out, excluded)
    source_bytes = small_bytes(source, SOURCE_CAP)
    payload = small_bytes(selection_report, JSON_CAP)
    require(sha(payload) == SELECTION_SHA256, "selection report hash differs from hard pin")
    records, selected, ledger_parent = bindings(payload)
    excluded.append(Path(os.path.abspath(selected["path"])).parent)  # lexical; never inspect selected file
    if ledger_parent is not None:
        excluded.append(Path(os.path.abspath(selection_report.parent / ledger_parent)))  # never open ledger
    outside(out, excluded)
    budget, files, identities = Budget(), [], []
    with metadata_limits():
        for record in records:
            row, ids = inspect_file(record, budget, transport=transport)
            files.append(row)
            identities.append(ids)
    all_ids = np.concatenate(identities)
    require(len(all_ids) == EXPECTED_ENTRIES and len(all_ids) <= MAX_TOTAL_ENTRIES, "observed total entries differ")
    require(len(np.unique(all_ids, axis=0)) == len(all_ids), "duplicate full IDs across files")
    result = dict(schema=SCHEMA, selection_report=dict(path=str(selection_report), sha256=sha(payload)),
        source=dict(path=str(source), sha256=sha(source_bytes), snapshot=source_bytes.decode("utf-8")),
        software=dict(python=platform.python_version(), numpy=np.__version__, uproot=uproot.__version__,
            **({"xrootd": xrootd_version()} if transport == "xrootd" else {})),
        transport=transport,
        resolved_urls={r["path"]: r["resolved_url"] for r in files} if transport == "xrootd" else {},
        transport_details=dict(api="XRootD.client.File" if transport == "xrootd" else "os.open/read/lseek/close",
            per_call_timeout_seconds={operation: XROOTD_TIMEOUT for operation in ("open", "read", "close")}
                if transport == "xrootd" else None,
            application_retries=0, automatic_fallback=False,
            short_read_policy="raise" if transport == "xrootd" else "return actual bytes to uproot"),
        files=files, aggregate=aggregate(files), total_entries=len(all_ids), full_ids_unique=True,
        read_bytes=budget.total, per_file_read_bytes=budget.files,
        bounds=dict(file_read=FILE_CAP, total_read=TOTAL_CAP, single_read=READ_CAP, keys=MAX_KEYS,
            branches=MAX_BRANCHES, baskets_per_branch=MAX_BASKETS, baskets_total=TOTAL_BASKETS,
            branch_payload=BRANCH_BYTES, payload_per_file=PAYLOAD_CAP, metadata_object=METADATA_CAP,
            entries_per_file=MAX_ENTRIES, entries_total=MAX_TOTAL_ENTRIES, version_object=VERSION_CAP,
            version_bins=VERSION_BINS, selection_json=JSON_CAP, source=SOURCE_CAP, output=OUTPUT_CAP),
        selected_track_provenance=dict(**selected, status="inherited only; not opened/stat'ed/hashed"),
        proof_limits=["No full-file SHA256; ROOT header/UUID and consumed arrays do not prove complete execution or immutable payloads.",
            "Header reopen checks UUID, entries, cycles and first 100 bytes only; basket changes with unchanged metadata are not excluded.",
            "No historical writer/configuration, calibration, filled-zero semantics or measured-intensity validity established.",
            "No selection, selected Parquet, old ledger, MC, tracks, CRV, old-audit execution, model, correlation, efficiency or F calculation.",
            "Byte/count caps are not a whole-process RSS limit. Scoped uproot metadata hooks require single-threaded use.",
            "XRootD timeouts are per client call, not a whole-run deadline; no application retries/fallback. NFS has no per-call timeout."],
        full_file_hashes=False, selection_executed=False, mc_read=False, prior_audit_executed=False,
        weight_model=None, F=None, finite_library_uncertainty=None,
        publication="exclusive x output; no atomic-completion or crash-durability claim")
    require(small_bytes(selection_report, JSON_CAP) == payload, "selection JSON changed")
    require(small_bytes(source, SOURCE_CAP) == source_bytes, "audit source changed")
    output = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(output) <= OUTPUT_CAP, "output byte cap")
    outside(out, excluded)
    with out.open("xb") as stream:
        stream.write(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_report")
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--transport", choices=("nfs", "xrootd"), default="nfs",
                        help="exact pinned files only; XRootD uses 30-second per-call timeouts; default: nfs")
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps(dict(schema=result["schema"], total_entries=result["total_entries"], read_bytes=result["read_bytes"])))


if __name__ == "__main__":
    main()
