"""Descriptive conditional-retained-event joint tables, never a CE correction.

Only local pinned saved JSON, Parquet and numeric NPZ inputs are consumed. No ROOT
reader, old selection audit, extractor, calibration or uncertainty code is used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import struct
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = "ce-joint-diagnostic-v1"
SELECTION_SHA256 = "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
SELECTED_SHA256 = "7f60b88899f35bea7cf0a438aa18250ff0021ad229bc3c000bfd180fa386a2ba"
LEDGER_SHA256 = "b9f94c948099cdeddb1e27634bb4f2f232619aa11d854ae5a96d4d5339e955cf"
AVAILABILITY_SHA256 = "2b59602667126391fe1b32bb6ce4051b83c88268f5f60c580922d1cfeef18325"
EXPECTED_ENTRIES = 91226
REPORT_CAP, SOURCE_CAP = 8 * 1024**2, 64 * 1024
SELECTED_CAP, LEDGER_CAP, ROWS_CAP = 64 * 1024**2, 8 * 1024**2, 16 * 1024**2
FOOTER_CAP, DECODED_CAP = 1024**2, 256 * 1024**2
MAX_ROWS = MAX_TRACKS = 2100000
MAX_TRACKS_PER_ROW, MAX_LEDGER_ROWS = 64, 100000
OUTPUT_CAP = 16 * 1024**2
IDS = ("run", "subrun", "event")
PROXIES = ("nProtonTCs", "nTrackerHits", "nCaloHitsD0", "nCaloHitsD1", "caloEnergy")
ROW_KEYS = (*IDS, "source_index", "measured_nprotons", "measured_pbtime", "measured_pbterr", *PROXIES, "caphriIsValid")
LEDGER_KEYS = (*IDS, "nprotons", "pbtime", "measured_nprotons", "measured_pbtime")
SELECTED_KEYS = (*IDS, "t_mid", "p_mid")
PB_EDGES = [-240, -230, -220, -210, -200, -190]
TIME_EDGES = list(range(0, 2001, 100))
TC_EDGES = [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 9.5]
HIT_EDGES = [0, 1, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256")
    return value


def local_path(value):
    require(isinstance(value, (str, Path)) and bool(str(value)) and ":" not in str(value)
            and not str(value).startswith("//"), "local files only")
    path = Path(os.path.abspath(value))
    require(not set(path.parts) & {"pnfs", "net", "nfs", "afs", "cvmfs"}
            and not any(p.lower().endswith(".root") for p in path.parts), "ROOT/network paths forbidden")
    return path


def no_symlinks(path):
    for component in [*reversed(path.parents), path]:
        require(not stat.S_ISLNK(component.lstat().st_mode), "symlink forbidden")


def bounded_bytes(path, cap):
    no_symlinks(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and 0 <= info.st_size <= cap, "regular file byte cap exceeded")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require((info.st_dev, info.st_ino, info.st_size) ==
                (opened.st_dev, opened.st_ino, opened.st_size), "file changed at open")
        data = stream.read(cap + 1)
    require(len(data) == info.st_size and len(data) <= cap, "file changed/byte cap exceeded")
    return data


def outside(out, paths):
    require(not os.path.lexists(out), "new output required")
    no_symlinks(out.parent)
    require(out.parent.is_dir(), "existing output directory required")
    for path in paths:
        require(not out.is_relative_to(path.parent), "output under input/code directory forbidden")


class Evidence:
    def __init__(self):
        self.records = {}

    def read(self, path, cap, expected=None):
        path = local_path(path)
        data = bounded_bytes(path, cap)
        digest = sha(data)
        if expected is not None:
            require(digest == pin(expected), "input SHA256 mismatch: " + str(path))
        self.records[path] = dict(path=str(path), sha256=digest, bytes=len(data), cap=cap)
        return data

    def recheck(self):
        for path, record in self.records.items():
            require(sha(bounded_bytes(path, record["cap"])) == record["sha256"], "input/source changed before output")


def json_object(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    require(type(result) is dict, "JSON object required")
    return result


def integer_array(array, name):
    require(array.ndim == 1 and array.dtype.kind in "iu" and array.dtype.itemsize <= 8,
            "integer scalar required: " + name)
    require(not len(array) or (np.all(array >= 0) and np.all(array <= np.iinfo(np.int64).max)),
            "nonnegative signed-int64 domain required: " + name)
    return array.astype(np.int64, copy=False)


def finite_array(array, name, nonnegative=False):
    require(array.ndim == 1 and array.dtype.kind in "biuf" and array.dtype.itemsize <= 8,
            "numeric scalar required: " + name)
    require(np.isfinite(array).all(), "nonfinite values: " + name)
    if nonnegative:
        require(np.all(array >= 0), "negative/reset value: " + name)
    return array


def identities(columns):
    arrays = [integer_array(columns[k], k) for k in IDS]
    require(len({len(a) for a in arrays}) == 1, "ID lengths differ")
    keys = np.empty(len(arrays[0]), dtype=[(k, "<i8") for k in IDS])
    for name, array in zip(IDS, arrays):
        keys[name] = array
    require(len(np.unique(keys)) == len(keys), "duplicate full event identity")
    return keys


def lookup(needles, haystack):
    """Full structured-tuple join; no packed IDs or truncation."""
    if not len(haystack):
        return np.full(len(needles), -1, dtype=np.int64)
    order = np.argsort(haystack)
    index = np.searchsorted(haystack[order], needles)
    safe = np.minimum(index, len(haystack) - 1)
    valid = (index < len(haystack)) & (haystack[order[safe]] == needles)
    return np.where(valid, order[safe], -1)


def zip_envelope(data, infos):
    # NumPy uses canonical local ZIP64 size extras even for small seekable files.
    # Admit that envelope only; no prefix, hidden members, descriptors or trailer.
    cursor = 0
    for info in infos:
        require(info.header_offset == cursor and cursor + 30 <= len(data), "ZIP hidden/prefix bytes")
        header = struct.unpack_from("<4s5H3I2H", data, cursor)
        signature, version, flags, method, _, _, crc, compressed, expanded, nlen, elen = header
        require(signature == b"PK\x03\x04" and flags == info.flag_bits and not flags & 8
                and method == info.compress_type and crc == info.CRC, "ZIP local header differs")
        end = cursor + 30 + nlen + elen
        require(end <= len(data) and data[cursor + 30:cursor + 30 + nlen] == info.filename.encode("ascii"),
                "ZIP member name/header differs")
        extra = data[cursor + 30 + nlen:end]
        if compressed == 0xffffffff or expanded == 0xffffffff:
            require(compressed == expanded == 0xffffffff and len(extra) == 20
                    and extra[:4] == b"\x01\x00\x10\x00", "ZIP local extras forbidden")
            expanded, compressed = struct.unpack("<QQ", extra[4:])
        else:
            require(not extra, "ZIP local extras forbidden")
        require((expanded, compressed) == (info.file_size, info.compress_size), "ZIP local sizes differ")
        cursor = end + compressed
    require(len(data) >= 22 and data[-22:-18] == b"PK\x05\x06", "ZIP trailing bytes forbidden")
    _, disk, cd_disk, n_disk, n_all, cd_size, cd_offset, comment = struct.unpack("<4s4H2IH", data[-22:])
    require(disk == cd_disk == comment == 0 and n_disk == n_all == len(infos)
            and cd_offset == cursor and cd_offset + cd_size == len(data) - 22, "ZIP directory bounds differ")


def load_rows(data, description):
    require(description.get("entries") == EXPECTED_ENTRIES, "rows entry count differs")
    columns = description.get("columns")
    require(type(columns) is dict and set(columns) == set(ROW_KEYS), "rows column allowlist differs")
    require(len(data) <= ROWS_CAP, "NPZ byte cap exceeded")
    arrays, total = {}, 0
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        require(len(infos) == len(ROW_KEYS) == 13 and {i.filename for i in infos} == {k + ".npy" for k in ROW_KEYS},
                "NPZ member allowlist differs")
        require(not archive.comment, "NPZ archive comment forbidden")
        zip_envelope(data, infos)
        for info in infos:
            require(not info.extra and not info.comment and not info.is_dir() and not info.flag_bits & 1,
                    "NPZ extra/encrypted member forbidden")
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "NPZ compression forbidden")
            require(0 <= info.file_size <= ROWS_CAP, "NPZ member byte cap")
            total += info.file_size
        require(total <= ROWS_CAP, "NPZ uncompressed byte cap")
        # Header validation precedes NumPy allocation; numeric-only, no pickle.
        for info in infos:
            name = info.filename[:-4]
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "NPY version forbidden")
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                require(shape == (EXPECTED_ENTRIES,) and dtype.fields is None and dtype.subdtype is None
                        and dtype.kind in "biuf" and dtype.itemsize <= 8 and not dtype.hasobject,
                        "NPY numeric dtype/shape required")
                require(stream.tell() + EXPECTED_ENTRIES * dtype.itemsize == info.file_size,
                        "NPY payload size differs")
            raw = archive.read(info)
            array = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096)
            record = columns[name]
            require(type(record) is dict and record.get("dtype") == array.dtype.str
                    and record.get("shape") == list(array.shape), "rows declared dtype/shape differs")
            if "sha256" in record:
                require(pin(record["sha256"]) == sha(array.tobytes(order="C")), "rows array hash differs")
            arrays[name] = array  # original dtype and values, never a resaved cast
    identities(arrays)
    integer_array(arrays["source_index"], "source_index")
    for name in ROW_KEYS[4:]:
        finite_array(arrays[name], name, nonnegative=name != "measured_pbtime")
    for name in PROXIES:
        integer_array(arrays[name], name)
    require(np.isin(arrays["caphriIsValid"], [0, 1]).all(), "validity must be 0/1")
    return arrays


def parquet_file(data, keys, row_cap):
    require(len(data) >= 12 and data[:4] == b"PAR1" and data[-4:] == b"PAR1", "invalid Parquet envelope")
    footer = struct.unpack("<I", data[-8:-4])[0]
    require(0 < footer <= FOOTER_CAP and footer <= len(data) - 12, "Parquet footer cap/bounds")
    parquet = pq.ParquetFile(pa.BufferReader(data), thrift_string_size_limit=FOOTER_CAP,
                             thrift_container_size_limit=100000, pre_buffer=False)
    meta = parquet.metadata
    require(0 <= meta.num_rows <= row_cap and 0 <= meta.num_row_groups <= 4096
            and 0 < meta.num_columns <= 256, "Parquet metadata bounds")
    names = parquet.schema_arrow.names
    require(len(names) == len(set(names)) and set(keys) <= set(names), "missing/duplicate Parquet columns")
    declared = group_rows = 0
    for i in range(meta.num_row_groups):
        group = meta.row_group(i)
        require(0 <= group.num_rows <= row_cap and group.num_columns == meta.num_columns,
                "Parquet row group bounds")
        group_rows += group.num_rows
        for j in range(group.num_columns):
            column = group.column(j)
            if column.path_in_schema.split(".")[0] in keys:
                require(0 <= column.num_values <= MAX_ROWS + MAX_TRACKS, "Parquet selected leaf value cap")
                require(0 <= column.total_uncompressed_size <= DECODED_CAP, "declared selected column cap")
                declared += column.total_uncompressed_size
    require(group_rows == meta.num_rows, "Parquet row group count mismatch")
    require(declared <= DECODED_CAP, "declared selected columns byte cap")
    return parquet


def storage(array):
    while isinstance(array, pa.ExtensionArray):
        require(array.type.extension_name == "awkward", "unsupported Arrow extension")
        array = array.storage
    return array


def scalar_column(array, name):
    array = storage(array)
    require(array.null_count == 0 and (pa.types.is_integer(array.type) or pa.types.is_floating(array.type)
            or pa.types.is_boolean(array.type)), "null/nonnumeric scalar: " + name)
    return array.to_numpy(zero_copy_only=False)


def read_ledger(data):
    parquet = parquet_file(data, LEDGER_KEYS, MAX_LEDGER_ROWS)
    require(parquet.metadata.num_rows == EXPECTED_ENTRIES, "ledger entry count differs")
    parts = {k: [] for k in LEDGER_KEYS}
    decoded = 0
    for batch in parquet.iter_batches(batch_size=16384, columns=list(LEDGER_KEYS), use_threads=False):
        decoded += batch.nbytes
        require(decoded <= DECODED_CAP, "decoded ledger byte cap")
        for name in LEDGER_KEYS:
            parts[name].append(scalar_column(batch.column(batch.schema.get_field_index(name)), name))
    arrays = {k: np.concatenate(v) if v else np.empty(0, dtype=np.int64) for k, v in parts.items()}
    identities(arrays)
    for name in LEDGER_KEYS[3:]:
        finite_array(arrays[name], name, nonnegative="nprotons" in name)
    return arrays


def ragged(array, name):
    array = storage(array)
    require(pa.types.is_list(array.type) or pa.types.is_large_list(array.type), "ragged list required: " + name)
    require(array.null_count == 0, "null ragged list: " + name)
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    lengths = np.diff(offsets)
    require(np.all((lengths >= 0) & (lengths <= MAX_TRACKS_PER_ROW)), "tracks per row cap")
    values = array.values.slice(int(offsets[0]), int(offsets[-1] - offsets[0]))
    flat = finite_array(scalar_column(values, name), name)
    return lengths, flat


def read_selected(data):
    parquet = parquet_file(data, SELECTED_KEYS, MAX_ROWS)
    ids = {k: [] for k in IDS}
    times, events, positions, lengths_parts, original_lengths = [], [], [], [], []
    rows = tracks = decoded = at85 = at200 = below = above = 0
    for batch in parquet.iter_batches(batch_size=16384, columns=list(SELECTED_KEYS), use_threads=False):
        decoded += batch.nbytes
        require(decoded <= DECODED_CAP, "decoded selected columns byte cap")
        arrays = {k: batch.column(batch.schema.get_field_index(k)) for k in SELECTED_KEYS}
        for k in IDS:
            ids[k].append(integer_array(scalar_column(arrays[k], k), k))
        nt, t = ragged(arrays["t_mid"], "t_mid")
        np_, p = ragged(arrays["p_mid"], "p_mid")
        require(np.array_equal(nt, np_), "misaligned time/momentum lists")
        original_lengths.append(nt)
        tracks += len(p)
        require(tracks <= MAX_TRACKS, "selected track cap")
        cut = (p > 85) & (p < 200)
        event = np.repeat(np.arange(batch.num_rows, dtype=np.int64), nt)
        pos = np.arange(len(p), dtype=np.int64) - np.repeat(np.cumsum(nt) - nt, nt)
        events.append(event[cut] + rows)
        positions.append(pos[cut])
        times.append(t[cut])
        lengths_parts.append(np.bincount(event[cut], minlength=batch.num_rows))
        at85 += int(np.count_nonzero(p == 85))
        at200 += int(np.count_nonzero(p == 200))
        below += int(np.count_nonzero(p < 85))
        above += int(np.count_nonzero(p > 200))
        rows += batch.num_rows
    require(rows == parquet.metadata.num_rows, "selected decoded row count differs")
    concat = lambda items, dtype: np.concatenate(items) if items else np.empty(0, dtype=dtype)
    keys = identities({k: concat(v, np.int64) for k, v in ids.items()})  # before any event cut
    multiplicity = concat(lengths_parts, np.int64)
    return dict(keys=keys, times=concat(times, float), events=concat(events, np.int64),
                positions=concat(positions, np.int64), multiplicity=multiplicity,
                original_multiplicity=concat(original_lengths, np.int64),
                counts=dict(full_selected_rows=rows, full_selected_tracks=tracks,
                    momentum_window_events=int(np.count_nonzero(multiplicity)),
                    momentum_window_tracks=int(multiplicity.sum()), exactly_85=at85,
                    exactly_200=at200, below_85=below, above_200=above,
                    zero_window_rows=int(np.count_nonzero(multiplicity == 0))))


def edges_from_report(report):
    strata = report["diagnostic"]["pbi_quantile_strata"]
    require(type(strata) is list and 1 <= len(strata) <= 32, "inherited MC strata bounds")
    edges = [strata[0]["pbi_lo"]]
    for i, row in enumerate(strata):
        require(row["pbi_lo"] == edges[-1] and row["pbi_hi"] > edges[-1]
                and row["upper_inclusive"] is (i == len(strata) - 1), "inherited MC strata not contiguous")
        edges.append(row["pbi_hi"])
    finite_array(np.asarray(edges), "MC edges", nonnegative=True)
    return edges


def bins(values, edges, last_inclusive=False):
    result = np.searchsorted(np.asarray(edges), values, side="right")
    if last_inclusive:
        result = np.where(values == edges[-1], len(edges) - 1, result)
    return result


def axis(edges, last_inclusive=False):
    return dict(edges=edges, bins=len(edges) + 1, underflow_index=0, overflow_index=len(edges),
                convention="left closed, right open; explicit underflow/overflow",
                last_finite_upper_inclusive=last_inclusive)


def divide(numerator, denominator):
    n, d = np.broadcast_arrays(numerator, denominator)
    result = np.full(n.shape, None, dtype=object)
    np.divide(n, d, out=result, where=d != 0, casting="unsafe")
    return result.tolist()


def grid(indices, shape, multiplicity, track_event, track_time_bin):
    cells = int(np.prod(shape))
    index = np.ravel_multi_index(tuple(indices), shape)
    def count(weights=None):
        if weights is None:
            return np.bincount(index, minlength=cells)
        result = np.zeros(cells, dtype=np.int64)
        np.add.at(result, index, np.asarray(weights, dtype=np.int64))
        return result
    n = count()
    a = count(multiplicity > 0)
    t = count(multiplicity)
    mult = np.bincount(index * 5 + np.minimum(multiplicity, 4), minlength=cells * 5).reshape(cells, 5)
    ntimes = len(TIME_EDGES) + 1
    hist = np.bincount(index[track_event] * ntimes + track_time_bin, minlength=cells * ntimes).reshape(cells, ntimes)
    require(np.array_equal(hist.sum(axis=1), t), "track histogram does not reconcile T")
    require(np.array_equal(mult.sum(axis=1), n) and np.array_equal(n - mult[:, 0], a), "multiplicity does not reconcile")
    event_support = n >= 100
    track_support = (a >= 20) & (t >= 30)
    def shaped(v, extra=()):
        return v.reshape((*shape, *extra)).tolist()
    unsupported = {}
    for name, mask in (("event_support", event_support), ("track_support", track_support),
                       ("joint_support", event_support & track_support)):
        unsupported[name] = dict(events=int(n[~mask].sum()), tracks=int(t[~mask].sum()),
            event_fraction=float(n[~mask].sum() / n.sum()) if n.sum() else None,
            track_fraction=float(t[~mask].sum() / t.sum()) if t.sum() else None)
    return dict(shape=list(shape), N_all_retained=shaped(n), A_events_with_tracks=shaped(a),
                T_tracks=shaped(t), multiplicity_0_1_2_3_ge4=shaped(mult, (5,)),
                selected_track_raw_time_histogram=shaped(hist, (ntimes,)),
                event_selection_probability_A_over_N=divide(a.reshape(shape), n.reshape(shape)),
                tracks_per_event_T_over_N_not_probability=divide(t.reshape(shape), n.reshape(shape)),
                tracks_per_selected_event_T_over_A=divide(t.reshape(shape), a.reshape(shape)),
                support_N_ge100=shaped(event_support), support_A_ge20_and_T_ge30=shaped(track_support),
                unsupported=unsupported)


def exact_measurement_equal(left, right, name):
    # Avoid float64 coercion falsely equating distinct >2**53 integers.
    if left.dtype.kind in "iu" and right.dtype.kind in "iu":
        equal = np.array_equal(integer_array(left, name), integer_array(right, name))
    elif left.dtype.kind == "f" and right.dtype.kind == "f":
        equal = np.array_equal(left, right)
    else:
        integer, floating = (left, right) if left.dtype.kind in "iu" else (right, left)
        require(integer.dtype.kind in "iu" and floating.dtype.kind == "f", "measurement dtype mismatch")
        ints = integer_array(integer, name)
        require(np.all((floating >= 0) & (floating < 2**63)) and np.all(floating == np.floor(floating)),
                "measurement cast not exact")
        equal = np.array_equal(ints, floating.astype(np.int64))
    require(equal, "ledger/proxy measured values differ: " + name)


def diagnose(selection_report, rows_report, rows, ledger, selected):
    mc_edges = edges_from_report(selection_report)
    row_keys, ledger_keys = identities(rows), identities(ledger)
    ledger_order = lookup(row_keys, ledger_keys)
    require(len(row_keys) == len(ledger_keys) and np.all(ledger_order >= 0), "proxy/ledger keyset anti-join")
    ledger = {k: v[ledger_order] for k, v in ledger.items()}
    for name in ("measured_nprotons", "measured_pbtime"):
        exact_measurement_equal(rows[name], ledger[name], name)
    selected_join = lookup(selected["keys"], row_keys)
    matched = selected_join >= 0
    active = selected["multiplicity"] > 0
    multiplicity = np.zeros(len(row_keys), dtype=np.int64)
    multiplicity[selected_join[matched]] = selected["multiplicity"][matched]
    track_matched = matched[selected["events"]]
    track_event = selected_join[selected["events"][track_matched]]
    track_time = selected["times"][track_matched]
    counts = dict(selected["counts"], retained_events=len(row_keys),
        selected_rows_matched_including_empty=int(matched.sum()),
        selected_rows_unmatched_including_empty=int((~matched).sum()),
        full_selected_tracks_matched_before_window=int(selected["original_multiplicity"][matched].sum()),
        full_selected_tracks_unmatched_before_window=int(selected["original_multiplicity"][~matched].sum()),
        selected_events_matched=int(np.count_nonzero(matched & active)),
        selected_tracks_matched=int(track_matched.sum()),
        selected_events_unmatched=int(np.count_nonzero(~matched & active)),
        selected_tracks_unmatched=int((~track_matched).sum()),
        retained_events_without_selected_row=int(len(row_keys) - matched.sum()),
        retained_events_without_window_tracks=int(np.count_nonzero(multiplicity == 0)),
        proxy_not_in_ledger=0, ledger_not_in_proxy=0)
    window = selection_report["momentum_window"]
    require(window.get("pmin") == 85 and window.get("pmax") == 200 and window.get("strict") is True,
            "original momentum window differs")
    for observed, inherited in (("momentum_window_events", "selected_events"), ("momentum_window_tracks", "selected_tracks")):
        require(counts[observed] == window[inherited], "original momentum window counts differ")
    diag = selection_report["diagnostic"]
    for observed, inherited in (("retained_events", "ntuple_events"), ("selected_events_matched", "selected_events_matched"),
            ("selected_tracks_matched", "selected_tracks_matched"),
            ("selected_events_unmatched", "selected_events_not_in_supplied_ntuples")):
        require(counts[observed] == diag[inherited], "original match counts differ")
    require(counts["selected_tracks_unmatched"] == window["selected_tracks"] - diag["selected_tracks_matched"],
            "original unmatched track counts differ")
    mc = bins(ledger["nprotons"], mc_edges, True)
    pb = bins(rows["measured_pbtime"], PB_EDGES)
    time = bins(track_time, TIME_EDGES)
    for i, inherited in enumerate(diag["pbi_quantile_strata"], 1):
        mask = mc == i
        for value, key in ((int(mask.sum()), "ntuple_events"),
                           (int(np.count_nonzero(multiplicity[mask])), "selected_events"),
                           (int(multiplicity[mask].sum()), "selected_tracks")):
            require(value == inherited[key], "original MC stratum counts differ")
    axes = dict(measured_pbtime=axis(PB_EDGES), raw_stored_t_mid=axis(TIME_EDGES),
                MC_truth_nprotons=axis(mc_edges, True))
    grids = {}
    for name in PROXIES:
        edges = TC_EDGES if name == "nProtonTCs" else HIT_EDGES
        axes[name] = axis(edges)
        grids[name + "__measured_pbtime"] = grid([bins(rows[name], edges), pb],
            (len(edges) + 1, len(PB_EDGES) + 1), multiplicity, track_event, time)
    grids["MC_truth_nprotons__measured_pbtime"] = grid([mc, pb], (len(mc_edges) + 1, len(PB_EDGES) + 1),
                                                         multiplicity, track_event, time)
    grids["nProtonTCs__MC_truth_nprotons__measured_pbtime"] = grid([bins(rows["nProtonTCs"], TC_EDGES), mc, pb],
        (len(TC_EDGES) + 1, len(mc_edges) + 1, len(PB_EDGES) + 1), multiplicity, track_event, time)
    source = integer_array(rows["source_index"], "source_index")
    sources = selection_report["ntuples"]
    files = rows_report.get("files")
    require(type(sources) is list and 0 < len(sources) <= 100 and type(files) is list
            and len(files) == len(sources), "source file records required")
    original_sources = {r["path"]: r for r in sources}
    require(len(original_sources) == len(sources), "duplicate original source paths")
    seen_indices, seen_paths = set(), set()
    summaries = []
    for record in sorted(files, key=lambda r: r["index"]):  # per-source, never per-event
        i, path = record["index"], record["path"]
        require(type(i) is int and 0 <= i <= 2**31 - 1 and i not in seen_indices
                and path in original_sources and path not in seen_paths, "source_index binding differs")
        seen_indices.add(i)
        seen_paths.add(path)
        original = original_sources[path]
        declared = original["trees"]["EventNtuple/ntuple"]["entries"]
        require(record["entries"] == declared and record["uuid"] == original["uuid"], "source file identity differs")
        mask = source == i
        require(int(mask.sum()) == declared, "source_index retained entry count differs")
        clock = rows["measured_pbtime"][mask]
        summaries.append(dict(source_index=i, path=record["path"], uuid=record["uuid"],
            retained_events=int(mask.sum()), events_with_tracks=int(np.count_nonzero(multiplicity[mask])),
            selected_tracks=int(multiplicity[mask].sum()),
            multiplicity_0_1_2_3_ge4=np.bincount(np.minimum(multiplicity[mask], 4), minlength=5).tolist(),
            measured_pbtime_min=float(clock.min()) if len(clock) else None,
            measured_pbtime_max=float(clock.max()) if len(clock) else None))
    require(np.isin(source, list(seen_indices)).all(), "source_index not bound by file records")
    # Hash identifies matched full tuple plus original frozen-list position, in canonical order.
    binding = np.empty(len(track_event), dtype=[(k, "<i8") for k in (*IDS, "position")])
    for name in IDS:
        binding[name] = rows[name][track_event]
    binding["position"] = selected["positions"][track_matched]
    binding.sort(order=[*IDS, "position"])
    validity = rows["caphriIsValid"]
    return dict(counts=counts, axes=axes, grids=grids, sources=summaries,
        matched_track_identity=dict(sha256=sha(binding.tobytes()), count=len(binding),
            encoding="C bytes of lexicographically sorted run,subrun,event,original_list_position; four little-endian signed int64"),
        caphriIsValid=dict(zero=int(np.count_nonzero(validity == 0)), one=int(np.count_nonzero(validity == 1)),
                          constant=bool(len(np.unique(validity)) <= 1)),
        measured_nprotons=dict(constant=bool(len(np.unique(rows["measured_nprotons"])) <= 1),
            min=float(rows["measured_nprotons"].min()) if len(row_keys) else None,
            max=float(rows["measured_nprotons"].max()) if len(row_keys) else None),
        measured_pbterr=dict(constant=bool(len(np.unique(rows["measured_pbterr"])) <= 1),
            min=float(rows["measured_pbterr"].min()) if len(row_keys) else None,
            max=float(rows["measured_pbterr"].max()) if len(row_keys) else None,
            interpretation="Observed stored error in ns; not an independent timing calibration."),
        MC_truth_fields=["old_ledger.nprotons", "old_ledger.pbtime"])


def run(selection_report, rows_report, out, *, expected_selection_sha256, expected_rows_sha256):
    require(pin(expected_selection_sha256) == SELECTION_SHA256, "selection pin must equal hardbound pin")
    pin(expected_rows_sha256)
    sp, rp, out = map(local_path, (selection_report, rows_report, out))
    code = local_path(__file__)
    evidence = Evidence()
    outside(out, [sp, rp, code, code.parents[2] / "__code_directory__"])
    source = evidence.read(code, SOURCE_CAP)
    original = json_object(evidence.read(sp, REPORT_CAP, expected_selection_sha256))
    extracted = json_object(evidence.read(rp, REPORT_CAP, expected_rows_sha256))
    require(extracted.get("schema") == "ce-proxy-rows-v1", "rows report schema differs")
    require(extracted.get("selection_report", {}).get("sha256") == SELECTION_SHA256,
            "rows selection binding differs")
    require(extracted.get("availability_report", {}).get("sha256") == AVAILABILITY_SHA256,
            "rows availability binding differs")
    require(original["selection"]["sha256"] == SELECTED_SHA256
            and original["exposure_ledger"]["sha256"] == LEDGER_SHA256, "hardbound selected/ledger hash differs")
    selected_path = local_path(original["selection"]["path"])
    ledger_path = local_path(sp.parent / original["exposure_ledger"]["path"])
    row_path = local_path(extracted["rows"]["path"])
    require(row_path == rp.parent / "rows.npz", "rows path must be report parent/rows.npz")
    paths = [sp, rp, selected_path, ledger_path, row_path, code, code.parents[2] / "__code_directory__"]
    outside(out, paths)
    rows = load_rows(evidence.read(row_path, ROWS_CAP, extracted["rows"]["sha256"]), extracted["rows"])
    ledger = read_ledger(evidence.read(ledger_path, LEDGER_CAP, LEDGER_SHA256))
    selected = read_selected(evidence.read(selected_path, SELECTED_CAP, SELECTED_SHA256))
    result = diagnose(original, extracted, rows, ledger, selected)
    result.update(schema=SCHEMA, software=dict(numpy=np.__version__, pyarrow=pa.__version__),
        bounds=dict(report_bytes=REPORT_CAP, source_bytes=SOURCE_CAP, selected_bytes=SELECTED_CAP,
            ledger_bytes=LEDGER_CAP, npz_bytes=ROWS_CAP, npz_uncompressed_bytes=ROWS_CAP,
            footer_bytes=FOOTER_CAP, selected_rows=MAX_ROWS, selected_tracks=MAX_TRACKS,
            tracks_per_row=MAX_TRACKS_PER_ROW, ledger_rows=MAX_LEDGER_ROWS,
            decoded_selected_columns_bytes=DECODED_CAP, output_bytes=OUTPUT_CAP),
        input_column_allowlists=dict(selected=list(SELECTED_KEYS), ledger=list(LEDGER_KEYS), rows=list(ROW_KEYS)),
        source=dict(path=str(code), sha256=sha(source), snapshot=source.decode("utf-8")),
        inputs=list(evidence.records.values()),
        scope="Descriptive joint diagnostic conditional on retained, trigger/output-conditioned events, not generated CE efficiency.",
        momentum_window=dict(pmin=85, pmax=200, strict=True, other_cuts="frozen upstream; not reapplied"),
        interpretation=[
            "Raw stored t_mid: no wrapping/rebasing; external producer provenance is not fully bound.",
            "EventInfoPBT recipe is MC-smeared with 0.5 ns error; not an independent data measurement.",
            "Per-source measured-pbtime ranges expose possible event-clock/source confounding, not a quantile model.",
            "MC truth strata inherited unchanged; no fit, tuned edges, merging, pseudocounts or independence inference.",
            "A/N is event selection probability conditional on retained events; T/N is tracks/event, not probability.",
            "Track-time histograms condition on selected tracks. Selection probability versus track time is undefined for unselected events.",
            "All zero-denominator rates are null. Track counts are not independent trials; uncertainty not estimated.",
            "No ROOT/MC payload is read. Saved ledger MC fields remain truth diagnostics, never measured corrections.",
            "Full-file input hashes and source snapshot rechecked before exclusive output; no atomicity/durability or filesystem-locality proof."],
        F=None, weight_model=None, calibration=None, finite_library_uncertainty=None,
        uncertainty="not estimated")
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")
    require(len(payload) <= OUTPUT_CAP, "output byte cap")
    evidence.recheck()
    outside(out, paths)
    with out.open("xb") as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_report")
    parser.add_argument("rows_report")
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-rows-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args.selection_report, args.rows_report, args.out,
        expected_selection_sha256=args.expected_selection_sha256,
        expected_rows_sha256=args.expected_rows_sha256)


if __name__ == "__main__":
    main()
