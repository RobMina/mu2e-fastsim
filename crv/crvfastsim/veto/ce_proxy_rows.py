"""Extract pinned raw CE scalar rows; no availability rerun or calibration.

Only report.json denotes completion. Failures may leave an exclusive partial
output directory; no atomic-publication or crash-durability claim is made.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re

import numpy as np
import uproot

from . import ce_proxy_audit as A

SCHEMA = "ce-proxy-rows-v1"
SELECTION_SHA256 = "f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3"
AVAILABILITY_SHA256 = "2b59602667126391fe1b32bb6ce4051b83c88268f5f60c580922d1cfeef18325"
# ce_proxy_audit.py revision that produced the frozen availability report (embedded in it as a source snapshot).
REPORT_AUDIT_SOURCE_SHA256 = "3765b8114f4128d7bae438d3b943cc6f8b61c1c4f393e0963fb4ede6494cc776"
# Current ce_proxy_audit.py: that revision with the dCache mount and xrootd door taken from the site configuration
# (CRVFS_DCACHE, CRVFS_XROOTD) instead of written into the source; paths and URLs are identical when they are set to
# the values the report was made with.
AUDIT_SOURCE_SHA256 = "f776977470c9ed6c62ddb94f902c368cde76ccb8e5f5b64c74ade0bbc364abc7"
JSON_CAP, SOURCE_CAP, REPORT_CAP, NPZ_CAP, PATH_CAP = 8 << 20, 64 << 10, 2 << 20, 16 << 20, 4096
COLUMNS = dict(zip(A.IDS, ("run", "subrun", "event"))) | {
    "evtinfo/nprotons": "measured_nprotons", "evtinfo/pbtime": "measured_pbtime",
    "evtinfo/pbterr": "measured_pbterr",
    **{"lumistream./lumistream." + name: name for name in
       ("nProtonTCs", "nTrackerHits", "nCaloHitsD0", "nCaloHitsD1", "caloEnergy", "caphriIsValid")}}
ROW_KEYS = ("run", "subrun", "event", "source_index", *tuple(COLUMNS.values())[3:])


def pathname(value, *, absolute=True):
    A.require(isinstance(value, (str, Path)) and 0 < len(str(value)) <= PATH_CAP and
              "\x00" not in str(value) and "://" not in str(value), "bounded pathname required")
    path = Path(value)
    A.require(".." not in path.parts and (not absolute or path.is_absolute()), "absolute nontraversing reference required")
    return path


def pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def availability_bindings(payload, records):
    old = json.loads(payload, object_pairs_hook=A.unique_object,
                     parse_constant=lambda x: A.require(False, "nonfinite JSON"))
    A.require(type(old) is dict and old.get("schema") == A.SCHEMA, "availability schema differs")
    selection, source = old.get("selection_report"), old.get("source")
    A.require(type(selection) is dict and selection.get("sha256") == SELECTION_SHA256,
              "availability selection pin differs")
    pathname(selection.get("path"))
    A.require(type(source) is dict and source.get("sha256") == REPORT_AUDIT_SOURCE_SHA256,
              "availability source pin differs")
    pathname(source.get("path"))
    snapshot = source.get("snapshot")
    A.require(type(snapshot) is str and len(snapshot.encode("utf-8")) <= SOURCE_CAP and
              A.sha(snapshot.encode("utf-8")) == REPORT_AUDIT_SOURCE_SHA256, "availability source snapshot differs")
    A.require(type(old.get("total_entries")) is int and old["total_entries"] == A.EXPECTED_ENTRIES and
              old.get("full_ids_unique") is True, "availability total/uniqueness differs")
    files = old.get("files")
    A.require(type(files) is list and len(files) == len(records) == 11, "availability exact file count differs")
    by_path = {}
    for saved in files:
        A.require(type(saved) is dict, "availability file object required")
        path = str(pathname(saved.get("path")))
        A.require(path not in by_path, "duplicate availability path")
        by_path[path] = saved
    A.require(set(by_path) == {r["path"] for r in records}, "availability file paths differ")
    for record in records:
        saved = by_path[record["path"]]
        A.require(all(saved.get(k) == v and type(saved.get(k)) is type(v) for k, v in record.items()),
                  "availability file identity differs")
        cycles = saved.get("chosen_cycles")
        A.require(type(cycles) is dict and set(cycles) == {"directory", "ntuple"} and
                  all(integer(v, 1, 32767) for v in cycles.values()), "availability cycles schema")
        A.require(saved.get("header_bytes") == 100 and type(saved.get("header_bytes")) is int and
                  saved.get("header_reopened") is True and pin(saved.get("header_sha256")), "availability header schema")
        scalars = saved.get("scalars")
        A.require(type(scalars) is dict and set(scalars) == set(A.KINDS) == set(COLUMNS), "availability scalar schema")
        for name, scalar in scalars.items():
            A.require(type(scalar) is dict and scalar.get("available") is True, "audited scalar must be available")
            meta, dig = scalar.get("metadata"), scalar.get("consumed_array")
            A.require(type(meta) is dict and type(dig) is dict, "audited metadata/digest required")
            A.require(meta.get("classname") in ("TBranch", "TBranchElement") and
                      type(meta.get("interpretation")) is str and len(meta["interpretation"]) <= 256,
                      "audited scalar metadata schema")
            A.require(type(dig.get("dtype")) is str and len(dig["dtype"]) <= 16, "audited dtype schema")
            dtype = np.dtype(dig["dtype"])
            kind = A.KINDS[name]
            A.require(dtype.fields is None and dtype.subdtype is None and dtype.itemsize <= 8 and
                      dtype.kind in ("iu" if kind in ("id", "energy") else "biu" if kind in ("count", "bool") else "f") and
                      dtype.str == dig["dtype"] == meta.get("numpy_dtype"), "audited dtype domain differs")
            A.require(pin(dig.get("sha256")) and dig.get("shape") == [record["entries"]] and
                      type(dig["shape"][0]) is int and dig.get("order") == "C" and
                      type(dig.get("bytes")) is int and dig["bytes"] == record["entries"] * dtype.itemsize,
                      "audited array digest schema")
            A.require(type(meta.get("entries")) is int and meta["entries"] == record["entries"] and
                      integer(meta.get("baskets"), 1, A.MAX_BASKETS) and
                      all(integer(meta.get(k), 0, A.BRANCH_BYTES) for k in
                          ("uncompressed_bytes", "compressed_bytes", "declared_uncompressed_bytes")),
                      "audited metadata count/byte cap")
        A.require(sum(s["metadata"]["baskets"] for s in scalars.values()) <= A.TOTAL_BASKETS and
                  sum(s["metadata"]["uncompressed_bytes"] for s in scalars.values()) <= A.PAYLOAD_CAP,
                  "audited total basket/payload cap")
    return old, by_path


def extract_file(record, saved, budget):
    executor = uproot.source.futures.TrivialExecutor()
    options = dict(object_cache=None, array_cache=None, decompression_executor=executor,
                   interpretation_executor=executor, use_threads=False, minimal_ttree_metadata=True)
    path, entries = record["path"], record["entries"]
    with A.XRootDBoundedFile(path, budget) as stream:
        header = A.sha(stream.read(100))
        A.require(header == saved["header_sha256"], "ROOT header differs from availability")
        stream.seek(0)
        with uproot.open(stream, **options) as root:
            _, tree, cycles = A.tree_header(root, record)
            A.require(cycles == saved["chosen_cycles"], "ROOT cycles differ from availability")
            names = tree.keys(recursive=True, full_paths=True)
            A.require(len(names) <= A.MAX_BRANCHES and len(names) == len(set(names)), "tree branch count/duplicate cap")
            A.require(all(n in names for n in COLUMNS), "required exact scalar missing; aliases forbidden")
            branches, metadata = {}, {}
            for name in COLUMNS:
                branches[name] = tree[name]
                metadata[name] = A.branch_metadata(branches[name], name, entries)
                A.require(metadata[name] == saved["scalars"][name]["metadata"], "ROOT branch metadata differs from availability")
            A.require(sum(m["baskets"] for m in metadata.values()) <= A.TOTAL_BASKETS, "total basket cap")
            A.require(sum(m["uncompressed_bytes"] for m in metadata.values()) <= A.PAYLOAD_CAP, "total scalar payload cap")
            arrays, digests = {}, {}
            for name, column in COLUMNS.items():
                array = branches[name].array(library="np", array_cache=None,
                    decompression_executor=executor, interpretation_executor=executor)
                A.require(isinstance(array, np.ndarray) and array.shape == (entries,) and
                          array.dtype.str == metadata[name]["numpy_dtype"], "consumed scalar shape/dtype differs")
                digests[name] = A.array_digest(array)
                A.require(digests[name] == saved["scalars"][name]["consumed_array"], "consumed array differs from availability")
                if name in A.IDS:
                    A.require(int(array.min()) >= 0 and int(array.max()) <= np.iinfo(np.int64).max, "invalid full ID domain")
                    array = array.astype(np.int64)
                arrays[column] = array
            ids = np.column_stack([arrays[n] for n in ("run", "subrun", "event")])
            A.require(len(np.unique(ids, axis=0)) == entries, "duplicate full IDs within file")
    before = budget.files[path]
    with A.XRootDBoundedFile(path, budget) as stream:
        A.require(A.sha(stream.read(100)) == header == saved["header_sha256"], "ROOT header changed on reopen")
        stream.seek(0)
        with uproot.open(stream, **options) as root:
            _, _, after_cycles = A.tree_header(root, record)
            A.require(after_cycles == cycles == saved["chosen_cycles"], "ROOT chosen cycles changed on reopen")
    arrays["source_index"] = np.full(entries, record["index"], dtype=np.int32)
    return arrays, dict(**record, chosen_cycles=cycles, header_sha256=header, header_bytes=100,
        header_reopened=True, resolved_url=A.xrootd_url(path), consumed_arrays=digests,
        version_inherited=saved.get("version"), version_reread=False,
        read_bytes_before_reopen=before, read_bytes=budget.files[path])


def run(selection_report, availability_report, out_dir, *, expected_selection_sha256, expected_availability_sha256):
    A.require(expected_selection_sha256 == SELECTION_SHA256, "mandatory hardbound selection SHA256 differs")
    A.require(expected_availability_sha256 == AVAILABILITY_SHA256, "mandatory hardbound availability SHA256 differs")
    selection_report, availability_report, out_dir = [pathname(A.local(pathname(p, absolute=False))) for p in
                                                   (selection_report, availability_report, out_dir)]
    source, audit_source = A.local(__file__), A.local(A.__file__)
    source_bytes, audit_bytes = A.small_bytes(source, SOURCE_CAP), A.small_bytes(audit_source, SOURCE_CAP)
    A.require(A.sha(audit_bytes) == AUDIT_SOURCE_SHA256, "imported audit source differs from frozen pin")
    selection_bytes = A.small_bytes(selection_report, JSON_CAP)
    availability_bytes = A.small_bytes(availability_report, JSON_CAP)
    A.require(A.sha(selection_bytes) == SELECTION_SHA256, "selection report hash differs from hard pin")
    A.require(A.sha(availability_bytes) == AVAILABILITY_SHA256, "availability report hash differs from hard pin")
    records, selected, ledger_parent = A.bindings(selection_bytes)
    pathname(selected["path"])
    if ledger_parent is not None:
        pathname(json.loads(selection_bytes)["exposure_ledger"]["path"], absolute=False)
    old, by_path = availability_bindings(availability_bytes, records)
    excluded = [source.parents[2], audit_source.parents[2], selection_report.parent, availability_report.parent,
                pathname(selected["path"]).parent, pathname(old["selection_report"]["path"]).parent,
                pathname(old["source"]["path"]).parent]
    if ledger_parent is not None:
        excluded.append(Path(os.path.abspath(selection_report.parent / pathname(ledger_parent, absolute=False))))
    A.outside(out_dir, excluded)
    out_dir.mkdir()
    budget, files, pieces = A.Budget(), [], {n: [] for n in ROW_KEYS}
    with A.metadata_limits():
        for record in records:
            arrays, file_report = extract_file(record, by_path[record["path"]], budget)
            files.append(file_report)
            for name in ROW_KEYS:
                A.require(not pieces[name] or pieces[name][0].dtype == arrays[name].dtype,
                          "cross-file scalar dtype differs; promotion forbidden")
                pieces[name].append(arrays[name])
    arrays = {name: np.concatenate(parts, dtype=parts[0].dtype) for name, parts in pieces.items()}
    total = len(arrays["run"])
    A.require(total == A.EXPECTED_ENTRIES and total <= A.MAX_TOTAL_ENTRIES, "observed total entries differ")
    ids = np.column_stack([arrays[n] for n in ("run", "subrun", "event")])
    A.require(len(np.unique(ids, axis=0)) == total, "duplicate full IDs across files")
    A.require(sum(a.nbytes for a in arrays.values()) <= NPZ_CAP, "NPZ scalar byte cap")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    npz_bytes = buffer.getvalue()
    A.require(len(npz_bytes) <= NPZ_CAP, "NPZ output byte cap")
    npz_path, report_path = out_dir / "rows.npz", out_dir / "report.json"
    result = dict(schema=SCHEMA, rows=dict(path=str(npz_path), sha256=A.sha(npz_bytes), entries=total,
        columns={n: dict(source_leaf=next((k for k, v in COLUMNS.items() if v == n), None),
                         **A.array_digest(a)) for n, a in arrays.items()}),
        selection_report=dict(path=str(selection_report), sha256=SELECTION_SHA256),
        availability_report=dict(path=str(availability_report), sha256=AVAILABILITY_SHA256),
        audit_source=dict(path=str(audit_source), sha256=AUDIT_SOURCE_SHA256),
        source=dict(path=str(source), sha256=A.sha(source_bytes), snapshot=source_bytes.decode("utf-8")),
        total_entries=total, full_ids_unique=True, files=files, read_bytes=budget.total,
        per_file_read_bytes=budget.files, transport="xrootd",
        transport_details=dict(api="XRootD.client.File", per_call_timeout_seconds={k: A.XROOTD_TIMEOUT for k in
            ("open", "read", "close")}, application_retries=0, automatic_fallback=False, short_read_policy="raise"),
        software=dict(numpy=np.__version__, uproot=uproot.__version__, xrootd=A.xrootd_version()),
        bounds=dict(report_json=JSON_CAP, source=SOURCE_CAP, output=REPORT_CAP, npz=NPZ_CAP, path=PATH_CAP,
            file_read=A.FILE_CAP, total_read=A.TOTAL_CAP, single_read=A.READ_CAP, metadata_object=A.METADATA_CAP,
            keys=A.MAX_KEYS, branches=A.MAX_BRANCHES, baskets_per_branch=A.MAX_BASKETS, baskets_total=A.TOTAL_BASKETS,
            branch_payload=A.BRANCH_BYTES, payload_per_file=A.PAYLOAD_CAP, entries_per_file=A.MAX_ENTRIES,
            entries_total=A.MAX_TOTAL_ENTRIES),
        selected_track_provenance=dict(**selected, status="inherited only; not opened/stat'ed/hashed"),
        column_convention="Original scalar dtypes; IDs signed int64 only after old digest verification; source_index is basename index int32, not file ordinal. Rows ordered by source_index then original entry.",
        interpretation="Raw measured proxies only; no units, calibration, intensity law or selection dependence inferred.",
        proof_limits=["Array C-byte/dtype/shape equality with the pinned audit is not a full-file hash.",
            "Reopens check header/UUID/entries/cycles only, not baskets after consumption.",
            "Per-operation timeouts are not a whole-run deadline; byte caps are not an RSS bound.",
            "No selected tracks, old exposure ledger, MC, CRV, coincidences or version payloads read."],
        availability_reexecuted=False, prior_audit_executed=False, selection_executed=False, mc_read=False,
        full_file_hashes=False, F=None, weights=None, weight_model=None, calibration=None, finite_library_uncertainty=None,
        publication="Exclusive directory and files; report.json written last marks completion; failures may leave partial directory; no atomic/durability claim.")
    output = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    A.require(len(output) <= REPORT_CAP, "report output byte cap")

    def recheck():
        for path, cap, original, label in ((selection_report, JSON_CAP, selection_bytes, "selection JSON"),
                (availability_report, JSON_CAP, availability_bytes, "availability JSON"),
                (audit_source, SOURCE_CAP, audit_bytes, "audit source"), (source, SOURCE_CAP, source_bytes, "rows source")):
            A.require(A.small_bytes(path, cap) == original, label + " changed")
        A.no_symlinks(out_dir)

    recheck()
    with npz_path.open("xb") as stream:
        A.require(stream.write(npz_bytes) == len(npz_bytes), "short NPZ write")
    A.require(A.small_bytes(npz_path, NPZ_CAP) == npz_bytes, "persisted NPZ differs")
    recheck()
    stream = report_path.open("xb")
    try:
        with stream:
            A.require(stream.write(output) == len(output), "short report write")
    except BaseException:
        report_path.unlink()
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_report")
    parser.add_argument("availability_report")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-availability-sha256", required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("schema", "total_entries", "read_bytes")}, separators=(",", ":")))


if __name__ == "__main__":
    main()
