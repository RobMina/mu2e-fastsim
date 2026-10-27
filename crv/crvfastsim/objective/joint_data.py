"""Bounded, passive measured-data capture from existing EventNtuples (no simulation).

    python -m crvfastsim.objective.joint_data --inputs FILE.root ... --out NEWDIR

Each read chunk gets a raw NPZ, so JointCapture never retains an entire input file.
Full integer event keys (including empty events) are mandatory. Only the compact
24-byte keys are retained across chunks for the exact global duplicate audit.
--reference-accumulator accepts one saved full-file JSON per input, in input order;
a partial capture records the references but cannot claim full-file equality.

Data times are UNCHANGED: crvresponse.accumulate_file defaults to time_corr=None;
timecal's mc_channel_shift is applied to MC in reco/r3.py, not to stock P4 data.
This command deliberately offers no time-shift or calibration-fitting option.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import awkward as ak
import numpy as np
import uproot

from . import bars, crvline as CL, crvresponse as R, joint, timecal

SCORING_FIELDS = ("events", "files", "exposure_ns", "lines_z", "lines_x", "lines_xz", "h", "oot_n")
KEY_ENCODING = "ordered run,subrun,event rows; little-endian signed int64; C order"


def digest(path):
    """Stream actual bytes, without stat/resolve (important for known /pnfs paths)."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discover_event_branches(tree):
    """Accept only complete, unambiguous real identity schemas, never entry IDs.

    evtinfo/run is the split EventNtuple spelling used by existing readers. Dot
    and bare spellings accommodate flattened exports; unrelated suffix matches
    (e.g. MC truth event numbers) are intentionally not guessed.
    """
    names = set(tree.keys(recursive=True))
    candidates = []
    for prefix in ("evtinfo/", "evtinfo.", "evtinfo/evtinfo.", ""):
        fields = tuple(prefix + field for field in ("run", "subrun", "event"))
        if all(field in names for field in fields):
            candidates.append(fields)
    if len(candidates) != 1:
        raise ValueError("need one unambiguous full run/subrun/event branch schema; "
                         f"found {candidates!r}")
    return dict(zip(("run", "subrun", "event"), candidates[0]))


def read_event_keys(tree, branches, start, stop):
    """Read scalar, nonmissing, nonnegative int64-compatible keys without float conversion."""
    columns = []
    # RNTuple literal slash fields need whole-record access (TTree supports branches).
    record = _arrays(tree, list(branches.values()), start, stop) if tree.classname == "ROOT::RNTuple" else None
    for name in ("run", "subrun", "event"):
        branch = branches[name]
        a = ak.to_numpy(record[branch] if record is not None else
                        tree[branch].array(entry_start=start, entry_stop=stop, library="ak"))
        if (np.ma.isMaskedArray(a) and np.any(np.ma.getmaskarray(a))) or a.ndim != 1 or len(a) != stop - start:
            raise ValueError(f"event identity {name} must be a nonmissing scalar per event")
        if a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > np.iinfo(np.int64).max):
            raise ValueError(f"event identity {name} must be a nonnegative int64-compatible integer")
        columns.append(np.asarray(a, dtype="<i8"))
    return np.ascontiguousarray(np.column_stack(columns), dtype="<i8")


def _arrays(tree, branches, start, stop):
    if tree.classname == "ROOT::RNTuple":
        # RNTuple interprets expressions as glob filters; literal [4] is not a glob.
        return tree.arrays(filter_name=lambda name: name in branches, entry_start=start, entry_stop=stop)
    return tree.arrays(branches, entry_start=start, entry_stop=stop)


def _unique_keys(keys):
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("duplicate full run/subrun/event identity in requested capture")


def compare_scoring_fields(actual, reference):
    """Exact equality of all exposure and scoring fields, not supplemental dtp/ftp."""
    for field in SCORING_FIELDS:
        if field not in reference or actual[field] != reference[field]:
            raise ValueError(f"reference accumulator differs in scoring field {field}")


def _json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _highest_tree(root):
    # Uproot's unqualified directory lookup selects its highest cycle as well.
    directory = root["EventNtuple"]
    cycles = [int(key.rsplit(";", 1)[1]) for key in directory.keys(cycle=True, recursive=False)
              if key.rsplit(";", 1)[0] == "ntuple"]
    if not cycles:
        raise ValueError("missing EventNtuple/ntuple tree")
    cycle = max(cycles)
    return directory[f"ntuple;{cycle}"], cycle


def capture_data(inputs, out, *, chunk=25000, entry_stop=None, window_ns=400000., reference_accumulator=None):
    """Create a NEW directory of raw NPZ shards, response state and provenance.

    Returns the JSON-compatible summary. Entry stop is per input file. References
    must be per-file states in matching order; exact comparisons are mandatory
    when entry_stop is None. An error may leave an incomplete directory, but no
    summary.json completion receipt is written until all audits succeed.
    ROOT files are streamed twice for SHA256 mutation checks in addition to the
    chunked branch reads; this intentionally costs I/O even for bounded captures.
    """
    inputs = [os.path.abspath(os.fspath(path)) for path in inputs]
    refs = [] if reference_accumulator is None else reference_accumulator
    if isinstance(refs, (str, os.PathLike)):
        refs = [refs]
    refs = [os.path.abspath(os.fspath(path)) for path in refs]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("inputs must be a nonempty list of distinct paths")
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk <= 0:
        raise ValueError("chunk must be a positive integer")
    if entry_stop is not None and (not isinstance(entry_stop, int) or isinstance(entry_stop, bool) or entry_stop < 0):
        raise ValueError("entry_stop must be a nonnegative integer")
    if not np.isfinite(window_ns) or window_ns <= 0:
        raise ValueError("window_ns must be finite and positive")
    if refs and len(refs) != len(inputs):
        raise ValueError("supply one reference accumulator per input, in input order")
    code = {module.__name__: {"path": os.path.abspath(module.__file__), "sha256": digest(module.__file__)}
            for module in (R, CL, bars, joint, timecal)}
    code[__name__] = {"path": os.path.abspath(__file__), "sha256": digest(__file__)}
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    total = R.new_state()
    all_keys, sources, shards = [], [], []
    key_hash = hashlib.sha256()
    for source_index, path in enumerate(inputs):
        started = time.monotonic()
        before = digest(path)
        reference = None
        if refs:
            ref_path = refs[source_index]
            ref_sha = digest(ref_path)
            with open(ref_path) as stream:
                reference = json.load(stream)
            if digest(ref_path) != ref_sha:
                raise ValueError("reference accumulator changed while reading")
        state = R.new_state()
        file_keys = []
        file_key_hash = hashlib.sha256()
        ranges = []
        # Passing the already-open stream avoids path discovery/stat on /pnfs.
        with open(path, "rb") as stream, uproot.open(stream) as root:
            tree, cycle = _highest_tree(root)
            branches = discover_event_branches(tree)
            available = int(tree.num_entries)
            stop = available if entry_stop is None else min(entry_stop, available)
            # One typed zero-entry chunk also preserves the empty-file schema.
            starts = range(0, stop, chunk) if stop else (0,)
            for start in starts:
                end = min(start + chunk, stop)
                keys = read_event_keys(tree, branches, start, end)
                _unique_keys(keys)
                c = _arrays(tree, CL.COINC_BR, start, end)
                p = _arrays(tree, CL.PULSE_BR, start, end)
                state.setdefault("window_ns", []).append(float(window_ns))
                measured = R.accumulate_arrays(c, p, state, window_ns)
                collector = joint.JointCapture()
                collector.add(c, p, measured, keys, source_index=source_index)
                name = f"raw_{source_index:04d}_{start:09d}_{end:09d}.npz"
                report = collector.write(destination / name)
                if report["events"] != end - start:
                    raise ValueError("capture event count differs from requested entries")
                shards.append({"path": name, "sha256": digest(destination / name),
                               "source_index": source_index, "entry_start": start,
                               "entry_stop": end, "summary": report})
                ranges.append([start, end])
                file_keys.append(keys)
                file_key_hash.update(keys.tobytes())
                key_hash.update(keys.tobytes())
        state["files"] = 1
        keys = np.concatenate(file_keys, axis=0)
        _unique_keys(keys)
        all_keys.append(keys)
        if state["events"] != stop or state["exposure_ns"] != stop * float(window_ns):
            raise ValueError("unfinished response event/exposure counts")
        comparison = {"status": "not_requested"}
        if refs:
            comparison = {"path": refs[source_index], "sha256": ref_sha,
                          "status": "skipped_partial_capture" if entry_stop is not None else "exact",
                          "ignored_supplemental_fields": ["dtp", "ftp"]}
            if entry_stop is None:
                compare_scoring_fields(state, reference)
                comparison["matched_scoring_fields"] = list(SCORING_FIELDS)
            if digest(refs[source_index]) != ref_sha:
                raise ValueError("reference accumulator changed during capture")
        if digest(path) != before:
            raise ValueError("ROOT input changed during capture")
        sources.append({"source_index": source_index, "path": path, "sha256": before,
                        "hash_scope": "entire ROOT file, before and after capture", "tree": R.TREE,
                        "tree_cycle": cycle, "tree_key": f"{R.TREE};{cycle}",
                        "event_branches": branches, "entries_available": available,
                        "entry_ranges": ranges, "entries_expected": stop,
                        "entries_finished": state["events"], "complete": True,
                        "full_file": stop == available, "event_key_count": len(keys),
                        "event_key_sha256": file_key_hash.hexdigest(), "duplicate_event_keys": 0,
                        "counts": {field: state[field] for field in SCORING_FIELDS if field not in ("h", "oot_n")},
                        "reference_comparison": comparison, "elapsed_seconds": time.monotonic() - started})
        total = R.merge_states([total, state])
    keys = np.concatenate(all_keys, axis=0)
    _unique_keys(keys)  # Across sources as well as within each source/chunk.
    for record in code.values():
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("source code changed during capture")
    counts = {field: total[field] for field in SCORING_FIELDS if field not in ("h", "oot_n")}
    provenance = {"schema_version": 1, "complete": True, "sample_role": "measured_data",
                  "inputs": sources, "source_code": code,
                  "software": {"numpy": np.__version__, "awkward": ak.__version__, "uproot": uproot.__version__},
                  "options": {"chunk": chunk, "entry_stop": entry_stop, "window_ns": float(window_ns)},
                  "time_shift_ns": None, "time_policy": "unshifted stock data; no recalibration",
                  "event_key_encoding": KEY_ENCODING, "event_key_sha256": key_hash.hexdigest(),
                  "event_key_count": len(keys), "duplicate_event_keys": 0,
                  "duplicate_guard": "exact full integer triplets across all captured inputs",
                  "identity_audit_scope": "captured entry ranges only, including empty events",
                  "counts": counts, "full_files": all(s["full_file"] for s in sources)}
    _json(destination / "response_state.json", total)
    provenance["response_state"] = {"path": "response_state.json", "sha256": digest(destination / "response_state.json")}
    _json(destination / "provenance.json", provenance)
    summary = {"schema_version": 1, "complete": True, "sample_role": "measured_data", "counts": counts,
               "event_key_count": len(keys), "event_key_sha256": key_hash.hexdigest(),
               "duplicate_event_keys": 0, "raw_shards": shards,
               "provenance": {"path": "provenance.json", "sha256": digest(destination / "provenance.json")}}
    _json(destination / "summary.json", summary)  # Completion receipt, written last.
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk", type=int, default=25000)
    parser.add_argument("--entry-stop", type=int)
    parser.add_argument("--window-ns", type=float, default=400000.)
    parser.add_argument("--reference-accumulator", nargs="+", help="one full-file JSON per input, in input order")
    args = parser.parse_args(argv)
    report = capture_data(args.inputs, args.out, chunk=args.chunk, entry_stop=args.entry_stop,
                          window_ns=args.window_ns, reference_accumulator=args.reference_accumulator)
    print(json.dumps({"complete": report["complete"], "counts": report["counts"],
                      "raw_shards": len(report["raw_shards"])}, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
