"""Certify completed named-reference dumps without transport, generation, or pnfs access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import uproot


CONFIGS = ("run1a", "run2")
RUN = 1400
KEY = ["run", "subrun", "event"]
IDENTITY = KEY + ["idx"]
VALUES = ["pdg", "px", "py", "pz", "x", "y", "z", "t", "e"]
SUMMARY = re.compile(
    r"TrigReport[ \t]+Events[ \t]+total[ \t]*=[ \t]*(\d+)"
    r"[ \t]+passed[ \t]*=[ \t]*(\d+)[ \t]+failed[ \t]*=[ \t]*(\d+)"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _local_path(path):
    # Reject dCache before any filesystem lookup, and do not follow symlinks into it.
    from .. import dcache   # lazily: the module itself imports no site configuration
    path = Path("/" + os.path.abspath(os.fspath(path)).lstrip("/"))
    _require(Path(dcache.mount()) not in (path, *path.parents), "pnfs paths are forbidden; use local dumps")
    for component in reversed((path, *path.parents)):
        _require(not component.is_symlink(), f"symlink paths are not allowed: {component}")
    return path


def fingerprint(path):
    path = _local_path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def _read_log(path, expected, worker=False):
    path = _local_path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    summaries = list(re.finditer(r"(?m)^[^\r\n]*TrigReport[ \t]+Events[^\r\n]*$", text))
    _require(len(summaries) == 1, f"{path}: require exactly one final TrigReport Events summary")
    summary = SUMMARY.fullmatch(summaries[0].group().strip())
    _require(summary is not None, f"{path}: malformed TrigReport Events summary")
    total, passed, failed = map(int, summary.groups())
    _require(total == expected, f"{path}: total {total} != expected {expected}")
    _require(passed > 0 and total == passed + failed, f"{path}: invalid passed/failed counts")
    marker = "Exit status:" if worker else "dump_exit="
    exits = list(re.finditer(r"(?m)^[^\r\n]*" + re.escape(marker) + r"[^\r\n]*$", text))
    _require(len(exits) == 1, f"{path}: require exactly one {marker} record")
    exit_pattern = r"Exit status:[ \t]*0" if worker else r"dump_exit=0"
    _require(re.fullmatch(exit_pattern, exits[0].group().strip()) is not None,
             f"{path}: missing or nonzero exit status")
    _require(exits[0].start() > summaries[0].end(), f"{path}: exit precedes final summary")
    if not worker:
        _require(not text[exits[0].end():].strip(), f"{path}: dump_exit=0 must end the dump log")
        _require(passed == total and failed == 0, f"{path}: dumper did not pass every input event")
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "summary": {"total": total, "passed": passed, "failed": failed}, "exit_status": 0}


def _read_root(path, names, subrun):
    artifact = fingerprint(path)
    frames = {}
    artifact["trees"] = {}
    with uproot.open(artifact["path"]) as root:
        _require("dump" in root, f"{path}: missing dump directory")
        directory = root["dump"]
        keys = directory.keys(recursive=False, cycle=True)
        for name in names:
            cycles = sorted(int(key.rsplit(";", 1)[1]) for key in keys if key.startswith(name + ";"))
            _require(bool(cycles), f"{path}: missing dump/{name}")
            tree = directory[f"{name};{cycles[-1]}"]
            _require(tree.classname == "TTree", f"{path}: dump/{name} must be a TTree")
            columns = IDENTITY + VALUES if name == "gen" else KEY
            _require(set(columns).issubset(tree.keys()), f"{path}: dump/{name} missing required columns")
            if name == "gen" and "genid" in tree.keys():
                columns = columns + ["genid"]
            arrays = tree.arrays(columns, library="np", how=dict)
            for column in columns:
                data = arrays[column]
                _require(data.ndim == 1 and data.dtype.kind in "iuf" and np.isfinite(data).all(),
                         f"{path}: dump/{name}/{column} must be finite scalar numeric data")
                if column in IDENTITY or column in ("pdg", "genid"):
                    _require(data.dtype.kind in "iu", f"{path}: dump/{name}/{column} must be integer")
            frame = pd.DataFrame(arrays)
            _require(len(frame) == tree.num_entries, f"{path}: dump/{name} row count mismatch")
            _require(frame.run.eq(RUN).all() and frame.subrun.eq(subrun).all(),
                     f"{path}: dump/{name} wrong run/subrun; expected {RUN}/{subrun}")
            if name == "gen":
                _require(not frame.duplicated(IDENTITY).any(), f"{path}: duplicate generator identity")
                _require(frame.idx.ge(0).all(), f"{path}: negative generator idx")
            frames[name] = frame
            artifact["trees"][name] = {"cycles": cycles, "selected_cycle": cycles[-1],
                                        "rows": len(frame), "unique_events": len(frame[KEY].drop_duplicates())}
    return frames, artifact


def _certify_shard(dumpdir, config, subrun, expected_events):
    base = dumpdir / config
    suffix = f"{RUN:06d}_{subrun:08d}"
    worker = _read_log(base / f"worker.{suffix}.log", expected_events, worker=True)
    accepted_count = worker["summary"]["passed"]
    gen_log = _read_log(base / f"dump_gen.{suffix}.log", expected_events)
    steps_log = _read_log(base / f"dump_crvsteps.{suffix}.log", accepted_count)
    source, gen_artifact = _read_root(base / f"gen.{suffix}.root", ("gen",), subrun)
    reference, steps_artifact = _read_root(base / f"crvsteps.{suffix}.root", ("gen", "particles", "steps"), subrun)
    original, accepted = source["gen"], reference["gen"]
    generated_ids = np.unique(original.event.to_numpy())
    accepted_ids = np.unique(accepted.event.to_numpy())
    particle_ids = np.unique(reference["particles"].event.to_numpy())
    step_ids = np.unique(reference["steps"].event.to_numpy())
    label = f"{config} subrun {subrun}"
    _require(len(generated_ids) == expected_events, f"{label}: generated event count mismatch")
    _require(np.array_equal(generated_ids, np.arange(1, expected_events + 1)),
             f"{label}: generated event IDs must be exactly 1..{expected_events}")
    _require(len(accepted_ids) == accepted_count and len(particle_ids) == accepted_count,
             f"{label}: accepted event count does not match worker passed")
    _require(np.array_equal(accepted_ids, particle_ids), f"{label}: accepted gen/particles event IDs differ")
    _require(np.isin(accepted_ids, generated_ids).all(), f"{label}: accepted event outside generator universe")
    _require(len(step_ids) > 0, f"{label}: steps must be nonempty")
    _require(np.isin(step_ids, accepted_ids).all(), f"{label}: steps event IDs outside accepted events")
    # Compare every primary of each accepted event, not just rows surviving an inner join.
    expected = original.loc[original.event.isin(accepted_ids)].set_index(IDENTITY).sort_index()
    actual = accepted.set_index(IDENTITY).sort_index()
    _require(expected.index.equals(actual.index), f"{label}: accepted generator identities differ from source")
    _require(set(expected.columns) == set(actual.columns), f"{label}: generator value columns differ")
    for column in expected.columns:
        _require(np.array_equal(expected[column].to_numpy(), actual[column].to_numpy()),
                 f"{label}: accepted generator value mismatch: {column}")
    return {"config": config, "run": RUN, "subrun": subrun,
            "counts": {"generated_events": len(generated_ids), "accepted_events": len(accepted_ids),
                       "particle_events": len(particle_ids), "step_events": len(step_ids),
                       "worker_passed": accepted_count, "worker_failed": worker["summary"]["failed"]},
            "artifacts": {"gen": gen_artifact, "crvsteps": steps_artifact, "worker_log": worker,
                          "dump_gen_log": gen_log, "dump_crvsteps_log": steps_log}}


def certify(dumpdir, first, last, expected_events, output):
    """Write a new report only after both configurations and every requested shard pass."""
    _require(first > 0 and last >= first, "require 0 < first <= last")
    _require(expected_events > 0, "expected-events must be positive")
    dumpdir, output = _local_path(dumpdir), _local_path(output)
    _require(not output.exists(), f"refusing to overwrite existing output: {output}")
    shards = []
    for config in CONFIGS:
        for subrun in range(first, last + 1):
            shards.append(_certify_shard(dumpdir, config, subrun, expected_events))
    # Recheck all file evidence after the full campaign; a growing log is not a completion certificate.
    for shard in shards:
        for artifact in shard["artifacts"].values():
            current = fingerprint(artifact["path"])
            _require(all(current[key] == artifact[key] for key in ("bytes", "sha256")),
                     f"{artifact['path']}: artifact changed during certification")
    report = {"schema_version": 1, "status": "PASS", "dumpdir": str(dumpdir), "run": RUN,
              "configs": list(CONFIGS), "first": first, "last": last, "expected_events": expected_events,
              "scope": "Local reference completeness and truth identity only; not transport or physics validation.",
              "shards": shards}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Exclusive creation also prevents a concurrent certifier from overwriting a report.
    stream = output.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(payload)
    except BaseException:
        output.unlink()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dumpdir", type=Path)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = certify(args.dumpdir, args.first, args.last, args.expected_events, args.output)
    except Exception as error:
        print(f"Reference certification failed: {error}", file=sys.stderr)
        return 1
    print(f"PASS: certified {len(report['shards'])} reference shards; report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
