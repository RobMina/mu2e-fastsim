"""Persistent finite-pool throughput harness; preparation, response and output remain timed."""
import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np
import uproot

from .counts import json_values
from .fast_detectors import save_sampling
from .inspect import summarize_fast


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def factory(spec):
    module, separator, symbol = spec.partition(":")
    if not separator or not module or not symbol:
        raise ValueError("backend must be module:factory")
    return getattr(importlib.import_module(module), symbol)


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(json_values(value), indent=2, allow_nan=False) + "\n")


def verify_batch(directory, ordinal_start, events, extra_tables=(), output_format="root"):
    """Count closed highest-cycle output entries, never log counters or expected input counts."""
    receipt = json.loads((directory / "sampling.json").read_text())
    with np.load(directory / "sampling.npz", allow_pickle=False) as arrays:
        expected_ids = arrays[receipt["signal_event_ids"]["npz_key"]]
    if output_format == "occupancy":
        from .device_occupancy import census
        return census(directory / "occupancy.npz", ordinal_start, events, expected_ids)
    census = summarize_fast(directory / "fast.root")
    if census["events"] != events or "tracker_digis" not in census:
        raise ValueError("finished combined detector census differs from batch")
    with uproot.open(directory / "fast.root") as root:
        identity = root["benchmark/events"].arrays(library="np")
        actual = root["crvfast/events"].arrays(["run", "subrun", "event"], library="np")
        if not np.array_equal(identity["realization"], np.arange(ordinal_start, ordinal_start + events, dtype=np.uint64)):
            raise ValueError("persisted realization identities differ")
        output_ids = np.column_stack([actual[field] for field in ("run", "subrun", "event")])
        if not np.array_equal(output_ids, expected_ids):
            raise ValueError("finished event identities differ from sampled source")
        for field in ("run", "subrun", "event"):
            if not np.array_equal(identity[field], actual[field]):
                raise ValueError("persisted original event identities differ")
        for table in extra_tables:
            rows = root[table].arrays(["run", "subrun", "event"], library="np")
            if any(not np.array_equal(rows[key], actual[key]) for key in actual):
                raise ValueError(f"extra stage event census differs: {table}")
    return census


def run(args, source_factory=None, backend_factory=None):
    """One resident source and backend; loop over batches, not individual events."""
    backend_factory = backend_factory or factory(args.backend)
    if source_factory is None:
        from .stream_source import StreamSource
        source_factory = getattr(backend_factory, "source_factory", StreamSource)
    if min(args.events, args.batch_events, args.max_selected_rows, args.chunk) < 1 or args.warmup_batches < 0:
        raise ValueError("positive workload/batch/row/chunk and nonnegative warmup required")
    if min(args.seed, args.tracker_seed, args.sampling_seed) < 0:
        raise ValueError("seeds must be nonnegative")
    if args.events + args.batch_events * args.warmup_batches > np.iinfo(np.int64).max:
        raise ValueError("realization count exceeds int64")
    args.out.mkdir(parents=True, exist_ok=False)
    job_start = time.perf_counter()
    source = source_factory(args.frames_manifest, seed=args.sampling_seed, max_selected_rows=args.max_selected_rows)
    if args.batch_events > source.population:
        raise ValueError("batch-events cannot exceed unique signal pool population")
    backend = backend_factory(args, source)
    metadata = backend.metadata
    for field in ("name", "support", "residency"):
        if field not in metadata:
            raise ValueError(f"backend metadata missing {field}")
    synthetic = getattr(args, "synthetic", False)
    if synthetic != (metadata.get("model_kind") == "synthetic"):
        raise ValueError("synthetic flag and backend model kind must agree")
    required = {*([] if synthetic else ["tracker_digis", "crv_reco"]), *args.require_stage}
    if not required <= set(metadata["support"]):
        raise ValueError("backend does not support requested stages")
    if getattr(args, "device_output", "root") == "occupancy" and metadata.get("output_format") != "occupancy":
        raise ValueError("occupancy output is implemented only by the device backend")
    extra_tables = metadata.get("output_event_tables", [])
    if "track_reco" in required and not extra_tables:
        raise ValueError("track_reco backend must declare persisted event census tables")
    write_json(args.out / "configuration.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source": source.metadata, "backend": metadata,
        "sampling": "finite source pool replay; repeated primaries are not independent new physics events",
        "rng": "batch-domain RNG; warmup advances source state; batch size/warmup are part of reproducibility",
    })
    backend.sync()
    setup_s = time.perf_counter() - job_start
    with (args.out / "batches.jsonl").open("x") as ledger:
        def process(phase, size):
            started = time.perf_counter()
            batch = source.next_batch(size)
            prepare_s = time.perf_counter() - started
            directory = args.out / f"{phase}_{batch.batch_index:08d}"
            directory.mkdir()
            if metadata.get("sampling_receipt_owner") != "backend":
                save_sampling(directory, batch.provenance)
            metrics = backend.run(batch, directory)
            backend.sync()
            if metrics["events"] != size:
                raise ValueError("backend returned wrong event count")
            row = {"phase": phase, "directory": directory.name, "batch_index": batch.batch_index,
                   "ordinal_start": batch.ordinal_start, "events": size, "prepare_s": prepare_s,
                   "batch_s": time.perf_counter() - started, "backend": metrics}
            ledger.write(json.dumps(json_values(row), allow_nan=False) + "\n")
            ledger.flush()
            return row
        pipelined = getattr(args, "device_pipeline", False)
        if pipelined and not hasattr(backend, "dispatch"):
            raise ValueError("--device-pipeline needs a backend with dispatch/collect (the device backend)")

        def dispatch(phase, size):
            started = time.perf_counter()
            batch = source.next_batch(size)
            prepare_s = time.perf_counter() - started
            directory = args.out / f"{phase}_{batch.batch_index:08d}"
            directory.mkdir()
            if metadata.get("sampling_receipt_owner") != "backend":
                save_sampling(directory, batch.provenance)
            return phase, size, batch, directory, prepare_s, started, backend.dispatch(batch)

        def collect(item):
            phase, size, batch, directory, prepare_s, started, handle = item
            metrics = backend.collect(handle, directory)
            if metrics["events"] != size:
                raise ValueError("backend returned wrong event count")
            row = {"phase": phase, "directory": directory.name, "batch_index": batch.batch_index,
                   "ordinal_start": batch.ordinal_start, "events": size, "prepare_s": prepare_s,
                   "batch_s": time.perf_counter() - started, "backend": metrics}
            ledger.write(json.dumps(json_values(row), allow_nan=False) + "\n")
            ledger.flush()

        def drive(phase, sizes):
            """--device-pipeline: batch k is collected on one worker thread while batch k + 1 is dispatched (a GPU
            execution holds its calling thread); same batches, outputs and ledger order."""
            from concurrent.futures import ThreadPoolExecutor
            pending = None
            with ThreadPoolExecutor(max_workers=1) as writer:
                for size in sizes:
                    item = dispatch(phase, size)
                    if pending is not None:
                        pending.result()
                    pending = writer.submit(collect, item)
                if pending is not None:
                    pending.result()

        sizes = [min(args.batch_events, args.events - start) for start in range(0, args.events, args.batch_events)]
        warm_start = time.perf_counter()
        if pipelined:
            drive("warmup", [args.batch_events] * args.warmup_batches)
        else:
            for _ in range(args.warmup_batches):
                process("warmup", args.batch_events)
        backend.sync()
        warmup_s = time.perf_counter() - warm_start
        begin = time.perf_counter()
        if pipelined:
            drive("measure", sizes)
        else:
            remaining = args.events
            while remaining:
                size = min(remaining, args.batch_events)
                process("measure", size)
                remaining -= size
        backend.sync()
    measured_s = time.perf_counter() - begin
    pipeline_s = time.perf_counter() - job_start
    verified, warm_events, measured_batches, next_ordinal = 0, 0, 0, 0
    check_start = time.perf_counter()
    with (args.out / "batches.jsonl").open() as ledger:
        for line in ledger:
            row = json.loads(line)
            if row["ordinal_start"] != next_ordinal:
                raise ValueError("batch realization sequence is not contiguous")
            census = verify_batch(args.out / row["directory"], row["ordinal_start"], row["events"], extra_tables,
                                  metadata.get("output_format", "root"))
            next_ordinal += census["events"]
            if row["phase"] == "measure":
                verified += census["events"]
                measured_batches += 1
            else:
                warm_events += census["events"]
    if verified != args.events or warm_events != args.warmup_batches * args.batch_events:
        raise ValueError("finished output population differs from requested workload")
    result = {"events": verified, "warmup_events": warm_events, "measured_batches": measured_batches,
              "setup_s": setup_s, "warmup_s": warmup_s, "measured_s": measured_s,
              "events_per_s": verified / measured_s, "pipeline_s": pipeline_s,
              "validation_s": time.perf_counter() - check_start, "backend": metadata,
              "boundary": ("persistent source selection/overlay, response, device occupancy reduction, synchronization, "
                           "sampling receipt and closed occupancy NPZ writes, ledger writes; excludes setup/warmup and post-run readback"
                           if metadata.get("output_format") == "occupancy" else
                           "persistent source selection/overlay, response, synchronization, sampling and closed ROOT writes, ledger writes; excludes setup/warmup and post-run readback"),
              "scope": metadata.get("scope", "finite-pool workload replay, not independent event generation"),
              "whole_job": "use mixing.process_timer externally; includes imports, setup, warmup, all writing, readback and shutdown",
              "baseline": None, "speedup": None}
    write_json(args.out / "stream_report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frames-manifest", "truth", "stock", "photon-csv", "expanded", "tracker-conditions", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--events", type=positive, required=True)
    parser.add_argument("--batch-events", type=positive, required=True)
    parser.add_argument("--max-selected-rows", type=positive, required=True)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--chunk", type=positive, default=32)
    parser.add_argument("--sampling-seed", type=int, default=20261106)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--tracker-seed", type=int, default=20261105)
    parser.add_argument("--backend", default="mixing.stream_backend:Backend")
    parser.add_argument("--require-stage", action="append", default=[])
    parser.add_argument("--device-kernel", help="pure JAX response factory module:symbol; no default physical model")
    parser.add_argument("--device-capacities", type=Path, help="JSON max_donors and per-collection rows")
    parser.add_argument("--device-platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--synthetic", action="store_true", help="explicit integration-only model opt-in")
    parser.add_argument("--device-output", choices=("root", "occupancy"), default="root",
                        help="device backend only: full detector tables to ROOT (default) or on-device occupancy NPZ")
    parser.add_argument("--occupancy-geometry", type=Path, help="bar_to_sector JSON (as counts --geometry) for occupancy output")
    parser.add_argument("--device-pipeline", action="store_true",
                        help="device backend only: dispatch the next batch before writing the previous one")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({key: result[key] for key in ("events", "measured_s", "events_per_s")}))


if __name__ == "__main__":
    main()
