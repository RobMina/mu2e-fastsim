"""Entry points for the shared warm-rate timer (`mu2efastsim.bench`) on the resident device backend.

    MIXING_BENCH_CONFIG=config.json python3 -m mu2efastsim.bench mixing.device_bench:run_pass \\
        --events N --prepare mixing.device_bench:prepare --output-included --workflow mixing \\
        --boundary "..." --out bench.json

`config.json` holds the `stream_benchmark` arguments by their long names (frames_manifest, expanded,
truth, stock, photon_csv, tracker_conditions, batch_events, max_selected_rows, device_capacities,
occupancy_geometry, out, and optionally seeds, device_kernel, device_output, device_platform, device_pipeline,
device_pipeline_writers).

Output is pipelined by default (`device_pipeline`, as `stream_benchmark --device-pipeline`): batch k is downloaded and
written on one worker thread while batch k + 1 runs on the device. The outputs are those of the serial loop;
`"device_pipeline": false` measures the serial loop. `"device_pipeline_writers": W` (default 1) collects up to W
batches at once, each on its own thread: the same files.

- `prepare(N)` reads and validates the five frame pools into host memory once (timer `prepare_s`).
- `prepare(N, seed=S)` (the timer's `--fresh-inputs`, called before every pass) reads the pools on its first call
  and shares them, and the resident backend, with later calls in the process. Each call gets its own output
  directory (`out/seed_<S>`) and its own sampling, response and tracker seeds (the configured ones plus S), which
  replace the backend's seeds on the device before the pass (same shapes, so no new compilation by themselves).
- Every pass streams N new realizations through one resident backend in fixed batches: device
  sampling, overlay, tracker/CRV response and (by default) device occupancy reduction, the one small
  terminal transfer and the closed per-batch output files. The first pass also constructs the
  backend (condition tables, upload) and pays compilation, so it lands in the timer's `setup_s`.
- Each pass writes `pass_metrics.json`: the summed compiled device step (dispatch to synchronisation) and the
  summed terminal transfer, host checks and file writes, and the largest demand of every sized buffer
  (`capacity_usage_max`, input to `python3 -m mixing.device_capacity`).
- Passes do the same work on different realizations: signal positions wrap through the pool and
  donor cursors continue, as in `stream_benchmark`. They are not repeats of identical events.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

PATHS = ("frames_manifest", "expanded", "truth", "stock", "photon_csv", "tracker_conditions",
         "device_capacities", "occupancy_geometry", "out")
DEFAULTS = {"device_kernel": "mixing.device_physics:model", "device_output": "occupancy", "device_platform": "gpu",
            "device_pipeline": True, "device_pipeline_writers": 1, "output_compression": None,
            "sampling_seed": 20261106, "seed": 20261106, "tracker_seed": 20261105, "chunk": 32, "synthetic": False}


def configuration(path=None):
    path = Path(path or os.environ["MIXING_BENCH_CONFIG"])
    values = {**DEFAULTS, **json.loads(path.read_text())}
    missing = [name for name in (*PATHS, "batch_events", "max_selected_rows") if values.get(name) is None]
    if missing:
        raise ValueError(f"bench configuration missing {missing}")
    for name in PATHS:
        values[name] = Path(values[name])
    return SimpleNamespace(**values)


SEED_NAMES = ("sampling_seed", "seed", "tracker_seed")


class Prepared:
    """Host pools plus the lazily constructed resident backend and its batch counter.

    ``shared`` (seeded prepares) is the first Prepared of the process for this configuration: its pools and backend
    are reused, and this object's seeds are put on the device before its pass."""

    def __init__(self, args, source, events, shared=None):
        self.args, self.source, self.events, self.backend, self.passes = args, source, events, None, 0
        self.shared = shared

    def ensure_backend(self):
        if self.shared is not None:
            backend = self.shared.ensure_backend()
            if self.backend is None:
                import numpy as np
                seeds = np.array([getattr(self.args, name) for name in SEED_NAMES], np.uint64)
                backend._seeds = backend.jax.device_put(seeds, backend.device)
                backend._arguments = (backend.pools, backend.state, backend.assets, backend._seeds)
                self.backend = backend
            return self.backend
        if self.backend is None:
            from .device_backend import Backend
            self.backend = Backend(self.args, self.source)
        return self.backend


_SHARED = {}


def prepare(events, config=None, seed=None):
    """Read the frame pools once; no device work. With ``seed`` (fresh inputs): see the module doc."""
    from .device_backend import Source
    if events < 1:
        raise ValueError("events must be positive")
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        key = str(Path(config or os.environ["MIXING_BENCH_CONFIG"]).resolve())
        shared = _SHARED.get(key)
        if shared is None:
            shared = _SHARED[key] = prepare(events, config)
        values = dict(vars(shared.args), out=shared.args.out / f"seed_{seed}")
        values.update({name: getattr(shared.args, name) + seed for name in SEED_NAMES})
        args = SimpleNamespace(**values)
        args.out.mkdir(exist_ok=False)
        return Prepared(args, shared.source, events, shared=shared)
    args = configuration(config)
    args.out.mkdir(parents=True, exist_ok=False)
    source = Source(args.frames_manifest, seed=args.sampling_seed, max_selected_rows=args.max_selected_rows)
    if args.batch_events > source.population:
        raise ValueError("batch_events cannot exceed the signal pool population")
    return Prepared(args, source, events)


def run_pass(prepared):
    """One timed pass: N realizations in batches through the resident backend; returns per-pass totals."""
    backend = prepared.ensure_backend()
    events = prepared.events
    directory = prepared.args.out / f"pass_{prepared.passes:02d}"
    directory.mkdir()
    remaining, batches, total = events, 0, 0
    step_s = transfer_s = 0.0
    usage = {}
    if prepared.args.device_pipeline:
        return _run_pipelined(prepared, backend, directory)
    while remaining:
        size = min(remaining, prepared.args.batch_events)
        request = prepared.source.next_batch(size)
        batch_dir = directory / f"batch_{request.batch_index:08d}"
        batch_dir.mkdir()
        metrics = backend.run(request, batch_dir)
        total += metrics["events"]
        step_s += metrics["response_s"]
        transfer_s += metrics["transfer_and_write_s"]
        for name, value in metrics.get("capacity_usage", {}).items():
            usage[name] = max(usage.get(name, 0), value)
        remaining -= size
        batches += 1
    backend.sync()
    prepared.passes += 1
    summary = {"events": total, "batches": batches, "device_step_s": step_s, "transfer_and_write_s": transfer_s,
               "capacity_usage_max": usage}
    (directory / "pass_metrics.json").write_text(json.dumps(summary) + "\n")
    return summary


def _run_pipelined(prepared, backend, directory):
    """device_pipeline: batch k is downloaded, checked and written (Backend.collect) on a worker thread while the
    main thread dispatches batch k + 1. A GPU execution holds its calling thread (the while-loop predicates are read
    back), so the overlap needs the thread. With device_pipeline_writers W, up to W batches are collected at once,
    each on its own thread (each batch has its own files). Same batches, keys and outputs as the serial loop."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from .stream_benchmark import pipeline_writers
    writers = pipeline_writers(prepared.args)
    remaining, batches, total, collect_s = prepared.events, 0, 0, 0.0
    usage, pending = {}, deque()

    def record(metrics):
        nonlocal total, collect_s, batches
        total += metrics["events"]
        collect_s += metrics["wait_transfer_and_write_s"]
        for name, value in metrics.get("capacity_usage", {}).items():
            usage[name] = max(usage.get(name, 0), value)
        batches += 1
    with ThreadPoolExecutor(max_workers=writers) as writer:
        while remaining:
            size = min(remaining, prepared.args.batch_events)
            request = prepared.source.next_batch(size)
            batch_dir = directory / f"batch_{request.batch_index:08d}"
            batch_dir.mkdir()
            handle = backend.dispatch(request)
            remaining -= size
            if len(pending) == writers:
                record(pending.popleft().result())
            pending.append(writer.submit(backend.collect, handle, batch_dir))
        while pending:
            record(pending.popleft().result())
    backend.sync()
    prepared.passes += 1
    summary = {"events": total, "batches": batches, "pipelined": True, "wait_transfer_and_write_s": collect_s,
               "capacity_usage_max": usage}
    (directory / "pass_metrics.json").write_text(json.dumps(summary) + "\n")
    return summary
