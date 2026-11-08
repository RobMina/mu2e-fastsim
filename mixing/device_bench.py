"""Entry points for the shared warm-rate timer (`mu2efastsim.bench`) on the resident device backend.

    MIXING_BENCH_CONFIG=config.json python3 -m mu2efastsim.bench mixing.device_bench:run_pass \\
        --events N --prepare mixing.device_bench:prepare --output-included --workflow mixing \\
        --boundary "..." --out bench.json

`config.json` holds the `stream_benchmark` arguments by their long names (frames_manifest, expanded,
truth, stock, photon_csv, tracker_conditions, batch_events, max_selected_rows, device_capacities,
occupancy_geometry, out, and optionally seeds, device_kernel, device_output, device_platform).

- `prepare(N)` reads and validates the five frame pools into host memory once (timer `prepare_s`).
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


class Prepared:
    """Host pools plus the lazily constructed resident backend and its batch counter."""

    def __init__(self, args, source, events):
        self.args, self.source, self.events, self.backend, self.passes = args, source, events, None, 0

    def ensure_backend(self):
        if self.backend is None:
            from .device_backend import Backend
            self.backend = Backend(self.args, self.source)
        return self.backend


def prepare(events, config=None):
    """Read the frame pools once; no device work."""
    from .device_backend import Source
    args = configuration(config)
    if events < 1:
        raise ValueError("events must be positive")
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
