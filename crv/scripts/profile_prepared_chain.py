"""Profile preparation and each audited pass without replacing the physics chain."""
import argparse
import cProfile
import functools
import json
import os
from pathlib import Path
import pstats
import time
from unittest.mock import patch

from crvfastsim.analyses import e2e_bench, runchain


def measured(function, directory, label):
    @functools.wraps(function)
    def call(*args, **kwargs):
        profiler = cProfile.Profile()
        start = time.perf_counter()
        try:
            return profiler.runcall(function, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            profiler.dump_stats(str(directory / f"{label}.pstats"))
            stats = pstats.Stats(profiler)
            rows = [{"file": key[0], "line": key[1], "function": key[2],
                     "primitive_calls": value[0], "calls": value[1],
                     "self_s": value[2], "cumulative_s": value[3]}
                    for key, value in stats.stats.items()]
            rows.sort(key=lambda row: row["cumulative_s"], reverse=True)
            (directory / f"{label}.json").write_text(json.dumps(
                {"elapsed_s": elapsed, "profiled": True, "functions": rows}, indent=2))
    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--inputs-root", required=True)
    parser.add_argument("--shards", nargs="+", type=int, default=[22])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--trace-pass", type=int)
    parser.add_argument("--reco-cpu", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--alternate-audit", action="store_true",
                        help="alternate 1/4 workers inside one prepared process; retain exact repeat checks")
    args = parser.parse_args()
    root = Path(args.inputs_root)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    options = dict(calib=str(root / "data/geometry/crvcalib_sim_best_v1_5_run1400.csv"),
                   deviation=str(root / "data/geometry/crvphoton_sim_best_v1_5_run1400.csv"),
                   gdml=str(root / "data/geometry/mu2e_extracted_v04.gdml"),
                   tier1b_set="strict_v4", tier1b_dir=str(root / "tier1b_strict_v2"),
                   jax=not args.cpu, max_events=args.max_events, prefetch=args.prefetch,
                   reco_cpu=args.reco_cpu)
    inputs = [str(root / f"gen_even/gen.001400_{shard:08d}.root") for shard in args.shards]
    prepare = runchain.prepare
    run = runchain.run
    index = 0

    def profiled_run(*a, **kw):
        nonlocal index
        label = f"pass_{index:03d}"
        index += 1
        return measured(run, out, label)(*a, **kw)

    if args.profile:
        with patch.object(runchain, "prepare", measured(prepare, out, "prepare")), \
             patch.object(runchain, "run", profiled_run):
            e2e_bench.run(inputs, outdir=str(out / "benchmark"), repeat=args.repeat,
                          seed=20261027, reuse_context=True, **options)
    elif args.trace_pass is not None:
        import jax
        from contextlib import ExitStack
        from crvfastsim import chain, tier1b, transport

        phase_intervals = []

        def traced(function, name):
            @functools.wraps(function)
            def call(*a, **kw):
                start = time.perf_counter_ns()
                try:
                    with jax.profiler.TraceAnnotation(name):
                        return function(*a, **kw)
                finally:
                    phase_intervals.append({"name": name, "start_ns": start,
                                            "end_ns": time.perf_counter_ns()})
            return call

        def traced_run(*a, **kw):
            nonlocal index
            step = index
            index += 1
            if step != args.trace_pass:
                return run(*a, **kw)
            jax.effects_barrier()
            before = jax.devices()[0].memory_stats()
            options = jax.profiler.ProfileOptions()
            options.host_tracer_level = 3
            options.python_tracer_level = 0
            with jax.profiler.trace(str(out / "trace"), create_perfetto_trace=True,
                                    profiler_options=options), ExitStack() as stack:
                for obj, name in ((tier1b, "sample"), (transport, "transport"), (chain, "charges"),
                                  (chain, "digis"), (chain, "reco_waveforms"), (chain, "clusters"),
                                  (runchain.PreparedContext, "audit")):
                    stack.enter_context(patch.object(obj, name, traced(getattr(obj, name), f"stage:{name}")))
                result = run(*a, **kw)
                jax.effects_barrier()
            (out / "device_memory.json").write_text(json.dumps(
                {"before": before, "after": jax.devices()[0].memory_stats(),
                 "phase_intervals_monotonic": phase_intervals}, indent=2))
            return result

        with patch.object(runchain, "run", traced_run):
            e2e_bench.run(inputs, outdir=str(out / "benchmark"), repeat=args.repeat,
                          seed=20261027, reuse_context=True, **options)
    elif args.alternate_audit:
        records = []
        def alternating_run(*a, **kw):
            nonlocal index
            step = index
            index += 1
            workers = (1, 4, 4, 1)[step % 4]
            os.environ["CRVFS_CONTEXT_HASH_WORKERS"] = str(workers)
            start = time.perf_counter()
            result = run(*a, **kw)
            records.append({"pass": step, "workers": workers, "elapsed_s": time.perf_counter() - start,
                            "audit_s": result["timers"]["context_audit_s"]})
            return result
        with patch.object(runchain, "run", alternating_run):
            e2e_bench.run(inputs, outdir=str(out / "benchmark"), repeat=args.repeat,
                          seed=20261027, reuse_context=True, **options)
        (out / "alternating_audit.json").write_text(json.dumps(records, indent=2))
    else:
        sync_root = os.environ.get("CRV_SPEED_SYNC_ROOT")
        if sync_root:
            def synchronized_run(*a, **kw):
                nonlocal index
                step = index
                index += 1
                folder = Path(sync_root) / f"barrier_{step:03d}"
                folder.mkdir(parents=True, exist_ok=True)
                worker = os.environ.get("CRV_SPEED_WORKER", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
                (folder / f"ready_{worker}").touch(exist_ok=False)
                deadline = time.monotonic() + 600
                while len(list(folder.glob("ready_*"))) != args.devices:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{args.devices}-device pass barrier timed out")
                    time.sleep(0.1)
                return run(*a, **kw)
            with patch.object(runchain, "run", synchronized_run):
                e2e_bench.run(inputs, outdir=str(out / "benchmark"), repeat=args.repeat,
                              seed=20261027, reuse_context=True, **options)
        else:
            e2e_bench.run(inputs, outdir=str(out / "benchmark"), repeat=args.repeat,
                          seed=20261027, reuse_context=True, **options)


if __name__ == "__main__":
    main()
