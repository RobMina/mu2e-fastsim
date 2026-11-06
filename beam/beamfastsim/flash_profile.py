"""Measure source setup, JIT compilation and fixed-shape execution without changing physics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def step_summary(pdg, nvac, nslab, nstall):
    """Saved counters give a lower bound on batch iterations, not exact termination times."""
    out = {}
    for code in (11, -11):
        mask = np.asarray(pdg) == code
        v, s, w = (np.asarray(a, np.int64)[mask] for a in (nvac, nslab, nstall))
        if not len(v):
            continue
        bound = int(np.maximum(v, s).max())
        work = v + s
        out[str(code)] = dict(parents=len(v), vacuum_steps=int(v.sum()), slab_steps=int(s.sum()),
                             stalls=int(w.sum()), iteration_lower_bound=bound,
                             vacuum_quantiles=np.quantile(v, [0, .5, .9, .99, 1]).tolist(),
                             active_step_fraction_upper_bound=float(work.sum() / (len(v) * max(bound, 1))),
                             parents_over_100k_steps=int(np.count_nonzero(work > 100000)))
    return out


def compare_saved(result, path):
    """Compare every saved state and photon array, preserving exact equality information."""
    comparisons = {}
    with np.load(path, allow_pickle=False) as old:
        for section, prefix in (("state", "state_"), ("records", "photon_")):
            for name, value in result[section].items():
                key = prefix + name
                if key not in old:
                    comparisons[key] = dict(exact=False, missing_reference=True)
                    continue
                a, b = np.asarray(value), old[key]
                exact = bool(np.array_equal(a, b, equal_nan=True))
                item = dict(exact=exact, shape_equal=a.shape == b.shape)
                if a.shape == b.shape:
                    item["different_elements"] = int(np.count_nonzero(a != b))
                if a.shape == b.shape and a.dtype.kind == "f":
                    item["max_abs_delta"] = float(np.nanmax(np.abs(a - b))) if a.size else 0.
                comparisons[key] = item
        for key in ("overflow", "capped", "loopers"):
            comparisons[key] = dict(exact=bool(np.array_equal(result[key], old[key])))
    return comparisons


def profile(args):
    import jax
    import jax.numpy as jnp
    from . import cechain as C
    from . import flash_navigator_jax as FN
    from .flash_flux import read_states
    from .flash_target_jax import target_source

    if args.cpu_vacuum:
        args.skip_empty = args.vacuum_fast = True
    report = dict(devices=list(map(str, jax.devices())), backend=jax.default_backend(),
                  checkpoints=args.checkpoints, skip_empty=args.skip_empty, vacuum_fast=args.vacuum_fast,
                  vacuum_bucket=args.vacuum_bucket, cpu_vacuum=args.cpu_vacuum, cpu_adaptive=args.cpu_adaptive,
                  species=[], source=str(args.states),
                  geometry=str(args.geom), source_events=args.source_events,
                  scope="Same shape/order/key across chunks; slab RNG iteration retained; no physics cuts")
    args.out.mkdir(parents=True, exist_ok=False)

    def emit(label, **values):
        print(json.dumps(dict(stage=label, **values), allow_nan=False), flush=True)
        (args.out / "profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    start = time.perf_counter()
    from .flash_geometry import from_offline
    geom = from_offline(str(args.geom), str(args.offline_root))
    source = read_states(args.states)
    maps = FN.device_maps([str(p) for p in args.maps], dtype=np.float64)
    jax.block_until_ready(maps)
    report["input_geometry_maps_seconds"] = time.perf_counter() - start
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as saved:
            report["saved_step_summary"] = step_summary(saved["source_pdg"], saved["state_nvac"],
                                                         saved["state_nslab"], saved["state_nstall"])
    emit("inputs", seconds=report["input_geometry_maps_seconds"], saved=report.get("saved_step_summary"))
    original = FN._build

    def measured_build(*a, **kw):
        wall, cpu = time.perf_counter(), time.process_time()
        if args.skip_empty:
            kw["skip_empty"] = True
        if args.vacuum_fast:
            kw["vacuum_fast"] = True
        if args.vacuum_bucket:
            kw["vacuum_bucket"] = args.vacuum_bucket
        init, run = original(*a, **kw)
        info = dict(pdg=-11 if kw.get("positron") else 11, build_seconds=time.perf_counter() - wall,
                    build_cpu_seconds=time.process_time() - cpu, chunks=[])
        report["species"].append(info)
        emit("tables", **info)

        def measured_init(*ia):
            wall = time.perf_counter()
            lowered = init.lower(*ia)
            info["init_lower_seconds"] = time.perf_counter() - wall
            wall = time.perf_counter()
            compiled = lowered.compile()
            info["init_compile_seconds"] = time.perf_counter() - wall
            wall = time.perf_counter()
            state = jax.block_until_ready(compiled(*ia))
            info["init_execute_seconds"] = time.perf_counter() - wall
            emit("init", **info)
            return state

        def measured_run(maps, key, st, limit, m_cand, m_slab):
            wall, cpu = time.perf_counter(), time.process_time()
            if args.cpu_vacuum:
                from .flash_hybrid import make_runner
                info["hybrid"] = {}
                compiled = make_runner(run, maps, key, st, m_cand, m_slab, args.cpu_vacuum, info["hybrid"],
                                       adaptive=args.cpu_adaptive)
                info["run_lower_seconds"] = 0.
                info["run_compile_seconds"] = time.perf_counter() - wall
            else:
                lowered = run.lower(maps, key, st, limit, m_cand, m_slab)
                info["run_lower_seconds"] = time.perf_counter() - wall
                wall = time.perf_counter()
                compiled = lowered.compile()
                info["run_compile_seconds"] = time.perf_counter() - wall
            info["lower_compile_cpu_seconds"] = time.process_time() - cpu
            info["parents"] = m_slab
            emit("compiled", **info)
            total = 0
            for end in sorted(set(args.checkpoints + [int(limit)])):
                if end <= total or end > int(limit):
                    continue
                previous_v, previous_s = np.asarray(st["nvac"]), np.asarray(st["nslab"])
                wall, cpu = time.perf_counter(), time.process_time()
                steps, st = jax.block_until_ready(compiled(maps, key, st, jnp.int32(end - total)))
                elapsed, cpu_s = time.perf_counter() - wall, time.process_time() - cpu
                count = int(steps)
                total += count
                live = np.asarray(st["fate"]) == C.LIVE
                dv = int((np.asarray(st["nvac"]) - previous_v).sum())
                ds = int((np.asarray(st["nslab"]) - previous_s).sum())
                chunk = dict(end_iteration=total, iterations=count, seconds=elapsed, cpu_seconds=cpu_s,
                             live=int(live.sum()), live_in_slab=int(np.count_nonzero(live & np.asarray(st["inl"]))),
                             vacuum_lane_steps=dv, slab_lane_steps=ds,
                             active_step_fraction=float((dv + ds) / max(count * m_slab, 1)),
                             photons=int(np.asarray(st["count"]).sum()))
                info["chunks"].append(chunk)
                emit("chunk", pdg=info["pdg"], **chunk)
                if not live.any():
                    break
            info["iterations"] = total
            info["execute_seconds"] = sum(c["seconds"] for c in info["chunks"])
            return jnp.int32(total), st
        return measured_init, measured_run

    p = np.linalg.norm(source["p"], axis=1)
    FN._build = measured_build
    try:
        wall = time.perf_counter()
        result = target_source(jax.random.PRNGKey(args.seed), geom, maps, source["x"], source["p"] / p[:, None],
                               p, source["time_ns"], source["pdg"], capacity=args.capacity,
                               max_steps=args.max_steps, h=args.h, s_max=args.s_max, unroll=args.unroll)
        jax.block_until_ready(result)
        report["instrumented_source_seconds"] = time.perf_counter() - wall
        wall = time.perf_counter()
        result = jax.tree_util.tree_map(np.asarray, result)
        report["final_host_copy_seconds"] = time.perf_counter() - wall
    finally:
        FN._build = original
    report["execute_seconds"] = sum(s["execute_seconds"] for s in report["species"])
    report["execute_input_events_per_second"] = args.source_events / report["execute_seconds"]
    if args.reference:
        report["reference_comparison"] = compare_saved(result, args.reference)
        report["reference_all_exact"] = all(v["exact"] for v in report["reference_comparison"].values())
    if args.flux_summary:
        from .flash_flux import exit_records, summarize
        payload = {"source_" + k: v for k, v in source.items()}
        payload.update({"state_" + k: v for k, v in result["state"].items()})
        payload.update({"photon_" + k: v for k, v in result["records"].items()})
        payload.update({k: result[k] for k in ("overflow", "capped", "loopers")})
        flux, _ = summarize(payload, exit_records(payload, geom.beam), source_events=args.source_events)
        (args.out / "flux.json").write_text(json.dumps(flux, indent=2, allow_nan=False) + "\n")
        report["flux_totals"] = flux["totals"]
    emit("finished", seconds=report["instrumented_source_seconds"], execution=report["execute_seconds"],
         exact=report.get("reference_all_exact"))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--geom", type=Path, required=True)
    p.add_argument("--offline-root", type=Path, required=True)
    p.add_argument("--maps", type=Path, nargs="+", required=True)
    p.add_argument("--source-events", type=int, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 10000, 50000, 100000, 200000, 500000])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--capacity", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=2000000)
    p.add_argument("--h", type=float, default=1.)
    p.add_argument("--s-max", type=float, default=30000.)
    p.add_argument("--unroll", action="store_true")
    p.add_argument("--skip-empty", action="store_true", help="Opt-in empty crossing/frame fast path")
    p.add_argument("--vacuum-fast", action="store_true", help="Bypass slab and crossing bookkeeping in free vacuum")
    p.add_argument("--vacuum-bucket", type=int, default=0, help="Compact vacuum-only tails into this many lanes")
    p.add_argument("--cpu-vacuum", type=int, default=0, help="CPU vacuum tail bucket; material remains on the original device")
    p.add_argument("--cpu-adaptive", action="store_true", help="Shrink CPU vacuum buckets as lanes finish")
    p.add_argument("--flux-summary", action="store_true", help="Also recompute foil exits and attenuation after timing")
    profile(p.parse_args(argv))


if __name__ == "__main__":
    main()
