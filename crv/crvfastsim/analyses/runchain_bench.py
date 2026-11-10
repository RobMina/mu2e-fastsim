"""Warm-rate measurement of runchain (generated CRY cosmics -> CRV coincidence clusters) with the shared timer.

    python3 -m crvfastsim.analyses.runchain_bench gen.A.root gen.B.root ... --calib C [--deviation D]
        [--run-config run1a_v01] [--window offspill] [--gdml G] [--tier1b SET --tier1b-dir DIR | --library L]
        [--seed 7] [--prefetch] [--resident-response] [--compile-cache DIR] [--passes 3] --out bench.json [--outdir DIR]
        [--fresh-inputs --files-per-pass K [--allow-file-reuse]]
        [--approx-preset default|fast|fidelity] [--approx NAME=LEVEL ...] [--approximations FILE]

The gen tables are read into host memory first (reported as prepare_s). Each pass is one runchain.run over all of
them in this process (the multi-file stream: every file through transport, tier 1b, response and reco). The
prepared setup context (geometry, conditions, world, library, optical tables) is built inside the first pass, so
mu2efastsim.bench reports it with the compile as setup_s; passes 2.. reuse it. Output: the timer's JSON report with
the last pass's runchain stats (clusters per generated event and the stage timers) under "runchain".

By default every pass runs all the files with the same seed. With --fresh-inputs (the shared timer's fresh-input
mode, mu2efastsim.bench.fresh_inputs) pass i runs its own K = --files-per-pass files, files[i*K : (i+1)*K], with
seed --seed + i, so no pass repeats the transport, response and reco of another. The files must cover passes x K
distinct files; with fewer, the run is refused unless --allow-file-reuse, which takes the files cyclically (a reused
file is generated-level identical but is transported with a new seed) and records the reuse in the report. All gen
tables are read once before the passes (prepare_s). Passes may then hold different event counts: the report's
warm_rate and n10 use the events of each pass (events_per_pass), and "events" is their mean.
"""
import argparse
import json
import time
from pathlib import Path

BOUNDARY = "generated CRY cosmics -> CRV coincidence clusters (runchain)"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="CRY gen tables (gen.*.root)")
    p.add_argument("--calib", required=True)
    p.add_argument("--deviation")
    p.add_argument("--run-config")
    p.add_argument("--window")
    p.add_argument("--gdml")
    p.add_argument("--tier1b", dest="tier1b_set")
    p.add_argument("--tier1b-dir")
    p.add_argument("--library")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--device-refit", action="store_true",
                   help="flagged waveforms refit on the device first (ChainConfig.device_refit)")
    p.add_argument("--resident-response", action=argparse.BooleanOptionalAction, default=None,
                   help="photons stay on the device into the SiPM stage (ChainConfig.resident_response); default "
                        "on (runchain's default with --jax); --no-resident-response: the two-stage path")
    p.add_argument("--compile-cache", metavar="DIR")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--out", required=True, help="timer report JSON")
    p.add_argument("--outdir", default=None, help="runchain stats directory (default: next to --out)")
    p.add_argument("--fresh-inputs", action="store_true",
                   help="each pass runs its own --files-per-pass files with seed --seed + pass index")
    p.add_argument("--files-per-pass", type=int, metavar="K", help="files per pass with --fresh-inputs")
    p.add_argument("--allow-file-reuse", action="store_true",
                   help="with --fresh-inputs: take files cyclically when there are fewer than passes x K")
    from mu2efastsim import approximations
    from .. import approximations as W5
    approximations.add_arguments(p)
    a = p.parse_args(argv)
    try:
        selection = approximations.from_args(a, W5.WORKFLOW)
    except ValueError as e:
        p.error(str(e))
    if a.fresh_inputs:
        if not a.files_per_pass or a.files_per_pass < 1:
            p.error("--fresh-inputs needs --files-per-pass K >= 1")
    elif a.files_per_pass or a.allow_file_reuse:
        p.error("--files-per-pass and --allow-file-reuse need --fresh-inputs")
    if a.compile_cache:
        from ..compile_cache import enable
        print(f"# compile cache: {enable(a.compile_cache)}", flush=True)
    from .. import tables
    from . import runchain
    from mu2efastsim import bench, precision
    files = list(tables.expand(a.inputs))
    plan = None
    if a.fresh_inputs:
        if not hasattr(bench, "fresh_inputs"):
            raise SystemExit("runchain_bench: --fresh-inputs needs a core with mu2efastsim.bench.fresh_inputs")
        plan = pass_files(files, a.passes, a.files_per_pass, a.allow_file_reuse)
    begin = time.perf_counter()
    gen_tables = {f: tables.read(f, "gen") for f in files}
    prepare_s = time.perf_counter() - begin
    counts = {f: int(g[tables.KEY].drop_duplicates().shape[0]) for f, g in gen_tables.items()}
    events = sum(counts.values())
    setup = dict(library=a.library, calib=a.calib, deviation=a.deviation, gdml=a.gdml, jax=True,
                 run_config=a.run_config, window=a.window, tier1b_set=a.tier1b_set, tier1b_dir=a.tier1b_dir)
    if a.resident_response is not None:
        setup["resident_response"] = a.resident_response
    if a.device_refit:
        setup["device_refit"] = True
    try:   # the levels' runchain options (none at the default levels); the bench runs on JAX, so jax=False is refused
        merged = selection.apply_kwargs(setup)
    except ValueError as e:
        p.error(str(e))
    run_kw = {k: v for k, v in merged.items() if k not in setup}   # run-time options only (not the prepared context)
    run_kw["approximations"] = selection
    outdir = a.outdir or str(Path(a.out).with_suffix("")) + "_runchain"
    state = {}

    def one_pass(drawn=None):
        if "context" not in state:
            state["context"] = runchain.prepare(**setup)
        run_files, run_tables, seed = drawn if drawn is not None else (files, gen_tables, a.seed)
        state["result"] = runchain.run(run_files, outdir, seed=seed, prepared=state["context"], gen_tables=run_tables,
                                       prefetch=a.prefetch, transport_backend="auto", tier1b_backend="auto", **setup,
                                       **run_kw)
        return None   # the chain hands host arrays between stages: the device is idle when run() returns
    if plan is None:
        report = bench.run(one_pass, events, passes=a.passes, workflow="W5", boundary=BOUNDARY, prepare_s=prepare_s,
                           dtype=precision.NAME)
    else:
        per_pass = [sum(counts[f] for f in chosen) for chosen in plan]

        def draw(n, seed):
            chosen = plan[seed - a.seed]
            return list(chosen), {f: gen_tables[f] for f in chosen}, seed
        seeds = [a.seed + i for i in range(a.passes)]
        report = bench.run(one_pass, round(sum(per_pass) / len(per_pass)), passes=a.passes, workflow="W5",
                           boundary=BOUNDARY, prepare_s=prepare_s, dtype=precision.NAME, seeds=seeds,
                           inputs=bench.fresh_inputs(draw, per_pass[0], seed=a.seed, spec="runchain_bench files"))
        warm_events, warm_s = sum(per_pass[1:]), sum(report["pass_s"][1:])
        report["warm_rate"] = warm_events / warm_s if warm_s > 0 else float("inf")
        report["n10"] = 9.0 * report["setup_s"] * report["warm_rate"]
        reused = len(set(f for chosen in plan for f in chosen)) < sum(len(chosen) for chosen in plan)
        report.update(events_per_pass=per_pass, files_per_pass=[[Path(f).name for f in chosen] for chosen in plan],
                      file_reuse=reused)
        if reused:
            print("# runchain_bench: --allow-file-reuse: some files run in more than one pass (new seed each pass)",
                  flush=True)
    report["command"] = "python3 -m crvfastsim.analyses.runchain_bench " + " ".join(
        argv if argv is not None else __import__("sys").argv[1:])
    res = state["result"]
    report["approximations"] = res.get("approximations")
    report["runchain"] = {k: res[k] for k in ("generated_events", "clusters_per_generated", "timers", "run_config",
                                              "tier1b", "conditions", "precision") if k in res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(bench.format_report(report), flush=True)


def pass_files(files, passes, k, allow_reuse=False):
    """The files of each pass for --fresh-inputs: disjoint K-file slices, or cyclic with ``allow_reuse``."""
    if k > len(files):
        raise SystemExit(f"runchain_bench: --files-per-pass {k} exceeds the {len(files)} input files")
    if passes * k > len(files) and not allow_reuse:
        raise SystemExit(f"runchain_bench: --fresh-inputs needs {passes} x {k} = {passes * k} distinct files for "
                         f"disjoint passes and got {len(files)}; add files, lower --files-per-pass, or pass "
                         f"--allow-file-reuse (files taken cyclically, new seed per pass, recorded in the report)")
    return [[files[(i * k + j) % len(files)] for j in range(k)] for i in range(passes)]


if __name__ == "__main__":
    main()
