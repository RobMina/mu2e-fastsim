"""Warm-rate measurement of runchain (generated CRY cosmics -> CRV coincidence clusters) with the shared timer.

    python3 -m crvfastsim.analyses.runchain_bench gen.A.root gen.B.root ... --calib C [--deviation D]
        [--run-config run1a_v01] [--window offspill] [--gdml G] [--tier1b SET --tier1b-dir DIR | --library L]
        [--seed 7] [--prefetch] [--resident-response] [--compile-cache DIR] [--passes 3] --out bench.json [--outdir DIR]

The gen tables are read into host memory first (reported as prepare_s). Each pass is one runchain.run over all of
them in this process (the multi-file stream: every file through transport, tier 1b, response and reco). The
prepared setup context (geometry, conditions, world, library, optical tables) is built inside the first pass, so
mu2efastsim.bench reports it with the compile as setup_s; passes 2.. reuse it. Output: the timer's JSON report with
the last pass's runchain stats (clusters per generated event and the stage timers) under "runchain".
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
    p.add_argument("--resident-response", action="store_true",
                   help="photons stay on the device into the SiPM stage (ChainConfig.resident_response)")
    p.add_argument("--compile-cache", metavar="DIR")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--out", required=True, help="timer report JSON")
    p.add_argument("--outdir", default=None, help="runchain stats directory (default: next to --out)")
    a = p.parse_args(argv)
    if a.compile_cache:
        from ..compile_cache import enable
        print(f"# compile cache: {enable(a.compile_cache)}", flush=True)
    from .. import tables
    from . import runchain
    from mu2efastsim import bench, precision
    files = list(tables.expand(a.inputs))
    begin = time.perf_counter()
    gen_tables = {f: tables.read(f, "gen") for f in files}
    prepare_s = time.perf_counter() - begin
    events = sum(int(g[tables.KEY].drop_duplicates().shape[0]) for g in gen_tables.values())
    setup = dict(library=a.library, calib=a.calib, deviation=a.deviation, gdml=a.gdml, jax=True,
                 run_config=a.run_config, window=a.window, tier1b_set=a.tier1b_set, tier1b_dir=a.tier1b_dir)
    if a.resident_response:
        setup["resident_response"] = True
    outdir = a.outdir or str(Path(a.out).with_suffix("")) + "_runchain"
    state = {}

    def one_pass():
        if "context" not in state:
            state["context"] = runchain.prepare(**setup)
        state["result"] = runchain.run(files, outdir, seed=a.seed, prepared=state["context"], gen_tables=gen_tables,
                                       prefetch=a.prefetch, transport_backend="auto", tier1b_backend="auto", **setup)
        return None   # the chain hands host arrays between stages: the device is idle when run() returns
    report = bench.run(one_pass, events, passes=a.passes, workflow="W5", boundary=BOUNDARY, prepare_s=prepare_s,
                       dtype=precision.NAME)
    report["command"] = "python3 -m crvfastsim.analyses.runchain_bench " + " ".join(
        argv if argv is not None else __import__("sys").argv[1:])
    res = state["result"]
    report["runchain"] = {k: res[k] for k in ("generated_events", "clusters_per_generated", "timers", "run_config",
                                              "tier1b", "conditions", "precision") if k in res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(bench.format_report(report), flush=True)


if __name__ == "__main__":
    main()
