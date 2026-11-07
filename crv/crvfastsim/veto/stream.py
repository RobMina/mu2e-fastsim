"""C4 chain over many selection batches in one process: setup once, then primaries -> per-track veto decision.

    python3 -m crvfastsim.veto.stream --selection S.parquet [S2.parquet ...] --primaries P.parquet [P2 ...]
        --gdml G --calib C --deviation D [--library-dir L] [--card veto_stats.json] [--batch-events N]
        [--split all|dev|heldout] [--compile-cache DIR] [--bench bench.json] --outdir OUT

primary_chain.run loads the geometry, conditions, transport world, optical tables and library inside every call
(about 25 s on a GPU node, most of a 5k-event pass). Here they are loaded once (primary_chain.load_setup), and the
selection files are streamed through the chain in batches of --batch-events selected events. Batch i of a file
draws from the seed (seed, file, i); with one file and a batch at least as large as the selection the draws, and the
output, are those of primary_chain.run with --seed. With --card the pass ends at the per-track veto decision
(veto.apply.vetoed, the C4 rule) and chain.json reports the vetoed fraction; veto.paired compares with stock.

--bench runs the whole stream (all batches and the veto decision) three times in this process with the shared
warm-rate timer (mu2efastsim.bench): setup, warm selected events/s, N10, device label. Inputs are read into memory
before the timer starts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import awkward as ak
import numpy as np
import pandas as pd

from .paired import KEY

BOUNDARY = "C4 selected-event cosmic primaries -> per-track CRV veto decision"


def batch_seed(seed, file_index, batch_index, single):
    """Seed of one batch: the run seed itself for a single batch of a single file (primary_chain.run parity)."""
    if single:
        return seed
    return int(np.random.SeedSequence([int(seed), int(file_index), int(batch_index)]).generate_state(1, np.uint64)[0])


def batches(selection, primaries, census, batch_events):
    """Contiguous event batches of a selection with their primaries (and census rows), in selection order."""
    n = len(selection)
    size = n if not batch_events or batch_events >= n else int(batch_events)
    if size < 1:
        raise ValueError("batch size must be positive")
    keys = pd.DataFrame({k: ak.to_numpy(selection[k]) for k in KEY})
    keys["batch"] = np.arange(n) // size
    owner = primaries[list(KEY)].merge(keys, on=list(KEY), how="left", validate="many_to_one").batch.to_numpy()
    if np.isnan(owner).any():
        raise ValueError("primary event absent from cached selection")
    cowner = None
    if census is not None:
        cowner = census[list(KEY)].merge(keys, on=list(KEY), how="left", validate="many_to_one").batch.to_numpy()
    for b, a in enumerate(range(0, n, size)):
        part_census = None if census is None else census[cowner == b]
        yield b, selection[a:a + size], primaries[owner == b], part_census


def decisions(selection, fast, card):
    """Per-track C4 veto decision ([E, T] bool) and the event gate ([E] bool) of the fast coincidences."""
    from .apply import gate, vetoed
    from .paired import align
    candidate = align(selection, fast)
    return vetoed(selection.t_mid, candidate.coincs, card), gate(candidate.coincs, card)


def stream(inputs, setup, gdml, calib, deviation, library_dir=None, seed=20261105, period=1695., batch_events=None,
           input_kind="represented", card=None):
    """Run the C4 chain over [(selection, primaries, census)] with one setup. Returns (fast, reports, veto) where fast
    concatenates every file's coincidence rows in input order and veto is the per-track decision (or None)."""
    from .primary_chain import run
    fast, reports, veto, gates = [], [], [], []
    single = len(inputs) == 1
    for f, (selection, primaries, census) in enumerate(inputs):
        parts = list(batches(selection, primaries, census, batch_events))
        for b, sel, prim, cen in parts:
            out, report = run(sel, prim, gdml, calib, deviation, batch_seed(seed, f, b, single and len(parts) == 1),
                              period, library_dir, input_kind, cen, setup=setup)
            report.update(file=f, batch=b)
            fast.append(out)
            reports.append(report)
            if card is not None:
                v, g = decisions(sel, out, card)
                veto.append(v)
                gates.append(g)
    joined = ak.concatenate(fast) if len(fast) > 1 else fast[0]
    if card is None:
        return joined, reports, None
    return joined, reports, (ak.concatenate(veto) if len(veto) > 1 else veto[0], np.concatenate(gates))


def load_card(path):
    saved = json.loads(Path(path).read_text())
    values = dict(saved.get("card", saved))
    values.pop("sha", None)
    from .card import VetoCard
    return VetoCard.from_dict(values)


def load_inputs(selections, primaries, split, split_at, input_kind, census_paths=None):
    """Read every selection/primaries pair into memory, with the split applied as primary_chain.main does."""
    from .generators import select_split
    if len(selections) != len(primaries) or (census_paths and len(census_paths) != len(selections)):
        raise ValueError("give one primaries (and census) file per selection file")
    out = []
    for i, (s, p) in enumerate(zip(selections, primaries)):
        sel, prim = ak.from_parquet(s), pd.read_parquet(p)
        census = pd.read_parquet(census_paths[i]) if census_paths else None
        sel = select_split(sel, split, split_at)
        if split != "all" and input_kind == "represented":
            keep = prim.shard < split_at
            prim = prim[keep] if split == "dev" else prim[~keep]
        if input_kind == "retained":
            from .generators import prepare
            prim, _ = prepare(sel, prim, census)
        out.append((sel, prim, census))
    return out


def summary(reports, veto, selection_events, wall):
    timers = {}
    for r in reports:
        for k, v in r["timers"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                timers[k] = timers.get(k, 0) + v
    res = dict(batches=len(reports), selected_events=selection_events, wall_s=wall,
               selected_events_per_s=selection_events / wall if wall > 0 else None,
               steps=sum(r["steps"] for r in reports), clusters=sum(r["clusters"] for r in reports),
               timers=timers, limitations=reports[0]["limitations"])
    if veto is not None:
        v, g = veto
        res["tracks"] = int(ak.sum(ak.num(v)))
        res["vetoed_tracks"] = int(ak.sum(v))
        res["gate_failed_events"] = int((~g).sum())
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", nargs="+", required=True)
    parser.add_argument("--primaries", nargs="+", required=True)
    for key in ("gdml", "calib", "deviation", "outdir"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--library-dir")
    parser.add_argument("--input-kind", choices=("represented", "retained"), default="represented")
    parser.add_argument("--generator-events", nargs="+", help="event census per selection file (retained inputs)")
    parser.add_argument("--card", help="VetoCard JSON or saved veto_stats.json: end each pass at the veto decision")
    parser.add_argument("--batch-events", type=int, help="selected events per chain call (default: one per file)")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--period", type=float, default=1695.)
    parser.add_argument("--split", choices=("all", "dev", "heldout"), default="all")
    parser.add_argument("--split-at", type=int, default=22)
    parser.add_argument("--compile-cache", metavar="DIR", help="JAX persistent compilation cache directory")
    parser.add_argument("--bench", metavar="JSON", help="time three passes with mu2efastsim.bench; write its report")
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args(argv)
    if args.input_kind == "retained" and not args.generator_events:
        parser.error("--input-kind retained requires --generator-events")
    if args.compile_cache:
        from ..compile_cache import enable
        print(f"# compile cache: {enable(args.compile_cache)}", flush=True)
    t = time.perf_counter()
    inputs = load_inputs(args.selection, args.primaries, args.split, args.split_at, args.input_kind,
                         args.generator_events)
    prepare_s = time.perf_counter() - t
    card = load_card(args.card) if args.card else None
    events = sum(len(s) for s, _, _ in inputs)
    from .primary_chain import load_setup
    common = (args.gdml, args.calib, args.deviation, args.library_dir)
    last = {}

    def one_pass(setup=None):
        begin = time.perf_counter()
        setup = setup if setup is not None else load_setup(*common)
        last["result"] = stream(inputs, setup, *common, seed=args.seed, period=args.period,
                                batch_events=args.batch_events, input_kind=args.input_kind, card=card)
        last["wall"] = time.perf_counter() - begin
        return None

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.bench:
        from mu2efastsim import bench
        cached = {}

        def timed_pass():   # setup belongs to the first pass (the timer reports it as setup_s)
            if "setup" not in cached:
                cached["setup"] = load_setup(*common)
            return one_pass(cached["setup"])
        from mu2efastsim import precision
        report = bench.run(timed_pass, events, passes=args.passes, workflow="W5", boundary=BOUNDARY,
                           prepare_s=prepare_s, dtype=precision.NAME)
        report["command"] = "python3 -m crvfastsim.veto.stream " + " ".join(argv if argv is not None else
                                                                             __import__("sys").argv[1:])
        Path(args.bench).write_text(json.dumps(report, indent=1) + "\n")
        print(bench.format_report(report), flush=True)
    else:
        one_pass()
    fast, reports, veto = last["result"]
    selection = ak.concatenate([s for s, _, _ in inputs]) if len(inputs) > 1 else inputs[0][0]
    ak.to_parquet(selection, out / "selection.parquet")
    ak.to_parquet(fast, out / "fast.parquet")
    res = summary(reports, veto, events, last["wall"])
    res.update(seed=args.seed, batch_events=args.batch_events, split=args.split, input_kind=args.input_kind,
               boundary=BOUNDARY if card is not None else "C4 selected-event cosmic primaries -> CRV coincidences",
               inputs=dict(selection=args.selection, primaries=args.primaries), prepare_s=prepare_s)
    (out / "chain.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in res if k not in ("timers", "limitations")}), flush=True)


if __name__ == "__main__":
    main()
