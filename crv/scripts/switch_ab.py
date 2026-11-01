"""Interleaved A/B of environment switches on the prepared chain: exact output identity and paired warm-pass times.

run: one prepared process alternates the arms between passes, so every arm sees the same host, device and warm caches
and e2e-bench's repeat gate requires identical fingerprints across arms. It suits switches read at call time. A switch
read while a kernel is traced, or at preparation, needs one process per arm (one --arm each); report then pairs them.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
from unittest.mock import patch

import numpy as np

from crvfastsim.analyses import e2e_bench, runchain
from profile_prepared_chain import chain_options

SEED = 20261027


def parse_arm(text):
    """name[:VAR=VALUE[,VAR=VALUE...]]"""
    name, _, spec = text.partition(":")
    pairs = [item.split("=", 1) for item in spec.split(",") if item]
    if not name or any(len(pair) != 2 or not pair[0] for pair in pairs):
        raise ValueError(f"bad arm: {text}")
    return name, dict(pairs)


def schedule(names, warmup, rounds):
    """Warm-up passes first (compilation, page cache), then rounds in alternating order: A B | B A | A B | ..."""
    order = [(name, -1) for name in names for _ in range(warmup)]
    for index in range(rounds):
        order += [(name, index) for name in (names if index % 2 == 0 else names[::-1])]
    return order


def set_arm(arms, name):
    for variable in {v for settings in arms.values() for v in settings}:
        os.environ.pop(variable, None)
    os.environ.update(arms[name])


def seconds(timers):
    return {k: v for k, v in timers.items() if k.endswith("_s") and isinstance(v, float)}


def summarize(records, reference):
    """Per-arm medians and paired ratios reference/arm over the measured rounds; warm-up passes are left out."""
    arms = {}
    measured = [r for r in records if r["round"] >= 0]
    for name in dict.fromkeys(r["arm"] for r in measured):
        rows = sorted((r for r in measured if r["arm"] == name), key=lambda r: r["round"])
        stages = sorted({k for r in rows for k in r["timers"]})
        arms[name] = {"rounds": len(rows), "elapsed_s": [r["elapsed_s"] for r in rows],
                      "median_s": statistics.median(r["elapsed_s"] for r in rows),
                      "stage_median_s": {k: statistics.median(r["timers"].get(k, 0.0) for r in rows) for k in stages}}
    base = {r["round"]: r["elapsed_s"] for r in measured if r["arm"] == reference}
    for name, row in arms.items():
        ratios = [base[r["round"]] / r["elapsed_s"] for r in measured if r["arm"] == name and r["round"] in base]
        row["paired_speedup_median"] = statistics.median(ratios) if ratios else None
        row["paired_speedup_range"] = [min(ratios), max(ratios)] if ratios else None
    return arms


def diff_npz(reference, candidate):
    """Named arrays that differ between two per-input outputs, with counts; first report for a non-identical arm."""
    result = {}
    with np.load(reference) as a, np.load(candidate) as b:
        for name in sorted(set(a.files) | set(b.files)):
            if name not in a.files or name not in b.files:
                result[name] = {"missing": True}
                continue
            x, y = a[name], b[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                result[name] = {"shape": [list(x.shape), list(y.shape)], "dtype": [x.dtype.str, y.dtype.str]}
            elif x.tobytes() != y.tobytes():
                changed = (x != y) & ~((x != x) & (y != y))
                row = {"size": int(x.size), "changed": int(changed.sum())}
                if x.dtype.kind == "f" and changed.any():
                    row["max_abs"] = float(np.max(np.abs(x[changed] - y[changed])))
                result[name] = row
    return result


def run(args):
    arms = dict(parse_arm(text) for text in args.arm)
    names = list(arms)
    order = schedule(names, args.warmup, args.rounds)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    records, passes, step = [], runchain.run, 0

    def alternating(*a, **kw):
        nonlocal step
        name, index = order[step]
        step += 1
        set_arm(arms, name)
        before = len(records)
        records.append({"pass": step - 1, "arm": name, "round": index})
        (out / "progress.json").write_text(json.dumps(records, indent=1))
        result = passes(*a, **kw)
        records[before]["timers"] = seconds(result["timers"])
        return result

    options = chain_options(Path(args.inputs_root), args.shards, cpu=args.cpu, max_events=args.max_events,
                            prefetch=args.prefetch, reco_cpu=args.reco_cpu)
    inputs = options.pop("inputs")
    summary = {"arms": arms, "reference": names[0], "seed": args.seed, "shards": args.shards,
               "max_events": args.max_events, "chain": "numpy" if args.cpu else "jax"}
    set_arm(arms, names[0])
    bench = out / "benchmark"
    try:
        with patch.object(runchain, "run", alternating):
            report = e2e_bench.run(inputs, outdir=str(bench), repeat=len(order), seed=args.seed,
                                   reuse_context=True, **options)
    except ValueError as exc:
        if "fingerprint" not in str(exc):
            raise
        last = records[-1]
        files = sorted((bench / "pass_000").glob("input_*.npz"))
        summary.update(identical=False, error=str(exc), differs_at_pass=last,
                       first_difference={f.name: diff_npz(f, bench / f"pass_{last['pass']:03d}" / f.name) for f in files})
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        raise SystemExit(f"arm {last['arm']} differs from arm {order[0][0]}: {out / 'summary.json'}")
    for record, item in zip(records, report["passes"]):
        record["elapsed_s"] = item["elapsed_s"]
    summary.update(identical=True, n_arms=len(names), device=report["device"],
                   host=report["hardware_software"]["hostname"], versions=report["hardware_software"]["versions"],
                   preparation_s=report.get("preparation_elapsed_s"), cold_s=report["passes"][0]["elapsed_s"],
                   counts=report["passes"][0]["counts"], fingerprints=[f["fingerprint"] for f in report["passes"][0]["files"]],
                   passes=records, summary=summarize(records, names[0]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def report(args):
    """Pair separate-process arms: reference and candidate directories alternate in the job (A B B A ...)."""
    rows = {}
    for text in args.arm:
        name, _, directory = text.partition("=")
        rows.setdefault(name, []).append(json.loads((Path(directory) / "summary.json").read_text()))
    reference = args.reference
    base = rows[reference][0]
    result = {"reference": reference, "arms": {}}
    for name, runs in rows.items():
        identical = all(r["fingerprints"] == base["fingerprints"] for r in runs)
        row = {"identical": identical, "hosts": sorted({r["host"] for r in runs}),
               "warm_median_s": [statistics.median(p["elapsed_s"] for p in r["passes"] if p["round"] >= 0) for r in runs],
               "preparation_s": [r["preparation_s"] for r in runs], "cold_s": [r["cold_s"] for r in runs]}
        row["median_of_runs_s"] = statistics.median(row["warm_median_s"])
        result["arms"][name] = row
    for row in result["arms"].values():
        row["speedup"] = result["arms"][reference]["median_of_runs_s"] / row["median_of_runs_s"]
    text = json.dumps(result, indent=2)
    Path(args.out).write_text(text + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("--outdir", required=True)
    one.add_argument("--inputs-root", required=True)
    one.add_argument("--arm", action="append", required=True, help="name[:VAR=VALUE,...]; the first is the reference")
    one.add_argument("--shards", nargs="+", type=int, default=[22])
    one.add_argument("--max-events", type=int, default=50000)
    one.add_argument("--seed", type=int, default=SEED)
    one.add_argument("--warmup", type=int, default=1)
    one.add_argument("--rounds", type=int, default=4)
    one.add_argument("--cpu", action="store_true", help="NumPy chain (the default chain without --jax)")
    one.add_argument("--prefetch", action="store_true")
    one.add_argument("--reco-cpu", action="store_true")
    one.set_defaults(function=run)
    pair = commands.add_parser("report")
    pair.add_argument("--reference", required=True)
    pair.add_argument("--arm", action="append", required=True, help="name=DIRECTORY of a finished run; repeat per process")
    pair.add_argument("--out", required=True)
    pair.set_defaults(function=report)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
