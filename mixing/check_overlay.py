"""Check sampled detector-step assembly on saved donors without redigitising."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np

from .frames import FrameBatch, overlay, read_frames, read_event_log
from .inspect import summarize
from .sampling import select_frames


def diagnostic(pool, count, seed=1):
    """Draw a synthetic multiplicity schedule; this is not a physical pileup rate."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=count)
    selection, _ = select_frames(rng, counts, len(pool.event_ids), mode="randomReplace")
    ids = np.column_stack((np.ones(count, dtype=np.uint32), np.ones(count, dtype=np.uint32),
                           np.arange(count, dtype=np.uint32)))
    empty = {name: ak.unflatten(ak.flatten(rows[:0]), np.zeros(count, dtype=np.int64))
             for name, rows in pool.collections.items()}
    result = overlay(FrameBatch(ids, empty), [(pool, selection)])
    for name, rows in pool.collections.items():
        expected = int(ak.sum(ak.num(rows[selection.donors])))
        if int(ak.sum(ak.num(result.collections[name]))) != expected:
            raise AssertionError(f"{name} rows changed during assembly")
    summary = summarize(result)
    summary.update({"diagnostic_only": True, "synthetic_mean_frames": 2.0,
                    "sampled_frames": int(counts.sum()), "seed": seed})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    pool = read_frames(args.input, event_ids=read_event_log(args.event_log))
    report = diagnostic(pool, args.events, args.seed)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
