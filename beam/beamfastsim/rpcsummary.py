"""Bundle the saved RPC comparisons and measured transport timing for a deck.

The timing comparison is stage-limited: fast primary-electron transport versus
stock all-particle generation/transport and output. It is not a full-chain
replacement factor. Inputs and raw results stay outside the source repository.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np


def stock_timing(path):
    """Read the completed stock art summary and its generated event denominator."""
    text = Path(path).read_text(errors="replace")
    events = re.findall(r"TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)", text)
    times = re.findall(r"TimeReport CPU\s*=\s*([\d.eE+-]+)\s+Real\s*=\s*([\d.eE+-]+)", text)
    module = re.findall(r"^\s*\S*:g4run:Mu2eG4\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                        r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)\s*$", text, re.M)
    if len(events) != 1 or len(times) != 1 or len(module) != 1:
        raise ValueError("need one completed stock job with trigger and G4 timing summaries")
    n, kept, failed = map(int, events[0])
    cpu, wall = map(float, times[0])
    g4_n = int(module[0][-1])
    g4_wall = float(module[0][1]) * g4_n
    if n <= 0 or n != kept + failed or wall <= 0 or g4_wall <= 0:
        raise ValueError("invalid stock exposure or times")
    return dict(generated=n, kept=kept, cpu_s=cpu, art_wall_s=wall,
                g4_events=g4_n, g4_wall_s=g4_wall, art_events_s=n / wall, g4_events_s=n / g4_wall)


def bundle(results, source, plane, starts, log, outdir):
    results, outdir = Path(results), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reports = {name: json.loads((results / (name + "_final.json")).read_text())
               for name in ("paired_dev", "paired_heldout", "generated_dev", "generated_heldout")}
    with np.load(plane, allow_pickle=False) as p, np.load(starts, allow_pickle=False) as s:
        n, wall = len(p["idx"]), float(p["wall"])
        if not np.array_equal(np.sort(p["idx"]), np.arange(len(s["pdg"]))):
            raise ValueError("transport result must retain every generated electron")
        if wall <= 0:
            raise ValueError("positive measured transport time required")
        source_wall = float(s["generator_wall_s"])
        fates = {str(int(k)): int(v) for k, v in zip(*np.unique(p["fate"], return_counts=True))}
    stock = stock_timing(log)
    rate = n / wall
    report = dict(source=json.loads(Path(source).read_text()), vd13=reports,
                  throughput=dict(electrons=n, transport_wall_s=wall, transport_events_s=rate,
                                  generator_wall_s=source_wall, source_plus_transport_events_s=n / (wall + source_wall),
                                  fates=fates, stock=stock,
                                  rate_ratio_vs_stock_art=rate / stock["art_events_s"],
                                  rate_ratio_vs_stock_g4=rate / stock["g4_events_s"],
                                  device="RTX PRO 6000 MIG 1g.24gb, four host CPUs",
                                  timing_scope="one million-electron call including map loading and JIT; excludes input/output; not a steady-state benchmark",
                                  comparison_scope="primary electron to tracker front versus stock pair/all-secondaries/all-detectors; not a full-chain replacement factor"))
    (outdir / "numbers.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(results / "generated_heldout_final.png", outdir / "rpc_vd13.png")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--source", required=True, help="rpccompare source-only numbers.json")
    ap.add_argument("--plane", required=True)
    ap.add_argument("--starts", required=True)
    ap.add_argument("--stock-log", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    report = bundle(a.results, a.source, a.plane, a.starts, a.stock_log, a.outdir)
    print(json.dumps(report["throughput"], indent=2))


if __name__ == "__main__":
    main()
