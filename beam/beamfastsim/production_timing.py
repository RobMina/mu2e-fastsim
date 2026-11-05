"""Summarize existing Offline production logs without conflating transport and whole-job rates."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def read(path):
    text = Path(path).read_text(errors="replace")
    events = re.findall(r"TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)", text)
    times = re.findall(r"TimeReport CPU\s*=\s*([\d.eE+-]+)\s+Real\s*=\s*([\d.eE+-]+)", text)
    module = re.findall(r"^\s*\S*:g4run:Mu2eG4\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                        r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)\s*$", text, re.M)
    if len(events) != 1 or len(times) != 1 or len(module) != 1:
        raise ValueError(f"{path}: expected exactly one completed job, trigger summary and G4 timing row")
    n, kept, failed = map(int, events[0])
    cpu, wall = map(float, times[0])
    g4_ms = 1000 * float(module[0][1])
    g4_events = int(module[0][-1])
    if min(n, cpu, wall, g4_ms) <= 0 or kept + failed != n or not 0 < g4_events <= n:
        raise ValueError(f"{path}: inconsistent event counts or timing")
    return dict(file=Path(path).name, generated_events=n, kept_events=kept, g4_events=g4_events,
                g4_ms_per_g4_event=g4_ms, g4_s_per_generated=g4_ms / 1000 * g4_events / n,
                cpu_s=cpu, wall_s=wall, cpu_ms_per_generated=1000 * cpu / n,
                wall_ms_per_generated=1000 * wall / n)


def summarize(paths):
    rows = [read(p) for p in paths]
    n = sum(r["generated_events"] for r in rows)
    ms = np.array([r["g4_ms_per_g4_event"] for r in rows])
    wall = sum(r["wall_s"] for r in rows)
    g4_s = sum(r["g4_s_per_generated"] * r["generated_events"] for r in rows)
    return dict(files=len(rows), generated_events=n, kept_events=sum(r["kept_events"] for r in rows),
                g4_ms_per_generated=1000 * g4_s / n, g4_ms_per_g4_event_range=[float(ms.min()), float(ms.max())],
                whole_art_wall_ms_per_generated=1000 * wall / n,
                stock_g4_gen_per_s=n / g4_s, stock_whole_art_gen_per_s=n / wall,
                scope="G4 all detectors/secondaries; whole-art includes setup and output, not input staging or queue",
                rows=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = summarize(a.logs)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
