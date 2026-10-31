"""Measure CPU pulse reconstruction on retained stock digis and validate exact workspace reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

import numpy as np

from crvfastsim.reco import pulsefit as PF
from crvfastsim.reco.cli import _prepare
from crvfastsim.reco import reftables


def fingerprint(result):
    h = hashlib.sha256()
    for key, value in sorted(result.items()):
        a = np.asarray(value)
        h.update(key.encode() + a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--stock-log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    cfg = PF.RecoConfig()
    W = _prepare([args.input], None, cfg)
    events = len(reftables.read([args.input], "events", ["event"]))
    modes = {"reference": False, "workspace": True, "normal_cache": True}
    timings = {mode: [] for mode in modes}
    expected = None
    count = None
    for rep in range(args.repeat):
        for mode in (list(modes) if rep % 2 == 0 else list(reversed(modes))):
            os.environ["CRVFS_CPU_FIT_WORKSPACE"] = str(int(modes[mode]))
            os.environ["CRVFS_CPU_FIT_NORMAL_CACHE"] = str(int(mode == "normal_cache"))
            start = time.perf_counter()
            result = PF.reconstruct_full(W["mat"], W["n"], W["start"], W["ped"], W["area"], W["height"], cfg)
            elapsed = time.perf_counter() - start
            actual = fingerprint(result)
            if expected is None:
                expected, count = actual, len(result["wf"])
            if actual != expected:
                raise ValueError("CPU pulse workspace changed outputs")
            timings[mode].append(elapsed)
            print(mode, rep, elapsed, actual, flush=True)
    stock = {}
    pattern = re.compile(r"^DigitizePath:(CrvRecoPulses|CrvCoincidenceClusterFinder):\S+\s+(.*)$")
    for line in Path(args.stock_log).read_text().splitlines():
        match = pattern.match(line)
        if match:
            values = match[2].split()
            n = int(values[-1])
            if n != events:
                raise ValueError("stock timing event count differs from dump")
            stock.setdefault(match[1], []).append({"mean_s": float(values[1]), "events": n,
                                                   "total_module_s": float(values[1]) * n})
    if set(stock) != {"CrvRecoPulses", "CrvCoincidenceClusterFinder"}:
        raise ValueError("stock TimeTracker rows missing")
    medians = {k: statistics.median(v) for k, v in timings.items()}
    with open(args.input, "rb") as stream:
        input_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"input": args.input, "input_sha256": input_digest, "stock_log": args.stock_log,
              "stock_log_sha256": hashlib.sha256(Path(args.stock_log).read_bytes()).hexdigest(), "events": events,
              "waveforms": len(W["n"]), "pulses": count, "exact_all_repeats": True,
              "fingerprint": expected, "seconds": timings, "medians_s": medians,
              "workspace_speedups": {mode: medians["reference"] / value for mode, value in medians.items()},
              "stock_module_observations": stock,
              "stock_pulse_total_s_median": statistics.median(q["total_module_s"] for q in stock["CrvRecoPulses"]),
              "scope": "identical retained stock digis; one CPU core each, different UVA CPU hosts/allocations; setup/IO outside fast timing; stock TimeTracker excludes framework overhead; not literal stock-float equality"}
    report["speedups_vs_saved_stock_pulse_module"] = {mode: report["stock_pulse_total_s_median"] / value
                                                     for mode, value in medians.items()}
    with open(args.out, "x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
