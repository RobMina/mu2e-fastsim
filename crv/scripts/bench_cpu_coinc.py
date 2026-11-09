"""Replay retained full-chain pulses through both CPU finders with byte-exact row checks."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from crvfastsim import runconfig
from crvfastsim.reco import coinc as C


def digest(rows):
    state = hashlib.sha256()
    for row in rows:
        for key, value in row.items():
            a = np.asarray(value)
            state.update(key.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return state.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--calib", help="conditions CSV for retained NPZ inputs; defaults to CRVFS_DATA/geometry calibration")
    args = parser.parse_args()
    rc = runconfig.get(runconfig.DEFAULT)
    G = C.sector_geom(rc.geometry(), rc.counters_cfg())
    cfg = rc.finder()
    from crvfastsim.chain import Conditions
    reports = []
    for path in args.inputs:
        if Path(path).suffix == ".root":
            from crvfastsim.reco.cli import _ref_pulses_sorted
            from crvfastsim.reco import reftables
            p = _ref_pulses_sorted([path])
            nevents = len(reftables.read([path], "events", ["event"]))
            status = reftables.calib([path])["status"]
        else:
            calib = args.calib or str(Path(os.environ["CRVFS_DATA"]) / "geometry/crvcalib_sim_best_v1_5_run1400.csv")
            status = Conditions.from_csv(calib).status
            with np.load(path, allow_pickle=False) as saved:
                p = {k: saved["chunk_0000__pulses__" + k] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")}
                nevents = len(saved["generated_keys"])
        events, lo = np.unique(p["ev"], return_index=True)
        hi = np.r_[lo[1:], len(p["ev"])]
        groups = [{k: v[a:b] for k, v in p.items() if k != "ev"} for a, b in zip(lo, hi)]
        timing = {"reference": [], "numpy": []}
        expected = None
        nclusters = None
        for rep in range(args.repeat):
            for mode in (("reference", "numpy") if rep % 2 == 0 else ("numpy", "reference")):
                os.environ["CRVFS_CPU_ATTACHMENT"] = mode
                start = time.perf_counter()
                rows = [row for group in groups for row in C.find_event(group, G, status, cfg)]
                elapsed = time.perf_counter() - start
                actual = digest(rows)
                if expected is None:
                    expected, nclusters = actual, len(rows)
                if actual != expected:
                    raise ValueError(f"CPU finder exact-output mismatch: {mode}")
                timing[mode].append(elapsed)
        medians = {k: statistics.median(v) for k, v in timing.items()}
        with open(path, "rb") as stream:
            input_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        reports.append({"input": path, "input_sha256": input_digest,
                        "generated_events": nevents, "events_with_pulses": len(events), "pulses": len(p["ev"]),
                        "clusters": nclusters, "row_fingerprint": expected, "exact_all_repeats": True,
                        "seconds": timing, "median_seconds": medians, "speedup": medians["reference"] / medians["numpy"],
                        "scope": "CPU coincidence finder only; retained inputs/geometry/status setup and fingerprint outside timing; not stock art"})
        print(json.dumps(reports[-1]), flush=True)
    with open(args.out, "x") as stream:
        json.dump(reports, stream, indent=2)


if __name__ == "__main__":
    main()
