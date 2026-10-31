"""Measure exact complete-payload hashes at bounded worker counts."""
import argparse
import json
from pathlib import Path
import resource
import statistics
import time

from crvfastsim.analyses.context_audit import content_fingerprint
from crvfastsim.analyses.runchain import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.inputs_root)
    context = prepare(calib=str(root / "data/geometry/crvcalib_sim_best_v1_5_run1400.csv"),
                      deviation=str(root / "data/geometry/crvphoton_sim_best_v1_5_run1400.csv"),
                      gdml=str(root / "data/geometry/mu2e_extracted_v04.gdml"),
                      tier1b_set="strict_v4", tier1b_dir=str(root / "tier1b_strict_v2"))
    state = (context.signature, dict(context.payload), context.propagation,
             context.template, context.template_args)
    samples = {workers: [] for workers in (1, 2, 4, 8)}
    for _ in range(args.repeat):
        for workers in samples:
            start = time.perf_counter()
            actual = content_fingerprint(state, workers=workers)
            elapsed = time.perf_counter() - start
            if actual != context.fingerprint:
                raise ValueError("parallel hash differs from prepared serial digest")
            samples[workers].append(elapsed)
    output = {"exact_serial_digest": context.fingerprint, "all_identical": True,
              "preparation_timers": context.preparation_timers,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "seconds": samples, "median_seconds": {n: statistics.median(v) for n, v in samples.items()}}
    with open(args.out, "x") as stream:
        json.dump(output, stream, indent=2)


if __name__ == "__main__":
    main()
