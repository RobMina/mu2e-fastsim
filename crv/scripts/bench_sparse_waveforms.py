"""Paired capacity-mode waveform benchmark using synthetic charges and the Offline template."""
import argparse
import json
import time

import numpy as np

from crvfastsim.capacity import Capacities, FORMAT
from crvfastsim.reco import cli, repro, waveform, waveform_jax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=150)
    parser.add_argument("--long", action="store_true", help="Spread each channel's charges across a long cluster")
    args = parser.parse_args()
    row, t, q, phase = repro._charges(args.events, seed=7)
    if args.long:
        first = np.searchsorted(row, row, side="left")
        t += 70 * ((np.arange(len(t)) - first) // 8)
        q *= 2  # Each spaced bunch is still above zero-suppression threshold.
    charges = dict(row=row, time=t, charge=q)
    cfg = waveform.WaveformConfig()
    caps = {k: Capacities(dict(format=FORMAT, waveform_kernel=k)) for k in ("dense", "scatter")}

    def run(k):
        return cli._synth_jax_fixed(waveform_jax, charges, phase, cfg, caps[k], noise=cfg.noise, seed=12)

    warm = {k: [] for k in caps}
    cold = {}
    results = {}
    for k in caps:
        t0 = time.perf_counter()
        results[k] = run(k)
        cold[k] = 1000 * (time.perf_counter() - t0)
    for _ in range(10):
        for k in caps:
            t0 = time.perf_counter()
            run(k)
            warm[k].append(1000 * (time.perf_counter() - t0))
    dense, scatter = results.values()
    if not len(dense["chan"]):
        raise RuntimeError("benchmark must include nonempty digis")
    for field in ("chan", "starttdc"):
        np.testing.assert_array_equal(dense[field], scatter[field])
    print(json.dumps(dict(long=args.long, charges=len(q), digis=len(dense["chan"]),
                          adc_mismatches=int(np.count_nonzero(dense["adc"] != scatter["adc"])),
                          max_abs_voltage_diff=float(np.max(np.abs(dense["volt"] - scatter["volt"]))),
                          first_ms=cold, median_ms={k: float(np.median(v)) for k, v in warm.items()}), indent=2))


if __name__ == "__main__":
    main()
