#!/usr/bin/env python3
"""Throughput of the fast calorimeter chain (steps -> clusters) on the current JAX device, warm, repeated."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from calofastsim import stockio
from calofastsim.chain import run_chain
from calofastsim.cluster import Geometry
from calofastsim.readout import ReadoutConfig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("steps", "geometry", "sim-crystal", "pulse", "noise", "propagation", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    events, timing, steps = stockio.load_steps(args.steps)
    sim = stockio.load_sim_crystal(args.sim_crystal)
    pulse, noise = stockio.load_pulse(args.pulse), stockio.load_noise(args.noise)
    table, geom = stockio.load_propagation(args.propagation), Geometry.load(args.geometry)

    def once(seed):
        t0 = time.perf_counter()
        out = run_chain(steps, timing, lambda s: stockio.adc_per_pe(sim, s), pulse, noise, geom,
                        np.random.default_rng(seed), ro_cfg=ReadoutConfig(), table=table, adc2mev=stockio.adc_to_mev)
        return time.perf_counter() - t0, int(out["clusters"]["event"].size), out["seconds"]

    first = once(0)[0]
    runs = [once(s + 1) for s in range(args.repeat)]
    warm = np.array([r[0] for r in runs])
    rep = {"events": int(events.size), "first_call_s": first, "warm_s": warm.tolist(),
           "events_per_s_warm_median": float(events.size / np.median(warm)),
           "device": str(jax.devices()[0]), "jax": jax.__version__, "host": platform.node(),
           "clusters": [r[1] for r in runs], "stage_seconds_last": runs[-1][2]}
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
