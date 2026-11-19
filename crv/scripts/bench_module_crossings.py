"""Paired CPU JAX crossing benchmark: public extracted or synthetic large geometry."""
import argparse
import json
import time

import jax
import numpy as np

from crvfastsim import crossings_jax as cj, geometry
from mu2efastsim import precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counters", help="Offline counters file; omit for synthetic 5,344-bar geometry")
    parser.add_argument("--tracks", type=int, default=1024)
    parser.add_argument("--slots", type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    if args.counters:
        geom = geometry.from_offline_config(geometry.find_counters(args.counters))
        center, half, sector, module = geom.center, geom.halflen, geom.sector, geom.module
    else:
        module = np.repeat(np.arange(167), 32)
        sector = np.zeros_like(module)
        center = np.column_stack((400 * (module % 17) + 10 * (np.arange(len(module)) % 32), 0 * module,
                                  400 * (module // 17)))
        half = np.broadcast_to([4, 10, 125], center.shape)
    lo, hi = center - half, center + half
    index = cj.module_index(lo, hi, sector, module)
    aim = center[rng.integers(len(center), size=args.tracks)]
    origin = aim + np.array([0, 1000, 0]) + rng.normal(0, 150, aim.shape)
    direction = aim + rng.normal(0, 30, aim.shape) - origin
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    o, d, lo, hi = map(jax.device_put, precision.dev((origin, direction, lo, hi)))
    ti, to = cj.slab(o, d, index.lo, index.hi)
    occupancy = np.asarray((to > np.maximum(ti, 0)).sum(axis=1))
    runs = {"dense": lambda: cj.crossings(o, d, lo, hi),
            "indexed": lambda: cj.crossings_by_module(o, d, lo, hi, index, module_slots=args.slots)}
    outputs = {}
    ms = {k: [] for k in runs}
    for arm, fn in runs.items():
        outputs[arm] = jax.device_get(fn())
    for _ in range(15):
        for arm, fn in runs.items():
            t = time.perf_counter()
            jax.block_until_ready(fn())
            ms[arm].append(1000 * (time.perf_counter() - t))
    a, b = outputs.values()
    for field in ("entry", "exit", "valid", "overflow"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(a.bar[a.valid], b.bar[b.valid])
    print(json.dumps(dict(bars=len(center), tracks=len(o), module_slots=args.slots, peak_modules=int(occupancy.max()),
                          boxes_tested_per_track_if_indexed=int(len(index.lo) + args.slots * index.bars.shape[1]),
                          crossings=int(a.valid.sum()), overflow=bool(a.overflow), exact=True,
                          median_ms={k: float(np.median(v)) for k, v in ms.items()}), indent=2))


if __name__ == "__main__":
    main()
