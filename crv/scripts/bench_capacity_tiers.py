"""Paired CPU JAX benchmark for the optical arrival-delay kernel's capacity tiers."""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from crvfastsim.capacity import Capacities, FORMAT
from crvfastsim.response.photons_jax import _kernel_b


def main():
    rng = np.random.default_rng(42)
    demands = [1024, 2500, 1800, 3900, 7600, 4600, 13800, 3800, 1600, 25000, 2300, 63000] * 2
    rng.shuffle(demands)
    spec = dict(format=FORMAT, photon_arrivals=65536)
    fixed = Capacities(spec)
    tiers = Capacities(dict(spec, tiers={"photon_arrivals": [4096, 16384, 32768]}))
    cum = jax.device_put(np.cumsum(rng.integers(1, 16, (64, 32)), axis=1))
    dsum = cum[:, -1]
    sb = 8192
    gbin = jax.device_put(rng.integers(64, size=(sb, 3)))
    t = jnp.arange(sb) * 0.5
    ri = jnp.arange(sb)
    args = [(jax.device_put(rng.multinomial(total, np.full(sb * 3, 1 / (sb * 3))).reshape(sb, 3)),
             gbin, t, ri, jax.random.PRNGKey(i), cum, dsum) for i, total in enumerate(demands)]

    def run(caps):
        return [jax.block_until_ready(_kernel_b(*a, caps.size("photon_arrivals", n, 2**(n - 1).bit_length())))
                for a, n in zip(args, demands)]

    t0 = time.perf_counter()
    a = run(fixed)
    fixed_first = time.perf_counter() - t0
    jax.clear_caches()
    t0 = time.perf_counter()
    b = run(tiers)
    tiers_first = time.perf_counter() - t0
    for raw, tiered, n in zip(a, b, demands):
        for lhs, rhs in zip(raw, tiered):
            np.testing.assert_array_equal(lhs[:n], rhs[:n])
    run(fixed)
    warm = [[], []]
    for _ in range(9):
        for caps, times in zip((fixed, tiers), warm):
            t0 = time.perf_counter()
            run(caps)
            times.append(time.perf_counter() - t0)
    padded = [sum(caps.size("photon_arrivals", n, 65536) for n in demands) for caps in (fixed, tiers)]
    print(json.dumps(dict(arrivals=sum(demands), batches=len(demands), exact=True, padded_slots=padded,
                          tier_calls=tiers.report()["tier_calls"], first_s=[fixed_first, tiers_first],
                          median_s=[float(np.median(v)) for v in warm]), indent=2))


if __name__ == "__main__":
    main()
