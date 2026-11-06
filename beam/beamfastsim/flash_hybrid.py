"""Opt-in compact CPU vacuum bursts with canonical full-shape material steps on the GPU."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import cechain as C

HOT = ("x", "u", "T", "t", "s", "fate", "inl", "nvac", "nstall", "nadapt")


def make_runner(run, maps, key, initial, m_cand, m_slab, bucket, diagnostics, adaptive=False):
    """Compile the device slow path and a compact host vacuum kernel without changing RNG calls."""
    cpu = jax.devices("cpu")[0]
    gpu = initial["T"].device
    start = time.perf_counter()
    slow = run.lower(maps, key, initial, jnp.int32(1), m_cand, m_slab, bucket).compile()
    diagnostics["slow_compile_seconds"] = time.perf_counter() - start
    cpu_maps = jax.device_put(maps, cpu)
    n = initial["T"].shape[0]
    sizes = sorted(set([bucket] + ([v for v in (1, 4, 16, 64) if v < bucket] if adaptive else [])))
    bursts = {}
    start = time.perf_counter()
    with jax.default_device(cpu):
        for size in sizes:
            template = {k: jax.device_put(np.repeat(np.asarray(initial[k])[:1], size, axis=0), cpu) for k in HOT}
            template["fate"] = jnp.full(size, C.PLANE, jnp.int32)
            bursts[size] = run.vacuum_burst.lower(cpu_maps, template, jnp.int32(1)).compile()
    diagnostics["vacuum_compile_seconds"] = time.perf_counter() - start
    diagnostics["bucket_sizes"] = sizes
    diagnostics.update(vacuum_seconds=0., device_seconds=0., transfer_seconds=0., bursts=0,
                       vacuum_iterations=0, device_iterations=0)

    @jax.jit
    def gather(st, idx):
        return {k: st[k][idx] for k in HOT}

    @jax.jit
    def scatter(st, idx, hot, used):
        updates = {k: st[k].at[idx].set(hot[k], mode="drop") for k in HOT}
        sl = dict(st["slab"], it=st["slab"]["it"] + used,
                  exited=st["slab"]["exited"] | ((used > 0) & ~st["inl"]))
        return dict(st, **updates, slab=sl)

    def execute(maps, key, st, limit):
        total, limit = 0, int(limit)
        while total < limit:
            live = np.asarray(st["fate"]) == C.LIVE
            if not live.any():
                break
            inl = np.asarray(st["inl"])
            ids = np.flatnonzero(live)
            blocked = True
            if len(ids) <= bucket and not np.any(live & inl):
                start = time.perf_counter()
                size = next(v for v in sizes if v >= len(ids))
                padded = np.pad(ids, (0, size - len(ids)), constant_values=n)
                hot = jax.tree_util.tree_map(np.asarray, gather(st, jax.device_put(np.minimum(padded, n - 1), gpu)))
                hot["fate"] = np.where(padded < n, hot["fate"], C.PLANE).astype(np.int32)
                hot = jax.device_put(hot, cpu)
                diagnostics["transfer_seconds"] += time.perf_counter() - start
                start = time.perf_counter()
                used, hot, blocked = jax.block_until_ready(bursts[size](cpu_maps, hot, jax.device_put(np.int32(limit - total), cpu)))
                diagnostics["vacuum_seconds"] += time.perf_counter() - start
                used, blocked = int(used), bool(blocked)
                diagnostics["bursts"] += 1
                diagnostics["vacuum_iterations"] += used
                start = time.perf_counter()
                st = jax.block_until_ready(scatter(st, jax.device_put(padded, gpu), jax.device_put(hot, gpu),
                                                   jax.device_put(np.int32(used), gpu)))
                diagnostics["transfer_seconds"] += time.perf_counter() - start
                total += used
            if blocked and total < limit:
                start = time.perf_counter()
                used, st = jax.block_until_ready(slow(maps, key, st, jax.device_put(np.int32(limit - total), gpu)))
                diagnostics["device_seconds"] += time.perf_counter() - start
                diagnostics["device_iterations"] += int(used)
                total += int(used)
        return jax.device_put(np.int32(total), gpu), st
    return execute
