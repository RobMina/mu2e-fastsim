"""Synchronized warm transport profile on an existing held-out generated shard.

Use --cuda-capture under Nsight Systems with --capture-range=cudaProfilerApi.
The capture excludes compilation. Component microbenchmarks are separate calls,
not additive estimates of the fused transport wall time.
"""
import argparse
import cProfile
import ctypes
import ctypes.util
import glob
import json
from pathlib import Path
import pstats
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from crvfastsim import geometry, tables, transport as nt, transport_jax as jt
from crvfastsim import crossings_jax as jc


def cudart():
    candidates = [ctypes.util.find_library("cudart"), "libcudart.so.12"]
    candidates += [p for root in sys.path for p in glob.glob(str(Path(root)/"nvidia/cuda_runtime/lib/libcudart.so*"))]
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass
    raise RuntimeError("CUDA runtime unavailable for explicit warm capture")


def timed(fn, repeat):
    a = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    cold = time.perf_counter() - a
    values = []
    for _ in range(repeat):
        a = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        values.append(time.perf_counter() - a)
    return dict(first_s=cold, warm_s=values), out


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    geom = geometry.from_offline_config(args.counters)
    world, ab = nt.load_world(args.gdml)
    dev = jt.DeviceTransport(geom, world, ab, batch_size=args.batch_size)
    gen = tables.read(args.gen, "gen")
    result = dict(devices=[str(d) for d in jax.devices()], batch_size=dev.batch_size,
                  primary_muons=len(nt.all_primary_muons(gen)), bars=geom.nbars,
                  band_pieces=[len(b[0][1]) for b in dev.bands],
                  band_materials=[list(b[1]) for b in dev.bands],
                  note="Uninstrumented repeats separate from warm CUDA capture; component calls do not sum to fused time.")
    def full():
        timers = {}
        a = time.perf_counter()
        steps, parts = jt.transport(gen, geom, np.random.default_rng([100022, 0]), stage="C", prepared=dev, timers=timers)
        return dict(total_s=time.perf_counter()-a, steps=len(steps), particles=len(parts), **timers)
    result["cold"] = full()
    result["warm"] = [full() for _ in range(args.repeat)]
    (out/"profile.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if args.cuda_capture:
        rt = cudart()
        if rt.cudaProfilerStart() != 0:
            raise RuntimeError("cudaProfilerStart failed")
        try:
            result["instrumented"] = [full() for _ in range(2)]
        finally:
            if rt.cudaProfilerStop() != 0:
                raise RuntimeError("cudaProfilerStop failed")
    prof = cProfile.Profile()
    prof.enable()
    full()
    prof.disable()
    with (out/"host_profile.txt").open("w") as stream:
        pstats.Stats(prof, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(60)
    if args.full_only:
        (out/"profile.json").write_text(json.dumps(result, indent=2))
        return result
    mu = nt.all_primary_muons(gen)
    o = mu[["x", "y", "z"]].to_numpy()
    mom = mu[["px", "py", "pz"]].to_numpy()
    p = np.linalg.norm(mom, axis=1)
    d = mom/p[:, None]
    near = nt.near_crv(o, d, geom, y_plane=dev.y)
    ti, to = nt.crossings.slab_crossings(o, d, (geom.center-geom.halflen).min(0)[None],
                                      (geom.center+geom.halflen).max(0)[None], backend="numpy")
    selected = np.flatnonzero(near | (to[:, 0] > np.maximum(ti[:, 0], 0.)))
    result["selected_muons"] = len(selected)
    if len(selected):
        ix = selected[:dev.batch_size]
        n = len(ix)
        op = np.zeros((dev.batch_size, 3)); op[:n] = o[ix]
        dp = np.zeros_like(op); dp[:, 1] = -1.; dp[:n] = d[ix]
        pp = np.ones(dev.batch_size); pp[:n] = p[ix]
        oj, dj, pj = map(jnp.asarray, (op, dp, pp))
        active = jnp.arange(dev.batch_size) < n
        nearj = jnp.asarray(np.r_[near[ix], np.zeros(dev.batch_size-n, bool)])
        key = jax.random.key(22)
        result["world_component"], wr = timed(jax.jit(lambda: dev.world(key, oj, dj, pj, nearj)), args.repeat)
        pm, alive, xx, bh = wr
        os, ds, ls = jt.scatter(key, oj, dj, jnp.maximum(jnp.sqrt(pj*jnp.maximum(pm, 1.)), 1.), dev.y, xx, bh)
        os, ds = jnp.where(nearj[:, None], os, oj), jnp.where(nearj[:, None], ds, dj)
        pm = jnp.where(nearj, jnp.where(alive, pm, 0.), pj)
        result["crossings_component"], xs = timed(lambda: jc.crossings(os, ds, dev.lo, dev.hi, dev.capacity, active), args.repeat)
        result["bar_component"], bars = timed(lambda: dev.bar_kernel(key, os, ds, pm, ls, active), args.repeat)
        result["first_batch_crossing_counts"] = np.asarray(xs.valid.sum(1)).tolist()
    (out/"profile.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen", required=True)
    p.add_argument("--gdml", required=True)
    p.add_argument("--counters", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--cuda-capture", action="store_true")
    p.add_argument("--full-only", action="store_true", help="skip independent component compilation")
    run(p.parse_args())


if __name__ == "__main__":
    main()
