"""Host stages (transport and tier 1b, NumPy) of runchain in worker processes: runchain --host-workers N.

The NumPy transport and tier 1b are Python-heavy, so in the prefetch thread they hold the GIL against the device
stages of the main thread (a second thread makes both slower). Worker processes run them beside the main process
instead. Each file keeps its own generator, default_rng([seed, file index]), and the workers run the same function as
the in-process path (numpy_host_stage), so the steps, the particles and the generator state handed back are the same
whichever process made them. The workers get the prepared context (geometry, world, absorbers, library) once, when
they start; they use JAX on the CPU only (the slab crossings of the transport), never the accelerator.
"""
import atexit
import os
import pickle
import time

import numpy as np

_CTX = None
_POOLS = {}


def numpy_host_stage(gen, rng, ctx, timers):
    """Transport (stage C) and tier 1b of one file's generated events with the NumPy backends. ctx: the prepared
    context (geom, world, ab, tagger, bar_cut, below, range_out, below_msc, lib, tier1b_options)."""
    from . import tier1b, transport as tr
    a = time.perf_counter()
    st, pa = tr.transport(gen, ctx["geom"], rng, stage="C", world=ctx["world"], absorbers=ctx["ab"],
                          tagger_bar=ctx["tagger"], bar_cut=ctx["bar_cut"], below=ctx["below"],
                          range_out=ctx["range_out"], below_msc=ctx["below_msc"])
    timers["transport_s"] = time.perf_counter() - a
    a = time.perf_counter()
    st, pa, _ = tier1b.sample(st, pa, gen, ctx["lib"], ctx["geom"], rng, **ctx["tier1b_options"])
    timers["tier1b_s"] = time.perf_counter() - a
    return st, pa


def _init(blob):
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_platforms", "cpu")
    global _CTX
    _CTX = pickle.loads(blob)


def _ready(_):
    return os.getpid()


def task(i, path, gen, seed, max_events):
    """Host stage of file i in a worker: (generated events, steps, particles, generator, timers)."""
    from . import tables
    rng = np.random.default_rng([int(seed), i])
    t = {}
    if gen is None:
        gen = tables.read(path, "gen")
    if max_events:
        keep = gen.event.isin(np.sort(gen.event.unique())[:int(max_events)])
        gen = gen[keep].reset_index(drop=True)
    ngen = int(gen[tables.KEY].drop_duplicates().shape[0])
    st, pa = numpy_host_stage(gen, rng, _CTX, t)
    return ngen, st, pa, rng, t


def pool(ctx, workers, key):
    """A process pool of `workers` started with ctx (kept per key: the passes of one prepared context share it)."""
    p = _POOLS.get(key)
    if p is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        p = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                initargs=(pickle.dumps(ctx, protocol=5),))
        list(p.map(_ready, range(workers)))
        _POOLS[key] = p
        atexit.register(p.shutdown)
    return p
