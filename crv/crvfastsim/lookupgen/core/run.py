"""Host driver: run the transport kernel over a set of bins in chunks of iterations.

Tail compaction: once the queue is empty, the remaining long chains (photons trapped in the fiber claddings travel
the whole counter bounce by bounce) would keep every lane iterating. When fewer than 1/16 of the lanes are alive,
they are moved into a kernel with 1/16 of the lanes (same bins, same random streams: a lane's randoms depend only on
its photon and step).
"""
import time

import numpy as np

import jax
import jax.numpy as jnp

from . import transport as T


def _compact(s, small):
    alive = np.nonzero(np.asarray(s["reg"]) != T.DEAD)[0]
    out = {}
    for k, v in s.items():
        a = np.asarray(v)
        b = np.zeros((small,) + a.shape[1:], a.dtype)
        b[:len(alive)] = a[alive]
        if k == "reg":
            b[len(alive):] = T.DEAD
        out[k] = jnp.asarray(b)
    return out


def device():
    """The device the kernel runs on, for the run stats: {"device": kind (e.g. "NVIDIA A100-SXM4-80GB", "cpu"),
    "platform": "gpu" / "cpu"}. Tables made on different devices agree within statistics; the record says which."""
    d = jax.devices()[0]
    return {"device": str(d.device_kind), "platform": str(d.platform)}


def generate(counter, binning, simtype, bins, nphot, dtype="float64", seed=1, lanes=1 << 16, chunk_iters=2000,
             refill_fraction=0.125, log=None, helix=True, compact=True, checkpoint=None,
             rough_reflection="stale", model=None, max_steps=None):
    """Counts for `bins` (numbers of `binning`) of `simtype`: dict(bins, gen, nall, hta, hea, ht, he, nphot, simtype,
    complete, stats)."""
    def kernel(nl):
        cfg = T.Config(counter=counter, binning=binning, simtype=simtype, dtype=dtype, seed=seed, lanes=nl,
                       refill_fraction=refill_fraction, helix=helix,
                       rough_reflection=rough_reflection, model=model or T.M.DEFAULT_MODEL,
                       max_steps=max_steps or (1 << T.STEP_BITS))
        return T.Kernel(cfg).build(bins, nphot)

    run, init, done = kernel(lanes)
    s, aux, acc = init()
    t0 = time.time()
    t_first, t_compact, n_calls, cur = None, None, 0, lanes
    while True:
        s, aux, acc = run(s, aux, acc, chunk_iters)
        aux["q"].block_until_ready()
        n_calls += 1
        if t_first is None:
            t_first = time.time() - t0
        nalive = int(jnp.sum(s["reg"] != T.DEAD))
        if log:
            log(f"call {n_calls}: queued {int(aux['q'])}/{len(bins) * nphot}, alive {nalive}/{cur}, "
                f"steps {int(aux['steps'])}, {time.time() - t0:.1f} s")
        if checkpoint is not None:
            checkpoint(_result(bins, nphot, simtype, s, aux, acc, {"partial": True}))
        if done(s, aux):
            break
        small = cur // 16
        if compact and int(aux["q"]) >= len(bins) * nphot and nalive < small and small >= 64:
            run, _, done = kernel(small)
            s = _compact(s, small)
            cur = small
            t_compact = time.time() - t0 if t_compact is None else t_compact
    wall = time.time() - t0
    return _result(bins, nphot, simtype, s, aux, acc,
                   dict(wall_s=wall, first_call_s=t_first, compact_s=t_compact, calls=n_calls, lanes=lanes,
                        dtype=dtype, seed=int(seed), photons=int(len(bins) * nphot),
                        model=model or T.M.DEFAULT_MODEL,
                        max_steps=max_steps or (1 << T.STEP_BITS), **device()))


def _result(bins, nphot, simtype, s, aux, acc, extra):
    """Counts and stats; ``complete`` marks the bins whose photons are all queued and finished."""
    stats = {k_: (np.asarray(v).tolist() if np.ndim(v) else int(v)) for k_, v in aux.items()}
    stats.update(extra)
    nb = len(bins)
    queued = np.minimum(np.maximum(int(aux["q"]) - np.arange(nb) * nphot, 0), nphot) == nphot
    busy = np.zeros(nb, bool)
    lb = np.asarray(s["lbin"])[np.asarray(s["reg"]) != T.DEAD]
    busy[np.unique(lb)] = True
    out = {k: np.asarray(v) for k, v in acc.items()}
    out.update(bins=np.asarray(bins), nphot=int(nphot), simtype=int(simtype), stats=stats, complete=queued & ~busy)
    return out
