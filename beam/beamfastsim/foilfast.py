"""Single-foil response of the production layer model (transport._layer) on frozen entries, for G4 slab closure.

Each entry (kinetic energy, direction cosine to the foil normal, charge) crosses one aluminium foil of the given
thickness through exactly the code the transport uses for a foil hit (same Params, tables, sampler and Highland kick).
Field, trajectory and the foil-plane geometry are not involved. Output: exit kinetic energy (0 for a stop), the
true path used, and whether the layer stopped the muon (exit below transport.T_STOP).
"""
import numpy as np

import jax
import jax.numpy as jnp

from mu2efastsim import eloss, eloss_jax

from . import geometry, layerphys, transport

_CACHE = {}


def _kernel(prm):
    if prm not in _CACHE:
        al = eloss_jax.consts(geometry.STOPPING_TARGET_AL)
        x0 = eloss.radiation_length_mm(geometry.STOPPING_TARGET_AL)
        tbs = transport.layer_tables(prm)

        @jax.jit
        def run(key, p, cosine, q, thickness):
            n = p.shape[0]
            mc = eloss_jax.MatConsts(*(jnp.full(n, v) for v in al))
            u = jnp.stack([jnp.sqrt(1 - cosine ** 2), jnp.zeros(n), cosine], axis=1)
            z = thickness / cosine
            kind = jnp.where(q > 0, 1, 0)
            p2, _, stop = transport._layer(key, mc, jnp.full(n, x0), p, u, z, jnp.ones(n, bool), prm, tbs, kind)
            t = layerphys.true_path(tbs, kind, layerphys.kinetic(p), z) if (prm.mean == "range" and prm.msc and prm.msc_path) \
                else z
            return layerphys.kinetic(p2), stop, t

        _CACHE[prm] = run
    return _CACHE[prm]


def respond(energy, cosine, thickness, prm, *, charge=-1, replicas=1, seed=1, batch=1 << 18, raw=False):
    """Arrays (replicas, n): exit kinetic energy [MeV] (0 when stopped), stop flag, true path [mm].

    raw=True appends outgoing kinetic energy before T_STOP (zero from range exhaustion or sampled-loss clipping).
    The returned path is the requested effective path, not the consumed stopping path."""
    energy, cosine = np.broadcast_arrays(np.asarray(energy, float), np.asarray(cosine, float))
    thickness = np.broadcast_to(np.asarray(thickness, float), energy.shape)
    q = np.broadcast_to(np.asarray(charge, float), energy.shape)
    if energy.ndim != 1 or np.any(energy <= 0) or np.any((cosine <= 0) | (cosine > 1)) or np.any(thickness <= 0):
        raise ValueError("one-dimensional positive energies, cosines in (0,1] and thicknesses required")
    run = _kernel(prm)
    n = len(energy)
    e, c, t, qq = (np.tile(v, replicas) for v in (energy, cosine, thickness, q))
    p = np.sqrt(e * (e + 2 * eloss.MMU))
    out = [np.empty(len(e)), np.empty(len(e), bool), np.empty(len(e))]
    key = jax.random.key(seed)
    for i, s in enumerate(range(0, len(e), batch)):
        sl = slice(s, s + batch)
        m = len(e[sl])
        pad = (1 << max(int(np.ceil(np.log2(m))), 8)) - m           # power-of-two shapes: few compilations
        args = [np.concatenate([v[sl], np.full(pad, v[sl][0])]) for v in (p, c, qq, t)]
        r = run(jax.random.fold_in(key, i), *(jnp.asarray(a) for a in args))
        for o, v in zip(out, r):
            o[sl] = np.asarray(v)[:m]
    exit_t = np.where(out[1], 0.0, out[0])
    shape = (replicas, n)
    result = (exit_t.reshape(shape), out[1].reshape(shape), out[2].reshape(shape))
    return result + (out[0].reshape(shape),) if raw else result
