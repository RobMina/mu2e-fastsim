"""JAX twin of sipm.simulate: SiPM charges on GPU, fixed shapes, one jitted scan over photon index.

Host: photons are sorted by (SiPM, time) and packed into a [S, K] matrix per bucket of SiPMs with K = max photons per
SiPM in the bucket (power of two); pixels are drawn from the photon map on the host (numpy, cheap). Device: a lax.scan
over k = 0..K-1 processes photon k of every SiPM at once; each step fires the primary pixel and then a fixed number of
breadth-first cross-talk generations (CT_GENERATIONS; a generation is ~4 % of the previous one, so 3 leave ~1e-4 of
cascades truncated relative to the numpy port, which runs them to the end). Each generation after the first keeps at
most CT_WIDTH candidate pixels per SiPM (the fired ones first): more than CT_WIDTH cross-talk hits from one generation
is ~(0.01)^CT_WIDTH per firing. Pixel state: last-discharge time per pixel,
[S, 41*41]. Outputs per (SiPM, k, generation, neighbour slot): fired flag, voltage; flattened on the host.
Thermal noise and trap afterpulses are not in the twin (both 0 in the production extracted configuration); it refuses
a configuration that has them.
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from .sipm import NPIX, draw_pixels

jax.config.update("jax_enable_x64", True)

CT_GENERATIONS = 3
CT_WIDTH = 4


def _bucket(n, lo=8):
    return max(lo, 1 << int(np.ceil(np.log2(max(n, 1)))))


@partial(jax.jit, static_argnames=("nx", "ny", "gens"))
def _scan(pix, t, valid, dead, key, vov, tau, p1, p2, q_ct, nx, ny, gens):
    """pix [S, K] int, t [S, K], valid [S, K]. Returns fired [K, G, S, 4or1...] flattened per generation and voltages."""
    S, K = pix.shape
    last0 = jnp.full((S, NPIX * NPIX), -jnp.inf)

    def pfire(v):
        return p1 * (1.0 - jnp.exp(-v / p2))

    rows = jnp.arange(S)

    def fire(last, p, tt, m, ok, k):
        v = vov * (1.0 - jnp.exp(-(tt - last[rows[:, None], p]) / tau))
        prob = 1.0 - (1.0 - pfire(v)) ** m
        f = ok & ~dead[p] & (jax.random.uniform(k, p.shape) < prob)
        last = last.at[rows[:, None], jnp.where(f, p, NPIX * NPIX - 1)].max(jnp.where(f, tt, -jnp.inf))
        return last, f, jnp.where(f, v, 0.0)

    def step(last, xs):
        p0, t0, ok0, k = xs
        ks = jax.random.split(k, 2 * gens + 1)
        p, tt = p0[:, None], t0[:, None]
        last, f, v = fire(last, p, tt, jnp.ones_like(p), ok0[:, None], ks[0])
        fired, volts = [f], [v]
        cur_p, cur_f = p, f
        for g in range(gens):
            x, y = cur_p // NPIX, cur_p % NPIX
            nxp = jnp.stack([x - 1, x + 1, x, x], -1)
            nyp = jnp.stack([y, y, y - 1, y + 1], -1)
            inb = jnp.stack([x > 0, x + 1 < nx, y > 0, y + 1 < ny], -1)
            go = cur_f[..., None] & inb & (jax.random.uniform(ks[2 * g + 1], inb.shape) < q_ct)
            npix = (nxp * NPIX + nyp).reshape(S, -1)
            go = go.reshape(S, -1)
            if npix.shape[1] > CT_WIDTH:  # compact: the fired candidates first, keep CT_WIDTH
                sel = jnp.argsort(~go, axis=1, stable=True)[:, :CT_WIDTH]
                npix = jnp.take_along_axis(npix, sel, 1)
                go = jnp.take_along_axis(go, sel, 1)
            # multiplicity of the same pixel within this generation (stock: processed one after another)
            same = (npix[:, :, None] == npix[:, None, :]) & go[:, None, :]
            mult = same.sum(-1)
            firstocc = jnp.argmax(same, axis=-1) == jnp.arange(npix.shape[1])[None, :]
            ok = go & firstocc
            last, f, v = fire(last, npix, jnp.broadcast_to(tt, npix.shape), mult, ok, ks[2 * g + 2])
            fired.append(f)
            volts.append(v)
            cur_p, cur_f = npix, f
        return last, (jnp.concatenate(fired, 1), jnp.concatenate(volts, 1))

    keys = jax.random.split(key, K)
    _, (fired, volts) = lax.scan(step, last0, (pix.T, t.T, valid.T, keys))
    return fired, volts  # [K, S, W]


def simulate(chan, time, photon, pmap, cfg, rng, key, t_end=1e5, max_sipms=1 << 15):
    """JAX equivalent of sipm.simulate (no thermal noise / traps). Returns the same dict."""
    if cfg.thermal_rate > 0 or cfg.trap0_prob > 0 or cfg.trap1_prob > 0:
        raise NotImplementedError("sipm_jax: thermal noise and trap afterpulses are not in the twin")
    chan, time, photon = np.asarray(chan), np.asarray(time, float), np.asarray(photon)
    keep = time <= t_end
    chan, time, photon = chan[keep], time[keep], photon[keep]
    px, py = draw_pixels(pmap, len(chan), rng)
    o = np.lexsort((time, chan))
    chan, time, photon, pix = chan[o], time[o], photon[o], (px * NPIX + py)[o]
    u, start, cnt = np.unique(chan, return_index=True, return_counts=True)
    order = np.argsort(-cnt, kind="stable")
    dead = np.zeros(NPIX * NPIX, bool)
    for x, y in cfg.inactive:
        dead[x * NPIX + y] = True
    dead[NPIX * NPIX - 1] = True  # the scatter sink for non-firing lanes (pixel (40, 40) never occurs)
    dead_d = jnp.asarray(dead)
    q_ct = (1.0 - (1.0 - cfg.cross_talk) ** 0.25) / cfg.p_fire(cfg.overvoltage)
    outs = []
    i = 0
    while i < len(order):
        K = _bucket(int(cnt[order[i]]))
        # SiPMs whose counts fit this K, up to max_sipms
        j = i
        while j < len(order) and j - i < max_sipms and _bucket(int(cnt[order[j]])) == K:
            j += 1
        sl = order[i:j]
        S = _bucket(len(sl), lo=256)
        P = np.full((S, K), NPIX * NPIX - 1, np.int64)
        T = np.zeros((S, K))
        V = np.zeros((S, K), bool)
        PH = np.full((S, K), -1, np.int64)
        r = np.repeat(np.arange(len(sl)), cnt[sl])
        k = np.arange(len(r)) - np.repeat(np.cumsum(cnt[sl]) - cnt[sl], cnt[sl])
        src = np.repeat(start[sl], cnt[sl]) + k
        P[r, k], T[r, k], V[r, k], PH[r, k] = pix[src], time[src], True, photon[src]
        key, sub = jax.random.split(key)
        fired, volts = _scan(jnp.asarray(P), jnp.asarray(T), jnp.asarray(V), dead_d, sub, cfg.overvoltage,
                             cfg.time_constant, cfg.p1, cfg.p2, q_ct, cfg.n_pixels_x, cfg.n_pixels_y, CT_GENERATIONS)
        fired, volts = np.asarray(fired), np.asarray(volts)
        kk, ss, ww = np.nonzero(fired)
        v = volts[kk, ss, ww]
        outs.append({"chan": u[sl][ss], "time": T[ss, kk], "charge": cfg.capacitance * v, "pe": v / cfg.overvoltage,
                     "photon": PH[ss, kk]})
        i = j
    if not outs:
        return {k: np.zeros(0) for k in ("chan", "time", "charge", "pe", "photon")}
    out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    o = np.lexsort((out["time"], out["chan"]))
    return {k: a[o] for k, a in out.items()}
