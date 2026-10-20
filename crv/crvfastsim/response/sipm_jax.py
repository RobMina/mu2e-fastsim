"""JAX twin of sipm.simulate, per-pixel streams: SiPM charges on GPU with a short scan.

A pixel's response depends only on the earlier hits on the same pixel (recovery) plus the cross-talk it receives, so:
  1. hits = photons (pixel drawn from the photon map) plus cross-talk *candidates*: each hit proposes each existing
     4-neighbour with probability q_ct, drawn once (3 generations: candidates of candidates); a candidate counts only
     if its parent hit is active and fires;
  2. all hits are sorted by (SiPM, pixel, time) on the device; the scan runs over the position inside a pixel stream
     (a few steps: ~130 photons spread over ~600 pixels), vectorised over all streams, with the pixel's last-discharge
     time as the carry; same-time hits on one pixel are handled one after the other, as the stock code does;
  3. the active set is solved by a fixed point: active(candidate) = active(parent) and fired(parent); fired depends on
     the active hits before it on its pixel; iterated until stable (at most generations + 1 rounds).
Every hit carries its own pre-drawn uniform, so the fixed point is deterministic. The previous twin scanned photon by
photon (13 k SiPMs/s per A100, slower than the numpy port). Thermal noise and traps are not in the twin (0 in the
production extracted configuration).
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from .sipm import NPIX

jax.config.update("jax_enable_x64", True)

GENERATIONS = 3
NP2 = NPIX * NPIX


def _bucket(n, lo=1 << 12):
    return max(lo, 1 << int(np.ceil(np.log2(max(n, 1)))))


@partial(jax.jit, static_argnames=("M", "nx", "ny"))
def _candidates(slot, pix, t, valid, key, q_ct, M, nx, ny):
    """Cross-talk candidates of the given hits: [M] compacted (slot, pixel, time, parent index, valid)."""
    x, y = pix // NPIX, pix % NPIX
    nxp = jnp.stack([x - 1, x + 1, x, x], 1)
    nyp = jnp.stack([y, y, y - 1, y + 1], 1)
    ok = jnp.stack([x > 0, x + 1 < nx, y > 0, y + 1 < ny], 1) & valid[:, None]
    ok &= jax.random.uniform(key, ok.shape) < q_ct
    r, j = jnp.nonzero(ok, size=M, fill_value=0)
    n = ok.sum()
    v = jnp.arange(M) < n
    return slot[r], (nxp * NPIX + nyp)[r, j], t[r], r, v, n


@partial(jax.jit, static_argnames=("L", "rounds"))
def _solve(key_sort, t, u, parent, valid, dead, vov, tau, p1, p2, L, rounds):
    """Sort hits into pixel streams, then the fixed point over the active set. Returns fired, v (per hit, input order)."""
    N = t.shape[0]
    o = jnp.lexsort((t, key_sort))
    ks, ts, us = key_sort[o], t[o], u[o]
    new = jnp.concatenate([jnp.array([True]), ks[1:] != ks[:-1]])
    start = jnp.where(new, jnp.arange(N), 0)
    start = lax.cummax(start, axis=0)
    pos = jnp.arange(N) - start
    nst = jnp.cumsum(new) - 1                      # stream id per sorted hit
    S = N                                          # upper bound on streams
    dense_idx = jnp.full((S, L), N, jnp.int64).at[nst, jnp.minimum(pos, L - 1)].set(jnp.arange(N))
    tsd = jnp.concatenate([ts, jnp.array([0.0])])[dense_idx]
    usd = jnp.concatenate([us, jnp.array([2.0])])[dense_idx]
    deadd = jnp.concatenate([dead[ks % NP2], jnp.array([True])])[dense_idx]
    inv = jnp.zeros(N, jnp.int64).at[o].set(jnp.arange(N))   # input index -> sorted index

    def scan_streams(act_sorted):
        actd = jnp.concatenate([act_sorted, jnp.array([False])])[dense_idx] & ~deadd

        def step(last, xs):
            tt, uu, aa = xs
            v = vov * (1.0 - jnp.exp(-(tt - last) / tau))
            f = aa & (uu < p1 * (1.0 - jnp.exp(-v / p2)))
            return jnp.where(f, tt, last), (f, v)

        _, (f, v) = lax.scan(step, jnp.full(S, -jnp.inf), (tsd.T, usd.T, actd.T))
        fired_sorted = jnp.zeros(N + 1, bool).at[dense_idx.T.ravel()].set(f.ravel())[:N]
        v_sorted = jnp.zeros(N + 1).at[dense_idx.T.ravel()].set(v.ravel())[:N]
        return fired_sorted, v_sorted

    def body(_, state):
        active, fired = state
        f_s, _ = scan_streams(active[o])
        fired = f_s[inv]
        par_ok = jnp.where(parent >= 0, active[jnp.maximum(parent, 0)] & fired[jnp.maximum(parent, 0)], True)
        return valid & par_ok, fired

    active0 = valid & (parent < 0)
    active, fired = lax.fori_loop(0, rounds, body, (active0, jnp.zeros(N, bool)))
    f_s, v_s = scan_streams(active[o])
    return f_s[inv] & active, v_s[inv]


CT_SAFETY = 1.25  # candidate buffer per generation = CT_SAFETY x expected + 8 sigma + 256 (no power-of-two rounding);
                  # overflow is flagged and the chunk redone on the host, never lost silently
STREAM_L = 8      # hits per (SiPM, pixel) stream in the first attempt; a longer stream (showers: up to ~2000 photons
                  # on one SiPM) is flagged and only the stream-length bucket is enlarged (L = 16, 32, ...), on device


@partial(jax.jit, static_argnames=("NB", "M1", "M2", "M3", "Ntot", "L", "nx", "ny", "gens"))
def device_sipm(slot, t, valid, key, cum, ncol, dead, q_ct, vov, tau, p1, p2, cap,
                NB, M1, M2, M3, Ntot, L, nx, ny, gens=GENERATIONS):
    """All of the SiPM stage in one jitted call, fixed shapes, no host round trip.

    slot [NB] SiPM slot per photon, t [NB], valid [NB]; cum: normalised cumulative photon map; ncol: its row length.
    Returns per hit (photons first, then cross-talk candidates): slot, time, root photon index, fired, charge, pe, and
    an overflow flag (a candidate buffer filled up, or a pixel stream exceeded L): rerun those inputs on the host path.
    """
    kp, kx, ku, *kg = jax.random.split(key, 3 + gens)
    b_ = jnp.searchsorted(cum, jax.random.uniform(kp, (NB,)), side="right")
    h = jax.random.uniform(kx, (2, NB)) >= 0.5                       # GetRandom2 + lrint
    pix = (b_ // ncol + h[0]) * NPIX + (b_ % ncol + h[1])
    hs, hp, ht, hpar, hv = [slot], [pix], [t], [jnp.full(NB, -1)], [valid]
    cs, cp, ctt, cv, off = slot, pix, t, valid, 0
    over = jnp.array(False)
    for M, k in zip((M1, M2, M3)[:gens], kg):
        s2, p2_, t2, par, v2, n = _candidates(cs, cp, ctt, cv, k, q_ct, M, nx, ny)
        over |= n > M
        hs.append(s2), hp.append(p2_), ht.append(t2), hpar.append(par + off), hv.append(v2)
        off += cs.shape[0]
        cs, cp, ctt, cv = s2, p2_, t2, v2
    S_, P_, T_, PAR, V_ = (jnp.concatenate(x) for x in (hs, hp, ht, hpar, hv))
    pad = Ntot - S_.shape[0]
    S_, P_, T_ = jnp.pad(S_, (0, pad)), jnp.pad(P_, (0, pad)), jnp.pad(T_, (0, pad))
    PAR, V_ = jnp.pad(PAR, (0, pad), constant_values=-1), jnp.pad(V_, (0, pad))
    U = jax.random.uniform(ku, (Ntot,))
    ksort = jnp.where(V_, S_ * NP2 + P_, jnp.iinfo(jnp.int64).max)
    sk = jnp.sort(ksort)
    run_start = jnp.concatenate([jnp.array([True]), sk[1:] != sk[:-1]])
    first = lax.cummax(jnp.where(run_start, jnp.arange(Ntot), 0), axis=0)
    stream_over = jnp.any(((jnp.arange(Ntot) - first) >= L) & (sk != jnp.iinfo(jnp.int64).max))
    fired, v = _solve(ksort, T_, U, PAR, V_, dead, vov, tau, p1, p2, L, gens + 1)
    root = jnp.arange(Ntot)
    for _ in range(gens):
        root = jnp.where(PAR[root] >= 0, PAR[root], root)
    return {"slot": S_, "time": T_, "root": root, "fired": fired, "charge": cap * v, "pe": v / vov,
            "overflow": over | stream_over, "candidate_overflow": over, "stream_overflow": stream_over}


def device_buckets(n_photons, cfg):
    """Static shapes for device_sipm from the photon count only (no data-dependent host sync)."""
    q_ct = (1.0 - (1.0 - cfg.cross_talk) ** 0.25) / cfg.p_fire(cfg.overvoltage)
    NB = _bucket(n_photons)
    Ms, prev = [], NB
    for _ in range(GENERATIONS):
        mu = 4 * q_ct * prev
        M = int(CT_SAFETY * mu + 8 * np.sqrt(mu) + 256)
        M = (M + 255) // 256 * 256
        Ms.append(M)
        prev = M
    Ntot = (NB + sum(Ms) + 1023) // 1024 * 1024
    return NB, Ms, Ntot, q_ct


def simulate(chan, time, photon, pmap, cfg, rng, key, t_end=1e5, max_photons=1 << 23):
    """JAX equivalent of sipm.simulate (no thermal noise / traps). Returns the same dict (host arrays).

    One device_sipm call per chunk of whole SiPMs; the only host work is labelling SiPMs (np.unique) and the final
    transfer. A chunk that overflows its static buffers is redone with the numpy port (sipm.simulate)."""
    from . import sipm as SI
    if cfg.thermal_rate > 0 or cfg.trap0_prob > 0 or cfg.trap1_prob > 0:
        raise NotImplementedError("sipm_jax: thermal noise and trap afterpulses are not in the twin")
    chan, time, photon = np.asarray(chan), np.asarray(time, float), np.asarray(photon)
    keep = time <= t_end
    chan, time, photon = chan[keep], time[keep], photon[keep]
    o = np.argsort(chan, kind="stable")
    chan, time, photon = chan[o], time[o], photon[o]
    dead = np.zeros(NP2, bool)
    for x, y in cfg.inactive:
        dead[x * NPIX + y] = True
    dead_d = jnp.asarray(dead)
    cum = jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum())
    bounds = np.flatnonzero(np.r_[True, chan[1:] != chan[:-1]])
    cuts = [0]
    while cuts[-1] < len(chan):
        j = int(np.searchsorted(bounds, cuts[-1] + max_photons, side="right")) - 1
        nxt = int(bounds[j]) if j < len(bounds) and bounds[j] > cuts[-1] else len(chan)
        cuts.append(nxt if nxt > cuts[-1] else len(chan))
    outs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        n = b - a
        NB, Ms, Ntot, q_ct = device_buckets(n, cfg)
        uc, slot_np = np.unique(chan[a:b], return_inverse=True)
        key, sub = jax.random.split(key)
        args = (jnp.asarray(np.pad(slot_np.ravel(), (0, NB - n))), jnp.asarray(np.pad(time[a:b], (0, NB - n))),
                jnp.arange(NB) < n, sub, cum, pmap.shape[1], dead_d, q_ct, cfg.overvoltage,
                cfg.time_constant, cfg.p1, cfg.p2, cfg.capacitance, NB, *Ms, Ntot)
        L = STREAM_L
        r = device_sipm(*args, L, cfg.n_pixels_x, cfg.n_pixels_y)
        while bool(r["stream_overflow"]) and L < 1024:
            L *= 2
            r = device_sipm(*args, L, cfg.n_pixels_x, cfg.n_pixels_y)
        if bool(r["overflow"]):
            outs.append(SI.simulate(chan[a:b], time[a:b], photon[a:b], pmap, cfg, rng, t_end=t_end))
            continue
        f = np.asarray(r["fired"])
        idx = np.flatnonzero(f)
        rt = np.asarray(r["root"])[idx]
        outs.append({"chan": uc[np.asarray(r["slot"])[idx]], "time": np.asarray(r["time"])[idx],
                     "charge": np.asarray(r["charge"])[idx], "pe": np.asarray(r["pe"])[idx],
                     "photon": photon[a:b][np.minimum(rt, n - 1)]})
    if not outs:
        return {k: np.zeros(0) for k in ("chan", "time", "charge", "pe", "photon")}
    out = {k: np.concatenate([o_[k] for o_ in outs]) for k in outs[0]}
    o = np.lexsort((out["time"], out["chan"]))
    return {k: a_[o] for k, a_ in out.items()}
