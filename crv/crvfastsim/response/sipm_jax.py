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


def _gather(arr_sorted, fill, di):
    return jnp.concatenate([arr_sorted, jnp.array([fill], arr_sorted.dtype)])[di]


@partial(jax.jit, static_argnames=("L", "SL", "LL", "rounds"))
def _solve(key_sort, t, u, parent, valid, dead, vov, tau, p1, p2, L, SL, LL, rounds):
    """Sort hits into pixel streams, then the fixed point over the active set.

    Streams of at most L hits go to a short table [N, L]; longer streams (rare) go to a long table [SL, LL], so a few
    long streams do not inflate every row. Returns fired, v (per hit, input order), the long-stream count and the
    longest stream; the long table overflowed if count > SL or longest > LL (caller retries with exact buckets)."""
    N = t.shape[0]
    pad_key = jnp.iinfo(jnp.int64).max
    o = jnp.lexsort((t, key_sort))
    ks, ts, us = key_sort[o], t[o], u[o]
    real = ks != pad_key
    new = jnp.concatenate([jnp.array([True]), ks[1:] != ks[:-1]])
    start = lax.cummax(jnp.where(new, jnp.arange(N), 0), axis=0)
    pos = jnp.arange(N) - start
    nst = jnp.cumsum(new) - 1
    slen = jax.ops.segment_sum(jnp.ones(N, jnp.int64), nst, num_segments=N)[nst]
    long_ = real & (slen > L)
    short = real & ~long_
    sid = jnp.cumsum(new & short) - 1
    lid = jnp.cumsum(new & long_) - 1
    n_long = (new & long_).sum()
    max_len = jnp.max(jnp.where(real, slen, 0))
    max_long = jnp.max(jnp.where(long_, slen, 0))
    hit = jnp.arange(N)
    di_s = jnp.full((N, L), N, jnp.int64).at[jnp.where(short, sid, N), jnp.minimum(pos, L - 1)].set(hit, mode="drop")
    di_l = jnp.full((SL, LL), N, jnp.int64).at[jnp.where(long_, lid, SL), jnp.minimum(pos, LL - 1)].set(hit, mode="drop")
    dead_s = dead[ks % NP2]
    tabs = [(di, _gather(ts, 0.0, di), _gather(us, 2.0, di), _gather(dead_s, True, di)) for di in (di_s, di_l)]
    inv = jnp.zeros(N, jnp.int64).at[o].set(hit)   # input index -> sorted index

    def step(last, xs):
        tt, uu, aa = xs
        v = vov * (1.0 - jnp.exp(-(tt - last) / tau))
        f = aa & (uu < p1 * (1.0 - jnp.exp(-v / p2)))
        return jnp.where(f, tt, last), (f, v)

    def scan_streams(act_sorted):
        fired_sorted, v_sorted = jnp.zeros(N + 1, bool), jnp.zeros(N + 1)
        for di, tsd, usd, deadd in tabs:
            actd = _gather(act_sorted, False, di) & ~deadd
            _, (f, v) = lax.scan(step, jnp.full(di.shape[0], -jnp.inf), (tsd.T, usd.T, actd.T))
            idx = di.T.ravel()                      # sentinel N collects the empty cells and is dropped
            fired_sorted = fired_sorted.at[idx].set(f.ravel())
            v_sorted = v_sorted.at[idx].set(v.ravel())
        return fired_sorted[:N], v_sorted[:N]

    def body(_, state):
        active, fired = state
        f_s, _ = scan_streams(active[o])
        fired = f_s[inv]
        par_ok = jnp.where(parent >= 0, active[jnp.maximum(parent, 0)] & fired[jnp.maximum(parent, 0)], True)
        return valid & par_ok, fired

    active0 = valid & (parent < 0)
    active, fired = lax.fori_loop(0, rounds, body, (active0, jnp.zeros(N, bool)))
    f_s, v_s = scan_streams(active[o])
    return f_s[inv] & active, v_s[inv], n_long, max_long, max_len


CT_SAFETY = 1.25  # candidate buffer per generation = CT_SAFETY x expected + 8 sigma + 256 (no power-of-two rounding);
                  # overflow is flagged and the chunk redone on the host, never lost silently
STREAM_L = 8      # short-table stream length: (SiPM, pixel) streams up to L hits (nearly all of them)
LONG_SL = 256     # first-attempt long table: streams longer than STREAM_L (showers) ...
LONG_LL = 64      # ... and their length; on overflow the caller retries once with buckets of the measured count/length
LONG_CELLS_MAX = 1 << 26  # long-table cells beyond which the chunk goes to the host path instead (memory guard)


@partial(jax.jit, static_argnames=("NB", "M1", "M2", "M3", "Ntot", "L", "SL", "LL", "nx", "ny", "gens"))
def device_sipm(slot, t, valid, key, cum, ncol, dead, q_ct, vov, tau, p1, p2, cap,
                NB, M1, M2, M3, Ntot, L, SL, LL, nx, ny, gens=GENERATIONS):
    """All of the SiPM stage in one jitted call, fixed shapes, no host round trip.

    slot [NB] SiPM slot per photon, t [NB], valid [NB]; cum: normalised cumulative photon map; ncol: its row length.
    Returns per hit (photons first, then cross-talk candidates): slot, time, root photon index, fired, charge, pe, and
    an overflow flag (a candidate buffer filled up, or the long-stream table [SL, LL] was too small: retry with
    long_buckets(r), else rerun those inputs on the host path).
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
    fired, v, n_long, max_long, max_len = _solve(ksort, T_, U, PAR, V_, dead, vov, tau, p1, p2, L, SL, LL, gens + 1)
    stream_over = (n_long > SL) | (max_long > LL)
    root = jnp.arange(Ntot)
    for _ in range(gens):
        root = jnp.where(PAR[root] >= 0, PAR[root], root)
    return {"slot": S_, "time": T_, "root": root, "fired": fired, "charge": cap * v, "pe": v / vov,
            "overflow": over | stream_over, "candidate_overflow": over, "stream_overflow": stream_over,
            "n_long_streams": n_long, "max_long_stream": max_long, "max_stream": max_len}


def long_buckets(r):
    """Long-table shapes (SL, LL) that fit the measured long streams of a device_sipm result (two scalar syncs), or
    None when that table would exceed LONG_CELLS_MAX (send the chunk to the host path)."""
    SL = _bucket(int(r["n_long_streams"]), LONG_SL)
    LL = _bucket(int(r["max_long_stream"]), LONG_LL)
    return None if SL * LL > LONG_CELLS_MAX else (SL, LL)


def run_device_sipm(args, cfg):
    """device_sipm with the default long table, retried once with measured buckets on a stream overflow."""
    r = device_sipm(*args, STREAM_L, LONG_SL, LONG_LL, cfg.n_pixels_x, cfg.n_pixels_y)
    if bool(r["stream_overflow"]):
        b = long_buckets(r)
        if b is not None:
            r = device_sipm(*args, STREAM_L, *b, cfg.n_pixels_x, cfg.n_pixels_y)
    return r


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
        r = run_device_sipm(args, cfg)
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
