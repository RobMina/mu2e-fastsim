"""Conversion electrons from birth to tracker StrawGasSteps in one device loop: the CE chain and the tracker transport
advance in the same loop iteration on two lane pools that are refilled as lanes finish (opt-in).

beamfastsim.cechain_jax carries each electron to the tracker entry plane; trkfastsim.sim.fieldsteps_jax steps it
through the tracker. Run one after the other (DeviceChain's default), each stage ends with a long tail of a few slow
lanes (electrons trapped in the target material, low-momentum curlers in the tracker) during which the device is
mostly idle. Here both stages share one loop: a CE pool steps births (refilled from the birth queue as lanes finish),
the electrons that reach the entry plane are queued, and a tracker pool steps them (refilled from that queue). The
tail of one stage overlaps the bulk of the other, and only one shared tail remains per batch.

Same kernels as the two transports (cechain_jax.lane_kernels step, fieldsteps_jax chord); the random streams differ
(keys per launch and iteration, lanes in another order), so results are statistically equivalent, not identical.
Outputs per birth: the cechain fields (device arrays) and a fieldsteps_jax.Result whose step records and final states
are indexed by birth.
"""
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from . import cechain as C
from . import cechain_jax as CJ

ME = CJ.ME
OUT = ("fate", "x", "u", "T", "t", "s", "cnt", "nvac", "nslab", "nstall")


def _pow2(n, lo=64):
    return max(lo, 1 << int(np.ceil(np.log2(max(int(n), 1)))))


_RUNS = {}       # compiled loops per (tracker, geometry, pool sizes, variant): built once per process


@jax.jit
def _ce_drain(ce, ce_id, ce_out, q, q_len, bt):
    """Finished CE lanes: per-birth outputs; the plane crossings appended to the tracker queue."""
    n = bt.shape[0]
    done = (ce_id >= 0) & (ce["fate"] != C.LIVE)
    tgt = jnp.where(done, ce_id, n)
    out = {k: ce_out[k].at[tgt].set(ce[k], mode="drop") for k in ce_out}
    plane = done & (ce["fate"] == C.PLANE)
    pos = q_len + jnp.cumsum(plane) - 1
    qt = jnp.where(plane, pos, n)
    Tp = ce["T"]
    pm = jnp.sqrt(Tp * (Tp + 2.0 * ME))
    un = ce["u"] / jnp.linalg.norm(ce["u"], axis=1, keepdims=True)
    bt_ = bt[jnp.clip(ce_id, 0, n - 1)]
    q2 = {"x": q["x"].at[qt].set(ce["x"], mode="drop"), "p": q["p"].at[qt].set(un * pm[:, None], mode="drop"),
          "t": q["t"].at[qt].set(bt_ + ce["t"], mode="drop"), "id": q["id"].at[qt].set(ce_id, mode="drop")}
    return out, q2, q_len + jnp.sum(plane), jnp.where(done, -1, ce_id), jnp.sum(ce_id >= 0)


@jax.jit
def _tr_drain(tr, tr_id, tfin):
    from trkfastsim.sim import fieldsteps_jax as FJ
    n = tfin["p"].shape[0]
    done = (tr_id < n) & (tr["status"] != FJ.LIVE)
    tgt = jnp.where(done, tr_id, n)
    out = {k: tfin[k].at[tgt].set(tr[k], mode="drop") for k in tfin}
    return out, jnp.where(done, n, tr_id), jnp.sum(tr_id < n)


def _tr_fill_fn(env):
    from trkfastsim.sim import fieldsteps_jax as FJ

    @jax.jit
    def run(tr, tr_id, q, qa, kq, T_, org):
        """Free tracker lanes take queue entries qa .. qa + kq - 1 (in lane order)."""
        n = q["t"].shape[0]
        free = tr_id >= n
        rank = jnp.cumsum(free) - 1
        take = free & (rank < kq)
        e = jnp.clip(qa + rank, 0, n - 1)
        xs = q["x"][e] - org
        pv = q["p"][e]
        pm = jnp.linalg.norm(pv, axis=1)
        new = FJ._state(xs, pv / pm[:, None], pm, q["t"][e], jnp.ones(tr_id.shape[0], bool))
        new["status"] = jnp.where(FJ._leaving(T_, env, xs, new["u"]), FJ.LEFT, new["status"])
        merged = {k: jnp.where(take.reshape((-1,) + (1,) * (v.ndim - 1)), new[k].astype(v.dtype), v)
                  for k, v in tr.items()}
        return merged, jnp.where(take, q["id"][e], tr_id), jnp.sum(take)
    return run


_FILLS = {}


def transport(key, cg, maps, trk, x0, u0, p0, t0, h=10.0, dtype=np.float64, unroll=True, ce_lanes=32768,
              trk_lanes=16384, n_it=64, cand_frac=0.125, slab_frac=0.5, shrink=4, stats=None):
    """Births (x0, u0 [n, 3], p0, t0 [n], host arrays, Mu2e frame) through the CE chain and the tracker in one device
    loop. trk: fieldsteps_jax.Tracker. Returns (ce: dict of per-birth device arrays with the cechain.Result fields,
    tracker: fieldsteps_jax.Result indexed by birth). n_it: loop iterations per launch; between launches finished
    lanes are written out and refilled (the host reads lane counts only)."""
    from trkfastsim.sim import fieldsteps_jax as FJ
    dt = np.dtype(dtype)
    init, step = CJ.lane_kernels(cg, maps, h=h, dtype=dt, unroll=unroll)
    x, u, T, k0, zl = CJ.start_tables(cg, x0, u0, p0)
    n = x.shape[0]
    tb = np.broadcast_to(np.asarray(t0, float), (n,))
    bx, bu, bT, bk, bz = (jnp.asarray(a, dt) if a.dtype.kind == "f" else jnp.asarray(a) for a in (x, u, T, k0, zl))
    bt = jnp.asarray(tb, dt)
    org = jnp.asarray(trk.origin, dt)
    tcfg = trk.cfg
    kw = dict(cfg=tcfg, fn=trk.fn, env=trk.env, pwidth=trk.pwidth, lnt0=trk.lnt0, dlnt=trk.dlnt)
    Lc = _pow2(min(n, ce_lanes))
    Lt = _pow2(min(n, trk_lanes))

    # per-birth outputs
    ce_out = {"fate": jnp.full(n, C.LOOPER, jnp.int32), "x": jnp.zeros((n, 3), dt), "u": jnp.zeros((n, 3), dt),
              "T": jnp.zeros(n, dt), "t": jnp.zeros(n, dt), "s": jnp.zeros(n, dt),
              "cnt": jnp.zeros((n, CJ.NCOUNT), jnp.int32), "nvac": jnp.zeros(n, jnp.int32),
              "nslab": jnp.zeros(n, jnp.int32), "nstall": jnp.zeros(n, jnp.int32)}
    q = {"x": jnp.zeros((n, 3), dt), "p": jnp.zeros((n, 3), dt), "t": jnp.zeros(n, dt), "id": jnp.zeros(n, jnp.int32)}
    rec = FJ._records(n, dt)
    tfin = {"x": jnp.zeros((n, 3), dt), "u": jnp.zeros((n, 3), dt), "p": jnp.zeros(n, dt), "t": jnp.zeros(n, dt),
            "path": jnp.zeros(n, dt), "chords": jnp.zeros(n, jnp.int32),
            "status": jnp.full(n, FJ.LEFT, jnp.int32)}

    def ce_block(ids, Lc):
        j = jnp.asarray(ids)
        return init(maps, bx[j], bu[j], bT[j], jnp.zeros(Lc, dt), bk[j], bz[j])

    # pools
    first = np.arange(min(n, Lc))
    ids = np.concatenate([first, np.zeros(Lc - first.size, np.int64)])
    ce = ce_block(ids, Lc)
    ce["fate"] = jnp.where(jnp.arange(Lc) < first.size, ce["fate"], C.LOOPER)     # padding lanes inert
    ce_id = jnp.asarray(np.where(np.arange(Lc) < first.size, ids, -1), jnp.int32)
    next_b = int(first.size)
    tr = FJ._state(jnp.zeros((Lt, 3), dt), jnp.tile(jnp.asarray([0.0, 0.0, 1.0], dt), (Lt, 1)), jnp.ones(Lt, dt),
                   jnp.zeros(Lt, dt), jnp.zeros(Lt, bool))
    tr_id = jnp.full(Lt, n, jnp.int32)                 # n: no birth (record writes dropped)
    q_len = q_next = 0
    tot = np.zeros(len(FJ.NSTAT), np.int64)
    it = launches = 0
    timeline = []

    def body_fn(Lc_, Lt_, do_ce, do_trk):
        """The loop for pool sizes (Lc_, Lt_) with the CE and/or tracker step compiled in (no device conditionals:
        a conditional would carry the record buffers; the host picks the variant from the busy-lane counts)."""
        m_cand = max(1, int(np.ceil(cand_frac * Lc_)))
        m_slab = max(1, int(np.ceil(slab_frac * Lc_)))

        @jax.jit
        def run(ce, tr, rec, tr_id, key, it0, maps_, T_, groups_, mc_):
            tk_ = ((T_, groups_, trk.env, trk.fn, trk.pwidth, trk.lnt0, trk.dlnt, mc_), tcfg)
            kce = jax.random.fold_in(key, 0)
            ktr = jax.random.fold_in(key, 1)

            def cond(c):
                i, ce_, tr_, _, _ = c
                live = jnp.zeros((), bool)
                if do_ce:
                    live = live | jnp.any(ce_["fate"] == C.LIVE)
                if do_trk:
                    live = live | jnp.any(tr_["status"] == FJ.LIVE)
                return (i < n_it) & live

            def body(c):
                i, ce_, tr_, rec_, stt = c
                if do_ce:
                    ce_ = step(maps_, jax.random.fold_in(kce, it0 + i), ce_, m_cand, m_slab)
                if do_trk:
                    tr_, rec_, st2 = FJ._chord(tk_, tr_, rec_, tr_id, ktr, it0 + i)
                    stt = stt + st2
                return i + 1, ce_, tr_, rec_, stt
            i, ce, tr, rec, stats_ = lax.while_loop(cond, body, (jnp.int32(0), ce, tr, rec,
                                                                  jnp.zeros(len(FJ.NSTAT), jnp.int64)))
            return i, ce, tr, rec, stats_
        return run

    def ce_fill(ce, ce_id, a, k, Lc_):
        free = np.flatnonzero(np.asarray(ce_id) < 0)[:k]
        ids = np.zeros(Lc_, np.int64)
        ids[free] = np.arange(a, a + len(free))
        mask = np.zeros(Lc_, bool)
        mask[free] = True
        new = ce_block(ids, Lc_)
        ce = CJ._merge(ce, new, jnp.asarray(mask))
        return ce, jnp.where(jnp.asarray(mask), jnp.asarray(ids, jnp.int32), ce_id), len(free)

    n_ce_busy, n_tr_busy = int(first.size), 0
    t_start = time.perf_counter()
    while True:
        do_ce, do_trk = n_ce_busy > 0, n_tr_busy > 0
        rk = (id(trk), id(cg), id(step), Lc, Lt, do_ce, do_trk, n_it, cand_frac, slab_frac)
        if rk not in _RUNS:
            _RUNS[rk] = body_fn(Lc, Lt, do_ce, do_trk)
        t_l = time.perf_counter()
        i, ce, tr, rec, st_ = _RUNS[rk](ce, tr, rec, tr_id, jax.random.fold_in(key, launches), jnp.int32(it), maps,
                                       trk.T, trk.groups, trk.mc)
        launches += 1
        it += n_it
        tot += np.asarray(st_)
        ce_out, q, q_len_d, ce_id, _ = _ce_drain(ce, ce_id, ce_out, q, jnp.int32(q_len), bt)
        q_len = int(q_len_d)
        tfin, tr_id, _ = _tr_drain(tr, tr_id, tfin)
        n_ce_busy = int(jnp.sum(ce_id >= 0))
        n_tr_busy = int(jnp.sum(tr_id < n))
        timeline.append((Lc, Lt, n_ce_busy, n_tr_busy, round(time.perf_counter() - t_l, 3)))
        # refill
        if next_b < n and n_ce_busy < Lc:
            k = min(Lc - n_ce_busy, n - next_b)
            ce, ce_id, got = ce_fill(ce, ce_id, next_b, k, Lc)
            next_b += got
            n_ce_busy += got
        if q_next < q_len and n_tr_busy < Lt:
            kq = min(Lt - n_tr_busy, q_len - q_next)
            if trk.env not in _FILLS:
                _FILLS[trk.env] = _tr_fill_fn(trk.env)
            tr, tr_id, got = _FILLS[trk.env](tr, tr_id, q, jnp.int32(q_next), jnp.int32(kq), trk.T, org)
            q_next += int(got)
            n_tr_busy += int(got)
        if next_b >= n and n_ce_busy == 0 and q_next >= q_len and n_tr_busy == 0:
            break
        # compaction once a pool has no more input
        if next_b >= n and _pow2(n_ce_busy) * shrink <= Lc:
            Lc2 = _pow2(n_ce_busy)
            keep = np.flatnonzero(np.asarray(ce_id) >= 0)
            sel = jnp.asarray(np.concatenate([keep, np.full(Lc2 - keep.size, keep[0] if keep.size else 0)]))
            ce = jax.tree_util.tree_map(lambda a: a[sel] if getattr(a, "ndim", 0) > 0 else a, ce)
            pad = jnp.arange(Lc2) >= keep.size
            ce["fate"] = jnp.where(pad, C.LOOPER, ce["fate"])
            ce_id = jnp.where(pad, -1, ce_id[sel])
            Lc = Lc2
        if next_b >= n and n_ce_busy == 0 and q_next >= q_len and _pow2(n_tr_busy) * shrink <= Lt:
            Lt2 = _pow2(n_tr_busy)
            keep = np.flatnonzero(np.asarray(tr_id) < n)
            sel = jnp.asarray(np.concatenate([keep, np.full(Lt2 - keep.size, keep[0] if keep.size else 0)]))
            tr = {k: v[sel] for k, v in tr.items()}
            pad = jnp.arange(Lt2) >= keep.size
            tr["status"] = jnp.where(pad, FJ.LEFT, tr["status"])
            tr_id = jnp.where(pad, n, tr_id[sel])
            Lt = Lt2
    # outputs
    cnt = ce_out["cnt"]
    ce_res = dict(fate=ce_out["fate"], x=ce_out["x"], u=ce_out["u"], T=ce_out["T"], t=bt + ce_out["t"],
                  path=ce_out["s"], nfoil=cnt[:, 0], nipa=cnt[:, 1], nopa=cnt[:, 2], nring=cnt[:, 3],
                  nvac=ce_out["nvac"], nslab=ce_out["nslab"], nstall=ce_out["nstall"])
    has = ce_res["fate"] == C.PLANE
    steps = FJ._rows(trk.T, trk.gas_mc, rec, FJ.s_nrec_of(rec), jax.random.fold_in(key, 1 << 30), tcfg, trk.lnt0,
                     trk.dlnt, trk.tcut_gas)
    # births that did not enter: the tracker final carries their CE end state, status 0 (as an inactive track)
    fx = jnp.where(has[:, None], tfin["x"] + org, ce_res["x"])
    fu = jnp.where(has[:, None], tfin["u"], ce_res["u"])
    fp = jnp.where(has, tfin["p"], jnp.sqrt(ce_res["T"] * (ce_res["T"] + 2.0 * ME)))
    fin = dict(x=fx[:, 0], y=fx[:, 1], z=fx[:, 2], px=fu[:, 0] * fp, py=fu[:, 1] * fp, pz=fu[:, 2] * fp,
               t=jnp.where(has, tfin["t"], ce_res["t"]), path=jnp.where(has, tfin["path"], 0.0),
               chords=jnp.where(has, tfin["chords"], 0), status=jnp.where(has, tfin["status"], FJ.LEFT))
    out_stats = dict(zip(FJ.NSTAT, (int(v) for v in tot)), iterations=it, launches=launches, timeline=timeline,
                     wall_s=round(time.perf_counter() - t_start, 3))
    if stats is not None:
        stats.update(out_stats)
    return ce_res, FJ.Result(steps, fin, out_stats)
