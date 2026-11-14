"""Pulses -> coincidences on the device in one pipelined pass (the R1 measurement).

Events are processed in chunks. For chunk k the jitted pulse reco is launched (asynchronously); meanwhile the host
collects chunk k-1's pulses, builds the coincidence-finder input (find_async: split, pack, launch) and collects chunk
k-2's first-choice clusters (coinc_jax.collect). The promotions of all chunks run together after the last chunk
(coinc_jax.complete_all: one call per tier level instead of one per chunk; the same clusters), then the numpy
fallback. On a GPU the host work of one chunk overlaps the device work of the next; on CPU the stages simply run in
turn. Waveforms longer than the kernel input, whose fit used all kernel
rounds or overflowed its window go to full-length numpy pulse reco (as in reco-vr2jax).
"""
import dataclasses
import time as _time

import numpy as np

from . import coinc as C
from . import coinc_jax as CJ
from . import pulsefit, pulsefit_jax as PJ
from mu2efastsim import precision as dprec

dprec.register(__name__)   # JAX creators/samplers here follow MU2EFS_DEVICE_FLOAT

PULSE_KEYS = ("found", "pe", "time", "rounds_exhausted", "window_overflow", "input_truncated")


def _pulse_dispatch(W, rows, cfg, batch, L, bounded=False):
    import jax.numpy as jnp
    outs = []
    for a in range(0, len(rows), batch):
        r = rows[a:a + batch]
        n = len(r)
        # Keep pulse-fit shapes fixed: changing their row capacity can change fit rounding.
        pad = lambda x, v: np.concatenate([x[r], np.full((batch - n,) + x.shape[1:], v, x.dtype)]) if n < batch else x[r]
        d = PJ.reconstruct_dense(jnp.asarray(pad(W["adc"][:, :L], 0)), jnp.asarray(pad(np.minimum(W["n"], L), 0)),
                                 jnp.asarray(pad(W["start"], 0)), jnp.asarray(pad(W["ped"].astype(float), 0.0)),
                                 jnp.asarray(pad(W["area"].astype(float), 1.0)),
                                 jnp.asarray(pad(W["height"].astype(float), 1.0)), cfg=cfg,
                                 **({"bounded": True} if bounded else {}))
        d = {**d, 'input_truncated': pad(W['n'] > L, False)}
        outs.append(({k: d[k] for k in PULSE_KEYS}, n))
    return outs


REFIT_ROUNDS = 16        # device refit: pulse rounds (static); a pulse in the last one still goes to numpy
REFIT_MIN_L = 16         # device refit: smallest padded waveform length


def _bucket(n, lo):
    return max(lo, 1 << int(np.ceil(np.log2(max(n, 1)))))


def _device_refit(W, g, cfg_np, caps=None):
    """Waveforms g (W rows) through the dense JAX fit at full length (power-of-two padded) with REFIT_ROUNDS rounds.
    Returns (flat pulses wf/seq/pe/time with wf indexing g, mask of g still needing the numpy refit)."""
    import jax.numpy as jnp
    n = W["n"][g]
    L2 = _bucket(int(n.max()), REFIT_MIN_L)
    R = _bucket(len(g), 1024)
    if caps is not None:   # fixed shapes (crvfastsim.capacity): rows only pad, samples must hold every waveform
        L2 = caps.size("refit_samples", int(n.max()), L2)
        R = caps.size("refit_rows", len(g), R)
    adc = np.zeros((R, L2), W["adc"].dtype)
    width = min(L2, W["adc"].shape[1])
    adc[:len(g), :width] = W["adc"][g][:, :width]
    pad = lambda x, v: np.concatenate([x, np.full(R - len(g), v, x.dtype)])
    cfg = dataclasses.replace(cfg_np, max_pulses=max(cfg_np.max_pulses, REFIT_ROUNDS))
    d = PJ.reconstruct_dense(jnp.asarray(adc), jnp.asarray(pad(n, 0)), jnp.asarray(pad(W["start"][g], 0)),
                             jnp.asarray(pad(W["ped"][g].astype(float), 0.0)),
                             jnp.asarray(pad(W["area"][g].astype(float), 1.0)),
                             jnp.asarray(pad(W["height"][g].astype(float), 1.0)), cfg=cfg,
                             **({"bounded": True} if caps is not None else {}))
    d = {k: dprec.host(d[k])[:len(g)] for k in ("found", "pe", "time", "rounds_exhausted", "window_overflow")}
    again = d["rounds_exhausted"] | d["window_overflow"]
    wf, seq = np.nonzero(d["found"] & ~again[:, None])
    return dict(wf=wf, seq=seq, pe=d["pe"][wf, seq], time=d["time"][wf, seq]), again


def _pulse_collect(outs, W, rows, cfg_np, device_refit=False, counts=None, caps=None):
    """Flat pulses (waveform row, seq, pe, time) in (waveform, seq) order; flagged waveforms redone in numpy, or first
    on the device at full length (device_refit) with only the ones that still overflow in numpy."""
    d = {k: np.concatenate([dprec.host(o[k])[:n] for o, n in outs]) for k in PULSE_KEYS}
    wf, seq = np.nonzero(d["found"])
    redo = d["rounds_exhausted"] | d["window_overflow"] | d["input_truncated"]
    pe, tm = d["pe"][wf, seq], d["time"][wf, seq]
    nredo = int(redo.sum())
    if nredo and device_refit:
        rr = np.flatnonzero(redo)
        dev, again = _device_refit(W, rows[rr], cfg_np, caps)
        parts = [dev]
        if again.any():
            ra = np.flatnonzero(again)
            g = rows[rr[ra]]
            s = pulsefit.reconstruct_full(W["adc"][g], W["n"][g], W["start"][g], W["ped"][g].astype(float),
                                          W["area"][g].astype(float), W["height"][g].astype(float), cfg_np)
            parts.append(dict(wf=ra[s["wf"]], seq=s["seq"], pe=s["pe"], time=s["time"]))
        sub = {k: np.concatenate([q[k] for q in parts]) for k in ("wf", "seq", "pe", "time")}
        if counts is not None:
            counts["pulse_device_refit"] = counts.get("pulse_device_refit", 0) + int(nredo - again.sum())
        keep = ~redo[wf]
        wf = np.concatenate([wf[keep], rr[sub["wf"]]])
        seq = np.concatenate([seq[keep], sub["seq"]])
        pe = np.concatenate([pe[keep], sub["pe"]])
        tm = np.concatenate([tm[keep], sub["time"]])
        o = np.lexsort((seq, wf))
        wf, seq, pe, tm = wf[o], seq[o], pe[o], tm[o]
        nredo = int(again.sum())
    elif nredo:
        rr = np.flatnonzero(redo)
        g = rows[rr]
        sub = pulsefit.reconstruct_full(W["adc"][g], W["n"][g], W["start"][g], W["ped"][g].astype(float),
                                   W["area"][g].astype(float), W["height"][g].astype(float), cfg_np)
        keep = ~redo[wf]
        wf = np.concatenate([wf[keep], rr[sub["wf"]]])
        seq = np.concatenate([seq[keep], sub["seq"]])
        pe = np.concatenate([pe[keep], sub["pe"]])
        tm = np.concatenate([tm[keep], sub["time"]])
        o = np.lexsort((seq, wf))
        wf, seq, pe, tm = wf[o], seq[o], pe[o], tm[o]
    return rows[wf], seq, pe, tm, nredo


def _deferrable(h):
    """A find_async handle whose promotions can wait for the other chunks (coinc_jax.collect / complete_all)."""
    return isinstance(h, dict) and "jobs" in h and hasattr(CJ, "collect")


def run(W, G, status, n_chunks=8, rounds=3, batch=65536, L=12, coinc_cfg=C.FinderConfig(), timers=None,
        keep_pulses=False, with_pos=False, add_pulses=None, pulse_cfg=None, device_refit=False, capacities=None):
    """W: waveforms sorted by (ev, channel, time) with pulsefit inputs and ev, bar, sipm; optional "toff" (per-waveform
    DB time offset [ns], added to the pulse times as CrvRecoPulsesFinder does with useTimeOffsetDB; offspill TDC0time
    is 0). Optional "event_t0" is a per-waveform bunch correction added independently of DB calibration.
    Returns per-chunk cluster outputs (list) and timers (pulse_dispatch_s, pulse_collect_s, coinc_async_s,
    coinc_finish_s, wall_s). keep_pulses: each chunk output also carries its flat pulses under "pulses" (ev, idx, bar,
    sipm, seq, pe, time; idx is the member index mem_idx refers to). with_pos: clusters carry cl_pos [n, 3].
    device_refit: flagged waveforms are refit on the device at full length first (_device_refit), numpy only after.
    capacities (crvfastsim.capacity): fixed refit shapes and coincidence hit buffers, and both pulse fits with bounded
    scans (the peak scans stop at the longest waveform, the rounds once no waveform found a pulse: the same pulses).
    add_pulses: callable(events) -> extra pulses (ev, bar, sipm, pe, time, seq) merged before the coincidence finder
    (reco.background: out-of-time light deposits the MC does not simulate)."""
    timers = timers if timers is not None else {}
    if not len(W['ev']):
        timers.update(events=0, waveforms=0, wall_s=0.0)
        return [], timers
    cfg_np = pulse_cfg if pulse_cfg is not None else pulsefit.RecoConfig()
    cfg = dataclasses.replace(cfg_np, max_pulses=rounds)
    params, tables = CJ.type_params(G, coinc_cfg), CJ.hit_tables(G, status, coinc_cfg)
    ue = np.unique(W["ev"])
    edges = np.searchsorted(W["ev"], ue[np.linspace(0, len(ue), n_chunks + 1).astype(int)[:-1]])
    edges = np.r_[edges, len(W["ev"])]
    chunks = [np.arange(edges[i], edges[i + 1]) for i in range(n_chunks) if edges[i + 1] > edges[i]]
    pul, coin, outs, kept, pulses = {}, {}, [], {}, {}
    tic = lambda: _time.perf_counter()
    add = lambda k, t0: timers.__setitem__(k, timers.get(k, 0.0) + tic() - t0)
    t_all = tic()
    for k in range(len(chunks) + 2):
        if k < len(chunks):
            t0 = tic()
            pul[k] = _pulse_dispatch(W, chunks[k], cfg, batch, L, **({"bounded": True} if capacities is not None else {}))
            add("pulse_dispatch_s", t0)
        if 0 <= k - 1 < len(chunks):
            t0 = tic()
            wrow, seq, pe, tm, nredo = (_pulse_collect(pul.pop(k - 1), W, chunks[k - 1], cfg_np, True, timers,
                                                       capacities)
                                        if device_refit else _pulse_collect(pul.pop(k - 1), W, chunks[k - 1], cfg_np))
            if cfg_np.ignore_channels:
                keep = (status[W['bar'][wrow] * 4 + W['sipm'][wrow]] & 0b010) == 0
                wrow, seq, pe, tm = wrow[keep], seq[keep], pe[keep], tm[keep]
            if "toff" in W and cfg_np.use_time_offset_db:
                tm = tm + W["toff"][wrow] * cfg_np.time_offset_scale
            if "event_t0" in W:  # bunch correction is independent of DB calibration enable/scale
                tm = tm + W["event_t0"][wrow]
            timers["pulse_numpy_redo"] = timers.get("pulse_numpy_redo", 0) + nredo
            ev = W["ev"][wrow]
            idx = np.arange(len(ev)) - np.searchsorted(ev, ev)
            bar, sipm = W["bar"][wrow], W["sipm"][wrow]
            if add_pulses is not None:
                from .background import merge_pulses
                B = add_pulses(np.unique(W["ev"][chunks[k - 1]]))
                ev, idx, bar, sipm, tm, pe, seq, _ = merge_pulses(ev, idx, bar, sipm, tm, pe, seq, B)
                timers["background_pulses"] = timers.get("background_pulses", 0) + len(B["ev"])
            add("pulse_collect_s", t0)
            t0 = tic()
            coin[k - 1] = CJ.find_async(ev, idx, bar, sipm, tm, pe, seq, G, status, coinc_cfg,
                                        params=params, tables=tables, timers=timers, with_pos=with_pos,
                                        **({"caps": capacities} if capacities is not None else {}))
            add("coinc_async_s", t0)
            timers["pulses"] = timers.get("pulses", 0) + len(ev)
            if keep_pulses:
                kept[k - 1] = dict(ev=ev, idx=idx, bar=bar, sipm=sipm, seq=seq, pe=pe, time=tm)
        if 0 <= k - 2 < len(chunks):
            t0 = tic()
            h = coin.pop(k - 2)
            if _deferrable(h):   # first-choice tiers now; promotions of all chunks together after the loop
                outs.append(CJ.collect(h))
            else:
                outs.append(CJ.finish(h))
            if keep_pulses:
                pulses[len(outs) - 1] = kept.pop(k - 2)
            add("coinc_finish_s", t0)
    t0 = tic()
    deferred = [i for i, o in enumerate(outs) if not isinstance(o, dict)]
    for i, o in zip(deferred, CJ.complete_all([outs[i] for i in deferred]) if deferred else []):
        outs[i] = o
    for i, p in pulses.items():
        outs[i]["pulses"] = p
    add("coinc_finish_s", t0)
    timers["wall_s"] = tic() - t_all
    timers["events"] = int(len(ue))
    timers["waveforms"] = int(len(W["ev"]))
    return outs, timers
