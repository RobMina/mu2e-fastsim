"""Pulses -> coincidences on the device in one pipelined pass (the R1 measurement).

Events are processed in chunks. For chunk k the jitted pulse reco is launched (asynchronously); meanwhile the host
collects chunk k-1's pulses, builds the coincidence-finder input (find_async: split, pack, launch) and finishes chunk
k-2 (collects clusters, promotions, numpy fallback). On a GPU the host work of one chunk overlaps the device work of
the next; on CPU the stages simply run in turn. Waveforms longer than the kernel input, whose fit used all kernel
rounds or overflowed its window go to full-length numpy pulse reco (as in reco-vr2jax).
"""
import dataclasses
import os
import time as _time

import numpy as np

from . import coinc as C
from . import coinc_jax as CJ
from . import pulsefit, pulsefit_jax as PJ

PULSE_KEYS = ("found", "pe", "time", "rounds_exhausted", "window_overflow", "input_truncated")


def _pulse_dispatch(W, rows, cfg, batch, L):
    import jax.numpy as jnp
    outs = CJ._DispatchOutputs()
    if os.environ.get('CRVFASTSIM_PVC_RECO') == '1' and CJ.is_oneapi():
        batch = min(batch, 4096)
    for a in range(0, len(rows), batch):
        r = rows[a:a + batch]
        n = len(r)
        # Keep pulse-fit shapes fixed: changing their row capacity can change fit rounding.
        pad = lambda x, v: np.concatenate([x[r], np.full((batch - n,) + x.shape[1:], v, x.dtype)]) if n < batch else x[r]
        d = PJ.reconstruct_dense(jnp.asarray(pad(W["adc"][:, :L], 0)), jnp.asarray(pad(np.minimum(W["n"], L), 0)),
                                 jnp.asarray(pad(W["start"], 0)), jnp.asarray(pad(W["ped"].astype(float), 0.0)),
                                 jnp.asarray(pad(W["area"].astype(float), 1.0)),
                                 jnp.asarray(pad(W["height"].astype(float), 1.0)), cfg=cfg)
        d = {**d, 'input_truncated': pad(W['n'] > L, False)}
        outs.append(({k: d[k] for k in PULSE_KEYS}, n))
    return outs


def _pulse_collect(outs, W, rows, cfg_np):
    """Flat pulses (waveform row, seq, pe, time) in (waveform, seq) order; flagged waveforms redone in numpy."""
    if getattr(outs, "compact", False):
        host = CJ._real_rows_to_host(outs, PULSE_KEYS)
        d = {k: np.concatenate([o[k] for o in host]) for k in PULSE_KEYS}
    else:
        d = {k: np.concatenate([np.asarray(o[k])[:n] for o, n in outs]) for k in PULSE_KEYS}
    wf, seq = np.nonzero(d["found"])
    redo = d["rounds_exhausted"] | d["window_overflow"] | d["input_truncated"]
    pe, tm = d["pe"][wf, seq], d["time"][wf, seq]
    nredo = int(redo.sum())
    if nredo:
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


def run(W, G, status, n_chunks=8, rounds=3, batch=65536, L=12, coinc_cfg=C.FinderConfig(), timers=None,
        keep_pulses=False, with_pos=False, add_pulses=None, pulse_cfg=None):
    """W: waveforms sorted by (ev, channel, time) with pulsefit inputs and ev, bar, sipm; optional "toff" (per-waveform
    DB time offset [ns], added to the pulse times as CrvRecoPulsesFinder does with useTimeOffsetDB; offspill TDC0time
    is 0). Returns per-chunk cluster outputs (list) and timers (pulse_dispatch_s, pulse_collect_s, coinc_async_s,
    coinc_finish_s, wall_s). keep_pulses: each chunk output also carries its flat pulses under "pulses" (ev, idx, bar,
    sipm, seq, pe, time; idx is the member index mem_idx refers to). with_pos: clusters carry cl_pos [n, 3].
    add_pulses: callable(events) -> extra pulses (ev, bar, sipm, pe, time, seq) merged before the coincidence finder
    (reco.background: out-of-time light deposits the MC does not simulate).
    CRVFASTSIM_COMPACT_DISPATCH=1 opts into real-row-only grouped transfers and compact coincidence
    row batches. Pulse-fit batch shapes stay fixed to preserve fit arithmetic."""
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
    pul, coin, outs, kept = {}, {}, [], {}
    tic = lambda: _time.perf_counter()
    add = lambda k, t0: timers.__setitem__(k, timers.get(k, 0.0) + tic() - t0)
    t_all = tic()
    for k in range(len(chunks) + 2):
        if k < len(chunks):
            t0 = tic()
            pul[k] = _pulse_dispatch(W, chunks[k], cfg, batch, L)
            add("pulse_dispatch_s", t0)
        if 0 <= k - 1 < len(chunks):
            t0 = tic()
            wrow, seq, pe, tm, nredo = _pulse_collect(pul.pop(k - 1), W, chunks[k - 1], cfg_np)
            if cfg_np.ignore_channels:
                keep = (status[W['bar'][wrow] * 4 + W['sipm'][wrow]] & 0b010) == 0
                wrow, seq, pe, tm = wrow[keep], seq[keep], pe[keep], tm[keep]
            if "toff" in W and cfg_np.use_time_offset_db:
                tm = tm + W["toff"][wrow] * cfg_np.time_offset_scale
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
                                        params=params, tables=tables, timers=timers, with_pos=with_pos)
            add("coinc_async_s", t0)
            timers["pulses"] = timers.get("pulses", 0) + len(ev)
            if keep_pulses:
                kept[k - 1] = dict(ev=ev, idx=idx, bar=bar, sipm=sipm, seq=seq, pe=pe, time=tm)
        if 0 <= k - 2 < len(chunks):
            t0 = tic()
            outs.append(CJ.finish(coin.pop(k - 2)))
            if keep_pulses:
                outs[-1]["pulses"] = kept.pop(k - 2)
            add("coinc_finish_s", t0)
    timers["wall_s"] = tic() - t_all
    timers["events"] = int(len(ue))
    timers["waveforms"] = int(len(W["ev"]))
    return outs, timers
