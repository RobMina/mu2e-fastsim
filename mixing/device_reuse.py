"""Donor-frame digi reuse for the resident tracker (an approximation; on in the W6 default capacity file
`run1a_mix1bb_batch256.json`, off in `run1a_mix1bb_batch256_exact.json` and the batch-64/128 files).

Most hit straws of a mixed event see steps of one donor frame only. With reuse, every donor frame of the pools is
digitised once at setup, as an event of its own at the signal events' mean time shift (mc_pbtime). A mixed batch
then
- digitises with the resident tracker only the straws hit by the primary or by two or more sources;
- takes the setup digis of each straw hit by a single donor frame, shifts their TDC by the event's time shift less
  the mean (about +-10 ns),
  applies the digitisation window after the shift, and keeps their TOT, ADC waveform and PMP.

What it freezes, per donor frame: the threshold, TDC and ADC noise draws, the ROC jitter and the ADC sampling phase
(the waveform does not move against the 20 ns ADC clock when the event shifts the frame). The setup window is widened
by the largest shift, so a digi that an event's shift moves into the real window is kept. Validate against stock
before use.
"""
from types import SimpleNamespace

import numpy as np

SRC_BITS = 22          # donor-frame code (global frame index + 1, 0 = primary) in the straw-source sort key


class ReuseCapacities(SimpleNamespace):
    """compute: tracker capacities for the straws digitised per batch; precompute: tracker capacities for the setup
    batches of donor frames; frames_per_batch: donor frames per setup batch; straws: reused straws per mixed batch;
    streams: the donor streams whose frames are reused (the others are digitised per batch: a small pool drawn many
    times per pass would repeat identical digis)."""
    def __init__(self, compute, precompute, frames_per_batch, straws, precompute_rows=None, streams=None):
        from .sampling import STREAMS
        streams = list(STREAMS) if streams is None else list(streams)
        if not streams or set(streams) - set(STREAMS):
            raise ValueError(f"tracker_reuse streams must be a non-empty subset of {list(STREAMS)}")
        if type(frames_per_batch) is not int or not 0 < frames_per_batch < 32768:
            raise ValueError("tracker_reuse frames_per_batch must be an integer in (0, 32768)")
        if type(straws) is not int or straws < 1:
            raise ValueError("tracker_reuse straws must be a positive integer")
        if precompute_rows is not None and (type(precompute_rows) is not int or precompute_rows < 1):
            raise ValueError("tracker_reuse precompute_rows must be a positive integer")
        super().__init__(compute=compute, precompute=precompute, frames_per_batch=frames_per_batch, straws=straws,
                         precompute_rows=precompute_rows, streams=streams)

    def reused_pools(self):
        """Pool indices (1-4, after the signal pool 0) of the reused streams."""
        from .sampling import STREAMS
        return [1 + STREAMS.index(name) for name in self.streams]


def _sid(columns):
    import jax.numpy as jnp
    if {"plane", "panel", "straw"} <= columns.keys():
        return ((columns["plane"].astype(jnp.int64) << 10) | (columns["panel"].astype(jnp.int64) << 7)
                | columns["straw"].astype(jnp.int64))
    return columns["sid"].astype(jnp.int64)


def widened(c, margin):
    """Conditions with the digitisation window (and so the step and crossing windows) widened by `margin` ns on each
    side: the setup keeps the digis that a mixed event's time shift moves into the real window."""
    from trkfastsim.digi.digicond import Conditions
    s = dict(c.s)
    s["digitizationStart"] -= margin
    s["digitizationEnd"] += margin
    ticks = int(np.ceil(margin / c.TDCLSB))
    s["digitizationStartTDC"] = int(c.digitizationStartTDC) - ticks
    s["digitizationEndTDC"] = int(c.digitizationEndTDC) + ticks
    return Conditions(s, c.a)


def precompute(c, capacities, data, length, seed, *, rows_capacity, shift=0.0, margin=0.0, log=None):
    """Digitise every donor frame with tracker steps once, as its own event at the time shift `shift` (the mean
    mc_pbtime of the signal events; each mixed event then moves the digis by its own mc_pbtime - shift), with the
    digitisation window widened by `margin` ns (the real window is applied after each event's shift).

    data: pack_source output (host arrays; the signal pool first, then the four donor pools). Returns the reuse
    table sorted by (frame, sid, analog time): frame (global pool index), sid, tdc [n, 2], tot [n, 2], adc [n, S],
    pmp, t_analog [n, 2]."""
    import time
    import jax
    from . import device_tracker
    caps = ReuseCapacities(**capacities)
    F = caps.frames_per_batch
    model = device_tracker.prepare(widened(c, margin) if margin else c, caps.precompute, F)
    assets = jax.device_put(model.assets)
    fn = jax.jit(model.apply_with_usage)
    trk = data["collections"]["tracker"]
    offsets, columns = trk["offsets"], trk["columns"]
    starts, sizes = data["starts"], data["sizes"]
    frames = np.concatenate([np.arange(starts[k], starts[k] + sizes[k]) for k in caps.reused_pools()])
    nrows = offsets[frames + 1] - offsets[frames]
    frames, nrows = frames[nrows > 0], nrows[nrows > 0]
    budget = caps.precompute_rows or rows_capacity
    if np.any(nrows > budget):
        raise ValueError("a donor frame exceeds the reuse precompute row capacity")
    # greedy batches: at most F frames and `budget` rows each, in pool order
    batches, start = [], 0
    while start < len(frames):
        cum = np.cumsum(nrows[start:start + F])
        stop = start + int(np.searchsorted(cum, budget, side="right"))
        batches.append((start, stop))
        start = stop
    timing = {"mc_pbtime": np.full(F, float(shift)), "event_length": np.full(F, float(length) + margin), "event_mask": np.ones(F, bool)}
    key = jax.random.PRNGKey(int(seed))
    parts, t0 = [], time.perf_counter()
    for b, (lo, hi) in enumerate(batches):
        fr = frames[lo:hi]
        src = np.concatenate([np.arange(offsets[f], offsets[f + 1]) for f in fr])
        ev = np.repeat(np.arange(len(fr)), nrows[lo:hi])
        pad = budget - len(src)
        sentinel = len(next(iter(columns.values()))) - 1
        rows = np.r_[src, np.full(pad, sentinel)]
        collection = {"columns": {name: values[rows] for name, values in columns.items()},
                      "mask": np.arange(budget) < len(src), "event": np.r_[ev, np.zeros(pad, ev.dtype)]}
        out, errors, _ = jax.device_get(fn(collection, timing, assets, jax.random.fold_in(key, b)))
        bad = [name for name, flag in errors.items() if np.any(flag)]
        if bad:
            raise RuntimeError(f"donor reuse precompute batch {b}: capacity or input errors {bad}")
        m = np.asarray(out["mask"])
        parts.append({"frame": fr[np.asarray(out["ev"])[m]], "sid": np.asarray(out["sid"])[m],
                      "tdc": np.asarray(out["tdc"])[m], "tot": np.asarray(out["tot"])[m],
                      "adc": np.asarray(out["adc"])[m], "pmp": np.asarray(out["pmp"])[m],
                      "t_analog": np.asarray(out["t_analog"])[m]})
    table = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
    order = np.lexsort((table["t_analog"][:, 0], table["sid"], table["frame"]))
    table = {name: value[order] for name, value in table.items()}
    table["shift"] = np.float64(shift)
    if log is not None:
        log(f"donor reuse precompute: {len(frames)} frames in {len(batches)} batches, {len(order)} digis, "
            f"{time.perf_counter() - t0:.1f} s")
    return table


def prepare(c, capacities, batch_events, table, starts):
    """Per-batch reuse apply around a reduced resident tracker (capacities["compute"])."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from . import device_tracker
    caps = ReuseCapacities(**capacities)
    compute = device_tracker.prepare(c, caps.compute, batch_events)
    pairs = compute.capacities.pairs_per_straw
    R = caps.straws
    sentinel = np.iinfo(np.int64).max
    key = (table["frame"].astype(np.int64) << 16) | table["sid"].astype(np.int64)
    if np.any(np.diff(key) < 0):
        raise ValueError("reuse table must be sorted by (frame, sid)")
    assets = {"compute": compute.assets, "table": {"key": key, **{k: v for k, v in table.items() if k in
                                                                   ("tdc", "tot", "adc", "pmp", "t_analog")}},
              "all_planes": np.asarray(c.allPlanes, np.int64), "shift": np.float64(table.get("shift", 0.0)),
              "starts": np.asarray(starts, np.int64),
              "reused": np.isin(np.arange(len(starts)), caps.reused_pools())}
    start_tdc, end_tdc = int(c.digitizationStartTDC), int(c.digitizationEndTDC)

    def apply(collection, timing, assets, key):
        st, mask, ev = collection["columns"], collection["mask"], collection["event"].astype(jnp.int64)
        occ = collection["occurrence"]
        sid = _sid(st)
        straw = jnp.where(mask, ev * 65536 + sid, -1)
        pool = jnp.searchsorted(assets["starts"], jnp.maximum(occ, 0), side="right") - 1
        # a step of the primary or of a stream not reused counts as the primary: its straw is digitised here
        code = jnp.where((occ >= 0) & assets["reused"][pool], occ + 1, 0)
        k = jnp.where(mask, (straw << SRC_BITS) | code, sentinel)
        order = jnp.argsort(k)
        ks = k[order]
        live = ks != sentinel
        straw_s = ks >> SRC_BITS          # sorted; dead entries (sentinel) sort last
        code_s = ks & ((1 << SRC_BITS) - 1)
        lo = jnp.searchsorted(straw_s, straw_s, side="left")
        hi = jnp.searchsorted(straw_s, straw_s, side="right") - 1
        single = live & (code_s[lo] == code_s[jnp.maximum(hi, 0)]) & (code_s[lo] > 0)
        reuse_step = jnp.zeros_like(mask).at[order].set(single)
        out, errors, usage = compute.apply_with_usage({**collection, "mask": mask & ~reuse_step}, timing,
                                                      assets["compute"], key)
        # one entry per reused straw: the first sorted step of each single-donor straw
        head = single & (jnp.arange(len(ks)) == lo)
        nreuse = head.sum()
        pos = jnp.nonzero(head, size=R, fill_value=0)[0]
        rlive = jnp.arange(R) < nreuse
        rstraw, rframe = straw_s[pos], code_s[pos] - 1
        rev, rsid = rstraw // 65536, rstraw % 65536
        T = assets["table"]
        tkey = (rframe << 16) | rsid
        tlo = jnp.searchsorted(T["key"], tkey, side="left")
        thi = jnp.searchsorted(T["key"], tkey, side="right")
        slot = tlo[:, None] + jnp.arange(pairs)
        valid = rlive[:, None] & (slot < thi[:, None])
        slot = jnp.clip(slot, 0, len(T["key"]) - 1).reshape(-1)
        valid = valid.reshape(-1)
        rev_p, rsid_p = jnp.repeat(rev, pairs), jnp.repeat(rsid, pairs)
        evc = jnp.clip(rev_p, 0, batch_events - 1)
        shift = timing["mc_pbtime"][evc] - assets["shift"]   # relative to the precompute's time shift
        tdc = T["tdc"][slot] + jnp.round(shift / c.TDCLSB).astype(jnp.int64)[:, None]
        tdc = jnp.clip(tdc, 0, c.maxTDC)
        readall = ((rsid_p & 127) >= c.allStraw) & jnp.isin(rsid_p >> 10, assets["all_planes"])
        ewend = jnp.clip(jnp.floor(timing["event_length"][evc] / c.TDCLSB), 0, c.maxTDC).astype(jnp.int64)
        upper = end_tdc if c.onspill else ewend[:, None]
        ok_norm = jnp.all((tdc > start_tdc) & (tdc < upper), 1)
        ok_all = jnp.all((tdc > 0) & (tdc < ewend[:, None]), 1)
        keep = valid & jnp.where(readall, ok_all, ok_norm) & timing["event_mask"][evc]
        reused = {"adc": T["adc"][slot].astype(out["adc"].dtype), "tdc": tdc.astype(out["tdc"].dtype),
                  "tot": T["tot"][slot].astype(out["tot"].dtype), "pmp": T["pmp"][slot].astype(out["pmp"].dtype),
                  "t_analog": (T["t_analog"][slot] + shift[:, None]).astype(out["t_analog"].dtype),
                  "mask": keep, "ev": rev_p.astype(out["ev"].dtype), "sid": rsid_p.astype(out["sid"].dtype)}
        names = [name for name in reused if name in out]
        merged = {name: jnp.concatenate([out[name], reused[name]]) for name in names}
        o = compute.output_order(merged["t_analog"][:, 0], merged["sid"],
                                 jnp.where(merged["mask"], merged["ev"], sentinel))
        merged = {name: value[o] for name, value in merged.items()}
        errors = {**errors, "reuse_straws": nreuse > R}
        usage = {**usage, "reuse_straws": nreuse}
        return merged, errors, usage

    def apply_plain(collection, timing, assets, key):
        output, errors, _ = apply(collection, timing, assets, key)
        return output, errors

    return SimpleNamespace(assets=assets, apply=apply_plain, apply_with_usage=apply, capacities=caps,
                           compute=compute,
                           metadata={**compute.metadata, "approximation": "donor-frame digi reuse"})
