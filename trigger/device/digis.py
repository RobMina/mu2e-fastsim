"""Flat digi arrays (one row per StrawDigi, as `trkfastsim.digi.digi_jax.digitize` returns them) -> the trigger's
dense event blocks, without a per-event Python loop.

Input columns: `ev` (event index 0..nev-1), `sid` (StrawId as uint16 value), `tdc` [N, 2] and `tot` [N, 2] (cal, hv),
`pmp`. Rows of one event keep their order, which must be the stock StrawDigi order (sorted by StrawId), as the
panel combine relies on it (CombineStrawHits with Unsorted false). Events without digis get an empty row, so the
event count and order are those of the input population.
"""
from __future__ import annotations

import numpy as np

from trkfastsim.reco.hitreco_jax import bucket


def check_sorted(ev, sid):
    """Events contiguous and ascending, StrawIds non-decreasing within each event."""
    ev, sid = np.asarray(ev), np.asarray(sid)
    if len(ev) and (np.any(np.diff(ev) < 0) or np.any((np.diff(ev) == 0) & (np.diff(sid) < 0))):
        raise ValueError("digis must be ordered by event, then StrawId (stock StrawDigi order)")


def blocks(dg, nev, size, events=None, pbt=None, spill=0, nh=None):
    """Yield (first event, events in block, block) for `chain.Chain.decide`, `size` events per block.

    events: event numbers (keys the response-layer draws), default 0..nev-1. pbt: proton bunch time per event
    (default 0, the off-spill and the device digitiser's on-spill default). spill: EventWindowMarker spill type
    (0 off-spill, 1 on-spill) for every event."""
    ev = np.asarray(dg["ev"], np.int64)
    sid = np.asarray(dg["sid"], np.int64)
    check_sorted(ev, sid)
    tdc, tot = np.asarray(dg["tdc"], np.int64), np.asarray(dg["tot"], np.int64)
    pmp = np.asarray(dg["pmp"], np.float64)
    count = np.bincount(ev, minlength=nev) if len(ev) else np.zeros(nev, np.int64)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    pos = np.arange(len(ev)) - start[ev]
    events = np.arange(nev, dtype=np.int64) if events is None else np.asarray(events, np.int64)
    pbt = np.zeros(nev) if pbt is None else np.asarray(pbt, np.float64)
    H = nh or bucket(int(count.max()) if nev and len(ev) else 1, 64)
    if nev and len(ev) and count.max() > H:
        raise ValueError(f"an event has {count.max()} digis, more than the block width {H}")
    for lo in range(0, nev, size):
        hi = min(nev, lo + size)
        m = (ev >= lo) & (ev < hi)
        r, c = ev[m] - lo, pos[m]
        b = {"valid": np.zeros((size, H), bool)}
        b["valid"][r, c] = True
        for k, v, dt in (("sid", sid, np.int64), ("tdc0", tdc[:, 0], np.int64), ("tdc1", tdc[:, 1], np.int64),
                         ("tot0", tot[:, 0], np.int64), ("tot1", tot[:, 1], np.int64), ("pmp", pmp, np.float64)):
            a = np.zeros((size, H), dt)
            a[r, c] = v[m]
            b[k] = a
        n = hi - lo
        b["pbt"] = np.zeros(size)
        b["pbt"][:n] = pbt[lo:hi]
        b["spill"] = np.full(size, spill, np.int64)
        b["valid_event"] = np.arange(size) < n
        b["event"] = np.zeros(size, np.int64)
        b["event"][:n] = events[lo:hi]
        yield lo, n, b
