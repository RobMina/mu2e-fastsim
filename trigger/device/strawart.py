"""Read stock StrawDigis, the proton bunch time and the trigger path status straight from an art file (uproot + NumPy),
in the layout `run.load_dump` returns, so the tracker chain runs on stored stock digis without a dumper job.

``StrawDigis`` are written memberwise in an unsplit branch: header 16 (bytecount 4, collection version 2, class
version 0, checksum 4, n 4), then, member by member over the n digis:

- ``_strawid``: n x 12 bytes (bytecount 4, version 0, checksum 4, ``_sid`` uint16);
- ``_tdc[2]``: n x 2 uint32; ``_tot[2]``: n x 2 uint16;
- ``_flag``: n x 21 bytes (BitMap: bytecount, version 0, checksum, the empty base class (bytecount, version 0,
  checksum), ``_value`` uint8);
- ``_pmp``: n x uint16.

So a blob is 16 + 47 n bytes; the decoder checks every length and header.
"""
from __future__ import annotations

import numpy as np

from .caloart import PBT, TRIGGER, _be, _blobs, _bytecount, decode_event_ids, decode_pbt, decode_trigger

DIGIS = "mu2e::StrawDigis_{label}.obj"
EWM = "mu2e::EventWindowMarker_{label}"
PER_DIGI = 47


def decode_straw_digis(data, offsets):
    """Per-event StrawDigi columns (lists of arrays): sid, tdc0, tdc1, tot0, tot1, pmp, flag."""
    start, stop = offsets[:-1], offsets[1:]
    _bytecount(data, start, stop)
    if np.any(_be(data, start + 4, ">u2") != 0x400A) or np.any(_be(data, start + 6, ">u2") != 0):
        raise ValueError("unexpected StrawDigi collection or class version")
    n = _be(data, start + 12, ">i4").astype(np.int64)
    if np.any(stop - start != 16 + PER_DIGI * n):
        raise ValueError("unexpected StrawDigi blob length")
    cols = {k: [] for k in ("sid", "tdc0", "tdc1", "tot0", "tot1", "pmp", "flag")}
    for e in range(n.size):
        k, at = int(n[e]), int(start[e]) + 16
        if k and np.any(_be(data, at + 12 * np.arange(k), ">u4") != 0x40000008):
            raise ValueError(f"unexpected StrawId layout (entry {e})")
        cols["sid"].append(_be(data, at + 12 * np.arange(k) + 10, ">u2").astype(np.int64)); at += 12 * k
        tdc = _be(data, at + 4 * np.arange(2 * k), ">u4").astype(np.int64).reshape(k, 2); at += 8 * k
        tot = _be(data, at + 2 * np.arange(2 * k), ">u2").astype(np.int64).reshape(k, 2); at += 4 * k
        if k and np.any(_be(data, at + 21 * np.arange(k), ">u4") != 0x40000011):
            raise ValueError(f"unexpected StrawDigi flag layout (entry {e})")
        cols["flag"].append(data[at + 21 * np.arange(k) + 20].astype(np.int64)); at += 21 * k
        cols["pmp"].append(_be(data, at + 2 * np.arange(k), ">u2").astype(np.int64)); at += 2 * k
        if at != stop[e]:
            raise ValueError(f"unexpected StrawDigi blob layout (entry {e})")
        cols["tdc0"].append(tdc[:, 0]); cols["tdc1"].append(tdc[:, 1])
        cols["tot0"].append(tot[:, 0]); cols["tot1"].append(tot[:, 1])
    return cols


def path_status(words):
    """HLTPathStatus words -> (state, index): state 0 ready, 1 pass, 2 fail, 3 exception; index = the path's module
    that decided (the last module run)."""
    w = np.asarray(words).astype(np.int64)
    return w & 3, w >> 2


def read(path, digis="makeSD__Digitize", pbt="EWMProducer__Digitize", ewm="EWMProducer__Digitize",
         process="Digitize"):
    """keys [(run, subrun, event)], ev (the `run.load_dump` digi columns, pbt, spill), and the TriggerResults words
    of `process` (None without one), in entry order."""
    import uproot
    with uproot.open(path) as f:
        t = f["Events"]
        ids = decode_event_ids(t["EventAuxiliary"])
        ev = decode_straw_digis(*_blobs(t[DIGIS.format(label=digis)]))
        ev["pbt"] = decode_pbt(*_blobs(t[PBT.format(label=pbt)])).astype(np.float64)
        obj = t[EWM.format(label=ewm) + ".obj"].array(library="np")      # strided objects: one per event
        ev["spill"] = np.array([int(o.member("_spillType")) for o in obj], np.int64)
        words = decode_trigger(*_blobs(t[TRIGGER.format(process=process)])) if process else None
    del ev["flag"]
    keys = [tuple(int(x) for x in r) for r in ids]
    return keys, ev, words
