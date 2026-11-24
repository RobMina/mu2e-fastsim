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

Fixed-size products (``EventWindowMarker``: 22 bytes, ``_spillType`` int32 at 10; ``IntensityInfoTimeCluster``:
8 bytes, the count uint16 at 6) are decoded from their big-endian bytes too: uproot's strided interpretation of these
classes reads their members with the wrong byte order.
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


def _fixed(branch, size):
    """Concatenated entries of a fixed-size object branch; each starts with its bytecount (size - 4)."""
    data = np.concatenate([np.frombuffer(bytes(branch.basket(i).data), np.uint8) for i in range(branch.num_baskets)])
    n = int(branch.num_entries)
    if data.size != n * size:
        raise ValueError(f"{branch.name}: {data.size} bytes for {n} entries of {size}")
    at = np.arange(n) * size
    if np.any((_be(data, at, ">u4") & 0x3FFFFFFF) != size - 4):
        raise ValueError(f"{branch.name}: unexpected bytecount")
    return data, at


def decode_spill(branch):
    data, at = _fixed(branch, 22)
    return _be(data, at + 10, ">i4").astype(np.int64)


def decode_intensity_tc(branch):
    """IntensityInfoTimeCluster: the time-cluster count of each event."""
    data, at = _fixed(branch, 8)
    return _be(data, at + 6, ">u2").astype(np.int64)


def stage_passed(word, position, n_modules):
    """Events that got past the module at `position` of a path of `n_modules` (the expanded configuration's list),
    from the HLTPathStatus words of that path. art's index runs one ahead of the listed sequence (a passed path ends at
    index n_modules); the offset is taken from the passed events and must be 0 or 1."""
    state, index = path_status(word)
    passed = state == 1
    off = set((index[passed] - (n_modules - 1)).tolist())
    if len(off) > 1 or not off <= {0, 1}:
        raise ValueError(f"unexpected path index offset {sorted(off)}")
    off = off.pop() if off else 1
    return passed | (state == 0) | (index > position + off)


HELIX_FIELDS = ("rcent", "fcent", "radius", "lambda", "fz0", "chi2xy", "chi2zphi")


def decode_helix_seeds(data, offsets):
    """Stock HelixSeeds (memberwise: header 16, then per member over the n seeds: ``_t0`` 26 bytes each (two doubles),
    ``_hhits`` (a bytecounted ComboHitCollection each; its ComboHits memberwise, ``_pos`` first, 32 bytes per hit with
    x, y, z float at 20), ``_helix`` 52 bytes each (7 floats at 10, helicity int at 48), the rest skipped).
    Returns per event a list of seeds {t0, helicity, rcent ... chi2zphi, pos [nh, 3]}."""
    out = []
    bc = lambda at: int(_be(data, np.array([at]), ">u4")[0] & 0x3FFFFFFF)
    for s, e in zip(offsets[:-1], offsets[1:]):
        s, e = int(s), int(e)
        if bc(s) + 4 != e - s or int(_be(data, np.array([s + 4]), ">u2")[0]) != 0x400A:
            raise ValueError("unexpected HelixSeed collection header")
        n = int(_be(data, np.array([s + 12]), ">i4")[0])
        at = s + 16
        seeds = [{"t0": float(_be(data, np.array([at + 26 * k + 10]), ">f8")[0])} for k in range(n)]
        at += 26 * n
        for k in range(n):
            b, inner = bc(at), at + 10
            if int(_be(data, np.array([inner + 4]), ">u2")[0]) != 0x400A:
                raise ValueError("unexpected ComboHitCollection layout")
            nh = int(_be(data, np.array([inner + 12]), ">i4")[0])
            h0 = inner + 16 + 32 * np.arange(nh)
            if nh and np.any(_be(data, h0, ">u4") != 0x4000001C):
                raise ValueError("unexpected ComboHit position layout")
            seeds[k]["pos"] = np.stack([_be(data, h0 + 20 + 4 * j, ">f4") for j in range(3)], 1) if nh else np.zeros((0, 3), np.float32)
            at += 4 + b
        for k in range(n):
            if bc(at) != 48:
                raise ValueError("unexpected RobustHelix layout")
            for j, name in enumerate(HELIX_FIELDS):
                seeds[k][name] = float(_be(data, np.array([at + 10 + 4 * j]), ">f4")[0])
            seeds[k]["helicity"] = int(_be(data, np.array([at + 48]), ">i4")[0])
            at += 52
        out.append(seeds)
    return out


def decode_time_clusters(data, offsets):
    """Stock TimeClusters (memberwise: header 16, then per member over the n clusters: ``_strawHitIdxs`` (one
    bytecounted block for all n, version 2 bytes, then per cluster a count and its uint16 indices), ``_t0`` 26 bytes
    each (t0 and its error, doubles at 10 and 18),
    ``_pos`` 32 bytes each (3 floats at 20), ``_nsh`` 2 bytes each (uint16), ``_caloCluster`` 36 bytes each (an
    art::Ptr: its key, the cluster index, as a uint64 at 28; all ones for a null Ptr)). Returns per event a list of
    clusters {hits (the ComboHit indices, in stock order), t0, t0err, pos [3], nsh, calo (the index, or -1)}."""
    out = []
    bc = lambda at: int(_be(data, np.array([at]), ">u4")[0] & 0x3FFFFFFF)
    for s, e in zip(offsets[:-1], offsets[1:]):
        s, e = int(s), int(e)
        if bc(s) + 4 != e - s or int(_be(data, np.array([s + 4]), ">u2")[0]) != 0x400A:
            raise ValueError("unexpected TimeCluster collection header")
        n = int(_be(data, np.array([s + 12]), ">i4")[0])
        at = s + 16
        tcs = [{} for _ in range(n)]
        if n:
            end = at + 4 + bc(at)
            at += 6
            for k in range(n):
                m = int(_be(data, np.array([at]), ">i4")[0])
                tcs[k]["hits"] = _be(data, at + 4 + 2 * np.arange(m), ">u2").astype(np.int64) if m else np.zeros(0, np.int64)
                at += 4 + 2 * m
            if at != end:
                raise ValueError("unexpected TimeCluster hit-index layout")
        for k in range(n):
            if bc(at) != 22:
                raise ValueError("unexpected TrkT0 layout")
            tcs[k]["t0"] = float(_be(data, np.array([at + 10]), ">f8")[0])
            tcs[k]["t0err"] = float(_be(data, np.array([at + 18]), ">f8")[0])
            at += 26
        for k in range(n):
            if bc(at) != 28 or bc(at + 10) != 18:
                raise ValueError("unexpected TimeCluster position layout")
            tcs[k]["pos"] = _be(data, at + 20 + 4 * np.arange(3), ">f4").astype(np.float32)
            at += 32
        for k in range(n):
            tcs[k]["nsh"] = int(_be(data, np.array([at]), ">u2")[0])
            at += 2
        for k in range(n):
            if bc(at) != 32:
                raise ValueError("unexpected art::Ptr layout")
            key = int(_be(data, np.array([at + 28]), ">u8")[0])
            tcs[k]["calo"] = -1 if key == 0xFFFFFFFFFFFFFFFF else key
            at += 36
        if at != e:
            raise ValueError("TimeCluster collection length differs from its members")
        out.append(tcs)
    return out


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
        ev["spill"] = decode_spill(t[EWM.format(label=ewm) + ".obj"])
        words = decode_trigger(*_blobs(t[TRIGGER.format(process=process)])) if process else None
    del ev["flag"]
    keys = [tuple(int(x) for x in r) for r in ids]
    return keys, ev, words
