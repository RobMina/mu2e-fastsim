"""Read ``CaloShowerStep``s straight from a stock art file (uproot + NumPy), and the stock on-spill event timing.

art writes ``std::vector<mu2e::CaloShowerStep>`` memberwise in one unsplit branch, which uproot cannot interpret.
Each event's blob is decoded here from its fixed layout (class version 1, members in streamer order)::

    header   bytecount(4) version|memberwise(2) class version(2) checksum(4) n(4)
    sim_     n x art::Ptr          36 bytes each (bytecounted object; ProductID value at +16, key at +28)
    volumeG4ID_ n x int32, nCompress_ n x uint32, time_ n x float64,
    energyDepG4_, energyDepBirks_, momentumIn_  n x float32 each
    pos_     n x CLHEP::Hep3Vector  30 bytes each (bytecounted object, 3 x float64 at +6)

All values are big-endian. The decoder checks the header bytecount and the total length of every event.
``pos_`` is the deposit-weighted position in the crystal frame; ``pos_.z`` is the depth used by the response.
"""
from __future__ import annotations

import numpy as np

PTR_BYTES, VEC_BYTES, HEADER = 36, 30, 16
BRANCH = "mu2e::CaloShowerSteps_{label}.obj"


def _baskets(branch):
    for i in range(branch.num_baskets):
        bk = branch.basket(i)
        yield np.frombuffer(bytes(bk.data), dtype=np.uint8), np.asarray(bk.byte_offsets, dtype=np.int64)


def decode_blobs(data, offsets):
    """Decode a concatenation of per-event blobs (``data[offsets[i]:offsets[i+1]]``) into flat step columns."""
    start, stop = offsets[:-1], offsets[1:]
    empty = (stop - start) == 0
    s = start[~empty]
    n = _be(data, s + 12, ">i4")
    if np.any((_be(data, s, ">u4") & 0x3FFFFFFF) + 4 != (stop - start)[~empty]):
        raise ValueError("unexpected CaloShowerStep blob bytecount")
    if np.any(HEADER + n * (PTR_BYTES + 4 + 4 + 8 + 4 + 4 + 4 + VEC_BYTES) != (stop - start)[~empty]):
        raise ValueError("unexpected CaloShowerStep blob layout (class version changed?)")
    ev = np.repeat(np.flatnonzero(~empty), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    base = np.repeat(s + HEADER, n)
    nn = np.repeat(n, n)
    col = {}
    ptr = base + k * PTR_BYTES
    col["sim_product"] = _be(data, ptr + 16, ">u4")
    col["sim_key"] = _be(data, ptr + 28, ">u8")
    off = base + nn * PTR_BYTES
    for name, fmt, size in (("volume", ">i4", 4), ("n_compress", ">u4", 4), ("time", ">f8", 8),
                            ("energy_g4", ">f4", 4), ("energy_birks", ">f4", 4), ("momentum_in", ">f4", 4)):
        col[name] = _be(data, off + k * size, fmt)
        off = off + nn * size
    vec = off + k * VEC_BYTES + 6
    for j, c in enumerate("xyz"):
        col[f"pos_{c}"] = _be(data, vec + 8 * j, ">f8")
    col["entry"] = ev
    return col


def _be(data, at, fmt):
    size = np.dtype(fmt).itemsize
    idx = np.asarray(at)[:, None] + np.arange(size)
    return data[idx].copy().view(fmt).ravel().astype(np.dtype(fmt).newbyteorder("="))


def read_steps(path, label="CaloShowerStepMaker__CaloLibraryCE"):
    """Steps and event numbers from an art file: dict of flat columns plus ``events`` (run, subrun, event)."""
    import uproot
    with uproot.open(path) as f:
        t = f["Events"]
        br = t[BRANCH.format(label=label)]
        datas, offs, shift, entry_base = [], [], 0, []
        for (d, o), first in zip(_baskets(br), br.entry_offsets[:-1]):
            datas.append(d)
            offs.append(o[:-1] + shift)
            entry_base.append(np.full(o.size - 1, first))
            shift += d.size
        offsets = np.r_[np.concatenate(offs), shift] if offs else np.zeros(1, np.int64)
        col = decode_blobs(np.concatenate(datas) if datas else np.zeros(0, np.uint8), offsets)
        ids = decode_event_ids(t["EventAuxiliary"])
    return col, ids


_ID_HEADERS = bytes.fromhex("4000001a000c40000010000a40000006000a")


def decode_event_ids(branch):
    """(run, subrun, event) per entry from the unsplit ``EventAuxiliary`` blobs.

    Layout: Auxiliary header (6), Hash<2> header (6), length-prefixed hash string (1 + len), then EventID,
    SubRunID and RunID headers (18) followed by run, subrun and event (big-endian uint32).
    """
    rows = []
    for d, o in _baskets(branch):
        s = o[:-1]
        base = s + 13 + d[s + 12].astype(np.int64)
        hdr = d[base[:, None] + np.arange(len(_ID_HEADERS))]
        if not np.all(hdr == np.frombuffer(_ID_HEADERS, np.uint8)):
            raise ValueError("unexpected EventAuxiliary layout")
        at = base + len(_ID_HEADERS)
        rows.append(np.stack([_be(d, at, ">u4"), _be(d, at + 4, ">u4"), _be(d, at + 8, ">u4")], axis=1))
    return np.concatenate(rows).astype(np.int64)


def onspill_timing(event_number, *, time_from_protons_to_dr_marker=200.0, initial_phase_shift=0.0,
                   clock_tick=25.0, daq_ticks=68, onspill_bins=5, dr_period=1695.0, crystal_length=200.0):
    """``EventWindowMarkerProducer`` on spill: MC proton-bunch time and event length per event (float32 storage)."""
    ev = np.asarray(event_number, dtype=np.int64)
    shift_per = -clock_tick / onspill_bins
    phase = initial_phase_shift * clock_tick + (ev % onspill_bins) * shift_per
    phase = np.where(phase < -clock_tick, phase + clock_tick, phase)
    length = np.where(phase < -clock_tick - shift_per, (daq_ticks - 1) * clock_tick, daq_ticks * clock_tick)
    n = ev.size
    return {"pbtime": (-time_from_protons_to_dr_marker + phase).astype(np.float32),
            "mbtime": np.full(n, dr_period, np.float32), "event_length": length.astype(np.float64),
            "onspill": np.ones(n, bool), "marker": np.full(n, time_from_protons_to_dr_marker, np.float32),
            "crystal_length": np.full(n, crystal_length, np.float32)}


def steps_from_art(path, sim_crystal, label="CaloShowerStepMaker__CaloLibraryCE", **timing_kw):
    """The ``(events, timing, steps)`` triple of :func:`calofastsim.stockio.load_steps`, from the art file itself."""
    col, ids = read_steps(path, label)
    order = np.argsort(ids[:, 2], kind="stable")
    events = ids[order, 2]
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    crystal = col["volume"].astype(np.int64)
    steps = {"event": rank[col["entry"]], "crystal": crystal, "energy_g4": col["energy_g4"],
             "energy_birks": col["energy_birks"], "time": col["time"], "z_local": col["pos_z"].astype(np.float32),
             "lru": sim_crystal["lru"][crystal], "yields": sim_crystal["pe_per_mev"][crystal]}
    return events, onspill_timing(events, **timing_kw), steps
