"""Read the stock calorimeter trigger inputs and outputs straight from an art file (uproot + NumPy).

art writes these collections memberwise in unsplit branches, which uproot cannot interpret, so each event's blob is
decoded here from its fixed layout (big-endian; the decoder checks every length):

- ``CaloDigis`` (class version 10): header 12 (bytecount 4, collection version 2, class version 2, n 4), then
  ``SiPMID_`` n x int32, ``t0tick_`` n x int32, ``waveform_`` (member header 6, then per digi n_samples int32 and the
  samples), ``peakpos_`` n x int32.
- ``CaloClusters`` (class version 1, written with a checksum): header 16 (bytecount, collection version, class
  version 0, checksum, n), then ``diskID_`` n x int32, ``time_``, ``timeErr_``, ``energyDep_``, ``energyDepErr_``
  n x float32 each, ``cog_`` n x Hep3Vector (30 bytes: bytecount, version, 3 x float64), ``caloHitsPtrVector_``
  (one bytecounted block, skipped), ``size_`` n x uint32, ``isSplit_`` n x bool.
- ``ProtonBunchTime`` (version 1): bytecount, version 0, checksum, ``pbtime_`` and ``pbterr_`` float32.
- ``TriggerResults``: the packed ``HLTPathStatus`` words of a process, one row per event (state = word & 3:
  0 ready, 1 pass, 2 fail, 3 exception; accepted = ready or pass, as ``HLTPathStatus::accept``).
"""
from __future__ import annotations

import numpy as np

DIGIS = "mu2e::CaloDigis_{label}.obj"
CLUSTERS = "mu2e::CaloClusters_{label}.obj"
PBT = "mu2e::ProtonBunchTime_{label}.obj"
TRIGGER = "art::TriggerResults_TriggerResults__{process}.obj"
_ID_HEADERS = bytes.fromhex("4000001a000c40000010000a40000006000a")


def _blobs(branch):
    datas, offs, shift = [], [], 0
    for i in range(branch.num_baskets):
        bk = branch.basket(i)
        d = np.frombuffer(bytes(bk.data), dtype=np.uint8)
        o = np.asarray(bk.byte_offsets, dtype=np.int64)
        datas.append(d)
        offs.append(o[:-1] + shift)
        shift += d.size
    data = np.concatenate(datas) if datas else np.zeros(0, np.uint8)
    return data, np.r_[np.concatenate(offs) if offs else np.zeros(0, np.int64), shift]


def _be(data, at, fmt):
    size = np.dtype(fmt).itemsize
    idx = np.asarray(at, dtype=np.int64)[..., None] + np.arange(size)
    return data[idx].copy().view(fmt)[..., 0].astype(np.dtype(fmt).newbyteorder("="))


def _bytecount(data, s, stop):
    if np.any((_be(data, s, ">u4") & 0x3FFFFFFF) + 4 != stop - s):
        raise ValueError("unexpected blob bytecount")


def decode_digis(data, offsets):
    """Flat CaloDigi columns plus ``entry``; ``samples`` holds every waveform, ``wf_start`` / ``wf_len`` index it."""
    start, stop = offsets[:-1], offsets[1:]
    _bytecount(data, start, stop)
    if np.any(_be(data, start + 4, ">u2") != 0x400A) or np.any(_be(data, start + 6, ">u2") != 10):
        raise ValueError("unexpected CaloDigi collection or class version")
    n = _be(data, start + 8, ">i4").astype(np.int64)
    cols = {k: [] for k in ("entry", "sipm", "t0", "peakpos", "wf_start", "wf_len")}
    samples, base = [], 0
    for e in range(n.size):
        k, at = int(n[e]), int(start[e]) + 12
        sipm = _be(data, at + 4 * np.arange(k), ">i4"); at += 4 * k
        t0 = _be(data, at + 4 * np.arange(k), ">i4"); at += 4 * k
        if k:
            at += 6
        lens = np.empty(k, np.int64)
        for j in range(k):
            m = int(_be(data, np.array([at]), ">i4")[0]); at += 4
            lens[j] = m
            samples.append(_be(data, at + 4 * np.arange(m), ">i4")); at += 4 * m
        peak = _be(data, at + 4 * np.arange(k), ">i4"); at += 4 * k
        if at != stop[e]:
            raise ValueError(f"unexpected CaloDigi blob layout (entry {e}, n {k}, at {at - start[e]}, len {stop[e] - start[e]})")
        cols["entry"].append(np.full(k, e)); cols["sipm"].append(sipm); cols["t0"].append(t0)
        cols["peakpos"].append(peak); cols["wf_len"].append(lens)
        cols["wf_start"].append(base + np.cumsum(lens) - lens); base += int(lens.sum())
    out = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)).astype(np.int64) for k, v in cols.items()}
    out["samples"] = np.concatenate(samples).astype(np.int64) if samples else np.zeros(0, np.int64)
    out["n"] = n
    return out


def decode_clusters(data, offsets):
    """Flat CaloCluster columns plus ``entry`` (collection order)."""
    start, stop = offsets[:-1], offsets[1:]
    _bytecount(data, start, stop)
    n = _be(data, start + 12, ">i4").astype(np.int64)
    names = ("disk", "time", "time_err", "energy", "energy_err", "cog_x", "cog_y", "cog_z", "size", "is_split")
    cols = {k: [] for k in ("entry",) + names}
    for e in range(n.size):
        k, at = int(n[e]), int(start[e]) + 16
        if k == 0:
            continue
        cols["entry"].append(np.full(k, e))
        cols["disk"].append(_be(data, at + 4 * np.arange(k), ">i4")); at += 4 * k
        for name in ("time", "time_err", "energy", "energy_err"):
            cols[name].append(_be(data, at + 4 * np.arange(k), ">f4")); at += 4 * k
        v = at + 30 * np.arange(k) + 6
        for j, c in enumerate("xyz"):
            cols[f"cog_{c}"].append(_be(data, v + 8 * j, ">f8"))
        at += 30 * k
        at += 4 + int(_be(data, np.array([at]), ">u4")[0] & 0x3FFFFFFF)
        cols["size"].append(_be(data, at + 4 * np.arange(k), ">u4")); at += 4 * k
        cols["is_split"].append(data[at:at + k].astype(bool)); at += k
        if at != stop[e]:
            raise ValueError(f"unexpected CaloCluster blob layout (entry {e}, n {k}, at {at - start[e]}, len {stop[e] - start[e]})")
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()} | {"n": n}


def decode_pbt(data, offsets):
    start, stop = offsets[:-1], offsets[1:]
    if np.any(stop - start != 18):
        raise ValueError("unexpected ProtonBunchTime blob")
    return _be(data, start + 10, ">f4")


def decode_event_ids(branch):
    """(run, subrun, event) per entry from the unsplit ``EventAuxiliary`` blobs."""
    data, off = _blobs(branch)
    s = off[:-1]
    base = s + 13 + data[s + 12].astype(np.int64)
    hdr = data[base[:, None] + np.arange(len(_ID_HEADERS))]
    if not np.all(hdr == np.frombuffer(_ID_HEADERS, np.uint8)):
        raise ValueError("unexpected EventAuxiliary layout")
    at = base + len(_ID_HEADERS)
    return np.stack([_be(data, at, ">u4"), _be(data, at + 4, ">u4"), _be(data, at + 8, ">u4")], 1).astype(np.int64)


def decode_trigger(data, offsets):
    """Packed HLTPathStatus words, (events, paths) uint16 (memberwise vector, version 10)."""
    start, stop = offsets[:-1], offsets[1:]
    rows = []
    for e in range(start.size):
        at = int(start[e])
        bc = int(_be(data, np.array([at]), ">u4")[0] & 0x3FFFFFFF)
        # TriggerResults (bytecount, version) > HLTGlobalStatus base (bytecount, version) > paths_ vector
        while True:
            ver = int(_be(data, np.array([at + 4]), ">u2")[0])
            if ver == 0x400A:
                break
            at += 6
            if at >= stop[e]:
                raise ValueError("HLTPathStatus vector not found")
        size = int(_be(data, np.array([at]), ">u4")[0] & 0x3FFFFFFF)
        m = int(_be(data, np.array([at + 8]), ">u4")[0])
        if int(_be(data, np.array([at + 6]), ">u2")[0]) != 10 or size != 8 + 2 * m:
            raise ValueError(f"unsupported HLTPathStatus framing (size {size}, n {m}, total {bc})")
        rows.append(_be(data, at + 12 + 2 * np.arange(m), ">u2"))
    if len({r.size for r in rows}) > 1:
        raise ValueError("path count changes between events")
    return np.stack(rows) if rows else np.zeros((0, 0), np.uint16)


def accepted(words):
    state = np.asarray(words) & 3
    return (state == 0) | (state == 1)


def read(path, digis="CaloDigiMaker__Digitize", clusters="CaloClusterFast__Digitize", pbt="EWMProducer__Digitize",
         process="Digitize"):
    """Everything the calo trigger comparison needs from one stock art file, in entry order."""
    import uproot
    with uproot.open(path) as f:
        t = f["Events"]
        out = {"ids": decode_event_ids(t["EventAuxiliary"]),
               "digis": decode_digis(*_blobs(t[DIGIS.format(label=digis)])),
               "pbt": decode_pbt(*_blobs(t[PBT.format(label=pbt)]))}
        if clusters:
            out["clusters"] = decode_clusters(*_blobs(t[CLUSTERS.format(label=clusters)]))
        if process:
            out["trigger"] = decode_trigger(*_blobs(t[TRIGGER.format(process=process)]))
    return out
