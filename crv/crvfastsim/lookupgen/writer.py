"""Version-6 lookup-table files (the format MakeCrvPhotons::LoadLookupTable reads) from generator counts.

Per bin the generator keeps the photons generated inside the scintillator (``gen``) and the SiPM-0 arrivals: all of
them (``nall``), histogrammed by rounded arrival time (150 ns) and by number of fiber emissions (10), once for the
arrivals inside each histogram (``hta``, ``hea``) and once for those inside both (``ht``, ``he``). The bin record is
encoded as WLSEventAction::EndOfEventAction does in the stock version that matches the optical model (bin_record).
"""
import struct

import numpy as np

from . import geometry as G
from . import materials as M

VERSION = (6, 0)


def header_bytes(c):
    """LookupConstants + LookupCerenkov + LookupBinDefinitions of counter c (struct padding written as zeros)."""
    out = bytearray()
    out += struct.pack("<3i4x", VERSION[0], VERSION[1], int(c.reflector))
    out += struct.pack("<10d", c.hx, c.hy, c.half_length, c.fiber_separation, c.hole_rx, c.hole_ry, c.r_clad2, c.rc,
                       M.BIRKS, 0.0)
    ck = M.cerenkov_map(M.RINDEX_PS)
    for _ in range(2):                         # scintillator and fiber share the polystyrene RINDEX
        out += struct.pack("<Q", len(ck)) + ck.astype("<f8").tobytes()
    e = G.edges(c)
    for k in ("x", "y", "z", "beta", "theta", "phi", "r"):
        out += struct.pack("<Q", len(e[k])) + np.asarray(e[k], "<f8").tobytes()
    return bytes(out)


def _encode(counts, total, min_len=0):
    """uint8(255.0f * (float(count) / float(total)) + 0.5) up to the last non-empty entry (at least min_len)."""
    nz = np.nonzero(counts)[0]
    m = max(nz[-1] + 1 if len(nz) else 0, min_len)
    if total == 0 or m == 0:
        return b""
    p = np.asarray(counts[:m]).astype(np.float32) / np.float32(total)
    v = (np.float32(255) * p).astype(np.float64) + 0.5
    return np.floor(v).astype(np.uint8).tobytes()


def bin_record(i, gen, counts, convention="v6"):
    """Bytes of one LookupBin. counts: nall (all SiPM-0 arrivals), hta / hea (arrivals inside the time / emission
    histograms), ht / he (arrivals inside both). Conventions of the stock writers:

    - ``v6`` (January 2019, the released tables): probability = all arrivals / generated; each histogram normalised
      to all arrivals; a bin with arrivals has at least one entry per histogram.
    - ``current`` (CRVStandalone since 2021): only arrivals inside both histograms count, for the probability too.
    """
    if convention == "v6":
        n = int(counts["nall"])
        if n == 0:
            d = e = b""
            prob = np.float32(0)
        else:
            prob = np.float32(n) / np.float32(gen)
            d, e = _encode(counts["hta"], n, 1), _encode(counts["hea"], n, 1)
    elif convention == "current":
        n = int(np.sum(counts["ht"]))
        if n != int(np.sum(counts["he"])):
            raise ValueError(f"bin {i}: time and emission histograms disagree")
        prob = np.float32(n) / np.float32(gen) if n > 0 else np.float32(0)
        d, e = _encode(counts["ht"], n), _encode(counts["he"], n)
    else:
        raise ValueError(convention)
    return struct.pack("<If", int(i), float(prob)) + struct.pack("<Q", len(d)) + d + struct.pack("<Q", len(e)) + e


KEYS = ("bins", "gen", "nall", "hta", "hea", "ht", "he")


def table_bytes(c, parts, convention="v6"):
    """The full file: parts[simtype] = dict(bins, gen, nall, hta, hea, ht, he) covering every bin once."""
    out = bytearray(header_bytes(c))
    e = G.edges(c)
    for st in (0, 1, 2):
        n = G.n_bins(c, st, e)
        p = parts[st]
        order = np.argsort(p["bins"])
        bins = np.asarray(p["bins"])[order]
        if len(bins) != n or np.any(bins != np.arange(n)):
            raise ValueError(f"sim type {st}: bins incomplete ({len(bins)} of {n})")
        for j, i in enumerate(order):
            out += bin_record(int(bins[j]), int(p["gen"][i]), {k: p[k][i] for k in KEYS[2:]}, convention)
    return bytes(out)


def merge_parts(pieces):
    """Concatenate generator outputs of one sim type (dicts with the KEYS); refuses duplicate bins."""
    bins = np.concatenate([np.asarray(p["bins"]) for p in pieces])
    if len(np.unique(bins)) != len(bins):
        raise ValueError("duplicate bins in the pieces")
    return {k: np.concatenate([np.asarray(p[k]) for p in pieces]) for k in KEYS}
