"""Readers for the CrvRecoTableDumper trees (art/CrvFastSim/src/CrvRecoTableDumper_module.cc).

Trees in the TFileService directory ``crvdump/``, keyed by (run, subrun, event); channel rows also carry ``idx`` (the
row's index in its art collection), ``bar`` and ``sipm``. Jagged columns (ADC samples, voltages, charges) come back as
awkward arrays; scalar columns as numpy.
"""
import awkward as ak
import numpy as np
import uproot

from ..tables import KEY, expand

DIR = "crvdump"
TREES = ("events", "charges", "digimc", "digis", "pulses", "clusters", "calib")


def read(paths, tree, columns=None, entry_stop=None):
    """Concatenate one tree over files as an awkward record array."""
    arrs = []
    for p in expand(paths):
        with uproot.open(p) as f:
            key = f"{DIR}/{tree}"
            if key not in f:
                continue
            arrs.append(f[key].arrays(columns, entry_stop=entry_stop, library="ak"))
    return ak.concatenate(arrs) if arrs else None


def calib(paths):
    """Per-channel conditions as dense numpy arrays indexed by offline channel (bar*4 + SiPM)."""
    c = read(paths, "calib")
    n = int(ak.max(c.channel)) + 1
    out = {k: np.zeros(n, dtype=float) for k in ("pedestal", "pulsearea", "pulseheight", "timeoffset")}
    for k in ("status", "roc", "feb", "febch"):
        out[k] = np.full(n, -1, dtype=np.int64)
    ch = ak.to_numpy(c.channel)
    for k in out:
        out[k][ch] = ak.to_numpy(c[k])
    return out


def event_key(a):
    """One int64 per row combining (run, subrun, event), for joins without pandas."""
    return (ak.to_numpy(a.run).astype(np.int64) << 40) | (ak.to_numpy(a.subrun).astype(np.int64) << 20) | \
        ak.to_numpy(a.event).astype(np.int64)


__all__ = ["DIR", "TREES", "KEY", "read", "calib", "event_key"]
