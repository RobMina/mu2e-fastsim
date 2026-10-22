"""Apply a VetoCard: vectorised over events, tracks and coincidences (awkward broadcasting, no Python loops)."""
from __future__ import annotations

import awkward as ak
import numpy as np

from .card import VetoCard


def coinc_mask(coincs, card: VetoCard):
    """[E, C] coincidences allowed to veto (PE, layer and sector cuts)."""
    m = ak.ones_like(coincs["time"], dtype=bool)
    if card.min_pe > 0 or card.min_pe_by_sector:
        thr = ak.full_like(coincs["PEs"], card.min_pe, dtype=float)
        for s, v in card.min_pe_by_sector.items():
            thr = ak.where(coincs["sectorType"] == s, v, thr)
        m = m & (coincs["PEs"] >= thr)
    if card.min_layers > 0:
        m = m & (coincs["nLayers"] >= card.min_layers)
    if card.sectors is not None:
        ok = ak.zeros_like(m)
        for s in card.sectors:
            ok = ok | (coincs["sectorType"] == s)
        m = m & ok
    return m


def gate(coincs, card: VetoCard):
    """[E] event gate (beam-flash timing): True where every coincidence is inside the gate (no coincidences: True)."""
    g = ak.ones_like(ak.num(coincs["time"], axis=1), dtype=bool)
    if card.gate_start_lo is not None:
        g = g & ak.all(coincs["timeStart"] > card.gate_start_lo, axis=1)
    if card.gate_end_hi is not None:
        g = g & ak.all(coincs["timeEnd"] < card.gate_end_hi, axis=1)
    return ak.to_numpy(g)


def vetoed(t_track, coincs, card: VetoCard, return_sector=False):
    """t_track: [E, T] track times (ns); coincs: [E, C] record with time, PEs, nLayers, sectorType (as needed).
    Returns [E, T] bool: any allowed coincidence with dt_lo < t_track - t_coinc < dt_hi. With return_sector, also the
    sectorType of the vetoing coincidence with the highest PEs (-1 if not vetoed)."""
    m = coinc_mask(coincs, card)
    dt = t_track[:, :, None] - coincs["time"][:, None, :]                     # [E, T, C]
    hit = (dt > card.dt_lo) & (dt < card.dt_hi) & m[:, None, :]
    v = ak.any(hit, axis=-1)
    if not return_sector:
        return v
    pe = ak.where(hit, coincs["PEs"][:, None, :], -1.0)
    best = ak.argmax(pe, axis=-1, keepdims=True)
    sec = ak.firsts(ak.broadcast_arrays(coincs["sectorType"][:, None, :], pe)[0][best], axis=-1)
    return v, ak.where(v, ak.fill_none(sec, -1), -1)
