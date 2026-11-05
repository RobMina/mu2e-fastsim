"""Vectorized source-conformance CaloHit grouping, without ntuple fit matching."""
from __future__ import annotations

import operator

import numpy as np

from .response import weighted_reco_digis


def make_hits(event_index, sipm_id, time, time_error, energy, energy_error, *,
              n_events, n_crystals, time4merge):
    """Return hits ordered by event/crystal/start time and original member indices.

    This mirrors the inspected CaloHitMaker algorithm, not TemplateFit: discard
    crystals with fewer than two TOTAL reco digis, sort by float32 time, and
    split relative to the FIRST member of each group. Equality merges and
    singleton temporal groups remain. SiPM identity never restricts membership.
    Time differences are float32 before comparison against a double threshold.
    A round handles all event/crystal cells in parallel; no Python event loop.
    Stable equal-time ordering is a deterministic convention, not C++ std::sort
    parity. The result flags affected hits; exact numeric closure needs an
    order-invariant fixture or the actual stock member order for those hits.
    """
    n_events, n_crystals = operator.index(n_events), operator.index(n_crystals)
    if n_events < 0 or n_crystals <= 0:
        raise ValueError("invalid event/crystal count")
    threshold = np.asarray(time4merge, dtype=np.float64)
    if threshold.ndim != 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("time4merge must be a finite nonnegative scalar")
    arrays = [np.asarray(value) for value in
              (event_index, sipm_id, time, time_error, energy, energy_error)]
    if any(array.ndim != 1 for array in arrays) or len({array.size for array in arrays}) != 1:
        raise ValueError("reco-digi columns must be one-dimensional with equal lengths")
    events, sipm = arrays[:2]
    if not np.issubdtype(events.dtype, np.integer) or not np.issubdtype(sipm.dtype, np.integer):
        raise ValueError("event_index and sipm_id must have integer dtype")
    if np.any((events < 0) | (events >= n_events)) or np.any((sipm < 0) | (sipm >= 2 * n_crystals)):
        raise ValueError("event/sensor ID outside supplied geometry")
    values = []
    for array in arrays[2:]:
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError("reco-digi values must be finite real numbers")
        with np.errstate(over="ignore"):
            stored = array.astype(np.float32)
        if not np.all(np.isfinite(stored)):
            raise ValueError("reco-digi values exceed float32 range")
        values.append(stored)
    times, terr, energies, eerr = values
    if np.any(terr <= 0) or np.any(eerr <= 0):
        raise ValueError("fit errors must be strictly positive")
    crystal = sipm.astype(np.int64) // 2
    order = np.lexsort((times, crystal, events))
    cell_keys = np.rec.fromarrays([events[order], crystal[order]], names="event,crystal")
    _, cell, counts = np.unique(cell_keys, return_inverse=True, return_counts=True)
    retained = counts[cell] >= 2
    order, cell = order[retained], cell[retained]
    _, cell = np.unique(cell, return_inverse=True)
    n_cells = int(np.max(cell, initial=-1)) + 1
    m = len(order)
    anchor = np.full(n_cells, m, dtype=np.int64)
    np.minimum.at(anchor, cell, np.arange(m))
    group_round = np.full(m, -1, dtype=np.int64)
    pending = np.ones(m, dtype=bool)
    round_index = 0
    while np.any(pending):
        current = np.flatnonzero(pending)
        active_cell = cell[current]
        with np.errstate(over="ignore"):
            delta = times[order[current]] - times[order[anchor[active_cell]]]
        take = current[delta.astype(np.float64) <= threshold]
        group_round[take] = round_index
        pending[take] = False
        anchor.fill(m)
        remaining = np.flatnonzero(pending)
        np.minimum.at(anchor, cell[remaining], remaining)
        round_index += 1
    group_keys = np.rec.fromarrays([cell, group_round], names="cell,round")
    _, group, members_per_hit = np.unique(group_keys, return_inverse=True, return_counts=True)
    n_hits = len(members_per_hit)
    member_order = np.argsort(group, kind="stable")
    offsets = np.r_[0, np.cumsum(members_per_hit)].astype(np.int64)
    first = member_order[offsets[:-1]]
    equal_time = ((cell[1:] == cell[:-1]) & (times[order][1:] == times[order][:-1]))
    tied = np.zeros(n_hits, dtype=bool)
    tied[group[1:][equal_time]] = True
    hit_time, hit_time_error = weighted_reco_digis(times[order], terr[order], group, n_hits)
    hit_energy, hit_energy_error = weighted_reco_digis(energies[order], eerr[order], group, n_hits)
    return {"event_index": events[order[first]], "crystal": crystal[order[first]],
            "nSiPMs": members_per_hit, "time": hit_time, "time_error": hit_time_error,
            "energy": hit_energy, "energy_error": hit_energy_error,
            "member_offsets": offsets, "member_index": order[member_order],
            "equal_time_order_unspecified": tied}
