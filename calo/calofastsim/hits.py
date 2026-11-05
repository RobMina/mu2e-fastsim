"""Vectorized source-conformance CaloHit grouping, without ntuple fit matching."""
from __future__ import annotations

import operator

import numpy as np

from .response import weighted_reco_digis


def make_hits(event_index, sipm_id, time, time_error, energy, energy_error, *,
              n_events, n_crystals, time4merge, backend="numpy"):
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
    ``backend="jax"`` runs the time-group rounds as a jitted ``lax.while_loop``
    (x64 required); sorting and the final reductions stay on the host.
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
    sorted_times = times[order]
    if backend == "jax":
        group_round = _rounds_jax(cell, sorted_times, n_cells, float(threshold))
    else:
        group_round = _rounds_numpy(cell, sorted_times, n_cells, threshold)
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


def _rounds_numpy(cell, times, n_cells, threshold):
    """Round r takes, in every cell, the pending members within ``threshold`` of the first pending member."""
    m = len(cell)
    anchor = np.full(n_cells, m, dtype=np.int64)
    np.minimum.at(anchor, cell, np.arange(m))
    group_round = np.full(m, -1, dtype=np.int64)
    pending = np.ones(m, dtype=bool)
    round_index = 0
    while np.any(pending):
        current = np.flatnonzero(pending)
        with np.errstate(over="ignore"):
            delta = times[current] - times[anchor[cell[current]]]
        take = current[delta.astype(np.float64) <= threshold]
        group_round[take] = round_index
        pending[take] = False
        anchor.fill(m)
        remaining = np.flatnonzero(pending)
        np.minimum.at(anchor, cell[remaining], remaining)
        round_index += 1
    return group_round


_ROUNDS = {}


def _rounds_jax(cell, times, n_cells, threshold):
    """The same rounds as a jitted ``lax.while_loop``, padded to power-of-two shapes to reuse compilations."""
    import jax
    import jax.numpy as jnp
    if not jax.config.x64_enabled:
        raise ValueError("hit grouping on JAX requires caller-enabled JAX x64")
    m = len(cell)
    if m == 0:
        return np.zeros(0, np.int64)
    size = 1 << int(np.ceil(np.log2(m)))
    cells = 1 << int(np.ceil(np.log2(n_cells + 1)))
    pad_cell = np.full(size, cells - 1, np.int64)
    pad_cell[:m] = cell
    pad_t = np.zeros(size, np.float32)
    pad_t[:m] = times
    key = (size, cells)
    if key not in _ROUNDS:
        def run(cell, t, valid, thr):
            idx = jnp.arange(size)

            def anchor_of(pending):
                a = jax.ops.segment_min(jnp.where(pending, idx, size), cell, num_segments=cells)
                return jnp.minimum(a[cell], size - 1)

            def body(state):
                pending, rnd, r = state
                delta = (t - t[anchor_of(pending)]).astype(jnp.float64)
                take = pending & (delta <= thr)
                return pending & ~take, jnp.where(take, r, rnd), r + 1

            pending, rnd, _ = jax.lax.while_loop(lambda s: jnp.any(s[0]), body,
                                                 (valid, jnp.full(size, -1, jnp.int64), jnp.int64(0)))
            return rnd

        _ROUNDS[key] = jax.jit(run)
    valid = np.arange(size) < m
    out = _ROUNDS[key](jnp.asarray(pad_cell), jnp.asarray(pad_t), jnp.asarray(valid), threshold)
    return np.asarray(out)[:m]
