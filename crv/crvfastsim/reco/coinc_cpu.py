"""Exact NumPy acceleration of the host coincidence attachment stage."""
from itertools import chain

import numpy as np


def _keys(cluster, bar, side):
    return np.rec.fromarrays((cluster, bar, side), names="cluster,bar,side")


def attach_opposite_side_pulses(clusters, hits, max_dt):
    """Attach from the native snapshot, preserving stock candidate ties and append order."""
    if not clusters or not hits:
        return
    members = list(chain.from_iterable(clusters))
    if not members:
        return
    fields = ("idx", "bar", "sipm", "side", "t", "seq")
    m = {k: np.fromiter((h[k] for h in members), float if k == "t" else np.int64) for k in fields}
    h = {k: np.fromiter((hit[k] for hit in hits), float if k == "t" else np.int64) for k in fields}
    mc = np.repeat(np.arange(len(clusters)), [len(cl) for cl in clusters])
    candidates = np.flatnonzero((h["seq"] == 0) & ~np.isin(h["idx"], m["idx"]))
    anchors = np.flatnonzero(m["seq"] == 0)
    if not len(candidates) or not len(anchors):
        return
    occupied = np.unique(_keys(mc, m["bar"], m["side"]))
    chosen = []
    block = max(1, 1_000_000 // len(anchors))
    for start in range(0, len(candidates), block):
        c = candidates[start:start + block]
        dt = np.abs(h["t"][c, None] - m["t"][anchors])
        matches = ((h["bar"][c, None] == m["bar"][anchors]) &
                   (h["side"][c, None] == 1 - m["side"][anchors]) & (dt < max_dt))
        row, col = np.nonzero(matches)
        candidate, cluster = c[row], mc[anchors[col]]
        delta = dt[row, col]
        available = ~np.isin(_keys(cluster, h["bar"][candidate], h["side"][candidate]), occupied)
        candidate, cluster, delta = candidate[available], cluster[available], delta[available]
        order = np.lexsort((cluster, delta, candidate))
        candidate, cluster, delta = candidate[order], cluster[order], delta[order]
        first = np.r_[True, candidate[1:] != candidate[:-1]] if len(candidate) else np.zeros(0, bool)
        chosen.append((candidate[first], cluster[first], delta[first]))
    candidate = np.concatenate([q[0] for q in chosen])
    cluster = np.concatenate([q[1] for q in chosen])
    delta = np.concatenate([q[2] for q in chosen])
    order = np.lexsort((candidate, delta, h["sipm"][candidate], h["bar"][candidate], cluster))
    candidate, cluster = candidate[order], cluster[order]
    keys = _keys(cluster, h["bar"][candidate], h["sipm"][candidate])
    first = np.r_[True, keys[1:] != keys[:-1]] if len(keys) else np.zeros(0, bool)
    for cl, c in zip(cluster[first], candidate[first]):
        clusters[cl].append(hits[c])
