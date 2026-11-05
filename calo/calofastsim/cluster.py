"""Stock calorimeter clustering (Offline CaloCluster: CaloProtoClusterMaker, ClusterFinder, CaloClusterMaker,
ClusterAssociator, ClusterUtils), vectorised over events. Python loops run over seed rounds and cluster slots only.

Source rules kept:
- seeds are hits above ``EminSeed`` taken in hit-collection order (``std::set`` of hit pointers), not energy order;
- a cluster grows from the seed crystal through neighbouring crystals; every unused hit within ``deltaTime`` of the
  seed time in a visited neighbour joins, and the crystal expands further when a joined hit is above ``ExpandCut``;
  the seed crystal's other hits never join;
- after the main clusters, hits not later than ``seed time - deltaTime`` are dropped and the rest form split
  clusters (any hit can seed); a split cluster joins the closest main cluster within ``deltaTime`` and
  ``maxDistSplit``, otherwise it is discarded; main clusters within ``deltaTime`` and ``maxDistMain`` merge;
- cluster time is that of the highest-energy hit, energy the sum, position the energy-weighted crystal centre of
  the hits on the highest-energy hit's disk, in that disk's front-face frame; clusters are sorted by energy.
Equal-energy and equal-time ordering ties follow NumPy's stable order, not ``std::sort``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterConfig:
    emin_seed: float = 10.0
    expand_cut: float = 1.0
    enoise_cut: float = 1.0
    delta_time: float = 2.0
    max_dist_split: float = 150.0
    max_dist_main: float = 150.0


@dataclass(frozen=True)
class Geometry:
    position: np.ndarray       # (n_crystals, 3) crystal centres, Mu2e frame
    disk: np.ndarray           # (n_crystals,)
    neighbors: np.ndarray      # (n_crystals, n_crystals) bool, first ring
    front_face: np.ndarray     # (n_disks, 3) front-face origins (axes are the Mu2e axes)

    @classmethod
    def from_dump(cls, crystals, disks):
        n = len(crystals["crystal"])
        order = np.argsort(crystals["crystal"])
        pos = np.stack([np.asarray(crystals[f"position_{c}"])[order] for c in "xyz"], axis=1)
        adj = np.zeros((n, n), bool)
        nb = [crystals["neighbors"][i] for i in order]
        rows = np.repeat(np.arange(n), [len(v) for v in nb])
        adj[rows, np.concatenate([np.asarray(v, dtype=np.int64) for v in nb])] = True
        ff = np.stack([np.asarray(disks[f"front_face_origin_{c}"]) for c in "xyz"], axis=1)
        return cls(pos, np.asarray(crystals["disk"])[order].astype(np.int64), adj, ff)

    def save(self, path):
        np.savez_compressed(path, position=self.position, disk=self.disk, neighbors=self.neighbors,
                            front_face=self.front_face)

    @classmethod
    def load(cls, path):
        """A saved ``.npz`` geometry, or a calorimeter dump ROOT file with ``dump/crystals`` and ``dump/disks``."""
        if str(path).endswith(".npz"):
            with np.load(path) as z:
                return cls(z["position"], z["disk"], z["neighbors"], z["front_face"])
        import uproot
        with uproot.open(path) as f:
            return cls.from_dump(f["dump/crystals"].arrays(library="np"), f["dump/disks"].arrays(library="np"))


def pad_hits(event, n_events, **cols):
    """Flat hits (grouped by event, in collection order) -> dense ``(n_events, H)`` arrays plus ``valid``."""
    event = np.asarray(event, dtype=np.int64)
    counts = np.bincount(event, minlength=n_events)
    width = max(int(counts.max(initial=0)), 1)
    slot = np.arange(event.size) - np.repeat(np.cumsum(counts) - counts, counts)
    out = {"valid": np.zeros((n_events, width), bool), "slot": slot}
    out["valid"][event, slot] = True
    for k, v in cols.items():
        v = np.asarray(v)
        a = np.zeros((n_events, width), v.dtype)
        a[event, slot] = v
        out[k] = a
    return out


def _grow(seed, avail, t, e, same, adj, cfg):
    """Cluster membership from one seed per event (``seed`` = hit slot or -1), BFS by layers."""
    n, width = t.shape
    ev = np.flatnonzero(seed >= 0)
    s = seed[ev]
    members = np.zeros((n, width), bool)
    if ev.size == 0:
        return members
    ts = t[ev, s].astype(np.float64)
    intime = avail[ev] & (np.abs(t[ev].astype(np.float64) - ts[:, None]) < cfg.delta_time)
    visited = same[ev, s]
    expanding = visited.copy()
    mem = np.zeros((ev.size, width), bool)
    mem[np.arange(ev.size), s] = True
    a = adj[ev]
    sm = same[ev]
    while True:
        reach = np.einsum("ei,eij->ej", expanding.astype(np.int8), a.astype(np.int8)) > 0
        new_crystal = reach & ~visited
        if not new_crystal.any():
            break
        visited |= new_crystal
        new = new_crystal & intime
        mem |= new
        grow = new & (e[ev] > cfg.expand_cut)
        expanding = np.einsum("ei,eij->ej", grow.astype(np.int8), sm.astype(np.int8)) > 0
    members[ev] = mem
    return members


def _rounds(seedable, avail, t, e, same, adj, cfg):
    """Repeated cluster formation; returns labels (round index or -1), seed times per round and the final avail."""
    n, width = t.shape
    label = np.full((n, width), -1, np.int64)
    seed_times = []
    avail = avail.copy()
    seedable = seedable & avail
    r = 0
    while seedable.any():
        has = seedable.any(axis=1)
        seed = np.where(has, np.argmax(seedable, axis=1), -1)
        members = _grow(seed, avail, t, e, same, adj, cfg)
        label[members] = r
        avail &= ~members
        seedable &= ~members
        seed_times.append(np.where(has, t[np.arange(n), np.maximum(seed, 0)].astype(np.float64), np.nan))
        r += 1
    return label, (np.stack(seed_times, axis=1) if seed_times else np.zeros((n, 0))), avail


def _cluster_props(label, n_cl, t, e, crystal):
    """Per (event, cluster): leading hit slot, its time, energy sum. Empty clusters have lead -1."""
    n, width = label.shape
    lead = np.full((n, n_cl), -1, np.int64)
    esum = np.zeros((n, n_cl))
    best = np.full((n, n_cl), -np.inf)
    ev, sl = np.nonzero(label >= 0)
    lab = label[ev, sl]
    np.add.at(esum, (ev, lab), e[ev, sl].astype(np.float64))
    order = np.lexsort((-sl, e[ev, sl]))
    best[ev[order], lab[order]] = e[ev, sl][order]
    lead[ev[order], lab[order]] = sl[order]
    time = np.where(lead >= 0, t[np.arange(n)[:, None], np.maximum(lead, 0)].astype(np.float64), np.inf)
    return lead, time, esum


def _closest(label_a, ia, label_b, ib, dist):
    """Min crystal distance between hits of cluster ``ia`` (labels a) and ``ib`` (labels b), per event."""
    ma = label_a == ia[:, None]
    mb = label_b == ib[:, None]
    d = np.where(ma[:, :, None] & mb[:, None, :], dist, np.inf)
    return d.min(axis=(1, 2), initial=np.inf)


def make_clusters(hits, n_events, geom: Geometry, cfg: ClusterConfig = ClusterConfig()):
    """``hits``: flat dict (event, crystal, time, energy, energy_error, time_error) in stock collection order.

    Returns per-cluster arrays (event, energy, time, x, y, z in the disk front-face frame, size, disk, split flag)
    sorted by event then decreasing energy, and the final cluster index of every input hit (-1 if unclustered).
    """
    P = pad_hits(hits["event"], n_events, crystal=np.asarray(hits["crystal"], np.int64),
                 t=np.asarray(hits["time"], np.float32), e=np.asarray(hits["energy"], np.float32))
    valid, crystal, t, e = P["valid"], P["crystal"], P["t"], P["e"]
    n, width = t.shape
    same = (crystal[:, :, None] == crystal[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    adj = geom.neighbors[crystal[:, :, None], crystal[:, None, :]] & valid[:, :, None] & valid[:, None, :]
    pos = geom.position[crystal]
    dist = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
    avail = valid & (e >= cfg.enoise_cut)

    main, seed_times, avail = _rounds(avail & (e > cfg.emin_seed), avail, t, e, same, adj, cfg)
    if seed_times.shape[1]:
        keep = np.any(seed_times[:, None, :] - t.astype(np.float64)[:, :, None] < cfg.delta_time, axis=2)
        avail &= keep
    else:
        avail &= False
    split, _, _ = _rounds(avail, avail, t, e, same, adj, cfg)

    nm, ns = max(seed_times.shape[1], 1), max(int(split.max(initial=-1)) + 1, 1)
    m_lead, m_time, _ = _cluster_props(main, nm, t, e, crystal)
    s_lead, s_time, _ = _cluster_props(split, ns, t, e, crystal)
    m_order = np.argsort(m_time, axis=1, kind="stable")      # time-sorted main clusters
    s_order = np.argsort(s_time, axis=1, kind="stable")
    rank = np.empty_like(m_order)
    np.put_along_axis(rank, m_order, np.arange(nm)[None, :].repeat(n, 0), axis=1)
    main_sorted = np.where(main >= 0, np.take_along_axis(rank, np.maximum(main, 0), axis=1), -1)
    ms_time = np.take_along_axis(m_time, m_order, axis=1)
    ms_lead = np.take_along_axis(m_lead, m_order, axis=1)
    rows = np.arange(n)

    # split -> closest main within time and distance (first minimum in time order wins)
    split_to = np.full((n, ns), -1, np.int64)
    for k in range(ns):
        js = s_order[:, k]
        ts = s_time[rows, js]
        lead_s = s_lead[rows, js]
        best = np.full(n, 1e6)
        for j in range(nm):
            tm = ms_time[:, j]
            lead_m = ms_lead[:, j]
            with np.errstate(invalid="ignore"):
                ok = (lead_s >= 0) & (lead_m >= 0) & (ts - tm <= cfg.delta_time) & (tm - ts <= cfg.delta_time)
            d0 = np.linalg.norm(geom.position[crystal[rows, np.maximum(lead_m, 0)]]
                                - geom.position[crystal[rows, np.maximum(lead_s, 0)]], axis=1)
            ok &= d0 <= 2.5 * cfg.max_dist_split
            d = _closest(main_sorted, np.full(n, j), split, js, dist)
            take = ok & (d < cfg.max_dist_split) & (d < best)
            best = np.where(take, d, best)
            split_to[rows[take], js[take]] = j

    # temp clusters = time-sorted mains plus their splits; main-main association (strategy 0)
    temp = main_sorted.copy()
    sj = np.where(split >= 0, np.take_along_axis(split_to, np.maximum(split, 0), axis=1), -1)
    temp = np.where((split >= 0) & (sj >= 0), sj, temp)
    parent = np.full((n, nm), -1, np.int64)
    for i in range(nm):
        for j in range(i + 1, nm):
            ok = (ms_lead[:, i] >= 0) & (ms_lead[:, j] >= 0) & (parent[:, j] < 0)
            with np.errstate(invalid="ignore"):
                ok &= (ms_time[:, j] - ms_time[:, i]) <= cfg.delta_time
            d = _closest(temp, np.full(n, i), temp, np.full(n, j), dist)
            ok &= d <= cfg.max_dist_main
            parent[rows[ok], j] = i
    root = np.broadcast_to(np.arange(nm), (n, nm)).copy()
    for _ in range(nm):
        p = np.take_along_axis(parent, root, axis=1)
        root = np.where(p >= 0, p, root)
    final = np.where(temp >= 0, np.take_along_axis(root, np.maximum(temp, 0), axis=1), -1)
    merged = np.zeros((n, nm), bool)
    merged[rows[:, None].repeat(nm, 1)[parent >= 0], root[parent >= 0]] = True
    has_split = np.zeros((n, nm), bool)
    ev_s, sl_s = np.nonzero((split >= 0) & (sj >= 0))
    has_split[ev_s, final[ev_s, sl_s]] = True

    # final cluster quantities
    lead, ctime, esum = _cluster_props(final, nm, t, e, crystal)
    exists = lead >= 0
    lead_disk = geom.disk[crystal[rows[:, None], np.maximum(lead, 0)]]
    hit_disk = geom.disk[crystal]
    ev, sl = np.nonzero(final >= 0)
    lab = final[ev, sl]
    on_disk = hit_disk[ev, sl] == lead_disk[ev, lab]
    w = np.where(on_disk, e[ev, sl].astype(np.float64), 0.0)
    sw = np.zeros((n, nm))
    sxyz = np.zeros((n, nm, 3))
    np.add.at(sw, (ev, lab), w)
    np.add.at(sxyz, (ev, lab), w[:, None] * pos[ev, sl])
    cog = sxyz / np.where(sw > 0, sw, 1.0)[..., None] - geom.front_face[lead_disk]
    size = np.zeros((n, nm), np.int64)
    np.add.at(size, (ev, lab), 1)

    ce, cc = np.nonzero(exists)
    order = np.lexsort((-esum[ce, cc], ce))
    ce, cc = ce[order], cc[order]
    out = {"event": ce, "energy": esum[ce, cc], "time": ctime[ce, cc], "x": cog[ce, cc, 0], "y": cog[ce, cc, 1],
           "z": cog[ce, cc, 2], "size": size[ce, cc], "disk": lead_disk[ce, cc],
           "split": has_split[ce, cc] | merged[ce, cc]}
    index = np.full((n, nm), -1, np.int64)
    index[ce, cc] = np.arange(ce.size)
    hit_cluster = np.where(final >= 0, index[rows[:, None], np.maximum(final, 0)], -1)
    out["hit_cluster"] = hit_cluster[np.asarray(hits["event"], np.int64), P["slot"]]
    return out
