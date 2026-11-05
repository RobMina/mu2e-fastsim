"""Vectorised clustering against a literal per-event transcription of CaloProtoClusterMaker/CaloClusterMaker."""
from collections import deque

import numpy as np
import pytest

from calofastsim.cluster import ClusterConfig, Geometry, make_clusters


def _geometry(nx=12, ny=12, pitch=34.3):
    """Two square disks of nx*ny crystals with 8-connected first-ring neighbours."""
    n = nx * ny
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    pos = np.zeros((2 * n, 3))
    disk = np.repeat([0, 1], n)
    adj = np.zeros((2 * n, 2 * n), bool)
    for d in (0, 1):
        sl = slice(d * n, (d + 1) * n)
        pos[sl, 0], pos[sl, 1], pos[sl, 2] = ix * pitch, iy * pitch, 700.0 * d
        near = (np.abs(ix[:, None] - ix[None, :]) <= 1) & (np.abs(iy[:, None] - iy[None, :]) <= 1)
        adj[sl, sl] = near & ~np.eye(n, dtype=bool)
    return Geometry(pos, disk, adj, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 700.0]]))


def _finder(seed, hit_map, hits, geom, cfg):
    t_seed = hits[seed][2]
    members = [seed]
    hit_map[hits[seed][1]].remove(seed)
    visited = np.zeros(len(geom.disk), bool)
    queue = deque([hits[seed][1]])
    while queue:
        cid = queue[0]
        visited[cid] = True
        for nid in np.flatnonzero(geom.neighbors[cid]):
            if visited[nid]:
                continue
            visited[nid] = True
            for h in list(hit_map[nid]):
                if abs(hits[h][2] - t_seed) < cfg.delta_time:
                    if hits[h][3] > cfg.expand_cut:
                        queue.append(nid)
                    members.append(h)
                    hit_map[nid].remove(h)
        queue.popleft()
    return sorted(members, key=lambda h: -hits[h][3])


def _reference(hits, geom, cfg):
    """hits: list of (index, crystal, time, energy) for one event, in collection order."""
    hit_map = {c: [] for c in range(len(geom.disk))}
    seeds = []
    for h, (_, c, t, e) in enumerate(hits):
        if e < cfg.enoise_cut:
            continue
        hit_map[c].append(h)
        if e > cfg.emin_seed:
            seeds.append(h)
    seeds = sorted(seeds)
    mains, times = [], []
    while seeds:
        s = seeds[0]
        m = _finder(s, hit_map, hits, geom, cfg)
        mains.append(m)
        times.append(hits[s][2])
        seeds = [x for x in seeds if x not in m]
    for c in hit_map:
        hit_map[c] = [h for h in hit_map[c] if any(T - hits[h][2] < cfg.delta_time for T in times)]
    seeds = sorted(h for c in hit_map for h in hit_map[c])
    splits = []
    while seeds:
        m = _finder(seeds[0], hit_map, hits, geom, cfg)
        splits.append(m)
        seeds = [x for x in seeds if x not in m]
    mains.sort(key=lambda m: hits[m[0]][2])
    splits.sort(key=lambda m: hits[m[0]][2])

    def closest(a, b):
        return min(np.linalg.norm(geom.position[hits[i][1]] - geom.position[hits[j][1]]) for i in a for j in b)

    temp = [list(m) for m in mains]
    for s in splits:
        best, jmin = 1e6, -1
        for j, m in enumerate(mains):
            ts, tm = hits[s[0]][2], hits[m[0]][2]
            if ts - tm > cfg.delta_time or tm - ts > cfg.delta_time:
                continue
            if np.linalg.norm(geom.position[hits[m[0]][1]] - geom.position[hits[s[0]][1]]) > 2.5 * cfg.max_dist_split:
                continue
            d = closest(m, s)
            if d < cfg.max_dist_split and d < best:
                best, jmin = d, j
        if jmin >= 0:
            temp[jmin] += s
    assoc_to = [-1] * len(temp)
    children = [[] for _ in temp]
    for i in range(len(temp)):
        for j in range(i + 1, len(temp)):
            if assoc_to[j] > -1:
                continue
            dt = hits[temp[j][0]][2] - hits[temp[i][0]][2]
            if dt > cfg.delta_time:
                break
            if closest(temp[i], temp[j]) > cfg.max_dist_main:
                continue
            assoc_to[j] = i
            children[i].append(j)
    out, used = [], [False] * len(temp)
    for i in range(len(temp)):
        if used[i]:
            continue
        members, todo = list(temp[i]), list(children[i])
        while todo:
            j = todo.pop()
            used[j] = True
            members += temp[j]
            todo += children[j]
        members.sort(key=lambda h: -hits[h][3])
        out.append((sum(hits[h][3] for h in members), hits[members[0]][2], len(members)))
    return sorted(out, key=lambda c: -c[0])


def _event(rng, geom, n_hits):
    crystals = rng.choice(len(geom.disk), size=n_hits, replace=True)
    base = rng.choice(np.flatnonzero(geom.disk == 0))
    near = np.flatnonzero(geom.neighbors[base])
    crystals[: n_hits // 2] = rng.choice(np.r_[near, base], size=n_hits // 2)
    crystals = np.sort(crystals)
    times = (rng.normal(600.0, 1.5, n_hits)).astype(np.float32)
    energy = rng.exponential(8.0, n_hits).astype(np.float32) + np.float32(0.3)
    return crystals, times, energy


@pytest.mark.parametrize("seed", range(6))
def test_clusters_match_reference(seed):
    rng = np.random.default_rng(seed)
    geom = _geometry()
    cfg = ClusterConfig()
    evs, cr, tt, ee, refs = [], [], [], [], []
    for ev in range(25):
        c, t, e = _event(rng, geom, int(rng.integers(1, 30)))
        evs.append(np.full(c.size, ev))
        cr.append(c)
        tt.append(t)
        ee.append(e)
        refs.append(_reference([(i, int(c[i]), float(t[i]), float(e[i])) for i in range(c.size)], geom, cfg))
    hits = {"event": np.concatenate(evs), "crystal": np.concatenate(cr), "time": np.concatenate(tt),
            "energy": np.concatenate(ee)}
    got = make_clusters(hits, 25, geom, cfg)
    for ev, ref in enumerate(refs):
        sel = got["event"] == ev
        assert sel.sum() == len(ref)
        np.testing.assert_allclose(got["energy"][sel], [r[0] for r in ref], rtol=1e-6)
        np.testing.assert_array_equal(got["time"][sel], np.array([r[1] for r in ref], np.float32))
        np.testing.assert_array_equal(got["size"][sel], [r[2] for r in ref])


def test_cog_is_energy_weighted_in_disk_frame():
    geom = _geometry()
    hits = {"event": np.zeros(2, int), "crystal": np.array([144 + 13, 144 + 14]),
            "time": np.array([500.0, 500.5], np.float32), "energy": np.array([30.0, 10.0], np.float32)}
    got = make_clusters(hits, 1, geom)
    assert got["event"].size == 1 and got["disk"][0] == 1
    x = (30 * geom.position[157, 0] + 10 * geom.position[158, 0]) / 40
    np.testing.assert_allclose([got["x"][0], got["z"][0]], [x, 0.0])
