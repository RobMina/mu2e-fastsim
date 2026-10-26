"""CRV coincidence finder: a port of Offline CRVReco CrvCoincidenceFinder (v13_39_00 = main 2026-10-16).

Per event and sector type (EX, T1, T2, taggers):
  1. hits = reco pulses (secondary pulses dropped if ignoreSecondaryPulses); time += fiber-length correction to the
     sector type's outermost readout end; PE doubled if the other SiPM on the same counter end has a compensate status
  2. filterHits per readout side: keep a hit if (same counter + one adjacent counter, same layer, |dt| <=
     maxTimeDifferenceAdjacentPulses) sum >= PEthreshold
  3. findClusters per side: greedy single-linkage in (|dx| <= maxDistance, |dt| < sector-type maxTimeDifference),
     in list order (order-dependent, as stock)
  4. checkCoincidence per cluster: combinations of 2/3/4 hits in distinct layers within the time window, slopes
     dx/dy within [minSlope, maxSlope] and pairwise within maxSlopeDifference; hits of passing combinations are kept,
     with maxDistance widened to the largest gap to a neighbour in a passing combination
  5. findClusters on the coincidence hits of both sides (opposite sides get + 2*halfLength/fiberSpeed)
  6. addOppositeSidePulses: attach filtered first pulses on missing opposite counter ends
  7. clusterProperties: PE sum, PE-weighted position, layers, slope, side PEs/times, hit time and position
This module is the reference implementation in numpy (loops over events and the few hits per sector; the event loop
is the parallel axis of the JAX twin). Output: one row per cluster with the pulse indices it contains.
"""
from dataclasses import dataclass, field
from itertools import chain, combinations, product

import numpy as np
import pandas as pd

N_LAYERS = 4


@dataclass(frozen=True)
class SectorCuts:
    pe_threshold: float
    max_dt_adjacent: float
    max_dt: float
    min_slope: float
    max_slope: float
    max_slope_diff: float
    coincidence_layers: int
    min_cluster_pes: float
    initial_max_distance: float
    final_max_distance: float


# epilog_extracted_v04 (the data operating point): EX 3 layers / 8 PE / 20 ns, T1/T2 and taggers 4 layers / 8 PE / 10 ns
EXTRACTED_V04 = {
    "EX": SectorCuts(8, 10, 20, -6, 6, 3, 3, 0, 250, 0),
    "T1": SectorCuts(8, 10, 10, -6, 6, 3, 4, 0, 250, 0),
    "T2": SectorCuts(8, 10, 10, -6, 6, 3, 4, 0, 250, 0),
    **{f"M{i}": SectorCuts(8, 10, 10, -6, 6, 3, 4, 0, 250, 0) for i in range(1, 9)},
}


@dataclass(frozen=True)
class FinderConfig:
    fiber_signal_speed: float = 140.0
    max_time_interval: float = 20.0
    time_offset: float = 34.5
    compensate_status_bits: tuple = (0, 1, 2)
    ignore_secondary_pulses: bool = True
    big_cluster_threshold: int = 50000
    cuts: dict = field(default_factory=lambda: dict(EXTRACTED_V04))
    add_opposite_side_pulses: bool = True


@dataclass
class SectorGeom:
    """Per-bar lookups the finder needs, from crvfastsim.geometry plus the counters-file sector settings."""
    sector: np.ndarray        # [nbars]
    layer: np.ndarray
    counter_in_type: np.ndarray   # precedingCounters + nCountersPerModule*module + counter
    x: np.ndarray             # position along the width direction
    y: np.ndarray             # position along the thickness direction
    pos: np.ndarray           # [nbars, 3] counter centre
    length_dir: np.ndarray    # [nsectors]
    sector_type: np.ndarray   # [nsectors]
    sector_names: list
    pos_side0: np.ndarray     # [nsectors] readout-end positions along the length direction
    pos_side1: np.ndarray
    sipms_side0: np.ndarray   # [nsectors] bool
    sipms_side1: np.ndarray


def sector_geom(geom, config_text_values):
    """Build SectorGeom from a CRVGeometry and the parsed counters file (geometry.parse_simpleconfig output)."""
    c = config_text_values
    names = geom.sector_names
    ns = len(names)
    st = np.array([int(c[f"crs.sectorType{n}"]) for n in names])
    prec = np.array([int(c[f"crs.precedingSectorFor{n}"]) for n in names])
    ncpm = np.array([int(c[f"crs.nCountersPerModule{n}"]) for n in names])
    nmod = np.array([int(c[f"crs.nModules{n}"]) for n in names])
    preceding = np.zeros(ns, np.int64)
    for i in range(ns):
        p = prec[i]
        while p != -1:
            preceding[i] += nmod[p] * ncpm[p]
            p = prec[p]
    dt, dw, dl = geom.local_to_world[:, 0], geom.local_to_world[:, 1], geom.local_to_world[:, 2]
    b = np.arange(geom.nbars)
    x = geom.center[b, dw]
    y = geom.center[b, dt]
    length_dir = np.array([dl[np.flatnonzero(geom.sector == s)[0]] for s in range(ns)])
    first = np.array([np.flatnonzero(geom.sector == s)[0] for s in range(ns)])
    half = geom.halflen[first, length_dir]
    centre_l = geom.center[first, length_dir]
    side0 = np.array([bool(c.get(f"crs.sipmsAtSide0{n}", True)) for n in names])
    side1 = np.array([bool(c.get(f"crs.sipmsAtSide1{n}", True)) for n in names])
    cit = preceding[geom.sector] + ncpm[geom.sector] * geom.module + geom.counter
    return SectorGeom(geom.sector, geom.layer, cit, x, y, geom.center, length_dir, st, list(names),
                      centre_l - half, centre_l + half, side0, side1)


def _type_limits(G, cfg):
    out = {}
    for t in np.unique(G.sector_type):
        secs = np.flatnonzero(G.sector_type == t)
        cuts = [cfg.cuts[G.sector_names[s]] for s in secs if G.sector_names[s] in cfg.cuts]
        if not cuts:
            continue
        p0, p1 = G.pos_side0[secs].min(), G.pos_side1[secs].max()
        out[int(t)] = dict(p0=p0, p1=p1, half=(p1 - p0) / 2, max_dt=max(c.max_dt for c in cuts),
                           side0=bool(G.sipms_side0[secs].any()), side1=bool(G.sipms_side1[secs].any()))
    return out


def _find_clusters(hits, max_dt, opposite=False, dt_opp=0.0):
    """Stock findClusters on a list of hit dicts (order matters)."""
    hits = list(hits)
    clusters = []
    while hits:
        cluster = []
        last = -1
        while last != len(cluster):
            last = len(cluster)
            i = 0
            while i < len(hits):
                h = hits[i]
                if not cluster:
                    cluster.append(h)
                    hits.pop(i)
                    continue
                joined = False
                for c in cluster:
                    md = max(h["maxd"], c["maxd"])
                    mt = max_dt + (dt_opp if opposite and h["side"] != c["side"] else 0.0)
                    if abs(h["x"] - c["x"]) <= md and abs(h["t"] - c["t"]) < mt:
                        cluster.append(h)
                        hits.pop(i)
                        joined = True
                        break
                if not joined:
                    i += 1
        clusters.append(cluster)
    return clusters


def _check_combination(combo):
    max_dt = max(h["cuts"].max_dt for h in combo)
    ts = [h["t"] for h in combo]
    if max(ts) - min(ts) > max_dt:
        return False
    lo = min(h["cuts"].min_slope for h in combo)
    hi = max(h["cuts"].max_slope for h in combo)
    msd = max(h["cuts"].max_slope_diff for h in combo)
    slopes = []
    for a, b in zip(combo[:-1], combo[1:]):
        s = (b["x"] - a["x"]) / (b["y"] - a["y"])
        if s < lo or s > hi:
            return False
        slopes.append(s)
    n = len(combo)
    if n > 2 and abs(slopes[0] - slopes[1]) > msd:
        return False
    if n > 3 and (abs(slopes[0] - slopes[2]) > msd or abs(slopes[1] - slopes[2]) > msd):
        return False
    srt = sorted(combo, key=lambda h: h["x"])
    for i, h in enumerate(srt):
        if i > 0:
            h["maxd"] = max(h["maxd"], h["x"] - srt[i - 1]["x"])
        if i + 1 < n:
            h["maxd"] = max(h["maxd"], srt[i + 1]["x"] - h["x"])
    return True


def _check_coincidence(cluster, cfg):
    if not cluster:
        return []
    by_layer = [[] for _ in range(N_LAYERS)]
    for h in cluster:
        hh = dict(h)
        hh["maxd"] = h["cuts"].final_max_distance
        by_layer[h["layer"]].append(hh)
    nmin = min(h["cuts"].coincidence_layers for h in cluster)
    nmax = max(h["cuts"].coincidence_layers for h in cluster)
    if len(cluster) > cfg.big_cluster_threshold and sum(1 for l in by_layer if l) >= nmin:
        return list(cluster)
    keep = {}
    for n, need in ((2, nmin == 2), (3, nmin <= 3 <= nmax), (4, nmax == 4)):
        if not need:
            continue
        for layers in combinations(range(N_LAYERS), n):
            for combo in product(*(by_layer[l] for l in layers)):
                if all(h["cuts"].coincidence_layers > n for h in combo):
                    continue
                if _check_combination(list(combo)):
                    for h in combo:
                        keep[h["idx"]] = h
    return [keep[k] for k in sorted(keep)]   # stock: a set ordered by the pulse Ptr (collection index)


def _check_coincidence_vec(cluster, cfg, chunk=2_000_000):
    """_check_coincidence with numpy broadcasting over the layer combinations (uniform cuts in the cluster only)."""
    c = cluster[0]["cuts"]
    if not cluster or any(h["cuts"] != c for h in cluster):
        return _check_coincidence(cluster, cfg)
    if len(cluster) > cfg.big_cluster_threshold and len({h["layer"] for h in cluster}) >= c.coincidence_layers:
        return list(cluster)
    n = c.coincidence_layers
    by_layer = [[i for i, h in enumerate(cluster) if h["layer"] == l] for l in range(N_LAYERS)]
    X = np.array([h["x"] for h in cluster]); Y = np.array([h["y"] for h in cluster]); T = np.array([h["t"] for h in cluster])
    wid = np.full(len(cluster), -np.inf)
    for layers in combinations(range(N_LAYERS), n):
        sets = [np.array(by_layer[l], np.int64) for l in layers]
        if any(len(q) == 0 for q in sets):
            continue
        grid = np.stack(np.meshgrid(*sets, indexing="ij"), -1).reshape(-1, n)
        for a in range(0, len(grid), chunk):
            cb = grid[a:a + chunk]
            x, y, t = X[cb], Y[cb], T[cb]
            ok = ~(t.max(1) - t.min(1) > c.max_dt)
            with np.errstate(divide="ignore", invalid="ignore"):
                sl = (x[:, 1:] - x[:, :-1]) / (y[:, 1:] - y[:, :-1])
            ok &= ~((sl < c.min_slope) | (sl > c.max_slope)).any(1)
            if n > 2:
                ok &= ~(np.abs(sl[:, 0] - sl[:, 1]) > c.max_slope_diff)
            if n > 3:
                ok &= ~((np.abs(sl[:, 0] - sl[:, 2]) > c.max_slope_diff) | (np.abs(sl[:, 1] - sl[:, 2]) > c.max_slope_diff))
            cb, x = cb[ok], x[ok]
            if not len(cb):
                continue
            o = np.argsort(x, axis=1, kind="stable")
            xs, hs = np.take_along_axis(x, o, 1), np.take_along_axis(cb, o, 1)
            gap = np.diff(xs, axis=1)
            w = np.zeros_like(xs)
            w[:, 1:] = gap
            w[:, :-1] = np.maximum(w[:, :-1], gap)
            np.maximum.at(wid, hs.ravel(), w.ravel())
    out = []
    for i in np.flatnonzero(wid > -np.inf):
        hh = dict(cluster[i])
        hh["maxd"] = max(c.final_max_distance, wid[i])
        out.append(hh)
    return sorted(out, key=lambda h: h["idx"])


def _attach_opposite_side_pulses(clusters, hits, max_dt):
    """Attach filtered hits using only original members, with stock cluster/candidate tie ordering."""
    if not clusters or not hits:
        return
    columns = ["idx", "bar", "sipm", "side", "t", "seq"]
    members = pd.DataFrame(chain.from_iterable(clusters), columns=columns)
    members["cluster"] = np.repeat(np.arange(len(clusters)), [len(cl) for cl in clusters])
    candidates = pd.DataFrame(hits, columns=columns)
    candidates["candidate"] = np.arange(len(hits))
    candidates = candidates.loc[(candidates.seq == 0) & ~candidates.idx.isin(members.idx)]
    if candidates.empty:
        return
    occupied = members[["cluster", "bar", "side"]].drop_duplicates().assign(occupied=True)
    anchors = members.loc[members.seq == 0, ["cluster", "bar", "side", "t"]].copy()
    anchors["side"] = 1 - anchors.side
    pairs = candidates.merge(anchors, on=["bar", "side"], suffixes=("", "_anchor"), sort=False)
    pairs = pairs.merge(occupied, on=["cluster", "bar", "side"], how="left", sort=False)
    pairs["dt"] = np.abs(pairs.t - pairs.t_anchor)
    pairs = pairs.loc[pairs.occupied.isna() & (pairs.dt < max_dt)]
    # First choose each candidate's nearest cluster; losing candidates are not reassigned.
    nearest = pairs.sort_values(["candidate", "dt", "cluster"], kind="stable").drop_duplicates("candidate")
    # Choose one pulse per SiPM, then append in the stock ordered-map key order.
    chosen = nearest.sort_values(["cluster", "bar", "sipm", "dt", "candidate"], kind="stable")
    chosen = chosen.drop_duplicates(["cluster", "bar", "sipm"])
    for cluster, candidate in chosen[["cluster", "candidate"]].itertuples(index=False, name=None):
        clusters[cluster].append(hits[candidate])


def find_event(pulses, G, status, cfg=FinderConfig(), vectorized=False):
    """pulses: dict of arrays for ONE event in collection order (idx, bar, sipm, time, pe, seq). Returns cluster rows.
    vectorized: numpy-broadcast combination check (same result; for the large events the JAX twin hands back)."""
    check = _check_coincidence_vec if vectorized else _check_coincidence
    lim = _type_limits(G, cfg)
    hits = {}
    for i in range(len(pulses["idx"])):
        if cfg.ignore_secondary_pulses and pulses["seq"][i] > 0:
            continue
        bar = int(pulses["bar"][i])
        s = int(G.sector[bar])
        name = G.sector_names[s]
        if name not in cfg.cuts:
            continue
        t_type = int(G.sector_type[s])
        sipm = int(pulses["sipm"][i])
        side = sipm % 2
        t = float(pulses["time"][i])
        if side == 0:
            t += (G.pos_side0[s] - lim[t_type]["p0"]) / cfg.fiber_signal_speed
        else:
            t += (lim[t_type]["p1"] - G.pos_side1[s]) / cfg.fiber_signal_speed
        pe = float(pulses["pe"][i])
        other = bar * 4 + sipm + (2 if sipm < 2 else -2)
        if any((int(status[other]) >> b) & 1 for b in cfg.compensate_status_bits):
            pe *= 2.0
        cuts = cfg.cuts[name]
        hits.setdefault(t_type, []).append(dict(idx=int(pulses["idx"][i]), bar=bar, sector=s, layer=int(G.layer[bar]),
                                               counter=int(G.counter_in_type[bar]), side=side, sipm=sipm,
                                               seq=int(pulses["seq"][i]), x=float(G.x[bar]),
                                               y=float(G.y[bar]), t=t, pe=pe, cuts=cuts,
                                               maxd=cuts.initial_max_distance))
    out = []
    for t_type in sorted(hits):
        H = hits[t_type]
        L = lim[t_type]
        coinc = []
        filtered = []
        for side in (0, 1):
            filt = []
            for h in H:
                if h["side"] != side:
                    continue
                same = adj1 = adj2 = 0.0
                for a in H:
                    if a["side"] != side or a["layer"] != h["layer"] or abs(a["t"] - h["t"]) > h["cuts"].max_dt_adjacent:
                        continue
                    d = a["counter"] - h["counter"]
                    same += a["pe"] if d == 0 else 0.0
                    adj1 += a["pe"] if d == -1 else 0.0
                    adj2 += a["pe"] if d == 1 else 0.0
                if same + adj1 >= h["cuts"].pe_threshold or same + adj2 >= h["cuts"].pe_threshold:
                    filt.append(h)
            filtered.extend(filt)
            for cl in _find_clusters(filt, L["max_dt"]):
                coinc += check(cl, cfg)
        both = L["side0"] and L["side1"]
        dt_opp = 2.0 * L["half"] / cfg.fiber_signal_speed
        clusters = _find_clusters(coinc, L["max_dt"], both, dt_opp)
        if both and cfg.add_opposite_side_pulses:
            _attach_opposite_side_pulses(clusters, filtered, L["max_dt"] + dt_opp)
        for cl in clusters:
            row = _properties(t_type, cl, G, L, cfg)
            if row is not None:
                out.append(row)
    return out


def _properties(t_type, cl, G, L, cfg):
    pe = np.array([h["pe"] for h in cl])
    tot = pe.sum()
    if tot < min(h["cuts"].min_cluster_pes for h in cl):
        return None
    x = np.array([h["x"] for h in cl])
    y = np.array([h["y"] for h in cl])
    pos = (np.array([G.pos[h["bar"]] for h in cl]) * pe[:, None]).sum(0) / tot
    slope = (tot * (pe * x * y).sum() - (pe * x).sum() * (pe * y).sum()) / (tot * (pe * y * y).sum() - (pe * y).sum() ** 2)
    t = np.array([h["t"] for h in cl])
    side = np.array([h["side"] for h in cl])
    ok = pe > 0
    nh = [int(((side == s) & ok).sum()) for s in (0, 1)]
    spe = [float(pe[(side == s) & ok].sum()) for s in (0, 1)]
    st = [float((t * pe)[(side == s) & ok].sum() / spe[s]) if spe[s] > 0 else 0.0 for s in (0, 1)]
    tmin = [t[(side == s) & ok].min() if nh[s] else 0.0 for s in (0, 1)]
    tmax = [t[(side == s) & ok].max() if nh[s] else 0.0 for s in (0, 1)]
    ld = int(G.length_dir[cl[0]["sector"]])
    calc = False
    if nh[0] and nh[1]:
        org = 0.5 * (cfg.fiber_signal_speed * (st[0] - st[1]) + (L["p0"] + L["p1"]))
        if L["p0"] < org < L["p1"] and tmax[0] - tmin[0] < cfg.max_time_interval and tmax[1] - tmin[1] < cfg.max_time_interval:
            calc = True
            pos[ld] = org
            avg_t = st[0] - (org - L["p0"]) / cfg.fiber_signal_speed
        else:
            org = pos[ld]
            avg_t = 0.5 * ((st[0] - (org - L["p0"]) / cfg.fiber_signal_speed) +
                           (st[1] - (L["p1"] - org) / cfg.fiber_signal_speed))
    else:
        org = pos[ld]
        avg_t = (st[0] - (org - L["p0"]) / cfg.fiber_signal_speed) if nh[0] else \
            (st[1] - (L["p1"] - org) / cfg.fiber_signal_speed)
    return dict(sectortype=t_type, start=float(t.min()), end=float(t.max()), pe=float(tot), slope=float(slope),
                layers=sorted({h["layer"] for h in cl}), pulses=[h["idx"] for h in cl], poscalc=calc,
                avgtime=float(avg_t - cfg.time_offset), pos=pos, sidepe=spe, sidetime=st, sidehits=nh)
