"""Muon crossings of a designed CRV and the per-muon probability that no sector makes a coincidence.

Crossings: per sector, the muon lines are rotated into the sector-local frame, where every bar is an axis-aligned box,
and crossed with the core slab method (tracks first culled by the sector's bounding box).

Veto probability, response fluctuations averaged analytically: per (muon, sector, layer, read-out end) the expected
PE of the crossed counters are summed (the finder's same-plus-adjacent-counter PE sum); the layer is hit if either
end has at least the sector's PE threshold (Poisson); the sector has a coincidence if at least k of its N layers are
hit (Poisson-binomial), k = Design.coincidence_layers, or N - Design.spare_layers, or by default N - (4 - stock
required layers), at least 1.
P_miss = prod over sectors (1 - P_coinc); a muon that crosses no counter has P_miss = 1. Optional coincidence groups
(p_miss `group`): sectors of one group are searched together, as CrvCoincidenceFinder searches all sectors of one
sector type (crs.sectorType) with counters numbered across them (crs.precedingSectorFor): a layer of the group is hit
if its summed crossed PE reaches the threshold, whichever member sector the counters belong to. With the finder cuts
(default in fom.evaluate) P_coinc also carries the finder's slope test and minimum cluster PE (finder_factor); its
time and distance cuts are not applied (one muon: hits in time, neighbouring counters).
"""
from __future__ import annotations

import numpy as np
from scipy.special import pdtr

from mu2efastsim import crossings as XC
from mu2efastsim import eloss, materials


def sector_crossings(dg, origin, direction, dtype=np.float64, chunk=4096):
    """Rows of crossings: track, bar, local entry [3], local exit [3] (counter-local, mm): (int [m, 2], float [m, 6])."""
    origin = np.asarray(origin, dtype)
    direction = np.asarray(direction, dtype)
    idx, pos = [], []
    for s in range(len(dg.sector_names)):
        bars = np.flatnonzero(dg.sector == s)
        if not len(bars):
            continue
        A = dg.axes[s].astype(dtype)
        o, d = origin @ A.T, direction @ A.T
        c = (dg.center[bars] @ dg.axes[s].T).astype(dtype)
        h = dg.halflen[bars].astype(dtype)
        lo, hi = c - h, c + h
        _, keep = _box_hits(o, d, lo.min(0), hi.max(0))
        tk = np.flatnonzero(keep)
        if not len(tk):
            continue
        x = XC.crossings(o[tk], d[tk], lo, hi, chunk=chunk)
        if not len(x):
            continue
        it = tk[x[:, 0].astype(np.int64)]
        ib = x[:, 1].astype(np.int64)
        p_in = o[it] + d[it] * x[:, 2:3].astype(dtype) - c[ib]
        p_out = o[it] + d[it] * x[:, 3:4].astype(dtype) - c[ib]
        idx.append(np.stack([it, bars[ib]], 1))
        pos.append(np.concatenate([p_in, p_out], 1))
    if not idx:
        return np.zeros((0, 2), np.int64), np.zeros((0, 6), dtype)
    return np.concatenate(idx), np.concatenate(pos).astype(dtype)


def _paths(axes, box_sector, center, halflen, origin, direction, chunk=4096):
    """Rows (track, box, t_in, t_out) of straight lines through boxes given in their sector-local frames."""
    out = []
    for s in np.unique(box_sector):
        boxes = np.flatnonzero(box_sector == s)
        A = axes[s]
        o, d = origin @ A.T, direction @ A.T
        c, h = center[boxes] @ A.T, halflen[boxes]
        lo, hi = c - h, c + h
        _, keep = _box_hits(o, d, lo.min(0), hi.max(0))
        tk = np.flatnonzero(keep)
        if not len(tk):
            continue
        x = XC.crossings(o[tk], d[tk], lo, hi, chunk=chunk)
        if len(x):
            out.append(np.stack([tk[x[:, 0].astype(np.int64)], boxes[x[:, 1].astype(np.int64)], x[:, 2], x[:, 3]], 1))
    return np.concatenate(out) if out else np.zeros((0, 4))


def stack_range(dg, origin, direction, p, idx, pos, t_stop=None):
    """Which bar crossings a muon still reaches through the designed stack, and which muons stop in it.

    Bar and absorber crossings are taken in path order; each costs the mean (Bethe) loss at the muon's entry
    momentum times its path (polystyrene, the design's absorber material); a bar crossing is kept while the loss
    before it is below the kinetic energy. stopped: the loss before t_stop (per muon, e.g. the entry into the CE-like
    region; default the whole line) reaches the kinetic energy. Returns (keep [m], stopped [n]).
    Approximation: no straggling, no loss outside the counters and absorbers (sheets, strongbacks, supports)."""
    origin, direction, p = (np.asarray(a, np.float64) for a in (origin, direction, p))
    n = len(p)
    cw = dg.center[idx[:, 1]] + np.einsum("mi,mij->mj", np.asarray(pos[:, :3], np.float64), dg.axes[dg.sector[idx[:, 1]]])
    t_bar = np.einsum("mi,mi->m", cw - origin[idx[:, 0]], direction[idx[:, 0]])
    path_bar = np.linalg.norm(np.asarray(pos[:, 3:] - pos[:, :3], np.float64), axis=1)
    ab = _paths(dg.axes, dg.abs_sector, dg.abs_center, dg.abs_halflen, origin, direction) if len(dg.abs_sector) \
        else np.zeros((0, 4))
    ta, tb = ab[:, 0].astype(np.int64), ab[:, 3] - ab[:, 2]
    trk = np.r_[idx[:, 0], ta]
    t = np.r_[t_bar, ab[:, 2]]
    is_bar = np.r_[np.ones(len(idx), bool), np.zeros(len(ab), bool)]
    dedx_s = eloss.dedx_restricted(materials.POLYSTYRENE, np.maximum(p, 1.0), np.inf)
    dedx_a = eloss.dedx_restricted(materials.get(dg.absorber_material), np.maximum(p, 1.0), np.inf)
    loss = np.r_[dedx_s[idx[:, 0]] * path_bar, dedx_a[ta] * tb]
    order = np.lexsort((t, trk))
    tr_o, l_o = trk[order], loss[order]
    cum = np.cumsum(l_o)
    start = np.r_[True, tr_o[1:] != tr_o[:-1]] if len(tr_o) else np.zeros(0, bool)
    seg = np.cumsum(start) - 1
    before = (cum - l_o) - (cum - l_o)[np.flatnonzero(start)][seg]     # loss before each element, per track
    ke = np.sqrt(p ** 2 + eloss.MMU ** 2) - eloss.MMU
    reached = np.empty(len(order), bool)
    reached[order] = before < ke[tr_o]
    keep = reached[:len(idx)]
    lim = np.full(n, np.inf) if t_stop is None else np.asarray(t_stop, np.float64)
    upto = np.zeros(n)
    w = t < lim[trk]
    np.add.at(upto, trk[w], loss[w])
    return keep, upto >= ke
def _box_hits(o, d, lo, hi):
    inv = 1.0 / np.where(d == 0, 1e-30, d)
    t1, t2 = (lo - o) * inv, (hi - o) * inv
    t_in = np.maximum(np.minimum(t1, t2).max(1), 0)
    t_out = np.maximum(t1, t2).min(1)
    return t_in, t_out > t_in


def required_layers(n_layers, stock_required, k=None, spare=None):
    """k of the k-of-N rule: ``k`` if given (Design.coincidence_layers), else N - ``spare`` (Design.spare_layers),
    else N minus the stock spare layers (stock: 3 of 4 or 4 of 4); at least 1. Raises if k exceeds N."""
    n = np.asarray(n_layers)
    if k is None:
        spare_n = 4 - np.asarray(stock_required) if spare is None else spare
        return np.maximum(1, n - spare_n)
    if np.any(k > n):
        raise ValueError(f"coincidence_layers {k} exceeds the layer count {n_layers}")
    return np.full(n.shape, int(k), np.int64)


def p_coincidence(mu_layer_side, thr, k):
    """mu_layer_side [g, N, 2] expected PE per group (muon, sector), layer and end; thr [g] PE; k [g] required layers.
    Returns P(at least k layers with an end >= thr) [g] (float64)."""
    mu = np.asarray(mu_layer_side, np.float64)
    kk = (np.ceil(np.asarray(thr, np.float64)) - 1.0)[:, None, None]
    below = np.where(mu > 0, pdtr(kk, mu), 1.0)            # P(N <= thr - 1) per end
    p = 1.0 - below.prod(2)                                 # [g, N]
    g, n = p.shape
    dist = np.zeros((g, n + 1))
    dist[:, 0] = 1.0
    for j in range(n):
        q = p[:, j:j + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - q) + dist[:, :-1] * q
        dist[:, 0] *= (1 - q[:, 0])
    cum = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]         # cum[:, m] = P(>= m)
    return cum[np.arange(g), np.asarray(k, np.int64)]


def finder_positions(dg, bars):
    """Hit positions of the stock finder for designed bars: (x, y) = the counter centre along the width and the
    thickness direction, in Mu2e coordinates of the sector's stock axes (CrvCoincidenceFinder uses the world
    components); a tilted module uses its own (rotated) width and thickness axes with the same signs."""
    bars = np.asarray(bars, np.int64)
    th = np.radians(dg.design.tilt_deg)
    A = dg.axes[dg.sector[bars]]                              # rows n_t, g_t, l (tilted)
    n0 = np.cos(th) * A[:, 0] - np.sin(th) * A[:, 1]          # untilted stock axes
    g0 = np.sin(th) * A[:, 0] + np.cos(th) * A[:, 1]
    sn = np.sign(n0[np.arange(len(bars)), np.abs(n0).argmax(1)])
    sg = np.sign(g0[np.arange(len(bars)), np.abs(g0).argmax(1)])
    c = dg.center[bars]
    return np.einsum("mi,mi->m", c, sg[:, None] * A[:, 1]), np.einsum("mi,mi->m", c, sn[:, None] * A[:, 0])


def _slope_ok(gid, x, y, lo, hi, msd, ng):
    """Per group (rows sorted by group, then layer; one row per hit layer): consecutive slopes dx/dy all in [lo, hi]
    with spread <= msd (the stock pairwise slope-difference test). Groups with one layer pass. lo/hi/msd per group."""
    ok = np.ones(ng, bool)
    same = gid[1:] == gid[:-1]
    if not same.any():
        return ok
    pg = gid[1:][same]
    sl = (x[1:] - x[:-1])[same] / (y[1:] - y[:-1])[same]
    bad = (sl < lo[pg]) | (sl > hi[pg])
    np.logical_and.at(ok, pg, ~bad)
    smin, smax = np.full(ng, np.inf), np.full(ng, -np.inf)
    np.minimum.at(smin, pg, sl)
    np.maximum.at(smax, pg, sl)
    return ok & ~((smax - smin) > msd)


def finder_factor(grp, layer, bar, pe_sides, dg, groups, k_grp, lo, hi, msd, min_pe):
    """Finder cuts the k-of-N rule leaves out, per (track, sector) group of `groups` (sorted keys of grp): the slope
    test on the most-lit crossed counter of each layer (if it fails and more than k layers are crossed, the first or
    the last layer may be dropped, as the finder may choose k of N) times P(cluster PE >= min_pe), Poisson about
    the summed expected PE. lo, hi, msd, min_pe, k_grp per group. Returns the factor per group."""
    gi = np.searchsorted(groups, np.asarray(grp, np.int64))
    pet = np.asarray(pe_sides, np.float64).sum(1)
    order = np.lexsort((-pet, np.asarray(layer), gi))
    g_o, l_o = gi[order], np.asarray(layer)[order]
    first = np.r_[True, (g_o[1:] != g_o[:-1]) | (l_o[1:] != l_o[:-1])]
    rows = order[first]                                         # one crossing per (group, layer), by group, layer
    x, y = finder_positions(dg, np.asarray(bar)[rows])
    g = gi[rows]
    ng = len(groups)
    nl = np.bincount(g, minlength=ng)
    ok = _slope_ok(g, x, y, lo, hi, msd, ng)
    start = np.r_[0, np.cumsum(nl)[:-1]]
    is_first, is_last = np.zeros(len(g), bool), np.zeros(len(g), bool)
    is_first[start[nl > 0]] = True
    is_last[(start + nl - 1)[nl > 0]] = True
    for drop in (is_first, is_last):
        keep = ~drop
        ok |= _slope_ok(g[keep], x[keep], y[keep], lo, hi, msd, ng) & (nl - 1 >= k_grp)
    tot = np.zeros(ng)
    np.add.at(tot, gi, pet)
    m = np.ceil(np.asarray(min_pe, np.float64))
    p_pe = np.where(m > 0, 1.0 - pdtr(np.maximum(m - 1.0, 0.0), tot), 1.0)
    return ok * p_pe


def group_index(names, groups=None):
    """Coincidence group of every sector [nsec] (int, the index of the group's first sector in `names`) from `groups`:
    a mapping (or pairs) sector name -> group label; sectors not listed are their own group. None or empty -> None
    (one group per sector). Raises on a name not in `names`."""
    if not groups:
        return None
    pairs = groups.items() if hasattr(groups, "items") else groups
    names = list(names)
    label = {n: n for n in names}
    for n, g in pairs:
        if n not in label:
            raise ValueError(f"coincidence group for unknown sector {n!r}")
        label[n] = g
    first = {}
    for i, n in enumerate(names):
        first.setdefault(label[n], i)
    return np.array([first[label[n]] for n in names], np.int64)


def _check_group_uniform(group, **per_sector):
    """Members of a coincidence group must share the parameters the group search uses (layers, threshold, k, cuts)."""
    for key, v in per_sector.items():
        v = np.asarray(v)
        bad = np.flatnonzero(v != v[group])
        if len(bad):
            raise ValueError(f"coincidence group of sector {int(bad[0])} mixes {key} values "
                             f"({v[bad[0]]} vs {v[group[bad[0]]]})")


def p_miss(n_tracks, track, sector, layer, pe_sides, n_layers_sec, thr_sec, k_sec, finder=None, group=None):
    """Per track P(no sector coincidence) [n_tracks] and per (track, sector) P_coinc rows (track, sector, p).
    track, sector, layer [m] per crossing; pe_sides [m, 2]. finder: None (k-of-N and threshold only) or dict(dg,
    bar [m], min_slope, max_slope, max_slope_diff, min_pe [nsec]) to apply the finder's slope and cluster-PE cuts
    (finder_factor). group: None (every sector alone) or [nsec] the coincidence group of each sector
    (group_index); a group's members must share n_layers, threshold, k and finder cuts. With groups, P_coinc is per
    (track, group) and the returned rows are per (track, crossed member sector), each with its group's P_coinc."""
    nsec = len(n_layers_sec)
    nmax = int(np.max(n_layers_sec)) if nsec else 1
    out = np.ones(n_tracks)
    if not len(track):
        return out, (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    sector = np.asarray(sector, np.int64)
    if group is not None:
        group = np.asarray(group, np.int64)
        if group.shape != (nsec,) or np.any(group[group] != group):
            raise ValueError("group must map every sector to a representative sector of its group")
        chk = dict(n_layers=n_layers_sec, threshold=thr_sec, k=k_sec)
        if finder is not None:
            chk.update({kk: finder[kk] for kk in ("min_slope", "max_slope", "max_slope_diff", "min_pe")})
        _check_group_uniform(group, **chk)
    gsec = sector if group is None else group[sector]
    grp = np.asarray(track, np.int64) * nsec + gsec
    ug, gi = np.unique(grp, return_inverse=True)
    mu = np.zeros((len(ug), nmax, 2))
    np.add.at(mu, (gi, np.asarray(layer, np.int64)), np.asarray(pe_sides, np.float64))
    gs = ug % nsec
    pc = p_coincidence(mu, np.asarray(thr_sec)[gs], np.asarray(k_sec)[gs])
    if finder is not None:
        f = finder
        pc = pc * finder_factor(grp, layer, f["bar"], pe_sides, f["dg"], ug, np.asarray(k_sec)[gs],
                                np.asarray(f["min_slope"])[gs], np.asarray(f["max_slope"])[gs],
                                np.asarray(f["max_slope_diff"])[gs], np.asarray(f["min_pe"])[gs])
    gt = ug // nsec
    np.multiply.at(out, gt, 1.0 - pc)
    if group is None:
        return out, (gt, gs, pc)
    mem = np.unique(np.asarray(track, np.int64) * nsec + sector)
    mt, ms = mem // nsec, mem % nsec
    return out, (mt, ms, pc[np.searchsorted(ug, mt * nsec + group[ms])])
