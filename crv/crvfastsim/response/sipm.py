"""SiPM charges from photon arrivals: a port of Offline CRVResponse MakeCrvSiPMCharges + CrvSiPMChargeGenerator.

Stock model, per SiPM (an event queue in time order):
  * each photon lands on a pixel drawn from the fibre photon map (TH2F 40 x 40, GetRandom2 then lrint, so a pixel id
    can be 40); pixels in the inactive 4 x 4 centre are dead; photons after the window end are dropped;
  * a pixel fires with probability p1 (1 - exp(-v / p2)), where v = Vov (1 - exp(-(t - t_last) / tau)) recovers after
    its previous discharge (v = Vov if never fired); a firing gives charge C v, i.e. v / Vov PE;
  * each firing schedules cross-talk at the same time on each existing 4-neighbour with probability
    (1 - (1 - P_ct)^(1/4)) / P_fire(Vov); trap afterpulses (probabilities 0 here) and thermal noise
    (ThermalRate, 0 in the production extracted digitisation: digitize/NoField.fcl) are also ported.
Vectorised across SiPMs: SiPMs are ordered by photon count, and iteration k processes the k-th photon of every SiPM
that has one, then its cross-talk cascade breadth-first (a pixel hit m times in one cascade generation fires with
1 - (1 - p)^m: the stock order processes them one after the other, the first firing leaves v = 0 for the rest).
Pixel state is a dense [SiPMs, 41, 41] array of last-discharge times, chunked over SiPMs.
"""
from dataclasses import dataclass, field

import numpy as np

NPIX = 41  # pixel ids 0..40 (lrint of a coordinate in [0, 40] can give 40)


@dataclass
class SiPMConfig:
    n_pixels_x: int = 40
    n_pixels_y: int = 40
    overvoltage: float = 3.0
    time_constant: float = 13.3
    capacitance: float = 8.84e-14
    p1: float = 0.607
    p2: float = 2.7
    cross_talk: float = 0.04
    thermal_rate: float = 0.0
    trap0_prob: float = 0.0
    trap1_prob: float = 0.0
    trap0_lifetime: float = 5.0
    trap1_lifetime: float = 50.0
    inactive: tuple = field(default_factory=lambda: tuple((x, y) for x in range(18, 22) for y in range(18, 22)))

    def p_fire(self, v):
        return self.p1 * (1.0 - np.exp(-v / self.p2))


def photon_map(path=None):
    """The fibre photon map as a [40, 40] array indexed [x, y] (bin contents of photonMap.root)."""
    import os
    import uproot
    from . import lookup
    path = path or os.path.join(lookup.lookup_dir(), "photonMap.root")
    with uproot.open(path) as f:
        return f["photonMap"].to_numpy()[0].astype(float)


def draw_pixels(pmap, n, rng):
    """GetRandom2 + lrint: a bin by content, a uniform point in it, rounded to the nearest integer pixel."""
    cum = np.cumsum(pmap.ravel())
    b = np.searchsorted(cum, rng.random(n) * cum[-1], side="right")
    bx, by = b // pmap.shape[1], b % pmap.shape[1]
    return bx + (rng.random(n) >= 0.5), by + (rng.random(n) >= 0.5)


def _neighbours(x, y, cfg):
    """[n, 4] neighbour pixel ids and validity (stock FindCrossTalkPixelIds bounds)."""
    nx = np.stack([x - 1, x + 1, x, x], 1)
    ny = np.stack([y, y, y - 1, y + 1], 1)
    ok = np.stack([x > 0, x + 1 < cfg.n_pixels_x, y > 0, y + 1 < cfg.n_pixels_y], 1)
    return nx, ny, ok


def simulate(chan, time, photon, pmap, cfg, rng, t_end=1e5, t_start=0.0, thermal_chans=None, chunk=20000):
    """SiPM charges for photon arrivals.

    chan [P] SiPM channel per photon, time [P] ns, photon [P] an id carried to the charges (e.g. the step index).
    thermal_chans: channels that get thermal noise (all simulated SiPMs in the stock module); ignored if the rate is 0.
    Returns dict of arrays per charge: chan, time, charge (C), pe, photon (-1 for dark noise).
    """
    chan, time, photon = np.asarray(chan), np.asarray(time, float), np.asarray(photon)
    keep = time <= t_end
    chan, time, photon = chan[keep], time[keep], photon[keep]
    dark = np.zeros(len(chan), bool)
    if cfg.thermal_rate > 0 and thermal_chans is not None:
        lam = cfg.thermal_rate / cfg.p_fire(cfg.overvoltage) * (t_end - t_start)
        nth = rng.poisson(lam, len(thermal_chans))
        tc = np.repeat(np.asarray(thermal_chans), nth)
        chan = np.r_[chan, tc]
        time = np.r_[time, t_start + (t_end - t_start) * rng.random(len(tc))]
        photon = np.r_[photon, np.full(len(tc), -1)]
        dark = np.r_[dark, np.ones(len(tc), bool)]
    px, py = draw_pixels(pmap, len(chan), rng)
    if dark.any():
        px[dark] = rng.integers(0, cfg.n_pixels_x, dark.sum())
        py[dark] = rng.integers(0, cfg.n_pixels_y, dark.sum())
    dead = np.zeros((NPIX, NPIX), bool)
    for x, y in cfg.inactive:
        dead[x, y] = True
    order = np.lexsort((time, chan))
    chan, time, photon, px, py = chan[order], time[order], photon[order], px[order], py[order]
    u, start, cnt = np.unique(chan, return_index=True, return_counts=True)
    by_cnt = np.argsort(-cnt, kind="stable")
    outs = []
    for c0 in range(0, len(u), chunk):
        sl = by_cnt[c0:c0 + chunk]
        outs.append(_simulate_chunk(u[sl], start[sl], cnt[sl], time, photon, px, py, dead, cfg, rng, t_end))
    if not outs:
        return {k: np.zeros(0) for k in ("chan", "time", "charge", "pe", "photon")}
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _simulate_chunk(chans, start, cnt, time, photon, px, py, dead, cfg, rng, t_end):
    S = len(chans)
    last = np.full((S, NPIX * NPIX), -np.inf)
    q_ct = (1.0 - (1.0 - cfg.cross_talk) ** 0.25) / cfg.p_fire(cfg.overvoltage)
    q_t0 = cfg.trap0_prob / cfg.p_fire(cfg.overvoltage)
    q_t1 = cfg.trap1_prob / cfg.p_fire(cfg.overvoltage)
    rec = {k: [] for k in ("slot", "time", "v", "photon")}
    traps = []  # (slot, pixel, time, photon) scheduled afterpulses
    for k in range(int(cnt.max())):
        n = int((cnt > k).sum())                              # slots sorted by count: active ones are a prefix
        idx = start[:n] + k
        slot = np.arange(n)
        pix = px[idx] * NPIX + py[idx]
        frontier = (slot, pix, time[idx], photon[idx], np.ones(n, np.int64))
        while len(frontier[0]):
            slot, pix, t, ph, m = frontier
            alive = ~dead.ravel()[pix]
            slot, pix, t, ph, m = slot[alive], pix[alive], t[alive], ph[alive], m[alive]
            v = cfg.overvoltage * (1.0 - np.exp(-(t - last[slot, pix]) / cfg.time_constant))
            p = 1.0 - (1.0 - cfg.p_fire(v)) ** m
            fire = rng.random(len(slot)) < p
            slot, pix, t, ph, v = slot[fire], pix[fire], t[fire], ph[fire], v[fire]
            last[slot, pix] = t
            rec["slot"].append(slot), rec["time"].append(t), rec["v"].append(v), rec["photon"].append(ph)
            if q_t0 > 0 or q_t1 > 0:
                for q, life in ((q_t0, cfg.trap0_lifetime), (q_t1, cfg.trap1_lifetime)):
                    tr = rng.random(len(slot)) < q
                    traps.append((slot[tr], pix[tr], t[tr] - life * np.log(rng.random(tr.sum())), ph[tr]))
            nx, ny, ok = _neighbours(pix // NPIX, pix % NPIX, cfg)
            ok &= rng.random(ok.shape) < q_ct
            r = np.nonzero(ok)
            cs, cp, ct, cph = slot[r[0]], (nx * NPIX + ny)[r], t[r[0]], ph[r[0]]
            if len(cs):
                key = cs * (NPIX * NPIX) + cp
                uk, first, mult = np.unique(key, return_index=True, return_counts=True)
                frontier = (cs[first], cp[first], ct[first], cph[first], mult)
            else:
                frontier = (cs, cp, ct, cph, np.zeros(0, np.int64))
    if traps:
        # afterpulses are rare (probabilities 0 in the production configuration): processed after the photons,
        # in time order per SiPM, without further cross-talk cascades between them and later photons
        ts = np.concatenate([x[0] for x in traps])
        tp, tt, tph = (np.concatenate([x[i] for x in traps]) for i in (1, 2, 3))
        o = np.lexsort((tt, ts))
        for sl, pp, t1, ph1 in zip(ts[o], tp[o], tt[o], tph[o]):
            if t1 > t_end:
                continue
            v = cfg.overvoltage * (1.0 - np.exp(-(t1 - last[sl, pp]) / cfg.time_constant))
            if rng.random() < cfg.p_fire(v):
                last[sl, pp] = t1
                rec["slot"].append(np.array([sl])), rec["time"].append(np.array([t1]))
                rec["v"].append(np.array([v])), rec["photon"].append(np.array([ph1]))
    slot = np.concatenate(rec["slot"])
    v = np.concatenate(rec["v"])
    out = {"chan": chans[slot], "time": np.concatenate(rec["time"]), "charge": cfg.capacitance * v,
           "pe": v / cfg.overvoltage, "photon": np.concatenate(rec["photon"])}
    o = np.lexsort((out["time"], out["chan"]))
    return {k: a[o] for k, a in out.items()}
