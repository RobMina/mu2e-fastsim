"""Synthetic CRV waveforms with KPP-data-like statistics, for benchmarks where the reference tables are not available
(Polaris: the Mu2e-internal reference data stay off the public repo and off ALCF).

Matched to the KPP run 124155 reference (2026-10-16): 20 waveforms per event; 99.6 % have 12 samples and 0.4 % 24;
56 % carry one pulse and 44 % two (the stock reco on data); first-pulse PE 4.6 / 16.7 / 57 / 129 at the 10/50/90/99 %
quantiles; beta 12.4 / 15.0 / 18.2 ns; pedestal ~2046 ADC with 2 ADC noise; pulse-area calibration ~500 ADC*ns/PE.
Only shapes and rates matter for timing; the pulse content is drawn, not taken from data.
"""
import numpy as np

from .pulsefit import PERIOD, gumbel

WF_PER_EVENT = 20
CALIB_AREA = 500.0
PEDESTAL = 2046.0


def waveforms(n_events, seed=1, L=12):
    """adc [W,L], n, start_tdc, pedestal, calib_area, calib_height for n_events events (fixed L; 24-sample waveforms
    are folded into L=12 batches, which is what dominates the cost)."""
    rng = np.random.default_rng(seed)
    W = n_events * WF_PER_EVENT
    start = rng.integers(10, 8000, W)
    t = (start[:, None] + np.arange(L)[None, :]) * PERIOD
    pe = np.exp(rng.normal(np.log(16.7), 1.0, W))
    beta = rng.normal(15.0, 2.2, W).clip(8, 30)
    A = pe * CALIB_AREA / beta
    mu = (start + 3) * PERIOD + rng.uniform(-6, 6, W)
    y = gumbel(t, A[:, None], mu[:, None], beta[:, None])
    two = rng.random(W) < 0.44
    pe2 = np.exp(rng.normal(np.log(6.0), 0.8, W))
    b2 = rng.normal(15.0, 2.2, W).clip(8, 30)
    y += np.where(two[:, None], gumbel(t, (pe2 * CALIB_AREA / b2)[:, None], mu[:, None] + rng.uniform(25, 60, W)[:, None],
                                       b2[:, None]), 0.0)
    adc = np.clip(np.round(y + PEDESTAL + rng.normal(0, 2.0, (W, L))), 0, 4095).astype(np.int64)
    return (adc, np.full(W, L), start, np.full(W, PEDESTAL), np.full(W, CALIB_AREA), np.full(W, CALIB_AREA / 15.0))


def coinc_events(n_events, G, seed=1, noise_mean=8.0, shower_frac=0.03, shower_mean=80.0):
    """Synthetic reco pulses for the coincidence-finder benchmark (no reference data on Polaris).

    Per event: one straight track (sector drawn with the KPP 124155 per-sector-type pulse shares, |slope| <= 1.5)
    giving both SiPMs of each readout side in every bar it crosses, PE lognormal around 15; Poisson(noise_mean)
    single noise pulses (PE ~3, random time); and in shower_frac of events Poisson(shower_mean) extra pulses in one
    module within +-15 ns (the large-group tail that sets the padded sizes). Returns flat arrays sorted by
    (event, channel) like the art collection: ev, idx, bar, sipm, time, pe, seq.
    """
    rng = np.random.default_rng(seed)
    nb = len(G.sector)
    share = {1: 0.71, 2: 0.07, 3: 0.18, 4: 0.04}
    sec_w = np.array([share[int(G.sector_type[s])] / np.sum(G.sector_type == G.sector_type[s])
                      for s in range(len(G.sector_names))])
    sec_w /= sec_w.sum()
    bars_of = [np.flatnonzero(G.sector == s) for s in range(len(G.sector_names))]
    b0 = bars_of[0][G.layer[bars_of[0]] == 0]
    width = np.median(np.diff(np.sort(G.x[b0])))
    rows = []

    def sides(s):
        return [k for k, on in ((0, G.sipms_side0[s]), (1, G.sipms_side1[s])) if on]
    secs = rng.choice(len(sec_w), n_events, p=sec_w)
    for e in range(n_events):
        s = secs[e]
        b = bars_of[s]
        x0 = rng.uniform(G.x[b].min(), G.x[b].max())
        slope = rng.uniform(-1.5, 1.5)
        y0 = G.y[b].mean()
        hit = b[np.abs(G.x[b] - (x0 + slope * (G.y[b] - y0))) < width / 2]
        t0 = rng.uniform(400, 1200)
        for bar in hit:
            for sd in sides(s):
                for sp in (sd, sd + 2):
                    rows.append((e, bar, sp, t0 + rng.normal(0, 2.0), rng.lognormal(np.log(15), 0.6)))
        for _ in range(rng.poisson(noise_mean)):
            bar = rng.integers(nb)
            rows.append((e, bar, rng.choice([k + 2 * j for k in sides(G.sector[bar]) for j in (0, 1)]),
                         rng.uniform(0, 2000), rng.lognormal(np.log(3), 0.8)))
        if rng.random() < shower_frac:
            bs = b[np.abs(G.x[b] - x0) < 8 * width]
            ts = rng.uniform(400, 1200)
            for _ in range(rng.poisson(shower_mean)):
                bar = rng.choice(bs)
                sd = rng.choice(sides(s))
                rows.append((e, bar, sd + 2 * rng.integers(2), ts + rng.uniform(-15, 15), rng.lognormal(np.log(10), 0.8)))
    a = np.array(rows, dtype=float)
    ev, bar, sipm = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64), a[:, 2].astype(np.int64)
    o = np.lexsort((a[:, 3], bar * 4 + sipm, ev))
    ev, bar, sipm, t, pe = ev[o], bar[o], sipm[o], a[o, 3], a[o, 4]
    first = np.searchsorted(ev, ev)
    return dict(ev=ev, idx=np.arange(len(ev)) - first, bar=bar, sipm=sipm, time=t, pe=pe, seq=np.zeros(len(ev), np.int64))


def chain_events(n_events, G, seed=1, L=12, **kw):
    """Synthetic digitised waveforms whose pulses make coincidences, for the pulses -> coincidences chain bench.

    The pulses of coinc_events (tracks, noise, showers) are each turned into one L-sample waveform (Gumbel of the
    pulse's PE and a drawn beta, KPP-like pedestal and noise, start 3 samples before the peak). Waveforms are ordered
    by (event, channel, time), the order the pulse reco hands to the coincidence finder. Returns the pulsefit inputs
    and per waveform: ev, bar, sipm.
    """
    rng = np.random.default_rng(seed + 7)
    p = coinc_events(n_events, G, seed=seed, **kw)
    W = len(p["ev"])
    start = np.floor(p["time"] / PERIOD).astype(np.int64) - 3
    t = (start[:, None] + np.arange(L)[None, :]) * PERIOD
    beta = rng.normal(15.0, 2.2, W).clip(8, 30)
    y = gumbel(t, (p["pe"] * CALIB_AREA / beta)[:, None], p["time"][:, None], beta[:, None])
    adc = np.clip(np.round(y + PEDESTAL + rng.normal(0, 2.0, (W, L))), 0, 4095).astype(np.int64)
    return dict(adc=adc, n=np.full(W, L), start=start, ped=np.full(W, PEDESTAL), area=np.full(W, CALIB_AREA),
                height=np.full(W, CALIB_AREA / 15.0), ev=p["ev"], bar=p["bar"], sipm=p["sipm"])
