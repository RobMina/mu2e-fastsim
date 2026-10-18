"""Noise pulses for the direct path: reco pulses with no SiPM charge behind them (dark counts, crosstalk, electronics).

A stock waveform is called a noise waveform when none of its pulses has a pulse on another SiPM of the same counter
within PARTNER_NS (signal light reaches both ends / both SiPMs of an end; a lone pulse does not). The model, fitted on
data, generates noise waveforms per event:
  * count per event: negative binomial (Poisson with a gamma-distributed event rate; shape from the measured
    overdispersion), mean = sum of per-channel rates, times NoiseConfig.rate_scale
  * channel: drawn from the per-channel rates (noisy channels stay noisy)
  * time of the first pulse: from the measured histogram over the readout window
  * content: the whole pulse list of one data noise waveform on the same channel (count, PE, spacing, beta, failed
    flag), resampled jointly; channels with few noise waveforms draw from the whole library
An MC reference can be subtracted: its isolated waveforms are signal (MC noise makes no pulses), so the data rate minus
the MC rate per channel is the rate the direct path has to add. All of it is a tunable model, not a fixed truth.
"""
from dataclasses import dataclass, field

import numpy as np

PARTNER_NS = 20.0
N_MODELLED = 3
T_BINS = 40


def partner_flag(ev, chan, time, window=PARTNER_NS, nchan_per_bar=4, group=None, other=None):
    """True for pulses with a partner within window in the same event: by default another SiPM of the same counter;
    with group [nbars] / other [nbars], a pulse in the same group (e.g. sector) whose `other` value (e.g. layer)
    differs."""
    _, ei = np.unique(ev, return_inverse=True)
    bar = chan // nchan_per_bar
    key = ei.astype(np.int64) * 100000 + (bar if group is None else group[bar])
    cval = chan if other is None else other[bar]
    o = np.lexsort((time, key))
    ks, ts, cs = key[o], time[o], cval[o]
    out = np.zeros(len(time), bool)
    n = len(o)
    for d in range(1, n):
        lo = np.arange(n - d)
        same = (ks[lo + d] == ks[lo]) & (ts[lo + d] - ts[lo] <= window)
        if not same.any():
            break
        m = same & (cs[lo + d] != cs[lo])
        out[o[lo[m]]] = True
        out[o[lo[m] + d]] = True
    return out


def noise_waveforms(ev, chan, time, wf, window=PARTNER_NS, sector=None, layer=None):
    """Per pulse: waveform id (dense) and whether its waveform is a noise waveform: no pulse on another SiPM of the
    counter and (if sector/layer [nbars] are given) none in another layer of the same sector within window."""
    part = partner_flag(ev, chan, time, window)
    if sector is not None:
        part |= partner_flag(ev, chan, time, window, group=sector, other=layer)
    wk = np.stack([ev, wf], 1)
    _, wi = np.unique(wk, axis=0, return_inverse=True)
    wi = wi.ravel()
    has = np.bincount(wi, weights=part, minlength=wi.max() + 1 if len(wi) else 0) > 0
    return wi, ~has[wi]


@dataclass(frozen=True)
class NoiseConfig:
    rate_scale: float = 1.0          # multiplies every channel rate (tuning knob)
    overdispersed: bool = True       # gamma-distributed event rate; False = plain Poisson


@dataclass
class NoiseModel:
    rate: np.ndarray = field(default=None)       # [nchan] noise waveforms per event
    gamma_k: float = np.inf                      # negative-binomial shape of the per-event count
    t_edges: np.ndarray = field(default=None)    # [T_BINS+1]
    t_cdf: np.ndarray = field(default=None)      # [T_BINS]
    lib: np.ndarray = field(default=None)        # [L, 1 + 4*N_MODELLED]: n, then per pulse pe, dt, beta, failed
    lib_start: np.ndarray = field(default=None)  # [nchan] library rows sorted by channel
    lib_count: np.ndarray = field(default=None)  # [nchan]
    min_per_channel: int = 200                   # fewer library rows on a channel -> draw from the whole library

    @classmethod
    def fit(cls, P, n_events, nchan, window=PARTNER_NS, subtract_rate=None, sector=None, layer=None):
        """P: flat stock pulses (ev, chan, time, pe, beta, failed, wf). subtract_rate: [nchan] per-event rate of
        isolated signal waveforms (from MC) to remove from the data rate."""
        wi, noise = noise_waveforms(P["ev"], P["chan"], P["time"], P["wf"], window, sector, layer)
        idx = np.flatnonzero(noise)
        idx = idx[np.lexsort((P["time"][idx], wi[idx]))]
        w = wi[idx]
        first = np.r_[True, w[1:] != w[:-1]]
        start = np.flatnonzero(first)
        rank = np.arange(len(idx)) - np.repeat(start, np.diff(np.r_[start, len(idx)]))
        nw = len(start)
        g = np.cumsum(first) - 1
        lib = np.full((nw, 1 + 4 * N_MODELLED), np.nan)
        lib[:, 0] = np.bincount(g, minlength=nw)
        t0 = P["time"][idx[start]]
        m = rank < N_MODELLED
        for j, col in enumerate((P["pe"], P["time"], P["beta"], P["failed"].astype(float))):
            v = col[idx].astype(float)
            if j == 1:
                v = v - t0[g]
            lib[g[m], 1 + 4 * rank[m] + j] = v[m]
        ch = P["chan"][idx[start]]
        rate = np.bincount(ch, minlength=nchan) / n_events
        if subtract_rate is not None:
            rate = np.clip(rate - subtract_rate, 0.0, None)
        evw = P["ev"][idx[start]]
        _, cnt = np.unique(evw, return_counts=True)
        cnt = np.r_[cnt, np.zeros(max(n_events - len(cnt), 0))]
        mu, var = cnt.mean(), cnt.var()
        k = mu ** 2 / (var - mu) if var > mu else np.inf
        h, e = np.histogram(t0, bins=T_BINS)
        o = np.argsort(ch, kind="stable")
        lc = np.bincount(ch, minlength=nchan)
        return cls(rate=rate, gamma_k=float(k), t_edges=e, t_cdf=np.cumsum(h) / max(h.sum(), 1), lib=lib[o],
                   lib_start=np.r_[0, np.cumsum(lc)[:-1]], lib_count=lc)

    def sample(self, n_events, rng, cfg=NoiseConfig()):
        """Flat noise pulses: ev (0..n_events-1), chan, time, pe, beta, failed, seq (time order within a waveform)."""
        tot = self.rate.sum() * cfg.rate_scale
        lam = np.full(n_events, tot)
        if cfg.overdispersed and np.isfinite(self.gamma_k):
            lam = rng.gamma(self.gamma_k, tot / self.gamma_k, n_events)
        nw = rng.poisson(lam)
        W = int(nw.sum())
        ev = np.repeat(np.arange(n_events), nw)
        p = self.rate / max(self.rate.sum(), 1e-30)
        chan = np.searchsorted(np.cumsum(p), rng.random(W) * p.sum(), side="right")
        chan = np.minimum(chan, len(p) - 1)
        b = np.searchsorted(self.t_cdf, rng.random(W), side="right")
        b = np.minimum(b, T_BINS - 1)
        t0 = self.t_edges[b] + rng.random(W) * (self.t_edges[b + 1] - self.t_edges[b])
        own = self.lib_count[chan] >= self.min_per_channel
        pick = np.where(own, self.lib_start[chan] + (rng.random(W) * self.lib_count[chan]).astype(np.int64),
                        rng.integers(0, len(self.lib), W))
        r = self.lib[pick]
        n = np.minimum(r[:, 0].astype(np.int64), N_MODELLED)
        wi = np.repeat(np.arange(W), n)
        k = np.arange(len(wi)) - np.repeat(np.cumsum(n) - n, n)
        col = lambda j: r[wi, 1 + 4 * k + j]
        return {"ev": ev[wi], "chan": chan[wi], "time": t0[wi] + col(1), "pe": col(0), "beta": col(2),
                "failed": col(3) > 0.5, "seq": k, "wf": wi}

    def save(self, path):
        np.savez(path, rate=self.rate, gamma_k=self.gamma_k, t_edges=self.t_edges, t_cdf=self.t_cdf, lib=self.lib,
                 lib_start=self.lib_start, lib_count=self.lib_count, partner_ns=PARTNER_NS)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(rate=z["rate"], gamma_k=float(z["gamma_k"]), t_edges=z["t_edges"], t_cdf=z["t_cdf"], lib=z["lib"],
                   lib_start=z["lib_start"], lib_count=z["lib_count"])
