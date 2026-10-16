"""Direct path: SiPM charges -> reco pulses without waveforms, through a response model learnt from stock pairs.

Unit = a charge cluster (the clustering of waveform.charge_clusters, which is what the stock chain digitises as one
waveform). Features per cluster: total charge in PE, charge-weighted time quantiles (10 %, 50 %) and spread. Targets,
from the stock pulses assigned to the cluster (same channel, time inside [t_first - 2 periods, t_last + 1PE length]):
  * n_pulses (0 = below threshold / zero-suppressed / not found)
  * for the first pulse: PE / Q_PE, t - t10 (TDC0 and DB time offset removed), beta, failed-fit flag
Model = conditional empirical resampling: clusters binned in (Q_PE, prompt fraction); per bin the pulse-count
probabilities and the residual vectors of the training clusters are stored and resampled. v0 binned (log Q, time
spread) with 1.37x-wide charge bins, too coarse at the 6-10 PE threshold; v1 (2026-10-16) uses 0.5 PE bins up to
20 PE and the prompt fraction (charge in the first PROMPT_NS of the cluster over its total), which sets the peak height
the zero suppression and the peak finder cut on. Vectorised throughout; no per-cluster Python.
"""
from dataclasses import dataclass, field

import numpy as np

from .pulsefit import PERIOD
from .waveform import WaveformConfig, charge_clusters

Q_EDGES = np.r_[0.0, np.arange(1.0, 20.01, 0.5), np.geomspace(21.0, 2000.0, 30), np.inf]
F_EDGES = np.array([0.0, 0.3, 0.45, 0.6, 0.75, 0.9, 1.01])
PROMPT_NS = 15.0
MAX_PULSES = 4


def cluster_features(chan, time, pe, cfg=WaveformConfig()):
    """Per-cluster features. Charges grouped by channel and time-ordered within it."""
    cid = charge_clusters(chan, time, cfg)
    C = int(cid[-1]) + 1 if len(cid) else 0
    w = np.maximum(pe, 0.0)
    q = np.bincount(cid, weights=w, minlength=C)
    first = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]]) if C else np.zeros(0, np.int64)
    last = np.r_[first[1:] - 1, len(cid) - 1] if C else np.zeros(0, np.int64)
    cw = np.cumsum(w)
    base = np.r_[0.0, cw][first]
    frac = (cw - base[cid]) / np.maximum(q[cid], 1e-12)

    def quant(f):
        hit = frac >= f
        idx = np.full(C, len(time) - 1)
        pos = np.flatnonzero(hit)
        np.minimum.at(idx, cid[pos], pos)
        return time[idx]

    t10, t50 = quant(0.1), quant(0.5)
    mean = np.bincount(cid, weights=w * time, minlength=C) / np.maximum(q, 1e-12)
    var = np.bincount(cid, weights=w * (time - mean[cid]) ** 2, minlength=C) / np.maximum(q, 1e-12)
    prompt = np.bincount(cid, weights=w * (time - time[first][cid] < PROMPT_NS), minlength=C)
    return {"cid": cid, "chan": chan[first], "q": q, "t10": t10, "t50": t50, "spread": np.sqrt(var),
            "pfrac": prompt / np.maximum(q, 1e-12), "tfirst": time[first], "tlast": time[last],
            "n": np.bincount(cid, minlength=C)}


def assign_pulses(feat, p_chan, p_time, cfg=WaveformConfig()):
    """Cluster index of each stock pulse (-1 if none): same channel, inside the cluster's time span."""
    C = len(feat["q"])
    order = np.lexsort((feat["tfirst"], feat["chan"]))
    keys = feat["chan"][order].astype(np.int64)
    lo = feat["tfirst"][order] - 2 * PERIOD
    hi = feat["tlast"][order] + cfg.single_pe_max_time
    out = np.full(len(p_time), -1, np.int64)
    if C == 0 or len(p_time) == 0:
        return out
    # last cluster of the same channel starting before the pulse
    comp = keys.astype(float) * 1e7 + lo
    pc = p_chan.astype(float) * 1e7 + p_time
    j = np.searchsorted(comp, pc, side="right") - 1
    ok = (j >= 0) & (keys[np.clip(j, 0, C - 1)] == p_chan) & (p_time <= hi[np.clip(j, 0, C - 1)])
    out[ok] = order[j[ok]]
    return out


NBINS = (len(Q_EDGES) - 1) * (len(F_EDGES) - 1)


def _bin(feat):
    iq = np.clip(np.searchsorted(Q_EDGES, feat["q"], side="right") - 1, 0, len(Q_EDGES) - 2)
    jf = np.clip(np.searchsorted(F_EDGES, feat["pfrac"], side="right") - 1, 0, len(F_EDGES) - 2)
    return iq * (len(F_EDGES) - 1) + jf


@dataclass
class DirectModel:
    """Conditional empirical resampling per (Q, spread) bin."""
    p_npulse: np.ndarray = field(default=None)       # [B, MAX_PULSES+1]
    res_start: np.ndarray = field(default=None)      # [B] start index into res
    res_count: np.ndarray = field(default=None)      # [B]
    res: np.ndarray = field(default=None)            # [R, 4]: pe_ratio, dt, beta, failed

    @classmethod
    def fit(cls, feat, n_pulses, first):
        """feat: cluster_features; n_pulses [C]; first: dict of first-pulse arrays per cluster (NaN where none)."""
        B = NBINS
        b = _bin(feat)
        npc = np.minimum(n_pulses, MAX_PULSES)
        cnt = np.zeros((B, MAX_PULSES + 1))
        np.add.at(cnt, (b, npc), 1.0)
        prior = cnt.sum(0) + 1.0
        p = (cnt + prior / prior.sum()) / (cnt.sum(1, keepdims=True) + 1.0)
        has = n_pulses > 0
        vec = np.stack([first["pe"] / np.maximum(feat["q"], 1e-9), first["dt"], first["beta"],
                        first["failed"].astype(float)], 1)[has]
        bb = b[has]
        order = np.argsort(bb, kind="stable")
        rc = np.bincount(bb, minlength=B)
        rs = np.r_[0, np.cumsum(rc)[:-1]]
        return cls(p_npulse=p, res_start=rs, res_count=rc, res=vec[order])

    def sample(self, feat, rng):
        """Draw (n_pulses, first-pulse pe, time, beta, failed) per cluster. Empty bins borrow the nearest filled bin."""
        b = _bin(feat)
        cum = np.cumsum(self.p_npulse[b], 1)
        npul = (rng.random(len(b))[:, None] > cum).sum(1)
        filled = np.flatnonzero(self.res_count > 0)
        nb = filled[np.clip(np.searchsorted(filled, b), 0, len(filled) - 1)]
        bb = np.where(self.res_count[b] > 0, b, nb)
        pick = self.res_start[bb] + (rng.random(len(b)) * self.res_count[bb]).astype(np.int64)
        r = self.res[pick]
        has = npul > 0
        return {"n": npul, "pe": np.where(has, r[:, 0] * feat["q"], np.nan),
                "time": np.where(has, feat["t10"] + r[:, 1], np.nan), "beta": np.where(has, r[:, 2], np.nan),
                "failed": has & (r[:, 3] > 0.5)}

    def save(self, path):
        np.savez(path, p_npulse=self.p_npulse, res_start=self.res_start, res_count=self.res_count, res=self.res,
                 q_edges=Q_EDGES, f_edges=F_EDGES, prompt_ns=PROMPT_NS)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(p_npulse=z["p_npulse"], res_start=z["res_start"], res_count=z["res_count"], res=z["res"])
