"""Out-of-time CRV background for the fast chain: single-bar light deposits that the cosmic MC does not simulate.

In KPP data (crvreco-005-001, runs 124336/124571/125026) the unclustered pulses far from every cluster are mostly real
light, not electronics noise: 62-78 % have a pulse on the other SiPM of the same bar within 20 ns, groups of pulses
close in time are 92 % one bar and one layer, the median is ~15 PE and they are flat in the readout window. The
stock and fast MC make ~10 % of that rate. reco/noise.py models lone-SiPM noise waveforms and misses the rest.

Model (fitted on data EventNtuples, MC subtracted):
  * a group = the out-of-time pulses (crvHitIndex < 0, more than OOT_DT from every cluster, all sequence indices) of
    one event and one sector slot, chained in time with gaps <= GAP_NS;
  * library: every data group, with each pulse's barId, SiPMId, PE, time relative to the group's first pulse and seq;
  * rate: data groups minus MC groups per event per ns (the MC's own background is already in the fast chain);
    count per event negative binomial with the data overdispersion; group time uniform in the event window;
  * sample(): groups drawn uniformly from the library, so the bar, sector and PE distributions follow data.
The library is data-derived (Mu2e-internal): keep its npz with the conditions (mode 700), not in the image.
"""
from dataclasses import dataclass, field

import awkward as ak
import numpy as np

from ..objective import bars

OOT_DT = 500.0          # ns from every cluster (crvresponse.OOT_DT)
GAP_NS = 30.0           # max time gap between consecutive pulses of one group
TREE = "EventNtuple/ntuple"
PULSE_BR = ["crvpulses.time", "crvpulses.PEs", "crvpulses.barId", "crvpulses.SiPMId", "crvpulses.crvHitIndex",
            "crvpulses.sequenceIndex"]


def oot_groups(p, c):
    """Out-of-time groups of one block of events. p, c: jagged crvpulses (PULSE_BR) / crvcoincs (crvcoincs.time).
    Returns flat per-pulse arrays (ev local index, group id dense, bar, sipm, pe, dt to the group start, seq) and the
    number of groups per event."""
    pt, ct = p["crvpulses.time"], c["crvcoincs.time"]
    a, b = ak.unzip(ak.cartesian([pt, ct], axis=1, nested=True))
    dmin = ak.fill_none(ak.min(np.abs(a - b), axis=2), np.inf)
    sel = ak.to_numpy(ak.flatten((p["crvpulses.crvHitIndex"] < 0) & (dmin > OOT_DT)))
    n = len(pt)
    ev = np.repeat(np.arange(n), ak.to_numpy(ak.num(pt)))[sel]
    f = lambda k: ak.to_numpy(ak.flatten(p[k]))[sel]
    t, bar, sipm, pe, seq = f("crvpulses.time"), f("crvpulses.barId").astype(np.int64), f("crvpulses.SiPMId").astype(
        np.int64), f("crvpulses.PEs").astype(np.float64), f("crvpulses.sequenceIndex").astype(np.int64)
    slot, _, _ = bars.bar_slot_body_layer(bar)
    o = np.lexsort((t, slot, ev))
    ev, t, bar, sipm, pe, seq, slot = ev[o], t[o], bar[o], sipm[o], pe[o], seq[o], slot[o]
    new = np.r_[True, (ev[1:] != ev[:-1]) | (slot[1:] != slot[:-1]) | (t[1:] - t[:-1] > GAP_NS)] if len(t) else np.zeros(0, bool)
    g = np.cumsum(new) - 1
    t0 = t[new][g] if len(t) else t
    return {"ev": ev, "group": g, "bar": bar, "sipm": sipm, "pe": pe, "dt": t - t0, "seq": seq,
            "groups_per_event": np.bincount(ev[new], minlength=n), "events": n}


LIB_KEYS = ("group", "bar", "sipm", "pe", "dt", "seq")


def scan_file(path, keep_frac=1.0, seed=0, entry_stop=None, chunk=100_000):
    """One EventNtuple file -> partial fit: events, exposure [event ns], groups, sum and sum of squares of the groups per
    event (overdispersion), and a library of a random keep_frac of the groups (bounded memory). Chunked reads."""
    import uproot
    rng = np.random.default_rng(seed)
    t = uproot.open(path)[TREE]
    n = t.num_entries if entry_stop is None else min(entry_stop, t.num_entries)
    part = {"events": 0, "exposure_ns": 0.0, "groups": 0, "cnt_sum": 0.0, "cnt_sumsq": 0.0, "windows": []}
    lib, off = {k: [] for k in LIB_KEYS}, 0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        p = t.arrays(PULSE_BR, entry_start=a, entry_stop=b)
        c = t.arrays(["crvcoincs.time"], entry_start=a, entry_stop=b)
        tmax = float(ak.max(p["crvpulses.time"], axis=None) or 0.0)
        w = max(1e5, np.round(tmax / 1e5) * 1e5)                   # readout window (data 400 us, MC 100 us)
        G = oot_groups(p, c)
        cnt = G["groups_per_event"]
        ng = int(cnt.sum())
        part["events"] += G["events"]
        part["exposure_ns"] += G["events"] * w
        part["groups"] += ng
        part["cnt_sum"] += float(cnt.sum())
        part["cnt_sumsq"] += float((cnt.astype(float) ** 2).sum())
        part["windows"].append(w)
        if keep_frac > 0 and ng:
            keep = rng.random(ng) < keep_frac
            m = keep[G["group"]]
            newid = np.cumsum(keep) - 1 + off
            for k in LIB_KEYS:
                lib[k].append(newid[G["group"][m]] if k == "group" else G[k][m])
            off += int(keep.sum())
    part["windows"] = sorted(set(part["windows"]))
    part["lib"] = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in lib.items()}
    part["keep_frac"] = keep_frac
    part["file"] = str(path)
    return part


def save_partial(path, part):
    import json
    meta = {k: v for k, v in part.items() if k != "lib"}
    np.savez(path, meta=json.dumps(meta), **{f"lib_{k}": v for k, v in part["lib"].items()})


def load_partial(path):
    import json
    z = np.load(path)
    part = json.loads(str(z["meta"]))
    part["lib"] = {k: z[f"lib_{k}"] for k in LIB_KEYS}
    return part


@dataclass
class BackgroundModel:
    rate_per_ns: float = 0.0                      # net (data - MC) groups per event per ns
    gamma_k: float = np.inf                       # negative-binomial shape of the per-event group count
    start: np.ndarray = field(default=None)       # [G] first pulse row of each library group
    count: np.ndarray = field(default=None)       # [G] pulses per group
    bar: np.ndarray = field(default=None)         # [P] per library pulse
    sipm: np.ndarray = field(default=None)
    pe: np.ndarray = field(default=None)
    dt: np.ndarray = field(default=None)
    seq: np.ndarray = field(default=None)
    info: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, data_paths, mc_paths=(), entry_stop=None, keep_frac=0.003, seed=20261022):
        """Small fits in one process: scan_file per file, then from_partials."""
        dp = [scan_file(p, keep_frac, seed + i, entry_stop) for i, p in enumerate(data_paths)]
        mp = [scan_file(p, 0.0, 0, entry_stop) for p in mc_paths]
        return cls.from_partials(dp, mp)

    @classmethod
    def from_partials(cls, data_parts, mc_parts=()):
        """Net rate (data - MC groups per event per ns, every group counted), overdispersion of the data group count, and
        the library (the kept groups of every data file; same keep_frac, so a uniform subset)."""
        sm = lambda parts, k: sum(p[k] for p in parts)
        nev, expo, ng = sm(data_parts, "events"), sm(data_parts, "exposure_ns"), sm(data_parts, "groups")
        r_data = ng / expo if expo else 0.0
        r_mc = (sm(mc_parts, "groups") / sm(mc_parts, "exposure_ns")) if mc_parts and sm(mc_parts, "exposure_ns") else 0.0
        mu = sm(data_parts, "cnt_sum") / max(nev, 1)
        var = sm(data_parts, "cnt_sumsq") / max(nev, 1) - mu ** 2
        k = float(mu ** 2 / (var - mu)) if var > mu else np.inf
        L, off = {kk: [] for kk in LIB_KEYS}, 0
        for p in data_parts:
            g = np.asarray(p["lib"]["group"], np.int64)
            for kk in LIB_KEYS:
                L[kk].append(g + off if kk == "group" else np.asarray(p["lib"][kk]))
            off += int(g.max()) + 1 if len(g) else 0
        L = {kk: np.concatenate(v) if v else np.zeros(0) for kk, v in L.items()}
        o = np.argsort(L["group"], kind="stable")
        L = {kk: v[o] for kk, v in L.items()}
        nlib = int(L["group"].max()) + 1 if len(L["group"]) else 0
        count = np.bincount(L["group"].astype(np.int64), minlength=nlib)
        start = np.r_[0, np.cumsum(count)[:-1]]
        info = {"data_window_ns": float(expo / max(nev, 1)), "data_files": [p["file"] for p in data_parts],
                "mc_files": [p["file"] for p in mc_parts], "data_events": int(nev), "data_groups": int(ng),
                "library_groups": nlib, "keep_frac": sorted({p["keep_frac"] for p in data_parts}),
                "data_rate_per_event_per_us": 1e3 * r_data, "mc_rate_per_event_per_us": 1e3 * r_mc,
                "pulses_per_group": float(len(L["bar"]) / max(nlib, 1)), "gamma_k_window": k, "gap_ns": GAP_NS,
                "oot_dt_ns": OOT_DT}
        return cls(rate_per_ns=max(r_data - r_mc, 0.0), gamma_k=k, start=start, count=count,
                   bar=L["bar"].astype(np.int64), sipm=L["sipm"].astype(np.int64), pe=L["pe"], dt=L["dt"],
                   seq=L["seq"].astype(np.int64), info=info)

    def sample(self, events, window, rng, rate_scale=1.0, overdispersed=True):
        """Background pulses for event keys `events` in the window (t0, t1) [ns]: dict ev, bar, sipm, pe, time, seq
        (flat, unsorted). rate_scale multiplies the net rate (the BO knob)."""
        events = np.asarray(events, np.int64)
        n, (w0, w1) = len(events), window
        empty = {"ev": np.zeros(0, np.int64), "bar": np.zeros(0, np.int64), "sipm": np.zeros(0, np.int64),
                 "pe": np.zeros(0), "time": np.zeros(0), "seq": np.zeros(0, np.int64)}
        if n == 0 or rate_scale <= 0 or self.rate_per_ns <= 0 or not len(self.count):
            return empty
        mu = self.rate_per_ns * rate_scale * (w1 - w0)
        lam = np.full(n, mu)
        if overdispersed and np.isfinite(self.gamma_k):
            k = self.gamma_k * (w1 - w0) / self.info.get("data_window_ns", w1 - w0)
            lam = rng.gamma(k, mu / k, n)
        ng = rng.poisson(lam)
        G = int(ng.sum())
        if G == 0:
            return empty
        gev = np.repeat(events, ng)
        pick = rng.integers(0, len(self.count), G)
        t0 = w0 + (w1 - w0) * rng.random(G)
        cnt = self.count[pick]
        gi = np.repeat(np.arange(G), cnt)
        row = np.repeat(self.start[pick], cnt) + (np.arange(len(gi)) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        return {"ev": gev[gi], "bar": self.bar[row], "sipm": self.sipm[row], "pe": self.pe[row],
                "time": t0[gi] + self.dt[row], "seq": self.seq[row]}

    def save(self, path):
        import json
        np.savez(path, rate_per_ns=self.rate_per_ns, gamma_k=self.gamma_k, start=self.start, count=self.count,
                 bar=self.bar, sipm=self.sipm, pe=self.pe, dt=self.dt, seq=self.seq, info=json.dumps(self.info))

    @classmethod
    def load(cls, path):
        import json
        z = np.load(path)
        return cls(rate_per_ns=float(z["rate_per_ns"]), gamma_k=float(z["gamma_k"]), start=z["start"], count=z["count"],
                   bar=z["bar"], sipm=z["sipm"], pe=z["pe"], dt=z["dt"], seq=z["seq"], info=json.loads(str(z["info"])))


def merge_pulses(ev, idx, bar, sipm, tm, pe, seq, B):
    """Signal pulses (sorted by event) plus background pulses B -> one table sorted by (event, bar, sipm, time) with
    idx recomputed per event. Returns the merged arrays and a flag per pulse (True = background)."""
    ev2 = np.r_[ev, B["ev"]]
    bar2, sipm2 = np.r_[bar, B["bar"]], np.r_[sipm, B["sipm"]]
    tm2, pe2, seq2 = np.r_[tm, B["time"]], np.r_[pe, B["pe"]], np.r_[seq, B["seq"]]
    isbg = np.r_[np.zeros(len(ev), bool), np.ones(len(B["ev"]), bool)]
    o = np.lexsort((tm2, sipm2, bar2, ev2))
    ev2, bar2, sipm2, tm2, pe2, seq2, isbg = ev2[o], bar2[o], sipm2[o], tm2[o], pe2[o], seq2[o], isbg[o]
    idx2 = np.arange(len(ev2)) - np.searchsorted(ev2, ev2)
    return ev2, idx2, bar2, sipm2, tm2, pe2, seq2, isbg
