"""Opt-in, staged zero-suppressed onspill response (no fused response, traps or background).

Supply the complete event population, including zero-step events, in EventTiming. All
internal event labels are row numbers in timing.keys, never chain.event_key. Keep that
mapping with staged outputs; run() returns it as event_keys. Times at digitisation are
local to the bunch; only reconstructed pulses receive the reconstructed bunch correction.
"""
from dataclasses import dataclass
import time as _time

import numpy as np

from . import chain as CH, readout, tables
from .response import photons as PH, sipm as SI


def _ids(keys):
    a = np.asarray(keys)
    if a.ndim != 2 or a.shape[1] != 3 or a.dtype.kind not in "iu" or np.any(a < 0):
        raise ValueError("event keys must be a nonnegative Nx3 integer array (run, subrun, event)")
    return np.array(a, dtype=np.uint64 if a.dtype.kind == "u" else np.int64, copy=True)


def _records(a):
    return np.ascontiguousarray(a).view(np.dtype([(k, a.dtype) for k in tables.KEY])).ravel()


@dataclass(frozen=True)
class EventTiming:
    """Full IDs and per-event times [ns]; S = -mc_pbtime, reco correction = -reco_pbtime."""
    keys: np.ndarray
    mc_pbtime: np.ndarray
    reco_pbtime: np.ndarray
    event_length: np.ndarray
    period: float = 1695.0

    def __post_init__(self):
        keys = _ids(self.keys)
        order = np.argsort(_records(keys), kind="stable")
        ordered = _records(keys[order])
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("event keys must be unique")
        keys.setflags(write=False)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "_order", order)
        object.__setattr__(self, "_ordered", ordered)
        for name in ("mc_pbtime", "reco_pbtime", "event_length"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.shape != (len(keys),) or not np.isfinite(a).all():
                raise ValueError(f"{name} must be a finite array of length N")
            if name == "event_length" and np.any(a <= 0):
                raise ValueError("event_length must be positive")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        p = np.asarray(self.period, dtype=float)
        if p.ndim or not np.isfinite(p) or p <= 0:
            raise ValueError("period must be finite and positive")
        object.__setattr__(self, "period", float(p))

    def lookup(self, keys):
        """Vectorised full-ID lookup, returning dense event rows in caller order."""
        a = _ids(keys)
        bounds = np.iinfo(self.keys.dtype)
        if a.size and (int(a.min()) < bounds.min or int(a.max()) > bounds.max):
            raise ValueError("step event keys absent from timing")
        rec = _records(a.astype(self.keys.dtype))
        loc = np.searchsorted(self._ordered, rec)
        if np.any(loc >= len(self.keys)):
            raise ValueError("step event keys absent from timing")
        if np.any(self._ordered[loc] != rec):
            raise ValueError("step event keys absent from timing")
        return self._order[loc]


def _validate(cfg, timing, geom, cond):
    from .response.contract import validate_window
    validate_window(cfg.window)
    lo, hi = cfg.window
    if lo < 100 or hi > timing.period or np.any(timing.event_length < hi):
        raise ValueError("onspill window needs a 100 ns prestart and must fit period and every event_length")
    if cfg.fused_response or cfg.background is not None:
        raise ValueError("onspill does not support fused_response or empirical background")
    s = cfg.sipm
    if s.trap0_prob != 0 or s.trap1_prob != 0:
        raise ValueError("onspill does not support trap afterpulses")
    for name in ("overvoltage", "time_constant", "capacitance", "p1", "p2"):
        if not np.isfinite(getattr(s, name)) or getattr(s, name) <= 0:
            raise ValueError(f"SiPM {name} must be finite and positive")
    if (not np.isfinite(s.thermal_rate) or s.thermal_rate < 0 or s.p1 > 1
            or not np.isfinite(s.cross_talk) or not 0 <= s.cross_talk <= 1):
        raise ValueError("invalid SiPM thermal_rate, p1 or cross_talk")
    for name in ("photon_yield_scale", "photon_yield_variation_scale"):
        if not np.isfinite(getattr(cfg.response, name)) or getattr(cfg.response, name) < 0:
            raise ValueError(f"invalid response {name}")
    if not np.isfinite(cfg.feb_phase_width) or not 0 <= cfg.feb_phase_width <= CH.PERIOD:
        raise ValueError("invalid FEB phase width")
    nchan = geom.nbars * 4
    if cond.readout_mask is None:
        raise ValueError("onspill charges require cond.readout_mask; use readout.apply with geometry and counters_cfg")
    status = np.asarray(readout.effective_status(cond))
    if (not isinstance(nchan, (int, np.integer)) or not 0 <= nchan <= CH.CHAN_MULT
            or status.ndim != 1 or status.dtype.kind not in "iu" or len(status) < nchan):
        raise ValueError("geometry/conditions channels must fit CHAN_MULT and integer status")
    return status[:nchan]


def _fold(t, period):
    t = np.fmod(t, period)
    return np.where(t < 0, t + period, t)


def charges(steps, pdg, geom, cond, cfg, rng, *, timing, key=None, tabs=None, pmap=None, timers=None):
    """Steps -> charges, sorted by (chan,time); photon is the ORIGINAL input row (-1 thermal).

    chan = dense_event * CH.CHAN_MULT + channel. All streams in one event share its
    SiPM history. Stock cfg.window=(200,1500) gives thermal start=100 and end=1500;
    photons before thermal start remain eligible. Caller chooses thermal_rate (stock 1e-4).
    cond.readout_mask must explicitly describe physical sensor presence, even for empty events.
    """
    status = _validate(cfg, timing, geom, cond)
    # DataFrame.to_numpy can promote mixed int64/uint64 columns to float64 and lose IDs.
    cols = [steps[k].to_numpy() for k in tables.KEY]
    bounds = np.iinfo(timing.keys.dtype)
    for col in cols:
        if col.dtype.kind not in "iu":
            raise ValueError("step event keys must have integer columns")
        if col.size and (int(col.min()) < bounds.min or int(col.max()) > bounds.max):
            raise ValueError("step event keys absent from timing")
    ev = timing.lookup(np.column_stack([col.astype(timing.keys.dtype) for col in cols]))
    pdg = np.asarray(pdg)
    if pdg.shape != (len(steps),) or pdg.dtype.kind not in "iu":
        raise ValueError("pdg must be an integer array aligned with every input step")
    ends = steps[["t0", "t1"]].to_numpy(dtype=float)
    if np.isinf(ends).any():
        raise ValueError("step endpoints must not be infinite")
    finite = np.isfinite(ends).all(axis=1)
    lo, hi = cfg.window
    folded = _fold(ends + timing.mc_pbtime[ev, None], timing.period)
    a, b = folded.T
    rows = np.flatnonzero(finite & ~(((a < lo - 100) & (b < lo - 100)) | ((a > hi) & (b > hi))
                                    | ((a > hi) & (b < lo - 100))))
    timers = timers if timers is not None else {}
    timers["nan_time_steps_skipped"] = timers.get("nan_time_steps_skipped", 0) + int((~finite).sum())
    timers["time_gate_steps_skipped"] = timers.get("time_gate_steps_skipped", 0) + int(finite.sum() - len(rows))
    start = _time.perf_counter()
    k2 = None
    if cfg.jax:
        import jax
        from .response import photons_jax, sipm_jax
        key = key if key is not None else jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
    if len(rows):
        # Never interpolate independently folded endpoints: optics sees the original interval.
        if cfg.jax:
            tabs = tabs if tabs is not None else photons_jax.load_device_tables(cfg.response)
            s, c, t = photons_jax.make_photons(steps.iloc[rows], pdg[rows], geom, cfg.response,
                                               tabs, cond.deviation, rng, k1)
        else:
            tabs = tabs if tabs is not None else PH.load_tables(cfg.response)
            s, c, t = PH.make_photons(steps.iloc[rows], pdg[rows], geom, cfg.response, tabs, cond.deviation, rng)
        s, c, t = rows[np.asarray(s, dtype=np.int64)], np.asarray(c, dtype=np.int64), np.asarray(t)
        if np.any(c < 0) or np.any(c >= len(status)):
            raise ValueError("photon channel outside geometry/conditions")
        keep = (status[c] & 0b101) == 0
        s, c, t = s[keep], c[keep], t[keep]
        t = _fold(t + timing.mc_pbtime[ev[s]], timing.period)
    else:
        s, c, t = np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    timers["photons_s"] = timers.get("photons_s", 0.0) + _time.perf_counter() - start
    labels = ev[s] * CH.CHAN_MULT + c
    live = np.flatnonzero((status & 0b101) == 0)
    thermal = (np.arange(len(timing.keys), dtype=np.int64)[:, None] * CH.CHAN_MULT + live).ravel()
    start = _time.perf_counter()
    if len(labels) or len(thermal):
        pmap = pmap if pmap is not None else SI.photon_map()
        args = (labels, t, s, pmap, cfg.sipm, rng)
        kw = dict(t_end=hi, t_start=lo - 100, thermal_chans=thermal)
        Q = sipm_jax.simulate(*args, k2, **kw) if cfg.jax else SI.simulate(*args, **kw)
    else:
        Q = {k: np.empty(0) for k in ("chan", "time", "charge", "pe", "photon")}
    Q = {k: np.asarray(v, dtype=np.int64 if k in ("chan", "photon") else float) for k, v in Q.items()}
    order = np.lexsort((Q["time"], Q["chan"]))
    Q = {k: v[order] for k, v in Q.items()}
    timers["sipm_s"] = timers.get("sipm_s", 0.0) + _time.perf_counter() - start
    timers["photons"] = timers.get("photons", 0) + len(s)
    timers["charges"] = timers.get("charges", 0) + len(Q["time"])
    return Q


def reco_timing(W, timing):
    """Return waveform rows carrying -reco_pbtime, independently of DB offset enable/scale."""
    ev = np.asarray(W["ev"])
    if ev.ndim != 1 or ev.dtype.kind not in "iu" or np.any(ev < 0) or np.any(ev >= len(timing.keys)):
        raise ValueError("waveform ev must contain dense timing row indices")
    return {**W, "event_t0": -timing.reco_pbtime[ev]}


def run(steps, pdg, geom, cond, cfg, rng, *, timing, counters_cfg, key=None, tabs=None, pmap=None,
        timers=None, detail=False):
    """Staged chain deriving a missing readout mask; event_keys maps dense outputs to original IDs."""
    timers = timers if timers is not None else {}
    if cond.readout_mask is None:
        cond = readout.apply(cond, geom, counters_cfg)
    Q = charges(steps, pdg, geom, cond, cfg, rng, timing=timing, key=key, tabs=tabs, pmap=pmap, timers=timers)
    D = CH.digis(Q, cond, cfg, rng, key=key, timers=timers, stock_scan=True)
    W = reco_timing(CH.reco_waveforms(D, cond), timing)
    clusters, timers = CH.clusters(W, cond, geom, counters_cfg, cfg, timers=timers, detail=detail, rng=rng)
    return dict(Q=Q, D=D, W=W, clusters=clusters, timers=timers, event_keys=timing.keys)
