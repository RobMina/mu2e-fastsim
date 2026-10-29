"""Per-call allocation admission for the frozen keyed SiPM diagnostic, not production."""
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from . import replay_sipm_control as S

CONTRACT = "diagnostic-sipm-bounded-v1"
_STATE_BYTES_PER_CHANNEL = 41 * 41 * 8
_neighbours = S.SI._neighbours


@dataclass(frozen=True)
class Limits:
    """Nonnegative Python-integer caps; generations counts primary generation zero."""

    photons: int = 300_000
    thermal_candidates: int = 300_000
    candidates: int = 1_000_000
    attempts: int = 1_000_000
    generations: int = 1682
    state_bytes: int = 300 * 1024**2
    live_channels: int = 4 * 5120

    def __post_init__(self):
        self.validate()

    def validate(self):
        maximum = int(np.iinfo(np.intp).max)
        for field in fields(self):
            value = getattr(self, field.name)
            upper = maximum if field.name == "state_bytes" else 10_000_000
            if field.name == "generations":
                upper = 1682
            if field.name == "live_channels":
                upper = min(upper, maximum // _STATE_BYTES_PER_CHANNEL)
            if type(value) is not int or not 0 <= value <= upper:
                raise ValueError(f"invalid SiPM resource limit: {field.name}")


def _limits(limits):
    if not isinstance(limits, Limits):
        raise ValueError("limits must be an immutable SiPM Limits instance")
    limits.validate()


def _check(name, size, limit):
    if size > limit:
        raise RuntimeError(f"SiPM {name} limit exceeded: {size} > {limit}; batch incomplete")


def _finite(name, *arrays):
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError(f"nonfinite SiPM {name}; batch incomplete")


def optical_candidates(photons, pmap, cfg, seed, pixel_policy="fixed_integer", *, limits=Limits()):
    """Check input photon/root budgets before delegating unchanged keyed pixel mapping."""
    _limits(limits)
    _check("photons", len(photons), limits.photons)
    _check("candidates", len(photons), limits.candidates)
    p = S._frame(photons, ("event", "chan", "photon_id"), ("time",))
    _finite("photon time", p.time.to_numpy())
    _check("live_channels", len(p[["event", "chan"]].drop_duplicates()), limits.live_channels)
    return S.optical_candidates(p, pmap, cfg, seed, pixel_policy)


def thermal_candidates(channels, cfg, seed, start=100.0, end=1500.0, *, limits=Limits()):
    """Independent Poisson counts via keyed exponential waits; times/pixels use separate domains.

    All supplied event/channel pairs are simulated, including optically quiet ones.
    The wait loop is vectorized over channels, not a per-channel RNG stream.
    """
    _limits(limits)
    _check("live_channels", len(channels), limits.live_channels)
    S._config(cfg)
    c = S._frame(channels, ("event", "chan"))
    if c.duplicated(["event", "chan"]).any():
        raise ValueError("duplicate thermal channel")
    if not np.isfinite([start, end]).all() or end < start or not np.isfinite(end - start):
        raise ValueError("invalid thermal time window")
    S.keyed_uniform(seed, "validate")
    mean = cfg.thermal_rate / cfg.p_fire(cfg.overvoltage) * (end - start) if cfg.thermal_rate else 0.0
    if not np.isfinite(mean):
        raise ValueError("nonfinite thermal mean")
    cap = min(limits.thermal_candidates, limits.candidates, limits.attempts)
    _check("thermal_candidates expected", mean * len(c), cap)
    event, chan = c.event.to_numpy(), c.chan.to_numpy()
    slots = np.arange(len(c)) if mean else np.zeros(0, dtype=int)
    elapsed = np.zeros(len(c))
    rows, ordinal, total = [], 0, 0
    while len(slots):
        elapsed[slots] += -np.log1p(-S.keyed_uniform(seed, "thermal-count", event[slots], chan[slots], ordinal))
        slots = slots[elapsed[slots] < mean]
        total += len(slots)
        _check("thermal_candidates actual", total, cap)
        if len(slots):
            rows.append((slots.copy(), np.full(len(slots), -1 - ordinal, dtype=np.int64)))
        ordinal += 1
    if not rows:
        return S._empty(S._CANDIDATE_COLUMNS)
    slots, root = (np.concatenate([r[i] for r in rows]) for i in (0, 1))
    keys = (event[slots], chan[slots], root)
    out = pd.DataFrame(dict(event=keys[0], chan=keys[1], root=root,
                           time=start + (end - start) * S.keyed_uniform(seed, "thermal-time", *keys),
                           px=(40 * S.keyed_uniform(seed, "thermal-x", *keys)).astype(int),
                           py=(40 * S.keyed_uniform(seed, "thermal-y", *keys)).astype(int), dark=True))
    _finite("thermal time", out.time.to_numpy())
    return out.sort_values(["chan", "event", "time", "root"]).reset_index(drop=True)


def simulate_candidates(candidates, cfg, seed, end=1500.0, *, limits=Limits()):
    """Return (charges, attempts), both sorted chan/event/time, retaining root ancestry.

    Candidates after end are discarded (end is inclusive); no lower time cut.
    Inactive pixels have ledger rows with p=0. Invalid/outside padded-grid pixels
    raise. Each root is unique within event/channel, regardless of dark flag.
    Attempt charge/pe are zero on failure; charges contains fired attempts only.
    """
    _limits(limits)
    _check("candidates", len(candidates), limits.candidates)
    q_ct, dead = S._config(cfg)
    S.keyed_uniform(seed, "validate")
    if not np.isfinite(end):
        raise ValueError("end must be finite")
    c = S._frame(candidates, ("event", "chan", "root", "px", "py", "dark"), ("time",))
    if c.duplicated(["event", "chan", "root"]).any():
        raise ValueError("duplicate candidate root")
    if not c.dark.isin([0, 1]).all() or ((c[["px", "py"]] < 0) | (c[["px", "py"]] > 40)).any().any():
        raise ValueError("invalid dark flag or pixel outside 0..40")
    if ((c.root < 0) & ~c.dark.astype(bool)).any():
        raise ValueError("optical roots must be nonnegative")
    _finite("candidate time", c.time.to_numpy())
    _check("photons", int((c.dark == 0).sum()), limits.photons)
    _check("thermal_candidates", int((c.dark == 1).sum()), limits.thermal_candidates)
    c = c[c.time <= end].sort_values(["event", "chan", "time", "dark", "root"]).reset_index(drop=True)
    if not len(c):
        return S._empty(S._ATTEMPT_COLUMNS), S._empty(S._ATTEMPT_COLUMNS)
    _check("attempts", len(c), limits.attempts)
    ec = c[["event", "chan"]].to_numpy()
    start = np.r_[0, np.flatnonzero(np.any(ec[1:] != ec[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[start, len(c)])
    _check("live_channels", len(start), limits.live_channels)
    _check("state_bytes", len(start) * _STATE_BYTES_PER_CHANNEL, limits.state_bytes)
    _check("generations", 1, limits.generations)
    last = np.full((len(start), 41 * 41), -np.inf)
    a = {name: c[name].to_numpy() for name in S._CANDIDATE_COLUMNS}
    records, total = [], 0
    for rank in range(int(counts.max())):
        slot = np.flatnonzero(counts > rank)
        idx = start[slot] + rank
        pixel = a["px"][idx] * 41 + a["py"][idx]
        mult = np.ones(len(idx), dtype=np.int64)
        generation = 0
        while len(slot):
            total += len(slot)
            _check("attempts ledger", total, limits.attempts)
            _check("generations", generation + 1, limits.generations)
            t = a["time"][idx]
            v = cfg.overvoltage * (-np.expm1(-(t - last[slot, pixel]) / cfg.time_constant))
            p = 1.0 - (1.0 - cfg.p_fire(v)) ** mult
            p[dead[pixel]] = 0.0
            dx, dy = pixel // 41 - a["px"][idx], pixel % 41 - a["py"][idx]
            keys = (a["event"][idx], a["chan"][idx], a["root"][idx], a["dark"][idx], generation, dx, dy)
            uniform = S.keyed_uniform(seed, "avalanche", *keys)
            fired = uniform < p
            charge = np.where(fired, cfg.capacitance * v, 0.0)
            _finite("avalanche arithmetic", v, p, uniform, charge)
            row = {name: a[name][idx] for name in S._CANDIDATE_COLUMNS}
            row.update(px=pixel // 41, py=pixel % 41, dark=a["dark"][idx].astype(bool),
                       generation=np.full(len(slot), generation), primary=np.full(len(slot), generation == 0),
                       multiplicity=mult, inactive=dead[pixel], v=v, p=p, uniform=uniform, fired=fired,
                       charge=charge, pe=np.where(fired, v / cfg.overvoltage, 0.0),
                       photon=np.where(a["dark"][idx], -1, a["root"][idx]),
                       root_time=t, root_px=a["px"][idx], root_py=a["py"][idx])
            records.append(pd.DataFrame(row))
            last[slot[fired], pixel[fired]] = t[fired]
            if not fired.any() or q_ct == 0:
                break
            # Bound all four directed neighbours before allocating expansion matrices.
            upper = 4 * int(np.count_nonzero(fired))
            _check("attempts neighbour upper", total + upper, limits.attempts)
            _check("generations neighbour upper", generation + 2, limits.generations)
            slot, pixel, idx = slot[fired], pixel[fired], idx[fired]
            nx, ny, ok = _neighbours(pixel // 41, pixel % 41, cfg)
            dx, dy = pixel // 41 - a["px"][idx], pixel % 41 - a["py"][idx]
            ct_keys = (a["event"][idx, None], a["chan"][idx, None], a["root"][idx, None],
                       a["dark"][idx, None], generation, dx[:, None], dy[:, None], np.arange(4))
            ok &= S.keyed_uniform(seed, "cross-talk", *ct_keys) < q_ct
            source, direction = np.nonzero(ok)
            slot, idx = slot[source], idx[source]
            pixel = (nx * 41 + ny)[source, direction]
            if len(slot):
                _, first, mult = np.unique(slot * (41 * 41) + pixel, return_index=True, return_counts=True)
                slot, pixel, idx = slot[first], pixel[first], idx[first]
            generation += 1
    attempts = pd.concat(records, ignore_index=True)[S._ATTEMPT_COLUMNS]
    attempts = attempts.sort_values(["chan", "event", "time", "dark", "root", "generation", "px", "py"]).reset_index(drop=True)
    return attempts[attempts.fired].reset_index(drop=True), attempts
