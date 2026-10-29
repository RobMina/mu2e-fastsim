"""Isolated keyed-common-random SiPM diagnostic, not stock bit/queue equivalence.

Hash version 1 uses FNV-1a UTF-8 domain tags followed by SplitMix64 integer
mixing, modulo 2**64. It is deterministic, non-cryptographic, and independent
of NumPy RNG state. Normal draws use Box-Muller (libm precision may vary).
Cross-talk is breadth-first: duplicate hits at one pixel in a generation use
1-(1-p)**multiplicity, not stock's sequential queue. Roots are ordered by
(time, dark, root), with optical first on ties. Removing unrelated roots never
shifts variates, but shared pixel recovery and cross-talk can change outcomes.
Contract v2 keys pixels relative to the root landing pixel, sharing primary
avalanche and translated cascade variates across the two pixel policies.
Optical root IDs must be nonnegative and preserved across arms; thermal IDs
are negative (-1-ordinal). The dark flag also separates their random keys.
Legacy rounded pixels 0..40 are supported; fixed and thermal pixels are 0..39.
Only zero-trap, 40x40 configurations are supported. Limits raise, never truncate.
"""
from numbers import Integral

import numpy as np
import pandas as pd

from crvfastsim.response import sipm as SI

HASH_VERSION = "sipm-keyed-v1"
RANDOM_CONTRACT = "root-relative-pixel-coupling-v2"
MAX_ATTEMPTS = 10_000_000
_MASK = (1 << 64) - 1
_CANDIDATE_COLUMNS = ["event", "chan", "root", "time", "px", "py", "dark"]
_ATTEMPT_COLUMNS = _CANDIDATE_COLUMNS + [
    "generation", "primary", "multiplicity", "inactive", "v", "p", "uniform",
    "fired", "charge", "pe", "photon", "root_time", "root_px", "root_py",
]


def _mix(x):
    with np.errstate(over="ignore"):
        x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def keyed_uniform(seed, domain, *integer_arrays):
    """Broadcast integer keys to uniforms in [0,1), with 53 random mantissa bits."""
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    if not isinstance(domain, str):
        raise ValueError("domain must be a string")
    tag = 0xCBF29CE484222325
    for b in (HASH_VERSION + ":" + domain).encode("utf8"):
        tag = ((tag ^ b) * 0x100000001B3) & _MASK
    arrays = [np.asarray(a) for a in integer_arrays]
    if any(a.dtype.kind not in "iub" for a in arrays):
        raise ValueError("random keys must be integer arrays")
    value = _mix(np.uint64(int(seed) & _MASK) ^ np.uint64(tag))
    for array in np.broadcast_arrays(*arrays) if arrays else []:
        value = _mix(value ^ _mix(array.astype(np.uint64)))
    return (value >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)


def keyed_normal(seed, domain, *integer_arrays):
    """One standard normal per key, independent of keyed_uniform's domain."""
    u = 1.0 - keyed_uniform(seed, domain + ":normal-radius", *integer_arrays)
    angle = keyed_uniform(seed, domain + ":normal-angle", *integer_arrays)
    return np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * angle)


def _config(cfg):
    if cfg.n_pixels_x != 40 or cfg.n_pixels_y != 40:
        raise ValueError("diagnostic requires a 40x40 SiPM")
    for name in ("overvoltage", "time_constant", "capacitance", "p2", "trap0_lifetime", "trap1_lifetime"):
        value = getattr(cfg, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("p1", "cross_talk", "trap0_prob", "trap1_prob"):
        value = getattr(cfg, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid {name}")
    if cfg.trap0_prob or cfg.trap1_prob:
        raise ValueError("this diagnostic supports zero traps only")
    if not np.isfinite(cfg.thermal_rate) or cfg.thermal_rate < 0:
        raise ValueError("thermal_rate must be finite and nonnegative")
    full = cfg.p_fire(cfg.overvoltage)
    if full == 0 and (cfg.thermal_rate or cfg.cross_talk):
        raise ValueError("thermal/cross-talk production requires nonzero avalanche probability")
    q = (1.0 - (1.0 - cfg.cross_talk) ** 0.25) / full if full else 0.0
    if q > 1:
        raise ValueError("cross-talk production probability exceeds one")
    dead = np.zeros((41, 41), dtype=bool)
    for pair in cfg.inactive:
        if len(pair) != 2 or any(not isinstance(x, Integral) or not 0 <= x < 40 for x in pair):
            raise ValueError("invalid inactive pixel")
        dead[tuple(pair)] = True
    return q, dead.ravel()


def _frame(frame, integers, floats=()):
    out = frame.copy()
    for name in (*integers, *floats):
        if name not in out:
            raise ValueError(f"missing column {name}")
        a = out[name].to_numpy()
        if a.dtype.kind not in "iufb" or not np.isfinite(a).all():
            if not len(a):
                out[name] = np.array([], dtype=np.int64 if name in integers else float)
                continue
            raise ValueError(f"{name} must be finite numeric values")
        if name in integers:
            invalid_float = a.dtype.kind == "f" and (
                np.any(a != np.floor(a)) or np.any(a < -float(2**63)) or np.any(a >= float(2**63)))
            invalid_unsigned = a.dtype.kind == "u" and np.any(a >= np.uint64(2**63))
            if invalid_float or invalid_unsigned:
                raise ValueError(f"{name} must contain int64 identities")
            out[name] = a.astype(np.int64)
        else:
            out[name] = a.astype(float)
    for name in ("event", "chan"):
        if name in out and (out[name] < 0).any():
            raise ValueError(f"{name} must be nonnegative")
    return out


def optical_candidates(photons, pmap, cfg, seed, pixel_policy="fixed_integer"):
    """Map stable photon_id roots using identical bin/within-bin keys in both policies."""
    _config(cfg)
    p = _frame(photons, ("event", "chan", "photon_id"), ("time",))
    if (p.photon_id < 0).any() or p.duplicated(["event", "chan", "photon_id"]).any():
        raise ValueError("optical photon IDs must be nonnegative and unique per channel/event")
    weights = np.asarray(pmap, dtype=float)
    if weights.shape != (40, 40) or not np.isfinite(weights).all() or (weights < 0).any() or not weights.max() > 0:
        raise ValueError("photon map must be finite, nonnegative, nonzero 40x40 weights")
    if pixel_policy not in ("legacy_round", "fixed_integer"):
        raise ValueError("unknown pixel policy")
    cum = np.cumsum((weights / weights.max()).ravel())
    keys = [p[c].to_numpy() for c in ("event", "chan", "photon_id")]
    bin_id = np.searchsorted(cum, keyed_uniform(seed, "optical-bin", *keys) * cum[-1], side="right")
    px, py = bin_id // 40, bin_id % 40
    if pixel_policy == "legacy_round":
        px += keyed_uniform(seed, "optical-x", *keys) >= 0.5
        py += keyed_uniform(seed, "optical-y", *keys) >= 0.5
    return pd.DataFrame(dict(event=p.event.to_numpy(), chan=p.chan.to_numpy(), root=p.photon_id.to_numpy(),
                             time=p.time.to_numpy(), px=px, py=py, dark=False))


def thermal_candidates(channels, cfg, seed, start=100.0, end=1500.0):
    """Independent Poisson counts via keyed exponential waits; times/pixels use separate domains.

    All supplied event/channel pairs are simulated, including optically quiet ones.
    The wait loop is vectorized over channels, not a per-channel RNG stream.
    """
    _config(cfg)
    c = _frame(channels, ("event", "chan"))
    if c.duplicated(["event", "chan"]).any():
        raise ValueError("duplicate thermal channel")
    if not np.isfinite([start, end]).all() or end < start:
        raise ValueError("invalid thermal time window")
    keyed_uniform(seed, "validate")
    mean = cfg.thermal_rate / cfg.p_fire(cfg.overvoltage) * (end - start) if cfg.thermal_rate else 0.0
    if not np.isfinite(mean) or mean * len(c) > MAX_ATTEMPTS:
        raise RuntimeError("thermal candidate limit exceeded")
    event, chan = c.event.to_numpy(), c.chan.to_numpy()
    slots = np.arange(len(c)) if mean else np.zeros(0, dtype=int)
    elapsed = np.zeros(len(c))
    rows, ordinal, total = [], 0, 0
    while len(slots):
        elapsed[slots] += -np.log1p(-keyed_uniform(seed, "thermal-count", event[slots], chan[slots], ordinal))
        slots = slots[elapsed[slots] < mean]
        total += len(slots)
        if total > MAX_ATTEMPTS:
            raise RuntimeError("thermal candidate limit exceeded")
        if len(slots):
            rows.append((slots.copy(), np.full(len(slots), -1 - ordinal, dtype=np.int64)))
        ordinal += 1
    if not rows:
        return _empty(_CANDIDATE_COLUMNS)
    slots, root = (np.concatenate([r[i] for r in rows]) for i in (0, 1))
    keys = (event[slots], chan[slots], root)
    out = pd.DataFrame(dict(event=keys[0], chan=keys[1], root=root,
                           time=start + (end - start) * keyed_uniform(seed, "thermal-time", *keys),
                           px=(40 * keyed_uniform(seed, "thermal-x", *keys)).astype(int),
                           py=(40 * keyed_uniform(seed, "thermal-y", *keys)).astype(int), dark=True))
    return out.sort_values(["chan", "event", "time", "root"]).reset_index(drop=True)


def _empty(columns):
    floats = {"time", "v", "p", "uniform", "charge", "pe", "root_time"}
    bools = {"dark", "primary", "inactive", "fired"}
    return pd.DataFrame({c: pd.Series(dtype=float if c in floats else bool if c in bools else np.int64) for c in columns})


def simulate_candidates(candidates, cfg, seed, end=1500.0):
    """Return (charges, attempts), both sorted chan/event/time, retaining root ancestry.

    Candidates after end are discarded (end is inclusive); no lower time cut.
    Inactive pixels have ledger rows with p=0. Invalid/outside padded-grid pixels
    raise. Each root is unique within event/channel, regardless of dark flag.
    Attempt charge/pe are zero on failure; charges contains fired attempts only.
    """
    q_ct, dead = _config(cfg)
    keyed_uniform(seed, "validate")
    if not np.isfinite(end):
        raise ValueError("end must be finite")
    c = _frame(candidates, ("event", "chan", "root", "px", "py", "dark"), ("time",))
    if c.duplicated(["event", "chan", "root"]).any():
        raise ValueError("duplicate candidate root")
    if not c.dark.isin([0, 1]).all() or ((c[["px", "py"]] < 0) | (c[["px", "py"]] > 40)).any().any():
        raise ValueError("invalid dark flag or pixel outside 0..40")
    if ((c.root < 0) & ~c.dark.astype(bool)).any():
        raise ValueError("optical roots must be nonnegative")
    c = c[c.time <= end].sort_values(["event", "chan", "time", "dark", "root"]).reset_index(drop=True)
    if not len(c):
        return _empty(_ATTEMPT_COLUMNS), _empty(_ATTEMPT_COLUMNS)
    if len(c) > MAX_ATTEMPTS:
        raise RuntimeError("SiPM attempt limit exceeded")
    ec = c[["event", "chan"]].to_numpy()
    start = np.r_[0, np.flatnonzero(np.any(ec[1:] != ec[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[start, len(c)])
    last = np.full((len(start), 41 * 41), -np.inf)
    a = {name: c[name].to_numpy() for name in _CANDIDATE_COLUMNS}
    records, total = [], 0
    for rank in range(int(counts.max())):
        slot = np.flatnonzero(counts > rank)
        idx = start[slot] + rank
        pixel = a["px"][idx] * 41 + a["py"][idx]
        mult = np.ones(len(idx), dtype=np.int64)
        generation = 0
        while len(slot):
            total += len(slot)
            if total > MAX_ATTEMPTS or generation > 41 * 41:
                raise RuntimeError("SiPM attempt/cascade limit exceeded")
            t = a["time"][idx]
            v = cfg.overvoltage * (-np.expm1(-(t - last[slot, pixel]) / cfg.time_constant))
            p = 1.0 - (1.0 - cfg.p_fire(v)) ** mult
            p[dead[pixel]] = 0.0
            dx, dy = pixel // 41 - a["px"][idx], pixel % 41 - a["py"][idx]
            keys = (a["event"][idx], a["chan"][idx], a["root"][idx], a["dark"][idx], generation, dx, dy)
            uniform = keyed_uniform(seed, "avalanche", *keys)
            fired = uniform < p
            charge = np.where(fired, cfg.capacitance * v, 0.0)
            row = {name: a[name][idx] for name in _CANDIDATE_COLUMNS}
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
            slot, pixel, idx = slot[fired], pixel[fired], idx[fired]
            nx, ny, ok = SI._neighbours(pixel // 41, pixel % 41, cfg)
            dx, dy = pixel // 41 - a["px"][idx], pixel % 41 - a["py"][idx]
            ct_keys = (a["event"][idx, None], a["chan"][idx, None], a["root"][idx, None],
                       a["dark"][idx, None], generation, dx[:, None], dy[:, None], np.arange(4))
            ok &= keyed_uniform(seed, "cross-talk", *ct_keys) < q_ct
            source, direction = np.nonzero(ok)
            slot, idx = slot[source], idx[source]
            pixel = (nx * 41 + ny)[source, direction]
            if len(slot):
                _, first, mult = np.unique(slot * (41 * 41) + pixel, return_index=True, return_counts=True)
                slot, pixel, idx = slot[first], pixel[first], idx[first]
            generation += 1
    attempts = pd.concat(records, ignore_index=True)[_ATTEMPT_COLUMNS]
    attempts = attempts.sort_values(["chan", "event", "time", "dark", "root", "generation", "px", "py"]).reset_index(drop=True)
    return attempts[attempts.fired].reset_index(drop=True), attempts
