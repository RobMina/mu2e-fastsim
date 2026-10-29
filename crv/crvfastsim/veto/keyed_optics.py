"""Diagnostic subset/permutation/chunk-invariant optical draws; no production defaults."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from numbers import Integral
import re

import numpy as np
import pandas as pd
from scipy.stats import binom, gamma, poisson

from ..response import lookup, photons as P
from .replay_sipm_control import keyed_normal, keyed_uniform

CONTRACT = "optical-full-step-substep-type-v1"
IDS = ("run", "subrun", "event", "source_row")
KEYS = ("input_sha256", "draw_seed", *IDS, "chan", "substep", "photon_type", "ordinal")
MAX_MEAN = 1e12


class Draws:
    """Full input digest plus seed; each call supplies immutable integer identity columns."""

    def __init__(self, input_sha256, seed):
        if not isinstance(input_sha256, str) or re.fullmatch("[0-9a-f]{64}", input_sha256) is None:
            raise ValueError("input_sha256 must be a lowercase SHA256")
        if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a uint64 integer")
        self.input_sha256, self.seed = input_sha256, int(seed)
        self.words = tuple(np.uint64(int(input_sha256[i:i + 16], 16)) for i in range(0, 64, 16))

    def uniform(self, domain, *keys):
        return keyed_uniform(self.seed, CONTRACT + ":" + domain, *self.words, *keys)

    def normal(self, domain, *keys):
        return keyed_normal(self.seed, CONTRACT + ":" + domain, *self.words, *keys)

    def quantile(self, domain, *keys):
        return np.clip(self.uniform(domain, *keys), 2.0**-53, 1.0 - 2.0**-53)


def _positive_int(value, name):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or not 0 < value <= 2**52:
        raise ValueError(f"{name} must be a positive integer no larger than 2**52")


def count_draws(mean, nsub, draws, *keys):
    """Production Gaussian/Poisson then nearest-even per-substep count law."""
    mean, nsub = np.broadcast_arrays(np.asarray(mean, float), np.asarray(nsub))
    if (not np.isfinite(mean).all() or np.any(mean < 0) or np.any(mean > MAX_MEAN)
            or nsub.dtype.kind not in "iu" or np.any(nsub <= 0)):
        raise ValueError("invalid optical count mean or substep count")
    keys = np.broadcast_arrays(*keys, mean)[:-1]
    big = mean > 10.0
    value = np.zeros(mean.shape)
    if big.any():
        value[big] = mean[big] + np.sqrt(mean[big]) * draws.normal("count-normal", *(k[big] for k in keys))
    small = ~big & (mean > 0)
    if small.any():
        value[small] = poisson.ppf(draws.quantile("count-poisson", *(k[small].astype(np.uint64) for k in keys)), mean[small])
    if not np.isfinite(value).all():
        raise ValueError("nonfinite photon count quantile")
    return np.maximum(np.rint(value / nsub), 0).astype(np.int64)


def arrival_counts(n, probability, draws, *keys):
    """Vectorized inverse-binomial draws with explicit zero/one probability endpoints."""
    n, probability = np.broadcast_arrays(np.asarray(n), np.asarray(probability, float))
    if (n.dtype.kind not in "iu" or np.any(n < 0) or np.any(n > MAX_MEAN)
            or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1)):
        raise ValueError("invalid binomial inputs")
    keys = np.broadcast_arrays(*keys, probability)[:-1]
    out = np.zeros(n.shape, dtype=np.int64)
    out[probability == 1] = n[probability == 1]
    active = (n > 0) & (probability > 0) & (probability < 1)
    if active.any():
        value = binom.ppf(draws.quantile("arrival-binomial", *(k[active].astype(np.uint64) for k in keys)),
                          n[active], probability[active])
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > n[active]):
            raise ValueError("invalid arrival count quantile")
        out[active] = value.astype(np.int64)
    return out


class _ZeroCounts:
    """Zero placeholders reuse production geometry without consuming random state."""

    @staticmethod
    def normal(loc, scale):
        return np.zeros_like(loc)

    @staticmethod
    def poisson(lam):
        return np.zeros_like(lam)


def _empty():
    out = {name: pd.Series(dtype="int64") for name in KEYS if name not in ("input_sha256", "draw_seed")}
    out["input_sha256"] = pd.Series(dtype="str")
    out["draw_seed"] = pd.Series(dtype="uint64")
    out["time"] = pd.Series(dtype="float64")
    return pd.DataFrame(out)[[*KEYS, "time"]]


def _validate(steps, pdg, geom, cfg, tables, deviation):
    for name in (*IDS, "bar"):
        a = steps[name].to_numpy()
        if a.dtype.kind not in "iu" or np.any(a < 0) or (a.dtype.kind == "u" and np.any(a >= np.uint64(2**63))):
            raise ValueError(f"{name} must contain nonnegative int64 identities")
    if steps.source_row.duplicated().any():
        raise ValueError("duplicate original source row")
    if np.any(steps.bar.to_numpy() >= geom.nbars):
        raise ValueError("bar outside geometry")
    pdg = np.asarray(pdg)
    if pdg.shape != (len(steps),) or pdg.dtype.kind not in "iu" or np.any(pdg > 2**31 - 1) or np.any(pdg < -(2**31 - 1)):
        raise ValueError("requires one signed-int32-range PDG per step")
    apdg = np.abs(pdg.astype(np.int64))
    ion = apdg >= 1_000_000_000
    z, a = (apdg // 10000) % 1000, (apdg // 10) % 1000
    if np.any(ion & ((apdg // 10000000 != 100) | (z <= 0) | (a < z))):
        raise ValueError("invalid ion PDG encoding")
    t0, t1 = steps.t0.to_numpy(), steps.t1.to_numpy()
    if np.isinf(t0).any() or np.isinf(t1).any():
        raise ValueError("infinite optical endpoint")
    active = np.isfinite(t0) & np.isfinite(t1)
    if not P.particle_props(pdg[active])[2].all():
        raise ValueError("unknown active particle PDG")
    for name in ("edep", "len", "p1", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0"):
        a = steps[name].to_numpy()[active]
        if not np.isfinite(a).all() or (name in ("edep", "len", "p1") and np.any(a < 0)):
            raise ValueError(f"invalid active optical input {name}")
    deviation = np.asarray(deviation, float)
    if deviation.shape != (geom.nbars * 4,) or not np.isfinite(deviation).all():
        raise ValueError("invalid photon deviation coverage")
    for scale in (cfg.photon_yield_scale, cfg.photon_yield_variation_scale):
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("invalid optical yield scale")
    if np.any(1.0 + deviation * cfg.photon_yield_variation_scale < 0):
        raise ValueError("negative photon deviation factor")
    for mapping in (cfg.speed_ratio, cfg.atten_ratio):
        if any(not np.isfinite(value) or value <= 0 for value in mapping.values()):
            raise ValueError("propagation ratios must be finite and positive")
        P.bar_ratios(mapping, geom)
    for name in geom.sector_names:
        sector = cfg.sectors[name]
        if not np.isfinite(sector.yield_per_mev) or sector.yield_per_mev < 0:
            raise ValueError("invalid sector photon yield")
        L = tables[sector.table]
        if (not np.isfinite(L.fiber_decay_time) or L.fiber_decay_time < 0
                or abs(sector.reflector) != L.reflector):
            raise ValueError("invalid lookup decay/reflector")
        for cmap in (L.ck_scint, L.ck_fiber):
            if not np.isfinite(cmap).all() or np.any(cmap < 0) or np.any(np.diff(cmap[:, 0]) <= 0):
                raise ValueError("invalid Cherenkov map")
    for L in tables.values():
        for name in ("half_thickness", "half_width", "half_length", "fiber_separation", "hole_radius_x",
                     "hole_radius_y", "fiber_radius", "corner_radius"):
            if not np.isfinite(getattr(L, name)) or getattr(L, name) <= 0:
                raise ValueError("invalid lookup geometry constant")
        nb = {}
        for name in ("x", "y", "z", "beta", "theta", "phi", "r"):
            edge = np.asarray(L.edges[name])
            if edge.ndim != 1 or len(edge) < 2 or not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0):
                raise ValueError("invalid lookup edges")
            nb[name] = len(edge) - 1
        sizes = (nb["x"]*nb["y"]*nb["z"], nb["x"]*nb["y"]*nb["z"]*nb["beta"],
                 nb["beta"]*nb["theta"]*nb["phi"]*nb["r"]*nb["z"])
        for name, size in zip(lookup.TABLES, sizes):
            T = L.tables[name]
            if T.prob.shape != (size,) or not np.isfinite(T.prob).all() or np.any(T.prob < 0) or np.any(T.prob > 1):
                raise ValueError("invalid lookup probability/domain")
            for weight, total in ((T.delays, T.delay_sum), (T.emis, T.emis_sum)):
                if (weight.ndim != 2 or weight.shape[1] < 1 or len(weight) != size or weight.dtype != np.uint8
                        or np.asarray(total).shape != (size,) or np.asarray(total).dtype.kind not in "iu"
                        or not np.array_equal(weight.sum(axis=1, dtype=np.int64), total)
                        or np.any(total > np.iinfo(np.int32).max)):
                    raise ValueError("invalid lookup weight/sum")
    return pdg, deviation, active


def _content_tables(tables):
    out = {}
    for name, L in tables.items():
        h = hashlib.sha256()
        for a in (L.edges["x"], L.edges["y"], L.edges["z"], L.tables["scint"].prob,
                  L.tables["scint"].delays, L.tables["scint"].delay_sum):
            a = np.asarray(a)
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        out[name] = replace(L, name="keyed-optics-propagation-" + h.hexdigest())
    return out


def _rows(steps, pdg, geom, cfg, tables, deviation, draws):
    for L, rows in P.iter_rows(steps, pdg, geom, cfg, tables, deviation, _ZeroCounts()):
        idx = rows["step"]
        yields = np.array([cfg.sectors[name].yield_per_mev for name in geom.sector_names])[geom.sector[steps.bar.to_numpy()[idx]]]
        _, charge, _ = P.particle_props(pdg[idx])
        means = (yields * steps.edep.to_numpy()[idx],
                 P.cerenkov_per_mm(rows["beta"], charge, L.ck_scint) * steps.len.to_numpy()[idx],
                 P.cerenkov_per_mm(rows["beta"], charge, L.ck_fiber) * steps.len.to_numpy()[idx])
        keys = tuple(steps[name].to_numpy()[idx] for name in IDS) + (rows["chan"],)
        for kind, (name, mean) in enumerate(zip(("n_scint", "n_ck_s", "n_ck_f"), means)):
            rows[name] = count_draws(mean, rows["nsub"], draws, *keys, np.int64(kind))
        yield L, rows


def _bins(L, rows, ri, k):
    E = L.edges
    nb = {key: len(edge) - 1 for key, edge in E.items()}
    frac = (k + 0.5) / rows["nsub"][ri]
    p = rows["s"][ri] + rows["d"][ri] * frac[:, None]
    t = rows["t0"][ri] + (rows["t1"][ri] - rows["t0"][ri]) * frac
    x, y, z = p.T
    insc = P._inside_scint(L, x, y, z)
    infb, r, phi = P._inside_fiber(L, x, y, z, rows["d"][ri, 0], rows["d"][ri, 1])
    infb &= ~insc
    bx, by, bz = P.find_bin(E["x"], abs(x)), P.find_bin(E["y"], y), P.find_bin(E["z"], z)
    bb = P.find_bin(E["beta"], rows["beta"][ri])
    geo = (bx >= 0) & (by >= 0) & (bz >= 0)
    sbin = np.where(insc & geo, bz + by * nb["z"] + bx * nb["y"] * nb["z"], -1)
    cbin = np.where(insc & geo & (bb >= 0), bb + bz * nb["beta"] + by * nb["z"] * nb["beta"]
                   + bx * nb["y"] * nb["z"] * nb["beta"], -1)
    bth, bph, br = P.find_bin(E["theta"], rows["theta"][ri]), P.find_bin(E["phi"], phi), P.find_bin(E["r"], r)
    valid = infb & (bb >= 0) & (bth >= 0) & (bph >= 0) & (br >= 0) & (bz >= 0)
    fbin = np.where(valid, bz + br * nb["z"] + bph * nb["r"] * nb["z"] + bth * nb["phi"] * nb["r"] * nb["z"]
                   + bb * nb["theta"] * nb["phi"] * nb["r"] * nb["z"], -1)
    return (sbin, cbin, fbin), t, z


def _weighted_index(weights, totals, bins, uniform):
    cum = np.cumsum(weights[bins], axis=1, dtype=np.int64)
    return (cum < (uniform * totals[bins])[:, None]).sum(axis=1)


def _arrival_times(L, T, bins, base_time, draws, keys, batch):
    out = np.empty(len(bins))
    for a in range(0, len(bins), batch):
        sl = slice(a, a + batch)
        kk = tuple(key[sl] for key in keys)
        delay = _weighted_index(T.delays, T.delay_sum, bins[sl], draws.uniform("delay", *kk))
        tt = base_time[sl] + delay.astype(float)
        if L.fiber_decay_time > 0:
            ne = _weighted_index(T.emis, T.emis_sum, bins[sl], draws.uniform("emission", *kk))
            positive = ne > 0
            extra = np.zeros(len(ne))
            extra[positive] = gamma.ppf(draws.quantile("emission-gamma", *(key[positive] for key in kk)),
                                        ne[positive], scale=L.fiber_decay_time)
            tt += extra
        if not np.isfinite(tt).all():
            raise ValueError("nonfinite optical arrival time")
        out[sl] = tt
    return out


def make_photons(steps, pdg, geom, cfg, tables, deviation, *, input_sha256, seed,
                 max_substeps=200_000, max_photons=5_000_000, max_total_substeps=20_000_000):
    """Return full immutable keys and raw arrival times; caller owns the all-event denominator."""
    draws = Draws(input_sha256, seed)
    _positive_int(max_substeps, "max_substeps")
    _positive_int(max_photons, "max_photons")
    _positive_int(max_total_substeps, "max_total_substeps")
    pdg, deviation, active = _validate(steps, pdg, geom, cfg, tables, deviation)
    if not active.any():
        return _empty()
    steps = steps.loc[active].reset_index(drop=True).astype({name: np.int64 for name in (*IDS, "bar")})
    pdg = pdg[active].astype(np.int64)
    s, e, _, _, _ = P.step_inputs(steps, pdg, geom)
    sub = np.maximum(np.floor(np.linalg.norm(e - s, axis=1) / P.PRECISION_MM), 1)
    reflectors = np.array([cfg.sectors[name].reflector for name in geom.sector_names])[geom.sector[steps.bar.to_numpy()]]
    if not np.isfinite(sub).all() or np.any(sub > MAX_MEAN) or np.sum(sub * np.where(reflectors == 0, 4, 2)) > max_total_substeps:
        raise ValueError("total substep work limit exceeded")
    tables = _content_tables(tables)
    products, total = [], 0
    for L, rows in _rows(steps, pdg, geom, cfg, tables, deviation, draws):
        nsub = rows["nsub"]
        if np.any(nsub <= 0) or np.any(nsub > MAX_MEAN) or np.sum(nsub, dtype=np.float64) > max_total_substeps:
            raise ValueError("invalid or overflowing substep expansion")
        starts = np.r_[np.int64(0), np.cumsum(nsub)]
        for a in range(0, int(starts[-1]), int(max_substeps)):
            rank = np.arange(a, min(a + max_substeps, int(starts[-1])), dtype=np.int64)
            ri = np.searchsorted(starts[1:], rank, side="right")
            k = rank - starts[ri]
            bins, time, z = _bins(L, rows, ri, k)
            for kind, name in enumerate(("n_scint", "n_ck_s", "n_ck_f")):
                m = bins[kind] >= 0
                if not m.any():
                    continue
                rr, ss = ri[m], rows["step"][ri[m]]
                keys = tuple(steps[key].to_numpy()[ss] for key in IDS) + (rows["chan"][rr], k[m], np.full(m.sum(), kind, np.int64))
                T = L.tables[lookup.TABLES[kind]]
                probability = T.prob[bins[kind][m]].astype(float) * rows["dev"][rr]
                if np.any(rows["mu"]):
                    probability *= np.exp(-rows["mu"][rr] * z[m])
                if not np.isfinite(probability).all():
                    raise ValueError("nonfinite optical probability")
                narr = arrival_counts(rows[name][rr], np.minimum(probability, 1.0), draws, *keys)
                if np.sum(narr, dtype=np.float64) > max_photons - total:
                    raise ValueError("photon resource limit exceeded; no truncated result")
                count = int(np.sum(narr, dtype=np.int64))
                if not count:
                    continue
                sel = np.repeat(np.arange(len(narr)), narr)
                ordinal = np.arange(count) - np.repeat(np.cumsum(narr) - narr, narr)
                kk = tuple(key[sel] for key in keys) + (ordinal,)
                tt = _arrival_times(L, T, bins[kind][m][sel], time[m][sel], draws, kk, min(max_substeps, 8192))
                out = pd.DataFrame(dict(zip((*IDS, "chan", "substep", "photon_type", "ordinal"), kk)))
                out["input_sha256"], out["draw_seed"], out["time"] = input_sha256, np.uint64(seed), tt
                products.append(out[[*KEYS, "time"]])
                total += count
    return pd.concat(products, ignore_index=True) if products else _empty()
