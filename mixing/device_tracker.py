"""Physical tracker digitisation with device-only compaction, ghosts and packing.

Reuse the existing ionisation, drift/charge and electronics JAX kernels unchanged.
Resource capacities are explicit; any overflow rejects the complete batch. No
NumPy fallback or approximation of an ionisation/electronics tail is permitted.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class Capacities:
    clusters_per_step: int
    drift_clusters: int
    straws: int
    clusters_per_end: int
    pairs_per_straw: int
    electronics_batch: int = 4

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"tracker {name} must be a positive integer")
        if self.straws % self.electronics_batch:
            raise ValueError("tracker straws must be a multiple of electronics_batch")


def prepare(c, capacities, batch_events):
    """Prepare immutable physical conditions/kernel closures once outside tracing.

    apply(collection, timing, assets, key) returns (masked digi columns, errors).
    timing contains mc_pbtime, event_length and event_mask, including empty events.
    Float64 is the supported tracker mode; float32 relative-frame handling is not
    silently omitted. The current conditions' no-noise/no-crosstalk scope is retained.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax, random
    from trkfastsim import devfloat as df
    from trkfastsim.digi import digi_jax as dj, digi_clusters_jax as dc

    from copy import deepcopy
    from trkfastsim.digi.digicond import Conditions
    c = Conditions(deepcopy(c.s), {name: np.array(value, copy=True) for name, value in c.a.items()})
    for value in c.a.values():
        value.setflags(write=False)
    cap = capacities if isinstance(capacities, Capacities) else Capacities(**capacities)
    if df.SINGLE or not jax.config.x64_enabled:
        raise ValueError("resident tracker currently requires float64 and JAX_ENABLE_X64=1")
    if type(batch_events) is not int or batch_events < 1:
        raise ValueError("batch_events must be positive")
    energies, probabilities = np.asarray(c.EIonize), np.asarray(c.intNProb)
    if (energies.ndim != 1 or not len(energies) or not np.isfinite(energies).all()
            or energies[0] <= 0 or np.any(np.diff(energies) <= 0)
            or probabilities.ndim != 1 or len(probabilities) < len(energies)
            or not np.isfinite(probabilities).all() or probabilities[0] <= 0
            or np.any(np.diff(probabilities) < 0)
            or not np.isfinite(c.meanFreePath) or c.meanFreePath <= 0
            or not np.isfinite(c.EAverage) or c.EAverage <= 0):
        raise ValueError("invalid tracker ionisation conditions")
    if not np.isfinite(c.mbtime) or c.mbtime <= 0:
        raise ValueError("tracker mbtime must be finite and positive")
    from .device_kernels import lift
    spec = jax.ShapeDtypeStruct
    charge_kernel = lift(dj._charge_kernel(c), spec((2,), np.uint32), spec((dj.CHARGE_BLOCK,), np.int64),
                         spec((dj.CHARGE_BLOCK, 3), np.float64), spec((dj.CHARGE_BLOCK,), np.int64),
                         spec((dj.CHARGE_BLOCK,), np.float64))
    batch = cap.electronics_batch
    wave = (2 * batch, cap.clusters_per_end)
    electronics_kernel = lift(dj._electronics_kernel(c, cap.clusters_per_end, cap.pairs_per_straw, batch),
        spec((2,), np.uint32), spec(wave, np.float64), spec(wave, np.float64), spec(wave, np.float64),
        spec(wave, np.int64), spec((2 * batch,), np.int32), spec((batch,), np.int64),
        spec((batch,), bool), spec((batch,), np.int64), spec((batch,), bool), spec((batch,), np.float64),
        event_end_tdc=spec((batch,), np.int64), window_end=spec((batch,), np.float64))
    assets = {"energies": energies, "probabilities": probabilities, "straw_dir": np.asarray(c.straw_dir),
              "all_planes": np.asarray(c.allPlanes, np.int64),
              "charge_kernel": charge_kernel.assets, "electronics_kernel": electronics_kernel.assets}
    ncharge = ((cap.drift_clusters + dj.CHARGE_BLOCK - 1) // dj.CHARGE_BLOCK) * dj.CHARGE_BLOCK
    sentinel = np.iinfo(np.int64).max

    def clusters(st, selected, assets, key):
        idx = jnp.arange(len(selected), dtype=jnp.int64)
        single = selected & ((st["shape"] == 3) | (st["len"] < c.meanFreePath))
        minion = st["ion"] == 0
        nc, single_ne = dc._counts(key, idx, st["eion"], st["len"], single, minion,
                                   c.meanFreePath, c.EAverage, energies[0], c.maxnclu)
        multi = selected & ~single & (nc > 0)
        overflow = jnp.any(multi & (nc > cap.clusters_per_step))
        bounded = jnp.where(multi, jnp.minimum(nc, cap.clusters_per_step), 0)
        p0 = jnp.stack([st[name] for name in ("x0", "y0", "z0")], axis=1)
        p1 = jnp.stack([st[name] for name in ("x1", "y1", "z1")], axis=1)
        direction = assets["straw_dir"][st["us"]] if c.randrad else jnp.zeros_like(p0)
        width = st["width"] if c.randrad else jnp.zeros_like(st["width"])
        pos, ne = dc._sample(key, idx, st["eion"], bounded, minion, p0, p1, direction, width,
                             c.EAverage, assets["energies"], assets["probabilities"],
                             cap.clusters_per_step, dc.MINION_UNROLL, dc.SEARCH_METHOD)
        mask = jnp.arange(cap.clusters_per_step) < bounded[:, None]
        repeats = jnp.concatenate([single.astype(jnp.int64),
                    jnp.where(mask, jnp.where(minion[:, None], ne, 1), 0).reshape(-1)])
        candidates = jnp.concatenate([idx, jnp.repeat(idx, cap.clusters_per_step)])
        positions = jnp.concatenate([p0, pos.reshape(-1, 3)])
        electrons = jnp.concatenate([single_ne, jnp.where(minion[:, None], 1, ne).reshape(-1)])
        ends = jnp.cumsum(repeats)
        row = jnp.arange(cap.drift_clusters, dtype=jnp.int64)
        owner = jnp.minimum(jnp.searchsorted(ends, row, side="right"), len(ends) - 1)
        valid = row < ends[-1]
        result = {"step": candidates[owner], "pos": positions[owner],
                  "ne": jnp.where(valid, electrons[owner], 1), "mask": valid}
        return result, {"clusters_per_step": overflow, "drift_clusters": ends[-1] > cap.drift_clusters}

    def drift(st, cluster, key, assets):
        pad = ncharge - cap.drift_clusters
        us = jnp.pad(st["us"][cluster["step"]], (0, pad))
        pos = jnp.pad(cluster["pos"], ((0, pad), (0, 0)))
        ne = jnp.pad(cluster["ne"], (0, pad), constant_values=1)
        times = jnp.pad(st["time"][cluster["step"]], (0, pad))
        arrays = (jnp.arange(ncharge // dj.CHARGE_BLOCK, dtype=jnp.int64) * dj.CHARGE_BLOCK,
                  us.reshape(-1, dj.CHARGE_BLOCK), pos.reshape(-1, dj.CHARGE_BLOCK, 3),
                  ne.reshape(-1, dj.CHARGE_BLOCK), times.reshape(-1, dj.CHARGE_BLOCK))
        def block(inputs):
            first, us_, pos_, ne_, time_ = inputs
            return charge_kernel.apply(assets["charge_kernel"],
                random.key_data(random.fold_in(random.fold_in(key, 1), first)), us_, pos_, ne_, time_)
        result = lax.map(block, arrays)
        gt, charge, wd = tuple(value.reshape((-1,) + value.shape[2:])[:cap.drift_clusters] for value in result)
        error = jnp.any(cluster["mask"] & (~jnp.isfinite(charge) | ~jnp.all(jnp.isfinite(gt) & jnp.isfinite(wd), axis=1)))
        return gt, charge, wd, error

    def pack(st, cluster, gt, charge, wd, timing, key, assets):
        cs, cmask = cluster["step"], cluster["mask"]
        ukey = st["ev"][cs] * 65536 + st["sid"][cs]
        sorted_key = jnp.sort(jnp.where(cmask, ukey, sentinel))
        first = jnp.concatenate([jnp.array([True]), sorted_key[1:] != sorted_key[:-1]]) & (sorted_key != sentinel)
        nstraw = first.sum()
        which = jnp.nonzero(first, size=cap.straws, fill_value=0)[0]
        live = jnp.arange(cap.straws) < nstraw
        strawkey = jnp.where(live, sorted_key[which], 0)
        straw_ev, straw_sid = strawkey // 65536, strawkey % 65536
        upanel = (straw_sid >> 10) * 6 + ((straw_sid >> 7) & 7)
        straw_us = upanel * 96 + (straw_sid & 127)
        readall = ((straw_sid & 127) >= c.allStraw) & jnp.isin(straw_sid >> 10, assets["all_planes"])
        base = jnp.arange(2 * cap.drift_clusters, dtype=jnp.int64)
        original = gt.reshape(-1).astype(jnp.float32)
        f64 = original.astype(jnp.float64)
        valid = jnp.repeat(cmask, 2)
        labels = jnp.repeat(ukey * 2, 2) + base % 2
        if c.onspill:
            buffer = (c.nADCSamples - c.nADCPresamples) * c.ADCPeriod
            ghost_src = jnp.tile(base, 3)
            ghost_time = jnp.concatenate([original, (f64 + c.mbtime).astype(jnp.float32),
                                          (f64 - c.mbtime).astype(jnp.float32)])
            valid = jnp.concatenate([valid, valid & (f64 < buffer), valid & (f64 > c.mbtime - buffer)])
            labels = jnp.tile(labels, 3)
        else:
            ghost_src, ghost_time = base, original
        labels = jnp.where(valid, labels, sentinel)
        order = jnp.lexsort((ghost_time, labels))
        sorted_label = labels[order]
        wanted = strawkey[:, None] * 2 + jnp.arange(2)
        lo = jnp.searchsorted(sorted_label, wanted, side="left")
        hi = jnp.searchsorted(sorted_label, wanted, side="right")
        counts = jnp.where(live[:, None], hi - lo, 0)
        gather = lo[:, :, None] + jnp.arange(cap.clusters_per_end)
        gi = order[jnp.minimum(gather, len(order) - 1)]
        index = ghost_src[gi] // 2
        ends = ghost_src[gi] % 2
        t, q, w, step = ghost_time[gi], charge[index], wd[index, ends], cs[index]
        valid_packed = live[:, None, None] & (jnp.arange(cap.clusters_per_end) < counts[:, :, None])
        t, q, w = tuple(jnp.where(valid_packed, a, 0).astype(jnp.float64) for a in (t, q, w))
        step = jnp.where(valid_packed, step, 0)
        nblocks = (batch_events + dj.JITTER_BATCH - 1) // dj.JITTER_BATCH
        jitter = lax.map(lambda block: dj._jitter_block(random.fold_in(random.fold_in(key, 2), block)),
                         jnp.arange(nblocks, dtype=jnp.int64))
        jitter = jitter.reshape(-1, 216)[straw_ev, upanel] * c.eventWindowMarkerROCJitter
        lengths = timing["event_length"][straw_ev]
        endtdc = jnp.clip(jnp.floor(lengths / c.TDCLSB), 0, c.maxTDC).astype(jnp.int64)
        packed = (t, q, w, step, jnp.minimum(counts, cap.clusters_per_end).astype(jnp.int32),
                  straw_us, readall, strawkey, live, jitter, endtdc, lengths)
        return packed, {"straws": nstraw > cap.straws,
                        "clusters_per_end": jnp.any(counts > cap.clusters_per_end)}

    def electronics(packed, key, assets):
        batch = cap.electronics_batch
        groups = tuple(value.reshape((-1, batch) + value.shape[1:]) for value in packed)
        def block(inputs):
            t, q, w, step, count, us, ra, skey, live, jitter, endtdc, lengths = inputs
            result = electronics_kernel.apply(assets["electronics_kernel"], random.key_data(random.fold_in(key, 3)),
                *tuple(value.reshape(2 * batch, cap.clusters_per_end) for value in (t, q, w, step)),
                count.reshape(-1), us, ra, skey, live, jitter, event_end_tdc=endtdc, window_end=lengths)
            return result
        result = lax.map(block, groups)
        output = {name: result[name].reshape((-1,) + result[name].shape[3:])
                  for name in ("adc", "tdc", "tot", "pmp", "step", "t_analog")}
        output["mask"] = result["keep"].reshape(-1)
        output["ev"] = jnp.repeat(packed[7] // 65536, cap.pairs_per_straw)
        output["sid"] = jnp.repeat(packed[7] % 65536, cap.pairs_per_straw)
        order = jnp.lexsort((output["t_analog"][:, 0], output["sid"],
                            jnp.where(output["mask"], output["ev"], sentinel)))
        output = {name: value[order] for name, value in output.items()}
        errors = {"electronics_search_or_pairs": jnp.any(result["failed"]),
                  "saturation_capacity": jnp.any(result["sat_overflow"])}
        return output, errors

    def apply(collection, timing, assets, key):
        st = dict(collection["columns"])
        required = ("t", "eion", "len", "width", "x0", "y0", "z0", "x1", "y1", "z1")
        mask = collection["mask"]
        identifiers = ("plane", "panel", "straw") if {"plane", "panel", "straw"} <= st.keys() else ("sid",)
        if any(st[name].dtype.kind not in "iu" for name in identifiers) or collection["event"].dtype.kind not in "iu":
            raise ValueError("tracker detector/event identifiers must have integer dtypes")
        invalid = jnp.any(jnp.stack([~jnp.isfinite(st[name]) for name in (*required, "shape", "ion")]), axis=0)
        if len(identifiers) == 3:
            for name, limit in (("plane", 36), ("panel", 6), ("straw", 96)):
                invalid |= (st[name] < 0) | (st[name] >= limit)
            sid = (st["plane"].astype(jnp.int64) << 10) | (st["panel"].astype(jnp.int64) << 7) | st["straw"].astype(jnp.int64)
        else:
            sid = st["sid"].astype(jnp.int64)
        ev = collection["event"].astype(jnp.int64)
        invalid |= (st["eion"] < 0) | (st["len"] < 0) | (st["eion"] / energies[0] >= 2**52)
        invalid |= (st["eion"] / c.EAverage >= 2**52) | (st["len"] / c.meanFreePath >= 2**52)
        invalid |= (sid < 0) | (sid >= 36 * 1024) | (((sid >> 7) & 7) >= 6) | ((sid & 127) >= 96)
        invalid |= (ev < 0) | (ev >= batch_events)
        input_error = jnp.any(mask & invalid)
        safe = mask & ~invalid
        for name in required:
            st[name] = jnp.where(safe, st[name], 0.).astype(jnp.float64)
        st["sid"], st["ev"] = jnp.where(safe, sid, 0), jnp.where(safe, ev, 0)
        st["us"] = ((st["sid"] >> 10) * 6 + ((st["sid"] >> 7) & 7)) * 96 + (st["sid"] & 127)
        time = st["t"] + timing["mc_pbtime"][st["ev"]]
        invalid_timing = (~jnp.isfinite(timing["mc_pbtime"]) | ~jnp.isfinite(timing["event_length"])
                          | (timing["event_length"] <= 0)
                          | (timing["event_length"] < c.digitizationEnd - c.timeFromProtonsToDRMarker))
        timing_error = jnp.any(invalid_timing & timing["event_mask"]) | jnp.any(safe & ~jnp.isfinite(time))
        safe &= timing["event_mask"][st["ev"]] & ~invalid_timing[st["ev"]] & jnp.isfinite(time)
        time = jnp.where(safe, time, 0.)
        if c.onspill:
            time = jnp.fmod(time, c.mbtime)
            time = jnp.where(time < 0, time + c.mbtime, time)
        st["time"] = time
        start = c.digitizationStart - c.timeFromProtonsToDRMarker
        end = c.digitizationEnd - c.timeFromProtonsToDRMarker if c.onspill else timing["event_length"][st["ev"]]
        inwin = ((time > start - c.electronicsTimeDelay - c.steptimebuf)
                 & (time < jnp.maximum(c.mbtime, end) - c.electronicsTimeDelay + c.steptimebuf))
        readall = ((st["sid"] & 127) >= c.allStraw) & jnp.isin(st["sid"] >> 10, assets["all_planes"])
        selected = safe & (st["eion"] > c.minstepE) & (inwin | readall)
        cluster, cluster_errors = clusters(st, selected, assets, random.fold_in(key, 0))
        gt, charge, wd, charge_error = drift(st, cluster, key, assets)
        packed, packing_errors = pack(st, cluster, gt, charge, wd, timing, key, assets)
        output, electronic_errors = electronics(packed, key, assets)
        errors = {"input": input_error, "timing": timing_error, "nonfinite_charge": charge_error,
                  **cluster_errors, **packing_errors, **electronic_errors}
        return output, errors

    return SimpleNamespace(assets=assets, apply=apply, clusters=clusters, drift=drift, pack=pack,
                           electronics=electronics, capacities=cap,
                           metadata={"physics": "existing tracker JAX ionisation/drift/electronics kernels",
                                     "precision": "float64", "capacity_policy": "reject; no fallback",
                                     "reconstruction": "not implemented"})
