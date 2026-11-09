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
    long_steps: int = None
    clusters_per_long_step: int = None
    end_tiers: tuple = None
    charge_blocks_per_launch: int = None
    flat_clusters: int = None
    minion_steps: int = None
    minion_unroll: int = None
    minion_search: str = None
    digit_entries_per_straw: float = None

    def __post_init__(self):
        optional = ("long_steps", "clusters_per_long_step", "end_tiers", "charge_blocks_per_launch", "flat_clusters",
                    "minion_steps", "minion_unroll", "minion_search")
        for name, value in vars(self).items():
            if name == "end_tiers" or (name in optional and value is None):
                continue
            if name == "minion_search":
                if value not in ("scan", "compare_all"):
                    raise ValueError("tracker minion_search must be 'scan' or 'compare_all'")
                continue
            if name == "digit_entries_per_straw":
                if value is not None and (type(value) not in (int, float) or not 0 < value <= self.pairs_per_straw):
                    raise ValueError("tracker digit_entries_per_straw must be in (0, pairs_per_straw]")
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"tracker {name} must be a positive integer")
        if self.straws % self.electronics_batch:
            raise ValueError("tracker straws must be a multiple of electronics_batch")
        if (self.long_steps is None) != (self.clusters_per_long_step is None):
            raise ValueError("tracker long_steps and clusters_per_long_step are set together")
        if self.long_steps is not None and self.clusters_per_long_step <= self.clusters_per_step:
            raise ValueError("tracker clusters_per_long_step must exceed clusters_per_step")
        if (self.flat_clusters is None) != (self.minion_steps is None):
            raise ValueError("tracker flat_clusters and minion_steps are set together")
        if self.flat_clusters is not None and self.long_steps is None:
            raise ValueError("tracker flat_clusters needs the long-step tier (long_steps)")
        if self.end_tiers is not None:
            tiers = tuple(tuple(tier) for tier in self.end_tiers)
            object.__setattr__(self, "end_tiers", tiers)
            widths = [self.clusters_per_end] + [tier[1] if len(tier) == 2 else 0 for tier in tiers]
            if not tiers or any(len(tier) != 2 or any(type(v) is not int or v < 1 for v in tier) for tier in tiers):
                raise ValueError("tracker end_tiers must be (straws, clusters_per_end) positive integer pairs")
            if any(b <= a for a, b in zip(widths, widths[1:])):
                raise ValueError("tracker end_tiers widths must increase and exceed clusters_per_end")
            if any(n % min(self.electronics_batch, n) for n, _ in tiers):
                raise ValueError("tracker end_tiers straws must be multiples of electronics_batch")

    def tier_batch(self, straws):
        return min(self.electronics_batch, straws)

    def digit_entries(self, batch):
        """Flat digitisation entries per electronics launch of `batch` straws (None: the pair-by-pair loop)."""
        if self.digit_entries_per_straw is None:
            return None
        return int(np.ceil(self.digit_entries_per_straw * batch))


def _entries(entries):
    return {} if entries is None else {"entries": entries}


def prepare(c, capacities, batch_events):
    """Prepare immutable physical conditions/kernel closures once outside tracing.

    apply(collection, timing, assets, key) returns (masked digi columns, errors).
    timing contains mc_pbtime, event_length and event_mask, including empty events.
    The tracker device float follows trkfastsim.devfloat: float64, or float32 with the staged
    digitiser's relative frames (positions to the straw middle, times to whole COMMON_PERIODs,
    per-straw n_ref in the electronics). The no-noise/no-crosstalk scope is retained.
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
    if not jax.config.x64_enabled:
        raise ValueError("resident tracker requires JAX_ENABLE_X64=1")
    rel = df.SINGLE   # float32: the staged digitiser's relative position and time frames
    if rel:
        df.check_common_period(c)
    period = df.COMMON_PERIOD
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
    # minimum-ionising loop knobs; both leave the outputs unchanged (digi_clusters_jax)
    unroll = cap.minion_unroll or dc.MINION_UNROLL
    method = cap.minion_search or dc.SEARCH_METHOD
    spec = jax.ShapeDtypeStruct
    charge_kernel = lift(dj._charge_kernel(c), spec((2,), np.uint32), spec((dj.CHARGE_BLOCK,), np.int64),
                         spec((dj.CHARGE_BLOCK, 3), np.float64), spec((dj.CHARGE_BLOCK,), np.int64),
                         spec((dj.CHARGE_BLOCK,), np.float64))
    batch = cap.electronics_batch
    wave = (2 * batch, cap.clusters_per_end)
    electronics_kernel = lift(dj._electronics_kernel(c, cap.clusters_per_end, cap.pairs_per_straw, batch,
                                                     **_entries(cap.digit_entries(batch))),
        spec((2,), np.uint32), spec(wave, np.float64), spec(wave, np.float64), spec(wave, np.float64),
        spec(wave, np.int64), spec((2 * batch,), np.int32), spec((batch,), np.int64),
        spec((batch,), bool), spec((batch,), np.int64), spec((batch,), bool), spec((batch,), np.float64),
        event_end_tdc=spec((batch,), np.int64), window_end=spec((batch,), np.float64),
        **({"n_ref": spec((batch,), np.int64)} if rel else {}))
    assets = {"energies": energies, "probabilities": probabilities, "straw_dir": np.asarray(c.straw_dir),
              "all_planes": np.asarray(c.allPlanes, np.int64),
              "charge_kernel": charge_kernel.assets, "electronics_kernel": electronics_kernel.assets}
    tier_kernels = []
    for straws, width in cap.end_tiers or ():
        lb = cap.tier_batch(straws)
        lw = (2 * lb, width)
        tier_kernels.append(lift(dj._electronics_kernel(c, width, cap.pairs_per_straw, lb,
                                                        **_entries(cap.digit_entries(lb))),
            spec((2,), np.uint32), spec(lw, np.float64), spec(lw, np.float64), spec(lw, np.float64),
            spec(lw, np.int64), spec((2 * lb,), np.int32), spec((lb,), np.int64),
            spec((lb,), bool), spec((lb,), np.int64), spec((lb,), bool), spec((lb,), np.float64),
            event_end_tdc=spec((lb,), np.int64), window_end=spec((lb,), np.float64),
            **({"n_ref": spec((lb,), np.int64)} if rel else {})))
    if tier_kernels:
        assets["electronics_tier_kernels"] = tuple(kernel.assets for kernel in tier_kernels)
    if rel:
        assets["straw_mid"] = np.asarray(c.straw_mid, np.float64)

    def sample_frame(st, idx, p0, p1, assets):
        """float32: step points relative to their straw middle, as digi_clusters_jax.make_clusters."""
        if not rel:
            return p0, p1, None
        ref = assets["straw_mid"][st["us"][idx]]
        return p0 - ref, p1 - ref, ref

    def absolute(pos, ref):
        return pos if ref is None else pos.astype(jnp.float64) + ref[:, None, :]

    def electronics_call(kernel, kernel_assets, key, t, q, w, step, count, width, batch, us, ra, skey, live,
                         jitter, endtdc, lengths):
        """One electronics block; float32 times relative to n_ref whole COMMON_PERIODs per straw, as digi_jax."""
        extra = {}
        if rel:
            valid = jnp.arange(width)[None, None, :] < count[:, :, None]
            first = jnp.min(jnp.where(valid, t, jnp.inf), axis=(1, 2))
            nref = jnp.where(jnp.isfinite(first), jnp.floor(first / period), 0).astype(jnp.int64)
            t = t - (nref * period)[:, None, None]
            extra = {"n_ref": nref}
        result = kernel.apply(kernel_assets, random.key_data(random.fold_in(key, 3)),
            *tuple(v.reshape(2 * batch, width) for v in (t, q, w, step)),
            count.reshape(-1), us, ra, skey, live, jitter, event_end_tdc=endtdc, window_end=lengths, **extra)
        if rel:
            result = {**result, "t_analog": result["t_analog"].astype(jnp.float64) + (nref * period)[:, None, None]}
        return result
    launch = dj.CHARGE_BLOCK * (cap.charge_blocks_per_launch or 1)
    ncharge = ((cap.drift_clusters + launch - 1) // launch) * launch
    sentinel = np.iinfo(np.int64).max

    def sample(st, idx, bounded, minion, assets, key, capacity):
        p0 = jnp.stack([st[name][idx] for name in ("x0", "y0", "z0")], axis=1)
        p1 = jnp.stack([st[name][idx] for name in ("x1", "y1", "z1")], axis=1)
        direction = assets["straw_dir"][st["us"][idx]] if c.randrad else jnp.zeros_like(p0)
        width = st["width"][idx] if c.randrad else jnp.zeros_like(st["width"][idx])
        p0, p1, ref = sample_frame(st, idx, p0, p1, assets)
        pos, ne = dc._sample(key, idx, st["eion"][idx], bounded, minion, p0, p1, direction, width,
                             c.EAverage, assets["energies"], assets["probabilities"],
                             capacity, unroll, method)
        return absolute(pos, ref), ne

    def clusters_tiered(st, selected, assets, key):
        """Two capacities with the same draws: per-slot keys do not depend on the padded width.

        Steps with at most clusters_per_step clusters use the dense [steps, clusters_per_step]
        path; up to long_steps longer steps are gathered into [long_steps, clusters_per_long_step].
        Clusters keep the single-capacity order (singles, then each step's clusters in step order).
        """
        n = len(selected)
        idx = jnp.arange(n, dtype=jnp.int64)
        single = selected & ((st["shape"] == 3) | (st["len"] < c.meanFreePath))
        minion = st["ion"] == 0
        nc, single_ne = dc._counts(key, idx, st["eion"], st["len"], single, minion,
                                   c.meanFreePath, c.EAverage, energies[0], c.maxnclu)
        multi = selected & ~single & (nc > 0)
        long = multi & (nc > cap.clusters_per_step)
        short_n = jnp.where(multi & ~long, nc, 0)
        pos_s, ne_s = sample(st, idx, short_n, minion, assets, key, cap.clusters_per_step)
        lidx = jnp.nonzero(long, size=cap.long_steps, fill_value=0)[0]
        llive = jnp.arange(cap.long_steps) < long.sum()
        long_n = jnp.where(llive, jnp.minimum(nc[lidx], cap.clusters_per_long_step), 0)
        pos_l, ne_l = sample(st, lidx, long_n, minion[lidx], assets, key, cap.clusters_per_long_step)
        errors = {"long_steps": long.sum() > cap.long_steps,
                  "clusters_per_step": jnp.any(llive & (nc[lidx] > cap.clusters_per_long_step))}

        def expand(n_, ne, mine):
            slot = jnp.arange(ne.shape[1]) < n_[:, None]
            return jnp.where(slot, jnp.where(mine[:, None], ne, 1), 0), jnp.where(mine[:, None], 1, ne)

        rep_s, el_s = expand(short_n, ne_s, minion)
        rep_l, el_l = expand(long_n, ne_l, minion[lidx])
        per_step = rep_s.sum(axis=1).at[jnp.where(llive, lidx, n)].add(rep_l.sum(axis=1), mode="drop")
        singles = single.astype(jnp.int64)
        base = singles.sum() + jnp.cumsum(per_step) - per_step
        start_s = base[:, None] + jnp.cumsum(rep_s, axis=1) - rep_s
        start_l = base[lidx][:, None] + jnp.cumsum(rep_l, axis=1) - rep_l
        start_1 = jnp.cumsum(singles) - singles
        total = singles.sum() + per_step.sum()
        # Candidate tables in one id space: singles, then short slots, then long slots.
        cand_step = jnp.concatenate([idx, jnp.repeat(idx, cap.clusters_per_step), jnp.repeat(lidx, cap.clusters_per_long_step)])
        cand_pos = jnp.concatenate([jnp.stack([st[name] for name in ("x0", "y0", "z0")], axis=1),
                                    pos_s.reshape(-1, 3), pos_l.reshape(-1, 3)])
        cand_el = jnp.concatenate([single_ne, el_s.reshape(-1), el_l.reshape(-1)])
        cand_rep = jnp.concatenate([singles, rep_s.reshape(-1), rep_l.reshape(-1)])
        cand_start = jnp.concatenate([start_1, start_s.reshape(-1), start_l.reshape(-1)])
        row = jnp.arange(cap.drift_clusters, dtype=jnp.int64)
        live = cand_rep > 0
        mark = jnp.full(cap.drift_clusters, -1, jnp.int64).at[jnp.where(live, cand_start, cap.drift_clusters)].set(
            jnp.arange(cand_rep.size, dtype=jnp.int64), mode="drop")
        anchor = lax.cummax(jnp.where(mark >= 0, row, 0), axis=0)
        owner = jnp.maximum(mark[anchor], 0)
        valid = row < total
        result = {"step": cand_step[owner], "pos": cand_pos[owner],
                  "ne": jnp.where(valid, cand_el[owner], 1), "mask": valid,
                  "usage": {"drift_clusters": total, "long_steps": long.sum(),
                            "clusters_per_step": jnp.max(jnp.where(multi & ~long, nc, 0)),
                            "clusters_per_long_step": jnp.max(jnp.where(long, nc, 0))}}
        return result, {**errors, "drift_clusters": total > cap.drift_clusters}

    def clusters_flat(st, selected, assets, key):
        """clusters_tiered with the short steps drawn as a flat list of live (step, slot) pairs, at most flat_clusters,
        and the minimum-ionising loops run on the short minimum-ionising steps only, at most minion_steps.

        Same per-slot and per-step keys, the same arithmetic and the same cluster order, so the same clusters; the
        dense path draws every row's clusters_per_step slots. Either overflow rejects the batch.
        """
        n = len(selected)
        idx = jnp.arange(n, dtype=jnp.int64)
        single = selected & ((st["shape"] == 3) | (st["len"] < c.meanFreePath))
        minion = st["ion"] == 0
        nc, single_ne = dc._counts(key, idx, st["eion"], st["len"], single, minion,
                                   c.meanFreePath, c.EAverage, energies[0], c.maxnclu)
        multi = selected & ~single & (nc > 0)
        long = multi & (nc > cap.clusters_per_step)
        short_n = jnp.where(multi & ~long, nc, 0)
        lidx = jnp.nonzero(long, size=cap.long_steps, fill_value=0)[0]
        llive = jnp.arange(cap.long_steps) < long.sum()
        long_n = jnp.where(llive, jnp.minimum(nc[lidx], cap.clusters_per_long_step), 0)
        pos_l, ne_l = sample(st, lidx, long_n, minion[lidx], assets, key, cap.clusters_per_long_step)
        # flat short slots, in step then slot order
        ends = jnp.cumsum(short_n)
        nshort = ends[-1]
        r = jnp.arange(cap.flat_clusters, dtype=jnp.int64)
        fstep = jnp.minimum(jnp.searchsorted(ends, r, side="right"), n - 1)
        first = ends[fstep] - short_n[fstep]
        fslot = r - first
        flive = r < nshort
        p0 = jnp.stack([st[name] for name in ("x0", "y0", "z0")], axis=1)
        p1 = jnp.stack([st[name] for name in ("x1", "y1", "z1")], axis=1)
        direction = assets["straw_dir"][st["us"]] if c.randrad else jnp.zeros_like(p0)
        width = st["width"] if c.randrad else jnp.zeros_like(st["width"])
        q0, q1, ref = sample_frame(st, idx, p0, p1, assets)
        pos_f, ne_f = dc._sample_flat(key, fstep, fslot, st["eion"], short_n, minion, q0, q1, direction, width,
                                      c.EAverage)
        pos_f = pos_f if ref is None else pos_f.astype(jnp.float64) + ref[fstep]
        # minimum-ionising electron counts: the per-step loops on the short minimum-ionising steps only
        mmask = multi & ~long & minion
        midx = jnp.nonzero(mmask, size=cap.minion_steps, fill_value=0)[0]
        mlive = jnp.arange(cap.minion_steps) < mmask.sum()
        mine, _ = dc._minion_clusters(dc._step_keys(key, midx), assets["energies"], assets["probabilities"],
                                      st["eion"][midx], jnp.where(mlive, short_n[midx], 0), cap.clusters_per_step,
                                      unroll, method)
        mrow = jnp.zeros(n, jnp.int64).at[jnp.where(mlive, midx, n)].set(jnp.arange(cap.minion_steps), mode="drop")
        fmin = minion[fstep]
        ne_f = jnp.where(fmin, mine[mrow[fstep], jnp.minimum(fslot, cap.clusters_per_step - 1)], ne_f)
        errors = {"long_steps": long.sum() > cap.long_steps,
                  "clusters_per_step": jnp.any(llive & (nc[lidx] > cap.clusters_per_long_step)),
                  "flat_clusters": nshort > cap.flat_clusters, "minion_steps": mmask.sum() > cap.minion_steps}

        def expand(n_, ne, mine):
            slot = jnp.arange(ne.shape[1]) < n_[:, None]
            return jnp.where(slot, jnp.where(mine[:, None], ne, 1), 0), jnp.where(mine[:, None], 1, ne)

        rep_f = jnp.where(flive, jnp.where(fmin, ne_f, 1), 0)
        el_f = jnp.where(fmin, 1, ne_f)
        rep_l, el_l = expand(long_n, ne_l, minion[lidx])
        per_step = (jnp.zeros(n, rep_f.dtype).at[jnp.where(flive, fstep, n)].add(rep_f, mode="drop")
                    .at[jnp.where(llive, lidx, n)].add(rep_l.sum(axis=1), mode="drop"))
        singles = single.astype(jnp.int64)
        base = singles.sum() + jnp.cumsum(per_step) - per_step
        within = jnp.cumsum(rep_f) - rep_f
        start_f = base[fstep] + within - within[jnp.minimum(first, cap.flat_clusters - 1)]
        start_l = base[lidx][:, None] + jnp.cumsum(rep_l, axis=1) - rep_l
        start_1 = jnp.cumsum(singles) - singles
        total = singles.sum() + per_step.sum()
        cand_step = jnp.concatenate([idx, fstep, jnp.repeat(lidx, cap.clusters_per_long_step)])
        cand_pos = jnp.concatenate([p0, pos_f, pos_l.reshape(-1, 3)])
        cand_el = jnp.concatenate([single_ne, el_f, el_l.reshape(-1)])
        cand_rep = jnp.concatenate([singles, rep_f, rep_l.reshape(-1)])
        cand_start = jnp.concatenate([start_1, start_f, start_l.reshape(-1)])
        row = jnp.arange(cap.drift_clusters, dtype=jnp.int64)
        live = cand_rep > 0
        mark = jnp.full(cap.drift_clusters, -1, jnp.int64).at[jnp.where(live, cand_start, cap.drift_clusters)].set(
            jnp.arange(cand_rep.size, dtype=jnp.int64), mode="drop")
        anchor = lax.cummax(jnp.where(mark >= 0, row, 0), axis=0)
        owner = jnp.maximum(mark[anchor], 0)
        valid = row < total
        result = {"step": cand_step[owner], "pos": cand_pos[owner],
                  "ne": jnp.where(valid, cand_el[owner], 1), "mask": valid,
                  "usage": {"drift_clusters": total, "long_steps": long.sum(),
                            "clusters_per_step": jnp.max(jnp.where(multi & ~long, nc, 0)),
                            "clusters_per_long_step": jnp.max(jnp.where(long, nc, 0)),
                            "flat_clusters": nshort, "minion_steps": mmask.sum()}}
        return result, {**errors, "drift_clusters": total > cap.drift_clusters}

    def clusters(st, selected, assets, key):
        if cap.flat_clusters is not None:
            return clusters_flat(st, selected, assets, key)
        if cap.long_steps is not None:
            return clusters_tiered(st, selected, assets, key)
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
        q0, q1, ref = sample_frame(st, idx, p0, p1, assets)
        pos, ne = dc._sample(key, idx, st["eion"], bounded, minion, q0, q1, direction, width,
                             c.EAverage, assets["energies"], assets["probabilities"],
                             cap.clusters_per_step, unroll, method)
        pos = absolute(pos, ref)
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
                  "ne": jnp.where(valid, electrons[owner], 1), "mask": valid,
                  "usage": {"drift_clusters": ends[-1], "clusters_per_step": jnp.max(jnp.where(multi, nc, 0))}}
        return result, {"clusters_per_step": overflow, "drift_clusters": ends[-1] > cap.drift_clusters}

    def drift(st, cluster, key, assets):
        pad = ncharge - cap.drift_clusters
        us = jnp.pad(st["us"][cluster["step"]], (0, pad))
        pos = jnp.pad(cluster["pos"], ((0, pad), (0, 0)))
        ne = jnp.pad(cluster["ne"], (0, pad), constant_values=1)
        times = jnp.pad(st["time"][cluster["step"]], (0, pad))
        if rel:   # float32: positions relative to the straw middle, times to whole COMMON_PERIODs, as digi_jax
            pos = pos - assets["straw_mid"][us]
            cref = period * jnp.floor(times / period)
            times = times - cref
        arrays = (jnp.arange(ncharge // dj.CHARGE_BLOCK, dtype=jnp.int64) * dj.CHARGE_BLOCK,
                  us.reshape(-1, dj.CHARGE_BLOCK), pos.reshape(-1, dj.CHARGE_BLOCK, 3),
                  ne.reshape(-1, dj.CHARGE_BLOCK), times.reshape(-1, dj.CHARGE_BLOCK))
        def block(inputs):
            first, us_, pos_, ne_, time_ = inputs
            return charge_kernel.apply(assets["charge_kernel"],
                random.key_data(random.fold_in(random.fold_in(key, 1), first)), us_, pos_, ne_, time_)
        if cap.charge_blocks_per_launch is None:
            result = lax.map(block, arrays)
        else:   # groups of blocks per launch (as digi_jax CHARGE_BLOCKS); each block keeps its own key
            g = cap.charge_blocks_per_launch
            grouped = tuple(a.reshape((-1, g) + a.shape[1:]) for a in arrays)
            result = lax.map(lambda group: jax.vmap(block)(group), grouped)
            result = tuple(r.reshape((-1,) + r.shape[2:]) for r in result)
        gt, charge, wd = tuple(value.reshape((-1,) + value.shape[2:])[:cap.drift_clusters] for value in result)
        if rel:
            gt = (cref[:cap.drift_clusters, None] + gt.astype(jnp.float64)).astype(jnp.float32)
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
            return electronics_call(electronics_kernel, assets["electronics_kernel"], key, t, q, w, step, count,
                                    cap.clusters_per_end, batch, us, ra, skey, live, jitter, endtdc, lengths)
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
        if cap.digit_entries_per_straw is not None:
            errors["digit_entries_per_straw"] = jnp.any(result["entry_overflow"])
            output["_digit_entries_per_straw"] = jnp.max(result["pairs"]) / batch
        return output, errors

    def pack_tiered(st, cluster, gt, charge, wd, timing, key, assets):
        """As pack, but a straw whose busiest end exceeds clusters_per_end goes to the narrowest end tier that holds it.

        Electronics draws are keyed per straw (event, sid), so the padded width does not change them.
        """
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
        nblocks = (batch_events + dj.JITTER_BATCH - 1) // dj.JITTER_BATCH
        jitter = lax.map(lambda block: dj._jitter_block(random.fold_in(random.fold_in(key, 2), block)),
                         jnp.arange(nblocks, dtype=jnp.int64))
        jitter = jitter.reshape(-1, 216)[straw_ev, upanel] * c.eventWindowMarkerROCJitter
        lengths = timing["event_length"][straw_ev]
        endtdc = jnp.clip(jnp.floor(lengths / c.TDCLSB), 0, c.maxTDC).astype(jnp.int64)
        busiest = jnp.where(live, counts.max(axis=1), 0)

        def gather(sel, slive, width):
            cnt = jnp.where(slive[:, None], counts[sel], 0)
            gi = order[jnp.minimum(lo[sel][:, :, None] + jnp.arange(width), len(order) - 1)]
            index, ends = ghost_src[gi] // 2, ghost_src[gi] % 2
            t, q, w, step = ghost_time[gi], charge[index], wd[index, ends], cs[index]
            vp = slive[:, None, None] & (jnp.arange(width) < cnt[:, :, None])
            t, q, w = tuple(jnp.where(vp, a, 0).astype(jnp.float64) for a in (t, q, w))
            return (t, q, w, jnp.where(vp, step, 0), jnp.minimum(cnt, width).astype(jnp.int32),
                    straw_us[sel], readall[sel], strawkey[sel], slive, jitter[sel], endtdc[sel], lengths[sel])

        edges = [cap.clusters_per_end] + [width for _, width in cap.end_tiers]
        tier = jnp.searchsorted(jnp.array(edges), busiest, side="left")   # 0 = narrow, k = end_tiers[k-1]
        packed = [gather(jnp.arange(cap.straws), live & (tier == 0), cap.clusters_per_end)]
        errors = {"straws": nstraw > cap.straws, "clusters_per_end": jnp.any(live & (busiest > edges[-1]))}
        for k, (straws, width) in enumerate(cap.end_tiers, start=1):
            member = live & (tier == k)
            sel = jnp.nonzero(member, size=straws, fill_value=0)[0]
            packed.append(gather(sel, jnp.arange(straws) < member.sum(), width))
            errors[f"end_tier_{k}_straws"] = member.sum() > straws
        return tuple(packed), errors

    def electronics_tiered(packed, key, assets):
        flat_flags, flat_usage = [], []

        def tier(values, kernel, kernel_assets, width, batch):
            groups = tuple(v.reshape((-1, batch) + v.shape[1:]) for v in values)

            def block(inputs):
                t, q, w, step, count, us, ra, skey, live, jitter, endtdc, lengths = inputs
                return electronics_call(kernel, kernel_assets, key, t, q, w, step, count, width, batch,
                                        us, ra, skey, live, jitter, endtdc, lengths)
            result = lax.map(block, groups)
            out = {name: result[name].reshape((-1,) + result[name].shape[3:])
                   for name in ("adc", "tdc", "tot", "pmp", "step", "t_analog")}
            out["mask"] = result["keep"].reshape(-1)
            out["ev"] = jnp.repeat(values[7] // 65536, cap.pairs_per_straw)
            out["sid"] = jnp.repeat(values[7] % 65536, cap.pairs_per_straw)
            if cap.digit_entries_per_straw is not None:
                flat_flags.append(jnp.any(result["entry_overflow"]))
                flat_usage.append(jnp.max(result["pairs"]) / batch)
            return out, jnp.any(result["failed"]), jnp.any(result["sat_overflow"])

        parts = [tier(packed[0], electronics_kernel, assets["electronics_kernel"], cap.clusters_per_end,
                      cap.electronics_batch)]
        for values, kernel, kernel_assets, (straws, width) in zip(packed[1:], tier_kernels,
                                                                   assets["electronics_tier_kernels"], cap.end_tiers):
            parts.append(tier(values, kernel, kernel_assets, width, cap.tier_batch(straws)))
        output = {name: jnp.concatenate([part[0][name] for part in parts]) for name in parts[0][0]}
        fa = jnp.any(jnp.stack([part[1] for part in parts]))
        sa = jnp.any(jnp.stack([part[2] for part in parts]))
        order = jnp.lexsort((output["t_analog"][:, 0], output["sid"],
                            jnp.where(output["mask"], output["ev"], sentinel)))
        output = {name: value[order] for name, value in output.items()}
        errors = {"electronics_search_or_pairs": fa, "saturation_capacity": sa}
        if flat_flags:
            errors["digit_entries_per_straw"] = jnp.any(jnp.stack(flat_flags))
            output["_digit_entries_per_straw"] = jnp.max(jnp.stack(flat_usage))
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
        usage = {"steps": mask.sum(), **cluster.pop("usage")}
        gt, charge, wd, charge_error = drift(st, cluster, key, assets)
        if cap.end_tiers is None:
            packed, packing_errors = pack(st, cluster, gt, charge, wd, timing, key, assets)
            output, electronic_errors = electronics(packed, key, assets)
            usage.update(straws=packed[8].sum(), clusters_per_end=jnp.max(packed[4]))
        else:
            packed, packing_errors = pack_tiered(st, cluster, gt, charge, wd, timing, key, assets)
            output, electronic_errors = electronics_tiered(packed, key, assets)
            # Straw and end counts are read from the packed tiers, so they saturate at a capacity that overflowed
            # (its flag then rejects the batch); otherwise they are the exact demand.
            usage.update(straws=sum(tier[8].sum() for tier in packed),
                         clusters_per_end=jnp.max(jnp.stack([jnp.max(tier[4]) for tier in packed])))
            for k, tier in enumerate(packed[1:], start=1):
                usage[f"end_tier_{k}_straws"] = tier[8].sum()
        if "_digit_entries_per_straw" in output:
            output = dict(output)
            usage["digit_entries_per_straw"] = output.pop("_digit_entries_per_straw")
        errors = {"input": input_error, "timing": timing_error, "nonfinite_charge": charge_error,
                  **cluster_errors, **packing_errors, **electronic_errors}
        return output, errors, usage

    def apply_plain(collection, timing, assets, key):
        output, errors, _ = apply(collection, timing, assets, key)
        return output, errors

    return SimpleNamespace(assets=assets, apply=apply_plain, apply_with_usage=apply, clusters=clusters, drift=drift, pack=pack,
                           electronics=electronics, capacities=cap,
                           metadata={"physics": "existing tracker JAX ionisation/drift/electronics kernels",
                                     "precision": df.NAME, "capacity_policy": "reject; no fallback",
                                     "reconstruction": "not implemented"})
