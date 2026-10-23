"""JAX twin of photons._substep_photons: the substep loop of the CRV photon stage on GPU (or CPU), fixed shapes.

The per-(step, SiPM) rows and their per-substep photon numbers come from photons.iter_rows (host, numpy rng: cheap, one
draw per row and photon type). On the device, per chunk of rows padded to a power-of-two bucket:
  kernel A: expand rows to 0.1 mm substeps, position and time per substep, scintillator / fibre membership, the three
            lookup bins, arrival probability x yield factor, and Binomial(n, p) arrivals per substep and photon type;
  kernel B: one entry per arrival (repeat with a padded total length): the time-delay draw from the bin's cumulative
            weights (the three tables are stacked into one [N, D] int32 array, offset per type) and the arrival time.
Only the arrival total crosses to the host between the kernels (to choose kernel B's bucket). float64 throughout (the
TF32 trap does not apply: no matmuls). The v6 tables have zero fibre decay time, which this twin requires.
"""
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from . import lookup, photons

jax.config.update("jax_enable_x64", True)

MIN_BUCKET = 1 << 12


def _bucket(n):
    return max(MIN_BUCKET, 1 << int(np.ceil(np.log2(max(n, 1)))))


class _NB(dict):
    """Hashable bin-count dict (a static jit argument)."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class DeviceTable:
    """A lookup table on the device: constants, bin edges, stacked probabilities and cumulative delay weights."""

    def __init__(self, L):
        if L.fiber_decay_time != 0:
            raise NotImplementedError("photons_jax assumes zero fibre decay time (v6 tables)")
        self.L = L
        T = [L.tables[k] for k in lookup.TABLES]
        D = max(t.delays.shape[1] for t in T)
        cum = [np.pad(np.cumsum(t.delays.astype(np.int32), axis=1), ((0, 0), (0, D - t.delays.shape[1])), mode="edge")
               for t in T]
        self.offsets = tuple(int(x) for x in np.r_[0, np.cumsum([len(t.prob) for t in T])[:-1]])
        self.cum = jnp.asarray(np.vstack(cum))
        self.prob = jnp.asarray(np.concatenate([t.prob for t in T]).astype(np.float64))
        self.dsum = jnp.asarray(np.concatenate([t.delay_sum for t in T]).astype(np.float64))
        self.edges = {k: jnp.asarray(v) for k, v in L.edges.items()}
        self.p_typical = 2.0 * float(L.tables["scint"].prob.mean())  # buffer estimate for device_photons (x2 margin)
        self.nb = _NB({k: len(v) - 1 for k, v in L.edges.items()})
        self.const = (L.half_thickness, L.half_width, L.half_length, L.fiber_separation, L.hole_radius_x,
                      L.hole_radius_y, L.fiber_radius, L.corner_radius)


ROW_KEYS = ("t0", "t1", "beta", "theta", "dev", "mu", "n_scint", "n_ck_s", "n_ck_f")


def _row(rows, k):
    """Per-row array k; rows without an attenuation shift (mu) get zeros."""
    return rows[k] if k in rows else np.zeros(len(rows["nsub"]))


def _find_bin(edges, v):
    i = jnp.searchsorted(edges, v, side="left")
    i = jnp.where(v == edges[0], 1, i)
    return jnp.where((i >= 1) & (i <= edges.shape[0] - 1), i - 1, -1)


@partial(jax.jit, static_argnames=("SB", "nb", "const", "offsets"))
def _kernel_a(rows, key, edges, prob, SB, nb, const, offsets):
    """Rows (padded) -> per substep: arrivals per type [SB, 3], their table bins [SB, 3], times [SB], row ids [SB]."""
    hT, hW, hL, fsep, hrx, hry, frad, crad = const
    R = rows["nsub"].shape[0]
    ri = jnp.repeat(jnp.arange(R), rows["nsub"], total_repeat_length=SB)
    first = jnp.cumsum(rows["nsub"]) - rows["nsub"]
    k = jnp.arange(SB) - first[ri]
    valid = jnp.arange(SB) < rows["nsub"].sum()
    frac = (k + 0.5) / rows["nsub"][ri]
    p = rows["s"][ri] + rows["d"][ri] * frac[:, None]
    t = rows["t0"][ri] + (rows["t1"][ri] - rows["t0"][ri]) * frac
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ax, ay = jnp.abs(x), jnp.abs(y)
    insc = (ax < hT) & (ay < hW) & (jnp.abs(z) < hL)
    insc &= (ax / hrx) ** 2 + ((ay - fsep / 2) / hry) ** 2 >= 1.0
    cx, cy = hT - crad, hW - crad
    insc &= ~((ax > cx) & (ay > cy) & (jnp.hypot(ax - cx, ay - cy) > crad))
    r0, r1 = jnp.hypot(x, y + fsep / 2), jnp.hypot(x, y - fsep / 2)
    in0, in1 = r0 <= frad, r1 <= frad
    ry = jnp.where(in1, y - fsep / 2, y + fsep / 2)
    r = jnp.where(in1, r1, r0)
    dx, dy = rows["d"][ri, 0], rows["d"][ri, 1]
    norm = jnp.hypot(dx, dy) * r
    phi = jnp.where(norm > 0, jnp.arccos(jnp.clip((dx * x + dy * ry) / jnp.where(norm > 0, norm, 1.0), -1.0, 1.0)), 0.0)
    infb = (jnp.abs(z) < hL) & (in0 | in1) & ~insc
    bx, by, bz = _find_bin(edges["x"], ax), _find_bin(edges["y"], y), _find_bin(edges["z"], z)
    bb = _find_bin(edges["beta"], rows["beta"][ri])
    geo = (bx >= 0) & (by >= 0) & (bz >= 0)
    sbin = jnp.where(insc & geo, bz + by * nb["z"] + bx * nb["y"] * nb["z"], -1)
    cbin = jnp.where(insc & geo & (bb >= 0), bb + bz * nb["beta"] + by * nb["z"] * nb["beta"]
                     + bx * nb["y"] * nb["z"] * nb["beta"], -1)
    bth, bph, br = _find_bin(edges["theta"], rows["theta"][ri]), _find_bin(edges["phi"], phi), _find_bin(edges["r"], r)
    fok = infb & (bb >= 0) & (bth >= 0) & (bph >= 0) & (br >= 0) & (bz >= 0)
    fbin = jnp.where(fok, bz + br * nb["z"] + bph * nb["r"] * nb["z"] + bth * nb["phi"] * nb["r"] * nb["z"]
                     + bb * nb["theta"] * nb["phi"] * nb["r"] * nb["z"], -1)
    bins = jnp.stack([sbin, cbin, fbin], 1)
    n = jnp.stack([rows["n_scint"][ri], rows["n_ck_s"][ri], rows["n_ck_f"][ri]], 1)
    gbin = jnp.where(bins >= 0, bins + jnp.asarray(offsets)[None, :], 0)
    pr = jnp.minimum(prob[gbin] * (rows["dev"][ri] * jnp.exp(-rows["mu"][ri] * z))[:, None], 1.0)
    n = jnp.where((bins >= 0) & valid[:, None], n, 0)
    narr = jax.random.binomial(key, n.astype(jnp.float64), pr).astype(jnp.int64)
    narr = jnp.where(n > 0, narr, 0)
    return narr, gbin, t, ri


@partial(jax.jit, static_argnames=("AB",))
def _kernel_b(narr, gbin, t, ri, key, cum, dsum, AB):
    """One entry per arrival: substep, delay draw, time; plus the validity mask of the padded tail."""
    flat = jnp.repeat(jnp.arange(narr.size), narr.ravel(), total_repeat_length=AB)
    valid = jnp.arange(AB) < narr.sum()
    sub = flat // 3
    g = gbin.ravel()[flat]
    thr = jax.random.uniform(key, (AB,)) * dsum[g]
    delay = (cum[g] < thr[:, None]).sum(axis=1)
    return t[sub] + delay, ri[sub], valid


def substep_photons(dt, rows, key, max_sub=1 << 22):
    """JAX equivalent of photons._substep_photons(L, rows, rng): (step index, channel, time) of arriving photons."""
    nsub = rows["nsub"]
    starts = np.r_[0, np.cumsum(nsub)]
    cuts = [0]
    while cuts[-1] < len(nsub):
        nxt = int(np.searchsorted(starts, starts[cuts[-1]] + max_sub, side="right")) - 1
        cuts.append(max(nxt, cuts[-1] + 1))
    out_s, out_c, out_t = [], [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        R = b - a
        RB = _bucket(R)
        pad = RB - R
        chunk = {}
        for kname in ("s", "d"):
            chunk[kname] = jnp.asarray(np.pad(rows[kname][a:b], ((0, pad), (0, 0))))
        for kname in ROW_KEYS:
            chunk[kname] = jnp.asarray(np.pad(_row(rows, kname)[a:b], (0, pad)))
        chunk["nsub"] = jnp.asarray(np.pad(nsub[a:b], (0, pad)))  # padded rows have 0 substeps
        SB = _bucket(int(nsub[a:b].sum()))
        key, ka, kb = jax.random.split(key, 3)
        narr, gbin, t, ri = _kernel_a(chunk, ka, dt.edges, dt.prob, SB, dt.nb, dt.const, dt.offsets)
        total = int(narr.sum())
        if total == 0:
            continue
        tt, rr, valid = _kernel_b(narr, gbin, t, ri, kb, dt.cum, dt.dsum, _bucket(total))
        tt, rr = np.asarray(tt)[:total], np.asarray(rr)[:total] + a
        out_s.append(rows["step"][rr])
        out_c.append(rows["chan"][rr])
        out_t.append(tt)
    if not out_s:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(out_s), np.concatenate(out_c), np.concatenate(out_t)


def make_photons(steps, pdg, geom, cfg, device_tables, deviation, rng, key):
    """JAX twin of photons.make_photons. device_tables: {table name: DeviceTable}; rng (numpy) draws the per-row photon
    numbers, key (JAX) the arrivals and delays."""
    outs = []
    for L, rows in photons.iter_rows(steps, pdg, geom, cfg, {k: v.L for k, v in device_tables.items()}, deviation, rng):
        key, sub = jax.random.split(key)
        outs.append(substep_photons(device_tables[L.name.replace("LookupTable_", "")], rows, sub))
    if not outs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return tuple(np.concatenate([o[i] for o in outs]) for i in range(3))


def load_device_tables(cfg, directory=None):
    return {k: DeviceTable(v) for k, v in photons.load_tables(cfg, directory).items()}


ARRIVAL_SAFETY = 1.25  # kernel-B buffer = 1.25 x expected arrivals + 8 sigma + 4096; overflow flagged


@partial(jax.jit, static_argnames=("SB", "AB", "nb", "const", "offsets"))
def _photons_device(chunk, step, chan, key, edges, prob, cum, dsum, SB, AB, nb, const, offsets):
    """Kernels A and B in one call with a static arrival buffer: (step, channel, time, valid, overflow) on device."""
    ka, kb = jax.random.split(key)
    narr, gbin, t, ri = _kernel_a(chunk, ka, edges, prob, SB, nb, const, offsets)
    tt, rr, valid = _kernel_b(narr, gbin, t, ri, kb, cum, dsum, AB)
    return step[rr], chan[rr], tt, valid, narr.sum() > AB


@jax.jit
def _take_rows(step, chan, rr, narr, AB):
    return step[rr], chan[rr], narr.sum() > AB


def _photons_device_split(chunk, step, chan, key, edges, prob, cum, dsum, SB, AB, nb, const, offsets):
    """_photons_device as two compiled kernels (no cross-kernel fusion); same keys, so the same photons."""
    ka, kb = jax.random.split(key)
    narr, gbin, t, ri = _kernel_a(chunk, ka, edges, prob, SB, nb, const, offsets)
    tt, rr, valid = _kernel_b(narr, gbin, t, ri, kb, cum, dsum, AB)
    s, c, over = _take_rows(step, chan, rr, narr, AB)
    return s, c, tt, valid, over


def _use_split():
    """oneAPI (Intel PVC): the fused kernel aborts the SPIR-V compiler (integer of a width other than 8/16/32/64)."""
    from ..reco.coinc_jax import is_oneapi
    return is_oneapi()


def device_photons(dt, rows, key, row_step_offset=0, split=None):
    """Photons of one iter_rows block, left on the device. The arrival buffer comes from the expected count (rows'
    photon numbers x mean arrival probability), not from a host sync. Returns dict of device arrays + overflow flag."""
    nsub = rows["nsub"]
    R = len(nsub)
    RB = _bucket(R)
    pad = RB - R
    chunk = {k: jnp.asarray(np.pad(rows[k], ((0, pad), (0, 0)))) for k in ("s", "d")}
    for k in ROW_KEYS:
        chunk[k] = jnp.asarray(np.pad(_row(rows, k), (0, pad)))
    chunk["nsub"] = jnp.asarray(np.pad(nsub, (0, pad)))
    SB = _bucket(int(nsub.sum()))
    ntot = float(((rows["n_scint"] + rows["n_ck_s"] + rows["n_ck_f"]) * nsub * np.minimum(rows["dev"], 1.0)).sum())
    mu = ntot * dt.p_typical                             # expected arrivals; overflow is flagged and redone
    AB = _bucket(int(ARRIVAL_SAFETY * mu + 8 * np.sqrt(mu) + 4096))
    step = jnp.asarray(np.pad(rows["step"] + row_step_offset, (0, pad)))
    chan = jnp.asarray(np.pad(rows["chan"], (0, pad)))
    fn = _photons_device_split if (_use_split() if split is None else split) else _photons_device
    s, c, t, v, over = fn(chunk, step, chan, key, dt.edges, dt.prob, dt.cum, dt.dsum, SB, AB, dt.nb, dt.const, dt.offsets)
    return {"step": s, "chan": c, "time": t, "valid": v, "overflow": over}
