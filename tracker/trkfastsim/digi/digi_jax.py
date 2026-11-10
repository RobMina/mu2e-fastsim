"""Opt-in fixed-shape JAX electronics with explicit random keys.

Host code packs complete straw instances into power-of-two cluster buckets. The
compiled kernel runs threshold searches and dead time in masked lock step. Pair
capacity is checked and retried, never silently truncated. NumPy digi.py remains
the independent default/reference. Same keys, inputs and bucket configuration
reproduce a run; random streams need not match NumPy or another JAX version.
"""
import os
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random

from trkfastsim import devfloat as df
from trkfastsim.digi import digitize as reference
df.register(__name__)   # follows the core device float when TRKFASTSIM_DEVICE_FLOAT=core

jax.config.update("jax_enable_x64", True)

GPU = jax.default_backend() == "gpu"
BATCH = 2048 if GPU else 32
MAX_ROW_CLUSTERS = 4096
WAVEFORM_CLUSTER_LANES = 262144 if GPU else 2048
SATURATION_TILE = 64 if GPU else 16
JITTER_BATCH = 128
CHARGE_BLOCK = 4096
CHARGE_BLOCKS = 16 if GPU else 1

# Opt-in half-precision storage of the preamp response tables (default off: every table float32, as shipped).
# TRKFASTSIM_TABLE_FLOAT=float16|bfloat16 stores the tables named in TRKFASTSIM_TABLE_PATHS (comma list of sat, adc,
# thresh; default sat, where "sat" also covers the ADC impulse table resp_imp) in that dtype; gathered values are
# promoted to float32 before any arithmetic, so only the table entries are rounded. float16 tables are scaled by an
# exact power of two first: resp_sat peaks at 5.8e-5, below float16's smallest normal (6.1e-5).
TABLE_FLOAT = os.environ.get("TRKFASTSIM_TABLE_FLOAT", "") or None
TABLE_PATHS = tuple(p for p in os.environ.get("TRKFASTSIM_TABLE_PATHS", "sat").split(",") if p)
if TABLE_FLOAT not in (None, "float16", "bfloat16"):
    raise ValueError(f"TRKFASTSIM_TABLE_FLOAT must be float16 or bfloat16, not {TABLE_FLOAT!r}")
if not set(TABLE_PATHS) <= {"sat", "adc", "thresh"}:
    raise ValueError(f"TRKFASTSIM_TABLE_PATHS must name sat, adc or thresh, not {TABLE_PATHS!r}")


def _half_table(values):
    """(table in TABLE_FLOAT, inverse scale): float16 entries scaled by 2**k so the peak sits near 2**14."""
    x = np.asarray(values, np.float64)
    peak = float(np.max(np.abs(x)))
    k = 14 - int(np.ceil(np.log2(peak))) if TABLE_FLOAT == "float16" and peak > 0 else 0
    return jnp.asarray(x * 2.0 ** k, dtype=TABLE_FLOAT), 2.0 ** -k


@df.jit
def _jitter_block(key):
    return df.draw(random.normal, key, (JITTER_BATCH, reference.NUPANELS))


def _panel_jitter(key, events, panels, scale):
    blocks = events // JITTER_BATCH
    out = np.empty(len(events))
    for block in np.unique(blocks):
        which = blocks == block
        values = np.asarray(_jitter_block(random.fold_in(key, int(block))))
        out[which] = values[events[which] % JITTER_BATCH, panels[which]] * scale
    return out


def _waveform_batch(cap):
    return max(1, min(BATCH, WAVEFORM_CLUSTER_LANES // cap))


def _empty(c):
    return {"ev": np.zeros(0, np.int64), "sid": np.zeros(0, np.int64),
            "tdc": np.zeros((0, 2), np.int64), "tot": np.zeros((0, 2), np.int64),
            "pmp": np.zeros(0, np.int64), "adc": np.zeros((0, c.nADCSamples), np.int64),
            "step": np.zeros((0, 2), np.int64), "t_analog": np.zeros((0, 2))}


def _gamma(key, a):
    if df.DRAW64:
        return random.gamma(key, a.astype(np.float64), dtype=np.float64).astype(df.FLOAT)
    return random.gamma(key, a)


def _relative_geometry(c):
    """float32 mode: cluster positions are relative to their straw's middle, so the wire middles are too."""
    a = dict(c.a)
    if df.SINGLE:
        a["wire_mid"] = np.asarray(c.a["wire_mid"]) - np.asarray(c.a["straw_mid"])
        a["straw_mid"] = np.zeros_like(c.a["straw_mid"])
    return df.dev({k: jnp.asarray(v) for k, v in a.items()})


def _d2t(rho, phi, dist, times, dD, dphi):
    """StrawDrift::D2T with radial extrapolation and folded-phi interpolation."""
    fphi = jnp.abs(jnp.fmod(phi, jnp.pi))
    fphi = jnp.where(fphi > jnp.pi / 2, jnp.pi - fphi, fphi)
    p0 = jnp.maximum(0, jnp.floor(fphi / dphi)).astype(jnp.int32)
    p1 = jnp.minimum(times.shape[1] - 1, jnp.ceil(fphi / dphi)).astype(jnp.int32)
    di = jnp.clip(jnp.floor(rho / dD).astype(jnp.int32), 0, len(dist) - 2)
    fr = (rho - dist[di]) / dD
    lo = times[di, p0] + fr * (times[di + 1, p0] - times[di, p0])
    hi = times[di, p1] + fr * (times[di + 1, p1] - times[di, p1])
    return jnp.where(phi == 0, lo, lo + (fphi - p0 * dphi) / dphi * (hi - lo))


@lru_cache(maxsize=8)
def _charge_kernel(c):
    a = _relative_geometry(c)
    dtv0, dtv1 = (float(x) for x in np.asarray(c.dtvar)[:2])
    field_free = np.all(c.drift_times == c.drift_times[:, :1])
    pdir = np.cross(np.asarray(c.wire_dir), np.array([0., 0., 1.]))
    pdir *= np.where(np.sum(pdir * c.wire_mid, axis=1) < 0, -1., 1.)[:, None]
    pdir = jnp.asarray(df.dev(pdir))

    @df.jit
    def run(key, us, pos, ne, time):
        k1, k2, k3 = random.split(key, 3)
        nef = ne.astype(df.FLOAT)
        aval = jnp.where(ne == 1, c.polyaA, nef * (c.polyaA + 1) - 1).astype(df.FLOAT)
        gain = jnp.where(ne < c.nGainGauss, _gamma(k1, aval) / aval,
                         1 + c.gainRMSSlope / jnp.sqrt(nef) * df.draw(random.normal, k2, ne.shape))
        charge = (c.Qe * nef * c.gasGain * gain).astype(jnp.float32)
        d = pos - a["straw_mid"][us]
        dw = jnp.sum(d * a["straw_dir"][us], 1)
        cp = d - dw[:, None] * a["straw_dir"][us]
        rho = jnp.linalg.norm(cp, axis=1)
        pos = pos - cp / jnp.maximum(rho, 1e-30)[:, None] * jnp.maximum(rho - c.strawInnerRadius, 0)[:, None]
        d = pos - a["wire_mid"][us]
        zw = jnp.sum(d * a["wire_dir"][us], 1)
        rho = jnp.linalg.norm(d - zw[:, None] * a["wire_dir"][us], axis=1)
        di = jnp.clip(jnp.floor(rho / c.drift_dD).astype(jnp.int32), 0, len(c.drift_dist) - 2)
        fr = (rho - a["drift_dist"][di]) / c.drift_dD
        if field_free:
            td = a["drift_times"][di, 0] + fr * (a["drift_times"][di + 1, 0] - a["drift_times"][di, 0])
        else:
            cp = d - zw[:, None] * a["wire_dir"][us]
            phi = jnp.arctan2(jnp.sum(cp * pdir[us], axis=1), cp[:, 2])
            td = _d2t(rho, phi, a["drift_dist"], a["drift_times"], c.drift_dD, c.drift_dphi)
        td = (td + (dtv0 + dtv1 * rho) * df.draw(random.normal, k3, ne.shape)).astype(jnp.float32)
        hl = a["wire_halflen"][us]
        wd = jnp.stack([hl - zw, hl + zw], 1).astype(jnp.float32)
        prop = (wd / c.vprop).astype(jnp.float32)
        gt = (time[:, None] + td[:, None].astype(df.FLOAT) + prop.astype(df.FLOAT)).astype(jnp.float32)
        return gt, (jnp.float32(0.5) * charge), wd
    return run


@lru_cache(maxsize=8)
def _charge_blocks_kernel(c):
    charge = _charge_kernel(c)

    @df.jit
    def run(key, first, us, pos, ne, time):
        offsets = first + jnp.arange(CHARGE_BLOCKS, dtype=jnp.int64) * CHARGE_BLOCK
        keys = jax.vmap(lambda offset: random.fold_in(random.fold_in(key, 1), offset))(offsets)
        args = (us.reshape(CHARGE_BLOCKS, CHARGE_BLOCK), pos.reshape(CHARGE_BLOCKS, CHARGE_BLOCK, 3),
                ne.reshape(CHARGE_BLOCKS, CHARGE_BLOCK), time.reshape(CHARGE_BLOCKS, CHARGE_BLOCK))
        result = jax.vmap(charge)(keys, *args)
        return tuple(a.reshape((-1,) + a.shape[2:]) for a in result)
    return run


def _saturation_rows(batch, saturation_rows=None):
    """Rows per compacted saturation chunk of the flat digitisation: the loop's 2 * batch, or with "auto" batch rows
    in float64 only (bit-identical there over full passes; float32 response sums drift, so float32 keeps 2 * batch)."""
    if saturation_rows not in (None, "auto"):
        raise ValueError("saturation_rows must be None or 'auto'")
    return batch if saturation_rows == "auto" and not df.SINGLE else 2 * batch


@lru_cache(maxsize=32)
def _electronics_kernel(c, cap, pair_cap, batch, entries=None, saturation_rows=None):
    """entries: None digitises pair by pair over every lane (default); an int digitises the flat list of present
    (straw, pair) entries, at most `entries` per call, with the same keys and arithmetic. Saturated ends are
    compacted into chunks of 2 * batch rows, the loop's lane count; saturation_rows="auto" uses batch rows in float64,
    where they were measured bit-identical over full passes (in float32 they drift by roundoff, so 2 * batch stays).
    An entry overflow is flagged in entry_overflow, never truncated."""
    return _electronics_parts(c, cap, pair_cap, batch, entries, saturation_rows).run


@lru_cache(maxsize=32)
def _electronics_parts(c, cap, pair_cap, batch, entries=None, saturation_rows=None, saturation_pool=False):
    """The electronics kernel `run` (as _electronics_kernel) and, with saturation_pool=True (flat digitisation only),
    the kernels that saturate the ends of many launches together.

    Pooled, `run` leaves saturation and the ADC to the caller: it returns "pool" (the saturated-end flags, their lane
    data, ADC sample times, pre-saturation volts and ADC noise) instead of "adc" and "pmp". `pool_saturate` takes
    the pools of every launch of one kernel shape, flattened, and saturates their saturated ends in chunks of the same
    srows rows as per launch, so each chunk computes as a per-launch chunk does, but fewer chunks run (a launch's
    padding is filled by the next launch's ends); `pool_finish(volts, noise, row, kk, straw_us)` is the per-launch
    ADC and PMP. Exact: the same keys and arithmetic, the same chunk shapes."""
    if saturation_pool and entries is None:
        raise ValueError("saturation_pool needs the flat digitisation (entries)")
    srows = _saturation_rows(batch, saturation_rows)
    rb, sr, LB = c.responseBins, c.sampleRate, c.clusterLookbackTime
    R = batch * 2
    rows = jnp.arange(R)
    ci = jnp.arange(cap)[None, :]
    cond = df.dev({k: jnp.asarray(v) for k, v in c.a.items()})
    rel = df.SINGLE   # float32: times relative to a per-straw reference tref = n_ref * COMMON_PERIOD
    if rel:
        df.check_common_period(c)
        per_adc, per_tdc = int(round(df.COMMON_PERIOD / c.ADCPeriod)), int(round(df.COMMON_PERIOD / c.TDCLSB))
    tables = {k: cond["resp_" + k].astype(jnp.float32) for k in ("thresh", "adc", "sat")}
    unscale = {}          # path -> exact power-of-two factor undoing the half-table scale (opt-in TABLE_FLOAT only)
    imp_unscale = None
    if TABLE_FLOAT:
        for k in TABLE_PATHS:
            tables[k], unscale[k] = _half_table(c.a["resp_" + k])
        if "sat" in TABLE_PATHS:
            imp_table, imp_unscale = _half_table(c.a["resp_imp"])

    def impulse(resp, ii):
        """sum over saturation steps of resp x the ADC impulse table at ii (half table: promoted, then unscaled)."""
        if imp_unscale is None:
            return jnp.sum(resp * cond["resp_imp"][ii], 1)
        return jnp.sum(resp * imp_table[ii].astype(resp.dtype), 1) * imp_unscale
    dts = c.saturationSampleFactor / sr
    sat_steps = int(np.ceil((c.ttrunc + c.nADCSamples * c.ADCPeriod + 2 * LB) / dts)) + 2
    ticks = jnp.arange(1, c.maxTOT)
    adc_k = jnp.arange(c.nADCSamples)
    padded_steps = ((sat_steps + SATURATION_TILE - 1) // SATURATION_TILE) * SATURATION_TILE

    def count_lanes(L, tlim, inclusive=False):
        cmp = L["t"] <= tlim[:, None] if inclusive else L["t"] < tlim[:, None]
        return jnp.sum(L["valid"] & cmp, axis=1).astype(jnp.int32)

    def volt_lanes(L, tq, path="thresh", cmin=None, lattice=None, lattice_origin=None, lattice_offset=0):
        t, tr, dk, aa, bb, sc, valid = (L[k] for k in ("t", "tr", "dk", "aa", "bb", "sc", "valid"))
        tq = tq.reshape(t.shape[0], -1)
        T = tables[path]
        if lattice is None:
            u = ((tq[:, None, :] - t[:, :, None]) * sr + rb / 2.).astype(jnp.float32)
            ix = jnp.clip(u, 0, rb - 1).astype(jnp.int32)
            ir = jnp.clip(u - (tr * sr).astype(jnp.float32)[:, :, None], 0, rb - 1).astype(jnp.int32)
        else:
            origin = tq[:, 0] if lattice_origin is None else lattice_origin
            u = (origin[:, None] - t) * sr + rb / 2.
            stride = (jnp.arange(tq.shape[1]) + lattice_offset) * lattice
            ix = jnp.clip(jnp.floor(u)[:, :, None] + stride, 0, rb - 1).astype(jnp.int32)
            ir = jnp.clip(jnp.floor(u - tr * sr)[:, :, None] + stride, 0, rb - 1).astype(jnp.int32)
        k = dk[:, :, None]
        v = aa[:, :, None] * (T[k, ix] + sc[:, :, None] * T[k, ir]) + bb[:, :, None] * (T[k + 1, ix] + sc[:, :, None] * T[k + 1, ir])
        ok = valid[:, :, None] & (t[:, :, None] - LB < tq[:, None, :])
        if cmin is not None:
            ok &= (ci >= cmin[:, None])[:, :, None]
        v = jnp.sum(jnp.where(ok, v, 0).astype(df.FLOAT), axis=1)
        if path in unscale:
            v = v * unscale[path]
        v *= (L["dv_a"] if path == "adc" else L["dv_t"])[:, None]
        return v

    def saturation_lanes(S, times):
        """Saturated ADC volts of the end rows S at ADC sample times `times`, and their step-capacity overflow."""
        n_ = S["n"]
        rws = jnp.arange(n_.shape[0])
        i0 = jnp.minimum(count_lanes(S, times[:, 0] + LB - c.ttrunc, inclusive=True), jnp.maximum(n_ - 1, 0))
        t0 = S["t"][rws, jnp.clip(i0, 0, cap - 1)] - LB
        nstep = jnp.maximum(0, jnp.ceil((times[:, -1] - t0) / dts).astype(jnp.int32))
        def tile(k, response):
            offset = k * SATURATION_TILE
            tick = offset + jnp.arange(SATURATION_TILE)
            tq = t0[:, None] + dts * tick[None, :].astype(df.FLOAT)
            values = volt_lanes(S, tq, "sat", cmin=i0, lattice=c.saturationSampleFactor,
                                lattice_origin=t0, lattice_offset=offset)
            values = jnp.where(tick[None, :] < nstep[:, None], values, 0)
            return lax.dynamic_update_slice(response, values, (0, offset))
        resp = lax.fori_loop(0, padded_steps // SATURATION_TILE, tile, jnp.zeros((n_.shape[0], padded_steps)))
        resp = jnp.minimum(resp[:, :sat_steps], c.saturationVoltage)
        tq = t0[:, None] + dts * jnp.arange(sat_steps)[None, :].astype(df.FLOAT)
        scale = c.saturationSampleFactor * S["dv_a"] / S["dv_t"]
        def sample(tm):
            ii = jnp.clip(jnp.trunc((tm[:, None] - tq) * sr + rb / 2.), 0, rb - 1).astype(jnp.int32)
            return impulse(resp, ii) * scale
        return jax.vmap(sample, in_axes=1, out_axes=1)(times), nstep > sat_steps

    def finish_adc(volts, noise, row, kk, us_e):
        """ADC words of the flat entries from their two ends' (saturated) volts and the ADC noise (adcAnalogNoise x
        the stage-5 normals, formed in the launch): [batch, pair_cap, samples]."""
        n_e = noise.shape[0]
        volts = volts.reshape(n_e, 2, c.nADCSamples).sum(1)
        wsum = volts + noise
        adc = jnp.clip(jnp.floor(wsum / c.ADCLSB).astype(jnp.int64) + cond["adc_ped"][us_e, None], 0, c.maxADC)
        return jnp.zeros((batch, pair_cap, c.nADCSamples), jnp.int64).at[row, kk].set(adc, mode="drop")

    def per_launch_saturation(L, times, sat, volts, noise, row, kk, us_e):
        """Saturated ends of one launch compacted into chunks of srows rows, enough for every entry; the response
        sums depend on the reduction shapes: 2 * batch (the loop's) is exact in both dtypes, batch rows in float64
        only (float32 drifts by roundoff over full passes); empty chunks are skipped."""
        n_rows = sat.shape[0]
        nsat = sat.sum()
        chunks = -(-n_rows // srows)
        sidx = jnp.nonzero(sat, size=chunks * srows, fill_value=0)[0]
        slive = jnp.arange(chunks * srows) < nsat
        sat_overflow = jnp.bool_(False)
        for chunk in range(chunks):
            cidx = sidx[chunk * srows:(chunk + 1) * srows]
            clive = slive[chunk * srows:(chunk + 1) * srows]

            def saturate(volts, cidx=cidx, clive=clive):
                result, of = saturation_lanes({name: value[cidx] for name, value in L.items()}, times[cidx])
                return (volts.at[jnp.where(clive, cidx, n_rows)].set(result, mode="drop"),
                        jnp.any(clive & of))
            volts, of = lax.cond(jnp.any(clive), saturate, lambda v: (v, jnp.bool_(False)), volts)
            sat_overflow |= of
        return finish_adc(volts, noise, row, kk, us_e), sat_overflow

    @df.jit
    def pool_saturate(lanes, times, sat, volts):
        """Saturated ends of many launches (pools flattened over launches) in chunks of srows rows: the chunk shape
        of one launch, filled across launches; chunks past the last saturated end are not run."""
        n_rows = sat.shape[0]
        nsat = sat.sum()
        chunks = -(-n_rows // srows)
        sidx = jnp.nonzero(sat, size=chunks * srows, fill_value=0)[0]

        def body(state):
            k, volts, overflow = state
            cidx = lax.dynamic_slice(sidx, (k * srows,), (srows,))
            clive = k * srows + jnp.arange(srows) < nsat
            result, of = saturation_lanes({name: value[cidx] for name, value in lanes.items()}, times[cidx])
            return (k + 1, volts.at[jnp.where(clive, cidx, n_rows)].set(result, mode="drop"),
                    overflow | jnp.any(clive & of))
        _, volts, overflow = lax.while_loop(lambda x: x[0] * srows < nsat, body,
                                            (jnp.int32(0), volts, jnp.bool_(False)))
        return volts, overflow

    @df.jit
    def pool_finish(volts, noise, row, kk, us_e):
        adc_out = finish_adc(volts, noise, row, kk, us_e)
        return adc_out, pmp_of(adc_out)

    def pmp_of(adc_out):
        return (adc_out.max(2) - adc_out[:, :, :c.nADCPresamples].sum(2) // c.nADCPresamples) & 0xffff

    @df.jit
    def run(key, t, q, wd, step, n, straw_us, straw_ra, straw_key, live_straw, jitter, n_ref=None,
            event_end_tdc=None, window_end=None):
        n = n.astype(jnp.int32)
        if rel and n_ref is None:     # absolute times (direct kernel callers)
            n_ref = jnp.zeros(batch, jnp.int64)
        tref = n_ref.astype(df.FLOAT) * df.COMMON_PERIOD if rel else 0.0     # [batch], exact multiples
        us = jnp.repeat(straw_us, 2)
        live = jnp.repeat(live_straw, 2)
        valid = (ci < n[:, None]) & live[:, None]
        wp = cond["wire_points"]
        dk = jnp.clip(jnp.searchsorted(wp, wd, side="right") - 1, 0, len(c.wire_points) - 2)
        frac = 1 - (wd - wp[dk]) / (wp[dk + 1] - wp[dk])
        path = 4 * cond["wire_halflen"][us, None] - 2 * wd
        tr = float(c.s["reflectionTimeShift"]) + path / float(c.s["reflectionVelocity"])
        sc = (float(c.s["reflectionFrac"]) * jnp.exp(-path / float(c.s["reflectionALength"]))).astype(jnp.float32)
        aa, bb = (q * frac).astype(jnp.float32), (q * (1 - frac)).astype(jnp.float32)
        dv_t, dv_a = cond["dvdi_thresh"][us], cond["dvdi_adc"][us]
        sn = us % 96
        tmax = cond["tmax"][dk, 0, sn[:, None]] * frac + cond["tmax"][dk + 1, 0, sn[:, None]] * (1 - frac)
        maxlin = (cond["linmax"][dk, 0, sn[:, None]] * frac + cond["linmax"][dk + 1, 0, sn[:, None]] * (1 - frac)) * q
        maxlin = jnp.where(valid, maxlin * dv_t[:, None], 0)
        sums = jnp.cumsum(maxlin, axis=1)
        maxposs = jnp.sum(maxlin, axis=1)
        keys = jax.vmap(lambda x: random.fold_in(random.fold_in(key, (x >> 32).astype(jnp.uint32)), x.astype(jnp.uint32)))(straw_key)
        base = {"t": t, "tr": tr, "dk": dk, "aa": aa, "bb": bb, "sc": sc, "valid": valid, "dv_a": dv_a, "dv_t": dv_t,
                "n": n}

        def normal(stage, iteration, shape):
            kk = jax.vmap(lambda k: random.fold_in(random.fold_in(k, stage), iteration))(keys)
            return jax.vmap(lambda k: df.draw(random.normal, k, shape))(kk)

        def at(x, i):
            return x[rows, jnp.clip(i, 0, cap - 1)]

        def count(tlim, inclusive=False, lanes=None):
            return count_lanes(lanes or base, tlim, inclusive)

        def volt(tq, path="thresh", cmin=None, lattice=None, lattice_origin=None, lattice_offset=0, lanes=None):
            return volt_lanes(lanes or base, tq, path, cmin, lattice, lattice_origin, lattice_offset)

        def crosses(ic, time, thr, vcross, active):
            ic = jnp.where(active, jnp.maximum(ic, count(time + LB)), ic)
            active &= ic < n
            vstart = volt(at(t, ic) - LB)[:, 0]
            retact = active & (vstart > thr)

            def retbody(state):
                ic, time, vs, act, it = state
                nxt = at(t, ic) + LB + 2 * at(tmax, ic)
                newic = jnp.maximum(ic, count(nxt + LB))
                ic = jnp.where(act, newic, ic)
                act &= ic < n
                time = jnp.where(act, at(t, ic) - LB, time)
                vs = jnp.where(act, volt(time)[:, 0], vs)
                return ic, time, vs, act & (vs > thr), it + 1
            ic, time, vstart, stuck, _ = lax.while_loop(
                lambda x: jnp.any(x[3]) & (x[4] <= cap), retbody,
                (ic, time, vstart, retact, jnp.int32(0)))
            active &= ic < n
            before = jnp.where(ic > 0, at(sums, ic - 1), 0.)
            j = jnp.maximum(ic, jnp.sum(valid & (sums <= (thr - vstart + before)[:, None]), 1).astype(jnp.int32))
            ic = jnp.where(active & (j >= n), n, ic)
            active &= j < n
            time = jnp.where(active, at(t, j) - LB, time)
            star = jnp.full(R, -1, jnp.int32)
            response = jnp.zeros(R)

            def peakbody(state):
                j, star, response, act = state
                cand = j[:, None] + jnp.arange(8)[None, :]
                gc = jnp.clip(cand, 0, cap - 1)
                mt = t[rows[:, None], gc] + tmax[rows[:, None], gc]
                v = volt(mt)
                hit = (cand < n[:, None]) & (v > thr[:, None]) & act[:, None]
                anyhit = jnp.any(hit, 1)
                first = jnp.argmax(hit, 1).astype(jnp.int32)
                star = jnp.where(anyhit, j + first, star)
                response = jnp.where(anyhit, v[rows, first], response)
                j = jnp.where(act & ~anyhit, j + 8, j)
                return j, star, response, act & ~anyhit & (j < n)
            _, star, response, _ = lax.while_loop(lambda x: jnp.any(x[3]), peakbody, (j, star, response, active))
            ic = jnp.where(active & (star < 0), n, ic)
            found = active & (star >= 0)
            pre, post = at(t, star) - LB, at(t, star) + at(tmax, star)
            pres, posts = volt(pre)[:, 0], response
            tt = pre + (post - pre) / (posts - pres) * (thr - pres)
            dt = post - pre

            def finebody(_, state):
                pre, post, pres, posts, tt, dt = state
                act = found & (jnp.abs(dt) > .020)
                smp = volt(tt)[:, 0]
                up = smp > thr
                post = jnp.where(act & up, tt, post)
                posts = jnp.where(act & up, smp, posts)
                pre = jnp.where(act & ~up, tt, pre)
                pres = jnp.where(act & ~up, smp, pres)
                newt = pre + (post - pre) / (posts - pres) * (thr - pres)
                return pre, post, pres, posts, jnp.where(act, newt, tt), jnp.where(act, newt - tt, dt)
            _, _, _, _, tt, _ = lax.fori_loop(0, 10, finebody, (pre, post, pres, posts, tt, dt))
            time = jnp.where(found, tt, time)
            vcross = jnp.where(found, thr, vcross)
            advance = jnp.maximum(star, count(tt + LB))
            ic = jnp.where(found, jnp.maximum(advance - 1, 0), ic)
            return ic, time, thr, vcross, found, jnp.any(stuck)

        noise = normal(0, 0, ()) * c.strawNoise
        threshold = (cond["vthresh"][straw_us] + noise[:, None] + c.thresholdAnalogNoise * normal(1, 0, (2,))).reshape(R)
        ts = jnp.where(straw_ra, df.FLOAT.type(-c.deadTimeAnalog),
                       df.FLOAT.type(c.digitizationStart - c.timeFromProtonsToDRMarker - c.flashBuffer))
        if rel:
            ts = ts - tref
        ic, tt, th, vv, cr, failed = crosses(jnp.zeros(R, jnp.int32), jnp.repeat(ts, 2), threshold, jnp.zeros(R), live)
        if c.onspill:
            end = max(c.digitizationEnd - c.timeFromProtonsToDRMarker, c.mbtime) - c.electronicsTimeDelay + c.tdcbuf
        elif window_end is not None:
            end = jnp.maximum(window_end, c.mbtime) - c.electronicsTimeDelay + c.tdcbuf
        else:
            end = max(c.eventWindowLength, c.mbtime) - c.electronicsTimeDelay + c.tdcbuf

        def active(cr, tt):
            t0, t1 = tt[::2], tt[1::2]
            return live_straw & cr[::2] & cr[1::2] & (jnp.where(t0 < t1, t1, t0) < (end - tref if rel else end))

        output_t = jnp.zeros((batch, pair_cap, 2))
        output_v = jnp.zeros_like(output_t)
        output_i = jnp.zeros((batch, pair_cap, 2), jnp.int32)
        number = jnp.zeros(batch, jnp.int32)

        def pairbody(state):
            ic, tt, th, vv, cr, ot, ov, oi, number, it, fail = state
            act = active(cr, tt)
            t0, t1 = tt[::2], tt[1::2]
            finite = jnp.isfinite(t0) & jnp.isfinite(t1)
            comb = act & finite & (jnp.abs(jnp.floor(t0 / c.ADCPeriod) - jnp.floor(t1 / c.ADCPeriod)) < c.maxThreshTimeSeparation)
            rec = comb & (jnp.minimum(t0, t1) > (-tref if rel else 0.))
            loc = jnp.minimum(number, pair_cap - 1)
            bi = jnp.arange(batch)
            ot = ot.at[bi, loc].set(jnp.where(rec[:, None], tt.reshape(batch, 2), ot[bi, loc]))
            ov = ov.at[bi, loc].set(jnp.where(rec[:, None], vv.reshape(batch, 2), ov[bi, loc]))
            oi = oi.at[bi, loc].set(jnp.where(rec[:, None], ic.reshape(batch, 2), oi[bi, loc]))
            fail |= jnp.any(rec & (number >= pair_cap))
            number += rec.astype(jnp.int32)
            early = jnp.where(t0 < t1, 0, 1)
            adv = jnp.repeat(comb, 2) | (jnp.repeat(act & ~comb, 2) & (rows % 2 == jnp.repeat(early, 2)))
            tt = jnp.where(adv, tt + c.deadTimeAnalog, tt)
            ic = ic + adv.astype(jnp.int32)
            fresh = (cond["vthresh"][straw_us] + c.thresholdAnalogNoise * normal(2, it, (2,))).reshape(R)
            th = jnp.where(jnp.repeat(comb, 2), fresh, th)
            ic, tt, th, vv, newcr, stuck = crosses(ic, tt, th, vv, adv)
            cr = jnp.where(adv, newcr, cr)
            return ic, tt, th, vv, cr, ot, ov, oi, number, it + 1, fail | stuck
        state = lax.while_loop(lambda x: jnp.any(active(x[4], x[1])) & (x[9] <= 2 * cap), pairbody,
                              (ic, tt, th, vv, cr, output_t, output_v, output_i, number, jnp.int32(0), failed))
        ic, tt, th, vv, cr, pt, pv, pi, number, _, failed = state
        failed |= jnp.any(active(cr, tt))
        entry_fail = jnp.bool_(False)
        if entries is None:
            ready = jnp.full(batch, -9e9)
            sat_overflow = jnp.bool_(False)
            adc_out = jnp.zeros((batch, pair_cap, c.nADCSamples), jnp.int64)
            tdc_out = jnp.zeros((batch, pair_cap, 2), jnp.int64)
            tot_out = jnp.zeros_like(tdc_out)
            accepted = jnp.zeros((batch, pair_cap), bool)

            def digitbody(k, state):
                ready, adc_out, tdc_out, tot_out, accepted, overflow = state
                present = live_straw & (k < number)
                T, V = pt[:, k], pv[:, k]
                xt = T + jitter[:, None] + c.TDCResolution * normal(3, k, (2,))
                tot_thr = V + c.thresholdAnalogNoise * normal(4, k, (2,))
                tq = (T + jitter[:, None]).reshape(R, 1) + ticks[None, :].astype(df.FLOAT) * c.TOTLSB
                volts_tot = volt(tq, lattice=int(round(c.TOTLSB * sr)))
                below = volts_tot < (tot_thr.reshape(R, 1) - c.triggerHysteresis)
                tot = jnp.where(jnp.any(below, 1), jnp.argmax(below, 1) + 1, c.maxTOT).reshape(batch, 2)
                if rel:
                    phase = jnp.maximum(0, jnp.ceil(T[:, 0] / c.ADCPeriod).astype(jnp.int64) + n_ref * per_adc
                                        - c.nADCPresamples) - n_ref * per_adc
                else:
                    phase = jnp.maximum(0, jnp.ceil(T[:, 0] / c.ADCPeriod).astype(jnp.int64) - c.nADCPresamples)
                times = jnp.repeat((phase[:, None] + adc_k).astype(df.FLOAT) * c.ADCPeriod + c.ADCOffset, 2, axis=0)
                volts = volt(times, "adc", lattice=int(round(c.ADCPeriod * sr)))
                sat = (maxposs > c.saturationVoltage) & jnp.repeat(present, 2)

                def saturation(v):
                    i0 = jnp.minimum(count(times[:, 0] + LB - c.ttrunc, inclusive=True), jnp.maximum(n - 1, 0))
                    t0 = at(t, i0) - LB
                    nstep = jnp.maximum(0, jnp.ceil((times[:, -1] - t0) / dts).astype(jnp.int32))
                    def tile(k, response):
                        offset = k * SATURATION_TILE
                        tick = offset + jnp.arange(SATURATION_TILE)
                        tq = t0[:, None] + dts * tick[None, :].astype(df.FLOAT)
                        values = volt(tq, "sat", cmin=i0, lattice=c.saturationSampleFactor,
                                      lattice_origin=t0, lattice_offset=offset)
                        values = jnp.where(tick[None, :] < nstep[:, None], values, 0)
                        return lax.dynamic_update_slice(response, values, (0, offset))
                    padded_steps = ((sat_steps + SATURATION_TILE - 1) // SATURATION_TILE) * SATURATION_TILE
                    resp = lax.fori_loop(0, padded_steps // SATURATION_TILE, tile, jnp.zeros((R, padded_steps)))
                    resp = jnp.minimum(resp[:, :sat_steps], c.saturationVoltage)
                    tq = t0[:, None] + dts * jnp.arange(sat_steps)[None, :].astype(df.FLOAT)
                    scale = c.saturationSampleFactor * dv_a / dv_t
                    def sample(tm):
                        ii = jnp.clip(jnp.trunc((tm[:, None] - tq) * sr + rb / 2.), 0, rb - 1).astype(jnp.int32)
                        return impulse(resp, ii) * scale
                    result = jax.vmap(sample, in_axes=1, out_axes=1)(times)
                    return jnp.where(sat[:, None], result, v), jnp.any(sat & (nstep > sat_steps))
                volts, of = lax.cond(jnp.any(sat), saturation, lambda v: (v, jnp.bool_(False)), volts)
                overflow |= of
                volts = volts.reshape(batch, 2, c.nADCSamples).sum(1)
                dtime = jnp.max(xt, 1)
                if not c.overrideDbTimeOffsets:
                    xt -= cond["time_offsets"][straw_us]
                wsum = volts + c.adcAnalogNoise * normal(5, k, (c.nADCSamples,))
                if rel:
                    tdc = jnp.clip(jnp.floor((xt + c.electronicsTimeDelay) / c.TDCLSB).astype(jnp.int64)
                                   + (n_ref * per_tdc)[:, None], 0, c.maxTDC)
                else:
                    tdc = jnp.clip(jnp.floor((xt + c.electronicsTimeDelay) / c.TDCLSB), 0, c.maxTDC).astype(jnp.int64)
                ewend = (int(c.tdc_response(c.eventWindowLength - c.electronicsTimeDelay))
                         if event_end_tdc is None else event_end_tdc[:, None])
                ok_all = jnp.all((xt + c.electronicsTimeDelay > (-tref[:, None] if rel else 0)) & (tdc < ewend), 1)
                upper = c.digitizationEndTDC if c.onspill else ewend
                ok_norm = jnp.all((tdc > c.digitizationStartTDC) & (tdc < upper), 1)
                keep = present & jnp.where(straw_ra, ok_all, ok_norm) & (dtime >= ready)
                ready = jnp.where(keep, dtime + c.deadTimeDigital, ready)
                adc = jnp.clip(jnp.floor(wsum / c.ADCLSB).astype(jnp.int64) + cond["adc_ped"][straw_us, None], 0, c.maxADC)
                return ready, adc_out.at[:, k].set(adc), tdc_out.at[:, k].set(tdc), tot_out.at[:, k].set(tot), accepted.at[:, k].set(keep), overflow
            _, adc_out, tdc_out, tot_out, accepted, sat_overflow = lax.fori_loop(
                0, jnp.minimum(pair_cap, jnp.max(number)), digitbody,
                (ready, adc_out, tdc_out, tot_out, accepted, sat_overflow))
        else:
            # Flat digitisation: one row per present (straw, pair) entry instead of pair_cap passes over every lane.
            # Same keys (per straw, stage, pair ordinal) and arithmetic; the digital dead time is applied after, per
            # straw in pair order, exactly as the pair loop does.
            npairs = jnp.where(live_straw, jnp.minimum(number, pair_cap), 0)
            cum = jnp.cumsum(npairs)
            total = cum[-1]

            def flat_digits(_):
                e = jnp.arange(entries)
                sb = jnp.minimum(jnp.searchsorted(cum, e, side="right"), batch - 1)
                present = e < total
                kk = jnp.where(present, e - (cum[sb] - npairs[sb]), 0).astype(jnp.int32)
                lane = jnp.stack([2 * sb, 2 * sb + 1], 1).reshape(-1)
                L = {name: value[lane] for name, value in base.items()}
                ekeys = keys[sb]

                def enormal(stage, shape):
                    kk2 = jax.vmap(lambda k, i: random.fold_in(random.fold_in(k, stage), i))(ekeys, kk)
                    return jax.vmap(lambda k: df.draw(random.normal, k, shape))(kk2)

                T, V = pt[sb, kk], pv[sb, kk]
                ejit = jitter[sb]
                xt = T + ejit[:, None] + c.TDCResolution * enormal(3, (2,))
                tot_thr = V + c.thresholdAnalogNoise * enormal(4, (2,))
                tq = (T + ejit[:, None]).reshape(2 * entries, 1) + ticks[None, :].astype(df.FLOAT) * c.TOTLSB
                volts_tot = volt(tq, lattice=int(round(c.TOTLSB * sr)), lanes=L)
                below = volts_tot < (tot_thr.reshape(2 * entries, 1) - c.triggerHysteresis)
                tot = jnp.where(jnp.any(below, 1), jnp.argmax(below, 1) + 1, c.maxTOT).reshape(entries, 2)
                if rel:
                    nr = n_ref[sb]
                    phase = jnp.maximum(0, jnp.ceil(T[:, 0] / c.ADCPeriod).astype(jnp.int64) + nr * per_adc
                                        - c.nADCPresamples) - nr * per_adc
                else:
                    phase = jnp.maximum(0, jnp.ceil(T[:, 0] / c.ADCPeriod).astype(jnp.int64) - c.nADCPresamples)
                times = jnp.repeat((phase[:, None] + adc_k).astype(df.FLOAT) * c.ADCPeriod + c.ADCOffset, 2, axis=0)
                volts = volt(times, "adc", lattice=int(round(c.ADCPeriod * sr)), lanes=L)
                sat = (maxposs[lane] > c.saturationVoltage) & jnp.repeat(present, 2)
                noise = c.adcAnalogNoise * enormal(5, (c.nADCSamples,))
                row = jnp.where(present, sb, batch)
                if saturation_pool:
                    adc_part = {"sat": sat, "lanes": L, "times": times, "volts": volts, "noise": noise, "row": row,
                                "kk": kk, "us": straw_us[sb]}
                else:
                    adc_part, sat_overflow = per_launch_saturation(L, times, sat, volts, noise, row, kk, straw_us[sb])
                dtime = jnp.max(xt, 1)
                if not c.overrideDbTimeOffsets:
                    xt -= cond["time_offsets"][straw_us[sb]]
                if rel:
                    tdc = jnp.clip(jnp.floor((xt + c.electronicsTimeDelay) / c.TDCLSB).astype(jnp.int64)
                                   + (nr * per_tdc)[:, None], 0, c.maxTDC)
                else:
                    tdc = jnp.clip(jnp.floor((xt + c.electronicsTimeDelay) / c.TDCLSB), 0, c.maxTDC).astype(jnp.int64)
                ewend = (int(c.tdc_response(c.eventWindowLength - c.electronicsTimeDelay))
                         if event_end_tdc is None else event_end_tdc[sb][:, None])
                ok_all = jnp.all((xt + c.electronicsTimeDelay > (-tref[sb][:, None] if rel else 0)) & (tdc < ewend), 1)
                upper = c.digitizationEndTDC if c.onspill else ewend
                ok_norm = jnp.all((tdc > c.digitizationStartTDC) & (tdc < upper), 1)
                ok = present & jnp.where(straw_ra[sb], ok_all, ok_norm)
                tdc_out = jnp.zeros((batch, pair_cap, 2), jnp.int64).at[row, kk].set(tdc, mode="drop")
                tot_out = jnp.zeros((batch, pair_cap, 2), jnp.int64).at[row, kk].set(tot, mode="drop")
                okm = jnp.zeros((batch, pair_cap), bool).at[row, kk].set(ok, mode="drop")
                dtm = jnp.zeros((batch, pair_cap), dtime.dtype).at[row, kk].set(dtime, mode="drop")

                def readybody(k, state):
                    ready, accepted = state
                    keep = okm[:, k] & (dtm[:, k] >= ready)
                    return jnp.where(keep, dtm[:, k] + c.deadTimeDigital, ready), accepted.at[:, k].set(keep)
                _, accepted = lax.fori_loop(0, jnp.minimum(pair_cap, jnp.max(number)), readybody,
                                            (jnp.full(batch, -9e9), jnp.zeros((batch, pair_cap), bool)))
                return adc_part, tdc_out, tot_out, accepted, sat_overflow if not saturation_pool else jnp.bool_(False)

            # an empty block (no pairs) skips the entry pass, as the pair loop runs no pass for it
            if saturation_pool:   # nothing saturated, every entry row dropped (row = batch)
                shapes = jax.eval_shape(flat_digits, None)[0]
                adc_empty = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), shapes)
                adc_empty["row"] = jnp.full(shapes["row"].shape, batch, shapes["row"].dtype)
            else:
                adc_empty = jnp.zeros((batch, pair_cap, c.nADCSamples), jnp.int64)
            empty = (adc_empty, jnp.zeros((batch, pair_cap, 2), jnp.int64),
                     jnp.zeros((batch, pair_cap, 2), jnp.int64), jnp.zeros((batch, pair_cap), bool), jnp.bool_(False))
            adc_out, tdc_out, tot_out, accepted, sat_overflow = lax.cond(total > 0, flat_digits, lambda _: empty, None)
            entry_fail = total > entries
        if saturation_pool:
            adc_fields = {"pool": adc_out}
        else:
            adc_fields = {"adc": adc_out, "pmp": pmp_of(adc_out)}
        step_out = jnp.stack([step[2 * jnp.arange(batch)[:, None], jnp.clip(pi[:, :, 0], 0, cap - 1)],
                              step[2 * jnp.arange(batch)[:, None] + 1, jnp.clip(pi[:, :, 1], 0, cap - 1)]], 2)
        return {**adc_fields, "tdc": tdc_out, "tot": tot_out, "step": step_out,
                "t_analog": pt, "keep": accepted, "failed": failed, "sat_overflow": sat_overflow,
                "pairs": number.sum(), "nan_ends": jnp.sum(jnp.isnan(tt) & live),
                "saturated_ends": jnp.sum((maxposs > c.saturationVoltage) & live),
                **({"entry_overflow": entry_fail} if entries is not None else {})}
    from types import SimpleNamespace
    return SimpleNamespace(run=run, pool_saturate=pool_saturate, pool_finish=pool_finish, srows=srows)


def digitize(st, nev, c, key, stats=None, *, event_timing=None):
    """Digitise flat steps with an explicit JAX key, returning host arrays like digi.digitize.

    Host boundaries select and pack variable-sized arrays; ionisation and electronics
    are JAX kernels. Conditions must be immutable for the life of the kernel cache.
    """
    from trkfastsim.digi.digi_clusters_jax import make_clusters
    from trkfastsim.digi.timing import step_times, ghost_indices
    random.key_data(key)
    st = {k: np.asarray(v) for k, v in st.items()}
    for field in ("t", "eion", "len", "width", "x0", "y0", "z0", "x1", "y1", "z1"):
        if not np.isfinite(st[field]).all():
            raise ValueError(f"nonfinite step {field}")
    sid, ev = st["sid"].astype(np.int64), st["ev"].astype(np.int64)
    if np.any(ev < 0) or np.any(ev >= nev):
        raise ValueError("step event index outside event population")
    us = reference.unique_straw(sid)
    if np.any(us < 0) or np.any(us >= len(c.wire_halflen)):
        raise ValueError("invalid StrawId")
    st["_us"] = us
    ra = ((sid & 0x7f) >= c.allStraw) & np.isin(sid >> 10, c.allPlanes)
    time, event_lengths = step_times(st, nev, c, event_timing)
    start = c.digitizationStart - c.timeFromProtonsToDRMarker
    end = c.digitizationEnd - c.timeFromProtonsToDRMarker if c.onspill else event_lengths[ev]
    inwin = (time > start - c.electronicsTimeDelay - c.steptimebuf) & (time < np.maximum(c.mbtime, end) - c.electronicsTimeDelay + c.steptimebuf)
    idx = np.flatnonzero((st["eion"] > c.minstepE) & (inwin | ra))
    if not len(idx):
        return _empty(c)
    cs, pos, ne = make_clusters(random.fold_in(key, 0), c, st, idx)
    if not len(cs):
        return _empty(c)
    ctime = time[cs]
    if df.SINGLE:   # float32: positions relative to the straw middle, times to whole COMMON_PERIODs (host float64)
        pos = pos - np.asarray(c.straw_mid)[us[cs]]
        cref = df.COMMON_PERIOD * np.floor(ctime / df.COMMON_PERIOD)
        ctime = ctime - cref
    charge_blocks = []
    group_size = CHARGE_BLOCK * CHARGE_BLOCKS
    for first in range(0, len(cs), group_size):
        last = min(first + group_size, len(cs))
        size = last - first
        pad = group_size - size
        args = (np.pad(us[cs[first:last]], (0, pad)), np.pad(pos[first:last], ((0, pad), (0, 0))),
                np.pad(ne[first:last], (0, pad), constant_values=1), np.pad(ctime[first:last], (0, pad)))
        result = _charge_blocks_kernel(c)(key, jnp.int64(first), *map(jnp.asarray, df.dev(args)))
        charge_blocks.append(tuple(np.asarray(x)[:size] for x in result))
    gt, charge, wd = (np.concatenate([b[i] for b in charge_blocks]) for i in range(3))
    if df.SINGLE:
        gt = (cref[:, None] + gt.astype(np.float64)).astype(np.float32)
    if not all(np.isfinite(x).all() for x in (gt, charge, wd)):
        raise ValueError("nonfinite drift/charge cluster before waveform packing")
    ukey, inst = np.unique(ev[cs] * 65536 + sid[cs], return_inverse=True)
    ns = len(ukey)
    straw_ev, straw_sid = ukey // 65536, ukey % 65536
    straw_us = reference.unique_straw(straw_sid)
    straw_ra = ((straw_sid & 0x7f) >= c.allStraw) & np.isin(straw_sid >> 10, c.allPlanes)
    panel = reference.unique_panel(straw_sid)
    jitter = _panel_jitter(random.fold_in(key, 2), straw_ev, panel, c.eventWindowMarkerROCJitter)
    counts = np.bincount(inst, minlength=ns)
    if c.onspill:
        buffer = (c.nADCSamples - c.nADCPresamples) * c.ADCPeriod
        ghost_src, ghost_time = ghost_indices(gt.ravel(), c.mbtime, buffer)
        ghost_row = 2 * inst[ghost_src // 2] + ghost_src % 2
        end_counts = np.bincount(ghost_row, minlength=2 * ns).reshape(ns, 2)
        ghost_order = np.argsort(ghost_row, kind="stable")
        ghost_starts = np.r_[0, np.cumsum(end_counts.ravel())[:-1]].reshape(ns, 2)
        counts = end_counts.max(axis=1)
    if counts.max() > MAX_ROW_CLUSTERS:
        raise OverflowError(f"waveform cluster count exceeds MAX_ROW_CLUSTERS={MAX_ROW_CLUSTERS}; no output returned")
    caps = np.left_shift(np.int64(1), np.ceil(np.log2(counts)).astype(np.int64))
    order = np.argsort(inst, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    outs = []
    for cap in np.unique(caps):
        bucket = np.flatnonzero(caps == cap)
        batch = _waveform_batch(int(cap))
        for b0 in range(0, len(bucket), batch):
            which = bucket[b0:b0 + batch]
            nb = len(which)
            if not c.onspill:
                gather = starts[which, None] + np.arange(cap)[None, :]
                mask = np.arange(cap)[None, :] < counts[which, None]
                ci_ = order[np.minimum(gather, len(order) - 1)]
            t = np.zeros((batch, 2, cap)); q = np.zeros_like(t); w = np.zeros_like(t)
            steps = np.zeros((batch, 2, cap), np.int64)
            n = np.zeros((batch, 2), np.int32)
            for end in (0, 1):
                if c.onspill:
                    gather = ghost_starts[which, end, None] + np.arange(cap)[None, :]
                    mask = np.arange(cap)[None, :] < end_counts[which, end, None]
                    gi = ghost_order[np.minimum(gather, len(ghost_order) - 1)]
                    oo = np.argsort(np.where(mask, ghost_time[gi], np.inf), axis=1, kind="stable")
                    gi = np.take_along_axis(gi, oo, 1)
                    ix = ghost_src[gi] // 2
                    t[:nb, end] = ghost_time[gi]
                    n[:nb, end] = end_counts[which, end]
                else:
                    oo = np.argsort(np.where(mask, gt[ci_, end], np.inf), axis=1, kind="stable")
                    ix = np.take_along_axis(ci_, oo, 1)
                    t[:nb, end] = gt[ix, end]
                    n[:nb, end] = counts[which]
                q[:nb, end] = charge[ix]
                w[:nb, end] = wd[ix, end]
                steps[:nb, end] = cs[ix]
            su = np.zeros(batch, np.int64); sr = np.zeros(batch, bool); sk = np.zeros(batch, np.int64)
            live = np.arange(batch) < nb; ji = np.zeros(batch)
            su[:nb], sr[:nb], sk[:nb], ji[:nb] = straw_us[which], straw_ra[which], ukey[which], jitter[which]
            extra = ()
            if df.SINGLE:   # float32: times relative to the straw's first cluster, floored to a COMMON_PERIOD
                valid = np.arange(cap)[None, None, :] < n[:nb, :, None]
                first_t = np.where(valid, t[:nb], np.inf).min(axis=(1, 2))
                nref = np.zeros(batch, np.int64)
                nref[:nb] = np.floor(first_t / df.COMMON_PERIOD).astype(np.int64)
                t = t - (nref * df.COMMON_PERIOD)[:, None, None]
                extra = (jnp.asarray(nref),)
            timing_args = {}
            if event_timing is not None:
                lengths = np.full(batch, c.eventWindowLength)
                lengths[:nb] = event_lengths[straw_ev[which]]
                timing_args = {"window_end": jnp.asarray(df.dev(lengths)),
                               "event_end_tdc": jnp.asarray(c.tdc_response(lengths - c.electronicsTimeDelay))}
            pair_cap = 4
            while True:
                result = _electronics_kernel(c, int(cap), pair_cap, batch)(random.fold_in(key, 3),
                    *[jnp.asarray(df.dev(x.reshape(2 * batch, cap))) for x in (t, q, w, steps)], jnp.asarray(n.ravel()),
                    jnp.asarray(su), jnp.asarray(sr), jnp.asarray(sk), jnp.asarray(live), jnp.asarray(df.dev(ji)), *extra, **timing_args)
                result = {k: np.asarray(v) for k, v in result.items()}
                if result["sat_overflow"]:
                    raise RuntimeError("JAX ADC saturation time bucket overflow; no output returned")
                if not result["failed"]:
                    break
                if pair_cap >= 2 * cap:
                    raise RuntimeError("JAX threshold search failed to progress; no output returned")
                pair_cap *= 2
            bi, pi = np.nonzero(result["keep"])
            out = {k: result[k][bi, pi] for k in ("adc", "tdc", "tot", "pmp", "step", "t_analog")}
            if df.SINGLE:
                out["t_analog"] = out["t_analog"].astype(np.float64) + (nref[bi] * df.COMMON_PERIOD)[:, None]
            out.update(ev=straw_ev[which[bi]], sid=straw_sid[which[bi]])
            outs.append(out)
            if stats is not None:
                for field, val in (("pairs", result["pairs"]), ("nan_crossing_ends", result["nan_ends"]),
                                   ("saturated_ends", result["saturated_ends"]), ("jax_buckets", 1)):
                    stats[field] = stats.get(field, 0) + int(val)
    out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    order = np.lexsort((out["t_analog"][:, 0], out["sid"], out["ev"]))
    if stats is not None:
        for field, val in (("clusters", len(cs)), ("straws", ns), ("ends", 2 * ns), ("digis", len(order))):
            stats[field] = stats.get(field, 0) + int(val)
    return {k: v[order] for k, v in out.items()}
