"""Bounded physical onspill CRV response; setup is host-side, apply is pure JAX."""
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from crvfastsim import chain as CH, onspill, readout
from crvfastsim.response import photons as PH, photons_jax as PJ, sipm as SI, sipm_jax as SJ
from crvfastsim.reco import waveform as WF, waveform_jax as WJ
from crvfastsim.reco import pulsefit as PF, pulsefit_jax as FJ, coinc as CO, coinc_jax as CJ
from mu2efastsim import precision as dprec


CAPACITIES = ("substeps", "photons", "thermal", "ct1", "ct2", "ct3", "pixel_short",
              "pixel_long_streams", "pixel_long_hits", "charge_clusters", "charges_per_cluster",
              "waveform_samples", "digis", "reco_waveforms", "reco_samples", "pulse_rounds",
              "coinc_hits", "coinc_combinations")
OPTIONS = {"waveform": ("dense", "scatter")}
STEP_FLOATS = ("edep", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "p1", "len")
PERIOD = 1695.0


def _fold(t):
    r = jnp.fmod(t, PERIOD)
    return jnp.where(r < 0, r + PERIOD, r)


def _step_gate(t0, t1, mc, lo=200.0, hi=1500.0):
    a, b = _fold(t0 + mc), _fold(t1 + mc)
    return (jnp.isfinite(t0) & jnp.isfinite(t1) &
            ~(((a < lo - 100) & (b < lo - 100)) | ((a > hi) & (b > hi)) | ((a > hi) & (b < lo - 100))))


def _groups(new, mask):
    row = jnp.arange(mask.size)
    first = lax.cummax(jnp.where(new, row, 0), axis=0)
    return jnp.cumsum(new) - 1, row - first


def _synth_scatter(time, weight, group, use, t_start, keep, tmpl, key, C, L, zs, precision, noise, min_voltage,
                   window_start, window_end):
    """Same waveforms and digi starts as `waveform_jax.synth_dense(stock_scan=True)`, built per charge.

    Each charge adds weight * template(t_sample - t_charge) to the few samples its template spans
    (a flat scatter-add of charges x span) instead of a dense [C, Q, L] gather, so the cost no
    longer scales with the largest cluster. Sample times, template indices, noise draws and the
    gated scan are computed exactly as in the dense kernel; only the floating-point summation order
    of overlapping charges differs.
    """
    period = WJ.PERIOD
    real = jnp.zeros(()).dtype                       # the dense kernel's T/W matrices use the default float
    time, weight = time.astype(real), weight.astype(real)
    span = int(np.ceil(tmpl.shape[0] * precision / period)) + 2
    g = jnp.clip(group, 0, C - 1)
    first = jnp.floor((time - t_start[g]) / period).astype(jnp.int64)
    j = jnp.maximum(first, 0)[:, None] + jnp.arange(span)[None, :]
    dt = (t_start[g][:, None] + j.astype(dprec.FLOAT) * period) - time[:, None]
    ti = jnp.rint(dt / precision).astype(jnp.int64)
    ok = use[:, None] & (dt >= 0) & (ti < tmpl.shape[0]) & (j < L)
    row = jnp.where(ok, g[:, None], C)
    col = jnp.clip(j, 0, L - 1)
    contrib = jnp.where(ok, tmpl[jnp.clip(ti, 0, tmpl.shape[0] - 1)] * weight[:, None], 0.0)
    wave = jnp.zeros((C, L), contrib.dtype).at[row, col].add(contrib, mode="drop")
    last = jnp.zeros(C, jnp.int64).at[row].max(jnp.where(ok, j + 1, 0), mode="drop")
    wave = wave + noise * jax.random.normal(key, wave.shape, dtype=dprec.FLOAT) * (jnp.arange(L)[None, :] < last[:, None])
    valid = jnp.arange(L)[None, :] < last[:, None]
    above = (wave > min_voltage) & valid
    pad = jnp.zeros((C, 2), bool)
    ap = jnp.concatenate([pad, above, pad], 1)
    win = (ap[:, 0:L] | ap[:, 1:L + 1] | ap[:, 2:L + 2] | ap[:, 3:L + 3] | ap[:, 4:L + 4]) & valid & keep[:, None]
    ts = t_start[:, None] + jnp.arange(L, dtype=dprec.FLOAT)[None, :] * period

    def step(next_i, i):
        start = win[:, i] & (next_i <= i) & (ts[:, i] >= window_start) & (ts[:, i] <= window_end)
        return jnp.where(start, i + zs, next_i), start

    _, starts = lax.scan(step, jnp.zeros(C, jnp.int64), jnp.arange(L))
    return wave, starts.T, last


def _counts_draw(avg, nsub, key):
    normal, poisson = jax.random.split(key)
    avg = jnp.maximum(avg, 0.0)
    draw = jnp.where(avg > 10, avg + jnp.sqrt(avg) * jax.random.normal(normal, avg.shape),
                     jax.random.poisson(poisson, jnp.where(avg <= 10, avg, 0.0)))
    return jnp.maximum(jnp.rint(draw / jnp.maximum(nsub, 1)), 0).astype(jnp.int64)


def _particle_props(pdg):
    mass, charge = jnp.zeros(pdg.shape), jnp.zeros(pdg.shape)
    for code, (m, q) in PH._PDG.items():
        mass, charge = jnp.where(pdg == code, m, mass), jnp.where(pdg == code, q, charge)
    absolute = jnp.abs(pdg)
    ion = absolute >= 1_000_000_000
    return (jnp.where(ion, ((absolute // 10) % 1000) * PH.AMU, mass),
            jnp.where(ion, ((absolute // 10000) % 1000) * jnp.sign(pdg), charge))


@dataclass(frozen=True)
class Adapter:
    assets: dict
    capacities: dict
    blocks: tuple
    table_static: tuple
    coincidence_static: tuple
    cfg: object
    wcfg: object
    pcfg: object
    fcfg: object
    nbars: int
    nfeb: int
    pmap_ncol: int
    waveform: str = "dense"
    kind: str = "physics"
    support: tuple = ("crv_digis", "crv_reco")

    @property
    def metadata(self):
        return dict(scope="bounded physical onspill photons/SiPM/ZS/pulses/coincidences",
                    period=PERIOD, window=list(self.cfg.window), capacities=dict(self.capacities),
                    sipm_cross_talk_generations=SJ.GENERATIONS, device_float=dprec.NAME,
                    rng="JAX keys; not NumPy/stock byte replay; shape-dependent",
                    waveform=self.waveform,
                    unsupported=["traps", "NZS", "background", "fused response"],
                    validation="local kernel/packing tests; no production closure or memory qualification")

    def _rows(self, columns, bar, active, assets, block, key):
        ti, reflector, sensor = block
        geo, tab = assets["geometry"], assets["tables"][ti]
        sl = jnp.stack([columns[a + "0"] for a in "xyz"], 1) - geo["center"][bar]
        el = jnp.stack([columns[a + "1"] for a in "xyz"], 1) - geo["center"][bar]
        sl = jnp.take_along_axis(sl, geo["l2w"][bar], 1)
        el = jnp.take_along_axis(el, geo["l2w"][bar], 1)
        signs = jnp.array([1, -1 if sensor in (2, 3) else 1,
                           (-1 if reflector < 0 else 1) * (-1 if sensor in (1, 3) else 1)])
        sl, el = sl * signs, el * signs
        delta = el - sl
        distance = jnp.linalg.norm(delta, axis=1)
        nsub = jnp.maximum(jnp.minimum(distance / PH.PRECISION_MM,
                                       self.capacities["substeps"] + 1).astype(jnp.int64), 1)
        selected = active & (geo["table"][bar] == ti) & (geo["reflector"][bar] == reflector)
        mass, charge = _particle_props(columns["pdg"])
        p0sq = sum(columns["p" + a + "0"] ** 2 for a in "xyz")
        energy = 0.5 * (jnp.sqrt(p0sq + mass ** 2) + jnp.sqrt(columns["p1"] ** 2 + mass ** 2))
        beta = jnp.sqrt(jnp.clip(1 - (mass / jnp.where(energy > 0, energy, 1)) ** 2, 0, 1))
        beta = jnp.where(mass == 0, 1., beta)
        theta = jnp.where(distance > 0, jnp.arccos(jnp.clip(delta[:, 2] / jnp.maximum(distance, 1e-300), -1, 1)), 0.)
        k = geo["speed"][bar]
        dev = (assets["deviation"][bar * 4 + sensor] * self.cfg.response.photon_yield_variation_scale + 1) * self.cfg.response.photon_yield_scale
        values = dict(s=sl, d=delta, t0=columns["t0"] + k * (sl[:, 2] + tab["half_length"]),
                      t1=columns["t1"] + k * (el[:, 2] + tab["half_length"]), beta=beta, theta=theta,
                      dev=dev, mu=geo["atten"][bar])
        ks, kc, kf = jax.random.split(key, 3)
        averages = [geo["yield"][bar] * columns["edep"]]
        for name in ("ck_scint", "ck_fiber"):
            cmap = tab[name]
            averages.append(jnp.interp(beta, cmap[:, 0], cmap[:, 1]) * jnp.abs(charge) * columns["len"])
        for name, avg, subkey in zip(("n_scint", "n_ck_s", "n_ck_f"), averages, (ks, kc, kf)):
            values[name] = _counts_draw(jnp.where(selected, avg, 0), nsub, subkey)
        values = {name: jnp.where(selected.reshape((-1,) + (1,) * (value.ndim - 1)), value, 0)
                  for name, value in values.items()}
        values["nsub"] = jnp.where(selected, nsub, 0)
        return values, bar * 4 + (sensor + 1 if reflector < 0 else sensor)

    def charges(self, collection, timing, assets, key):
        """Resident physical photons and thermal attempts through shared pixel histories."""
        c, cap = collection["columns"], self.capacities
        n = collection["mask"].shape[0]
        b = timing["event_mask"].shape[0]
        if n < 1 or b < 1:
            raise ValueError("fixed buffers need at least one lane; represent empty input with mask=False")
        if collection["mask"].shape != (n,) or collection["mask"].dtype.kind != "b":
            raise TypeError("CRV mask must be a boolean vector")
        if collection["event"].shape != (n,) or collection["event"].dtype.kind not in "iu":
            raise TypeError("CRV event must be an integer vector")
        for name in (*STEP_FLOATS, "t0", "t1", "bar", "pdg"):
            if c[name].shape != (n,):
                raise ValueError(f"CRV {name} must align with the row mask")
        for name in ("bar", "pdg"):
            if c[name].dtype.kind not in "iu":
                raise TypeError(f"CRV {name} must be an integer column")
        for name in ("event_mask", "mc_pbtime", "reco_pbtime", "event_length"):
            if timing[name].shape != (b,):
                raise ValueError(f"CRV timing {name} must align with event_mask")
        if timing["event_mask"].dtype.kind != "b":
            raise TypeError("CRV event_mask must be boolean")
        ev = jnp.clip(collection["event"], 0, b - 1)
        bar = jnp.clip(c["bar"], 0, self.nbars - 1).astype(jnp.int64)
        live = collection["mask"] & timing["event_mask"][ev]
        bad_index = collection["mask"] & ((collection["event"] < 0) | (collection["event"] >= b) |
                                         ~timing["event_mask"][ev] | (c["bar"] < 0) | (c["bar"] >= self.nbars))
        bad_payload = live & (jnp.isinf(c["t0"]) | jnp.isinf(c["t1"]))
        for name in STEP_FLOATS:
            bad_payload |= live & ~jnp.isfinite(c[name])
        for name in ("edep", "p1", "len"):
            bad_payload |= live & (c[name] < 0)
        bad_timing = jnp.zeros(b, bool)
        for name in ("mc_pbtime", "reco_pbtime", "event_length"):
            bad_timing |= ~jnp.isfinite(timing[name])
        bad_timing |= timing["event_length"] < self.cfg.window[1]
        active = live & ~bad_index & ~bad_payload & _step_gate(c["t0"], c["t1"], timing["mc_pbtime"][ev], *self.cfg.window)
        clean = {name: jnp.where(active, value, 0) for name, value in c.items()}
        keys = jax.random.split(key, len(self.blocks) + 2)
        photons, sub_over, arrival_over = [], jnp.array(False), jnp.array(False)
        numeric_error = jnp.array(False)
        for block, blockkey in zip(self.blocks, keys):
            kr, ka, kb = jax.random.split(blockkey, 3)
            rows, channel = self._rows(clean, bar, active, assets, block, kr)
            for value in rows.values():
                if value.dtype.kind == "f":
                    numeric_error |= jnp.any(~jnp.isfinite(value))
            numeric_error |= jnp.any(rows["nsub"] < 0)
            ti = block[0]
            tab = assets["tables"][ti]
            nb, const, offsets = self.table_static[ti]
            sub_over |= rows["nsub"].sum() > cap["substeps"]
            narr, bins, times, ri = PJ._kernel_a(rows, ka, tab["edges"], tab["prob"], cap["substeps"], nb, const, offsets)
            numeric_error |= jnp.any(narr < 0)
            narr = jnp.maximum(narr, 0)
            arrival_over |= narr.sum() > cap["photons"]
            times, rr, valid = PJ._kernel_b(narr, bins, times, ri, kb, tab["cum"], tab["dsum"], cap["photons"])
            channel = channel[rr]
            times = _fold(times + timing["mc_pbtime"][ev[rr]])
            numeric_error |= jnp.any(valid & ~jnp.isfinite(times))
            valid &= ((assets["status"][channel] & 0b101) == 0) & (times <= self.cfg.window[1])
            photons.append(dict(chan=ev[rr] * CH.CHAN_MULT + channel, time=times, photon=rr, mask=valid))
        optical = {name: jnp.concatenate([p[name] for p in photons]) for name in photons[0]}
        dark, thermal_over = self._thermal(timing, assets, keys[-2])
        slot = jnp.concatenate([optical["chan"], dark["chan"]])
        time = jnp.concatenate([optical["time"], dark["time"]])
        mask = jnp.concatenate([optical["mask"], dark["mask"]])
        root = jnp.concatenate([optical["photon"], jnp.full(cap["thermal"], -1, jnp.int64)])
        pixels = jnp.concatenate([jnp.full(optical["time"].size, -1, jnp.int64), dark["pixel"]])
        s = self.cfg.sipm
        total = slot.size + cap["ct1"] + cap["ct2"] + cap["ct3"]
        result = SJ.device_sipm(slot, time, mask, keys[-1], assets["pixel_cum"], self.pmap_ncol, assets["dead"],
                               assets["q_ct"], s.overvoltage, s.time_constant, s.p1, s.p2, s.capacitance,
                               slot.size, cap["ct1"], cap["ct2"], cap["ct3"], total, cap["pixel_short"],
                               cap["pixel_long_streams"], cap["pixel_long_hits"], s.n_pixels_x, s.n_pixels_y,
                               pixel_override=pixels)
        out = dict(chan=result["slot"], time=result["time"], charge=result["charge"], pe=result["pe"],
                   photon=root[jnp.minimum(result["root"], root.size - 1)], mask=result["fired"])
        order = jnp.lexsort((out["time"], out["chan"], ~out["mask"]))
        out = {name: values[order] for name, values in out.items()}
        errors = dict(crv_input=jnp.any(bad_index | bad_payload),
                      crv_timing=jnp.any(bad_timing & timing["event_mask"]),
                      crv_substeps=sub_over, crv_photons=arrival_over, crv_thermal=thermal_over,
                      crv_optics_numeric=numeric_error,
                      crv_sipm_candidates=result["candidate_overflow"], crv_pixel_streams=result["stream_overflow"])
        return out, errors

    def _thermal(self, timing, assets, key):
        kc, kt, kx, ky = jax.random.split(key, 4)
        live = ((assets["status"] & 0b101) == 0)[None, :] & timing["event_mask"][:, None]
        lo, hi = self.cfg.window
        lam = self.cfg.sipm.thermal_rate / self.cfg.sipm.p_fire(self.cfg.sipm.overvoltage) * (hi - lo + 100)
        count = jax.random.poisson(kc, jnp.where(live, lam, 0.), dtype=jnp.int64).ravel()
        size = self.capacities["thermal"]
        flat = jnp.repeat(jnp.arange(count.size), count, total_repeat_length=size)
        channel = flat % assets["status"].size
        event = flat // assets["status"].size
        pixel = (jax.random.randint(kx, (size,), 0, self.cfg.sipm.n_pixels_x) * SI.NPIX +
                 jax.random.randint(ky, (size,), 0, self.cfg.sipm.n_pixels_y))
        return dict(chan=event * CH.CHAN_MULT + channel,
                    time=lo - 100 + (hi - lo + 100) * jax.random.uniform(kt, (size,)),
                    pixel=pixel, mask=jnp.arange(size) < count.sum()), count.sum() > size

    def digis(self, q, timing, assets, key):
        """Device stock charge clustering, waveform synthesis, gated ZS and digi packing."""
        cap, w = self.capacities, self.wcfg
        C, Q, L, D = (cap[n] for n in ("charge_clusters", "charges_per_cluster", "waveform_samples", "digis"))
        order = jnp.lexsort((q["time"], q["chan"], ~q["mask"]))
        q = {k: v[order] for k, v in q.items()}
        mask = q["mask"]
        new = mask & jnp.concatenate([jnp.array([True]), (q["chan"][1:] != q["chan"][:-1]) |
                    (q["time"][1:] - q["time"][:-1] > w.single_pe_max_time + 4 * CH.PERIOD)])
        group, pos = _groups(new, mask)
        where = jnp.where(mask & (group < C) & (pos < Q), group, C)
        T = jnp.zeros((C, Q)).at[where, jnp.minimum(pos, Q - 1)].set(q["time"], mode="drop")
        W = jnp.zeros((C, Q)).at[where, jnp.minimum(pos, Q - 1)].set(q["charge"] / w.single_pe_reference_charge, mode="drop")
        counts = jnp.zeros(C, jnp.int64).at[jnp.where(mask, group, C)].add(1, mode="drop")
        first = jnp.nonzero(new, size=C, fill_value=0)[0]
        chan = q["chan"][first]
        kp, kn = jax.random.split(key)
        phases = jax.random.uniform(kp, (timing["event_mask"].size, self.nfeb)) * self.cfg.feb_phase_width
        tdc0 = phases[chan // CH.CHAN_MULT, assets["feb_index"][chan % CH.CHAN_MULT]]
        start = jnp.ceil((q["time"][first] - CH.PERIOD - tdc0) / CH.PERIOD) * CH.PERIOD + tdc0
        span = jnp.zeros(C).at[jnp.where(mask, group, C)].max(q["time"] - q["time"][first[jnp.clip(group, 0, C - 1)]], mode="drop")
        keep = (counts > 0) & (counts * assets["template"].max() >= w.min_voltage)
        need = jnp.ceil((span + w.single_pe_max_time) / CH.PERIOD).astype(jnp.int64) + 2 + w.samples_zs
        if self.waveform == "scatter":
            wave, starts, last = _synth_scatter(q["time"], q["charge"] / w.single_pe_reference_charge, group,
                    mask & (group < C), start, keep, assets["template"], kn, C, L, w.samples_zs, w.single_pe_precision,
                    w.noise if self.cfg.noise else 0., w.min_voltage, self.cfg.window[0], self.cfg.window[1])
        else:
            wave, _, starts, last = WJ.synth_dense(T, W, jnp.minimum(counts, Q), start, keep, assets["template"], kn,
                    L, w.samples_zs, w.single_pe_precision, w.noise if self.cfg.noise else 0., w.min_voltage,
                    w.adc_conversion, w.pedestal, w.min_adc, w.max_adc,
                    window_start=self.cfg.window[0], window_end=self.cfg.window[1], stock_scan=True)
        r, i = jnp.nonzero(starts, size=D, fill_value=0)
        cols = i[:, None] + jnp.arange(w.samples_zs)[None, :]
        length = jnp.maximum(jnp.floor((span + w.single_pe_max_time) / CH.PERIOD).astype(jnp.int64), last)
        volt = jnp.where(cols < jnp.minimum(length[r, None], L), wave[r[:, None], jnp.minimum(cols, L - 1)], 0.)
        adc = jnp.clip((volt * w.adc_conversion + w.pedestal + .5).astype(jnp.int16), w.min_adc, w.max_adc)
        st = start[r] + i * CH.PERIOD
        out = dict(mask=jnp.arange(D) < starts.sum(), ev=chan[r] // CH.CHAN_MULT,
                   bar=chan[r] % CH.CHAN_MULT // 4, sipm=chan[r] % 4,
                   starttdc=jnp.rint((st - tdc0[r]) / CH.PERIOD).astype(jnp.int64), adc=adc)
        order = jnp.lexsort((out["starttdc"], out["sipm"], out["bar"], out["ev"], ~out["mask"]))
        return {k: v[order] for k, v in out.items()}, dict(crv_charge_clusters=new.sum() > C,
                    crv_cluster_charges=(jnp.array(False) if self.waveform == "scatter" else jnp.any(mask & (pos >= Q))),
                    crv_waveform_samples=jnp.any((counts > 0) & (need > L)), crv_digis=starts.sum() > D)

    def pulses(self, d, timing, assets):
        """Concatenate contiguous digis before fitting; never fit each digi independently."""
        cap = self.capacities
        R, L = cap["reco_waveforms"], cap["reco_samples"]
        ch = d["bar"] * 4 + d["sipm"]
        new = d["mask"] & jnp.concatenate([jnp.array([True]), (d["ev"][1:] != d["ev"][:-1]) |
                    (ch[1:] != ch[:-1]) | (d["starttdc"][1:] != d["starttdc"][:-1] + self.wcfg.samples_zs)])
        group, pos = _groups(new, d["mask"])
        sample = pos[:, None] * self.wcfg.samples_zs + jnp.arange(self.wcfg.samples_zs)[None, :]
        row = jnp.where(d["mask"][:, None] & (group[:, None] < R) & (sample < L), group[:, None], R)
        adc = jnp.zeros((R, L), jnp.int64).at[row, jnp.minimum(sample, L - 1)].set(d["adc"], mode="drop")
        n = jnp.zeros(R, jnp.int64).at[jnp.where(d["mask"], group, R)].add(self.wcfg.samples_zs, mode="drop")
        first = jnp.nonzero(new, size=R, fill_value=0)[0]
        channels, event = ch[first], d["ev"][first]
        p = FJ.reconstruct_dense(adc, jnp.minimum(n, L), d["starttdc"][first], assets["pedestal"][channels],
                                assets["pulsearea"][channels], assets["pulseheight"][channels], cfg=self.pcfg)
        valid = p["found"] & (n > 0)[:, None]
        if self.pcfg.ignore_channels:
            valid &= ((assets["raw_status"][channels] & 0b010) == 0)[:, None]
        correction = -timing["reco_pbtime"][event]
        if self.pcfg.use_time_offset_db:
            correction += assets["timeoffset"][channels] * self.pcfg.time_offset_scale
        shape = p["found"].shape
        out = dict(mask=valid.ravel(), ev=jnp.broadcast_to(event[:, None], shape).ravel(),
                   bar=jnp.broadcast_to(channels[:, None] // 4, shape).ravel(),
                   sipm=jnp.broadcast_to(channels[:, None] % 4, shape).ravel(),
                   seq=jnp.broadcast_to(jnp.arange(shape[1]), shape).ravel(),
                   pe=p["pe"].ravel(), time=(p["time"] + correction[:, None]).ravel())
        order = jnp.lexsort((jnp.arange(valid.size), out["ev"], ~out["mask"]))
        out = {k: v[order] for k, v in out.items()}
        new_event = out["mask"] & jnp.concatenate([jnp.array([True]), out["ev"][1:] != out["ev"][:-1]])
        _, index = _groups(new_event, out["mask"])
        out["idx"] = index
        return out, dict(crv_reco_waveforms=new.sum() > R, crv_reco_samples=jnp.any(n > L),
                         crv_pulse_rounds=jnp.any((n > 0) & p["rounds_exhausted"]),
                         crv_pulse_window=jnp.any((n > 0) & p["window_overflow"]))

    def clusters(self, p, timing, assets):
        """Pack event/type groups on device and call the physical coincidence kernel."""
        B, H = timing["event_mask"].size, self.capacities["coinc_hits"]
        a = assets["coincidence"]
        ch = p["bar"] * 4 + p["sipm"]
        outputs, hit_over, comb_over = [], jnp.array(False), jnp.array(False)
        for ti, (stype, ld, ns, uniform) in enumerate(self.coincidence_static):
            mask = p["mask"] & (a["stype"][ch] == stype)
            if self.fcfg.ignore_secondary_pulses:
                mask &= p["seq"] == 0
            counts = jnp.zeros(B, jnp.int64).at[p["ev"]].add(mask.astype(jnp.int64))
            start = jnp.cumsum(counts) - counts
            rank = jnp.cumsum(mask) - 1 - start[p["ev"]]
            row = jnp.where(mask & (rank < H), p["ev"], B)
            col = jnp.clip(rank, 0, H - 1)
            def dense(value):
                return jnp.zeros((B, H) + value.shape[1:], value.dtype).at[row, col].set(value, mode="drop")
            bar = p["bar"]
            args = dict(valid=dense(mask), idx=dense(p["idx"]), side=dense(p["sipm"] % 2),
                        bar=dense(bar), sipm=dense(p["sipm"]), first=dense(p["seq"] == 0),
                        layer=dense(a["layer"][bar]), counter=dense(a["counter"][bar]),
                        x=dense(a["x"][bar]), y=dense(a["y"][bar]),
                        t=dense(p["time"] + a["tcorr"][ch]), pe=dense(p["pe"] * a["pefac"][ch]),
                        pos=dense(a["pos"][bar]), cut=dense(a["cuts"][ch]))
            reference = args["t"][:, 0]
            if dprec.SINGLE:
                for name in ("t", "x", "y"):
                    args[name] = jnp.where(args["valid"], args[name] - args[name][:, :1], args[name])
            result = CJ.find_dense(**args, tp=a["types"][ti]["tp"], cu=a["types"][ti]["cu"],
                                   ld=ld, ns=ns, uniform=uniform, K=self.capacities["coinc_combinations"])
            hit_over |= jnp.any(counts > H)
            comb_over |= jnp.any(result["overflow"])
            position = result["pos"].reshape(-1, 3)
            time = result["avgtime"] + (reference[:, None] if dprec.SINGLE else 0.)
            outputs.append(dict(mask=(result["root"] & timing["event_mask"][:, None]).ravel(),
                    ev=jnp.repeat(jnp.arange(B), H), sectortype=jnp.full(B * H, stype, jnp.int64),
                    pe=result["pe"].ravel(), time=time.ravel(), poscalc=result["poscalc"].ravel(),
                    x=position[:, 0], y=position[:, 1], z=position[:, 2]))
        out = {name: jnp.concatenate([o[name] for o in outputs]) for name in outputs[0]}
        return out, dict(crv_coinc_hits=hit_over, crv_coinc_combinations=comb_over)

    def apply(self, collection, timing, assets, key):
        """Return fixed-size digis/pulses/clusters and fatal batch error flags."""
        kq, kd = jax.random.split(key)
        q, eq = self.charges(collection, timing, assets, kq)
        d, ed = self.digis(q, timing, assets, kd)
        p, ep = self.pulses(d, timing, assets)
        c, ec = self.clusters(p, timing, assets)
        errors = {**eq, **ed, **ep, **ec}
        tables = {"digis": d, "pulses": p, "clusters": c}
        nonfinite = jnp.array(False)
        for table in tables.values():
            for name, value in table.items():
                if value.dtype.kind == "f":
                    finite = jnp.isfinite(value).reshape(value.shape[0], -1).all(axis=1)
                    nonfinite |= jnp.any(table["mask"] & ~finite)
        errors["crv_nonfinite"] = nonfinite
        rejected = jnp.any(jnp.stack(list(errors.values())))
        tables = {name: {**table, "mask": table["mask"] & ~rejected} for name, table in tables.items()}
        return tables, errors


def prepare(geom, cond, cfg, counters, capacities, tabs, pmap):
    """Prepare immutable numeric assets; caller uploads once and JITs adapter.apply.

    `capacities` may also carry the option `waveform`: "dense" (default, the shared
    `synth_dense` kernel) or "scatter" (`_synth_scatter`; `charges_per_cluster` is then unused).
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("JAX_ENABLE_X64=1 is required")
    cfg = deepcopy(cfg)
    capacities = dict(capacities)
    options = {name: capacities.pop(name, choices[0]) for name, choices in OPTIONS.items()}
    for name, value in options.items():
        if value not in OPTIONS[name]:
            raise ValueError(f"CRV option {name} must be one of {OPTIONS[name]}")
    missing, extra = set(CAPACITIES) - set(capacities), set(capacities) - set(CAPACITIES)
    if missing or extra:
        raise ValueError(f"explicit CRV capacities required; missing={sorted(missing)}, extra={sorted(extra)}")
    cap = dict(capacities)
    if any(type(v) is not int or v < 1 for v in cap.values()):
        raise ValueError("CRV capacities must be positive Python integers")
    if cap["reco_samples"] < 3 or cap["waveform_samples"] < 3:
        raise ValueError("waveform buffers need at least three samples")
    cond = readout.apply(cond, geom, counters)
    tm = onspill.EventTiming(np.array([[1, 0, 0]]), np.zeros(1), np.zeros(1), np.array([PERIOD]))
    status = onspill._validate(cfg, tm, geom, cond)
    if tuple(cfg.window) != (200., 1500.):
        raise NotImplementedError("resident adapter supports the stock onspill [200,1500] ZS gate only")
    if geom.nbars < 1:
        raise ValueError("CRV geometry must contain bars")
    center, l2w = np.asarray(geom.center), np.asarray(geom.local_to_world)
    if center.shape != (geom.nbars, 3) or not np.isfinite(center).all():
        raise ValueError("invalid CRV geometry centers")
    if l2w.shape != center.shape or l2w.dtype.kind not in "iu" or not np.all(np.sort(l2w, axis=1) == np.arange(3)):
        raise ValueError("invalid CRV local-to-world axis permutations")
    for name in ("pedestal", "pulsearea", "pulseheight", "timeoffset", "deviation", "roc", "feb"):
        v = np.asarray(getattr(cond, name))
        if v.ndim != 1 or len(v) < 4 * geom.nbars or not np.isfinite(v).all():
            raise ValueError(f"invalid CRV conditions {name}")
    if np.any(cond.pulsearea <= 0) or np.any(cond.pulseheight <= 0):
        raise ValueError("CRV pulse calibration must be positive")
    if np.any((cond.deviation * cfg.response.photon_yield_variation_scale + 1) < 0):
        raise ValueError("negative photon probability scale")
    s = cfg.sipm
    for v in (s.n_pixels_x, s.n_pixels_y):
        if type(v) is not int or not 1 <= v <= SI.NPIX:
            raise ValueError("invalid SiPM pixel dimensions")
    inactive = np.asarray(s.inactive)
    if inactive.size and (inactive.ndim != 2 or inactive.shape[1] != 2 or inactive.dtype.kind not in "iu"):
        raise ValueError("inactive SiPM pixels must be integer pairs")
    inactive = inactive.astype(np.int64).reshape(-1, 2)
    if np.any(inactive < 0) or np.any(inactive >= SI.NPIX):
        raise ValueError("invalid inactive SiPM pixels")
    if not 0 < s.p_fire(s.overvoltage) <= 1:
        raise ValueError("invalid fully charged SiPM firing probability")
    if not np.isfinite(1400 * s.thermal_rate / s.p_fire(s.overvoltage)):
        raise ValueError("nonfinite thermal Poisson mean")
    pmap = np.asarray(pmap, float)
    if pmap.ndim != 2 or min(pmap.shape) < 1 or max(pmap.shape) >= SI.NPIX or not np.isfinite(pmap).all() or np.any(pmap < 0) or pmap.sum() <= 0:
        raise ValueError("invalid optical pixel map")
    w = cfg.waveform if cfg.waveform is not None else WF.WaveformConfig()
    if w.samples_zs != 12:
        raise NotImplementedError("stock 12-sample ZS is required; NZS is not supported")
    for name in ("single_pe_precision", "single_pe_stretch", "single_pe_max_time", "single_pe_reference_charge", "adc_conversion"):
        if not np.isfinite(getattr(w, name)) or getattr(w, name) <= 0:
            raise ValueError(f"invalid waveform setting {name}")
    for name in ("noise", "min_voltage"):
        if not np.isfinite(getattr(w, name)) or getattr(w, name) < 0:
            raise ValueError(f"invalid waveform setting {name}")
    pcfg = replace(cfg.pulse_reco if cfg.pulse_reco is not None else PF.RecoConfig(), max_pulses=cap["pulse_rounds"])
    fcfg = cfg.finder if cfg.finder is not None else CO.FinderConfig()
    names = list(geom.sector_names)
    tnames = sorted({cfg.response.sectors[n].table for n in names})
    bysector = [cfg.response.sectors[n] for n in names]
    table_id = np.array([tnames.index(x.table) for x in bysector])[geom.sector]
    reflectors = np.array([x.reflector for x in bysector])[geom.sector]
    yields = np.array([x.yield_per_mev for x in bysector])[geom.sector]
    if not np.isfinite(yields).all() or np.any(yields < 0):
        raise ValueError("invalid scintillation yield")
    speed = PH.bar_ratios(cfg.response.speed_ratio, geom)
    atten = PH.bar_ratios(cfg.response.atten_ratio, geom)
    if not np.isfinite(speed).all() or not np.isfinite(atten).all():
        raise ValueError("nonfinite CRV propagation ratios")
    tables, statics, blocks = [], [], []
    for ti, name in enumerate(tnames):
        dt = tabs[name]
        if not isinstance(dt, PJ.DeviceTable):
            dt = PJ.DeviceTable(dt)
        lookup = dt.L
        inv_v, att = PH.table_propagation(lookup)
        use = table_id == ti
        speed[use] = inv_v * (1 / speed[use] - 1)
        atten[use] = (1 / atten[use] - 1) / att
        for refl in sorted(set(reflectors[use])):
            if abs(refl) != lookup.reflector:
                raise ValueError(f"reflector does not match table {name}")
            blocks.extend((ti, int(refl), sensor) for sensor in ((0, 1, 2, 3) if refl == 0 else (0, 2)))
        tables.append(dict(edges={k: np.asarray(v) for k, v in dt.edges.items()}, prob=np.asarray(dt.prob),
                    cum=np.asarray(dt.cum), dsum=np.asarray(dt.dsum), half_length=np.array(lookup.half_length),
                    ck_scint=np.vstack([np.zeros((1, 2)), lookup.ck_scint]),
                    ck_fiber=np.vstack([np.zeros((1, 2)), lookup.ck_fiber])))
        statics.append((dt.nb, dt.const, dt.offsets))
    nchan = 4 * geom.nbars
    febs = np.column_stack([np.asarray(cond.roc[:nchan]), np.asarray(cond.feb[:nchan])])
    ufeb, feb_index = np.unique(febs, axis=0, return_inverse=True)
    dead = np.zeros(SI.NPIX * SI.NPIX, bool)
    dead[inactive[:, 0] * SI.NPIX + inactive[:, 1]] = True
    G = CO.sector_geom(geom, counters)
    params = CJ.type_params(G, fcfg)
    if not params:
        raise ValueError("no configured CRV coincidence sector types")
    coincidence = CJ.hit_tables(G, cond.status, fcfg)
    coincidence.update(layer=G.layer, counter=G.counter_in_type, x=G.x, y=G.y, pos=G.pos,
                       types=tuple(dict(tp=p.tp, cu=p.cu) for p in params.values()))
    assets = dict(geometry=dict(center=np.asarray(geom.center), l2w=np.asarray(geom.local_to_world),
                    table=table_id, reflector=reflectors, **{"yield": yields}, speed=speed, atten=atten),
                  tables=tuple(tables), status=status, raw_status=np.asarray(cond.status[:nchan]),
                  deviation=np.asarray(cond.deviation[:nchan]), feb_index=feb_index,
                  pixel_cum=np.cumsum(pmap.ravel()) / pmap.sum(), dead=dead,
                  q_ct=np.array((1 - (1 - s.cross_talk) ** .25) / s.p_fire(s.overvoltage)),
                  template=WF.single_pe_template(w.single_pe_file, w.single_pe_precision, w.single_pe_stretch, w.single_pe_max_time),
                  coincidence=coincidence,
                  **{name: np.asarray(getattr(cond, name)[:nchan]) for name in ("pedestal", "pulsearea", "pulseheight", "timeoffset")})
    assets = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), assets)
    for leaf in jax.tree_util.tree_leaves(assets):
        leaf.setflags(write=False)
    return Adapter(assets, cap, tuple(blocks), tuple(statics),
                   tuple((int(t), p.ld, p.ns, p.uniform) for t, p in params.items()),
                   cfg, w, pcfg, fcfg, geom.nbars, len(ufeb), pmap.shape[1], options["waveform"])
