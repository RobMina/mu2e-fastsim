"""Photons -> SiPM charges with the photons resident on the device (ChainConfig.resident_response; runchain --jax).

The two-stage JAX twins hand every arrival photon to the host between the stages: photons_jax.make_photons returns
host arrays (about 19 M photons per 50k generated Run 1A cosmics), chain.charges masks and labels them on the host,
and sipm_jax.simulate sorts, groups and pads them on the host and uploads them again. Here the same kernels run with
the same keys and shapes, but the photons stay on the device: the status mask, labels, window cut, stable sort by
label, grouping into SiPM slots, padding and the fired-hit compaction run as device ops, and only the cut positions
(a few scalars), the fired charges and the final order cross to the host. The draws, the chunking (max_photons) and
the stream-overflow retries are those of sipm_jax.simulate, so the charges are the same arrays; a chunk that still
overflows goes to the numpy port with the same rng, as there. Thermal noise and the fused response are not covered.
"""
import numpy as np

import jax
import jax.numpy as jnp

from mu2efastsim import precision as dprec

from . import photons, photons_jax, sipm_jax

dprec.register(__name__)

I64_MAX = np.iinfo(np.int64).max


def _bucket(n, lo=1 << 12):
    return max(lo, 1 << int(np.ceil(np.log2(max(n, 1)))))


@jax.jit
def _take_rows(step, chan, rr, a):
    return step[rr + a], chan[rr + a]


def _photon_pieces(steps, pdg, geom, cfg, device_tables, deviation, rng, key, caps=None):
    """photons_jax.make_photons with the arrivals left on the device: [(step, chan, time, count)] in its order."""
    pieces = []
    for L, rows in photons.iter_rows(steps, pdg, geom, cfg, {k: v.L for k, v in device_tables.items()}, deviation, rng):
        key, sub = jax.random.split(key)
        dt = device_tables[L.name.replace("LookupTable_", "")]
        nsub = rows["nsub"]
        starts = np.r_[0, np.cumsum(nsub)]
        cuts = [0]
        max_sub = 1 << 22                                   # photons_jax.substep_photons default
        while cuts[-1] < len(nsub):
            nxt = int(np.searchsorted(starts, starts[cuts[-1]] + max_sub, side="right")) - 1
            cuts.append(max(nxt, cuts[-1] + 1))
        step_d, chan_d = jnp.asarray(rows["step"]), jnp.asarray(rows["chan"])
        k = sub
        for a, b in zip(cuts[:-1], cuts[1:]):
            R = b - a
            RB = photons_jax._bucket(R) if caps is None else caps.size("photon_rows", R, photons_jax._bucket(R))
            pad = RB - R
            chunk = {n: jnp.asarray(np.pad(rows[n][a:b], ((0, pad), (0, 0)))) for n in ("s", "d")}
            for n in photons_jax.ROW_KEYS:
                chunk[n] = jnp.asarray(np.pad(photons_jax._row(rows, n)[a:b], (0, pad)))
            chunk["nsub"] = jnp.asarray(np.pad(nsub[a:b], (0, pad)))
            ns = int(nsub[a:b].sum())
            SB = photons_jax._bucket(ns) if caps is None else caps.size("photon_substeps", ns, photons_jax._bucket(ns))
            k, ka, kb = jax.random.split(k, 3)
            narr, gbin, t, ri = photons_jax._kernel_a(chunk, ka, dt.edges, dt.prob, SB, dt.nb, dt.const, dt.offsets)
            total = int(narr.sum())
            if total == 0:
                continue
            AB = photons_jax._bucket(total)
            if caps is not None:
                AB = caps.size("photon_arrivals", total, AB)
            tt, rr, _ = photons_jax._kernel_b(narr, gbin, t, ri, kb, dt.cum, dt.dsum, AB)
            s, c = _take_rows(step_d, chan_d, rr, a)
            pieces.append((s, c, tt, total))
    return pieces


@jax.jit
def _place(buf, piece, off):
    return jax.lax.dynamic_update_slice(buf, piece, (off,))


def _concat(pieces, size):
    """Pieces [(s, c, t, count)] written one after the other into buffers of length size (tails overwritten)."""
    tdtype = pieces[0][2].dtype
    bufs = [jnp.zeros(size, jnp.int64), jnp.zeros(size, jnp.int64), jnp.zeros(size, tdtype)]
    off = 0
    for s, c, t, n in pieces:
        for i, x in enumerate((s, c, t)):
            bufs[i] = _place(bufs[i], x.astype(bufs[i].dtype), off)
        off += n
    return bufs, off


@jax.jit
def _label_sort(s, c, t, n, status, evkey, chan_mult, t_end):
    """Status mask, labels, window cut and the stable sort by label: kept photons first, in sipm_jax order."""
    i = jnp.arange(s.shape[0])
    keep = (i < n) & ((status[c] & 0b101) == 0)
    n_photons = keep.sum()
    label = evkey[s] * chan_mult + c
    keep &= t <= t_end
    o = jnp.argsort(jnp.where(keep, label, I64_MAX), stable=True)
    lab, tim, pho = label[o], t[o], s[o]
    nk = keep.sum()
    j = jnp.arange(s.shape[0])
    first = (j == 0) | (lab != jnp.roll(lab, 1))
    first &= j < nk
    rank = jnp.cumsum(first) - 1
    lastb = jax.lax.cummax(jnp.where(first, j, 0))
    return lab, tim, pho, rank, lastb, nk, n_photons


@jax.jit
def _chunk_inputs(lab, tim, pho, rank, a, n, NBarr):
    """Slot, time and photon of one chunk [a, a + n) padded to NB (zeros past n, as np.pad)."""
    NB = NBarr.shape[0]
    sl = lambda x: jax.lax.dynamic_slice(x, (a,), (NB,))
    valid = jnp.arange(NB) < n
    slot = jnp.where(valid, sl(rank) - rank[a], 0)
    time = jnp.where(valid, sl(tim), 0)
    return slot, time, valid, sl(lab), sl(pho)


@jax.jit
def _fired(r_fired, r_slot, r_root, r_time, r_charge, r_pe, lab_c, pho_c, rank_c, n):
    """Fired hits in index order: their label (the chunk's unique labels by slot), time, charge, pe and photon."""
    idx = jnp.nonzero(r_fired, size=r_fired.shape[0], fill_value=0)[0]
    slot = r_slot[idx]
    first = jnp.searchsorted(jnp.where(jnp.arange(rank_c.shape[0]) < n, rank_c, I64_MAX), slot, side="left")
    return (lab_c[first], r_time[idx], r_charge[idx], r_pe[idx], pho_c[jnp.minimum(r_root[idx], n - 1)])


def charges(steps, pdg, geom, cond, cfg, rng, key, tabs, pmap, status, event_keys, chan_mult, timers,
            max_photons=1 << 23, caps=None):
    """chain.charges (JAX two-stage path) with resident photons. Returns the same dict of host arrays.
    caps (crvfastsim.capacity.Capacities): fixed shapes from a capacity file and SiPM draws per hit section, so the
    charges do not depend on the shapes (another realization than caps=None)."""
    import time as _time
    from . import sipm as SI
    scfg = cfg.sipm
    if scfg.thermal_rate != 0:
        raise NotImplementedError("resident response: thermal noise is not covered (use the two-stage path)")
    if scfg.trap0_prob != 0 or scfg.trap1_prob != 0:
        raise NotImplementedError("sipm_jax: trap afterpulses are not in the twin")
    t_end = cfg.window[1]
    if caps is not None and caps.cap("sipm_photons"):
        max_photons = caps.cap("sipm_photons")       # SiPM chunks of the fixed size: only a file's last one is short
    t0 = _time.perf_counter()
    k1, k2 = jax.random.split(key)
    pieces = _photon_pieces(steps, pdg, geom, cfg.response, tabs, cond.deviation, rng, k1, caps)
    empty = {k: np.zeros(0) for k in ("chan", "time", "charge", "pe", "photon")}
    if not pieces:
        timers["photons_s"] = timers.get("photons_s", 0.0) + _time.perf_counter() - t0
        return empty
    total = sum(p[3] for p in pieces)
    size = _bucket(total) + _bucket(min(max(total, 1), max_photons))   # room for every chunk window [a, a + NB)
    if caps is not None:
        NBmax = caps.cap("sipm_photons") or _bucket(min(max(total, 1), max_photons))
        size = caps.size("photon_buffer", total + NBmax, size)
    (s, c, t), n = _concat(pieces, size)
    if int(jnp.max(jnp.where(jnp.arange(size) < n, c, 0))) >= chan_mult:
        raise ValueError(f"channel >= CHAN_MULT {chan_mult}: SiPM labels would overlap the next event")
    lab, tim, pho, rank, lastb, nk, n_photons = _label_sort(s, c, t, n, jnp.asarray(status), jnp.asarray(event_keys),
                                                             chan_mult, t_end)
    nk, n_photons = int(nk), int(n_photons)
    timers["photons_s"] = timers.get("photons_s", 0.0) + _time.perf_counter() - t0
    timers["photons"] = timers.get("photons", 0) + n_photons
    t0 = _time.perf_counter()
    dead = np.zeros(sipm_jax.NP2, bool)
    for x, y in scfg.inactive:
        dead[x * sipm_jax.NPIX + y] = True
    dead_d = jnp.asarray(dead)
    cum = jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum())
    cuts = [0]
    while cuts[-1] < nk:                                 # sipm_jax.simulate's cut rule on the device's boundaries
        b = int(lastb[min(cuts[-1] + max_photons, nk - 1)])
        cuts.append(b if b > cuts[-1] else nk)
    outs, on_host = [], False
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = b - a
        NB, Ms, Ntot, q_ct = sipm_jax.device_buckets(m, scfg)
        if caps is not None:
            NB, Ms, Ntot, q_ct = sipm_jax.device_buckets(m, scfg, NB=caps.size("sipm_photons", m, NB))
        if a + NB > size:   # one SiPM with more than max_photons photons: no room for its window
            raise RuntimeError("resident response: chunk window exceeds the photon buffer")
        slot, time, valid, lab_c, pho_c = _chunk_inputs(lab, tim, pho, rank, a, m, jnp.zeros(NB, bool))
        k2, sub = jax.random.split(k2)
        args = (slot, time, valid, sub, cum, pmap.shape[1], dead_d, q_ct, scfg.overvoltage,
                scfg.time_constant, scfg.p1, scfg.p2, scfg.capacitance, NB, *Ms, Ntot)
        r = sipm_jax.run_device_sipm(args, scfg, draws="offset" if caps is None else "section")
        if bool(r["overflow"]):
            ch, tm, ph = (np.asarray(x)[:m] for x in (lab_c, time, pho_c))
            outs.append(SI.simulate(ch, dprec.host(tm), ph, pmap, scfg, rng, t_end=t_end))
            on_host = True
            continue
        nf = int(r["fired"].sum())
        rank_c = jax.lax.dynamic_slice(rank, (a,), (NB,)) - rank[a]
        f = _fired(r["fired"], r["slot"], r["root"], r["time"], r["charge"], r["pe"], lab_c, pho_c, rank_c, m)
        outs.append(dict(zip(("chan", "time", "charge", "pe", "photon"), (dprec.host(np.asarray(x)[:nf]) for x in f))))
    out = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    o = np.lexsort((out["time"], out["chan"]))
    out = {k: v[o] for k, v in out.items()}
    timers["sipm_s"] = timers.get("sipm_s", 0.0) + _time.perf_counter() - t0
    timers["charges"] = timers.get("charges", 0) + len(out["time"])
    timers["resident_host_fallback"] = timers.get("resident_host_fallback", 0) + int(on_host)
    return out
